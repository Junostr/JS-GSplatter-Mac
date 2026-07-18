import Foundation
import CoreVideo
import Metal

// MARK: - Baseline Metal analyzer (stage 2)
//
// Kernel source is compiled at runtime with makeLibrary(source:) instead of
// shipping a .metallib: this project has three build paths (SPM, Xcode, raw
// swiftc for distribution) and only some of them can bundle resources. A
// one-time ~10 ms compile at init is a fair price for kernels that work
// identically on all of them. Revisit when stage 5's kernel count grows.
//
// Baseline constraints honored throughout (this must run on a GT 750M):
//  - dispatchThreadgroups + in-kernel bounds checks, never dispatchThreads:
//    non-uniform threadgroup dispatch needs Metal family mac2/apple4, which
//    legacy Nvidia (mac1) lacks.
//  - No float atomics (not universally supported): per-threadgroup partial
//    sums are reduced in threadgroup memory and finished on the CPU.
//  - Threadgroup memory use peaks at 2 KB (256 floats × 2 arrays) — far
//    inside the 48 KB, even Kepler-era, limit.
//  - Buffers use .storageModeShared (valid for buffers on every macOS GPU,
//    including discrete); the input texture uses the default .managed mode
//    on discrete GPUs, with replaceRegion handling the CPU→GPU sync.
private let kAnalyzerKernelSource = """
#include <metal_stdlib>
using namespace metal;

// Rec. 709 luma weights — matches the CPU analyzer exactly so tier choice
// never changes scores.
constant float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

// Luma stored as half. NOT because it is lossless — it isn't: luma is a
// Rec.709 weighted combination, not an 8-bit sample, and half's ULP near 0.5
// is ~4.9e-4. The justification is that the ingestion path always delivers
// 8-bit BGRA (kCVPixelFormatType_32BGRA), whose own quantization step is
// ~3.9e-3 — about 8x COARSER than half's. So the storage error is always
// dominated by quantization already present in the input, and it stays well
// inside the 5% cross-tier blur tolerance even in the worst case for this
// (a smooth gradient, where the true Laplacian is ~0 and any staircase shows
// up directly): measured 0.02% divergence vs the fp32 CPU path, pinned by the
// "smooth-gradient blur agrees" assertion in selftest. Halving this buffer is
// worth keeping on the 2 GB legacy tier, where it is read back twice.
// If a future ingestion path ever delivers >8-bit frames, revisit: the
// dominance argument disappears and this should become float.
kernel void luma_from_bgra(
    texture2d<float, access::read> src [[texture(0)]],
    device half *outLuma [[buffer(0)]],
    constant uint2 &size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    float4 c = src.read(gid);
    outLuma[gid.y * size.x + gid.x] = half(dot(c.rgb, kLumaWeights));
}

// 5-point Laplacian + first/second moment accumulation.
// Accumulation is fp32 on purpose: summing lap^2 over an 8-megapixel frame
// overflows half almost immediately, and E[x^2]-E[x]^2 cancellation needs
// the full mantissa. Border pixels contribute 0 (identical to the CPU path).
kernel void laplacian_stats(
    device const half *luma [[buffer(0)]],
    constant uint2 &size [[buffer(1)]],
    device float2 *partials [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgPerGrid [[threadgroups_per_grid]])
{
    threadgroup float sums[256];
    threadgroup float sqs[256];

    float lap = 0.0;
    if (gid.x >= 1 && gid.y >= 1 && gid.x + 1 < size.x && gid.y + 1 < size.y) {
        uint idx = gid.y * size.x + gid.x;
        float c = float(luma[idx]);
        float l = float(luma[idx - 1]);
        float r = float(luma[idx + 1]);
        float u = float(luma[idx - size.x]);
        float d = float(luma[idx + size.x]);
        lap = 4.0 * c - l - r - u - d;
    }
    sums[tid] = lap;
    sqs[tid] = lap * lap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sqs[tid] += sqs[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[tgid.y * tgPerGrid.x + tgid.x] = float2(sums[0], sqs[0]);
    }
}

// 8x8 grid of mean luma — the scene-change signature. One threadgroup per
// cell (8x8 threadgroups of 64 threads), each striding over its cell.
kernel void cell_means(
    device const half *luma [[buffer(0)]],
    constant uint2 &size [[buffer(1)]],
    device float *cells [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float sums[64];
    uint x0 = tgid.x * size.x / 8;
    uint x1 = (tgid.x + 1) * size.x / 8;
    uint y0 = tgid.y * size.y / 8;
    uint y1 = (tgid.y + 1) * size.y / 8;
    uint w = x1 - x0;
    uint total = w * (y1 - y0);

    float sum = 0.0;
    for (uint i = tid; i < total; i += 64) {
        uint x = x0 + (i % w);
        uint y = y0 + (i / w);
        sum += float(luma[y * size.x + x]);
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) { sums[tid] += sums[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        cells[tgid.y * 8 + tgid.x] = total > 0 ? sums[0] / float(total) : 0.0;
    }
}
"""

public final class MetalFrameAnalyzer: FrameAnalyzer {

    public var descriptionForLog: String { "Metal analyzer on \(device.name)" }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lumaPipeline: MTLComputePipelineState
    private let laplacianPipeline: MTLComputePipelineState
    private let cellsPipeline: MTLComputePipelineState

    // Per-resolution scratch, reused across frames of the same size. VRAM
    // cost for 4K: ~33 MB texture + ~17 MB half luma + <1 MB partials —
    // comfortably inside the 2 GB legacy budget.
    private var cachedSize: (width: Int, height: Int) = (0, 0)
    private var texture: MTLTexture?
    private var lumaBuffer: MTLBuffer?
    private var partialsBuffer: MTLBuffer?
    private var partialsCount = 0
    private var cellsBuffer: MTLBuffer?

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
            throw FrameAnalyzerError.metalUnavailable("no Metal device")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw FrameAnalyzerError.metalUnavailable("cannot create command queue")
        }
        self.queue = queue

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kAnalyzerKernelSource, options: nil)
        } catch {
            throw FrameAnalyzerError.metalUnavailable("kernel compile failed: \(error.localizedDescription)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw FrameAnalyzerError.metalUnavailable("missing kernel \(name)")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.lumaPipeline = try pipeline("luma_from_bgra")
        self.laplacianPipeline = try pipeline("laplacian_stats")
        self.cellsPipeline = try pipeline("cell_means")
    }

    public func analyze(index: Int, timestamp: Double?, pixelBuffer: CVPixelBuffer) throws -> FrameScore {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw FrameAnalyzerError.unsupportedPixelFormat(format)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        try ensureScratch(width: width, height: height)
        guard let texture = texture, let lumaBuffer = lumaBuffer,
              let partialsBuffer = partialsBuffer, let cellsBuffer = cellsBuffer else {
            throw FrameAnalyzerError.kernelFailure("scratch allocation failed")
        }

        // Upload. A plain replaceRegion copy for now; the zero-copy
        // CVMetalTextureCache path arrives with stage 5, where per-frame
        // upload cost actually matters (training touches frames repeatedly;
        // scoring touches each once and is decode-bound in practice).
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: base,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer)
            )
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FrameAnalyzerError.kernelFailure("cannot create command buffer")
        }

        var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
        let tg16 = MTLSize(width: 16, height: 16, depth: 1)
        let grid16 = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)

        encoder.setComputePipelineState(lumaPipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(lumaBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.dispatchThreadgroups(grid16, threadsPerThreadgroup: tg16)

        encoder.setComputePipelineState(laplacianPipeline)
        encoder.setBuffer(lumaBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.setBuffer(partialsBuffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(grid16, threadsPerThreadgroup: tg16)

        encoder.setComputePipelineState(cellsPipeline)
        encoder.setBuffer(lumaBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.setBuffer(cellsBuffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: 8, height: 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw FrameAnalyzerError.kernelFailure(error.localizedDescription)
        }

        // Finish the reduction on the CPU: a few thousand float2s, cheaper
        // than a second GPU pass and avoids float atomics (see header).
        let partials = partialsBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: partialsCount)
        var sum: Double = 0
        var sumSq: Double = 0
        for i in 0..<partialsCount {
            sum += Double(partials[i].x)
            sumSq += Double(partials[i].y)
        }
        let n = Double(width * height)
        let mean = sum / n
        let variance = max(0, sumSq / n - mean * mean)

        let cells = cellsBuffer.contents().bindMemory(to: Float.self, capacity: 64)
        let signature = Array(UnsafeBufferPointer(start: cells, count: 64))

        return FrameScore(index: index, timestamp: timestamp, blurScore: variance, signature: signature)
    }

    private func ensureScratch(width: Int, height: Int) throws {
        guard (width, height) != cachedSize else { return }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw FrameAnalyzerError.kernelFailure("texture allocation \(width)x\(height) failed")
        }

        partialsCount = ((width + 15) / 16) * ((height + 15) / 16)
        // 2 bytes per pixel = MSL half. (Not MemoryLayout<Float16> — Swift's
        // Float16 doesn't exist on x86_64, and this file compiles universally.)
        guard let luma = device.makeBuffer(length: width * height * 2, options: .storageModeShared),
              let partials = device.makeBuffer(length: partialsCount * MemoryLayout<SIMD2<Float>>.size, options: .storageModeShared),
              let cells = device.makeBuffer(length: 64 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw FrameAnalyzerError.kernelFailure("buffer allocation \(width)x\(height) failed")
        }

        self.texture = texture
        self.lumaBuffer = luma
        self.partialsBuffer = partials
        self.cellsBuffer = cells
        self.cachedSize = (width, height)
    }
}
