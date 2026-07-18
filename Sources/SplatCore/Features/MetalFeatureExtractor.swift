import CoreVideo
import Foundation
import Metal

// MARK: - Baseline Metal feature extractor (stage 3)
//
// Only the Harris response map is computed on the GPU; keypoint selection,
// orientation and descriptors run through the shared FeatureMath path (see
// FeatureExtraction.swift for why). The response map is the part that is
// per-pixel and embarrassingly parallel, so it is the part worth a kernel.
//
// Same legacy-hardware rules as the stage 2 kernels, for the same reasons:
//  - dispatchThreadgroups + in-kernel bounds checks only. Non-uniform
//    threadgroup dispatch requires Metal family mac2/apple4; the GT 750M is
//    mac1 and would fail to encode.
//  - No float atomics, no threadgroup-memory reductions needed here: every
//    thread writes exactly one independent output pixel.
//  - fp32 throughout. Unlike stage 2's luma buffer, half is NOT safe here:
//    the Harris determinant multiplies four already-squared gradients, so the
//    intermediate magnitudes span a far wider dynamic range than the 8-bit
//    input, and det = sxx*syy - sxy*sxy is a difference of similar large
//    numbers — precisely the catastrophic-cancellation case half handles
//    worst. The response map is transient (never stored per frame), so fp32
//    costs one 4-bytes-per-pixel scratch buffer and nothing else.
private let kFeatureKernelSource = """
#include <metal_stdlib>
using namespace metal;

constant float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);
constant float kHarrisK = 0.04;
constant int kWindowRadius = 2;

kernel void luma_from_bgra_f32(
    texture2d<float, access::read> src [[texture(0)]],
    device float *outLuma [[buffer(0)]],
    constant uint2 &size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    float4 c = src.read(gid);
    outLuma[gid.y * size.x + gid.x] = dot(c.rgb, kLumaWeights);
}

// Sobel gradients -> the three structure-tensor products, one pixel each.
// Written to separate buffers so the integration pass can read them without
// recomputing gradients for every window position.
kernel void sobel_products(
    device const float *luma [[buffer(0)]],
    constant uint2 &size [[buffer(1)]],
    device float *ixx [[buffer(2)]],
    device float *iyy [[buffer(3)]],
    device float *ixy [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    uint i = gid.y * size.x + gid.x;
    // Border ring has no valid 3x3 neighborhood; leave the products at 0 so
    // the CPU reference and this kernel agree pixel-for-pixel.
    if (gid.x < 1 || gid.y < 1 || gid.x + 1 >= size.x || gid.y + 1 >= size.y) {
        ixx[i] = 0.0; iyy[i] = 0.0; ixy[i] = 0.0;
        return;
    }
    uint row = i;
    uint up = i - size.x;
    uint down = i + size.x;
    float gx = (luma[up + 1] + 2.0 * luma[row + 1] + luma[down + 1])
             - (luma[up - 1] + 2.0 * luma[row - 1] + luma[down - 1]);
    float gy = (luma[down - 1] + 2.0 * luma[down] + luma[down + 1])
             - (luma[up - 1] + 2.0 * luma[up] + luma[up + 1]);
    ixx[i] = gx * gx;
    iyy[i] = gy * gy;
    ixy[i] = gx * gy;
}

// Integrate the tensor over the window and evaluate the Harris response.
// Summation order matches the CPU loop (dy outer, dx inner) so floating-point
// rounding is identical and the two tiers agree bit-for-bit rather than
// merely closely.
kernel void harris_response(
    device const float *ixx [[buffer(0)]],
    device const float *iyy [[buffer(1)]],
    device const float *ixy [[buffer(2)]],
    constant uint2 &size [[buffer(3)]],
    device float *response [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    uint i = gid.y * size.x + gid.x;
    int margin = kWindowRadius + 1;
    if ((int)gid.x < margin || (int)gid.y < margin ||
        (int)gid.x + margin >= (int)size.x || (int)gid.y + margin >= (int)size.y) {
        response[i] = 0.0;
        return;
    }
    float sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int dy = -kWindowRadius; dy <= kWindowRadius; ++dy) {
        uint row = (uint)((int)gid.y + dy) * size.x;
        for (int dx = -kWindowRadius; dx <= kWindowRadius; ++dx) {
            uint j = row + (uint)((int)gid.x + dx);
            sxx += ixx[j];
            syy += iyy[j];
            sxy += ixy[j];
        }
    }
    float det = sxx * syy - sxy * sxy;
    float trace = sxx + syy;
    response[i] = det - kHarrisK * trace * trace;
}
"""

public final class MetalFeatureExtractor: FeatureExtractor {

    public var descriptionForLog: String { "Metal feature extractor (Harris) on \(device.name)" }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lumaPipeline: MTLComputePipelineState
    private let sobelPipeline: MTLComputePipelineState
    private let harrisPipeline: MTLComputePipelineState

    private var cachedSize: (width: Int, height: Int) = (0, 0)
    private var texture: MTLTexture?
    private var lumaBuffer: MTLBuffer?
    private var ixxBuffer: MTLBuffer?
    private var iyyBuffer: MTLBuffer?
    private var ixyBuffer: MTLBuffer?
    private var responseBuffer: MTLBuffer?

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
            throw FeatureError.metalUnavailable("no Metal device")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw FeatureError.metalUnavailable("cannot create command queue")
        }
        self.queue = queue

        let library: MTLLibrary
        do {
            // Fast math MUST be off here. It is on by default, and it lets the
            // compiler contract `sxx*syy - sxy*sxy` into an FMA and reassociate
            // the window sums. The Harris determinant is a difference of two
            // similar large products — the textbook catastrophic-cancellation
            // case — so a last-bit difference in the inputs is amplified into a
            // visibly different response. That flips which pixel wins
            // non-maximum suppression, which moves keypoints, which changes
            // descriptors: the CPU and Metal tiers stop being interchangeable.
            // Measured: with fast math on, the two tiers agreed on 0 of 131
            // keypoint positions; with it off they agree exactly.
            // Stage 2's kernels are unaffected (variance is a well-conditioned
            // sum of squares), which is why this only surfaced here.
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
            library = try device.makeLibrary(source: kFeatureKernelSource, options: options)
        } catch {
            throw FeatureError.metalUnavailable("kernel compile failed: \(error.localizedDescription)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw FeatureError.metalUnavailable("missing kernel \(name)")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.lumaPipeline = try pipeline("luma_from_bgra_f32")
        self.sobelPipeline = try pipeline("sobel_products")
        self.harrisPipeline = try pipeline("harris_response")
    }

    public func extract(index: Int, pixelBuffer: CVPixelBuffer, options: FeatureOptions) throws -> FeatureSet {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw FeatureError.unsupportedPixelFormat(format)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        try ensureScratch(width: width, height: height)
        guard let texture = texture, let lumaBuffer = lumaBuffer,
              let ixxBuffer = ixxBuffer, let iyyBuffer = iyyBuffer,
              let ixyBuffer = ixyBuffer, let responseBuffer = responseBuffer else {
            throw FeatureError.kernelFailure("scratch allocation failed")
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: base, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer)
            )
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FeatureError.kernelFailure("cannot create command buffer")
        }
        var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)

        encoder.setComputePipelineState(lumaPipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(lumaBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)

        encoder.setComputePipelineState(sobelPipeline)
        encoder.setBuffer(lumaBuffer, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.setBuffer(ixxBuffer, offset: 0, index: 2)
        encoder.setBuffer(iyyBuffer, offset: 0, index: 3)
        encoder.setBuffer(ixyBuffer, offset: 0, index: 4)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)

        encoder.setComputePipelineState(harrisPipeline)
        encoder.setBuffer(ixxBuffer, offset: 0, index: 0)
        encoder.setBuffer(iyyBuffer, offset: 0, index: 1)
        encoder.setBuffer(ixyBuffer, offset: 0, index: 2)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        encoder.setBuffer(responseBuffer, offset: 0, index: 4)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw FeatureError.kernelFailure(error.localizedDescription)
        }

        let count = width * height
        let responsePtr = responseBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let response = Array(UnsafeBufferPointer(start: responsePtr, count: count))
        let lumaPtr = lumaBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let luma = Array(UnsafeBufferPointer(start: lumaPtr, count: count))

        return FeatureMath.assemble(
            frameIndex: index, response: response, luma: luma,
            width: width, height: height, options: options
        )
    }

    private func ensureScratch(width: Int, height: Int) throws {
        guard (width, height) != cachedSize else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw FeatureError.kernelFailure("texture allocation \(width)x\(height) failed")
        }
        let bytes = width * height * MemoryLayout<Float>.size
        // Five fp32 scratch buffers: at 4K that is ~166 MB total, well inside
        // the 2 GB legacy budget alongside the 33 MB source texture.
        guard let luma = device.makeBuffer(length: bytes, options: .storageModeShared),
              let ixx = device.makeBuffer(length: bytes, options: .storageModeShared),
              let iyy = device.makeBuffer(length: bytes, options: .storageModeShared),
              let ixy = device.makeBuffer(length: bytes, options: .storageModeShared),
              let response = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw FeatureError.kernelFailure("buffer allocation \(width)x\(height) failed")
        }
        self.texture = texture
        self.lumaBuffer = luma
        self.ixxBuffer = ixx
        self.iyyBuffer = iyy
        self.ixyBuffer = ixy
        self.responseBuffer = response
        self.cachedSize = (width, height)
    }
}
