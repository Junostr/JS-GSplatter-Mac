import Foundation
import Metal

// MARK: - Stage 5: Metal forward rasterizer (baseline tier)
//
// Work split deliberately across CPU and GPU by what each is good at:
//
//   CPU — project each splat, depth sort, and work out which tiles it touches.
//         O(splats), a few thousand items, trivial.
//   GPU — blend every pixel against its tile's splat list.
//         O(pixels x splats-per-tile), the part that actually costs.
//
// The reference CUDA implementation of 3DGS instead sorts splats into tiles on
// the GPU with a radix sort. That is the wrong shape for the baseline tier
// here: a GPU sort needs either heavy atomics or multi-pass scan machinery, and
// this must run on a GeForce GT 750M (GPUFamilyMac1) with limited atomics and
// no non-uniform threadgroup dispatch. Sorting is not the bottleneck anyway —
// per-pixel blending is — so moving it to the CPU costs almost nothing and
// removes the entire class of legacy-GPU compatibility problems.
//
// Same legacy rules as every other kernel in this project:
//  - dispatchThreadgroups only, never dispatchThreads.
//  - No atomics, no threadgroup reductions: each thread owns one pixel and
//    writes only that pixel.
//  - fp32 throughout. Alpha compositing multiplies a long chain of
//    (1 - alpha) terms, and half's 11-bit mantissa visibly quantises
//    transmittance after a few dozen overlapping splats.
private let kRasterizerSource = """
#include <metal_stdlib>
using namespace metal;

struct RasterParams {
    uint width;
    uint height;
    uint tilesX;
    uint tileSize;
    float backgroundR;
    float backgroundG;
    float backgroundB;
    float transmittanceCutoff;
};

// One threadgroup per tile, one thread per pixel.
//
// Splat attributes arrive as separate flat float arrays rather than a struct.
// A struct would need its layout to match byte-for-byte between Swift and MSL,
// and MSL's float3 is 16-byte aligned while Swift's SIMD3 packing differs —
// exactly the kind of mismatch that produces plausible-looking garbage rather
// than a clean failure. Flat arrays with explicit strides cannot drift.
kernel void rasterize_tiles(
    device const float *centres      [[buffer(0)]],   // 2 per splat
    device const float *conics       [[buffer(1)]],   // 3 per splat
    device const float *colors       [[buffer(2)]],   // 3 per splat
    device const float *opacities    [[buffer(3)]],   // 1 per splat
    device const float *radii        [[buffer(9)]],   // 1 per splat
    device const uint  *tileRanges   [[buffer(4)]],   // 2 per tile: start, count
    device const uint  *tileSplats   [[buffer(5)]],   // flattened splat indices
    device float       *outColor     [[buffer(6)]],   // 3 per pixel
    device float       *outTrans     [[buffer(7)]],   // 1 per pixel
    constant RasterParams &params    [[buffer(8)]],
    uint2 tileID   [[threadgroup_position_in_grid]],
    uint2 localID  [[thread_position_in_threadgroup]])
{
    uint x = tileID.x * params.tileSize + localID.x;
    uint y = tileID.y * params.tileSize + localID.y;
    // Bounds check in-kernel: the grid is rounded up to whole tiles because
    // non-uniform dispatch is unavailable on the baseline GPU.
    if (x >= params.width || y >= params.height) { return; }

    uint pixelIndex = y * params.width + x;
    uint tileIndex = tileID.y * params.tilesX + tileID.x;
    uint start = tileRanges[tileIndex * 2 + 0];
    uint count = tileRanges[tileIndex * 2 + 1];

    float3 accumulated = float3(0.0);
    float transmittance = 1.0;
    float px = float(x);
    float py = float(y);

    for (uint k = 0; k < count; ++k) {
        if (transmittance <= params.transmittanceCutoff) { break; }
        uint s = tileSplats[start + k];

        float cx = centres[s * 2 + 0];
        float cy = centres[s * 2 + 1];
        // Same bounding-box test the CPU reference applies.
        //
        // Tile assignment is a superset of per-splat coverage: a splat is
        // listed for every tile its box overlaps, so without this check the GPU
        // would blend it at pixels inside the tile but OUTSIDE the box. Beyond
        // 3 sigma a high-opacity splat can still clear the 1/255 alpha floor,
        // so those extra contributions are real, not rounding — measured as a
        // 5.3e-3 peak divergence from the reference. Arguably the GPU was the
        // more correct of the two, but the tiers must agree, and the reference
        // is the one the gradients were verified against.
        float r = radii[s];
        if (px < floor(cx - r) || px > ceil(cx + r) ||
            py < floor(cy - r) || py > ceil(cy + r)) { continue; }
        float dx = px - cx;
        float dy = py - cy;
        float ca = conics[s * 3 + 0];
        float cb = conics[s * 3 + 1];
        float cc = conics[s * 3 + 2];
        float power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy;
        if (power > 0.0) { continue; }

        float alpha = min(0.99, opacities[s] * exp(power));
        if (alpha < 1.0 / 255.0) { continue; }

        float weight = alpha * transmittance;
        accumulated += float3(colors[s * 3 + 0], colors[s * 3 + 1], colors[s * 3 + 2]) * weight;
        transmittance *= (1.0 - alpha);
    }

    float3 background = float3(params.backgroundR, params.backgroundG, params.backgroundB);
    accumulated += background * transmittance;

    outColor[pixelIndex * 3 + 0] = accumulated.x;
    outColor[pixelIndex * 3 + 1] = accumulated.y;
    outColor[pixelIndex * 3 + 2] = accumulated.z;
    outTrans[pixelIndex] = transmittance;
}
"""

public final class MetalSplatRasterizer {

    public var descriptionForLog: String { "Metal splat rasterizer on \(device.name)" }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    /// Threads per threadgroup edge. 16 gives 256 threads, inside every Metal
    /// family's limit including mac1, and matches the tier parameters' default.
    private let tileSize: Int

    public init(device: MTLDevice? = nil, tileSize: Int = 16) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
            throw TrainingEngineError.deviceUnavailable("no Metal device")
        }
        self.device = device
        self.tileSize = tileSize
        guard let queue = device.makeCommandQueue() else {
            throw TrainingEngineError.deviceUnavailable("cannot create command queue")
        }
        self.queue = queue
        do {
            // Fast math off, as everywhere else in this project. Alpha
            // compositing is a long product of (1 - alpha) terms and the
            // reference CPU path must be reproducible against it; reassociation
            // would put the two tiers permanently slightly apart.
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: kRasterizerSource, options: options)
            guard let function = library.makeFunction(name: "rasterize_tiles") else {
                throw TrainingEngineError.deviceUnavailable("missing kernel rasterize_tiles")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as TrainingEngineError {
            throw error
        } catch {
            throw TrainingEngineError.deviceUnavailable("kernel compile failed: \(error.localizedDescription)")
        }
    }

    /// Assign projected splats to the tiles their screen extent overlaps.
    ///
    /// Splats are visited in depth order and appended per tile, so each tile's
    /// list is already front-to-back — the GPU can then blend in list order
    /// with no sorting of its own.
    static func buildTiles(
        projected: [ProjectedSplat], width: Int, height: Int, tileSize: Int
    ) -> (ranges: [UInt32], indices: [UInt32], order: [Int]) {
        let tilesX = (width + tileSize - 1) / tileSize
        let tilesY = (height + tileSize - 1) / tileSize
        let tileCount = tilesX * tilesY

        let order = projected.indices.sorted {
            projected[$0].depth != projected[$1].depth
                ? projected[$0].depth < projected[$1].depth
                : projected[$0].index < projected[$1].index
        }

        var perTile = [[UInt32]](repeating: [], count: tileCount)
        for (slot, projectedIndex) in order.enumerated() {
            let splat = projected[projectedIndex]
            let minX = max(0, Int((splat.centre.x - splat.radius).rounded(.down)))
            let maxX = min(width - 1, Int((splat.centre.x + splat.radius).rounded(.up)))
            let minY = max(0, Int((splat.centre.y - splat.radius).rounded(.down)))
            let maxY = min(height - 1, Int((splat.centre.y + splat.radius).rounded(.up)))
            guard minX <= maxX, minY <= maxY else { continue }

            let tileX0 = minX / tileSize, tileX1 = maxX / tileSize
            let tileY0 = minY / tileSize, tileY1 = maxY / tileSize
            for ty in tileY0...tileY1 {
                for tx in tileX0...tileX1 {
                    perTile[ty * tilesX + tx].append(UInt32(slot))
                }
            }
        }

        var ranges = [UInt32](repeating: 0, count: tileCount * 2)
        var indices: [UInt32] = []
        indices.reserveCapacity(perTile.reduce(0) { $0 + $1.count })
        for tile in 0..<tileCount {
            ranges[tile * 2] = UInt32(indices.count)
            ranges[tile * 2 + 1] = UInt32(perTile[tile].count)
            indices.append(contentsOf: perTile[tile])
        }
        return (ranges, indices, order)
    }

    /// Render on the GPU. Output matches `SplatRasterizer.render` — the CPU
    /// implementation is the reference this is verified against.
    public func render(
        cloud: SplatCloud, pose: CameraPose, intrinsics: CameraIntrinsics,
        width: Int, height: Int,
        background: SIMD3<Float> = SIMD3<Float>(repeating: 0),
        options: RasterizerOptions = RasterizerOptions()
    ) throws -> RenderTarget {
        var target = RenderTarget(width: width, height: height)
        let projected = SplatRasterizer.project(cloud: cloud, pose: pose, intrinsics: intrinsics,
                                                width: width, height: height, options: options)

        let tilesX = (width + tileSize - 1) / tileSize
        let tilesY = (height + tileSize - 1) / tileSize
        let (ranges, tileIndices, order) = MetalSplatRasterizer.buildTiles(
            projected: projected, width: width, height: height, tileSize: tileSize)

        // Splat attributes in DEPTH ORDER, so the tile lists index directly.
        let n = max(order.count, 1)
        var centres = [Float](repeating: 0, count: n * 2)
        var conics = [Float](repeating: 0, count: n * 3)
        var colors = [Float](repeating: 0, count: n * 3)
        var opacities = [Float](repeating: 0, count: n)
        var radii = [Float](repeating: 0, count: n)
        for (slot, projectedIndex) in order.enumerated() {
            let splat = projected[projectedIndex]
            centres[slot * 2] = splat.centre.x
            centres[slot * 2 + 1] = splat.centre.y
            conics[slot * 3] = splat.conic.0
            conics[slot * 3 + 1] = splat.conic.1
            conics[slot * 3 + 2] = splat.conic.2
            colors[slot * 3] = splat.color.x
            colors[slot * 3 + 1] = splat.color.y
            colors[slot * 3 + 2] = splat.color.z
            opacities[slot] = splat.opacity
            radii[slot] = splat.radius
        }

        func buffer(_ array: [Float]) throws -> MTLBuffer {
            guard let b = device.makeBuffer(bytes: array, length: max(array.count, 1) * MemoryLayout<Float>.stride,
                                            options: .storageModeShared) else {
                throw TrainingEngineError.deviceUnavailable("buffer allocation failed")
            }
            return b
        }
        func uintBuffer(_ array: [UInt32]) throws -> MTLBuffer {
            let padded = array.isEmpty ? [UInt32(0)] : array
            guard let b = device.makeBuffer(bytes: padded, length: padded.count * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared) else {
                throw TrainingEngineError.deviceUnavailable("buffer allocation failed")
            }
            return b
        }

        let centreBuffer = try buffer(centres)
        let conicBuffer = try buffer(conics)
        let colorBuffer = try buffer(colors)
        let opacityBuffer = try buffer(opacities)
        let radiusBuffer = try buffer(radii)
        let rangeBuffer = try uintBuffer(ranges)
        let indexBuffer = try uintBuffer(tileIndices)
        guard let outColor = device.makeBuffer(length: width * height * 3 * MemoryLayout<Float>.stride,
                                               options: .storageModeShared),
              let outTrans = device.makeBuffer(length: width * height * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
            throw TrainingEngineError.deviceUnavailable("output allocation failed")
        }

        var params = RasterParams(
            width: UInt32(width), height: UInt32(height),
            tilesX: UInt32(tilesX), tileSize: UInt32(tileSize),
            backgroundR: background.x, backgroundG: background.y, backgroundB: background.z,
            transmittanceCutoff: options.transmittanceCutoff)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TrainingEngineError.deviceUnavailable("cannot create command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(centreBuffer, offset: 0, index: 0)
        encoder.setBuffer(conicBuffer, offset: 0, index: 1)
        encoder.setBuffer(colorBuffer, offset: 0, index: 2)
        encoder.setBuffer(opacityBuffer, offset: 0, index: 3)
        encoder.setBuffer(rangeBuffer, offset: 0, index: 4)
        encoder.setBuffer(indexBuffer, offset: 0, index: 5)
        encoder.setBuffer(outColor, offset: 0, index: 6)
        encoder.setBuffer(outTrans, offset: 0, index: 7)
        encoder.setBytes(&params, length: MemoryLayout<RasterParams>.stride, index: 8)
        encoder.setBuffer(radiusBuffer, offset: 0, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: tilesX, height: tilesY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tileSize, height: tileSize, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw TrainingEngineError.deviceUnavailable(error.localizedDescription)
        }

        let colorPtr = outColor.contents().bindMemory(to: Float.self, capacity: width * height * 3)
        let transPtr = outTrans.contents().bindMemory(to: Float.self, capacity: width * height)
        target.pixels = Array(UnsafeBufferPointer(start: colorPtr, count: width * height * 3))
        target.transmittance = Array(UnsafeBufferPointer(start: transPtr, count: width * height))
        return target
    }

    /// Mirrors the MSL struct. Kept adjacent to the shader so the two are
    /// edited together; field order and types must match exactly.
    private struct RasterParams {
        var width: UInt32
        var height: UInt32
        var tilesX: UInt32
        var tileSize: UInt32
        var backgroundR: Float
        var backgroundG: Float
        var backgroundB: Float
        var transmittanceCutoff: Float
    }
}
