import Foundation
import Metal

// MARK: - Stage 5: Metal backward kernel (baseline tier)
//
// The backward pass scatters: every pixel a splat covers contributes to that
// splat's gradient. On a GPU that is a many-to-one accumulation, which normally
// wants float atomics — and `atomic_float` add is not available on
// GPUFamilyMac1, the tier this must run on.
//
// Two ways round that, and the choice matters:
//
//   Float atomics emulated with a compare-and-swap loop on uint. Portable, but
//   every contending thread retries, and a splat covering hundreds of pixels
//   inside one tile is maximum contention. Rejected.
//
//   THREADGROUP REDUCTION per splat, used here. One threadgroup owns one tile;
//   for each splat in that tile's list, every thread computes its own pixel's
//   contribution, the threadgroup reduces them in threadgroup memory, and one
//   thread writes a single per-(tile, splat) partial. No atomics at any point,
//   and the reduction mechanism is the same one stage 2's Laplacian kernel
//   already runs on this hardware.
//
// The CPU then sums those partials per splat and runs the screen-space ->
// parameter-space chain, which is per-splat rather than per-pixel and so cheap
// that moving it to the GPU would buy nothing — while re-deriving it in MSL
// would duplicate the one piece of maths in this project that finite
// differences had to be brought in to verify.
private let kBackwardSource = """
#include <metal_stdlib>
using namespace metal;

struct BackwardParams {
    uint width;
    uint height;
    uint tilesX;
    uint tileSize;
    float transmittanceCutoff;
};

// 9 accumulated values per (tile, splat):
//   0,1  dL/d(centre.x), dL/d(centre.y)
//   2,3,4 dL/d(conic A, B, C)
//   5    dL/d(opacity)
//   6,7,8 dL/d(colour r, g, b)
constant uint kSlots = 9;

kernel void backward_tiles(
    device const float *centres     [[buffer(0)]],
    device const float *conics      [[buffer(1)]],
    device const float *colors      [[buffer(2)]],
    device const float *opacities   [[buffer(3)]],
    device const float *radii       [[buffer(4)]],
    device const uint  *tileRanges  [[buffer(5)]],
    device const uint  *tileSplats  [[buffer(6)]],
    device const float *finalColor  [[buffer(7)]],   // forward output, 3/pixel
    device const float *dLdPixel    [[buffer(8)]],   // loss gradient, 3/pixel
    device float       *partials    [[buffer(9)]],   // kSlots per tile-splat entry
    constant BackwardParams &params [[buffer(10)]],
    uint2 tileID      [[threadgroup_position_in_grid]],
    uint2 localID     [[thread_position_in_threadgroup]],
    uint2 groupSize   [[threads_per_threadgroup]])
{
    // MSL requires every thread-position attribute in a kernel to share a
    // vector type, so the linear index and count are derived rather than taken
    // from the scalar `thread_index_in_threadgroup` attribute.
    uint threadIndex = localID.y * groupSize.x + localID.x;
    uint threadCount = groupSize.x * groupSize.y;
    // 256 threads x 9 slots x 4 bytes = 9 KB, well inside the 32 KB floor.
    threadgroup float scratch[256 * 9];

    uint x = tileID.x * params.tileSize + localID.x;
    uint y = tileID.y * params.tileSize + localID.y;
    bool inside = (x < params.width && y < params.height);
    uint pixelIndex = inside ? (y * params.width + x) : 0;

    uint tileIndex = tileID.y * params.tilesX + tileID.x;
    uint start = tileRanges[tileIndex * 2 + 0];
    uint count = tileRanges[tileIndex * 2 + 1];

    // Each thread replays the forward walk for its own pixel, front to back,
    // exactly as the CPU reference does. Replaying is cheaper than storing:
    // caching per-pixel per-splat transmittance would be O(pixels x splats).
    float transmittance = inside ? 1.0 : 0.0;
    float3 accumulated = float3(0.0);
    float px = float(x);
    float py = float(y);

    float3 pixelGrad = float3(0.0);
    float3 finalPixel = float3(0.0);
    if (inside) {
        pixelGrad = float3(dLdPixel[pixelIndex * 3 + 0],
                           dLdPixel[pixelIndex * 3 + 1],
                           dLdPixel[pixelIndex * 3 + 2]);
        finalPixel = float3(finalColor[pixelIndex * 3 + 0],
                            finalColor[pixelIndex * 3 + 1],
                            finalColor[pixelIndex * 3 + 2]);
    }

    for (uint k = 0; k < count; ++k) {
        uint s = tileSplats[start + k];
        float contribution[9];
        for (uint slot = 0; slot < kSlots; ++slot) { contribution[slot] = 0.0; }

        bool active = inside && (transmittance > params.transmittanceCutoff);
        if (active) {
            float cx = centres[s * 2 + 0];
            float cy = centres[s * 2 + 1];
            float r = radii[s];
            if (px < floor(cx - r) || px > ceil(cx + r) ||
                py < floor(cy - r) || py > ceil(cy + r)) {
                active = false;
            } else {
                float dx = px - cx;
                float dy = py - cy;
                float ca = conics[s * 3 + 0];
                float cb = conics[s * 3 + 1];
                float cc = conics[s * 3 + 2];
                float power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy;
                if (power > 0.0) {
                    active = false;
                } else {
                    float g = exp(power);
                    float opacity = opacities[s];
                    float rawAlpha = opacity * g;
                    float alpha = min(0.99, rawAlpha);
                    if (alpha < 1.0 / 255.0) {
                        active = false;
                    } else {
                        float weight = alpha * transmittance;
                        float3 splatColor = float3(colors[s * 3 + 0], colors[s * 3 + 1], colors[s * 3 + 2]);

                        contribution[6] = pixelGrad.x * weight;
                        contribution[7] = pixelGrad.y * weight;
                        contribution[8] = pixelGrad.z * weight;

                        float3 mine = splatColor * weight;
                        float3 behind = finalPixel - accumulated - mine;
                        float oneMinusAlpha = max(1.0 - alpha, 1e-6);
                        float3 term = splatColor * transmittance - behind / oneMinusAlpha;
                        float dLdAlpha = dot(pixelGrad, term);
                        float dAlphaDRaw = (rawAlpha <= 0.99) ? 1.0 : 0.0;
                        float dLdRawAlpha = dLdAlpha * dAlphaDRaw;

                        contribution[5] = dLdRawAlpha * g;
                        float dLdPower = dLdRawAlpha * opacity * g;

                        contribution[2] = dLdPower * (-0.5 * dx * dx);
                        contribution[3] = dLdPower * (-dx * dy);
                        contribution[4] = dLdPower * (-0.5 * dy * dy);

                        float dPowerDdx = -(ca * dx) - cb * dy;
                        float dPowerDdy = -(cc * dy) - cb * dx;
                        contribution[0] = dLdPower * (-dPowerDdx);
                        contribution[1] = dLdPower * (-dPowerDdy);

                        // Advance this pixel's forward state, as the CPU does.
                        accumulated += mine;
                        transmittance *= (1.0 - alpha);
                    }
                }
            }
        }

        // Reduce the threadgroup's contributions for this splat.
        for (uint slot = 0; slot < kSlots; ++slot) {
            scratch[threadIndex * kSlots + slot] = contribution[slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threadCount / 2; stride > 0; stride >>= 1) {
            if (threadIndex < stride) {
                for (uint slot = 0; slot < kSlots; ++slot) {
                    scratch[threadIndex * kSlots + slot] +=
                        scratch[(threadIndex + stride) * kSlots + slot];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (threadIndex == 0) {
            for (uint slot = 0; slot < kSlots; ++slot) {
                partials[(start + k) * kSlots + slot] = scratch[slot];
            }
        }
        // Required: the next iteration overwrites scratch, and without this
        // a fast thread could start writing while a slow one is still reading.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""

public final class MetalSplatBackward {

    public var descriptionForLog: String { "Metal splat backward on \(device.name)" }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let tileSize: Int
    private static let slots = 9

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
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: kBackwardSource, options: options)
            guard let function = library.makeFunction(name: "backward_tiles") else {
                throw TrainingEngineError.deviceUnavailable("missing kernel backward_tiles")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as TrainingEngineError {
            throw error
        } catch {
            throw TrainingEngineError.deviceUnavailable("kernel compile failed: \(error.localizedDescription)")
        }
    }

    /// Loss and gradients, with the per-pixel work on the GPU.
    ///
    /// Output must match `SplatBackward.lossAndGradients`, which is the
    /// reference the finite-difference tests verified.
    public func lossAndGradients(
        cloud: SplatCloud, pose: CameraPose, intrinsics: CameraIntrinsics,
        width: Int, height: Int, reference: [Float],
        background: SIMD3<Float> = SIMD3<Float>(repeating: 0),
        options: RasterizerOptions = RasterizerOptions()
    ) throws -> (loss: Double, gradients: SplatGradients) {

        var gradients = SplatGradients(count: cloud.count)
        let projected = SplatRasterizer.project(cloud: cloud, pose: pose, intrinsics: intrinsics,
                                                width: width, height: height, options: options)
        let pixelCount = width * height
        guard reference.count == pixelCount * 3 else { return (0, gradients) }

        // Forward on the CPU rasterizer: the backward kernel needs the final
        // colour per pixel, and the two paths agree to float round-off.
        let target = SplatRasterizer.render(projected: projected, width: width, height: height,
                                            background: background, options: options)
        var dLdPixel = [Float](repeating: 0, count: pixelCount * 3)
        var loss = 0.0
        let inverseN = Float(1.0 / Double(pixelCount * 3))
        for i in 0..<(pixelCount * 3) {
            let diff = target.pixels[i] - reference[i]
            loss += Double(abs(diff))
            dLdPixel[i] = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) * inverseN
        }
        loss /= Double(pixelCount * 3)
        guard !projected.isEmpty else { return (loss, gradients) }

        let tilesX = (width + tileSize - 1) / tileSize
        let tilesY = (height + tileSize - 1) / tileSize
        let (ranges, tileIndices, order) = MetalSplatRasterizer.buildTiles(
            projected: projected, width: width, height: height, tileSize: tileSize)

        let n = max(order.count, 1)
        var centres = [Float](repeating: 0, count: n * 2)
        var conics = [Float](repeating: 0, count: n * 3)
        var colors = [Float](repeating: 0, count: n * 3)
        var opacities = [Float](repeating: 0, count: n)
        var radii = [Float](repeating: 0, count: n)
        for (slot, projectedIndex) in order.enumerated() {
            let splat = projected[projectedIndex]
            centres[slot * 2] = splat.centre.x; centres[slot * 2 + 1] = splat.centre.y
            conics[slot * 3] = splat.conic.0
            conics[slot * 3 + 1] = splat.conic.1
            conics[slot * 3 + 2] = splat.conic.2
            colors[slot * 3] = splat.color.x
            colors[slot * 3 + 1] = splat.color.y
            colors[slot * 3 + 2] = splat.color.z
            opacities[slot] = splat.opacity
            radii[slot] = splat.radius
        }

        func floatBuffer(_ array: [Float]) throws -> MTLBuffer {
            let padded = array.isEmpty ? [Float(0)] : array
            guard let b = device.makeBuffer(bytes: padded, length: padded.count * MemoryLayout<Float>.stride,
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

        let entryCount = max(tileIndices.count, 1)
        guard let partials = device.makeBuffer(
            length: entryCount * MetalSplatBackward.slots * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw TrainingEngineError.deviceUnavailable("partial allocation failed")
        }
        memset(partials.contents(), 0, entryCount * MetalSplatBackward.slots * MemoryLayout<Float>.stride)

        var params = BackwardParams(width: UInt32(width), height: UInt32(height),
                                    tilesX: UInt32(tilesX), tileSize: UInt32(tileSize),
                                    transmittanceCutoff: options.transmittanceCutoff)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TrainingEngineError.deviceUnavailable("cannot create command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(try floatBuffer(centres), offset: 0, index: 0)
        encoder.setBuffer(try floatBuffer(conics), offset: 0, index: 1)
        encoder.setBuffer(try floatBuffer(colors), offset: 0, index: 2)
        encoder.setBuffer(try floatBuffer(opacities), offset: 0, index: 3)
        encoder.setBuffer(try floatBuffer(radii), offset: 0, index: 4)
        encoder.setBuffer(try uintBuffer(ranges), offset: 0, index: 5)
        encoder.setBuffer(try uintBuffer(tileIndices), offset: 0, index: 6)
        encoder.setBuffer(try floatBuffer(target.pixels), offset: 0, index: 7)
        encoder.setBuffer(try floatBuffer(dLdPixel), offset: 0, index: 8)
        encoder.setBuffer(partials, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<BackwardParams>.stride, index: 10)
        encoder.dispatchThreadgroups(
            MTLSize(width: tilesX, height: tilesY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tileSize, height: tileSize, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw TrainingEngineError.deviceUnavailable(error.localizedDescription)
        }

        // Sum the per-(tile, splat) partials into per-splat screen gradients.
        let slots = MetalSplatBackward.slots
        let ptr = partials.contents().bindMemory(to: Float.self, capacity: entryCount * slots)
        var dLdCentre = [SIMD2<Float>](repeating: .zero, count: order.count)
        var dLdConic = [SIMD3<Float>](repeating: .zero, count: order.count)
        var dLdOpacity = [Float](repeating: 0, count: order.count)
        for entry in 0..<tileIndices.count {
            let slot = Int(tileIndices[entry])
            guard slot < order.count else { continue }
            let base = entry * slots
            dLdCentre[slot] += SIMD2<Float>(ptr[base], ptr[base + 1])
            dLdConic[slot] += SIMD3<Float>(ptr[base + 2], ptr[base + 3], ptr[base + 4])
            dLdOpacity[slot] += ptr[base + 5]
            let splatIndex = projected[order[slot]].index
            gradients.colors[splatIndex] += SIMD3<Float>(ptr[base + 6], ptr[base + 7], ptr[base + 8])
        }

        // Screen space -> parameter space, reusing the verified CPU chain.
        SplatBackward.applyScreenGradients(
            cloud: cloud, pose: pose, intrinsics: intrinsics,
            projected: order.map { projected[$0] },
            dLdCentre: dLdCentre, dLdConic: dLdConic, dLdOpacity: dLdOpacity,
            into: &gradients)
        return (loss, gradients)
    }

    private struct BackwardParams {
        var width: UInt32
        var height: UInt32
        var tilesX: UInt32
        var tileSize: UInt32
        var transmittanceCutoff: Float
    }
}
