import Foundation

// MARK: - Stage 5: optimizer and adaptive density control
//
// Adam, plus the clone/split/prune cycle that lets the splat count adapt to the
// scene instead of being fixed at whatever SfM happened to triangulate.
//
// Density control is what makes 3D Gaussian Splatting work at all. SfM gives a
// few thousand sparse points; a detailed scene needs hundreds of thousands of
// Gaussians. No amount of gradient descent on a fixed set can create detail
// where there are no primitives, so the optimiser has to be allowed to add and
// remove them.

/// Adam moment state for one parameter group.
///
/// Adam rather than plain SGD because the parameters have wildly different
/// natural scales and curvatures — a colour channel lives in [0,1] while a
/// position spans the whole scene — and Adam's per-parameter normalisation is
/// what lets a single learning rate per group work at all.
struct AdamState {
    var m: [Float]
    var v: [Float]

    init(count: Int) {
        m = Array(repeating: 0, count: count)
        v = Array(repeating: 0, count: count)
    }

    mutating func resize(to count: Int) {
        if m.count < count {
            m.append(contentsOf: Array(repeating: 0, count: count - m.count))
            v.append(contentsOf: Array(repeating: 0, count: count - v.count))
        } else if m.count > count {
            m.removeLast(m.count - count)
            v.removeLast(v.count - count)
        }
    }

    /// Keep only the entries at `indices`, in order. Used when pruning, so the
    /// moment state stays aligned with the splats it belongs to.
    mutating func keep(_ indices: [Int]) {
        m = indices.map { $0 < m.count ? m[$0] : 0 }
        v = indices.map { $0 < v.count ? v[$0] : 0 }
    }
}

public struct OptimizerOptions {
    /// Per-group learning rates. These differ by orders of magnitude because
    /// the parameters do: moving a splat across the scene and nudging its
    /// colour are not comparable operations, and a single rate would either
    /// freeze position or make colour diverge.
    public var positionLR: Float
    public var scaleLR: Float
    public var rotationLR: Float
    public var opacityLR: Float
    public var colorLR: Float
    public var beta1: Float
    public var beta2: Float
    public var epsilon: Float

    public init(positionLR: Float = 0.00016, scaleLR: Float = 0.005,
                rotationLR: Float = 0.001, opacityLR: Float = 0.05,
                colorLR: Float = 0.0025, beta1: Float = 0.9,
                beta2: Float = 0.999, epsilon: Float = 1e-15) {
        self.positionLR = positionLR
        self.scaleLR = scaleLR
        self.rotationLR = rotationLR
        self.opacityLR = opacityLR
        self.colorLR = colorLR
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    /// Scale the position learning rate to the scene's size.
    ///
    /// Positions are in whatever units SfM produced, and that scale is
    /// arbitrary — two views fix geometry only up to a similarity. A fixed
    /// position learning rate is therefore meaningless on its own: the same
    /// value is a crawl in one reconstruction and a catastrophe in another.
    public mutating func scaleToScene(extent: Float) {
        positionLR *= max(extent, 1e-6)
    }
}

public struct DensityOptions {
    /// Fraction of visible splats to densify each pass, chosen by ranking
    /// their average screen-space gradient.
    ///
    /// A PERCENTILE rather than an absolute gradient threshold, because an
    /// absolute one is not portable across loss normalisations. The published
    /// 3DGS value (0.0002) assumes their loss scaling; this pipeline averages
    /// L1 over every pixel and channel, so at 240x135 the gradients are ~1e-5
    /// and that threshold is never reached. Measured consequence: 0 clones and
    /// 0 splits across an entire 200-iteration run, so the splat count could
    /// only shrink and no detail could ever be added — density control present
    /// but silently inert.
    ///
    /// Ranking is invariant to that scaling: the top few percent of splats by
    /// gradient are the under-reconstructed ones whatever the units.
    public var densifyFraction: Float
    /// Absolute floor, so a converged scene with uniformly tiny gradients is
    /// not densified purely because some splat has to be in the top percentile.
    public var gradientThreshold: Float
    /// World-space size, as a fraction of scene extent, separating "too small,
    /// clone it" from "too large, split it".
    public var sizeThreshold: Float
    /// Splats below this opacity are removed — they contribute nothing visible
    /// but still cost memory and blending work.
    public var minOpacity: Float
    /// Splats larger than this fraction of the scene extent are removed. These
    /// are usually degenerate: one enormous Gaussian smeared over the whole
    /// scene can lower the loss slightly while destroying all detail.
    public var maxWorldSize: Float
    /// How far apart split children are placed, as a fraction of the parent's
    /// scale.
    public var splitSeparation: Float
    /// Children of a split are shrunk by this factor. 1.6 is the 3DGS value:
    /// two Gaussians at ~62% the parent's size cover a similar footprint
    /// together while each resolving finer detail.
    public var splitScaleDivisor: Float

    public init(densifyFraction: Float = 0.05, gradientThreshold: Float = 1e-9,
                sizeThreshold: Float = 0.01,
                minOpacity: Float = 0.005, maxWorldSize: Float = 0.1,
                splitSeparation: Float = 1.0, splitScaleDivisor: Float = 1.6) {
        self.densifyFraction = densifyFraction
        self.gradientThreshold = gradientThreshold
        self.sizeThreshold = sizeThreshold
        self.minOpacity = minOpacity
        self.maxWorldSize = maxWorldSize
        self.splitSeparation = splitSeparation
        self.splitScaleDivisor = splitScaleDivisor
    }
}

public struct DensityReport: Equatable {
    public let cloned: Int
    public let split: Int
    public let pruned: Int
    public let finalCount: Int
}

public final class SplatOptimizer {

    public private(set) var options: OptimizerOptions
    public private(set) var step: Int = 0

    private var positionState: AdamState
    private var scaleState: AdamState
    private var rotationState: AdamState
    private var opacityState: AdamState
    private var colorState: AdamState
    private var rng: SplitMix64

    public init(splatCount: Int, options: OptimizerOptions = OptimizerOptions(), seed: UInt64 = 0x5DEECE66D) {
        self.options = options
        // Vector groups hold 3 or 4 scalars per splat; Adam state is per scalar.
        positionState = AdamState(count: splatCount * 3)
        scaleState = AdamState(count: splatCount * 3)
        rotationState = AdamState(count: splatCount * 4)
        opacityState = AdamState(count: splatCount)
        colorState = AdamState(count: splatCount * 3)
        rng = SplitMix64(seed: seed)
    }

    /// One Adam update over every parameter group.
    public func apply(gradients: SplatGradients, to cloud: inout SplatCloud) {
        guard cloud.count > 0 else { return }
        step += 1
        syncCapacity(to: cloud.count)

        // Bias correction. Adam's moments start at zero, so early steps are
        // biased toward zero; dividing by (1 - beta^t) removes exactly that.
        let correction1 = 1 - powf(options.beta1, Float(step))
        let correction2 = 1 - powf(options.beta2, Float(step))

        func update(_ state: inout AdamState, index: Int, gradient: Float, lr: Float) -> Float {
            state.m[index] = options.beta1 * state.m[index] + (1 - options.beta1) * gradient
            state.v[index] = options.beta2 * state.v[index] + (1 - options.beta2) * gradient * gradient
            let mHat = state.m[index] / correction1
            let vHat = state.v[index] / correction2
            return -lr * mHat / (vHat.squareRoot() + options.epsilon)
        }

        for i in 0..<cloud.count {
            for axis in 0..<3 {
                let k = i * 3 + axis
                cloud.positions[i][axis] += update(&positionState, index: k,
                                                   gradient: gradients.positions[i][axis], lr: options.positionLR)
                cloud.logScales[i][axis] += update(&scaleState, index: k,
                                                   gradient: gradients.logScales[i][axis], lr: options.scaleLR)
                cloud.colors[i][axis] += update(&colorState, index: k,
                                                gradient: gradients.colors[i][axis], lr: options.colorLR)
            }
            for axis in 0..<4 {
                let k = i * 4 + axis
                cloud.rotations[i][axis] += update(&rotationState, index: k,
                                                   gradient: gradients.rotations[i][axis], lr: options.rotationLR)
            }
            cloud.opacityLogits[i] += update(&opacityState, index: i,
                                             gradient: gradients.opacityLogits[i], lr: options.opacityLR)
            // Renormalise so the quaternion cannot drift far from unit length.
            // The rotation matrix normalises on use, so this is housekeeping
            // rather than correctness — but leaving it to grow makes the
            // effective learning rate shrink over time.
            cloud.rotations[i] = SplatMath.normalizeQuaternion(cloud.rotations[i])
            // Colour is a reflectance and cannot be negative; clamping here is
            // cheaper and more stable than optimising a sigmoid of it.
            cloud.colors[i] = SIMD3<Float>(max(0, min(1, cloud.colors[i].x)),
                                           max(0, min(1, cloud.colors[i].y)),
                                           max(0, min(1, cloud.colors[i].z)))
        }
    }

    /// Clone, split and prune, then keep optimizer state aligned with the
    /// surviving splats.
    ///
    /// Order matters: densify first, then prune. Doing it the other way round
    /// would let a splat be pruned for low opacity in the same pass that its
    /// gradient marked it as needing more detail.
    @discardableResult
    public func densifyAndPrune(
        cloud: inout SplatCloud,
        gradients: SplatGradients,
        sceneExtent: Float,
        options densityOptions: DensityOptions = DensityOptions()
    ) -> DensityReport {
        guard cloud.count > 0 else { return DensityReport(cloned: 0, split: 0, pruned: 0, finalCount: 0) }

        let sizeLimit = densityOptions.sizeThreshold * sceneExtent

        // Rank visible splats by average screen gradient and take the top
        // fraction, subject to the absolute floor.
        var ranked: [(index: Int, gradient: Float)] = []
        ranked.reserveCapacity(cloud.count)
        for i in 0..<cloud.count where gradients.visibleCount[i] > 0 {
            let average = gradients.screenGradient[i] / Float(gradients.visibleCount[i])
            if average > densityOptions.gradientThreshold { ranked.append((i, average)) }
        }
        // Sorted by gradient, index breaking ties so densification is
        // deterministic rather than depending on sort stability.
        ranked.sort { $0.gradient != $1.gradient ? $0.gradient > $1.gradient : $0.index < $1.index }
        let densifyCount = min(ranked.count,
                               max(0, Int((Float(ranked.count) * densityOptions.densifyFraction).rounded())))
        let selected = Set(ranked.prefix(densifyCount).map { $0.index })

        var cloned = 0, split = 0
        var doomed = Set<Int>()
        var additions: [Splat] = []

        let originalCount = cloud.count
        for i in 0..<originalCount {
            guard selected.contains(i) else { continue }
            let splat = cloud[i]
            let maxScale = max(splat.scale.x, max(splat.scale.y, splat.scale.z))

            if maxScale <= sizeLimit {
                // CLONE: the splat is small but the image still wants to move
                // it, which means it is under-covering its region. Duplicate it
                // so one copy can migrate while the other holds the position.
                additions.append(splat)
                cloned += 1
            } else {
                // SPLIT: the splat is large and being pulled, meaning it spans
                // detail it cannot represent. Replace it with two smaller
                // children offset along its own principal axes, so together
                // they cover a similar footprint at higher resolution.
                let shrunk = SIMD3<Float>(
                    splat.logScale.x - logf(densityOptions.splitScaleDivisor),
                    splat.logScale.y - logf(densityOptions.splitScaleDivisor),
                    splat.logScale.z - logf(densityOptions.splitScaleDivisor))
                let rotation = SplatMath.rotationMatrix(SplatMath.normalizeQuaternion(splat.rotation))
                for _ in 0..<2 {
                    // Offset sampled from the parent's own distribution, so
                    // children land where the parent actually had mass rather
                    // than in an arbitrary direction.
                    let local = SIMD3<Float>(
                        (rng.nextGaussian()) * splat.scale.x * densityOptions.splitSeparation,
                        (rng.nextGaussian()) * splat.scale.y * densityOptions.splitSeparation,
                        (rng.nextGaussian()) * splat.scale.z * densityOptions.splitSeparation)
                    let world = SIMD3<Float>(
                        rotation[0] * local.x + rotation[1] * local.y + rotation[2] * local.z,
                        rotation[3] * local.x + rotation[4] * local.y + rotation[5] * local.z,
                        rotation[6] * local.x + rotation[7] * local.y + rotation[8] * local.z)
                    additions.append(Splat(position: splat.position + world,
                                           logScale: shrunk, rotation: splat.rotation,
                                           opacityLogit: splat.opacityLogit, color: splat.color))
                }
                doomed.insert(i)     // the parent is replaced by its children
                split += 1
            }
        }

        for splat in additions { cloud.append(splat) }

        // Prune: invisible or degenerate splats.
        let worldSizeLimit = densityOptions.maxWorldSize * sceneExtent
        for i in 0..<cloud.count {
            let splat = cloud[i]
            if splat.opacity < densityOptions.minOpacity { doomed.insert(i); continue }
            let maxScale = max(splat.scale.x, max(splat.scale.y, splat.scale.z))
            if maxScale > worldSizeLimit { doomed.insert(i) }
        }

        let kept = (0..<cloud.count).filter { !doomed.contains($0) }
        let prunedCount = cloud.count - kept.count
        // Never empty the scene: with nothing left there is no gradient and no
        // way to recover, so a too-aggressive threshold would end training
        // permanently rather than just badly.
        if kept.isEmpty {
            syncCapacity(to: cloud.count)
            return DensityReport(cloned: cloned, split: split, pruned: 0, finalCount: cloud.count)
        }

        var compacted = SplatCloud()
        compacted.reserveCapacity(kept.count)
        for i in kept { compacted.append(cloud[i]) }
        cloud = compacted

        // Adam state must follow its parameters. New splats get zero moments —
        // inheriting the parent's would apply a momentum built from a different
        // geometry, and the first few steps would move them in a direction that
        // was only ever right for the splat they replaced.
        resizeState(keeping: kept, originalCount: originalCount + additions.count)
        return DensityReport(cloned: cloned, split: split, pruned: prunedCount, finalCount: cloud.count)
    }

    /// Periodically damp all opacities toward zero.
    ///
    /// Standard 3DGS practice, and it exists to fight a specific failure: large
    /// near-opaque splats close to the camera can hide errors behind them, so
    /// the loss stops pushing on anything they cover. Resetting opacity forces
    /// every splat to re-earn its visibility, and ones that cannot are pruned
    /// on the next pass.
    public func resetOpacity(cloud: inout SplatCloud, to value: Float = 0.01) {
        let logit = logf(value / (1 - value))
        for i in 0..<cloud.count {
            cloud.opacityLogits[i] = min(cloud.opacityLogits[i], logit)
        }
        opacityState = AdamState(count: cloud.count)
    }

    private func syncCapacity(to count: Int) {
        positionState.resize(to: count * 3)
        scaleState.resize(to: count * 3)
        rotationState.resize(to: count * 4)
        opacityState.resize(to: count)
        colorState.resize(to: count * 3)
    }

    private func resizeState(keeping kept: [Int], originalCount: Int) {
        // Expand to cover the appended splats (zeroed), then keep survivors.
        positionState.resize(to: originalCount * 3)
        scaleState.resize(to: originalCount * 3)
        rotationState.resize(to: originalCount * 4)
        opacityState.resize(to: originalCount)
        colorState.resize(to: originalCount * 3)

        var vector3: [Int] = [], vector4: [Int] = []
        vector3.reserveCapacity(kept.count * 3)
        vector4.reserveCapacity(kept.count * 4)
        for i in kept {
            for axis in 0..<3 { vector3.append(i * 3 + axis) }
            for axis in 0..<4 { vector4.append(i * 4 + axis) }
        }
        positionState.keep(vector3)
        scaleState.keep(vector3)
        colorState.keep(vector3)
        rotationState.keep(vector4)
        opacityState.keep(kept)
    }

    /// Rough scene extent: the radius of the splat cloud about its centroid.
    /// Used to scale learning rates and density thresholds into whatever units
    /// this particular reconstruction happens to use.
    public static func sceneExtent(of cloud: SplatCloud) -> Float {
        guard cloud.count > 0 else { return 1 }
        var centroid = SIMD3<Float>.zero
        for p in cloud.positions { centroid += p }
        centroid /= Float(cloud.count)
        var maxDistance: Float = 0
        for p in cloud.positions {
            let d = p - centroid
            maxDistance = max(maxDistance, (d.x * d.x + d.y * d.y + d.z * d.z).squareRoot())
        }
        return max(maxDistance, 1e-6)
    }
}
