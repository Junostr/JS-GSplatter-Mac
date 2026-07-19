import Foundation

// MARK: - Stage 5: the training loop
//
// Ties the splatting pieces into one driver: forward/backward each view,
// accumulate gradients, Adam step, periodic density control, checkpointing.
//
// The forward and backward passes go through a tier-selecting shim so the loop
// itself is identical whether the GPU kernels are present or not — the Metal
// path when a device is available, the verified CPU reference otherwise. This
// is the same shape as every other stage: one loop, two conformant backends,
// selected once.

/// One camera's contribution to a training step: pose, intrinsics, and its
/// reference image at render resolution (planar RGB float).
public struct TrainingView {
    public let frameIndex: Int
    public let pose: CameraPose
    public let intrinsics: CameraIntrinsics
    public let reference: [Float]
    public let width: Int
    public let height: Int

    public init(frameIndex: Int, pose: CameraPose, intrinsics: CameraIntrinsics,
                reference: [Float], width: Int, height: Int) {
        self.frameIndex = frameIndex
        self.pose = pose
        self.intrinsics = intrinsics
        self.reference = reference
        self.width = width
        self.height = height
    }
}

/// Selects the GPU backward path when a Metal device exists, the CPU reference
/// otherwise. Both were verified to agree to ~1e-8, so the loop is oblivious
/// to which one ran.
public final class SplatGradientBackend {
    public let usesMetal: Bool
    public let descriptionForLog: String
    private let metal: MetalSplatBackward?

    public init(forceCPU: Bool = false) {
        if !forceCPU, let metal = try? MetalSplatBackward() {
            self.metal = metal
            self.usesMetal = true
            self.descriptionForLog = metal.descriptionForLog
        } else {
            self.metal = nil
            self.usesMetal = false
            self.descriptionForLog = "CPU splat backward (reference)"
        }
    }

    public func lossAndGradients(
        cloud: SplatCloud, pose: CameraPose, intrinsics: CameraIntrinsics,
        width: Int, height: Int, reference: [Float], background: SIMD3<Float>,
        options: RasterizerOptions
    ) -> (loss: Double, gradients: SplatGradients) {
        if let metal = metal,
           let result = try? metal.lossAndGradients(
               cloud: cloud, pose: pose, intrinsics: intrinsics,
               width: width, height: height, reference: reference,
               background: background, options: options) {
            return result
        }
        // Fall through to the reference if Metal is absent OR throws mid-run —
        // a transient device error should degrade, not end training.
        return SplatBackward.lossAndGradients(
            cloud: cloud, pose: pose, intrinsics: intrinsics,
            width: width, height: height, reference: reference,
            background: background, options: options)
    }
}

public struct TrainerOptions {
    public var optimizer: OptimizerOptions
    public var density: DensityOptions
    /// Run density control every N iterations, within the densify window.
    public var densifyInterval: Int
    /// Do not densify before this iteration (early gradients are noise) or
    /// within this many of the end (new splats need time to settle).
    public var densifyStart: Int
    public var densifyEnd: Int
    /// Reset opacity every N iterations, 0 to disable.
    public var opacityResetInterval: Int
    public var background: SIMD3<Float>

    public init(optimizer: OptimizerOptions = OptimizerOptions(),
                density: DensityOptions = DensityOptions(),
                densifyInterval: Int = 50, densifyStart: Int = 50,
                densifyEnd: Int = 0, opacityResetInterval: Int = 0,
                background: SIMD3<Float> = SIMD3<Float>(repeating: 0.05)) {
        self.optimizer = optimizer
        self.density = density
        self.densifyInterval = densifyInterval
        self.densifyStart = densifyStart
        self.densifyEnd = densifyEnd
        self.opacityResetInterval = opacityResetInterval
        self.background = background
    }
}

public struct TrainerIterationReport {
    public let iteration: Int
    public let loss: Double
    public let splatCount: Int
    public let density: DensityReport?
}

public final class SplatTrainer {

    public private(set) var cloud: SplatCloud
    public let backend: SplatGradientBackend
    private let optimizer: SplatOptimizer
    private let views: [TrainingView]
    private let options: TrainerOptions
    private let sceneExtent: Float
    private let rasterOptions = RasterizerOptions()
    public private(set) var iteration: Int = 0

    public init(cloud: SplatCloud, views: [TrainingView], options: TrainerOptions = TrainerOptions(),
                forceCPU: Bool = false) {
        self.cloud = cloud
        self.views = views
        self.options = options
        self.sceneExtent = SplatOptimizer.sceneExtent(of: cloud)
        var opts = options.optimizer
        opts.scaleToScene(extent: sceneExtent)
        self.optimizer = SplatOptimizer(splatCount: cloud.count, options: opts)
        self.backend = SplatGradientBackend(forceCPU: forceCPU)
    }

    /// Run one iteration over all views. Returns the mean loss and any density
    /// change, so the caller can log or checkpoint without the loop knowing how.
    @discardableResult
    public func step() -> TrainerIterationReport {
        iteration += 1
        var accumulated = SplatGradients(count: cloud.count)
        var totalLoss = 0.0
        for view in views {
            let (loss, gradients) = backend.lossAndGradients(
                cloud: cloud, pose: view.pose, intrinsics: view.intrinsics,
                width: view.width, height: view.height, reference: view.reference,
                background: options.background, options: rasterOptions)
            totalLoss += loss
            accumulated.add(gradients)
        }
        totalLoss /= Double(max(views.count, 1))

        optimizer.apply(gradients: accumulated, to: &cloud)

        var densityReport: DensityReport?
        let end = options.densifyEnd > 0 ? options.densifyEnd : Int.max
        if iteration % options.densifyInterval == 0,
           iteration >= options.densifyStart, iteration < end {
            densityReport = optimizer.densifyAndPrune(
                cloud: &cloud, gradients: accumulated,
                sceneExtent: sceneExtent, options: options.density)
        }
        if options.opacityResetInterval > 0, iteration % options.opacityResetInterval == 0 {
            optimizer.resetOpacity(cloud: &cloud)
        }

        return TrainerIterationReport(iteration: iteration, loss: totalLoss,
                                      splatCount: cloud.count, density: densityReport)
    }

    /// Render the current cloud from a view — for previews and checkpoint
    /// thumbnails.
    public func render(view: TrainingView) -> RenderTarget {
        SplatRasterizer.render(cloud: cloud, pose: view.pose, intrinsics: view.intrinsics,
                               width: view.width, height: view.height,
                               background: options.background, options: rasterOptions)
    }
}
