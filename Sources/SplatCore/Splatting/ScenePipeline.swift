import CoreVideo
import Foundation

// MARK: - Stage 5/6 glue: the whole pipeline as one driver
//
// Ingestion -> filtering -> features -> focal -> SfM -> initial splats, returned
// ready to train. Extracted here so the CLI and the app run the SAME code with
// the same progress and cancellation hooks, rather than each re-orchestrating
// the stages and drifting apart. The training loop itself stays with the caller
// because the two want different things from it (the app renders a live preview
// and publishes to the main thread; the CLI writes checkpoints), but everything
// expensive and order-sensitive before training is shared.

public struct ScenePipelineOptions {
    /// Target frame count after stage-2 selection.
    public var targetFrames: Int
    /// 1/scale render resolution for training references.
    public var renderScale: Int
    public var maxFeatures: Int

    public init(targetFrames: Int = 40, renderScale: Int = 12, maxFeatures: Int = 1500) {
        self.targetFrames = targetFrames
        self.renderScale = renderScale
        self.maxFeatures = maxFeatures
    }
}

/// Coarse progress phases, for a status line and a progress bar. Deliberately
/// not tied to any UI type.
public enum ScenePipelinePhase: Equatable {
    case analyzingFrames(count: Int)
    case selectingFrames
    case extractingFeatures(done: Int, total: Int)
    case estimatingFocal
    case reconstructing
    case preparingSplats(count: Int)
}

public struct PreparedScene {
    public let reconstruction: Reconstruction
    public let views: [TrainingView]
    public let initialCloud: SplatCloud
    public let renderWidth: Int
    public let renderHeight: Int
    public let registeredCameras: Int
    public let totalCameras: Int
}

public enum ScenePipelineError: Error, CustomStringConvertible {
    case cancelled
    case noUsableFrames
    case reconstructionFailed
    case noSplats

    public var description: String {
        switch self {
        case .cancelled: return "Cancelled"
        case .noUsableFrames: return "No usable frames — need at least two with enough overlap"
        case .reconstructionFailed: return "Reconstruction failed — not enough parallax or too few matches"
        case .noSplats: return "Reconstruction produced no points to initialize splats from"
        }
    }
}

public enum ScenePipeline {

    /// Run everything up to (but not including) training, reporting progress and
    /// checking for cancellation at each safe point. `shouldCancel` is polled
    /// often enough that a stop feels immediate everywhere except inside the
    /// single SfM solve, which is not interruptible mid-call.
    public static func prepare(
        source: IngestionSource,
        options: ScenePipelineOptions = ScenePipelineOptions(),
        forceCPU: Bool = false,
        progress: (ScenePipelinePhase) -> Void = { _ in },
        shouldCancel: () -> Bool = { false }
    ) throws -> PreparedScene {

        func checkCancel() throws { if shouldCancel() { throw ScenePipelineError.cancelled } }

        let analyzer = FrameAnalyzerFactory.make(forceCPU: forceCPU)
        let extractor = FeatureExtractorFactory.make(forceCPU: forceCPU)

        // Pass 1 — blur/signature scores for every frame.
        var scores: [FrameScore] = []
        try FrameIngestor.ingest(source) { frame in
            scores.append(try analyzer.analyze(index: frame.index, timestamp: frame.timestamp,
                                               pixelBuffer: frame.pixelBuffer))
            if frame.index % 8 == 0 { progress(.analyzingFrames(count: scores.count)) }
            return !shouldCancel()
        }
        try checkCancel()

        progress(.selectingFrames)
        // Finer dedup than the stage-2 default: SfM wants many well-separated
        // views, not just a non-redundant few (see FrameSelector notes).
        let selection = FrameSelector.select(
            scores: scores,
            options: FilterOptions(targetFrameCount: options.targetFrames, dedupMinDistance: 0.001))
        let wanted = Set(selection.selected.map { $0.index })
        guard wanted.count >= 2 else { throw ScenePipelineError.noUsableFrames }

        // Pass 2 — features on the surviving frames, plus their downsampled
        // reference images for training.
        var featureSets: [FeatureSet] = []
        var intrinsicsByFrame: [Int: CameraIntrinsics] = [:]
        var references: [Int: [Float]] = [:]
        var frameWidth = 0, frameHeight = 0
        var renderWidth = 0, renderHeight = 0
        var extracted = 0
        try FrameIngestor.ingest(source) { frame in
            guard wanted.contains(frame.index) else { return !shouldCancel() }
            featureSets.append(try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                                     options: FeatureOptions(maxFeatures: options.maxFeatures)))
            frameWidth = frame.width; frameHeight = frame.height
            renderWidth = max(1, frame.width / options.renderScale)
            renderHeight = max(1, frame.height / options.renderScale)
            intrinsicsByFrame[frame.index] = CameraIntrinsics.guess(width: frame.width, height: frame.height)
            references[frame.index] = downsampledRGB(frame, width: renderWidth, height: renderHeight)
            extracted += 1
            progress(.extractingFeatures(done: extracted, total: wanted.count))
            return !shouldCancel()
        }
        try checkCancel()
        guard featureSets.count >= 2 else { throw ScenePipelineError.noUsableFrames }

        progress(.estimatingFocal)
        if let estimate = FocalEstimation.estimate(featureSets: featureSets,
                                                   imageWidth: frameWidth, imageHeight: frameHeight) {
            for key in intrinsicsByFrame.keys { intrinsicsByFrame[key]?.focalLength = estimate.focalLength }
        }
        try checkCancel()

        progress(.reconstructing)
        guard let (reconstruction, report) = StructureFromMotion.reconstruct(
            featureSets: featureSets, intrinsics: intrinsicsByFrame) else {
            throw ScenePipelineError.reconstructionFailed
        }
        try checkCancel()

        let cloud = SplatCloud.fromReconstruction(reconstruction)
        guard cloud.count > 0 else { throw ScenePipelineError.noSplats }
        progress(.preparingSplats(count: cloud.count))

        // Views that both registered AND have a reference image, with intrinsics
        // scaled to render resolution.
        let views = reconstruction.cameras.keys.sorted().compactMap { frame -> TrainingView? in
            guard let camera = reconstruction.cameras[frame], let reference = references[frame] else { return nil }
            var scaled = camera.intrinsics
            scaled.focalLength /= Double(options.renderScale)
            scaled.cx /= Double(options.renderScale); scaled.cy /= Double(options.renderScale)
            return TrainingView(frameIndex: frame, pose: camera.pose, intrinsics: scaled,
                                reference: reference, width: renderWidth, height: renderHeight)
        }
        guard !views.isEmpty else { throw ScenePipelineError.noUsableFrames }

        return PreparedScene(
            reconstruction: reconstruction, views: views, initialCloud: cloud,
            renderWidth: renderWidth, renderHeight: renderHeight,
            registeredCameras: report.registeredCameras, totalCameras: report.totalCameras)
    }

    /// Box-downsample a BGRA pixel buffer into planar float RGB at the given
    /// size. Training references are at reduced resolution because the
    /// rasterizer is O(pixels x splats).
    public static func downsampledRGB(_ frame: IngestedFrame, width: Int, height: Int) -> [Float] {
        let sourceWidth = frame.width, sourceHeight = frame.height
        guard sourceWidth >= 1, sourceHeight >= 1, width > 0, height > 0 else {
            return [Float](repeating: 0, count: max(width, 0) * max(height, 0) * 3)
        }
        CVPixelBufferLockBaseAddress(frame.pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(frame.pixelBuffer) else {
            return [Float](repeating: 0, count: width * height * 3)
        }
        let stride = CVPixelBufferGetBytesPerRow(frame.pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        var out = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            let y0 = y * sourceHeight / height
            let y1 = max(y0 + 1, (y + 1) * sourceHeight / height)
            for x in 0..<width {
                let x0 = x * sourceWidth / width
                let x1 = max(x0 + 1, (x + 1) * sourceWidth / width)
                var r = 0.0, g = 0.0, b = 0.0, n = 0.0
                for sy in y0..<y1 {
                    let row = sy * stride
                    for sx in x0..<x1 {
                        b += Double(bytes[row + sx * 4 + 0])
                        g += Double(bytes[row + sx * 4 + 1])
                        r += Double(bytes[row + sx * 4 + 2])
                        n += 1
                    }
                }
                let o = (y * width + x) * 3
                out[o] = Float(r / n / 255); out[o + 1] = Float(g / n / 255); out[o + 2] = Float(b / n / 255)
            }
        }
        return out
    }
}
