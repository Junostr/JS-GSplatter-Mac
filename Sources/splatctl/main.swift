import CoreVideo
import Foundation
import ImageIO
import SplatCore
import UniformTypeIdentifiers

// splatctl — CLI driver for the SplatCore pipeline.
//
// Usage:
//   splatctl [probe] [--force-baseline] [--json]
//   splatctl ingest <photo-folder|video> [--max-frames N] [--max-dim N] [--save <dir>]
//   splatctl filter <photo-folder|video> [--target N] [--cpu] [--save <dir>] [--verbose]

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

func gib(_ bytes: UInt64) -> String {
    String(format: "%.1f GiB", Double(bytes) / Double(1 << 30))
}

var args = Array(CommandLine.arguments.dropFirst())
let command: String
if let first = args.first, !first.hasPrefix("--") {
    command = first
    args.removeFirst()
} else {
    command = "probe"
}

switch command {
case "probe":
    runProbe(args)
case "ingest":
    runIngest(args)
case "filter":
    runFilter(args)
case "sfm":
    runSfM(args)
case "features":
    runFeatures(args)
case "render":
    runRender(args)
case "train":
    runTrain(args)
default:
    fail("Unknown command '\(command)'. Commands: probe, ingest, filter, sfm, features, render, train", code: 2)
}

// MARK: - probe

func runProbe(_ args: [String]) {
    let forceBaseline = args.contains("--force-baseline")
    let jsonOutput = args.contains("--json")
    let unknown = args.filter { $0 != "--force-baseline" && $0 != "--json" }
    if !unknown.isEmpty {
        fail("Unknown arguments: \(unknown.joined(separator: " "))\nUsage: splatctl probe [--force-baseline] [--json]", code: 2)
    }

    let probe = HardwareProbe.run()
    let decision = TierSelector.decide(probe: probe, forceBaseline: forceBaseline)

    if jsonOutput {
        struct Report: Codable {
            let probe: SystemProbe
            let decision: TierDecision
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(Report(probe: probe, decision: decision)),
              let json = String(data: data, encoding: .utf8) else {
            fail("Failed to encode probe report")
        }
        print(json)
        return
    }

    print("=== Capability probe ===")
    print("macOS:        \(probe.osVersion)")
    print("Architecture: \(probe.architecture.rawValue)\(probe.isTranslated ? " (Rosetta 2 translation)" : "")")
    if probe.gpus.isEmpty {
        print("GPUs:         none (Metal unavailable)")
    } else {
        for gpu in probe.gpus {
            var traits: [String] = []
            if gpu.hasUnifiedMemory { traits.append("unified memory") }
            if gpu.isLowPower { traits.append("low power") }
            if gpu.isRemovable { traits.append("removable") }
            if gpu.isHeadless { traits.append("headless") }
            let suffix = traits.isEmpty ? "" : " [\(traits.joined(separator: ", "))]"
            print("GPU:          \(gpu.name) (\(gpu.vendor.rawValue)), working set \(gib(gpu.recommendedWorkingSetBytes))\(suffix)")
        }
    }

    print("\n=== Tier decision ===")
    print("Tier:         \(decision.tier.label)\(forceBaseline ? "  [--force-baseline]" : "")")
    let p = decision.parameters
    print("Parameters:   maxSplats=\(p.maxSplatCount), tile=\(p.tileSize)px, storage=\(p.storagePrecision.rawValue), compute=\(p.computePrecision.rawValue), gpuBudget=\(gib(p.gpuMemoryBudgetBytes))")
    for reason in decision.reasons {
        print("  - \(reason)")
    }

    print("\n=== Engine smoke test ===")
    let engine = EngineFactory.makeEngine(for: decision)
    print("Engine:       \(engine.descriptionForLog)")
    do {
        try engine.prepare(configuration: TrainingConfiguration(parameters: decision.parameters))
        for _ in 0..<3 {
            let result = try engine.step()
            print("  step \(result.iteration): loss=\(String(format: "%.4f", result.loss)), splats=\(result.splatCount)")
        }
        engine.teardown()
        print("Engine stub OK.")
    } catch {
        fail("Engine smoke test failed: \(error)")
    }
}

// MARK: - ingest

func runIngest(_ args: [String]) {
    var inputPath: String?
    var maxFrames: Int?
    var maxDimension: Int?
    var saveDir: String?

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--max-frames":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--max-frames needs a positive integer", code: 2) }
            maxFrames = n
        case "--max-dim":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--max-dim needs a positive integer", code: 2) }
            maxDimension = n
        case "--save":
            saveDir = value(for: arg)
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            guard inputPath == nil else { fail("Multiple inputs given: \(inputPath!), \(arg)", code: 2) }
            inputPath = arg
        }
        i += 1
    }

    guard let inputPath = inputPath else {
        fail("Usage: splatctl ingest <photo-folder|video> [--max-frames N] [--max-dim N] [--save <dir>]", code: 2)
    }
    let inputURL = URL(fileURLWithPath: inputPath)

    let source: IngestionSource
    do {
        source = try IngestionSource.detect(at: inputURL)
    } catch {
        fail("\(error)")
    }

    var saveURL: URL?
    if let saveDir = saveDir {
        let url = URL(fileURLWithPath: saveDir, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        } catch {
            fail("Cannot create output directory \(url.path): \(error.localizedDescription)")
        }
        saveURL = url
    }

    print("=== Ingest ===")
    print("Source:       \(source.frameCountEstimateLabel)")
    if let maxFrames = maxFrames { print("Max frames:   \(maxFrames)") }
    if let maxDimension = maxDimension { print("Max dim:      \(maxDimension)px (photos)") }

    let options = IngestionOptions(maxFrames: maxFrames, maxDimension: maxDimension)
    let start = Date()
    do {
        let summary = try FrameIngestor.ingest(source, options: options) { frame in
            if let saveURL = saveURL {
                let name = String(format: "frame_%05d.jpg", frame.index)
                try writeJPEG(frame, to: saveURL.appendingPathComponent(name))
            }
            return true
        }
        let elapsed = Date().timeIntervalSince(start)
        print("\n=== Summary ===")
        print("Frames:       \(summary.deliveredFrames) delivered / \(summary.decodedFrames) decoded")
        print("Resolution:   \(summary.width)x\(summary.height)")
        if let duration = summary.duration, let fps = summary.nominalFrameRate {
            print("Video:        \(String(format: "%.2f", duration))s @ \(String(format: "%.2f", fps)) fps")
        }
        print("Elapsed:      \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", Double(summary.deliveredFrames) / max(elapsed, 0.001))) frames/s)")
        if let saveURL = saveURL {
            print("Saved to:     \(saveURL.path)")
        }
    } catch {
        fail("Ingestion failed: \(error)")
    }
}

// MARK: - filter

func runFilter(_ args: [String]) {
    var inputPath: String?
    var target = 150
    var forceCPU = false
    var saveDir: String?
    var verbose = false

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--target":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--target needs a positive integer", code: 2) }
            target = n
        case "--cpu":
            forceCPU = true
        case "--save":
            saveDir = value(for: arg)
        case "--verbose":
            verbose = true
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            guard inputPath == nil else { fail("Multiple inputs given: \(inputPath!), \(arg)", code: 2) }
            inputPath = arg
        }
        i += 1
    }
    guard let inputPath = inputPath else {
        fail("Usage: splatctl filter <photo-folder|video> [--target N] [--cpu] [--save <dir>] [--verbose]", code: 2)
    }

    let source: IngestionSource
    do {
        source = try IngestionSource.detect(at: URL(fileURLWithPath: inputPath))
    } catch {
        fail("\(error)")
    }

    var saveURL: URL?
    if let saveDir = saveDir {
        let url = URL(fileURLWithPath: saveDir, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        } catch {
            fail("Cannot create output directory \(url.path): \(error.localizedDescription)")
        }
        saveURL = url
    }

    let analyzer = FrameAnalyzerFactory.make(forceCPU: forceCPU)
    print("=== Filter ===")
    print("Source:       \(source.frameCountEstimateLabel)")
    print("Analyzer:     \(analyzer.descriptionForLog)")
    print("Target:       \(target) frames")

    // Two passes over the source: pass 1 scores every frame (cheap, GPU),
    // pass 2 re-decodes only the selected ones to save them. Keeping every
    // candidate frame in RAM instead would be gigabytes.
    var scores: [FrameScore] = []
    let start = Date()
    do {
        try FrameIngestor.ingest(source) { frame in
            let score = try analyzer.analyze(index: frame.index, timestamp: frame.timestamp, pixelBuffer: frame.pixelBuffer)
            scores.append(score)
            return true
        }
    } catch {
        fail("Analysis failed: \(error)")
    }
    let analyzeElapsed = Date().timeIntervalSince(start)

    let result = FrameSelector.select(scores: scores, options: FilterOptions(targetFrameCount: target))

    if verbose {
        print("\nidx   time      blur          keep")
        let kept = Set(result.selected.map { $0.index })
        for score in scores {
            let time = score.timestamp.map { String(format: "%7.2fs", $0) } ?? "      —"
            print(String(format: "%-5d %@  %12.6g  %@", score.index, time, score.blurScore, kept.contains(score.index) ? "✓" : ""))
        }
    }

    print("\n=== Selection ===")
    print("Analyzed:     \(scores.count) frames in \(String(format: "%.2f", analyzeElapsed))s (\(String(format: "%.1f", Double(scores.count) / max(analyzeElapsed, 0.001))) frames/s)")
    print("Selected:     \(result.selected.count)")
    print("Rejected:     \(result.rejectedBlurry) blurry, \(result.rejectedDuplicates) near-duplicates, \(result.rejectedOverBudget) over budget")

    if let saveURL = saveURL {
        let wanted = Set(result.selected.map { $0.index })
        do {
            try FrameIngestor.ingest(source) { frame in
                if wanted.contains(frame.index) {
                    let name = String(format: "frame_%05d.jpg", frame.index)
                    try writeJPEG(frame, to: saveURL.appendingPathComponent(name))
                }
                return true
            }
            print("Saved:        \(result.selected.count) frames to \(saveURL.path)")
        } catch {
            fail("Saving selected frames failed: \(error)")
        }
    }
}

// MARK: - train (stage 5 end to end)

/// Box-downsample a BGRA pixel buffer straight into planar float RGB.
///
/// Training runs at reduced resolution: the CPU rasterizer costs O(pixels x
/// splats), and full 4K per view per iteration is not a useful place to
/// discover whether the pipeline converges.
func downsampledRGB(_ frame: IngestedFrame, width: Int, height: Int) -> [Float]? {
    let sourceWidth = frame.width, sourceHeight = frame.height
    guard sourceWidth >= width, sourceHeight >= height, width > 0, height > 0 else { return nil }
    CVPixelBufferLockBaseAddress(frame.pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(frame.pixelBuffer) else { return nil }
    let stride = CVPixelBufferGetBytesPerRow(frame.pixelBuffer)
    let bytes = base.assumingMemoryBound(to: UInt8.self)

    var out = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        let y0 = y * sourceHeight / height, y1 = max(y0 + 1, (y + 1) * sourceHeight / height)
        for x in 0..<width {
            let x0 = x * sourceWidth / width, x1 = max(x0 + 1, (x + 1) * sourceWidth / width)
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

func runTrain(_ args: [String]) {
    var inputPath: String?
    var target = 24
    var iterations = 200
    var renderScale = 12
    var saveDir: String?
    var forceCPU = false
    var checkpointPath: String?
    var resumePath: String?

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--target":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--target needs a positive integer", code: 2) }
            target = n
        case "--iterations":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--iterations needs a positive integer", code: 2) }
            iterations = n
        case "--scale":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--scale needs a positive integer", code: 2) }
            renderScale = n
        case "--save": saveDir = value(for: arg)
        case "--cpu": forceCPU = true
        case "--checkpoint": checkpointPath = value(for: arg)
        case "--resume": resumePath = value(for: arg)
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            inputPath = arg
        }
        i += 1
    }
    guard let inputPath = inputPath else {
        fail("Usage: splatctl train <input> [--target N] [--iterations N] [--scale N] [--save dir] [--cpu] [--checkpoint file] [--resume file]", code: 2)
    }

    let source: IngestionSource
    do { source = try IngestionSource.detect(at: URL(fileURLWithPath: inputPath)) }
    catch { fail("\(error)") }

    print("=== Train ===")
    let analyzer = FrameAnalyzerFactory.make()
    let extractor = FeatureExtractorFactory.make()

    var scores: [FrameScore] = []
    do {
        try FrameIngestor.ingest(source) { frame in
            scores.append(try analyzer.analyze(index: frame.index, timestamp: frame.timestamp,
                                               pixelBuffer: frame.pixelBuffer))
            return true
        }
    } catch { fail("Analysis failed: \(error)") }
    let selection = FrameSelector.select(
        scores: scores, options: FilterOptions(targetFrameCount: target, dedupMinDistance: 0.001))
    let wanted = Set(selection.selected.map { $0.index })

    var featureSets: [FeatureSet] = []
    var intrinsicsByFrame: [Int: CameraIntrinsics] = [:]
    var references: [Int: [Float]] = [:]
    var frameWidth = 0, frameHeight = 0
    var renderWidth = 0, renderHeight = 0
    do {
        try FrameIngestor.ingest(source) { frame in
            guard wanted.contains(frame.index) else { return true }
            featureSets.append(try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                                     options: FeatureOptions(maxFeatures: 1500)))
            frameWidth = frame.width; frameHeight = frame.height
            renderWidth = max(1, frame.width / renderScale)
            renderHeight = max(1, frame.height / renderScale)
            intrinsicsByFrame[frame.index] = CameraIntrinsics.guess(width: frame.width, height: frame.height)
            references[frame.index] = downsampledRGB(frame, width: renderWidth, height: renderHeight)
            return true
        }
    } catch { fail("Feature extraction failed: \(error)") }

    if let estimate = FocalEstimation.estimate(featureSets: featureSets,
                                               imageWidth: frameWidth, imageHeight: frameHeight) {
        for key in intrinsicsByFrame.keys { intrinsicsByFrame[key]?.focalLength = estimate.focalLength }
    }
    guard let (reconstruction, report) = StructureFromMotion.reconstruct(
        featureSets: featureSets, intrinsics: intrinsicsByFrame) else { fail("Reconstruction failed.") }
    print("Cameras:      \(report.registeredCameras)/\(report.totalCameras), \(report.points) points")

    var cloud = SplatCloud.fromReconstruction(reconstruction)
    guard cloud.count > 0 else { fail("No splats.") }

    // Resume from a checkpoint if one was given and exists.
    var startIteration = 0
    if let resumePath = resumePath, FileManager.default.fileExists(atPath: resumePath) {
        do {
            let (restored, restoredIteration) = try SplatCheckpoint.read(from: URL(fileURLWithPath: resumePath))
            cloud = restored; startIteration = restoredIteration
            print("Resumed:      \(cloud.count) splats from iteration \(startIteration)")
        } catch { fail("Could not read checkpoint: \(error)") }
    }

    let extent = SplatOptimizer.sceneExtent(of: cloud)
    print("Splats:       \(cloud.count) initial, scene extent \(String(format: "%.2f", extent))")
    print("Resolution:   \(renderWidth) x \(renderHeight) (1/\(renderScale) scale)")

    let trainingViews = reconstruction.cameras.keys.sorted().compactMap { frame -> TrainingView? in
        guard let camera = reconstruction.cameras[frame], let reference = references[frame] else { return nil }
        var scaled = camera.intrinsics
        scaled.focalLength /= Double(renderScale)
        scaled.cx /= Double(renderScale); scaled.cy /= Double(renderScale)
        return TrainingView(frameIndex: frame, pose: camera.pose, intrinsics: scaled,
                            reference: reference, width: renderWidth, height: renderHeight)
    }
    guard !trainingViews.isEmpty else { fail("No views with both a pose and an image.") }
    print("Views:        \(trainingViews.count)")

    var trainerOptions = TrainerOptions()
    trainerOptions.background = SIMD3<Float>(repeating: 0.05)
    trainerOptions.densifyEnd = iterations - 20
    let trainer = SplatTrainer(cloud: cloud, views: trainingViews, options: trainerOptions, forceCPU: forceCPU)
    print("Backend:      \(trainer.backend.descriptionForLog)")

    var outURL: URL?
    if let saveDir = saveDir {
        let url = URL(fileURLWithPath: saveDir, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        outURL = url
    }

    let start = Date()
    var firstLoss = 0.0, lastLoss = 0.0
    for iteration in 1...iterations {
        let report = trainer.step()
        if iteration == 1 { firstLoss = report.loss }
        lastLoss = report.loss
        if let density = report.density {
            print(String(format: "  iter %-5d loss %.5f  (+%d cloned, +%d split, -%d pruned -> %d splats)",
                         startIteration + iteration, report.loss, density.cloned, density.split,
                         density.pruned, density.finalCount))
        } else if iteration % 25 == 0 || iteration == 1 {
            print(String(format: "  iter %-5d loss %.5f  (%d splats)",
                         startIteration + iteration, report.loss, report.splatCount))
        }
        // Periodic checkpoint, so a long run survives interruption.
        if let checkpointPath = checkpointPath, iteration % 100 == 0 {
            try? SplatCheckpoint.write(trainer.cloud, iteration: startIteration + iteration,
                                       to: URL(fileURLWithPath: checkpointPath))
        }
    }

    print("\n=== Result ===")
    print(String(format: "Loss:         %.5f -> %.5f (%.0f%% reduction)",
                 firstLoss, lastLoss, (1 - lastLoss / max(firstLoss, 1e-12)) * 100))
    print("Splats:       \(trainer.cloud.count)")
    print("Elapsed:      \(String(format: "%.1f", Date().timeIntervalSince(start)))s")

    if let checkpointPath = checkpointPath {
        try? SplatCheckpoint.write(trainer.cloud, iteration: startIteration + iterations,
                                   to: URL(fileURLWithPath: checkpointPath))
        print("Checkpoint:   \(checkpointPath)")
    }

    if let outURL = outURL {
        for view in trainingViews.prefix(3) {
            let image = trainer.render(view: view)
            writeRenderPNG(image, to: outURL.appendingPathComponent(String(format: "trained_%05d.png", view.frameIndex)))
            var ref = RenderTarget(width: renderWidth, height: renderHeight)
            ref.pixels = view.reference
            writeRenderPNG(ref, to: outURL.appendingPathComponent(String(format: "reference_%05d.png", view.frameIndex)))
        }
        print("Saved:        renders and references to \(outURL.path)")
    }
}


// MARK: - render (stage 5 forward rasterizer)

/// Reconstruct, initialize splats from the sparse cloud, and render from each
/// registered camera. Proves the stage 3 -> stage 5 handoff end to end: if the
/// poses and the splat projection disagree, the render will not resemble the
/// input frames.
func runRender(_ args: [String]) {
    var inputPath: String?
    var target = 40
    var saveDir: String?

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--target":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--target needs a positive integer", code: 2) }
            target = n
        case "--save": saveDir = value(for: arg)
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            inputPath = arg
        }
        i += 1
    }
    guard let inputPath = inputPath else {
        fail("Usage: splatctl render <photo-folder|video> [--target N] [--save <dir>]", code: 2)
    }

    let source: IngestionSource
    do { source = try IngestionSource.detect(at: URL(fileURLWithPath: inputPath)) }
    catch { fail("\(error)") }

    let analyzer = FrameAnalyzerFactory.make()
    let extractor = FeatureExtractorFactory.make()
    print("=== Render ===")
    print("Source:       \(source.frameCountEstimateLabel)")

    var scores: [FrameScore] = []
    do {
        try FrameIngestor.ingest(source) { frame in
            scores.append(try analyzer.analyze(index: frame.index, timestamp: frame.timestamp,
                                               pixelBuffer: frame.pixelBuffer))
            return true
        }
    } catch { fail("Analysis failed: \(error)") }
    let selection = FrameSelector.select(
        scores: scores, options: FilterOptions(targetFrameCount: target, dedupMinDistance: 0.001))
    let wanted = Set(selection.selected.map { $0.index })

    var featureSets: [FeatureSet] = []
    var intrinsicsByFrame: [Int: CameraIntrinsics] = [:]
    var frameWidth = 0, frameHeight = 0
    do {
        try FrameIngestor.ingest(source) { frame in
            guard wanted.contains(frame.index) else { return true }
            featureSets.append(try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                                     options: FeatureOptions(maxFeatures: 1500)))
            frameWidth = frame.width; frameHeight = frame.height
            intrinsicsByFrame[frame.index] = CameraIntrinsics.guess(width: frame.width, height: frame.height)
            return true
        }
    } catch { fail("Feature extraction failed: \(error)") }

    if let estimate = FocalEstimation.estimate(featureSets: featureSets,
                                               imageWidth: frameWidth, imageHeight: frameHeight) {
        for key in intrinsicsByFrame.keys { intrinsicsByFrame[key]?.focalLength = estimate.focalLength }
        print("Intrinsics:   focal ≈ \(Int(estimate.focalLength)) px (estimated)")
    }

    guard let (reconstruction, report) = StructureFromMotion.reconstruct(
        featureSets: featureSets, intrinsics: intrinsicsByFrame) else {
        fail("Reconstruction failed.")
    }
    print("Cameras:      \(report.registeredCameras)/\(report.totalCameras)")
    print("Points:       \(report.points)")

    let cloud = SplatCloud.fromReconstruction(reconstruction)
    print("Splats:       \(cloud.count) initialized from the sparse cloud")
    guard cloud.count > 0 else { fail("No splats to render.") }

    // Render at a reduced size: this is the CPU reference rasterizer, and 4K
    // per camera is pointless for a sanity check.
    let scale = 8
    let rw = max(1, frameWidth / scale), rh = max(1, frameHeight / scale)
    var outURL: URL?
    if let saveDir = saveDir {
        let url = URL(fileURLWithPath: saveDir, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        outURL = url
    }

    let start = Date()
    var rendered = 0
    for frame in reconstruction.cameras.keys.sorted() {
        guard let camera = reconstruction.cameras[frame] else { continue }
        var intr = camera.intrinsics
        intr.focalLength /= Double(scale); intr.cx /= Double(scale); intr.cy /= Double(scale)
        let image = SplatRasterizer.render(cloud: cloud, pose: camera.pose, intrinsics: intr,
                                           width: rw, height: rh,
                                           background: SIMD3<Float>(repeating: 0.05))
        let covered = image.transmittance.filter { $0 < 0.99 }.count
        print(String(format: "  frame %-5d %d x %d, %.1f%% of pixels covered by splats",
                     frame, rw, rh, 100.0 * Double(covered) / Double(rw * rh)))
        if let outURL = outURL {
            writeRenderPNG(image, to: outURL.appendingPathComponent(String(format: "render_%05d.png", frame)))
        }
        rendered += 1
    }
    print("Rendered:     \(rendered) views in \(String(format: "%.2f", Date().timeIntervalSince(start)))s")
    if let outURL = outURL { print("Saved to:     \(outURL.path)") }
}

func writeRenderPNG(_ image: RenderTarget, to url: URL) {
    guard let ctx = CGContext(data: nil, width: image.width, height: image.height,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                        | CGBitmapInfo.byteOrder32Little.rawValue),
          let data = ctx.data else { return }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    for y in 0..<image.height {
        for x in 0..<image.width {
            let p = image.pixel(x: x, y: y)
            let o = y * ctx.bytesPerRow + x * 4
            bytes[o + 0] = UInt8(max(0, min(255, p.z * 255)))
            bytes[o + 1] = UInt8(max(0, min(255, p.y * 255)))
            bytes[o + 2] = UInt8(max(0, min(255, p.x * 255)))
            bytes[o + 3] = 255
        }
    }
    guard let cg = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(dest, cg, nil)
    _ = CGImageDestinationFinalize(dest)
}

// MARK: - features (visual diagnostic)

/// Render detected keypoints onto frames. Where features actually land decides
/// whether a failed reconstruction is a pipeline bug or an unreconstructable
/// scene — specular highlights and glass reflections produce strong, stable-
/// looking corners that are not attached to any real 3D point.
func runFeatures(_ args: [String]) {
    var inputPath: String?
    var saveDir: String?
    var every = 40

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--save": saveDir = value(for: arg)
        case "--every":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--every needs a positive integer", code: 2) }
            every = n
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            inputPath = arg
        }
        i += 1
    }
    guard let inputPath = inputPath, let saveDir = saveDir else {
        fail("Usage: splatctl features <photo-folder|video> --save <dir> [--every N]", code: 2)
    }
    let outURL = URL(fileURLWithPath: saveDir, isDirectory: true)
    try? FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

    let source: IngestionSource
    do { source = try IngestionSource.detect(at: URL(fileURLWithPath: inputPath)) }
    catch { fail("\(error)") }

    let extractor = FeatureExtractorFactory.make()
    print("Extractor:    \(extractor.descriptionForLog)")
    var written = 0
    do {
        try FrameIngestor.ingest(source) { frame in
            guard frame.index % every == 0, written < 6 else { return true }
            let set = try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                            options: FeatureOptions(maxFeatures: 1200))
            let image = try frame.makeCGImage()
            let w = image.width, h = image.height
            guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                                | CGBitmapInfo.byteOrder32Little.rawValue) else { return true }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            // Keypoints are top-left origin; CGContext is bottom-left.
            let maxResponse = set.keypoints.first?.response ?? 1
            ctx.setLineWidth(max(2, Double(w) / 700))
            for kp in set.keypoints {
                let strength = maxResponse > 0 ? Double(kp.response / maxResponse) : 0
                // Strong corners red, weak ones blue.
                ctx.setStrokeColor(CGColor(red: 0.2 + 0.8 * strength, green: 0.9 - 0.7 * strength,
                                           blue: 1.0 - 0.9 * strength, alpha: 0.95))
                let r = max(4.0, Double(w) / 260)
                ctx.strokeEllipse(in: CGRect(x: Double(kp.x) - r, y: Double(h) - Double(kp.y) - r,
                                             width: r * 2, height: r * 2))
            }
            guard let out = ctx.makeImage() else { return true }
            let url = outURL.appendingPathComponent(String(format: "features_%05d.jpg", frame.index))
            if let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) {
                CGImageDestinationAddImage(dest, out, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
                _ = CGImageDestinationFinalize(dest)
            }
            print("  frame \(frame.index): \(set.count) keypoints → \(url.lastPathComponent)")
            written += 1
            return true
        }
    } catch { fail("Feature visualization failed: \(error)") }
}

// MARK: - sfm

func runSfM(_ args: [String]) {
    var inputPath: String?
    var target = 40
    var forceCPU = false
    var plyPath: String?
    var focalOverride: Double?
    var descriptorKind: DescriptorKind = .sift
    var loopClosure = false

    var i = 0
    while i < args.count {
        let arg = args[i]
        func value(for flag: String) -> String {
            i += 1
            guard i < args.count else { fail("Missing value for \(flag)", code: 2) }
            return args[i]
        }
        switch arg {
        case "--target":
            guard let n = Int(value(for: arg)), n > 0 else { fail("--target needs a positive integer", code: 2) }
            target = n
        case "--focal":
            guard let v = Double(value(for: arg)), v > 0 else { fail("--focal needs a positive number", code: 2) }
            focalOverride = v
        case "--cpu": forceCPU = true
        case "--brief": descriptorKind = .brief
        case "--loop": loopClosure = true
        case "--ply": plyPath = value(for: arg)
        default:
            if arg.hasPrefix("--") { fail("Unknown flag \(arg)", code: 2) }
            guard inputPath == nil else { fail("Multiple inputs given", code: 2) }
            inputPath = arg
        }
        i += 1
    }
    guard let inputPath = inputPath else {
        fail("Usage: splatctl sfm <photo-folder|video> [--target N] [--cpu] [--ply out.ply]", code: 2)
    }

    let source: IngestionSource
    do { source = try IngestionSource.detect(at: URL(fileURLWithPath: inputPath)) }
    catch { fail("\(error)") }

    let analyzer = FrameAnalyzerFactory.make(forceCPU: forceCPU)
    let extractor = FeatureExtractorFactory.make(forceCPU: forceCPU)
    print("=== SfM ===")
    print("Source:       \(source.frameCountEstimateLabel)")
    print("Analyzer:     \(analyzer.descriptionForLog)")
    print("Extractor:    \(extractor.descriptionForLog)")
    print("Descriptor:   \(descriptorKind.rawValue)")

    // Stage 2 first: pose estimation on blurry or duplicate frames is wasted
    // work at best and actively harmful at worst.
    var scores: [FrameScore] = []
    do {
        try FrameIngestor.ingest(source) { frame in
            scores.append(try analyzer.analyze(index: frame.index, timestamp: frame.timestamp,
                                               pixelBuffer: frame.pixelBuffer))
            return true
        }
    } catch { fail("Analysis failed: \(error)") }
    // Finer dedup spacing than the stage-2 default. That default asks "is this
    // frame redundant for viewing?"; SfM asks "do I have enough well-separated
    // views to triangulate?", and wants far more of them. At the default 0.01
    // this scene collapsed 72 frames to 5, leaving only adjacent pairs with no
    // usable baseline.
    let selection = FrameSelector.select(
        scores: scores,
        options: FilterOptions(targetFrameCount: target, dedupMinDistance: 0.001)
    )
    let wanted = Set(selection.selected.map { $0.index })
    print("Filtered:     \(scores.count) → \(selection.selected.count) frames")

    // Stage 3: features on the surviving frames.
    var featureSets: [FeatureSet] = []
    var intrinsicsByFrame: [Int: CameraIntrinsics] = [:]
    var frameWidth = 0, frameHeight = 0
    let start = Date()
    do {
        try FrameIngestor.ingest(source) { frame in
            guard wanted.contains(frame.index) else { return true }
            let set = try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                            options: FeatureOptions(maxFeatures: 1500,
                                                                    descriptorKind: descriptorKind))
            featureSets.append(set)
            frameWidth = frame.width; frameHeight = frame.height
            var intr = CameraIntrinsics.guess(width: frame.width, height: frame.height)
            if let focalOverride = focalOverride { intr.focalLength = focalOverride }
            intrinsicsByFrame[frame.index] = intr
            return true
        }
    } catch { fail("Feature extraction failed: \(error)") }
    let featureElapsed = Date().timeIntervalSince(start)
    let totalFeatures = featureSets.reduce(0) { $0 + $1.count }
    print("Features:     \(totalFeatures) over \(featureSets.count) frames in \(String(format: "%.2f", featureElapsed))s")

    // Estimate focal from the geometry unless overridden. The fixed heuristic
    // is badly wrong for phone cameras and a wrong focal fails confidently
    // rather than obviously -- see FocalEstimation.swift.
    if focalOverride == nil, let first = featureSets.first, first.count > 0 {
        let w = frameWidth, h = frameHeight
        if let estimate = FocalEstimation.estimate(featureSets: featureSets, imageWidth: w, imageHeight: h) {
            let multiplier = estimate.focalLength / Double(max(w, h))
            print("Intrinsics:   focal ≈ \(Int(estimate.focalLength)) px (estimated, \(String(format: "%.2f", multiplier))× long side, "
                  + "\(estimate.supportingPoints) supporting points, median \(String(format: "%.2f", estimate.medianReprojectionError)) px)")
            for key in intrinsicsByFrame.keys { intrinsicsByFrame[key]?.focalLength = estimate.focalLength }
        } else {
            print("Intrinsics:   focal ≈ \(Int(intrinsicsByFrame.values.first?.focalLength ?? 0)) px (guess — estimation failed)")
        }
    } else {
        print("Intrinsics:   focal ≈ \(Int(intrinsicsByFrame.values.first?.focalLength ?? 0)) px \(focalOverride != nil ? "(--focal override)" : "(guessed)")")
    }

    guard featureSets.count >= 2 else { fail("Need at least 2 usable frames, got \(featureSets.count)") }

    let sfmStart = Date()
    guard let (reconstruction, report) = StructureFromMotion.reconstruct(
        featureSets: featureSets, intrinsics: intrinsicsByFrame,
        options: SfMOptions(loopClosure: loopClosure),
        log: { print("  \($0)") }
    ) else {
        fail("Reconstruction failed — not enough parallax or too few matches.")
    }
    let sfmElapsed = Date().timeIntervalSince(sfmStart)

    print("\n=== Reconstruction ===")
    print("Cameras:      \(report.registeredCameras)/\(report.totalCameras) registered")
    print("Points:       \(report.points)")
    print("RMSE:         \(String(format: "%.3f", report.rmseBefore)) → \(String(format: "%.3f", report.rmseAfter)) px")
    print("Elapsed:      \(String(format: "%.2f", sfmElapsed))s")
    if report.registeredCameras < report.totalCameras {
        print("Note:         \(report.totalCameras - report.registeredCameras) frame(s) could not be registered.")
    }

    if let plyPath = plyPath {
        do {
            try writePLY(reconstruction: reconstruction, to: URL(fileURLWithPath: plyPath))
            print("Wrote:        \(plyPath)")
        } catch {
            fail("Could not write PLY: \(error)")
        }
    }
}

/// Minimal ASCII PLY of the sparse cloud plus camera centres. A debugging aid
/// for inspecting a reconstruction in MeshLab/CloudCompare — the real export
/// pipeline is stage 7.
func writePLY(reconstruction: Reconstruction, to url: URL) throws {
    var lines: [String] = []
    let cameraCentres = reconstruction.cameras.keys.sorted().compactMap { reconstruction.cameras[$0]?.pose.center }
    let total = reconstruction.points.count + cameraCentres.count
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append("comment sparse reconstruction from GaussianSplatter stage 3")
    lines.append("element vertex \(total)")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append("property uchar red")
    lines.append("property uchar green")
    lines.append("property uchar blue")
    lines.append("end_header")
    for point in reconstruction.points {
        lines.append("\(point.position.x) \(point.position.y) \(point.position.z) 200 200 200")
    }
    // Cameras in red so the path is visible against the cloud.
    for centre in cameraCentres {
        lines.append("\(centre.x) \(centre.y) \(centre.z) 255 40 40")
    }
    try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
}

func writeJPEG(_ frame: IngestedFrame, to url: URL) throws {
    let image = try frame.makeCGImage()
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.jpeg.identifier as CFString, 1, nil
    ) else {
        throw IngestionError.decodeFailed("cannot create \(url.lastPathComponent)")
    }
    CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
        throw IngestionError.decodeFailed("cannot write \(url.lastPathComponent)")
    }
}
