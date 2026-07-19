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
default:
    fail("Unknown command '\(command)'. Commands: probe, ingest, filter, sfm, features", code: 2)
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
