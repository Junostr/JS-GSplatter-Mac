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
default:
    fail("Unknown command '\(command)'. Commands: probe, ingest, filter", code: 2)
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
