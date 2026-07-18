import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import SplatCore
import UniformTypeIdentifiers

// Self-test executable instead of an XCTest/Swift Testing target: this
// machine builds with Command Line Tools only, which ship neither test
// framework, and pulling in swift-testing as a package dependency would
// violate the no-new-dependencies-without-check-in policy. The assertions
// are 1:1 portable to XCTest once full Xcode is in play.
//
// Run with: swift run selftest   (exit code 0 = all passed)

var failures = 0
var passed = 0

func expect(_ condition: Bool, _ label: String, file: String = #file, line: Int = #line) {
    if condition {
        passed += 1
        print("  ok  \(label)")
    } else {
        failures += 1
        print("FAIL  \(label)  (\((file as NSString).lastPathComponent):\(line))")
    }
}

func makeProbe(os: (Int, Int, Int), arch: ProcessArchitecture,
               translated: Bool = false, gpus: [GPUInfo]) -> SystemProbe {
    SystemProbe(osVersion: OSVersion(os.0, os.1, os.2), architecture: arch,
                isTranslated: translated, gpus: gpus)
}

// MARK: Synthetic hardware

let gt750M = GPUInfo(
    name: "NVIDIA GeForce GT 750M", vendor: .nvidia,
    recommendedWorkingSetBytes: 1_610_612_736, // ~1.5 GiB advisory on a 2 GB card
    hasUnifiedMemory: false, isLowPower: false, isRemovable: false, isHeadless: false
)
let irisPro = GPUInfo(
    name: "Intel Iris Pro Graphics", vendor: .intel,
    recommendedWorkingSetBytes: 1_610_612_736,
    hasUnifiedMemory: true, isLowPower: true, isRemovable: false, isHeadless: false
)
let radeonPro5500M = GPUInfo(
    name: "AMD Radeon Pro 5500M", vendor: .amd,
    recommendedWorkingSetBytes: 4_294_967_296,
    hasUnifiedMemory: false, isLowPower: false, isRemovable: false, isHeadless: false
)
let m1Max = GPUInfo(
    name: "Apple M1 Max", vendor: .apple,
    recommendedWorkingSetBytes: 21_474_836_480,
    hasUnifiedMemory: true, isLowPower: false, isRemovable: false, isHeadless: false
)

// MARK: The reference legacy machine (2014 15" MBP: Iris Pro + GT 750M, Big Sur)

print("Legacy Nvidia on Big Sur:")
do {
    let probe = makeProbe(os: (11, 7, 10), arch: .x86_64, gpus: [irisPro, gt750M])
    let d = TierSelector.decide(probe: probe, enhancedCompiledIn: false)
    expect(d.tier == .baseline(.legacyNVIDIA), "picks legacyNVIDIA sub-tier")
    expect(d.selectedGPU?.name == "NVIDIA GeForce GT 750M", "picks discrete card over integrated")
    expect(d.parameters.computePrecision == .fp32, "Kepler computes in fp32")
    expect(d.parameters.storagePrecision == .fp16, "fp16 storage to halve VRAM traffic")
    expect(d.parameters.gpuMemoryBudgetBytes < gt750M.recommendedWorkingSetBytes,
           "budget fits inside the 2 GB card's working set")
}

// MARK: Enhanced gating

print("Enhanced-tier gates:")
do {
    let modern = makeProbe(os: (14, 4, 0), arch: .arm64, gpus: [m1Max])
    expect(TierSelector.decide(probe: modern, enhancedCompiledIn: true).tier == .enhanced,
           "Apple Silicon + modern macOS → enhanced")

    let forced = TierSelector.decide(probe: modern, forceBaseline: true, enhancedCompiledIn: true)
    expect(forced.tier == .baseline(.appleSilicon), "--force-baseline overrides enhanced")
    expect(forced.reasons.contains { $0.contains("force-baseline") }, "override is stated in reasons")

    let bigSurM1 = makeProbe(os: (11, 6, 0), arch: .arm64, gpus: [m1Max])
    let d2 = TierSelector.decide(probe: bigSurM1, enhancedCompiledIn: true)
    expect(d2.tier == .baseline(.appleSilicon), "M1 on Big Sur fails availability gate → baseline")
    expect(d2.reasons.contains { $0.contains("availability gate") }, "availability gate stated in reasons")

    let atFloor = makeProbe(os: (13, 3, 0), arch: .arm64, gpus: [m1Max])
    expect(TierSelector.decide(probe: atFloor, enhancedCompiledIn: true).tier == .enhanced,
           "macOS exactly at the floor passes the availability gate")

    let rosetta = makeProbe(os: (14, 4, 0), arch: .x86_64, translated: true, gpus: [m1Max])
    let d3 = TierSelector.decide(probe: rosetta, enhancedCompiledIn: false)
    expect(d3.tier == .baseline(.appleSilicon), "Rosetta process stays baseline (arch gate)")
    expect(d3.reasons.contains { $0.contains("Rosetta") }, "Rosetta hint stated in reasons")
}

// MARK: Other baseline sub-tiers

print("Baseline sub-tiers:")
do {
    let amd = makeProbe(os: (12, 6, 0), arch: .x86_64, gpus: [radeonPro5500M])
    expect(TierSelector.decide(probe: amd, enhancedCompiledIn: false).tier == .baseline(.discreteAMD),
           "discrete AMD → discreteAMD")

    let intel = makeProbe(os: (11, 0, 0), arch: .x86_64, gpus: [irisPro])
    expect(TierSelector.decide(probe: intel, enhancedCompiledIn: false).tier == .baseline(.integratedIntel),
           "integrated Intel only → integratedIntel")

    let none = makeProbe(os: (11, 0, 0), arch: .x86_64, gpus: [])
    let d = TierSelector.decide(probe: none, enhancedCompiledIn: false)
    expect(d.tier == .baseline(.cpuFallback), "no Metal device → cpuFallback")
    expect(d.selectedGPU == nil, "cpuFallback has no selected GPU")
}

// MARK: Vendor parsing

print("Vendor detection:")
do {
    expect(HardwareProbe.vendor(fromDeviceName: "NVIDIA GeForce GT 750M") == .nvidia, "NVIDIA")
    expect(HardwareProbe.vendor(fromDeviceName: "AMD Radeon Pro Vega 20") == .amd, "AMD")
    expect(HardwareProbe.vendor(fromDeviceName: "Apple M3 Pro") == .apple, "Apple")
    expect(HardwareProbe.vendor(fromDeviceName: "Intel(R) UHD Graphics 630") == .intel, "Intel UHD")
    expect(HardwareProbe.vendor(fromDeviceName: "Iris Pro Graphics 5200") == .intel, "Iris without 'Intel'")
}

// MARK: Engine factory degradation (runs on the real host hardware)

print("Engine factory:")
do {
    // Feed the factory a decision computed as if enhanced were compiled in.
    // Whatever this host is, the factory must return a working engine —
    // enhanced only if both real gates pass, baseline otherwise.
    let probe = HardwareProbe.run()
    let forcedDecision = TierSelector.decide(probe: probe, enhancedCompiledIn: true)
    let engine = EngineFactory.makeEngine(for: forcedDecision)
    do {
        try engine.prepare(configuration: TrainingConfiguration(parameters: forcedDecision.parameters))
        let result = try engine.step()
        expect(result.iteration == 1, "factory engine runs a step (\(engine.descriptionForLog))")
        engine.teardown()
    } catch {
        expect(false, "factory engine prepare/step threw: \(error)")
    }

    let cpu = CPUFallbackTrainingEngine()
    do {
        _ = try cpu.step()
        expect(false, "step before prepare should throw")
    } catch TrainingEngineError.notPrepared {
        expect(true, "step before prepare throws notPrepared")
    } catch {
        expect(false, "step before prepare threw the wrong error: \(error)")
    }
}

// MARK: - Stage 1: ingestion
// Fixtures are synthesized on the fly (PNGs via ImageIO, an H.264 clip via
// AVAssetWriter) so the real decode paths run without binary test assets.

func makeTestImage(width: Int, height: Int, seed: Int) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    context.setFillColor(CGColor(red: CGFloat(seed % 7) / 7.0, green: 0.4, blue: 0.6, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 10 + seed * 5, y: 10, width: 40, height: 40))
    return context.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    _ = CGImageDestinationFinalize(dest)
}

func writeTestVideo(to url: URL, frames: Int, width: Int, height: Int, fps: Int32) throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
    ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
    ])
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    for i in 0..<frames {
        while !input.isReadyForMoreMediaData { usleep(1000) }
        var maybeBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &maybeBuffer)
        let buffer = maybeBuffer!
        CVPixelBufferLockBaseAddress(buffer, [])
        // Per-frame varying fill so the encoder has real deltas to encode.
        let base = CVPixelBufferGetBaseAddress(buffer)!
        memset(base, Int32(40 + (i * 8) % 200), CVPixelBufferGetDataSize(buffer))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
    }
    input.markAsFinished()
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting { semaphore.signal() }
    semaphore.wait()
    guard writer.status == .completed else {
        throw writer.error ?? IngestionError.decodeFailed("video write failed")
    }
}

let workDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("splatcore-selftest-\(ProcessInfo.processInfo.processIdentifier)")
try? FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: workDir) }

print("Ingestion — source detection:")
do {
    let missing = workDir.appendingPathComponent("nope")
    do {
        _ = try IngestionSource.detect(at: missing)
        expect(false, "missing path should throw")
    } catch is IngestionError {
        expect(true, "missing path throws IngestionError")
    }

    let emptyDir = workDir.appendingPathComponent("empty")
    try? FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
    do {
        _ = try IngestionSource.detect(at: emptyDir)
        expect(false, "folder without images should throw")
    } catch is IngestionError {
        expect(true, "folder without images throws IngestionError")
    }

    let lonePhoto = workDir.appendingPathComponent("lone.png")
    writePNG(makeTestImage(width: 32, height: 32, seed: 0), to: lonePhoto)
    do {
        _ = try IngestionSource.detect(at: lonePhoto)
        expect(false, "single image file should throw")
    } catch is IngestionError {
        expect(true, "single image file throws (folder required)")
    }
}

print("Ingestion — photo folder:")
do {
    let photoDir = workDir.appendingPathComponent("photos")
    try FileManager.default.createDirectory(at: photoDir, withIntermediateDirectories: true)
    // Deliberately unordered creation; img_10 must sort after img_3.
    for name in ["img_10", "img_1", "img_3", "img_2"] {
        writePNG(makeTestImage(width: 320, height: 240, seed: name.count),
                 to: photoDir.appendingPathComponent("\(name).png"))
    }

    let source = try IngestionSource.detect(at: photoDir)
    var names: [String] = []
    let summary = try FrameIngestor.ingest(source) { frame in
        names.append(frame.sourceURL!.deletingPathExtension().lastPathComponent)
        return true
    }
    expect(summary.deliveredFrames == 4, "all 4 photos delivered")
    expect(names == ["img_1", "img_2", "img_3", "img_10"], "numeric name ordering (got \(names))")
    expect(summary.width == 320 && summary.height == 240, "native photo resolution")

    let small = try FrameIngestor.ingest(source, options: IngestionOptions(maxDimension: 160)) { _ in true }
    expect(small.width == 160 && small.height == 120, "maxDimension downsamples with aspect kept (got \(small.width)x\(small.height))")

    var seen = 0
    let stopped = try FrameIngestor.ingest(source) { _ in
        seen += 1
        return seen < 2
    }
    expect(stopped.deliveredFrames == 2, "handler returning false stops ingestion")
}

print("Ingestion — video (AVFoundation/VideoToolbox):")
do {
    let videoURL = workDir.appendingPathComponent("clip.mov")
    try writeTestVideo(to: videoURL, frames: 24, width: 320, height: 240, fps: 12)

    let source = try IngestionSource.detect(at: videoURL)
    var timestamps: [Double] = []
    let summary = try FrameIngestor.ingest(source) { frame in
        timestamps.append(frame.timestamp ?? -1)
        return true
    }
    expect(summary.deliveredFrames == 24, "all 24 frames delivered (got \(summary.deliveredFrames))")
    expect(summary.width == 320 && summary.height == 240, "video resolution matches")
    expect(timestamps == timestamps.sorted(), "timestamps monotonically increasing")
    expect(abs((summary.duration ?? 0) - 2.0) < 0.1, "duration ≈ 2 s (got \(summary.duration ?? -1))")

    let sampled = try FrameIngestor.ingest(source, options: IngestionOptions(maxFrames: 6)) { _ in true }
    expect(sampled.deliveredFrames == 6, "maxFrames subsamples to 6 (got \(sampled.deliveredFrames))")
    expect(sampled.decodedFrames > sampled.deliveredFrames, "subsampling skips delivery, not decode")
} catch {
    expect(false, "video ingestion threw: \(error)")
}

// MARK: - Stage 2: frame filtering

func makeCheckerboard(width: Int, height: Int, cell: Int) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    for y in Swift.stride(from: 0, to: height, by: cell) {
        for x in Swift.stride(from: 0, to: width, by: cell) {
            if ((x / cell) + (y / cell)) % 2 == 0 {
                context.fill(CGRect(x: x, y: y, width: cell, height: cell))
            }
        }
    }
    return context.makeImage()!
}

func makeFlat(width: Int, height: Int, gray: CGFloat) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    context.setFillColor(CGColor(red: gray, green: gray, blue: gray, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
}

/// Smooth horizontal luma ramp. The true Laplacian of a linear ramp is ~0
/// everywhere, so this is the worst case for any quantization in the luma
/// buffer: staircase error from low-precision storage shows up as spurious
/// second differences. Used to measure Metal(half/float) vs CPU(float32)
/// blur-score divergence on exactly the near-uniform frames the blur filter
/// has to rank.
func makeGradient(width: Int, height: Int) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    for x in 0..<width {
        let g = CGFloat(x) / CGFloat(Swift.max(1, width - 1))
        context.setFillColor(CGColor(red: g, green: g, blue: g, alpha: 1))
        context.fill(CGRect(x: x, y: 0, width: 1, height: height))
    }
    return context.makeImage()!
}

/// Left half `left` gray, right half `right` gray. Two mirror images have
/// identical mean luma but very different 8×8 signatures — the case a global
/// brightness check would miss and spatial-composition signatures catch.
func makeHalfSplit(width: Int, height: Int, left: CGFloat, right: CGFloat) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    context.setFillColor(CGColor(red: left, green: left, blue: left, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width / 2, height: height))
    context.setFillColor(CGColor(red: right, green: right, blue: right, alpha: 1))
    context.fill(CGRect(x: width / 2, y: 0, width: width - width / 2, height: height))
    return context.makeImage()!
}

print("Frame analyzers (Metal vs CPU):")
do {
    let analyzerDir = workDir.appendingPathComponent("analyzer")
    try FileManager.default.createDirectory(at: analyzerDir, withIntermediateDirectories: true)
    // Fixtures are name-sorted; indices below follow this order:
    //   0 a_sharp    fine checkerboard (high texture → high blur score)
    //   1 b_flat     flat mid-gray     (no texture  → ~zero blur score)
    //   2 c_dark     flat dark gray    (brightness change vs b_flat)
    //   3 d_leftdark left dark/right light  ┐ mirror pair: equal mean luma,
    //   4 e_rightdark left light/right dark  ┘ opposite composition
    writePNG(makeCheckerboard(width: 256, height: 192, cell: 2), to: analyzerDir.appendingPathComponent("a_sharp.png"))
    writePNG(makeFlat(width: 256, height: 192, gray: 0.5), to: analyzerDir.appendingPathComponent("b_flat.png"))
    writePNG(makeFlat(width: 256, height: 192, gray: 0.15), to: analyzerDir.appendingPathComponent("c_dark.png"))
    writePNG(makeHalfSplit(width: 256, height: 192, left: 0.15, right: 0.85), to: analyzerDir.appendingPathComponent("d_leftdark.png"))
    writePNG(makeHalfSplit(width: 256, height: 192, left: 0.85, right: 0.15), to: analyzerDir.appendingPathComponent("e_rightdark.png"))

    let cpu = CPUFrameAnalyzer()
    let metal = try? MetalFrameAnalyzer()
    if metal == nil {
        print("  --  (no Metal device here — Metal analyzer tests skipped, CPU still verified)")
    }

    var cpuScores: [FrameScore] = []
    var metalScores: [FrameScore] = []
    let source = try IngestionSource.detect(at: analyzerDir)
    try FrameIngestor.ingest(source) { frame in
        cpuScores.append(try cpu.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer))
        if let metal = metal {
            metalScores.append(try metal.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer))
        }
        return true
    }

    // Blur score: texture vs flatness. The checkerboard (0) must dwarf every
    // flat/near-flat frame (1,2), and the half-splits (3,4) sit in between —
    // one interior edge contributes, but far less than a full checkerboard.
    expect(cpuScores[0].blurScore > cpuScores[1].blurScore * 100,
           "CPU: checkerboard scores far sharper than flat (\(cpuScores[0].blurScore) vs \(cpuScores[1].blurScore))")
    expect(cpuScores[0].blurScore > cpuScores[3].blurScore * 10,
           "CPU: full-texture checkerboard beats a single-edge half-split")

    // Signature invariant 1 — a flat frame's fingerprint is internally
    // uniform (this is what the absolute-0.5 assumption should have been).
    let flatSig = cpuScores[1].signature
    let flatSpread = (flatSig.max() ?? 0) - (flatSig.min() ?? 0)
    expect(flatSpread < 0.01, "CPU: flat-gray signature is uniform across all 64 cells (spread \(flatSpread))")
    expect(cpuScores[1].signatureDistance(to: cpuScores[1]) == 0, "CPU: signatureDistance is zero for identical frames")

    // Signature invariant 2 — brightness sensitivity: mid-gray vs dark-gray
    // flats are a clear scene change.
    expect(cpuScores[1].signatureDistance(to: cpuScores[2]) > 0.2,
           "CPU: signature separates a brightness change (\(cpuScores[1].signatureDistance(to: cpuScores[2])))")

    // Signature invariant 3 — spatial composition: the two half-splits have
    // (near) identical MEAN luma but opposite layout, so a mean-only metric
    // would call them equal. The 8×8 signature must not.
    let meanDelta = abs((cpuScores[3].signature.reduce(0, +) - cpuScores[4].signature.reduce(0, +)))
    expect(meanDelta < 1.0, "CPU: half-split mirror pair has near-equal total luma (Δ \(meanDelta))")
    expect(cpuScores[3].signatureDistance(to: cpuScores[4]) > 0.3,
           "CPU: signature separates mirrored composition despite equal mean (\(cpuScores[3].signatureDistance(to: cpuScores[4])))")

    if let metal = metal {
        expect(metalScores[0].blurScore > metalScores[1].blurScore * 100,
               "Metal: checkerboard scores far sharper than flat")
        // Cross-tier agreement across ALL fixtures — the contract that lets
        // the viewer/selector ignore which tier produced a score.
        for i in 0..<cpuScores.count {
            let c = cpuScores[i].blurScore, m = metalScores[i].blurScore
            expect(abs(c - m) <= Swift.max(0.05 * Swift.max(c, m), 1e-6),
                   "blur agreement frame \(i): CPU \(c) vs Metal \(m)")
            let maxSigDiff = zip(cpuScores[i].signature, metalScores[i].signature)
                .map { abs($0 - $1) }.max() ?? 1
            expect(maxSigDiff < 0.02, "signature agreement frame \(i) (max diff \(maxSigDiff))")
        }
        _ = metal.descriptionForLog
    }
}

print("Analyzer robustness — degenerate frame dimensions:")
do {
    // Regression: `1..<(dim-1)` becomes the reversed range `1..<0` when a
    // dimension is 1, which TRAPS in Swift. The CPU tier used to abort the
    // whole process here while Metal returned a score.
    let degenerateDir = workDir.appendingPathComponent("degenerate")
    try FileManager.default.createDirectory(at: degenerateDir, withIntermediateDirectories: true)
    writePNG(makeFlat(width: 64, height: 1, gray: 0.5), to: degenerateDir.appendingPathComponent("a_1tall.png"))
    writePNG(makeFlat(width: 1, height: 64, gray: 0.5), to: degenerateDir.appendingPathComponent("b_1wide.png"))
    writePNG(makeFlat(width: 1, height: 1, gray: 0.5), to: degenerateDir.appendingPathComponent("c_1x1.png"))
    writePNG(makeFlat(width: 2, height: 2, gray: 0.5), to: degenerateDir.appendingPathComponent("d_2x2.png"))

    let cpu = CPUFrameAnalyzer()
    let metal = try? MetalFrameAnalyzer()
    var cpuOK = 0, agree = 0, frames = 0
    let source = try IngestionSource.detect(at: degenerateDir)
    try FrameIngestor.ingest(source) { frame in
        frames += 1
        // Reaching the next line at all is the assertion: pre-fix this trapped.
        let c = try cpu.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer)
        cpuOK += 1
        expect(c.blurScore == 0, "degenerate \(frame.width)x\(frame.height): no interior ⇒ blur 0 (got \(c.blurScore))")
        expect(c.signature.count == 64, "degenerate \(frame.width)x\(frame.height): signature still 64 cells")
        if let metal = metal {
            let m = try metal.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer)
            if m.blurScore == c.blurScore { agree += 1 }
        }
        return true
    }
    expect(frames == 4 && cpuOK == 4, "CPU analyzer survives all degenerate sizes without trapping (\(cpuOK)/4)")
    if metal != nil {
        expect(agree == 4, "Metal and CPU agree on degenerate sizes (\(agree)/4)")
    }
}

print("Tier parity on smooth gradients (luma precision):")
do {
    // The blur filter's whole job is ranking near-uniform frames, so any
    // luma-storage quantization on the GPU shows up HERE, not on textured
    // frames where the real Laplacian swamps it. Measured, not assumed.
    let gradientDir = workDir.appendingPathComponent("gradient")
    try FileManager.default.createDirectory(at: gradientDir, withIntermediateDirectories: true)
    writePNG(makeGradient(width: 512, height: 384), to: gradientDir.appendingPathComponent("a_ramp.png"))

    let cpu = CPUFrameAnalyzer()
    if let metal = try? MetalFrameAnalyzer() {
        let source = try IngestionSource.detect(at: gradientDir)
        try FrameIngestor.ingest(source) { frame in
            let c = try cpu.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer)
            let m = try metal.analyze(index: frame.index, timestamp: nil, pixelBuffer: frame.pixelBuffer)
            let denom = Swift.max(c.blurScore, m.blurScore)
            let relative = denom > 0 ? abs(c.blurScore - m.blurScore) / denom : 0
            print(String(format: "      smooth ramp: CPU %.6g  Metal %.6g  (relative %.4f)", c.blurScore, m.blurScore, relative))
            expect(relative <= 0.05,
                   "smooth-gradient blur agrees within the 5% tier tolerance (relative \(relative))")
            return true
        }
    } else {
        print("  --  (no Metal device — gradient parity check skipped)")
    }
}

print("Frame selector (pure logic):")
do {
    func sig(_ v: Float) -> [Float] { [Float](repeating: v, count: 64) }
    func score(_ i: Int, blur: Double, sig sigValue: Float) -> FrameScore {
        FrameScore(index: i, timestamp: Double(i), blurScore: blur, signature: sig(sigValue))
    }

    // Blur rejection: median 10, factor 0.4 → threshold 4; the 0.1 frame dies.
    let blurry = [score(0, blur: 10, sig: 0.1), score(1, blur: 10, sig: 0.3),
                  score(2, blur: 0.1, sig: 0.5), score(3, blur: 10, sig: 0.7)]
    let r1 = FrameSelector.select(scores: blurry, options: FilterOptions(targetFrameCount: 10))
    expect(r1.rejectedBlurry == 1 && r1.selected.count == 3, "blurry frame rejected relative to median")

    // Dedup: three near-identical frames collapse to the sharpest one.
    let dupes = [score(0, blur: 5, sig: 0.5), score(1, blur: 9, sig: 0.5005),
                 score(2, blur: 6, sig: 0.501), score(3, blur: 7, sig: 0.9)]
    let r2 = FrameSelector.select(scores: dupes, options: FilterOptions(targetFrameCount: 10))
    expect(r2.selected.map { $0.index } == [1, 3],
           "duplicate cluster keeps sharpest member (got \(r2.selected.map { $0.index }))")
    expect(r2.rejectedDuplicates == 2, "two duplicates rejected")

    // Static hold (tripod / video freeze): identical composition, varying
    // sharpness. Must collapse to exactly one frame — the sharpest.
    let hold = (0..<15).map { score($0, blur: Double($0), sig: 0.5) }
    let rHold = FrameSelector.select(scores: hold, options: FilterOptions(targetFrameCount: 100))
    expect(rHold.selected.count == 1, "static hold collapses to one frame (got \(rHold.selected.count))")
    expect(rHold.selected.first?.index == 14, "static hold keeps the sharpest member")

    // Continuous slow drift — the splat-capture case. Each consecutive step
    // is 0.006, BELOW the 0.01 dedup threshold, but drift accumulates. The
    // old previous-frame comparison would have collapsed all 20 to 1; the
    // anchor-based comparison must keep a keyframe every ~0.01 of drift.
    let drift = (0..<20).map { score($0, blur: 10, sig: Float($0) * 0.006) }
    let rDrift = FrameSelector.select(scores: drift, options: FilterOptions(targetFrameCount: 100, dedupMinDistance: 0.01))
    expect(rDrift.selected.count == 10,
           "continuous drift keeps evenly-spaced keyframes, not one (got \(rDrift.selected.count))")
    expect(rDrift.selected == rDrift.selected.sorted { $0.index < $1.index },
           "drift keyframes stay in temporal order")

    // Budget: 20 distinct frames, target 5 → 5 picks spread over the range.
    let many = (0..<20).map { score($0, blur: Double(10 + $0 % 3), sig: Float($0) / 20.0) }
    let r3 = FrameSelector.select(scores: many, options: FilterOptions(targetFrameCount: 5))
    expect(r3.selected.count == 5, "budget respected (got \(r3.selected.count))")
    expect(r3.selected.first!.index < 4 && r3.selected.last!.index >= 16,
           "budgeted picks span the whole sequence")

    // Order-independence: shuffled input gives the same selection.
    let r4 = FrameSelector.select(scores: many.shuffled(), options: FilterOptions(targetFrameCount: 5))
    expect(r4.selected.map { $0.index } == r3.selected.map { $0.index }, "selection is input-order independent")

    // Empty input.
    let r5 = FrameSelector.select(scores: [], options: FilterOptions())
    expect(r5.selected.isEmpty, "empty input yields empty selection")
}

print("\n\(passed) passed, \(failures) failed")
exit(failures == 0 ? 0 : 1)
