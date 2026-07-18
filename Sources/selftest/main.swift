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

// MARK: - Stage 3: feature extraction and matching

/// Scattered bright squares on a dark field: unambiguous corners at known
/// locations, and shifting the whole pattern gives ground-truth correspondence.
func makeCorners(width: Int, height: Int, offsetX: Int, offsetY: Int) -> CGImage {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    context.setFillColor(CGColor(red: 0.08, green: 0.08, blue: 0.1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    // Deterministic pseudo-scatter; varied sizes so responses differ and the
    // strongest-first ordering is actually exercised.
    var seed = 12345
    func nextInt(_ bound: Int) -> Int {
        seed = (seed &* 1103515245 &+ 12345) & 0x7FFF_FFFF
        return seed % bound
    }
    for _ in 0..<40 {
        let x = 30 + nextInt(width - 90) + offsetX
        let y = 30 + nextInt(height - 90) + offsetY
        let size = 8 + nextInt(10)
        let shade = 0.55 + Double(nextInt(45)) / 100.0
        context.setFillColor(CGColor(red: shade, green: shade * 0.95, blue: shade * 0.9, alpha: 1))
        context.fill(CGRect(x: x, y: y, width: size, height: size))
    }
    return context.makeImage()!
}

print("Feature descriptor pattern (determinism):")
do {
    // The BRIEF pattern must be identical across runs and machines, or
    // descriptors extracted today cannot match ones extracted yesterday.
    expect(FeatureMath.pattern.count == 256, "pattern has 256 tests (got \(FeatureMath.pattern.count))")
    let allInPatch = FeatureMath.pattern.allSatisfy { t in
        let r = Int8(FeatureMath.patchRadius)
        return t.dx1 >= -r && t.dx1 <= r && t.dy1 >= -r && t.dy1 <= r
            && t.dx2 >= -r && t.dx2 <= r && t.dy2 >= -r && t.dy2 <= r
    }
    expect(allInPatch, "every sampling point lies inside the patch radius")
    let degenerate = FeatureMath.pattern.contains { $0.dx1 == $0.dx2 && $0.dy1 == $0.dy2 }
    expect(!degenerate, "no test compares a pixel with itself")
    // Regenerating from the same fixed seed must reproduce the pattern.
    var rng = SplitMix64(seed: 0x5F3A_9C21_7E44_B0D9)
    let firstDraw = rng.next()
    var rng2 = SplitMix64(seed: 0x5F3A_9C21_7E44_B0D9)
    expect(rng2.next() == firstDraw, "SplitMix64 is reproducible from a fixed seed")
}

print("Feature extraction (Metal vs CPU parity):")
do {
    let featureDir = workDir.appendingPathComponent("features")
    try FileManager.default.createDirectory(at: featureDir, withIntermediateDirectories: true)
    writePNG(makeCorners(width: 320, height: 240, offsetX: 0, offsetY: 0),
             to: featureDir.appendingPathComponent("a_base.png"))

    let cpu = CPUFeatureExtractor()
    let metal = try? MetalFeatureExtractor()
    if metal == nil { print("  --  (no Metal device — GPU extractor tests skipped)") }

    let source = try IngestionSource.detect(at: featureDir)
    try FrameIngestor.ingest(source) { frame in
        let options = FeatureOptions(maxFeatures: 500)
        let c = try cpu.extract(index: frame.index, pixelBuffer: frame.pixelBuffer, options: options)
        expect(c.count > 20, "CPU finds a meaningful number of corners (got \(c.count))")
        expect(c.descriptors.count == c.count * FeatureSet.descriptorBytes,
               "descriptor buffer size matches keypoint count")
        // Responses arrive strongest-first, but only up to the sort's
        // quantization grid: within one bucket the order is spatial, so the
        // raw response may tick upward by less than one grid step. Asserting
        // strict monotonicity would be asserting the bug the quantization
        // deliberately fixes.
        let responses = c.keypoints.map { $0.response }
        let quantStep = (responses.first ?? 0) / Float(1 << 20)
        let monotonic = zip(responses, responses.dropFirst()).allSatisfy { $1 <= $0 + quantStep * 2 }
        expect(monotonic, "keypoints are ordered strongest-first (within the sort quantization step)")
        expect(responses.first == responses.max(), "the first keypoint is the global maximum")

        if let metal = metal {
            let m = try metal.extract(index: frame.index, pixelBuffer: frame.pixelBuffer, options: options)
            expect(m.count == c.count, "same keypoint count across tiers (CPU \(c.count), Metal \(m.count))")
            // The contract is that the two tiers produce INTERCHANGEABLE
            // feature sets, matched by position — not bit-identical array
            // ordering. The response maps legitimately differ by ~1 ULP
            // (see the sort-quantization note in FeatureExtraction.swift),
            // and demanding identical ordering across heterogeneous GPUs
            // would be a promise we cannot keep on real Nvidia/AMD hardware.
            // NMS guarantees unique positions, but uniquing defensively keeps
            // a duplicate from trapping the whole suite.
            // Keyed by position AND octave: with a pyramid the same
            // full-resolution coordinate can be produced by two different
            // levels, and those are genuinely different features with
            // different descriptors.
            let cByPos = Dictionary(c.keypoints.enumerated().map { ("\($1.x)|\($1.y)|\($1.octave)", $0) },
                                    uniquingKeysWith: { first, _ in first })
            let mByPos = Dictionary(m.keypoints.enumerated().map { ("\($1.x)|\($1.y)|\($1.octave)", $0) },
                                    uniquingKeysWith: { first, _ in first })
            // Near-exact rather than exact. The tiers' response maps differ by
            // ~1 ULP, and a pyramid multiplies the chances that some corner
            // sits exactly on a threshold or NMS boundary and tips one way on
            // one tier. Demanding a perfect set match would be asserting a
            // promise the hardware cannot keep; what matters is that the
            // feature sets are interchangeable.
            let shared = Set(cByPos.keys).intersection(mByPos.keys).count
            let overlap = Double(shared) / Double(Swift.max(c.count, m.count))
            expect(overlap >= 0.97,
                   "keypoint sets agree across tiers (\(shared)/\(c.count) shared, \(Int(overlap * 100))%)")

            // For every shared keypoint, orientation and descriptor must match
            // exactly — these are computed by shared code from the same luma,
            // so any difference here would be a real defect.
            var maxAngleDiff: Float = 0
            var descriptorMismatches = 0
            var worstDescriptorBits = 0
            for (key, ci) in cByPos {
                guard let mi = mByPos[key] else { continue }
                maxAngleDiff = Swift.max(maxAngleDiff, abs(c.keypoints[ci].angle - m.keypoints[mi].angle))
                let cd = Array(c.descriptor(at: ci)), md = Array(m.descriptor(at: mi))
                if cd != md {
                    descriptorMismatches += 1
                    worstDescriptorBits = Swift.max(worstDescriptorBits,
                        zip(cd, md).reduce(0) { $0 + ($1.0 ^ $1.1).nonzeroBitCount })
                }
            }
            expect(maxAngleDiff < 1e-4, "identical orientations for shared keypoints (max Δ \(maxAngleDiff))")
            // Descriptors are near-identical rather than always bit-identical.
            // Steered BRIEF rounds each rotated sample to a pixel, so an angle
            // difference of ~5e-7 (which is what the tiers' ~1 ULP response
            // difference produces) can move a single sample across a rounding
            // boundary and flip one bit. What matters for matching is that the
            // Hamming distance stays negligible against the 80-bit accept
            // threshold — a 1-2 bit difference out of 256 cannot change a match.
            let mismatchFraction = Double(descriptorMismatches) / Double(Swift.max(1, cByPos.count))
            expect(mismatchFraction < 0.02,
                   "descriptors agree for shared keypoints (\(descriptorMismatches)/\(cByPos.count) differ)")
            expect(worstDescriptorBits <= 4,
                   "any descriptor difference is negligible vs the match threshold (worst \(worstDescriptorBits)/256 bits)")
            expect(c.keypoints.allSatisfy { $0.octave >= 0 && $0.scale == Float(1 << $0.octave) },
                   "octave and scale are consistent")
            expect(Set(c.keypoints.map { $0.octave }).count > 1, "features come from multiple pyramid levels")

            // Ordering is quantization-stabilized, so it should also agree in
            // practice; report it rather than asserting it, since it is a
            // best-effort property across GPU vendors, not a guarantee.
            let sameOrder = zip(c.keypoints, m.keypoints).filter { $0.x == $1.x && $0.y == $1.y }.count
            print("      order agreement: \(sameOrder)/\(c.count) positions in identical rank")
        }
        return true
    }
}

print("Feature matching (recovers a known translation):")
do {
    let matchDir = workDir.appendingPathComponent("matching")
    try FileManager.default.createDirectory(at: matchDir, withIntermediateDirectories: true)
    let shift = 12
    writePNG(makeCorners(width: 320, height: 240, offsetX: 0, offsetY: 0),
             to: matchDir.appendingPathComponent("a_frame0.png"))
    writePNG(makeCorners(width: 320, height: 240, offsetX: shift, offsetY: 0),
             to: matchDir.appendingPathComponent("b_frame1.png"))

    let extractor = FeatureExtractorFactory.make()
    var sets: [FeatureSet] = []
    let source = try IngestionSource.detect(at: matchDir)
    try FrameIngestor.ingest(source) { frame in
        sets.append(try extractor.extract(index: frame.index, pixelBuffer: frame.pixelBuffer,
                                          options: FeatureOptions(maxFeatures: 500)))
        return true
    }
    expect(sets.count == 2, "two frames extracted")

    let matches = FeatureMatcher.match(query: sets[0], train: sets[1])
    expect(matches.count > 15, "matcher finds a usable number of correspondences (got \(matches.count))")

    // Ground truth: frame 1 is frame 0 shifted right by `shift` px. Note the
    // CGContext y-axis is bottom-up while keypoints are top-down, so only the
    // x-shift is asserted directionally; dy must simply be ~0.
    let dxs = matches.map { sets[1].keypoints[$0.trainIndex].x - sets[0].keypoints[$0.queryIndex].x }
    let dys = matches.map { sets[1].keypoints[$0.trainIndex].y - sets[0].keypoints[$0.queryIndex].y }
    let correct = zip(dxs, dys).filter { abs($0 - Float(shift)) < 1.5 && abs($1) < 1.5 }.count
    let fraction = Double(correct) / Double(matches.count)
    print(String(format: "      %d matches, %.0f%% recover the true (+%d, 0) shift", matches.count, fraction * 100, shift))
    expect(fraction > 0.85, "≥85% of matches recover the ground-truth shift (got \(Int(fraction * 100))%)")

    // Self-matching must be perfect and identity.
    let selfMatches = FeatureMatcher.match(query: sets[0], train: sets[0])
    let identity = selfMatches.allSatisfy { $0.queryIndex == $0.trainIndex && $0.distance == 0 }
    expect(identity, "matching a frame against itself yields identity at distance 0")

    // Degenerate inputs must not crash.
    let empty = FeatureSet(frameIndex: 0, keypoints: [], descriptors: [])
    expect(FeatureMatcher.match(query: empty, train: sets[0]).isEmpty, "empty query yields no matches")
    expect(FeatureMatcher.match(query: sets[0], train: empty).isEmpty, "empty train yields no matches")
}

print("Hamming distance:")
do {
    let a: [UInt8] = [0b0000_0000, 0xFF]
    let b: [UInt8] = [0b0000_1111, 0xFF]
    expect(FeatureMatcher.hamming(a[0...1], b[0...1]) == 4, "hamming counts differing bits")
    expect(FeatureMatcher.hamming(a[0...1], a[0...1]) == 0, "hamming of a value with itself is 0")
}

// MARK: - Stage 3: two-view geometry
//
// Synthetic scenes with exactly known poses. Real imagery cannot test this
// layer: without ground truth you can only check that a reconstruction is
// self-consistent, which a mirrored or wrongly-scaled one also is.

func rotationAboutY(_ degrees: Double) -> [Double] {
    let t = degrees * .pi / 180
    return [cos(t), 0, sin(t), 0, 1, 0, -sin(t), 0, cos(t)]
}

func rotationAngleBetween(_ a: [Double], _ b: [Double]) -> Double {
    // Geodesic angle on SO(3): acos((trace(A Bᵀ) − 1) / 2)
    let abt = LinearAlgebra.matMul3(a, LinearAlgebra.transpose3(b))
    let trace = abt[0] + abt[4] + abt[8]
    return acos(Swift.max(-1, Swift.min(1, (trace - 1) / 2))) * 180 / .pi
}

/// Deterministic synthetic scene: 3D points, two cameras, exact projections.
func makeSyntheticPair(
    pointCount: Int, rotationDegrees: Double, baseline: SIMD3<Double>,
    intrinsics: CameraIntrinsics, seed: UInt64
) -> (points: [SIMD3<Double>], pose2: CameraPose, kp1: [Keypoint], kp2: [Keypoint], matches: [FeatureMatch]) {
    var rng = SplitMix64(seed: seed)
    let rotation = rotationAboutY(rotationDegrees)
    // Pose is world->camera, so t = -R * C for camera centre C.
    let rc = LinearAlgebra.matVec3(rotation, baseline)
    let pose2 = CameraPose(rotation: rotation, translation: SIMD3<Double>(-rc.x, -rc.y, -rc.z))
    let pose1 = CameraPose.identity

    var points: [SIMD3<Double>] = []
    var kp1: [Keypoint] = []
    var kp2: [Keypoint] = []
    var matches: [FeatureMatch] = []

    while points.count < pointCount {
        // Spread in a slab in front of both cameras; varied depth gives the
        // parallax the essential matrix needs.
        let x = (Double(rng.nextUniform()) - 0.5) * 4.0
        let y = (Double(rng.nextUniform()) - 0.5) * 3.0
        let z = 3.0 + Double(rng.nextUniform()) * 4.0
        let p = SIMD3<Double>(x, y, z)
        guard let a = pose1.project(p, intrinsics: intrinsics),
              let b = pose2.project(p, intrinsics: intrinsics) else { continue }
        // Keep only points that land inside a plausible image.
        guard a.x > 0, a.x < 2 * intrinsics.cx, a.y > 0, a.y < 2 * intrinsics.cy,
              b.x > 0, b.x < 2 * intrinsics.cx, b.y > 0, b.y < 2 * intrinsics.cy else { continue }
        let index = points.count
        points.append(p)
        kp1.append(Keypoint(x: Float(a.x), y: Float(a.y), response: 1, angle: 0))
        kp2.append(Keypoint(x: Float(b.x), y: Float(b.y), response: 1, angle: 0))
        matches.append(FeatureMatch(queryIndex: index, trainIndex: index, distance: 0))
    }
    return (points, pose2, kp1, kp2, matches)
}

print("Linear algebra:")
do {
    // Symmetric eigen against a known decomposition.
    let m: [Double] = [2, 1, 0, 1, 2, 0, 0, 0, 3]
    let (values, _) = LinearAlgebra.symmetricEigen(m, n: 3)
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3; plus the isolated 3.
    expect(abs(values[0] - 1) < 1e-9, "smallest eigenvalue is 1 (got \(values[0]))")
    expect(abs(values[1] - 3) < 1e-9 && abs(values[2] - 3) < 1e-9, "remaining eigenvalues are 3")

    // SVD on a GENERAL matrix. A diagonal fixture is near-useless here: it
    // has U = V = I, so a transpose or sign error reconstructs perfectly and
    // the test passes while the decomposition is wrong.
    func checkSVD(_ a: [Double], _ label: String, expectRankDeficient: Bool) {
        let (u, s, vt) = LinearAlgebra.svd3x3(a)
        let sMat: [Double] = [s[0], 0, 0, 0, s[1], 0, 0, 0, s[2]]
        let recon = LinearAlgebra.matMul3(u, LinearAlgebra.matMul3(sMat, vt))
        let maxErr = zip(recon, a).map { abs($0 - $1) }.max() ?? 1
        expect(maxErr < 1e-9, "\(label): U·S·Vᵀ reconstructs A (max err \(maxErr))")
        expect(s[0] >= s[1] && s[1] >= s[2], "\(label): singular values descend")

        // U must be a genuine orthonormal basis. This is the property that
        // actually broke: for a rank-deficient A the third column collapsed
        // to the zero vector, so |col| == 0 and every downstream use of U
        // silently produced garbage.
        for col in 0..<3 {
            let v = SIMD3<Double>(u[col], u[3 + col], u[6 + col])
            expect(abs(LinearAlgebra.length(v) - 1) < 1e-9,
                   "\(label): U column \(col) is unit length (got \(LinearAlgebra.length(v)))")
        }
        let uut = LinearAlgebra.matMul3(u, LinearAlgebra.transpose3(u))
        let offDiag = [uut[1], uut[2], uut[5]].map { abs($0) }.max() ?? 1
        expect(offDiag < 1e-9, "\(label): U is orthogonal (max off-diagonal \(offDiag))")
        expect(abs(abs(LinearAlgebra.determinant3(u)) - 1) < 1e-9,
               "\(label): |det(U)| == 1 (got \(LinearAlgebra.determinant3(u)))")
        if expectRankDeficient {
            expect(s[2] == 0, "\(label): a numerically-zero singular value is reported as exactly 0 (got \(s[2]))")
        }
    }
    checkSVD([1, 2, 3, 4, 5, 6, 7, 8, 10], "general", expectRankDeficient: false)
    checkSVD([3, 0, 0, 0, 2, 0, 0, 0, 0], "diagonal rank-2", expectRankDeficient: true)
    // The real regression case: an essential matrix, [t]× R, exactly rank 2.
    // Its zero singular value arrives as sqrt(~1e-18) ≈ 1e-9, which an
    // absolute 1e-12 tolerance fails to recognize.
    let rGT = rotationAboutY(8)
    let tHat = SIMD3<Double>(0.9965, 0.0830, 0)
    let tCross: [Double] = [0, -tHat.z, tHat.y, tHat.z, 0, -tHat.x, -tHat.y, tHat.x, 0]
    checkSVD(LinearAlgebra.matMul3(tCross, rGT), "essential matrix", expectRankDeficient: true)

    // nearestRotation on a scaled rotation must return the rotation.
    let r = rotationAboutY(30)
    let scaled = r.map { $0 * 2.5 }
    let recovered = LinearAlgebra.nearestRotation(scaled)
    expect(rotationAngleBetween(recovered, r) < 1e-6, "nearestRotation strips scale")
    expect(abs(LinearAlgebra.determinant3(recovered) - 1) < 1e-9, "nearestRotation has det +1")

    // SPD solve.
    let spd: [Double] = [4, 1, 1, 3]
    let rhs: [Double] = [1, 2]
    if let x = LinearAlgebra.solveSPD(spd, rhs, n: 2) {
        let r0 = spd[0] * x[0] + spd[1] * x[1]
        let r1 = spd[2] * x[0] + spd[3] * x[1]
        expect(abs(r0 - 1) < 1e-12 && abs(r1 - 2) < 1e-12, "solveSPD solves the system")
    } else {
        expect(false, "solveSPD returned nil on an SPD matrix")
    }
    expect(LinearAlgebra.solveSPD([1, 0, 0, -1], [1, 1], n: 2) == nil, "solveSPD rejects non-PD input")
}

print("Camera model:")
do {
    let intrinsics = CameraIntrinsics.guess(width: 1920, height: 1080)
    expect(abs(intrinsics.cx - 960) < 1e-9 && abs(intrinsics.cy - 540) < 1e-9, "principal point centres the image")
    let round = intrinsics.project(intrinsics.normalize(x: 123, y: 456))
    expect(abs(round.x - 123) < 1e-9 && abs(round.y - 456) < 1e-9, "normalize/project round-trip")

    // A point behind the camera must not project.
    let pose = CameraPose.identity
    expect(pose.project(SIMD3<Double>(0, 0, -5), intrinsics: intrinsics) == nil,
           "points behind the camera do not project")

    // Camera centre of a translated pose.
    let c = SIMD3<Double>(1, 2, 3)
    let r = rotationAboutY(25)
    let rc = LinearAlgebra.matVec3(r, c)
    let posed = CameraPose(rotation: r, translation: SIMD3<Double>(-rc.x, -rc.y, -rc.z))
    let centre = posed.center
    expect(abs(centre.x - 1) < 1e-9 && abs(centre.y - 2) < 1e-9 && abs(centre.z - 3) < 1e-9,
           "camera centre recovers -Rᵀt (got \(centre))")
}

print("Two-view geometry (exact synthetic data):")
do {
    let intrinsics = CameraIntrinsics(focalLength: 1200, cx: 960, cy: 540)
    let baseline = SIMD3<Double>(0.6, 0.05, 0)
    let scene = makeSyntheticPair(pointCount: 120, rotationDegrees: 8,
                                  baseline: baseline, intrinsics: intrinsics, seed: 42)

    guard let result = TwoViewGeometry.estimate(
        matches: scene.matches, keypoints1: scene.kp1, keypoints2: scene.kp2,
        intrinsics1: intrinsics, intrinsics2: intrinsics
    ) else {
        expect(false, "two-view estimation returned a result")
        exit(1)
    }

    let rotationError = rotationAngleBetween(result.pose.rotation, scene.pose2.rotation)
    print(String(format: "      rotation error %.4f deg, inliers %d/%d, points %d",
                 rotationError, result.inliers.count, scene.matches.count, result.points.count))
    expect(rotationError < 0.5, "recovered rotation matches ground truth (\(rotationError) deg)")

    // Translation is recovered only up to scale, so compare directions.
    let tTrue = scene.pose2.translation
    let tTrueNorm = LinearAlgebra.length(tTrue)
    let tEst = result.pose.translation
    let dot = (tTrue.x * tEst.x + tTrue.y * tEst.y + tTrue.z * tEst.z) / tTrueNorm
    let directionError = acos(Swift.max(-1, Swift.min(1, abs(dot)))) * 180 / .pi
    expect(directionError < 1.0, "recovered translation direction matches (\(directionError) deg)")

    expect(result.inliers.count >= scene.matches.count - 2,
           "essentially all exact correspondences are inliers (\(result.inliers.count)/\(scene.matches.count))")
    expect(result.points.count > 100, "triangulation produced points (\(result.points.count))")

    // Triangulated structure must match ground truth up to a single global
    // scale (two views cannot fix scale).
    var ratios: [Double] = []
    for (i, matchIndex) in result.pointMatchIndices.enumerated() {
        let truth = scene.points[scene.matches[matchIndex].queryIndex]
        ratios.append(LinearAlgebra.length(truth) / Swift.max(1e-12, LinearAlgebra.length(result.points[i])))
    }
    let meanRatio = ratios.reduce(0, +) / Double(ratios.count)
    let maxDeviation = ratios.map { abs($0 / meanRatio - 1) }.max() ?? 1
    expect(maxDeviation < 0.02, "triangulated structure matches truth up to one global scale (max dev \(maxDeviation))")
}

print("Two-view geometry (noise and outliers):")
do {
    let intrinsics = CameraIntrinsics(focalLength: 1200, cx: 960, cy: 540)
    var scene = makeSyntheticPair(pointCount: 140, rotationDegrees: 6,
                                  baseline: SIMD3<Double>(0.5, 0, 0.05), intrinsics: intrinsics, seed: 7)

    // Corrupt 25% of the matches into random wrong correspondences —
    // RANSAC's whole purpose.
    var rng = SplitMix64(seed: 99)
    let outlierCount = scene.matches.count / 4
    for i in 0..<outlierCount {
        let victim = Int(rng.next() % UInt64(scene.matches.count))
        let wrongTarget = Int(rng.next() % UInt64(scene.kp2.count))
        scene.matches[victim] = FeatureMatch(queryIndex: scene.matches[victim].queryIndex,
                                             trainIndex: wrongTarget, distance: 50)
        _ = i
    }

    guard let result = TwoViewGeometry.estimate(
        matches: scene.matches, keypoints1: scene.kp1, keypoints2: scene.kp2,
        intrinsics1: intrinsics, intrinsics2: intrinsics
    ) else {
        expect(false, "estimation survived 25% outliers")
        exit(1)
    }
    let rotationError = rotationAngleBetween(result.pose.rotation, scene.pose2.rotation)
    print(String(format: "      with 25%% outliers: rotation error %.4f deg, inliers %d/%d",
                 rotationError, result.inliers.count, scene.matches.count))
    expect(rotationError < 1.0, "rotation still recovered under 25% outliers (\(rotationError) deg)")
    expect(result.inliers.count > scene.matches.count / 2, "RANSAC kept the consensus set")
}

print("Two-view degenerate inputs:")
do {
    let intrinsics = CameraIntrinsics.guess(width: 640, height: 480)
    expect(TwoViewGeometry.estimate(matches: [], keypoints1: [], keypoints2: [],
                                    intrinsics1: intrinsics, intrinsics2: intrinsics) == nil,
           "no matches yields no result")
    let few = (0..<4).map { FeatureMatch(queryIndex: $0, trainIndex: $0, distance: 0) }
    let kps = (0..<4).map { Keypoint(x: Float($0 * 10), y: Float($0 * 10), response: 1, angle: 0) }
    expect(TwoViewGeometry.estimate(matches: few, keypoints1: kps, keypoints2: kps,
                                    intrinsics1: intrinsics, intrinsics2: intrinsics) == nil,
           "fewer than 8 matches yields no result")
}

print("Guided epipolar matching:")
do {
    let intrinsics = CameraIntrinsics(focalLength: 1200, cx: 960, cy: 540)
    let scene = makeSyntheticPair(pointCount: 90, rotationDegrees: 7,
                                  baseline: SIMD3<Double>(0.55, 0.03, 0), intrinsics: intrinsics, seed: 4242)

    // Descriptors that are deliberately AMBIGUOUS: every point gets one of
    // only 6 distinct patterns, so many features look alike and the plain
    // ratio test throws most of them away. This is the situation guided
    // matching exists for — wide-viewpoint frames where appearance is no
    // longer distinctive on its own.
    var rng = SplitMix64(seed: 11)
    var patterns: [[UInt8]] = []
    for _ in 0..<6 {
        var d = [UInt8](repeating: 0, count: FeatureSet.descriptorBytes)
        for b in 0..<FeatureSet.descriptorBytes { d[b] = UInt8(rng.next() & 0xFF) }
        patterns.append(d)
    }
    var descriptors1: [UInt8] = [], descriptors2: [UInt8] = []
    for i in 0..<scene.kp1.count {
        descriptors1.append(contentsOf: patterns[i % 6])
        descriptors2.append(contentsOf: patterns[i % 6])
    }
    let set1 = FeatureSet(frameIndex: 0, keypoints: scene.kp1, descriptors: descriptors1)
    let set2 = FeatureSet(frameIndex: 1, keypoints: scene.kp2, descriptors: descriptors2)

    let plain = FeatureMatcher.match(query: set1, train: set2)
    let guided = FeatureMatcher.matchGuided(
        query: set1, train: set2,
        queryPose: .identity, trainPose: scene.pose2,
        queryIntrinsics: intrinsics, trainIntrinsics: intrinsics
    )
    // Ground truth is index-to-index by construction.
    let plainCorrect = plain.filter { $0.queryIndex == $0.trainIndex }.count
    let guidedCorrect = guided.filter { $0.queryIndex == $0.trainIndex }.count
    print("      ambiguous descriptors: plain \(plainCorrect)/\(plain.count) correct, guided \(guidedCorrect)/\(guided.count)")

    expect(guided.count > plain.count,
           "guided matching recovers more correspondences (\(guided.count) vs \(plain.count))")
    expect(guidedCorrect > plainCorrect,
           "guided matching recovers more CORRECT correspondences (\(guidedCorrect) vs \(plainCorrect))")
    let precision = guided.isEmpty ? 0 : Double(guidedCorrect) / Double(guided.count)
    expect(precision > 0.9, "guided matches stay precise (\(Int(precision * 100))%)")

    // A wrong pose must NOT produce a confident match set — the epipolar
    // constraint has to be doing real work, not rubber-stamping everything.
    let wrongPose = CameraPose(rotation: rotationAboutY(75), translation: SIMD3<Double>(2, 1, 0.5))
    let bogus = FeatureMatcher.matchGuided(
        query: set1, train: set2,
        queryPose: .identity, trainPose: wrongPose,
        queryIntrinsics: intrinsics, trainIntrinsics: intrinsics
    )
    let bogusCorrect = bogus.filter { $0.queryIndex == $0.trainIndex }.count
    expect(bogusCorrect < guidedCorrect / 2,
           "a wrong pose does not yield correct matches (\(bogusCorrect) vs \(guidedCorrect))")
}

print("Bundle adjustment:")
do {
    let intrinsics = CameraIntrinsics(focalLength: 1200, cx: 960, cy: 540)
    let baseline = SIMD3<Double>(0.6, 0.05, 0)
    let scene = makeSyntheticPair(pointCount: 100, rotationDegrees: 8,
                                  baseline: baseline, intrinsics: intrinsics, seed: 2024)

    // Ground-truth reconstruction: reprojection error must be ~0. This is
    // first a check on the whole camera/projection convention stack — if the
    // conventions disagree anywhere, exact inputs would not give exact output.
    var keypointsByFrame: [Int: [Keypoint]] = [0: scene.kp1, 1: scene.kp2]
    func makeReconstruction(pose2: CameraPose, points: [SIMD3<Double>]) -> Reconstruction {
        var r = Reconstruction()
        r.cameras[0] = RegisteredCamera(frameIndex: 0, pose: .identity, intrinsics: intrinsics)
        r.cameras[1] = RegisteredCamera(frameIndex: 1, pose: pose2, intrinsics: intrinsics)
        r.points = points.enumerated().map {
            ScenePoint(position: $1, observations: [(frame: 0, keypoint: $0), (frame: 1, keypoint: $0)])
        }
        return r
    }

    let truth = makeReconstruction(pose2: scene.pose2, points: scene.points)
    let truthRMSE = truth.reprojectionRMSE(keypoints: keypointsByFrame)
    expect(truthRMSE < 1e-3, "ground-truth reconstruction reprojects exactly (RMSE \(truthRMSE) px; Keypoint stores Float, so ~1e-5 is the floor)")

    // Now perturb points and the second camera's rotation, and let BA recover.
    var rng = SplitMix64(seed: 555)
    let perturbedPoints = scene.points.map { p in
        SIMD3<Double>(p.x + (Double(rng.nextUniform()) - 0.5) * 0.30,
                      p.y + (Double(rng.nextUniform()) - 0.5) * 0.30,
                      p.z + (Double(rng.nextUniform()) - 0.5) * 0.30)
    }
    let perturbedPose2 = scene.pose2.rotated(byAxisAngle: SIMD3<Double>(0.010, -0.015, 0.008))
    var reconstruction = makeReconstruction(pose2: perturbedPose2, points: perturbedPoints)

    let before = reconstruction.reprojectionRMSE(keypoints: keypointsByFrame)
    let rotationErrorBefore = rotationAngleBetween(reconstruction.cameras[1]!.pose.rotation, scene.pose2.rotation)
    let result = BundleAdjustment.refine(reconstruction: &reconstruction, keypoints: keypointsByFrame)
    let after = reconstruction.reprojectionRMSE(keypoints: keypointsByFrame)
    let rotationErrorAfter = rotationAngleBetween(reconstruction.cameras[1]!.pose.rotation, scene.pose2.rotation)

    print(String(format: "      RMSE %.4f -> %.4f px in %d iters (rotation err %.4f -> %.4f deg)",
                 before, after, result.iterations, rotationErrorBefore, rotationErrorAfter))
    expect(after < before, "bundle adjustment reduces reprojection error")
    expect(after < 0.05, "converges to sub-0.05px reprojection error (got \(after))")
    expect(rotationErrorAfter < rotationErrorBefore, "camera rotation moves toward truth")
    expect(rotationErrorAfter < 0.05, "recovers the true rotation (\(rotationErrorAfter) deg)")
    expect(result.initialRMSE >= result.finalRMSE, "reported RMSE is non-increasing")

    // The first camera is the gauge anchor and must not move at all.
    let anchor = reconstruction.cameras[0]!.pose
    expect(anchor.rotation == CameraPose.identity.rotation && anchor.translation == .zero,
           "the fixed first camera is untouched")

    // Baseline length is the free scale DOF and must be preserved.
    let baselineAfter = LinearAlgebra.length(reconstruction.cameras[1]!.pose.translation)
    let baselineBefore = LinearAlgebra.length(perturbedPose2.translation)
    expect(abs(baselineAfter - baselineBefore) / baselineBefore < 0.02,
           "scale gauge holds the baseline length (\(baselineBefore) -> \(baselineAfter))")

    // Degenerate input must not crash or report nonsense.
    var empty = Reconstruction()
    let emptyResult = BundleAdjustment.refine(reconstruction: &empty, keypoints: [:])
    expect(emptyResult.iterations == 0, "empty reconstruction is a no-op")
}

print("Absolute pose (PnP):")
do {
    let intrinsics = CameraIntrinsics(focalLength: 1100, cx: 800, cy: 600)
    // A known camera looking at a known 3D cloud.
    let rotation = rotationAboutY(20)
    let centre = SIMD3<Double>(1.2, -0.4, -0.5)
    let rc = LinearAlgebra.matVec3(rotation, centre)
    let truePose = CameraPose(rotation: rotation, translation: SIMD3<Double>(-rc.x, -rc.y, -rc.z))

    var rng = SplitMix64(seed: 31337)
    var worldPoints: [SIMD3<Double>] = []
    var imagePoints: [SIMD2<Double>] = []
    while worldPoints.count < 60 {
        let p = SIMD3<Double>((Double(rng.nextUniform()) - 0.5) * 5,
                              (Double(rng.nextUniform()) - 0.5) * 4,
                              3.0 + Double(rng.nextUniform()) * 5)
        guard let projected = truePose.project(p, intrinsics: intrinsics) else { continue }
        worldPoints.append(p)
        imagePoints.append(projected)
    }

    guard let clean = PoseEstimation.estimatePose(worldPoints: worldPoints, imagePoints: imagePoints,
                                                  intrinsics: intrinsics) else {
        expect(false, "PnP returned a pose")
        exit(1)
    }
    let rotationError = rotationAngleBetween(clean.pose.rotation, truePose.rotation)
    let centreError = LinearAlgebra.length(clean.pose.center - truePose.center)
    print(String(format: "      clean: rotation err %.5f deg, centre err %.5f, inliers %d/%d",
                 rotationError, centreError, clean.inliers.count, worldPoints.count))
    expect(rotationError < 0.01, "PnP recovers rotation (\(rotationError) deg)")
    expect(centreError < 0.01, "PnP recovers camera centre (\(centreError))")
    expect(clean.inliers.count >= worldPoints.count - 2, "clean data is almost entirely inliers")

    // 30% of the 2D observations corrupted.
    var corrupted = imagePoints
    for i in 0..<(worldPoints.count * 3 / 10) {
        let victim = Int(rng.next() % UInt64(corrupted.count))
        corrupted[victim] = SIMD2<Double>(Double(rng.nextUniform()) * 1600,
                                          Double(rng.nextUniform()) * 1200)
        _ = i
    }
    guard let robust = PoseEstimation.estimatePose(worldPoints: worldPoints, imagePoints: corrupted,
                                                   intrinsics: intrinsics) else {
        expect(false, "PnP survived 30% outliers")
        exit(1)
    }
    let robustRotationError = rotationAngleBetween(robust.pose.rotation, truePose.rotation)
    print(String(format: "      30%% outliers: rotation err %.5f deg, inliers %d/%d",
                 robustRotationError, robust.inliers.count, worldPoints.count))
    expect(robustRotationError < 0.5, "PnP still recovers rotation under 30% outliers (\(robustRotationError) deg)")

    expect(PoseEstimation.estimatePose(worldPoints: Array(worldPoints.prefix(3)),
                                       imagePoints: Array(imagePoints.prefix(3)),
                                       intrinsics: intrinsics) == nil,
           "fewer than 6 correspondences yields no pose")
}

print("Incremental structure from motion (end-to-end, synthetic):")
do {
    // A camera arc around a 3D cloud, with synthetic descriptors that make
    // matching unambiguous. This exercises the real pipeline — matching,
    // track building, initial pair selection, PnP registration, triangulation
    // and bundle adjustment — against poses we know exactly.
    let intrinsics = CameraIntrinsics(focalLength: 1000, cx: 640, cy: 360)
    let frameCount = 6
    var rng = SplitMix64(seed: 8080)

    var world: [SIMD3<Double>] = []
    while world.count < 220 {
        world.append(SIMD3<Double>((Double(rng.nextUniform()) - 0.5) * 4,
                                   (Double(rng.nextUniform()) - 0.5) * 3,
                                   4.0 + Double(rng.nextUniform()) * 3))
    }

    // Cameras translate sideways with a slight inward rotation.
    var truePoses: [CameraPose] = []
    for i in 0..<frameCount {
        let t = Double(i)
        let rotation = rotationAboutY(-3.0 * t)
        let centre = SIMD3<Double>(0.35 * t, 0.02 * t, 0)
        let rc = LinearAlgebra.matVec3(rotation, centre)
        truePoses.append(CameraPose(rotation: rotation, translation: SIMD3<Double>(-rc.x, -rc.y, -rc.z)))
    }

    // Each world point gets a unique descriptor, reused across every frame it
    // appears in — a stand-in for a perfect detector, isolating the geometry.
    var descriptorFor: [[UInt8]] = []
    for _ in 0..<world.count {
        var d = [UInt8](repeating: 0, count: FeatureSet.descriptorBytes)
        for b in 0..<FeatureSet.descriptorBytes { d[b] = UInt8(rng.next() & 0xFF) }
        descriptorFor.append(d)
    }

    var featureSets: [FeatureSet] = []
    var intrinsicsByFrame: [Int: CameraIntrinsics] = [:]
    var visibleWorldIndex: [Int: [Int]] = [:]
    for frame in 0..<frameCount {
        var keypoints: [Keypoint] = []
        var descriptors: [UInt8] = []
        var visible: [Int] = []
        for (worldIndex, p) in world.enumerated() {
            guard let projected = truePoses[frame].project(p, intrinsics: intrinsics),
                  projected.x > 0, projected.x < 1280, projected.y > 0, projected.y < 720 else { continue }
            keypoints.append(Keypoint(x: Float(projected.x), y: Float(projected.y), response: 1, angle: 0))
            descriptors.append(contentsOf: descriptorFor[worldIndex])
            visible.append(worldIndex)
        }
        featureSets.append(FeatureSet(frameIndex: frame, keypoints: keypoints, descriptors: descriptors))
        intrinsicsByFrame[frame] = intrinsics
        visibleWorldIndex[frame] = visible
    }
    expect(featureSets.allSatisfy { $0.count > 60 }, "every synthetic frame sees enough points")

    guard let (reconstruction, report) = StructureFromMotion.reconstruct(
        featureSets: featureSets, intrinsics: intrinsicsByFrame,
        options: SfMOptions(matchWindow: 3, minPairInliers: 30, minInitialAngleDegrees: 1.0)
    ) else {
        expect(false, "SfM produced a reconstruction")
        exit(1)
    }

    print(String(format: "      registered %d/%d cameras, %d points, RMSE %.4f -> %.4f px",
                 report.registeredCameras, report.totalCameras, report.points,
                 report.rmseBefore, report.rmseAfter))
    expect(report.registeredCameras == frameCount,
           "all \(frameCount) cameras registered (got \(report.registeredCameras))")
    expect(report.points > 100, "a substantial point cloud was triangulated (got \(report.points))")
    expect(report.rmseAfter < 1.0, "final reprojection RMSE is sub-pixel (\(report.rmseAfter))")

    // Geometry check: the reconstruction is only defined up to a similarity
    // transform, so compare the SHAPE of the camera path — ratios of
    // consecutive baselines — against the truth, which is scale-invariant.
    let orderedFrames = reconstruction.cameras.keys.sorted()
    if orderedFrames.count == frameCount {
        var estimatedGaps: [Double] = []
        var trueGaps: [Double] = []
        for i in 1..<orderedFrames.count {
            let a = reconstruction.cameras[orderedFrames[i - 1]]!.pose.center
            let b = reconstruction.cameras[orderedFrames[i]]!.pose.center
            estimatedGaps.append(LinearAlgebra.length(b - a))
            trueGaps.append(LinearAlgebra.length(truePoses[i].center - truePoses[i - 1].center))
        }
        let scale = estimatedGaps[0] / trueGaps[0]
        let maxDeviation = zip(estimatedGaps, trueGaps).map { abs($0 / ($1 * scale) - 1) }.max() ?? 1
        print(String(format: "      camera-path shape: max baseline deviation %.4f", maxDeviation))
        expect(maxDeviation < 0.05,
               "camera path matches truth up to a global scale (max dev \(maxDeviation))")
    }

    // Reproducibility: the same input must give bit-identical output. This is
    // what the union-find identity fix protects — a hash-keyed track table
    // would vary per process because Swift seeds hashValue randomly.
    guard let (again, report2) = StructureFromMotion.reconstruct(
        featureSets: featureSets, intrinsics: intrinsicsByFrame,
        options: SfMOptions(matchWindow: 3, minPairInliers: 30, minInitialAngleDegrees: 1.0)
    ) else {
        expect(false, "second SfM run produced a reconstruction")
        exit(1)
    }
    expect(report2.points == report.points && report2.registeredCameras == report.registeredCameras,
           "SfM is reproducible across runs (\(report.points) vs \(report2.points) points)")
    let samePoses = orderedFrames.allSatisfy { frame in
        guard let a = reconstruction.cameras[frame], let b = again.cameras[frame] else { return false }
        return rotationAngleBetween(a.pose.rotation, b.pose.rotation) < 1e-9
    }
    expect(samePoses, "SfM camera poses are identical across runs")
}

print("\n\(passed) passed, \(failures) failed")
exit(failures == 0 ? 0 : 1)
