import Foundation
import CoreVideo

// MARK: - Stage 2: frame filtering — shared interface
//
// Same tiered pattern as TrainingEngine: a protocol with a Metal baseline
// and a CPU (Accelerate) fallback. There is deliberately no enhanced-tier
// analyzer — Laplacian variance and an 8×8 luma signature are memory-bound
// trivia on any GPU; MLX/MPSGraph would add overhead, not speed.

/// Per-frame measurements used by the selector. Producing these is the
/// expensive part (GPU/CPU); consuming them is pure logic in FrameSelector.
public struct FrameScore: Equatable, Codable {
    public let index: Int
    public let timestamp: Double?
    /// Variance of the 3×3 Laplacian over full-resolution luma. Higher =
    /// sharper. Only comparable within one capture session — absolute values
    /// depend on scene content, so all thresholds downstream are relative.
    public let blurScore: Double
    /// 8×8 grid of mean luma values (row-major, 0…1) — a tiny perceptual
    /// signature for scene-change/duplicate detection. 64 floats per frame
    /// keeps a full session's signatures negligible in memory.
    public let signature: [Float]

    public init(index: Int, timestamp: Double?, blurScore: Double, signature: [Float]) {
        self.index = index
        self.timestamp = timestamp
        self.blurScore = blurScore
        self.signature = signature
    }

    /// Mean absolute difference between two signatures (0…1 scale).
    public func signatureDistance(to other: FrameScore) -> Double {
        guard signature.count == other.signature.count, !signature.isEmpty else { return 1 }
        var total: Double = 0
        for i in 0..<signature.count {
            total += Double(abs(signature[i] - other.signature[i]))
        }
        return total / Double(signature.count)
    }
}

public enum FrameAnalyzerError: Error, CustomStringConvertible {
    case unsupportedPixelFormat(OSType)
    case metalUnavailable(String)
    case kernelFailure(String)

    public var description: String {
        switch self {
        case .unsupportedPixelFormat(let format):
            return "Frame analyzer needs BGRA input, got FourCC \(format)"
        case .metalUnavailable(let detail):
            return "Metal analyzer unavailable: \(detail)"
        case .kernelFailure(let detail):
            return "Analyzer kernel failed: \(detail)"
        }
    }
}

/// Both implementations must produce scores that agree closely enough that
/// tier choice never changes which frames get selected (verified in selftest).
public protocol FrameAnalyzer: AnyObject {
    var descriptionForLog: String { get }
    func analyze(index: Int, timestamp: Double?, pixelBuffer: CVPixelBuffer) throws -> FrameScore
}

public enum FrameAnalyzerFactory {
    /// Metal when any device exists, CPU otherwise (or when forced).
    public static func make(forceCPU: Bool = false) -> FrameAnalyzer {
        if !forceCPU, let analyzer = try? MetalFrameAnalyzer() {
            return analyzer
        }
        return CPUFrameAnalyzer()
    }
}
