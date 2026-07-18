import Foundation
import CoreVideo

// MARK: - Stage 3: feature extraction — shared interface and shared math
//
// Tiering follows the same shape as stage 2, with one deliberate difference:
// only CORNER DETECTION is tier-specific. Orientation assignment and
// descriptor computation are shared code called by both tiers.
//
// Why: detection is the expensive, embarrassingly-parallel part (a response
// value per pixel over the whole frame) and is worth a GPU kernel. Descriptors
// are cheap by comparison — 2000 features x 256 binary tests is ~512K samples,
// trivial next to a 2 MP response map — so duplicating that logic in MSL would
// buy nothing and would introduce a second place for the tiers to disagree.
// Sharing it makes descriptor parity structural rather than something we have
// to keep testing into existence.

/// A detected corner. Coordinates are in pixels of the analyzed frame.
public struct Keypoint: Equatable {
    public let x: Float
    public let y: Float
    /// Harris response — corner strength. Comparable only within one frame.
    public let response: Float
    /// Dominant orientation in radians, from the intensity centroid. Used to
    /// steer the descriptor so matching survives camera roll.
    public let angle: Float

    public init(x: Float, y: Float, response: Float, angle: Float) {
        self.x = x
        self.y = y
        self.response = response
        self.angle = angle
    }
}

/// Keypoints plus their packed binary descriptors for one frame.
public struct FeatureSet {
    public static let descriptorBits = 256
    public static let descriptorBytes = descriptorBits / 8   // 32

    public let frameIndex: Int
    public let keypoints: [Keypoint]
    /// Row-major, `keypoints.count * descriptorBytes` bytes.
    public let descriptors: [UInt8]

    public init(frameIndex: Int, keypoints: [Keypoint], descriptors: [UInt8]) {
        self.frameIndex = frameIndex
        self.keypoints = keypoints
        self.descriptors = descriptors
    }

    public var count: Int { keypoints.count }

    public func descriptor(at index: Int) -> ArraySlice<UInt8> {
        let start = index * FeatureSet.descriptorBytes
        return descriptors[start..<(start + FeatureSet.descriptorBytes)]
    }
}

public enum FeatureError: Error, CustomStringConvertible {
    case unsupportedPixelFormat(OSType)
    case metalUnavailable(String)
    case kernelFailure(String)

    public var description: String {
        switch self {
        case .unsupportedPixelFormat(let f): return "Feature extractor needs BGRA input, got FourCC \(f)"
        case .metalUnavailable(let d): return "Metal feature extractor unavailable: \(d)"
        case .kernelFailure(let d): return "Feature kernel failed: \(d)"
        }
    }
}

public struct FeatureOptions {
    /// Upper bound on returned keypoints, strongest-first.
    public var maxFeatures: Int
    /// Harris response floor, relative to the frame's maximum response.
    /// Relative for the same reason stage 2's blur threshold is: absolute
    /// corner strength scales with scene contrast, so a fixed cutoff would
    /// return nothing on a low-contrast capture.
    public var relativeThreshold: Float
    /// Non-maximum suppression radius in pixels. Keeps features spread out
    /// instead of clumping on one high-contrast edge.
    public var nmsRadius: Int

    public init(maxFeatures: Int = 2000, relativeThreshold: Float = 0.01, nmsRadius: Int = 4) {
        self.maxFeatures = maxFeatures
        self.relativeThreshold = relativeThreshold
        self.nmsRadius = nmsRadius
    }
}

public protocol FeatureExtractor: AnyObject {
    var descriptionForLog: String { get }
    func extract(index: Int, pixelBuffer: CVPixelBuffer, options: FeatureOptions) throws -> FeatureSet
}

public enum FeatureExtractorFactory {
    public static func make(forceCPU: Bool = false) -> FeatureExtractor {
        if !forceCPU, let metal = try? MetalFeatureExtractor() {
            return metal
        }
        return CPUFeatureExtractor()
    }
}

// MARK: - Shared math used by both tiers

public enum FeatureMath {

    /// Patch radius for orientation and descriptor sampling. 15 px matches
    /// ORB's default and keeps the sampling pattern inside a 31x31 window.
    public static let patchRadius = 15

    /// BRIEF sampling pattern: 256 point pairs inside the patch.
    ///
    /// Generated once from a FIXED seed with a deterministic PRNG rather than
    /// hardcoding a table or using a system RNG. Two properties matter and
    /// both are required for correctness, not convenience:
    ///  - identical across tiers, so Metal and CPU descriptors are bit-equal;
    ///  - identical across RUNS and machines, or descriptors extracted today
    ///    could not be matched against ones extracted yesterday.
    /// A Gaussian-ish distribution concentrates tests near the center where
    /// the patch is most reliable under small misalignment.
    public static let pattern: [(dx1: Int8, dy1: Int8, dx2: Int8, dy2: Int8)] = {
        var rng = SplitMix64(seed: 0x5F3A_9C21_7E44_B0D9)
        var result: [(Int8, Int8, Int8, Int8)] = []
        result.reserveCapacity(descriptorTests)
        let sigma = Float(patchRadius) / 2.5
        func sample() -> Int8 {
            // Box-Muller, clamped into the patch.
            let value = rng.nextGaussian() * sigma
            return Int8(max(Float(-patchRadius), min(Float(patchRadius), value.rounded())))
        }
        while result.count < descriptorTests {
            let a = (sample(), sample())
            let b = (sample(), sample())
            // Reject degenerate tests that compare a pixel with itself.
            if a == b { continue }
            result.append((a.0, a.1, b.0, b.1))
        }
        return result
    }()

    static let descriptorTests = FeatureSet.descriptorBits

    /// Intensity-centroid orientation (Rosin), the same construction ORB uses.
    /// atan2 of the first-order moments about the patch center: the vector
    /// from the center to the patch's brightness centroid defines a repeatable
    /// direction, so the descriptor can be steered by it.
    public static func orientation(luma: [Float], width: Int, height: Int, x: Int, y: Int) -> Float {
        var m01: Float = 0
        var m10: Float = 0
        let r = patchRadius
        for dy in -r...r {
            let py = y + dy
            guard py >= 0 && py < height else { continue }
            let row = py * width
            for dx in -r...r {
                let px = x + dx
                guard px >= 0 && px < width else { continue }
                guard dx * dx + dy * dy <= r * r else { continue }
                let value = luma[row + px]
                m10 += Float(dx) * value
                m01 += Float(dy) * value
            }
        }
        return atan2(m01, m10)
    }

    /// Steered BRIEF: rotate the sampling pattern by the keypoint's angle,
    /// then compare pairs. Rotating the PATTERN rather than the image patch
    /// avoids resampling the image and keeps the operation exact.
    /// Out-of-bounds samples read 0, matching the border convention used by
    /// the analyzers in stage 2.
    public static func describe(luma: [Float], width: Int, height: Int, keypoint: Keypoint) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: FeatureSet.descriptorBytes)
        let cosA = cos(keypoint.angle)
        let sinA = sin(keypoint.angle)
        let cx = Int(keypoint.x.rounded())
        let cy = Int(keypoint.y.rounded())

        @inline(__always)
        func sample(_ dx: Int8, _ dy: Int8) -> Float {
            let fdx = Float(dx), fdy = Float(dy)
            let rx = cosA * fdx - sinA * fdy
            let ry = sinA * fdx + cosA * fdy
            let px = cx + Int(rx.rounded())
            let py = cy + Int(ry.rounded())
            guard px >= 0, px < width, py >= 0, py < height else { return 0 }
            return luma[py * width + px]
        }

        for i in 0..<descriptorTests {
            let t = pattern[i]
            if sample(t.dx1, t.dy1) < sample(t.dx2, t.dy2) {
                bytes[i >> 3] |= UInt8(1 << (i & 7))
            }
        }
        return bytes
    }

    /// Convert a Harris response map into keypoints: relative threshold,
    /// non-maximum suppression, strongest-first, capped at maxFeatures.
    /// Shared by both tiers so selection can never diverge — only the
    /// response map itself is tier-specific.
    public static func selectKeypoints(
        response: [Float], width: Int, height: Int,
        luma: [Float], options: FeatureOptions
    ) -> [Keypoint] {
        guard width > 2, height > 2, !response.isEmpty else { return [] }
        let maxResponse = response.max() ?? 0
        guard maxResponse > 0 else { return [] }
        let threshold = maxResponse * options.relativeThreshold
        let r = max(1, options.nmsRadius)

        var candidates: [(index: Int, x: Int, y: Int, value: Float)] = []
        for y in r..<(height - r) {
            let row = y * width
            for x in r..<(width - r) {
                let value = response[row + x]
                guard value >= threshold else { continue }
                // Strict non-maximum suppression over the (2r+1)^2 window.
                // Ties are broken by scan order via the `>=` on earlier
                // pixels, so the result is deterministic.
                var isMax = true
                var dy = -r
                nms: while dy <= r {
                    let py = (y + dy) * width
                    var dx = -r
                    while dx <= r {
                        if dx != 0 || dy != 0 {
                            let other = response[py + x + dx]
                            if other > value || (other == value && (dy < 0 || (dy == 0 && dx < 0))) {
                                isMax = false
                                break nms
                            }
                        }
                        dx += 1
                    }
                    dy += 1
                }
                if isMax {
                    candidates.append((row + x, x, y, value))
                }
            }
        }

        // Strongest first — but sorted on a QUANTIZED response, not the raw
        // float.
        //
        // The CPU and GPU response maps agree to within ~1 ULP (a 25-term
        // window summation feeding a cancellation-prone determinant; the
        // difference is inherent, not a bug). Sorting on the raw value means
        // two distinct keypoints whose responses differ by 1 ULP can order
        // differently on the two tiers. That is not cosmetic: `maxFeatures`
        // truncates this list, so an unstable order lets the tiers keep
        // different SUBSETS of features at the cut line, and the frames stop
        // being matchable across a tier switch.
        //
        // Quantizing onto a 2^20 grid relative to the frame maximum collapses
        // sub-ULP disagreements to equal keys, and `index` (row-major, so
        // spatial) then breaks the tie identically on both tiers. Note this is
        // a genuine total order — an epsilon-tolerant comparator would not be
        // transitive and would make sort() undefined behavior.
        let sortScale = Float(1 << 20) / maxResponse
        candidates.sort { a, b in
            let ka = (a.value * sortScale).rounded()
            let kb = (b.value * sortScale).rounded()
            return ka != kb ? ka > kb : a.index < b.index
        }
        if candidates.count > options.maxFeatures {
            candidates.removeSubrange(options.maxFeatures...)
        }

        return candidates.map { candidate in
            let angle = orientation(luma: luma, width: width, height: height, x: candidate.x, y: candidate.y)
            return Keypoint(x: Float(candidate.x), y: Float(candidate.y), response: candidate.value, angle: angle)
        }
    }

    /// Build the full FeatureSet from a response map. Shared tail of both
    /// tiers' extract().
    public static func assemble(
        frameIndex: Int, response: [Float], luma: [Float],
        width: Int, height: Int, options: FeatureOptions
    ) -> FeatureSet {
        let keypoints = selectKeypoints(response: response, width: width, height: height, luma: luma, options: options)
        var descriptors = [UInt8]()
        descriptors.reserveCapacity(keypoints.count * FeatureSet.descriptorBytes)
        for keypoint in keypoints {
            descriptors.append(contentsOf: describe(luma: luma, width: width, height: height, keypoint: keypoint))
        }
        return FeatureSet(frameIndex: frameIndex, keypoints: keypoints, descriptors: descriptors)
    }
}

/// Small deterministic PRNG. Used only to build the fixed BRIEF pattern, so
/// it must be reproducible across machines and OS versions — which rules out
/// SystemRandomNumberGenerator and anything seeded from time.
/// Public so the reproducibility contract itself is testable.
public struct SplitMix64 {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func nextUniform() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }

    /// Standard normal via Box-Muller.
    public mutating func nextGaussian() -> Float {
        let u1 = max(nextUniform(), 1e-7)
        let u2 = nextUniform()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
