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

/// A detected corner. Coordinates are always in pixels of the FULL-RESOLUTION
/// frame, even when the corner was detected on a smaller pyramid level, so
/// every consumer downstream (matching, triangulation, bundle adjustment) can
/// stay ignorant of the pyramid.
public struct Keypoint: Equatable {
    public let x: Float
    public let y: Float
    /// Harris response — corner strength. Comparable only within one frame.
    public let response: Float
    /// Dominant orientation in radians, from the intensity centroid. Used to
    /// steer the descriptor so matching survives camera roll.
    public let angle: Float
    /// Pyramid level this corner was found on (0 = full resolution).
    public let octave: Int
    /// Linear size ratio of that level to full resolution (e.g. 2 means the
    /// corner was found on a half-size image, so its descriptor covers twice
    /// the area in original pixels).
    public let scale: Float

    public init(x: Float, y: Float, response: Float, angle: Float, octave: Int = 0, scale: Float = 1) {
        self.x = x
        self.y = y
        self.response = response
        self.angle = angle
        self.octave = octave
        self.scale = scale
    }
}

/// Which descriptor a FeatureSet holds, and therefore which distance metric
/// the matcher must use. Carried on the data rather than assumed globally, so
/// mixing sets built with different settings is a type-level impossibility
/// rather than a silent wrong answer.
public enum DescriptorKind: String, Codable {
    /// 256-bit steered BRIEF, compared with Hamming distance.
    case brief
    /// 128-dimension gradient-orientation histogram, compared with squared L2.
    case sift

    public var byteCount: Int {
        switch self {
        case .brief: return 32
        case .sift: return SIFTDescriptor.dimensions
        }
    }
}

/// Keypoints plus their packed descriptors for one frame.
public struct FeatureSet {
    public static let descriptorBits = 256
    /// BRIEF's size. Prefer `kind.byteCount` — this remains for the binary path.
    public static let descriptorBytes = descriptorBits / 8   // 32

    public let frameIndex: Int
    public let keypoints: [Keypoint]
    /// Row-major, `keypoints.count * kind.byteCount` bytes.
    public let descriptors: [UInt8]
    public let kind: DescriptorKind

    public init(frameIndex: Int, keypoints: [Keypoint], descriptors: [UInt8],
                kind: DescriptorKind = .brief) {
        self.frameIndex = frameIndex
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.kind = kind
    }

    public var count: Int { keypoints.count }
    public var descriptorByteCount: Int { kind.byteCount }

    public func descriptor(at index: Int) -> ArraySlice<UInt8> {
        let bytes = kind.byteCount
        let start = index * bytes
        return descriptors[start..<(start + bytes)]
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

    /// Grid resolution for spatial bucketing (N means an NxN grid). 0 disables
    /// it and reverts to pure global ranking.
    ///
    /// Without this, `maxFeatures` is filled by global Harris response, and one
    /// high-contrast region takes the entire budget. Observed on a real
    /// capture of a plant on an ornately engraved brass tray: essentially all
    /// 1200 keypoints landed on the tray and the plant, while richly textured
    /// floor, furniture and fabric elsewhere in frame got almost none. Features
    /// packed into one small, near-planar region give a poorly conditioned
    /// reconstruction — it is close to the planar-degeneracy case even though
    /// the room itself is fully three-dimensional.
    ///
    /// Bucketing takes the strongest few per cell first, so coverage is spread
    /// across the frame before raw strength is allowed to dominate. Every
    /// mature SfM implementation does some version of this.
    public var spatialBuckets: Int

    /// Absolute floor for the per-cell threshold, as a fraction of the frame's
    /// global maximum response.
    ///
    /// The threshold is computed PER CELL (see `relativeThreshold`), which is
    /// what lets a low-contrast or defocused region still contribute its best
    /// corners. But a cell containing only flat wall or sky has a maximum that
    /// is pure noise, and 1% of noise is still noise — without a floor those
    /// cells would fill up with garbage. This floor is deliberately very low
    /// (0.05% of the global max): high enough to reject sensor noise, low
    /// enough not to reintroduce the global-threshold problem it exists to fix.
    public var noiseFloorFraction: Float

    /// Number of pyramid levels, each half the linear size of the previous
    /// (1 = no pyramid, full resolution only).
    ///
    /// BRIEF descriptors are computed over a fixed pixel patch, so they are
    /// not scale invariant: the same physical surface photographed from twice
    /// the distance produces a completely different bit pattern and simply
    /// fails to match. Detecting and describing on a pyramid means a feature
    /// seen small in one frame and large in another can still match, because
    /// one of the levels puts them at comparable apparent size.
    ///
    /// 4 levels covers an 8x scale range, which comfortably spans the distance
    /// variation in a handheld orbit. Levels use exact 2x2 box downsampling —
    /// non-integer factors (ORB's 1.2) sample finer but need resampling, and
    /// the coarse steps are enough for the failure being addressed here.
    public var pyramidLevels: Int

    /// Which descriptor to compute. SIFT-like is the default: BRIEF compares
    /// raw intensities at point pairs and degrades sharply once appearance
    /// changes, which is what limited registration at the wide viewpoint
    /// changes late in an orbit. A gradient-orientation histogram discards
    /// absolute intensity and records only edge-direction distribution, so it
    /// survives lighting change and moderate warps. It costs 4x the bytes and
    /// more compute per keypoint; both are cheap next to the response map.
    public var descriptorKind: DescriptorKind

    public init(maxFeatures: Int = 2000, relativeThreshold: Float = 0.01, nmsRadius: Int = 4,
                spatialBuckets: Int = 8, noiseFloorFraction: Float = 0.0005,
                pyramidLevels: Int = 4, descriptorKind: DescriptorKind = .sift) {
        self.descriptorKind = descriptorKind
        self.maxFeatures = maxFeatures
        self.relativeThreshold = relativeThreshold
        self.nmsRadius = nmsRadius
        self.spatialBuckets = spatialBuckets
        self.noiseFloorFraction = noiseFloorFraction
        self.pyramidLevels = pyramidLevels
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
        let r = max(1, options.nmsRadius)

        // PER-CELL thresholds, not one global cutoff.
        //
        // A single `relativeThreshold * globalMax` cutoff means the frame's
        // most contrasty object decides what counts as a feature everywhere.
        // Observed on a real capture: one ornately engraved brass tray set the
        // global maximum so high that a fully textured room — wood floor,
        // furniture, patterned fabric — fell entirely below the floor and
        // contributed almost no features, leaving the reconstruction to rely
        // on a single small, near-planar object.
        //
        // Scaling the threshold to each cell's own maximum lets every region
        // contribute its best corners on its own terms, which is what spreads
        // features across the frame. (Spatial bucketing alone does not do this:
        // it only rebalances once the maxFeatures cap binds, and if the
        // threshold already rejected a region there is nothing left to
        // rebalance.)
        let grid = max(1, options.spatialBuckets)
        var cellMax = [Float](repeating: 0, count: grid * grid)
        for y in r..<(height - r) {
            let row = y * width
            let cy = min(grid - 1, y * grid / height)
            for x in r..<(width - r) {
                let value = response[row + x]
                let cx = min(grid - 1, x * grid / width)
                let cell = cy * grid + cx
                if value > cellMax[cell] { cellMax[cell] = value }
            }
        }
        let noiseFloor = maxResponse * options.noiseFloorFraction
        var cellThreshold = [Float](repeating: 0, count: grid * grid)
        for i in 0..<(grid * grid) {
            cellThreshold[i] = max(cellMax[i] * options.relativeThreshold, noiseFloor)
        }

        var candidates: [(index: Int, x: Int, y: Int, value: Float)] = []
        for y in r..<(height - r) {
            let row = y * width
            let cy = min(grid - 1, y * grid / height)
            for x in r..<(width - r) {
                let value = response[row + x]
                let cx = min(grid - 1, x * grid / width)
                guard value >= cellThreshold[cy * grid + cx] else { continue }
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
        func rank(_ a: (index: Int, x: Int, y: Int, value: Float),
                  _ b: (index: Int, x: Int, y: Int, value: Float)) -> Bool {
            let ka = (a.value * sortScale).rounded()
            let kb = (b.value * sortScale).rounded()
            return ka != kb ? ka > kb : a.index < b.index
        }
        candidates.sort(by: rank)

        if candidates.count > options.maxFeatures {
            let grid = options.spatialBuckets
            if grid > 1 {
                // Round-robin over grid cells: take the strongest remaining
                // feature from each occupied cell in turn, so coverage spreads
                // before strength dominates. Cells are visited in a fixed
                // order and each cell's list is already rank-sorted, so the
                // result stays deterministic and cross-tier-stable.
                var cells: [Int: [Int]] = [:]   // cell -> candidate indices, strongest first
                for (i, candidate) in candidates.enumerated() {
                    let cx = min(grid - 1, candidate.x * grid / width)
                    let cy = min(grid - 1, candidate.y * grid / height)
                    cells[cy * grid + cx, default: []].append(i)
                }
                var cursor = [Int: Int]()
                var chosen: [Int] = []
                chosen.reserveCapacity(options.maxFeatures)
                let cellOrder = cells.keys.sorted()
                var exhausted = false
                while chosen.count < options.maxFeatures && !exhausted {
                    exhausted = true
                    for cell in cellOrder {
                        guard chosen.count < options.maxFeatures else { break }
                        let list = cells[cell]!
                        let position = cursor[cell] ?? 0
                        guard position < list.count else { continue }
                        chosen.append(list[position])
                        cursor[cell] = position + 1
                        exhausted = false
                    }
                }
                // Restore strongest-first ordering across the selected set.
                candidates = chosen.sorted().map { candidates[$0] }
                candidates.sort(by: rank)
            } else {
                candidates.removeSubrange(options.maxFeatures...)
            }
        }

        return candidates.map { candidate in
            let angle = orientation(luma: luma, width: width, height: height, x: candidate.x, y: candidate.y)
            return Keypoint(x: Float(candidate.x), y: Float(candidate.y), response: candidate.value, angle: angle)
        }
    }

    /// 2x2 box downsample. Exact and separable-free; at these sizes the cost is
    /// irrelevant next to the Harris pass that consumes the result.
    public static func downsample(luma: [Float], width: Int, height: Int)
        -> (luma: [Float], width: Int, height: Int) {
        let w = width / 2, h = height / 2
        guard w > 0, h > 0 else { return ([], 0, 0) }
        var out = [Float](repeating: 0, count: w * h)
        out.withUnsafeMutableBufferPointer { dst in
            luma.withUnsafeBufferPointer { src in
                for y in 0..<h {
                    let row0 = (y * 2) * width
                    let row1 = row0 + width
                    let outRow = y * w
                    for x in 0..<w {
                        let x0 = x * 2
                        dst[outRow + x] = (src[row0 + x0] + src[row0 + x0 + 1]
                                         + src[row1 + x0] + src[row1 + x0 + 1]) * 0.25
                    }
                }
            }
        }
        return (out, w, h)
    }

    /// Detect and describe across a scale pyramid.
    ///
    /// `responseProvider` is the only tier-specific part — it returns the
    /// Harris response map for a luma buffer at a given size. Everything else
    /// (level construction, per-level selection, orientation, descriptors,
    /// coordinate mapping, the global feature budget) is shared, so the tiers
    /// cannot diverge in any of it.
    ///
    /// The feature budget is split across levels by AREA, matching the number
    /// of distinguishable corners each level can actually support: a level with
    /// a quarter the pixels gets roughly a quarter the allocation. Without
    /// that, coarse levels — where responses are stronger because downsampling
    /// suppresses noise — would crowd out fine detail.
    public static func extractPyramid(
        frameIndex: Int, luma: [Float], width: Int, height: Int,
        options: FeatureOptions,
        responseProvider: (_ luma: [Float], _ width: Int, _ height: Int) throws -> [Float]
    ) rethrows -> FeatureSet {
        let levelCount = max(1, options.pyramidLevels)
        var levels: [(luma: [Float], width: Int, height: Int)] = [(luma, width, height)]
        for _ in 1..<levelCount {
            guard let last = levels.last, last.width >= 64, last.height >= 64 else { break }
            let next = downsample(luma: last.luma, width: last.width, height: last.height)
            guard next.width > 0 else { break }
            levels.append(next)
        }

        let totalArea = levels.reduce(0.0) { $0 + Double($1.width * $1.height) }
        var keypoints: [Keypoint] = []
        var descriptors: [UInt8] = []

        for (octave, level) in levels.enumerated() {
            let share = totalArea > 0 ? Double(level.width * level.height) / totalArea : 1
            var levelOptions = options
            levelOptions.maxFeatures = max(16, Int((Double(options.maxFeatures) * share).rounded()))
            // NMS radius is in level pixels; keeping it constant would make
            // coarse levels suppress far more of the original image than fine
            // ones, so it shrinks with the level (floored at 2).
            levelOptions.nmsRadius = max(2, options.nmsRadius >> octave)

            let response = try responseProvider(level.luma, level.width, level.height)
            let levelKeypoints = selectKeypoints(
                response: response, width: level.width, height: level.height,
                luma: level.luma, options: levelOptions
            )
            let scale = Float(1 << octave)
            for keypoint in levelKeypoints {
                // Descriptor is sampled on THIS level, so it covers a patch
                // `scale` times larger in original pixels — which is exactly
                // what gives scale invariance.
                switch options.descriptorKind {
                case .brief:
                    descriptors.append(contentsOf: describe(
                        luma: level.luma, width: level.width, height: level.height, keypoint: keypoint))
                case .sift:
                    descriptors.append(contentsOf: SIFTDescriptor.describe(
                        luma: level.luma, width: level.width, height: level.height, keypoint: keypoint))
                }
                // Map back to full-resolution coordinates. The +0.5 centring
                // accounts for a level pixel covering a `scale`-wide block.
                keypoints.append(Keypoint(
                    x: (keypoint.x + 0.5) * scale - 0.5,
                    y: (keypoint.y + 0.5) * scale - 0.5,
                    response: keypoint.response, angle: keypoint.angle,
                    octave: octave, scale: scale
                ))
            }
        }
        // Levels were appended in order, so the merged list is only sorted
        // WITHIN each level. Re-sort globally to preserve the strongest-first
        // contract callers rely on, using the same quantized key and
        // deterministic tiebreak as single-level selection.
        //
        // Cross-level responses are not strictly comparable (downsampling
        // averages gradients, so coarse levels score lower), which in practice
        // means fine detail sorts first. That is the right bias: the per-level
        // budget has already guaranteed each level its share, so this only
        // decides presentation order, not which features survive.
        if !keypoints.isEmpty {
            let maxResponse = keypoints.map { $0.response }.max() ?? 1
            let sortScale = maxResponse > 0 ? Float(1 << 20) / maxResponse : 1
            let order = keypoints.indices.sorted { a, b in
                let ka = (keypoints[a].response * sortScale).rounded()
                let kb = (keypoints[b].response * sortScale).rounded()
                if ka != kb { return ka > kb }
                if keypoints[a].octave != keypoints[b].octave { return keypoints[a].octave < keypoints[b].octave }
                if keypoints[a].y != keypoints[b].y { return keypoints[a].y < keypoints[b].y }
                return keypoints[a].x < keypoints[b].x
            }
            let bytes = options.descriptorKind.byteCount
            var sortedKeypoints = [Keypoint](); sortedKeypoints.reserveCapacity(order.count)
            var sortedDescriptors = [UInt8](); sortedDescriptors.reserveCapacity(descriptors.count)
            for index in order {
                sortedKeypoints.append(keypoints[index])
                sortedDescriptors.append(contentsOf: descriptors[(index * bytes)..<((index + 1) * bytes)])
            }
            keypoints = sortedKeypoints
            descriptors = sortedDescriptors
        }
        return FeatureSet(frameIndex: frameIndex, keypoints: keypoints,
                          descriptors: descriptors, kind: options.descriptorKind)
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
