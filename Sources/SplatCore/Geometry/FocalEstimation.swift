import Foundation

// MARK: - Stage 3: focal length estimation
//
// Why this exists: the focal length guess dominates real-world reconstruction
// quality, and no fixed heuristic covers the range of cameras this app has to
// accept.
//
// The previous default was 1.2 x the longest image side (COLMAP's fallback,
// aimed at DSLR-ish photos), which implies roughly a 45 degree horizontal
// field of view. Phone main cameras shoot far wider — an iPhone at 4K is about
// 69 degrees, i.e. ~0.72 x width. Measured on a real 3840x2160 iPhone capture:
//
//   focal 4608 (the 1.2x guess) -> 120 inliers, 14.29 px RMSE,  0 points
//   focal 3200                  -> 301 inliers,  1.02 px RMSE, 125 points
//   focal 2770 (~0.72 x width)  -> 298 inliers,  0.60 px RMSE, 263 points
//
// A wrong focal does not merely scale the result: the essential matrix is only
// an essential matrix under correct calibration, so RANSAC fits a model that
// is not physically realizable, cheirality rejects most points, and bundle
// adjustment then deletes what little survives. It fails confidently rather
// than obviously, which is the worst failure mode.
//
// Rather than swap one heuristic for another, this sweeps candidate focals and
// lets the geometry decide, scoring each by how much structure actually
// survives cheirality and the triangulation-angle test. That is self-
// correcting for any camera, including ones with no usable metadata (video
// almost never carries EXIF focal length).

public struct FocalEstimate {
    public let focalLength: Double
    /// Points that survived triangulation at this focal, summed over the
    /// sampled pairs — the score that selected it.
    public let supportingPoints: Int
    /// Median reprojection error of those points, in pixels.
    public let medianReprojectionError: Double
    public let candidatesTried: Int
}

public enum FocalEstimation {

    /// Candidate scoring criteria, exposed so the synthetic harness can plot
    /// each one's curve and confirm it actually has a minimum at the true
    /// focal before it is trusted on real footage.
    public enum Criterion: String {
        /// Count of triangulated points (the original). Threshold-biased.
        case pointCount
        /// Median Sampson epipolar error, in pixels, over ALL matches.
        /// Threshold-free: no gate, so nothing to bias the score, and the
        /// median tolerates up to 50% outliers.
        case medianEpipolarError
        /// Singular-value asymmetry of E = KᵀFK for an F fitted once from
        /// pixel coordinates.
        case asymmetry
    }

    /// Score one calibration under one criterion. Lower is better for
    /// `medianEpipolarError` and `asymmetry`; higher is better for
    /// `pointCount` (negated here so every criterion minimises).
    public static func score(
        criterion: Criterion,
        fundamental: [Double]?,
        pixels1: [SIMD2<Double>], pixels2: [SIMD2<Double>],
        intrinsics: CameraIntrinsics
    ) -> Double? {
        switch criterion {
        case .asymmetry:
            guard let f = fundamental else { return nil }
            return TwoViewGeometry.calibrationAsymmetry(fundamental: f, intrinsics: intrinsics)
        case .medianEpipolarError:
            guard let f = fundamental else { return nil }
            // Sampson distance under the essential matrix implied by this K,
            // measured in PIXELS over every match. No inlier gate is involved,
            // so no threshold policy can tilt the comparison.
            let k: [Double] = [intrinsics.focalLength, 0, intrinsics.cx,
                               0, intrinsics.focalLength, intrinsics.cy,
                               0, 0, 1]
            let kInv: [Double] = [1 / intrinsics.focalLength, 0, -intrinsics.cx / intrinsics.focalLength,
                                  0, 1 / intrinsics.focalLength, -intrinsics.cy / intrinsics.focalLength,
                                  0, 0, 1]
            let e = LinearAlgebra.matMul3(LinearAlgebra.transpose3(k), LinearAlgebra.matMul3(f, k))
            // Force the essential structure (s, s, 0), then map back to pixels.
            let (u, _, vt) = LinearAlgebra.svd3x3(e)
            let d: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 0]
            let eIdeal = LinearAlgebra.matMul3(u, LinearAlgebra.matMul3(d, vt))
            let fIdeal = LinearAlgebra.matMul3(LinearAlgebra.transpose3(kInv),
                                               LinearAlgebra.matMul3(eIdeal, kInv))
            var errors: [Double] = []
            errors.reserveCapacity(pixels1.count)
            for i in 0..<pixels1.count {
                let d = TwoViewGeometry.sampsonDistance(e: fIdeal, x1: pixels1[i], x2: pixels2[i])
                if d.isFinite { errors.append(d) }
            }
            guard !errors.isEmpty else { return nil }
            // Divide by the candidate focal so the curve is measured in
            // NORMALIZED units rather than pixels.
            //
            // Pixel error scales with focal, which makes the curve steeply
            // asymmetric — measured on clean synthetic data it runs 2.88 at
            // 0.5x but 21.65 at 1.9x around a true 0.80x. Any noise floor then
            // slides the minimum toward the flatter low side, producing a
            // systematic UNDERestimate (measured: -19% to -23% across three
            // true focals, all biased the same direction). Normalizing removes
            // the scale term the asymmetry comes from.
            return errors.sorted()[errors.count / 2] / intrinsics.focalLength
        case .pointCount:
            return nil   // handled by the existing path
        }
    }

    /// Multipliers of the longer image side to try.
    ///
    /// Spans roughly 90 degrees horizontal FOV (0.5) down to 30 degrees (1.9),
    /// which covers ultrawide phone lenses through short telephoto. Sampling
    /// is denser in the 0.6-1.0 region where phone and action cameras actually
    /// sit, since that is where most captures land and where the penalty for
    /// being wrong is steepest.
    /// Extended down to 0.30 (~117 degrees horizontal) to cover ULTRA-WIDE
    /// phone lenses, not just main cameras.
    ///
    /// The old range started at 0.50 and a real capture kept landing exactly
    /// there — which looked like a broken estimator hitting a boundary. It may
    /// instead have been a correct estimate clipped by too narrow a sweep: an
    /// iPhone ultra-wide is around 13 mm equivalent, roughly 0.38x the long
    /// side, well below the old floor. A sweep that cannot express the answer
    /// guarantees a boundary result no matter how good the criterion is.
    public static let defaultMultipliers: [Double] = [
        0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.58, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.35, 1.55, 1.90,
    ]

    /// Estimate focal length by AGGREGATING score curves across many pairs.
    ///
    /// The single-pair criterion is correct but weak. Measured on synthetic
    /// pairs with a known 0.80x focal: with clean keypoints both the epipolar
    /// error and the essential-matrix asymmetry form a sharp V with its minimum
    /// exactly at the truth, but at 1.5 px of localisation noise the error curve
    /// varies only 4.4-4.7 px across the ENTIRE focal range — a 7% spread. The
    /// minimum is then decided by noise, not geometry, which is why one capture
    /// returned 0.50x, 0.58x, 0.75x, 0.85x and 0.90x depending only on which
    /// frames happened to be sampled.
    ///
    /// That is an observability problem, not a scoring bug, and no cleverer
    /// single-pair criterion fixes it. The answer is more evidence: each pair
    /// contributes a weak but independently-noisy curve, so normalising and
    /// summing them lets the shared true minimum reinforce while the noise
    /// averages out.
    ///
    /// Curves are normalised by their own median before summing. Absolute error
    /// scale varies hugely between pairs (a wide-baseline pair has larger
    /// residuals everywhere), so without normalisation a few high-error pairs
    /// would dominate the sum and the aggregate would just track them.
    public static func estimate(
        pairs: [(keypoints1: [Keypoint], keypoints2: [Keypoint], matches: [FeatureMatch])],
        imageWidth: Int, imageHeight: Int,
        multipliers: [Double] = defaultMultipliers
    ) -> FocalEstimate? {
        guard !pairs.isEmpty else { return nil }
        let longSide = Double(max(imageWidth, imageHeight))
        let cx = Double(imageWidth) / 2, cy = Double(imageHeight) / 2

        var aggregate = [Double](repeating: 0, count: multipliers.count)
        var contributing = 0
        var totalSupport = 0

        for pair in pairs {
            guard pair.matches.count >= 16 else { continue }
            // FINEST-OCTAVE MATCHES ONLY.
            //
            // Focal accuracy is limited purely by keypoint localisation noise
            // (measured: exact recovery at 0.5 px, collapsing toward a
            // curve-shape prior of ~0.70-0.75x by 3 px, for every true focal).
            // A corner detected on pyramid level k has its coordinates
            // multiplied by 2^k to reach full resolution, so its localisation
            // error is multiplied too — an octave-2 keypoint carries roughly 4x
            // the positional error of an octave-0 one. Feeding those into the
            // estimator is self-inflicted noise.
            //
            // The pyramid earns its place in MATCHING, where scale invariance
            // is what makes a correspondence findable at all. It has no such
            // role here: the estimator needs precision, not recall, and there
            // are plenty of octave-0 matches to work with.
            var px1: [SIMD2<Double>] = [], px2: [SIMD2<Double>] = []
            var coarse1: [SIMD2<Double>] = [], coarse2: [SIMD2<Double>] = []
            for match in pair.matches {
                guard match.queryIndex < pair.keypoints1.count,
                      match.trainIndex < pair.keypoints2.count else { continue }
                let a = pair.keypoints1[match.queryIndex], b = pair.keypoints2[match.trainIndex]
                if a.octave == 0 && b.octave == 0 {
                    px1.append(SIMD2<Double>(Double(a.x), Double(a.y)))
                    px2.append(SIMD2<Double>(Double(b.x), Double(b.y)))
                } else {
                    coarse1.append(SIMD2<Double>(Double(a.x), Double(a.y)))
                    coarse2.append(SIMD2<Double>(Double(b.x), Double(b.y)))
                }
            }
            // Fall back to everything only if the fine matches alone are too
            // few to fit — a noisy estimate beats none.
            if px1.count < 16 {
                px1.append(contentsOf: coarse1)
                px2.append(contentsOf: coarse2)
            }
            guard px1.count >= 16,
                  let f = TwoViewGeometry.fundamentalRANSAC(p1: px1, p2: px2) else { continue }
            // Score only on the F inliers: gross mismatches carry no calibration
            // information and would flatten the curve further.
            let inlier1 = f.inliers.map { px1[$0] }
            let inlier2 = f.inliers.map { px2[$0] }
            guard inlier1.count >= 12 else { continue }

            var curve = [Double](repeating: 0, count: multipliers.count)
            var usable = true
            for (i, multiplier) in multipliers.enumerated() {
                let intrinsics = CameraIntrinsics(focalLength: multiplier * longSide, cx: cx, cy: cy)
                guard let value = score(criterion: .medianEpipolarError, fundamental: f.matrix,
                                        pixels1: inlier1, pixels2: inlier2, intrinsics: intrinsics),
                      value.isFinite else { usable = false; break }
                curve[i] = value
            }
            guard usable else { continue }
            let median = curve.sorted()[curve.count / 2]
            guard median > 1e-12 else { continue }
            for i in curve.indices { aggregate[i] += curve[i] / median }
            contributing += 1
            totalSupport += inlier1.count
        }

        if ProcessInfo.processInfo.environment["SPLAT_FOCAL_DEBUG"] != nil {
            var line = "FOCALCURVE pairs=\(contributing) "
            for (i, m) in multipliers.enumerated() {
                line += String(format: "%.2f:%.4f ", m, aggregate[i] / Double(max(1, contributing)))
            }
            FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
        }
        guard contributing > 0 else { return nil }
        var bestIndex = 0
        for i in aggregate.indices where aggregate[i] < aggregate[bestIndex] { bestIndex = i }

        // A minimum at either END of the sweep is not a measurement — it means
        // the curve is monotonic over the whole range and the criterion never
        // discriminated. Returning the boundary value would dress a failure up
        // as a confident answer, which is exactly how the old estimator did
        // damage.
        //
        // This is the real-capture case, and the curve makes it plain:
        //   0.50:0.289  0.58:0.473  0.65:0.628 ... 1.55:1.404  1.90:1.477
        // strictly increasing end to end, with no interior minimum to find. On
        // clean synthetic data the same criterion forms a sharp V at the truth,
        // so the geometry is recoverable in principle; on this footage the
        // residual is dominated by terms that scale with the calibration
        // transform rather than with fit quality, and the signal is buried.
        //
        // Reporting nil lets the caller fall back to a documented prior, which
        // is worth more than a number the data does not support.
        if bestIndex == 0 || bestIndex == multipliers.count - 1 { return nil }

        let focal = multipliers[bestIndex] * longSide
        return FocalEstimate(
            focalLength: focal,
            supportingPoints: totalSupport,
            medianReprojectionError: aggregate[bestIndex] / Double(contributing),

            candidatesTried: multipliers.count
        )
    }

    /// Convenience: pick the best-matched pairs out of a feature set list and
    /// estimate from those.
    public static func estimate(
        featureSets: [FeatureSet], imageWidth: Int, imageHeight: Int,
        samplePairs: Int = 24, gaps: [Int] = [1, 2, 3]
    ) -> FocalEstimate? {
        guard featureSets.count >= 2 else { return nil }
        let ordered = featureSets.sorted { $0.frameIndex < $1.frameIndex }
        var pairs: [(keypoints1: [Keypoint], keypoints2: [Keypoint], matches: [FeatureMatch])] = []

        // These feature sets have already been through stage-2 selection, so
        // CONSECUTIVE entries are typically tens of original frames apart and
        // already carry real baseline. Striding further on top of that
        // overshoots: on a 17 s handheld 4K clip, a gap of 4 selected frames
        // is an ~80-frame jump and matching returns almost nothing, so
        // estimation silently found no usable pair and fell back to the bad
        // guess. Small gaps first, larger only as a fallback.
        outer: for gap in gaps {
            guard gap < ordered.count else { continue }
            let step = max(1, (ordered.count - gap) / max(1, samplePairs))
            var i = 0
            while i + gap < ordered.count {
                let a = ordered[i], b = ordered[i + gap]
                let matches = FeatureMatcher.match(query: a, train: b)
                if matches.count >= 16 {
                    pairs.append((a.keypoints, b.keypoints, matches))
                    if pairs.count >= samplePairs { break outer }
                }
                i += step
            }
        }
        guard !pairs.isEmpty else { return nil }
        return estimate(pairs: pairs, imageWidth: imageWidth, imageHeight: imageHeight)
    }
}
