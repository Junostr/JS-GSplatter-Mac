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

    /// Multipliers of the longer image side to try.
    ///
    /// Spans roughly 90 degrees horizontal FOV (0.5) down to 30 degrees (1.9),
    /// which covers ultrawide phone lenses through short telephoto. Sampling
    /// is denser in the 0.6-1.0 region where phone and action cameras actually
    /// sit, since that is where most captures land and where the penalty for
    /// being wrong is steepest.
    public static let defaultMultipliers: [Double] = [
        0.50, 0.58, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        0.95, 1.00, 1.10, 1.20, 1.35, 1.55, 1.90,
    ]

    /// Estimate focal length from the match graph.
    ///
    /// `pairs` should be a handful of well-matched frame pairs; sampling a few
    /// rather than all of them keeps this to a fraction of a second, and the
    /// signal is strong enough that more pairs do not change the answer.
    public static func estimate(
        pairs: [(keypoints1: [Keypoint], keypoints2: [Keypoint], matches: [FeatureMatch])],
        imageWidth: Int, imageHeight: Int,
        multipliers: [Double] = defaultMultipliers
    ) -> FocalEstimate? {
        guard !pairs.isEmpty else { return nil }
        let longSide = Double(max(imageWidth, imageHeight))
        let cx = Double(imageWidth) / 2, cy = Double(imageHeight) / 2

        var best: FocalEstimate?
        for multiplier in multipliers {
            let focal = multiplier * longSide
            let intrinsics = CameraIntrinsics(focalLength: focal, cx: cx, cy: cy)
            var totalPoints = 0
            var errors: [Double] = []

            for pair in pairs {
                guard pair.matches.count >= 16 else { continue }
                guard let result = TwoViewGeometry.estimate(
                    matches: pair.matches,
                    keypoints1: pair.keypoints1, keypoints2: pair.keypoints2,
                    intrinsics1: intrinsics, intrinsics2: intrinsics
                ) else { continue }
                totalPoints += result.points.count

                // Reprojection error of the triangulated points in BOTH views.
                // Point count alone can be gamed by a degenerate model that
                // places everything at a plausible depth; requiring the points
                // to also reproject tightly is what makes the score honest.
                for (i, matchIndex) in result.pointMatchIndices.enumerated() {
                    let match = pair.matches[matchIndex]
                    guard match.queryIndex < pair.keypoints1.count,
                          match.trainIndex < pair.keypoints2.count else { continue }
                    let point = result.points[i]
                    if let p1 = CameraPose.identity.project(point, intrinsics: intrinsics) {
                        let kp = pair.keypoints1[match.queryIndex]
                        errors.append(((p1.x - Double(kp.x)) * (p1.x - Double(kp.x))
                                     + (p1.y - Double(kp.y)) * (p1.y - Double(kp.y))).squareRoot())
                    }
                    if let p2 = result.pose.project(point, intrinsics: intrinsics) {
                        let kp = pair.keypoints2[match.trainIndex]
                        errors.append(((p2.x - Double(kp.x)) * (p2.x - Double(kp.x))
                                     + (p2.y - Double(kp.y)) * (p2.y - Double(kp.y))).squareRoot())
                    }
                }
            }

            guard totalPoints > 0, !errors.isEmpty else { continue }
            let median = errors.sorted()[errors.count / 2]
            let candidate = FocalEstimate(
                focalLength: focal, supportingPoints: totalPoints,
                medianReprojectionError: median, candidatesTried: multipliers.count
            )
            // Prefer more surviving structure; break near-ties on lower
            // reprojection error. The 5% band keeps a marginally larger point
            // count from beating a clearly better-fitting model.
            if let current = best {
                let betterCount = Double(candidate.supportingPoints) > Double(current.supportingPoints) * 1.05
                let comparableCount = Double(candidate.supportingPoints) >= Double(current.supportingPoints) * 0.95
                if betterCount || (comparableCount && candidate.medianReprojectionError < current.medianReprojectionError) {
                    best = candidate
                }
            } else {
                best = candidate
            }
        }
        return best
    }

    /// Convenience: pick the best-matched pairs out of a feature set list and
    /// estimate from those.
    public static func estimate(
        featureSets: [FeatureSet], imageWidth: Int, imageHeight: Int,
        samplePairs: Int = 3, gaps: [Int] = [1, 2]
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
