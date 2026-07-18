import Foundation

// MARK: - Stage 3: two-view geometry
//
// Recovers relative pose from matched features. Pure math over correspondences
// — no GPU, no tier split. It runs once per image pair on a few hundred
// matches, which is nothing next to the per-pixel work in stages 2 and 5.

public struct TwoViewResult {
    /// Pose of camera 2 relative to camera 1 (camera 1 is the identity).
    /// Translation is a UNIT vector: two views alone cannot determine scale.
    public let pose: CameraPose
    /// Indices into the input match array that agreed with the model.
    public let inliers: [Int]
    /// Points triangulated from the inliers, in camera-1 (world) coordinates.
    public let points: [SIMD3<Double>]
    /// Parallel to `points`: which input match produced each.
    public let pointMatchIndices: [Int]
}

public struct TwoViewOptions {
    /// Sampson-distance inlier threshold, in pixels.
    public var inlierThresholdPixels: Double
    /// RANSAC iteration cap.
    public var maxIterations: Int
    /// Stop early once the inlier ratio implies this confidence.
    public var confidence: Double
    /// Deterministic RANSAC seed. Reconstruction must be reproducible run to
    /// run, so sampling never draws from a system RNG.
    public var seed: UInt64
    /// Minimum triangulation angle (degrees) for a point to be kept. Points
    /// seen along nearly the same ray from both cameras have depth that is
    /// numerically meaningless — they explode to huge coordinates and poison
    /// bundle adjustment.
    public var minTriangulationAngleDegrees: Double

    public init(inlierThresholdPixels: Double = 2.0, maxIterations: Int = 2000,
                confidence: Double = 0.9999, seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
                minTriangulationAngleDegrees: Double = 1.0) {
        self.inlierThresholdPixels = inlierThresholdPixels
        self.maxIterations = maxIterations
        self.confidence = confidence
        self.seed = seed
        self.minTriangulationAngleDegrees = minTriangulationAngleDegrees
    }
}

public enum TwoViewGeometry {

    /// Estimate relative pose from matched keypoints.
    ///
    /// Pipeline: normalize by intrinsics -> RANSAC over the 8-point essential
    /// matrix using Sampson distance -> recover (R, t) by cheirality ->
    /// re-triangulate the final inlier set.
    ///
    /// KNOWN LIMITATION — planar scenes. The 8-point algorithm is degenerate
    /// when the observed points are dominated by a single plane: the epipolar
    /// constraint no longer determines E uniquely, so RANSAC happily reports a
    /// large consistent inlier set while the recovered pose is wrong and most
    /// points fail the cheirality test. Observed on a plane-dominant rendered
    /// scene: 99 inliers, of which 4 triangulated. The standard fix is to fit
    /// BOTH a homography and a fundamental/essential matrix, choose between
    /// them with a model-selection score (ORB-SLAM's approach), and recover
    /// pose from the homography when the scene is planar. Not implemented yet;
    /// captures that orbit a real object with depth variation are unaffected.
    ///
    /// The 8-point algorithm is used rather than the minimal 5-point solver.
    /// 5-point needs fewer samples per RANSAC hypothesis (so it tolerates more
    /// outliers) and handles pure rotation better, but it requires solving a
    /// 10th-degree polynomial — several hundred lines whose failure modes are
    /// subtle and hard to test. With the cross-checked, ratio-tested matches
    /// coming out of the matcher, outlier rates are low enough that 8-point
    /// with enough iterations converges reliably. If pure-rotation or
    /// high-outlier captures become a problem, 5-point is the upgrade.
    public static func estimate(
        matches: [FeatureMatch],
        keypoints1: [Keypoint], keypoints2: [Keypoint],
        intrinsics1: CameraIntrinsics, intrinsics2: CameraIntrinsics,
        options: TwoViewOptions = TwoViewOptions()
    ) -> TwoViewResult? {
        guard matches.count >= 8 else { return nil }

        // Normalized (calibration-removed) correspondences.
        var p1 = [SIMD2<Double>](), p2 = [SIMD2<Double>]()
        p1.reserveCapacity(matches.count)
        p2.reserveCapacity(matches.count)
        for match in matches {
            guard match.queryIndex < keypoints1.count, match.trainIndex < keypoints2.count else { return nil }
            let a = keypoints1[match.queryIndex], b = keypoints2[match.trainIndex]
            p1.append(intrinsics1.normalize(x: Double(a.x), y: Double(a.y)))
            p2.append(intrinsics2.normalize(x: Double(b.x), y: Double(b.y)))
        }

        // Threshold is given in pixels but Sampson distance is computed in
        // normalized units, so convert once using the mean focal length.
        let meanFocal = (intrinsics1.focalLength + intrinsics2.focalLength) / 2
        let threshold = options.inlierThresholdPixels / meanFocal

        var rng = SplitMix64(seed: options.seed)
        var bestInliers: [Int] = []
        var iterations = options.maxIterations
        var iteration = 0

        while iteration < iterations {
            defer { iteration += 1 }
            // Sample 8 distinct correspondences.
            var indices = Set<Int>()
            var guard_ = 0
            while indices.count < 8 && guard_ < 100 {
                indices.insert(Int(rng.next() % UInt64(matches.count)))
                guard_ += 1
            }
            guard indices.count == 8 else { continue }
            let sample = Array(indices)

            guard let e = essentialFromEightPoints(
                p1: sample.map { p1[$0] }, p2: sample.map { p2[$0] }
            ) else { continue }

            var inliers: [Int] = []
            for i in 0..<p1.count where sampsonDistance(e: e, x1: p1[i], x2: p2[i]) < threshold {
                inliers.append(i)
            }
            if inliers.count > bestInliers.count {
                bestInliers = inliers
                // Adaptive iteration count: once a good model is found, the
                // remaining budget can shrink dramatically.
                let ratio = Double(inliers.count) / Double(p1.count)
                if ratio > 0 {
                    let denominator = log(max(1e-12, 1 - pow(ratio, 8)))
                    if denominator < 0 {
                        let needed = Int(log(1 - options.confidence) / denominator) + 1
                        iterations = min(iterations, max(needed, 20))
                    }
                }
            }
        }

        guard bestInliers.count >= 8 else { return nil }

        // Refit on all inliers — the minimal sample only had to find them.
        guard let refined = essentialFromEightPoints(
            p1: bestInliers.map { p1[$0] }, p2: bestInliers.map { p2[$0] }
        ) else { return nil }

        guard let (pose, points, keptIndices) = recoverPose(
            e: refined,
            p1: bestInliers.map { p1[$0] },
            p2: bestInliers.map { p2[$0] },
            minAngleDegrees: options.minTriangulationAngleDegrees
        ) else { return nil }

        return TwoViewResult(
            pose: pose,
            inliers: bestInliers,
            points: points,
            pointMatchIndices: keptIndices.map { bestInliers[$0] }
        )
    }

    /// Normalized 8-point algorithm on already-calibrated points.
    ///
    /// Hartley normalization (centroid to origin, mean distance sqrt(2)) is
    /// applied even though the points are calibration-normalized: the
    /// constraint matrix is built from products of coordinates, so forming
    /// AᵀA squares an already-poor condition number. Without this the solution
    /// degrades visibly.
    static func essentialFromEightPoints(p1: [SIMD2<Double>], p2: [SIMD2<Double>]) -> [Double]? {
        guard p1.count >= 8, p1.count == p2.count else { return nil }
        guard let (n1, t1) = hartleyNormalize(p1), let (n2, t2) = hartleyNormalize(p2) else { return nil }

        var rows: [[Double]] = []
        rows.reserveCapacity(n1.count)
        for i in 0..<n1.count {
            let a = n1[i], b = n2[i]
            // Epipolar constraint x2ᵀ E x1 = 0, expanded into the 9 unknowns.
            rows.append([
                b.x * a.x, b.x * a.y, b.x,
                b.y * a.x, b.y * a.y, b.y,
                a.x,       a.y,       1,
            ])
        }
        let solution = LinearAlgebra.smallestSingularVector(rows: rows)
        guard solution.count == 9 else { return nil }

        // Undo normalization: E = T2ᵀ Ê T1
        let eHat = solution
        let e = LinearAlgebra.matMul3(LinearAlgebra.transpose3(t2), LinearAlgebra.matMul3(eHat, t1))

        // Project onto the essential-matrix manifold: a valid E has singular
        // values (s, s, 0). The linear solution above does not respect that,
        // and skipping this step is a classic source of a subtly wrong pose.
        let (u, _, vt) = LinearAlgebra.svd3x3(e)
        let d: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 0]
        return LinearAlgebra.matMul3(u, LinearAlgebra.matMul3(d, vt))
    }

    /// Translate to centroid and scale so mean distance from origin is sqrt(2).
    /// Returns the transformed points and the 3x3 transform applied.
    static func hartleyNormalize(_ points: [SIMD2<Double>]) -> ([SIMD2<Double>], [Double])? {
        guard !points.isEmpty else { return nil }
        var cx = 0.0, cy = 0.0
        for p in points { cx += p.x; cy += p.y }
        cx /= Double(points.count); cy /= Double(points.count)

        var meanDistance = 0.0
        for p in points {
            let dx = p.x - cx, dy = p.y - cy
            meanDistance += (dx * dx + dy * dy).squareRoot()
        }
        meanDistance /= Double(points.count)
        guard meanDistance > 1e-12 else { return nil }
        let scale = 2.0.squareRoot() / meanDistance

        let transformed = points.map { SIMD2<Double>(($0.x - cx) * scale, ($0.y - cy) * scale) }
        let transform: [Double] = [
            scale, 0, -scale * cx,
            0, scale, -scale * cy,
            0, 0, 1,
        ]
        return (transformed, transform)
    }

    /// First-order geometric error of the epipolar constraint.
    /// Preferred over the raw algebraic residual x2ᵀEx1, which is biased by
    /// how far a point sits from the epipole.
    static func sampsonDistance(e: [Double], x1: SIMD2<Double>, x2: SIMD2<Double>) -> Double {
        let a = SIMD3<Double>(x1.x, x1.y, 1)
        let b = SIMD3<Double>(x2.x, x2.y, 1)
        let ea = LinearAlgebra.matVec3(e, a)
        let etb = LinearAlgebra.matVec3(LinearAlgebra.transpose3(e), b)
        let numerator = b.x * ea.x + b.y * ea.y + b.z * ea.z
        let denominator = ea.x * ea.x + ea.y * ea.y + etb.x * etb.x + etb.y * etb.y
        guard denominator > 1e-18 else { return .infinity }
        return abs(numerator) / denominator.squareRoot()
    }

    /// Decompose E into the physically valid (R, t) and triangulate.
    ///
    /// E admits four (R, t) combinations — two rotations times a sign on the
    /// translation — of which exactly one puts points in front of BOTH
    /// cameras. That cheirality test is the disambiguation.
    static func recoverPose(
        e: [Double], p1: [SIMD2<Double>], p2: [SIMD2<Double>], minAngleDegrees: Double
    ) -> (CameraPose, [SIMD3<Double>], [Int])? {
        let (u, _, vt) = LinearAlgebra.svd3x3(e)

        let w: [Double] = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        var r1 = LinearAlgebra.matMul3(u, LinearAlgebra.matMul3(w, vt))
        var r2 = LinearAlgebra.matMul3(u, LinearAlgebra.matMul3(LinearAlgebra.transpose3(w), vt))

        // U and V are only determined up to sign, so U·W·Vᵀ can come out as a
        // reflection (det = −1). The fix is to NEGATE THE WHOLE MATRIX — for
        // 3x3, det(−R) = −det(R), so this restores a proper rotation while
        // preserving the epipolar geometry.
        //
        // What must NOT happen here is running a reflection through
        // `nearestRotation`: that projects onto SO(3) by flipping a single
        // column, which turns the reflection into a *different, valid-looking*
        // rotation roughly 180 degrees from the truth. It then passes the
        // cheirality test on most points and produces a confidently wrong,
        // mirrored reconstruction. (Measured before this fix: essential matrix
        // correct with 120/120 inliers, yet rotation error exactly 180.0000
        // deg and translation exactly 90 deg off.)
        if LinearAlgebra.determinant3(r1) < 0 { r1 = r1.map { -$0 } }
        if LinearAlgebra.determinant3(r2) < 0 { r2 = r2.map { -$0 } }

        let t = SIMD3<Double>(u[2], u[5], u[8])   // third column of U
        let candidates: [(rotation: [Double], translation: SIMD3<Double>)] = [
            (r1, t), (r1, SIMD3<Double>(-t.x, -t.y, -t.z)),
            (r2, t), (r2, SIMD3<Double>(-t.x, -t.y, -t.z)),
        ]

        let camera1 = CameraPose.identity
        var best: (pose: CameraPose, points: [SIMD3<Double>], indices: [Int])?

        for candidate in candidates {
            // No re-orthogonalization here: the rotation is already a proper
            // rotation by construction (orthogonal U, W, V with the sign fixed
            // above), and projecting it again could only move it off the
            // correct answer.
            let pose = CameraPose(rotation: candidate.rotation, translation: candidate.translation)
            var points: [SIMD3<Double>] = []
            var indices: [Int] = []
            for i in 0..<p1.count {
                guard let x = triangulate(p1: p1[i], p2: p2[i], pose1: camera1, pose2: pose) else { continue }
                // In front of both cameras.
                guard x.z > 0, pose.transform(x).z > 0 else { continue }
                // Sufficient parallax.
                guard triangulationAngleDegrees(point: x, pose1: camera1, pose2: pose) >= minAngleDegrees else { continue }
                points.append(x)
                indices.append(i)
            }
            if best == nil || points.count > best!.points.count {
                best = (pose, points, indices)
            }
        }
        guard let result = best, !result.points.isEmpty else { return nil }
        return (result.pose, result.points, result.indices)
    }

    /// Angle at the 3D point between the rays to the two camera centres.
    static func triangulationAngleDegrees(point: SIMD3<Double>, pose1: CameraPose, pose2: CameraPose) -> Double {
        let c1 = pose1.center, c2 = pose2.center
        let a = SIMD3<Double>(point.x - c1.x, point.y - c1.y, point.z - c1.z)
        let b = SIMD3<Double>(point.x - c2.x, point.y - c2.y, point.z - c2.z)
        let la = LinearAlgebra.length(a), lb = LinearAlgebra.length(b)
        guard la > 1e-12, lb > 1e-12 else { return 0 }
        let cosine = (a.x * b.x + a.y * b.y + a.z * b.z) / (la * lb)
        return acos(max(-1, min(1, cosine))) * 180 / .pi
    }

    /// Linear (DLT) triangulation from two normalized observations.
    /// Each view contributes two rows of the form (x * P_row3 - P_row1); the
    /// 3D point is the null vector of the stacked system.
    public static func triangulate(
        p1: SIMD2<Double>, p2: SIMD2<Double>, pose1: CameraPose, pose2: CameraPose
    ) -> SIMD3<Double>? {
        func projectionRows(_ pose: CameraPose, _ p: SIMD2<Double>) -> [[Double]] {
            let r = pose.rotation, t = pose.translation
            let row0 = [r[0], r[1], r[2], t.x]
            let row1 = [r[3], r[4], r[5], t.y]
            let row2 = [r[6], r[7], r[8], t.z]
            return [
                (0..<4).map { p.x * row2[$0] - row0[$0] },
                (0..<4).map { p.y * row2[$0] - row1[$0] },
            ]
        }
        let rows = projectionRows(pose1, p1) + projectionRows(pose2, p2)
        let solution = LinearAlgebra.smallestSingularVector(rows: rows)
        guard solution.count == 4, abs(solution[3]) > 1e-12 else { return nil }
        let w = solution[3]
        let point = SIMD3<Double>(solution[0] / w, solution[1] / w, solution[2] / w)
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { return nil }
        return point
    }
}
