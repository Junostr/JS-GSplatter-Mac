import Foundation

// MARK: - Stage 3: absolute pose (PnP)
//
// Registers a NEW camera against already-triangulated 3D points. Two-view
// geometry bootstraps the reconstruction; this is how every camera after the
// first two joins it.

public struct PnPOptions {
    public var inlierThresholdPixels: Double
    public var maxIterations: Int
    public var seed: UInt64
    public var minInliers: Int

    public init(inlierThresholdPixels: Double = 4.0, maxIterations: Int = 500,
                seed: UInt64 = 0xBF58_476D_1CE4_E5B9, minInliers: Int = 12) {
        self.inlierThresholdPixels = inlierThresholdPixels
        self.maxIterations = maxIterations
        self.seed = seed
        self.minInliers = minInliers
    }
}

public struct PnPResult {
    public let pose: CameraPose
    /// Indices into the input correspondence arrays that agreed.
    public let inliers: [Int]
}

public enum PoseEstimation {

    /// Estimate a camera pose from 3D-2D correspondences, with RANSAC.
    ///
    /// Uses the linear DLT formulation (6-point minimum) rather than a minimal
    /// P3P solver. P3P samples only 3 points per hypothesis so it tolerates
    /// more outliers, but it needs a quartic solve plus disambiguation. Here
    /// the 3D points come from an already-verified two-view reconstruction and
    /// the 2D matches are cross-checked, so the outlier rate is low, and the
    /// pose is immediately refined by bundle adjustment anyway. DLT keeps the
    /// failure modes inspectable.
    public static func estimatePose(
        worldPoints: [SIMD3<Double>], imagePoints: [SIMD2<Double>],
        intrinsics: CameraIntrinsics, options: PnPOptions = PnPOptions()
    ) -> PnPResult? {
        guard worldPoints.count == imagePoints.count, worldPoints.count >= 6 else { return nil }

        // Work in normalized coordinates so the DLT is well-conditioned and
        // the inlier threshold converts cleanly.
        let normalized = imagePoints.map { intrinsics.normalize(x: $0.x, y: $0.y) }
        let threshold = options.inlierThresholdPixels / intrinsics.focalLength

        var rng = SplitMix64(seed: options.seed)
        var bestInliers: [Int] = []
        var bestPose: CameraPose?

        for _ in 0..<options.maxIterations {
            var indices = Set<Int>()
            var attempts = 0
            while indices.count < 6 && attempts < 100 {
                indices.insert(Int(rng.next() % UInt64(worldPoints.count)))
                attempts += 1
            }
            guard indices.count == 6 else { continue }
            let sample = Array(indices)
            guard let pose = poseFromDLT(
                worldPoints: sample.map { worldPoints[$0] },
                normalized: sample.map { normalized[$0] }
            ) else { continue }

            var inliers: [Int] = []
            for i in 0..<worldPoints.count {
                let camera = pose.transform(worldPoints[i])
                guard camera.z > 1e-9 else { continue }
                let dx = camera.x / camera.z - normalized[i].x
                let dy = camera.y / camera.z - normalized[i].y
                if (dx * dx + dy * dy).squareRoot() < threshold { inliers.append(i) }
            }
            if inliers.count > bestInliers.count {
                bestInliers = inliers
                bestPose = pose
            }
        }

        guard bestInliers.count >= options.minInliers, bestPose != nil else { return nil }
        // Refit on the full inlier set.
        if let refined = poseFromDLT(
            worldPoints: bestInliers.map { worldPoints[$0] },
            normalized: bestInliers.map { normalized[$0] }
        ) {
            var inliers: [Int] = []
            for i in 0..<worldPoints.count {
                let camera = refined.transform(worldPoints[i])
                guard camera.z > 1e-9 else { continue }
                let dx = camera.x / camera.z - normalized[i].x
                let dy = camera.y / camera.z - normalized[i].y
                if (dx * dx + dy * dy).squareRoot() < threshold { inliers.append(i) }
            }
            if inliers.count >= bestInliers.count {
                return PnPResult(pose: refined, inliers: inliers)
            }
        }
        return PnPResult(pose: bestPose!, inliers: bestInliers)
    }

    /// Direct linear transform for the 3x4 projection matrix [R | t], then
    /// project the rotation block back onto SO(3).
    static func poseFromDLT(worldPoints: [SIMD3<Double>], normalized: [SIMD2<Double>]) -> CameraPose? {
        guard worldPoints.count >= 6 else { return nil }

        // Condition the 3D points (translate to centroid, scale to mean
        // distance sqrt(3)) — the DLT matrix mixes coordinate products, so
        // raw world coordinates make AᵀA badly conditioned.
        var centroid = SIMD3<Double>(0, 0, 0)
        for p in worldPoints { centroid += p }
        centroid /= Double(worldPoints.count)
        var meanDistance = 0.0
        for p in worldPoints { meanDistance += LinearAlgebra.length(p - centroid) }
        meanDistance /= Double(worldPoints.count)
        guard meanDistance > 1e-12 else { return nil }
        let scale = 3.0.squareRoot() / meanDistance
        let conditioned = worldPoints.map { ($0 - centroid) * scale }

        var rows: [[Double]] = []
        rows.reserveCapacity(worldPoints.count * 2)
        for i in 0..<worldPoints.count {
            let p = conditioned[i]
            let x = normalized[i].x, y = normalized[i].y
            rows.append([p.x, p.y, p.z, 1, 0, 0, 0, 0, -x * p.x, -x * p.y, -x * p.z, -x])
            rows.append([0, 0, 0, 0, p.x, p.y, p.z, 1, -y * p.x, -y * p.y, -y * p.z, -y])
        }
        let solution = LinearAlgebra.smallestSingularVector(rows: rows)
        guard solution.count == 12 else { return nil }

        var rotation = [
            solution[0], solution[1], solution[2],
            solution[4], solution[5], solution[6],
            solution[8], solution[9], solution[10],
        ]
        var translation = SIMD3<Double>(solution[3], solution[7], solution[11])

        // The DLT solution is defined up to sign; choose the one that puts the
        // conditioned points in front of the camera.
        let testPoint = conditioned[0]
        let testDepth = rotation[6] * testPoint.x + rotation[7] * testPoint.y + rotation[8] * testPoint.z + translation.z
        if testDepth < 0 {
            rotation = rotation.map { -$0 }
            translation = SIMD3<Double>(-translation.x, -translation.y, -translation.z)
        }

        // Recover scale from the rotation block, which must be orthonormal.
        let rowNorm = LinearAlgebra.length(SIMD3<Double>(rotation[0], rotation[1], rotation[2]))
        guard rowNorm > 1e-12 else { return nil }
        translation /= rowNorm
        let properRotation = LinearAlgebra.nearestRotation(rotation)

        // Undo the conditioning: the pose was solved for p' = (p - c) * s, so
        // t must absorb the centroid shift and the scale.
        //   Xc = R·(p − c)·s + t'   =>   Xc = R·p + (t'/s − R·c)
        let tWorld = translation / scale - LinearAlgebra.matVec3(properRotation, centroid)
        guard tWorld.x.isFinite, tWorld.y.isFinite, tWorld.z.isFinite else { return nil }
        return CameraPose(rotation: properRotation, translation: tWorld)
    }
}
