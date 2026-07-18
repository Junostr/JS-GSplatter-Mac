import Foundation

// MARK: - Stage 3: bundle adjustment
//
// Refines camera poses and 3D points together by minimizing total reprojection
// error with Levenberg-Marquardt.
//
// Structure of the implementation: this is the classic SPARSE bundle problem,
// where the Hessian has a block-arrowhead shape (cameras couple to points, but
// no camera couples directly to another camera). Rather than build the full
// (6C + 3P) system — which is enormous and mostly zeros — this uses the Schur
// complement: eliminate the point blocks analytically to get a small dense
// system in the camera parameters only, solve that, then back-substitute the
// point updates. For C cameras that is a 6C x 6C solve regardless of how many
// thousands of points there are.
//
// Poses update in the tangent space (a 3-vector axis-angle increment composed
// onto the rotation) rather than by perturbing matrix entries, so rotations
// stay on SO(3) by construction instead of drifting off and needing repair.

public struct BundleAdjustmentOptions {
    public var maxIterations: Int
    /// Stop when a step improves total cost by less than this fraction.
    public var convergenceTolerance: Double
    /// Initial LM damping. Large = gradient-descent-like and safe, small =
    /// Gauss-Newton-like and fast. This is adapted every iteration.
    public var initialDamping: Double
    /// Hold the first camera fixed. A reconstruction has 7 gauge degrees of
    /// freedom (3 rotation, 3 translation, 1 scale); without pinning something
    /// the normal equations are singular and the solve is meaningless.
    public var fixFirstCamera: Bool
    /// Also hold the second camera's translation SCALE by keeping it fixed —
    /// two-view reconstruction defines translation only up to scale.
    public var fixSecondCameraScale: Bool
    /// Reject observations worse than this (pixels) before optimizing.
    ///
    /// Defaults to infinity — i.e. no rejection — and that default is
    /// load-bearing. Filtering on the error of the INITIAL estimate is
    /// backwards: a fresh triangulation can easily start tens of pixels off,
    /// so a tight threshold discards most of the problem and leaves BA
    /// optimizing a small, biased subset while total error gets worse.
    /// (Measured with an 8 px default: 37.46 px RMSE in, 37.62 px out.)
    /// Outlier rejection belongs AFTER a first unfiltered pass, once the
    /// residuals mean something — which is what `refineWithOutlierRejection`
    /// does.
    public var outlierThresholdPixels: Double

    public init(maxIterations: Int = 60, convergenceTolerance: Double = 1e-8,
                initialDamping: Double = 1e-4, fixFirstCamera: Bool = true,
                fixSecondCameraScale: Bool = true, outlierThresholdPixels: Double = .infinity) {
        self.maxIterations = maxIterations
        self.convergenceTolerance = convergenceTolerance
        self.initialDamping = initialDamping
        self.fixFirstCamera = fixFirstCamera
        self.fixSecondCameraScale = fixSecondCameraScale
        self.outlierThresholdPixels = outlierThresholdPixels
    }
}

public struct BundleAdjustmentResult {
    public let initialRMSE: Double
    public let finalRMSE: Double
    public let iterations: Int
    public let converged: Bool
}

public enum BundleAdjustment {

    /// One observation: a 2D measurement of a point in a camera.
    struct Observation {
        let cameraSlot: Int      // index into the ordered camera list
        let pointIndex: Int
        let measured: SIMD2<Double>
    }

    /// Refine `reconstruction` in place.
    ///
    /// `keypoints` maps frame index -> that frame's keypoints, used to look up
    /// the measured image position of each observation.
    @discardableResult
    public static func refine(
        reconstruction: inout Reconstruction,
        keypoints: [Int: [Keypoint]],
        options: BundleAdjustmentOptions = BundleAdjustmentOptions()
    ) -> BundleAdjustmentResult {

        // Deterministic camera ordering: dictionary iteration order is not
        // stable, and an unstable parameter layout would make results
        // irreproducible run to run.
        let cameraFrames = reconstruction.cameras.keys.sorted()
        guard cameraFrames.count >= 2, !reconstruction.points.isEmpty else {
            let rmse = reconstruction.reprojectionRMSE(keypoints: keypoints)
            return BundleAdjustmentResult(initialRMSE: rmse, finalRMSE: rmse, iterations: 0, converged: true)
        }
        var slotOfFrame: [Int: Int] = [:]
        for (slot, frame) in cameraFrames.enumerated() { slotOfFrame[frame] = slot }

        // Gather observations, dropping gross outliers up front so a handful
        // of bad matches cannot dominate a least-squares cost.
        var observations: [Observation] = []
        for (pointIndex, point) in reconstruction.points.enumerated() {
            for observation in point.observations {
                guard let slot = slotOfFrame[observation.frame],
                      let frameKeypoints = keypoints[observation.frame],
                      observation.keypoint < frameKeypoints.count else { continue }
                let kp = frameKeypoints[observation.keypoint]
                let measured = SIMD2<Double>(Double(kp.x), Double(kp.y))
                let camera = reconstruction.cameras[observation.frame]!
                if let projected = camera.pose.project(point.position, intrinsics: camera.intrinsics) {
                    let dx = projected.x - measured.x, dy = projected.y - measured.y
                    if (dx * dx + dy * dy).squareRoot() > options.outlierThresholdPixels { continue }
                }
                observations.append(Observation(cameraSlot: slot, pointIndex: pointIndex, measured: measured))
            }
        }
        guard !observations.isEmpty else {
            let rmse = reconstruction.reprojectionRMSE(keypoints: keypoints)
            return BundleAdjustmentResult(initialRMSE: rmse, finalRMSE: rmse, iterations: 0, converged: false)
        }

        // Which camera slots are free to move.
        let firstFree = options.fixFirstCamera ? 1 : 0
        let freeCameras = Array(firstFree..<cameraFrames.count)
        let cameraParamCount = freeCameras.count * 6

        var poses = cameraFrames.map { reconstruction.cameras[$0]!.pose }
        let intrinsics = cameraFrames.map { reconstruction.cameras[$0]!.intrinsics }
        var points = reconstruction.points.map { $0.position }

        let initialRMSE = cost(observations: observations, poses: poses, intrinsics: intrinsics, points: points).rmse
        var damping = options.initialDamping
        var currentCost = cost(observations: observations, poses: poses, intrinsics: intrinsics, points: points).total
        var iterations = 0
        var converged = false

        for iteration in 0..<options.maxIterations {
            iterations = iteration + 1

            // Accumulators for the Schur complement.
            // U: per-camera 6x6 blocks. V: per-point 3x3 blocks.
            // W: camera-point 6x3 coupling. g: gradient.
            var u = [Double](repeating: 0, count: cameraParamCount * cameraParamCount)
            var gCam = [Double](repeating: 0, count: cameraParamCount)
            var vBlocks = [[Double]](repeating: [Double](repeating: 0, count: 9), count: points.count)
            var gPoint = [SIMD3<Double>](repeating: .zero, count: points.count)
            // W blocks keyed by (pointIndex, freeCameraSlot).
            var wBlocks: [Int: [Int: [Double]]] = [:]

            var freeSlotOf = [Int: Int]()
            for (i, slot) in freeCameras.enumerated() { freeSlotOf[slot] = i }

            for observation in observations {
                let pose = poses[observation.cameraSlot]
                let intr = intrinsics[observation.cameraSlot]
                let point = points[observation.pointIndex]
                let camPoint = pose.transform(point)
                guard camPoint.z > 1e-9 else { continue }

                let invZ = 1.0 / camPoint.z
                let projected = SIMD2<Double>(
                    intr.focalLength * camPoint.x * invZ + intr.cx,
                    intr.focalLength * camPoint.y * invZ + intr.cy
                )
                let residual = SIMD2<Double>(projected.x - observation.measured.x,
                                             projected.y - observation.measured.y)

                // d(projection) / d(camera-space point)
                let f = intr.focalLength
                let dpdX: [Double] = [
                    f * invZ, 0, -f * camPoint.x * invZ * invZ,
                    0, f * invZ, -f * camPoint.y * invZ * invZ,
                ]

                // d(camera-space point) / d(point in world) = R
                var jPoint = [Double](repeating: 0, count: 6)   // 2x3
                for r in 0..<2 {
                    for c in 0..<3 {
                        var sum = 0.0
                        for k in 0..<3 { sum += dpdX[r * 3 + k] * pose.rotation[k * 3 + c] }
                        jPoint[r * 3 + c] = sum
                    }
                }

                // d(camera-space point) / d(pose increment).
                // Translation part is identity; rotation part is -[Xc]× for a
                // left-multiplied axis-angle increment on the rotation.
                var jPose: [Double]? = nil
                if let freeIndex = freeSlotOf[observation.cameraSlot] {
                    let x = camPoint
                    let skew: [Double] = [
                        0, x.z, -x.y,
                        -x.z, 0, x.x,
                        x.y, -x.x, 0,
                    ]
                    var j = [Double](repeating: 0, count: 12)   // 2x6
                    for r in 0..<2 {
                        // rotation columns 0..2
                        for c in 0..<3 {
                            var sum = 0.0
                            for k in 0..<3 { sum += dpdX[r * 3 + k] * skew[k * 3 + c] }
                            j[r * 6 + c] = sum
                        }
                        // translation columns 3..5
                        for c in 0..<3 { j[r * 6 + 3 + c] = dpdX[r * 3 + c] }
                    }
                    jPose = j
                    _ = freeIndex
                }

                // Accumulate V (point block) and point gradient.
                for r in 0..<3 {
                    for c in 0..<3 {
                        vBlocks[observation.pointIndex][r * 3 + c] +=
                            jPoint[0 * 3 + r] * jPoint[0 * 3 + c] + jPoint[1 * 3 + r] * jPoint[1 * 3 + c]
                    }
                }
                let gp = SIMD3<Double>(
                    jPoint[0] * residual.x + jPoint[3] * residual.y,
                    jPoint[1] * residual.x + jPoint[4] * residual.y,
                    jPoint[2] * residual.x + jPoint[5] * residual.y
                )
                gPoint[observation.pointIndex] += gp

                guard let j = jPose, let freeIndex = freeSlotOf[observation.cameraSlot] else { continue }
                let base = freeIndex * 6
                // U block and camera gradient.
                for r in 0..<6 {
                    for c in 0..<6 {
                        u[(base + r) * cameraParamCount + (base + c)] +=
                            j[0 * 6 + r] * j[0 * 6 + c] + j[1 * 6 + r] * j[1 * 6 + c]
                    }
                    gCam[base + r] += j[0 * 6 + r] * residual.x + j[1 * 6 + r] * residual.y
                }
                // W coupling block (6x3).
                var w = wBlocks[observation.pointIndex]?[freeIndex] ?? [Double](repeating: 0, count: 18)
                for r in 0..<6 {
                    for c in 0..<3 {
                        w[r * 3 + c] += j[0 * 6 + r] * jPoint[0 * 3 + c] + j[1 * 6 + r] * jPoint[1 * 3 + c]
                    }
                }
                wBlocks[observation.pointIndex, default: [:]][freeIndex] = w
            }

            // Schur complement: S = U - Σ_p W_p V_p⁻¹ W_pᵀ,  b = g_cam - Σ_p W_p V_p⁻¹ g_p
            var s = u
            var b = gCam.map { -$0 }
            var vInverses = [[Double]?](repeating: nil, count: points.count)

            for pointIndex in 0..<points.count {
                var v = vBlocks[pointIndex]
                guard v[0] != 0 || v[4] != 0 || v[8] != 0 else { continue }
                for d in 0..<3 { v[d * 3 + d] *= (1 + damping) }
                guard let vInv = invert3x3(v) else { continue }
                vInverses[pointIndex] = vInv

                guard let cameraBlocks = wBlocks[pointIndex] else { continue }
                // W V⁻¹ g_p  and  W V⁻¹ Wᵀ
                let g = gPoint[pointIndex]
                let vInvG = LinearAlgebra.matVec3(vInv, g)
                for (freeIndex, w) in cameraBlocks {
                    let base = freeIndex * 6
                    for r in 0..<6 {
                        let contribution = w[r * 3 + 0] * vInvG.x + w[r * 3 + 1] * vInvG.y + w[r * 3 + 2] * vInvG.z
                        b[base + r] += contribution
                    }
                    for (otherIndex, w2) in cameraBlocks {
                        let otherBase = otherIndex * 6
                        for r in 0..<6 {
                            // (W V⁻¹)_r = row r of W times V⁻¹
                            var wv = [Double](repeating: 0, count: 3)
                            for c in 0..<3 {
                                wv[c] = w[r * 3 + 0] * vInv[0 * 3 + c]
                                      + w[r * 3 + 1] * vInv[1 * 3 + c]
                                      + w[r * 3 + 2] * vInv[2 * 3 + c]
                            }
                            for c in 0..<6 {
                                let value = wv[0] * w2[c * 3 + 0] + wv[1] * w2[c * 3 + 1] + wv[2] * w2[c * 3 + 2]
                                s[(base + r) * cameraParamCount + (otherBase + c)] -= value
                            }
                        }
                    }
                }
            }

            // LM damping on the camera block.
            for d in 0..<cameraParamCount {
                s[d * cameraParamCount + d] *= (1 + damping)
                s[d * cameraParamCount + d] += 1e-12
            }

            guard cameraParamCount == 0 || LinearAlgebra.solveSPD(s, b, n: cameraParamCount) != nil else {
                damping *= 10
                if damping > 1e12 { break }
                continue
            }
            let deltaCam = cameraParamCount > 0
                ? LinearAlgebra.solveSPD(s, b, n: cameraParamCount)!
                : []

            // Back-substitute point updates:  Δp = V⁻¹ (-g_p - Wᵀ Δc)
            var candidatePoints = points
            for pointIndex in 0..<points.count {
                guard let vInv = vInverses[pointIndex] else { continue }
                var rhs = SIMD3<Double>(-gPoint[pointIndex].x, -gPoint[pointIndex].y, -gPoint[pointIndex].z)
                if let cameraBlocks = wBlocks[pointIndex] {
                    for (freeIndex, w) in cameraBlocks {
                        let base = freeIndex * 6
                        guard base + 5 < deltaCam.count else { continue }
                        for c in 0..<3 {
                            var sum = 0.0
                            for r in 0..<6 { sum += w[r * 3 + c] * deltaCam[base + r] }
                            if c == 0 { rhs.x -= sum } else if c == 1 { rhs.y -= sum } else { rhs.z -= sum }
                        }
                    }
                }
                let delta = LinearAlgebra.matVec3(vInv, rhs)
                candidatePoints[pointIndex] = points[pointIndex] + delta
            }

            // Apply camera updates on a trial basis.
            var candidatePoses = poses
            for (i, slot) in freeCameras.enumerated() {
                let base = i * 6
                guard base + 5 < deltaCam.count else { continue }
                let omega = SIMD3<Double>(deltaCam[base], deltaCam[base + 1], deltaCam[base + 2])
                var translationDelta = SIMD3<Double>(deltaCam[base + 3], deltaCam[base + 4], deltaCam[base + 5])
                // Gauge fix for the free global scale. Only the component of
                // the update ALONG the current baseline changes its length, so
                // that one component is projected out and the direction stays
                // free to improve. Zeroing the whole translation would pin
                // three degrees of freedom to remove one, and would stop the
                // second camera from correcting its direction at all.
                if options.fixSecondCameraScale && slot == 1 {
                    let t = candidatePoses[slot].translation
                    let n = LinearAlgebra.length(t)
                    if n > 1e-12 {
                        let unit = SIMD3<Double>(t.x / n, t.y / n, t.z / n)
                        let along = translationDelta.x * unit.x + translationDelta.y * unit.y + translationDelta.z * unit.z
                        translationDelta -= SIMD3<Double>(unit.x * along, unit.y * along, unit.z * along)
                    }
                }
                var pose = candidatePoses[slot].rotated(byAxisAngle: omega)
                pose.translation = pose.translation + translationDelta
                candidatePoses[slot] = pose
            }

            let candidateCost = cost(observations: observations, poses: candidatePoses,
                                     intrinsics: intrinsics, points: candidatePoints).total
            if candidateCost < currentCost {
                let improvement = (currentCost - candidateCost) / max(currentCost, 1e-30)
                poses = candidatePoses
                points = candidatePoints
                currentCost = candidateCost
                damping = max(damping * 0.3, 1e-12)
                if improvement < options.convergenceTolerance {
                    converged = true
                    break
                }
            } else {
                // Step rejected: increase damping and retry.
                damping *= 10
                if damping > 1e12 { break }
            }
        }

        for (slot, frame) in cameraFrames.enumerated() {
            reconstruction.cameras[frame]?.pose = poses[slot]
        }
        for i in 0..<points.count {
            reconstruction.points[i].position = points[i]
        }

        let finalRMSE = cost(observations: observations, poses: poses, intrinsics: intrinsics, points: points).rmse
        return BundleAdjustmentResult(initialRMSE: initialRMSE, finalRMSE: finalRMSE,
                                      iterations: iterations, converged: converged)
    }

    /// Two-pass refinement: optimize on everything, then drop observations
    /// that are still bad once the residuals are meaningful, then re-optimize.
    /// This is the ordering that makes outlier rejection safe — see the note
    /// on `outlierThresholdPixels`.
    @discardableResult
    public static func refineWithOutlierRejection(
        reconstruction: inout Reconstruction,
        keypoints: [Int: [Keypoint]],
        options: BundleAdjustmentOptions = BundleAdjustmentOptions(),
        rejectionThresholdPixels: Double = 4.0
    ) -> BundleAdjustmentResult {
        var firstPass = options
        firstPass.outlierThresholdPixels = .infinity
        let initial = refine(reconstruction: &reconstruction, keypoints: keypoints, options: firstPass)

        // Drop individual bad observations, then any point left with fewer
        // than two views — a point seen once is not constrained and would
        // wander freely in the next pass.
        for pointIndex in reconstruction.points.indices {
            let position = reconstruction.points[pointIndex].position
            reconstruction.points[pointIndex].observations.removeAll { observation in
                guard let camera = reconstruction.cameras[observation.frame],
                      let frameKeypoints = keypoints[observation.frame],
                      observation.keypoint < frameKeypoints.count else { return true }
                guard let projected = camera.pose.project(position, intrinsics: camera.intrinsics) else { return true }
                let kp = frameKeypoints[observation.keypoint]
                let dx = projected.x - Double(kp.x), dy = projected.y - Double(kp.y)
                return (dx * dx + dy * dy).squareRoot() > rejectionThresholdPixels
            }
        }
        reconstruction.points.removeAll { $0.observations.count < 2 }
        guard !reconstruction.points.isEmpty else { return initial }

        var secondPass = options
        secondPass.outlierThresholdPixels = .infinity
        let final = refine(reconstruction: &reconstruction, keypoints: keypoints, options: secondPass)
        return BundleAdjustmentResult(
            initialRMSE: initial.initialRMSE, finalRMSE: final.finalRMSE,
            iterations: initial.iterations + final.iterations, converged: final.converged
        )
    }

    static func cost(observations: [Observation], poses: [CameraPose],
                     intrinsics: [CameraIntrinsics], points: [SIMD3<Double>]) -> (total: Double, rmse: Double) {
        var total = 0.0
        var count = 0
        for observation in observations {
            let pose = poses[observation.cameraSlot]
            guard let projected = pose.project(points[observation.pointIndex],
                                               intrinsics: intrinsics[observation.cameraSlot]) else { continue }
            let dx = projected.x - observation.measured.x
            let dy = projected.y - observation.measured.y
            total += dx * dx + dy * dy
            count += 1
        }
        return (total, count > 0 ? (total / Double(count)).squareRoot() : 0)
    }

    static func invert3x3(_ m: [Double]) -> [Double]? {
        let det = LinearAlgebra.determinant3(m)
        guard abs(det) > 1e-18 else { return nil }
        let invDet = 1 / det
        return [
            (m[4] * m[8] - m[5] * m[7]) * invDet,
            (m[2] * m[7] - m[1] * m[8]) * invDet,
            (m[1] * m[5] - m[2] * m[4]) * invDet,
            (m[5] * m[6] - m[3] * m[8]) * invDet,
            (m[0] * m[8] - m[2] * m[6]) * invDet,
            (m[2] * m[3] - m[0] * m[5]) * invDet,
            (m[3] * m[7] - m[4] * m[6]) * invDet,
            (m[1] * m[6] - m[0] * m[7]) * invDet,
            (m[0] * m[4] - m[1] * m[3]) * invDet,
        ]
    }
}
