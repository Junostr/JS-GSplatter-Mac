import Foundation

// MARK: - Stage 3: descriptor matching
//
// Pure logic over descriptors, so it is tier-independent and unit-testable
// without a GPU (same split as FrameSelector in stage 2). Binary descriptors
// make this cheap: Hamming distance is xor + popcount, which is a single
// hardware instruction per 64 bits on both arm64 and x86_64.

public struct FeatureMatch: Equatable {
    /// Index into the query frame's keypoints.
    public let queryIndex: Int
    /// Index into the train frame's keypoints.
    public let trainIndex: Int
    /// Hamming distance between the two descriptors (0…256).
    public let distance: Int

    public init(queryIndex: Int, trainIndex: Int, distance: Int) {
        self.queryIndex = queryIndex
        self.trainIndex = trainIndex
        self.distance = distance
    }
}

public struct MatchOptions {
    /// Lowe's ratio test: keep a match only if the best candidate is
    /// meaningfully closer than the second-best. This is what separates a
    /// distinctive correspondence from one that matches many similar patches
    /// equally well (repeated texture, foliage, brickwork). 0.8 is Lowe's
    /// published value and rejects ~90% of false matches for ~5% of true ones.
    public var ratioThreshold: Float
    /// Require the match to be mutually best in both directions. Catches the
    /// many-to-one failure the ratio test alone permits, at the cost of a
    /// second pass.
    public var crossCheck: Bool
    /// Hard ceiling on Hamming distance (0…256). Beyond ~30% of the bits
    /// differing, two BRIEF descriptors carry essentially no shared signal.
    public var maxDistance: Int

    public init(ratioThreshold: Float = 0.8, crossCheck: Bool = true, maxDistance: Int = 80) {
        self.ratioThreshold = ratioThreshold
        self.crossCheck = crossCheck
        self.maxDistance = maxDistance
    }
}

public enum FeatureMatcher {

    /// Hamming distance between two 256-bit descriptors.
    @inline(__always)
    public static func hamming(_ a: ArraySlice<UInt8>, _ b: ArraySlice<UInt8>) -> Int {
        var distance = 0
        var i = a.startIndex
        var j = b.startIndex
        while i < a.endIndex {
            distance += (a[i] ^ b[j]).nonzeroBitCount
            i += 1
            j += 1
        }
        return distance
    }

    /// Match two frames whose poses are already known, using epipolar geometry
    /// to constrain the search.
    ///
    /// Unguided matching has to be conservative: with a few thousand candidates
    /// and no geometric context, Lowe's ratio test is the only defence against
    /// false matches, and it rejects a great many correct ones — precisely the
    /// ones where a surface has changed appearance, which is exactly what
    /// happens at the wide viewpoint changes late in an orbit.
    ///
    /// Once both cameras are posed, a feature in one image must lie on its
    /// epipolar line in the other. That single constraint eliminates almost
    /// every accidental descriptor collision, so the ratio test can be relaxed
    /// and matches that were previously discarded as ambiguous become usable.
    /// This is the standard way to densify correspondences after registration.
    ///
    /// Only candidates within `epipolarThresholdPixels` of the line are
    /// considered, which also makes it cheaper than brute force despite being
    /// more permissive on descriptors.
    public static func matchGuided(
        query: FeatureSet, train: FeatureSet,
        queryPose: CameraPose, trainPose: CameraPose,
        queryIntrinsics: CameraIntrinsics, trainIntrinsics: CameraIntrinsics,
        options: MatchOptions = MatchOptions(ratioThreshold: 0.95, crossCheck: false, maxDistance: 100),
        epipolarThresholdPixels: Double = 4.0
    ) -> [FeatureMatch] {
        guard query.count > 0, train.count > 0 else { return [] }

        // Relative pose train <- query, then E = [t]x R in normalized coords.
        let rQuery = queryPose.rotation, rTrain = trainPose.rotation
        let rRel = LinearAlgebra.matMul3(rTrain, LinearAlgebra.transpose3(rQuery))
        let rotatedQueryT = LinearAlgebra.matVec3(rRel, queryPose.translation)
        let tRel = trainPose.translation - rotatedQueryT
        let tx: [Double] = [
            0, -tRel.z, tRel.y,
            tRel.z, 0, -tRel.x,
            -tRel.y, tRel.x, 0,
        ]
        let e = LinearAlgebra.matMul3(tx, rRel)

        // Pre-normalize the train keypoints once.
        let trainNormalized = train.keypoints.map {
            trainIntrinsics.normalize(x: Double($0.x), y: Double($0.y))
        }
        // Threshold converted to normalized units via the train focal length.
        let epipolarThreshold = epipolarThresholdPixels / trainIntrinsics.focalLength

        var matches: [FeatureMatch] = []
        matches.reserveCapacity(query.count)
        var claimed = [Bool](repeating: false, count: train.count)

        for q in 0..<query.count {
            let kp = query.keypoints[q]
            let a = queryIntrinsics.normalize(x: Double(kp.x), y: Double(kp.y))
            let line = LinearAlgebra.matVec3(e, SIMD3<Double>(a.x, a.y, 1))
            let norm = (line.x * line.x + line.y * line.y).squareRoot()
            guard norm > 1e-12 else { continue }

            let qd = query.descriptor(at: q)
            var best = Int.max, secondBest = Int.max, bestIndex = -1
            for t in 0..<train.count where !claimed[t] {
                let b = trainNormalized[t]
                let distanceToLine = abs(b.x * line.x + b.y * line.y + line.z) / norm
                guard distanceToLine < epipolarThreshold else { continue }
                let distance = hamming(qd, train.descriptor(at: t))
                if distance < best {
                    secondBest = best
                    best = distance
                    bestIndex = t
                } else if distance < secondBest {
                    secondBest = distance
                }
            }
            guard bestIndex >= 0, best <= options.maxDistance else { continue }
            // The ratio test still runs, but only among candidates that already
            // satisfy the epipolar constraint — so it is discriminating between
            // genuinely plausible correspondences rather than the whole image.
            if secondBest < Int.max {
                guard Float(best) < options.ratioThreshold * Float(secondBest) else { continue }
            }
            // One-to-one: a train feature cannot serve two query features.
            claimed[bestIndex] = true
            matches.append(FeatureMatch(queryIndex: q, trainIndex: bestIndex, distance: best))
        }
        return matches
    }

    /// Brute-force match `query` against `train`.
    ///
    /// O(n·m) by design. For the frame counts stage 2 hands us (a few hundred
    /// frames, ~2000 features each) an exact search is well under a second per
    /// pair and avoids the approximate-index accuracy loss that would
    /// propagate into pose estimation. If pair counts grow, the escape hatch
    /// is a GPU kernel over the same exact computation, not an approximate
    /// index — correctness here is worth more than speed.
    public static func match(
        query: FeatureSet, train: FeatureSet, options: MatchOptions = MatchOptions()
    ) -> [FeatureMatch] {
        guard query.count > 0, train.count > 0 else { return [] }

        var forward = [FeatureMatch]()
        forward.reserveCapacity(query.count)

        for q in 0..<query.count {
            let qd = query.descriptor(at: q)
            var best = Int.max, secondBest = Int.max, bestIndex = -1
            for t in 0..<train.count {
                let distance = hamming(qd, train.descriptor(at: t))
                if distance < best {
                    secondBest = best
                    best = distance
                    bestIndex = t
                } else if distance < secondBest {
                    secondBest = distance
                }
            }
            guard bestIndex >= 0, best <= options.maxDistance else { continue }
            // With only one train feature there is no second-best to compare
            // against; accept on the distance ceiling alone rather than
            // dividing by Int.max and rejecting everything.
            if train.count > 1 {
                guard Float(best) < options.ratioThreshold * Float(secondBest) else { continue }
            }
            forward.append(FeatureMatch(queryIndex: q, trainIndex: bestIndex, distance: best))
        }

        guard options.crossCheck else { return forward }

        // Reverse pass: for each train feature find its best query feature,
        // then keep only mutually-best pairs.
        var bestQueryForTrain = [Int](repeating: -1, count: train.count)
        var bestDistanceForTrain = [Int](repeating: Int.max, count: train.count)
        for t in 0..<train.count {
            let td = train.descriptor(at: t)
            for q in 0..<query.count {
                let distance = hamming(td, query.descriptor(at: q))
                if distance < bestDistanceForTrain[t] {
                    bestDistanceForTrain[t] = distance
                    bestQueryForTrain[t] = q
                }
            }
        }
        return forward.filter { bestQueryForTrain[$0.trainIndex] == $0.queryIndex }
    }
}
