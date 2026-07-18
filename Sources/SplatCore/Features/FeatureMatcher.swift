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
