import Foundation

// MARK: - Stage 3: incremental structure from motion
//
// Ties the pieces together: match graph -> feature tracks -> two-view
// initialization -> incremental PnP registration -> triangulation -> bundle
// adjustment. Output is camera poses plus a sparse point cloud, which is what
// stage 5 needs to initialize gaussians.

public struct SfMOptions {
    /// Match each frame against this many following frames. Captures are
    /// sequential (a video or an ordered photo walk-around), so an exhaustive
    /// O(n²) match is wasted work — but the window must exceed 1 so the
    /// reconstruction has redundancy and loop-ish overlap to lean on.
    public var matchWindow: Int
    public var maxFeaturesPerFrame: Int
    /// Minimum two-view inliers for a pair to be usable at all.
    public var minPairInliers: Int
    /// Minimum median triangulation angle (degrees) for the INITIAL pair.
    /// Picking a pair with a large inlier count but tiny baseline is the
    /// classic way to start a reconstruction that never recovers: the
    /// initial points are numerically meaningless and everything registers
    /// against them.
    public var minInitialAngleDegrees: Double
    public var bundleAdjust: Bool

    public init(matchWindow: Int = 6, maxFeaturesPerFrame: Int = 1500,
                minPairInliers: Int = 30, minInitialAngleDegrees: Double = 1.2,
                bundleAdjust: Bool = true) {
        self.matchWindow = matchWindow
        self.maxFeaturesPerFrame = maxFeaturesPerFrame
        self.minPairInliers = minPairInliers
        self.minInitialAngleDegrees = minInitialAngleDegrees
        self.bundleAdjust = bundleAdjust
    }
}

public struct SfMReport {
    public let registeredCameras: Int
    public let totalCameras: Int
    public let points: Int
    public let initialPair: (Int, Int)?
    public let rmseBefore: Double
    public let rmseAfter: Double
}

public enum StructureFromMotion {

    /// A feature track: one physical 3D point seen in several frames.
    struct Track {
        var observations: [Int: Int] = [:]   // frameIndex -> keypointIndex
        var pointIndex: Int? = nil           // index into Reconstruction.points once triangulated
    }

    public static func reconstruct(
        featureSets: [FeatureSet],
        intrinsics: [Int: CameraIntrinsics],
        options: SfMOptions = SfMOptions(),
        log: ((String) -> Void)? = nil
    ) -> (reconstruction: Reconstruction, report: SfMReport)? {
        guard featureSets.count >= 2 else { return nil }
        let setsByFrame = Dictionary(uniqueKeysWithValues: featureSets.map { ($0.frameIndex, $0) })
        let frames = featureSets.map { $0.frameIndex }.sorted()
        var keypointsByFrame: [Int: [Keypoint]] = [:]
        for set in featureSets { keypointsByFrame[set.frameIndex] = set.keypoints }

        // 1. Match within a sliding window.
        var pairMatches: [Pair: [FeatureMatch]] = [:]
        for (i, frameA) in frames.enumerated() {
            for offset in 1...max(1, options.matchWindow) {
                let j = i + offset
                guard j < frames.count else { break }
                let frameB = frames[j]
                guard let a = setsByFrame[frameA], let b = setsByFrame[frameB] else { continue }
                let matches = FeatureMatcher.match(query: a, train: b)
                if matches.count >= 8 { pairMatches[Pair(frameA, frameB)] = matches }
            }
        }
        guard !pairMatches.isEmpty else { return nil }
        log?("matched \(pairMatches.count) frame pairs")

        // 2. Build tracks by union-find over (frame, keypoint) observations.
        var unionFind = UnionFind()
        for (pair, matches) in pairMatches {
            for match in matches {
                unionFind.union(Observation(frame: pair.a, keypoint: match.queryIndex),
                                Observation(frame: pair.b, keypoint: match.trainIndex))
            }
        }
        var tracks: [Observation: Track] = [:]
        // Sorted iteration, not dictionary order. When a track collects two
        // keypoints from the same frame the rule below keeps whichever is seen
        // first — so an arbitrary iteration order would make track CONTENTS
        // vary run to run, and could drop the exact keypoint the initial pair
        // needs to seed its points.
        for observation in unionFind.parent.keys.sorted(by: {
            ($0.frame, $0.keypoint) < ($1.frame, $1.keypoint)
        }) {
            let root = unionFind.find(observation)
            var track = tracks[root] ?? Track()
            // A track must not contain two keypoints from the SAME frame:
            // that means the match graph merged distinct physical points
            // (common with repeated texture). Dropping the conflict keeps the
            // track usable rather than poisoning it.
            if track.observations[observation.frame] == nil {
                track.observations[observation.frame] = observation.keypoint
            }
            tracks[root] = track
        }
        tracks = tracks.filter { $0.value.observations.count >= 2 }
        log?("built \(tracks.count) tracks")

        // 3. Pick the initial pair: most two-view inliers, subject to enough
        //    parallax.
        // Score candidates on TRIANGULATED POINT COUNT, not match count, and
        // evaluate a bounded set of them rather than stopping at the first
        // that qualifies.
        //
        // Sorting by match count and taking the first qualifying pair is
        // actively wrong: adjacent frames always have the most matches and the
        // least parallax, so that rule reliably seeds the reconstruction on a
        // near-zero baseline. Measured on a real pan: it chose frames 84 and
        // 86, whose 33 verified inliers yielded exactly 1 point that survived
        // the triangulation-angle test — and the whole reconstruction then
        // failed with 2/14 cameras registered.
        //
        // `result.points` is already filtered by cheirality and minimum
        // triangulation angle, so its size is a direct measure of how much
        // usable structure a pair actually provides.
        var bestPair: (pair: Pair, result: TwoViewResult)?
        var bestScore = -Double.infinity
        var evaluated = 0
        let candidateLimit = 25
        // Build a candidate pool that SPANS baselines, rather than sorting
        // by any single key.
        //
        // Both naive orderings fail, and they fail in opposite directions:
        //   - by match count: near-adjacent pairs always match most densely,
        //     so the whole bounded pool is short-baseline and the seed stays
        //     pinned to frames 4 apart -- 169 points at 0.42 px RMSE, then
        //     ZERO further cameras registered because the structure is too
        //     shallow to register against.
        //   - by separation: the widest pairs barely overlap, every candidate
        //     fails, the budget is exhausted before reaching usable pairs, and
        //     initialization reports "no pair had enough parallax" outright.
        // Taking the best few pairs at EACH separation walks the whole range
        // within the same evaluation budget, and the points x parallax score
        // then picks among genuinely different options.
        let orderedFrameIndex = Dictionary(uniqueKeysWithValues: frames.enumerated().map { ($1, $0) })
        var bySeparation: [Int: [(key: Pair, value: [FeatureMatch])]] = [:]
        for (pair, matches) in pairMatches {
            let separation = abs((orderedFrameIndex[pair.b] ?? 0) - (orderedFrameIndex[pair.a] ?? 0))
            bySeparation[separation, default: []].append((pair, matches))
        }
        var candidates: [(key: Pair, value: [FeatureMatch])] = []
        let perSeparation = 3
        for separation in bySeparation.keys.sorted(by: >) {
            let group = bySeparation[separation]!.sorted {
                $0.value.count != $1.value.count ? $0.value.count > $1.value.count
                                                 : ($0.key.a, $0.key.b) < ($1.key.a, $1.key.b)
            }
            candidates.append(contentsOf: group.prefix(perSeparation))
        }
        for (pair, matches) in candidates {
            guard matches.count >= options.minPairInliers else { continue }
            guard evaluated < candidateLimit else { break }
            evaluated += 1
            guard let kpA = keypointsByFrame[pair.a], let kpB = keypointsByFrame[pair.b],
                  let intrA = intrinsics[pair.a], let intrB = intrinsics[pair.b] else { continue }
            guard let result = TwoViewGeometry.estimate(
                matches: matches, keypoints1: kpA, keypoints2: kpB,
                intrinsics1: intrA, intrinsics2: intrB
            ), result.inliers.count >= options.minPairInliers, !result.points.isEmpty else { continue }

            let angles = result.points.map {
                TwoViewGeometry.triangulationAngleDegrees(point: $0, pose1: .identity, pose2: result.pose)
            }.sorted()
            let medianAngle = angles[angles.count / 2]
            guard medianAngle >= options.minInitialAngleDegrees else { continue }

            // Score on points x parallax, not points alone.
            //
            // Point count by itself systematically favours SMALL baselines:
            // near-adjacent frames match more densely, so more points clear
            // cheirality. But the resulting points have enormous depth
            // uncertainty, and every subsequent camera has to register against
            // them -- so PnP is ill-conditioned and finds too few inliers.
            // Measured on a real 4K capture: a 4-frame-apart seed produced 169
            // points at 0.42 px RMSE and then registered ZERO further cameras,
            // despite candidates having 20-28 correspondences available.
            // Weighting by median triangulation angle prefers a seed whose
            // structure is actually well-conditioned for what comes next.
            let score = Double(result.points.count) * medianAngle
            if bestPair == nil || score > bestScore {
                bestScore = score
                bestPair = (pair, result)
            }
        }
        guard let seed = bestPair else {
            log?("no pair had enough parallax to initialize")
            return nil
        }
        log?("initial pair \(seed.pair.a)-\(seed.pair.b) with \(seed.result.inliers.count) inliers")

        // 4. Initialize the reconstruction from that pair.
        var reconstruction = Reconstruction()
        reconstruction.cameras[seed.pair.a] = RegisteredCamera(
            frameIndex: seed.pair.a, pose: .identity, intrinsics: intrinsics[seed.pair.a]!)
        reconstruction.cameras[seed.pair.b] = RegisteredCamera(
            frameIndex: seed.pair.b, pose: seed.result.pose, intrinsics: intrinsics[seed.pair.b]!)

        // Seed points DIRECTLY from the verified two-view inliers, one point
        // per inlier match, and record an observation -> point map.
        //
        // An earlier version looked each inlier up in the union-find track
        // table instead. That collapsed the reconstruction: with a sliding
        // match window, chains like (a,k)->(b,j)->(c,m) merge many distinct
        // physical points into a single track, so after the first inlier
        // claimed that track every other inlier was skipped as
        // already-triangulated. Measured: 33 verified inliers seeded 1 point.
        // The two-view inliers are already geometrically verified, so they are
        // a better source of truth here than the transitively-merged tracks.
        let seedMatches = pairMatches[seed.pair]!
        var pointForObservation: [Observation: Int] = [:]
        for (i, matchIndex) in seed.result.pointMatchIndices.enumerated() {
            let match = seedMatches[matchIndex]
            let obsA = Observation(frame: seed.pair.a, keypoint: match.queryIndex)
            let obsB = Observation(frame: seed.pair.b, keypoint: match.trainIndex)
            guard pointForObservation[obsA] == nil, pointForObservation[obsB] == nil else { continue }
            let pointIndex = reconstruction.points.count
            reconstruction.points.append(ScenePoint(
                position: seed.result.points[i],
                observations: [(frame: seed.pair.a, keypoint: match.queryIndex),
                               (frame: seed.pair.b, keypoint: match.trainIndex)]
            ))
            pointForObservation[obsA] = pointIndex
            pointForObservation[obsB] = pointIndex
        }
        log?("seeded \(reconstruction.points.count) points")

        // Register remaining frames, most-supported first. Support and
        // correspondence both come from PAIRWISE matches against already
        // registered frames, not from global tracks.
        var remaining = Set(frames).subtracting([seed.pair.a, seed.pair.b])
        var progress = true
        while progress && !remaining.isEmpty {
            progress = false

            // 3D-2D correspondences for a candidate frame.
            func correspondences(for frame: Int) -> (world: [SIMD3<Double>], image: [SIMD2<Double>], keypoints: [Int]) {
                guard let keypoints = keypointsByFrame[frame] else { return ([], [], []) }
                var world: [SIMD3<Double>] = [], image: [SIMD2<Double>] = [], kpIndices: [Int] = []
                var seen = Set<Int>()
                for registered in reconstruction.cameras.keys {
                    guard let matches = pairMatches[Pair(frame, registered)] else { continue }
                    let frameIsQuery = frame < registered
                    for match in matches {
                        let frameKp = frameIsQuery ? match.queryIndex : match.trainIndex
                        let otherKp = frameIsQuery ? match.trainIndex : match.queryIndex
                        guard !seen.contains(frameKp), frameKp < keypoints.count else { continue }
                        guard let pointIndex = pointForObservation[Observation(frame: registered, keypoint: otherKp)]
                        else { continue }
                        seen.insert(frameKp)
                        world.append(reconstruction.points[pointIndex].position)
                        image.append(SIMD2<Double>(Double(keypoints[frameKp].x), Double(keypoints[frameKp].y)))
                        kpIndices.append(frameKp)
                    }
                }
                return (world, image, kpIndices)
            }

            let ordered = remaining.sorted { correspondences(for: $0).world.count > correspondences(for: $1).world.count }
            for frame in ordered {
                guard let intr = intrinsics[frame], let keypoints = keypointsByFrame[frame] else { continue }
                let c = correspondences(for: frame)
                guard c.world.count >= 6 else { continue }
                guard let pnp = PoseEstimation.estimatePose(worldPoints: c.world, imagePoints: c.image,
                                                           intrinsics: intr)
                else { continue }

                reconstruction.cameras[frame] = RegisteredCamera(frameIndex: frame, pose: pnp.pose, intrinsics: intr)
                remaining.remove(frame)
                progress = true

                // Link this frame's inlier observations to their points.
                for i in pnp.inliers {
                    let keypointIndex = c.keypoints[i]
                    let observation = Observation(frame: frame, keypoint: keypointIndex)
                    guard pointForObservation[observation] == nil else { continue }
                    // Find which point this correspondence referred to.
                    for registered in reconstruction.cameras.keys where registered != frame {
                        guard let matches = pairMatches[Pair(frame, registered)] else { continue }
                        let frameIsQuery = frame < registered
                        for match in matches {
                            let frameKp = frameIsQuery ? match.queryIndex : match.trainIndex
                            guard frameKp == keypointIndex else { continue }
                            let otherKp = frameIsQuery ? match.trainIndex : match.queryIndex
                            guard let pointIndex = pointForObservation[Observation(frame: registered, keypoint: otherKp)]
                            else { continue }
                            if !reconstruction.points[pointIndex].observations.contains(where: { $0.frame == frame }) {
                                reconstruction.points[pointIndex].observations.append((frame: frame, keypoint: keypointIndex))
                            }
                            pointForObservation[observation] = pointIndex
                            break
                        }
                        if pointForObservation[observation] != nil { break }
                    }
                }

                // Triangulate NEW points from matches to registered frames
                // where neither endpoint is triangulated yet.
                var added = 0
                for registered in reconstruction.cameras.keys where registered != frame {
                    guard let matches = pairMatches[Pair(frame, registered)],
                          let camA = reconstruction.cameras[registered],
                          let keypointsA = keypointsByFrame[registered] else { continue }
                    let camB = reconstruction.cameras[frame]!
                    let frameIsQuery = frame < registered
                    for match in matches {
                        let frameKp = frameIsQuery ? match.queryIndex : match.trainIndex
                        let otherKp = frameIsQuery ? match.trainIndex : match.queryIndex
                        guard frameKp < keypoints.count, otherKp < keypointsA.count else { continue }
                        let obsNew = Observation(frame: frame, keypoint: frameKp)
                        let obsOld = Observation(frame: registered, keypoint: otherKp)
                        guard pointForObservation[obsNew] == nil, pointForObservation[obsOld] == nil else { continue }

                        let nA = camA.intrinsics.normalize(x: Double(keypointsA[otherKp].x), y: Double(keypointsA[otherKp].y))
                        let nB = camB.intrinsics.normalize(x: Double(keypoints[frameKp].x), y: Double(keypoints[frameKp].y))
                        guard let position = TwoViewGeometry.triangulate(
                            p1: nA, p2: nB, pose1: camA.pose, pose2: camB.pose) else { continue }
                        guard camA.pose.transform(position).z > 0, camB.pose.transform(position).z > 0 else { continue }
                        guard TwoViewGeometry.triangulationAngleDegrees(
                            point: position, pose1: camA.pose, pose2: camB.pose) >= 1.0 else { continue }
                        // Reject points that do not actually reproject well.
                        guard let pa = camA.pose.project(position, intrinsics: camA.intrinsics),
                              let pb = camB.pose.project(position, intrinsics: camB.intrinsics) else { continue }
                        let ea = (pa - SIMD2<Double>(Double(keypointsA[otherKp].x), Double(keypointsA[otherKp].y)))
                        let eb = (pb - SIMD2<Double>(Double(keypoints[frameKp].x), Double(keypoints[frameKp].y)))
                        guard (ea.x*ea.x + ea.y*ea.y).squareRoot() < 4.0,
                              (eb.x*eb.x + eb.y*eb.y).squareRoot() < 4.0 else { continue }

                        let pointIndex = reconstruction.points.count
                        reconstruction.points.append(ScenePoint(
                            position: position,
                            observations: [(frame: registered, keypoint: otherKp), (frame: frame, keypoint: frameKp)]
                        ))
                        pointForObservation[obsNew] = pointIndex
                        pointForObservation[obsOld] = pointIndex
                        added += 1
                    }
                }
                log?("registered frame \(frame) (\(pnp.inliers.count)/\(c.world.count) PnP inliers, +\(added) points)")
                break
            }
        }

        let rmseBefore = reconstruction.reprojectionRMSE(keypoints: keypointsByFrame)
        var rmseAfter = rmseBefore
        if options.bundleAdjust && reconstruction.cameras.count >= 2 && !reconstruction.points.isEmpty {
            let result = BundleAdjustment.refineWithOutlierRejection(
                reconstruction: &reconstruction, keypoints: keypointsByFrame)
            rmseAfter = result.finalRMSE
            log?(String(format: "bundle adjustment: %.3f -> %.3f px over %d iterations",
                        result.initialRMSE, result.finalRMSE, result.iterations))
        }

        let report = SfMReport(
            registeredCameras: reconstruction.cameras.count,
            totalCameras: frames.count,
            points: reconstruction.points.count,
            initialPair: (seed.pair.a, seed.pair.b),
            rmseBefore: rmseBefore,
            rmseAfter: rmseAfter
        )
        return (reconstruction, report)
    }

    /// How many already-triangulated points a candidate frame can see.
    static func support(frame: Int, tracks: [Observation: Track], unionFind: UnionFind,
                        keypointsByFrame: [Int: [Keypoint]]) -> Int {
        guard let keypoints = keypointsByFrame[frame] else { return 0 }
        var count = 0
        var uf = unionFind
        for keypointIndex in 0..<keypoints.count {
            let observation = Observation(frame: frame, keypoint: keypointIndex)
            guard uf.parent[observation] != nil else { continue }
            if let track = tracks[uf.find(observation)], track.pointIndex != nil { count += 1 }
        }
        return count
    }

    // MARK: Supporting types

    struct Observation: Hashable {
        let frame: Int
        let keypoint: Int
    }

    struct Pair: Hashable {
        let a: Int
        let b: Int
        init(_ a: Int, _ b: Int) { self.a = min(a, b); self.b = max(a, b) }
    }

    /// Union-find over observations, keyed by an integer id so tracks can be
    /// referenced compactly.
    struct UnionFind {
        var parent: [Observation: Observation] = [:]

        /// Returns the representative OBSERVATION, never a hash.
        ///
        /// An earlier version returned `root.hashValue` as a compact track id.
        /// That is doubly wrong: Swift seeds hashValue randomly per process,
        /// so reconstructions would not be reproducible between runs, and any
        /// hash collision would silently merge two unrelated feature tracks
        /// into one 3D point.
        mutating func find(_ x: Observation) -> Observation {
            if parent[x] == nil { parent[x] = x; return x }
            var root = x
            while let next = parent[root], next != root { root = next }
            var current = x
            while let next = parent[current], next != root {
                parent[current] = root
                current = next
            }
            return root
        }

        mutating func union(_ a: Observation, _ b: Observation) {
            let rootA = find(a)
            let rootB = find(b)
            if rootA != rootB { parent[rootB] = rootA }
        }
    }
}
