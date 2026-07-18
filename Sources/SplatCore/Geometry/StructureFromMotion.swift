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
    /// Run an interim bundle adjustment after this many newly registered
    /// cameras. Keeps structure tight so later PnP stays well conditioned.
    public var bundleEveryNCameras: Int

    public init(matchWindow: Int = 6, maxFeaturesPerFrame: Int = 1500,
                minPairInliers: Int = 30, minInitialAngleDegrees: Double = 1.2,
                bundleAdjust: Bool = true, bundleEveryNCameras: Int = 3) {
        self.matchWindow = matchWindow
        self.maxFeaturesPerFrame = maxFeaturesPerFrame
        self.minPairInliers = minPairInliers
        self.minInitialAngleDegrees = minInitialAngleDegrees
        self.bundleAdjust = bundleAdjust
        self.bundleEveryNCameras = bundleEveryNCameras
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
        // Sorted: union order decides which observation becomes a set's root.
        // The resulting PARTITION is order-independent, but the roots are not,
        // and tracks are keyed by root — so leaving this to Dictionary order
        // would make the track table vary per process for no benefit.
        for pair in pairMatches.keys.sorted(by: { ($0.a, $0.b) < ($1.a, $1.b) }) {
            let matches = pairMatches[pair]!
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
        // Guided matches, keyed by pair, computed once both cameras are posed.
        //
        // These serve two purposes at once. Densification: the plain ratio test
        // has to be conservative with no geometric context and discards many
        // correct correspondences, exactly where appearance has changed most —
        // the wide viewpoint changes late in an orbit. And LOOP CLOSURE: the
        // sliding match window never pairs the end of an orbit with its start,
        // but once both ends are registered the epipolar constraint links them
        // directly, which is what stops drift accumulating all the way round.
        var refinedMatches: [Pair: [FeatureMatch]] = [:]

        /// Matches to use for structure growth: guided if both cameras are
        /// posed, otherwise the plain windowed matches.
        func matchesFor(_ a: Int, _ b: Int) -> [FeatureMatch]? {
            refinedMatches[Pair(a, b)] ?? pairMatches[Pair(a, b)]
        }

        /// Re-match `frame` against every other registered camera using
        /// epipolar guidance. Cheap enough to run on every registration
        /// because the epipolar constraint prunes candidates before any
        /// descriptor comparison.
        func refineMatches(for frame: Int) {
            guard let camNew = reconstruction.cameras[frame], let setNew = setsByFrame[frame] else { return }
            // Sorted: Dictionary order is randomized per process in Swift.
            for other in reconstruction.cameras.keys.sorted() where other != frame {
                guard let camOther = reconstruction.cameras[other], let setOther = setsByFrame[other] else { continue }
                let pair = Pair(frame, other)
                // Keep the ordering convention: query is the lower frame index.
                let (qSet, qCam, tSet, tCam) = pair.a == frame
                    ? (setNew, camNew, setOther, camOther)
                    : (setOther, camOther, setNew, camNew)
                let guided = FeatureMatcher.matchGuided(
                    query: qSet, train: tSet,
                    queryPose: qCam.pose, trainPose: tCam.pose,
                    queryIntrinsics: qCam.intrinsics, trainIntrinsics: tCam.intrinsics
                )
                // Only adopt guided matches when they beat what we already had;
                // a bad pose would otherwise poison a good plain match set.
                if guided.count > (matchesFor(pair.a, pair.b)?.count ?? 0) {
                    refinedMatches[pair] = guided
                }
            }
        }

        // Does `position` land close enough to `keypoint` in `camera`?
        func reprojectsWell(_ position: SIMD3<Double>, _ camera: RegisteredCamera,
                            _ keypoint: Keypoint, tolerance: Double = 2.5) -> Bool {
            guard let projected = camera.pose.project(position, intrinsics: camera.intrinsics) else { return false }
            let dx = projected.x - Double(keypoint.x), dy = projected.y - Double(keypoint.y)
            return (dx * dx + dy * dy).squareRoot() < tolerance
        }

        /// Extend existing points with new observations and triangulate new
        /// ones, over every match between two registered cameras.
        func growStructure() -> (extended: Int, created: Int) {
            var extended = 0, created = 0
            let registered = reconstruction.cameras.keys.sorted()
            for (i, a) in registered.enumerated() {
                for b in registered[(i + 1)...] {
                    guard let camA = reconstruction.cameras[a], let camB = reconstruction.cameras[b],
                          let kpA = keypointsByFrame[a], let kpB = keypointsByFrame[b] else { continue }
                    // Guided matches EXTEND existing points; only plain,
                    // ratio-tested and cross-checked matches may CREATE them.
                    //
                    // The epipolar constraint is a line, not a point: a wrong
                    // feature further along that line satisfies it perfectly.
                    // Creating a point from such a match is unrecoverable,
                    // because a point triangulated from two views always
                    // reprojects well INTO THOSE TWO VIEWS — the reprojection
                    // check cannot catch its own input. Extending an existing
                    // point is different: it tests the match against a 3D
                    // position established independently, which is a real
                    // check. Measured when guided matches were allowed to
                    // create points: 3962 points but PnP inlier ratios fell to
                    // 10/219 and registered cameras dropped from 17 to 9.
                    let extendMatches = matchesFor(a, b) ?? []
                    let createMatches = pairMatches[Pair(a, b)] ?? []
                    let matches = extendMatches
                    // Pair(a,b) keeps a < b, and matches were built with the
                    // lower frame as query, so query indexes a and train b.
                    for match in matches {
                        let ka = match.queryIndex, kb = match.trainIndex
                        guard ka < kpA.count, kb < kpB.count else { continue }
                        let obsA = Observation(frame: a, keypoint: ka)
                        let obsB = Observation(frame: b, keypoint: kb)
                        let pointA = pointForObservation[obsA]
                        let pointB = pointForObservation[obsB]

                        if let p = pointA, pointB == nil {
                            guard reprojectsWell(reconstruction.points[p].position, camB, kpB[kb]) else { continue }
                            if !reconstruction.points[p].observations.contains(where: { $0.frame == b }) {
                                reconstruction.points[p].observations.append((frame: b, keypoint: kb))
                            }
                            pointForObservation[obsB] = p
                            extended += 1
                        } else if let p = pointB, pointA == nil {
                            guard reprojectsWell(reconstruction.points[p].position, camA, kpA[ka]) else { continue }
                            if !reconstruction.points[p].observations.contains(where: { $0.frame == a }) {
                                reconstruction.points[p].observations.append((frame: a, keypoint: ka))
                            }
                            pointForObservation[obsA] = p
                            extended += 1
                        }
                    }
                    // Creation pass, restricted to plain matches.
                    for match in createMatches {
                        let ka = match.queryIndex, kb = match.trainIndex
                        guard ka < kpA.count, kb < kpB.count else { continue }
                        let obsA = Observation(frame: a, keypoint: ka)
                        let obsB = Observation(frame: b, keypoint: kb)
                        if pointForObservation[obsA] == nil && pointForObservation[obsB] == nil {
                            let nA = camA.intrinsics.normalize(x: Double(kpA[ka].x), y: Double(kpA[ka].y))
                            let nB = camB.intrinsics.normalize(x: Double(kpB[kb].x), y: Double(kpB[kb].y))
                            guard let position = TwoViewGeometry.triangulate(
                                p1: nA, p2: nB, pose1: camA.pose, pose2: camB.pose) else { continue }
                            guard camA.pose.transform(position).z > 0,
                                  camB.pose.transform(position).z > 0 else { continue }
                            guard TwoViewGeometry.triangulationAngleDegrees(
                                point: position, pose1: camA.pose, pose2: camB.pose) >= 1.0 else { continue }
                            guard reprojectsWell(position, camA, kpA[ka], tolerance: 4.0),
                                  reprojectsWell(position, camB, kpB[kb], tolerance: 4.0) else { continue }
                            let pointIndex = reconstruction.points.count
                            reconstruction.points.append(ScenePoint(
                                position: position,
                                observations: [(frame: a, keypoint: ka), (frame: b, keypoint: kb)]
                            ))
                            pointForObservation[obsA] = pointIndex
                            pointForObservation[obsB] = pointIndex
                            created += 1
                        }
                    }
                }
            }
            return (extended, created)
        }

        var registeredSinceBA = 0
        // Seed the structure once before any registration, so the very first
        // candidate sees everything the initial pair can support.
        _ = growStructure()

        var remaining = Set(frames).subtracting([seed.pair.a, seed.pair.b])
        var progress = true
        while progress && !remaining.isEmpty {
            progress = false

            // 3D-2D correspondences for a candidate frame.
            func correspondences(for frame: Int) -> (world: [SIMD3<Double>], image: [SIMD2<Double>], keypoints: [Int]) {
                guard let keypoints = keypointsByFrame[frame] else { return ([], [], []) }
                var world: [SIMD3<Double>] = [], image: [SIMD2<Double>] = [], kpIndices: [Int] = []
                var seen = Set<Int>()
                // Sorted, and load-bearing: the `seen` set below makes the
                // FIRST match for a keypoint win, so iteration order decides which
                // correspondences PnP receives. Unsorted, the same input produced
                // 33 registered cameras on one run and 14 on the next.
                for registered in reconstruction.cameras.keys.sorted() {
                    guard let matches = matchesFor(frame, registered) else { continue }
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

            // Total order, and counts computed ONCE.
            //
            // Two independent bugs lived in the previous one-liner. It sorted a
            // Set — whose iteration order Swift randomizes per process — and
            // Swift's sort is NOT stable, so any tie in correspondence count
            // resolved arbitrarily. Since the loop registers only the first
            // frame and then restarts, a tie decided which camera joined next,
            // and that choice cascades through every later registration.
            // Proven: SWIFT_DETERMINISTIC_HASHING=1 gave 29/60 registered
            // cameras where the default gave 25/60, on an identical binary.
            // It also called correspondences() inside the comparator, which is
            // O(n log n) recomputations of an expensive function AND risks an
            // inconsistent comparator if the value could ever vary mid-sort —
            // undefined behaviour for sort().
            let ordered = remaining
                .map { (frame: $0, support: correspondences(for: $0).world.count) }
                .sorted { $0.support != $1.support ? $0.support > $1.support : $0.frame < $1.frame }
                .map { $0.frame }
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

                // Grow structure across ALL registered pairs, not just the
                // pairs involving the frame that was just added.
                //
                // The previous version did two narrower things and coverage
                // could not propagate around an orbit: it linked observations
                // only for PnP INLIERS (11 of 18 on a real frame), and when
                // triangulating it skipped any match where EITHER endpoint
                // already had a point. Since seed-frame keypoints all have
                // points, that skipped exactly the matches that would have
                // carried structure forward — so the next frame round the arc
                // was left with ~7 correspondences and could not register.
                //
                // Two passes, run to a fixed point:
                //   - extend: a match where one side has a point and the other
                //     does not attaches the new observation (subject to
                //     reprojection), which is what propagates tracks.
                //   - create: a match where neither side has a point
                //     triangulates a new one.
                // Pose is known now, so re-match this camera against the rest
                // with epipolar guidance before growing structure.
                refineMatches(for: frame)
                let (extended, added) = growStructure()

                log?("registered frame \(frame) (\(pnp.inliers.count)/\(c.world.count) PnP inliers, "
                     + "+\(added) points, +\(extended) observations)")

                // Periodic bundle adjustment. Structure triangulated from a
                // two-view seed still carries depth error, and every camera
                // registered afterwards inherits it; tightening as we go keeps
                // later PnP well conditioned instead of letting drift compound.
                registeredSinceBA += 1
                if options.bundleAdjust && registeredSinceBA >= options.bundleEveryNCameras {
                    registeredSinceBA = 0
                    let interim = BundleAdjustment.refine(reconstruction: &reconstruction,
                                                          keypoints: keypointsByFrame)
                    log?(String(format: "  interim BA: %.3f -> %.3f px", interim.initialRMSE, interim.finalRMSE))
                }
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
