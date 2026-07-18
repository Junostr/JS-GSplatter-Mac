import Foundation

// MARK: - Stage 2: frame selection — pure logic
//
// Separated from the analyzers for the same reason TierSelector is separate
// from HardwareProbe: everything here is a pure function over scores, so the
// interesting decisions are unit-testable without a GPU or real images.

public struct FilterOptions: Equatable {
    /// Upper bound on selected frames. The selector may return fewer if the
    /// input has fewer distinct, sharp frames than requested.
    public var targetFrameCount: Int
    /// Frames whose blur score falls below this fraction of the session
    /// median are considered blurry. Relative, never absolute — Laplacian
    /// variance scales with scene texture, so a fixed cutoff would reject
    /// entire low-texture sessions (walls, sky) wholesale.
    public var blurRejectFactor: Double
    /// Minimum signature distance (0…1) from the last kept frame for a frame
    /// to count as new content. Below it, the frame is a near-duplicate —
    /// tripod shots, video freezes, burst photos of the same pose.
    public var dedupMinDistance: Double

    public init(targetFrameCount: Int = 150,
                blurRejectFactor: Double = 0.4,
                dedupMinDistance: Double = 0.01) {
        self.targetFrameCount = targetFrameCount
        self.blurRejectFactor = blurRejectFactor
        self.dedupMinDistance = dedupMinDistance
    }
}

public struct SelectionResult: Equatable {
    public let selected: [FrameScore]
    public let rejectedBlurry: Int
    public let rejectedDuplicates: Int
    public let rejectedOverBudget: Int
}

public enum FrameSelector {

    /// Three passes, in a deliberate order:
    ///  1. blur rejection (relative to the session median),
    ///  2. near-duplicate collapse (keeping the sharpest of each run),
    ///  3. even-coverage downsampling to the target count.
    /// Dedup runs before budgeting so a long static stretch costs one slot,
    /// not a proportional share of the budget.
    public static func select(scores: [FrameScore], options: FilterOptions) -> SelectionResult {
        guard !scores.isEmpty else {
            return SelectionResult(selected: [], rejectedBlurry: 0, rejectedDuplicates: 0, rejectedOverBudget: 0)
        }
        let ordered = scores.sorted { $0.index < $1.index }

        // Pass 1 — blur. Median, not mean: one perfectly sharp frame in a
        // shaky session would drag a mean-based threshold up and reject
        // everything else.
        let sortedBlur = ordered.map { $0.blurScore }.sorted()
        let median = sortedBlur[sortedBlur.count / 2]
        let threshold = median * options.blurRejectFactor
        let sharp = ordered.filter { $0.blurScore >= threshold }
        let rejectedBlurry = ordered.count - sharp.count

        // Pass 2 — collapse near-duplicates, keeping each cluster's sharpest
        // member (not its first: the first frame after the camera stops moving
        // is often the most motion-blurred of the cluster).
        //
        // Distance is measured against the cluster ANCHOR (its first frame),
        // not the immediately preceding frame. This is the crucial choice for
        // splat capture: the camera is in continuous orbital motion, so
        // consecutive frames are always near-identical and a previous-frame
        // comparison would collapse an entire sweep into one keyframe. Anchor
        // comparison lets drift accumulate, so a new keyframe is taken every
        // time the view has changed by `dedupMinDistance` — even, content-
        // driven spacing. A genuine static hold (tripod, video freeze) never
        // drifts from its anchor and correctly collapses to a single frame.
        var deduped: [FrameScore] = []
        var cluster: [FrameScore] = []
        var anchor: FrameScore?
        func flushCluster() {
            if let best = cluster.max(by: { $0.blurScore < $1.blurScore }) {
                deduped.append(best)
            }
            cluster = []
        }
        for score in sharp {
            if let anchor = anchor, anchor.signatureDistance(to: score) >= options.dedupMinDistance {
                flushCluster()
            }
            if cluster.isEmpty { anchor = score }
            cluster.append(score)
        }
        flushCluster()
        let rejectedDuplicates = sharp.count - deduped.count

        // Pass 3 — budget. Even buckets over the *remaining* sequence, best
        // blur score per bucket: keeps temporal coverage uniform (SfM needs
        // views spread across the whole path) while still preferring sharp
        // frames within each neighborhood.
        let target = max(1, options.targetFrameCount)
        let selected: [FrameScore]
        if deduped.count <= target {
            selected = deduped
        } else {
            var picked: [FrameScore] = []
            picked.reserveCapacity(target)
            for bucket in 0..<target {
                let start = bucket * deduped.count / target
                let end = (bucket + 1) * deduped.count / target
                guard end > start else { continue }
                // Search only the CENTRAL portion of each bucket.
                //
                // Taking the sharpest frame anywhere in the bucket lets two
                // consecutive buckets pick frames on either side of their
                // shared boundary — producing selected frames that are
                // adjacent in the original video while every other pair is a
                // full bucket apart. That wrecks downstream SfM: the initial
                // pair is chosen for having the most matches, adjacent frames
                // always win that, and the resulting near-zero baseline gives
                // points too shallow to register any other camera against.
                // Measured on a real capture: seed frames 4 apart, nearest
                // other frame 20 apart, and PnP then failed on every single
                // frame despite having 13-34 correspondences available.
                //
                // Restricting to the middle 60% guarantees a minimum spacing
                // of ~40% of a bucket between picks, at the cost of sometimes
                // taking a slightly less sharp frame — a good trade, since
                // even coverage matters more to reconstruction than the last
                // few percent of sharpness.
                let span = end - start
                let inset = span >= 5 ? span / 5 : 0
                let searchStart = start + inset
                let searchEnd = max(searchStart + 1, end - inset)
                if let best = deduped[searchStart..<searchEnd].max(by: { $0.blurScore < $1.blurScore }) {
                    picked.append(best)
                }
            }
            selected = picked
        }
        let rejectedOverBudget = deduped.count - selected.count

        return SelectionResult(
            selected: selected,
            rejectedBlurry: rejectedBlurry,
            rejectedDuplicates: rejectedDuplicates,
            rejectedOverBudget: rejectedOverBudget
        )
    }
}
