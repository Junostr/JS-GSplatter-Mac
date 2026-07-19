# Gaussian Splatter Mac

Native macOS app that converts a photo series or video into a 3D Gaussian
Splat. Universal binary (arm64 + x86_64), deployment target **macOS 11.0
(Big Sur)**, GPU compute via raw Metal — no CUDA anywhere.

## Status: stages 1 (ingestion), 2 (filtering), 3 (SfM), 4 (probe)

Implemented:
- **Ingestion** (`SplatCore/Ingestion/`) — photo folders (ImageIO: EXIF
  orientation applied, decode-time downsampling via `--max-dim`, numeric
  filename ordering) and video (AVFoundation/VideoToolbox hardware decode,
  BGRA output, even subsampling via `--max-frames`). Frames stream one at a
  time through a handler — nothing materializes a whole clip in RAM. CLI:
  `splatctl ingest <folder|video> [--max-frames N] [--max-dim N] [--save dir]`.
- **Frame filtering** (`SplatCore/Filtering/`) — the first real tiered
  compute stage. A `FrameAnalyzer` protocol with a **Metal baseline**
  (`MetalFrameAnalyzer`, runtime-compiled MSL kernels) and an **Accelerate/
  vDSP CPU fallback** (`CPUFrameAnalyzer`), producing per-frame Laplacian-
  variance blur scores and an 8×8 mean-luma scene signature. Both tiers are
  held to a numerical-agreement contract (verified in selftest: blur within
  5%, signature within 0.02). `FrameSelector` is pure logic over the scores:
  relative-to-median blur rejection, anchor-based near-duplicate collapse
  (keeps the sharpest of each cluster; drift accumulates so continuous camera
  motion is *not* collapsed while static holds are), and even-coverage
  downsampling to a target count. CLI: `splatctl filter <folder|video>
  [--target N] [--cpu] [--save dir] [--verbose]`. On an M1 Pro the Metal
  path runs ~37× faster than the CPU fallback (510 vs 14 frames/s at 640×480).
- **Feature extraction + matching** (`SplatCore/Features/`) — first half of
  stage 3. `FeatureExtractor` protocol with a Metal baseline (Harris corner
  response) and an Accelerate CPU fallback; ORB-style intensity-centroid
  orientation and steered 256-bit BRIEF descriptors; brute-force Hamming
  matching with Lowe's ratio test and cross-check. Only *detection* is
  tier-specific — orientation and descriptors are shared code, so descriptor
  parity is structural rather than tested-into-existence.

  Two precision landmines this stage hit, both worth knowing before touching
  the kernels:
  - **Metal's fast math is on by default** and contracts the Harris
    determinant (`sxx*syy − sxy*sxy`, a difference of similar large products)
    into an FMA. That is catastrophic cancellation, so last-bit input noise
    became visibly different responses. It is explicitly disabled. Stage 2's
    kernels are unaffected — variance is a well-conditioned sum of squares.
  - **The tiers' response maps still differ by ~1 ULP**, which is inherent.
    Keypoints are therefore sorted on a response *quantized* to a 2²⁰ grid
    with a spatial tiebreak. Not cosmetic: `maxFeatures` truncates that list,
    so an unstable order would let the tiers keep different feature *subsets*
    at the cut line. (A total order, deliberately — an epsilon-tolerant
    comparator is non-transitive and makes `sort()` undefined behavior.)

  The cross-tier contract is consequently *interchangeable feature sets
  matched by position*, not identical array ordering — the latter isn't a
  promise that can hold across Nvidia/AMD/Apple GPUs. Verified identical on
  two different GPUs (M1 Pro and CI's paravirtual device).
- **App shell** (`App/`) — SwiftUI drag-and-drop UI (macOS 11-safe API
  surface only): shows the tier decision with reasons and a force-baseline
  toggle, accepts a dropped photo folder or video, runs ingestion off-main
  with live progress and a first-frame preview. Same sources build via the
  Xcode project (dev) and `Scripts/build-app.sh` (distribution).
- **Hardware/OS capability probe** (`SplatCore/Capabilities/HardwareProbe.swift`)
  — GPU vendor/working set via Metal, CPU architecture, macOS version,
  Rosetta 2 detection.
- **Tier selection** (`SplatCore/Capabilities/TierSelection.swift`) — pure
  function from probe → decision, with a human-readable reason trail.
  Baseline sub-tiers: `legacyNVIDIA` / `discreteAMD` / `integratedIntel` /
  `appleSilicon` / `cpuFallback`, each with default parameters (splat
  ceiling, tile size, precision, GPU memory budget).
- **`TrainingEngine` protocol** with three conformances:
  `BaselineMetalTrainingEngine` (stub, always present),
  `CPUFallbackTrainingEngine` (stub, Accelerate-backed later), and
  `EnhancedTrainingEngine` (stub, arm64-only + `@available(macOS 13.3, *)`).
- **`splatctl`** CLI that prints the probe, the tier decision with reasons,
  and drives the selected engine stub. `--force-baseline` overrides enhanced
  tier for testing; `--json` emits the full report machine-readably.

- **Structure from motion** (`SplatCore/Geometry/`) — completes stage 3.
  Self-contained linear algebra (Jacobi eigensolver, 3x3 SVD, Cholesky —
  deliberately not LAPACK, whose Accelerate interface changed across exactly
  the SDK range this project straddles), a pinhole camera model, two-view
  geometry (normalized 8-point essential matrix under RANSAC with Sampson
  distance, pose recovery by cheirality, DLT triangulation), PnP registration
  for additional views, and sparse Levenberg-Marquardt bundle adjustment using
  the Schur complement. `StructureFromMotion.reconstruct` drives the whole
  thing; `splatctl sfm <input> [--target N] [--ply out.ply]` runs it end to end
  and can dump the sparse cloud plus camera path as PLY.

  Verified against synthetic scenes with exactly known ground truth (real
  imagery cannot test this layer — without ground truth you can only check
  that a reconstruction is self-consistent, which a mirrored one also is):
  rotation recovered to 0.0 deg, structure matching truth to 8e-7 relative,
  robust to 25% outliers, 6/6 cameras registered end to end at 1.3e-5 px RMSE,
  and bit-identical across runs.

  **Focal length is estimated from the geometry**, not guessed
  (`FocalEstimation`). The old `1.2 x long side` heuristic implies ~45 deg FOV
  and is badly wrong for phone cameras (~69 deg at 4K). A wrong focal is not a
  scale error: the essential matrix is only an essential matrix under correct
  calibration, so it fails *confidently* — RANSAC fits an unrealizable model,
  cheirality rejects nearly everything, and BA deletes the rest. Measured on a
  real 3840x2160 iPhone capture:

  | focal | inliers | RMSE after BA | points |
  |---|---|---|---|
  | 4608 (old 1.2x guess) | 120 | 14.29 px | 0 |
  | 3200 | 301 | 1.02 px | 125 |
  | 2880 (auto-estimated, 0.75x) | 169 | **0.42 px** | 169 |

  **Feature selection is spatially bucketed** (`FeatureOptions.spatialBuckets`)
  so one high-contrast region cannot take the whole `maxFeatures` budget.
  Note this addresses only the *cap*; see the threshold limitation below.

  **Known limitations (real captures are NOT yet reliable):**
  - *Incremental registration still stalls part-way round an orbit.* A
    structure-growth pass now runs after every registration over ALL pairs of
    registered cameras — extending existing points with new observations as
    well as triangulating new ones — plus an interim bundle adjustment every 3
    cameras. That took the real capture from 2 to 5 registered cameras, with
    PnP inlier counts going from 10/19 to 104/214, but it does not yet complete
    a full orbit.

    A 4-level scale pyramid (`FeatureOptions.pyramidLevels`) confirmed that
    diagnosis and produced the single largest improvement so far — features are
    now detected and described per level, so a surface seen at different
    apparent sizes still matches:

    | | cameras | points | RMSE |
    |---|---|---|---|
    | before pyramid | 5/60 | 304 | 1.42 px |
    | with pyramid | **17/60** | **1809** | **1.34 px** |

    Early PnP inlier ratios went to 92/93, 117/118, 133/137.

    **Guided epipolar matching** (`FeatureMatcher.matchGuided`) re-matches every
    pair of already-posed cameras using the epipolar constraint, which lets the
    ratio test relax because geometry has already eliminated most accidental
    descriptor collisions. On synthetic frames with deliberately ambiguous
    descriptors it recovers 86/86 correct matches where plain matching finds 0,
    and a deliberately wrong pose yields 0 — the constraint does real work.
    Reconstruction: 17 -> 20/60 cameras, RMSE 1.31 px.

    Two constraints on how it may be used, both learned by measurement:
    - Guided matches may EXTEND existing points but not CREATE them. The
      epipolar constraint is a line, not a point, so a wrong feature further
      along it satisfies the test — and a point triangulated from a bad match
      always reprojects well into the two views that created it, so the
      reprojection check cannot catch its own input. Letting guided matches
      create points gave 3962 points but collapsed PnP inlier ratios to 10/219
      and dropped registration from 17 to 9 cameras.
    - The extension reprojection tolerance is tight (2.5 px). At 6 px, wrong
      observations attach to good points and bias bundle adjustment.

    Loop closure comes free from the same mechanism — the sliding match window
    never pairs an orbit's end with its start, but guided matching links any two
    posed cameras that overlap. It has NOT yet been demonstrated closing a real
    loop, because the orbit does not register all the way round for both ends to
    exist simultaneously.

    **SIFT-like descriptors** (`SIFTDescriptor`, now the default;
    `FeatureOptions.descriptorKind` selects, `splatctl sfm --brief` reverts).
    BRIEF compares raw intensities at point pairs and degrades once appearance
    changes — exactly the wide-viewpoint case that limited registration. A
    128-dimension gradient-orientation histogram (4x4 cells x 8 bins, trilinear
    interpolation, normalize/clip/renormalize) discards absolute intensity and
    records only edge direction. Verified: under a contrast x0.55 plus
    brightness +0.2 change the descriptor distance is **0**, against 408939 for
    a genuinely different patch.

    A/B on the real capture, same everything else:

    | descriptor | cameras | points | RMSE |
    |---|---|---|---|
    | BRIEF (256-bit, Hamming) | 17/60 | 2048 | 1.18 px |
    | SIFT-like (128-D, L2) | **25-29/60** | 3326 | 1.46 px |

    Descriptor kind is carried on `FeatureSet`, so the matcher picks Hamming or
    squared-L2 from the data rather than a global assumption. Note the ratio
    test is squared for the L2 path — Lowe's 0.8 is defined on Euclidean
    distance, and applying it unsquared to squared distances would silently
    impose a much stricter 0.894 and discard many correct matches.

    Registration reaches roughly half the orbit (29/60). Late-orbit PnP ratios
    still decay, consistent with accumulating drift at the widest viewpoint
    changes.

    **Loop closure** (`SfMOptions.loopClosure`, `splatctl sfm --loop`, still off
    by default). A cheap probe on the strongest ~120 descriptors scores every
    pair beyond the sliding window; the best candidates are fully matched and
    then **geometrically verified** — a two-view model with at least 25 inliers
    AND a 50% inlier ratio — before entering the match graph.

    That verification is the load-bearing part. Accepting candidates on
    descriptor match count alone actively harmed the reconstruction: 30 pairs
    added, registration DOWN from 20/40 to 16/40, even with the focal correct.
    On an orbiting capture of a furnished room, distant frames share plenty of
    locally-similar texture (fabric, wood grain, foliage) and produce dozens of
    confident-looking matches corresponding to nothing.

    With verification the result on the real capture is unambiguous: **273
    candidates, 0 verified, 60 rejected** — and reconstruction is byte-identical
    to loop closure being off (20/40, 1953 points, 1.252 px). Every candidate
    was a descriptor coincidence.

    The reason is the footage, not the code: that 17-second clip does not
    complete a full orbit, so there is no loop to close. Loop closure is
    therefore *correct but unexercised* — it now provably does no harm when
    there is no loop, but it has never been shown to help. **Validating it needs
    a capture that returns to its starting viewpoint**, ideally a full 360
    degrees with recognisable overlap between the first and last frames.

    Two things were fixed getting that far, and both are worth keeping:
    - A reversed-range trap. For the last `matchWindow + 1` frames the inner
      loop's start index exceeds the frame count, and `25..<24` is a Swift
      precondition failure rather than an empty sequence — the whole run
      crashed with SIGTRAP. (Same trap class as the degenerate-dimension bug in
      `CPUFrameAnalyzer`.)
    - Loop pairs must be excluded from INITIAL-PAIR candidacy. They are by
      construction the widest-baseline pairs in the graph, so they flood the
      bounded seed-candidate pool and starve it of the moderate baselines that
      seed well; with them included the seed search reported "no pair had
      enough parallax" and failed outright.

    Also unresolved and probably prior to loop closure: **focal estimation is
    frame-set sensitive** — the same capture estimates 0.50x, 0.58x, 0.75x,
    0.85x and 0.90x of the long side at different `--target` values, against a
    true value near 0.72x for an iPhone at 4K. A bad focal degrades everything
    downstream, so this likely has to be fixed before loop closure (or any
    other change) can be judged fairly.

    An attempt was made and REVERTED; the findings are worth keeping:

    - The root flaw is real and identified. The current scorer re-fits the
      essential matrix for each candidate focal, so the RANSAC inlier set moves
      with the focal — "which focal is right?" is confounded with "which points
      are inliers?". Scoring by triangulated point count compounds it, because a
      wrong focal still admits a self-consistent model with plenty of surviving
      points, so the count has no sharp peak at the truth.
    - The principled fix is Hartley's: fit the FUNDAMENTAL matrix once from
      pixel coordinates (calibration-independent), then sweep K and minimise the
      singular-value asymmetry (s0 − s1) / (s0 + s1) of E = KᵀFK, since a valid
      essential matrix has singular values (s, s, 0).
    - Implemented, and it did make the estimate **consistent** — 0.50x at every
      `--target`. But 0.50x is the bottom of the sweep range, i.e. the score
      decreases monotonically toward small focals instead of having an interior
      minimum at the truth. Consistently wrong is more dangerous than noisily
      right, so it was reverted rather than shipped.
    **The synthetic harness now exists** (`selftest`: "Focal estimation
    (synthetic, known ground truth)" and "under realistic degradation"), and it
    changed the diagnosis completely:

    - On CLEAN synthetic pairs the current estimator recovers 0.65x, 0.80x,
      1.00x and 1.20x **exactly, 0% error**, and the estimates track the truth
      rather than being constant. So the criterion is not inherently broken.
    - Degrading one property at a time isolates the cause. 10% and 25% outliers:
      fine. Depth spread reduced from 4.0 to 0.3 (an orbit at near-constant
      distance): fine. **Keypoint localisation noise is the whole story** —
      0.5 px and 1.5 px recover 0.80x, 3.0 px collapses to 0.50x, the bottom of
      the sweep and exactly the real-capture symptom.
    - Mechanism: `TwoViewGeometry` converts its gate with
      `threshold = pixels / focal`, so a smaller candidate focal literally
      loosens the geometric test, admits more inliers and triangulates more
      points. The score rewards small focals for being lenient, not for being
      right — invisible on clean data, dominant once keypoints are noisy.
    - Holding the gate constant in NORMALIZED units was tried and made things
      worse: it simply inverts the bias, and noise then drives the estimate to
      the sweep CEILING (1.5 px noise -> 1.90x, worse than the 0.80x it managed
      before). Reverted.

    **Rewritten using the harness. Now STABLE, still BIASED.**

    Plotting the score curves against a known focal was decisive. Both
    candidate criteria (median epipolar error; essential-matrix asymmetry) form
    a sharp V with the minimum exactly at the truth on clean data — but at
    1.5 px of keypoint noise the curve goes nearly FLAT (error varies 4.4-4.7 px
    across the entire focal range). Focal is only weakly observable from one
    noisy pair, so no single-pair criterion, however clever, can work. That is
    an observability limit, not a scoring bug.

    The estimator therefore now: fits F once per pair from pixel coordinates
    (calibration-independent), scores a threshold-free median epipolar error
    over the F inliers, normalises each pair's curve by its own median, and
    SUMS across ~24 pairs so the shared true minimum reinforces while
    independent noise averages out.

    Result on the real capture: **0.50x at `--target` 24, 40 and 60** — the same
    answer every time, where the old estimator gave 0.50x, 0.58x, 0.75x, 0.85x
    and 0.90x on the same footage. Measurements are repeatable again, which was
    the point.

    **Bias characterised, and one contributing cause removed.** Sweeping noise
    against true focal in the harness gives:

    | true \\ noise | 0.5 px | 1.0 px | 1.5 px | 2.0 px | 3.0 px |
    |---|---|---|---|---|---|
    | 0.65x | 0.65 | 0.58 | 0.65 | 0.65 | 0.50 |
    | 0.80x | 0.80 | 0.80 | 0.80 | 0.70 | 0.75 |
    | 1.00x | 1.00 | 0.90 | 0.85 | 0.70 | 0.70 |
    | 1.20x | 1.20 | 1.10 | 0.85 | 0.80 | 0.75 |

    It is **exact for every focal at 0.5 px** and degrades toward a ~0.70-0.75x
    curve-shape prior as noise grows. So accuracy is bounded by keypoint
    localisation noise, not by the criterion — meaning the fix is to feed it
    less noise, not to add a correction factor.

    Accordingly the estimator now uses **finest-octave (level 0) matches only**.
    A corner found on pyramid level k has its coordinates multiplied by 2^k to
    reach full resolution, and its localisation error with them — an octave-2
    keypoint carries ~4x the positional error of an octave-0 one. The pyramid
    earns its place in matching, where scale invariance makes a correspondence
    findable at all; it has no role here, where the estimator needs precision
    rather than recall. Coarse matches are used only as a fallback if fewer
    than 16 fine ones exist. **Not yet validated on the real capture** — macOS
    revoked this session's access to the source video partway through
    (`cp` reports "Operation not permitted"), so the synthetic suite is the only
    evidence for it so far.

    **Resolved: the sweep range was too narrow, not the estimator.** The real
    capture kept returning 0.50x — the bottom of the sweep — which looked like a
    broken estimator hitting a boundary. Dumping the aggregate curve
    (`SPLAT_FOCAL_DEBUG=1`) showed it strictly monotonic with no interior
    minimum, confirming the criterion could not discriminate. Extending the
    sweep down to 0.30x made a clear V appear:

    ```
    0.30:0.907  0.34:0.609  0.38:0.378  0.42:0.217  0.46:0.210 <- min
    0.50:0.313  0.58:0.519  ...  1.90:1.678
    ```

    The true focal is ~0.46x (~95 deg horizontal) — an iPhone **ultra-wide**
    lens, not the main camera. The estimator had been correct all along; a
    sweep that cannot express the answer guarantees a boundary result no matter
    how good the criterion is. Effect on the real capture at `--target 40`:

    | focal | cameras | points | RMSE |
    |---|---|---|---|
    | 0.72x (assumed "true", prior) | 6/40 | 911 | 1.65 px |
    | 0.50x (old sweep floor) | 9/40 | 1013 | 1.10 px |
    | **0.46x (estimated, extended sweep)** | **20/40** | **1953** | 1.25 px |

    An estimate landing on either END of the sweep is now rejected outright
    (`return nil`) so the caller falls back to a documented prior rather than a
    boundary value dressed up as a measurement. The fallback prior itself was
    also corrected from COLMAP's 1.2x (~45 deg, DSLR-oriented) to 0.72x, since
    1.2x produced 14.29 px RMSE and zero surviving points on this footage.

    Trade-off, recorded rather than hidden: the wider range gives noise more
    room to pull a weak estimate downward, and the noisiest synthetic case moved
    from 0.58x to 0.50x against a true 0.65x. Worth it — a range that cannot
    express the answer is wrong for *every* capture from that camera, while the
    noise sensitivity is bounded and characterised below.

    **Residual: a noise-dependent UNDERestimate.** On synthetic pairs at
    1.5 px noise it returns 0.58x for a true 0.65x, 0.65x for 0.80x, and 0.90x
    for 1.10x — tracking the truth but biased ~11-19% low. Real 4K footage is
    noisier still, so it lands on 0.50x (the sweep floor) against a true ~0.72x.
    Scoring in normalised rather than pixel units removed part of the bias (the
    pixel-error curve is steeply asymmetric — 2.88 at 0.5x versus 21.65 at 1.9x
    — so a noise floor slides the minimum toward the flatter low side), but not
    all of it. Next step: characterise the residual bias against noise level in
    the harness and correct or reparameterise it; the tests to do that now
    exist.

    **Resolved (was: 25/60 vs 29/60 from an identical binary).** Root cause was
    the registration queue: `remaining.sorted { support > support }` sorted a
    **Set** — whose iteration order Swift randomizes per process — using a
    comparator that only compared support counts, and Swift's sort is **not
    stable**. Any tie therefore resolved arbitrarily, and since the loop
    registers only the first frame before restarting, one tie decided which
    camera joined next and cascaded through every later registration.

    Proven rather than guessed: `SWIFT_DETERMINISTIC_HASHING=1` produced 29/60
    where the default produced 25/60 on the same binary. After adding a total
    order (support, then frame index) and computing each support count once
    instead of inside the comparator, both hashing modes agree at 29/60 — and
    29 was the correct answer all along. Union-find construction was also
    pinned to a sorted pair order, since union order decides set roots and
    tracks are keyed by root.

    Note the existing "reproducible across runs" test could not catch this: the
    hash seed is fixed for a process lifetime, so running twice in-process
    proves nothing about hash-order dependence. The new guard reconstructs from
    **shuffled input feature sets**, which perturbs the order data lands in
    every internal Set and Dictionary and fails if any dependence survives.

    Practical note meanwhile: prefer `--target 60` or higher on real captures.
    Denser sampling costs little and matters more than any tuning knob here.
  - *Plane-dominant scenes.* The 8-point algorithm cannot determine E uniquely
    when the points lie near a plane; RANSAC reports a large inlier set but the
    pose is wrong (measured: 99 inliers, 4 triangulated). Fix is
    homography/essential model selection as in ORB-SLAM.
  - *Focal estimation is not yet robust.* It recovered a plausible 0.75x long
    side (2880 px) on one feature set but 0.50x (1920 px) on a nearly identical
    one — and 0.50 is the edge of the sweep range, which means the score stopped
    discriminating rather than found a minimum. Consistent with the tray-
    dominated, near-planar feature distribution above: a degenerate
    configuration fits many focal lengths about equally well. Needs a
    stability check across pairs, and probably the planar fix first.

- **Splat model and forward rasterizer** (`SplatCore/Splatting/`) — first
  half of stage 5. Struct-of-arrays `SplatCloud` (position, log-scale,
  quaternion, opacity logit, colour), 3D covariance as Σ = R S Sᵀ Rᵀ, EWA
  projection to screen space, and a front-to-back alpha-compositing rasterizer
  with early termination. `SplatCloud.fromReconstruction` is the stage 3 → 5
  handoff: one Gaussian per sparse point, initial radius from nearest-neighbour
  distance so density follows structure.

  Parameters are stored in *unconstrained* form — log for scale, logit for
  opacity — so gradient steps need no projection back onto a valid range, and a
  fixed learning rate behaves the same for a huge splat as a tiny one.

  Verified against analytically known answers rather than self-consistency:
  identity rotation gives exactly `diag(s²)`, a 90° rotation about Z swaps the
  x/y extents exactly, doubling depth halves the screen radius to within 0.4%,
  a nearer splat occludes a farther one, and the render is bit-identical under
  reordered input.

  `splatctl render <input> --save <dir>` runs ingestion → filtering → SfM →
  splat init → render from every registered pose. On the real capture it
  produces coherent grey fog: correct, since an untrained cloud is by
  definition uniform mid-grey blobs. Making it resemble the input needs the
  backward pass.

- **Backward pass** (`SplatCore/Splatting/SplatBackward.swift`) — analytic
  gradients of an L1 image loss w.r.t. every splat parameter: colour, opacity
  logit, position, log-scale and rotation quaternion. The chain runs pixel →
  alpha blend → (colour, alpha) → G = exp(power) → conic → 2D covariance → 3D
  covariance → scale/rotation, and position → camera → world.

  **Every gradient is verified against finite differences**, which is the only
  test that distinguishes a derivative from something merely shaped like one: a
  sign error or dropped term still yields smooth, plausible gradients that
  decrease the loss for a while and then train to something subtly wrong.
  Measured agreement: colour, opacity and position.x to <0.1%; position.z,
  log-scale and rotation to 0.3–2%.

  Three things the derivation gets right that are easy to miss:
  - Alpha's gradient has an *indirect* term. Raising a splat's alpha also
    attenuates everything behind it, so the gradient is
    `c·T − (colour behind)/(1−α)`. The "colour behind" is recovered as
    `final − accumulated_in_front − this_splat` rather than cached, keeping
    memory at O(pixels) instead of O(pixels × splats) — tens of GB at 4K.
  - Position moves the 2D **covariance** as well as the centre, through the
    projection Jacobian. Omitting that term is silent: gradients stay plausible
    while position updates are wrong wherever a splat is off-axis.
  - The quaternion gradient includes the normalisation Jacobian
    `(I − q̂q̂ᵀ)/|q|`. Without it the optimiser spends updates changing the
    quaternion's *length*, which the rotation matrix ignores entirely.

  Note the tests use deliberately large splats. Finite differences on a splat
  covering few pixels are dominated by rasterization discretization (integer
  pixel bounds, alpha and transmittance cutoffs) — shrinking the test splats
  moved rotation agreement from 2% to 54% while the analytic value barely
  moved, which is a noisy *reference*, not a wrong derivative.

- **Optimizer and adaptive density control** (`SplatOptimizer`) — Adam with
  per-group learning rates, plus the clone / split / prune cycle that lets the
  splat count adapt to the scene. Density control is what makes 3DGS work at
  all: SfM yields a few thousand sparse points where a detailed scene needs
  hundreds of thousands of Gaussians, and no amount of gradient descent on a
  fixed set can create detail where there are no primitives.

  - **Clone** a small splat with a high screen-space gradient (under-covering
    its region), **split** a large one (spanning detail it cannot represent)
    into two children at 1/1.6 scale, offset by sampling the parent's own
    distribution, **prune** anything below `minOpacity` or above `maxWorldSize`.
  - Density keys off the **screen-space** gradient, not the world one: that
    signal is depth-independent, whereas a distant splat needs a far larger
    world movement to shift the same number of pixels.
  - Position learning rate scales with scene extent, because SfM fixes geometry
    only up to a similarity — a fixed rate is a crawl in one reconstruction and
    a catastrophe in another.
  - New splats get **zero** Adam moments rather than inheriting the parent's,
    which would apply momentum built from a different geometry.

  Verified end to end: starting from a displaced, mis-sized, grey cloud and
  training against renders of a known target from 3 viewpoints, **loss falls
  92% in 60 iterations** and the splats learn the *correct* colours (red → 1.0
  red, green → 1.0 green) rather than reducing loss by blurring to the mean.

  One threshold relationship is load-bearing and pinned by a test:
  `sizeThreshold` (split above this) must stay well below `maxWorldSize` (prune
  above this). Set equal, every splat large enough to split is also large enough
  to prune, so each split is undone in the same pass and its children
  discarded — measured as 4 pruned instead of 2. Defaults leave a 10× band.

Not yet implemented: checkpoint/resume, the Metal kernels for the splatting
path, viewer (stage 6), export (stage 7), and homography-based initialization
for planar scenes.

## Tier architecture

Two independent gates decide baseline vs enhanced, both must pass:

1. **Compile-time architecture gate** — enhanced-tier code sits inside
   `#if arch(arm64)`; it does not exist in the x86_64 slice (verified:
   `nm` shows zero enhanced symbols there). Apple-Silicon-only dependencies
   (MLX, later) will live exclusively inside that block.
2. **Runtime OS gate** — `EnhancedTierRequirements.minimumOS` (13.3, the MLX
   floor) checked at selection time, plus `if #available` at every
   construction site with the baseline path as the `else`.

`EngineFactory` re-checks both gates before instantiating, so even a stale
or deserialized "enhanced" decision degrades safely to baseline.

## Building

Three build paths, each with a distinct purpose:

**1. Everyday development — SPM (CLT is enough for arm64):**
```sh
swift run splatctl [probe|ingest|filter] ...
swift run selftest        # test suite (161 assertions, synthesizes its own fixtures)
```
Note: on macOS 27 tooling these local builds get a 12.0 deployment floor
(see below) — fine for development, not for release. The arm64 slice builds
with CLT alone; the **x86_64 slice needs `DEVELOPER_DIR=Xcode-beta`** because
the CLT 27 beta's static Swift compatibility libraries dropped their x86_64
slices (`swift build --triple x86_64-apple-macosx` fails to link otherwise).

**2. Xcode — open `App/GaussianSplatter.xcodeproj`** for UI work and
debugging. Its deployment target is 12.0 because Xcode 27 refuses anything
lower. Dev convenience only; never ships.

**3. Distribution — `Scripts/build-app.sh`:** produces
`dist/GaussianSplatter.app`, universal (arm64 + x86_64) with a verified
**macOS 11.0** floor, by driving `swiftc` directly against the macOS 26.2 SDK.

### Why the split (macOS 27 toolchain drift)

Every high-level build system on current tooling clamps the macOS deployment
target to 12.0: xcodebuild errors below 12.0, and SwiftPM 6.4 silently
raises the package's `.macOS(.v11)` floor (even with an explicit
`--triple arm64-apple-macosx11.0`). Raw `swiftc -target ...-apple-macos11.0`
still works, and the MacOSX26.2 SDK (shipping inside CLT 27) supports
deployment back to 10.13 — so the distribution script uses exactly that pair
and hard-fails its own verification if a slice comes out above 11.0.
Two further beta gotchas the script works around: the CLT 27 beta's static
Swift compatibility libraries lost their x86_64 slices (Xcode-beta's copies
still have them, so the script prefers `DEVELOPER_DIR=Xcode-beta`), and the
27.0 SDK's SwiftUI property wrappers are compiler macros whose plugins the
CLT doesn't ship (the 26.2 SDK's SwiftUI is pre-macro).

Tests are a plain `selftest` executable rather than a `.testTarget` because
the project must stay buildable with CLT alone (no XCTest/swift-testing).

### CI (`.github/workflows/ci.yml`)

Two jobs. `test` runs `swift run selftest` on `macos-latest` for fast
feedback. `deployment-floor` runs on `macos-26` — a runner whose toolchain
predates the 12.0 clamp — and executes `Scripts/ci-verify-floor.sh`, which
builds both `splatctl` and the app for arm64 **and** x86_64 via plain SPM,
asserts every slice reports `minos 11.0`, and checks enhanced-tier arch
gating. Running that script locally on macOS 27 is *expected to fail* the
minos assertion (SwiftPM clamps to 12.0) — that failure is the drift signal
it exists to produce.

Three things it does deliberately, each of which was a bug first:

- **Queries `--show-bin-path` instead of hardcoding output paths.** SwiftPM's
  layout is build-system dependent: the 6.4 `swiftbuild` system emits *every*
  triple to the same `.build/out/Products/Release` directory, so an x86_64
  build overwrites the arm64 one. Each slice is inspected and stashed
  immediately after its own build, before the next arch can clobber it.
- **Fails loudly on a missing binary.** `nm missing-file | grep -c` returns 0,
  which would otherwise sail through the "no enhanced symbols in x86_64"
  assertion and report a passing gate for something never built.
- **Asymmetric gate assertions.** x86_64 containing enhanced-tier symbols is a
  hard failure (the `#if arch(arm64)` gate is broken). arm64 *lacking* them is
  only a note — under `-O` the optimizer can legitimately dead-strip them, so
  failing on that would make CI permanently red for a non-defect. If no
  product retains them, the run is reported INCONCLUSIVE rather than passing.

**Scope limit, stated plainly:** this proves the floor for the *SwiftPM* path.
Releases ship via `Scripts/build-app.sh` (raw swiftc + pinned local SDK +
`DEVELOPER_DIR=Xcode-beta`), which cannot run on a GitHub runner — that script
carries its own equivalent minos assertion. Neither check alone covers both
paths, and the uploaded artifact is an unsigned SPM executable, not a release
bundle.

## Hard constraints (do not violate)

- macOS 11.0 deployment target, universal binary.
- No CUDA. All baseline GPU compute is hand-written Metal Shading Language.
- Baseline must fit 2 GB VRAM (GeForce GT 750M class) and never assume
  unified memory or Apple-Silicon-only Metal features.
- CPU (Accelerate/vDSP) fallback when Metal is unavailable.
- Narrow-range dependencies (MLX, PyTorch-MPS, …) only inside enhanced-tier
  conformances, gated as above. New dependencies need explicit sign-off.
