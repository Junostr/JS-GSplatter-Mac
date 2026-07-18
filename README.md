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
  - *Incremental registration stalls after the first camera.* Registration now
    works (2 -> 3 cameras on the real capture, RMSE 0.878 px) but stops there:
    a new frame only gets points triangulated where it matched an
    ALREADY-registered frame, so coverage does not propagate around an orbit —
    the next frame is left with ~7 correspondences, below what PnP needs. The
    fix is a re-triangulation loop that re-triangulates all match pairs among
    registered cameras after each registration, with periodic bundle
    adjustment. This is the main outstanding item.
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

Not yet implemented: real training kernels (stage 5), viewer (stage 6),
export (stage 7), and homography-based initialization for planar scenes.

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
