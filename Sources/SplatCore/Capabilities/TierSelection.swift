import Foundation

// MARK: - Tier model

/// Sub-tiers of the baseline (hand-written MSL) path. These do not change
/// *which* code runs — the baseline kernels are one code path — only the
/// parameters it runs with (splat ceiling, tile size, precision, memory budget).
public enum BaselineSubTier: String, Codable {
    /// Legacy discrete Nvidia (Kepler-era, e.g. GeForce GT 750M, 2 GB VRAM).
    /// The most constrained device in scope; everything must fit its limits.
    case legacyNVIDIA
    /// Discrete AMD (Radeon Pro / Vega found in 2015–2020 Macs).
    case discreteAMD
    /// Intel integrated (Iris / UHD). Shared memory but weak ALU throughput.
    case integratedIntel
    /// Apple Silicon running the baseline kernels (either forced, or the OS
    /// is too old for the enhanced tier's dependencies).
    case appleSilicon
    /// No usable Metal device — Accelerate/vDSP on the CPU.
    case cpuFallback
}

public enum ComputeTier: Equatable, Codable {
    case baseline(BaselineSubTier)
    case enhanced

    public var label: String {
        switch self {
        case .baseline(let sub): return "baseline (\(sub.rawValue))"
        case .enhanced: return "enhanced"
        }
    }
}

public enum SplatPrecision: String, Codable {
    case fp16
    case fp32
}

/// Default knobs for the baseline kernels, set per sub-tier. These are
/// starting points for real-hardware tuning later; the reasoning for each
/// choice is documented at the assignment site in `TierSelector`.
public struct BaselineParameters: Equatable, Codable {
    /// Hard ceiling on the number of Gaussians during training.
    public let maxSplatCount: Int
    /// Rasterizer tile edge in pixels (tiles are square).
    public let tileSize: Int
    /// Precision for stored splat attributes (positions stay fp32 always —
    /// fp16 positions visibly quantize scene geometry).
    public let storagePrecision: SplatPrecision
    /// Precision for in-kernel arithmetic.
    public let computePrecision: SplatPrecision
    /// Budget for persistent GPU allocations, derived from the device's
    /// recommended working set — never from total system RAM.
    public let gpuMemoryBudgetBytes: UInt64

    public init(maxSplatCount: Int, tileSize: Int, storagePrecision: SplatPrecision,
                computePrecision: SplatPrecision, gpuMemoryBudgetBytes: UInt64) {
        self.maxSplatCount = maxSplatCount
        self.tileSize = tileSize
        self.storagePrecision = storagePrecision
        self.computePrecision = computePrecision
        self.gpuMemoryBudgetBytes = gpuMemoryBudgetBytes
    }
}

/// The full, explainable result of tier selection.
public struct TierDecision: Equatable, Codable {
    public let tier: ComputeTier
    public let parameters: BaselineParameters
    /// The GPU the decision is based on (nil in the CPU-fallback case).
    public let selectedGPU: GPUInfo?
    /// Human-readable trail of why this tier was chosen — surfaced in logs
    /// so decisions can be verified against real hardware.
    public let reasons: [String]
}

// MARK: - Enhanced-tier gates

public enum EnhancedTierRequirements {
    /// Runtime OS gate for the enhanced tier. macOS 13.3 is the floor for the
    /// dependencies we intend to use there (MLX requires ≥ 13.3; the MPSGraph
    /// features worth having over hand-written kernels also land around 13.x).
    /// Raising this constant never affects the baseline path. Keep the
    /// @available annotations in EnhancedTrainingEngine.swift in sync.
    public static let minimumOS = OSVersion(13, 3)

    /// Compile-time architecture gate: whether enhanced-tier code was compiled
    /// into *this* slice of the universal binary at all. On x86_64 (including
    /// Rosetta) this is false and no Apple-Silicon-only dependency is linked.
    public static var compiledIntoThisSlice: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }
}

// MARK: - Selector

public enum TierSelector {

    /// Pure function: facts in, decision out. `forceBaseline` is the manual
    /// override for testing/debugging on machines that would pick enhanced.
    /// `enhancedCompiledIn` defaults to the real compile-time gate but is
    /// injectable so tests can exercise both slices from either architecture.
    public static func decide(
        probe: SystemProbe,
        forceBaseline: Bool = false,
        enhancedCompiledIn: Bool = EnhancedTierRequirements.compiledIntoThisSlice
    ) -> TierDecision {
        var reasons: [String] = []
        let gpu = pickGPU(from: probe.gpus, reasons: &reasons)

        // --- Enhanced-tier gating: both gates must pass, and nothing forces baseline.
        if let gpu = gpu, gpu.vendor == .apple {
            let archOK = probe.architecture == .arm64 && enhancedCompiledIn
            let osOK = probe.osVersion >= EnhancedTierRequirements.minimumOS

            if forceBaseline {
                reasons.append("Enhanced tier available but disabled by --force-baseline override.")
            } else if !archOK {
                if probe.isTranslated {
                    reasons.append("Running x86_64 slice under Rosetta 2; enhanced tier needs the native arm64 slice. Relaunch natively to enable it.")
                } else {
                    reasons.append("Enhanced tier not compiled into this slice (arch gate).")
                }
            } else if !osOK {
                reasons.append("macOS \(probe.osVersion) is below the enhanced tier's floor of \(EnhancedTierRequirements.minimumOS) (availability gate).")
            } else {
                reasons.append("Apple Silicon + macOS \(probe.osVersion) ≥ \(EnhancedTierRequirements.minimumOS): both enhanced-tier gates pass.")
                return TierDecision(
                    tier: .enhanced,
                    // Enhanced still carries baseline parameters: the baseline
                    // engine must remain constructible at any moment (e.g. if an
                    // enhanced dependency fails to initialize at runtime).
                    parameters: appleSiliconParameters(for: gpu),
                    selectedGPU: gpu,
                    reasons: reasons
                )
            }
        }

        // --- Baseline sub-tier selection.
        guard let gpu = gpu else {
            reasons.append("No Metal device found; using Accelerate/vDSP CPU fallback.")
            return TierDecision(
                tier: .baseline(.cpuFallback),
                // CPU path: tileSize is the vDSP block edge, budget is a
                // conservative slice of RAM rather than a GPU working set.
                parameters: BaselineParameters(
                    maxSplatCount: 200_000,
                    tileSize: 16,
                    storagePrecision: .fp32,
                    computePrecision: .fp32,
                    gpuMemoryBudgetBytes: 1 << 30
                ),
                selectedGPU: nil,
                reasons: reasons
            )
        }

        let subTier: BaselineSubTier
        let params: BaselineParameters
        switch gpu.vendor {
        case .nvidia:
            subTier = .legacyNVIDIA
            reasons.append("Legacy discrete Nvidia GPU (\(gpu.name)): most constrained baseline sub-tier.")
            // GT 750M reasoning: Kepler has no fast fp16 ALUs, so compute stays
            // fp32; fp16 is storage-only to halve VRAM traffic. 16 px tiles keep
            // per-tile sort lists small enough for 2 GB VRAM. Splat ceiling and
            // budget leave headroom for the OS/WindowServer share of VRAM.
            params = BaselineParameters(
                maxSplatCount: 1_000_000,
                tileSize: 16,
                storagePrecision: .fp16,
                computePrecision: .fp32,
                gpuMemoryBudgetBytes: scaledBudget(gpu, fraction: 0.6)
            )
        case .amd:
            subTier = .discreteAMD
            reasons.append("Discrete AMD GPU (\(gpu.name)).")
            // GCN/RDNA: fp16 storage is free bandwidth savings; fp16 *compute*
            // gains vary wildly across GCN generations, so default fp32 until
            // tuned on hardware. 32 px tiles suit the wider wavefronts (64 lanes).
            params = BaselineParameters(
                maxSplatCount: splatCeiling(gpu, perGiB: 1_000_000, cap: 4_000_000),
                tileSize: 32,
                storagePrecision: .fp16,
                computePrecision: .fp32,
                gpuMemoryBudgetBytes: scaledBudget(gpu, fraction: 0.7)
            )
        case .intel:
            subTier = .integratedIntel
            reasons.append("Intel integrated GPU (\(gpu.name)): shared memory, limited ALU throughput.")
            // Shared memory means the working set competes with the app itself;
            // stay conservative. Small tiles bound threadgroup memory, which is
            // scarce on Gen9/Gen11.
            params = BaselineParameters(
                maxSplatCount: 750_000,
                tileSize: 16,
                storagePrecision: .fp16,
                computePrecision: .fp32,
                gpuMemoryBudgetBytes: scaledBudget(gpu, fraction: 0.5)
            )
        case .apple:
            subTier = .appleSilicon
            reasons.append("Apple Silicon GPU (\(gpu.name)) on the baseline path.")
            params = appleSiliconParameters(for: gpu)
        case .unknown:
            subTier = .integratedIntel
            reasons.append("Unrecognized GPU vendor for \(gpu.name); using conservative integrated-class parameters.")
            params = BaselineParameters(
                maxSplatCount: 500_000,
                tileSize: 16,
                storagePrecision: .fp32,
                computePrecision: .fp32,
                gpuMemoryBudgetBytes: scaledBudget(gpu, fraction: 0.5)
            )
        }

        return TierDecision(tier: .baseline(subTier), parameters: params, selectedGPU: gpu, reasons: reasons)
    }

    // Apple GPUs run fp16 at double rate and the unified working set is large;
    // fp16 compute is the right default (positions still fp32, see
    // BaselineParameters.storagePrecision doc). 32 px tiles match the 32-wide
    // SIMD groups and generous threadgroup memory.
    static func appleSiliconParameters(for gpu: GPUInfo) -> BaselineParameters {
        BaselineParameters(
            maxSplatCount: splatCeiling(gpu, perGiB: 750_000, cap: 6_000_000),
            tileSize: 32,
            storagePrecision: .fp16,
            computePrecision: .fp16,
            gpuMemoryBudgetBytes: scaledBudget(gpu, fraction: 0.5)
        )
    }

    /// Prefer the most capable device: discrete/high-power first, then the
    /// largest recommended working set. External (eGPU) devices participate
    /// on equal terms; headless compute devices are fine for training.
    static func pickGPU(from gpus: [GPUInfo], reasons: inout [String]) -> GPUInfo? {
        guard !gpus.isEmpty else { return nil }
        let best = gpus.sorted { a, b in
            if a.isLowPower != b.isLowPower { return !a.isLowPower }
            return a.recommendedWorkingSetBytes > b.recommendedWorkingSetBytes
        }.first!
        if gpus.count > 1 {
            reasons.append("Multiple Metal devices (\(gpus.map { $0.name }.joined(separator: ", "))); selected \(best.name).")
        }
        return best
    }

    static func splatCeiling(_ gpu: GPUInfo, perGiB: Int, cap: Int) -> Int {
        let gib = Double(gpu.recommendedWorkingSetBytes) / Double(1 << 30)
        return min(cap, max(250_000, Int(gib * Double(perGiB))))
    }

    static func scaledBudget(_ gpu: GPUInfo, fraction: Double) -> UInt64 {
        UInt64(Double(gpu.recommendedWorkingSetBytes) * fraction)
    }

}
