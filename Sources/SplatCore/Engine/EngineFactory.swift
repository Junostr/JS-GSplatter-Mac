import Foundation
import Metal

public enum EngineFactory {

    /// Build the training engine for a tier decision. Defense in depth: even
    /// if a decision says `.enhanced`, this factory re-checks both gates and
    /// silently degrades to the baseline engine — the baseline path must be
    /// reachable from every state the app can get into.
    public static func makeEngine(for decision: TierDecision) -> TrainingEngine {
        guard let gpuInfo = decision.selectedGPU,
              let device = metalDevice(named: gpuInfo.name) else {
            return CPUFallbackTrainingEngine()
        }

        switch decision.tier {
        case .enhanced:
            #if arch(arm64)
            if #available(macOS 13.3, *) {
                return EnhancedTrainingEngine(device: device)
            }
            #endif
            // Gate re-check failed (shouldn't happen if the decision came from
            // TierSelector, but never trust a serialized/stale decision).
            return BaselineMetalTrainingEngine(device: device, subTier: .appleSilicon)

        case .baseline(.cpuFallback):
            return CPUFallbackTrainingEngine()

        case .baseline(let subTier):
            return BaselineMetalTrainingEngine(device: device, subTier: subTier)
        }
    }

    static func metalDevice(named name: String) -> MTLDevice? {
        let devices = MTLCopyAllDevices()
        return devices.first { $0.name == name } ?? devices.first
    }
}
