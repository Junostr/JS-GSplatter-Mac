import Foundation
import Metal

// ============================================================================
// Enhanced tier — BOTH gates demonstrated here:
//
//  1. Compile-time architecture gate: the entire type only exists in the
//     arm64 slice of the universal binary. When MLX (Apple Silicon-only) is
//     eventually imported here, it will not even attempt to link on x86_64.
//  2. Runtime OS gate: @available matches EnhancedTierRequirements.minimumOS;
//     construction sites additionally use `if #available` so the baseline
//     path is always the `else`.
//
// No MLX/MPSGraph dependency is imported yet — adding one requires the
// project's dependency check-in. Today this stub only proves the gating and
// the shared interface.
// ============================================================================

#if arch(arm64)

@available(macOS 13.3, *)
public final class EnhancedTrainingEngine: TrainingEngine {

    public let tier: ComputeTier = .enhanced
    private let device: MTLDevice
    private var configuration: TrainingConfiguration?
    private var iteration = 0

    public var descriptionForLog: String {
        "Enhanced engine (MLX/MPSGraph, stub) on \(device.name)"
    }

    public init(device: MTLDevice) {
        self.device = device
    }

    public func prepare(configuration: TrainingConfiguration) throws {
        self.configuration = configuration
        self.iteration = 0
    }

    public func step() throws -> TrainingStepResult {
        guard let config = configuration else { throw TrainingEngineError.notPrepared }
        iteration += 1
        return TrainingStepResult(
            iteration: iteration,
            loss: 1.0 / Double(iteration),
            splatCount: min(iteration * 25_000, config.parameters.maxSplatCount)
        )
    }

    public func teardown() {
        configuration = nil
    }
}

#endif // arch(arm64)
