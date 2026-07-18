import Foundation
import Metal

/// Baseline tier: hand-written Metal Shading Language compute kernels.
/// This is the only engine guaranteed to exist on every machine in scope
/// (Big Sur+, Intel with legacy Nvidia, Apple Silicon). Real forward/backward
/// rasterization kernels arrive with pipeline stage 5; today this stub proves
/// the interface, device setup, and tier plumbing end to end.
public final class BaselineMetalTrainingEngine: TrainingEngine {

    public let tier: ComputeTier
    private let device: MTLDevice
    private var commandQueue: MTLCommandQueue?
    private var configuration: TrainingConfiguration?
    private var iteration = 0

    public var descriptionForLog: String {
        "Baseline Metal engine on \(device.name)"
    }

    public init(device: MTLDevice, subTier: BaselineSubTier) {
        self.device = device
        self.tier = .baseline(subTier)
    }

    public func prepare(configuration: TrainingConfiguration) throws {
        guard let queue = device.makeCommandQueue() else {
            throw TrainingEngineError.deviceUnavailable("Could not create command queue on \(device.name)")
        }
        self.commandQueue = queue
        self.configuration = configuration
        self.iteration = 0
    }

    public func step() throws -> TrainingStepResult {
        guard commandQueue != nil, let config = configuration else {
            throw TrainingEngineError.notPrepared
        }
        // Placeholder dynamics so callers can already exercise progress
        // reporting; replaced by real kernel dispatches in stage 5.
        iteration += 1
        let fakeLoss = 1.0 / Double(iteration)
        return TrainingStepResult(
            iteration: iteration,
            loss: fakeLoss,
            splatCount: min(iteration * 10_000, config.parameters.maxSplatCount)
        )
    }

    public func teardown() {
        commandQueue = nil
        configuration = nil
    }
}

/// CPU fallback for machines where Metal compute is not viable at all.
/// Will be backed by Accelerate/vDSP; the same stub contract as above for now.
public final class CPUFallbackTrainingEngine: TrainingEngine {

    public let tier: ComputeTier = .baseline(.cpuFallback)
    private var configuration: TrainingConfiguration?
    private var iteration = 0

    public var descriptionForLog: String {
        "CPU fallback engine (Accelerate/vDSP)"
    }

    public init() {}

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
            splatCount: min(iteration * 2_000, config.parameters.maxSplatCount)
        )
    }

    public func teardown() {
        configuration = nil
    }
}
