import Foundation

// MARK: - Shared engine interface
//
// Everything downstream of training (viewer, export, UI) talks to this
// protocol and must never need to know which tier produced a result.
// Both tiers conform; the baseline conformance always exists.

public struct TrainingConfiguration: Equatable {
    public let parameters: BaselineParameters

    public init(parameters: BaselineParameters) {
        self.parameters = parameters
    }
}

public struct TrainingStepResult: Equatable {
    public let iteration: Int
    public let loss: Double
    public let splatCount: Int

    public init(iteration: Int, loss: Double, splatCount: Int) {
        self.iteration = iteration
        self.loss = loss
        self.splatCount = splatCount
    }
}

public enum TrainingEngineError: Error, Equatable {
    case notPrepared
    case deviceUnavailable(String)
}

public protocol TrainingEngine: AnyObject {
    /// Short human-readable identity ("Baseline Metal on NVIDIA GeForce GT 750M").
    var descriptionForLog: String { get }
    /// The tier this engine implements, for surfacing in logs/UI.
    var tier: ComputeTier { get }

    /// Allocate device resources for a run. Must be callable again after
    /// `teardown` (checkpoint/resume will rely on this later).
    func prepare(configuration: TrainingConfiguration) throws
    /// One optimization step. Stubbed for now: proves the interface and tier
    /// plumbing end to end without real training logic.
    func step() throws -> TrainingStepResult
    func teardown()
}
