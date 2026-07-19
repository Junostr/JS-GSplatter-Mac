import AppKit
import Combine
import Foundation
import SplatCore

/// Bridges SplatCore to the UI. All pipeline work runs off the main thread;
/// only @Published mutations hop back to main.
final class PipelineModel: ObservableObject {

    let probe: SystemProbe

    @Published var forceBaseline = false {
        didSet { recomputeDecision() }
    }
    @Published private(set) var decision: TierDecision

    enum IngestState {
        case idle
        case running(frames: Int)
        case done(summaryLines: [String])
        case failed(String)
    }
    @Published private(set) var ingestState: IngestState = .idle
    @Published private(set) var previewImage: NSImage?

    /// A trained splat scene loaded for interactive viewing (stage 6).
    @Published private(set) var loadedCloud: SplatCloud?

    // MARK: Full-pipeline processing (video -> trained scene, in the app)

    enum ProcessState: Equatable {
        case idle
        case running(status: String, fraction: Double)   // fraction in [0, 1]
        case failed(String)
    }
    @Published private(set) var processState: ProcessState = .idle
    /// Live preview of the scene as it trains.
    @Published private(set) var trainingPreview: NSImage?

    /// How many training iterations a run does. Kept modest for a preview-grade
    /// result in reasonable time; a full-quality run is a longer, checkpointed
    /// affair better suited to the CLI for now.
    private let trainingIterations = 200
    private var cancelRequested = false

    private let workQueue = DispatchQueue(label: "splat.pipeline", qos: .userInitiated)

    /// Load a `.splt` checkpoint and hand it to the viewer. Decode runs off the
    /// main thread — a mature scene is hundreds of thousands of splats — and
    /// only the published assignment hops back.
    func loadScene(url: URL) {
        workQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let (cloud, _) = try SplatCheckpoint.read(from: url)
                DispatchQueue.main.async { self.loadedCloud = cloud }
            } catch {
                DispatchQueue.main.async { self.ingestState = .failed("Could not open scene: \(error)") }
            }
        }
    }

    func closeViewer() {
        loadedCloud = nil
    }

    func cancelProcessing() {
        cancelRequested = true
    }

    /// Run the full pipeline on a dropped capture and land in the viewer.
    /// Everything heavy is on the work queue; only @Published updates hop to
    /// main. Progress is coarse-grained by phase, with the long training phase
    /// mapped across most of the bar since it dominates wall-clock.
    func process(url: URL) {
        cancelRequested = false
        trainingPreview = nil
        processState = .running(status: "Reading capture…", fraction: 0.01)

        workQueue.async { [weak self] in
            guard let self = self else { return }

            func publish(_ state: ProcessState) { DispatchQueue.main.async { self.processState = state } }

            do {
                let source = try IngestionSource.detect(at: url)

                // Prep (ingest -> filter -> features -> focal -> SfM) is the
                // first ~30% of the bar; training is the rest.
                let prepared = try ScenePipeline.prepare(
                    source: source,
                    options: ScenePipelineOptions(targetFrames: 40),
                    progress: { phase in
                        publish(.running(status: Self.describe(phase), fraction: Self.fraction(phase)))
                    },
                    shouldCancel: { self.cancelRequested })

                // Training loop, owned here so a live preview can be rendered and
                // published as it converges.
                var trainerOptions = TrainerOptions()
                trainerOptions.densifyEnd = self.trainingIterations - 20
                let trainer = SplatTrainer(cloud: prepared.initialCloud, views: prepared.views,
                                           options: trainerOptions)
                let previewView = prepared.views.first

                for iteration in 1...self.trainingIterations {
                    if self.cancelRequested { throw ScenePipelineError.cancelled }
                    let report = trainer.step()
                    let fraction = 0.3 + 0.68 * Double(iteration) / Double(self.trainingIterations)
                    publish(.running(status: String(format: "Training… %d/%d  (loss %.3f, %d splats)",
                                                    iteration, self.trainingIterations, report.loss, report.splatCount),
                                     fraction: fraction))
                    // A live preview every so often — cheap on the GPU, and it
                    // makes the wait legible instead of a spinning bar.
                    if iteration % 15 == 0 || iteration == self.trainingIterations, let view = previewView {
                        let image = trainer.render(view: view)
                        if let cg = SplatOrbitView.cgImage(from: image) {
                            let ns = NSImage(cgImage: cg, size: NSSize(width: image.width, height: image.height))
                            DispatchQueue.main.async { self.trainingPreview = ns }
                        }
                    }
                }

                let finalCloud = trainer.cloud
                DispatchQueue.main.async {
                    self.processState = .idle
                    self.trainingPreview = nil
                    self.loadedCloud = finalCloud     // straight into the viewer
                }
            } catch is ScenePipelineError where self.cancelRequested {
                publish(.idle)
            } catch {
                publish(.failed("\(error)"))
            }
        }
    }

    private static func describe(_ phase: ScenePipelinePhase) -> String {
        switch phase {
        case .analyzingFrames(let n): return "Analyzing frames… (\(n))"
        case .selectingFrames: return "Selecting the sharpest, best-spread frames…"
        case .extractingFeatures(let done, let total): return "Finding features… (\(done)/\(total))"
        case .estimatingFocal: return "Estimating the lens…"
        case .reconstructing: return "Reconstructing camera poses (this is the slow part)…"
        case .preparingSplats(let n): return "Initialising \(n) splats…"
        }
    }

    private static func fraction(_ phase: ScenePipelinePhase) -> Double {
        switch phase {
        case .analyzingFrames: return 0.05
        case .selectingFrames: return 0.10
        case .extractingFeatures(let done, let total):
            return 0.10 + 0.12 * Double(done) / Double(max(total, 1))
        case .estimatingFocal: return 0.24
        case .reconstructing: return 0.27
        case .preparingSplats: return 0.30
        }
    }

    init() {
        let probe = HardwareProbe.run()
        self.probe = probe
        self.decision = TierSelector.decide(probe: probe)
    }

    private func recomputeDecision() {
        decision = TierSelector.decide(probe: probe, forceBaseline: forceBaseline)
    }

    /// Entry point the drop zone calls — same one `splatctl ingest` uses.
    func ingest(url: URL) {
        ingestState = .running(frames: 0)
        previewImage = nil

        workQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let source = try IngestionSource.detect(at: url)
                var preview: NSImage?
                let start = Date()
                let summary = try FrameIngestor.ingest(
                    source,
                    // Cap dropped videos: stage 2 (frame filtering) will pick
                    // the good frames; ingestion just needs candidates.
                    options: IngestionOptions(maxFrames: 300)
                ) { frame in
                    if frame.index == 0 {
                        // Copy now — video pixel buffers are only valid
                        // during the handler call.
                        if let cgImage = try? frame.makeCGImage() {
                            preview = NSImage(cgImage: cgImage, size: .zero)
                        }
                    }
                    if frame.index % 10 == 0 {
                        let count = frame.index + 1
                        DispatchQueue.main.async {
                            self.ingestState = .running(frames: count)
                        }
                    }
                    return true
                }
                let elapsed = Date().timeIntervalSince(start)

                var lines = [
                    "\(summary.deliveredFrames) frames (\(summary.decodedFrames) decoded) in \(String(format: "%.2f", elapsed)) s",
                    "\(summary.width) × \(summary.height)",
                ]
                if let duration = summary.duration, let fps = summary.nominalFrameRate {
                    lines.append("\(String(format: "%.1f", duration)) s video @ \(String(format: "%.1f", fps)) fps")
                }
                DispatchQueue.main.async {
                    self.previewImage = preview
                    self.ingestState = .done(summaryLines: lines)
                }
            } catch {
                DispatchQueue.main.async {
                    self.ingestState = .failed("\(error)")
                }
            }
        }
    }
}
