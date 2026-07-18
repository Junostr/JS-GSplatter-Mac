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

    private let workQueue = DispatchQueue(label: "splat.ingest", qos: .userInitiated)

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
