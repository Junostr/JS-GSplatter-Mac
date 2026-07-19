import SwiftUI
import SplatCore
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = PipelineModel()
    @State private var isDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            tierSection
            Divider()
            if let cloud = model.loadedCloud {
                viewerSection(cloud: cloud)
            } else {
                dropZone
                statusSection
            }
            Spacer()
        }
        .padding(20)
    }

    private func viewerSection(cloud: SplatCloud) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Viewing \(cloud.count) splats")
                    .font(.callout).bold()
                Spacer()
                Button("Close") { model.closeViewer() }
            }
            SplatViewerRepresentable(cloud: cloud)
                .frame(minWidth: 480, minHeight: 360)
                .cornerRadius(8)
            Text("Drag to orbit · scroll to zoom · ⌥-drag to pan")
                .font(.caption).foregroundColor(.secondary)
        }
    }

    private var tierSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Compute tier: \(model.decision.tier.label)")
                    .font(.headline)
                Spacer()
                Toggle("Force baseline", isOn: $model.forceBaseline)
                    .toggleStyle(SwitchToggleStyle())
            }
            if let gpu = model.decision.selectedGPU {
                Text("\(gpu.name) — working set \(String(format: "%.1f", Double(gpu.recommendedWorkingSetBytes) / Double(1 << 30))) GiB")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            ForEach(model.decision.reasons, id: \.self) { reason in
                Text("• \(reason)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var dropZone: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: 2, dash: [8])
                )
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isDropTargeted ? Color.accentColor.opacity(0.1) : Color.clear)
                )
            VStack(spacing: 8) {
                if let preview = model.previewImage {
                    Image(nsImage: preview)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 180)
                        .cornerRadius(6)
                }
                Text("Drop a photo folder, a video, or a .splt scene here")
                    .foregroundColor(.secondary)
            }
            .padding(12)
        }
        .frame(minHeight: 220)
        .onDrop(of: [UTType.fileURL], isTargeted: $isDropTargeted) { providers in
            guard let provider = providers.first else { return false }
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                let url: URL?
                if let data = item as? Data {
                    url = URL(dataRepresentation: data, relativeTo: nil)
                } else {
                    url = item as? URL
                }
                if let url = url {
                    DispatchQueue.main.async {
                        // A trained scene opens straight into the viewer;
                        // anything else is treated as capture input.
                        if url.pathExtension.lowercased() == "splt" {
                            model.loadScene(url: url)
                        } else {
                            model.ingest(url: url)
                        }
                    }
                }
            }
            return true
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch model.ingestState {
        case .idle:
            EmptyView()
        case .running(let frames):
            HStack(spacing: 8) {
                ProgressView()
                    .scaleEffect(0.6)
                Text("Ingesting… \(frames) frames")
                    .font(.callout)
            }
        case .done(let lines):
            VStack(alignment: .leading, spacing: 2) {
                Text("Ingestion complete")
                    .font(.callout).bold()
                ForEach(lines, id: \.self) { line in
                    Text(line).font(.caption).foregroundColor(.secondary)
                }
            }
        case .failed(let message):
            Text(message)
                .font(.callout)
                .foregroundColor(.red)
        }
    }
}
