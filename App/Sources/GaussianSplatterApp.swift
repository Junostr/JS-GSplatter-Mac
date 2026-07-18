import SwiftUI

// App shell. Deployment target is macOS 11.0, so everything in App/Sources
// must stick to SwiftUI's Big Sur surface (WindowGroup, onDrop(of:[UTType]),
// ProgressView are all 11.0-safe; .task, Table, etc. are not).
@main
struct GaussianSplatterApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 640, minHeight: 480)
        }
    }
}
