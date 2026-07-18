// swift-tools-version:5.9
// Tools version 5.9 keeps us in the Swift 5 language mode: no strict-concurrency
// requirements leak into baseline code that must stay simple and macOS 11-compatible.
import PackageDescription

let package = Package(
    name: "GaussianSplatterMac",
    // HARD CONSTRAINT: deployment target macOS 11.0 (Big Sur), universal binary.
    // Nothing in this manifest may raise this floor. Apple-Silicon-only or
    // newer-OS dependencies belong exclusively to enhanced-tier conformances,
    // gated at compile time (#if arch(arm64)) and runtime (#available).
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .library(name: "SplatCore", targets: ["SplatCore"]),
        .executable(name: "splatctl", targets: ["splatctl"]),
        .executable(name: "selftest", targets: ["selftest"]),
        // Declared explicitly rather than relying on SwiftPM's implicit
        // product for executable targets: CI builds it by name
        // (`swift build --product GaussianSplatterApp`), and an implicit
        // product is not a contract worth betting the release check on.
        .executable(name: "GaussianSplatterApp", targets: ["GaussianSplatterApp"]),
    ],
    // No external dependencies. Per project policy, any new dependency that could
    // affect the Big Sur/Intel/Nvidia baseline or the enhanced tier requires
    // explicit sign-off before being added here.
    dependencies: [],
    targets: [
        .target(
            name: "SplatCore",
            dependencies: [],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .executableTarget(
            name: "splatctl",
            dependencies: ["SplatCore"]
        ),
        // Plain executable instead of a .testTarget: this project must build
        // with Command Line Tools alone, which ship neither XCTest nor
        // swift-testing. Run with `swift run selftest`.
        .executableTarget(
            name: "selftest",
            dependencies: ["SplatCore"]
        ),
        // The GUI app, built from the SAME sources the Xcode project uses.
        // This SPM target exists because Xcode 27 refuses deployment targets
        // below 12.0, while SPM+CLT still emits minos 11.0 — so distribution
        // builds go through Scripts/build-app.sh (SPM build + manual bundle
        // assembly), and the Xcode project is a dev-only convenience.
        .executableTarget(
            name: "GaussianSplatterApp",
            dependencies: ["SplatCore"],
            path: "App/Sources"
        ),
    ]
)
