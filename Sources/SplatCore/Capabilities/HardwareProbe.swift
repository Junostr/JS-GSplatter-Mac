import Foundation
import Metal

// MARK: - Probe data model
//
// The probe is deliberately split from the *decision* (TierSelection.swift):
// the probe only gathers facts about the machine; the selector turns facts
// into a tier. That keeps the selection logic a pure function we can unit-test
// against synthetic probes for hardware we don't have on the build machine
// (e.g. a 2014 MacBook Pro with a GeForce GT 750M).

public enum GPUVendor: String, Codable {
    case apple = "Apple"
    case amd = "AMD"
    case nvidia = "NVIDIA"
    case intel = "Intel"
    case unknown = "Unknown"
}

public enum ProcessArchitecture: String, Codable {
    case arm64
    case x86_64

    /// Architecture of the *running slice* of the universal binary.
    /// Under Rosetta 2 this is `x86_64` even on Apple Silicon hardware —
    /// which is exactly what we want for tier selection, because an x86_64
    /// process cannot load arm64-only enhanced-tier code (e.g. MLX).
    public static var current: ProcessArchitecture {
        #if arch(arm64)
        return .arm64
        #else
        return .x86_64
        #endif
    }
}

public struct GPUInfo: Codable, Equatable {
    public let name: String
    public let vendor: GPUVendor
    /// Metal's advisory budget for GPU-resident data. On discrete GPUs this
    /// tracks VRAM (~1.5 GB on a 2 GB GT 750M); on Apple Silicon it is a
    /// slice of unified memory. Baseline kernels must size all persistent
    /// allocations against this, never against total system RAM.
    public let recommendedWorkingSetBytes: UInt64
    public let hasUnifiedMemory: Bool
    public let isLowPower: Bool
    public let isRemovable: Bool
    /// True for headless compute devices (no display attached).
    public let isHeadless: Bool

    public init(name: String, vendor: GPUVendor, recommendedWorkingSetBytes: UInt64,
                hasUnifiedMemory: Bool, isLowPower: Bool, isRemovable: Bool, isHeadless: Bool) {
        self.name = name
        self.vendor = vendor
        self.recommendedWorkingSetBytes = recommendedWorkingSetBytes
        self.hasUnifiedMemory = hasUnifiedMemory
        self.isLowPower = isLowPower
        self.isRemovable = isRemovable
        self.isHeadless = isHeadless
    }
}

/// Our own version triple instead of Foundation's OperatingSystemVersion,
/// which conforms to neither Codable nor Equatable (and retroactive
/// conformances on imported types are fragile).
public struct OSVersion: Codable, Equatable, Comparable, CustomStringConvertible {
    public let major: Int
    public let minor: Int
    public let patch: Int

    public init(_ major: Int, _ minor: Int, _ patch: Int = 0) {
        self.major = major
        self.minor = minor
        self.patch = patch
    }

    public init(_ v: OperatingSystemVersion) {
        self.init(v.majorVersion, v.minorVersion, v.patchVersion)
    }

    public static func < (lhs: OSVersion, rhs: OSVersion) -> Bool {
        (lhs.major, lhs.minor, lhs.patch) < (rhs.major, rhs.minor, rhs.patch)
    }

    public var description: String {
        "\(major).\(minor)" + (patch > 0 ? ".\(patch)" : "")
    }
}

public struct SystemProbe: Codable, Equatable {
    public let osVersion: OSVersion
    public let architecture: ProcessArchitecture
    /// True when the x86_64 slice is running under Rosetta 2 on Apple Silicon.
    /// The tier stays baseline (the process genuinely is x86_64), but we
    /// surface it so the user knows a native launch would do better.
    public let isTranslated: Bool
    /// All Metal devices, in probe order. Empty means Metal is unavailable
    /// and the CPU (Accelerate/vDSP) fallback is the only option.
    public let gpus: [GPUInfo]

    public init(osVersion: OSVersion, architecture: ProcessArchitecture,
                isTranslated: Bool, gpus: [GPUInfo]) {
        self.osVersion = osVersion
        self.architecture = architecture
        self.isTranslated = isTranslated
        self.gpus = gpus
    }
}

// MARK: - Live probe

public enum HardwareProbe {

    /// Gather facts about the current machine. Every API used here exists on
    /// macOS 11.0 (most since 10.13–10.15), so the probe itself is baseline-safe.
    public static func run() -> SystemProbe {
        let devices = MTLCopyAllDevices()
        let gpus = devices.map { describe($0) }
        return SystemProbe(
            osVersion: OSVersion(ProcessInfo.processInfo.operatingSystemVersion),
            architecture: .current,
            isTranslated: isRosettaTranslated(),
            gpus: gpus
        )
    }

    static func describe(_ device: MTLDevice) -> GPUInfo {
        GPUInfo(
            name: device.name,
            vendor: vendor(fromDeviceName: device.name),
            recommendedWorkingSetBytes: device.recommendedMaxWorkingSetSize,
            hasUnifiedMemory: device.hasUnifiedMemory,
            isLowPower: device.isLowPower,
            isRemovable: device.isRemovable,
            isHeadless: device.isHeadless
        )
    }

    /// Vendor from the device name string. MTLDevice has no vendor-ID property
    /// on macOS 11, and IOKit registry walking is overkill for four vendors —
    /// Metal device names are stable marketing names ("Apple M1", "AMD Radeon
    /// Pro 5500M", "NVIDIA GeForce GT 750M", "Intel(R) Iris(TM) Plus Graphics").
    public static func vendor(fromDeviceName name: String) -> GPUVendor {
        let lowered = name.lowercased()
        if lowered.contains("apple") { return .apple }
        if lowered.contains("nvidia") || lowered.contains("geforce") || lowered.contains("quadro") {
            return .nvidia
        }
        if lowered.contains("amd") || lowered.contains("radeon") || lowered.contains("firepro") || lowered.contains("vega") {
            return .amd
        }
        if lowered.contains("intel") || lowered.contains("iris") || lowered.contains("hd graphics") || lowered.contains("uhd graphics") {
            return .intel
        }
        return .unknown
    }

    /// sysctl.proc_translated is 1 under Rosetta 2, 0 native, and the sysctl
    /// does not exist at all on Intel hardware or pre-11.0 systems (-1 path).
    static func isRosettaTranslated() -> Bool {
        var translated: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let result = sysctlbyname("sysctl.proc_translated", &translated, &size, nil, 0)
        return result == 0 && translated == 1
    }
}
