import Foundation

// MARK: - Stage 5: checkpoint / resume
//
// Persist a splat cloud so a training run survives interruption. A full-quality
// run is hours to days on the baseline tier, so "the machine slept and lost
// everything" has to be a recoverable event, not a restart.
//
// Format is a small binary container, not JSON: a mature scene is hundreds of
// thousands of splats at 60 bytes each, tens of megabytes, and JSON would
// triple that and parse slowly. The layout is a fixed header (magic, version,
// count) followed by each attribute as a contiguous little-endian float block,
// in the same struct-of-arrays order the cloud already uses — so writing and
// reading are each a handful of bulk copies.
//
// Little-endian is written explicitly rather than dumping native memory,
// because the universal binary runs on both arm64 and x86_64. They happen to
// share byte order today, but relying on that is the kind of assumption that
// silently corrupts a file the first time it is wrong.

public enum SplatCheckpoint {

    static let magic: UInt32 = 0x53504C54          // "SPLT"
    static let version: UInt32 = 1
    /// Floats per splat: pos(3) logScale(3) rot(4) opacity(1) color(3) = 14.
    static let floatsPerSplat = 14

    public enum CheckpointError: Error, CustomStringConvertible {
        case tooShort
        case badMagic
        case unsupportedVersion(UInt32)
        case truncated(expected: Int, got: Int)

        public var description: String {
            switch self {
            case .tooShort: return "checkpoint file is too small to contain a header"
            case .badMagic: return "not a splat checkpoint (bad magic)"
            case .unsupportedVersion(let v): return "unsupported checkpoint version \(v)"
            case .truncated(let e, let g): return "checkpoint truncated: expected \(e) floats, got \(g)"
            }
        }
    }

    public static func encode(_ cloud: SplatCloud, iteration: Int) -> Data {
        var data = Data()
        func putUInt32(_ value: UInt32) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        putUInt32(magic)
        putUInt32(version)
        putUInt32(UInt32(cloud.count))
        putUInt32(UInt32(max(0, iteration)))

        // Attribute blocks, each fully contiguous. Reserving up front avoids
        // repeated reallocation while appending a large scene.
        var floats = [Float]()
        floats.reserveCapacity(cloud.count * floatsPerSplat)
        for i in 0..<cloud.count {
            let p = cloud.positions[i], s = cloud.logScales[i]
            let r = cloud.rotations[i], c = cloud.colors[i]
            floats.append(contentsOf: [p.x, p.y, p.z, s.x, s.y, s.z,
                                       r.x, r.y, r.z, r.w, cloud.opacityLogits[i],
                                       c.x, c.y, c.z])
        }
        for value in floats {
            var le = value.bitPattern.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }

    public static func decode(_ data: Data) throws -> (cloud: SplatCloud, iteration: Int) {
        guard data.count >= 16 else { throw CheckpointError.tooShort }
        // Copy out of the possibly-unaligned Data before interpreting: reading
        // a UInt32 straight from an arbitrary Data offset can trap on a strict-
        // alignment target, and this binary must run on both architectures.
        func readUInt32(_ offset: Int) -> UInt32 {
            var value: UInt32 = 0
            withUnsafeMutableBytes(of: &value) { dst in
                data.copyBytes(to: dst, from: offset..<(offset + 4))
            }
            return UInt32(littleEndian: value)
        }
        guard readUInt32(0) == magic else { throw CheckpointError.badMagic }
        let fileVersion = readUInt32(4)
        guard fileVersion == version else { throw CheckpointError.unsupportedVersion(fileVersion) }
        let count = Int(readUInt32(8))
        let iteration = Int(readUInt32(12))

        let expectedFloats = count * floatsPerSplat
        let available = (data.count - 16) / 4
        guard available >= expectedFloats else {
            throw CheckpointError.truncated(expected: expectedFloats, got: available)
        }

        var floats = [Float](repeating: 0, count: expectedFloats)
        for i in 0..<expectedFloats {
            var bits: UInt32 = 0
            let offset = 16 + i * 4
            withUnsafeMutableBytes(of: &bits) { dst in
                data.copyBytes(to: dst, from: offset..<(offset + 4))
            }
            floats[i] = Float(bitPattern: UInt32(littleEndian: bits))
        }

        var cloud = SplatCloud()
        cloud.reserveCapacity(count)
        for i in 0..<count {
            let b = i * floatsPerSplat
            cloud.append(Splat(
                position: SIMD3<Float>(floats[b], floats[b + 1], floats[b + 2]),
                logScale: SIMD3<Float>(floats[b + 3], floats[b + 4], floats[b + 5]),
                rotation: SIMD4<Float>(floats[b + 6], floats[b + 7], floats[b + 8], floats[b + 9]),
                opacityLogit: floats[b + 10],
                color: SIMD3<Float>(floats[b + 11], floats[b + 12], floats[b + 13])))
        }
        return (cloud, iteration)
    }

    public static func write(_ cloud: SplatCloud, iteration: Int, to url: URL) throws {
        // Atomic write: a crash mid-save must not replace a good checkpoint
        // with a half-written one — which is exactly when a checkpoint matters.
        try encode(cloud, iteration: iteration).write(to: url, options: .atomic)
    }

    public static func read(from url: URL) throws -> (cloud: SplatCloud, iteration: Int) {
        try decode(try Data(contentsOf: url))
    }
}
