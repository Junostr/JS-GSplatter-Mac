import Foundation
import AVFoundation

// MARK: - Stage 1: ingestion source detection
//
// The engine takes a URL — a folder of photos or a video file — and the
// future app target's drag-and-drop handler will call exactly this entry
// point. Everything here is macOS 11-safe (ImageIO and AVFoundation APIs
// that exist since 10.x); no UI dependencies.

public enum IngestionError: Error, CustomStringConvertible {
    case notFound(URL)
    case emptyFolder(URL)
    case unsupportedFile(URL, hint: String)
    case videoUnreadable(URL, underlying: String)
    case decodeFailed(String)

    public var description: String {
        switch self {
        case .notFound(let url):
            return "No file or folder at \(url.path)"
        case .emptyFolder(let url):
            return "Folder contains no supported images: \(url.path)"
        case .unsupportedFile(let url, let hint):
            return "Unsupported input \(url.lastPathComponent): \(hint)"
        case .videoUnreadable(let url, let underlying):
            return "Could not read video \(url.lastPathComponent): \(underlying)"
        case .decodeFailed(let detail):
            return "Frame decode failed: \(detail)"
        }
    }
}

public enum IngestionSource {
    /// A folder of still photos; URLs are pre-scanned and name-sorted.
    case photoFolder(URL, imageURLs: [URL])
    /// A single video file.
    case video(URL)

    public var frameCountEstimateLabel: String {
        switch self {
        case .photoFolder(_, let urls): return "\(urls.count) photos"
        case .video(let url): return "video \(url.lastPathComponent)"
        }
    }

    /// Formats ImageIO decodes on macOS 11 across all our targets. WebP is
    /// deliberately absent: ImageIO gained it in 11.0 but flakily on some
    /// point releases; revisit if it ever matters for capture workflows.
    static let imageExtensions: Set<String> = [
        "jpg", "jpeg", "png", "heic", "heif", "tiff", "tif", "bmp", "dng",
    ]

    static let videoExtensions: Set<String> = [
        "mov", "mp4", "m4v", "avi", "mts", "m2ts",
    ]

    /// Detect what kind of source a dropped/passed URL is.
    public static func detect(at url: URL) throws -> IngestionSource {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw IngestionError.notFound(url)
        }

        if isDirectory.boolValue {
            let urls = try scanPhotoFolder(url)
            guard !urls.isEmpty else { throw IngestionError.emptyFolder(url) }
            return .photoFolder(url, imageURLs: urls)
        }

        let ext = url.pathExtension.lowercased()
        if videoExtensions.contains(ext) {
            return .video(url)
        }
        if imageExtensions.contains(ext) {
            throw IngestionError.unsupportedFile(url, hint: "a single image is not enough — pass the containing folder")
        }
        throw IngestionError.unsupportedFile(url, hint: "expected a photo folder or a video (\(videoExtensions.sorted().joined(separator: "/")))")
    }

    /// Shallow scan (no recursion — capture sessions are flat folders),
    /// name-sorted with numeric awareness so frame_2 < frame_10.
    static func scanPhotoFolder(_ folder: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        return contents
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { a, b in
                a.lastPathComponent.localizedStandardCompare(b.lastPathComponent) == .orderedAscending
            }
    }
}
