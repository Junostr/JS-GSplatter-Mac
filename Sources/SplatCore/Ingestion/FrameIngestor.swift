import Foundation
import AVFoundation
import CoreVideo
import ImageIO
import VideoToolbox

// MARK: - Stage 1: frame extraction
//
// Streams frames one at a time through a handler instead of returning an
// array: a 4K video at 30 fps is ~24 MB per BGRA frame, so materializing a
// whole clip would blow past even generous RAM budgets — and the 2 GB-VRAM
// baseline machines are exactly the ones with 8–16 GB RAM. Downstream stages
// (blur scoring, feature extraction) are per-frame anyway.

public struct IngestionOptions {
    /// Evenly subsample a video down to at most this many frames.
    /// nil = every decoded frame. Ignored for photo folders (photos are
    /// deliberate captures; dropping them is stage 2's judgment call).
    public var maxFrames: Int?
    /// Downscale photos so the longest edge is at most this. nil = original
    /// size. Video frames are currently delivered at native resolution —
    /// scaling them belongs on the GPU next to blur scoring (stage 2), not
    /// in a CPU copy here.
    public var maxDimension: Int?

    public init(maxFrames: Int? = nil, maxDimension: Int? = nil) {
        self.maxFrames = maxFrames
        self.maxDimension = maxDimension
    }
}

public struct IngestedFrame {
    /// Position in the delivered sequence (0-based, contiguous).
    public let index: Int
    /// Presentation time in seconds (video sources only).
    public let timestamp: Double?
    /// Originating file (photo sources only).
    public let sourceURL: URL?
    /// BGRA pixel data. Only valid during the handler call for video
    /// sources — the decoder recycles buffers from a pool. Copy (or finish
    /// consuming) before returning if the data must outlive the call.
    public let pixelBuffer: CVPixelBuffer

    public var width: Int { CVPixelBufferGetWidth(pixelBuffer) }
    public var height: Int { CVPixelBufferGetHeight(pixelBuffer) }

    /// Snapshot as CGImage (copies). Handy for writing frames to disk and
    /// for the viewer later; not on the hot training path.
    public func makeCGImage() throws -> CGImage {
        var image: CGImage?
        let status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &image)
        guard status == noErr, let image = image else {
            throw IngestionError.decodeFailed("VTCreateCGImageFromCVPixelBuffer status \(status)")
        }
        return image
    }
}

public struct IngestionSummary {
    public let deliveredFrames: Int
    public let decodedFrames: Int
    public let width: Int
    public let height: Int
    /// Video only.
    public let duration: Double?
    public let nominalFrameRate: Double?
}

public enum FrameIngestor {

    /// Stream all frames from a source. The handler returns `false` to stop
    /// early (still yields a valid summary — the UI's cancel button hooks in
    /// here).
    @discardableResult
    public static func ingest(
        _ source: IngestionSource,
        options: IngestionOptions = IngestionOptions(),
        handler: (IngestedFrame) throws -> Bool
    ) throws -> IngestionSummary {
        switch source {
        case .photoFolder(_, let imageURLs):
            return try ingestPhotos(imageURLs, options: options, handler: handler)
        case .video(let url):
            return try ingestVideo(url, options: options, handler: handler)
        }
    }

    // MARK: Photos (ImageIO)

    static func ingestPhotos(
        _ urls: [URL],
        options: IngestionOptions,
        handler: (IngestedFrame) throws -> Bool
    ) throws -> IngestionSummary {
        var delivered = 0
        var width = 0, height = 0

        for url in urls {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                throw IngestionError.decodeFailed("cannot open \(url.lastPathComponent)")
            }
            // Thumbnail API instead of CGImageSourceCreateImageAtIndex:
            // it applies EXIF orientation (kCGImageSourceCreateThumbnailWithTransform)
            // and downsamples during decode — an oriented, resized image in one
            // pass without ever materializing the full-size bitmap. Orientation
            // matters: SfM (stage 3) must see pixels the way the camera saw them.
            var thumbOptions: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
            ]
            if let maxDim = options.maxDimension {
                thumbOptions[kCGImageSourceThumbnailMaxPixelSize] = maxDim
            } else {
                // "Thumbnail" at full size: cap at the source's own long edge.
                thumbOptions[kCGImageSourceThumbnailMaxPixelSize] = 1 << 16
            }
            guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions as CFDictionary) else {
                throw IngestionError.decodeFailed("cannot decode \(url.lastPathComponent)")
            }

            let buffer = try makePixelBuffer(from: cgImage)
            if delivered == 0 {
                width = cgImage.width
                height = cgImage.height
            }
            let frame = IngestedFrame(index: delivered, timestamp: nil, sourceURL: url, pixelBuffer: buffer)
            delivered += 1
            if try !handler(frame) { break }
        }

        return IngestionSummary(
            deliveredFrames: delivered, decodedFrames: delivered,
            width: width, height: height, duration: nil, nominalFrameRate: nil
        )
    }

    /// Render a CGImage into a fresh BGRA CVPixelBuffer so both source types
    /// hand downstream stages the same pixel format.
    static func makePixelBuffer(from image: CGImage) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            // IOSurface backing lets Metal wrap the buffer zero-copy later.
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, image.width, image.height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer
        )
        guard status == kCVReturnSuccess, let buffer = buffer else {
            throw IngestionError.decodeFailed("CVPixelBufferCreate status \(status)")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: image.width, height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            // BGRA little-endian == premultipliedFirst + 32Little host order.
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw IngestionError.decodeFailed("CGContext creation failed")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return buffer
    }

    // MARK: Video (AVFoundation → VideoToolbox)

    static func ingestVideo(
        _ url: URL,
        options: IngestionOptions,
        handler: (IngestedFrame) throws -> Bool
    ) throws -> IngestionSummary {
        // AVAsset(url:) + synchronous tracks API: the async load(.tracks)
        // replacement is macOS 12+, so the deprecated-but-present calls are
        // the correct choice for the 11.0 deployment target.
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw IngestionError.videoUnreadable(url, underlying: "no video track")
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw IngestionError.videoUnreadable(url, underlying: error.localizedDescription)
        }

        // BGRA output to match the photo path. The decode itself goes through
        // VideoToolbox and uses the hardware decoder wherever one exists —
        // including Intel QuickSync on the 2013–2015 baseline machines.
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        // No defensive copies; we promise handlers the buffer is call-scoped.
        output.alwaysCopiesSampleData = false
        reader.add(output)

        let duration = asset.duration.seconds
        let fps = Double(track.nominalFrameRate)

        // Even subsampling: estimate the total from duration × fps and take
        // every stride-th decoded frame. Decode cost is unavoidable (H.264/HEVC
        // must decode sequentially anyway); we only skip delivery.
        var stride = 1
        if let maxFrames = options.maxFrames, maxFrames > 0, duration.isFinite, fps > 0 {
            let estimated = Int(duration * fps)
            if estimated > maxFrames {
                stride = (estimated + maxFrames - 1) / maxFrames
            }
        }

        guard reader.startReading() else {
            throw IngestionError.videoUnreadable(url, underlying: reader.error?.localizedDescription ?? "startReading failed")
        }

        var decoded = 0
        var delivered = 0
        var width = 0, height = 0
        var stopped = false

        while let sample = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            defer { decoded += 1 }
            guard decoded % stride == 0 else { continue }
            if let maxFrames = options.maxFrames, delivered >= maxFrames { break }

            if delivered == 0 {
                width = CVPixelBufferGetWidth(pixelBuffer)
                height = CVPixelBufferGetHeight(pixelBuffer)
            }
            let frame = IngestedFrame(
                index: delivered,
                timestamp: CMSampleBufferGetPresentationTimeStamp(sample).seconds,
                sourceURL: nil,
                pixelBuffer: pixelBuffer
            )
            delivered += 1
            if try !handler(frame) {
                stopped = true
                break
            }
        }

        if !stopped {
            if reader.status == .failed {
                throw IngestionError.videoUnreadable(url, underlying: reader.error?.localizedDescription ?? "reader failed")
            }
        } else {
            reader.cancelReading()
        }

        return IngestionSummary(
            deliveredFrames: delivered, decodedFrames: decoded,
            width: width, height: height,
            duration: duration.isFinite ? duration : nil,
            nominalFrameRate: fps > 0 ? fps : nil
        )
    }
}
