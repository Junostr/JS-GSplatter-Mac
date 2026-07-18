import Accelerate
import CoreVideo
import Foundation

/// Shared BGRA→luma conversion. Kept in one place because stage 2's analyzers
/// and stage 3's extractors must see byte-identical luma: a keypoint detected
/// at a slightly different intensity would shift its descriptor, and the two
/// stages' results are consumed together downstream.
public enum LumaBuffer {

    /// Rec. 709 luma, normalized to 0…1. Same weights and normalization as
    /// the `luma_from_bgra` kernel and CPUFrameAnalyzer.
    public static func make(from pixelBuffer: CVPixelBuffer) throws -> (luma: [Float], width: Int, height: Int) {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw FeatureError.unsupportedPixelFormat(format)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw FeatureError.kernelFailure("pixel buffer has no base address")
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        var luma = [Float](repeating: 0, count: width * height)
        luma.withUnsafeMutableBufferPointer { out in
            for y in 0..<height {
                let row = bytes + y * bytesPerRow
                let dst = y * width
                for x in 0..<width {
                    let b = Float(row[x * 4 + 0])
                    let g = Float(row[x * 4 + 1])
                    let r = Float(row[x * 4 + 2])
                    out[dst + x] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                }
            }
        }
        return (luma, width, height)
    }
}

/// CPU fallback feature extractor. This is the reference implementation: the
/// Metal tier's response map is verified against this one in selftest.
public final class CPUFeatureExtractor: FeatureExtractor {

    public var descriptionForLog: String { "CPU feature extractor (Harris, Accelerate)" }

    /// Harris free parameter. 0.04 is the canonical value from the original
    /// paper and is what every reference implementation uses; changing it
    /// shifts the corner/edge boundary, so it is fixed rather than exposed.
    static let harrisK: Float = 0.04
    /// Structure-tensor integration window radius. 2 (a 5x5 window) is large
    /// enough to be stable under noise without merging nearby corners.
    static let windowRadius = 2

    public init() {}

    public func extract(index: Int, pixelBuffer: CVPixelBuffer, options: FeatureOptions) throws -> FeatureSet {
        let (luma, width, height) = try LumaBuffer.make(from: pixelBuffer)
        let response = CPUFeatureExtractor.harrisResponse(luma: luma, width: width, height: height)
        return FeatureMath.assemble(
            frameIndex: index, response: response, luma: luma,
            width: width, height: height, options: options
        )
    }

    /// Harris corner response over the whole frame.
    ///
    /// R = det(M) − k·trace(M)²  where M is the structure tensor
    /// [[ΣIx², ΣIxIy], [ΣIxIy, ΣIy²]] summed over the integration window.
    /// Borders (where either the Sobel or the window would read outside the
    /// image) stay 0, matching the GPU kernel's bounds behavior exactly.
    public static func harrisResponse(luma: [Float], width: Int, height: Int) -> [Float] {
        var response = [Float](repeating: 0, count: width * height)
        guard width > 2, height > 2 else { return response }

        // Sobel gradients. Separate arrays rather than fusing into the tensor
        // loop: each gradient is read by up to (2w+1)² window positions, so
        // computing them once is a large win over recomputing per window.
        var ixx = [Float](repeating: 0, count: width * height)
        var iyy = [Float](repeating: 0, count: width * height)
        var ixy = [Float](repeating: 0, count: width * height)

        luma.withUnsafeBufferPointer { src in
            ixx.withUnsafeMutableBufferPointer { dxx in
                iyy.withUnsafeMutableBufferPointer { dyy in
                    ixy.withUnsafeMutableBufferPointer { dxy in
                        for y in 1..<(height - 1) {
                            let row = y * width
                            let up = row - width
                            let down = row + width
                            for x in 1..<(width - 1) {
                                let gx =
                                    (src[up + x + 1] + 2 * src[row + x + 1] + src[down + x + 1]) -
                                    (src[up + x - 1] + 2 * src[row + x - 1] + src[down + x - 1])
                                let gy =
                                    (src[down + x - 1] + 2 * src[down + x] + src[down + x + 1]) -
                                    (src[up + x - 1] + 2 * src[up + x] + src[up + x + 1])
                                let i = row + x
                                dxx[i] = gx * gx
                                dyy[i] = gy * gy
                                dxy[i] = gx * gy
                            }
                        }
                    }
                }
            }
        }

        let w = windowRadius
        let margin = w + 1   // +1 because the Sobel ring is already invalid
        guard width > 2 * margin, height > 2 * margin else { return response }

        ixx.withUnsafeBufferPointer { axx in
            iyy.withUnsafeBufferPointer { ayy in
                ixy.withUnsafeBufferPointer { axy in
                    response.withUnsafeMutableBufferPointer { out in
                        for y in margin..<(height - margin) {
                            for x in margin..<(width - margin) {
                                var sxx: Float = 0, syy: Float = 0, sxy: Float = 0
                                for dy in -w...w {
                                    let row = (y + dy) * width
                                    for dx in -w...w {
                                        let i = row + x + dx
                                        sxx += axx[i]
                                        syy += ayy[i]
                                        sxy += axy[i]
                                    }
                                }
                                let det = sxx * syy - sxy * sxy
                                let trace = sxx + syy
                                out[y * width + x] = det - harrisK * trace * trace
                            }
                        }
                    }
                }
            }
        }
        return response
    }
}
