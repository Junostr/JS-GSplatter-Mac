import Accelerate
import CoreVideo
import Foundation

/// CPU fallback analyzer (Accelerate/vDSP) for machines where Metal compute
/// is not viable. Mirrors the Metal kernels operation-for-operation — same
/// Rec. 709 weights, same 5-point Laplacian with zeroed borders, same 8×8
/// mean-luma signature — so scores agree across tiers (asserted in selftest).
public final class CPUFrameAnalyzer: FrameAnalyzer {

    public var descriptionForLog: String { "CPU analyzer (Accelerate/vDSP)" }

    // Reused across frames of the same size, like the Metal scratch buffers.
    private var luma: [Float] = []
    private var laplacian: [Float] = []
    private var cachedSize: (width: Int, height: Int) = (0, 0)

    public init() {}

    public func analyze(index: Int, timestamp: Double?, pixelBuffer: CVPixelBuffer) throws -> FrameScore {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw FrameAnalyzerError.unsupportedPixelFormat(format)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        if (width, height) != cachedSize {
            luma = [Float](repeating: 0, count: width * height)
            laplacian = [Float](repeating: 0, count: width * height)
            cachedSize = (width, height)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw FrameAnalyzerError.kernelFailure("pixel buffer has no base address")
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // BGRA → luma, normalized to 0…1 like the texture read on the GPU.
        luma.withUnsafeMutableBufferPointer { lumaPtr in
            for y in 0..<height {
                let row = bytes + y * bytesPerRow
                let out = y * width
                for x in 0..<width {
                    let b = Float(row[x * 4 + 0])
                    let g = Float(row[x * 4 + 1])
                    let r = Float(row[x * 4 + 2])
                    lumaPtr[out + x] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                }
            }
        }

        // 5-point Laplacian, interior only; borders stay 0 to match the GPU
        // path exactly. Hand-rolled instead of vDSP_f3x3 because that API's
        // border behavior is unspecified and we need bit-for-bit-comparable
        // statistics across tiers.
        //
        // The dimension guard is load-bearing, not defensive noise: with
        // width or height == 1, `1..<(dim - 1)` is `1..<0`, and Swift TRAPS
        // constructing a reversed Range — the whole process aborts. The GPU
        // kernel's bounds test simply selects no interior pixels and yields
        // variance 0, so without this guard a 1-pixel-tall frame would crash
        // the CPU tier while the Metal tier returned a score, breaking the
        // tiers-are-interchangeable contract. (dim == 2 is already safe: the
        // range is empty but well-formed.)
        if width > 2 && height > 2 {
            luma.withUnsafeBufferPointer { src in
                laplacian.withUnsafeMutableBufferPointer { dst in
                    for y in 1..<(height - 1) {
                        let row = y * width
                        for x in 1..<(width - 1) {
                            let i = row + x
                            dst[i] = 4 * src[i] - src[i - 1] - src[i + 1] - src[i - width] - src[i + width]
                        }
                    }
                }
            }
        }

        // Variance via vDSP moments: E[x²] − E[x]².
        var mean: Float = 0
        var meanSquare: Float = 0
        vDSP_meanv(laplacian, 1, &mean, vDSP_Length(laplacian.count))
        vDSP_measqv(laplacian, 1, &meanSquare, vDSP_Length(laplacian.count))
        let variance = max(0, Double(meanSquare) - Double(mean) * Double(mean))

        // 8×8 signature with the same integer cell boundaries as the kernel.
        var signature = [Float](repeating: 0, count: 64)
        for cy in 0..<8 {
            let y0 = cy * height / 8, y1 = (cy + 1) * height / 8
            for cx in 0..<8 {
                let x0 = cx * width / 8, x1 = (cx + 1) * width / 8
                let count = (x1 - x0) * (y1 - y0)
                guard count > 0 else { continue }
                var sum: Float = 0
                for y in y0..<y1 {
                    var rowSum: Float = 0
                    luma.withUnsafeBufferPointer { ptr in
                        vDSP_sve(ptr.baseAddress! + y * width + x0, 1, &rowSum, vDSP_Length(x1 - x0))
                    }
                    sum += rowSum
                }
                signature[cy * 8 + cx] = sum / Float(count)
            }
        }

        return FrameScore(index: index, timestamp: timestamp, blurScore: variance, signature: signature)
    }
}
