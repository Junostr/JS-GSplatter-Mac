import Foundation

// MARK: - Stage 3: SIFT-like gradient-histogram descriptor
//
// BRIEF compares raw intensities at pairs of points. That is fast and compact,
// but it degrades sharply once a surface's appearance changes — which is
// exactly what happens at the wide viewpoint changes late in an orbit, and is
// where registration was observed to fade (PnP inlier ratios decaying to
// 14/124 while correspondences were still plentiful).
//
// A gradient-orientation histogram is far more robust because it discards
// absolute intensity entirely and records only the DISTRIBUTION of edge
// directions in each part of the patch. That survives lighting change,
// moderate blur, and small geometric warps that would scramble a pairwise
// intensity test.
//
// Layout follows Lowe: a 16x16 sample window around the keypoint, rotated to
// the keypoint's orientation, split into 4x4 spatial cells, each holding an
// 8-bin orientation histogram — 4*4*8 = 128 dimensions. Samples are spread
// with trilinear interpolation across neighbouring cells and bins so a sample
// drifting one pixel cannot abruptly move all its weight, which is what makes
// the descriptor stable rather than merely descriptive.

public enum SIFTDescriptor {

    public static let cells = 4          // 4x4 spatial grid
    public static let bins = 8           // orientation bins per cell
    public static let dimensions = cells * cells * bins   // 128
    /// Samples per cell edge; the full window is 16x16 at the keypoint's level.
    public static let samplesPerCell = 4

    /// Compute the 128-byte descriptor for one keypoint on one pyramid level.
    ///
    /// `luma` is the level image; the keypoint's coordinates are in that
    /// level's pixels. Working on the level rather than the full-resolution
    /// image is what gives scale invariance — the same physical patch occupies
    /// a similar number of pixels whichever level detected it.
    public static func describe(luma: [Float], width: Int, height: Int, keypoint: Keypoint) -> [UInt8] {
        var histogram = [Float](repeating: 0, count: dimensions)
        let windowSamples = cells * samplesPerCell        // 16
        let half = Float(windowSamples) / 2               // 8
        let cosA = cos(keypoint.angle), sinA = sin(keypoint.angle)
        let cx = keypoint.x, cy = keypoint.y
        // Gaussian falloff over the window: samples near the edge are least
        // reliable under small misalignment, so they contribute least.
        let sigma = Float(windowSamples) * 0.5
        let expDenominator = 2 * sigma * sigma

        for sy in 0..<windowSamples {
            for sx in 0..<windowSamples {
                // Sample position in window coordinates, centred on the keypoint.
                let wx = Float(sx) - half + 0.5
                let wy = Float(sy) - half + 0.5
                // Rotate into image space so the descriptor is measured in the
                // keypoint's own frame — this is what makes it rotation
                // invariant.
                let px = cx + cosA * wx - sinA * wy
                let py = cy + sinA * wx + cosA * wy
                let ix = Int(px.rounded()), iy = Int(py.rounded())
                guard ix >= 1, iy >= 1, ix + 1 < width, iy + 1 < height else { continue }

                // Central differences on the level image.
                let gx = luma[iy * width + ix + 1] - luma[iy * width + ix - 1]
                let gy = luma[(iy + 1) * width + ix] - luma[(iy - 1) * width + ix]
                let magnitude = (gx * gx + gy * gy).squareRoot()
                guard magnitude > 0 else { continue }
                // Gradient direction relative to the keypoint orientation.
                var theta = atan2(gy, gx) - keypoint.angle
                while theta < 0 { theta += 2 * .pi }
                while theta >= 2 * .pi { theta -= 2 * .pi }

                let weight = expf(-(wx * wx + wy * wy) / expDenominator) * magnitude

                // Continuous coordinates in (cell_x, cell_y, bin) space, then
                // trilinear spread. Without the interpolation a sample that
                // shifts by one pixel jumps its whole weight into a different
                // cell, which is precisely the instability the descriptor is
                // supposed to avoid.
                let fx = (Float(sx) + 0.5) / Float(samplesPerCell) - 0.5
                let fy = (Float(sy) + 0.5) / Float(samplesPerCell) - 0.5
                let fo = theta / (2 * .pi) * Float(bins)

                let x0 = Int(floor(fx)), y0 = Int(floor(fy)), o0 = Int(floor(fo))
                let dx = fx - Float(x0), dy = fy - Float(y0), doo = fo - Float(o0)

                for cxOffset in 0...1 {
                    let cellX = x0 + cxOffset
                    guard cellX >= 0, cellX < cells else { continue }
                    let wxWeight = cxOffset == 0 ? (1 - dx) : dx
                    for cyOffset in 0...1 {
                        let cellY = y0 + cyOffset
                        guard cellY >= 0, cellY < cells else { continue }
                        let wyWeight = cyOffset == 0 ? (1 - dy) : dy
                        for oOffset in 0...1 {
                            // Orientation wraps — bin 7 and bin 0 are adjacent.
                            let bin = (o0 + oOffset) % bins
                            let woWeight = oOffset == 0 ? (1 - doo) : doo
                            let index = (cellY * cells + cellX) * bins + bin
                            histogram[index] += weight * wxWeight * wyWeight * woWeight
                        }
                    }
                }
            }
        }

        // Normalize -> clip -> renormalize.
        //
        // The clip is the reason this tolerates lighting change: a specular
        // highlight or a hard shadow edge produces one enormous gradient that
        // would otherwise dominate the whole vector. Capping any single
        // component at 0.2 of the norm bounds how much one sample can matter,
        // then renormalizing restores unit length.
        func normalize(_ v: inout [Float]) {
            var sum: Float = 0
            for value in v { sum += value * value }
            let norm = sum.squareRoot()
            guard norm > 1e-12 else { return }
            for i in v.indices { v[i] /= norm }
        }
        normalize(&histogram)
        for i in histogram.indices { histogram[i] = min(histogram[i], 0.2) }
        normalize(&histogram)

        // Quantize to bytes. 512x matches Lowe's convention and keeps typical
        // components well inside the range after the 0.2 clip.
        var bytes = [UInt8](repeating: 0, count: dimensions)
        for i in 0..<dimensions {
            bytes[i] = UInt8(max(0, min(255, (histogram[i] * 512).rounded())))
        }
        return bytes
    }

    /// Squared L2 distance between two descriptors.
    ///
    /// Squared rather than Euclidean: the square root is monotonic, so it
    /// changes no ordering and no ratio-test outcome (the ratio test compares
    /// distances, and sqrt(a)/sqrt(b) < t is equivalent to a/b < t²) while
    /// keeping the whole computation in integers.
    @inline(__always)
    public static func squaredDistance(_ a: ArraySlice<UInt8>, _ b: ArraySlice<UInt8>) -> Int {
        var total = 0
        var i = a.startIndex, j = b.startIndex
        while i < a.endIndex {
            let d = Int(a[i]) - Int(b[j])
            total += d * d
            i += 1
            j += 1
        }
        return total
    }
}
