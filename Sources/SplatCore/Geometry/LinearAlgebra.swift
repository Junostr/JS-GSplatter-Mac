import Foundation

// MARK: - Small dense linear algebra for SfM
//
// Deliberately hand-written rather than calling LAPACK through Accelerate.
//
// Accelerate's LAPACK interface changed shape (ACCELERATE_NEW_LAPACK, the
// __CLPK_* types being deprecated) across exactly the SDK range this project
// straddles — 11.0 deployment, built variously against the 26.2 and 27.0 SDKs
// by three different build paths. Given how much trouble the toolchain has
// already caused, a self-contained solver removes an entire class of
// build-configuration risk, and every decomposition SfM needs here is on a
// matrix of fixed, tiny size (3x3, 4x4, 9x9) where LAPACK's asymptotic
// advantages are irrelevant. Revisit only if profiling says otherwise.
//
// Everything is Double: the normal equations formed below square the input
// condition number, and bundle adjustment accumulates over many observations.
// Float would be false economy here — this is not the per-pixel hot path.

public enum LinearAlgebra {

    /// Cyclic Jacobi eigenvalue decomposition for a symmetric n x n matrix.
    ///
    /// Returns eigenvalues ascending, with `vectors` column-major-by-index:
    /// eigenvector i occupies `vectors[i*n ..< (i+1)*n]`.
    ///
    /// Jacobi rather than a Householder/QR pipeline because it is short,
    /// unconditionally stable for symmetric input, and needs no pivoting
    /// bookkeeping — at these sizes its extra sweeps cost nothing.
    public static func symmetricEigen(_ input: [Double], n: Int, sweeps: Int = 64) -> (values: [Double], vectors: [Double]) {
        precondition(input.count == n * n, "matrix must be n*n")
        var a = input
        // vectors starts as identity; rotations accumulate into it.
        var v = [Double](repeating: 0, count: n * n)
        for i in 0..<n { v[i * n + i] = 1 }

        for _ in 0..<sweeps {
            // Convergence test: sum of squared off-diagonal entries.
            var off = 0.0
            for i in 0..<n {
                for j in (i + 1)..<n {
                    off += a[i * n + j] * a[i * n + j]
                }
            }
            if off < 1e-30 { break }

            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = a[p * n + q]
                    if abs(apq) < 1e-300 { continue }
                    let app = a[p * n + p]
                    let aqq = a[q * n + q]
                    // Rotation angle that zeroes the (p,q) entry.
                    let theta = (aqq - app) / (2 * apq)
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    for k in 0..<n {
                        let akp = a[k * n + p]
                        let akq = a[k * n + q]
                        a[k * n + p] = c * akp - s * akq
                        a[k * n + q] = s * akp + c * akq
                    }
                    for k in 0..<n {
                        let apk = a[p * n + k]
                        let aqk = a[q * n + k]
                        a[p * n + k] = c * apk - s * aqk
                        a[q * n + k] = s * apk + c * aqk
                    }
                    for k in 0..<n {
                        let vkp = v[k * n + p]
                        let vkq = v[k * n + q]
                        v[k * n + p] = c * vkp - s * vkq
                        v[k * n + q] = s * vkp + c * vkq
                    }
                }
            }
        }

        var values = (0..<n).map { a[$0 * n + $0] }
        // Sort ascending, permuting eigenvectors with them, and transpose the
        // accumulated matrix so each eigenvector is contiguous.
        let order = (0..<n).sorted { values[$0] < values[$1] }
        var sortedValues = [Double](repeating: 0, count: n)
        var sortedVectors = [Double](repeating: 0, count: n * n)
        for (newIndex, oldIndex) in order.enumerated() {
            sortedValues[newIndex] = values[oldIndex]
            for k in 0..<n {
                sortedVectors[newIndex * n + k] = v[k * n + oldIndex]
            }
        }
        values = sortedValues
        return (values, sortedVectors)
    }

    /// Unit vector x minimizing |Ax| — the right singular vector of A with the
    /// smallest singular value, found as the smallest eigenvector of AᵀA.
    ///
    /// Forming AᵀA squares the condition number, which is why callers must
    /// pre-condition their data (see `Normalization` in TwoViewGeometry).
    /// With that normalization in place the accuracy is ample for the
    /// homogeneous systems here, and it keeps this file to one decomposition.
    public static func smallestSingularVector(rows: [[Double]]) -> [Double] {
        guard let first = rows.first else { return [] }
        let n = first.count
        var ata = [Double](repeating: 0, count: n * n)
        for row in rows {
            for i in 0..<n {
                let ri = row[i]
                if ri == 0 { continue }
                for j in 0..<n {
                    ata[i * n + j] += ri * row[j]
                }
            }
        }
        let (_, vectors) = symmetricEigen(ata, n: n)
        return Array(vectors[0..<n])   // eigenvector of the smallest eigenvalue
    }

    /// SVD of a 3x3 matrix, returned as U, singular values (descending), Vᵀ.
    ///
    /// Built from the symmetric eigendecomposition of AᵀA: V comes from the
    /// eigenvectors, singular values from the square-rooted eigenvalues, and
    /// U from A·V normalized. A degenerate (near-zero) singular value leaves
    /// the corresponding U column undetermined by that route, so it is filled
    /// in as the cross product of the other two — which is exactly the case
    /// that matters for an essential matrix, whose third singular value is 0.
    public static func svd3x3(_ a: [Double]) -> (u: [Double], s: [Double], vt: [Double]) {
        precondition(a.count == 9)
        var ata = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var sum = 0.0
                for k in 0..<3 { sum += a[k * 3 + i] * a[k * 3 + j] }
                ata[i * 3 + j] = sum
            }
        }
        let (values, vectors) = symmetricEigen(ata, n: 3)
        // symmetricEigen gives ascending; SVD convention is descending.
        let order = [2, 1, 0]
        var v = [Double](repeating: 0, count: 9)      // columns = right singular vectors
        var s = [Double](repeating: 0, count: 3)
        for (col, src) in order.enumerated() {
            s[col] = max(0, values[src]).squareRoot()
            for r in 0..<3 { v[r * 3 + col] = vectors[src * 3 + r] }
        }

        // "Numerically zero" must be RELATIVE to the largest singular value,
        // and must account for the square root in this construction.
        //
        // Singular values here are sqrt(eigenvalues of AᵀA). An exactly-zero
        // singular value therefore arrives as sqrt(machine-epsilon-scale
        // eigenvalue) — sqrt(1e-18) is 1e-9, NOT 1e-18. An absolute 1e-12 test
        // silently fails to fire, and the column is then computed as
        // (A·v)/s = 0/1e-9, collapsing U's third column to the zero vector and
        // making U singular. That is fatal downstream: essential-matrix
        // recovery reads the translation from U's third column and builds
        // rotations from U, so it produced det(R) = 0 and a pose 180 degrees
        // from the truth while the essential matrix itself was perfect.
        // 1e-7 relative sits well above that ~1e-8 noise floor and far below
        // any meaningful singular value.
        let zeroTolerance = max(s[0], 1e-300) * 1e-7
        var u = [Double](repeating: 0, count: 9)
        for col in 0..<3 {
            if s[col] > zeroTolerance {
                for r in 0..<3 {
                    var sum = 0.0
                    for k in 0..<3 { sum += a[r * 3 + k] * v[k * 3 + col] }
                    u[r * 3 + col] = sum / s[col]
                }
            } else {
                // Report it as exactly zero so callers (and the rank-deficient
                // fill below) see a consistent value.
                s[col] = 0
            }
        }
        // Fill any undetermined U column via orthogonality.
        for col in 0..<3 where s[col] == 0 {
            let c1 = (col + 1) % 3, c2 = (col + 2) % 3
            let x = SIMD3<Double>(u[0 * 3 + c1], u[1 * 3 + c1], u[2 * 3 + c1])
            let y = SIMD3<Double>(u[0 * 3 + c2], u[1 * 3 + c2], u[2 * 3 + c2])
            let z = cross(x, y)
            let n = length(z)
            if n > 1e-12 {
                u[0 * 3 + col] = z.x / n
                u[1 * 3 + col] = z.y / n
                u[2 * 3 + col] = z.z / n
            }
        }

        var vt = [Double](repeating: 0, count: 9)
        for r in 0..<3 { for c in 0..<3 { vt[r * 3 + c] = v[c * 3 + r] } }
        return (u, s, vt)
    }

    // MARK: 3x3 helpers (row-major)

    public static func matMul3(_ a: [Double], _ b: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                var sum = 0.0
                for k in 0..<3 { sum += a[r * 3 + k] * b[k * 3 + c] }
                out[r * 3 + c] = sum
            }
        }
        return out
    }

    public static func transpose3(_ a: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: 9)
        for r in 0..<3 { for c in 0..<3 { out[r * 3 + c] = a[c * 3 + r] } }
        return out
    }

    public static func matVec3(_ a: [Double], _ v: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3<Double>(
            a[0] * v.x + a[1] * v.y + a[2] * v.z,
            a[3] * v.x + a[4] * v.y + a[5] * v.z,
            a[6] * v.x + a[7] * v.y + a[8] * v.z
        )
    }

    public static func determinant3(_ a: [Double]) -> Double {
        a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    public static func cross(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3<Double>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    public static func length(_ v: SIMD3<Double>) -> Double {
        (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    }

    /// Nearest rotation matrix (orthogonal, det +1) to `a`, via SVD.
    /// Bundle adjustment updates rotations incrementally, so they drift off
    /// SO(3); this projects them back.
    public static func nearestRotation(_ a: [Double]) -> [Double] {
        let (u, _, vt) = svd3x3(a)
        var r = matMul3(u, vt)
        if determinant3(r) < 0 {
            // Flip the sign of the column tied to the smallest singular value.
            var uFlipped = u
            for row in 0..<3 { uFlipped[row * 3 + 2] = -uFlipped[row * 3 + 2] }
            r = matMul3(uFlipped, vt)
        }
        return r
    }

    /// Solve a small dense symmetric positive-definite system via Cholesky
    /// with a diagonal fallback. Used for the bundle-adjustment normal
    /// equations, which are SPD by construction once damped.
    /// Returns nil if the matrix is not positive definite even after damping.
    public static func solveSPD(_ a: [Double], _ b: [Double], n: Int) -> [Double]? {
        var l = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                var sum = a[i * n + j]
                for k in 0..<j { sum -= l[i * n + k] * l[j * n + k] }
                if i == j {
                    if sum <= 0 { return nil }
                    l[i * n + i] = sum.squareRoot()
                } else {
                    l[i * n + j] = sum / l[j * n + j]
                }
            }
        }
        // Forward then back substitution.
        var y = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = b[i]
            for k in 0..<i { sum -= l[i * n + k] * y[k] }
            y[i] = sum / l[i * n + i]
        }
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = y[i]
            for k in (i + 1)..<n { sum -= l[k * n + i] * x[k] }
            x[i] = sum / l[i * n + i]
        }
        return x
    }
}
