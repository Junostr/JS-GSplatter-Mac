import Foundation

// MARK: - Stage 5: the Gaussian splat model
//
// A scene is a cloud of 3D Gaussians, each with position, anisotropic scale,
// orientation, opacity and colour. Rendering projects each Gaussian to a 2D
// screen-space ellipse and alpha-composites them front to back.
//
// Storage is STRUCT-OF-ARRAYS rather than an array of structs. Every stage of
// training touches one attribute across all splats at a time (project all
// positions, then blend all colours, then apply gradients to all scales), so
// SoA keeps each of those passes reading contiguous memory. It is also the
// layout a Metal buffer wants: one `MTLBuffer` per attribute, bound directly,
// with no interleaving or padding rules to get wrong across architectures.

/// Parameters of one Gaussian, used for construction and inspection.
/// The training path works on the SoA arrays, not on this.
public struct Splat: Equatable {
    public var position: SIMD3<Float>
    /// Per-axis extent BEFORE rotation, stored as log(scale).
    ///
    /// Log space because scale must stay strictly positive and spans orders of
    /// magnitude between a floor plane and a leaf edge. Optimising log(s) makes
    /// the parameter unbounded (no projection needed after each step) and makes
    /// gradient steps multiplicative, so a fixed learning rate behaves the same
    /// for a huge splat as for a tiny one.
    public var logScale: SIMD3<Float>
    /// Orientation as a quaternion (x, y, z, w), normalised on use.
    public var rotation: SIMD4<Float>
    /// Opacity BEFORE the sigmoid, i.e. logit space, for the same reason scale
    /// is logged: it keeps the optimised parameter unconstrained while the
    /// value it represents stays in (0, 1).
    public var opacityLogit: Float
    /// View-independent colour (spherical-harmonic DC term). Higher SH bands
    /// are a later addition; the layout leaves room for them.
    public var color: SIMD3<Float>

    public init(position: SIMD3<Float>, logScale: SIMD3<Float>, rotation: SIMD4<Float>,
                opacityLogit: Float, color: SIMD3<Float>) {
        self.position = position
        self.logScale = logScale
        self.rotation = rotation
        self.opacityLogit = opacityLogit
        self.color = color
    }

    public var opacity: Float { 1 / (1 + expf(-opacityLogit)) }
    public var scale: SIMD3<Float> {
        SIMD3<Float>(expf(logScale.x), expf(logScale.y), expf(logScale.z))
    }
}

/// The scene, in struct-of-arrays form.
public struct SplatCloud {
    public var positions: [SIMD3<Float>] = []
    public var logScales: [SIMD3<Float>] = []
    public var rotations: [SIMD4<Float>] = []
    public var opacityLogits: [Float] = []
    public var colors: [SIMD3<Float>] = []

    public var count: Int { positions.count }

    public init() {}

    public init(splats: [Splat]) {
        for splat in splats { append(splat) }
    }

    public mutating func append(_ splat: Splat) {
        positions.append(splat.position)
        logScales.append(splat.logScale)
        rotations.append(splat.rotation)
        opacityLogits.append(splat.opacityLogit)
        colors.append(splat.color)
    }

    public subscript(index: Int) -> Splat {
        get {
            Splat(position: positions[index], logScale: logScales[index],
                  rotation: rotations[index], opacityLogit: opacityLogits[index],
                  color: colors[index])
        }
        set {
            positions[index] = newValue.position
            logScales[index] = newValue.logScale
            rotations[index] = newValue.rotation
            opacityLogits[index] = newValue.opacityLogit
            colors[index] = newValue.color
        }
    }

    public mutating func remove(atOffsets doomed: Set<Int>) {
        guard !doomed.isEmpty else { return }
        var kept = SplatCloud()
        kept.reserveCapacity(count - doomed.count)
        for i in 0..<count where !doomed.contains(i) { kept.append(self[i]) }
        self = kept
    }

    public mutating func reserveCapacity(_ n: Int) {
        positions.reserveCapacity(n); logScales.reserveCapacity(n)
        rotations.reserveCapacity(n); opacityLogits.reserveCapacity(n)
        colors.reserveCapacity(n)
    }

    /// Initialise from an SfM reconstruction — the handoff from stage 3.
    ///
    /// Each sparse point becomes one Gaussian. Initial scale is set from the
    /// distance to nearby points so density follows the structure: a splat in a
    /// dense, well-observed region starts small, one in a sparse region starts
    /// large enough to cover the gap. Starting everything at a fixed size makes
    /// training spend its early iterations undoing that choice.
    ///
    /// Opacity starts low (0.1). Splats that earn their place get pushed up by
    /// the gradient; ones that do not are pruned. Starting near-opaque instead
    /// makes early renders a soup that the optimiser has to dig out of.
    public static func fromReconstruction(
        _ reconstruction: Reconstruction,
        neighbourCount: Int = 3,
        initialOpacity: Float = 0.1
    ) -> SplatCloud {
        var cloud = SplatCloud()
        let points = reconstruction.points.map { $0.position }
        guard !points.isEmpty else { return cloud }
        cloud.reserveCapacity(points.count)

        for (i, p) in points.enumerated() {
            // Mean distance to the nearest few neighbours, as the initial
            // radius. O(n²) but this runs once, and n is a sparse cloud.
            var distances: [Double] = []
            distances.reserveCapacity(points.count - 1)
            for (j, q) in points.enumerated() where j != i {
                distances.append(LinearAlgebra.length(p - q))
            }
            distances.sort()
            let k = min(neighbourCount, distances.count)
            let radius: Double = k > 0
                ? distances[0..<k].reduce(0, +) / Double(k)
                : 0.01
            // Guard against coincident points collapsing the scale to zero,
            // which would make log(scale) negative infinity.
            let safeRadius = Float(max(radius, 1e-4))

            cloud.append(Splat(
                position: SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z)),
                logScale: SIMD3<Float>(repeating: logf(safeRadius)),
                rotation: SIMD4<Float>(0, 0, 0, 1),      // identity quaternion
                opacityLogit: logf(initialOpacity / (1 - initialOpacity)),
                color: SIMD3<Float>(repeating: 0.5)      // mid grey; colour is learned
            ))
        }
        return cloud
    }
}

// MARK: - Gaussian geometry

public enum SplatMath {

    /// 3D covariance from scale and rotation: Σ = R S Sᵀ Rᵀ.
    ///
    /// Built from the factors rather than optimised directly, because a raw
    /// 3x3 covariance would have to be kept symmetric AND positive
    /// semi-definite after every gradient step. Scale and a quaternion are
    /// unconstrained (given normalisation), so the product is valid by
    /// construction.
    ///
    /// Returned as the 6 upper-triangular terms (xx, xy, xz, yy, yz, zz),
    /// since a covariance is symmetric and storing 9 wastes a third of the
    /// bandwidth on every read.
    public static func covariance3D(logScale: SIMD3<Float>, rotation: SIMD4<Float>) -> (Float, Float, Float, Float, Float, Float) {
        let r = normalizeQuaternion(rotation)
        let m = rotationMatrix(r)
        let s = SIMD3<Float>(expf(logScale.x), expf(logScale.y), expf(logScale.z))

        // M = R * S, then Σ = M Mᵀ.
        var mScaled = [Float](repeating: 0, count: 9)
        for row in 0..<3 {
            mScaled[row * 3 + 0] = m[row * 3 + 0] * s.x
            mScaled[row * 3 + 1] = m[row * 3 + 1] * s.y
            mScaled[row * 3 + 2] = m[row * 3 + 2] * s.z
        }
        func dot(_ a: Int, _ b: Int) -> Float {
            mScaled[a * 3 + 0] * mScaled[b * 3 + 0]
            + mScaled[a * 3 + 1] * mScaled[b * 3 + 1]
            + mScaled[a * 3 + 2] * mScaled[b * 3 + 2]
        }
        return (dot(0, 0), dot(0, 1), dot(0, 2), dot(1, 1), dot(1, 2), dot(2, 2))
    }

    public static func normalizeQuaternion(_ q: SIMD4<Float>) -> SIMD4<Float> {
        let n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).squareRoot()
        return n > 1e-12 ? q / n : SIMD4<Float>(0, 0, 0, 1)
    }

    /// Row-major 3x3 rotation from a normalised quaternion (x, y, z, w).
    public static func rotationMatrix(_ q: SIMD4<Float>) -> [Float] {
        let x = q.x, y = q.y, z = q.z, w = q.w
        return [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ]
    }

    /// Project a 3D covariance into 2D screen space (EWA splatting).
    ///
    /// Σ2D = J W Σ Wᵀ Jᵀ, where W is the view rotation and J is the Jacobian
    /// of the perspective projection at this point. J is a LOCAL LINEARISATION:
    /// perspective projection is not affine, so a Gaussian does not stay
    /// Gaussian under it. Linearising at the splat centre is the standard
    /// approximation and is accurate while the splat is small relative to its
    /// depth — which is also exactly when it stops holding, for huge splats
    /// very close to the camera.
    ///
    /// Returns the 3 upper-triangular terms of the 2x2 result.
    public static func covariance2D(
        cov3D: (Float, Float, Float, Float, Float, Float),
        cameraPoint: SIMD3<Float>,
        viewRotation: [Float],          // row-major 3x3, world -> camera
        focalX: Float, focalY: Float
    ) -> (Float, Float, Float) {
        let z = max(cameraPoint.z, 1e-6)
        let invZ = 1 / z
        let invZ2 = invZ * invZ

        // d(pixel)/d(camera point).
        let j: [Float] = [
            focalX * invZ, 0, -focalX * cameraPoint.x * invZ2,
            0, focalY * invZ, -focalY * cameraPoint.y * invZ2,
        ]
        // T = J * W  (2x3)
        var t = [Float](repeating: 0, count: 6)
        for row in 0..<2 {
            for col in 0..<3 {
                var sum: Float = 0
                for k in 0..<3 { sum += j[row * 3 + k] * viewRotation[k * 3 + col] }
                t[row * 3 + col] = sum
            }
        }
        // Σ2D = T Σ Tᵀ, with Σ expanded from its 6 stored terms.
        let (c00, c01, c02, c11, c12, c22) = cov3D
        let sigma: [Float] = [c00, c01, c02, c01, c11, c12, c02, c12, c22]
        func quad(_ a: Int, _ b: Int) -> Float {
            var sum: Float = 0
            for i in 0..<3 {
                var inner: Float = 0
                for k in 0..<3 { inner += sigma[i * 3 + k] * t[b * 3 + k] }
                sum += t[a * 3 + i] * inner
            }
            return sum
        }
        // Dilate by a small isotropic term so a splat can never collapse below
        // one pixel and vanish between samples — the standard 3DGS guard, and
        // also what keeps the 2x2 invertible.
        return (quad(0, 0) + 0.3, quad(0, 1), quad(1, 1) + 0.3)
    }

    /// Screen-space extent (radius in pixels) covering ~3 standard deviations.
    /// Used to decide which tiles a splat touches; 3σ captures >99% of the mass
    /// and past that the exponential is below the 8-bit noise floor anyway.
    public static func screenRadius(cov2D: (Float, Float, Float)) -> Float {
        let (a, b, c) = cov2D
        // Larger eigenvalue of a symmetric 2x2.
        let mid = 0.5 * (a + c)
        let disc = max(0, mid * mid - (a * c - b * b))
        let lambda = mid + disc.squareRoot()
        return 3 * max(lambda, 0).squareRoot()
    }
}
