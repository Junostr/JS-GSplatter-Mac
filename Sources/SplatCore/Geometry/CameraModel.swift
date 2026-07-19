import Foundation

// MARK: - Camera model
//
// Conventions, fixed once here because getting them inconsistent is the
// classic way SfM silently produces a mirrored or inside-out reconstruction:
//   - Camera looks down +Z in camera space.
//   - Pose is WORLD -> CAMERA:  Xc = R * Xw + t
//   - Image coords are pixels, origin top-left, +y down (matching how the
//     ingestion stage hands us frames and where keypoints are detected).
//   - A point is in front of the camera iff Xc.z > 0.

/// Pinhole intrinsics with a single focal length (square pixels, no skew).
/// Distortion is deliberately absent for now: estimating it well needs either
/// a calibration target or a mature bundle adjuster, and a badly-estimated
/// distortion term is worse than none. It belongs in the BA parameter block
/// once poses are reliable.
public struct CameraIntrinsics: Equatable {
    public var focalLength: Double
    public var cx: Double
    public var cy: Double

    public init(focalLength: Double, cx: Double, cy: Double) {
        self.focalLength = focalLength
        self.cx = cx
        self.cy = cy
    }

    /// Initial guess when the capture carries no calibration.
    ///
    /// 0.72 x the longer image side, i.e. about a 69 degree horizontal field
    /// of view — a phone main camera at 4K.
    ///
    /// COLMAP's 1.2x default (roughly 45 degrees) is aimed at DSLR-style
    /// photos and is badly wrong for the footage this app actually receives.
    /// Measured on a real iPhone capture: 1.2x gave 14.29 px RMSE and ZERO
    /// surviving points, while ~0.72x gave 0.42 px and a working
    /// reconstruction. When geometric estimation cannot discriminate (see
    /// FocalEstimation), this prior is what the pipeline falls back to, so it
    /// needs to be right for the common case rather than for COLMAP's.
    public static func guess(width: Int, height: Int) -> CameraIntrinsics {
        CameraIntrinsics(
            focalLength: 0.72 * Double(max(width, height)),
            cx: Double(width) / 2,
            cy: Double(height) / 2
        )
    }

    /// Pixel -> normalized image coordinates (the ray direction with z = 1).
    public func normalize(x: Double, y: Double) -> SIMD2<Double> {
        SIMD2<Double>((x - cx) / focalLength, (y - cy) / focalLength)
    }

    /// Normalized -> pixel.
    public func project(_ normalized: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2<Double>(normalized.x * focalLength + cx, normalized.y * focalLength + cy)
    }
}

/// A rigid world-to-camera transform.
public struct CameraPose: Equatable {
    /// Row-major 3x3 rotation, world -> camera.
    public var rotation: [Double]
    /// Translation, world -> camera.
    public var translation: SIMD3<Double>

    public init(rotation: [Double], translation: SIMD3<Double>) {
        precondition(rotation.count == 9)
        self.rotation = rotation
        self.translation = translation
    }

    public static let identity = CameraPose(
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: SIMD3<Double>(0, 0, 0)
    )

    /// World point -> camera space.
    public func transform(_ worldPoint: SIMD3<Double>) -> SIMD3<Double> {
        LinearAlgebra.matVec3(rotation, worldPoint) + translation
    }

    /// Camera centre in world coordinates: C = -Rᵀ t.
    public var center: SIMD3<Double> {
        let rt = LinearAlgebra.transpose3(rotation)
        let rtT = LinearAlgebra.matVec3(rt, translation)
        return SIMD3<Double>(-rtT.x, -rtT.y, -rtT.z)
    }

    /// Project a world point to pixels. Returns nil when the point is behind
    /// the camera, which callers must treat as "no observation" rather than
    /// letting a negative depth fold the point back into the image.
    public func project(_ worldPoint: SIMD3<Double>, intrinsics: CameraIntrinsics) -> SIMD2<Double>? {
        let camera = transform(worldPoint)
        guard camera.z > 1e-9 else { return nil }
        return intrinsics.project(SIMD2<Double>(camera.x / camera.z, camera.y / camera.z))
    }

    /// Rotate by a small axis-angle increment (Rodrigues). Used by bundle
    /// adjustment, which parameterizes rotation updates as a 3-vector in the
    /// tangent space to avoid the gimbal/normalization problems of updating
    /// matrix entries directly.
    public func rotated(byAxisAngle omega: SIMD3<Double>) -> CameraPose {
        let theta = LinearAlgebra.length(omega)
        var delta: [Double]
        if theta < 1e-12 {
            delta = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        } else {
            let k = SIMD3<Double>(omega.x / theta, omega.y / theta, omega.z / theta)
            let c = cos(theta), s = sin(theta), t = 1 - c
            delta = [
                t * k.x * k.x + c,       t * k.x * k.y - s * k.z, t * k.x * k.z + s * k.y,
                t * k.x * k.y + s * k.z, t * k.y * k.y + c,       t * k.y * k.z - s * k.x,
                t * k.x * k.z - s * k.y, t * k.y * k.z + s * k.x, t * k.z * k.z + c,
            ]
        }
        return CameraPose(
            rotation: LinearAlgebra.nearestRotation(LinearAlgebra.matMul3(delta, rotation)),
            translation: translation
        )
    }
}

/// One camera's registered state in a reconstruction.
public struct RegisteredCamera {
    public let frameIndex: Int
    public var pose: CameraPose
    public var intrinsics: CameraIntrinsics

    public init(frameIndex: Int, pose: CameraPose, intrinsics: CameraIntrinsics) {
        self.frameIndex = frameIndex
        self.pose = pose
        self.intrinsics = intrinsics
    }
}

/// A triangulated 3D point and the observations that produced it.
public struct ScenePoint {
    public var position: SIMD3<Double>
    /// (frameIndex, keypointIndex) pairs.
    public var observations: [(frame: Int, keypoint: Int)]

    public init(position: SIMD3<Double>, observations: [(frame: Int, keypoint: Int)]) {
        self.position = position
        self.observations = observations
    }
}

/// The output of stage 3: camera poses plus a sparse point cloud, which is
/// exactly what stage 5 needs to initialize gaussians.
public struct Reconstruction {
    public var cameras: [Int: RegisteredCamera]
    public var points: [ScenePoint]

    public init(cameras: [Int: RegisteredCamera] = [:], points: [ScenePoint] = []) {
        self.cameras = cameras
        self.points = points
    }

    /// Root-mean-square reprojection error in pixels over all observations.
    /// The headline health metric for a reconstruction: under ~1 px is good,
    /// several px means the poses or matches are wrong.
    public func reprojectionRMSE(keypoints: [Int: [Keypoint]]) -> Double {
        var sum = 0.0
        var count = 0
        for point in points {
            for observation in point.observations {
                guard let camera = cameras[observation.frame],
                      let projected = camera.pose.project(point.position, intrinsics: camera.intrinsics),
                      let frameKeypoints = keypoints[observation.frame],
                      observation.keypoint < frameKeypoints.count else { continue }
                let kp = frameKeypoints[observation.keypoint]
                let dx = projected.x - Double(kp.x)
                let dy = projected.y - Double(kp.y)
                sum += dx * dx + dy * dy
                count += 1
            }
        }
        return count > 0 ? (sum / Double(count)).squareRoot() : 0
    }
}
