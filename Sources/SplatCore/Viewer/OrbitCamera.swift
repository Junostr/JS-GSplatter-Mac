import Foundation

// MARK: - Stage 6: orbit camera
//
// The camera model for the interactive viewer and for turntable rendering.
// Parameterised by a target point and spherical coordinates about it, because
// that is what an orbit control manipulates directly: drag changes azimuth and
// elevation, scroll changes distance, and the target stays put unless panned.
//
// Camera conventions are fixed once in CameraModel.swift and this must obey
// them exactly, or the viewer silently mirrors or turns inside-out — the same
// failure mode that cost real time in stage 3. The tests pin the convention
// against known geometry rather than trusting the derivation.

public struct OrbitCamera {
    /// World point the camera looks at and orbits around.
    public var target: SIMD3<Double>
    /// Distance from target to eye.
    public var distance: Double
    /// Horizontal orbit angle (radians). 0 places the eye on the +Z side of
    /// the target looking back toward -Z, matching the camera's +Z-forward
    /// convention at azimuth 0.
    public var azimuth: Double
    /// Vertical orbit angle (radians), clamped away from the poles.
    public var elevation: Double
    /// Vertical field of view (radians).
    public var verticalFOV: Double

    /// How close to straight up/down the elevation may get. At exactly ±90°
    /// the view direction is parallel to world-up and the right vector is
    /// undefined (cross product collapses), so the basis would be garbage.
    public static let maxElevation = 89.0 * .pi / 180.0

    public init(target: SIMD3<Double>, distance: Double,
                azimuth: Double = 0, elevation: Double = 0.3,
                verticalFOV: Double = 50.0 * .pi / 180.0) {
        self.target = target
        self.distance = max(distance, 1e-4)
        self.azimuth = azimuth
        self.elevation = min(max(elevation, -Self.maxElevation), Self.maxElevation)
        self.verticalFOV = verticalFOV
    }

    /// Eye position in world coordinates.
    public var eye: SIMD3<Double> {
        let ce = cos(elevation)
        let offset = SIMD3<Double>(ce * sin(azimuth), sin(elevation), ce * cos(azimuth))
        return target + offset * distance
    }

    /// World -> camera pose obeying the project convention: camera looks down
    /// +Z, +y is down in the image, Xc = R Xw + t.
    public var pose: CameraPose {
        let eyePosition = eye
        // Forward (+Z): from eye toward target.
        var forward = target - eyePosition
        let forwardLength = LinearAlgebra.length(forward)
        forward = forwardLength > 1e-12
            ? SIMD3<Double>(forward.x / forwardLength, forward.y / forwardLength, forward.z / forwardLength)
            : SIMD3<Double>(0, 0, 1)

        // Right (+X): forward × world-up. The order matters and is not
        // arbitrary — `worldUp × forward` gives the opposite sign and flips the
        // image upside down. Verified by hand and pinned in the tests: at
        // azimuth 0 the eye is at world +Z looking toward -Z, and a point above
        // the target (+Y) must land in the UPPER half of the image. World-up is
        // +Y; the elevation clamp guarantees these are not parallel.
        let worldUp = SIMD3<Double>(0, 1, 0)
        var right = LinearAlgebra.cross(forward, worldUp)
        let rightLength = LinearAlgebra.length(right)
        right = rightLength > 1e-12
            ? SIMD3<Double>(right.x / rightLength, right.y / rightLength, right.z / rightLength)
            : SIMD3<Double>(1, 0, 0)

        // Down (+Y in image): forward × right completes a right-handed basis
        // (right × down = forward) with +Y pointing DOWN in world when the
        // camera is upright, matching the top-left image origin.
        let down = LinearAlgebra.cross(forward, right)

        // Rows of R are the camera axes expressed in world coordinates.
        let rotation: [Double] = [
            right.x, right.y, right.z,
            down.x, down.y, down.z,
            forward.x, forward.y, forward.z,
        ]
        // t = -R * eye, so the eye maps to the camera-space origin.
        let rEye = LinearAlgebra.matVec3(rotation, eyePosition)
        return CameraPose(rotation: rotation, translation: SIMD3<Double>(-rEye.x, -rEye.y, -rEye.z))
    }

    /// Intrinsics for a viewport. Focal length follows from the vertical FOV so
    /// the framing is independent of resolution: resizing the window changes
    /// the pixel count, not what is visible.
    public func intrinsics(width: Int, height: Int) -> CameraIntrinsics {
        let focal = Double(height) / 2 / tan(verticalFOV / 2)
        return CameraIntrinsics(focalLength: focal, cx: Double(width) / 2, cy: Double(height) / 2)
    }

    // MARK: Interaction

    /// Orbit by pixel-scaled deltas. Elevation is clamped; azimuth wraps freely.
    public mutating func orbit(deltaAzimuth: Double, deltaElevation: Double) {
        azimuth += deltaAzimuth
        elevation = min(max(elevation + deltaElevation, -Self.maxElevation), Self.maxElevation)
    }

    /// Multiplicative zoom: factor < 1 moves closer, > 1 further. Multiplicative
    /// rather than additive so a scroll notch feels the same whether the camera
    /// is near or far, and distance can never cross zero.
    public mutating func zoom(factor: Double) {
        distance = max(distance * factor, 1e-4)
    }

    /// Pan the target within the camera's image plane, scaled so a drag moves
    /// the scene by roughly the same screen distance at any zoom level.
    public mutating func pan(dx: Double, dy: Double) {
        let p = pose
        // Camera right and down axes are rows 0 and 1 of R.
        let right = SIMD3<Double>(p.rotation[0], p.rotation[1], p.rotation[2])
        let down = SIMD3<Double>(p.rotation[3], p.rotation[4], p.rotation[5])
        let scale = distance * tan(verticalFOV / 2)
        target += right * (dx * scale) + down * (dy * scale)
    }

    /// Auto-frame a point set: centre on its centroid and back off far enough
    /// that its bounding sphere fits the vertical field of view. The default
    /// starting view for opening a scene.
    public static func framing(points: [SIMD3<Double>], verticalFOV: Double = 50.0 * .pi / 180.0) -> OrbitCamera {
        guard !points.isEmpty else {
            return OrbitCamera(target: .zero, distance: 1, verticalFOV: verticalFOV)
        }
        var centroid = SIMD3<Double>.zero
        for p in points { centroid += p }
        centroid /= Double(points.count)

        // Radius from the 90th-percentile distance, not the maximum.
        //
        // A reconstruction always has a few badly-triangulated outlier splats
        // far from the real scene, and a max-distance radius lets one of them
        // dictate the whole framing — the camera backs off to fit the outlier
        // and the actual content shrinks to a dot. Same lesson as focal
        // estimation: a robust statistic over the raw extremum. Measured on a
        // real trained scene, the outliers put the true content at a few
        // percent of the frame; the percentile fixes it.
        var distances = points.map { LinearAlgebra.length($0 - centroid) }
        distances.sort()
        let percentileIndex = min(distances.count - 1, Int(Double(distances.count) * 0.9))
        let radius = max(distances[percentileIndex], 1e-3)
        // Distance so the bounding sphere subtends the vertical FOV, with a
        // little margin so the scene is not flush against the frame edge.
        let distance = radius / sin(verticalFOV / 2) * 1.2
        return OrbitCamera(target: centroid, distance: distance, verticalFOV: verticalFOV)
    }

    /// Frame a reconstruction directly — the common entry point from stage 3/5.
    public static func framing(reconstruction: Reconstruction,
                               verticalFOV: Double = 50.0 * .pi / 180.0) -> OrbitCamera {
        framing(points: reconstruction.points.map { $0.position }, verticalFOV: verticalFOV)
    }

    /// Frame a splat cloud.
    public static func framing(cloud: SplatCloud,
                               verticalFOV: Double = 50.0 * .pi / 180.0) -> OrbitCamera {
        framing(points: cloud.positions.map { SIMD3<Double>(Double($0.x), Double($0.y), Double($0.z)) },
                verticalFOV: verticalFOV)
    }
}
