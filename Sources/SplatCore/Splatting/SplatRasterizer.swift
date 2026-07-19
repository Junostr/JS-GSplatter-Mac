import Foundation

// MARK: - Stage 5: forward rasterization
//
// Projects every Gaussian to screen space and alpha-composites them front to
// back. This is the CPU reference implementation: the Metal kernel is verified
// against it, exactly as the stage 2 and 3 tiers were.
//
// Front-to-back (rather than back-to-front) because it allows early
// termination: once accumulated transmittance falls below a threshold, nothing
// behind can contribute a visible amount and the remaining splats for that
// pixel can be skipped. It is also the order the backward pass needs, so the
// two stay symmetric.

public struct RenderTarget {
    public let width: Int
    public let height: Int
    /// RGB, row-major, 3 floats per pixel.
    public var pixels: [Float]
    /// Final transmittance per pixel — how much background shows through.
    /// Kept because the backward pass needs it, and it doubles as an alpha
    /// channel for compositing.
    public var transmittance: [Float]

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.pixels = [Float](repeating: 0, count: width * height * 3)
        self.transmittance = [Float](repeating: 1, count: width * height)
    }

    public func pixel(x: Int, y: Int) -> SIMD3<Float> {
        let i = (y * width + x) * 3
        return SIMD3<Float>(pixels[i], pixels[i + 1], pixels[i + 2])
    }
}

public struct RasterizerOptions {
    /// Stop compositing a pixel once transmittance drops below this.
    /// 1/255 is the point where further contributions cannot change an 8-bit
    /// output, so continuing is pure waste.
    public var transmittanceCutoff: Float
    /// Splats closer than this to the camera are dropped. The EWA Jacobian is
    /// a linearisation about the splat centre and degenerates as z approaches
    /// zero, producing enormous screen-space ellipses from a near-zero divide.
    public var nearPlane: Float
    /// Reject splats whose screen radius exceeds this many pixels. A single
    /// splat covering most of the frame is nearly always a degenerate one that
    /// would otherwise dominate every tile it touches.
    public var maxScreenRadius: Float

    public init(transmittanceCutoff: Float = 1.0 / 255.0,
                nearPlane: Float = 0.2,
                maxScreenRadius: Float = 512) {
        self.transmittanceCutoff = transmittanceCutoff
        self.nearPlane = nearPlane
        self.maxScreenRadius = maxScreenRadius
    }
}

/// One splat prepared for rasterization: everything the per-pixel loop needs,
/// computed once per splat rather than per pixel.
public struct ProjectedSplat {
    public let index: Int
    public let depth: Float
    public let centre: SIMD2<Float>
    /// Inverse of the 2D covariance (the "conic"), upper-triangular.
    /// Inverted once here because the per-pixel test needs Σ⁻¹, and a splat
    /// covers many pixels.
    public let conic: (Float, Float, Float)
    public let radius: Float
    public let color: SIMD3<Float>
    public let opacity: Float
}

public enum SplatRasterizer {

    /// Project all splats into screen space, dropping those that cannot
    /// contribute. Shared by both tiers and by the backward pass, so culling
    /// rules can never differ between them.
    public static func project(
        cloud: SplatCloud,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        width: Int, height: Int,
        options: RasterizerOptions = RasterizerOptions()
    ) -> [ProjectedSplat] {
        var out: [ProjectedSplat] = []
        out.reserveCapacity(cloud.count)

        let r = pose.rotation.map { Float($0) }
        let t = SIMD3<Float>(Float(pose.translation.x), Float(pose.translation.y), Float(pose.translation.z))
        let fx = Float(intrinsics.focalLength), fy = Float(intrinsics.focalLength)
        let cx = Float(intrinsics.cx), cy = Float(intrinsics.cy)

        for i in 0..<cloud.count {
            let p = cloud.positions[i]
            let camera = SIMD3<Float>(
                r[0] * p.x + r[1] * p.y + r[2] * p.z + t.x,
                r[3] * p.x + r[4] * p.y + r[5] * p.z + t.y,
                r[6] * p.x + r[7] * p.y + r[8] * p.z + t.z
            )
            guard camera.z > options.nearPlane else { continue }

            let screenX = fx * camera.x / camera.z + cx
            let screenY = fy * camera.y / camera.z + cy

            let cov3 = SplatMath.covariance3D(logScale: cloud.logScales[i], rotation: cloud.rotations[i])
            let cov2 = SplatMath.covariance2D(cov3D: cov3, cameraPoint: camera,
                                              viewRotation: r, focalX: fx, focalY: fy)
            let radius = SplatMath.screenRadius(cov2D: cov2)
            guard radius >= 0.5, radius <= options.maxScreenRadius else { continue }

            // Off-screen test with the radius as margin.
            guard screenX + radius >= 0, screenX - radius < Float(width),
                  screenY + radius >= 0, screenY - radius < Float(height) else { continue }

            let (a, b, c) = cov2
            let det = a * c - b * b
            guard det > 1e-12 else { continue }
            let invDet = 1 / det
            let conic = (c * invDet, -b * invDet, a * invDet)

            let opacity = 1 / (1 + expf(-cloud.opacityLogits[i]))
            out.append(ProjectedSplat(
                index: i, depth: camera.z,
                centre: SIMD2<Float>(screenX, screenY),
                conic: conic, radius: radius,
                color: cloud.colors[i], opacity: opacity
            ))
        }
        return out
    }

    /// Rasterize projected splats into a target, front to back.
    public static func render(
        projected: [ProjectedSplat],
        width: Int, height: Int,
        background: SIMD3<Float> = SIMD3<Float>(repeating: 0),
        options: RasterizerOptions = RasterizerOptions()
    ) -> RenderTarget {
        var target = RenderTarget(width: width, height: height)
        guard !projected.isEmpty else {
            // Nothing to draw: the background shows through everywhere.
            for y in 0..<height {
                for x in 0..<width {
                    let i = (y * width + x) * 3
                    target.pixels[i] = background.x
                    target.pixels[i + 1] = background.y
                    target.pixels[i + 2] = background.z
                }
            }
            return target
        }

        // Depth sort, nearest first. Ties broken by index so the result is
        // deterministic — with equal depths the blend order changes the output,
        // and unstable sorts would make renders vary run to run.
        let order = projected.indices.sorted {
            projected[$0].depth != projected[$1].depth
                ? projected[$0].depth < projected[$1].depth
                : projected[$0].index < projected[$1].index
        }

        for slot in order {
            let splat = projected[slot]
            let minX = max(0, Int((splat.centre.x - splat.radius).rounded(.down)))
            let maxX = min(width - 1, Int((splat.centre.x + splat.radius).rounded(.up)))
            let minY = max(0, Int((splat.centre.y - splat.radius).rounded(.down)))
            let maxY = min(height - 1, Int((splat.centre.y + splat.radius).rounded(.up)))
            guard minX <= maxX, minY <= maxY else { continue }

            for y in minY...maxY {
                for x in minX...maxX {
                    let pixelIndex = y * width + x
                    let transmittance = target.transmittance[pixelIndex]
                    guard transmittance > options.transmittanceCutoff else { continue }

                    // Gaussian falloff: exp(-0.5 * dᵀ Σ⁻¹ d).
                    let dx = Float(x) - splat.centre.x
                    let dy = Float(y) - splat.centre.y
                    let (ca, cb, cc) = splat.conic
                    let power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy
                    guard power <= 0 else { continue }
                    let alpha = min(0.99, splat.opacity * expf(power))
                    guard alpha >= 1.0 / 255.0 else { continue }

                    let weight = alpha * transmittance
                    let base = pixelIndex * 3
                    target.pixels[base] += splat.color.x * weight
                    target.pixels[base + 1] += splat.color.y * weight
                    target.pixels[base + 2] += splat.color.z * weight
                    target.transmittance[pixelIndex] = transmittance * (1 - alpha)
                }
            }
        }

        // Composite whatever background remains visible.
        for i in 0..<(width * height) {
            let remaining = target.transmittance[i]
            let base = i * 3
            target.pixels[base] += background.x * remaining
            target.pixels[base + 1] += background.y * remaining
            target.pixels[base + 2] += background.z * remaining
        }
        return target
    }

    /// Convenience: project and render in one call.
    public static func render(
        cloud: SplatCloud, pose: CameraPose, intrinsics: CameraIntrinsics,
        width: Int, height: Int,
        background: SIMD3<Float> = SIMD3<Float>(repeating: 0),
        options: RasterizerOptions = RasterizerOptions()
    ) -> RenderTarget {
        let projected = project(cloud: cloud, pose: pose, intrinsics: intrinsics,
                                width: width, height: height, options: options)
        return render(projected: projected, width: width, height: height,
                      background: background, options: options)
    }

    /// Mean absolute error between a render and a reference image (RGB float,
    /// same layout). This is the L1 term of the 3DGS loss; the SSIM term comes
    /// with the backward pass.
    public static func meanAbsoluteError(_ target: RenderTarget, reference: [Float]) -> Double {
        guard reference.count == target.pixels.count, !reference.isEmpty else { return 0 }
        var sum = 0.0
        for i in 0..<target.pixels.count {
            sum += Double(abs(target.pixels[i] - reference[i]))
        }
        return sum / Double(target.pixels.count)
    }
}
