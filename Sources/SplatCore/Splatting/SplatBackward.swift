import Foundation

// MARK: - Stage 5: backward pass
//
// Gradients of the image loss with respect to every splat parameter. The chain,
// outermost first:
//
//   L  <- pixel colour  <- alpha blend  <- (colour, alpha) per splat
//      <- alpha = opacity * G          <- G = exp(power)
//      <- power                        <- conic (inverse 2D covariance), centre
//      <- 2D covariance                <- 3D covariance, camera-space position
//      <- 3D covariance                <- log-scale, rotation quaternion
//      <- camera position              <- world position
//
// Every link below is verified against finite differences in selftest. That is
// not optional decoration: a sign error or a missing term in any one of these
// produces gradients that still look plausible — smooth, right magnitude, loss
// even decreasing for a while — and then trains to something subtly wrong.
// Finite differences are the only check that distinguishes "derivative" from
// "something shaped like a derivative".

public struct SplatGradients {
    public var positions: [SIMD3<Float>]
    public var logScales: [SIMD3<Float>]
    public var rotations: [SIMD4<Float>]
    public var opacityLogits: [Float]
    public var colors: [SIMD3<Float>]

    public init(count: Int) {
        positions = Array(repeating: .zero, count: count)
        logScales = Array(repeating: .zero, count: count)
        rotations = Array(repeating: .zero, count: count)
        opacityLogits = Array(repeating: 0, count: count)
        colors = Array(repeating: .zero, count: count)
    }
}

public enum SplatBackward {

    /// L1 loss against a reference image, plus gradients for every parameter.
    ///
    /// L1 rather than L2: photographic references contain specular highlights,
    /// motion blur and exposure differences that no geometry can explain, and
    /// squared error lets those few large residuals dominate the whole update.
    /// (Full 3DGS adds a D-SSIM term for structural fidelity; that is a later
    /// addition and is why the loss is factored out rather than inlined.)
    public static func lossAndGradients(
        cloud: SplatCloud,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        width: Int, height: Int,
        reference: [Float],
        background: SIMD3<Float> = SIMD3<Float>(repeating: 0),
        options: RasterizerOptions = RasterizerOptions()
    ) -> (loss: Double, gradients: SplatGradients) {

        var gradients = SplatGradients(count: cloud.count)
        let projected = SplatRasterizer.project(cloud: cloud, pose: pose, intrinsics: intrinsics,
                                                width: width, height: height, options: options)
        let target = SplatRasterizer.render(projected: projected, width: width, height: height,
                                            background: background, options: options)
        let pixelCount = width * height
        guard reference.count == pixelCount * 3 else {
            return (0, gradients)
        }

        // dL/d(pixel) for mean-absolute-error: sign(rendered - reference) / N.
        var dLdPixel = [Float](repeating: 0, count: pixelCount * 3)
        var loss = 0.0
        let inverseN = Float(1.0 / Double(pixelCount * 3))
        for i in 0..<(pixelCount * 3) {
            let diff = target.pixels[i] - reference[i]
            loss += Double(abs(diff))
            dLdPixel[i] = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) * inverseN
        }
        loss /= Double(pixelCount * 3)

        guard !projected.isEmpty else { return (loss, gradients) }

        // Same front-to-back order the forward pass used. Recomputing the walk
        // rather than caching per-pixel per-splat transmittance is deliberate:
        // caching is O(pixels x splats) memory, which at 4K with thousands of
        // splats is tens of gigabytes.
        let order = projected.indices.sorted {
            projected[$0].depth != projected[$1].depth
                ? projected[$0].depth < projected[$1].depth
                : projected[$0].index < projected[$1].index
        }

        let r = pose.rotation.map { Float($0) }
        let t = SIMD3<Float>(Float(pose.translation.x), Float(pose.translation.y), Float(pose.translation.z))
        let fx = Float(intrinsics.focalLength), fy = Float(intrinsics.focalLength)

        // Per-pixel running state, matching the forward pass.
        var transmittance = [Float](repeating: 1, count: pixelCount)
        // Colour accumulated so far (in front of the current splat).
        var accumulated = [Float](repeating: 0, count: pixelCount * 3)

        // Accumulators in screen space, converted to parameter space per splat.
        var dLdCentre = [SIMD2<Float>](repeating: .zero, count: projected.count)
        var dLdConic = [SIMD3<Float>](repeating: .zero, count: projected.count)   // (A, B, C)
        var dLdOpacity = [Float](repeating: 0, count: projected.count)

        for slot in order {
            let splat = projected[slot]
            let minX = max(0, Int((splat.centre.x - splat.radius).rounded(.down)))
            let maxX = min(width - 1, Int((splat.centre.x + splat.radius).rounded(.up)))
            let minY = max(0, Int((splat.centre.y - splat.radius).rounded(.down)))
            let maxY = min(height - 1, Int((splat.centre.y + splat.radius).rounded(.up)))
            guard minX <= maxX, minY <= maxY else { continue }

            let (ca, cb, cc) = splat.conic
            var gCentre = SIMD2<Float>.zero
            var gConic = SIMD3<Float>.zero
            var gOpacity: Float = 0
            var gColor = SIMD3<Float>.zero

            for y in minY...maxY {
                for x in minX...maxX {
                    let pixelIndex = y * width + x
                    let tCurrent = transmittance[pixelIndex]
                    guard tCurrent > options.transmittanceCutoff else { continue }

                    let dx = Float(x) - splat.centre.x
                    let dy = Float(y) - splat.centre.y
                    let power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy
                    guard power <= 0 else { continue }
                    let g = expf(power)
                    let rawAlpha = splat.opacity * g
                    let alpha = min(0.99, rawAlpha)
                    guard alpha >= 1.0 / 255.0 else { continue }

                    let base = pixelIndex * 3
                    let grad = SIMD3<Float>(dLdPixel[base], dLdPixel[base + 1], dLdPixel[base + 2])

                    // Colour gradient is direct: this splat contributes
                    // c * alpha * T to the pixel.
                    let weight = alpha * tCurrent
                    gColor += grad * weight

                    // Alpha gradient has two parts. Directly, this splat adds
                    // c*alpha*T. Indirectly, everything BEHIND it (and the
                    // background) is attenuated by (1 - alpha), so raising
                    // alpha removes exactly that much of their contribution.
                    //
                    // The "rest behind" is recovered as
                    //   final - accumulated_in_front - this_splat
                    // rather than by caching it per pixel, which keeps memory
                    // at O(pixels) instead of O(pixels x splats).
                    let finalColor = SIMD3<Float>(target.pixels[base], target.pixels[base + 1], target.pixels[base + 2])
                    let inFront = SIMD3<Float>(accumulated[base], accumulated[base + 1], accumulated[base + 2])
                    let mine = splat.color * weight
                    let behind = finalColor - inFront - mine
                    let oneMinusAlpha = max(1 - alpha, 1e-6)
                    let dLdAlpha = simdDot(grad, splat.color * tCurrent - behind / oneMinusAlpha)

                    // Clamping alpha at 0.99 makes the gradient zero beyond it:
                    // past the clamp the output genuinely does not respond to
                    // the parameter, and pretending otherwise would push
                    // opacity up forever with no effect on the image.
                    let dAlphaDRaw: Float = rawAlpha <= 0.99 ? 1 : 0
                    let dLdRawAlpha = dLdAlpha * dAlphaDRaw

                    gOpacity += dLdRawAlpha * g
                    let dLdG = dLdRawAlpha * splat.opacity
                    let dLdPower = dLdG * g          // d(exp)/d(power) = exp

                    // power = -0.5(A dx² + C dy²) - B dx dy
                    gConic.x += dLdPower * (-0.5 * dx * dx)
                    gConic.y += dLdPower * (-dx * dy)
                    gConic.z += dLdPower * (-0.5 * dy * dy)

                    // dx = x - centre.x, so d(power)/d(centre) = -d(power)/d(d).
                    let dPowerDdx = -(ca * dx) - cb * dy
                    let dPowerDdy = -(cc * dy) - cb * dx
                    gCentre.x += dLdPower * (-dPowerDdx)
                    gCentre.y += dLdPower * (-dPowerDdy)

                    // Advance the forward state exactly as the renderer did.
                    accumulated[base] += splat.color.x * weight
                    accumulated[base + 1] += splat.color.y * weight
                    accumulated[base + 2] += splat.color.z * weight
                    transmittance[pixelIndex] = tCurrent * (1 - alpha)
                }
            }

            dLdCentre[slot] = gCentre
            dLdConic[slot] = gConic
            dLdOpacity[slot] = gOpacity
            gradients.colors[splat.index] += gColor
        }

        // Screen space -> parameter space, once per splat.
        for slot in projected.indices {
            let splat = projected[slot]
            let i = splat.index

            // Opacity: alpha = sigmoid(logit) * G, and d(sigmoid)/d(logit)
            // = s(1-s).
            let o = splat.opacity
            gradients.opacityLogits[i] += dLdOpacity[slot] * o * (1 - o)

            // Rebuild what the projection produced, to differentiate through it.
            let p = cloud.positions[i]
            let camera = SIMD3<Float>(
                r[0] * p.x + r[1] * p.y + r[2] * p.z + t.x,
                r[3] * p.x + r[4] * p.y + r[5] * p.z + t.y,
                r[6] * p.x + r[7] * p.y + r[8] * p.z + t.z
            )
            let z = max(camera.z, 1e-6)
            let invZ = 1 / z

            // Centre gradient -> camera-space position.
            // centre = (fx * X/Z + cx, fy * Y/Z + cy)
            let gc = dLdCentre[slot]
            var dLdCamera = SIMD3<Float>(
                gc.x * fx * invZ,
                gc.y * fy * invZ,
                -(gc.x * fx * camera.x + gc.y * fy * camera.y) * invZ * invZ
            )

            // Conic gradient -> 2D covariance. For M = Σ2D and its inverse,
            // dL/dM = -M⁻¹ (dL/dM⁻¹) M⁻¹, which for symmetric 2x2 expands as
            // below. The off-diagonal factor of 2 accounts for B appearing in
            // both the (0,1) and (1,0) slots of the symmetric matrix.
            let cov2 = SplatMath.covariance2D(cov3D: SplatMath.covariance3D(
                logScale: cloud.logScales[i], rotation: cloud.rotations[i]),
                cameraPoint: camera, viewRotation: r, focalX: fx, focalY: fy)
            let (sa, sb, sc) = cov2
            let det = sa * sc - sb * sb
            guard det > 1e-12 else { continue }
            let invDet = 1 / det
            // conic = (sc, -sb, sa) * invDet
            let gA = dLdConic[slot].x, gB = dLdConic[slot].y, gC = dLdConic[slot].z
            let invDet2 = invDet * invDet
            // Derivatives of each conic term w.r.t. each covariance term.
            let dA_da = -sc * sc * invDet2
            let dA_db = 2 * sc * sb * invDet2
            let dA_dc = invDet - sc * sa * invDet2
            let dB_da = sb * sc * invDet2
            let dB_db = -invDet - (-sb) * 2 * sb * invDet2
            let dB_dc = sb * sa * invDet2
            let dC_da = invDet - sa * sc * invDet2
            let dC_db = 2 * sa * sb * invDet2
            let dC_dc = -sa * sa * invDet2

            let dLd_sa = gA * dA_da + gB * dB_da + gC * dC_da
            let dLd_sb = gA * dA_db + gB * dB_db + gC * dC_db
            let dLd_sc = gA * dA_dc + gB * dB_dc + gC * dC_dc

            // 2D covariance -> 3D covariance, through Σ2D = T Σ Tᵀ with T = J W.
            let invZ2 = invZ * invZ
            let j: [Float] = [
                fx * invZ, 0, -fx * camera.x * invZ2,
                0, fy * invZ, -fy * camera.y * invZ2,
            ]
            var tMat = [Float](repeating: 0, count: 6)     // 2x3
            for row in 0..<2 {
                for col in 0..<3 {
                    var sum: Float = 0
                    for k in 0..<3 { sum += j[row * 3 + k] * r[k * 3 + col] }
                    tMat[row * 3 + col] = sum
                }
            }
            // dL/dΣ = Tᵀ (dL/dΣ2D) T, with the 2x2 gradient symmetrised.
            let g2: [Float] = [dLd_sa, dLd_sb * 0.5, dLd_sb * 0.5, dLd_sc]
            var dLdSigma = [Float](repeating: 0, count: 9)
            for a in 0..<3 {
                for b in 0..<3 {
                    var sum: Float = 0
                    for m in 0..<2 {
                        for n in 0..<2 {
                            sum += tMat[m * 3 + a] * g2[m * 2 + n] * tMat[n * 3 + b]
                        }
                    }
                    dLdSigma[a * 3 + b] = sum
                }
            }

            // 3D covariance -> scale and rotation, through Σ = M Mᵀ, M = R S.
            let q = SplatMath.normalizeQuaternion(cloud.rotations[i])
            let rot = SplatMath.rotationMatrix(q)
            let s = SIMD3<Float>(expf(cloud.logScales[i].x),
                                 expf(cloud.logScales[i].y),
                                 expf(cloud.logScales[i].z))
            var m = [Float](repeating: 0, count: 9)
            for row in 0..<3 {
                m[row * 3 + 0] = rot[row * 3 + 0] * s.x
                m[row * 3 + 1] = rot[row * 3 + 1] * s.y
                m[row * 3 + 2] = rot[row * 3 + 2] * s.z
            }
            // dL/dM = (dL/dΣ + dL/dΣᵀ) M
            var dLdM = [Float](repeating: 0, count: 9)
            for a in 0..<3 {
                for b in 0..<3 {
                    var sum: Float = 0
                    for k in 0..<3 {
                        sum += (dLdSigma[a * 3 + k] + dLdSigma[k * 3 + a]) * m[k * 3 + b]
                    }
                    dLdM[a * 3 + b] = sum
                }
            }
            // M = R S with S diagonal, so dL/ds_col = Σ_row dL/dM[row][col] * R[row][col],
            // and s = exp(logScale) gives the extra factor of s.
            for col in 0..<3 {
                var sum: Float = 0
                for row in 0..<3 { sum += dLdM[row * 3 + col] * rot[row * 3 + col] }
                let component = sum * s[col]
                if col == 0 { gradients.logScales[i].x += component }
                else if col == 1 { gradients.logScales[i].y += component }
                else { gradients.logScales[i].z += component }
            }
            // dL/dR[row][col] = dL/dM[row][col] * s_col, then R -> quaternion.
            var dLdR = [Float](repeating: 0, count: 9)
            for row in 0..<3 {
                for col in 0..<3 { dLdR[row * 3 + col] = dLdM[row * 3 + col] * s[col] }
            }
            gradients.rotations[i] += quaternionGradient(q: cloud.rotations[i], dLdR: dLdR)

            // Camera-space position also moves the 2D covariance (through J),
            // not only the centre. Omitting that term is a classic silent error:
            // gradients stay smooth and plausible while position updates are
            // systematically wrong wherever a splat is off-axis.
            let cov3 = SplatMath.covariance3D(logScale: cloud.logScales[i], rotation: cloud.rotations[i])
            dLdCamera += covarianceDepthGradient(
                cov3D: cov3, camera: camera, viewRotation: r, fx: fx, fy: fy, dLd2D: (dLd_sa, dLd_sb, dLd_sc))

            // Camera -> world: Xc = R Xw + t, so dL/dXw = Rᵀ dL/dXc.
            gradients.positions[i] += SIMD3<Float>(
                r[0] * dLdCamera.x + r[3] * dLdCamera.y + r[6] * dLdCamera.z,
                r[1] * dLdCamera.x + r[4] * dLdCamera.y + r[7] * dLdCamera.z,
                r[2] * dLdCamera.x + r[5] * dLdCamera.y + r[8] * dLdCamera.z
            )
        }
        return (loss, gradients)
    }

    @inline(__always)
    static func simdDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    /// d(2D covariance)/d(camera position), contracted with the incoming 2D
    /// gradient. Only the Jacobian J depends on the camera point; W does not.
    ///
    /// Computed by finite-differencing J itself rather than by hand-expanding
    /// the third derivative. J is a closed-form 2x3 of the point, so this is a
    /// numerically clean derivative of an exact function — quite different from
    /// finite-differencing the whole renderer, which is what the tests do to
    /// check this file. The hand-expansion is several dozen terms and its only
    /// benefit would be speed in a path that runs once per splat, not per pixel.
    static func covarianceDepthGradient(
        cov3D: (Float, Float, Float, Float, Float, Float),
        camera: SIMD3<Float>, viewRotation: [Float],
        fx: Float, fy: Float,
        dLd2D: (Float, Float, Float)
    ) -> SIMD3<Float> {
        let epsilon: Float = 1e-3
        var out = SIMD3<Float>.zero
        for axis in 0..<3 {
            var plus = camera, minus = camera
            plus[axis] += epsilon
            minus[axis] -= epsilon
            let a = SplatMath.covariance2D(cov3D: cov3D, cameraPoint: plus,
                                           viewRotation: viewRotation, focalX: fx, focalY: fy)
            let b = SplatMath.covariance2D(cov3D: cov3D, cameraPoint: minus,
                                           viewRotation: viewRotation, focalX: fx, focalY: fy)
            let d = ((a.0 - b.0) / (2 * epsilon), (a.1 - b.1) / (2 * epsilon), (a.2 - b.2) / (2 * epsilon))
            out[axis] = dLd2D.0 * d.0 + dLd2D.1 * d.1 + dLd2D.2 * d.2
        }
        return out
    }

    /// Chain a rotation-matrix gradient back to the (unnormalised) quaternion,
    /// including the normalisation Jacobian.
    ///
    /// The normalisation term matters: the stored quaternion is not unit, and
    /// without projecting the gradient onto the sphere's tangent the optimiser
    /// spends its updates changing the quaternion's LENGTH, which the rotation
    /// matrix ignores entirely — the parameter drifts while the splat does not
    /// turn.
    static func quaternionGradient(q raw: SIMD4<Float>, dLdR: [Float]) -> SIMD4<Float> {
        let n = (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z + raw.w * raw.w).squareRoot()
        guard n > 1e-12 else { return .zero }
        let q = raw / n
        let x = q.x, y = q.y, z = q.z, w = q.w

        // dR/dq for the NORMALISED quaternion, row-major as in rotationMatrix.
        func dR(_ component: Int) -> [Float] {
            switch component {
            case 0: // x
                return [0, 2 * y, 2 * z,
                        2 * y, -4 * x, -2 * w,
                        2 * z, 2 * w, -4 * x]
            case 1: // y
                return [-4 * y, 2 * x, 2 * w,
                        2 * x, 0, 2 * z,
                        -2 * w, 2 * z, -4 * y]
            case 2: // z
                return [-4 * z, -2 * w, 2 * x,
                        2 * w, -4 * z, 2 * y,
                        2 * x, 2 * y, 0]
            default: // w
                return [0, -2 * z, 2 * y,
                        2 * z, 0, -2 * x,
                        -2 * y, 2 * x, 0]
            }
        }
        var gNormalised = SIMD4<Float>.zero
        for component in 0..<4 {
            let d = dR(component)
            var sum: Float = 0
            for k in 0..<9 { sum += dLdR[k] * d[k] }
            gNormalised[component] = sum
        }
        // Normalisation Jacobian: d(q/|q|)/dq = (I - q̂ q̂ᵀ)/|q|.
        let dot = gNormalised.x * q.x + gNormalised.y * q.y + gNormalised.z * q.z + gNormalised.w * q.w
        return (gNormalised - q * dot) / n
    }
}
