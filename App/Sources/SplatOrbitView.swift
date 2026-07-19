import AppKit
import SwiftUI
import SplatCore

// MARK: - Stage 6: interactive orbit viewer
//
// An NSView that renders a splat cloud from an OrbitCamera and turns drag /
// scroll / pan into camera moves. It deliberately reuses the verified
// rasterizer (MetalSplatRasterizer, CPU fallback) and draws the resulting
// RenderTarget as a CGImage, rather than standing up an MTKView + a
// drawable-writing kernel: the rasterizer is already checked against the CPU
// reference to float round-off, and a preview does not need the last few
// milliseconds a direct-to-drawable path would save.
//
// Responsiveness comes from rendering at a capped resolution WHILE interacting
// and a full one when the gesture ends — a 2000-splat scene at 960px is a few
// milliseconds on the GPU, but the CPU fallback needs the smaller size to stay
// usable on the legacy tier this must also serve.

final class SplatOrbitView: NSView {

    var cloud: SplatCloud? {
        didSet {
            if let cloud = cloud { camera = OrbitCamera.framing(cloud: cloud) }
            renderAndDisplay(interactive: false)
        }
    }

    private var camera = OrbitCamera(target: .zero, distance: 3)
    private let rasterizer = try? MetalSplatRasterizer()
    private let background = SIMD3<Float>(repeating: 0.05)
    private var cachedImage: CGImage?
    private var lastDragPoint: NSPoint?

    /// Longest render edge during a drag vs. at rest. The interactive cap keeps
    /// the CPU fallback usable; the GPU could go higher but there is no reason
    /// to when the frame is about to be replaced by the next drag event.
    private let interactiveMaxEdge = 480
    private let restMaxEdge = 960

    override var isFlipped: Bool { true }   // top-left origin, matching image coords
    override var acceptsFirstResponder: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }

    // MARK: Interaction

    override func mouseDown(with event: NSEvent) {
        lastDragPoint = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let last = lastDragPoint else { return }
        let point = convert(event.locationInWindow, from: nil)
        let dx = Double(point.x - last.x), dy = Double(point.y - last.y)
        lastDragPoint = point

        if event.modifierFlags.contains(.option) {
            // Pan: scale by a fraction of the view so a drag tracks the cursor.
            let scale = 1.0 / Double(max(bounds.height, 1))
            camera.pan(dx: -dx * scale, dy: -dy * scale)
        } else {
            // Orbit: a full view width is ~one full turn, which feels natural
            // and is resolution-independent.
            let perPixel = 2 * Double.pi / Double(max(bounds.width, 1))
            camera.orbit(deltaAzimuth: -dx * perPixel, deltaElevation: -dy * perPixel)
        }
        renderAndDisplay(interactive: true)
    }

    override func mouseUp(with event: NSEvent) {
        lastDragPoint = nil
        renderAndDisplay(interactive: false)   // sharpen once the gesture ends
    }

    override func scrollWheel(with event: NSEvent) {
        // Each notch scales distance multiplicatively, so zoom feels the same
        // near and far and can never cross zero.
        let factor = pow(1.1, -Double(event.scrollingDeltaY) / 10.0)
        camera.zoom(factor: factor)
        renderAndDisplay(interactive: true)
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .openHand)
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        renderAndDisplay(interactive: false)
    }

    // MARK: Rendering

    private func renderAndDisplay(interactive: Bool) {
        guard let cloud = cloud, cloud.count > 0, bounds.width > 1, bounds.height > 1 else {
            cachedImage = nil
            needsDisplay = true
            return
        }
        let maxEdge = interactive ? interactiveMaxEdge : restMaxEdge
        let aspect = bounds.width / bounds.height
        var renderW = Int(bounds.width), renderH = Int(bounds.height)
        if max(renderW, renderH) > maxEdge {
            if aspect >= 1 { renderW = maxEdge; renderH = max(1, Int(Double(maxEdge) / Double(aspect))) }
            else { renderH = maxEdge; renderW = max(1, Int(Double(maxEdge) * Double(aspect))) }
        }
        let intrinsics = camera.intrinsics(width: renderW, height: renderH)
        let target: RenderTarget
        if let rasterizer = rasterizer,
           let gpu = try? rasterizer.render(cloud: cloud, pose: camera.pose, intrinsics: intrinsics,
                                            width: renderW, height: renderH, background: background) {
            target = gpu
        } else {
            target = SplatRasterizer.render(cloud: cloud, pose: camera.pose, intrinsics: intrinsics,
                                            width: renderW, height: renderH, background: background)
        }
        cachedImage = SplatOrbitView.cgImage(from: target)
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        context.setFillColor(CGColor(gray: 0.05, alpha: 1))
        context.fill(bounds)
        if let image = cachedImage {
            // Fit the render into the view, preserving aspect (letterbox).
            context.interpolationQuality = .low
            context.draw(image, in: aspectFit(imageWidth: image.width, imageHeight: image.height, into: bounds))
        }
    }

    private func aspectFit(imageWidth: Int, imageHeight: Int, into rect: NSRect) -> NSRect {
        let imageAspect = CGFloat(imageWidth) / CGFloat(imageHeight)
        let rectAspect = rect.width / rect.height
        if imageAspect >= rectAspect {
            let h = rect.width / imageAspect
            return NSRect(x: rect.minX, y: rect.midY - h / 2, width: rect.width, height: h)
        } else {
            let w = rect.height * imageAspect
            return NSRect(x: rect.midX - w / 2, y: rect.minY, width: w, height: rect.height)
        }
    }

    /// RenderTarget (planar RGB float) -> CGImage. Kept here rather than in the
    /// core so SplatCore stays free of AppKit/CoreGraphics.
    static func cgImage(from target: RenderTarget) -> CGImage? {
        let width = target.width, height = target.height
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let r = target.pixels[i * 3 + 0], g = target.pixels[i * 3 + 1], b = target.pixels[i * 3 + 2]
            bytes[i * 4 + 0] = UInt8(max(0, min(255, r * 255)))
            bytes[i * 4 + 1] = UInt8(max(0, min(255, g * 255)))
            bytes[i * 4 + 2] = UInt8(max(0, min(255, b * 255)))
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

/// SwiftUI wrapper so the orbit view drops into the app's layout.
struct SplatViewerRepresentable: NSViewRepresentable {
    let cloud: SplatCloud?

    func makeNSView(context: Context) -> SplatOrbitView {
        let view = SplatOrbitView()
        view.cloud = cloud
        return view
    }

    func updateNSView(_ nsView: SplatOrbitView, context: Context) {
        // Reference-identity check would be ideal, but SplatCloud is a value
        // type; the parent only swaps it when a new scene is loaded, so a plain
        // assignment (which reframes and redraws) is correct and cheap.
        nsView.cloud = cloud
    }
}
