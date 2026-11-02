package com.allstar.faithid.kiosk.recognition

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import com.allstar.faithid.core.geometry.GuideEllipse
import com.allstar.faithid.core.recognition.HeadPose
import com.allstar.faithid.core.recognition.Pose
import com.allstar.faithid.core.recognition.PoseLimits
import com.allstar.faithid.kiosk.flow.Observation
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.google.mlkit.vision.face.Face

/**
 * Turns camera frames into [Observation]s for the flow. Runs on the camera thread.
 *
 * Only every n-th frame is analyzed (n from the settings), and frames are skipped entirely on
 * screens that do not use the camera, so the tablet does not spend its battery on unused work.
 *
 * While registering it also reports several faces and a face outside the circle, measures the face
 * against the normal circle as drawn, and rates the frame's sharpness.
 */
class FrameAnalyzer(
    private val engine: FaceEngine,
    private val settings: () -> KioskSettings,
    private val screen: () -> Screen,
    private val mirrored: () -> Boolean,
    /** Size of the face circle on screen relative to the configured one (registration step and fit). */
    private val guideScale: () -> Float,
    /** Size of the normal registration circle on screen relative to the configured one (the fit). */
    private val normalScale: () -> Float,
    private val deliver: (Observation) -> Unit,
    private val log: (String) -> Unit,
) {
    private var tick = 0L
    private val poseLimits = PoseLimits()

    @Volatile
    var lastFaceAt: Long = 0L
        private set

    fun onFrame(frame: Bitmap) {
        try {
            val current = screen()
            if (current != Screen.MONITOR && current != Screen.REGISTER_CAPTURE) return
            val s = settings()
            val stride = (if (current == Screen.MONITOR) s.analyzeEveryNFrames else s.registerCaptureEveryNFrames).coerceAtLeast(1)
            if (tick++ % stride != 0L) return
            if (!engine.isReady) return
            val registering = current == Screen.REGISTER_CAPTURE
            val scale = if (registering) guideScale().coerceIn(0.4f, 1.6f) else 1f
            val ellipse = GuideEllipse.forFrame(frame.width, frame.height, s.registerEllipseAxisXRatio * scale, s.registerEllipseAxisYRatio * scale, s.registerEllipseCenterYOffsetRatio)
            val faces = runCatching { engine.detect(frame) }
                .onFailure { log("얼굴 검출 실패: ${it.javaClass.simpleName}") }
                .getOrNull()
                .orEmpty()
            val face = faces
                .filter { ellipse.contains(it.boundingBox.exactCenterX(), it.boundingBox.exactCenterY()) }
                .maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
            if (registering) {
                // People of about the same size count; a small face far behind does not.
                val largest = faces.maxOfOrNull { it.boundingBox.width() } ?: 0
                val count = faces.count { it.boundingBox.width() >= largest * SEVERAL_MIN_RATIO }
                if (count >= 2) {
                    deliver(Observation.Several(count))
                    return
                }
                val visible = faces.maxByOrNull { it.boundingBox.width() }
                if (face == null && visible != null) {
                    val dx = (visible.boundingBox.exactCenterX() - ellipse.centerX) / ellipse.radiusX
                    deliver(Observation.Outside(if (mirrored()) -dx else dx))
                    return
                }
            }
            if (face == null) {
                deliver(Observation.NoFace)
                return
            }
            lastFaceAt = System.currentTimeMillis()
            val masked = maskAround(frame, face, faces)
            val embedding = try {
                engine.embed(face, masked, s.faceAlignmentEnabled)
            } finally {
                masked.recycle()
            } ?: return
            val angles = FaceEngine.landmarksOf(face)?.let { HeadPose.measure(it, mirrored()) }
            val pose = angles?.let { poseLimits.classify(it) }
            deliver(
                Observation.Face(
                    embedding = embedding,
                    direction = (pose as? Pose.Facing)?.direction,
                    size = if (registering) sizeAgainstNormalCircle(face, frame, s) else null,
                    pose = pose,
                    angles = angles,
                    sharpness = if (registering) sharpness(frame, face.boundingBox) else 0f,
                ),
            )
        } finally {
            frame.recycle()
        }
    }

    /** True if no face has been in front of the camera for [seconds]; used before applying server changes. */
    fun idleFor(seconds: Int): Boolean = System.currentTimeMillis() - lastFaceAt >= seconds.coerceAtLeast(1) * 1000L

    /**
     * The face height relative to the height of a face that fits the normal circle as drawn
     * (1 = fits it). A fitting face is as wide as the circle, so its height is measured against the
     * circle's width ([KioskSettings.registerFaceFill]); the circle's height depends on the screen's
     * shape. Height changes less than width when the head turns left or right.
     */
    private fun sizeAgainstNormalCircle(face: Face, frame: Bitmap, s: KioskSettings): Float? {
        val circleWidth = 2f * frame.width * s.registerEllipseAxisXRatio * normalScale().coerceIn(0.4f, 1.6f)
        val fitted = circleWidth * s.registerFaceFill
        return if (fitted > 0f) face.boundingBox.height() / fitted else null
    }

    /**
     * Keeps an oval a little larger than the chosen face and blacks out the rest, including any
     * other face, so people nearby do not leak into the embedding.
     */
    private fun maskAround(frame: Bitmap, face: Face, all: List<Face>): Bitmap {
        val copy = frame.copy(Bitmap.Config.ARGB_8888, true)
        val box = face.boundingBox
        val rx = box.width() * KEEP_SCALE / 2f
        val ry = box.height() * KEEP_SCALE / 2f
        val keep = RectF(box.exactCenterX() - rx, box.exactCenterY() - ry, box.exactCenterX() + rx, box.exactCenterY() + ry)
        val outside = Path().apply {
            fillType = Path.FillType.EVEN_ODD
            addRect(0f, 0f, copy.width.toFloat(), copy.height.toFloat(), Path.Direction.CW)
            addOval(keep, Path.Direction.CCW)
        }
        val black = Paint().apply { color = Color.BLACK; style = Paint.Style.FILL; isAntiAlias = true }
        val canvas = Canvas(copy)
        canvas.drawPath(outside, black)
        all.filter { it !== face }.forEach { canvas.drawRect(RectF(it.boundingBox), black) }
        return copy
    }

    /** Variance of a Laplacian over the face, shrunk to 48×48 grey: higher is sharper. */
    private fun sharpness(frame: Bitmap, box: Rect): Float {
        val left = box.left.coerceIn(0, frame.width - 1)
        val top = box.top.coerceIn(0, frame.height - 1)
        val w = (box.right.coerceAtMost(frame.width) - left).coerceAtLeast(1)
        val h = (box.bottom.coerceAtMost(frame.height) - top).coerceAtLeast(1)
        val crop = Bitmap.createBitmap(frame, left, top, w, h)
        val small = Bitmap.createScaledBitmap(crop, SHARP_SIZE, SHARP_SIZE, true)
        if (small !== crop) crop.recycle()
        val px = IntArray(SHARP_SIZE * SHARP_SIZE)
        small.getPixels(px, 0, SHARP_SIZE, 0, 0, SHARP_SIZE, SHARP_SIZE)
        small.recycle()
        val grey = FloatArray(px.size) { i -> val p = px[i]; 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF) }
        var sum = 0.0
        var sumSq = 0.0
        var n = 0
        for (y in 1 until SHARP_SIZE - 1) for (x in 1 until SHARP_SIZE - 1) {
            val i = y * SHARP_SIZE + x
            val lap = grey[i - 1] + grey[i + 1] + grey[i - SHARP_SIZE] + grey[i + SHARP_SIZE] - 4 * grey[i]
            sum += lap
            sumSq += lap * lap
            n++
        }
        val mean = sum / n
        return (sumSq / n - mean * mean).toFloat()
    }

    private companion object {
        /** Faces at least this wide relative to the widest one count as another person in front of the kiosk. */
        const val SEVERAL_MIN_RATIO = 0.4f

        /** The kept oval around the chosen face, relative to its detection box. */
        const val KEEP_SCALE = 1.4f
        const val SHARP_SIZE = 48
    }
}
