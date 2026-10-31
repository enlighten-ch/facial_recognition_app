package com.allstar.faithid.kiosk.recognition

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import com.allstar.faithid.core.geometry.GuideEllipse
import com.allstar.faithid.core.recognition.HeadPose
import com.allstar.faithid.kiosk.flow.Observation
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.google.mlkit.vision.face.Face

/**
 * Turns camera frames into [Observation]s for the flow. Runs on the camera thread.
 *
 * Only every n-th frame is analyzed (n from the settings), and frames are skipped entirely on
 * screens that do not use the camera, so the tablet does not spend its battery on unused work.
 */
class FrameAnalyzer(
    private val engine: FaceEngine,
    private val settings: () -> KioskSettings,
    private val screen: () -> Screen,
    private val mirrored: () -> Boolean,
    /** Size of the face circle on screen relative to the configured one (registration steps). */
    private val guideScale: () -> Float,
    private val deliver: (Observation) -> Unit,
    private val log: (String) -> Unit,
) {
    private var tick = 0L

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
            val scale = if (current == Screen.REGISTER_CAPTURE) guideScale().coerceIn(0.4f, 1.6f) else 1f
            val ellipse = GuideEllipse.forFrame(frame.width, frame.height, s.registerEllipseAxisXRatio * scale, s.registerEllipseAxisYRatio * scale, s.registerEllipseCenterYOffsetRatio)
            val face = runCatching { engine.detect(frame) }
                .onFailure { log("얼굴 검출 실패: ${it.javaClass.simpleName}") }
                .getOrNull()
                ?.filter { ellipse.contains(it.boundingBox.exactCenterX(), it.boundingBox.exactCenterY()) }
                ?.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
            if (face == null) {
                deliver(Observation.NoFace)
                return
            }
            lastFaceAt = System.currentTimeMillis()
            val masked = maskOutsideEllipse(frame, ellipse)
            val embedding = try {
                engine.embed(face, masked, s.faceAlignmentEnabled)
            } finally {
                masked.recycle()
            } ?: return
            deliver(Observation.Face(embedding, direction(face), face.boundingBox.width().toFloat() / frame.width))
        } finally {
            frame.recycle()
        }
    }

    /** True if no face has been in front of the camera for [seconds]; used before applying server changes. */
    fun idleFor(seconds: Int): Boolean = System.currentTimeMillis() - lastFaceAt >= seconds.coerceAtLeast(1) * 1000L

    private fun direction(face: Face) = FaceEngine.landmarksOf(face)?.let { HeadPose.classify(it, mirrored()) }

    /** Blacks out everything outside the guide oval so people behind the user do not leak into the embedding. */
    private fun maskOutsideEllipse(frame: Bitmap, ellipse: GuideEllipse): Bitmap {
        val copy = frame.copy(Bitmap.Config.ARGB_8888, true)
        val oval = RectF(ellipse.centerX - ellipse.radiusX, ellipse.centerY - ellipse.radiusY, ellipse.centerX + ellipse.radiusX, ellipse.centerY + ellipse.radiusY)
        val outside = Path().apply {
            fillType = Path.FillType.EVEN_ODD
            addRect(0f, 0f, copy.width.toFloat(), copy.height.toFloat(), Path.Direction.CW)
            addOval(oval, Path.Direction.CCW)
        }
        Canvas(copy).drawPath(outside, Paint().apply { color = Color.BLACK; style = Paint.Style.FILL; isAntiAlias = true })
        return copy
    }
}
