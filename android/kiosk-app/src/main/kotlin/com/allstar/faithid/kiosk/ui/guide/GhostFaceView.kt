package com.allstar.faithid.kiosk.ui.guide

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.os.SystemClock
import android.provider.Settings
import android.util.AttributeSet
import android.view.Choreographer
import android.view.View
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The white face of the registration screen, drawn over the camera: a translucent round head with
 * ears and two face lines (the middle line and the eye line) that turns to the direction and grows
 * to the circle size to copy. Where the lines cross is where it looks. Around it: the hold ring,
 * which fills only once the face shows what is being collected; a dashed outline that grows or
 * shrinks from the person's face size to the target; and, if the distance is still wrong after a
 * while, shoe-sole footprints that step forward or back. The head is ray-cast into a small bitmap.
 */
class GhostFaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs), Choreographer.FrameCallback {

    private val density = resources.displayMetrics.density
    private val bw = 243
    private val bh = 198
    private val pixels = IntArray(bw * bh)
    private val bitmap = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val dst = RectF()
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.BUTT }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; color = GOLD }
    private val dashPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = 0xFFFFFFFF.toInt() }
    private val footPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFFFFFFF.toInt() }
    private val footPath = Path()
    private val oval = RectF()
    private val tmp = DoubleArray(3)
    private val reduceMotion = runCatching {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }.getOrDefault(false)

    /** The normal face circle in this view's coordinates; the face is centred on it. */
    var normalOval: RectF = RectF()
        set(value) {
            field = value
            invalidate()
        }
    var bigScale = 1.18f
    var smallScale = 0.73f

    // What to show.
    private var active = false
    private var targetYaw = 0.0
    private var targetPitch = 0.0
    private var targetScale = 1.0
    private var holding = 0f
    private var demo = false
    private var dim = false
    private var sizeHint: Float? = null
    private var steps = 0
    private var demoStart = 0L

    // What is shown, eased toward the targets.
    private var yaw = 0.0
    private var pitch = 0.0
    private var scale = 1.0
    private var ring = 0f
    private var flashAt = 0L
    private var gold = false
    private var drawnKey = ""
    private var lastFrameNanos = 0L
    private var running = false

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
        dashPaint.pathEffect = DashPathEffect(floatArrayOf(10 * density, 10 * density), 0f)
    }

    /**
     * Shows [direction] at [circle]. [holding] (0..1) fills the ring; [demo] repeats the turn from
     * the front; [sizeHint] is the person's face size relative to [circle] (null: no hint);
     * [steps] shows footprints (1 forward, -1 back, 0 none); [dim] fades the face (several people).
     */
    fun show(direction: HeadDirection, circle: FaceCircle, holding: Float, demo: Boolean, sizeHint: Float?, steps: Int, dim: Boolean) {
        if (!active) {
            active = true
            yaw = yawOf(direction)
            pitch = pitchOf(direction)
            scale = scaleOf(circle).toDouble()
        }
        targetYaw = yawOf(direction)
        targetPitch = pitchOf(direction)
        targetScale = scaleOf(circle).toDouble()
        if (demo && !this.demo) demoStart = SystemClock.uptimeMillis()
        this.holding = holding
        this.demo = demo && direction != HeadDirection.FRONT
        this.sizeHint = sizeHint
        this.steps = steps
        this.dim = dim
        visibility = VISIBLE
        start()
    }

    /** A sample was kept: the ring closes and fades. */
    fun collected() {
        flashAt = SystemClock.uptimeMillis()
        ring = 0f
    }

    fun hide() {
        active = false
        visibility = GONE
        stop()
    }

    private fun scaleOf(circle: FaceCircle) = when (circle) {
        FaceCircle.NORMAL -> 1f
        FaceCircle.BIG -> bigScale
        FaceCircle.SMALL -> smallScale
    }

    /** Turned a little further than the person needs to, so the direction is unmistakable. */
    private fun yawOf(d: HeadDirection) = when (d) {
        HeadDirection.LEFT -> -TURN
        HeadDirection.RIGHT -> TURN
        else -> 0.0
    }

    private fun pitchOf(d: HeadDirection) = when (d) {
        HeadDirection.UP -> -NOD
        HeadDirection.DOWN -> NOD
        else -> 0.0
    }

    override fun onVisibilityAggregated(isVisible: Boolean) {
        super.onVisibilityAggregated(isVisible)
        if (isVisible && active) start() else stop()
    }

    override fun onDetachedFromWindow() {
        stop()
        super.onDetachedFromWindow()
    }

    private fun start() {
        if (running) return
        running = true
        lastFrameNanos = 0L
        Choreographer.getInstance().postFrameCallback(this)
    }

    private fun stop() {
        if (!running) return
        running = false
        Choreographer.getInstance().removeFrameCallback(this)
    }

    override fun doFrame(frameTimeNanos: Long) {
        if (!running) return
        val dt = if (lastFrameNanos == 0L) 0.0 else min(0.1, (frameTimeNanos - lastFrameNanos) / 1e9)
        lastFrameNanos = frameTimeNanos
        var wantYaw = targetYaw
        var wantPitch = targetPitch
        if (demo && !reduceMotion) {
            // Turns from the front to the target again and again: "like this".
            val ph = ((SystemClock.uptimeMillis() - demoStart) % DEMO_MS) / DEMO_MS.toDouble()
            val e = when {
                ph < 0.15 -> 0.0
                ph < 0.65 -> 0.5 - 0.5 * cos((ph - 0.15) / 0.5 * PI)
                else -> 1.0
            }
            wantYaw *= e
            wantPitch *= e
        }
        val quick = demo || holding > 0f
        val k = if (reduceMotion) 1.0 else 1 - exp(-dt * if (quick) 12.0 else 5.0)
        yaw += (wantYaw - yaw) * k
        pitch += (wantPitch - pitch) * k
        scale += (targetScale - scale) * (if (reduceMotion) 1.0 else 1 - exp(-dt * if (quick) 12.0 else 4.0))
        // The ring fills only once the face shows the target.
        val matched = abs(yaw - targetYaw) < 0.12 && abs(pitch - targetPitch) < 0.12 && abs(scale / targetScale - 1) < 0.06
        val wantRing = if (matched) holding else 0f
        ring += (wantRing - ring) * (1 - exp(-dt * 10)).toFloat()
        gold = matched && holding > 0f
        val key = "%.3f,%.3f,%b,%b".format(yaw, pitch, gold, dim)
        if (key != drawnKey) {
            render()
            drawnKey = key
        }
        invalidate()
        Choreographer.getInstance().postFrameCallback(this)
    }

    /** Ray-casts the head and ears into [bitmap]; x spans ±1.35 head radii, y ±1.1. */
    private fun render() {
        val ca = cos(yaw)
        val sa = sin(yaw)
        val cb = cos(pitch)
        val sb = sin(pitch)
        // Camera axes in head-local space (the inverse turn), for rays toward -z.
        val ex = unrot(DoubleArray(3), 1.0, 0.0, 0.0, ca, sa, cb, sb)
        val ey = unrot(DoubleArray(3), 0.0, 1.0, 0.0, ca, sa, cb, sb)
        val d = unrot(DoubleArray(3), 0.0, 0.0, -1.0, ca, sa, cb, sb)
        val unit = 2.7 / bw
        val (cr, cg, cbl) = if (gold) Triple(255, 214, 102) else Triple(255, 255, 255)
        val alphaScale = if (dim) 0.35 else 1.0
        for (j in 0 until bh) {
            val v = (j + 0.5) * unit - 1.1
            for (i in 0 until bw) {
                val u = (i + 0.5) * unit - 1.35
                val idx = j * bw + i
                if (u * u + v * v > 1.9) {
                    pixels[idx] = 0
                    continue
                }
                val ox = u * ex[0] + v * ey[0]
                val oy = u * ex[1] + v * ey[1]
                val oz = u * ex[2] + v * ey[2]
                var bestT = Double.MAX_VALUE
                var chord = 0.0
                var part = -1
                for (p in PARTS.indices) {
                    val c = PARTS[p]
                    val px = (ox - c[0]) / c[3]
                    val py = (oy - c[1]) / c[4]
                    val pz = (oz - c[2]) / c[5]
                    val dx = d[0] / c[3]
                    val dy = d[1] / c[4]
                    val dz = d[2] / c[5]
                    val a = dx * dx + dy * dy + dz * dz
                    val b = 2 * (px * dx + py * dy + pz * dz)
                    val cc = px * px + py * py + pz * pz - 1
                    val disc = b * b - 4 * a * cc
                    if (disc <= 0) continue
                    val sq = sqrt(disc)
                    val t = (-b - sq) / (2 * a)
                    if (t < bestT) {
                        bestT = t
                        chord = sq / a
                        part = p
                    }
                }
                if (part < 0) {
                    pixels[idx] = 0
                    continue
                }
                val c = PARTS[part]
                val hx = ox + bestT * d[0]
                val hy = oy + bestT * d[1]
                val hz = oz + bestT * d[2]
                // Normal in head space, turned back to the camera's.
                val nx0 = (hx - c[0]) / (c[3] * c[3])
                val ny0 = (hy - c[1]) / (c[4] * c[4])
                val nz0 = (hz - c[2]) / (c[5] * c[5])
                val n = rot(tmp, nx0, ny0, nz0, ca, sa, cb, sb)
                val nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                val facing = max(0.0, n[2] / nn)
                val lit = max(0.0, (n[0] * LX + n[1] * LY + n[2] * LZ) / nn)
                var o = 0.14 + 0.2 * lit + 0.42 * (1 - facing).pow(3)
                o += if (part == 0) 0.3 * max(0.0, hz).pow(2.4) else 0.08
                val edge = min(1.0, chord / (unit * 3))
                val alpha = (255 * min(1.0, o * alphaScale) * edge).toInt().coerceIn(0, 255)
                pixels[idx] = (alpha shl 24) or (cr shl 16) or (cg shl 8) or cbl
            }
        }
        bitmap.setPixels(pixels, 0, bw, 0, 0, bw, bh)
    }

    override fun onDraw(canvas: Canvas) {
        if (!active || normalOval.isEmpty) return
        val cx = normalOval.centerX()
        val cy = normalOval.centerY()
        // The head is as wide as a face that fits the circle: a little narrower than the circle.
        val r = (normalOval.width() / 2f * FACE_FILL * scale).toFloat()
        val ry = r * 1.08f
        dst.set(cx - r * 1.35f, cy - r * 1.1f, cx + r * 1.35f, cy + r * 1.1f)
        canvas.drawBitmap(bitmap, null, dst, bitmapPaint)
        drawLines(canvas, cx, cy, r, ry)
        val now = SystemClock.uptimeMillis()
        if (!dim) {
            sizeHint?.let { hint -> if (abs(hint - 1f) > 0.12f && !reduceMotion) drawSizePulse(canvas, cx, cy, r, ry, hint, now) }
            if (steps != 0) drawSteps(canvas, now, steps > 0)
            val pad = 14 * density
            oval.set(cx - r - pad, cy - ry - pad, cx + r + pad, cy + ry + pad)
            ringPaint.strokeWidth = 9 * density
            if (ring > 0.01f) {
                ringPaint.alpha = 255
                canvas.drawArc(oval, -90f, 360f * ring.coerceAtMost(1f), false, ringPaint)
            }
            val since = now - flashAt
            if (flashAt > 0 && since < FLASH_MS) {
                ringPaint.alpha = (255 * (1 - since / FLASH_MS.toFloat())).toInt()
                canvas.drawArc(oval, -90f, 360f, false, ringPaint)
            }
        }
    }

    /** The middle line and the eye line, on the front of the head, fading toward its edge. */
    private fun drawLines(canvas: Canvas, cx: Float, cy: Float, r: Float, ry: Float) {
        val ca = cos(yaw)
        val sa = sin(yaw)
        val cb = cos(pitch)
        val sb = sin(pitch)
        linePaint.strokeWidth = 4.5f * density
        val base = if (gold) GOLD and 0xFFFFFF else 0xFFFFFF
        for (line in 0..1) {
            var px = 0f
            var py = 0f
            for (i in 0..SEGMENTS) {
                val t = -1.15 + 2.3 * i / SEGMENTS
                val p = if (line == 0) rot(tmp, 0.0, sin(t), cos(t), ca, sa, cb, sb)
                else rot(tmp, sin(t) * 0.995, -0.1, cos(t) * 0.995, ca, sa, cb, sb)
                val x = cx + (p[0] * r).toFloat()
                val y = cy + (p[1] * ry).toFloat()
                val fade = (min(1.0, max(0.0, cos((i.toDouble() / SEGMENTS - 0.5) * PI) - 0.05) * 1.3) * min(1.0, max(0.0, p[2] * 2.5))).toFloat()
                if (i > 0 && fade > 0.02f) {
                    val a = (0.9f * fade * (if (dim) 0.35f else 1f) * 255).toInt()
                    linePaint.color = (a shl 24) or base
                    canvas.drawLine(px, py, x, y, linePaint)
                }
                px = x
                py = y
            }
        }
    }

    /** A dashed outline growing (or shrinking) from the person's face size to the target, again and again. */
    private fun drawSizePulse(canvas: Canvas, cx: Float, cy: Float, r: Float, ry: Float, hint: Float, now: Long) {
        val ph = (now % PULSE_MS) / PULSE_MS.toFloat()
        val e = if (ph < 0.6f) (0.5f - 0.5f * cos(ph / 0.6f * PI.toFloat())) else 1f
        val s = hint + (1 - hint) * e
        dashPaint.strokeWidth = 3 * density
        dashPaint.alpha = (255 * 0.55f * (1 - e * 0.6f)).toInt()
        oval.set(cx - r * s, cy - ry * s, cx + r * s, cy + ry * s)
        canvas.drawOval(oval, dashPaint)
    }

    /** Three shoe-sole prints, toes toward the viewer: coming closer (growing) or stepping back. */
    private fun drawSteps(canvas: Canvas, now: Long, closer: Boolean) {
        val t = (now % STEPS_MS) / STEPS_MS.toFloat()
        val baseY = max(normalOval.bottom + 40 * density, height - 150 * density)
        for (i in 0..2) {
            val life = t - i * 0.22f
            if (life < 0) continue
            val k = if (closer) i else 2 - i
            val alpha = min(1f, life * 6) * max(0f, 1 - max(0f, t - 0.75f) * 4) * 0.85f
            val s = density * (1.05f + k * 0.25f)
            val dx = if (i % 2 == 0) -26f else 26f
            sole(canvas, width / 2f + dx * s, baseY + k * 42 * density, s, dx > 0, alpha)
        }
    }

    private fun sole(canvas: Canvas, x: Float, y: Float, s: Float, flip: Boolean, alpha: Float) {
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(if (flip) -s else s, s)
        footPaint.alpha = (alpha * 255).toInt()
        footPaint.setShadowLayer(8f, 0f, 0f, 0x66281C30)
        footPath.reset()
        footPath.moveTo(-8f, -4f)
        footPath.cubicTo(-13f, 6f, -16f, 18f, -14f, 26f)
        footPath.cubicTo(-12f, 38f, 12f, 40f, 14f, 27f)
        footPath.cubicTo(16f, 16f, 12f, 5f, 8f, -4f)
        footPath.close()
        canvas.drawPath(footPath, footPaint)
        canvas.drawRoundRect(-9f, -30f, 9f, -13f, 6f, 6f, footPaint)
        canvas.restore()
    }

    private companion object {
        const val GOLD = 0xFFFFC93C.toInt()
        const val TURN = 0.74
        const val NOD = 0.67
        const val FACE_FILL = 0.9f
        const val SEGMENTS = 48
        const val DEMO_MS = 2400L
        const val PULSE_MS = 1600L
        const val STEPS_MS = 2400L
        const val FLASH_MS = 350L

        // Light from the upper left, toward the viewer.
        val LX = -0.45 / 1.0062
        val LY = -0.55 / 1.0062
        val LZ = 0.7 / 1.0062

        /** Head and ears: centre x, y, z and radii x, y, z, in head radii (x right, y down, z toward the camera). */
        val PARTS = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.08, 1.0),
            doubleArrayOf(1.0, 0.04, -0.04, 0.19, 0.25, 0.13),
            doubleArrayOf(-1.0, 0.04, -0.04, 0.19, 0.25, 0.13),
        )

        /** Turns a head-local point by yaw, then pitch, into the camera's view; writes into [out]. */
        fun rot(out: DoubleArray, x: Double, y: Double, z: Double, ca: Double, sa: Double, cb: Double, sb: Double): DoubleArray {
            val z0 = -x * sa + z * ca
            out[0] = x * ca + z * sa
            out[1] = y * cb + z0 * sb
            out[2] = -y * sb + z0 * cb
            return out
        }

        fun unrot(out: DoubleArray, x: Double, y: Double, z: Double, ca: Double, sa: Double, cb: Double, sb: Double): DoubleArray {
            val z0 = y * sb + z * cb
            out[0] = x * ca - z0 * sa
            out[1] = y * cb - z * sb
            out[2] = x * sa + z0 * ca
            return out
        }
    }
}
