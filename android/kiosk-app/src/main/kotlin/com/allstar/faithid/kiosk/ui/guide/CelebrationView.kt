package com.allstar.faithid.kiosk.ui.guide

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Full-screen effects over the camera: the celebration when all registration samples are in (a big
 * star that bounces in, sparks and confetti for children; a check mark for adults), or confetti
 * alone when a child is recognized. Draws nothing when idle.
 */
class CelebrationView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private enum class Mode { NONE, CELEBRATE, CONFETTI }

    private val density = resources.displayMetrics.density
    private val painter = StarPainter()
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = 0xFFFFFFFF.toInt()
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF33263F.toInt()
        textAlign = Paint.Align.CENTER
    }
    private val pill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xE6FFFFFF.toInt() }
    private val rect = RectF()
    private val check = Path()
    private val colors = intArrayOf(0xFFFF9A62.toInt(), 0xFFFFC24B.toInt(), 0xFF7CC7A4.toInt(), 0xFFA98BE0.toInt(), 0xFF6FA8E8.toInt())

    private var mode = Mode.NONE
    private var started = 0L
    private var kids = true
    private var message = ""
    private var centerX = 0f
    private var centerY = 0f
    private var onDone: (() -> Unit)? = null

    /** Jua for the message; set by the screen. */
    var typeface: Typeface? = null
        set(value) {
            field = value
            text.typeface = value
        }

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    val isPlaying: Boolean get() = mode != Mode.NONE

    /** Plays the registration celebration around ([cx], [cy]) and calls [done] after it. */
    fun celebrate(cx: Float, cy: Float, kids: Boolean, message: String, done: () -> Unit) {
        centerX = cx
        centerY = cy
        this.kids = kids
        this.message = message
        onDone = done
        start(Mode.CELEBRATE)
    }

    fun confetti() = start(Mode.CONFETTI)

    fun stop() {
        mode = Mode.NONE
        onDone = null
        invalidate()
    }

    private fun start(next: Mode) {
        mode = next
        started = SystemClock.uptimeMillis()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        if (mode == Mode.NONE) return
        val t = (SystemClock.uptimeMillis() - started) / 1000f
        val length = if (mode == Mode.CELEBRATE) CELEBRATION_SECONDS else 1.9f
        if (t > length) {
            val done = onDone
            mode = Mode.NONE
            onDone = null
            invalidate()
            done?.invoke()
            return
        }
        if (mode == Mode.CELEBRATE) {
            val fade = if (t > length - 0.3f) (length - t) / 0.3f else min(1f, t / 0.3f)
            fill.shader = RadialGradient(centerX, centerY, width * 0.6f, 0x8CFFE9B4.toInt(), 0x5933263F, Shader.TileMode.CLAMP)
            fill.alpha = (255 * fade).toInt()
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), fill)
            fill.shader = null
            if (kids) drawStars(canvas, t) else drawCheck(canvas, t)
            drawMessage(canvas, t)
        }
        if (kids) drawConfetti(canvas, t)
        postInvalidateOnAnimation()
    }

    private fun drawStars(canvas: Canvas, t: Float) {
        // Sparks fly out from the centre.
        for (i in 0 until 14) {
            val p = min(1f, t / 1.2f)
            if (p >= 1f) break
            val a = i / 14.0 * Math.PI * 2
            val x = centerX + (cos(a) * 420 * density * p).toFloat()
            val y = centerY + (sin(a) * 300 * density * p).toFloat()
            painter.draw(canvas, x, y, 40 * density * (0.3f + 0.7f * p), on = true, face = false, alpha = ((1 - p) * 255).toInt())
        }
        // The big star bounces in.
        val p = min(1f, t / 1.1f)
        val s = if (p < 0.6f) 1.15f * (p / 0.6f) else 1.15f - 0.15f * ((p - 0.6f) / 0.4f)
        canvas.save()
        canvas.rotate((1 - min(1f, p / 0.6f)) * -180f, centerX, centerY)
        painter.draw(canvas, centerX, centerY - 20 * density, 230 * density * s, on = true, face = true)
        canvas.restore()
    }

    private fun drawCheck(canvas: Canvas, t: Float) {
        val r = 90 * density
        stroke.strokeWidth = 10 * density
        val circle = min(1f, t / 0.7f)
        rect.set(centerX - r, centerY - r - 20 * density, centerX + r, centerY + r - 20 * density)
        canvas.drawArc(rect, -90f, 360f * circle, false, stroke)
        val tick = ((t - 0.55f) / 0.45f).coerceIn(0f, 1f)
        if (tick > 0f) {
            check.reset()
            val x0 = centerX - 0.4f * r; val y0 = rect.centerY() + 0.04f * r
            val x1 = centerX - 0.14f * r; val y1 = rect.centerY() + 0.3f * r
            val x2 = centerX + 0.4f * r; val y2 = rect.centerY() - 0.28f * r
            check.moveTo(x0, y0)
            if (tick < 0.4f) check.lineTo(x0 + (x1 - x0) * tick / 0.4f, y0 + (y1 - y0) * tick / 0.4f)
            else {
                check.lineTo(x1, y1)
                val q = (tick - 0.4f) / 0.6f
                check.lineTo(x1 + (x2 - x1) * q, y1 + (y2 - y1) * q)
            }
            canvas.drawPath(check, stroke)
        }
    }

    private fun drawMessage(canvas: Canvas, t: Float) {
        val p = ((t - 0.35f) / 0.5f).coerceIn(0f, 1f)
        if (p <= 0f) return
        text.textSize = (if (kids) 44 else 32) * density
        text.alpha = (255 * p).toInt()
        pill.alpha = (230 * p).toInt()
        val y = centerY + 150 * density + (1 - p) * 30 * density
        val w = text.measureText(message) / 2 + 36 * density
        rect.set(centerX - w, y - 40 * density, centerX + w, y + 26 * density)
        canvas.drawRoundRect(rect, 40 * density, 40 * density, pill)
        canvas.drawText(message, centerX, y + 6 * density, text)
    }

    private fun drawConfetti(canvas: Canvas, t: Float) {
        val fall = height + 40 * density
        for (i in 0 until 26) {
            val delay = (i % 7) * 0.08f
            val p = ((t - delay) / 1.8f).coerceIn(0f, 1f)
            if (p <= 0f || p >= 1f) continue
            val x = width * ((i * 37) % 100) / 100f
            val y = -20 * density + fall * p * p
            fill.color = colors[i % colors.size]
            canvas.save()
            canvas.rotate(540f * p, x, y)
            rect.set(x - 7 * density, y - 11 * density, x + 7 * density, y + 11 * density)
            canvas.drawRoundRect(rect, 4 * density, 4 * density, fill)
            canvas.restore()
        }
    }

    companion object {
        const val CELEBRATION_SECONDS = 2.6f
    }
}
