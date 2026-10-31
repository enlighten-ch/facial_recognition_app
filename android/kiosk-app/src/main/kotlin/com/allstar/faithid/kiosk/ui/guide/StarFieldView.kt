package com.allstar.faithid.kiosk.ui.guide

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * The registration stars: three per direction, on the side of the face circle the person is asked
 * to look at. Star sizes follow the circles (normal, big, small). A collected star pops; the next
 * one breathes and its group glows. Adults see plain dots instead of stars.
 */
class StarFieldView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val density = resources.displayMetrics.density
    private val painter = StarPainter()
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG)
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG)
    private val popStart = LongArray(12)

    private var counts = IntArray(4)
    private var perDirection = 3
    private var next = -1
    private var kids = true

    /** The biggest face circle in this view's coordinates; the stars sit just outside it. */
    var circle: RectF = RectF()
        set(value) {
            field = value
            invalidate()
        }

    /** [collected] per direction (LEFT, RIGHT, UP, DOWN); [nextDirection] = index or -1 when done. */
    fun show(collected: IntArray, perDirection: Int, nextDirection: Int, kids: Boolean) {
        val now = SystemClock.uptimeMillis()
        for (d in 0 until 4) for (k in 0 until 3) {
            if (collected[d] > k && counts[d] <= k && visibility == VISIBLE) popStart[d * 3 + k] = now
        }
        counts = collected.copyOf()
        this.perDirection = perDirection
        next = nextDirection
        this.kids = kids
        invalidate()
    }

    fun reset() {
        counts = IntArray(4)
        popStart.fill(0)
        next = -1
        invalidate()
    }

    /** Star centre for direction [d] (0..3) and step [k] (0 normal, 1 big, 2 small). */
    private fun position(d: Int, k: Int, out: FloatArray) {
        // Clear of the biggest circle by the largest star's radius plus a margin.
        val gap = (sizeFor(1) / 2) + 18 * density
        val o = k - 1
        val a = abs(o)
        val cx = circle.centerX()
        val cy = circle.centerY()
        when (d) {
            0 -> { out[0] = circle.left - gap + a * 12 * density; out[1] = cy + o * 64 * density }
            1 -> { out[0] = circle.right + gap - a * 12 * density; out[1] = cy + o * 64 * density }
            2 -> { out[0] = cx + o * 72 * density; out[1] = circle.top - gap + a * 12 * density }
            else -> { out[0] = cx + o * 72 * density; out[1] = circle.bottom + gap - a * 12 * density }
        }
    }

    private fun sizeFor(k: Int): Float = (if (kids) floatArrayOf(54f, 64f, 44f) else floatArrayOf(30f, 36f, 24f))[k] * density

    override fun onDraw(canvas: Canvas) {
        if (circle.isEmpty) return
        val now = SystemClock.uptimeMillis()
        val p = FloatArray(2)
        var animating = false
        if (next in 0..3) {
            // Soft glow behind the group the person should look at.
            val c = FloatArray(2)
            position(next, 1, c)
            val r = 120 * density
            val pulse = 0.75f + 0.25f * sin(now / 260.0).toFloat()
            glow.shader = RadialGradient(c[0], c[1], r, (((if (kids) 0x8C else 0x66) * pulse).toInt() shl 24) or (if (kids) 0xFFD66E else 0xFFFFFF), 0, Shader.TileMode.CLAMP)
            canvas.drawCircle(c[0], c[1], r, glow)
            animating = true
        }
        for (d in 0 until 4) for (k in 0 until 3) {
            position(d, k, p)
            val on = counts[d] > k
            var size = sizeFor(k)
            val since = now - popStart[d * 3 + k]
            if (on && popStart[d * 3 + k] > 0 && since < 600) {
                // Pop: from small to a little too big and back.
                val t = since / 600f
                size *= if (t < 0.7f) 0.2f + 1.1f * (t / 0.7f) else 1.3f - 0.3f * ((t - 0.7f) / 0.3f)
                ring.color = 0xFFFFE08A.toInt()
                ring.alpha = ((1 - t) * 255).toInt()
                ring.strokeWidth = 4 * density
                canvas.drawCircle(p[0], p[1], sizeFor(k) * (0.6f + 0.6f * t), ring)
                animating = true
            } else if (!on && d == next && k == counts[d]) {
                size *= 1f + 0.12f * ((sin(now / 190.0) + 1) / 2).toFloat()
                animating = true
            }
            if (kids) {
                painter.draw(canvas, p[0], p[1], size, on, face = true)
            } else {
                dot.style = Paint.Style.FILL
                dot.color = if (on) 0xFFFFFFFF.toInt() else 0x40FFFFFF
                canvas.drawCircle(p[0], p[1], size / 2, dot)
                ring.color = if (on) 0x8C5B3F8C.toInt() else 0xD9FFFFFF.toInt()
                ring.alpha = 255
                ring.strokeWidth = (if (on) 5 else 3) * density
                canvas.drawCircle(p[0], p[1], size / 2 + if (on) 3 * density else 0f, ring)
            }
        }
        if (animating && isShown) postInvalidateOnAnimation()
    }

    /** The area the stars use, for keeping other views clear of them. */
    fun bounds(): RectF {
        val r = RectF(circle)
        val reach = sizeFor(1) + 18 * density
        r.inset(-reach, -reach)
        return RectF(max(0f, r.left), max(0f, r.top), min(width.toFloat(), r.right), min(height.toFloat(), r.bottom))
    }
}
