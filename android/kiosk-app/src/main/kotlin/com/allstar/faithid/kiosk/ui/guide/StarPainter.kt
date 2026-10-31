package com.allstar.faithid.kiosk.ui.guide

import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import kotlin.math.cos
import kotlin.math.sin

/**
 * A chubby star: a fat inner radius and round joins, with a friendly face once collected (dark,
 * slightly high eyes with highlights, a small smile and cheeks). Drawn in a 24-unit box.
 */
internal class StarPainter {
    private val star = Path().apply {
        for (i in 0 until 10) {
            val a = -Math.PI / 2 + i * Math.PI / 5
            val r = if (i % 2 == 1) 5.6 else 10.0
            val x = (12 + r * cos(a)).toFloat()
            val y = (12.6 + r * sin(a)).toFloat()
            if (i == 0) moveTo(x, y) else lineTo(x, y)
        }
        close()
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }
    private val dashed = DashPathEffect(floatArrayOf(3f, 3.2f), 0f)
    private val oval = RectF()
    private val smile = Path().apply {
        moveTo(11.2f, 13.25f)
        quadTo(12f, 14f, 12.8f, 13.25f)
    }

    /** Draws a star of [size] px centred at ([cx], [cy]); [on] = collected. */
    fun draw(canvas: Canvas, cx: Float, cy: Float, size: Float, on: Boolean, face: Boolean, alpha: Int = 255) {
        canvas.save()
        canvas.translate(cx - size / 2, cy - size / 2)
        canvas.scale(size / 24f, size / 24f)
        if (on) {
            fill.color = GOLD
            fill.alpha = alpha
            stroke.color = GOLD_EDGE
            stroke.alpha = alpha
            stroke.strokeWidth = 3.4f
            stroke.pathEffect = null
        } else {
            fill.color = 0x52FFFFFF
            stroke.color = 0xD9FFFFFF.toInt()
            stroke.strokeWidth = 2.6f
            stroke.pathEffect = dashed
        }
        canvas.drawPath(star, fill)
        canvas.drawPath(star, stroke)
        if (on && face) drawFace(canvas, alpha)
        canvas.restore()
    }

    private fun drawFace(canvas: Canvas, alpha: Int) {
        fill.color = 0xB3FFFFFF.toInt()
        canvas.save()
        canvas.rotate(-30f, 8.4f, 8.9f)
        ellipse(canvas, 8.4f, 8.9f, 1.5f, 0.9f)
        canvas.restore()
        fill.color = 0x8CFF7A8A.toInt()
        ellipse(canvas, 8.5f, 13.2f, 1.15f, 0.65f)
        ellipse(canvas, 15.5f, 13.2f, 1.15f, 0.65f)
        fill.color = EYE
        fill.alpha = alpha
        ellipse(canvas, 10.1f, 11.5f, 1.05f, 1.25f)
        ellipse(canvas, 13.9f, 11.5f, 1.05f, 1.25f)
        fill.color = 0xFFFFFFFF.toInt()
        canvas.drawCircle(10.45f, 11.05f, 0.42f, fill)
        canvas.drawCircle(14.25f, 11.05f, 0.42f, fill)
        stroke.color = EYE
        stroke.strokeWidth = 0.62f
        stroke.pathEffect = null
        canvas.drawPath(smile, stroke)
    }

    private fun ellipse(canvas: Canvas, cx: Float, cy: Float, rx: Float, ry: Float) {
        oval.set(cx - rx, cy - ry, cx + rx, cy + ry)
        canvas.drawOval(oval, fill)
    }

    companion object {
        const val GOLD = 0xFFFFC93C.toInt()
        const val GOLD_EDGE = 0xFFF4A81D.toInt()
        const val EYE = 0xFF1D120C.toInt()
    }
}
