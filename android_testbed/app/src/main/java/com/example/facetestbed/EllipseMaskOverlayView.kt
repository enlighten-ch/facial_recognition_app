package com.example.facetestbed

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class EllipseMaskOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = 0xFFFFFFFF.toInt()
    }
    private val path = Path()

    var axisXRatio: Float = 0.22f
        set(value) {
            field = value
            invalidate()
        }
    var axisYRatio: Float = 0.30f
        set(value) {
            field = value
            invalidate()
        }
    var centerYOffsetRatio: Float = 0.0f
        set(value) {
            field = value
            invalidate()
        }
    var outsideDimAlpha: Float = 0.60f
        set(value) {
            field = value
            invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val cx = width / 2f
        val cy = (height * (0.5f + centerYOffsetRatio))
        val rx = max(40f, width * axisXRatio)
        val ry = max(60f, height * axisYRatio)

        val alpha = (outsideDimAlpha.coerceIn(0f, 1f) * 255).toInt()
        dimPaint.color = (alpha shl 24)

        path.reset()
        path.fillType = Path.FillType.EVEN_ODD
        path.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
        path.addOval(cx - rx, cy - ry, cx + rx, cy + ry, Path.Direction.CW)
        canvas.drawPath(path, dimPaint)
        canvas.drawOval(cx - rx, cy - ry, cx + rx, cy + ry, guidePaint)
    }
}
