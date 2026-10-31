package com.allstar.faithid.kiosk.camera

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.OvershootInterpolator
import com.allstar.faithid.core.geometry.GuideEllipse

/**
 * Darkens everything outside the oval face guide on the camera preview. During registration the
 * oval grows and shrinks ([scale]) to bring the person closer or further away.
 */
class GuideOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f * resources.displayMetrics.density
        color = 0xFFFFFFFF.toInt()
    }
    private val path = Path()
    private var animator: ValueAnimator? = null

    var axisXRatio: Float = 0.17f
        set(value) {
            field = value
            invalidate()
        }
    var axisYRatio: Float = 0.27f
        set(value) {
            field = value
            invalidate()
        }
    var centerYOffsetRatio: Float = 0.06f
        set(value) {
            field = value
            invalidate()
        }
    var outsideDimAlpha: Float = 0.35f
        set(value) {
            field = value
            invalidate()
        }

    /** Outline colour: white while waiting, warm when a face is in the oval. */
    var outlineColor: Int = 0xFFFFFFFF.toInt()
        set(value) {
            if (field == value) return
            field = value
            guidePaint.color = value
            invalidate()
        }

    /** Current size relative to the configured oval. */
    var scale: Float = 1f
        private set

    /** Called on every frame of a size change, so views placed around the oval can follow. */
    var onOvalChanged: (() -> Unit)? = null

    /** Grows or shrinks the oval to [target] with a small bounce. */
    fun animateScale(target: Float) {
        if (animator?.isRunning == true && animator?.animatedValue == target) return
        if (scale == target && animator?.isRunning != true) return
        animator?.cancel()
        animator = ValueAnimator.ofFloat(scale, target).apply {
            duration = 650
            interpolator = OvershootInterpolator(1.4f)
            addUpdateListener {
                scale = it.animatedValue as Float
                invalidate()
                onOvalChanged?.invoke()
            }
            start()
        }
    }

    /** The oval in this view's coordinates at [atScale] (default: the current size). */
    fun ovalRect(atScale: Float = scale): RectF = ovalRect(width, height, axisXRatio, axisYRatio, centerYOffsetRatio, atScale)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        val oval = ovalRect()
        val alpha = (outsideDimAlpha.coerceIn(0f, 1f) * 255).toInt()
        dimPaint.color = (alpha shl 24) or 0x33263F
        path.reset()
        path.fillType = Path.FillType.EVEN_ODD
        path.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
        path.addOval(oval, Path.Direction.CW)
        canvas.drawPath(path, dimPaint)
        canvas.drawOval(oval, guidePaint)
    }

    override fun onDetachedFromWindow() {
        animator?.cancel()
        super.onDetachedFromWindow()
    }

    companion object {
        /** Same formula as the analysis ellipse ([GuideEllipse.forFrame]), scaled around its centre. */
        fun ovalRect(width: Int, height: Int, axisX: Float, axisY: Float, offsetY: Float, scale: Float): RectF {
            val e = GuideEllipse.forFrame(width, height, axisX, axisY, offsetY)
            val rx = e.radiusX * scale
            val ry = e.radiusY * scale
            return RectF(e.centerX - rx, e.centerY - ry, e.centerX + rx, e.centerY + ry)
        }
    }
}
