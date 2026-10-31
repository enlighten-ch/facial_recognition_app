package com.allstar.faithid.kiosk.ui.guide

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** 빛이, the guide: a warm round face with two eyes and a smile. Children and adults meet the same 빛이. */
class GuideAvatarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val face = Paint(Paint.ANTI_ALIAS_FLAG)
    private val ink = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF33263F.toInt() }
    private val smile = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF33263F.toInt()
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val rect = RectF()

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val r = min(w, h) / 2f
        face.shader = RadialGradient(w / 2f - r * 0.3f, h / 2f - r * 0.4f, r * 1.4f, 0xFFFFD2B4.toInt(), 0xFFFF9A62.toInt(), Shader.TileMode.CLAMP)
    }

    override fun onDraw(canvas: Canvas) {
        val s = min(width, height).toFloat()
        val cx = width / 2f
        val cy = height / 2f
        canvas.drawCircle(cx, cy, s / 2, face)
        // Eyes and smile, in a 96-unit face.
        val u = s / 96f
        rect.set(cx - 18 * u, cy - 8 * u, cx - 8 * u, cy + 6 * u)
        canvas.drawRoundRect(rect, 5 * u, 5 * u, ink)
        rect.set(cx + 8 * u, cy - 8 * u, cx + 18 * u, cy + 6 * u)
        canvas.drawRoundRect(rect, 5 * u, 5 * u, ink)
        smile.strokeWidth = 4 * u
        rect.set(cx - 13 * u, cy + 4 * u, cx + 13 * u, cy + 20 * u)
        canvas.drawArc(rect, 20f, 140f, false, smile)
    }
}
