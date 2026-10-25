package com.allstar.faithid.core.geometry

data class Point(val x: Float, val y: Float)

data class Rect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
    val area: Float get() = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
}

/**
 * The oval face guide drawn on the camera preview. Ratios are relative to the frame size, so the
 * same settings work for the preview view and the analysis bitmap.
 */
data class GuideEllipse(
    val centerX: Float,
    val centerY: Float,
    val radiusX: Float,
    val radiusY: Float,
) {
    fun contains(x: Float, y: Float): Boolean {
        if (radiusX <= 0f || radiusY <= 0f) return false
        val dx = (x - centerX) / radiusX
        val dy = (y - centerY) / radiusY
        return dx * dx + dy * dy <= 1f
    }

    companion object {
        /** Same formula and minimum radii (40 px, 60 px) as the original kiosk code. */
        fun forFrame(width: Int, height: Int, axisXRatio: Float, axisYRatio: Float, centerYOffsetRatio: Float): GuideEllipse {
            val w = width.coerceAtLeast(1)
            val h = height.coerceAtLeast(1)
            return GuideEllipse(
                centerX = (w / 2).toFloat(),
                centerY = (h * (0.5f + centerYOffsetRatio)).toInt().toFloat(),
                radiusX = (w * axisXRatio).toInt().coerceAtLeast(40).toFloat(),
                radiusY = (h * axisYRatio).toInt().coerceAtLeast(60).toFloat(),
            )
        }
    }
}
