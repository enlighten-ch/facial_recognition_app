package com.allstar.faithid.kiosk.ui.guide

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * A collected sample: a star flies from the face to the star count in an arc (0.7 s, easing out,
 * shrinking a little), while small sparks burst around the face. [onArrive] lets the count react.
 * Draws only; it never takes touches. With animations off (reduce motion) only [onArrive] runs.
 */
class StarFlightView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val density = resources.displayMetrics.density
    private val painter = StarPainter()
    private val spark = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFFFC93C.toInt() }

    private class Flight(val x0: Float, val y0: Float, val x1: Float, val y1: Float, var t: Float = 0f)

    private val flights = mutableListOf<Flight>()
    private val running = mutableListOf<ValueAnimator>()

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
        isClickable = false
        isFocusable = false
    }

    /** Sends a star from ([fromX], [fromY]) to ([toX], [toY]), both in this view's coordinates. */
    fun fly(fromX: Float, fromY: Float, toX: Float, toY: Float, onArrive: () -> Unit) {
        if (!ValueAnimator.areAnimatorsEnabled()) {
            onArrive()
            return
        }
        val flight = Flight(fromX, fromY, toX, toY)
        flights += flight
        running += ValueAnimator.ofFloat(0f, 1f).apply {
            duration = FLIGHT_MS
            addUpdateListener {
                flight.t = it.animatedValue as Float
                invalidate()
            }
            addListener(object : android.animation.AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: android.animation.Animator) {
                    running.remove(animation)
                    flights.remove(flight)
                    invalidate()
                    onArrive()
                }
            })
            start()
        }
    }

    fun stop() {
        running.toList().forEach { it.removeAllListeners(); it.cancel() }
        running.clear()
        flights.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        flights.toList().forEach { f ->
            val k = f.t
            // Sparks: eight dots flying out from the face and fading in the first part of the flight.
            if (k < 0.85f) {
                val s = k / 0.85f
                spark.alpha = ((1 - s) * 255).toInt()
                for (i in 0 until 8) {
                    val a = i / 8f * 2 * PI
                    val d = (24 + 70 * s) * density
                    canvas.drawCircle(f.x0 + (cos(a) * d).toFloat(), f.y0 + (sin(a) * d).toFloat(), (5 * (1 - s) + 1.5f) * density, spark)
                }
            }
            // The star: eases out along the line, lifted in an arc, shrinking a little on the way.
            val e = 1 - (1 - k).pow(3)
            val x = f.x0 + (f.x1 - f.x0) * e
            val y = f.y0 + (f.y1 - f.y0) * e - sin(k * PI).toFloat() * ARC_DP * density
            painter.draw(canvas, x, y, STAR_DP * density * (1 - k * 0.4f), on = true, face = true)
        }
    }

    private companion object {
        const val FLIGHT_MS = 700L
        const val ARC_DP = 90f
        const val STAR_DP = 44f
    }
}
