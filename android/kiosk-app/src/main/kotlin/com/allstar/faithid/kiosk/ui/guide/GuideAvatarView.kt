package com.allstar.faithid.kiosk.ui.guide

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.AnimatorSet
import android.animation.Keyframe
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.View
import kotlin.math.min
import kotlin.random.Random

/**
 * 빛이, the guide: a warm round face with two eyes and a smile. It blinks on its own every few
 * seconds and plays [Gesture]s on request, one after another (see [GuideGestures]). Children and
 * adults meet the same 빛이.
 */
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
    private val density = resources.displayMetrics.density
    private val main = Handler(Looper.getMainLooper())

    /** 1 = eyes open, 0 = closed. */
    private var eyeOpen = 1f
    /** -1 looks left, 1 looks right. */
    private var eyeShift = 0f

    private val queue = ArrayDeque<Gesture>()
    private var playing: Animator? = null

    private val blinkLoop = object : Runnable {
        override fun run() {
            if (isShown && playing == null) play(Gesture.BLINK)
            main.postDelayed(this, Random.nextLong(GuideGestures.BLINK_MIN_MS, GuideGestures.BLINK_MAX_MS))
        }
    }

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    /** The last gesture started, for tests and logs. */
    var lastGesture: Gesture? = null
        private set

    fun play(gestures: List<Gesture>) = gestures.forEach { play(it) }

    fun play(gesture: Gesture) {
        if (queue.size > 3) return
        queue.addLast(gesture)
        if (playing == null) next()
    }

    private fun next() {
        val gesture = queue.removeFirstOrNull() ?: run {
            playing = null
            return
        }
        lastGesture = gesture
        playing = animatorFor(gesture).apply {
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    reset()
                    next()
                }
            })
            start()
        }
    }

    private fun reset() {
        translationY = 0f
        rotation = 0f
        scaleX = 1f
        scaleY = 1f
        eyeOpen = 1f
        eyeShift = 0f
        invalidate()
    }

    private fun eyes(duration: Long, vararg open: Float) = ValueAnimator.ofFloat(*open).apply {
        this.duration = duration
        addUpdateListener { eyeOpen = it.animatedValue as Float; invalidate() }
    }

    private fun look(duration: Long, vararg shift: Float) = ValueAnimator.ofFloat(*shift).apply {
        this.duration = duration
        addUpdateListener { eyeShift = it.animatedValue as Float; invalidate() }
    }

    /** A property animation through evenly spaced key values. */
    private fun keys(property: Property, duration: Long, vararg values: Float): Animator {
        val frames = values.mapIndexed { i, v -> Keyframe.ofFloat(i / (values.size - 1f), v) }.toTypedArray()
        return ObjectAnimator.ofPropertyValuesHolder(this, PropertyValuesHolder.ofKeyframe(property.view, *frames)).apply { this.duration = duration }
    }

    private enum class Property(val view: android.util.Property<View, Float>) {
        Y(View.TRANSLATION_Y), TURN(View.ROTATION), WIDE(View.SCALE_X), TALL(View.SCALE_Y)
    }

    private fun hop(height: Float) = AnimatorSet().apply {
        playTogether(
            keys(Property.Y, 420, 0f, -height, -height * 0.6f, 0f, 0f),
            keys(Property.TALL, 420, 1f, 1.04f, 1f, 0.9f, 1f),
            keys(Property.WIDE, 420, 1f, 0.97f, 1f, 1.08f, 1f),
        )
    }

    private fun animatorFor(gesture: Gesture): Animator {
        pivotX = width / 2f
        pivotY = height.toFloat()
        val d = density
        return when (gesture) {
            Gesture.BLINK -> eyes(160, 1f, 0.08f, 1f)
            Gesture.DOUBLE_BLINK -> AnimatorSet().apply { playSequentially(eyes(150, 1f, 0.08f, 1f), eyes(150, 1f, 0.08f, 1f)) }
            Gesture.GLANCE -> look(1300, 0f, -1f, -1f, 1f, 1f, 0f)
            Gesture.HOP -> hop(14 * d)
            Gesture.DOUBLE_HOP -> AnimatorSet().apply { playSequentially(hop(16 * d), hop(12 * d)) }
            Gesture.TILT -> AnimatorSet().apply { playTogether(keys(Property.TURN, 1100, 0f, 14f, 14f, 14f, 0f), look(1100, 0f, 0.5f, 0.5f, 0f)) }
            Gesture.NOD -> keys(Property.Y, 520, 0f, 5 * d, 0f, 5 * d, 0f)
            Gesture.WIGGLE -> keys(Property.TURN, 800, 0f, -9f, 9f, -7f, 7f, 0f)
            Gesture.SQUISH -> AnimatorSet().apply { playTogether(keys(Property.TALL, 500, 1f, 0.85f, 1.06f, 1f), keys(Property.WIDE, 500, 1f, 1.1f, 0.96f, 1f)) }
            Gesture.SLEEPY -> AnimatorSet().apply { playTogether(eyes(1800, 1f, 0.35f, 0.35f, 0.35f, 1f), keys(Property.Y, 1800, 0f, 4 * d, 4 * d, 0f)) }
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        main.postDelayed(blinkLoop, Random.nextLong(GuideGestures.BLINK_MIN_MS, GuideGestures.BLINK_MAX_MS))
    }

    override fun onDetachedFromWindow() {
        main.removeCallbacks(blinkLoop)
        playing?.cancel()
        queue.clear()
        super.onDetachedFromWindow()
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
        val dx = eyeShift * 4 * u
        val half = 7 * u * eyeOpen.coerceIn(0.08f, 1f)
        rect.set(cx - 18 * u + dx, cy - 1 * u - half, cx - 8 * u + dx, cy - 1 * u + half)
        canvas.drawRoundRect(rect, 5 * u, 5 * u, ink)
        rect.set(cx + 8 * u + dx, cy - 1 * u - half, cx + 18 * u + dx, cy - 1 * u + half)
        canvas.drawRoundRect(rect, 5 * u, 5 * u, ink)
        smile.strokeWidth = 4 * u
        rect.set(cx - 13 * u, cy + 4 * u, cx + 13 * u, cy + 20 * u)
        canvas.drawArc(rect, 20f, 140f, false, smile)
    }
}
