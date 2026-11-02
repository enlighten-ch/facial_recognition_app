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
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 빛이, the guide: a warm round face with two eyes and a smile. It blinks on its own every few
 * seconds and plays [Gesture]s on request, one after another (see [GuideGestures]). Children and
 * adults meet the same 빛이. The face is a ball: [turnTo] turns it like the registration's white
 * face, and [faceScale] makes it bigger or smaller with the distance to keep.
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
    private val smilePath = Path()
    private val point = FloatArray(3)
    private val density = resources.displayMetrics.density
    private val main = Handler(Looper.getMainLooper())

    /** 1 = eyes open, 0 = closed. */
    private var eyeOpen = 1f
    /** A glance: -1 looks left, 1 looks right. The whole face turns, eyes and smile together. */
    private var glance = 0f

    // The head turn (radians) and size, eased toward the targets.
    private var yaw = 0f
    private var pitch = 0f
    private var scaleNow = 1f
    private var targetYaw = 0f
    private var targetPitch = 0f
    private var targetScale = 1f
    private var lastDrawNanos = 0L
    private var shaderRadius = 0f

    /** The size [faceScale] 1 is drawn at relative to the view: the view leaves room up to [maxScale]. */
    var maxScale = 1f
        set(value) {
            field = value
            invalidate()
        }

    /** Turns the face toward [yaw] (right positive) and [pitch] (down positive) and sizes it, smoothly. */
    fun turnTo(yaw: Float, pitch: Float, faceScale: Float = 1f) {
        if (targetYaw == yaw && targetPitch == pitch && targetScale == faceScale) return
        targetYaw = yaw
        targetPitch = pitch
        targetScale = faceScale
        invalidate()
    }

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
        glance = 0f
        invalidate()
    }

    private fun eyes(duration: Long, vararg open: Float) = ValueAnimator.ofFloat(*open).apply {
        this.duration = duration
        addUpdateListener { eyeOpen = it.animatedValue as Float; invalidate() }
    }

    private fun look(duration: Long, vararg shift: Float) = ValueAnimator.ofFloat(*shift).apply {
        this.duration = duration
        addUpdateListener { glance = it.animatedValue as Float; invalidate() }
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
        shaderRadius = 0f
    }

    private fun shade(cx: Float, cy: Float, r: Float) {
        if (abs(r - shaderRadius) < 0.5f) return
        shaderRadius = r
        face.shader = RadialGradient(cx - r * 0.3f, cy - r * 0.4f, r * 1.4f, 0xFFFFD2B4.toInt(), 0xFFFF9A62.toInt(), Shader.TileMode.CLAMP)
    }

    override fun onDraw(canvas: Canvas) {
        val now = System.nanoTime()
        val dt = if (lastDrawNanos == 0L) 0f else min(0.1f, (now - lastDrawNanos) / 1e9f)
        lastDrawNanos = now
        val k = 1 - exp(-dt * 6f)
        yaw += (targetYaw - yaw) * k
        pitch += (targetPitch - pitch) * k
        scaleNow += (targetScale - scaleNow) * k
        val settling = abs(targetYaw - yaw) > 0.002f || abs(targetPitch - pitch) > 0.002f || abs(targetScale - scaleNow) > 0.002f

        val cx = width / 2f
        val cy = height / 2f
        val r = min(width, height) / 2f * scaleNow / maxScale
        shade(cx, cy, r)
        canvas.drawCircle(cx, cy, r, face)
        // Eyes and smile live on the ball's surface and turn with it as one face, glances included.
        val faceYaw = yaw + glance * 0.35f
        for (sx in floatArrayOf(-0.27f, 0.27f)) {
            val p = turn(sx, -0.02f, sqrt(1 - sx * sx - 0.0004f), faceYaw, pitch)
            if (p[2] < 0.2f) continue
            val hw = 0.104f * r * sqrt(max(0f, 1 - p[0] * p[0]))
            val hh = 0.146f * r * eyeOpen.coerceIn(0.08f, 1f) * sqrt(max(0f, 1 - p[1] * p[1]))
            rect.set(cx + p[0] * r - hw, cy + p[1] * r - hh, cx + p[0] * r + hw, cy + p[1] * r + hh)
            canvas.drawRoundRect(rect, hw, hw, ink)
        }
        smile.strokeWidth = r / 12f
        smilePath.reset()
        var started = false
        for (i in 0..16) {
            val x = -0.27f + 0.54f * i / 16
            val y = 0.26f + 0.14f * (1 - (x / 0.27f) * (x / 0.27f))
            val p = turn(x, y, sqrt(max(0f, 1 - x * x - y * y)), faceYaw, pitch)
            if (p[2] < 0.05f) {
                started = false
                continue
            }
            if (started) smilePath.lineTo(cx + p[0] * r, cy + p[1] * r) else smilePath.moveTo(cx + p[0] * r, cy + p[1] * r)
            started = true
        }
        canvas.drawPath(smilePath, smile)
        if (settling) postInvalidateOnAnimation() else lastDrawNanos = 0L
    }

    /** A point on the unit ball turned by yaw, then pitch (the same turn as the white face). */
    private fun turn(x: Float, y: Float, z: Float, yaw: Float, pitch: Float): FloatArray {
        val z0 = -x * sin(yaw) + z * cos(yaw)
        point[0] = x * cos(yaw) + z * sin(yaw)
        point[1] = y * cos(pitch) + z0 * sin(pitch)
        point[2] = -y * sin(pitch) + z0 * cos(pitch)
        return point
    }
}
