package com.allstar.faithid.kiosk.ui.guide

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.provider.Settings
import android.util.AttributeSet
import android.view.Choreographer
import android.view.View
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A cartoon head that shows the move to make: it turns towards [direction] and comes closer or
 * steps back with [circle]. Eyes and cheeks show which way it faces; there is no nose or mouth.
 * Ray-cast into a small bitmap each frame while it moves.
 */
class HeadView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs), Choreographer.FrameCallback {

    private val size = 200
    private val pixels = IntArray(size * size)
    private val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val dst = Rect()
    private val reduceMotion = runCatching {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }.getOrDefault(false)

    private var yaw = 0.0
    private var pitch = 0.0
    private var zoom = 1.0
    private var lastFrameNanos = 0L
    private var drawnKey = ""
    private var running = false

    var direction: HeadDirection? = null
    var circle: FaceCircle? = null
    /** Nods happily instead of turning (registration finished). */
    var done = false
    var kids = true
        set(value) {
            field = value
            drawnKey = ""
        }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        start()
    }

    override fun onDetachedFromWindow() {
        stop()
        super.onDetachedFromWindow()
    }

    override fun onVisibilityAggregated(isVisible: Boolean) {
        super.onVisibilityAggregated(isVisible)
        if (isVisible) start() else stop()
    }

    private fun start() {
        if (running) return
        running = true
        lastFrameNanos = 0L
        Choreographer.getInstance().postFrameCallback(this)
    }

    private fun stop() {
        running = false
        Choreographer.getInstance().removeFrameCallback(this)
    }

    override fun doFrame(frameTimeNanos: Long) {
        if (!running) return
        val dt = if (lastFrameNanos == 0L) 0.0 else min(0.1, (frameTimeNanos - lastFrameNanos) / 1e9)
        lastFrameNanos = frameTimeNanos
        val t = frameTimeNanos / 1e9
        var targetYaw = 0.0
        var targetPitch = 0.0
        var targetZoom = 1.0
        val d = direction
        if (done) {
            targetPitch = if (reduceMotion) 0.0 else sin(t * 6.25) * 0.22
        } else if (d != null) {
            val sway = if (reduceMotion) 1.0 else 0.86 + 0.14 * sin(t * 2.4)
            when (d) {
                HeadDirection.LEFT -> targetYaw = -0.98 * sway
                HeadDirection.RIGHT -> targetYaw = 0.98 * sway
                HeadDirection.UP -> targetPitch = -0.6 * sway
                HeadDirection.DOWN -> targetPitch = 0.55 * sway
            }
            targetZoom = when (circle) {
                FaceCircle.BIG -> 1.24
                FaceCircle.SMALL -> 0.78
                else -> 1.0
            }
        }
        val ease = if (reduceMotion) 1.0 else 1 - exp(-dt * 4.5)
        yaw += (targetYaw - yaw) * ease
        pitch += (targetPitch - pitch) * ease
        zoom += (targetZoom - zoom) * (if (reduceMotion) 1.0 else 1 - exp(-dt * 3))
        val key = "%.3f,%.3f,%.3f,%b".format(yaw, pitch, zoom, kids)
        if (key != drawnKey) {
            drawnKey = key
            render()
            invalidate()
        }
        Choreographer.getInstance().postFrameCallback(this)
    }

    override fun onDraw(canvas: Canvas) {
        val side = min(width, height)
        dst.set((width - side) / 2, (height - side) / 2, (width + side) / 2, (height + side) / 2)
        canvas.drawBitmap(bitmap, null, dst, paint)
    }

    // ----- ray casting ----------------------------------------------------------------------

    private var hitT = 0.0

    /** Ray o + t·d against an axis-aligned ellipsoid; sets [hitT] and returns true on a hit in front. */
    private fun hit(ox: Double, oy: Double, oz: Double, dx: Double, dy: Double, dz: Double,
                    cx: Double, cy: Double, cz: Double, rx: Double, ry: Double, rz: Double): Boolean {
        val px = (ox - cx) / rx; val py = (oy - cy) / ry; val pz = (oz - cz) / rz
        val qx = dx / rx; val qy = dy / ry; val qz = dz / rz
        val a = qx * qx + qy * qy + qz * qz
        val b = 2 * (px * qx + py * qy + pz * qz)
        val c = px * px + py * py + pz * pz - 1
        val disc = b * b - 4 * a * c
        if (disc < 0) return false
        hitT = (-b - sqrt(disc)) / (2 * a)
        return hitT > 0
    }

    private fun render() {
        val skinR = if (kids) 255.0 else 236.0; val skinG = if (kids) 214.0 else 203.0; val skinB = if (kids) 186.0 else 178.0
        val hairR = if (kids) 120.0 else 86.0; val hairG = if (kids) 78.0 else 70.0; val hairB = if (kids) 64.0 else 78.0
        val shirtR = if (kids) 255.0 else 104.0; val shirtG = if (kids) 160.0 else 78.0; val shirtB = if (kids) 110.0 else 150.0
        val radius = size * 0.26 * zoom
        val centerX = size / 2.0
        val centerY = size * 0.43
        val cyw = cos(yaw); val syw = sin(yaw); val cp = cos(pitch); val sp = sin(pitch)
        // World ray direction (0, 0, -1) in head coordinates: undo yaw, then pitch.
        val dLx = syw; val dLz0 = -cyw
        val dLy = sp * dLz0; val dLz = cp * dLz0
        // Light and half vector.
        val lx = -0.45; val ly = 0.6; val lz = 0.66
        val ll = sqrt(lx * lx + ly * ly + lz * lz)
        val Lx = lx / ll; val Ly = ly / ll; val Lz = lz / ll
        val hl = sqrt(Lx * Lx + Ly * Ly + (Lz + 1) * (Lz + 1))
        val Hx = Lx / hl; val Hy = Ly / hl; val Hz = (Lz + 1) / hl

        for (py in 0 until size) {
            for (px in 0 until size) {
                val i = py * size + px
                val wx = (px - centerX) / radius
                val wy = -(py - centerY) / radius
                if (wx * wx + wy * wy > 7) { pixels[i] = 0; continue }
                // Ray origin in head coordinates.
                val x1 = cyw * wx - syw * 5.0
                val z1 = syw * wx + cyw * 5.0
                val oLx = x1; val oLy = cp * wy + sp * z1; val oLz = -sp * wy + cp * z1

                var best = Double.MAX_VALUE
                var r = 0.0; var g = 0.0; var b = 0.0
                var nx = 0.0; var ny = 0.0; var nz = 0.0
                var gloss = 0.0
                var found = false
                var local = false

                if (hit(oLx, oLy, oLz, dLx, dLy, dLz, 0.0, 0.0, 0.0, 1.0, 1.04, 0.96) && hitT < best) {
                    best = hitT; found = true; local = true
                    val hx = oLx + hitT * dLx; val hy = oLy + hitT * dLy; val hz = oLz + hitT * dLz
                    val n1 = hx; val n2 = hy / 1.0816; val n3 = hz / 0.9216
                    var l = sqrt(n1 * n1 + n2 * n2 + n3 * n3); nx = n1 / l; ny = n2 / l; nz = n3 / l
                    l = sqrt(hx * hx + hy * hy + hz * hz)
                    val ux = hx / l; val uy = hy / l; val uz = hz / l
                    val fringe = 0.5 + 0.07 * cos(ux * 11)
                    val line = if (uz > 0) fringe + 0.05 * uz else 0.5 + uz * 0.95
                    if (uy > line) {
                        r = hairR; g = hairG; b = hairB; gloss = 0.35
                    } else {
                        r = skinR; g = skinG; b = skinB; gloss = 0.08
                        var eye = false
                        for (sx in SIDES) {
                            val ex = ux - sx * 0.3; val ey = uy - 0.06
                            if (uz > 0.5 && (ex / 0.105).pow(2) + (ey / 0.135).pow(2) < 1) {
                                eye = true; r = 34.0; g = 24.0; b = 26.0; gloss = 0.6
                                if (kids && ((ex - 0.03) / 0.04).pow(2) + ((ey - 0.045) / 0.04).pow(2) < 1) { r = 255.0; g = 255.0; b = 255.0 }
                            }
                        }
                        if (kids && !eye && uz > 0.4) for (sx in SIDES) {
                            val bx = ux - sx * 0.5; val by = uy + 0.17
                            val d2 = (bx / 0.13).pow(2) + (by / 0.08).pow(2)
                            if (d2 < 1) { val w = (1 - d2) * 0.6; r += (255 - r) * w; g += (140 - g) * w; b += (140 - b) * w }
                        }
                    }
                }
                for (sx in SIDES) {
                    if (hit(oLx, oLy, oLz, dLx, dLy, dLz, sx * 0.97, -0.05, -0.04, 0.2, 0.24, 0.16) && hitT < best) {
                        best = hitT; found = true; local = true
                        val hx = oLx + hitT * dLx - sx * 0.97; val hy = oLy + hitT * dLy + 0.05; val hz = oLz + hitT * dLz + 0.04
                        val n1 = hx / 0.04; val n2 = hy / 0.0576; val n3 = hz / 0.0256
                        val l = sqrt(n1 * n1 + n2 * n2 + n3 * n3); nx = n1 / l; ny = n2 / l; nz = n3 / l
                        r = skinR; g = skinG; b = skinB; gloss = 0.05
                    }
                }
                if (hit(oLx, oLy, oLz, dLx, dLy, dLz, 0.12, 1.04, -0.12, 0.2, 0.2, 0.2) && hitT < best) {
                    best = hitT; found = true; local = true
                    val hx = oLx + hitT * dLx - 0.12; val hy = oLy + hitT * dLy - 1.04; val hz = oLz + hitT * dLz + 0.12
                    val l = sqrt(hx * hx + hy * hy + hz * hz); nx = hx / l; ny = hy / l; nz = hz / l
                    r = hairR; g = hairG; b = hairB; gloss = 0.35
                }
                if (local) {
                    // Head normal back to world: pitch, then yaw.
                    val y1 = cp * ny - sp * nz
                    val z2 = sp * ny + cp * nz
                    val wnx = cyw * nx + syw * z2
                    val wnz = -syw * nx + cyw * z2
                    nx = wnx; ny = y1; nz = wnz
                }
                // The body does not turn.
                if (hit(wx, wy, 5.0, 0.0, 0.0, -1.0, 0.0, -1.05, -0.08, 0.36, 0.42, 0.34) && hitT < best) {
                    best = hitT; found = true
                    val hx = wx; val hy = wy + 1.05; val hz = 5.0 - hitT + 0.08
                    val n1 = hx / 0.1296; val n2 = hy / 0.1764; val n3 = hz / 0.1156
                    val l = sqrt(n1 * n1 + n2 * n2 + n3 * n3); nx = n1 / l; ny = n2 / l; nz = n3 / l
                    r = skinR * 0.9; g = skinG * 0.9; b = skinB * 0.9; gloss = 0.05
                }
                if (hit(wx, wy, 5.0, 0.0, 0.0, -1.0, 0.0, -1.78, -0.2, 1.45, 0.78, 0.8) && hitT < best) {
                    found = true
                    val hx = wx; val hy = wy + 1.78; val hz = 5.0 - hitT + 0.2
                    val n1 = hx / 2.1025; val n2 = hy / 0.6084; val n3 = hz / 0.64
                    val l = sqrt(n1 * n1 + n2 * n2 + n3 * n3); nx = n1 / l; ny = n2 / l; nz = n3 / l
                    r = shirtR; g = shirtG; b = shirtB; gloss = 0.05
                }
                if (!found) { pixels[i] = 0; continue }
                val diff = max(0.0, nx * Lx + ny * Ly + nz * Lz)
                val spec = max(0.0, nx * Hx + ny * Hy + nz * Hz).pow(28) * gloss * 255
                val rim = (1 - max(0.0, nz)).pow(3) * 0.18
                val k = 0.52 + 0.55 * diff
                val cr = min(255.0, r * k + spec + 255 * rim).toInt()
                val cg = min(255.0, g * k + spec + 230 * rim).toInt()
                val cb = min(255.0, b * k + spec + 210 * rim).toInt()
                pixels[i] = (0xFF shl 24) or (cr shl 16) or (cg shl 8) or cb
            }
        }
        bitmap.setPixels(pixels, 0, size, 0, 0, size, size)
    }

    companion object {
        private val SIDES = intArrayOf(-1, 1)

        /** Where the centre of the head is drawn, as a fraction of this view's height (aligns it with the face circle). */
        const val HEAD_CENTER_FRACTION = 0.43f
    }
}
