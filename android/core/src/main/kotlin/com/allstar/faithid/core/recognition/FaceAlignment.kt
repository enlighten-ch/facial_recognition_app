package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.geometry.Point

/**
 * Similarity transform `x' = a·x − b·y + tx`, `y' = b·x + a·y + ty` (rotation, uniform scale,
 * translation). On Android it maps to `Matrix.setValues(floatArrayOf(a, -b, tx, b, a, ty, 0, 0, 1))`.
 */
data class SimilarityTransform(val a: Float, val b: Float, val tx: Float, val ty: Float) {
    fun apply(p: Point): Point = Point(a * p.x - b * p.y + tx, b * p.x + a * p.y + ty)

    val scale: Float get() = kotlin.math.sqrt(a * a + b * b)
}

/**
 * Aligns a face to the 112×112 ArcFace template using the five ML Kit landmarks, the input layout
 * the Buffalo-L recognition model was trained on.
 */
object FaceAlignment {
    const val SIZE = 112

    /** Reference positions from InsightFace: left eye, right eye, nose, left and right mouth corner (image left/right). */
    val ARCFACE_TEMPLATE = listOf(
        Point(38.2946f, 51.6963f),
        Point(73.5318f, 51.5014f),
        Point(56.0252f, 71.7366f),
        Point(41.5493f, 92.3655f),
        Point(70.7299f, 92.2041f),
    )

    /**
     * Transform that maps image coordinates onto the template. Eyes and mouth corners are ordered by
     * their x coordinate, so it does not matter whether the image is mirrored or which eye ML Kit
     * calls "left".
     */
    fun transformFor(landmarks: FaceLandmarks): SimilarityTransform {
        val eyes = listOf(landmarks.leftEye, landmarks.rightEye).sortedBy { it.x }
        val mouth = listOf(landmarks.mouthLeft, landmarks.mouthRight).sortedBy { it.x }
        return estimate(listOf(eyes[0], eyes[1], landmarks.noseBase, mouth[0], mouth[1]), ARCFACE_TEMPLATE)
    }

    /** Least-squares similarity transform from [src] to [dst] (Umeyama, without reflection). */
    fun estimate(src: List<Point>, dst: List<Point>): SimilarityTransform {
        require(src.size == dst.size && src.size >= 2) { "need at least two point pairs" }
        val n = src.size.toFloat()
        val srcMeanX = src.sumOf { it.x.toDouble() }.toFloat() / n
        val srcMeanY = src.sumOf { it.y.toDouble() }.toFloat() / n
        val dstMeanX = dst.sumOf { it.x.toDouble() }.toFloat() / n
        val dstMeanY = dst.sumOf { it.y.toDouble() }.toFloat() / n
        var dotSum = 0.0
        var crossSum = 0.0
        var normSum = 0.0
        for (i in src.indices) {
            val ax = (src[i].x - srcMeanX).toDouble()
            val ay = (src[i].y - srcMeanY).toDouble()
            val bx = (dst[i].x - dstMeanX).toDouble()
            val by = (dst[i].y - dstMeanY).toDouble()
            dotSum += ax * bx + ay * by
            crossSum += ax * by - ay * bx
            normSum += ax * ax + ay * ay
        }
        require(normSum > 1e-9) { "source points are degenerate" }
        val a = (dotSum / normSum).toFloat()
        val b = (crossSum / normSum).toFloat()
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)
        return SimilarityTransform(a, b, tx, ty)
    }
}
