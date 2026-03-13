package com.example.facetestbed

import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlin.math.abs
import kotlin.math.sqrt

class FaceRecognitionEngine {
    private val detector: FaceDetector

    init {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        detector = FaceDetection.getClient(options)
    }

    fun detectLargestFace(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    fun makeEmbedding(face: Face, imageWidth: Int, imageHeight: Int): FloatArray {
        val iw = imageWidth.toFloat().coerceAtLeast(1f)
        val ih = imageHeight.toFloat().coerceAtLeast(1f)
        val box: Rect = face.boundingBox

        val fx = mutableListOf<Float>()
        fx.add(box.centerX() / iw)
        fx.add(box.centerY() / ih)
        fx.add(box.width() / iw)
        fx.add(box.height() / ih)
        fx.add(face.headEulerAngleX / 90f)
        fx.add(face.headEulerAngleY / 90f)
        fx.add(face.headEulerAngleZ / 90f)

        appendLandmark(fx, face, FaceLandmark.LEFT_EYE, iw, ih)
        appendLandmark(fx, face, FaceLandmark.RIGHT_EYE, iw, ih)
        appendLandmark(fx, face, FaceLandmark.NOSE_BASE, iw, ih)
        appendLandmark(fx, face, FaceLandmark.MOUTH_LEFT, iw, ih)
        appendLandmark(fx, face, FaceLandmark.MOUTH_RIGHT, iw, ih)
        appendLandmark(fx, face, FaceLandmark.MOUTH_BOTTOM, iw, ih)
        appendLandmark(fx, face, FaceLandmark.LEFT_EAR, iw, ih)
        appendLandmark(fx, face, FaceLandmark.RIGHT_EAR, iw, ih)
        appendLandmark(fx, face, FaceLandmark.LEFT_CHEEK, iw, ih)
        appendLandmark(fx, face, FaceLandmark.RIGHT_CHEEK, iw, ih)

        val emb = FloatArray(fx.size)
        for (i in fx.indices) emb[i] = fx[i]
        return l2Normalize(emb)
    }

    fun classifyDirection(face: Face): String? {
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position ?: return null
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position ?: return null
        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position ?: return null
        val mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position ?: return null
        val mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position ?: return null

        val eyeCenterX = (leftEye.x + rightEye.x) / 2f
        val eyeCenterY = (leftEye.y + rightEye.y) / 2f
        val eyeDx = rightEye.x - leftEye.x
        val eyeDy = rightEye.y - leftEye.y
        val eyeDist = sqrt(eyeDx * eyeDx + eyeDy * eyeDy).coerceAtLeast(1e-6f)

        val mouthCenterY = (mouthLeft.y + mouthRight.y) / 2f
        val eyeToMouth = (mouthCenterY - eyeCenterY)
        if (abs(eyeToMouth) < 1e-6f) return null

        val yaw = (nose.x - eyeCenterX) / eyeDist
        val pitch = (nose.y - eyeCenterY) / eyeToMouth

        return when {
            yaw <= -0.10f -> "left"
            yaw >= 0.10f -> "right"
            pitch <= 0.42f -> "up"
            pitch >= 0.62f -> "down"
            else -> null
        }
    }

    fun close() {
        detector.close()
    }

    private fun appendLandmark(out: MutableList<Float>, face: Face, kind: Int, iw: Float, ih: Float) {
        val lm = face.getLandmark(kind)
        if (lm == null) {
            out.add(0f)
            out.add(0f)
            return
        }
        out.add(lm.position.x / iw)
        out.add(lm.position.y / ih)
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += (x * x).toDouble()
        val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(v.size) { i -> v[i] / norm }
    }
}
