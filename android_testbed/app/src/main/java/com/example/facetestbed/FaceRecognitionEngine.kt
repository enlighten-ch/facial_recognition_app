package com.example.facetestbed

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.abs
import kotlin.math.sqrt

class FaceRecognitionEngine(private val context: Context) {
    private val detector: FaceDetector
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var outputName: String? = null
    private var buffaloReady = false

    init {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build()
        detector = FaceDetection.getClient(options)
        initBuffaloModel()
    }

    fun detectLargestFace(image: InputImage): Task<List<Face>> = detector.process(image)

    fun isBuffaloReady(): Boolean = buffaloReady

    fun makeEmbedding(face: Face, frameBitmap: Bitmap): FloatArray {
        val onnx = makeBuffaloEmbedding(face, frameBitmap)
        if (onnx != null) return onnx
        return makeGeometryEmbedding(face, frameBitmap.width, frameBitmap.height)
    }

    fun classifyDirection(face: Face, mirroredPreview: Boolean = false): String? {
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
        val eyeToMouth = mouthCenterY - eyeCenterY
        if (abs(eyeToMouth) < 1e-6f) return null

        var yaw = (nose.x - eyeCenterX) / eyeDist
        if (mirroredPreview) {
            yaw = -yaw
        }
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
        runCatching { ortSession?.close() }
    }

    private fun initBuffaloModel() {
        runCatching {
            val assetPath = "models/buffalo_l/w600k_r50.onnx"
            val modelFile = copyAssetToFiles(assetPath)
            val session = ortEnv.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
            val firstOutput = session.outputNames.firstOrNull()
            ortSession = session
            outputName = firstOutput
            buffaloReady = firstOutput != null
        }.onFailure {
            buffaloReady = false
        }
    }

    private fun copyAssetToFiles(assetPath: String): File {
        val outFile = File(context.filesDir, assetPath.replace('/', '_'))
        if (outFile.exists() && outFile.length() > 0L) return outFile
        context.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    private fun makeBuffaloEmbedding(face: Face, frameBitmap: Bitmap): FloatArray? {
        val session = ortSession ?: return null
        val out = outputName ?: return null

        val crop = cropFace(frameBitmap, face.boundingBox) ?: return null
        val resized = Bitmap.createScaledBitmap(crop, 112, 112, true)
        if (crop != resized) crop.recycle()

        val inputTensorData = bitmapToNchwFloat(resized)
        resized.recycle()

        val tensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(inputTensorData),
            longArrayOf(1, 3, 112, 112),
        )

        tensor.use { t ->
            session.run(Collections.singletonMap(session.inputNames.first(), t)).use { results ->
                val value = results[0].value
                val emb = when (value) {
                    is Array<*> -> {
                        val row = (value.getOrNull(0) as? FloatArray) ?: return null
                        row
                    }
                    is FloatArray -> value
                    else -> return null
                }
                return l2Normalize(emb.copyOf())
            }
        }
    }

    private fun cropFace(bitmap: Bitmap, bbox: Rect): Bitmap? {
        val x1 = bbox.left.coerceAtLeast(0)
        val y1 = bbox.top.coerceAtLeast(0)
        val x2 = bbox.right.coerceAtMost(bitmap.width)
        val y2 = bbox.bottom.coerceAtMost(bitmap.height)
        if (x2 <= x1 || y2 <= y1) return null
        return Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)
    }

    private fun bitmapToNchwFloat(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = FloatArray(3 * w * h)

        val hw = w * h
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF).toFloat()
            val g = ((p shr 8) and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            out[i] = (r - 127.5f) / 127.5f
            out[hw + i] = (g - 127.5f) / 127.5f
            out[2 * hw + i] = (b - 127.5f) / 127.5f
        }
        return out
    }

    private fun makeGeometryEmbedding(face: Face, imageWidth: Int, imageHeight: Int): FloatArray {
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
