package com.allstar.faithid.kiosk.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import com.allstar.faithid.core.geometry.Point
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.recognition.FaceAlignment
import com.allstar.faithid.core.recognition.FaceLandmarks
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections

/**
 * Face detection (Google ML Kit) and face embeddings (InsightFace Buffalo-L ArcFace R50, ONNX).
 * Call from one background thread.
 */
class FaceEngine(private val context: Context, private val log: (String) -> Unit) : AutoCloseable {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build(),
    )
    private val environment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    /** Why the model could not be loaded, or null when it is ready. */
    var modelError: String? = null
        private set

    val modelPath: String get() = File(context.filesDir, MODEL_ASSET).absolutePath

    init {
        runCatching {
            val file = extractModel()
            session = environment.createSession(file.absolutePath, OrtSession.SessionOptions())
            log("얼굴 인식 모델 준비 완료 (${file.length() / 1_000_000} MB)")
        }.onFailure {
            modelError = "얼굴 인식 모델을 불러오지 못했습니다: ${it.javaClass.simpleName} ${it.message.orEmpty()}"
            log(modelError!!)
        }
    }

    val isReady: Boolean get() = session != null

    fun detect(bitmap: Bitmap): List<Face> = Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0)))

    /**
     * 512-dimensional, L2-normalized embedding. With [align] the face is warped onto the ArcFace
     * template using its landmarks; otherwise the detection box is cropped, as in the first version.
     */
    fun embed(face: Face, frame: Bitmap, align: Boolean): FloatArray? {
        val model = session ?: return null
        val input = (if (align) alignedCrop(face, frame) else null) ?: boxCrop(face, frame) ?: return null
        val tensorData = toNchw(input)
        input.recycle()
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensorData), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            model.run(Collections.singletonMap(model.inputNames.first(), tensor)).use { result ->
                val raw = when (val value = result[0].value) {
                    is Array<*> -> value.firstOrNull() as? FloatArray
                    is FloatArray -> value
                    else -> null
                } ?: return null
                return Vectors.l2Normalize(raw.copyOf())
            }
        }
    }

    override fun close() {
        detector.close()
        runCatching { session?.close() }
    }

    private fun boxCrop(face: Face, frame: Bitmap): Bitmap? {
        val box = face.boundingBox
        val left = box.left.coerceAtLeast(0)
        val top = box.top.coerceAtLeast(0)
        val right = box.right.coerceAtMost(frame.width)
        val bottom = box.bottom.coerceAtMost(frame.height)
        if (right <= left || bottom <= top) return null
        val crop = Bitmap.createBitmap(frame, left, top, right - left, bottom - top)
        val scaled = Bitmap.createScaledBitmap(crop, SIZE, SIZE, true)
        if (scaled !== crop) crop.recycle()
        return scaled
    }

    private fun alignedCrop(face: Face, frame: Bitmap): Bitmap? {
        val landmarks = landmarksOf(face) ?: return null
        val t = FaceAlignment.transformFor(landmarks)
        val out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val matrix = Matrix().apply { setValues(floatArrayOf(t.a, -t.b, t.tx, t.b, t.a, t.ty, 0f, 0f, 1f)) }
        Canvas(out).drawBitmap(frame, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        return out
    }

    private fun toNchw(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(SIZE * SIZE)
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        val out = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
            out[plane + i] = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
            out[2 * plane + i] = ((p and 0xFF) - 127.5f) / 127.5f
        }
        return out
    }

    /** Copies the model from the APK assets to internal storage once (ONNX Runtime needs a file path). */
    private fun extractModel(): File {
        val target = File(context.filesDir, MODEL_ASSET)
        val assetSize = runCatching { context.assets.openFd(MODEL_ASSET).use { it.length } }.getOrDefault(-1L)
        if (!target.isFile || target.length() <= 0L || (assetSize > 0L && target.length() != assetSize)) {
            target.parentFile?.mkdirs()
            log("얼굴 인식 모델 복사 중: $MODEL_ASSET")
            context.assets.open(MODEL_ASSET).use { input -> target.outputStream().use { input.copyTo(it) } }
        }
        return target
    }

    companion object {
        const val MODEL_ASSET = "models/buffalo_l/w600k_r50.onnx"
        private const val SIZE = FaceAlignment.SIZE

        fun landmarksOf(face: Face): FaceLandmarks? {
            fun point(type: Int) = face.getLandmark(type)?.position?.let { Point(it.x, it.y) }
            return FaceLandmarks(
                leftEye = point(FaceLandmark.LEFT_EYE) ?: return null,
                rightEye = point(FaceLandmark.RIGHT_EYE) ?: return null,
                noseBase = point(FaceLandmark.NOSE_BASE) ?: return null,
                mouthLeft = point(FaceLandmark.MOUTH_LEFT) ?: return null,
                mouthRight = point(FaceLandmark.MOUTH_RIGHT) ?: return null,
            )
        }
    }
}
