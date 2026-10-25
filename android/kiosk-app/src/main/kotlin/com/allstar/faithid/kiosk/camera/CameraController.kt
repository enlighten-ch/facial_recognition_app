package com.allstar.faithid.kiosk.camera

import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Binds the preview and frame analysis to the activity lifecycle. Frames arrive on a background
 * thread; while one frame is being analyzed newer frames are dropped (keep only latest).
 */
class CameraController(
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    private val onFrame: (Bitmap) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "faith-camera") }
    private var provider: ProcessCameraProvider? = null

    /** True when the front camera is in use; its preview is shown mirrored. */
    @Volatile
    var mirrored: Boolean = false
        private set

    fun start(preferFront: Boolean) {
        val context = previewView.context
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = runCatching { future.get() }.getOrElse {
                onError("카메라를 준비하지 못했습니다: ${it.javaClass.simpleName}")
                return@addListener
            }
            provider = cameraProvider
            val selector = listOf(
                if (preferFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA,
                if (preferFront) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA,
            ).firstOrNull { runCatching { cameraProvider.hasCamera(it) }.getOrDefault(false) }
            if (selector == null) {
                onError("카메라를 찾을 수 없습니다. 설정 › 인식에서 카메라 번호를 바꿔 보세요.")
                return@addListener
            }
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image ->
                val bitmap = try {
                    FrameConverter.toUprightBitmap(image)
                } finally {
                    image.close()
                }
                if (bitmap != null) onFrame(bitmap)
            }
            runCatching {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(owner, selector, preview, analysis)
                mirrored = camera.cameraInfo.lensFacing == CameraSelector.LENS_FACING_FRONT
                previewView.scaleX = if (mirrored) -1f else 1f
            }.onFailure { onError("카메라 연결 실패: ${it.javaClass.simpleName}") }
        }, ContextCompat.getMainExecutor(context))
    }

    fun restart(preferFront: Boolean) {
        provider?.unbindAll()
        start(preferFront)
    }

    fun shutdown() {
        provider?.unbindAll()
        analysisExecutor.shutdown()
    }
}
