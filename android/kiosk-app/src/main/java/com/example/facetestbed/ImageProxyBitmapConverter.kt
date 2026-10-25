package com.example.facetestbed

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object ImageProxyBitmapConverter {
    fun toUprightBitmap(image: ImageProxy): Bitmap? {
        val nv21 = yuv420ToNv21(image) ?: return null
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        val base = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        return rotate(base, image.imageInfo.rotationDegrees)
    }

    private fun rotate(src: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return src
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        if (rotated != src) src.recycle()
        return rotated
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray? {
        val yPlane = image.planes.getOrNull(0) ?: return null
        val uPlane = image.planes.getOrNull(1) ?: return null
        val vPlane = image.planes.getOrNull(2) ?: return null

        val ySize = yPlane.buffer.remaining()
        val uSize = uPlane.buffer.remaining()
        val vSize = vPlane.buffer.remaining()
        val out = ByteArray(ySize + uSize + vSize)

        yPlane.buffer.get(out, 0, ySize)

        val chromaRowStride = uPlane.rowStride
        val chromaPixelStride = uPlane.pixelStride
        val width = image.width
        val height = image.height

        var offset = ySize
        val vBuf = vPlane.buffer
        val uBuf = uPlane.buffer

        val chromaHeight = height / 2
        val chromaWidth = width / 2

        val vRow = ByteArray(vPlane.rowStride)
        val uRow = ByteArray(uPlane.rowStride)

        for (row in 0 until chromaHeight) {
            val vPos = row * vPlane.rowStride
            val uPos = row * uPlane.rowStride
            vBuf.position(vPos)
            uBuf.position(uPos)
            vBuf.get(vRow, 0, minOf(vPlane.rowStride, vBuf.remaining()))
            uBuf.get(uRow, 0, minOf(uPlane.rowStride, uBuf.remaining()))

            for (col in 0 until chromaWidth) {
                val vuIndex = col * chromaPixelStride
                out[offset++] = vRow[vuIndex]
                out[offset++] = uRow[vuIndex]
            }
        }
        return out
    }
}
