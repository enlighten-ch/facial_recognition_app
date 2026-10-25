package com.allstar.faithid.kiosk.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/**
 * Turns a camera frame into an upright ARGB bitmap. The analysis use case delivers RGBA frames
 * (see [CameraController]), so no JPEG round trip is needed as in the first version.
 */
object FrameConverter {
    fun toUprightBitmap(image: ImageProxy): Bitmap? {
        val raw = runCatching { image.toBitmap() }.getOrNull() ?: return null
        val degrees = image.imageInfo.rotationDegrees
        if (degrees % 360 == 0) return raw
        val rotated = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)
        if (rotated !== raw) raw.recycle()
        return rotated
    }
}
