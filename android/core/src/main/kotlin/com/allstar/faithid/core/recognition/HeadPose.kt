package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.geometry.Point
import kotlin.math.abs
import kotlin.math.sqrt

/** Head directions requested during registration, in the order they are collected. */
enum class HeadDirection(val label: String, val arrow: String, val prompt: String, val progressLabel: String) {
    LEFT("좌", "←", "얼굴을 화면 기준 왼쪽으로 돌려주세요.", "왼쪽 얼굴"),
    RIGHT("우", "→", "얼굴을 화면 기준 오른쪽으로 돌려주세요.", "오른쪽 얼굴"),
    UP("위", "↑", "턱을 살짝 들고 위를 봐주세요.", "위쪽 얼굴"),
    DOWN("아래", "↓", "턱을 살짝 내리고 아래를 봐주세요.", "아래쪽 얼굴"),
}

/** The five landmarks ML Kit reports, in image coordinates. */
data class FaceLandmarks(
    val leftEye: Point,
    val rightEye: Point,
    val noseBase: Point,
    val mouthLeft: Point,
    val mouthRight: Point,
)

object HeadPose {
    /** Nose offset from the eye midpoint, in eye-distance units, beyond which the head counts as turned. */
    const val YAW_THRESHOLD = 0.10f

    /** Nose height between eyes (0) and mouth (1) at or below which the head counts as tilted up. */
    const val PITCH_UP_MAX = 0.42f

    /** ... and at or above which it counts as tilted down. */
    const val PITCH_DOWN_MIN = 0.62f

    /**
     * Classifies the head direction as seen on screen, or null for a frontal face. [mirroredPreview]
     * flips left and right for the front camera, whose preview is shown mirrored.
     */
    fun classify(landmarks: FaceLandmarks, mirroredPreview: Boolean): HeadDirection? {
        val eyeCenterX = (landmarks.leftEye.x + landmarks.rightEye.x) / 2f
        val eyeCenterY = (landmarks.leftEye.y + landmarks.rightEye.y) / 2f
        val eyeDx = landmarks.rightEye.x - landmarks.leftEye.x
        val eyeDy = landmarks.rightEye.y - landmarks.leftEye.y
        val eyeDistance = sqrt(eyeDx * eyeDx + eyeDy * eyeDy).coerceAtLeast(1e-6f)
        val mouthCenterY = (landmarks.mouthLeft.y + landmarks.mouthRight.y) / 2f
        val eyeToMouth = mouthCenterY - eyeCenterY
        if (abs(eyeToMouth) < 1e-6f) return null

        var yaw = (landmarks.noseBase.x - eyeCenterX) / eyeDistance
        if (mirroredPreview) yaw = -yaw
        val pitch = (landmarks.noseBase.y - eyeCenterY) / eyeToMouth
        return when {
            yaw <= -YAW_THRESHOLD -> HeadDirection.LEFT
            yaw >= YAW_THRESHOLD -> HeadDirection.RIGHT
            pitch <= PITCH_UP_MAX -> HeadDirection.UP
            pitch >= PITCH_DOWN_MIN -> HeadDirection.DOWN
            else -> null
        }
    }
}
