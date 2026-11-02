package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.geometry.Point
import kotlin.math.abs
import kotlin.math.sqrt

/** Head directions collected during registration, in the order the guide suggests them. */
enum class HeadDirection(val label: String, val arrow: String, val prompt: String, val progressLabel: String) {
    FRONT("정면", "•", "정면을 바라봐 주세요.", "정면 얼굴"),
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

/**
 * How far the head is turned, from the landmarks. [yaw] is the nose offset from the eye midpoint in
 * eye distances, as seen on screen (negative = turned to the screen's left; about 0.2 at 20°, 0.55
 * at 45°). [pitch] is the nose height between the eyes (0) and the mouth (1); lower is looking up.
 */
data class HeadAngles(val yaw: Float, val pitch: Float)

/** What a head pose counts as during registration. */
sealed class Pose {
    data class Facing(val direction: HeadDirection) : Pose()

    /** Between two directions (a little turned, not enough for either). */
    data object Between : Pose()

    /** Turned so far that the landmarks, and so the sample, are unreliable. */
    data object TooFar : Pose()
}

/**
 * Where each direction starts and ends. Front is a narrow band around looking straight ahead; the
 * turned directions have an outer limit, past which ML Kit's landmarks get unreliable.
 */
data class PoseLimits(
    val frontYaw: Float = 0.08f,
    val turnMin: Float = 0.18f,
    val turnMax: Float = 0.60f,
    val upFrom: Float = 0.42f,
    val upMax: Float = 0.12f,
    val downFrom: Float = 0.62f,
    val downMax: Float = 0.88f,
    val frontPitchLow: Float = 0.45f,
    val frontPitchHigh: Float = 0.59f,
) {
    fun classify(angles: HeadAngles): Pose {
        val yaw = angles.yaw
        val pitch = angles.pitch
        if (abs(yaw) > turnMax || pitch < upMax || pitch > downMax) return Pose.TooFar
        return when {
            yaw <= -turnMin -> Pose.Facing(HeadDirection.LEFT)
            yaw >= turnMin -> Pose.Facing(HeadDirection.RIGHT)
            pitch <= upFrom -> Pose.Facing(HeadDirection.UP)
            pitch >= downFrom -> Pose.Facing(HeadDirection.DOWN)
            abs(yaw) <= frontYaw && pitch in frontPitchLow..frontPitchHigh -> Pose.Facing(HeadDirection.FRONT)
            else -> Pose.Between
        }
    }
}

object HeadPose {
    /**
     * The head angles as seen on screen, or null when the landmarks are degenerate.
     * [mirroredPreview] flips left and right for the front camera, whose preview is shown mirrored.
     */
    fun measure(landmarks: FaceLandmarks, mirroredPreview: Boolean): HeadAngles? {
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
        return HeadAngles(yaw, (landmarks.noseBase.y - eyeCenterY) / eyeToMouth)
    }

    fun classify(landmarks: FaceLandmarks, mirroredPreview: Boolean, limits: PoseLimits = PoseLimits()): Pose? =
        measure(landmarks, mirroredPreview)?.let { limits.classify(it) }
}
