package com.allstar.faithid.kiosk.flow

import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadAngles
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.core.recognition.Pose
import com.allstar.faithid.core.recognition.RecognitionThresholds
import com.allstar.faithid.core.recognition.RegistrationConfig
import com.allstar.faithid.core.recognition.RegistrationCell

enum class Screen { MONITOR, REGISTER_CAPTURE, REGISTER_INFO, OUTPUT }

/** Texts the administrator can change in the kiosk settings ("UI 문구"). */
data class FlowMessages(
    val welcome: String = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요",
    val serverDisconnected: String = "서버 앱 실행 및 연동 후 사용해주세요",
    val serverDisconnectedDetail: String = "DB 서버 연결 이후 인식 기능이 자동으로 활성화됩니다.",
    val awaitingApproval: String = "이 키오스크({clientId})는 서버 승인을 기다리고 있습니다",
    val awaitingApprovalDetail: String = "서버 태블릿의 Faith DB Server 앱 › 현황 탭에서 승인해 주세요.",
    val unregistered: String = "얼굴이 등록되어 있지 않아요. 등록을 시작할까요?",
    val unregisteredDetail: String = "등록 시작하기 또는 이름표만 출력하기를 선택해 주세요.",
    val recognized: String = "반가워요 {name}! 이름 출력을 위해 버튼을 눌러주세요",
    val ambiguous: String = "유사한 얼굴이 있습니다. 이름을 선택해 주세요.",
    val ambiguousDetail: String = "후보 중 이름 선택 후 이름 출력하기 버튼을 눌러주세요.",
    val selectedCandidate: String = "선택됨: {label} / 이름 출력하기 버튼을 눌러주세요.",
    val registerComplete: String = "얼굴등록이 완료되었어요! 이름을 입력해주세요",
    val labelOnlyInput: String = "등록없이 이름표만 출력할게요.\n출력할 내용을 입력해주세요.",
    val duplicateFound: String = "이미 등록된 분과 얼굴이 매우 비슷해요: {label}",
    val duplicateDetail: String = "본인이 맞으면 '본인이 맞아요'를, 다른 사람이면 '처음 등록이에요'를 눌러 주세요.",
    val returnCountdown: String = "{seconds}초 후 기본 화면으로 복귀합니다",
)

data class FlowConfig(
    val thresholds: RecognitionThresholds = RecognitionThresholds(),
    val registration: RegistrationConfig = RegistrationConfig(),
    val candidateSwitchFrames: Int = 3,
    val selectedGraceFrames: Int = 5,
    val adaptiveEnabled: Boolean = true,
    val adaptiveMaxPerSession: Int = 20,
    val adaptiveMinIntervalFrames: Int = 30,
    val outputReturnSeconds: Int = 3,
    val messages: FlowMessages = FlowMessages(),
)

/** Whether the kiosk can recognize people right now. */
sealed class Readiness {
    /** Recognition works (online, offline with a local database, or synchronization turned off). */
    data object Ready : Readiness()

    /** No local database yet and the server has not approved this kiosk. */
    data class WaitingForApproval(val clientId: String) : Readiness()

    /** No local database yet and the server cannot be reached. */
    data class NoServer(val reason: String) : Readiness()
}

/** One analyzed camera frame, as far as the flow is concerned. */
sealed class Observation {
    data object NoFace : Observation()

    /**
     * A face inside the guide ellipse. [direction] is the classified direction (FRONT for a frontal
     * face, null when unknown or in between); [pose] adds "turned too far" for registration.
     * [size] is the face size relative to the normal face circle as drawn (1 = fits it), if known.
     * [sharpness] ranks the frames of a registration hold.
     */
    class Face(
        val embedding: FloatArray,
        val direction: HeadDirection?,
        val size: Float? = null,
        val pose: Pose? = direction?.let { Pose.Facing(it) },
        val angles: HeadAngles? = null,
        val sharpness: Float = 0f,
    ) : Observation()

    /** Registration only: a face is visible but outside the circle; [screenDx] < 0 means it is left of the circle on screen. */
    data class Outside(val screenDx: Float) : Observation()

    /** Registration only: [count] faces are in front of the camera. */
    data class Several(val count: Int) : Observation()
}

data class CandidateView(
    val personKey: String,
    val label: String,
    val score: Float,
    val selected: Boolean,
    val metadata: PersonMetadata? = null,
)

/** Why the last registration frame was not kept, for the guide's words. */
enum class CaptureProblem { NO_FACE, OUTSIDE_LEFT, OUTSIDE_RIGHT, SEVERAL, TOO_TURNED, DIRECTION, COME_CLOSER, STEP_BACK, TOO_SIMILAR, OTHER_PERSON }

data class RegistrationView(
    /** Every cell to collect, in the order the guide suggests them, and whether it is collected. */
    val cells: List<Pair<RegistrationCell, Boolean>>,
    val total: Int,
    val target: Int,
    /** The suggested cell (null when complete). */
    val next: HeadDirection?,
    val circle: FaceCircle? = null,
    /** 0..1 while the face stays in an empty cell; the sample is kept at 1. */
    val holding: Float = 0f,
    val problem: CaptureProblem? = null,
    /** Several frames in a row were not kept; the screen gives the reason more weight. */
    val stalled: Boolean = false,
) {
    val complete: Boolean get() = total >= target

    fun collected(direction: HeadDirection): Int = cells.count { it.first.direction == direction && it.second }
}

data class OutputView(val metadata: PersonMetadata, val secondsLeft: Int, val printStatus: String)

/** Everything the screen shows. Rendering is a pure function of this value. */
data class UiState(
    val screen: Screen = Screen.MONITOR,
    val message: String = "",
    val detail: String = "",
    val candidates: List<CandidateView> = emptyList(),
    val showRegisterChoice: Boolean = false,
    val printButtonVisible: Boolean = false,
    val printButtonEnabled: Boolean = false,
    val registration: RegistrationView? = null,
    val duplicateOf: CandidateView? = null,
    val labelOnly: Boolean = false,
    val saving: Boolean = false,
    val output: OutputView? = null,
    val banner: String = "",
    /** False while the kiosk waits for the server (no database yet, or not approved). */
    val ready: Boolean = true,
    /** Set on the first screen after a registration was saved, so the screen can congratulate. */
    val registered: PersonMetadata? = null,
    /** After "나 아닌데?": recognition pauses for a moment, then looks again. */
    val retrying: Boolean = false,
)

/** A check-in, recorded when the print button is pressed (whether or not printing succeeds). */
data class CheckIn(val personKey: String, val metadata: PersonMetadata, val source: CheckInSource, val score: Float)

/** Side effects of the flow, implemented by the app (database, printer, log). */
interface FlowEffects {
    fun addSamples(personKey: String, samples: List<FloatArray>)

    /** Registers a person; returns the new key, or null if saving failed. */
    fun register(metadata: PersonMetadata, samples: List<FloatArray>): String?

    fun checkIn(checkIn: CheckIn)

    fun print(metadata: PersonMetadata)

    fun log(message: String)
}

internal fun String.fill(vararg values: Pair<String, String>): String =
    values.fold(this) { text, (key, value) -> text.replace("{$key}", value) }
