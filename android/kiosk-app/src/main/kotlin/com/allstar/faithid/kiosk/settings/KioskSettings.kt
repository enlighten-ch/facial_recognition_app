package com.allstar.faithid.kiosk.settings

import com.allstar.faithid.core.recognition.RecognitionThresholds
import com.allstar.faithid.core.recognition.RegistrationConfig
import com.allstar.faithid.kiosk.flow.FlowConfig
import com.allstar.faithid.kiosk.flow.FlowMessages

data class PaperProfile(
    val id: String,
    val name: String,
    val widthMm: Float,
    val heightMm: Float,
)

object PrinterModel {
    const val SYSTEM = "system"
    const val NEMONIC = "nemonic"
    const val BROTHER_QL820NWB = "brother_ql820nwb"

    fun sanitize(value: String?): String = when (value) {
        NEMONIC, BROTHER_QL820NWB -> value
        else -> SYSTEM
    }
}

object GuideAudience {
    const val KIDS = "kids"
    const val ADULTS = "adults"

    fun sanitize(value: String?): String = if (value == ADULTS) ADULTS else KIDS
}

/** One text line on the label ("속성"): which column, how big, and where. */
data class LabelLine(
    val column: String,
    val textSizeSp: Int,
    val offsetX: Int,
    val offsetY: Int,
)

/** All kiosk settings. Stored as JSON in shared preferences by [KioskSettingsStore]. */
data class KioskSettings(
    // Screen
    val developerViewEnabled: Boolean = false,
    val hideSystemBars: Boolean = true,
    /** Who the operation screen talks to: [GuideAudience.KIDS] (빛이, stars, playful words) or [GuideAudience.ADULTS]. */
    val guideAudience: String = GuideAudience.KIDS,

    // Recognition
    val authSimThreshold: Float = 0.40f,
    val authMarginThreshold: Float = 0.10f,
    val topK: Int = 3,
    val analyzeEveryNFrames: Int = 6,
    val ambiguousCandidateSwitchFrames: Int = 3,
    val selectedCandidateGraceFrames: Int = 5,
    val cameraIndex: Int = 0,
    /** Aligns faces to the ArcFace template before embedding. Changes embeddings: re-register everyone after enabling. */
    val faceAlignmentEnabled: Boolean = false,
    /** Face circle (guide oval) half-widths relative to the screen, set below the guide's speech bubble. */
    val registerEllipseAxisXRatio: Float = 0.17f,
    val registerEllipseAxisYRatio: Float = 0.27f,
    val registerEllipseCenterYOffsetRatio: Float = 0.06f,
    val registerEllipseOutsideDimAlpha: Float = 0.35f,

    // Registration
    val registerSamplesPerDirection: Int = 3,
    /** Normal, big and small face circle for the samples of each direction (closer, then further). */
    val registerCircleSteps: Boolean = true,
    val registerBigCircleScale: Float = 1.18f,
    val registerSmallCircleScale: Float = 0.73f,
    val registerCaptureEveryNFrames: Int = 4,
    val registerSimilarityLowerBoundary: Float = 0.35f,
    val registerSimilarityHigherBoundary: Float = 0.85f,
    val duplicateThreshold: Float = 0.45f,
    val maxEmbeddingsPerPerson: Int = 100,
    val adaptiveUpdateEnabled: Boolean = true,
    val adaptiveUpdateMaxSim: Float = 0.62f,
    val adaptiveUpdateMinIntervalFrames: Int = 30,
    val adaptiveUpdateMaxSamplesPerSession: Int = 20,
    /** Downloaded database changes are applied once no face has been seen for this long. */
    val adaptiveUpdateIdleSeconds: Int = 5,

    // Server
    val dbSyncEnabled: Boolean = true,
    val dbServerBaseUrl: String = "http://127.0.0.1:18080",
    val dbConnectRetryDelaySeconds: Int = 3,
    val dbConnectTimeoutSeconds: Int = 6,
    val dbClientId: String = "",
    val dbClientSecret: String = "",
    val dbDeviceName: String = "",

    // Printer
    val printEnabled: Boolean = true,
    val printerModel: String = PrinterModel.SYSTEM,
    val printerName: String = "",
    val brotherPrinterMacAddress: String = "",
    val nemonicPrinterMacAddress: String = "",
    val printerTimeoutSec: Int = 10,
    val printerCopies: Int = 1,
    val printerJobTitlePrefix: String = "FaceRecognition",
    val paperProfiles: List<PaperProfile> = DEFAULT_PAPER_PROFILES,
    val selectedPaperId: String = DEFAULT_PAPER_ID,
    val printLandscape: Boolean = false,

    // Label template
    val printerTemplateImageUri: String = "",
    val templateImageRotationQuarterTurns: Int = 0,
    val templateShrinkToPaperIfOversized: Boolean = true,
    val paperLandscape: Boolean = false,
    val templateTextRegionOffsetX: Int = 24,
    val templateTextRegionOffsetY: Int = 24,
    val templateTextRegionWidth: Int = 528,
    val templateTextRegionHeight: Int = 352,
    /** Lines are placed one below the other by their real height. Off for kiosks configured before 2026-10 (fixed positions). */
    val printLayoutStacked: Boolean = true,
    val printTextScaleMultiplier: Int = 2,
    val printAttributeCount: Int = 3,
    val labelLines: List<LabelLine> = DEFAULT_LABEL_LINES,

    // Messages
    val printStatusReadyMessage: String = "출력을 준비하고 있습니다.",
    val printStatusPermissionMessage: String = "출력 권한을 확인하고 있습니다.",
    val printStatusConnectingMessage: String = "프린터와 연결하고 있습니다.",
    val printStatusPreparingMessage: String = "출력 내용을 준비하고 있습니다.",
    val printStatusSendingMessage: String = "출력 중입니다. 잠시만 기다려 주세요.",
    val printStatusSuccessMessage: String = "출력 요청을 보냈습니다.",
    val printStatusErrorMessage: String = "출력에 문제가 있습니다. 오류로그를 확인해 주세요.",
    val outputReturnDelaySeconds: Int = 3,
    val messages: FlowMessages = FlowMessages(),
    /** The guide's lines changed in "안내 문구", keyed "<audience>.<line>" (see GuideCopy.LINES); blank keys use the defaults. */
    val guideLines: Map<String, String> = emptyMap(),
    val uiSavingMessage: String = "저장 중입니다. 잠시만 기다려주세요...",
    val uiOutputButtonText: String = "이름표 출력하기",

    /** Folder of the pre-2026-10 kiosk database, imported once when synchronization is off. */
    val legacyFaceDataDir: String = "/sdcard/faceDB",
) {
    val selectedPaper: PaperProfile
        get() = paperProfiles.firstOrNull { it.id == selectedPaperId } ?: paperProfiles.firstOrNull() ?: DEFAULT_PAPER_PROFILES.first()

    val activeLabelLines: List<LabelLine> get() = labelLines.take(printAttributeCount.coerceIn(1, 3))

    fun flowConfig(): FlowConfig = FlowConfig(
        thresholds = RecognitionThresholds(
            minScore = authSimThreshold,
            minMargin = authMarginThreshold,
            candidateCount = topK.coerceAtLeast(2),
            adaptiveMaxScore = adaptiveUpdateMaxSim,
        ),
        registration = RegistrationConfig(
            samplesPerDirection = registerSamplesPerDirection.coerceAtLeast(1),
            minSimilarity = registerSimilarityLowerBoundary,
            maxSimilarity = registerSimilarityHigherBoundary,
            duplicateThreshold = duplicateThreshold,
            circleSteps = registerCircleSteps,
            bigCircleScale = registerBigCircleScale,
            smallCircleScale = registerSmallCircleScale,
        ),
        candidateSwitchFrames = ambiguousCandidateSwitchFrames,
        selectedGraceFrames = selectedCandidateGraceFrames,
        adaptiveEnabled = adaptiveUpdateEnabled,
        adaptiveMaxPerSession = adaptiveUpdateMaxSamplesPerSession,
        adaptiveMinIntervalFrames = adaptiveUpdateMinIntervalFrames,
        outputReturnSeconds = outputReturnDelaySeconds,
        messages = messages,
    )

    companion object {
        const val DEFAULT_PAPER_ID = "nemonic_72x50"
        val DEFAULT_PAPER_PROFILES = listOf(PaperProfile(DEFAULT_PAPER_ID, "Nemonic 72x50", 72f, 50f))
        val DEFAULT_LABEL_LINES = listOf(
            LabelLine("name", 44, 0, 0),
            LabelLine("department", 24, 0, 0),
            LabelLine("grade", 20, 0, 0),
        )
        val LABEL_COLUMNS = listOf("name" to "이름", "department" to "부서", "grade" to "학년")
    }
}
