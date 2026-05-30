package com.example.facetestbed

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class PaperProfile(
    val id: String,
    val name: String,
    val widthMm: Float,
    val heightMm: Float,
)

private const val DEFAULT_PAPER_ID = "nemonic_72x50"
private val DEFAULT_PAPER_PROFILES = listOf(
    PaperProfile(
        id = DEFAULT_PAPER_ID,
        name = "Nemonic 72x50",
        widthMm = 72f,
        heightMm = 50f,
    ),
)

data class AppSettings(
    val faceDataDir: String = "/sdcard/faceDB",
    val developerViewEnabled: Boolean = false,
    val authSimThreshold: Float = 0.40f,
    val authMarginThreshold: Float = 0.10f,
    val topK: Int = 3,
    val analyzeEveryNFrames: Int = 6,
    val ambiguousCandidateSwitchFrames: Int = 3,
    val selectedCandidateGraceFrames: Int = 5,
    val cameraIndex: Int = 0,
    val recognitionUseEllipseMask: Boolean = false,
    val registerSamplesPerDirection: Int = 3,
    val registerCaptureEveryNFrames: Int = 4,
    val registerSimilarityLowerBoundary: Float = 0.35f,
    val registerSimilarityHigherBoundary: Float = 0.85f,
    val maxEmbeddingsPerPerson: Int = 100,
    val registerEllipseAxisXRatio: Float = 0.22f,
    val registerEllipseAxisYRatio: Float = 0.30f,
    val registerEllipseCenterYOffsetRatio: Float = 0.00f,
    val registerEllipseOutsideDimAlpha: Float = 0.60f,
    val adaptiveUpdateEnabled: Boolean = true,
    val adaptiveUpdateMaxSim: Float = 0.62f,
    val adaptiveUpdateMinIntervalFrames: Int = 30,
    val adaptiveUpdateMaxSamplesPerSession: Int = 20,
    val adaptiveUpdateIdleSeconds: Int = 5,
    val dbSyncEnabled: Boolean = true,
    val dbServerBaseUrl: String = "http://127.0.0.1:18080",
    val dbConnectRetryCount: Int = 10,
    val dbConnectRetryDelaySeconds: Int = 3,
    val dbConnectTimeoutSeconds: Int = 6,
    val dbClientId: String = "",
    val dbAuthToken: String = "",
    val dbDeviceName: String = "",
    val printEnabled: Boolean = true,
    val printerModel: String = PrinterModel.SYSTEM,
    val printerName: String = "",
    val brotherPrinterMacAddress: String = "",
    val nemonicPrinterMacAddress: String = "",
    val printerTimeoutSec: Int = 10,
    val printerCopies: Int = 1,
    val printerJobTitlePrefix: String = "FaceRecognition",
    val printerContentTemplate: String = "Name: {name}\nDepartment: {department}\n",
    val printerTemplateImageUri: String = "",
    val templateImageRotationQuarterTurns: Int = 0,
    val templateShrinkToPaperIfOversized: Boolean = true,
    val paperProfiles: List<PaperProfile> = DEFAULT_PAPER_PROFILES,
    val selectedPaperId: String = DEFAULT_PAPER_ID,
    val printLandscape: Boolean = false,
    val paperLandscape: Boolean = false,
    val templateTextRegionOffsetX: Int = 24,
    val templateTextRegionOffsetY: Int = 24,
    val templateTextRegionWidth: Int = 360,
    val templateTextRegionHeight: Int = 180,
    val printTextBaseFontSize: Int = 48,
    val printTextScaleMultiplier: Int = 5,
    val printAttributeCount: Int = 2,
    val printAttribute1Column: String = "name",
    val printAttribute1TextSizeSp: Int = 40,
    val printAttribute1OffsetX: Int = 0,
    val printAttribute1OffsetY: Int = 0,
    val printAttribute2Column: String = "department",
    val printAttribute2TextSizeSp: Int = 24,
    val printAttribute2OffsetX: Int = 0,
    val printAttribute2OffsetY: Int = 0,
    val printAttribute3Column: String = "grade",
    val printAttribute3TextSizeSp: Int = 20,
    val printAttribute3OffsetX: Int = 0,
    val printAttribute3OffsetY: Int = 36,
    val outputNameTextSizeSp: Int = 40,
    val outputNameOffsetX: Int = 0,
    val outputNameOffsetY: Int = 0,
    val outputDepartmentTextSizeSp: Int = 24,
    val outputDepartmentOffsetX: Int = 0,
    val outputDepartmentOffsetY: Int = 0,
    val printStatusReadyMessage: String = "출력을 준비하고 있습니다.",
    val printStatusPermissionMessage: String = "출력 권한을 확인하고 있습니다.",
    val printStatusConnectingMessage: String = "프린터와 연결하고 있습니다.",
    val printStatusPreparingMessage: String = "출력 내용을 준비하고 있습니다.",
    val printStatusSendingMessage: String = "출력 중입니다. 잠시만 기다려 주세요.",
    val printStatusSuccessMessage: String = "출력 요청을 보냈습니다.",
    val printStatusErrorMessage: String = "출력에 문제가 있습니다. 오류로그를 확인해 주세요.",
    val outputReturnDelaySeconds: Int = 3,
    val outputReturnMessageTemplate: String = "{seconds}초 후 기본 화면으로 복귀합니다",
    val uiWelcomeMessage: String = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요",
    val uiServerDisconnectedMessage: String = "서버 앱 실행 및 연동 후 사용해주세요",
    val uiServerDisconnectedDetail: String = "DB 서버 연결 이후 인식 기능이 자동으로 활성화됩니다.",
    val uiUnregisteredMessage: String = "얼굴이 등록되어 있지 않아요. 등록을 시작할까요?",
    val uiUnregisteredDetail: String = "등록 시작하기 또는 이름표만 출력하기를 선택해 주세요.",
    val uiRecognizedMessageTemplate: String = "반가워요 {name}! 이름 출력을 위해 버튼을 눌러주세요",
    val uiAmbiguousMessage: String = "유사한 얼굴이 있습니다. 이름을 선택해 주세요.",
    val uiAmbiguousDetail: String = "후보 중 이름 선택 후 이름 출력하기 버튼을 눌러주세요.",
    val uiSelectedCandidateTemplate: String = "선택됨: {label} / 이름 출력하기 버튼을 눌러주세요.",
    val uiRegisterCompleteMessage: String = "얼굴등록이 완료되었어요! 이름을 입력해주세요",
    val uiLabelOnlyInputMessage: String = "등록없이 이름표만 출력할게요.\n출력할 내용을 입력해주세요.",
    val uiSavingMessage: String = "저장 중입니다. 잠시만 기다려주세요...",
    val uiOutputButtonText: String = "이름표 출력하기",
)

object PrinterModel {
    const val SYSTEM = "system"
    const val NEMONIC = "nemonic"
    const val BROTHER_QL820NWB = "brother_ql820nwb"

    fun sanitize(value: String?): String {
        return when (value) {
            NEMONIC -> NEMONIC
            BROTHER_QL820NWB -> BROTHER_QL820NWB
            else -> SYSTEM
        }
    }
}

class AppSettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)

    fun load(): AppSettings {
        val raw = prefs.getString(KEY_JSON, null) ?: return AppSettings()
        return runCatching {
            val o = JSONObject(raw)
            AppSettings(
                faceDataDir = o.optString("FACE_DATA_DIR", "/sdcard/faceDB"),
                developerViewEnabled = o.optBoolean("DEVELOPER_VIEW_ENABLED", false),
                authSimThreshold = o.optDouble("AUTH_SIM_THRESHOLD", 0.40).toFloat(),
                authMarginThreshold = o.optDouble("AUTH_MARGIN_THRESHOLD", 0.10).toFloat(),
                topK = o.optInt("TOP_K", 3),
                analyzeEveryNFrames = o.optInt("ANALYZE_EVERY_N_FRAMES", 6),
                ambiguousCandidateSwitchFrames = o.optInt("AMBIGUOUS_CANDIDATE_SWITCH_FRAMES", 3),
                selectedCandidateGraceFrames = o.optInt("SELECTED_CANDIDATE_GRACE_FRAMES", 5),
                cameraIndex = o.optInt("CAMERA_INDEX", 0),
                recognitionUseEllipseMask = o.optBoolean("RECOGNITION_USE_ELLIPSE_MASK", false),
                registerSamplesPerDirection = o.optInt("REGISTER_SAMPLES_PER_DIRECTION", 3),
                registerCaptureEveryNFrames = o.optInt("REGISTER_CAPTURE_EVERY_N_FRAMES", 4),
                registerSimilarityLowerBoundary = o.optDouble("REGISTER_SIMILARITY_LOWER_BOUNDARY", 0.35).toFloat(),
                registerSimilarityHigherBoundary = o.optDouble("REGISTER_SIMILARITY_HIGHER_BOUNDARY", 0.85).toFloat(),
                maxEmbeddingsPerPerson = o.optInt("MAX_EMBEDDINGS_PER_PERSON", 100),
                registerEllipseAxisXRatio = o.optDouble("REGISTER_ELLIPSE_AXIS_X_RATIO", 0.22).toFloat(),
                registerEllipseAxisYRatio = o.optDouble("REGISTER_ELLIPSE_AXIS_Y_RATIO", 0.30).toFloat(),
                registerEllipseCenterYOffsetRatio = o.optDouble("REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO", 0.0).toFloat(),
                registerEllipseOutsideDimAlpha = o.optDouble("REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA", 0.60).toFloat(),
                adaptiveUpdateEnabled = o.optBoolean("ADAPTIVE_UPDATE_ENABLED", true),
                adaptiveUpdateMaxSim = o.optDouble("ADAPTIVE_UPDATE_MAX_SIM", 0.62).toFloat(),
                adaptiveUpdateMinIntervalFrames = o.optInt("ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES", 30),
                adaptiveUpdateMaxSamplesPerSession = o.optInt("ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION", 20),
                adaptiveUpdateIdleSeconds = o.optInt("ADAPTIVE_UPDATE_IDLE_SECONDS", 5),
                dbSyncEnabled = o.optBoolean("DB_SYNC_ENABLED", true),
                dbServerBaseUrl = o.optString("DB_SERVER_BASE_URL", "http://127.0.0.1:18080"),
                dbConnectRetryCount = o.optInt("DB_CONNECT_RETRY_COUNT", 10),
                dbConnectRetryDelaySeconds = o.optInt("DB_CONNECT_RETRY_DELAY_SECONDS", 3),
                dbConnectTimeoutSeconds = o.optInt("DB_CONNECT_TIMEOUT_SECONDS", 6),
                dbClientId = o.optString("DB_CLIENT_ID", ""),
                dbAuthToken = o.optString("DB_AUTH_TOKEN", ""),
                dbDeviceName = o.optString("DB_DEVICE_NAME", ""),
                printEnabled = o.optBoolean("PRINT_ENABLED", true),
                printerModel = PrinterModel.sanitize(o.optString("PRINTER_MODEL", PrinterModel.SYSTEM)),
                printerName = o.optString("PRINTER_NAME", ""),
                brotherPrinterMacAddress = o.optString("BROTHER_PRINTER_MAC_ADDRESS", ""),
                nemonicPrinterMacAddress = o.optString("NEMONIC_PRINTER_MAC_ADDRESS", ""),
                printerTimeoutSec = o.optInt("PRINTER_TIMEOUT_SEC", 10),
                printerCopies = o.optInt("PRINTER_COPIES", 1),
                printerJobTitlePrefix = o.optString("PRINTER_JOB_TITLE_PREFIX", "FaceRecognition"),
                printerContentTemplate = o.optString("PRINTER_CONTENT_TEMPLATE", "Name: {name}\nDepartment: {department}\n"),
                printerTemplateImageUri = o.optString("PRINTER_TEMPLATE_IMAGE_URI", ""),
                templateImageRotationQuarterTurns = o.optInt("TEMPLATE_IMAGE_ROTATION_QUARTER_TURNS", 0).mod(4),
                templateShrinkToPaperIfOversized = o.optBoolean("TEMPLATE_SHRINK_TO_PAPER_IF_OVERSIZED", true),
                paperProfiles = parsePaperProfiles(o.optJSONArray("PAPER_PROFILES")),
                selectedPaperId = o.optString("SELECTED_PAPER_ID", DEFAULT_PAPER_ID),
                printLandscape = o.optBoolean("PRINT_LANDSCAPE", false),
                paperLandscape = o.optBoolean("PAPER_LANDSCAPE", false),
                templateTextRegionOffsetX = o.optInt("TEMPLATE_TEXT_REGION_OFFSET_X", 24),
                templateTextRegionOffsetY = o.optInt("TEMPLATE_TEXT_REGION_OFFSET_Y", 24),
                templateTextRegionWidth = o.optInt("TEMPLATE_TEXT_REGION_WIDTH", 360),
                templateTextRegionHeight = o.optInt("TEMPLATE_TEXT_REGION_HEIGHT", 180),
                printTextBaseFontSize = o.optInt("PRINT_TEXT_BASE_FONT_SIZE", 48),
                printTextScaleMultiplier = o.optInt("PRINT_TEXT_SCALE_MULTIPLIER", 5),
                printAttributeCount = o.optInt("PRINT_ATTRIBUTE_COUNT", 2),
                printAttribute1Column = o.optString("PRINT_ATTRIBUTE_1_COLUMN", "name"),
                printAttribute1TextSizeSp = o.optInt("PRINT_ATTRIBUTE_1_TEXT_SIZE_SP", o.optInt("OUTPUT_NAME_TEXT_SIZE_SP", 40)),
                printAttribute1OffsetX = o.optInt("PRINT_ATTRIBUTE_1_OFFSET_X", o.optInt("OUTPUT_NAME_OFFSET_X", 0)),
                printAttribute1OffsetY = o.optInt("PRINT_ATTRIBUTE_1_OFFSET_Y", o.optInt("OUTPUT_NAME_OFFSET_Y", 0)),
                printAttribute2Column = o.optString("PRINT_ATTRIBUTE_2_COLUMN", "department"),
                printAttribute2TextSizeSp = o.optInt("PRINT_ATTRIBUTE_2_TEXT_SIZE_SP", o.optInt("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", 24)),
                printAttribute2OffsetX = o.optInt("PRINT_ATTRIBUTE_2_OFFSET_X", o.optInt("OUTPUT_DEPARTMENT_OFFSET_X", 0)),
                printAttribute2OffsetY = o.optInt("PRINT_ATTRIBUTE_2_OFFSET_Y", o.optInt("OUTPUT_DEPARTMENT_OFFSET_Y", 0)),
                printAttribute3Column = o.optString("PRINT_ATTRIBUTE_3_COLUMN", "grade"),
                printAttribute3TextSizeSp = o.optInt("PRINT_ATTRIBUTE_3_TEXT_SIZE_SP", 20),
                printAttribute3OffsetX = o.optInt("PRINT_ATTRIBUTE_3_OFFSET_X", 0),
                printAttribute3OffsetY = o.optInt("PRINT_ATTRIBUTE_3_OFFSET_Y", 36),
                outputNameTextSizeSp = o.optInt("OUTPUT_NAME_TEXT_SIZE_SP", 40),
                outputNameOffsetX = o.optInt("OUTPUT_NAME_OFFSET_X", 0),
                outputNameOffsetY = o.optInt("OUTPUT_NAME_OFFSET_Y", 0),
                outputDepartmentTextSizeSp = o.optInt("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", 24),
                outputDepartmentOffsetX = o.optInt("OUTPUT_DEPARTMENT_OFFSET_X", 0),
                outputDepartmentOffsetY = o.optInt("OUTPUT_DEPARTMENT_OFFSET_Y", 0),
                printStatusReadyMessage = o.optString("PRINT_STATUS_READY_MESSAGE", "출력을 준비하고 있습니다."),
                printStatusPermissionMessage = o.optString("PRINT_STATUS_PERMISSION_MESSAGE", "출력 권한을 확인하고 있습니다."),
                printStatusConnectingMessage = o.optString("PRINT_STATUS_CONNECTING_MESSAGE", "프린터와 연결하고 있습니다."),
                printStatusPreparingMessage = o.optString("PRINT_STATUS_PREPARING_MESSAGE", "출력 내용을 준비하고 있습니다."),
                printStatusSendingMessage = o.optString("PRINT_STATUS_SENDING_MESSAGE", "출력 중입니다. 잠시만 기다려 주세요."),
                printStatusSuccessMessage = o.optString("PRINT_STATUS_SUCCESS_MESSAGE", "출력 요청을 보냈습니다."),
                printStatusErrorMessage = o.optString("PRINT_STATUS_ERROR_MESSAGE", "출력에 문제가 있습니다. 오류로그를 확인해 주세요."),
                outputReturnDelaySeconds = o.optInt("OUTPUT_RETURN_DELAY_SECONDS", 3),
                outputReturnMessageTemplate = o.optString("OUTPUT_RETURN_MESSAGE_TEMPLATE", "{seconds}초 후 기본 화면으로 복귀합니다"),
                uiWelcomeMessage = o.optString("UI_WELCOME_MESSAGE", "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요"),
                uiServerDisconnectedMessage = o.optString("UI_SERVER_DISCONNECTED_MESSAGE", "서버 앱 실행 및 연동 후 사용해주세요"),
                uiServerDisconnectedDetail = o.optString("UI_SERVER_DISCONNECTED_DETAIL", "DB 서버 연결 이후 인식 기능이 자동으로 활성화됩니다."),
                uiUnregisteredMessage = o.optString("UI_UNREGISTERED_MESSAGE", "얼굴이 등록되어 있지 않아요. 등록을 시작할까요?"),
                uiUnregisteredDetail = o.optString("UI_UNREGISTERED_DETAIL", "등록 시작하기 또는 이름표만 출력하기를 선택해 주세요."),
                uiRecognizedMessageTemplate = o.optString("UI_RECOGNIZED_MESSAGE_TEMPLATE", "반가워요 {name}! 이름 출력을 위해 버튼을 눌러주세요"),
                uiAmbiguousMessage = o.optString("UI_AMBIGUOUS_MESSAGE", "유사한 얼굴이 있습니다. 이름을 선택해 주세요."),
                uiAmbiguousDetail = o.optString("UI_AMBIGUOUS_DETAIL", "후보 중 이름 선택 후 이름 출력하기 버튼을 눌러주세요."),
                uiSelectedCandidateTemplate = o.optString("UI_SELECTED_CANDIDATE_TEMPLATE", "선택됨: {label} / 이름 출력하기 버튼을 눌러주세요."),
                uiRegisterCompleteMessage = o.optString("UI_REGISTER_COMPLETE_MESSAGE", "얼굴등록이 완료되었어요! 이름을 입력해주세요"),
                uiLabelOnlyInputMessage = o.optString("UI_LABEL_ONLY_INPUT_MESSAGE", "등록없이 이름표만 출력할게요.\n출력할 내용을 입력해주세요.")
                    .let { if (it == "이름표에 들어갈 정보를 입력해 주세요.\n얼굴 임베딩은 등록하거나 업데이트하지 않습니다.") "등록없이 이름표만 출력할게요.\n출력할 내용을 입력해주세요." else it },
                uiSavingMessage = o.optString("UI_SAVING_MESSAGE", "저장 중입니다. 잠시만 기다려주세요..."),
                uiOutputButtonText = o.optString("UI_OUTPUT_BUTTON_TEXT", "이름표 출력하기"),
            )
        }.getOrDefault(AppSettings())
    }

    fun save(settings: AppSettings) {
        val o = JSONObject()
        o.put("FACE_DATA_DIR", settings.faceDataDir)
        o.put("DEVELOPER_VIEW_ENABLED", settings.developerViewEnabled)
        o.put("AUTH_SIM_THRESHOLD", settings.authSimThreshold.toDouble())
        o.put("AUTH_MARGIN_THRESHOLD", settings.authMarginThreshold.toDouble())
        o.put("TOP_K", settings.topK)
        o.put("ANALYZE_EVERY_N_FRAMES", settings.analyzeEveryNFrames)
        o.put("AMBIGUOUS_CANDIDATE_SWITCH_FRAMES", settings.ambiguousCandidateSwitchFrames)
        o.put("SELECTED_CANDIDATE_GRACE_FRAMES", settings.selectedCandidateGraceFrames)
        o.put("CAMERA_INDEX", settings.cameraIndex)
        o.put("RECOGNITION_USE_ELLIPSE_MASK", settings.recognitionUseEllipseMask)
        o.put("REGISTER_SAMPLES_PER_DIRECTION", settings.registerSamplesPerDirection)
        o.put("REGISTER_CAPTURE_EVERY_N_FRAMES", settings.registerCaptureEveryNFrames)
        o.put("REGISTER_SIMILARITY_LOWER_BOUNDARY", settings.registerSimilarityLowerBoundary.toDouble())
        o.put("REGISTER_SIMILARITY_HIGHER_BOUNDARY", settings.registerSimilarityHigherBoundary.toDouble())
        o.put("MAX_EMBEDDINGS_PER_PERSON", settings.maxEmbeddingsPerPerson)
        o.put("REGISTER_ELLIPSE_AXIS_X_RATIO", settings.registerEllipseAxisXRatio.toDouble())
        o.put("REGISTER_ELLIPSE_AXIS_Y_RATIO", settings.registerEllipseAxisYRatio.toDouble())
        o.put("REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO", settings.registerEllipseCenterYOffsetRatio.toDouble())
        o.put("REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA", settings.registerEllipseOutsideDimAlpha.toDouble())
        o.put("ADAPTIVE_UPDATE_ENABLED", settings.adaptiveUpdateEnabled)
        o.put("ADAPTIVE_UPDATE_MAX_SIM", settings.adaptiveUpdateMaxSim.toDouble())
        o.put("ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES", settings.adaptiveUpdateMinIntervalFrames)
        o.put("ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION", settings.adaptiveUpdateMaxSamplesPerSession)
        o.put("ADAPTIVE_UPDATE_IDLE_SECONDS", settings.adaptiveUpdateIdleSeconds)
        o.put("DB_SYNC_ENABLED", settings.dbSyncEnabled)
        o.put("DB_SERVER_BASE_URL", settings.dbServerBaseUrl)
        o.put("DB_CONNECT_RETRY_COUNT", settings.dbConnectRetryCount)
        o.put("DB_CONNECT_RETRY_DELAY_SECONDS", settings.dbConnectRetryDelaySeconds)
        o.put("DB_CONNECT_TIMEOUT_SECONDS", settings.dbConnectTimeoutSeconds)
        o.put("DB_CLIENT_ID", settings.dbClientId)
        o.put("DB_AUTH_TOKEN", settings.dbAuthToken)
        o.put("DB_DEVICE_NAME", settings.dbDeviceName)
        o.put("PRINT_ENABLED", settings.printEnabled)
        o.put("PRINTER_MODEL", settings.printerModel)
        o.put("PRINTER_NAME", settings.printerName)
        o.put("BROTHER_PRINTER_MAC_ADDRESS", settings.brotherPrinterMacAddress)
        o.put("NEMONIC_PRINTER_MAC_ADDRESS", settings.nemonicPrinterMacAddress)
        o.put("PRINTER_TIMEOUT_SEC", settings.printerTimeoutSec)
        o.put("PRINTER_COPIES", settings.printerCopies)
        o.put("PRINTER_JOB_TITLE_PREFIX", settings.printerJobTitlePrefix)
        o.put("PRINTER_CONTENT_TEMPLATE", settings.printerContentTemplate)
        o.put("PRINTER_TEMPLATE_IMAGE_URI", settings.printerTemplateImageUri)
        o.put("TEMPLATE_IMAGE_ROTATION_QUARTER_TURNS", settings.templateImageRotationQuarterTurns.mod(4))
        o.put("TEMPLATE_SHRINK_TO_PAPER_IF_OVERSIZED", settings.templateShrinkToPaperIfOversized)
        val papers = JSONArray()
        settings.paperProfiles.forEach { profile ->
            papers.put(
                JSONObject().apply {
                    put("id", profile.id)
                    put("name", profile.name)
                    put("width_mm", profile.widthMm.toDouble())
                    put("height_mm", profile.heightMm.toDouble())
                },
            )
        }
        o.put("PAPER_PROFILES", papers)
        o.put("SELECTED_PAPER_ID", settings.selectedPaperId)
        o.put("PRINT_LANDSCAPE", settings.printLandscape)
        o.put("PAPER_LANDSCAPE", settings.paperLandscape)
        o.put("TEMPLATE_TEXT_REGION_OFFSET_X", settings.templateTextRegionOffsetX)
        o.put("TEMPLATE_TEXT_REGION_OFFSET_Y", settings.templateTextRegionOffsetY)
        o.put("TEMPLATE_TEXT_REGION_WIDTH", settings.templateTextRegionWidth)
        o.put("TEMPLATE_TEXT_REGION_HEIGHT", settings.templateTextRegionHeight)
        o.put("PRINT_TEXT_BASE_FONT_SIZE", settings.printTextBaseFontSize)
        o.put("PRINT_TEXT_SCALE_MULTIPLIER", settings.printTextScaleMultiplier)
        o.put("PRINT_ATTRIBUTE_COUNT", settings.printAttributeCount)
        o.put("PRINT_ATTRIBUTE_1_COLUMN", settings.printAttribute1Column)
        o.put("PRINT_ATTRIBUTE_1_TEXT_SIZE_SP", settings.printAttribute1TextSizeSp)
        o.put("PRINT_ATTRIBUTE_1_OFFSET_X", settings.printAttribute1OffsetX)
        o.put("PRINT_ATTRIBUTE_1_OFFSET_Y", settings.printAttribute1OffsetY)
        o.put("PRINT_ATTRIBUTE_2_COLUMN", settings.printAttribute2Column)
        o.put("PRINT_ATTRIBUTE_2_TEXT_SIZE_SP", settings.printAttribute2TextSizeSp)
        o.put("PRINT_ATTRIBUTE_2_OFFSET_X", settings.printAttribute2OffsetX)
        o.put("PRINT_ATTRIBUTE_2_OFFSET_Y", settings.printAttribute2OffsetY)
        o.put("PRINT_ATTRIBUTE_3_COLUMN", settings.printAttribute3Column)
        o.put("PRINT_ATTRIBUTE_3_TEXT_SIZE_SP", settings.printAttribute3TextSizeSp)
        o.put("PRINT_ATTRIBUTE_3_OFFSET_X", settings.printAttribute3OffsetX)
        o.put("PRINT_ATTRIBUTE_3_OFFSET_Y", settings.printAttribute3OffsetY)
        o.put("OUTPUT_NAME_TEXT_SIZE_SP", settings.outputNameTextSizeSp)
        o.put("OUTPUT_NAME_OFFSET_X", settings.outputNameOffsetX)
        o.put("OUTPUT_NAME_OFFSET_Y", settings.outputNameOffsetY)
        o.put("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", settings.outputDepartmentTextSizeSp)
        o.put("OUTPUT_DEPARTMENT_OFFSET_X", settings.outputDepartmentOffsetX)
        o.put("OUTPUT_DEPARTMENT_OFFSET_Y", settings.outputDepartmentOffsetY)
        o.put("PRINT_STATUS_READY_MESSAGE", settings.printStatusReadyMessage)
        o.put("PRINT_STATUS_PERMISSION_MESSAGE", settings.printStatusPermissionMessage)
        o.put("PRINT_STATUS_CONNECTING_MESSAGE", settings.printStatusConnectingMessage)
        o.put("PRINT_STATUS_PREPARING_MESSAGE", settings.printStatusPreparingMessage)
        o.put("PRINT_STATUS_SENDING_MESSAGE", settings.printStatusSendingMessage)
        o.put("PRINT_STATUS_SUCCESS_MESSAGE", settings.printStatusSuccessMessage)
        o.put("PRINT_STATUS_ERROR_MESSAGE", settings.printStatusErrorMessage)
        o.put("OUTPUT_RETURN_DELAY_SECONDS", settings.outputReturnDelaySeconds)
        o.put("OUTPUT_RETURN_MESSAGE_TEMPLATE", settings.outputReturnMessageTemplate)
        o.put("UI_WELCOME_MESSAGE", settings.uiWelcomeMessage)
        o.put("UI_SERVER_DISCONNECTED_MESSAGE", settings.uiServerDisconnectedMessage)
        o.put("UI_SERVER_DISCONNECTED_DETAIL", settings.uiServerDisconnectedDetail)
        o.put("UI_UNREGISTERED_MESSAGE", settings.uiUnregisteredMessage)
        o.put("UI_UNREGISTERED_DETAIL", settings.uiUnregisteredDetail)
        o.put("UI_RECOGNIZED_MESSAGE_TEMPLATE", settings.uiRecognizedMessageTemplate)
        o.put("UI_AMBIGUOUS_MESSAGE", settings.uiAmbiguousMessage)
        o.put("UI_AMBIGUOUS_DETAIL", settings.uiAmbiguousDetail)
        o.put("UI_SELECTED_CANDIDATE_TEMPLATE", settings.uiSelectedCandidateTemplate)
        o.put("UI_REGISTER_COMPLETE_MESSAGE", settings.uiRegisterCompleteMessage)
        o.put("UI_LABEL_ONLY_INPUT_MESSAGE", settings.uiLabelOnlyInputMessage)
        o.put("UI_SAVING_MESSAGE", settings.uiSavingMessage)
        o.put("UI_OUTPUT_BUTTON_TEXT", settings.uiOutputButtonText)
        prefs.edit().putString(KEY_JSON, o.toString()).apply()
    }

    companion object {
        private const val KEY_JSON = "runtime_settings_json"

        private fun parsePaperProfiles(raw: JSONArray?): List<PaperProfile> {
            val parsed = mutableListOf<PaperProfile>()
            if (raw != null) {
                for (idx in 0 until raw.length()) {
                    val o = raw.optJSONObject(idx) ?: continue
                    val id = o.optString("id", "").trim()
                    val name = o.optString("name", "").trim()
                    val width = o.optDouble("width_mm", -1.0).toFloat()
                    val height = o.optDouble("height_mm", -1.0).toFloat()
                    if (id.isBlank() || name.isBlank() || width <= 0f || height <= 0f) continue
                    parsed += PaperProfile(id = id, name = name, widthMm = width, heightMm = height)
                }
            }
            if (parsed.isEmpty()) return DEFAULT_PAPER_PROFILES
            if (parsed.none { it.id == DEFAULT_PAPER_ID }) {
                parsed.add(0, DEFAULT_PAPER_PROFILES.first())
            }
            return parsed.distinctBy { it.id }
        }
    }
}
