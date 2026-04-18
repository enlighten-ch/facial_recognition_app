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
    val faceDataDir: String = "",
    val authSimThreshold: Float = 0.40f,
    val authMarginThreshold: Float = 0.10f,
    val topK: Int = 3,
    val analyzeEveryNFrames: Int = 6,
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
    val nemonicPrinterMacAddress: String = "",
    val printerTimeoutSec: Int = 10,
    val printerCopies: Int = 1,
    val printerJobTitlePrefix: String = "FaceRecognition",
    val printerContentTemplate: String = "Name: {name}\nDepartment: {department}\n",
    val printerTemplateImageUri: String = "",
    val templateShrinkToPaperIfOversized: Boolean = true,
    val paperProfiles: List<PaperProfile> = DEFAULT_PAPER_PROFILES,
    val selectedPaperId: String = DEFAULT_PAPER_ID,
    val printLandscape: Boolean = false,
    val templateTextRegionOffsetX: Int = 24,
    val templateTextRegionOffsetY: Int = 24,
    val templateTextRegionWidth: Int = 360,
    val templateTextRegionHeight: Int = 180,
    val printTextBaseFontSize: Int = 48,
    val printTextScaleMultiplier: Int = 5,
    val outputNameTextSizeSp: Int = 40,
    val outputDepartmentTextSizeSp: Int = 24,
    val outputDepartmentOffsetX: Int = 0,
    val outputDepartmentOffsetY: Int = 0,
    val outputReturnDelaySeconds: Int = 3,
    val outputReturnMessageTemplate: String = "{seconds}초 후 기본 화면으로 복귀합니다",
)

object PrinterModel {
    const val SYSTEM = "system"
    const val NEMONIC = "nemonic"

    fun sanitize(value: String?): String {
        return when (value) {
            NEMONIC -> NEMONIC
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
                faceDataDir = o.optString("FACE_DATA_DIR", ""),
                authSimThreshold = o.optDouble("AUTH_SIM_THRESHOLD", 0.40).toFloat(),
                authMarginThreshold = o.optDouble("AUTH_MARGIN_THRESHOLD", 0.10).toFloat(),
                topK = o.optInt("TOP_K", 3),
                analyzeEveryNFrames = o.optInt("ANALYZE_EVERY_N_FRAMES", 6),
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
                nemonicPrinterMacAddress = o.optString("NEMONIC_PRINTER_MAC_ADDRESS", ""),
                printerTimeoutSec = o.optInt("PRINTER_TIMEOUT_SEC", 10),
                printerCopies = o.optInt("PRINTER_COPIES", 1),
                printerJobTitlePrefix = o.optString("PRINTER_JOB_TITLE_PREFIX", "FaceRecognition"),
                printerContentTemplate = o.optString("PRINTER_CONTENT_TEMPLATE", "Name: {name}\nDepartment: {department}\n"),
                printerTemplateImageUri = o.optString("PRINTER_TEMPLATE_IMAGE_URI", ""),
                templateShrinkToPaperIfOversized = o.optBoolean("TEMPLATE_SHRINK_TO_PAPER_IF_OVERSIZED", true),
                paperProfiles = parsePaperProfiles(o.optJSONArray("PAPER_PROFILES")),
                selectedPaperId = o.optString("SELECTED_PAPER_ID", DEFAULT_PAPER_ID),
                printLandscape = o.optBoolean("PRINT_LANDSCAPE", false),
                templateTextRegionOffsetX = o.optInt("TEMPLATE_TEXT_REGION_OFFSET_X", 24),
                templateTextRegionOffsetY = o.optInt("TEMPLATE_TEXT_REGION_OFFSET_Y", 24),
                templateTextRegionWidth = o.optInt("TEMPLATE_TEXT_REGION_WIDTH", 360),
                templateTextRegionHeight = o.optInt("TEMPLATE_TEXT_REGION_HEIGHT", 180),
                printTextBaseFontSize = o.optInt("PRINT_TEXT_BASE_FONT_SIZE", 48),
                printTextScaleMultiplier = o.optInt("PRINT_TEXT_SCALE_MULTIPLIER", 5),
                outputNameTextSizeSp = o.optInt("OUTPUT_NAME_TEXT_SIZE_SP", 40),
                outputDepartmentTextSizeSp = o.optInt("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", 24),
                outputDepartmentOffsetX = o.optInt("OUTPUT_DEPARTMENT_OFFSET_X", 0),
                outputDepartmentOffsetY = o.optInt("OUTPUT_DEPARTMENT_OFFSET_Y", 0),
                outputReturnDelaySeconds = o.optInt("OUTPUT_RETURN_DELAY_SECONDS", 3),
                outputReturnMessageTemplate = o.optString("OUTPUT_RETURN_MESSAGE_TEMPLATE", "{seconds}초 후 기본 화면으로 복귀합니다"),
            )
        }.getOrDefault(AppSettings())
    }

    fun save(settings: AppSettings) {
        val o = JSONObject()
        o.put("FACE_DATA_DIR", settings.faceDataDir)
        o.put("AUTH_SIM_THRESHOLD", settings.authSimThreshold.toDouble())
        o.put("AUTH_MARGIN_THRESHOLD", settings.authMarginThreshold.toDouble())
        o.put("TOP_K", settings.topK)
        o.put("ANALYZE_EVERY_N_FRAMES", settings.analyzeEveryNFrames)
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
        o.put("NEMONIC_PRINTER_MAC_ADDRESS", settings.nemonicPrinterMacAddress)
        o.put("PRINTER_TIMEOUT_SEC", settings.printerTimeoutSec)
        o.put("PRINTER_COPIES", settings.printerCopies)
        o.put("PRINTER_JOB_TITLE_PREFIX", settings.printerJobTitlePrefix)
        o.put("PRINTER_CONTENT_TEMPLATE", settings.printerContentTemplate)
        o.put("PRINTER_TEMPLATE_IMAGE_URI", settings.printerTemplateImageUri)
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
        o.put("TEMPLATE_TEXT_REGION_OFFSET_X", settings.templateTextRegionOffsetX)
        o.put("TEMPLATE_TEXT_REGION_OFFSET_Y", settings.templateTextRegionOffsetY)
        o.put("TEMPLATE_TEXT_REGION_WIDTH", settings.templateTextRegionWidth)
        o.put("TEMPLATE_TEXT_REGION_HEIGHT", settings.templateTextRegionHeight)
        o.put("PRINT_TEXT_BASE_FONT_SIZE", settings.printTextBaseFontSize)
        o.put("PRINT_TEXT_SCALE_MULTIPLIER", settings.printTextScaleMultiplier)
        o.put("OUTPUT_NAME_TEXT_SIZE_SP", settings.outputNameTextSizeSp)
        o.put("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", settings.outputDepartmentTextSizeSp)
        o.put("OUTPUT_DEPARTMENT_OFFSET_X", settings.outputDepartmentOffsetX)
        o.put("OUTPUT_DEPARTMENT_OFFSET_Y", settings.outputDepartmentOffsetY)
        o.put("OUTPUT_RETURN_DELAY_SECONDS", settings.outputReturnDelaySeconds)
        o.put("OUTPUT_RETURN_MESSAGE_TEMPLATE", settings.outputReturnMessageTemplate)
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
