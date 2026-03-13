package com.example.facetestbed

import android.content.Context
import org.json.JSONObject

data class AppSettings(
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
    val printEnabled: Boolean = true,
    val printerName: String = "",
    val printerTimeoutSec: Int = 10,
    val printerCopies: Int = 1,
    val printerJobTitlePrefix: String = "FaceRecognition",
    val printerContentTemplate: String = "Name: {name}\n",
    val printTextBaseFontSize: Int = 48,
    val printTextScaleMultiplier: Int = 5,
)

class AppSettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)

    fun load(): AppSettings {
        val raw = prefs.getString(KEY_JSON, null) ?: return AppSettings()
        return runCatching {
            val o = JSONObject(raw)
            AppSettings(
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
                printEnabled = o.optBoolean("PRINT_ENABLED", true),
                printerName = o.optString("PRINTER_NAME", ""),
                printerTimeoutSec = o.optInt("PRINTER_TIMEOUT_SEC", 10),
                printerCopies = o.optInt("PRINTER_COPIES", 1),
                printerJobTitlePrefix = o.optString("PRINTER_JOB_TITLE_PREFIX", "FaceRecognition"),
                printerContentTemplate = o.optString("PRINTER_CONTENT_TEMPLATE", "Name: {name}\n"),
                printTextBaseFontSize = o.optInt("PRINT_TEXT_BASE_FONT_SIZE", 48),
                printTextScaleMultiplier = o.optInt("PRINT_TEXT_SCALE_MULTIPLIER", 5),
            )
        }.getOrDefault(AppSettings())
    }

    fun save(settings: AppSettings) {
        val o = JSONObject()
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
        o.put("PRINT_ENABLED", settings.printEnabled)
        o.put("PRINTER_NAME", settings.printerName)
        o.put("PRINTER_TIMEOUT_SEC", settings.printerTimeoutSec)
        o.put("PRINTER_COPIES", settings.printerCopies)
        o.put("PRINTER_JOB_TITLE_PREFIX", settings.printerJobTitlePrefix)
        o.put("PRINTER_CONTENT_TEMPLATE", settings.printerContentTemplate)
        o.put("PRINT_TEXT_BASE_FONT_SIZE", settings.printTextBaseFontSize)
        o.put("PRINT_TEXT_SCALE_MULTIPLIER", settings.printTextScaleMultiplier)
        prefs.edit().putString(KEY_JSON, o.toString()).apply()
    }

    companion object {
        private const val KEY_JSON = "runtime_settings_json"
    }
}
