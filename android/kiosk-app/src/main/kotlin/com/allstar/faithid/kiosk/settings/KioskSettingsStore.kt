package com.allstar.faithid.kiosk.settings

import android.content.Context
import com.allstar.faithid.kiosk.flow.FlowMessages
import org.json.JSONArray
import org.json.JSONObject

/**
 * Loads and saves [KioskSettings] as one JSON document in shared preferences. Key names are the
 * ones used since the first release, so settings survive the upgrade.
 */
class KioskSettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)

    fun load(): KioskSettings {
        val raw = prefs.getString(KEY_JSON, null) ?: return KioskSettings()
        return runCatching { fromJson(JSONObject(raw)) }.getOrDefault(KioskSettings())
    }

    fun save(settings: KioskSettings) {
        prefs.edit().putString(KEY_JSON, toJson(settings).toString()).apply()
    }

    companion object {
        private const val KEY_JSON = "runtime_settings_json"

        fun fromJson(o: JSONObject): KioskSettings = upgradeUntouchedLabelLayout(o, parse(o))

        /**
         * Kiosks that never changed the label layout get the new stacked layout: the old factory
         * values printed overlapping text (report issue I-11). Layouts someone tuned stay as they are.
         */
        private fun upgradeUntouchedLabelLayout(o: JSONObject, s: KioskSettings): KioskSettings {
            if (o.has("PRINT_LAYOUT_STACKED")) return s
            val untouched = s.templateTextRegionOffsetX == 24 && s.templateTextRegionOffsetY == 24 &&
                s.templateTextRegionWidth == 360 && s.templateTextRegionHeight == 180 &&
                s.printTextScaleMultiplier == 5 &&
                s.labelLines.map { it.textSizeSp } == listOf(40, 24, 20) &&
                s.labelLines.map { it.offsetX to it.offsetY } == listOf(0 to 0, 0 to 0, 0 to 36)
            if (!untouched) return s
            val d = KioskSettings()
            return s.copy(
                printLayoutStacked = true,
                templateTextRegionWidth = d.templateTextRegionWidth,
                templateTextRegionHeight = d.templateTextRegionHeight,
                printTextScaleMultiplier = d.printTextScaleMultiplier,
                labelLines = s.labelLines.mapIndexed { i, line -> d.labelLines[i].copy(column = line.column) },
            )
        }

        private fun parse(o: JSONObject): KioskSettings {
            val d = KioskSettings()
            val dm = FlowMessages()
            fun str(key: String, default: String) = o.optString(key, default)
            fun int(key: String, default: Int) = o.optInt(key, default)
            fun float(key: String, default: Float) = o.optDouble(key, default.toDouble()).toFloat()
            fun bool(key: String, default: Boolean) = o.optBoolean(key, default)
            // Before 2026-10 the kiosk stored a server-issued token; it becomes the kiosk's secret so the
            // migrated server still recognizes the kiosk as approved.
            val legacyToken = str("DB_AUTH_TOKEN", "")
            val lines = listOf(
                LabelLine(
                    str("PRINT_ATTRIBUTE_1_COLUMN", "name"),
                    int("PRINT_ATTRIBUTE_1_TEXT_SIZE_SP", int("OUTPUT_NAME_TEXT_SIZE_SP", 40)),
                    int("PRINT_ATTRIBUTE_1_OFFSET_X", int("OUTPUT_NAME_OFFSET_X", 0)),
                    int("PRINT_ATTRIBUTE_1_OFFSET_Y", int("OUTPUT_NAME_OFFSET_Y", 0)),
                ),
                LabelLine(
                    str("PRINT_ATTRIBUTE_2_COLUMN", "department"),
                    int("PRINT_ATTRIBUTE_2_TEXT_SIZE_SP", int("OUTPUT_DEPARTMENT_TEXT_SIZE_SP", 24)),
                    int("PRINT_ATTRIBUTE_2_OFFSET_X", int("OUTPUT_DEPARTMENT_OFFSET_X", 0)),
                    int("PRINT_ATTRIBUTE_2_OFFSET_Y", int("OUTPUT_DEPARTMENT_OFFSET_Y", 0)),
                ),
                LabelLine(
                    str("PRINT_ATTRIBUTE_3_COLUMN", "grade"),
                    int("PRINT_ATTRIBUTE_3_TEXT_SIZE_SP", 20),
                    int("PRINT_ATTRIBUTE_3_OFFSET_X", 0),
                    int("PRINT_ATTRIBUTE_3_OFFSET_Y", 36),
                ),
            )
            return KioskSettings(
                developerViewEnabled = bool("DEVELOPER_VIEW_ENABLED", d.developerViewEnabled),
                hideSystemBars = bool("HIDE_SYSTEM_BARS", d.hideSystemBars),
                authSimThreshold = float("AUTH_SIM_THRESHOLD", d.authSimThreshold),
                authMarginThreshold = float("AUTH_MARGIN_THRESHOLD", d.authMarginThreshold),
                topK = int("TOP_K", d.topK),
                analyzeEveryNFrames = int("ANALYZE_EVERY_N_FRAMES", d.analyzeEveryNFrames),
                ambiguousCandidateSwitchFrames = int("AMBIGUOUS_CANDIDATE_SWITCH_FRAMES", d.ambiguousCandidateSwitchFrames),
                selectedCandidateGraceFrames = int("SELECTED_CANDIDATE_GRACE_FRAMES", d.selectedCandidateGraceFrames),
                cameraIndex = int("CAMERA_INDEX", d.cameraIndex),
                faceAlignmentEnabled = bool("FACE_ALIGNMENT_ENABLED", d.faceAlignmentEnabled),
                registerEllipseAxisXRatio = float("REGISTER_ELLIPSE_AXIS_X_RATIO", d.registerEllipseAxisXRatio),
                registerEllipseAxisYRatio = float("REGISTER_ELLIPSE_AXIS_Y_RATIO", d.registerEllipseAxisYRatio),
                registerEllipseCenterYOffsetRatio = float("REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO", d.registerEllipseCenterYOffsetRatio),
                registerEllipseOutsideDimAlpha = float("REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA", d.registerEllipseOutsideDimAlpha),
                registerSamplesPerDirection = int("REGISTER_SAMPLES_PER_DIRECTION", d.registerSamplesPerDirection),
                registerCaptureEveryNFrames = int("REGISTER_CAPTURE_EVERY_N_FRAMES", d.registerCaptureEveryNFrames),
                registerSimilarityLowerBoundary = float("REGISTER_SIMILARITY_LOWER_BOUNDARY", d.registerSimilarityLowerBoundary),
                registerSimilarityHigherBoundary = float("REGISTER_SIMILARITY_HIGHER_BOUNDARY", d.registerSimilarityHigherBoundary),
                duplicateThreshold = float("DUPLICATE_THRESHOLD", d.duplicateThreshold),
                maxEmbeddingsPerPerson = int("MAX_EMBEDDINGS_PER_PERSON", d.maxEmbeddingsPerPerson),
                adaptiveUpdateEnabled = bool("ADAPTIVE_UPDATE_ENABLED", d.adaptiveUpdateEnabled),
                adaptiveUpdateMaxSim = float("ADAPTIVE_UPDATE_MAX_SIM", d.adaptiveUpdateMaxSim),
                adaptiveUpdateMinIntervalFrames = int("ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES", d.adaptiveUpdateMinIntervalFrames),
                adaptiveUpdateMaxSamplesPerSession = int("ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION", d.adaptiveUpdateMaxSamplesPerSession),
                adaptiveUpdateIdleSeconds = int("ADAPTIVE_UPDATE_IDLE_SECONDS", d.adaptiveUpdateIdleSeconds),
                dbSyncEnabled = bool("DB_SYNC_ENABLED", d.dbSyncEnabled),
                dbServerBaseUrl = str("DB_SERVER_BASE_URL", d.dbServerBaseUrl),
                dbConnectRetryDelaySeconds = int("DB_CONNECT_RETRY_DELAY_SECONDS", d.dbConnectRetryDelaySeconds),
                dbConnectTimeoutSeconds = int("DB_CONNECT_TIMEOUT_SECONDS", d.dbConnectTimeoutSeconds),
                dbClientId = str("DB_CLIENT_ID", ""),
                dbClientSecret = str("DB_CLIENT_SECRET", "").ifBlank { legacyToken },
                dbDeviceName = str("DB_DEVICE_NAME", ""),
                printEnabled = bool("PRINT_ENABLED", d.printEnabled),
                printerModel = PrinterModel.sanitize(str("PRINTER_MODEL", PrinterModel.SYSTEM)),
                printerName = str("PRINTER_NAME", ""),
                brotherPrinterMacAddress = str("BROTHER_PRINTER_MAC_ADDRESS", ""),
                nemonicPrinterMacAddress = str("NEMONIC_PRINTER_MAC_ADDRESS", ""),
                printerTimeoutSec = int("PRINTER_TIMEOUT_SEC", d.printerTimeoutSec),
                printerCopies = int("PRINTER_COPIES", d.printerCopies),
                printerJobTitlePrefix = str("PRINTER_JOB_TITLE_PREFIX", d.printerJobTitlePrefix),
                paperProfiles = parsePaperProfiles(o.optJSONArray("PAPER_PROFILES")),
                selectedPaperId = str("SELECTED_PAPER_ID", KioskSettings.DEFAULT_PAPER_ID),
                printLandscape = bool("PRINT_LANDSCAPE", d.printLandscape),
                printerTemplateImageUri = str("PRINTER_TEMPLATE_IMAGE_URI", ""),
                templateImageRotationQuarterTurns = int("TEMPLATE_IMAGE_ROTATION_QUARTER_TURNS", 0).mod(4),
                templateShrinkToPaperIfOversized = bool("TEMPLATE_SHRINK_TO_PAPER_IF_OVERSIZED", true),
                paperLandscape = bool("PAPER_LANDSCAPE", d.paperLandscape),
                templateTextRegionOffsetX = int("TEMPLATE_TEXT_REGION_OFFSET_X", 24),
                templateTextRegionOffsetY = int("TEMPLATE_TEXT_REGION_OFFSET_Y", 24),
                templateTextRegionWidth = int("TEMPLATE_TEXT_REGION_WIDTH", 360),
                templateTextRegionHeight = int("TEMPLATE_TEXT_REGION_HEIGHT", 180),
                // Settings saved by older versions keep their fixed line positions.
                printLayoutStacked = bool("PRINT_LAYOUT_STACKED", false),
                printTextScaleMultiplier = int("PRINT_TEXT_SCALE_MULTIPLIER", 5),
                printAttributeCount = int("PRINT_ATTRIBUTE_COUNT", 2),
                labelLines = lines,
                printStatusReadyMessage = str("PRINT_STATUS_READY_MESSAGE", d.printStatusReadyMessage),
                printStatusPermissionMessage = str("PRINT_STATUS_PERMISSION_MESSAGE", d.printStatusPermissionMessage),
                printStatusConnectingMessage = str("PRINT_STATUS_CONNECTING_MESSAGE", d.printStatusConnectingMessage),
                printStatusPreparingMessage = str("PRINT_STATUS_PREPARING_MESSAGE", d.printStatusPreparingMessage),
                printStatusSendingMessage = str("PRINT_STATUS_SENDING_MESSAGE", d.printStatusSendingMessage),
                printStatusSuccessMessage = str("PRINT_STATUS_SUCCESS_MESSAGE", d.printStatusSuccessMessage),
                printStatusErrorMessage = str("PRINT_STATUS_ERROR_MESSAGE", d.printStatusErrorMessage),
                outputReturnDelaySeconds = int("OUTPUT_RETURN_DELAY_SECONDS", d.outputReturnDelaySeconds),
                messages = FlowMessages(
                    welcome = str("UI_WELCOME_MESSAGE", dm.welcome),
                    serverDisconnected = str("UI_SERVER_DISCONNECTED_MESSAGE", dm.serverDisconnected),
                    serverDisconnectedDetail = str("UI_SERVER_DISCONNECTED_DETAIL", dm.serverDisconnectedDetail),
                    awaitingApproval = str("UI_AWAITING_APPROVAL_MESSAGE", dm.awaitingApproval),
                    awaitingApprovalDetail = str("UI_AWAITING_APPROVAL_DETAIL", dm.awaitingApprovalDetail),
                    unregistered = str("UI_UNREGISTERED_MESSAGE", dm.unregistered),
                    unregisteredDetail = str("UI_UNREGISTERED_DETAIL", dm.unregisteredDetail),
                    recognized = str("UI_RECOGNIZED_MESSAGE_TEMPLATE", dm.recognized),
                    ambiguous = str("UI_AMBIGUOUS_MESSAGE", dm.ambiguous),
                    ambiguousDetail = str("UI_AMBIGUOUS_DETAIL", dm.ambiguousDetail),
                    selectedCandidate = str("UI_SELECTED_CANDIDATE_TEMPLATE", dm.selectedCandidate),
                    registerComplete = str("UI_REGISTER_COMPLETE_MESSAGE", dm.registerComplete),
                    labelOnlyInput = str("UI_LABEL_ONLY_INPUT_MESSAGE", dm.labelOnlyInput),
                    duplicateFound = str("UI_DUPLICATE_FOUND_MESSAGE", dm.duplicateFound),
                    duplicateDetail = str("UI_DUPLICATE_DETAIL", dm.duplicateDetail),
                    returnCountdown = str("OUTPUT_RETURN_MESSAGE_TEMPLATE", dm.returnCountdown),
                ),
                uiSavingMessage = str("UI_SAVING_MESSAGE", d.uiSavingMessage),
                uiOutputButtonText = str("UI_OUTPUT_BUTTON_TEXT", d.uiOutputButtonText),
                legacyFaceDataDir = str("FACE_DATA_DIR", d.legacyFaceDataDir),
            )
        }

        fun toJson(s: KioskSettings): JSONObject = JSONObject().apply {
            put("DEVELOPER_VIEW_ENABLED", s.developerViewEnabled)
            put("HIDE_SYSTEM_BARS", s.hideSystemBars)
            put("AUTH_SIM_THRESHOLD", s.authSimThreshold.toDouble())
            put("AUTH_MARGIN_THRESHOLD", s.authMarginThreshold.toDouble())
            put("TOP_K", s.topK)
            put("ANALYZE_EVERY_N_FRAMES", s.analyzeEveryNFrames)
            put("AMBIGUOUS_CANDIDATE_SWITCH_FRAMES", s.ambiguousCandidateSwitchFrames)
            put("SELECTED_CANDIDATE_GRACE_FRAMES", s.selectedCandidateGraceFrames)
            put("CAMERA_INDEX", s.cameraIndex)
            put("FACE_ALIGNMENT_ENABLED", s.faceAlignmentEnabled)
            put("REGISTER_ELLIPSE_AXIS_X_RATIO", s.registerEllipseAxisXRatio.toDouble())
            put("REGISTER_ELLIPSE_AXIS_Y_RATIO", s.registerEllipseAxisYRatio.toDouble())
            put("REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO", s.registerEllipseCenterYOffsetRatio.toDouble())
            put("REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA", s.registerEllipseOutsideDimAlpha.toDouble())
            put("REGISTER_SAMPLES_PER_DIRECTION", s.registerSamplesPerDirection)
            put("REGISTER_CAPTURE_EVERY_N_FRAMES", s.registerCaptureEveryNFrames)
            put("REGISTER_SIMILARITY_LOWER_BOUNDARY", s.registerSimilarityLowerBoundary.toDouble())
            put("REGISTER_SIMILARITY_HIGHER_BOUNDARY", s.registerSimilarityHigherBoundary.toDouble())
            put("DUPLICATE_THRESHOLD", s.duplicateThreshold.toDouble())
            put("MAX_EMBEDDINGS_PER_PERSON", s.maxEmbeddingsPerPerson)
            put("ADAPTIVE_UPDATE_ENABLED", s.adaptiveUpdateEnabled)
            put("ADAPTIVE_UPDATE_MAX_SIM", s.adaptiveUpdateMaxSim.toDouble())
            put("ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES", s.adaptiveUpdateMinIntervalFrames)
            put("ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION", s.adaptiveUpdateMaxSamplesPerSession)
            put("ADAPTIVE_UPDATE_IDLE_SECONDS", s.adaptiveUpdateIdleSeconds)
            put("DB_SYNC_ENABLED", s.dbSyncEnabled)
            put("DB_SERVER_BASE_URL", s.dbServerBaseUrl)
            put("DB_CONNECT_RETRY_DELAY_SECONDS", s.dbConnectRetryDelaySeconds)
            put("DB_CONNECT_TIMEOUT_SECONDS", s.dbConnectTimeoutSeconds)
            put("DB_CLIENT_ID", s.dbClientId)
            put("DB_CLIENT_SECRET", s.dbClientSecret)
            put("DB_DEVICE_NAME", s.dbDeviceName)
            put("PRINT_ENABLED", s.printEnabled)
            put("PRINTER_MODEL", s.printerModel)
            put("PRINTER_NAME", s.printerName)
            put("BROTHER_PRINTER_MAC_ADDRESS", s.brotherPrinterMacAddress)
            put("NEMONIC_PRINTER_MAC_ADDRESS", s.nemonicPrinterMacAddress)
            put("PRINTER_TIMEOUT_SEC", s.printerTimeoutSec)
            put("PRINTER_COPIES", s.printerCopies)
            put("PRINTER_JOB_TITLE_PREFIX", s.printerJobTitlePrefix)
            put(
                "PAPER_PROFILES",
                JSONArray().also { array ->
                    s.paperProfiles.forEach { p ->
                        array.put(JSONObject().put("id", p.id).put("name", p.name).put("width_mm", p.widthMm.toDouble()).put("height_mm", p.heightMm.toDouble()))
                    }
                },
            )
            put("SELECTED_PAPER_ID", s.selectedPaperId)
            put("PRINT_LANDSCAPE", s.printLandscape)
            put("PRINTER_TEMPLATE_IMAGE_URI", s.printerTemplateImageUri)
            put("TEMPLATE_IMAGE_ROTATION_QUARTER_TURNS", s.templateImageRotationQuarterTurns.mod(4))
            put("TEMPLATE_SHRINK_TO_PAPER_IF_OVERSIZED", s.templateShrinkToPaperIfOversized)
            put("PAPER_LANDSCAPE", s.paperLandscape)
            put("TEMPLATE_TEXT_REGION_OFFSET_X", s.templateTextRegionOffsetX)
            put("TEMPLATE_TEXT_REGION_OFFSET_Y", s.templateTextRegionOffsetY)
            put("TEMPLATE_TEXT_REGION_WIDTH", s.templateTextRegionWidth)
            put("TEMPLATE_TEXT_REGION_HEIGHT", s.templateTextRegionHeight)
            put("PRINT_LAYOUT_STACKED", s.printLayoutStacked)
            put("PRINT_TEXT_SCALE_MULTIPLIER", s.printTextScaleMultiplier)
            put("PRINT_ATTRIBUTE_COUNT", s.printAttributeCount)
            s.labelLines.forEachIndexed { i, line ->
                val n = i + 1
                put("PRINT_ATTRIBUTE_${n}_COLUMN", line.column)
                put("PRINT_ATTRIBUTE_${n}_TEXT_SIZE_SP", line.textSizeSp)
                put("PRINT_ATTRIBUTE_${n}_OFFSET_X", line.offsetX)
                put("PRINT_ATTRIBUTE_${n}_OFFSET_Y", line.offsetY)
            }
            put("PRINT_STATUS_READY_MESSAGE", s.printStatusReadyMessage)
            put("PRINT_STATUS_PERMISSION_MESSAGE", s.printStatusPermissionMessage)
            put("PRINT_STATUS_CONNECTING_MESSAGE", s.printStatusConnectingMessage)
            put("PRINT_STATUS_PREPARING_MESSAGE", s.printStatusPreparingMessage)
            put("PRINT_STATUS_SENDING_MESSAGE", s.printStatusSendingMessage)
            put("PRINT_STATUS_SUCCESS_MESSAGE", s.printStatusSuccessMessage)
            put("PRINT_STATUS_ERROR_MESSAGE", s.printStatusErrorMessage)
            put("OUTPUT_RETURN_DELAY_SECONDS", s.outputReturnDelaySeconds)
            val m = s.messages
            put("UI_WELCOME_MESSAGE", m.welcome)
            put("UI_SERVER_DISCONNECTED_MESSAGE", m.serverDisconnected)
            put("UI_SERVER_DISCONNECTED_DETAIL", m.serverDisconnectedDetail)
            put("UI_AWAITING_APPROVAL_MESSAGE", m.awaitingApproval)
            put("UI_AWAITING_APPROVAL_DETAIL", m.awaitingApprovalDetail)
            put("UI_UNREGISTERED_MESSAGE", m.unregistered)
            put("UI_UNREGISTERED_DETAIL", m.unregisteredDetail)
            put("UI_RECOGNIZED_MESSAGE_TEMPLATE", m.recognized)
            put("UI_AMBIGUOUS_MESSAGE", m.ambiguous)
            put("UI_AMBIGUOUS_DETAIL", m.ambiguousDetail)
            put("UI_SELECTED_CANDIDATE_TEMPLATE", m.selectedCandidate)
            put("UI_REGISTER_COMPLETE_MESSAGE", m.registerComplete)
            put("UI_LABEL_ONLY_INPUT_MESSAGE", m.labelOnlyInput)
            put("UI_DUPLICATE_FOUND_MESSAGE", m.duplicateFound)
            put("UI_DUPLICATE_DETAIL", m.duplicateDetail)
            put("OUTPUT_RETURN_MESSAGE_TEMPLATE", m.returnCountdown)
            put("UI_SAVING_MESSAGE", s.uiSavingMessage)
            put("UI_OUTPUT_BUTTON_TEXT", s.uiOutputButtonText)
            put("FACE_DATA_DIR", s.legacyFaceDataDir)
        }

        private fun parsePaperProfiles(raw: JSONArray?): List<PaperProfile> {
            val parsed = mutableListOf<PaperProfile>()
            if (raw != null) {
                for (i in 0 until raw.length()) {
                    val o = raw.optJSONObject(i) ?: continue
                    val id = o.optString("id", "").trim()
                    val name = o.optString("name", "").trim()
                    val width = o.optDouble("width_mm", -1.0).toFloat()
                    val height = o.optDouble("height_mm", -1.0).toFloat()
                    if (id.isNotBlank() && name.isNotBlank() && width > 0f && height > 0f) parsed += PaperProfile(id, name, width, height)
                }
            }
            if (parsed.isEmpty()) return KioskSettings.DEFAULT_PAPER_PROFILES
            if (parsed.none { it.id == KioskSettings.DEFAULT_PAPER_ID }) parsed.add(0, KioskSettings.DEFAULT_PAPER_PROFILES.first())
            return parsed.distinctBy { it.id }
        }
    }
}
