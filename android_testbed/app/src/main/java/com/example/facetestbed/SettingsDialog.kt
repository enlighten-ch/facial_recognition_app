package com.example.facetestbed

import android.app.AlertDialog
import android.content.Context
import android.text.InputType
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView

object SettingsDialog {
    fun show(context: Context, current: AppSettings, onSave: (AppSettings) -> Unit) {
        val root = ScrollView(context)
        val content = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(context, 16), dp(context, 12), dp(context, 16), dp(context, 12))
        }
        root.addView(content, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        section(content, context, "Recognition")
        val authSim = numberField(content, context, "AUTH_SIM_THRESHOLD", current.authSimThreshold.toString(), false)
        val authMargin = numberField(content, context, "AUTH_MARGIN_THRESHOLD", current.authMarginThreshold.toString(), false)
        val topK = numberField(content, context, "TOP_K", current.topK.toString(), true)
        val analyzeEvery = numberField(content, context, "ANALYZE_EVERY_N_FRAMES", current.analyzeEveryNFrames.toString(), true)
        val cameraIndex = numberField(content, context, "CAMERA_INDEX (0=back,1=front)", current.cameraIndex.toString(), true)
        val useEllipseMask = boolField(content, context, "RECOGNITION_USE_ELLIPSE_MASK", current.recognitionUseEllipseMask)

        section(content, context, "Registration")
        val regSamples = numberField(content, context, "REGISTER_SAMPLES_PER_DIRECTION", current.registerSamplesPerDirection.toString(), true)
        val regCaptureEvery = numberField(content, context, "REGISTER_CAPTURE_EVERY_N_FRAMES", current.registerCaptureEveryNFrames.toString(), true)
        val regLow = numberField(content, context, "REGISTER_SIMILARITY_LOWER_BOUNDARY", current.registerSimilarityLowerBoundary.toString(), false)
        val regHigh = numberField(content, context, "REGISTER_SIMILARITY_HIGHER_BOUNDARY", current.registerSimilarityHigherBoundary.toString(), false)
        val maxEmb = numberField(content, context, "MAX_EMBEDDINGS_PER_PERSON", current.maxEmbeddingsPerPerson.toString(), true)

        section(content, context, "GuideEllipse")
        val ellipseX = numberField(content, context, "REGISTER_ELLIPSE_AXIS_X_RATIO", current.registerEllipseAxisXRatio.toString(), false)
        val ellipseY = numberField(content, context, "REGISTER_ELLIPSE_AXIS_Y_RATIO", current.registerEllipseAxisYRatio.toString(), false)
        val ellipseOffset = numberField(content, context, "REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO", current.registerEllipseCenterYOffsetRatio.toString(), false)
        val ellipseAlpha = numberField(content, context, "REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA", current.registerEllipseOutsideDimAlpha.toString(), false)

        section(content, context, "AdaptiveUpdate")
        val adaptiveEnabled = boolField(content, context, "ADAPTIVE_UPDATE_ENABLED", current.adaptiveUpdateEnabled)
        val adaptiveMax = numberField(content, context, "ADAPTIVE_UPDATE_MAX_SIM", current.adaptiveUpdateMaxSim.toString(), false)
        val adaptiveMinInterval = numberField(content, context, "ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES", current.adaptiveUpdateMinIntervalFrames.toString(), true)
        val adaptiveMaxSamples = numberField(content, context, "ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION", current.adaptiveUpdateMaxSamplesPerSession.toString(), true)

        section(content, context, "Printer")
        val printEnabled = boolField(content, context, "PRINT_ENABLED", current.printEnabled)
        val printerName = textField(content, context, "PRINTER_NAME", current.printerName)
        val printerTimeout = numberField(content, context, "PRINTER_TIMEOUT_SEC", current.printerTimeoutSec.toString(), true)
        val printerCopies = numberField(content, context, "PRINTER_COPIES", current.printerCopies.toString(), true)
        val printerTitlePrefix = textField(content, context, "PRINTER_JOB_TITLE_PREFIX", current.printerJobTitlePrefix)
        val printerTemplate = textField(content, context, "PRINTER_CONTENT_TEMPLATE", current.printerContentTemplate)
        val printBaseFont = numberField(content, context, "PRINT_TEXT_BASE_FONT_SIZE", current.printTextBaseFontSize.toString(), true)
        val printScale = numberField(content, context, "PRINT_TEXT_SCALE_MULTIPLIER", current.printTextScaleMultiplier.toString(), true)

        AlertDialog.Builder(context)
            .setTitle("Settings")
            .setView(root)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save") { _, _ ->
                val updated = AppSettings(
                    authSimThreshold = authSim.asFloat(current.authSimThreshold),
                    authMarginThreshold = authMargin.asFloat(current.authMarginThreshold),
                    topK = topK.asInt(current.topK).coerceAtLeast(1),
                    analyzeEveryNFrames = analyzeEvery.asInt(current.analyzeEveryNFrames).coerceAtLeast(1),
                    cameraIndex = cameraIndex.asInt(current.cameraIndex).coerceIn(0, 1),
                    recognitionUseEllipseMask = useEllipseMask.isChecked,
                    registerSamplesPerDirection = regSamples.asInt(current.registerSamplesPerDirection).coerceAtLeast(1),
                    registerCaptureEveryNFrames = regCaptureEvery.asInt(current.registerCaptureEveryNFrames).coerceAtLeast(1),
                    registerSimilarityLowerBoundary = regLow.asFloat(current.registerSimilarityLowerBoundary),
                    registerSimilarityHigherBoundary = regHigh.asFloat(current.registerSimilarityHigherBoundary),
                    maxEmbeddingsPerPerson = maxEmb.asInt(current.maxEmbeddingsPerPerson).coerceAtLeast(1),
                    registerEllipseAxisXRatio = ellipseX.asFloat(current.registerEllipseAxisXRatio),
                    registerEllipseAxisYRatio = ellipseY.asFloat(current.registerEllipseAxisYRatio),
                    registerEllipseCenterYOffsetRatio = ellipseOffset.asFloat(current.registerEllipseCenterYOffsetRatio),
                    registerEllipseOutsideDimAlpha = ellipseAlpha.asFloat(current.registerEllipseOutsideDimAlpha),
                    adaptiveUpdateEnabled = adaptiveEnabled.isChecked,
                    adaptiveUpdateMaxSim = adaptiveMax.asFloat(current.adaptiveUpdateMaxSim),
                    adaptiveUpdateMinIntervalFrames = adaptiveMinInterval.asInt(current.adaptiveUpdateMinIntervalFrames).coerceAtLeast(1),
                    adaptiveUpdateMaxSamplesPerSession = adaptiveMaxSamples.asInt(current.adaptiveUpdateMaxSamplesPerSession).coerceAtLeast(1),
                    printEnabled = printEnabled.isChecked,
                    printerName = printerName.text.toString(),
                    printerTimeoutSec = printerTimeout.asInt(current.printerTimeoutSec).coerceAtLeast(1),
                    printerCopies = printerCopies.asInt(current.printerCopies).coerceAtLeast(1),
                    printerJobTitlePrefix = printerTitlePrefix.text.toString(),
                    printerContentTemplate = printerTemplate.text.toString(),
                    printTextBaseFontSize = printBaseFont.asInt(current.printTextBaseFontSize).coerceAtLeast(1),
                    printTextScaleMultiplier = printScale.asInt(current.printTextScaleMultiplier).coerceAtLeast(1),
                )
                onSave(updated)
            }
            .show()
    }

    private fun section(parent: LinearLayout, context: Context, title: String) {
        val t = TextView(context).apply {
            text = title
            textSize = 18f
            setPadding(0, dp(context, 12), 0, dp(context, 8))
        }
        parent.addView(t)
    }

    private fun numberField(parent: LinearLayout, context: Context, label: String, value: String, integer: Boolean): EditText {
        val field = textField(parent, context, label, value)
        field.inputType = if (integer) {
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
        } else {
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
        }
        return field
    }

    private fun textField(parent: LinearLayout, context: Context, label: String, value: String): EditText {
        val box = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(context, 4), 0, dp(context, 4))
        }
        val tv = TextView(context).apply { text = label }
        val et = EditText(context).apply {
            setText(value)
            setSingleLine(true)
        }
        box.addView(tv)
        box.addView(et)
        parent.addView(box)
        return et
    }

    private fun boolField(parent: LinearLayout, context: Context, label: String, value: Boolean): Switch {
        val sw = Switch(context).apply {
            text = label
            isChecked = value
            setPadding(0, dp(context, 4), 0, dp(context, 4))
        }
        parent.addView(sw)
        return sw
    }

    private fun EditText.asInt(default: Int): Int = text.toString().trim().toIntOrNull() ?: default
    private fun EditText.asFloat(default: Float): Float = text.toString().trim().toFloatOrNull() ?: default
    private fun dp(context: Context, v: Int): Int = (v * context.resources.displayMetrics.density).toInt()
}
