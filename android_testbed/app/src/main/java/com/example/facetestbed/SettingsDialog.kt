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

        section(content, context, "저장 위치")
        val faceDataDir = textField(content, context, "얼굴 정보 저장 폴더 경로 (비우면 기본 폴더)", current.faceDataDir)

        section(content, context, "인식")
        val authSim = numberField(content, context, "인식 최소 유사도 (높을수록 엄격)", current.authSimThreshold.toString(), false)
        val authMargin = numberField(content, context, "1, 2위 점수 차 최소값", current.authMarginThreshold.toString(), false)
        val topK = numberField(content, context, "후보 표시 개수", current.topK.toString(), true)
        val analyzeEvery = numberField(content, context, "몇 프레임마다 분석할지", current.analyzeEveryNFrames.toString(), true)
        val cameraIndex = numberField(content, context, "카메라 번호 (0=후면, 1=전면 우선)", current.cameraIndex.toString(), true)
        val useEllipseMask = boolField(content, context, "인식 시 타원 마스크 사용", current.recognitionUseEllipseMask)

        section(content, context, "등록")
        val regSamples = numberField(content, context, "방향별 수집 장수", current.registerSamplesPerDirection.toString(), true)
        val regCaptureEvery = numberField(content, context, "몇 프레임마다 샘플 수집", current.registerCaptureEveryNFrames.toString(), true)
        val regLow = numberField(content, context, "등록 유사도 하한", current.registerSimilarityLowerBoundary.toString(), false)
        val regHigh = numberField(content, context, "등록 유사도 상한 (중복 기준)", current.registerSimilarityHigherBoundary.toString(), false)
        val maxEmb = numberField(content, context, "사람별 최대 임베딩 수", current.maxEmbeddingsPerPerson.toString(), true)

        section(content, context, "가이드 타원")
        val ellipseX = numberField(content, context, "타원 너비 비율", current.registerEllipseAxisXRatio.toString(), false)
        val ellipseY = numberField(content, context, "타원 높이 비율", current.registerEllipseAxisYRatio.toString(), false)
        val ellipseOffset = numberField(content, context, "타원 중심 세로 오프셋 비율", current.registerEllipseCenterYOffsetRatio.toString(), false)
        val ellipseAlpha = numberField(content, context, "타원 밖 어둡게 처리 강도", current.registerEllipseOutsideDimAlpha.toString(), false)

        section(content, context, "자동 보정")
        val adaptiveEnabled = boolField(content, context, "자동 보정 사용", current.adaptiveUpdateEnabled)
        val adaptiveMax = numberField(content, context, "자동 보정 유사도 상한", current.adaptiveUpdateMaxSim.toString(), false)
        val adaptiveMinInterval = numberField(content, context, "자동 보정 최소 프레임 간격", current.adaptiveUpdateMinIntervalFrames.toString(), true)
        val adaptiveMaxSamples = numberField(content, context, "세션당 자동 보정 최대 횟수", current.adaptiveUpdateMaxSamplesPerSession.toString(), true)

        section(content, context, "프린터")
        val printEnabled = boolField(content, context, "프린터 사용", current.printEnabled)
        val printerName = textField(content, context, "프린터 이름 (비우면 기본)", current.printerName)
        val printerTimeout = numberField(content, context, "출력 대기시간(초)", current.printerTimeoutSec.toString(), true)
        val printerCopies = numberField(content, context, "출력 매수", current.printerCopies.toString(), true)
        val printerTitlePrefix = textField(content, context, "출력 작업 제목 접두어", current.printerJobTitlePrefix)
        val printerTemplate = textField(content, context, "출력 문구 템플릿", current.printerContentTemplate)
        val printBaseFont = numberField(content, context, "출력 기본 글자 크기", current.printTextBaseFontSize.toString(), true)
        val printScale = numberField(content, context, "출력 글자 확대 배율", current.printTextScaleMultiplier.toString(), true)

        AlertDialog.Builder(context)
            .setTitle("설정")
            .setView(root)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                val updated = AppSettings(
                    faceDataDir = faceDataDir.text.toString().trim(),
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
