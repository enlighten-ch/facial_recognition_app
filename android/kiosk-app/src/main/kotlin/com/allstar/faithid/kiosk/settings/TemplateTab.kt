package com.allstar.faithid.kiosk.settings

import android.content.Context
import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.kiosk.print.LabelRenderer

/**
 * "출력 템플릿": template image, text region and the lines on the label, with a live preview.
 *
 * Region and offsets are stored for the portrait template; while the landscape template is
 * selected the fields show them rotated, as the administrator sees them in the preview.
 */
class TemplateTab(
    context: Context,
    private val initial: KioskSettings,
    private val renderer: LabelRenderer,
    private val draft: () -> KioskSettings,
    private val pickTemplate: ((String?) -> Unit) -> Unit,
) : SettingsTab {
    override val title = "출력 템플릿"
    private val form = SettingsForm(context)
    private val previewImage = ImageView(context).apply {
        adjustViewBounds = true
        scaleType = ImageView.ScaleType.FIT_CENTER
    }
    private val guideImage = ImageView(context).apply {
        adjustViewBounds = true
        scaleType = ImageView.ScaleType.FIT_CENTER
    }
    private val previewStatus = TextView(context)

    override val view = LinearLayout(context).apply {
        orientation = LinearLayout.HORIZONTAL
        val pad = form.dp(8)
        setPadding(pad, pad, pad, pad)
        val preview = SettingsForm.verticalLayout(context)
        preview.addView(
            FrameLayout(context).apply {
                setBackgroundColor(0xFFECEFF4.toInt())
                setPadding(pad, pad, pad, pad)
                addView(previewImage, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
                addView(guideImage, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            },
        )
        preview.addView(previewStatus)
        addView(preview, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = pad })
        addView(ScrollView(context).apply { addView(form.root) }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
    }

    private var templateUri = initial.printerTemplateImageUri
    private var quarterTurns = initial.templateImageRotationQuarterTurns.mod(4)
    private var landscape = initial.paperLandscape
    private var previewBitmap: Bitmap? = null
    private var guideBitmap: Bitmap? = null

    init { form.section("템플릿 이미지") }
    private val templateStatus = form.note(templateLabel())

    init {
        form.button("이미지 불러오기") {
            pickTemplate { picked ->
                if (picked != null) {
                    templateUri = picked
                    templateStatus.text = templateLabel()
                    refresh()
                }
            }
        }
        form.button("이미지 지우기") {
            templateUri = ""
            templateStatus.text = templateLabel()
            refresh()
        }
    }

    private val rotationStatus = form.note("이미지 회전: ${quarterTurns * 90}도")

    init {
        form.button("이미지 시계방향 90도 회전") {
            quarterTurns = (quarterTurns + 1) % 4
            rotationStatus.text = "이미지 회전: ${quarterTurns * 90}도"
            refresh()
        }
    }

    private val orientation = form.choice("템플릿 방향", listOf("세로", "가로"), if (landscape) 1 else 0) { position ->
        val next = position == 1
        if (next != landscape) {
            val stored = storedValues()
            landscape = next
            showValues(stored)
        }
        refresh()
    }
    private val shrink = form.switch("이미지가 용지보다 크면 줄이기", initial.templateShrinkToPaperIfOversized)

    init { form.section("글자 영역 (점선)") }
    private val regionXY = form.pair("X 위치(px)", 0, "Y 위치(px)", 0)
    private val regionSize = form.pair("너비(px)", 0, "높이(px)", 0)
    private val stacked = form.switch("줄 간격 자동 맞춤 (끄면 이전 버전처럼 고정 위치)", initial.printLayoutStacked)

    init { form.section("이름표에 들어갈 줄") }
    private val count = form.choice("줄 수", listOf("1줄", "2줄", "3줄"), initial.printAttributeCount.coerceIn(1, 3) - 1) { refresh() }
    private val scale = form.number("글자 확대 배율", initial.printTextScaleMultiplier)
    private val lines = initial.labelLines.mapIndexed { i, line ->
        val column = form.choice(
            "${i + 1}번째 줄 내용",
            KioskSettings.LABEL_COLUMNS.map { it.second },
            KioskSettings.LABEL_COLUMNS.indexOfFirst { it.first == line.column }.coerceAtLeast(0),
        ) { refresh() }
        val size = form.number("${i + 1}번째 줄 글자 크기", line.textSizeSp)
        val offset = form.pair("${i + 1}번째 줄 X 이동", 0, "Y 이동", 0)
        LineFields(column, size, offset.first, offset.second)
    }

    private class LineFields(val column: android.widget.Spinner, val size: EditText, val x: EditText, val y: EditText)

    /** Region and offsets in stored (portrait) coordinates. */
    private data class Stored(val x: Int, val y: Int, val w: Int, val h: Int, val offsets: List<Pair<Int, Int>>)

    init {
        showValues(
            Stored(
                initial.templateTextRegionOffsetX,
                initial.templateTextRegionOffsetY,
                initial.templateTextRegionWidth,
                initial.templateTextRegionHeight,
                initial.labelLines.map { it.offsetX to it.offsetY },
            ),
        )
        (listOf(regionXY.first, regionXY.second, regionSize.first, regionSize.second, scale) + lines.flatMap { listOf(it.size, it.x, it.y) })
            .forEach { it.onChanged { refresh() } }
        shrink.setOnCheckedChangeListener { _, _ -> refresh() }
        stacked.setOnCheckedChangeListener { _, _ -> refresh() }
    }

    fun refresh() {
        val settings = runCatching { draft() }.getOrNull() ?: return
        val rendered = runCatching { renderer.render(PersonMetadata.of("홍길동", "올스타반", "3학년"), settings) }.getOrNull()
        previewBitmap?.recycle()
        guideBitmap?.recycle()
        previewBitmap = rendered
        guideBitmap = rendered?.let { renderer.regionGuide(it.width, it.height, settings) }
        previewImage.setImageBitmap(previewBitmap)
        guideImage.setImageBitmap(guideBitmap)
        previewStatus.text = if (rendered == null) "미리보기를 만들지 못했습니다." else "${settings.selectedPaper.name} · ${if (landscape) "가로" else "세로"} 템플릿 미리보기 (점선은 화면에만 보입니다)"
    }

    fun release() {
        previewBitmap?.recycle()
        guideBitmap?.recycle()
        previewBitmap = null
        guideBitmap = null
    }

    override fun applyTo(settings: KioskSettings): KioskSettings {
        val stored = storedValues(settings)
        return settings.copy(
            printerTemplateImageUri = templateUri,
            templateImageRotationQuarterTurns = quarterTurns,
            templateShrinkToPaperIfOversized = shrink.isChecked,
            paperLandscape = orientation.selectedItemPosition == 1,
            templateTextRegionOffsetX = stored.x,
            templateTextRegionOffsetY = stored.y,
            templateTextRegionWidth = stored.w,
            templateTextRegionHeight = stored.h,
            printLayoutStacked = stacked.isChecked,
            printTextScaleMultiplier = scale.intValue(settings.printTextScaleMultiplier).coerceIn(1, 10),
            printAttributeCount = count.selectedItemPosition + 1,
            labelLines = lines.mapIndexed { i, f ->
                LabelLine(
                    column = KioskSettings.LABEL_COLUMNS.getOrNull(f.column.selectedItemPosition)?.first ?: "name",
                    textSizeSp = f.size.intValue(settings.labelLines.getOrNull(i)?.textSizeSp ?: 24).coerceIn(8, 120),
                    offsetX = stored.offsets[i].first,
                    offsetY = stored.offsets[i].second,
                )
            },
        )
    }

    private fun templateLabel() = if (templateUri.isBlank()) "이미지 없음 (흰 바탕에 글자만 출력)" else "선택된 이미지: $templateUri"

    private fun storedValues(settings: KioskSettings = draftOrInitial()): Stored {
        val portraitHeight = LabelRenderer.paperPixels(settings).second
        val x = regionXY.first.intValue(0)
        val y = regionXY.second.intValue(0)
        val w = regionSize.first.intValue(1).coerceAtLeast(1)
        val h = regionSize.second.intValue(1).coerceAtLeast(1)
        val offsets = lines.map { f ->
            val dx = f.x.intValue(0)
            val dy = f.y.intValue(0)
            if (landscape) dy to -dx else dx to dy
        }
        return if (landscape) Stored(y, portraitHeight - (x + w), h, w, offsets) else Stored(x, y, w, h, offsets)
    }

    private fun showValues(stored: Stored) {
        val portraitHeight = LabelRenderer.paperPixels(draftOrInitial()).second
        val shown = if (landscape) intArrayOf(portraitHeight - (stored.y + stored.h), stored.x, stored.h, stored.w) else intArrayOf(stored.x, stored.y, stored.w, stored.h)
        regionXY.first.setText(shown[0].toString())
        regionXY.second.setText(shown[1].toString())
        regionSize.first.setText(shown[2].toString())
        regionSize.second.setText(shown[3].toString())
        lines.forEachIndexed { i, f ->
            val (ox, oy) = stored.offsets.getOrElse(i) { 0 to 0 }
            val (dx, dy) = if (landscape) -oy to ox else ox to oy
            f.x.setText(dx.toString())
            f.y.setText(dy.toString())
        }
    }

    private var drafting = false

    /** The draft includes this tab; while building it, fall back to the initial paper size. */
    private fun draftOrInitial(): KioskSettings {
        if (drafting) return initial
        drafting = true
        return try {
            runCatching { draft() }.getOrDefault(initial)
        } finally {
            drafting = false
        }
    }
}
