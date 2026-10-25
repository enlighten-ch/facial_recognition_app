package com.allstar.faithid.kiosk.print

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.text.TextPaint
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.kiosk.settings.KioskSettings
import java.io.File
import java.io.FileInputStream
import java.io.InputStream

/**
 * Draws the name label: optional template image, then the configured lines (name, department,
 * grade) inside the text region. Thermal printers print at about 8 dots per millimetre.
 */
class LabelRenderer(private val context: Context) {

    /** The label as it looks on paper (template orientation), used for printing and the settings preview. */
    fun render(metadata: PersonMetadata, settings: KioskSettings): Bitmap {
        val (baseWidth, baseHeight) = paperPixels(settings)
        val landscape = settings.paperLandscape
        val width = if (landscape) baseHeight else baseWidth
        val height = if (landscape) baseWidth else baseHeight
        val bitmap = loadTemplate(settings, width, height, landscape) ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            Canvas(it).drawColor(Color.WHITE)
        }
        drawLines(Canvas(bitmap), metadata, settings, textRegion(settings, width, height))
        return bitmap
    }

    /** [render] rotated for the printer when the printer orientation differs from the template. */
    fun renderForPrinter(metadata: PersonMetadata, settings: KioskSettings): Bitmap {
        val label = render(metadata, settings)
        if (settings.paperLandscape == settings.printLandscape) return label
        val rotated = Bitmap.createBitmap(label, 0, 0, label.width, label.height, Matrix().apply { postRotate(90f) }, true)
        if (rotated !== label) label.recycle()
        return rotated
    }

    /** Dashed outline of the text region, drawn over the preview in the settings dialog only. */
    fun regionGuide(width: Int, height: Int, settings: KioskSettings): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val paint = Paint().apply {
            color = Color.argb(220, 50, 50, 50)
            style = Paint.Style.STROKE
            strokeWidth = 2f
            pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
            isAntiAlias = true
        }
        Canvas(bitmap).drawRect(textRegion(settings, width, height), paint)
        return bitmap
    }

    private fun drawLines(canvas: Canvas, metadata: PersonMetadata, settings: KioskSettings, rect: RectF) {
        val scale = settings.printTextScaleMultiplier.coerceAtLeast(1)
        val paints = mutableMapOf<Float, TextPaint>()
        fun paintFor(size: Float, bold: Boolean) = paints.getOrPut(if (bold) -size else size) {
            TextPaint().apply {
                color = Color.BLACK
                textSize = size
                isAntiAlias = true
                isFakeBoldText = bold
            }
        }
        val texts = settings.activeLabelLines.mapIndexed { i, line ->
            val (dx, dy) = renderOffsets(line.offsetX, line.offsetY, settings.paperLandscape)
            LabelText(
                text = metadata.valueOf(line.column),
                sizePx = (line.textSizeSp * scale).toFloat().coerceIn(10f, 240f),
                offsetX = dx,
                offsetY = dy,
                fixedBaseline = FIXED_BASELINES.getOrElse(i) { 0f },
            )
        }
        val region = Region(rect.left, rect.top, rect.right, rect.bottom)
        val firstLineBold = texts.firstOrNull()?.text
        val measure = { text: String, size: Float ->
            val paint = paintFor(size, text == firstLineBold)
            val fm = paint.fontMetrics
            TextMetrics(paint.measureText(text), fm.ascent, fm.descent)
        }
        val placed = if (settings.printLayoutStacked) LabelLayout.stacked(texts, region, measure) else LabelLayout.fixed(texts, region, measure)
        placed.forEachIndexed { i, p -> canvas.drawText(p.text, p.x, p.baseline, paintFor(p.sizePx, i == 0)) }
    }

    private fun loadTemplate(settings: KioskSettings, width: Int, height: Int, landscape: Boolean): Bitmap? {
        val source = settings.printerTemplateImageUri.trim()
        if (source.isEmpty()) return null
        return runCatching {
            val decoded = open(source)?.use { BitmapFactory.decodeStream(it) } ?: return null
            val rotation = ((settings.templateImageRotationQuarterTurns.mod(4) * 90) + if (landscape) 90 else 0) % 360
            val oriented = if (rotation != 0) {
                Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
                    .also { if (it !== decoded) decoded.recycle() }
            } else {
                decoded
            }
            val target = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(target)
            canvas.drawColor(Color.WHITE)
            val fit = minOf(width.toFloat() / oriented.width, height.toFloat() / oriented.height)
            val scale = if (settings.templateShrinkToPaperIfOversized || fit < 1f) fit else 1f
            val drawWidth = (oriented.width * scale).toInt().coerceAtLeast(1)
            val drawHeight = (oriented.height * scale).toInt().coerceAtLeast(1)
            val scaled = Bitmap.createScaledBitmap(oriented, drawWidth, drawHeight, true)
            canvas.drawBitmap(scaled, (width - drawWidth) / 2f, (height - drawHeight) / 2f, null)
            if (scaled !== oriented) scaled.recycle()
            oriented.recycle()
            target
        }.getOrNull()
    }

    private fun open(source: String): InputStream? = runCatching {
        if (source.startsWith("content://") || source.startsWith("file://")) {
            context.contentResolver.openInputStream(Uri.parse(source))
        } else {
            File(source).takeIf { it.isFile }?.let { FileInputStream(it) } ?: context.contentResolver.openInputStream(Uri.parse(source))
        }
    }.getOrNull()

    companion object {
        private const val DOTS_PER_MM = 8.0f

        /** Baselines of the pre-2026-10 fixed layout, relative to the first line. */
        private val FIXED_BASELINES = listOf(0f, 46f, 92f)

        fun paperPixels(settings: KioskSettings): Pair<Int, Int> {
            val paper = settings.selectedPaper
            return (paper.widthMm * DOTS_PER_MM).toInt().coerceIn(200, 1200) to (paper.heightMm * DOTS_PER_MM).toInt().coerceIn(120, 1600)
        }

        /** Offsets are stored for the portrait template and rotated with it. */
        fun renderOffsets(offsetX: Int, offsetY: Int, landscape: Boolean): Pair<Float, Float> =
            if (landscape) -offsetY.toFloat() to offsetX.toFloat() else offsetX.toFloat() to offsetY.toFloat()

        /** The text region is stored for the portrait template and rotated with it. */
        fun textRegion(settings: KioskSettings, width: Int, height: Int): RectF {
            if (!settings.paperLandscape) {
                val x = settings.templateTextRegionOffsetX.coerceIn(0, (width - 1).coerceAtLeast(0))
                val y = settings.templateTextRegionOffsetY.coerceIn(0, (height - 1).coerceAtLeast(0))
                val right = (x + settings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(width).let { if (it <= x) (x + 1).coerceAtMost(width) else it }
                val bottom = (y + settings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(height).let { if (it <= y) (y + 1).coerceAtMost(height) else it }
                return RectF(x.toFloat(), y.toFloat(), right.toFloat(), bottom.toFloat())
            }
            val portraitWidth = height
            val portraitHeight = width
            val sx = settings.templateTextRegionOffsetX.coerceIn(0, (portraitWidth - 1).coerceAtLeast(0))
            val sy = settings.templateTextRegionOffsetY.coerceIn(0, (portraitHeight - 1).coerceAtLeast(0))
            val sRight = (sx + settings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(portraitWidth)
            val sBottom = (sy + settings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(portraitHeight)
            val left = (portraitHeight - sBottom).coerceIn(0, width - 1)
            val top = sx.coerceIn(0, height - 1)
            return RectF(left.toFloat(), top.toFloat(), (portraitHeight - sy).coerceIn(left + 1, width).toFloat(), sRight.coerceIn(top + 1, height).toFloat())
        }
    }
}
