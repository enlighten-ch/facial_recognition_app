package com.allstar.faithid.kiosk.print

/** Font metrics of one text at one size (ascent is negative, as in Android's FontMetrics). */
data class TextMetrics(val width: Float, val ascent: Float, val descent: Float)

/** A text line to place on the label. [fixedBaseline] is used only by the fixed layout. */
data class LabelText(
    val text: String,
    val sizePx: Float,
    val offsetX: Float,
    val offsetY: Float,
    val fixedBaseline: Float = 0f,
)

data class Region(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

data class PlacedText(val text: String, val sizePx: Float, val x: Float, val baseline: Float)

/**
 * Where the name, department and grade go on the label.
 *
 * - [stacked] (default since 2026-10): lines one below the other by their real height, each shrunk
 *   to fit the region width, then shifted by the user's offsets.
 * - [fixed] (older kiosks): the original fixed baselines (0, 46, 92 px below the first line top),
 *   which overlap when the text is large, kept so existing label setups print unchanged.
 *
 * All positions stay inside the region.
 */
object LabelLayout {
    const val PADDING = 8f
    const val LINE_GAP = 6f
    const val MIN_TEXT_PX = 12f

    fun stacked(lines: List<LabelText>, region: Region, measure: (String, Float) -> TextMetrics): List<PlacedText> {
        val visible = lines.filter { it.text.isNotBlank() }
        val maxWidth = (region.width - 2 * PADDING).coerceAtLeast(1f)
        val maxHeight = (region.height - 2 * PADDING).coerceAtLeast(1f)
        var sizes = visible.map { fitSize(it.text, it.sizePx, maxWidth, measure) }
        // Shrink all lines together until the stack fits the region height.
        repeat(8) {
            val total = stackHeight(visible, sizes, measure)
            if (total <= maxHeight) return@repeat
            val factor = maxHeight / total
            sizes = sizes.map { (it * factor).coerceAtLeast(MIN_TEXT_PX) }
        }
        var cursorTop = region.top + PADDING
        return visible.mapIndexed { i, line ->
            val size = sizes[i]
            val metrics = measure(line.text, size)
            val baseline = cursorTop - metrics.ascent
            cursorTop = baseline + metrics.descent + LINE_GAP
            clamp(line.text, size, region.left + PADDING + line.offsetX, baseline + line.offsetY, metrics, region)
        }
    }

    fun fixed(lines: List<LabelText>, region: Region, measure: (String, Float) -> TextMetrics): List<PlacedText> =
        lines.filter { it.text.isNotBlank() }.map { line ->
            val metrics = measure(line.text, line.sizePx)
            val baseline = region.top + PADDING + line.sizePx + line.fixedBaseline + line.offsetY
            clamp(line.text, line.sizePx, region.left + PADDING + line.offsetX, baseline, metrics, region)
        }

    private fun stackHeight(lines: List<LabelText>, sizes: List<Float>, measure: (String, Float) -> TextMetrics): Float =
        lines.indices.sumOf { i -> measure(lines[i].text, sizes[i]).let { (it.descent - it.ascent).toDouble() } }.toFloat() +
            LINE_GAP * (lines.size - 1).coerceAtLeast(0)

    /** Largest size not above [requested] whose width fits [maxWidth]. */
    fun fitSize(text: String, requested: Float, maxWidth: Float, measure: (String, Float) -> TextMetrics): Float {
        var size = requested
        while (size > MIN_TEXT_PX && measure(text, size).width > maxWidth) size *= 0.92f
        return size.coerceAtLeast(MIN_TEXT_PX)
    }

    private fun clamp(text: String, size: Float, x: Float, baseline: Float, metrics: TextMetrics, region: Region): PlacedText {
        val minX = region.left + PADDING
        val maxX = (region.right - PADDING - metrics.width).coerceAtLeast(minX)
        val minBaseline = region.top + PADDING - metrics.ascent
        val maxBaseline = (region.bottom - PADDING - metrics.descent).coerceAtLeast(minBaseline)
        return PlacedText(text, size, x.coerceIn(minX, maxX), baseline.coerceIn(minBaseline, maxBaseline))
    }
}
