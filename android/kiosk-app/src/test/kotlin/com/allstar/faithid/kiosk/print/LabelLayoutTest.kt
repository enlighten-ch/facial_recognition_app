package com.allstar.faithid.kiosk.print

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LabelLayoutTest {
    /** Monospace approximation: each character is 0.9 em wide; ascent 0.9 em, descent 0.25 em. */
    private val measure = { text: String, size: Float -> TextMetrics(text.length * size * 0.9f, -size * 0.9f, size * 0.25f) }
    private val region = Region(24f, 24f, 552f, 376f) // default region on a 72x50 mm label (576x400 px)

    private fun lines(nameSize: Float = 88f) = listOf(
        LabelText("홍길동", nameSize, 0f, 0f, 0f),
        LabelText("청년부", 48f, 0f, 0f, 46f),
        LabelText("3학년", 40f, 0f, 0f, 92f),
    )

    private fun bounds(p: PlacedText): ClosedFloatingPointRange<Float> {
        val m = measure(p.text, p.sizePx)
        return (p.baseline + m.ascent)..(p.baseline + m.descent)
    }

    @Test
    fun `stacked lines never overlap and stay inside the region`() {
        val placed = LabelLayout.stacked(lines(), region, measure)
        assertEquals(3, placed.size)
        placed.zipWithNext().forEach { (upper, lower) ->
            assertTrue("${upper.text} overlaps ${lower.text}", bounds(upper).endInclusive <= bounds(lower).start)
        }
        placed.forEach { p ->
            assertTrue(bounds(p).start >= region.top && bounds(p).endInclusive <= region.bottom)
            assertTrue(p.x + measure(p.text, p.sizePx).width <= region.right)
        }
    }

    @Test
    fun `a small region shrinks all lines so they still do not overlap`() {
        val small = Region(24f, 24f, 384f, 204f) // the pre-2026-10 default region (360 x 180)
        val placed = LabelLayout.stacked(lines(nameSize = 200f), small, measure)
        placed.zipWithNext().forEach { (upper, lower) -> assertTrue(bounds(upper).endInclusive <= bounds(lower).start + 0.01f) }
        placed.forEach { assertTrue(bounds(it).endInclusive <= small.bottom) }
    }

    @Test
    fun `the old fixed layout overlaps with large text, which is why stacking is the default`() {
        val placed = LabelLayout.fixed(lines(nameSize = 200f), region, measure)
        val overlapping = bounds(placed[0]).endInclusive > bounds(placed[1]).start
        assertTrue(overlapping)
    }

    @Test
    fun `long text shrinks to the region width`() {
        val longName = LabelText("가".repeat(20), 88f, 0f, 0f)
        val placed = LabelLayout.stacked(listOf(longName), region, measure).single()
        assertTrue(placed.sizePx < 88f)
        assertTrue(measure(placed.text, placed.sizePx).width <= region.width - 2 * LabelLayout.PADDING)
    }

    @Test
    fun `offsets move a line but cannot push it off the label, and blank lines are skipped`() {
        val placed = LabelLayout.stacked(
            listOf(LabelText("A", 40f, 10_000f, 10_000f), LabelText(" ", 40f, 0f, 0f)),
            region,
            measure,
        )
        assertEquals(1, placed.size)
        val p = placed.single()
        assertTrue(p.x + measure("A", 40f).width <= region.right - LabelLayout.PADDING + 0.01f)
        assertTrue(p.baseline + measure("A", 40f).descent <= region.bottom - LabelLayout.PADDING + 0.01f)
    }
}
