package com.allstar.faithid.kiosk.ui.guide

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GuideGesturesTest {
    @Test
    fun `every moment has at least one gesture`() {
        GuideMoment.entries.forEach { assertTrue("$it", GuideGestures.TABLE[it].orEmpty().isNotEmpty()) }
    }

    @Test
    fun `good news is answered with a positive gesture, trouble with a tilt`() {
        val positive = setOf(Gesture.HOP, Gesture.DOUBLE_HOP, Gesture.NOD)
        listOf(GuideMoment.RECOGNIZED, GuideMoment.REGISTER_DONE, GuideMoment.SAVED, GuideMoment.STAR_COLLECTED).forEach {
            assertTrue("$it", GuideGestures.gesturesFor(it).any { g -> g in positive })
        }
        listOf(GuideMoment.UNKNOWN_FACE, GuideMoment.STALLED).forEach {
            assertEquals("$it", Gesture.TILT, GuideGestures.gesturesFor(it).first())
        }
    }

    @Test
    fun `idle plays one gesture picked from its list`() {
        assertEquals(listOf(Gesture.HOP), GuideGestures.gesturesFor(GuideMoment.IDLE) { 1 })
    }
}
