package com.allstar.faithid.kiosk.settings

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KioskSettingsStoreTest {

    /** Settings JSON as saved by the 2026-05 kiosk with factory label values. */
    private fun legacy(extra: Map<String, Any> = emptyMap()) = JSONObject(
        mapOf(
            "DB_CLIENT_ID" to "client-edc621e6",
            "DB_AUTH_TOKEN" to "bb348175fd294d1b919547797557f1af",
            "TEMPLATE_TEXT_REGION_OFFSET_X" to 24, "TEMPLATE_TEXT_REGION_OFFSET_Y" to 24,
            "TEMPLATE_TEXT_REGION_WIDTH" to 360, "TEMPLATE_TEXT_REGION_HEIGHT" to 180,
            "PRINT_TEXT_SCALE_MULTIPLIER" to 5, "PRINT_ATTRIBUTE_COUNT" to 2,
            "PRINT_ATTRIBUTE_1_TEXT_SIZE_SP" to 40, "PRINT_ATTRIBUTE_2_TEXT_SIZE_SP" to 24, "PRINT_ATTRIBUTE_3_TEXT_SIZE_SP" to 20,
            "PRINT_ATTRIBUTE_3_OFFSET_Y" to 36,
            "UI_WELCOME_MESSAGE" to "어서 오세요",
        ) + extra,
    )

    @Test
    fun `old token becomes the kiosk secret and messages are kept`() {
        val s = KioskSettingsStore.fromJson(legacy())
        assertEquals("client-edc621e6", s.dbClientId)
        assertEquals("bb348175fd294d1b919547797557f1af", s.dbClientSecret)
        assertEquals("어서 오세요", s.messages.welcome)
    }

    @Test
    fun `untouched factory label layout is upgraded, a tuned one is kept`() {
        val upgraded = KioskSettingsStore.fromJson(legacy())
        assertTrue(upgraded.printLayoutStacked)
        assertEquals(2, upgraded.printTextScaleMultiplier)
        assertEquals(528, upgraded.templateTextRegionWidth)

        val tuned = KioskSettingsStore.fromJson(legacy(mapOf("PRINT_ATTRIBUTE_2_OFFSET_Y" to 50)))
        assertFalse(tuned.printLayoutStacked)
        assertEquals(5, tuned.printTextScaleMultiplier)
        assertEquals(50, tuned.labelLines[1].offsetY)
    }

    @Test
    fun `untouched factory oval moves below the guide's bubble, a tuned one is kept`() {
        val old = mapOf("REGISTER_ELLIPSE_AXIS_X_RATIO" to 0.22, "REGISTER_ELLIPSE_AXIS_Y_RATIO" to 0.30, "REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO" to 0.0, "REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA" to 0.60)
        val upgraded = KioskSettingsStore.fromJson(legacy(old))
        assertEquals(0.17f, upgraded.registerEllipseAxisXRatio)
        assertEquals(0.06f, upgraded.registerEllipseCenterYOffsetRatio)
        assertTrue(upgraded.registerCircleSteps)

        val tuned = KioskSettingsStore.fromJson(legacy(old + ("REGISTER_ELLIPSE_AXIS_X_RATIO" to 0.25)))
        assertEquals(0.25f, tuned.registerEllipseAxisXRatio)
        assertEquals(0.30f, tuned.registerEllipseAxisYRatio)
    }

    @Test
    fun `save and load round trip`() {
        val original = KioskSettings(
            dbClientId = "k1", dbClientSecret = "s".repeat(32), printLayoutStacked = false, topK = 4,
            guideAudience = GuideAudience.ADULTS, registerCircleSteps = false, registerBigCircleScale = 1.3f,
            guideLines = mapOf("kids.idle" to "어서 와!", "adults.print" to "출력"),
        )
        val back = KioskSettingsStore.fromJson(KioskSettingsStore.toJson(original))
        assertEquals(original, back)
    }
}
