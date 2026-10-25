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
    fun `save and load round trip`() {
        val original = KioskSettings(dbClientId = "k1", dbClientSecret = "s".repeat(32), printLayoutStacked = false, topK = 4)
        val back = KioskSettingsStore.fromJson(KioskSettingsStore.toJson(original))
        assertEquals(original, back)
    }
}
