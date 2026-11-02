package com.allstar.faithid.core.model

/**
 * Family and given name of a person. The full name stays in [PersonMetadata.name]; people
 * registered with separate fields also carry [PersonMetadata.familyName], and for everyone else the
 * split of a Korean name is guessed.
 */
object PersonNames {
    /** Two-syllable family names, for guessing the split of four-syllable names ("남궁민수"). */
    private val DOUBLE_FAMILY_NAMES = setOf("남궁", "황보", "제갈", "선우", "독고", "사공", "서문", "동방", "어금", "망절")

    /** "김" + "하은" → "김하은"; names in other scripts keep a space ("Kim Haeun"). */
    fun join(family: String, given: String): String {
        val f = family.trim()
        val g = given.trim()
        if (f.isEmpty() || g.isEmpty()) return f + g
        return if (isHangul(f) && isHangul(g)) f + g else "$f $g"
    }

    fun family(metadata: PersonMetadata): String = split(metadata).first

    /** "하은" for 김하은; the whole name when no split is known (two-syllable names, other scripts). */
    fun given(metadata: PersonMetadata): String = split(metadata).second

    /**
     * The name with the chosen parts: both = the full name, given only = "하은", family only = "김".
     * Falls back to the full name when the chosen part is unknown or nothing is chosen.
     */
    fun display(metadata: PersonMetadata, family: Boolean, given: Boolean): String {
        val (f, g) = split(metadata)
        return when {
            family && !given -> f.ifEmpty { metadata.name }
            given && !family -> g.ifEmpty { metadata.name }
            else -> metadata.name
        }
    }

    private fun split(metadata: PersonMetadata): Pair<String, String> {
        val name = metadata.name.trim()
        val stored = metadata.familyName.trim()
        if (stored.isNotEmpty() && name.length > stored.length && name.startsWith(stored)) {
            return stored to name.removePrefix(stored).trim()
        }
        if (!isHangul(name) || name.length !in 3..4) return "" to name
        val familyLength = if (name.length == 4 && name.take(2) in DOUBLE_FAMILY_NAMES) 2 else 1
        return name.take(familyLength) to name.drop(familyLength)
    }

    private fun isHangul(text: String) = text.isNotEmpty() && text.all { it in '가'..'힣' }
}
