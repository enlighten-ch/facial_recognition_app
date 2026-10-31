package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.core.model.PersonMetadata

/** A stable colour per department name, from the same six colours as the server app. */
object DepartmentColors {
    private val COLORS = intArrayOf(
        0xFFF2A7B5.toInt(), 0xFFF6C453.toInt(), 0xFF7CC7A4.toInt(),
        0xFF6FA8E8.toInt(), 0xFFA98BE0.toInt(), 0xFF9AA3B5.toInt(),
    )

    fun of(department: String): Int = COLORS[Math.floorMod(department.trim().hashCode(), COLORS.size)]
}

/**
 * The department and grade chips of the name form, taken from the people already on this kiosk:
 * the most used first, so a teacher usually only types the name.
 */
object DepartmentChoices {
    const val MAX_CHIPS = 10

    fun departments(people: List<PersonMetadata>): List<String> =
        people.map { it.department.trim() }.filter { it.isNotEmpty() }
            .groupingBy { it }.eachCount()
            .entries.sortedWith(compareByDescending<Map.Entry<String, Int>> { it.value }.thenBy { it.key })
            .take(MAX_CHIPS).map { it.key }

    /** Grades used in [department], in natural order ("2학년" before "10학년", "중1" before "중2"). */
    fun grades(people: List<PersonMetadata>, department: String): List<String> =
        people.filter { it.department.trim() == department.trim() }
            .map { it.grade.trim() }.filter { it.isNotEmpty() }.distinct()
            .sortedWith(compareBy<String>({ it.replace(DIGITS, "") }, { DIGITS.find(it)?.value?.toIntOrNull() ?: Int.MAX_VALUE }, { it }))
            .take(MAX_CHIPS)

    private val DIGITS = Regex("\\d+")
}
