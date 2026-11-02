package com.allstar.faithid.kiosk.ui

/**
 * Labels of the name form, after the column names set on the server ("department" → "반").
 * Columns the server did not rename keep the kiosk's own words.
 */
class FormLabels(labels: Map<String, String>) {
    val familyName: String = labels["family_name"] ?: "성"
    val name: String = labels["given_name"] ?: labels["name"] ?: "이름"
    val department: String = labels["department"] ?: "부서"
    val grade: String = labels["grade"] ?: "학년 · 나이"
}
