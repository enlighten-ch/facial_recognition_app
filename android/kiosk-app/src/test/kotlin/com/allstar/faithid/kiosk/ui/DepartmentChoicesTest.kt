package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.core.model.PersonMetadata
import org.junit.Assert.assertEquals
import org.junit.Test

class DepartmentChoicesTest {
    private val people = listOf(
        PersonMetadata.of("가", "유년부", "2학년"), PersonMetadata.of("나", "유년부", "10학년"), PersonMetadata.of("다", "유년부", "1학년"),
        PersonMetadata.of("라", "중고등부", "중2"), PersonMetadata.of("마", "중고등부", "고1"), PersonMetadata.of("바", "중고등부", "중1"),
        PersonMetadata.of("사", "청년부", ""), PersonMetadata.of("아", "", "3학년"),
    )

    @Test
    fun `most used departments come first`() {
        assertEquals(listOf("유년부", "중고등부", "청년부"), DepartmentChoices.departments(people))
    }

    @Test
    fun `grades are in natural order`() {
        assertEquals(listOf("1학년", "2학년", "10학년"), DepartmentChoices.grades(people, "유년부"))
        assertEquals(listOf("고1", "중1", "중2"), DepartmentChoices.grades(people, "중고등부"))
        assertEquals(emptyList<String>(), DepartmentChoices.grades(people, "청년부"))
    }

    @Test
    fun `a department always gets the same colour`() {
        assertEquals(DepartmentColors.of("유년부"), DepartmentColors.of(" 유년부 "))
    }
}
