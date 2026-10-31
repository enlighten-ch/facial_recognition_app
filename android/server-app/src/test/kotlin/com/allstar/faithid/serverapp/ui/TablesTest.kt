package com.allstar.faithid.serverapp.ui

import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TablesTest {
    private fun person(key: String, name: String, dept: String, grade: String, samples: Int = 12) =
        Person(key, PersonMetadata.of(name, dept, grade), List(samples) { FloatArray(4) })

    private fun row(key: String, name: String, dept: String, grade: String, first: Long? = null, onRoster: Boolean = true, kiosk: String? = null) =
        AttendanceRow(key, name, dept, grade, onRoster, if (first == null) 0 else 1, first, first, listOfNotNull(kiosk), emptyList())

    private val people = listOf(
        person("a", "김하은", "유년부", "2학년", 20),
        person("b", "이준호", "초등부", "5학년", 12),
        person("c", "박서윤", "유치부", "6세", 31),
        person("d", "최도윤", "유년부", "1학년", 15),
    )

    @Test
    fun `sort state reverses on the same column and resets on another`() {
        val s = SortState(PeopleTable.NAME)
        assertFalse(s.tap(PeopleTable.NAME).ascending)
        assertEquals(SortState(PeopleTable.SAMPLES, true), s.tap(PeopleTable.NAME).tap(PeopleTable.SAMPLES))
        assertEquals(" ▲", s.arrow(PeopleTable.NAME))
        assertEquals("", s.arrow(PeopleTable.GRADE))
    }

    @Test
    fun `known departments keep their colour and order, others hash stably`() {
        assertEquals(Departments.COLORS[2], Departments.color("유년부"))
        assertEquals(Departments.color("올스타반"), Departments.color(" 올스타반 "))
        assertEquals(Departments.NONE_COLOR, Departments.color(""))
        assertEquals(listOf("유치부", "유년부", "초등부", "가나반"), Departments.sorted(listOf("초등부", "가나반", "유년부", "유치부", "", "유년부")))
    }

    @Test
    fun `people search, filter and sort`() {
        val present = setOf("a", "c")
        assertEquals(listOf("김하은", "박서윤", "이준호", "최도윤"), PeopleTable.rows(people, present, "", null, SortState(PeopleTable.NAME)).map { it.person.name })
        assertEquals(listOf("최도윤", "김하은"), PeopleTable.rows(people, present, "", "유년부", SortState(PeopleTable.NAME, false)).map { it.person.name })
        assertEquals(listOf("이준호"), PeopleTable.rows(people, present, "5학년", null, SortState(PeopleTable.NAME)).map { it.person.name })
        assertEquals(listOf("박서윤", "최도윤", "김하은", "이준호"), PeopleTable.rows(people, present, "", null, SortState(PeopleTable.DEPARTMENT)).map { it.person.name })
        assertEquals("박서윤", PeopleTable.rows(people, present, "", null, SortState(PeopleTable.SAMPLES, false)).first().person.name)
        assertTrue(PeopleTable.rows(people, present, "", null, SortState(PeopleTable.TODAY)).take(2).all { it.presentToday })
    }

    @Test
    fun `attendance default order puts arrivals first by time then the absent by department`() {
        val rows = listOf(
            row("a", "김하은", "유년부", "2학년", first = 300, kiosk = "본당 로비"),
            row("b", "이준호", "초등부", "5학년"),
            row("c", "박서윤", "유치부", "6세", first = 100, kiosk = "유년부 입구"),
            row("d", "최도윤", "유년부", "1학년"),
            row("", "손님", "", "", first = 200, onRoster = false),
        )
        assertEquals(listOf("박서윤", "손님", "김하은", "최도윤", "이준호"), AttendanceTable.rows(rows, "", null, AttendanceShow.ALL, SortState(AttendanceTable.STATUS)).map { it.name })
        assertEquals(listOf("최도윤", "이준호"), AttendanceTable.rows(rows, "", null, AttendanceShow.ABSENT, SortState(AttendanceTable.STATUS)).map { it.name })
        assertEquals(listOf("김하은", "최도윤"), AttendanceTable.rows(rows, "", "유년부", AttendanceShow.ALL, SortState(AttendanceTable.NAME)).map { it.name })
        assertEquals(
            listOf(DepartmentSummary("유치부", 1, 1), DepartmentSummary("유년부", 1, 2), DepartmentSummary("초등부", 0, 1)),
            AttendanceTable.summary(rows),
        )
    }

    @Test
    fun `log lines are classified like the developer screen chips`() {
        assertEquals(LogKind.SERVER, LogKind.of("[2026-10-31 18:09:08] server started on port 18080"))
        assertEquals(LogKind.KIOSK, LogKind.of("approval requested: client=client-3f90, device=SM-T975N"))
        assertEquals(LogKind.PEOPLE, LogKind.of("person edited: key=p1, name=조예린, revision=1529"))
        assertEquals(LogKind.ATTENDANCE, LogKind.of("[2026-10-31 19:53:08] check-ins stored: client=client-7a1c, new=1, received=1"))
        assertEquals(LogKind.ERROR, LogKind.of("attendance sheet not written: Permission denied"))
        assertEquals(LogKind.ERROR, LogKind.of("server start failed: BindException: in use"))
    }

    @Test
    fun `config view hides the password hash and salt`() {
        val shown = ConfigView.masked("admin_id: \"admin\"\nadmin_password_hash: \"hash9f3\"\nadmin_password_salt: \"salt7c1\"\ndefault_port: \"18080\"\n")
        assertFalse(shown.contains("hash9f3") || shown.contains("salt7c1"))
        assertTrue(shown.contains("admin_id: \"admin\"") && shown.contains("default_port"))
    }
}
