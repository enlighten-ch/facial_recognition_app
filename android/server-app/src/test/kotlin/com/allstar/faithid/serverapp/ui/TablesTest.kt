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
        assertEquals("", s.arrow(PeopleTable.SAMPLES))
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
        val columns = RosterColumns.shown(people, emptyMap(), emptySet())
        fun rows(query: String = "", group: String? = null, sort: SortState) = PeopleTable.rows(people, present, query, columns, group, sort).map { it.person.name }
        assertEquals(listOf("김하은", "박서윤", "이준호", "최도윤"), rows(sort = SortState(PeopleTable.NAME)))
        assertEquals(listOf("최도윤", "김하은"), rows(group = "유년부", sort = SortState(PeopleTable.NAME, false)))
        assertEquals(listOf("이준호"), rows(query = "5학년", sort = SortState(PeopleTable.NAME)))
        assertEquals(listOf("박서윤", "최도윤", "김하은", "이준호"), rows(sort = SortState("department")))
        assertEquals("박서윤", rows(sort = SortState(PeopleTable.SAMPLES, false)).first())
        assertTrue(PeopleTable.rows(people, present, "", columns, null, SortState(PeopleTable.TODAY)).take(2).all { it.presentToday })
    }

    @Test
    fun `roster columns follow the people's columns, headers and hidden columns`() {
        val defaults = RosterColumns.shown(people, emptyMap(), setOf("family_name"))
        assertEquals(listOf("부서", "학년 · 나이"), defaults.map { it.label })

        // face_db.csv with the department column renamed to 반 and a new 메모 column, after "CSV 편집 반영".
        val edited = people.mapIndexed { i, p ->
            Person(p.key, PersonMetadata.of(linkedMapOf("name" to p.name, "반" to "${i + 1}반", "grade" to p.grade, "메모" to if (i == 0) "새가족" else "")), p.embeddings)
        } + Person("e", PersonMetadata.of("강", "율", "", ""), emptyList())
        val columns = RosterColumns.shown(edited, emptyMap(), setOf("family_name"))
        assertEquals(listOf("반", "grade", "메모"), columns.map { it.key })
        // The empty department column is left out but stays in the developer screen's list.
        assertTrue(RosterColumns.all(edited, emptyMap(), emptySet()).any { it.key == "department" && !it.shown })
        assertTrue(RosterColumns.all(edited, emptyMap(), emptySet()).any { it.key == "family_name" && it.shown })

        // Headers and hidden columns from the developer screen; the first column gives the filter chips.
        val renamed = RosterColumns.shown(edited, mapOf("department" to "반", "반" to "우리 반"), setOf("grade", "family_name"))
        assertEquals(listOf("우리 반", "메모"), renamed.map { it.label })
        assertEquals(listOf("김하은"), PeopleTable.rows(edited, emptySet(), "", renamed, "1반", SortState(PeopleTable.NAME)).map { it.person.name })
        assertEquals(listOf("김하은"), PeopleTable.rows(edited, emptySet(), "새가족", renamed, null, SortState(PeopleTable.NAME)).map { it.person.name })
    }

    @Test
    fun `tabs can group by any column but the name, falling back to the department`() {
        val withClass = people.mapIndexed { i, p -> p.copy(metadata = p.metadata.withValue("반", if (i < 2) "1반" else "2반")) }
        val choices = Grouping.choices(RosterColumns.all(withClass, mapOf("반" to "우리 반"), emptySet()))
        assertEquals(listOf("department", "grade", "반"), choices.map { it.key })
        assertEquals("반", Grouping.pick("반", choices)!!.key)
        assertEquals("department", Grouping.pick("없는열", choices)!!.key)

        val grouping = Grouping("반", "우리 반", withClass.associateBy { it.key })
        val rows = listOf(
            row("a", "김하은", "유년부", "2학년", first = 300),
            row("b", "이준호", "초등부", "5학년"),
            row("c", "박서윤", "유치부", "6세", first = 100),
            row("d", "최도윤", "유년부", "1학년"),
            row("", "손님", "유년부", "", first = 200, onRoster = false),
        )
        assertEquals(listOf(GroupSummary("1반", 1, 2), GroupSummary("2반", 1, 2)), AttendanceTable.summary(rows, grouping))
        assertEquals(listOf("김하은", "이준호"), AttendanceTable.rows(rows, "", "1반", AttendanceShow.ALL, SortState(AttendanceTable.NAME), grouping).map { it.name })
        // 명단: the chips use the chosen column.
        val columns = RosterColumns.shown(withClass, emptyMap(), emptySet())
        assertEquals(listOf("박서윤", "최도윤"), PeopleTable.rows(withClass, emptySet(), "", columns, "2반", SortState(PeopleTable.NAME), groupColumn = "반").map { it.person.name })
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
            listOf(GroupSummary("유치부", 1, 1), GroupSummary("유년부", 1, 2), GroupSummary("초등부", 0, 1)),
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
