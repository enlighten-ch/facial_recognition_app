package com.allstar.faithid.core.attendance

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

class AttendanceReportTest {
    private val zone = ZoneId.of("Asia/Seoul")
    private val report = AttendanceReport(zone)
    private val faces = FakeFaces(seed = 1, dim = 8)
    private val day = LocalDate.of(2026, 10, 25)

    private val kimA = faces.person("김철수", "청년부", "1학년", key = "kimA")
    private val kimB = faces.person("김철수", "중등부", "2학년", key = "kimB")
    private val hong = faces.person("홍길동", "청년부", "3학년", key = "hong")
    private val roster = listOf(hong, kimA, kimB)

    private fun at(hour: Int, minute: Int, date: LocalDate = day) = ZonedDateTime.of(date.atTime(hour, minute), zone).toInstant().toEpochMilli()

    private fun event(id: String, key: String, name: String, time: Long, kiosk: String = "k1", source: CheckInSource = CheckInSource.RECOGNIZED, dept: String = "", grade: String = "") =
        AttendanceEvent(id, key, name, dept, grade, kiosk, time, source, 0.6f)

    @Test
    fun `homonyms are counted separately and repeat check-ins keep first and last time`() {
        val events = listOf(
            event("1", "kimA", "김철수", at(9, 0)),
            event("2", "kimA", "김철수", at(10, 30), kiosk = "k2"),
            event("3", "kimB", "김철수", at(9, 5)),
        )
        val rows = report.daily(day, roster, events).associateBy { it.personKey }
        assertEquals(2, rows.getValue("kimA").checkIns)
        assertEquals(at(9, 0), rows.getValue("kimA").firstAt)
        assertEquals(at(10, 30), rows.getValue("kimA").lastAt)
        assertEquals(listOf("k1", "k2"), rows.getValue("kimA").kiosks)
        assertEquals(1, rows.getValue("kimB").checkIns)
        assertFalse(rows.getValue("hong").present)
    }

    @Test
    fun `events are bucketed by local date`() {
        val lateNight = at(23, 59)
        val nextMorning = at(0, 1, day.plusDays(1))
        val rows = report.daily(day, roster, listOf(event("1", "hong", "홍길동", lateNight), event("2", "hong", "홍길동", nextMorning)))
        assertEquals(1, rows.first { it.personKey == "hong" }.checkIns)
    }

    @Test
    fun `label-only check-ins attach to a unique roster match, otherwise appear off roster`() {
        val events = listOf(
            event("1", "", "홍길동", at(9, 0), source = CheckInSource.LABEL_ONLY, dept = "청년부", grade = "3학년"),
            event("2", "", "김철수", at(9, 0), source = CheckInSource.LABEL_ONLY, dept = "", grade = ""),
            event("3", "", "방문자", at(9, 10), source = CheckInSource.LABEL_ONLY),
        )
        val rows = report.daily(day, roster, events)
        assertTrue(rows.first { it.personKey == "hong" }.present)
        val offRoster = rows.filter { !it.onRoster }
        assertEquals(setOf("김철수", "방문자"), offRoster.map { it.name }.toSet())
        assertEquals(rows.size, roster.size + 2)
        assertTrue(rows.indexOfFirst { !it.onRoster } > rows.indexOfLast { it.onRoster })
    }

    @Test
    fun `events of deleted people stay visible off roster`() {
        val rows = report.daily(day, listOf(hong), listOf(event("1", "kimA", "김철수", at(9, 0), dept = "청년부")))
        val row = rows.single { it.personKey == "kimA" }
        assertFalse(row.onRoster)
        assertTrue(row.present)
    }

    @Test
    fun `daily csv has a summary line and one row per person`() {
        val rows = report.daily(day, roster, listOf(event("1", "hong", "홍길동", at(9, 7))))
        val table = Csv.parse(report.dailyCsv(day, rows))
        assertEquals("출석부", table[0][0])
        assertTrue(table[0][2].contains("출석 1 명"))
        assertEquals("이름", table[1][0])
        val hongRow = table.first { it[0] == "홍길동" }
        assertEquals("출석", hongRow[3])
        assertEquals("09:07:00", hongRow[4])
        assertEquals(2 + roster.size, table.size)
        assertEquals("attendance_2026-10-25.csv", AttendanceReport.fileName(day))
    }

    @Test
    fun `range csv marks each attended day`() {
        val events = listOf(
            event("1", "hong", "홍길동", at(9, 0)),
            event("2", "hong", "홍길동", at(9, 0, day.plusDays(2))),
            event("3", "kimA", "김철수", at(9, 0, day.plusDays(1))),
        )
        val table = Csv.parse(report.rangeCsv(day, day.plusDays(2), roster, events))
        assertEquals(listOf("이름", "부서", "학년", "출석 일수", "10-25", "10-26", "10-27"), table[1])
        val hongRow = table.first { it[0] == "홍길동" }
        assertEquals(listOf("2", "O", "", "O"), hongRow.subList(3, 7))
        val kimRow = table.first { it[0] == "김철수" && it[1] == "청년부" }
        assertEquals("1", kimRow[3])
    }
}
