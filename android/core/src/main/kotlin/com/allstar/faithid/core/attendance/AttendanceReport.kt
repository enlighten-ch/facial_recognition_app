package com.allstar.faithid.core.attendance

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** One line of the daily attendance sheet. */
data class AttendanceRow(
    val personKey: String,
    val name: String,
    val department: String,
    val grade: String,
    val onRoster: Boolean,
    val checkIns: Int,
    val firstAt: Long?,
    val lastAt: Long?,
    val kiosks: List<String>,
    val sources: List<CheckInSource>,
) {
    val present: Boolean get() = checkIns > 0
}

/**
 * Builds attendance sheets from check-in events.
 *
 * Rows are keyed by person key, so two people with the same name stay separate. Label-only
 * check-ins carry no key; they are attributed to a registered person only when exactly one person
 * on the roster has the same name, department and grade, otherwise they appear as "명단 외".
 */
class AttendanceReport(private val zone: ZoneId) {

    fun dateOf(event: AttendanceEvent): LocalDate = Instant.ofEpochMilli(event.occurredAt).atZone(zone).toLocalDate()

    fun daily(date: LocalDate, roster: List<Person>, events: List<AttendanceEvent>): List<AttendanceRow> {
        val dayEvents = events.filter { dateOf(it) == date }
        val rosterByKey = roster.associateBy { it.key }
        val rosterByIdentity = roster.groupBy { it.metadata.identityKey() }

        val grouped = LinkedHashMap<String, MutableList<AttendanceEvent>>()
        dayEvents.forEach { event ->
            val groupKey = when {
                event.personKey.isNotBlank() -> event.personKey
                else -> {
                    val identity = PersonMetadata.of(event.name, event.department, event.grade).identityKey()
                    rosterByIdentity[identity]?.singleOrNull()?.key ?: "label:$identity"
                }
            }
            grouped.getOrPut(groupKey) { mutableListOf() } += event
        }

        val rows = mutableListOf<AttendanceRow>()
        roster.forEach { person ->
            rows += buildRow(person.key, person.name, person.department, person.grade, onRoster = true, grouped[person.key].orEmpty())
        }
        grouped.filterKeys { it !in rosterByKey }.forEach { (key, list) ->
            val latest = list.maxBy { it.occurredAt }
            rows += buildRow(
                personKey = if (key.startsWith("label:")) "" else key,
                name = latest.name,
                department = latest.department,
                grade = latest.grade,
                onRoster = false,
                events = list,
            )
        }
        return rows.sortedWith(ROW_ORDER)
    }

    fun dailyCsv(date: LocalDate, rows: List<AttendanceRow>): String {
        val present = rows.count { it.present }
        val table = mutableListOf(
            listOf("출석부", date.toString(), "출석 $present 명 / 명단 ${rows.count { it.onRoster }} 명"),
            listOf("이름", "부서", "학년", "출석여부", "첫 출석", "마지막 출석", "횟수", "키오스크", "확인 방법", "비고"),
        )
        rows.forEach { row ->
            table += listOf(
                row.name,
                row.department,
                row.grade,
                if (row.present) "출석" else "미출석",
                row.firstAt?.let { formatTime(it) }.orEmpty(),
                row.lastAt?.let { formatTime(it) }.orEmpty(),
                row.checkIns.toString(),
                row.kiosks.joinToString(" "),
                row.sources.joinToString(" ") { it.label },
                if (row.onRoster) "" else "명단 외",
            )
        }
        return Csv.write(table)
    }

    /** Person × date matrix for a period, for monthly or quarterly records. */
    fun rangeCsv(from: LocalDate, to: LocalDate, roster: List<Person>, events: List<AttendanceEvent>): String {
        require(!to.isBefore(from)) { "invalid range $from..$to" }
        val dates = generateSequence(from) { it.plusDays(1) }.takeWhile { !it.isAfter(to) }.toList()
        val daysByRowKey = LinkedHashMap<String, Pair<AttendanceRow, MutableSet<LocalDate>>>()
        dates.forEach { date ->
            daily(date, roster, events).forEach { row ->
                val rowKey = row.personKey.ifBlank { "label:" + PersonMetadata.of(row.name, row.department, row.grade).identityKey() }
                val entry = daysByRowKey.getOrPut(rowKey) { row to mutableSetOf() }
                if (row.present) entry.second += date
            }
        }
        val header = listOf("이름", "부서", "학년", "출석 일수") + dates.map { it.format(SHORT_DATE) }
        val table = mutableListOf(listOf("출석 기록", "$from ~ $to"), header)
        daysByRowKey.values
            .sortedWith(compareBy<Pair<AttendanceRow, MutableSet<LocalDate>>, AttendanceRow>(ROW_ORDER) { it.first })
            .forEach { (row, days) ->
                table += listOf(row.name, row.department, row.grade, days.size.toString()) +
                    dates.map { if (it in days) "O" else "" }
            }
        return Csv.write(table)
    }

    fun formatTime(epochMillis: Long): String = Instant.ofEpochMilli(epochMillis).atZone(zone).format(TIME)

    private fun buildRow(
        personKey: String,
        name: String,
        department: String,
        grade: String,
        onRoster: Boolean,
        events: List<AttendanceEvent>,
    ): AttendanceRow {
        val sorted = events.sortedBy { it.occurredAt }
        return AttendanceRow(
            personKey = personKey,
            name = name,
            department = department,
            grade = grade,
            onRoster = onRoster,
            checkIns = sorted.size,
            firstAt = sorted.firstOrNull()?.occurredAt,
            lastAt = sorted.lastOrNull()?.occurredAt,
            kiosks = sorted.map { it.kioskId }.distinct(),
            sources = sorted.map { it.source }.distinct(),
        )
    }

    companion object {
        private val TIME = DateTimeFormatter.ofPattern("HH:mm:ss")
        private val SHORT_DATE = DateTimeFormatter.ofPattern("MM-dd")

        /** Roster first, then by department, grade and name. */
        val ROW_ORDER: Comparator<AttendanceRow> =
            compareBy<AttendanceRow> { !it.onRoster }
                .thenBy { it.department }
                .thenBy { it.grade }
                .thenBy { it.name }
                .thenBy { it.personKey }

        fun fileName(date: LocalDate): String = "attendance_$date.csv"
    }
}
