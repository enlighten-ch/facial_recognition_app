package com.allstar.faithid.serverapp.ui

import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.core.model.Person

/** Which column a table is sorted by; tapping the same header again reverses it. */
internal data class SortState(val key: String, val ascending: Boolean = true) {
    fun tap(column: String): SortState = if (column == key) copy(ascending = !ascending) else SortState(column)

    fun arrow(column: String): String = if (column != key) "" else if (ascending) " ▲" else " ▼"
}

/** Department order and colours, shared by every list so one department always looks the same. */
internal object Departments {
    /** The six colours of the design, used for known departments and, by hash, for any other name. */
    val COLORS = intArrayOf(0xFFF2A7B5.toInt(), 0xFFF6C453.toInt(), 0xFF7CC7A4.toInt(), 0xFF6FA8E8.toInt(), 0xFFA98BE0.toInt(), 0xFF9AA3B5.toInt())
    const val NONE_COLOR = 0xFFC9BCD2.toInt()
    private val KNOWN = listOf("영아부", "유치부", "유년부", "초등부", "중고등부", "청년부")

    fun color(name: String): Int {
        val n = name.trim()
        if (n.isEmpty()) return NONE_COLOR
        val known = KNOWN.indexOf(n)
        return COLORS[if (known >= 0) known else Math.floorMod(n.hashCode(), COLORS.size)]
    }

    /** Known departments first in the usual age order, then the others by name. */
    fun order(name: String): Int = KNOWN.indexOf(name.trim()).let { if (it >= 0) it else KNOWN.size }

    val comparator: Comparator<String> = compareBy<String>({ order(it) }, { it.isBlank() }, { it })

    fun sorted(names: Collection<String>): List<String> = names.filter { it.isNotBlank() }.distinct().sortedWith(comparator)
}

internal data class PersonRow(val person: Person, val presentToday: Boolean)

/** The 명단 table: search, department filter and the sortable columns. */
internal object PeopleTable {
    const val NAME = "name"
    const val DEPARTMENT = "department"
    const val GRADE = "grade"
    const val SAMPLES = "samples"
    const val TODAY = "today"

    fun rows(people: List<Person>, present: Set<String>, query: String, department: String?, sort: SortState): List<PersonRow> {
        val q = query.trim()
        val filtered = people.filter { p ->
            (department == null || p.department == department) &&
                (q.isEmpty() || listOf(p.name, p.department, p.grade).any { it.contains(q, ignoreCase = true) })
        }.map { PersonRow(it, it.key in present) }
        val byName = compareBy<PersonRow> { it.person.name }
        val byDepartment = compareBy<PersonRow, String>(Departments.comparator) { it.person.department }.thenBy { it.person.grade }
        val primary: Comparator<PersonRow> = when (sort.key) {
            DEPARTMENT, GRADE -> byDepartment
            SAMPLES -> compareBy { it.person.embeddings.size }
            TODAY -> compareBy { !it.presentToday }
            else -> byName
        }
        return filtered.sortedWith((if (sort.ascending) primary else primary.reversed()).then(byName).thenBy { it.person.key })
    }
}

internal enum class AttendanceShow { ALL, PRESENT, ABSENT }

internal data class DepartmentSummary(val department: String, val present: Int, val roster: Int)

/** The 출석 table: department summary, filters and the sortable columns. */
internal object AttendanceTable {
    const val STATUS = "status"
    const val NAME = "name"
    const val DEPARTMENT = "department"
    const val TIME = "time"
    const val KIOSK = "kiosk"

    fun summary(rows: List<AttendanceRow>): List<DepartmentSummary> =
        rows.filter { it.onRoster }.groupBy { it.department }
            .map { (dept, list) -> DepartmentSummary(dept, list.count { it.present }, list.size) }
            .sortedWith(compareBy(Departments.comparator) { it.department })

    fun rows(rows: List<AttendanceRow>, query: String, department: String?, show: AttendanceShow, sort: SortState): List<AttendanceRow> {
        val q = query.trim()
        val filtered = rows.filter { r ->
            (department == null || (r.onRoster && r.department == department)) &&
                (q.isEmpty() || r.name.contains(q, ignoreCase = true)) &&
                when (show) {
                    AttendanceShow.ALL -> true
                    AttendanceShow.PRESENT -> r.present
                    AttendanceShow.ABSENT -> !r.present
                }
        }
        val byName = compareBy<AttendanceRow> { it.name }
        val byDepartment = compareBy<AttendanceRow> { !it.onRoster }.then(compareBy(Departments.comparator) { it.department }).thenBy { it.grade }
        val byTime = compareBy<AttendanceRow> { it.firstAt ?: Long.MAX_VALUE }
        val primary: Comparator<AttendanceRow> = when (sort.key) {
            NAME -> byName
            DEPARTMENT -> byDepartment
            TIME -> byTime
            KIOSK -> compareBy<AttendanceRow> { it.kiosks.isEmpty() }.thenBy { it.kiosks.joinToString(" ") }
            // Present first by arrival, then the absent by department, grade and name.
            else -> compareBy<AttendanceRow> { !it.present }.then(byTime).then(byDepartment)
        }
        return filtered.sortedWith((if (sort.ascending) primary else primary.reversed()).then(byName).thenBy { it.personKey })
    }
}

/** Kinds of server log lines, for the filter chips of the developer screen. */
internal enum class LogKind(val label: String) {
    SERVER("서버"), KIOSK("키오스크 연결"), PEOPLE("명단 변경"), ATTENDANCE("출석"), ERROR("오류");

    companion object {
        private val ERROR_RE = Regex("fail|not written|rolled back|dropped|error|exception", RegexOption.IGNORE_CASE)
        private val SERVER_RE = Regex("^(server|people loaded|legacy)")
        private val KIOSK_RE = Regex("approval|client approved|client removed|full download")
        private val PEOPLE_RE = Regex("person|CSV|kiosk changes")
        private val ATTENDANCE_RE = Regex("check-ins|attendance")

        /** Classifies one line, with or without the "[yyyy-MM-dd HH:mm:ss] " prefix of the log file. */
        fun of(line: String): LogKind {
            val message = line.substringAfter("] ", line)
            return when {
                ERROR_RE.containsMatchIn(message) -> ERROR
                SERVER_RE.containsMatchIn(message) -> SERVER
                KIOSK_RE.containsMatchIn(message) -> KIOSK
                PEOPLE_RE.containsMatchIn(message) -> PEOPLE
                ATTENDANCE_RE.containsMatchIn(message) -> ATTENDANCE
                else -> SERVER
            }
        }
    }
}

/** config.yaml as shown on the developer screen: the password hash and salt never appear. */
internal object ConfigView {
    private val SECRET_KEYS = setOf("admin_password_hash", "admin_password_salt", "admin_password")

    fun masked(text: String): String = text.lines().joinToString("\n") { line ->
        val key = line.substringBefore(':').trim()
        if (key in SECRET_KEYS) "$key: \"••••••••\"" else line
    }.trimEnd()
}
