package com.allstar.faithid.serverapp.ui

import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.PersonNames
import com.allstar.faithid.core.storage.PeopleCsv

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

/** Names as the server app shows them: "김 하은" when the family name is known, else the name as it is. */
internal object DisplayNames {
    fun of(metadata: PersonMetadata): String {
        val family = PersonNames.family(metadata)
        return if (family.isEmpty()) metadata.name else "$family ${PersonNames.given(metadata)}"
    }
}

/** One column of the 명단 besides the name: a metadata column of the CSV ("department", "반", …). */
internal data class RosterColumn(val key: String, val label: String, val hidden: Boolean, val hasValues: Boolean) {
    val shown: Boolean get() = !hidden && hasValues

    fun valueOf(person: Person): String = person.metadata.valueOf(key)
}

/**
 * The 명단's columns come from the people's own columns, so a column added to or renamed in
 * face_db.csv shows up once the CSV is applied. Headers and hidden columns come from the
 * developer screen; columns no one has a value in stay out of the table.
 */
internal object RosterColumns {
    const val NAME = PersonMetadata.CANONICAL_NAME
    val DEFAULT_LABELS = mapOf(
        PersonMetadata.CANONICAL_NAME to "이름",
        PersonMetadata.CANONICAL_DEPARTMENT to "부서",
        PersonMetadata.CANONICAL_GRADE to "학년 · 나이",
        PersonMetadata.CANONICAL_FAMILY_NAME to "성",
    )

    fun label(key: String, labels: Map<String, String>): String = labels[key]?.takeIf { it.isNotBlank() } ?: DEFAULT_LABELS[key] ?: key

    /** Every column but the name, in the order the people's columns first appear (CSV order). */
    fun all(people: List<Person>, labels: Map<String, String>, hidden: Set<String>): List<RosterColumn> {
        val keys = LinkedHashSet<String>()
        // An empty roster still offers department and grade; otherwise the CSV's order decides, so a
        // renamed column (부서 → 반) keeps its place and still gives the filter chips.
        if (people.isEmpty()) {
            keys += PersonMetadata.CANONICAL_DEPARTMENT
            keys += PersonMetadata.CANONICAL_GRADE
        }
        people.forEach { p ->
            p.metadata.entries.keys.forEach { raw ->
                val key = PersonMetadata.canonicalColumn(raw)
                if (key != NAME && keys.none { it.equals(key, ignoreCase = true) }) keys += key
            }
        }
        return keys.map { key ->
            RosterColumn(
                key = key,
                label = label(key, labels),
                hidden = key in hidden,
                // An empty roster still shows department and grade.
                hasValues = people.isEmpty() && key in DEFAULT_LABELS || people.any { it.metadata.valueOf(key).isNotBlank() },
            )
        }
    }

    fun shown(people: List<Person>, labels: Map<String, String>, hidden: Set<String>): List<RosterColumn> =
        all(people, labels, hidden).filter { it.shown }
}

/**
 * The 명단 table: search, the filter chips of the first column (부서, or what replaced it) and the
 * sortable columns.
 */
internal object PeopleTable {
    const val NAME = "name"
    const val SAMPLES = "samples"
    const val TODAY = "today"

    /** [groupColumn] is the column of the filter chips ([group] is the chosen value); by default the first column. */
    fun rows(people: List<Person>, present: Set<String>, query: String, columns: List<RosterColumn>, group: String?, sort: SortState, groupColumn: String? = null): List<PersonRow> {
        val q = query.trim()
        val groupKey = groupColumn ?: columns.firstOrNull()?.key
        fun groupOf(p: Person) = groupKey?.let { PeopleCsv.cellOf(p.metadata, it) }.orEmpty()
        val filtered = people.filter { p ->
            (group == null || groupOf(p) == group) &&
                (q.isEmpty() || (listOf(p.name) + columns.map { it.valueOf(p) }).any { it.contains(q, ignoreCase = true) })
        }.map { PersonRow(it, it.key in present) }
        val byName = compareBy<PersonRow> { it.person.name }
        val column = columns.firstOrNull { it.key == sort.key }
        val primary: Comparator<PersonRow> = when {
            sort.key == SAMPLES -> compareBy { it.person.embeddings.size }
            sort.key == TODAY -> compareBy { !it.presentToday }
            column != null && column.key == groupKey ->
                compareBy<PersonRow, String>(Departments.comparator) { column.valueOf(it.person) }
                    .thenBy { r -> columns.drop(1).joinToString("\u0001") { it.valueOf(r.person) } }
            column != null -> compareBy<PersonRow> { column.valueOf(it.person).isBlank() }.thenBy { column.valueOf(it.person) }
            else -> byName
        }
        return filtered.sortedWith((if (sort.ascending) primary else primary.reversed()).then(byName).thenBy { it.person.key })
    }
}

internal enum class AttendanceShow { ALL, PRESENT, ABSENT }

internal data class GroupSummary(val value: String, val present: Int, val roster: Int)

/**
 * The column a tab groups people by (부서 by default, or any other CSV column but the name).
 * Roster rows take the value from the person; people not on the roster only have a department and
 * a grade from their label.
 */
internal class Grouping(val column: String, val label: String, private val people: Map<String, Person>) {
    fun of(row: AttendanceRow): String =
        people[row.personKey]?.let { PeopleCsv.cellOf(it.metadata, column) } ?: when (column) {
            PersonMetadata.CANONICAL_DEPARTMENT -> row.department
            PersonMetadata.CANONICAL_GRADE -> row.grade
            else -> ""
        }

    fun of(person: Person): String = PeopleCsv.cellOf(person.metadata, column)

    /** The row's name, split into family and given name when the person is on the roster. */
    fun nameOf(row: AttendanceRow): String = people[row.personKey]?.let { DisplayNames.of(it.metadata) } ?: row.name

    companion object {
        /** Columns to group by: every column with values except the name and its parts (현황, 명단, 출석). */
        fun choices(columns: List<RosterColumn>): List<RosterColumn> =
            columns.filter { it.hasValues && it.key != PersonMetadata.CANONICAL_FAMILY_NAME }

        /** The saved column if it still exists, else the department, else the first choice. */
        fun pick(saved: String, choices: List<RosterColumn>): RosterColumn? =
            choices.firstOrNull { it.key == saved } ?: choices.firstOrNull { it.key == PersonMetadata.CANONICAL_DEPARTMENT } ?: choices.firstOrNull()
    }
}

/** The 출석 table: department summary, filters and the sortable columns. */
internal object AttendanceTable {
    const val STATUS = "status"
    const val NAME = "name"
    const val DEPARTMENT = "department"
    const val TIME = "time"
    const val KIOSK = "kiosk"

    fun summary(rows: List<AttendanceRow>, grouping: Grouping? = null): List<GroupSummary> =
        rows.filter { it.onRoster }.groupBy { grouping?.of(it) ?: it.department }
            .map { (value, list) -> GroupSummary(value, list.count { it.present }, list.size) }
            .sortedWith(compareBy(Departments.comparator) { it.value })

    fun rows(rows: List<AttendanceRow>, query: String, group: String?, show: AttendanceShow, sort: SortState, grouping: Grouping? = null): List<AttendanceRow> {
        val q = query.trim()
        val filtered = rows.filter { r ->
            (group == null || (r.onRoster && (grouping?.of(r) ?: r.department) == group)) &&
                (q.isEmpty() || r.name.contains(q, ignoreCase = true)) &&
                when (show) {
                    AttendanceShow.ALL -> true
                    AttendanceShow.PRESENT -> r.present
                    AttendanceShow.ABSENT -> !r.present
                }
        }
        val byName = compareBy<AttendanceRow> { it.name }
        val byDepartment = compareBy<AttendanceRow> { !it.onRoster }.then(compareBy(Departments.comparator) { grouping?.of(it) ?: it.department }).thenBy { it.grade }
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
