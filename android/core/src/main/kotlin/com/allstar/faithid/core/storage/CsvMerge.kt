package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.codec.CsvHeader
import com.allstar.faithid.core.codec.cell
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.PersonNames
import java.util.Locale

/**
 * What an administrator changed in `face_db.csv`, found by comparing the file with the copy the
 * server wrote last ([base]). Only differences from that copy count, so changes made in the app
 * since then (registrations, edits) are kept unless the same cell was edited in the file too.
 *
 * - a changed cell changes that value; a cell of a new column adds the column
 * - a header renamed in place (same position) renames the column: only its name, the data stays
 *   under the column the kiosks fill in
 * - the name comes as family_name and given_name; a change in either sets the name (see the
 *   repository); taking either column out is ignored, so a slip cannot rename everyone
 * - a column taken out of the file is removed (department and grade are emptied)
 * - a row taken out of the file deletes that person, unless the file has no rows at all or more
 *   than [MASS_DELETE_COUNT] people and [MASS_DELETE_SHARE] of the rows would go at once
 *
 * Rows are matched by person_key, then by person_idx.
 */
object CsvMerge {
    const val MASS_DELETE_COUNT = 10
    const val MASS_DELETE_SHARE = 0.3

    /** Columns that make up the name; they are never removed. */
    private val NAME_COLUMNS = setOf(PersonMetadata.CANONICAL_NAME, PersonMetadata.CANONICAL_FAMILY_NAME, PersonMetadata.CSV_GIVEN_NAME)

    /** The CSV the server wrote last, with the column key behind each metadata column in header order. */
    data class Base(val text: String, val columns: List<String>)

    data class Changes(
        /** Values to set per person key, by column key. */
        val values: Map<String, Map<String, String>> = emptyMap(),
        /** New names of columns, by column key. */
        val renamed: Map<String, String> = emptyMap(),
        val removedColumns: Set<String> = emptySet(),
        val deleted: Set<String> = emptySet(),
        /** Deletions found but not applied (too many at once, or a file without rows). */
        val heldDeletions: Int = 0,
        /** Rows that match no one (new rows cannot become people: they have no face). */
        val unmatched: List<String> = emptyList(),
    ) {
        val isEmpty: Boolean get() = values.isEmpty() && renamed.isEmpty() && removedColumns.isEmpty() && deleted.isEmpty()

        /** How much differs; the copy an edited file came from gives the smallest. */
        val size: Int get() = values.values.sumOf { it.size } + renamed.size + removedColumns.size + deleted.size + heldDeletions + unmatched.size
    }

    /** True when two written CSVs have the same people, columns and values (face sample counts aside). */
    fun sameContent(a: Base, b: Base): Boolean = a.columns == b.columns && content(a) == content(b)

    private fun content(base: Base): List<List<String>> {
        val t = table(base.text)
        return t.rows.map { row -> listOf(row.key) + t.columns.map { row.cells.cell(it.index) } }
    }

    private class Table(val columns: List<Column>, val rows: List<Row>)
    private class Column(val index: Int, val header: String)
    private class Row(val cells: List<String>, val key: String, val number: Long?)

    /**
     * [base] is the CSV the server wrote last. [older] are earlier copies, newest first: a
     * spreadsheet may save a copy it opened before the server wrote a newer one. From them the
     * copy the file came from is found (the oldest one that knows every row of the file); people
     * not in that copy are never taken as deleted, and a cell still holding that copy's value is
     * not an edit.
     */
    fun diff(
        base: Base,
        edited: String,
        people: List<Person>,
        displayNumbers: Map<String, Long>,
        labels: Map<String, String>,
        older: List<Base> = emptyList(),
    ): Changes {
        val old = table(base.text)
        val new = table(edited)
        val fileKeys = new.rows.map { it.key }.filter { it.isNotEmpty() }.toSet()
        val origin = (listOf(base) + older).map { table(it.text) }
            .lastOrNull { copy -> copy.rows.map { it.key }.toSet().containsAll(fileKeys) } ?: old
        val originRows = origin.rows.associateBy { it.key }
        val originColumns = origin.columns.associate { it.header.trim() to it.index }
        val byKey = people.associateBy { it.key }
        val keyByNumber = displayNumbers.entries.associate { (key, number) -> number to key }

        // Column keys of the edited file: names given in the app, then the usual aliases.
        val byLabel = labels.entries.associate { (key, label) -> label.trim().lowercase(Locale.US) to key }
        fun resolve(header: String) = byLabel[header.trim().lowercase(Locale.US)] ?: PersonMetadata.canonicalColumn(header)
        val baseKeys = base.columns
        val newKeys = new.columns.map { resolve(it.header) }.toMutableList()
        val renamed = LinkedHashMap<String, String>()
        // A named column written under its own name again ("반" back to "department"): the name goes.
        new.columns.forEachIndexed { i, column ->
            val key = newKeys[i]
            val label = labels[key] ?: return@forEachIndexed
            if (!column.header.trim().equals(label.trim(), ignoreCase = true)) renamed[key] = column.header.trim()
        }
        val baseRowsByKey = old.rows.associateBy { it.key }
        // A header that is not a known column, while a column of the base is gone and the values
        // mostly stayed: a rename (wherever the column moved). Mostly different values: a column
        // replaced by a new one.
        newKeys.indices.forEach { i ->
            val key = newKeys[i]
            if (key in baseKeys) return@forEach
            val match = baseKeys.indices
                .filter { b -> baseKeys[b] !in newKeys && baseKeys[b] !in NAME_COLUMNS }
                .map { b -> b to sameShare(old, b, new, i, baseRowsByKey) }
                .filter { (_, share) -> share >= 0.5 }
                .maxByOrNull { (_, share) -> share }
                ?: return@forEach
            val baseKey = baseKeys[match.first]
            newKeys[i] = baseKey
            renamed[baseKey] = new.columns[i].header.trim()
        }
        val removed = baseKeys.filter { it !in newKeys && it !in NAME_COLUMNS }.toSet()

        val baseRows = baseRowsByKey
        val values = LinkedHashMap<String, Map<String, String>>()
        val seen = HashSet<String>()
        val unmatched = mutableListOf<String>()
        new.rows.forEach { row ->
            val key = row.key.takeIf { it in byKey } ?: row.number?.let { keyByNumber[it] }
            if (key == null || key in seen) {
                val name = listOf(PersonMetadata.CANONICAL_FAMILY_NAME, PersonMetadata.CSV_GIVEN_NAME, PersonMetadata.CANONICAL_NAME)
                    .mapNotNull { key -> newKeys.indexOf(key).takeIf { it >= 0 }?.let { row.cells.cell(new.columns[it].index) } }
                    .let { parts -> if (parts.size >= 2) PersonNames.join(parts[0], parts[1]) else parts.firstOrNull() }
                if (!name.isNullOrBlank()) unmatched += name
                return@forEach
            }
            seen += key
            val person = byKey.getValue(key)
            val before = baseRows[key]
            val changed = LinkedHashMap<String, String>()
            new.columns.forEachIndexed { i, column ->
                val columnKey = newKeys[i]
                val value = row.cells.cell(column.index)
                val baseIndex = baseKeys.indexOf(columnKey)
                // Without a base row (or a new column) the person's current value is the reference.
                val current = PeopleCsv.cellOf(person.metadata, columnKey)
                val reference = if (before != null && baseIndex >= 0) before.cells.cell(old.columns[baseIndex].index) else current
                if (value == reference) return@forEachIndexed
                // Still the value of the copy the file came from: changed since in the app, not in the file.
                val stale = originRows[key]?.let { r -> originColumns[column.header.trim()]?.let { r.cells.cell(it) } }
                if (origin !== old && stale != null && value == stale) return@forEachIndexed
                if ((columnKey == PersonMetadata.CANONICAL_NAME || columnKey == PersonMetadata.CSV_GIVEN_NAME) && value.isBlank()) return@forEachIndexed
                if (baseIndex < 0 && value.isBlank() && current.isBlank()) return@forEachIndexed
                changed[columnKey] = value
            }
            if (changed.isNotEmpty()) values[key] = changed
        }

        val gone = old.rows.map { it.key }.filter { it in byKey && it !in seen && it in originRows }.toSet()
        val tooMany = gone.size > MASS_DELETE_COUNT && gone.size > old.rows.size * MASS_DELETE_SHARE
        val hold = new.rows.isEmpty() || tooMany
        return Changes(
            values = values,
            renamed = renamed,
            removedColumns = removed,
            deleted = if (hold) emptySet() else gone,
            heldDeletions = if (hold) gone.size else 0,
            unmatched = unmatched,
        )
    }

    /** Share of rows (of those in both files) where base column [b] and edited column [i] hold the same value. */
    private fun sameShare(old: Table, b: Int, new: Table, i: Int, baseRows: Map<String, Row>): Double {
        val pairs = new.rows.mapNotNull { row -> baseRows[row.key]?.let { it.cells.cell(old.columns[b].index) to row.cells.cell(new.columns[i].index) } }
            .filter { (a, n) -> a.isNotEmpty() || n.isNotEmpty() }
        return if (pairs.isEmpty()) 0.0 else pairs.count { (a, n) -> a == n }.toDouble() / pairs.size
    }

    private fun table(text: String): Table {
        val rows = Csv.parse(text)
        if (rows.isEmpty()) return Table(emptyList(), emptyList())
        val header = CsvHeader(rows.first())
        val keyIndex = header.indexOf("person_key")
        val numberIndex = header.indexOf("person_idx", "person_id")
        val columns = header.columns.withIndex()
            .filter { (_, name) -> name.isNotBlank() && name.lowercase(Locale.US) !in LegacyImport.NON_METADATA_COLUMNS }
            .map { (index, name) -> Column(index, name) }
        val data = rows.drop(1)
            .filter { row -> row.any { it.isNotBlank() } }
            .map { Row(it, it.cell(keyIndex), it.cell(numberIndex).toLongOrNull()) }
        return Table(columns, data)
    }
}
