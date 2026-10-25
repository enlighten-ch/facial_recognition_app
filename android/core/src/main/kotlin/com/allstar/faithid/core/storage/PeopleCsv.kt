package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.codec.CsvHeader
import com.allstar.faithid.core.codec.cell
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata

/**
 * `face_db.csv`: the people list as a spreadsheet. Administrators may edit names, departments,
 * grades and add their own columns in Excel; [parseEdits] and [applyEdits] bring those edits back.
 * Face samples are never part of the CSV.
 */
object PeopleCsv {

    fun export(people: List<Person>, displayNumbers: Map<String, Long>): String {
        val metadataColumns = LinkedHashSet<String>()
        people.forEach { person -> person.metadata.entries.keys.forEach { key -> if (metadataColumns.none { it.equals(key, true) }) metadataColumns += key } }
        val header = listOf("person_idx") + metadataColumns + listOf("embedding_count", "updated_at", "person_key")
        val rows = people
            .sortedWith(compareBy<Person> { displayNumbers[it.key] ?: Long.MAX_VALUE }.thenBy { it.name })
            .map { person ->
                listOf((displayNumbers[person.key] ?: 0L).toString()) +
                    metadataColumns.map { person.metadata[it] } +
                    listOf(person.embeddings.size.toString(), person.updatedAt.toString(), person.key)
            }
        return Csv.write(listOf(header) + rows)
    }

    data class Edit(val personKey: String, val displayNumber: Long?, val metadata: PersonMetadata)

    fun parseEdits(text: String): List<Edit> {
        val rows = Csv.parse(text)
        if (rows.isEmpty()) return emptyList()
        val header = CsvHeader(rows.first())
        val keyIndex = header.indexOf("person_key")
        val idxIndex = header.indexOf("person_idx", "person_id")
        val metadataColumns = header.columns.withIndex().filter { (_, name) -> name.lowercase() !in LegacyImport.NON_METADATA_COLUMNS }
        return rows.drop(1).mapNotNull { row ->
            val metadata = PersonMetadata.of(metadataColumns.associate { (index, name) -> name to row.cell(index) })
            if (metadata.name.isBlank()) null else Edit(row.cell(keyIndex), row.cell(idxIndex).toLongOrNull(), metadata)
        }
    }

    data class ApplyResult(val changed: List<Person>, val unmatched: List<Edit>)

    /**
     * Matches each edit to a person by key, then by display number, then by a unique
     * name/department/grade, and returns the people whose metadata actually changed.
     */
    fun applyEdits(people: List<Person>, displayNumbers: Map<String, Long>, edits: List<Edit>, now: Long): ApplyResult {
        val byKey = people.associateBy { it.key }
        val byNumber = people.associateBy { displayNumbers[it.key] }
        val byIdentity = people.groupBy { it.metadata.identityKey() }.filterValues { it.size == 1 }.mapValues { it.value.single() }
        val changed = LinkedHashMap<String, Person>()
        val unmatched = mutableListOf<Edit>()
        edits.forEach { edit ->
            val target = byKey[edit.personKey]
                ?: edit.displayNumber?.let { byNumber[it] }
                ?: byIdentity[edit.metadata.identityKey()]
            if (target == null || target.key in changed) {
                unmatched += edit
            } else if (target.metadata != edit.metadata) {
                changed[target.key] = target.copy(metadata = edit.metadata, updatedAt = now)
            }
        }
        return ApplyResult(changed.values.toList(), unmatched)
    }
}
