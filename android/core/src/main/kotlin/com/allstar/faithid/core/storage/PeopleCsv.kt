package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.PersonNames

/**
 * `face_db.csv`: the people list as a spreadsheet. Administrators may edit it in Excel; [CsvMerge]
 * brings those edits back. Face samples are never part of the CSV.
 */
object PeopleCsv {

    fun export(people: List<Person>, displayNumbers: Map<String, Long>, labels: Map<String, String> = emptyMap()): String =
        exportTable(people, displayNumbers, labels).text

    /** The written CSV and, in header order, the column key behind each metadata column. */
    data class Export(val text: String, val columns: List<String>)

    /**
     * The CSV with one column per metadata column (aliases such as "부서" folded into "department").
     * The name is written as two columns, family_name and given_name, so screens and labels can use
     * either part; the full name is both together. A column with a name from the administrator
     * ([labels], "department" → "반") uses it as header.
     */
    fun exportTable(people: List<Person>, displayNumbers: Map<String, Long>, labels: Map<String, String> = emptyMap()): Export {
        val metadataColumns = LinkedHashSet<String>()
        people.forEach { person ->
            person.metadata.entries.keys.forEach { raw ->
                when (val key = PersonMetadata.canonicalColumn(raw)) {
                    PersonMetadata.CANONICAL_FAMILY_NAME -> Unit
                    PersonMetadata.CANONICAL_NAME -> { metadataColumns += PersonMetadata.CANONICAL_FAMILY_NAME; metadataColumns += PersonMetadata.CSV_GIVEN_NAME }
                    else -> if (metadataColumns.none { it.equals(key, true) }) metadataColumns += key
                }
            }
        }
        val header = listOf("person_idx") + metadataColumns.map { labels[it]?.takeIf { l -> l.isNotBlank() } ?: it } + listOf("embedding_count", "updated_at", "person_key")
        val rows = people
            .sortedWith(compareBy<Person> { displayNumbers[it.key] ?: Long.MAX_VALUE }.thenBy { it.name })
            .map { person ->
                listOf((displayNumbers[person.key] ?: 0L).toString()) +
                    metadataColumns.map { cellOf(person.metadata, it) } +
                    listOf(person.embeddings.size.toString(), person.updatedAt.toString(), person.key)
            }
        return Export(Csv.write(listOf(header) + rows), metadataColumns.toList())
    }

    /** A CSV cell: the name's parts as [PersonNames] splits it, other columns as stored. */
    fun cellOf(metadata: PersonMetadata, column: String): String = when (column) {
        PersonMetadata.CANONICAL_FAMILY_NAME -> PersonNames.family(metadata)
        PersonMetadata.CSV_GIVEN_NAME -> PersonNames.given(metadata)
        else -> metadata.valueOf(column)
    }
}
