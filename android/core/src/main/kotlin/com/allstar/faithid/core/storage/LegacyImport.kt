package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.codec.CsvHeader
import com.allstar.faithid.core.codec.FloatPacking
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.codec.cell
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonKeys
import com.allstar.faithid.core.model.PersonMetadata
import java.io.File

/**
 * Reads databases written before schema version 3:
 *
 * - `face_meta.db` with a people table pointing into `face_embeddings.bin` (2026-03 … 2026-05), or
 * - only `face_db.csv` with an `embeddings_json` column (first prototype exports).
 */
internal object LegacyImport {

    fun readIfLegacy(directory: File, factory: SqlDatabaseFactory, log: (String) -> Unit): List<Person>? {
        val dbFile = File(directory, PeopleStore.DB_FILE)
        if (dbFile.exists()) {
            val db = factory.open(dbFile)
            try {
                if ("people" !in db.tables()) return null
                if ("embeddings" in db.columns("people")) return null
                log("legacy database found (people table + ${PeopleStore.LEGACY_BIN_FILE})")
                return readV2(db, File(directory, PeopleStore.LEGACY_BIN_FILE), log)
            } finally {
                db.close()
            }
        }
        val csvFile = File(directory, PeopleStore.CSV_FILE)
        if (csvFile.exists()) {
            val people = readCsvWithEmbeddings(csvFile.readText())
            if (people != null) {
                log("legacy CSV with embeddings found: ${people.size} people")
                return people
            }
        }
        return null
    }

    fun backupLegacyFiles(directory: File, log: (String) -> Unit) {
        listOf(PeopleStore.DB_FILE, PeopleStore.LEGACY_BIN_FILE, PeopleStore.DB_FILE + "-journal").forEach { name ->
            val file = File(directory, name)
            if (!file.exists()) return@forEach
            var backup = File(directory, name + PeopleStore.BACKUP_SUFFIX)
            var n = 1
            while (backup.exists()) backup = File(directory, name + PeopleStore.BACKUP_SUFFIX + "." + n++)
            if (file.renameTo(backup)) log("kept ${file.name} as ${backup.name}") else log("could not back up ${file.name}")
        }
    }

    private fun readV2(db: SqlDatabase, binFile: File, log: (String) -> Unit): List<Person> {
        val columns = db.columns("people")
        val bin = if (binFile.exists()) binFile.readBytes() else ByteArray(0)
        val select = buildList {
            add("name")
            add("department")
            add(if ("grade" in columns) "grade" else "'' AS grade")
            add(if ("person_key" in columns) "person_key" else "'' AS person_key")
            add(if ("metadata_blob" in columns) "metadata_blob" else "NULL AS metadata_blob")
            add("offset_bytes")
            add("length_bytes")
            add("embedding_count")
            add("embedding_dim")
            add(if ("updated_at" in columns) "updated_at" else "0 AS updated_at")
        }.joinToString(", ")
        val seenKeys = HashSet<String>()
        return db.query("SELECT $select FROM people") { row ->
            val name = row.string("name").orEmpty().trim()
            if (name.isEmpty()) return@query null
            val department = row.string("department").orEmpty().trim()
            val grade = row.string("grade").orEmpty().trim()
            val metadata = PersonCodec.decodeMetadata(row.bytes("metadata_blob")) ?: PersonMetadata.of(name, department, grade)
            val key = row.string("person_key")?.takeIf { it.isNotBlank() } ?: PersonKeys.legacyKey(name, department, grade)
            val embeddings = runCatching {
                FloatPacking.unpack(
                    bin,
                    count = (row.long("embedding_count") ?: 0L).toInt(),
                    dim = (row.long("embedding_dim") ?: 0L).toInt(),
                    offset = (row.long("offset_bytes") ?: 0L).toInt(),
                )
            }.getOrElse {
                log("skipped $name: ${it.message}")
                emptyList()
            }
            if (embeddings.isEmpty() || !seenKeys.add(key)) null else Person(key, metadata, embeddings, 0L, row.long("updated_at") ?: 0L)
        }.filterNotNull()
    }

    /** People from a CSV export that still carries an `embeddings_json` column, or null if it has none. */
    fun readCsvWithEmbeddings(text: String): List<Person>? {
        val rows = Csv.parse(text)
        if (rows.isEmpty()) return null
        val header = CsvHeader(rows.first())
        val embeddingsIndex = header.indexOf("embeddings_json", "embeddings", "embedding_json")
        if (embeddingsIndex < 0) return null
        val keyIndex = header.indexOf("person_key")
        val metadataColumns = header.columns.withIndex().filter { (index, name) ->
            index != embeddingsIndex && index != keyIndex && name.lowercase() !in NON_METADATA_COLUMNS
        }
        val seenKeys = HashSet<String>()
        return rows.drop(1).mapNotNull { row ->
            val metadata = PersonMetadata.of(metadataColumns.associate { (index, name) -> name to row.cell(index) })
            if (metadata.name.isBlank()) return@mapNotNull null
            val embeddings = FloatJson.parseMatrix(row.cell(embeddingsIndex))
            if (embeddings.isEmpty()) return@mapNotNull null
            val key = row.cell(keyIndex).ifBlank { PersonKeys.legacyKey(metadata.name, metadata.department, metadata.grade) }
            if (!seenKeys.add(key)) return@mapNotNull null
            Person(key, metadata, embeddings)
        }
    }

    val NON_METADATA_COLUMNS = setOf(
        "person_key", "person_idx", "person_id", "embeddings_json", "embeddings", "embedding_json",
        "embedding_count", "updated_at", "updatedat", "hash", "record_hash", "metadata_hash",
    )
}

/** Parses `[[0.1, -0.2, ...], [...]]` without a JSON library. Returns an empty list on malformed input. */
internal object FloatJson {
    fun parseMatrix(raw: String): List<FloatArray> {
        val text = raw.trim()
        if (!text.startsWith("[")) return emptyList()
        val out = mutableListOf<FloatArray>()
        var depth = 0
        var current: MutableList<Float>? = null
        val number = StringBuilder()
        fun flushNumber(): Boolean {
            if (number.isEmpty()) return true
            val value = number.toString().toFloatOrNull() ?: return false
            current?.add(value) ?: return false
            number.setLength(0)
            return true
        }
        for (c in text) {
            when {
                c == '[' -> {
                    depth++
                    if (depth == 2) current = mutableListOf()
                    if (depth > 2) return emptyList()
                }
                c == ']' -> {
                    if (!flushNumber()) return emptyList()
                    if (depth == 2) {
                        current?.let { list -> if (list.isNotEmpty()) out += list.toFloatArray() }
                        current = null
                    }
                    depth--
                }
                c == ',' -> if (!flushNumber()) return emptyList()
                c.isWhitespace() -> Unit
                else -> number.append(c)
            }
        }
        val dim = out.firstOrNull()?.size ?: return emptyList()
        return out.filter { it.size == dim }
    }
}
