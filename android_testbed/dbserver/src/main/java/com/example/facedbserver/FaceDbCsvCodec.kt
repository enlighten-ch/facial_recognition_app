package com.example.facedbserver

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.util.LinkedHashMap
import java.util.Locale

data class CsvMetaRow(
    val personId: Long,
    val personKey: String,
    val name: String,
    val department: String,
    val grade: String,
    val updatedAt: Long,
    val metadata: LinkedHashMap<String, String>,
    val metadataHash: String,
)

data class AttendanceCsvRow(
    val name: String,
    val affiliation: String,
    val department: String,
)

class FaceDbCsvCodec {
    fun hasEmbeddingsColumn(rawCsv: String): Boolean {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) return false
        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        return indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json")) >= 0
    }

    fun decode(rawCsv: String): JSONObject? {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) {
            return JSONObject().put("people", JSONArray())
        }

        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "이름"))
        if (nameIndex < 0) return null
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val gradeIndex = indexOfAny(header, listOf("grade", "year", "학년"))
        val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))
        val hashIndex = indexOfAny(header, listOf("hash", "record_hash"))
        val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))
        val personKeyIndex = indexOfAny(header, listOf("person_key"))
        val personIdIndex = indexOfAny(header, listOf("person_idx", "person_id"))

        val people = JSONArray()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val grade = row.getOrNull(gradeIndex)?.trim().orEmpty()
            val embeddings = parseEmbeddings(row.getOrNull(embeddingsIndex).orEmpty())
            val metadata = normalizeMetadataMap(
                csvMetadataFromRow(rows.first(), row),
                fallbackName = name,
                fallbackDepartment = department,
                fallbackGrade = grade,
            )
            val item = JSONObject()
                .put("name", metadataName(metadata, name))
                .put("department", metadataDepartment(metadata, department))
                .put("grade", metadataGrade(metadata, grade))
                .put("personIdx", parsePersonId(row.getOrNull(personIdIndex), name, department, grade))
                .put("personKey", parsePersonKey(row.getOrNull(personKeyIndex), name, department, grade))
                .put("metadata", JSONObject(metadata as Map<*, *>))
                .put("embeddings", embeddings)
            val hash = row.getOrNull(hashIndex)?.trim().orEmpty()
            if (hash.isNotBlank()) item.put("hash", hash)
            val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: 0L
            if (updatedAt > 0L) item.put("updatedAt", updatedAt)
            people.put(item)
        }
        return JSONObject().put("people", people)
    }

    fun encode(db: JSONObject): String {
        val people = extractPeopleArray(db)
        val metadataHeaders = linkedSetOf<String>()
        for (i in 0 until people.length()) {
            val person = people.optJSONObject(i) ?: continue
            val metadataJson = person.optJSONObject("metadata")
            if (metadataJson != null) {
                val keys = metadataJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    if (key.isNotBlank()) metadataHeaders += key
                }
            }
        }
        if (metadataHeaders.none { it.equals("name", ignoreCase = true) }) metadataHeaders.add("name")
        if (metadataHeaders.none { it.equals("department", ignoreCase = true) }) metadataHeaders.add("department")
        if (metadataHeaders.none { it.equals("grade", ignoreCase = true) }) metadataHeaders.add("grade")
        val out = StringBuilder()
        out.append((mutableListOf("person_idx").apply { addAll(metadataHeaders); add("embedding_count"); add("updated_at"); add("person_key") }).joinToString(",") { escapeCell(it) }).append('\n')
        for (i in 0 until people.length()) {
            val person = people.optJSONObject(i) ?: continue
            val name = person.optString("name", "").trim()
            if (name.isBlank()) continue
            val department = person.optString("department", person.optString("folder", "")).trim()
            val grade = person.optString("grade", "").trim()
            val metadataJson = person.optJSONObject("metadata")
            val metadata = linkedMapOf<String, String>()
            if (metadataJson != null) {
                val keys = metadataJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    metadata[key] = metadataJson.optString(key, "")
                }
            }
            val normalizedMetadata = normalizeMetadataMap(metadata, name, department, grade)
            val embeddings = person.optJSONArray("embeddings") ?: JSONArray()
            val updatedAt = person.optLong("updatedAt", 0L)
            val row = mutableListOf<String>()
            row += parsePersonId(person.optString("personIdx", person.optString("personId", "")), name, department, grade).toString()
            metadataHeaders.forEach { key ->
                row += normalizedMetadata.entries.firstOrNull { it.key.equals(key, ignoreCase = true) }?.value.orEmpty()
            }
            row += embeddings.length().toString()
            row += if (updatedAt > 0L) updatedAt.toString() else System.currentTimeMillis().toString()
            row += parsePersonKey(person.optString("personKey", ""), name, department, grade)
            out.append(row.joinToString(",") { escapeCell(it) }).append('\n')
        }
        return out.toString()
    }

    fun encodePeople(people: List<ServerPersonRecord>): String {
        val out = StringBuilder()
        val metadataHeaders = linkedSetOf<String>()
        val sortedPeople = people.sortedWith(compareBy<ServerPersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
        sortedPeople
            .forEach { person ->
                person.metadata.keys.forEach { key ->
                    if (key.isNotBlank()) metadataHeaders += key
                }
            }
        if (metadataHeaders.none { it.equals("name", ignoreCase = true) }) metadataHeaders.add("name")
        if (metadataHeaders.none { it.equals("department", ignoreCase = true) }) metadataHeaders.add("department")
        if (metadataHeaders.none { it.equals("grade", ignoreCase = true) }) metadataHeaders.add("grade")
        val header = mutableListOf("person_idx")
        header.addAll(metadataHeaders)
        header += listOf("embedding_count", "updated_at", "person_key")
        out.append(header.joinToString(",") { escapeCell(it) }).append('\n')
        sortedPeople
            .forEachIndexed { index, person ->
                val row = mutableListOf<String>()
                row += (index + 1).toString()
                metadataHeaders.forEach { key ->
                    row += person.metadata.entries.firstOrNull { it.key.equals(key, ignoreCase = true) }?.value.orEmpty()
                }
                row += person.embeddings.size.toString()
                row += person.updatedAt.toString()
                row += person.personKey
                out.append(row.joinToString(",") { escapeCell(it) }).append('\n')
            }
        return out.toString()
    }

    fun parseMetadataRows(rawCsv: String): List<CsvMetaRow> {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) return emptyList()
        val rawHeader = rows.first().map { it.trim() }
        val header = rawHeader.map { it.lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "이름"))
        if (nameIndex < 0) return emptyList()
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val gradeIndex = indexOfAny(header, listOf("grade", "year", "학년"))
        val personKeyIndex = indexOfAny(header, listOf("person_key"))
        val personIdIndex = indexOfAny(header, listOf("person_idx", "person_id"))
        val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))

        val out = mutableListOf<CsvMetaRow>()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val grade = row.getOrNull(gradeIndex)?.trim().orEmpty()
            val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: System.currentTimeMillis()
            val metadata = normalizeMetadataMap(
                csvMetadataFromRow(rawHeader, row),
                fallbackName = name,
                fallbackDepartment = department,
                fallbackGrade = grade,
            )
            out.add(
                CsvMetaRow(
                    personId = parsePersonId(row.getOrNull(personIdIndex), name, department, grade),
                    personKey = parsePersonKey(row.getOrNull(personKeyIndex), name, department, grade),
                    name = metadataName(metadata, name),
                    department = metadataDepartment(metadata, department),
                    grade = metadataGrade(metadata, grade),
                    updatedAt = updatedAt,
                    metadata = metadata,
                    metadataHash = metadataHash(metadata),
                ),
            )
        }
        return out
    }

    fun parseAttendanceRows(rawCsv: String): List<AttendanceCsvRow> {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) return emptyList()
        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "person", "이름"))
        if (nameIndex < 0) return emptyList()
        val affiliationIndex = indexOfAny(header, listOf("affiliation", "organization", "group", "소속", "grade", "year", "학년"))
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val out = mutableListOf<AttendanceCsvRow>()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val affiliation = row.getOrNull(affiliationIndex)?.trim().orEmpty()
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            out.add(AttendanceCsvRow(name = name, affiliation = affiliation, department = department))
        }
        return out
    }

    fun parseAttendanceMarkedNames(rawCsv: String): Set<String> {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) return emptySet()
        val dataRows =
            if (rows.firstOrNull()?.firstOrNull()?.contains("출석부", ignoreCase = true) == true) {
                rows.drop(1)
            } else {
                rows
            }
        if (dataRows.isEmpty()) return emptySet()
        val header = dataRows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "person", "이름"))
        val statusIndex = indexOfAny(header, listOf("attendance", "status", "출석여부"))
        if (nameIndex < 0 || statusIndex < 0) return emptySet()
        val out = linkedSetOf<String>()
        for (i in 1 until dataRows.size) {
            val row = dataRows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            val status = row.getOrNull(statusIndex)?.trim().orEmpty()
            if (name.isNotBlank() && (status == "출석" || status.equals("present", ignoreCase = true))) {
                out += name
            }
        }
        return out
    }

    private fun parseEmbeddings(raw: String): JSONArray {
        if (raw.isBlank()) return JSONArray()
        val parsed = runCatching { JSONTokener(raw).nextValue() }.getOrNull()
        if (parsed !is JSONArray) return JSONArray()
        val normalized = JSONArray()
        for (i in 0 until parsed.length()) {
            val emb = parsed.optJSONArray(i) ?: continue
            val vec = JSONArray()
            for (j in 0 until emb.length()) {
                vec.put(emb.optDouble(j, 0.0))
            }
            normalized.put(vec)
        }
        return normalized
    }

    private fun extractPeopleArray(root: JSONObject): JSONArray {
        root.optJSONArray("people")?.let { return it }
        root.optJSONArray("persons")?.let { return it }
        root.optJSONArray("faces")?.let { return it }
        return JSONArray()
    }

    private fun escapeCell(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        val requiresQuote = escaped.any { ch -> ch == ',' || ch == '"' || ch == '\n' || ch == '\r' }
        return if (requiresQuote) "\"$escaped\"" else escaped
    }

    private fun parseCsv(raw: String): List<List<String>> {
        val rows = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val cell = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < raw.length) {
            val c = raw[i]
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < raw.length && raw[i + 1] == '"') {
                        cell.append('"')
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    cell.append(c)
                }
                i += 1
                continue
            }

            when (c) {
                '"' -> inQuotes = true
                ',' -> {
                    row.add(cell.toString())
                    cell.setLength(0)
                }
                '\n' -> {
                    row.add(cell.toString())
                    rows.add(row)
                    row = mutableListOf()
                    cell.setLength(0)
                }
                '\r' -> {
                    row.add(cell.toString())
                    rows.add(row)
                    row = mutableListOf()
                    cell.setLength(0)
                    if (i + 1 < raw.length && raw[i + 1] == '\n') {
                        i += 1
                    }
                }
                else -> cell.append(c)
            }
            i += 1
        }
        if (cell.isNotEmpty() || row.isNotEmpty()) {
            row.add(cell.toString())
            rows.add(row)
        }
        return rows.filterNot { fields ->
            fields.size == 1 && fields[0].isBlank()
        }
    }

    private fun indexOfAny(header: List<String>, candidates: List<String>): Int {
        candidates.forEach { candidate ->
            val idx = header.indexOf(candidate)
            if (idx >= 0) return idx
        }
        return -1
    }
}
