package com.example.facedbserver

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.util.Locale

data class CsvMetaRow(
    val name: String,
    val department: String,
    val updatedAt: Long,
)

class FaceDbCsvCodec {
    fun decode(rawCsv: String): JSONObject? {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) {
            return JSONObject().put("people", JSONArray())
        }

        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "이름"))
        if (nameIndex < 0) return null
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))
        val hashIndex = indexOfAny(header, listOf("hash", "record_hash"))
        val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))

        val people = JSONArray()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val embeddings = parseEmbeddings(row.getOrNull(embeddingsIndex).orEmpty())
            val item = JSONObject()
                .put("name", name)
                .put("department", department)
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
        val out = StringBuilder()
        out.append("name,department,embedding_count,updated_at\n")
        for (i in 0 until people.length()) {
            val person = people.optJSONObject(i) ?: continue
            val name = person.optString("name", "").trim()
            if (name.isBlank()) continue
            val department = person.optString("department", person.optString("folder", "")).trim()
            val embeddings = person.optJSONArray("embeddings") ?: JSONArray()
            val updatedAt = person.optLong("updatedAt", 0L)
            val row = listOf(
                name,
                department,
                embeddings.length().toString(),
                if (updatedAt > 0L) updatedAt.toString() else System.currentTimeMillis().toString(),
            )
            out.append(row.joinToString(",") { escapeCell(it) }).append('\n')
        }
        return out.toString()
    }

    fun parseMetadataRows(rawCsv: String): List<CsvMetaRow> {
        val rows = parseCsv(rawCsv)
        if (rows.isEmpty()) return emptyList()
        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "이름"))
        if (nameIndex < 0) return emptyList()
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))

        val out = mutableListOf<CsvMetaRow>()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: System.currentTimeMillis()
            out.add(CsvMetaRow(name = name, department = department, updatedAt = updatedAt))
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
