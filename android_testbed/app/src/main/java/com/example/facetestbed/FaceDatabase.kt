package com.example.facetestbed

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.sqrt

data class PersonRecord(
    val name: String,
    val department: String,
    val centroid: FloatArray,
    val embeddings: MutableList<FloatArray>,
)

data class FaceMatch(
    val name: String,
    val department: String,
    val score: Float,
)

class FaceDatabase private constructor(private val backend: StorageBackend) {
    constructor(dbFile: File) : this(FileStorageBackend(dbFile))

    constructor(context: Context, treeUri: Uri, fileName: String = "face_db.json") : this(
        TreeUriStorageBackend(context, treeUri, fileName)
    )

    private val people = mutableListOf<PersonRecord>()

    init {
        load()
    }

    @Synchronized
    fun isEmpty(): Boolean = people.isEmpty()

    @Synchronized
    fun personCount(): Int = people.size

    @Synchronized
    fun upsertPerson(name: String, department: String, samples: List<FloatArray>, maxEmbeddingsPerPerson: Int = 100) {
        if (samples.isEmpty()) return

        val cleanedName = sanitizeName(name)
        val cleanedDepartment = sanitizeDepartment(department)
        val pruned = pruneToLimit(samples.map { it.copyOf() }.toMutableList(), maxEmbeddingsPerPerson)
        val centroid = l2Normalize(meanVector(pruned))

        people.removeAll { it.name == cleanedName }
        people.add(PersonRecord(cleanedName, cleanedDepartment, centroid, pruned.toMutableList()))
        save()
    }

    @Synchronized
    fun rank(embedding: FloatArray, topK: Int = 3): List<FaceMatch> {
        if (people.isEmpty()) return emptyList()

        return people
            .map { FaceMatch(it.name, it.department, dot(it.centroid, embedding)) }
            .sortedByDescending { it.score }
            .take(topK.coerceAtLeast(1))
    }

    @Synchronized
    fun departmentOf(name: String): String {
        return people.firstOrNull { it.name == name }?.department.orEmpty()
    }

    @Synchronized
    fun appendSampleToPerson(name: String, embedding: FloatArray, maxEmbeddingsPerPerson: Int = 100): Boolean {
        val person = people.firstOrNull { it.name == name } ?: return false
        person.embeddings.add(embedding.copyOf())
        val pruned = pruneToLimit(person.embeddings, maxEmbeddingsPerPerson).toMutableList()
        person.embeddings.clear()
        person.embeddings.addAll(pruned)
        val newCentroid = l2Normalize(meanVector(person.embeddings))
        for (i in person.centroid.indices) {
            person.centroid[i] = if (i < newCentroid.size) newCentroid[i] else 0f
        }
        save()
        return true
    }

    @Synchronized
    fun exportJson(pretty: Boolean = true): String {
        val root = toJsonObject()
        return if (pretty) root.toString(2) else root.toString()
    }

    @Synchronized
    fun exportDiffSummaryJson(): String {
        val arr = JSONArray()
        people.sortedBy { it.name }.forEach { person ->
            arr.put(
                JSONObject()
                    .put("name", person.name)
                    .put("hash", personHash(person)),
            )
        }
        return JSONObject().put("people", arr).toString()
    }

    @Synchronized
    fun replaceWithJson(rawJson: String): Boolean {
        val parsed = runCatching { JSONObject(rawJson) }.getOrNull() ?: return false
        val nextPeople = parsePeople(parsed) ?: return false
        people.clear()
        people.addAll(nextPeople)
        save()
        return true
    }

    @Synchronized
    fun applyDiffPatch(upsertsRaw: String, deletesRaw: String): Boolean {
        val upserts = runCatching { JSONArray(upsertsRaw) }.getOrNull() ?: return false
        val deletes = runCatching { JSONArray(deletesRaw) }.getOrDefault(JSONArray())
        val next = people.associateBy { it.name }.toMutableMap()

        for (i in 0 until deletes.length()) {
            val name = deletes.optString(i, "").trim()
            if (name.isNotBlank()) {
                next.remove(name)
            }
        }

        for (i in 0 until upserts.length()) {
            val item = upserts.optJSONObject(i) ?: continue
            val parsed = parsePerson(item) ?: continue
            next[parsed.name] = parsed
        }

        people.clear()
        people.addAll(next.values.sortedBy { it.name })
        save()
        return true
    }

    private fun load() {
        val raw = backend.readText() ?: return
        val root = runCatching { JSONObject(raw) }.getOrNull() ?: return
        val loaded = parsePeople(root) ?: return
        people.clear()
        people.addAll(loaded)
    }

    private fun save() {
        val root = toJsonObject()
        backend.writeText(root.toString(2))
    }

    private fun toJsonObject(): JSONObject {
        val root = JSONObject()
        val arr = JSONArray()
        for (p in people) {
            val obj = JSONObject()
            obj.put("name", p.name)
            obj.put("department", p.department)
            obj.put("centroid", floatArrayToJsonArray(p.centroid))
            val embs = JSONArray()
            for (e in p.embeddings) {
                embs.put(floatArrayToJsonArray(e))
            }
            obj.put("embeddings", embs)
            arr.put(obj)
        }
        root.put("people", arr)
        return root
    }

    private fun parsePeople(root: JSONObject): MutableList<PersonRecord>? {
        val arr = root.optJSONArray("people") ?: return mutableListOf()
        val out = mutableListOf<PersonRecord>()
        for (i in 0 until arr.length()) {
            val item = arr.optJSONObject(i) ?: continue
            val person = parsePerson(item) ?: continue
            out.add(person)
        }
        return out
    }

    private fun parsePerson(item: JSONObject): PersonRecord? {
        val name = item.optString("name", "").trim()
        if (name.isEmpty()) return null
        val department = item.optString("department", item.optString("folder", "")).trim()

        val embeddingsJson = item.optJSONArray("embeddings") ?: JSONArray()
        val embeddings = mutableListOf<FloatArray>()
        for (j in 0 until embeddingsJson.length()) {
            val embJson = embeddingsJson.optJSONArray(j) ?: continue
            embeddings.add(jsonArrayToFloatArray(embJson))
        }
        if (embeddings.isEmpty()) return null
        val centroid = l2Normalize(meanVector(embeddings))
        return PersonRecord(name, department, centroid, embeddings)
    }

    private fun pruneToLimit(embeddings: MutableList<FloatArray>, limit: Int): List<FloatArray> {
        val safeLimit = limit.coerceAtLeast(1)
        while (embeddings.size > safeLimit) {
            val centroid = l2Normalize(meanVector(embeddings))
            var dropIdx = 0
            var minScore = Float.MAX_VALUE
            for (i in embeddings.indices) {
                val score = dot(embeddings[i], centroid)
                if (score < minScore) {
                    minScore = score
                    dropIdx = i
                }
            }
            embeddings.removeAt(dropIdx)
        }
        return embeddings
    }

    private fun sanitizeName(raw: String): String {
        val cleaned = buildString {
            raw.trim().forEach { ch ->
                if (ch.isLetterOrDigit() || ch == '_' || ch == '-' || ch.code in 0xAC00..0xD7A3) {
                    append(ch)
                } else {
                    append('_')
                }
            }
        }
        return if (cleaned.isBlank()) "unknown" else cleaned
    }

    private fun sanitizeDepartment(raw: String): String {
        return raw.trim().replace(Regex("\\s+"), " ")
    }

    private fun meanVector(vectors: List<FloatArray>): FloatArray {
        val dim = vectors.firstOrNull()?.size ?: 0
        val out = FloatArray(dim)
        if (dim == 0) return out
        for (v in vectors) {
            for (i in 0 until dim) out[i] += v[i]
        }
        for (i in 0 until dim) out[i] /= vectors.size.toFloat()
        return out
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += (x * x).toDouble()
        val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(v.size) { i -> v[i] / norm }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var s = 0f
        for (i in 0 until n) s += a[i] * b[i]
        return s
    }

    private fun floatArrayToJsonArray(v: FloatArray): JSONArray {
        val arr = JSONArray()
        for (x in v) arr.put(x.toDouble())
        return arr
    }

    private fun jsonArrayToFloatArray(arr: JSONArray): FloatArray {
        val out = FloatArray(arr.length())
        for (i in 0 until arr.length()) {
            out[i] = arr.optDouble(i, 0.0).toFloat()
        }
        return out
    }

    private fun personHash(person: PersonRecord): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(person.name.trim().toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(person.department.trim().toByteArray(Charsets.UTF_8))
        digest.update(0)
        updateDigestInt(digest, person.embeddings.size)
        person.embeddings.forEach { emb ->
            updateDigestInt(digest, emb.size)
            emb.forEach { v ->
                updateDigestInt(digest, java.lang.Float.floatToIntBits(v))
            }
        }
        return digest.digest().joinToString("") { b -> "%02x".format(b) }
    }

    private fun updateDigestInt(digest: MessageDigest, value: Int) {
        digest.update(((value ushr 24) and 0xFF).toByte())
        digest.update(((value ushr 16) and 0xFF).toByte())
        digest.update(((value ushr 8) and 0xFF).toByte())
        digest.update((value and 0xFF).toByte())
    }

    private interface StorageBackend {
        fun readText(): String?
        fun writeText(raw: String): Boolean
    }

    private class FileStorageBackend(private val dbFile: File) : StorageBackend {
        override fun readText(): String? {
            if (!dbFile.exists()) return null
            return runCatching { dbFile.readText() }.getOrNull()
        }

        override fun writeText(raw: String): Boolean {
            dbFile.parentFile?.mkdirs()
            return runCatching {
                dbFile.writeText(raw)
            }.isSuccess
        }
    }

    private class TreeUriStorageBackend(
        private val context: Context,
        private val treeUri: Uri,
        private val fileName: String,
    ) : StorageBackend {
        override fun readText(): String? {
            val file = resolveDocumentFile(createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(file.uri)?.bufferedReader()?.use { it.readText() }
            }.getOrNull()
        }

        override fun writeText(raw: String): Boolean {
            val file = resolveDocumentFile(createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "wt")?.bufferedWriter()?.use { it.write(raw) } != null
            }.getOrDefault(false)
        }

        private fun resolveDocumentFile(createIfMissing: Boolean): DocumentFile? {
            val root = DocumentFile.fromTreeUri(context, treeUri) ?: return null
            val existing = root.findFile(fileName)
            if (existing != null) return existing
            if (!createIfMissing) return null
            return root.createFile("application/json", fileName)
        }
    }
}
