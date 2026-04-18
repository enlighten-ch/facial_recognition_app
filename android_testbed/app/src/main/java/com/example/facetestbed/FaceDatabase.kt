package com.example.facetestbed

import android.content.Context
import android.net.Uri
import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
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
    constructor(dbFile: File) : this(
        FileStorageBackend(
            csvFile = resolveCsvFile(dbFile),
            metaDbFile = resolveMetaDbFile(dbFile),
            embeddingsBinFile = resolveEmbeddingsBinFile(dbFile),
        ),
    )

    constructor(
        context: Context,
        treeUri: Uri,
        fileName: String = "face_db.csv",
    ) : this(
        TreeUriStorageBackend(
            context = context,
            treeUri = treeUri,
            csvFileName = fileName,
            metaDbFileName = "face_meta.db",
            embeddingsBinFileName = "face_embeddings.bin",
            metaDbFile = resolveTreeMetaDbFile(context, treeUri),
            embeddingsBinFile = resolveTreeEmbeddingsBinFile(context, treeUri),
        ),
    )

    private val people = mutableListOf<PersonRecord>()
    @Volatile
    private var rankIndex: List<RankIndexEntry> = emptyList()
    @Volatile
    private var departmentIndex: Map<String, String> = emptyMap()
    private val persistExecutor = Executors.newSingleThreadExecutor()
    private val persistStateLock = Any()
    private var persistRunning = false
    private var persistDirty = false

    init {
        load()
    }

    fun isEmpty(): Boolean = rankIndex.isEmpty()

    fun personCount(): Int = rankIndex.size

    @Synchronized
    fun totalEmbeddingCount(): Int = people.sumOf { it.embeddings.size }

    @Synchronized
    fun upsertPerson(name: String, department: String, samples: List<FloatArray>, maxEmbeddingsPerPerson: Int = 100) {
        if (samples.isEmpty()) return

        val cleanedName = sanitizeName(name)
        val cleanedDepartment = sanitizeDepartment(department)
        val pruned = pruneToLimit(samples.map { it.copyOf() }.toMutableList(), maxEmbeddingsPerPerson)
        val centroid = l2Normalize(meanVector(pruned))

        people.removeAll { it.name == cleanedName }
        people.add(PersonRecord(cleanedName, cleanedDepartment, centroid, pruned.toMutableList()))
        rebuildIndexesLocked()
        save()
    }

    fun rank(embedding: FloatArray, topK: Int = 3): List<FaceMatch> {
        val snapshot = rankIndex
        if (snapshot.isEmpty()) return emptyList()
        return snapshot
            .map { FaceMatch(it.name, it.department, dot(it.centroid, embedding)) }
            .sortedByDescending { it.score }
            .take(topK.coerceAtLeast(1))
    }

    fun departmentOf(name: String): String {
        return departmentIndex[name].orEmpty()
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
        rebuildIndexesLocked()
        save()
        return true
    }

    @Synchronized
    fun appendSamplesToPerson(name: String, embeddings: List<FloatArray>, maxEmbeddingsPerPerson: Int = 100): Int {
        if (embeddings.isEmpty()) return 0
        val person = people.firstOrNull { it.name == name } ?: return 0
        var added = 0
        embeddings.forEach { emb ->
            if (emb.isNotEmpty()) {
                person.embeddings.add(emb.copyOf())
                added += 1
            }
        }
        if (added <= 0) return 0
        val pruned = pruneToLimit(person.embeddings, maxEmbeddingsPerPerson).toMutableList()
        person.embeddings.clear()
        person.embeddings.addAll(pruned)
        val newCentroid = l2Normalize(meanVector(person.embeddings))
        for (i in person.centroid.indices) {
            person.centroid[i] = if (i < newCentroid.size) newCentroid[i] else 0f
        }
        rebuildIndexesLocked()
        save()
        return added
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

    fun replaceWithJson(rawJson: String): Boolean {
        val parsed = runCatching { JSONObject(rawJson) }.getOrNull() ?: return false
        val nextPeople = parsePeople(parsed) ?: return false
        synchronized(this) {
            people.clear()
            people.addAll(nextPeople)
            rebuildIndexesLocked()
        }
        save()
        return true
    }

    fun replaceWithStructuredBytes(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
        if (metaDbBytes.isEmpty() || embeddingsBinBytes.isEmpty()) return false
        if (!backend.writeStructured(metaDbBytes, embeddingsBinBytes)) return false
        val nextPeople = backend.readPeople() ?: return false
        synchronized(this) {
            people.clear()
            people.addAll(nextPeople)
            rebuildIndexesLocked()
        }
        return true
    }

    fun exportStructuredBytes(): Pair<ByteArray, ByteArray>? {
        val metaBytes = backend.readStructuredMetaBytes() ?: return null
        val embeddingsBytes = backend.readStructuredEmbeddingsBytes() ?: return null
        if (metaBytes.isEmpty() || embeddingsBytes.isEmpty()) return null
        return metaBytes to embeddingsBytes
    }

    fun applyDiffPatch(upsertsRaw: String, deletesRaw: String): Boolean {
        val upserts = runCatching { JSONArray(upsertsRaw) }.getOrNull() ?: return false
        val deletes = runCatching { JSONArray(deletesRaw) }.getOrDefault(JSONArray())
        val deleteNames = mutableSetOf<String>()
        for (i in 0 until deletes.length()) {
            val name = deletes.optString(i, "").trim()
            if (name.isNotBlank()) {
                deleteNames += name
            }
        }

        val upsertPeople = mutableListOf<PersonRecord>()
        for (i in 0 until upserts.length()) {
            val item = upserts.optJSONObject(i) ?: continue
            val parsed = parsePerson(item) ?: continue
            upsertPeople += parsed
        }

        synchronized(this) {
            val next = people.associateBy { it.name }.toMutableMap()
            deleteNames.forEach { next.remove(it) }
            upsertPeople.forEach { parsed ->
                next[parsed.name] = parsed
            }
            people.clear()
            people.addAll(next.values.sortedBy { it.name })
            rebuildIndexesLocked()
        }
        save()
        return true
    }

    private fun load() {
        val startedAt = System.currentTimeMillis()
        logLoad("DB load start")
        val loaded = backend.readPeople() ?: return
        synchronized(this) {
            people.clear()
            people.addAll(loaded)
            rebuildIndexesLocked()
        }
        val totalElapsedMs = System.currentTimeMillis() - startedAt
        logLoad("DB load complete: people=${loaded.size}, total=${totalElapsedMs}ms")
    }

    private fun logLoad(message: String) {
        Log.i(LOGCAT_TAG, message)
    }

    private fun save() {
        synchronized(persistStateLock) {
            persistDirty = true
            if (persistRunning) return
            persistRunning = true
        }
        persistExecutor.execute {
            while (true) {
                synchronized(persistStateLock) {
                    persistDirty = false
                }
                val snapshot = synchronized(this@FaceDatabase) { toJsonObject() }
                runCatching { backend.writeDb(snapshot) }
                val hasMore = synchronized(persistStateLock) { persistDirty }
                if (!hasMore) {
                    synchronized(persistStateLock) {
                        persistRunning = false
                    }
                    break
                }
            }
        }
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

    private fun rebuildIndexesLocked() {
        rankIndex = people.map { person ->
            RankIndexEntry(
                name = person.name,
                department = person.department,
                centroid = person.centroid.copyOf(),
            )
        }
        departmentIndex = people.associate { person -> person.name to person.department }
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

    private fun parsePeopleFromCsv(rawCsv: String): MutableList<PersonRecord>? {
        val rows = parseCsvRows(rawCsv)
        if (rows.isEmpty()) return mutableListOf()

        val header = rows.first().map { it.trim().lowercase(Locale.US) }
        val nameIndex = indexOfAny(header, listOf("name", "이름"))
        if (nameIndex < 0) return null
        val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
        val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))

        val out = mutableListOf<PersonRecord>()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val embeddings = parseEmbeddingsFromCsvCell(row.getOrNull(embeddingsIndex).orEmpty())
            if (embeddings.isEmpty()) continue
            out.add(
                PersonRecord(
                    name = sanitizeName(name),
                    department = sanitizeDepartment(department),
                    centroid = l2Normalize(meanVector(embeddings)),
                    embeddings = embeddings.toMutableList(),
                ),
            )
        }
        return out
    }

    private fun parseEmbeddingsFromCsvCell(raw: String): List<FloatArray> {
        if (raw.isBlank()) return emptyList()
        val parsed = runCatching { JSONArray(raw) }.getOrNull() ?: return emptyList()
        val embeddings = mutableListOf<FloatArray>()
        for (i in 0 until parsed.length()) {
            val embJson = parsed.optJSONArray(i) ?: continue
            embeddings.add(jsonArrayToFloatArray(embJson))
        }
        return embeddings
    }

    private fun toCsvText(): String {
        val out = StringBuilder()
        out.append("name,department,embedding_count,updated_at\n")
        val now = System.currentTimeMillis().toString()
        people.sortedBy { it.name }.forEach { person ->
            val row = listOf(
                person.name,
                person.department,
                person.embeddings.size.toString(),
                now,
            )
            out.append(row.joinToString(",") { csvEscape(it) }).append('\n')
        }
        return out.toString()
    }

    private fun embeddingsToJsonText(embeddings: List<FloatArray>): String {
        val arr = JSONArray()
        embeddings.forEach { emb ->
            arr.put(floatArrayToJsonArray(emb))
        }
        return arr.toString()
    }

    private fun csvEscape(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        val needsQuote = escaped.any { ch -> ch == ',' || ch == '"' || ch == '\n' || ch == '\r' }
        return if (needsQuote) "\"$escaped\"" else escaped
    }

    private fun parseCsvRows(raw: String): List<List<String>> {
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

    private data class RankIndexEntry(
        val name: String,
        val department: String,
        val centroid: FloatArray,
    )

    private interface StorageBackend {
        fun readPeople(): MutableList<PersonRecord>?
        fun writeDb(root: JSONObject): Boolean
        fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean
        fun readStructuredMetaBytes(): ByteArray?
        fun readStructuredEmbeddingsBytes(): ByteArray?
    }

    private class FileStorageBackend(
        private val csvFile: File,
        private val metaDbFile: File,
        private val embeddingsBinFile: File,
    ) : StorageBackend {
        override fun readPeople(): MutableList<PersonRecord>? {
            val structuredExists = metaDbFile.exists() && embeddingsBinFile.exists()
            val csvNeedsImport = csvFile.exists() &&
                (
                    !structuredExists ||
                        csvFile.lastModified() > maxOf(metaDbFile.lastModified(), embeddingsBinFile.lastModified())
                    )

            if (csvNeedsImport) {
                val raw = runCatching { csvFile.readText() }.getOrNull()
                if (raw != null) {
                    if (structuredExists && !StructuredCodec.hasEmbeddingsColumn(raw)) {
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                    val root =
                        if (structuredExists) {
                            val base = StructuredCodec.loadFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredCodec.mergeWithCsvMetadata(base, raw) else null
                        } else {
                            StructuredCodec.parseCsvRoot(raw)
                        }
                    if (root != null && StructuredCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) {
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                }
            }

            if (structuredExists) {
                StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
            }
            return null
        }

        override fun writeDb(root: JSONObject): Boolean {
            if (!StructuredCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) return false
            return writeCsvSnapshot(root)
        }

        override fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaDbBytes)
                embeddingsBinFile.writeBytes(embeddingsBinBytes)
                true
            }.getOrDefault(false)
        }

        override fun readStructuredMetaBytes(): ByteArray? {
            return runCatching { metaDbFile.readBytes() }.getOrNull()
        }

        override fun readStructuredEmbeddingsBytes(): ByteArray? {
            return runCatching { embeddingsBinFile.readBytes() }.getOrNull()
        }

        private fun writeCsvSnapshot(root: JSONObject): Boolean {
            return runCatching {
                csvFile.parentFile?.mkdirs()
                csvFile.writeText(StructuredCodec.toCsv(root))
                true
            }.getOrDefault(false)
        }
    }

    private class TreeUriStorageBackend(
        private val context: Context,
        private val treeUri: Uri,
        private val csvFileName: String,
        private val metaDbFileName: String,
        private val embeddingsBinFileName: String,
        private val metaDbFile: File,
        private val embeddingsBinFile: File,
    ) : StorageBackend {
        override fun readPeople(): MutableList<PersonRecord>? {
            val metaDoc = resolveDocumentFile(metaDbFileName, createIfMissing = false)
            val binDoc = resolveDocumentFile(embeddingsBinFileName, createIfMissing = false)
            val docStructuredExists = (metaDoc != null && binDoc != null) && syncStructuredFromDocuments(metaDoc, binDoc)
            val localStructuredExists = metaDbFile.exists() && embeddingsBinFile.exists()
            val structuredExists = docStructuredExists || localStructuredExists
            val csvDoc = resolveDocumentFile(csvFileName, createIfMissing = false)
            val csvNeedsImport = csvDoc != null &&
                (
                    !structuredExists ||
                        csvDoc.lastModified() > maxOf(metaDoc?.lastModified() ?: 0L, binDoc?.lastModified() ?: 0L)
                    )

            if (csvNeedsImport) {
                val csvRaw = runCatching {
                    context.contentResolver.openInputStream(csvDoc!!.uri)?.bufferedReader()?.use { it.readText() }
                }.getOrNull()
                if (csvRaw != null) {
                    if (structuredExists && !StructuredCodec.hasEmbeddingsColumn(csvRaw)) {
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                    val root =
                        if (structuredExists) {
                            val base = StructuredCodec.loadFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredCodec.mergeWithCsvMetadata(base, csvRaw) else null
                        } else {
                            StructuredCodec.parseCsvRoot(csvRaw)
                        }
                    if (root != null && StructuredCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) {
                        syncStructuredToDocuments()
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                }
            }

            if (structuredExists) {
                StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
            }
            return null
        }

        override fun writeDb(root: JSONObject): Boolean {
            if (!StructuredCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) return false
            syncStructuredToDocuments()
            return writeDocumentText(csvFileName, StructuredCodec.toCsv(root))
        }

        override fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
            val localOk = runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaDbBytes)
                embeddingsBinFile.writeBytes(embeddingsBinBytes)
                true
            }.getOrDefault(false)
            if (!localOk) return false
            syncStructuredToDocuments()
            return true
        }

        override fun readStructuredMetaBytes(): ByteArray? {
            if (metaDbFile.exists()) {
                runCatching { metaDbFile.readBytes() }.getOrNull()?.let { return it }
            }
            val metaDoc = resolveDocumentFile(metaDbFileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(metaDoc.uri)?.use { it.readBytes() }
            }.getOrNull()
        }

        override fun readStructuredEmbeddingsBytes(): ByteArray? {
            if (embeddingsBinFile.exists()) {
                runCatching { embeddingsBinFile.readBytes() }.getOrNull()?.let { return it }
            }
            val binDoc = resolveDocumentFile(embeddingsBinFileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(binDoc.uri)?.use { it.readBytes() }
            }.getOrNull()
        }

        private fun readDocumentText(fileName: String): String? {
            val file = resolveDocumentFile(fileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(file.uri)?.bufferedReader()?.use { it.readText() }
            }.getOrNull()
        }

        private fun writeDocumentText(fileName: String, raw: String): Boolean {
            val file = resolveDocumentFile(fileName, createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "wt")?.bufferedWriter()?.use { it.write(raw) } != null
            }.getOrDefault(false)
        }

        private fun writeDocumentBytes(fileName: String, bytes: ByteArray): Boolean {
            val file = resolveDocumentFile(fileName, createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "w")?.use { it.write(bytes) } != null
            }.getOrDefault(false)
        }

        private fun syncStructuredFromDocuments(metaDoc: DocumentFile?, binDoc: DocumentFile?): Boolean {
            if (metaDoc == null || binDoc == null) return false
            val metaBytes = runCatching {
                context.contentResolver.openInputStream(metaDoc.uri)?.use { it.readBytes() }
            }.getOrNull() ?: return false
            val binBytes = runCatching {
                context.contentResolver.openInputStream(binDoc.uri)?.use { it.readBytes() }
            }.getOrNull() ?: return false
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaBytes)
                embeddingsBinFile.writeBytes(binBytes)
                true
            }.getOrDefault(false)
        }

        private fun syncStructuredToDocuments(): Boolean {
            val metaBytes = runCatching { metaDbFile.readBytes() }.getOrNull() ?: return false
            val binBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return false
            val metaOk = writeDocumentBytes(metaDbFileName, metaBytes)
            val binOk = writeDocumentBytes(embeddingsBinFileName, binBytes)
            return metaOk && binOk
        }

        private fun resolveDocumentFile(fileName: String, createIfMissing: Boolean): DocumentFile? {
            val root = DocumentFile.fromTreeUri(context, treeUri) ?: return null
            val existing = root.findFile(fileName)
            if (existing != null) return existing
            if (!createIfMissing) return null
            val lower = fileName.lowercase(Locale.US)
            val mimeType = if (lower.endsWith(".csv")) "text/csv" else "application/octet-stream"
            return root.createFile(mimeType, fileName)
        }
    }

    private object StructuredCodec {
        private const val TABLE = "people"

        fun hasEmbeddingsColumn(raw: String): Boolean {
            val rows = parseCsvRows(raw)
            if (rows.isEmpty()) return false
            val header = rows.first().map { it.trim().lowercase(Locale.US) }
            return indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json")) >= 0
        }

        fun toCsv(root: JSONObject): String {
            val people = normalizeRoot(root).optJSONArray("people") ?: JSONArray()
            val out = StringBuilder()
            out.append("name,department,embedding_count,updated_at\n")
            val now = System.currentTimeMillis()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                val department = person.optString("department", person.optString("folder", "")).trim()
                val embeddings = normalizeEmbeddings(person.optJSONArray("embeddings") ?: JSONArray())
                val updatedAt = person.optLong("updatedAt", now)
                val row = listOf(
                    name,
                    department,
                    embeddings.length().toString(),
                    updatedAt.toString(),
                )
                out.append(row.joinToString(",") { csvEscape(it) }).append('\n')
            }
            return out.toString()
        }

        fun mergeWithCsvMetadata(currentRoot: JSONObject, csvRaw: String): JSONObject {
            val base = normalizeRoot(currentRoot)
            val rows = parseMetadataRows(csvRaw)
            if (rows.isEmpty()) return JSONObject().put("people", JSONArray())
            val people = base.optJSONArray("people") ?: JSONArray()
            val byName = mutableMapOf<String, JSONObject>()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                byName[name] = person
            }
            val out = JSONArray()
            rows.forEach { row ->
                val basePerson = byName[row.name] ?: return@forEach
                val obj = JSONObject(basePerson.toString())
                obj.put("name", row.name)
                obj.put("department", row.department)
                obj.put("updatedAt", row.updatedAt)
                val embeddings = normalizeEmbeddings(obj.optJSONArray("embeddings") ?: JSONArray())
                obj.put("embeddings", embeddings)
                obj.put("hash", computeHash(row.name, row.department, embeddings))
                out.put(obj)
            }
            return JSONObject().put("people", out)
        }

        fun loadFromStructured(metaDbFile: File, embeddingsBinFile: File): JSONObject? {
            if (!metaDbFile.exists() || !embeddingsBinFile.exists()) return null
            val embeddingsBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val arr = JSONArray()
                db.query(
                    TABLE,
                    arrayOf("name", "department", "offset_bytes", "length_bytes", "embedding_count", "embedding_dim", "hash", "updated_at"),
                    null,
                    null,
                    null,
                    null,
                    "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val name = cursor.getString(0)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(1)?.trim().orEmpty()
                        val offset = cursor.getLong(2).toInt()
                        val length = cursor.getLong(3).toInt()
                        val count = cursor.getInt(4).coerceAtLeast(0)
                        val dim = cursor.getInt(5).coerceAtLeast(0)
                        val hash = cursor.getString(6)?.trim().orEmpty()
                        val updatedAt = cursor.getLong(7)
                        val embeddings = decodeEmbeddings(embeddingsBytes, offset, length, count, dim)
                        val obj = JSONObject()
                            .put("name", name)
                            .put("department", department)
                            .put("embeddings", embeddings)
                        if (updatedAt > 0L) obj.put("updatedAt", updatedAt)
                        obj.put("hash", if (hash.isNotBlank()) hash else computeHash(name, department, embeddings))
                        arr.put(obj)
                    }
                }
                JSONObject().put("people", arr)
            }.getOrNull().also { db.close() }
        }

        fun loadPeopleFromStructured(metaDbFile: File, embeddingsBinFile: File): MutableList<PersonRecord>? {
            if (!metaDbFile.exists() || !embeddingsBinFile.exists()) return null
            val embeddingsBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val out = mutableListOf<PersonRecord>()
                db.query(
                    TABLE,
                    arrayOf("name", "department", "offset_bytes", "length_bytes", "embedding_count", "embedding_dim"),
                    null,
                    null,
                    null,
                    null,
                    "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val name = cursor.getString(0)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(1)?.trim().orEmpty()
                        val offset = cursor.getLong(2).toInt()
                        val length = cursor.getLong(3).toInt()
                        val count = cursor.getInt(4).coerceAtLeast(0)
                        val dim = cursor.getInt(5).coerceAtLeast(0)
                        val embeddings = decodeEmbeddingsToFloatArrays(embeddingsBytes, offset, length, count, dim)
                        if (embeddings.isEmpty()) continue
                        out.add(
                            PersonRecord(
                                name = name,
                                department = department,
                                centroid = l2NormalizeFloat(meanVectorFloat(embeddings)),
                                embeddings = embeddings.toMutableList(),
                            ),
                        )
                    }
                }
                out
            }.getOrNull().also { db.close() }
        }

        fun saveToStructured(root: JSONObject, metaDbFile: File, embeddingsBinFile: File): Boolean {
            val normalized = normalizeRoot(root)
            val people = normalized.optJSONArray("people") ?: JSONArray()
            val records = mutableListOf<Record>()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                val department = person.optString("department", person.optString("folder", "")).trim()
                val embeddings = normalizeEmbeddings(person.optJSONArray("embeddings") ?: JSONArray())
                if (embeddings.length() <= 0) continue
                val updatedAt = person.optLong("updatedAt", System.currentTimeMillis())
                val hash = person.optString("hash", "").trim().ifBlank { computeHash(name, department, embeddings) }
                records.add(Record(name, department, embeddings, hash, updatedAt))
            }
            val encoded = encodeEmbeddings(records)
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                embeddingsBinFile.writeBytes(encoded.bytes)
                val db = SQLiteDatabase.openOrCreateDatabase(metaDbFile, null)
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS $TABLE(" +
                        "name TEXT PRIMARY KEY," +
                        "department TEXT," +
                        "offset_bytes INTEGER," +
                        "length_bytes INTEGER," +
                        "embedding_count INTEGER," +
                        "embedding_dim INTEGER," +
                        "hash TEXT," +
                        "updated_at INTEGER" +
                        ")",
                )
                db.beginTransaction()
                try {
                    db.delete(TABLE, null, null)
                    encoded.rows.forEach { row ->
                        val values = ContentValues().apply {
                            put("name", row.name)
                            put("department", row.department)
                            put("offset_bytes", row.offsetBytes)
                            put("length_bytes", row.lengthBytes)
                            put("embedding_count", row.embeddingCount)
                            put("embedding_dim", row.embeddingDim)
                            put("hash", row.hash)
                            put("updated_at", row.updatedAt)
                        }
                        db.insert(TABLE, null, values)
                    }
                    db.setTransactionSuccessful()
                } finally {
                    db.endTransaction()
                    db.close()
                }
                true
            }.getOrDefault(false)
        }

        fun parseCsvRoot(raw: String): JSONObject? {
            val rows = parseCsvRows(raw)
            if (rows.isEmpty()) return JSONObject().put("people", JSONArray())
            val header = rows.first().map { it.trim().lowercase(Locale.US) }
            val nameIndex = indexOfAny(header, listOf("name", "이름"))
            if (nameIndex < 0) return null
            val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
            val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))
            val hashIndex = indexOfAny(header, listOf("hash", "record_hash"))
            val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))
            val arr = JSONArray()
            for (i in 1 until rows.size) {
                val row = rows[i]
                val name = row.getOrNull(nameIndex)?.trim().orEmpty()
                if (name.isBlank()) continue
                val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
                val embeddings = runCatching { JSONArray(row.getOrNull(embeddingsIndex).orEmpty()) }.getOrNull() ?: JSONArray()
                if (embeddings.length() <= 0) continue
                val hash = row.getOrNull(hashIndex)?.trim().orEmpty()
                val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: System.currentTimeMillis()
                val obj = JSONObject()
                    .put("name", name)
                    .put("department", department)
                    .put("embeddings", normalizeEmbeddings(embeddings))
                    .put("updatedAt", updatedAt)
                if (hash.isNotBlank()) obj.put("hash", hash)
                arr.put(obj)
            }
            return JSONObject().put("people", arr)
        }

        private fun parseMetadataRows(raw: String): List<CsvMetaRow> {
            val rows = parseCsvRows(raw)
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
                out.add(CsvMetaRow(name, department, updatedAt))
            }
            return out
        }

        private fun normalizeRoot(root: JSONObject): JSONObject {
            root.optJSONArray("people")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
            root.optJSONArray("persons")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
            root.optJSONArray("faces")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
            return JSONObject().put("people", JSONArray())
        }

        private fun normalizeEmbeddings(arr: JSONArray): JSONArray {
            val out = JSONArray()
            for (i in 0 until arr.length()) {
                val emb = arr.optJSONArray(i) ?: continue
                val vec = JSONArray()
                for (j in 0 until emb.length()) {
                    vec.put(emb.optDouble(j, 0.0))
                }
                out.put(vec)
            }
            return out
        }

        private fun encodeEmbeddings(records: List<Record>): EncodedData {
            val rows = mutableListOf<RowMeta>()
            val out = ByteArrayOutputStream()
            records.sortedBy { it.name }.forEach { record ->
                val offset = out.size()
                val count = record.embeddings.length()
                val dim = if (count > 0) (record.embeddings.optJSONArray(0)?.length() ?: 0) else 0
                for (i in 0 until count) {
                    val emb = record.embeddings.optJSONArray(i) ?: JSONArray()
                    for (j in 0 until dim) {
                        writeFloatLe(out, emb.optDouble(j, 0.0).toFloat())
                    }
                }
                val length = out.size() - offset
                rows.add(RowMeta(record.name, record.department, offset, length, count, dim, record.hash, record.updatedAt))
            }
            return EncodedData(rows, out.toByteArray())
        }

        private fun decodeEmbeddings(bytes: ByteArray, offset: Int, length: Int, count: Int, dim: Int): JSONArray {
            val vectors = decodeEmbeddingsToFloatArrays(bytes, offset, length, count, dim)
            val arr = JSONArray()
            vectors.forEach { vec ->
                val emb = JSONArray()
                vec.forEach { value -> emb.put(value.toDouble()) }
                arr.put(emb)
            }
            return arr
        }

        private fun decodeEmbeddingsToFloatArrays(
            bytes: ByteArray,
            offset: Int,
            length: Int,
            count: Int,
            dim: Int,
        ): List<FloatArray> {
            if (offset < 0 || length < 0 || offset + length > bytes.size || count <= 0 || dim <= 0) return emptyList()
            val out = ArrayList<FloatArray>(count)
            var p = offset
            val limit = offset + length
            for (i in 0 until count) {
                if (p + dim * 4 > limit) break
                val emb = FloatArray(dim)
                for (j in 0 until dim) {
                    val bits =
                        (bytes[p].toInt() and 0xFF) or
                            ((bytes[p + 1].toInt() and 0xFF) shl 8) or
                            ((bytes[p + 2].toInt() and 0xFF) shl 16) or
                            ((bytes[p + 3].toInt() and 0xFF) shl 24)
                    p += 4
                    emb[j] = Float.fromBits(bits)
                }
                out.add(emb)
            }
            return out
        }

        private fun meanVectorFloat(vectors: List<FloatArray>): FloatArray {
            val dim = vectors.firstOrNull()?.size ?: 0
            val out = FloatArray(dim)
            if (dim == 0) return out
            for (v in vectors) {
                for (i in 0 until dim) out[i] += v[i]
            }
            for (i in 0 until dim) out[i] /= vectors.size.toFloat()
            return out
        }

        private fun l2NormalizeFloat(v: FloatArray): FloatArray {
            var sum = 0.0
            for (x in v) sum += (x * x).toDouble()
            val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
            return FloatArray(v.size) { i -> v[i] / norm }
        }

        private fun parseCsvRows(raw: String): List<List<String>> {
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
                        if (i + 1 < raw.length && raw[i + 1] == '\n') i += 1
                    }
                    else -> cell.append(c)
                }
                i += 1
            }
            if (cell.isNotEmpty() || row.isNotEmpty()) {
                row.add(cell.toString())
                rows.add(row)
            }
            return rows.filterNot { fields -> fields.size == 1 && fields[0].isBlank() }
        }

        private fun csvEscape(value: String): String {
            val escaped = value.replace("\"", "\"\"")
            val needsQuote = escaped.any { ch -> ch == ',' || ch == '"' || ch == '\n' || ch == '\r' }
            return if (needsQuote) "\"$escaped\"" else escaped
        }

        private fun indexOfAny(header: List<String>, candidates: List<String>): Int {
            candidates.forEach { candidate ->
                val idx = header.indexOf(candidate)
                if (idx >= 0) return idx
            }
            return -1
        }

        private fun writeFloatLe(out: ByteArrayOutputStream, value: Float) {
            val bits = java.lang.Float.floatToIntBits(value)
            out.write(bits and 0xFF)
            out.write((bits ushr 8) and 0xFF)
            out.write((bits ushr 16) and 0xFF)
            out.write((bits ushr 24) and 0xFF)
        }

        private fun computeHash(name: String, department: String, embeddings: JSONArray): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            updateDigestInt(digest, embeddings.length())
            for (i in 0 until embeddings.length()) {
                val emb = embeddings.optJSONArray(i) ?: JSONArray()
                updateDigestInt(digest, emb.length())
                for (j in 0 until emb.length()) {
                    val v = emb.optDouble(j, 0.0).toFloat()
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

        private data class Record(
            val name: String,
            val department: String,
            val embeddings: JSONArray,
            val hash: String,
            val updatedAt: Long,
        )

        private data class CsvMetaRow(
            val name: String,
            val department: String,
            val updatedAt: Long,
        )

        private data class RowMeta(
            val name: String,
            val department: String,
            val offsetBytes: Int,
            val lengthBytes: Int,
            val embeddingCount: Int,
            val embeddingDim: Int,
            val hash: String,
            val updatedAt: Long,
        )

        private data class EncodedData(
            val rows: List<RowMeta>,
            val bytes: ByteArray,
        )
    }

    companion object {
        private const val LOGCAT_TAG = "FaceTestDb"

        private fun resolveCsvFile(dbFile: File): File {
            val lowerName = dbFile.name.lowercase(Locale.US)
            if (lowerName.endsWith(".csv")) return dbFile
            val parent = dbFile.parentFile ?: File(".")
            return File(parent, "face_db.csv")
        }

        private fun resolveMetaDbFile(dbFile: File): File {
            val parent = dbFile.parentFile ?: File(".")
            return File(parent, "face_meta.db")
        }

        private fun resolveEmbeddingsBinFile(dbFile: File): File {
            val parent = dbFile.parentFile ?: File(".")
            return File(parent, "face_embeddings.bin")
        }

        private fun resolveTreeMetaDbFile(context: Context, treeUri: Uri): File {
            val key = Integer.toHexString(treeUri.toString().hashCode())
            return File(context.filesDir, "face_data_meta/face_meta_$key.db")
        }

        private fun resolveTreeEmbeddingsBinFile(context: Context, treeUri: Uri): File {
            val key = Integer.toHexString(treeUri.toString().hashCode())
            return File(context.filesDir, "face_data_meta/face_embeddings_$key.bin")
        }
    }
}
