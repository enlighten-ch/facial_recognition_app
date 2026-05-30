package com.example.facetestbed

import android.content.Context
import android.net.Uri
import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import com.example.faithid.common.csvMetadataFromRow
import com.example.faithid.common.decodeMetadataBlob
import com.example.faithid.common.defaultMetadataMap
import com.example.faithid.common.encodeMetadataBlob
import com.example.faithid.common.metadataDepartment
import com.example.faithid.common.metadataGrade
import com.example.faithid.common.metadataHash
import com.example.faithid.common.metadataIdentityKey
import com.example.faithid.common.metadataName
import com.example.faithid.common.normalizeMetadataMap
import com.example.faithid.common.parsePersonId
import com.example.faithid.common.parsePersonKey
import com.example.faithid.common.personIdFromIdentity
import com.example.faithid.common.personKeyFromIdentity
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.MessageDigest
import java.util.LinkedHashMap
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.sqrt

data class PersonRecord(
    val name: String,
    val department: String,
    val grade: String = "",
    val centroid: FloatArray,
    val embeddings: MutableList<FloatArray>,
    val hash: String,
    val updatedAt: Long,
    val personId: Long = personIdFromIdentity(name, department, grade),
    val personKey: String = personKeyFromIdentity(name, department, grade),
    val metadata: LinkedHashMap<String, String> = defaultMetadataMap(name, department, grade),
    val metadataHash: String = metadataHash(metadata),
)

data class FaceMatch(
    val name: String,
    val department: String,
    val grade: String = "",
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
    @Volatile
    private var gradeIndex: Map<String, String> = emptyMap()
    @Volatile
    private var personIdIndex: Map<Long, PersonRecord> = emptyMap()
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
    fun upsertPerson(name: String, department: String, grade: String = "", samples: List<FloatArray>, maxEmbeddingsPerPerson: Int = 100) {
        if (samples.isEmpty()) return

        val cleanedName = sanitizeName(name)
        val cleanedDepartment = sanitizeDepartment(department)
        val cleanedGrade = sanitizeGrade(grade)
        val pruned = pruneToLimit(samples.map { it.copyOf() }.toMutableList(), maxEmbeddingsPerPerson)
        val centroid = l2Normalize(meanVector(pruned))

        people.removeAll { it.name == cleanedName }
        val updatedAt = System.currentTimeMillis()
        val metadata = defaultMetadataMap(cleanedName, cleanedDepartment, cleanedGrade)
        people.add(
            PersonRecord(
                name = cleanedName,
                department = cleanedDepartment,
                grade = cleanedGrade,
                centroid = centroid,
                embeddings = pruned.toMutableList(),
                hash = personHash(cleanedName, cleanedDepartment, cleanedGrade, pruned),
                updatedAt = updatedAt,
                metadata = metadata,
                metadataHash = metadataHash(metadata),
            ),
        )
        rebuildIndexesLocked()
        save()
    }

    fun rank(embedding: FloatArray, topK: Int = 3): List<FaceMatch> {
        val snapshot = rankIndex
        if (snapshot.isEmpty()) return emptyList()
        return snapshot
            .map { FaceMatch(it.name, it.department, it.grade, dot(it.centroid, embedding)) }
            .sortedByDescending { it.score }
            .take(topK.coerceAtLeast(1))
    }

    fun departmentOf(name: String): String {
        return departmentIndex[name].orEmpty()
    }

    fun gradeOf(name: String): String {
        return gradeIndex[name].orEmpty()
    }

    @Synchronized
    fun appendSampleToPerson(name: String, embedding: FloatArray, maxEmbeddingsPerPerson: Int = 100): Boolean {
        val person = people.firstOrNull { it.name == name } ?: return false
        person.embeddings.add(embedding.copyOf())
        val pruned = pruneToLimit(person.embeddings, maxEmbeddingsPerPerson).toMutableList()
        person.embeddings.clear()
        person.embeddings.addAll(pruned)
        val refreshed = person.copy(
            centroid = l2Normalize(meanVector(person.embeddings)),
            embeddings = person.embeddings.toMutableList(),
            hash = personHash(person.name, person.department, person.grade, person.embeddings),
            updatedAt = System.currentTimeMillis(),
            metadataHash = metadataHash(person.metadata),
        )
        people[people.indexOf(person)] = refreshed
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
        val refreshed = person.copy(
            centroid = l2Normalize(meanVector(person.embeddings)),
            embeddings = person.embeddings.toMutableList(),
            hash = personHash(person.name, person.department, person.grade, person.embeddings),
            updatedAt = System.currentTimeMillis(),
            metadataHash = metadataHash(person.metadata),
        )
        people[people.indexOf(person)] = refreshed
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
            val normalized = assignSequentialPersonIdsLocked(nextPeople)
            people.clear()
            people.addAll(normalized)
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
            val normalized = assignSequentialPersonIdsLocked(nextPeople)
            people.clear()
            people.addAll(normalized)
            rebuildIndexesLocked()
        }
        save()
        return true
    }

    fun isStructuredSchemaCurrent(): Boolean {
        return !backend.needsStructuredSchemaUpgrade()
    }

    fun exportStructuredBytes(): Pair<ByteArray, ByteArray>? {
        val metaBytes = backend.readStructuredMetaBytes() ?: return null
        val embeddingsBytes = backend.readStructuredEmbeddingsBytes() ?: return null
        if (metaBytes.isEmpty() || embeddingsBytes.isEmpty()) return null
        return metaBytes to embeddingsBytes
    }

    @Synchronized
    fun exportHashSummary(): Map<String, String> {
        return people.associate { it.personKey to it.hash }
    }

    @Synchronized
    fun totalEmbeddingCountForNames(names: Set<String>): Int {
        if (names.isEmpty()) return 0
        return people.filter { it.personKey in names }.sumOf { it.embeddings.size }
    }

    @Synchronized
    fun exportDeltaBundle(serverHashes: Map<String, String>): Pair<ByteArray, Int>? {
        val upserts = people
            .filter { person -> serverHashes[person.personKey] != person.hash }
            .sortedBy { it.personKey }
        if (upserts.isEmpty()) return null
        return DeltaCodec.encode(upserts, emptyList()) to upserts.sumOf { it.embeddings.size }
    }

    fun applyDeltaBundle(bundle: ByteArray): Boolean {
        val delta = DeltaCodec.decode(bundle) ?: return false
        if (delta.upserts.isEmpty() && delta.deletes.isEmpty()) return true
        synchronized(this) {
            val next = people.associateBy { it.personKey }.toMutableMap()
            val legacyIdentity = next.values.associateBy { metadataIdentityKey(it.metadata) }.toMutableMap()
            delta.deletes.forEach { personKey ->
                val removed = next.remove(personKey)
                if (removed != null) {
                    legacyIdentity.remove(metadataIdentityKey(removed.metadata))
                }
            }
            delta.upserts.forEach { parsed ->
                val identityKey = metadataIdentityKey(parsed.metadata)
                val existingPersonKey = next[parsed.personKey]?.personKey ?: legacyIdentity[identityKey]?.personKey
                if (existingPersonKey != null && existingPersonKey != parsed.personKey) {
                    val removed = next.remove(existingPersonKey)
                    if (removed != null) legacyIdentity.remove(metadataIdentityKey(removed.metadata))
                }
                next[parsed.personKey] = parsed
                legacyIdentity[identityKey] = parsed
            }
            people.clear()
            people.addAll(assignSequentialPersonIdsLocked(next.values.toList()))
            rebuildIndexesLocked()
        }
        save()
        return true
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
        if (backend.needsStructuredSchemaUpgrade()) {
            if (backend.writePeople(loaded)) {
                logLoad("Structured DB schema upgraded to person_key + person_idx + metadata_blob")
            } else {
                logLoad("Structured DB schema upgrade failed")
            }
        }
        synchronized(this) {
            val normalized = assignSequentialPersonIdsLocked(loaded)
            people.clear()
            people.addAll(normalized)
            rebuildIndexesLocked()
            if (loaded.map { it.personId } != normalized.map { it.personId }) {
                save()
            }
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
                val snapshot = synchronized(this@FaceDatabase) {
                    val normalized = assignSequentialPersonIdsLocked(people)
                    people.clear()
                    people.addAll(normalized)
                    rebuildIndexesLocked()
                    normalized
                }
                runCatching { backend.writePeople(snapshot) }
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

    private fun assignSequentialPersonIdsLocked(records: List<PersonRecord>): List<PersonRecord> {
        return records
            .asSequence()
            .filter { it.name.isNotBlank() }
            .sortedWith(compareBy<PersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
            .mapIndexed { index, person ->
                person.copy(
                    personId = (index + 1).toLong(),
                    centroid = person.centroid.copyOf(),
                    embeddings = person.embeddings.map { it.copyOf() }.toMutableList(),
                )
            }
            .toList()
    }

    private fun toJsonObject(): JSONObject {
        val root = JSONObject()
        val arr = JSONArray()
        for (p in people) {
            val obj = JSONObject()
            obj.put("name", p.name)
            obj.put("department", p.department)
            obj.put("grade", p.grade)
            obj.put("centroid", floatArrayToJsonArray(p.centroid))
            val embs = JSONArray()
            for (e in p.embeddings) {
                embs.put(floatArrayToJsonArray(e))
            }
            obj.put("embeddings", embs)
            obj.put("hash", p.hash)
            obj.put("updatedAt", p.updatedAt)
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
                grade = person.grade,
                centroid = person.centroid.copyOf(),
            )
        }
        departmentIndex = people.associate { person -> person.name to person.department }
        gradeIndex = people.associate { person -> person.name to person.grade }
        personIdIndex = people.associateBy { it.personId }
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
        val gradeIndex = indexOfAny(header, listOf("grade", "year", "학년"))
        val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))

        val out = mutableListOf<PersonRecord>()
        for (i in 1 until rows.size) {
            val row = rows[i]
            val name = row.getOrNull(nameIndex)?.trim().orEmpty()
            if (name.isBlank()) continue
            val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
            val grade = row.getOrNull(gradeIndex)?.trim().orEmpty()
            val embeddings = parseEmbeddingsFromCsvCell(row.getOrNull(embeddingsIndex).orEmpty())
            if (embeddings.isEmpty()) continue
            val metadata = normalizeMetadataMap(
                csvMetadataFromRow(rows.first(), row),
                fallbackName = name,
                fallbackDepartment = department,
                fallbackGrade = grade,
            )
            out.add(
                PersonRecord(
                    name = sanitizeName(name),
                    department = sanitizeDepartment(department),
                    grade = sanitizeGrade(grade),
                    centroid = l2Normalize(meanVector(embeddings)),
                    embeddings = embeddings.toMutableList(),
                    hash = personHash(name, department, grade, embeddings),
                    updatedAt = System.currentTimeMillis(),
                    personId = parsePersonId(row.getOrNull(indexOfAny(header, listOf("person_idx"))), name, department, grade),
                    metadata = metadata,
                    metadataHash = metadataHash(metadata),
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
        val metadataHeaders = linkedSetOf<String>()
        people.sortedBy { it.name }.forEach { person ->
            person.metadata.keys.forEach { key -> if (key.isNotBlank()) metadataHeaders += key }
        }
        if (metadataHeaders.none { it.equals("name", ignoreCase = true) }) metadataHeaders.add("name")
        if (metadataHeaders.none { it.equals("department", ignoreCase = true) }) metadataHeaders.add("department")
        if (metadataHeaders.none { it.equals("grade", ignoreCase = true) }) metadataHeaders.add("grade")
        val header = mutableListOf("person_key", "person_idx")
        header.addAll(metadataHeaders)
        header += listOf("embedding_count", "updated_at")
        out.append(header.joinToString(",") { csvEscape(it) }).append('\n')
        val now = System.currentTimeMillis().toString()
        people.sortedBy { it.name }.forEach { person ->
            val row = mutableListOf<String>()
            row += person.personId.toString()
            metadataHeaders.forEach { key ->
                row += person.metadata.entries.firstOrNull { it.key.equals(key, ignoreCase = true) }?.value.orEmpty()
            }
            row += person.embeddings.size.toString()
            row += now
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
        val grade = item.optString("grade", "").trim()

        val embeddingsJson = item.optJSONArray("embeddings") ?: JSONArray()
        val embeddings = mutableListOf<FloatArray>()
        for (j in 0 until embeddingsJson.length()) {
            val embJson = embeddingsJson.optJSONArray(j) ?: continue
            embeddings.add(jsonArrayToFloatArray(embJson))
        }
        if (embeddings.isEmpty()) return null
        val centroid = l2Normalize(meanVector(embeddings))
        val hash = item.optString("hash", "").trim().ifBlank { personHash(name, department, grade, embeddings) }
        val updatedAt = item.optLong("updatedAt", System.currentTimeMillis())
        val metadataJson = item.optJSONObject("metadata")
        val metadata = linkedMapOf<String, String>()
        if (metadataJson != null) {
            val keys = metadataJson.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                metadata[key] = metadataJson.optString(key, "")
            }
        }
        val normalizedMetadata = normalizeMetadataMap(metadata, name, department, grade)
        return PersonRecord(
            name,
            department,
            grade,
            centroid,
            embeddings,
            hash,
            updatedAt,
            parsePersonId(item.optString("personIdx", item.optString("personId", "")), name, department, grade),
            parsePersonKey(item.optString("personKey", ""), name, department, grade),
            normalizedMetadata,
            metadataHash(normalizedMetadata),
        )
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

    private fun sanitizeGrade(raw: String): String {
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

    private fun personHash(name: String, department: String, grade: String = "", embeddings: List<FloatArray>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(name.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(department.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(grade.toByteArray(Charsets.UTF_8))
        digest.update(0)
        updateDigestInt(digest, embeddings.size)
        embeddings.forEach { emb ->
            updateDigestInt(digest, emb.size)
            emb.forEach { value ->
                updateDigestInt(digest, java.lang.Float.floatToIntBits(value))
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
        return personHash(person.name.trim(), person.department.trim(), person.grade.trim(), person.embeddings)
    }

    private data class RankIndexEntry(
        val name: String,
        val department: String,
        val grade: String,
        val centroid: FloatArray,
    )

    private interface StorageBackend {
        fun readPeople(): MutableList<PersonRecord>?
        fun writePeople(people: List<PersonRecord>): Boolean
        fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean
        fun readStructuredMetaBytes(): ByteArray?
        fun readStructuredEmbeddingsBytes(): ByteArray?
        fun needsStructuredSchemaUpgrade(): Boolean
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
                    val people =
                        if (structuredExists) {
                            val base = StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredCodec.mergeMetadataIntoPeople(base, raw) else null
                        } else {
                            StructuredCodec.parseCsvToPeople(raw)
                        }
                    if (people != null && StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)) {
                        csvFile.writeText(StructuredCodec.toCsvFromPeople(people))
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                }
            }

            if (structuredExists) {
                StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { people ->
                    if (StructuredCodec.needsSchemaUpgrade(metaDbFile)) {
                        StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)
                    }
                    return people
                }
            }
            return null
        }

        override fun writePeople(people: List<PersonRecord>): Boolean {
            if (!StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)) return false
            return runCatching {
                csvFile.parentFile?.mkdirs()
                csvFile.writeText(StructuredCodec.toCsvFromPeople(people))
                StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { people ->
                    csvFile.parentFile?.mkdirs()
                    csvFile.writeText(StructuredCodec.toCsvFromPeople(people))
                }
                true
            }.getOrDefault(false)
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

        override fun needsStructuredSchemaUpgrade(): Boolean = StructuredCodec.needsSchemaUpgrade(metaDbFile)
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
                    val people =
                        if (structuredExists) {
                            val base = StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredCodec.mergeMetadataIntoPeople(base, csvRaw) else null
                        } else {
                            StructuredCodec.parseCsvToPeople(csvRaw)
                        }
                    if (people != null && StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)) {
                        syncStructuredToDocuments()
                        writeDocumentText(csvFileName, StructuredCodec.toCsvFromPeople(people))
                        StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                }
            }

            if (structuredExists) {
                StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { people ->
                    if (StructuredCodec.needsSchemaUpgrade(metaDbFile)) {
                        StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)
                        syncStructuredToDocuments()
                    }
                    return people
                }
            }
            return null
        }

        override fun writePeople(people: List<PersonRecord>): Boolean {
            if (!StructuredCodec.saveToStructuredFromPeople(people, metaDbFile, embeddingsBinFile)) return false
            syncStructuredToDocuments()
            return writeDocumentText(csvFileName, StructuredCodec.toCsvFromPeople(people))
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
            val people = StructuredCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile) ?: return true
            return writeDocumentText(csvFileName, StructuredCodec.toCsvFromPeople(people))
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

        override fun needsStructuredSchemaUpgrade(): Boolean = StructuredCodec.needsSchemaUpgrade(metaDbFile)

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

        fun parseCsvToPeople(raw: String): MutableList<PersonRecord>? {
            val rows = parseCsvRows(raw)
            if (rows.isEmpty()) return mutableListOf()
            val header = rows.first().map { it.trim().lowercase(Locale.US) }
            val nameIndex = indexOfAny(header, listOf("name", "이름"))
            if (nameIndex < 0) return null
            val departmentIndex = indexOfAny(header, listOf("department", "dept", "folder", "부서"))
            val gradeIndex = indexOfAny(header, listOf("grade", "year", "학년"))
            val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))
            val hashIndex = indexOfAny(header, listOf("hash", "record_hash"))
            val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))
            val out = mutableListOf<PersonRecord>()
            for (i in 1 until rows.size) {
                val row = rows[i]
                val name = row.getOrNull(nameIndex)?.trim().orEmpty()
                if (name.isBlank()) continue
                val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
                val grade = row.getOrNull(gradeIndex)?.trim().orEmpty()
                val embeddings = parseEmbeddingsList(row.getOrNull(embeddingsIndex).orEmpty())
                if (embeddings.isEmpty()) continue
                val hash = row.getOrNull(hashIndex)?.trim().orEmpty().ifBlank {
                    computeFloatHash(name, department, grade, embeddings)
                }
                val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: System.currentTimeMillis()
                out += PersonRecord(
                    name = name,
                    department = department,
                    grade = grade,
                    centroid = l2NormalizeFloat(meanVectorFloat(embeddings)),
                    embeddings = embeddings.toMutableList(),
                    hash = hash,
                    updatedAt = updatedAt,
                )
            }
            return out
        }

        fun mergeMetadataIntoPeople(
            currentPeople: List<PersonRecord>,
            csvRaw: String,
        ): MutableList<PersonRecord> {
            val rows = parseMetadataRows(csvRaw)
            if (rows.isEmpty()) return mutableListOf()
            val byPersonKey = currentPeople.associateBy { it.personKey }
            val byPersonId = currentPeople.associateBy { it.personId }
            val byIdentity = currentPeople.associateBy { metadataIdentityKey(it.metadata) }
            val byUniqueName = currentPeople
                .groupBy { metadataName(it.metadata, it.name).trim().lowercase(Locale.US) }
                .filterValues { it.size == 1 }
                .mapValues { it.value.first() }
            return rows.mapNotNull { row ->
                val baseByPersonKey = byPersonKey[row.personKey]
                val baseByPersonId = byPersonId[row.personId]
                val baseByIdentity = byIdentity[metadataIdentityKey(row.metadata)]
                val baseByName = byUniqueName[row.name.trim().lowercase(Locale.US)]
                val base = when {
                    baseByPersonKey != null -> baseByPersonKey
                    baseByPersonId != null && baseByName != null &&
                        !metadataName(baseByPersonId.metadata, baseByPersonId.name).equals(row.name, ignoreCase = true) -> baseByName
                    baseByPersonId != null && metadataIdentityKey(baseByPersonId.metadata) == metadataIdentityKey(row.metadata) -> baseByPersonId
                    else -> baseByIdentity ?: baseByName
                } ?: return@mapNotNull null
                base.copy(
                    name = row.name,
                    department = row.department,
                    grade = row.grade,
                    centroid = base.centroid.copyOf(),
                    embeddings = base.embeddings.map { it.copyOf() }.toMutableList(),
                    hash = computeFloatHash(row.name, row.department, row.grade, base.embeddings),
                    updatedAt = row.updatedAt,
                    personId = row.personId,
                    personKey = row.personKey,
                    metadata = row.metadata,
                    metadataHash = row.metadataHash,
                )
            }.toMutableList()
        }

        fun hasEmbeddingsColumn(raw: String): Boolean {
            val rows = parseCsvRows(raw)
            if (rows.isEmpty()) return false
            val header = rows.first().map { it.trim().lowercase(Locale.US) }
            return indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json")) >= 0
        }

        fun needsSchemaUpgrade(metaDbFile: File): Boolean {
            if (!metaDbFile.exists()) return false
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return false
            return runCatching {
                !(hasColumn(db, TABLE, "person_key") && hasColumn(db, TABLE, "person_idx") && hasColumn(db, TABLE, "metadata_blob") && hasColumn(db, TABLE, "metadata_hash"))
            }.getOrDefault(false).also { db.close() }
        }

        fun toCsv(root: JSONObject): String {
            val people = normalizeRoot(root).optJSONArray("people") ?: JSONArray()
            val out = StringBuilder()
            out.append("name,department,grade,embedding_count,updated_at\n")
            val now = System.currentTimeMillis()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                val department = person.optString("department", person.optString("folder", "")).trim()
                val grade = person.optString("grade", "").trim()
                val embeddings = normalizeEmbeddings(person.optJSONArray("embeddings") ?: JSONArray())
                val updatedAt = person.optLong("updatedAt", now)
                val row = listOf(
                    name,
                    department,
                    grade,
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
                obj.put("grade", row.grade)
                obj.put("updatedAt", row.updatedAt)
                val embeddings = normalizeEmbeddings(obj.optJSONArray("embeddings") ?: JSONArray())
                obj.put("embeddings", embeddings)
                obj.put("hash", computeHash(row.name, row.department, row.grade, embeddings))
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
                val hasGrade = hasColumn(db, TABLE, "grade")
                val hasPersonId = hasColumn(db, TABLE, "person_idx")
                val hasMetadataBlob = hasColumn(db, TABLE, "metadata_blob")
                val hasMetadataHash = hasColumn(db, TABLE, "metadata_hash")
                val projection = if (hasGrade) {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("grade")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                } else {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                }
                db.query(
                    TABLE,
                    projection,
                    null,
                    null,
                    null,
                    null,
                    "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val personId = cursor.getString(0)?.trim().orEmpty()
                        val name = cursor.getString(1)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(2)?.trim().orEmpty()
                        val grade = if (hasGrade) cursor.getString(3)?.trim().orEmpty() else ""
                        val base = if (hasGrade) 0 else -1
                        val offset = cursor.getLong(4 + base).toInt()
                        val length = cursor.getLong(5 + base).toInt()
                        val count = cursor.getInt(6 + base).coerceAtLeast(0)
                        val dim = cursor.getInt(7 + base).coerceAtLeast(0)
                        val hash = cursor.getString(8 + base)?.trim().orEmpty()
                        val updatedAt = cursor.getLong(9 + base)
                        val metadataBlob = if (hasMetadataBlob) cursor.getBlob(10 + base) else null
                        val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
                        val embeddings = decodeEmbeddings(embeddingsBytes, offset, length, count, dim)
                        val obj = JSONObject()
                            .put("name", name)
                            .put("department", department)
                            .put("grade", grade)
                            .put("embeddings", embeddings)
                            .put("personId", parsePersonId(personId, name, department, grade))
                            .put("metadata", JSONObject(metadata as Map<*, *>))
                        if (updatedAt > 0L) obj.put("updatedAt", updatedAt)
                        obj.put("hash", if (hash.isNotBlank()) hash else computeHash(name, department, grade, embeddings))
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
                val hasGrade = hasColumn(db, TABLE, "grade")
                val hasPersonId = hasColumn(db, TABLE, "person_idx")
                val hasMetadataBlob = hasColumn(db, TABLE, "metadata_blob")
                val hasMetadataHash = hasColumn(db, TABLE, "metadata_hash")
                val projection = if (hasGrade) {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("grade")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                } else {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                }
                db.query(
                    TABLE,
                    projection,
                    null,
                    null,
                    null,
                    null,
                    "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val personId = cursor.getString(0)?.trim().orEmpty()
                        val name = cursor.getString(1)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(2)?.trim().orEmpty()
                        val grade = if (hasGrade) cursor.getString(3)?.trim().orEmpty() else ""
                        val base = if (hasGrade) 0 else -1
                        val offset = cursor.getLong(4 + base).toInt()
                        val length = cursor.getLong(5 + base).toInt()
                        val count = cursor.getInt(6 + base).coerceAtLeast(0)
                        val dim = cursor.getInt(7 + base).coerceAtLeast(0)
                        val embeddings = decodeEmbeddingsToFloatArrays(embeddingsBytes, offset, length, count, dim)
                        if (embeddings.isEmpty()) continue
                        val metadataBlob = if (hasMetadataBlob) cursor.getBlob(10 + base) else null
                        val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
                        out.add(
                            PersonRecord(
                                name = name,
                                department = department,
                                grade = grade,
                                centroid = l2NormalizeFloat(meanVectorFloat(embeddings)),
                                embeddings = embeddings.toMutableList(),
                                hash = cursor.getString(8 + base)?.trim().orEmpty().ifBlank {
                                    computeFloatHash(name, department, grade, embeddings)
                                },
                                updatedAt = cursor.getLong(9 + base),
                                personId = parsePersonId(personId, name, department, grade),
                                metadata = metadata,
                                metadataHash = if (hasMetadataHash) {
                                    cursor.getString(11 + base)?.trim().orEmpty().ifBlank { metadataHash(metadata) }
                                } else {
                                    metadataHash(metadata)
                                },
                            ),
                        )
                    }
                }
                out
            }.getOrNull().also { db.close() }
        }

        fun saveToStructuredFromPeople(
            people: List<PersonRecord>,
            metaDbFile: File,
            embeddingsBinFile: File,
        ): Boolean {
            val sorted = people
                .sortedWith(compareBy<PersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
                .mapIndexed { index, person -> person.copy(personId = (index + 1).toLong()) }
            val rows = mutableListOf<RowMeta>()
            val out = ByteArrayOutputStream()
            sorted.forEach { person ->
                val offset = out.size()
                val count = person.embeddings.size
                val dim = person.embeddings.firstOrNull()?.size ?: 0
                person.embeddings.forEach { emb ->
                    for (j in 0 until dim) {
                        writeFloatLe(out, if (j < emb.size) emb[j] else 0f)
                    }
                }
                val length = out.size() - offset
                rows.add(
                    RowMeta(
                        personId = person.personId,
                        personKey = person.personKey,
                        name = person.name,
                        department = person.department,
                        grade = person.grade,
                        offsetBytes = offset,
                        lengthBytes = length,
                        embeddingCount = count,
                        embeddingDim = dim,
                        hash = person.hash,
                        updatedAt = person.updatedAt,
                        metadataBlob = encodeMetadataBlob(person.metadata),
                        metadataHash = person.metadataHash,
                    ),
                )
            }
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                embeddingsBinFile.writeBytes(out.toByteArray())
                val db = SQLiteDatabase.openOrCreateDatabase(metaDbFile, null)
                recreatePeopleTable(db)
                db.beginTransaction()
                try {
                    db.delete(TABLE, null, null)
                    rows.forEach { row ->
                        db.insert(
                            TABLE,
                            null,
                            ContentValues().apply {
                                put("person_key", row.personKey)
                                put("person_idx", row.personId)
                                put("name", row.name)
                                put("department", row.department)
                                put("grade", row.grade)
                                put("offset_bytes", row.offsetBytes)
                                put("length_bytes", row.lengthBytes)
                                put("embedding_count", row.embeddingCount)
                                put("embedding_dim", row.embeddingDim)
                                put("hash", row.hash)
                                put("updated_at", row.updatedAt)
                                put("metadata_blob", row.metadataBlob)
                                put("metadata_hash", row.metadataHash)
                            },
                        )
                    }
                    db.setTransactionSuccessful()
                } finally {
                    db.endTransaction()
                    db.close()
                }
                true
            }.getOrDefault(false)
        }

        fun toCsvFromPeople(people: List<PersonRecord>): String {
            val out = StringBuilder()
            val metadataHeaders = linkedSetOf<String>()
            val sortedPeople = people.sortedWith(compareBy<PersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
            sortedPeople.forEach { person ->
                person.metadata.keys.forEach { key -> if (key.isNotBlank()) metadataHeaders += key }
            }
            if (metadataHeaders.none { it.equals("name", ignoreCase = true) }) metadataHeaders.add("name")
            if (metadataHeaders.none { it.equals("department", ignoreCase = true) }) metadataHeaders.add("department")
            if (metadataHeaders.none { it.equals("grade", ignoreCase = true) }) metadataHeaders.add("grade")
            val header = mutableListOf("person_idx")
            header.addAll(metadataHeaders)
            header += listOf("embedding_count", "updated_at", "person_key")
            out.append(header.joinToString(",") { csvEscape(it) }).append('\n')
            val now = System.currentTimeMillis()
            sortedPeople.forEachIndexed { index, person ->
                val row = mutableListOf<String>()
                row += (index + 1).toString()
                metadataHeaders.forEach { key ->
                    row += person.metadata.entries.firstOrNull { it.key.equals(key, ignoreCase = true) }?.value.orEmpty()
                }
                row += person.embeddings.size.toString()
                row += if (person.updatedAt > 0L) person.updatedAt.toString() else now.toString()
                row += person.personKey
                out.append(row.joinToString(",") { csvEscape(it) }).append('\n')
            }
            return out.toString()
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
                val grade = person.optString("grade", "").trim()
                val embeddings = normalizeEmbeddings(person.optJSONArray("embeddings") ?: JSONArray())
                if (embeddings.length() <= 0) continue
                val updatedAt = person.optLong("updatedAt", System.currentTimeMillis())
                val hash = person.optString("hash", "").trim().ifBlank { computeHash(name, department, grade, embeddings) }
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
                records.add(
                    Record(
                        personId = parsePersonId(person.optString("personIdx", person.optString("personId", "")), name, department, grade),
                        personKey = parsePersonKey(person.optString("personKey", ""), name, department, grade),
                        name = name,
                        department = department,
                        grade = grade,
                        embeddings = embeddings,
                        hash = hash,
                        updatedAt = updatedAt,
                        metadataBlob = encodeMetadataBlob(normalizedMetadata),
                        metadataHash = metadataHash(normalizedMetadata),
                    ),
                )
            }
            val encoded = encodeEmbeddings(records)
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                embeddingsBinFile.writeBytes(encoded.bytes)
                val db = SQLiteDatabase.openOrCreateDatabase(metaDbFile, null)
                recreatePeopleTable(db)
                db.beginTransaction()
                try {
                    db.delete(TABLE, null, null)
                    encoded.rows.forEach { row ->
                        val values = ContentValues().apply {
                            put("person_key", row.personKey)
                            put("person_idx", row.personId)
                            put("name", row.name)
                            put("department", row.department)
                            put("grade", row.grade)
                            put("offset_bytes", row.offsetBytes)
                            put("length_bytes", row.lengthBytes)
                            put("embedding_count", row.embeddingCount)
                            put("embedding_dim", row.embeddingDim)
                            put("hash", row.hash)
                            put("updated_at", row.updatedAt)
                            put("metadata_blob", row.metadataBlob)
                            put("metadata_hash", row.metadataHash)
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
            val gradeIndex = indexOfAny(header, listOf("grade", "year", "학년"))
            val embeddingsIndex = indexOfAny(header, listOf("embeddings_json", "embeddings", "embedding_json"))
            val hashIndex = indexOfAny(header, listOf("hash", "record_hash"))
            val updatedAtIndex = indexOfAny(header, listOf("updated_at", "updatedat"))
            val arr = JSONArray()
            for (i in 1 until rows.size) {
                val row = rows[i]
                val name = row.getOrNull(nameIndex)?.trim().orEmpty()
                if (name.isBlank()) continue
                val department = row.getOrNull(departmentIndex)?.trim().orEmpty()
                val grade = row.getOrNull(gradeIndex)?.trim().orEmpty()
                val embeddings = runCatching { JSONArray(row.getOrNull(embeddingsIndex).orEmpty()) }.getOrNull() ?: JSONArray()
                if (embeddings.length() <= 0) continue
                val hash = row.getOrNull(hashIndex)?.trim().orEmpty()
                val updatedAt = row.getOrNull(updatedAtIndex)?.trim()?.toLongOrNull() ?: System.currentTimeMillis()
                val obj = JSONObject()
                    .put("name", name)
                    .put("department", department)
                    .put("grade", grade)
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

        private fun parseEmbeddingsList(raw: String): List<FloatArray> {
            if (raw.isBlank()) return emptyList()
            val parsed = runCatching { JSONArray(raw) }.getOrNull() ?: return emptyList()
            val out = mutableListOf<FloatArray>()
            for (i in 0 until parsed.length()) {
                val emb = parsed.optJSONArray(i) ?: continue
                val vec = FloatArray(emb.length())
                for (j in 0 until emb.length()) {
                    vec[j] = emb.optDouble(j, 0.0).toFloat()
                }
                out += vec
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
                rows.add(
                    RowMeta(
                        personId = record.personId,
                        personKey = record.personKey,
                        name = record.name,
                        department = record.department,
                        grade = record.grade,
                        offsetBytes = offset,
                        lengthBytes = length,
                        embeddingCount = count,
                        embeddingDim = dim,
                        hash = record.hash,
                        updatedAt = record.updatedAt,
                        metadataBlob = record.metadataBlob,
                        metadataHash = record.metadataHash,
                    ),
                )
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

        private fun ensurePeopleTableSchema(db: SQLiteDatabase) {
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN grade TEXT DEFAULT ''") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_key TEXT") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_idx INTEGER") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_id INTEGER") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN metadata_blob BLOB") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN metadata_hash TEXT") }
        }

        private fun recreatePeopleTable(db: SQLiteDatabase) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE")
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS $TABLE(" +
                    "person_key TEXT PRIMARY KEY,person_idx INTEGER," +
                    "name TEXT," +
                    "department TEXT," +
                    "grade TEXT," +
                    "offset_bytes INTEGER," +
                    "length_bytes INTEGER," +
                    "embedding_count INTEGER," +
                    "embedding_dim INTEGER," +
                    "hash TEXT," +
                    "updated_at INTEGER," +
                    "metadata_blob BLOB," +
                    "metadata_hash TEXT" +
                    ")",
            )
        }

        private fun hasColumn(db: SQLiteDatabase, table: String, column: String): Boolean {
            return runCatching {
                db.rawQuery("PRAGMA table_info($table)", null).use { cursor ->
                    val nameIndex = cursor.getColumnIndex("name")
                    while (cursor.moveToNext()) {
                        if (nameIndex >= 0 && cursor.getString(nameIndex).equals(column, ignoreCase = true)) {
                            return@use true
                        }
                    }
                    false
                }
            }.getOrDefault(false)
        }

        private fun computeHash(name: String, department: String, grade: String, embeddings: JSONArray): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(grade.toByteArray(Charsets.UTF_8))
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

        private fun computeFloatHash(name: String, department: String, grade: String, embeddings: List<FloatArray>): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(grade.toByteArray(Charsets.UTF_8))
            digest.update(0)
            updateDigestInt(digest, embeddings.size)
            embeddings.forEach { emb ->
                updateDigestInt(digest, emb.size)
                emb.forEach { value ->
                    updateDigestInt(digest, java.lang.Float.floatToIntBits(value))
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
            val personId: Long,
            val personKey: String,
            val name: String,
            val department: String,
            val grade: String,
            val embeddings: JSONArray,
            val hash: String,
            val updatedAt: Long,
            val metadataBlob: ByteArray,
            val metadataHash: String,
        )

        private data class CsvMetaRow(
            val personId: Long,
            val personKey: String,
            val name: String,
            val department: String,
            val grade: String,
            val updatedAt: Long,
            val metadata: LinkedHashMap<String, String>,
            val metadataHash: String,
        )

        private data class RowMeta(
            val personId: Long,
            val personKey: String,
            val name: String,
            val department: String,
            val grade: String,
            val offsetBytes: Int,
            val lengthBytes: Int,
            val embeddingCount: Int,
            val embeddingDim: Int,
            val hash: String,
            val updatedAt: Long,
            val metadataBlob: ByteArray,
            val metadataHash: String,
        )

        private data class EncodedData(
            val rows: List<RowMeta>,
            val bytes: ByteArray,
        )
    }

    private object DeltaCodec {
        fun encode(upserts: List<PersonRecord>, deletes: List<String>): ByteArray {
            val out = ByteArrayOutputStream()
            out.write(byteArrayOf('F'.code.toByte(), 'D'.code.toByte(), 'D'.code.toByte(), '1'.code.toByte()))
            writeIntLe(out, deletes.size)
            deletes.sorted().forEach { personKey -> writeString(out, personKey) }
            writeIntLe(out, upserts.size)
            upserts.sortedBy { it.personKey }.forEach { person ->
                writeString(out, person.personKey)
                writeString(out, person.personId.toString())
                writeString(out, person.name)
                writeString(out, person.department)
                writeString(out, person.grade)
                writeString(out, person.hash)
                writeLongLe(out, person.updatedAt)
                val metadataBlob = encodeMetadataBlob(person.metadata)
                writeIntLe(out, metadataBlob.size)
                out.write(metadataBlob)
                writeString(out, person.metadataHash)
                val embCount = person.embeddings.size
                val dim = person.embeddings.firstOrNull()?.size ?: 0
                writeIntLe(out, embCount)
                writeIntLe(out, dim)
                person.embeddings.forEach { emb ->
                    emb.forEach { value -> writeFloatLe(out, value) }
                }
            }
            return out.toByteArray()
        }

        fun decode(bytes: ByteArray): DeltaPayload? {
            if (bytes.size < 4) return null
            if (bytes[0] != 'F'.code.toByte() || bytes[1] != 'D'.code.toByte() || bytes[2] != 'D'.code.toByte() || bytes[3] != '1'.code.toByte()) return null
            val cursor = Cursor(bytes, 4)
            val deleteCount = cursor.readInt() ?: return null
            val deletes = MutableList(deleteCount.coerceAtLeast(0)) { cursor.readString().orEmpty() }.filter { it.isNotBlank() }
            val upsertCount = cursor.readInt() ?: return null
            val upserts = mutableListOf<PersonRecord>()
            repeat(upsertCount.coerceAtLeast(0)) {
                val personKeyRaw = cursor.readString()
                val personId = parsePersonId(cursor.readString())
                val name = cursor.readString() ?: return@repeat
                val department = cursor.readString() ?: return@repeat
                val grade = cursor.readString() ?: return@repeat
                val hash = cursor.readString() ?: return@repeat
                val updatedAt = cursor.readLong() ?: return@repeat
                val metadataBlobLength = cursor.readInt() ?: return@repeat
                val metadataBlob = cursor.readBytes(metadataBlobLength) ?: return@repeat
                val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
                val metadataHash = cursor.readString() ?: metadataHash(metadata)
                val embCount = cursor.readInt() ?: return@repeat
                val dim = cursor.readInt() ?: return@repeat
                val embeddings = mutableListOf<FloatArray>()
                repeat(embCount.coerceAtLeast(0)) {
                    val emb = FloatArray(dim.coerceAtLeast(0))
                    for (i in emb.indices) {
                        emb[i] = cursor.readFloat() ?: return@repeat
                    }
                    embeddings += emb
                }
                if (name.isBlank() || embeddings.isEmpty()) return@repeat
                upserts += PersonRecord(
                    name = name,
                    department = department,
                    grade = grade,
                    centroid = normalize(mean(embeddings)),
                    embeddings = embeddings,
                    hash = hash.ifBlank { computeHash(name, department, grade, embeddings) },
                    updatedAt = updatedAt,
                    personId = if (personId > 0L) personId else personIdFromIdentity(name, department, grade),
                    personKey = parsePersonKey(personKeyRaw, name, department, grade),
                    metadata = metadata,
                    metadataHash = metadataHash.ifBlank { metadataHash(metadata) },
                )
            }
            return DeltaPayload(upserts, deletes)
        }

        private fun writeString(out: ByteArrayOutputStream, value: String) {
            val bytes = value.toByteArray(Charsets.UTF_8)
            writeIntLe(out, bytes.size)
            out.write(bytes)
        }

        private fun writeIntLe(out: ByteArrayOutputStream, value: Int) {
            out.write(value and 0xFF)
            out.write((value ushr 8) and 0xFF)
            out.write((value ushr 16) and 0xFF)
            out.write((value ushr 24) and 0xFF)
        }

        private fun writeLongLe(out: ByteArrayOutputStream, value: Long) {
            for (shift in 0 until 64 step 8) {
                out.write(((value ushr shift) and 0xFF).toInt())
            }
        }

        private fun writeFloatLe(out: ByteArrayOutputStream, value: Float) {
            writeIntLe(out, java.lang.Float.floatToIntBits(value))
        }

        private fun mean(vectors: List<FloatArray>): FloatArray {
            val dim = vectors.firstOrNull()?.size ?: 0
            val out = FloatArray(dim)
            if (dim == 0) return out
            vectors.forEach { vector ->
                for (i in 0 until dim) out[i] += vector[i]
            }
            for (i in 0 until dim) out[i] /= vectors.size.toFloat()
            return out
        }

        private fun normalize(v: FloatArray): FloatArray {
            var sum = 0.0
            v.forEach { sum += (it * it).toDouble() }
            val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
            return FloatArray(v.size) { idx -> v[idx] / norm }
        }

        private fun computeHash(name: String, department: String, grade: String, embeddings: List<FloatArray>): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(grade.toByteArray(Charsets.UTF_8))
            digest.update(0)
            writeDigestInt(digest, embeddings.size)
            embeddings.forEach { emb ->
                writeDigestInt(digest, emb.size)
                emb.forEach { value ->
                    writeDigestInt(digest, java.lang.Float.floatToIntBits(value))
                }
            }
            return digest.digest().joinToString("") { b -> "%02x".format(b) }
        }

        private fun writeDigestInt(digest: MessageDigest, value: Int) {
            digest.update(((value ushr 24) and 0xFF).toByte())
            digest.update(((value ushr 16) and 0xFF).toByte())
            digest.update(((value ushr 8) and 0xFF).toByte())
            digest.update((value and 0xFF).toByte())
        }

        private class Cursor(private val bytes: ByteArray, private var offset: Int) {
            fun readInt(): Int? {
                if (offset + 4 > bytes.size) return null
                val value =
                    (bytes[offset].toInt() and 0xFF) or
                        ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                        ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                        ((bytes[offset + 3].toInt() and 0xFF) shl 24)
                offset += 4
                return value
            }

            fun readLong(): Long? {
                if (offset + 8 > bytes.size) return null
                var value = 0L
                for (i in 0 until 8) {
                    value = value or ((bytes[offset + i].toLong() and 0xFFL) shl (8 * i))
                }
                offset += 8
                return value
            }

            fun readFloat(): Float? {
                return readInt()?.let { Float.fromBits(it) }
            }

            fun readString(): String? {
                val size = readInt() ?: return null
                if (size < 0 || offset + size > bytes.size) return null
                val value = bytes.copyOfRange(offset, offset + size).toString(Charsets.UTF_8)
                offset += size
                return value
            }

            fun readBytes(size: Int): ByteArray? {
                if (size < 0 || offset + size > bytes.size) return null
                val value = bytes.copyOfRange(offset, offset + size)
                offset += size
                return value
            }
        }

        data class DeltaPayload(
            val upserts: List<PersonRecord>,
            val deletes: List<String>,
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
