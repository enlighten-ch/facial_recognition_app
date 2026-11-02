package com.allstar.faithid.server.people

import com.allstar.faithid.core.model.EmbeddingSets
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.PersonNames
import com.allstar.faithid.core.storage.CsvMerge
import com.allstar.faithid.core.storage.PeopleCsv
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.ChangeSet
import com.allstar.faithid.core.sync.ColumnLabels
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.OpsResponse
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.Tombstone
import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerLog
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction
import java.util.UUID
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * The server's authoritative people list.
 *
 * Every change gets the next revision number; kiosks ask for "everything after revision N". Kiosk
 * operations only add (register, add samples) and merge by set union, so operations from many
 * kiosks can arrive in any order, be retried, or be duplicated without losing or corrupting data.
 * Deletions leave a tombstone, so a kiosk that was offline cannot bring a deleted person back.
 *
 * All methods are synchronized; each change is persisted in one SQLite transaction before it
 * becomes visible.
 */
class PeopleRepository(
    private val store: PeopleStore,
    private val maxEmbeddingsPerPerson: Int,
    private val clock: Clock,
    private val log: ServerLog,
) {
    private val people = LinkedHashMap<String, Person>()
    private val tombstones = LinkedHashMap<String, Tombstone>()

    /** Writers queue fairly, so an administrator's edit is not starved by a busy kiosk fleet. */
    private val lock = ReentrantLock(true)
    private val csvLock = Any()

    /** Immutable view published after every commit; readers never wait for writers. */
    private class Snapshot(
        val people: Map<String, Person>,
        val tombstones: Map<String, Tombstone>,
        val epoch: String,
        val revision: Long,
        val displayNumbers: Map<String, Long>,
        val labels: Map<String, String>,
    )

    @Volatile
    private var snapshot = Snapshot(emptyMap(), emptyMap(), "", 0, emptyMap(), emptyMap())

    /** Names of columns set by the administrator ("department" → "반"); the data keeps its column. */
    private var labels: Map<String, String> = emptyMap()

    /**
     * Fingerprint of the face_db.csv the server wrote last; other content means someone edited it.
     * The file's time and size alone are not enough: shared storage may report a stale time for a
     * file just replaced. They only decide when the file needs reading again.
     */
    @Volatile
    private var lastWrittenHash: String? = null
    @Volatile
    private var checked: Pair<Pair<Long, Long>, Boolean>? = null
    private var editLogged = false

    private var currentEpoch: String = ""
    private var currentRevision: Long = 0

    val epoch: String get() = snapshot.epoch
    val revision: Long get() = snapshot.revision

    /** Number of deletion records kept so offline kiosks cannot bring deleted people back. */
    val deletedCount: Int get() = snapshot.tombstones.size

    /** Called after every committed change with the new revision. */
    @Volatile
    var onChanged: (Long) -> Unit = {}

    /** Called (with a reason) before changes that touch many people at once, to back up the database first. */
    @Volatile
    var beforeBulkChange: (String) -> Unit = {}

    /**
     * True while face_db.csv has edits not applied yet; the server then leaves the file alone.
     * Looks at the file's time and size, so a screen showing it is current without any polling.
     */
    val csvEditPending: Boolean
        get() = editedOnDisk(File(store.directory, PeopleStore.CSV_FILE))

    private fun editedOnDisk(file: File): Boolean {
        val written = lastWrittenHash ?: return false
        if (!file.exists()) return false
        val stamp = file.lastModified() to file.length()
        checked?.let { (seen, edited) -> if (seen == stamp) return edited }
        val edited = runCatching { sha256(file.readBytes()) != written }.getOrDefault(false)
        checked = stamp to edited
        return edited
    }

    val columnLabels: Map<String, String> get() = snapshot.labels

    init {
        lock.withLock { loadFromStore(newEpoch = false) }
    }

    fun list(): List<Person> = snapshot.people.values.toList()

    fun get(key: String): Person? = snapshot.people[key]

    fun isDeleted(key: String): Boolean = key in snapshot.tombstones

    fun displayNumbers(): Map<String, Long> = snapshot.displayNumbers

    fun changesSince(since: Long, clientEpoch: String?): ChangeSet {
        val view = snapshot
        val full = clientEpoch != view.epoch || since <= 0 || since > view.revision
        return if (full) {
            ChangeSet(view.epoch, view.revision, fullSnapshot = true, persons = view.people.values.toList(), tombstones = view.tombstones.values.toList())
        } else {
            ChangeSet(
                view.epoch,
                view.revision,
                fullSnapshot = false,
                persons = view.people.values.filter { it.revision > since },
                tombstones = view.tombstones.values.filter { it.revision > since },
            )
        }
    }

    /** Applies kiosk operations in order. Returns one verdict per operation id. */
    fun applyOps(ops: List<PersonOp>, clientId: String): OpsResponse {
        val (response, changed) = lock.withLock {
            val results = LinkedHashMap<String, OpResult>()
            val before = currentRevision
            atomically {
                ops.forEach { op -> results[op.opId] = applyOne(op) }
                store.setMeta(META_REVISION, currentRevision.toString())
            }
            publish()
            OpsResponse(currentRevision, results) to (currentRevision != before)
        }
        if (changed) {
            log.log("kiosk changes applied: client=$clientId, ops=${ops.size}, revision=${response.serverRevision}")
            onChanged(response.serverRevision)
        }
        return response
    }

    fun updateMetadata(key: String, name: String, department: String, grade: String): Boolean =
        updateMetadata(key, linkedMapOf(PersonMetadata.CANONICAL_NAME to name, PersonMetadata.CANONICAL_DEPARTMENT to department, PersonMetadata.CANONICAL_GRADE to grade))

    /** Sets the given columns (any CSV column, "부서" and "department" alike); the others stay as they are. */
    fun updateMetadata(key: String, values: Map<String, String>): Boolean {
        val revisionAfter = lock.withLock {
            val person = people[key] ?: return false
            val metadata = values.entries.fold(person.metadata) { m, (column, value) -> m.withValue(column, value.trim()) }
            if (metadata == person.metadata) return true
            atomically { commit(person.copy(metadata = metadata, revision = nextRevision(), updatedAt = clock.now())) }
            publish()
            currentRevision
        }
        log.log("person edited: key=$key, name=${people[key]?.name}, revision=$revisionAfter")
        onChanged(revisionAfter)
        return true
    }

    fun delete(key: String): Boolean {
        val revisionAfter = lock.withLock {
            val person = people[key] ?: return false
            atomically {
                val tombstone = Tombstone(key, nextRevision())
                store.putTombstone(tombstone, clock.now())
                store.setMeta(META_REVISION, currentRevision.toString())
                people.remove(key)
                tombstones[key] = tombstone
            }
            publish()
            log.log("person deleted: key=$key, name=${person.name}, revision=$currentRevision")
            currentRevision
        }
        onChanged(revisionAfter)
        return true
    }

    /**
     * Names a column ("department" → "반"; blank = back to its own name). Only the name changes:
     * the data stays in the column the kiosks fill in. Kiosks get the new name with the next download.
     */
    fun setColumnLabel(column: String, label: String?) {
        val key = PersonMetadata.canonicalColumn(column)
        val name = label?.trim()?.takeIf { it.isNotEmpty() && it != key }
        val revisionAfter = lock.withLock {
            val next = if (name == null) labels - key else labels + (key to name)
            if (next == labels) return
            atomically {
                nextRevision()
                saveLabels(next)
                store.setMeta(META_REVISION, currentRevision.toString())
            }
            publish()
            currentRevision
        }
        log.log("column named: $key=${name ?: "(default)"}, revision=$revisionAfter")
        onChanged(revisionAfter)
    }

    /** What reading face_db.csv did. */
    data class CsvResult(val changes: CsvMerge.Changes, val changedPeople: Int, val found: Boolean)

    /**
     * Reads face_db.csv and applies what was edited in it since the server wrote it (see
     * [CsvMerge]), then writes it again. Called when the server starts and by "지금 반영".
     */
    fun applyCsvFile(): CsvResult = synchronized(csvLock) {
        val file = File(store.directory, PeopleStore.CSV_FILE)
        val text = if (file.exists()) runCatching { readCsv(file.readBytes()) }.getOrNull() else null
        val result = if (text == null) CsvResult(CsvMerge.Changes(), 0, found = false) else applyCsvText(text)
        writeCsv(file)
        result
    }

    /** Applies edits of a CSV text against the copy the server wrote last. */
    fun applyCsvText(text: String): CsvResult {
        val (result, revisionAfter) = lock.withLock {
            // Compared with the last copy written; earlier copies tell which one a spreadsheet may
            // have kept open meanwhile, so people registered since are not taken as deleted.
            val bases = csvBases().ifEmpty { listOf(PeopleCsv.exportTable(people.values.toList(), store.displayNumbers(), labels).let { CsvMerge.Base(it.text, it.columns) }) }
            if (bases.first().text == text) return CsvResult(CsvMerge.Changes(), 0, found = true)
            val changes = CsvMerge.diff(bases.first(), text, people.values.toList(), store.displayNumbers(), labels, older = bases.drop(1))
            if (changes.isEmpty) return CsvResult(changes, 0, found = true).also { logCsv(it) }
            beforeBulkChange("csv")
            var changed = 0
            val now = clock.now()
            atomically {
                val rev = nextRevision()
                people.values.toList().forEach { person ->
                    if (person.key in changes.deleted) return@forEach
                    var metadata = person.metadata
                    changes.removedColumns.forEach { metadata = metadata.without(it) }
                    changes.values[person.key]?.let { edits ->
                        // The name's parts (family_name, given_name) together set the name.
                        val family = edits[PersonMetadata.CANONICAL_FAMILY_NAME]
                        val given = edits[PersonMetadata.CSV_GIVEN_NAME]
                        if (family != null || given != null) {
                            metadata = metadata.withName(family ?: PersonNames.family(metadata), given ?: PersonNames.given(metadata))
                        }
                        edits.filterKeys { it != PersonMetadata.CANONICAL_FAMILY_NAME && it != PersonMetadata.CSV_GIVEN_NAME }
                            .forEach { (column, value) -> metadata = metadata.withValue(column, value) }
                    }
                    if (metadata != person.metadata && metadata.name.isNotBlank()) {
                        commit(person.copy(metadata = metadata, revision = rev, updatedAt = now))
                        changed++
                    }
                }
                changes.deleted.forEach { key ->
                    if (people.remove(key) != null) {
                        val tombstone = Tombstone(key, rev)
                        store.putTombstone(tombstone, now)
                        tombstones[key] = tombstone
                    }
                }
                // A name equal to the column's own key means "back to the default".
                if (changes.renamed.isNotEmpty()) saveLabels((labels + changes.renamed).filter { (key, label) -> !label.equals(key, ignoreCase = true) })
                store.setMeta(META_REVISION, currentRevision.toString())
            }
            publish()
            CsvResult(changes, changed, found = true) to currentRevision
        }
        logCsv(result)
        onChanged(revisionAfter)
        return result
    }

    /**
     * Writes `face_db.csv` next to the database (metadata only, for viewing and editing in Excel).
     * When the file was edited since the server wrote it, it is left alone until the edits are
     * applied ([applyCsvFile]), unless [force]. Serialized: exports share one temporary file.
     */
    fun exportCsv(force: Boolean = false): File? = synchronized(csvLock) {
        val file = File(store.directory, PeopleStore.CSV_FILE)
        if (!force && editedOnDisk(file)) {
            if (!editLogged) log.log("face_db.csv was edited: kept until the server starts again or the edits are applied")
            editLogged = true
            return null
        }
        writeCsv(file)
        file
    }

    private fun writeCsv(file: File) {
        val view = snapshot
        val export = PeopleCsv.exportTable(view.people.values.toList(), view.displayNumbers, view.labels)
        val tmp = File(store.directory, PeopleStore.CSV_FILE + ".tmp")
        tmp.writeText(export.text)
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
        lastWrittenHash = sha256(export.text.toByteArray(Charsets.UTF_8))
        checked = null
        editLogged = false
        lock.withLock {
            // Kept only when names or columns changed (not for new face samples), the newest first.
            val bases = csvBases()
            val latest = bases.firstOrNull()
            val entry = CsvMerge.Base(export.text, export.columns)
            val next = if (latest != null && CsvMerge.sameContent(latest, entry)) listOf(entry) + bases.drop(1) else (listOf(entry) + bases).take(CSV_BASES_KEPT)
            store.setMeta(META_CSV_BASES, next.joinToString(ENTRY) { it.columns.joinToString("\n") + PART + it.text })
        }
    }

    private fun csvBases(): List<CsvMerge.Base> =
        store.meta(META_CSV_BASES).orEmpty().split(ENTRY).filter { PART in it }.map { entry ->
            CsvMerge.Base(entry.substringAfter(PART), entry.substringBefore(PART).split('\n').filter { it.isNotEmpty() })
        }

    private fun logCsv(result: CsvResult) {
        val c = result.changes
        log.log(
            "CSV edits applied: changed=${result.changedPeople}, deleted=${c.deleted.size}, renamed=${c.renamed}, " +
                "removed columns=${c.removedColumns}, unmatched=${c.unmatched.size}" +
                if (c.heldDeletions > 0) ", deletions held back=${c.heldDeletions}" else "",
        )
    }

    private fun saveLabels(next: Map<String, String>) {
        store.setMeta(META_COLUMN_LABELS, ColumnLabels.encode(next).ifEmpty { null })
        labels = next
    }

    /**
     * Replaces everyone with a backup's people, deletion records and column names. Starts a new
     * epoch, so every kiosk downloads the whole list again.
     */
    fun restore(people: List<Person>, tombstones: List<Tombstone>, labels: Map<String, String>) {
        val revisionAfter = lock.withLock {
            beforeBulkChange("restore")
            atomically {
                store.replaceAll(people, tombstones)
                saveLabels(labels)
            }
            loadFromStore(newEpoch = true)
            currentRevision
        }
        log.log("database restored: ${people.size} people, revision=$revisionAfter")
        onChanged(revisionAfter)
    }

    /** A consistent copy of the database, taken while no change is written. */
    fun backupTo(target: File) = lock.withLock { store.backupTo(target) }

    /**
     * Re-reads the database file (e.g. after it was replaced by a backup). Starts a new epoch so
     * every kiosk replaces its cache instead of trusting revision numbers from the old file.
     */
    fun reload() {
        val revisionAfter = lock.withLock {
            loadFromStore(newEpoch = true)
            currentRevision
        }
        onChanged(revisionAfter)
    }

    private fun applyOne(op: PersonOp): OpResult {
        if (op.personKey.isBlank()) return OpResult.REJECTED_INVALID
        if (op.personKey in tombstones) return OpResult.REJECTED_DELETED
        val incoming = when (op) {
            is PersonOp.Register -> op.embeddings
            is PersonOp.AddEmbeddings -> op.embeddings
        }
        if (incoming.isEmpty() || !dimensionAccepted(incoming)) return OpResult.REJECTED_INVALID
        val existing = people[op.personKey]
        val now = clock.now()
        if (existing == null) {
            if (op !is PersonOp.Register) return OpResult.REJECTED_UNKNOWN
            if (op.metadata.name.isBlank()) return OpResult.REJECTED_INVALID
            val samples = EmbeddingSets.prune(EmbeddingSets.union(emptyList(), incoming), maxEmbeddingsPerPerson)
            commit(Person(op.personKey, op.metadata, samples, nextRevision(), now))
            return OpResult.APPLIED
        }
        // The server's metadata is authoritative (administrators edit it here); kiosks only add samples.
        val merged = EmbeddingSets.prune(EmbeddingSets.union(existing.embeddings, incoming), maxEmbeddingsPerPerson)
        if (!EmbeddingSets.sameSet(merged, existing.embeddings)) {
            commit(existing.copy(embeddings = merged, revision = nextRevision(), updatedAt = now))
        }
        return OpResult.APPLIED
    }

    private fun dimensionAccepted(samples: List<FloatArray>): Boolean {
        val dim = samples.first().size
        if (dim == 0 || samples.any { it.size != dim }) return false
        val existingDim = people.values.firstOrNull()?.embeddingDim ?: return true
        return existingDim == dim
    }

    private fun publish() {
        snapshot = Snapshot(LinkedHashMap(people), LinkedHashMap(tombstones), currentEpoch, currentRevision, store.displayNumbers(), labels)
    }

    private fun nextRevision(): Long = ++currentRevision

    /** Runs a change in one transaction; on failure memory is reloaded so it matches the database again. */
    private fun <T> atomically(block: () -> T): T =
        try {
            store.transaction(block)
        } catch (t: Throwable) {
            log.log("change rolled back: ${t.javaClass.simpleName}: ${t.message}")
            loadFromStore(newEpoch = false)
            throw t
        }

    private fun commit(person: Person) {
        store.upsert(person)
        store.setMeta(META_REVISION, currentRevision.toString())
        people[person.key] = person
        tombstones.remove(person.key)
    }

    private fun loadFromStore(newEpoch: Boolean) {
        val loadedPeople = store.loadPeople()
        val loadedTombstones = store.loadTombstones()
        people.clear()
        loadedPeople.forEach { people[it.key] = it }
        tombstones.clear()
        loadedTombstones.forEach { tombstones[it.key] = it }
        val highest = maxOf(loadedPeople.maxOfOrNull { it.revision } ?: 0L, loadedTombstones.maxOfOrNull { it.revision } ?: 0L)
        currentRevision = maxOf(store.meta(META_REVISION)?.toLongOrNull() ?: 0L, highest)
        currentEpoch = store.meta(META_EPOCH)?.takeIf { it.isNotBlank() && !newEpoch } ?: UUID.randomUUID().toString()
        labels = ColumnLabels.decode(store.meta(META_COLUMN_LABELS))
        store.transaction {
            store.setMeta(META_EPOCH, currentEpoch)
            store.setMeta(META_REVISION, currentRevision.toString())
        }
        publish()
        log.log("people loaded: ${people.size} people, ${tombstones.size} deleted, revision=$currentRevision")
    }

    companion object {
        const val META_EPOCH = "server_epoch"
        const val META_REVISION = "server_revision"
        const val META_COLUMN_LABELS = "column_labels"

        /** Recent CSVs the server wrote, with their column keys: what edits are compared with. */
        const val META_CSV_BASES = "csv_bases"
        const val CSV_BASES_KEPT = 20
        private const val ENTRY = "\u0001"
        private const val PART = "\u0002"

        /**
         * Spreadsheets save CSV as UTF-8 (with or without BOM) or, in Korean Excel's plain "CSV",
         * as CP949. Only text that is valid UTF-8 is read as UTF-8.
         */
        private fun sha256(bytes: ByteArray): String =
            java.security.MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

        fun readCsv(bytes: ByteArray): String {
            val utf8 = Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
            return runCatching { utf8.decode(ByteBuffer.wrap(bytes)).toString() }
                .getOrElse { String(bytes, Charset.forName(if (Charset.isSupported("x-windows-949")) "x-windows-949" else "EUC-KR")) }
        }
    }
}
