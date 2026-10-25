package com.allstar.faithid.server.people

import com.allstar.faithid.core.model.EmbeddingSets
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.storage.PeopleCsv
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.ChangeSet
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.OpsResponse
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.Tombstone
import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerLog
import java.io.File
import java.util.UUID

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

    @Volatile
    var epoch: String = ""
        private set

    @Volatile
    var revision: Long = 0
        private set

    /** Called after every committed change with the new revision. */
    @Volatile
    var onChanged: (Long) -> Unit = {}

    init {
        loadFromStore(newEpoch = false)
    }

    @Synchronized
    fun list(): List<Person> = people.values.toList()

    @Synchronized
    fun get(key: String): Person? = people[key]

    @Synchronized
    fun isDeleted(key: String): Boolean = key in tombstones

    @Synchronized
    fun displayNumbers(): Map<String, Long> = store.displayNumbers()

    @Synchronized
    fun changesSince(since: Long, clientEpoch: String?): ChangeSet {
        val full = clientEpoch != epoch || since <= 0 || since > revision
        return if (full) {
            ChangeSet(epoch, revision, fullSnapshot = true, persons = people.values.toList(), tombstones = tombstones.values.toList())
        } else {
            ChangeSet(
                epoch,
                revision,
                fullSnapshot = false,
                persons = people.values.filter { it.revision > since },
                tombstones = tombstones.values.filter { it.revision > since },
            )
        }
    }

    /** Applies kiosk operations in order. Returns one verdict per operation id. */
    fun applyOps(ops: List<PersonOp>, clientId: String): OpsResponse {
        val (response, changed) = synchronized(this) {
            val results = LinkedHashMap<String, OpResult>()
            val before = revision
            atomically {
                ops.forEach { op -> results[op.opId] = applyOne(op) }
                store.setMeta(META_REVISION, revision.toString())
            }
            OpsResponse(revision, results) to (revision != before)
        }
        if (changed) {
            log.log("kiosk changes applied: client=$clientId, ops=${ops.size}, revision=${response.serverRevision}")
            onChanged(response.serverRevision)
        }
        return response
    }

    fun updateMetadata(key: String, name: String, department: String, grade: String): Boolean {
        val revisionAfter = synchronized(this) {
            val person = people[key] ?: return false
            val metadata = person.metadata.with(name.trim(), department.trim(), grade.trim())
            if (metadata == person.metadata) return true
            atomically { commit(person.copy(metadata = metadata, revision = nextRevision(), updatedAt = clock.now())) }
            revision
        }
        log.log("person edited: key=$key, name=$name, revision=$revisionAfter")
        onChanged(revisionAfter)
        return true
    }

    fun delete(key: String): Boolean {
        val revisionAfter = synchronized(this) {
            val person = people[key] ?: return false
            atomically {
                val tombstone = Tombstone(key, nextRevision())
                store.putTombstone(tombstone, clock.now())
                store.setMeta(META_REVISION, revision.toString())
                people.remove(key)
                tombstones[key] = tombstone
            }
            log.log("person deleted: key=$key, name=${person.name}, revision=$revision")
            revision
        }
        onChanged(revisionAfter)
        return true
    }

    /** Applies name/department/grade edits made to `face_db.csv` in a spreadsheet. */
    fun importCsvEdits(csvText: String): PeopleCsv.ApplyResult {
        val (result, revisionAfter) = synchronized(this) {
            val result = PeopleCsv.applyEdits(people.values.toList(), store.displayNumbers(), PeopleCsv.parseEdits(csvText), clock.now())
            if (result.changed.isNotEmpty()) {
                atomically {
                    val rev = nextRevision()
                    result.changed.forEach { commit(it.copy(revision = rev)) }
                }
            }
            result to revision
        }
        log.log("CSV edits applied: changed=${result.changed.size}, unmatched=${result.unmatched.size}")
        if (result.changed.isNotEmpty()) onChanged(revisionAfter)
        return result
    }

    /** Writes `face_db.csv` next to the database (metadata only, for viewing and editing in Excel). */
    @Synchronized
    fun exportCsv(): File {
        val file = File(store.directory, PeopleStore.CSV_FILE)
        val tmp = File(store.directory, PeopleStore.CSV_FILE + ".tmp")
        tmp.writeText(PeopleCsv.export(people.values.toList(), store.displayNumbers()))
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
        return file
    }

    /**
     * Re-reads the database file (e.g. after it was replaced by a backup). Starts a new epoch so
     * every kiosk replaces its cache instead of trusting revision numbers from the old file.
     */
    fun reload() {
        val revisionAfter = synchronized(this) {
            loadFromStore(newEpoch = true)
            revision
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

    private fun nextRevision(): Long = ++revision

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
        store.setMeta(META_REVISION, revision.toString())
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
        revision = maxOf(store.meta(META_REVISION)?.toLongOrNull() ?: 0L, highest)
        epoch = store.meta(META_EPOCH)?.takeIf { it.isNotBlank() && !newEpoch } ?: UUID.randomUUID().toString()
        store.transaction {
            store.setMeta(META_EPOCH, epoch)
            store.setMeta(META_REVISION, revision.toString())
        }
        log.log("people loaded: ${people.size} people, ${tombstones.size} deleted, revision=$revision")
    }

    companion object {
        const val META_EPOCH = "server_epoch"
        const val META_REVISION = "server_revision"
    }
}
