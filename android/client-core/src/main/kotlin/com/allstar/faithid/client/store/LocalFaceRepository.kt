package com.allstar.faithid.client.store

import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.model.EmbeddingSets
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonKeys
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.ChangeSet
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.PersonOp
import java.io.Closeable
import java.util.UUID

/**
 * The kiosk's face database: the server's people plus this kiosk's not-yet-confirmed changes.
 *
 * Local changes are applied immediately (so a person can be recognized right after registering)
 * and queued in the outbox in the same transaction. When server changes arrive, they are applied
 * and the queued changes are applied again on top, so nothing done on this kiosk is lost while it
 * is offline or while a response is lost.
 *
 * Thread-safe. [index] is an immutable snapshot that the camera thread can use without locking.
 */
class LocalFaceRepository(
    private val store: PeopleStore,
    private val maxEmbeddingsPerPerson: Int = 100,
    private val now: () -> Long = System::currentTimeMillis,
) : Closeable {
    private val outbox = Outbox(store.database)
    private val people = LinkedHashMap<String, Person>()
    private val listeners = mutableListOf<(FaceIndex) -> Unit>()

    @Volatile
    var index: FaceIndex = FaceIndex.EMPTY
        private set

    init {
        synchronized(this) {
            store.loadPeople().forEach { people[it.key] = it }
            rebuildIndex()
        }
    }

    data class SyncPosition(val epoch: String?, val revision: Long)

    @Synchronized
    fun syncPosition(): SyncPosition = SyncPosition(store.meta(META_EPOCH), store.meta(META_REVISION)?.toLongOrNull() ?: 0L)

    @Synchronized
    fun people(): List<Person> = people.values.toList()

    @Synchronized
    fun get(key: String): Person? = people[key]

    @Synchronized
    fun pendingOps(limit: Int = Int.MAX_VALUE): List<PersonOp> = outbox.ops(limit)

    @Synchronized
    fun pendingAttendance(limit: Int = Int.MAX_VALUE): List<AttendanceEvent> = outbox.attendance(limit)

    @Synchronized
    fun pendingCounts(): Pair<Int, Int> = outbox.opCount() to outbox.attendanceCount()

    fun addIndexListener(listener: (FaceIndex) -> Unit) {
        synchronized(listeners) { listeners += listener }
    }

    /** Registers a new person on this kiosk. The key is random, so namesakes stay separate. */
    fun register(metadata: PersonMetadata, samples: List<FloatArray>): Person {
        require(samples.isNotEmpty()) { "registration needs samples" }
        require(metadata.name.isNotBlank()) { "registration needs a name" }
        val person = synchronized(this) {
            val key = PersonKeys.newKey()
            val created = Person(key, metadata, EmbeddingSets.prune(EmbeddingSets.union(emptyList(), samples), maxEmbeddingsPerPerson), 0L, now())
            store.transaction {
                store.upsert(created)
                outbox.addOp(PersonOp.Register(newOpId(), key, metadata, samples, now()), now())
            }
            people[key] = created
            rebuildIndex()
            created
        }
        notifyListeners()
        return person
    }

    /** Adds recognized-face samples to a person ("자동 보정" or a confirmed duplicate). */
    fun addSamples(personKey: String, samples: List<FloatArray>): Boolean {
        if (samples.isEmpty()) return false
        synchronized(this) {
            val person = people[personKey] ?: return false
            val merged = EmbeddingSets.prune(EmbeddingSets.union(person.embeddings, samples), maxEmbeddingsPerPerson)
            store.transaction {
                if (!EmbeddingSets.sameSet(merged, person.embeddings)) {
                    val updated = person.copy(embeddings = merged, updatedAt = now())
                    store.upsert(updated)
                    people[personKey] = updated
                }
                outbox.addOp(PersonOp.AddEmbeddings(newOpId(), personKey, samples), now())
            }
            rebuildIndex()
        }
        notifyListeners()
        return true
    }

    @Synchronized
    fun recordAttendance(event: AttendanceEvent) {
        outbox.addAttendance(event, now())
    }

    /** Removes acknowledged operations. Rejected registrations disappear locally as well. */
    fun acknowledgeOps(results: Map<String, OpResult>) {
        if (results.isEmpty()) return
        val changed = synchronized(this) {
            val pending = outbox.ops().associateBy { it.opId }
            var changed = false
            store.transaction {
                results.forEach { (opId, result) ->
                    val op = pending[opId] ?: return@forEach
                    if (result != OpResult.APPLIED) {
                        val local = people[op.personKey]
                        // Only drop people the server never confirmed; confirmed ones are fixed by the next pull.
                        if (local != null && local.revision == 0L && op is PersonOp.Register) {
                            store.delete(op.personKey)
                            people.remove(op.personKey)
                            changed = true
                        }
                    }
                }
                outbox.removeOps(results.keys)
            }
            if (changed) rebuildIndex()
            changed
        }
        if (changed) notifyListeners()
    }

    @Synchronized
    fun acknowledgeAttendance(eventIds: Collection<String>) = outbox.removeAttendance(eventIds)

    /**
     * Applies changes pulled from the server, then re-applies this kiosk's queued operations so the
     * local view is always "server state + my pending changes". Returns the number of people changed.
     */
    fun applyServerChanges(changes: ChangeSet): Int {
        val count = synchronized(this) {
            store.transaction {
                if (changes.fullSnapshot) {
                    store.replaceAll(changes.persons, changes.tombstones)
                    people.clear()
                    changes.persons.forEach { people[it.key] = it }
                } else {
                    changes.tombstones.forEach { tombstone ->
                        store.putTombstone(tombstone, now())
                        people.remove(tombstone.key)
                    }
                    changes.persons.forEach { person ->
                        store.upsert(person)
                        people[person.key] = person
                    }
                }
                val deleted = changes.tombstones.mapTo(HashSet()) { it.key }
                reapplyPendingOps(deleted)
                store.setMeta(META_EPOCH, changes.serverEpoch)
                store.setMeta(META_REVISION, changes.serverRevision.toString())
            }
            rebuildIndex()
            changes.persons.size + changes.tombstones.size
        }
        if (count > 0 || changes.fullSnapshot) notifyListeners()
        return count
    }

    override fun close() = store.close()

    private fun reapplyPendingOps(deleted: Set<String>) {
        outbox.ops().forEach { op ->
            if (op.personKey in deleted) return@forEach
            val current = people[op.personKey]
            val updated = when (op) {
                is PersonOp.Register ->
                    if (current == null) {
                        Person(op.personKey, op.metadata, EmbeddingSets.prune(op.embeddings, maxEmbeddingsPerPerson), 0L, op.createdAt)
                    } else {
                        current.copy(embeddings = EmbeddingSets.prune(EmbeddingSets.union(current.embeddings, op.embeddings), maxEmbeddingsPerPerson))
                    }
                is PersonOp.AddEmbeddings ->
                    current?.copy(embeddings = EmbeddingSets.prune(EmbeddingSets.union(current.embeddings, op.embeddings), maxEmbeddingsPerPerson))
            } ?: return@forEach
            if (current == null || !EmbeddingSets.sameSet(current.embeddings, updated.embeddings)) {
                store.upsert(updated)
                people[updated.key] = updated
            }
        }
    }

    private fun rebuildIndex() {
        index = FaceIndex.of(people.values)
    }

    private fun notifyListeners() {
        val snapshot = index
        val copy = synchronized(listeners) { listeners.toList() }
        copy.forEach { it(snapshot) }
    }

    private fun newOpId(): String = UUID.randomUUID().toString()

    companion object {
        const val META_EPOCH = "sync_epoch"
        const val META_REVISION = "sync_revision"
    }
}
