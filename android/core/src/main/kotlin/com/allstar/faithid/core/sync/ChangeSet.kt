package com.allstar.faithid.core.sync

import com.allstar.faithid.core.codec.BinaryReader
import com.allstar.faithid.core.codec.BinaryWriter
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.model.Person

/** A person removed on the server at [revision]. Kept forever so late kiosks learn about it. */
data class Tombstone(val key: String, val revision: Long)

/**
 * What a kiosk receives when it pulls: every person and tombstone changed after the kiosk's last
 * known revision.
 *
 * [serverEpoch] identifies one server database. When the kiosk's stored epoch differs (new server,
 * restored backup) the server sends a [fullSnapshot] and the kiosk replaces its whole cache.
 */
data class ChangeSet(
    val serverEpoch: String,
    val serverRevision: Long,
    val fullSnapshot: Boolean,
    val persons: List<Person>,
    val tombstones: List<Tombstone>,
) {
    fun isEmpty(): Boolean = persons.isEmpty() && tombstones.isEmpty() && !fullSnapshot
}

object ChangeSetCodec {
    private const val MAGIC = "FCS1"

    fun encode(changeSet: ChangeSet): ByteArray {
        val estimate = 64 + changeSet.persons.sumOf { 128 + it.embeddings.size * it.embeddingDim * 4 }
        val w = BinaryWriter(estimate.coerceAtMost(64 * 1024 * 1024))
            .magic(MAGIC)
            .string(changeSet.serverEpoch)
            .long(changeSet.serverRevision)
            .byte(if (changeSet.fullSnapshot) 1 else 0)
        w.int(changeSet.tombstones.size)
        changeSet.tombstones.forEach { w.string(it.key).long(it.revision) }
        w.int(changeSet.persons.size)
        changeSet.persons.forEach { PersonCodec.writePerson(w, it) }
        return w.toByteArray()
    }

    fun decode(bytes: ByteArray): ChangeSet {
        val r = BinaryReader(bytes)
        r.expectMagic(MAGIC)
        val epoch = r.string()
        val revision = r.long()
        val full = r.byte() == 1
        val tombstones = List(r.count(minBytesPerItem = 12)) { Tombstone(r.string(), r.long()) }
        val persons = List(r.count(minBytesPerItem = 32)) { PersonCodec.readPerson(r) }
        return ChangeSet(epoch, revision, full, persons, tombstones)
    }
}
