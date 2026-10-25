package com.allstar.faithid.core.sync

import com.allstar.faithid.core.codec.BinaryReader
import com.allstar.faithid.core.codec.BinaryWriter
import com.allstar.faithid.core.codec.CodecException
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.model.PersonMetadata

/**
 * A change made on a kiosk, queued in its outbox until the server acknowledges it.
 *
 * Kiosks never overwrite or delete; they only register people and add samples. Both operations
 * are idempotent on the server (sample sets merge by union), so a kiosk can safely resend an
 * operation whose response was lost.
 */
sealed class PersonOp {
    abstract val opId: String
    abstract val personKey: String

    data class Register(
        override val opId: String,
        override val personKey: String,
        val metadata: PersonMetadata,
        val embeddings: List<FloatArray>,
        val createdAt: Long,
    ) : PersonOp()

    data class AddEmbeddings(
        override val opId: String,
        override val personKey: String,
        val embeddings: List<FloatArray>,
    ) : PersonOp()
}

/** Server verdict for one operation. Every verdict is final: the kiosk removes the op from its outbox. */
enum class OpResult {
    /** The change was applied (or had already been applied; replays are harmless). */
    APPLIED,

    /** The person was deleted on the server; the kiosk drops the operation. */
    REJECTED_DELETED,

    /** Samples for a person the server never saw (its registration was rejected or lost). */
    REJECTED_UNKNOWN,

    /** Malformed operation, e.g. no samples or a wrong embedding dimension. */
    REJECTED_INVALID,
}

object PersonOpCodec {
    private const val MAGIC = "FOP1"
    private const val TYPE_REGISTER = 1
    private const val TYPE_ADD = 2

    fun encode(ops: List<PersonOp>): ByteArray {
        val w = BinaryWriter().magic(MAGIC).int(ops.size)
        ops.forEach { writeOp(w, it) }
        return w.toByteArray()
    }

    fun decode(bytes: ByteArray): List<PersonOp> {
        val r = BinaryReader(bytes)
        r.expectMagic(MAGIC)
        return List(r.count(minBytesPerItem = 9)) { readOp(r) }
    }

    /** Single operation, used to persist outbox entries one row at a time. */
    fun encodeOne(op: PersonOp): ByteArray = BinaryWriter().also { writeOp(it, op) }.toByteArray()

    fun decodeOne(bytes: ByteArray): PersonOp = readOp(BinaryReader(bytes))

    private fun writeOp(w: BinaryWriter, op: PersonOp) {
        when (op) {
            is PersonOp.Register -> {
                w.byte(TYPE_REGISTER).string(op.opId).string(op.personKey).long(op.createdAt)
                w.bytes(PersonCodec.encodeMetadata(op.metadata))
                PersonCodec.writeEmbeddings(w, op.embeddings)
            }
            is PersonOp.AddEmbeddings -> {
                w.byte(TYPE_ADD).string(op.opId).string(op.personKey)
                PersonCodec.writeEmbeddings(w, op.embeddings)
            }
        }
    }

    private fun readOp(r: BinaryReader): PersonOp {
        return when (val type = r.byte()) {
            TYPE_REGISTER -> {
                val opId = r.string()
                val key = r.string()
                val createdAt = r.long()
                val metadata = PersonCodec.decodeMetadata(r.bytes()) ?: throw CodecException("invalid metadata in op $opId")
                PersonOp.Register(opId, key, metadata, PersonCodec.readEmbeddings(r), createdAt)
            }
            TYPE_ADD -> {
                val opId = r.string()
                val key = r.string()
                PersonOp.AddEmbeddings(opId, key, PersonCodec.readEmbeddings(r))
            }
            else -> throw CodecException("unknown op type $type")
        }
    }
}
