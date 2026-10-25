package com.allstar.faithid.core.codec

import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata

/** Binary encodings of [PersonMetadata] and [Person], shared by storage and the sync protocol. */
object PersonCodec {
    private const val METADATA_MAGIC = "MDT1"

    /** Same "MDT1" layout as the metadata_blob column written by earlier app versions. */
    fun encodeMetadata(metadata: PersonMetadata): ByteArray {
        val w = BinaryWriter(256).magic(METADATA_MAGIC).int(metadata.entries.size)
        metadata.entries.forEach { (key, value) -> w.string(key).string(value) }
        return w.toByteArray()
    }

    fun decodeMetadata(bytes: ByteArray?): PersonMetadata? {
        if (bytes == null || bytes.size < 8) return null
        return runCatching {
            val r = BinaryReader(bytes)
            r.expectMagic(METADATA_MAGIC)
            val count = r.count(minBytesPerItem = 8)
            val fields = LinkedHashMap<String, String>()
            repeat(count) { fields[r.string()] = r.string() }
            PersonMetadata.of(fields)
        }.getOrNull()
    }

    fun writePerson(w: BinaryWriter, person: Person) {
        w.string(person.key)
        w.long(person.revision)
        w.long(person.updatedAt)
        w.bytes(encodeMetadata(person.metadata))
        writeEmbeddings(w, person.embeddings)
    }

    fun readPerson(r: BinaryReader): Person {
        val key = r.string()
        val revision = r.long()
        val updatedAt = r.long()
        val metadata = decodeMetadata(r.bytes()) ?: throw CodecException("invalid metadata for $key")
        val embeddings = readEmbeddings(r)
        if (key.isBlank()) throw CodecException("blank person key")
        return Person(key, metadata, embeddings, revision, updatedAt)
    }

    fun writeEmbeddings(w: BinaryWriter, embeddings: List<FloatArray>) {
        val dim = embeddings.firstOrNull()?.size ?: 0
        w.int(embeddings.size).int(dim)
        embeddings.forEach { v ->
            if (v.size != dim) throw CodecException("embeddings must share one dimension")
            w.floats(v)
        }
    }

    fun readEmbeddings(r: BinaryReader): List<FloatArray> {
        val count = r.count(minBytesPerItem = 0)
        val dim = r.int()
        if (dim < 0 || dim > MAX_DIM) throw CodecException("implausible embedding dimension $dim")
        if (count.toLong() * dim * 4 > r.remaining) throw CodecException("embedding block truncated")
        return List(count) { r.floats(dim) }
    }

    const val MAX_DIM = 4096
}
