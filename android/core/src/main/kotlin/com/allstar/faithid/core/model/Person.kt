package com.allstar.faithid.core.model

import java.security.MessageDigest
import java.util.Locale
import java.util.UUID

/**
 * A registered person: a stable [key], display [metadata] and face [embeddings].
 *
 * Instances are immutable; callers must not mutate the embedding arrays after construction.
 * [revision] is the server revision at which this version of the person was last changed (0 for a
 * change that only exists on a kiosk so far).
 */
class Person(
    val key: String,
    val metadata: PersonMetadata,
    val embeddings: List<FloatArray>,
    val revision: Long = 0L,
    val updatedAt: Long = 0L,
) {
    init {
        require(key.isNotBlank()) { "person key must not be blank" }
    }

    val name: String get() = metadata.name
    val department: String get() = metadata.department
    val grade: String get() = metadata.grade
    val embeddingDim: Int get() = embeddings.firstOrNull()?.size ?: 0

    val centroid: FloatArray by lazy(LazyThreadSafetyMode.PUBLICATION) { Vectors.centroid(embeddings) }

    fun copy(
        metadata: PersonMetadata = this.metadata,
        embeddings: List<FloatArray> = this.embeddings,
        revision: Long = this.revision,
        updatedAt: Long = this.updatedAt,
    ): Person = Person(key, metadata, embeddings, revision, updatedAt)

    /** Hash of everything that is synchronized (metadata and the sample set, order-independent). */
    fun contentHash(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(key.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(metadata.contentHash().toByteArray(Charsets.UTF_8))
        embeddings.map { Vectors.fingerprint(it) }.sorted().forEach { fp ->
            for (shift in 0 until 64 step 8) digest.update(((fp ushr shift) and 0xFF).toByte())
        }
        return digest.digest().toHex()
    }

    fun sameContentAs(other: Person): Boolean =
        key == other.key && metadata == other.metadata && EmbeddingSets.sameSet(embeddings, other.embeddings)

    override fun toString(): String =
        "Person(key=$key, name=$name, department=$department, grade=$grade, samples=${embeddings.size}, rev=$revision)"
}

object PersonKeys {
    private const val PREFIX = "p_"

    /** Key for a newly registered person. Random, so people with the same name stay distinct. */
    fun newKey(): String = PREFIX + UUID.randomUUID().toString().replace("-", "")

    /**
     * Key used by app versions before 2026-10: SHA-256 of name, department and grade. Kept so that
     * existing databases and CSV exports keep their identities after the upgrade.
     */
    fun legacyKey(name: String, department: String, grade: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        listOf(name.trim(), department.trim(), grade.trim()).forEach { value ->
            digest.update(value.lowercase(Locale.US).toByteArray(Charsets.UTF_8))
            digest.update(0)
        }
        return digest.digest().toHex()
    }
}
