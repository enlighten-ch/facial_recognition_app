package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.Vectors

/** One ranked candidate for a query face. */
data class Match(
    val personKey: String,
    val metadata: PersonMetadata,
    val score: Float,
) {
    val name: String get() = metadata.name
    val label: String get() = metadata.label()
}

/**
 * Immutable snapshot used for matching: one centroid per person. Building it once per database
 * change keeps per-frame matching allocation-free and safe to call from the camera thread while the
 * database is being updated.
 */
class FaceIndex private constructor(private val entries: List<Entry>) {

    private class Entry(val key: String, val metadata: PersonMetadata, val centroid: FloatArray)

    val size: Int get() = entries.size

    fun isEmpty(): Boolean = entries.isEmpty()

    fun rank(embedding: FloatArray, topK: Int): List<Match> {
        if (entries.isEmpty()) return emptyList()
        val k = topK.coerceAtLeast(1)
        return entries
            .map { Match(it.key, it.metadata, Vectors.dot(it.centroid, embedding)) }
            .sortedWith(compareByDescending<Match> { it.score }.thenBy { it.personKey })
            .take(k)
    }

    fun best(embedding: FloatArray): Match? = rank(embedding, 1).firstOrNull()

    fun metadataOf(personKey: String): PersonMetadata? = entries.firstOrNull { it.key == personKey }?.metadata

    companion object {
        val EMPTY = FaceIndex(emptyList())

        fun of(people: Collection<Person>): FaceIndex =
            FaceIndex(people.filter { it.embeddings.isNotEmpty() }.map { Entry(it.key, it.metadata, it.centroid) })
    }
}
