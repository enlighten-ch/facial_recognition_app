package com.allstar.faithid.core.testing

import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonKeys
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.Vectors
import kotlin.random.Random

/**
 * Synthetic face embeddings. Each fake identity has a random base direction; its samples are the
 * base plus noise, normalized like real ArcFace outputs. Same-identity samples score ~0.8, different
 * identities ~0.0, which mirrors the separation the real model gives.
 */
class FakeFaces(seed: Long = 42, val dim: Int = 512) {
    private val random = Random(seed)
    private val bases = HashMap<String, FloatArray>()

    @Synchronized
    fun base(identity: String): FloatArray = bases.getOrPut(identity) {
        Vectors.l2Normalize(FloatArray(dim) { gaussian() })
    }

    /** [noise] is the norm of the added noise vector: 0.6 gives sample-to-sample similarity ≈ 0.74. */
    @Synchronized
    fun sample(identity: String, noise: Float = 0.6f): FloatArray {
        val base = base(identity)
        val scale = noise / kotlin.math.sqrt(dim.toFloat())
        return Vectors.l2Normalize(FloatArray(dim) { base[it] + scale * gaussian() })
    }

    fun samples(identity: String, count: Int, noise: Float = 0.6f): List<FloatArray> = List(count) { sample(identity, noise) }

    fun person(
        name: String,
        department: String = "청년부",
        grade: String = "1학년",
        samples: Int = 12,
        key: String = PersonKeys.newKey(),
        identity: String = key,
    ): Person = Person(key, PersonMetadata.of(name, department, grade), samples(identity, samples))

    private fun gaussian(): Float {
        var u = 0.0
        var v = 0.0
        while (u == 0.0) u = random.nextDouble()
        while (v == 0.0) v = random.nextDouble()
        return (kotlin.math.sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * Math.PI * v)).toFloat()
    }
}
