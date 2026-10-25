package com.allstar.faithid.core.model

import kotlin.math.sqrt

/** Small vector helpers for L2-normalized face embeddings. */
object Vectors {

    fun dot(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var sum = 0f
        for (i in 0 until n) sum += a[i] * b[i]
        return sum
    }

    fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += (x * x).toDouble()
        val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(v.size) { i -> v[i] / norm }
    }

    fun mean(vectors: List<FloatArray>): FloatArray {
        val dim = vectors.firstOrNull()?.size ?: return FloatArray(0)
        val out = FloatArray(dim)
        for (v in vectors) {
            for (i in 0 until minOf(dim, v.size)) out[i] += v[i]
        }
        for (i in 0 until dim) out[i] /= vectors.size.toFloat()
        return out
    }

    /** Normalized mean, used as the single representative vector of a person. */
    fun centroid(vectors: List<FloatArray>): FloatArray = l2Normalize(mean(vectors))

    /**
     * 64-bit FNV-1a over the raw float bits. Two embeddings with the same fingerprint are treated as
     * the same sample, which makes merging sample sets idempotent.
     */
    fun fingerprint(v: FloatArray): Long {
        var hash = -0x340d631b7bdddcdbL
        hash = (hash xor v.size.toLong()) * 0x100000001b3L
        for (x in v) {
            val bits = java.lang.Float.floatToIntBits(x)
            for (shift in 0 until 32 step 8) {
                hash = (hash xor ((bits ushr shift) and 0xFF).toLong()) * 0x100000001b3L
            }
        }
        return hash
    }
}
