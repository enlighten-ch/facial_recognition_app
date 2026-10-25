package com.allstar.faithid.core.model

/**
 * Set operations on a person's embedding samples.
 *
 * Kiosks only ever add samples, so merging is a union keyed by [Vectors.fingerprint]. A union is
 * commutative and idempotent: replaying or reordering uploads from many kiosks converges to the same
 * set. [prune] is deterministic so the server and every kiosk agree on which samples are dropped.
 */
object EmbeddingSets {

    /** Existing samples first (original order), then incoming samples not already present. */
    fun union(existing: List<FloatArray>, incoming: List<FloatArray>): List<FloatArray> {
        if (incoming.isEmpty()) return existing
        val dim = existing.firstOrNull()?.size ?: incoming.first().size
        val seen = HashSet<Long>(existing.size + incoming.size)
        val out = ArrayList<FloatArray>(existing.size + incoming.size)
        existing.forEach { if (seen.add(Vectors.fingerprint(it))) out += it }
        incoming.forEach { if (it.size == dim && seen.add(Vectors.fingerprint(it))) out += it }
        return out
    }

    /** True when every incoming sample is already in [existing]. */
    fun containsAll(existing: List<FloatArray>, incoming: List<FloatArray>): Boolean {
        val seen = existing.mapTo(HashSet()) { Vectors.fingerprint(it) }
        return incoming.all { Vectors.fingerprint(it) in seen }
    }

    /**
     * Drops the samples farthest from the centroid until at most [limit] remain. Outliers (bad
     * lighting, wrong person) go first. Ties are broken by fingerprint so the result does not depend
     * on input order.
     */
    fun prune(samples: List<FloatArray>, limit: Int): List<FloatArray> {
        val safeLimit = limit.coerceAtLeast(1)
        if (samples.size <= safeLimit) return samples
        val remaining = samples.toMutableList()
        while (remaining.size > safeLimit) {
            val centroid = Vectors.centroid(remaining)
            var dropIndex = 0
            var dropScore = Float.MAX_VALUE
            var dropFingerprint = Long.MAX_VALUE
            remaining.forEachIndexed { index, sample ->
                val score = Vectors.dot(sample, centroid)
                val fingerprint = Vectors.fingerprint(sample)
                if (score < dropScore || (score == dropScore && fingerprint < dropFingerprint)) {
                    dropScore = score
                    dropFingerprint = fingerprint
                    dropIndex = index
                }
            }
            remaining.removeAt(dropIndex)
        }
        return remaining
    }

    /** Order-independent equality of two sample sets. */
    fun sameSet(a: List<FloatArray>, b: List<FloatArray>): Boolean =
        a.size == b.size && a.mapTo(HashSet()) { Vectors.fingerprint(it) } == b.mapTo(HashSet()) { Vectors.fingerprint(it) }
}
