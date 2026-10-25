package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.model.Vectors

/** Settings from the kiosk "등록" tab. */
data class RegistrationConfig(
    val samplesPerDirection: Int = 3,
    /** A new sample must be at least this similar to the samples already taken (same person). */
    val minSimilarity: Float = 0.35f,
    /** ... and at most this similar (otherwise it adds nothing new). */
    val maxSimilarity: Float = 0.85f,
    /** Finished samples this similar to an already registered person trigger the "is this you?" check. */
    val duplicateThreshold: Float = 0.45f,
)

/**
 * Collects face samples for one registration: [RegistrationConfig.samplesPerDirection] samples for
 * each [HeadDirection], in order. Not thread-safe; drive it from one thread.
 */
class RegistrationSession(private val config: RegistrationConfig) {

    sealed class Outcome {
        data class Accepted(val direction: HeadDirection, val maxSimilarity: Float?) : Outcome()
        data class WrongDirection(val expected: HeadDirection, val detected: HeadDirection?) : Outcome()
        data class TooDifferent(val maxSimilarity: Float) : Outcome()
        data class TooSimilar(val maxSimilarity: Float) : Outcome()
        data object AlreadyComplete : Outcome()
    }

    private val perDirection = HeadDirection.entries.associateWith { mutableListOf<FloatArray>() }

    val target: Int get() = config.samplesPerDirection.coerceAtLeast(1)
    val totalTarget: Int get() = target * HeadDirection.entries.size
    val totalCollected: Int get() = perDirection.values.sumOf { it.size }

    val nextDirection: HeadDirection?
        get() = HeadDirection.entries.firstOrNull { (perDirection[it]?.size ?: 0) < target }

    val isComplete: Boolean get() = nextDirection == null

    fun collected(direction: HeadDirection): Int = perDirection[direction]?.size ?: 0

    val samples: List<FloatArray> get() = HeadDirection.entries.flatMap { perDirection[it].orEmpty() }

    fun offer(detected: HeadDirection?, embedding: FloatArray): Outcome {
        val expected = nextDirection ?: return Outcome.AlreadyComplete
        if (detected != expected) return Outcome.WrongDirection(expected, detected)
        val existing = samples
        val maxSimilarity = existing.maxOfOrNull { Vectors.dot(it, embedding) }
        if (maxSimilarity != null) {
            if (maxSimilarity > config.maxSimilarity) return Outcome.TooSimilar(maxSimilarity)
            if (maxSimilarity < config.minSimilarity) return Outcome.TooDifferent(maxSimilarity)
        }
        perDirection.getValue(expected) += embedding
        return Outcome.Accepted(expected, maxSimilarity)
    }

    /**
     * The registered person these samples most likely belong to, if any. Checked when the samples
     * are complete, before asking for a name, so the same person is not registered twice.
     */
    fun likelyDuplicateOf(index: FaceIndex): Match? {
        val all = samples
        if (all.isEmpty()) return null
        val best = index.best(Vectors.centroid(all)) ?: return null
        return best.takeIf { it.score >= config.duplicateThreshold }
    }

    fun reset() {
        perDirection.values.forEach { it.clear() }
    }
}

/**
 * Decides which recognized faces become extra samples ("자동 보정"). Limits how many are taken per
 * kiosk session and how often, so one person standing in front of the camera does not flood the
 * database.
 */
class AdaptiveSampler(
    private val enabled: Boolean,
    private val maxPerSession: Int = 20,
    private val minIntervalFrames: Int = 30,
) {
    private var takenThisSession = 0
    private var lastTakenFrame = Long.MIN_VALUE / 2

    /** Returns true if the sample at [frame] should be kept. */
    fun shouldTake(frame: Long): Boolean {
        if (!enabled) return false
        if (takenThisSession >= maxPerSession) return false
        if (frame - lastTakenFrame < minIntervalFrames) return false
        takenThisSession++
        lastTakenFrame = frame
        return true
    }

    /** Starts a new session (the kiosk went back to its idle screen). */
    fun resetSession() {
        takenThisSession = 0
        lastTakenFrame = Long.MIN_VALUE / 2
    }
}
