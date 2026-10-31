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
    /**
     * Asks for the samples of each direction with a normal, a big and a small face circle, so the
     * person also comes closer and steps back. Off: every sample uses the normal circle.
     */
    val circleSteps: Boolean = true,
    /** Size of the big and the small circle relative to the normal one (drawn on screen). */
    val bigCircleScale: Float = 1.18f,
    val smallCircleScale: Float = 0.73f,
) {
    /** A big-circle sample needs a face at least this much larger than the direction's first sample. */
    val minGrowth: Float get() = 1f + (bigCircleScale - 1f).coerceAtLeast(0f) / 2f

    /** A small-circle sample needs a face at most this large relative to the direction's first sample. */
    val maxShrink: Float get() = 1f - (1f - smallCircleScale).coerceAtLeast(0f) / 2f
}

/** Which face circle the next sample is taken with. */
enum class FaceCircle { NORMAL, BIG, SMALL }

/**
 * Collects face samples for one registration: [RegistrationConfig.samplesPerDirection] samples for
 * each [HeadDirection], in order. With [RegistrationConfig.circleSteps] the samples of a direction
 * cycle through the normal, big and small circle, checked by the face size the camera reports
 * relative to the direction's first sample. Not thread-safe; drive it from one thread.
 */
class RegistrationSession(private val config: RegistrationConfig) {

    sealed class Outcome {
        data class Accepted(val direction: HeadDirection, val maxSimilarity: Float?) : Outcome()
        data class WrongDirection(val expected: HeadDirection, val detected: HeadDirection?) : Outcome()
        data class TooDifferent(val maxSimilarity: Float) : Outcome()
        data class TooSimilar(val maxSimilarity: Float) : Outcome()

        /** The face does not fit the circle yet: [expected] BIG means come closer, SMALL step back. */
        data class WrongSize(val expected: FaceCircle, val ratio: Float) : Outcome()
        data object AlreadyComplete : Outcome()
    }

    private val perDirection = HeadDirection.entries.associateWith { mutableListOf<FloatArray>() }

    /** Face size of each direction's first (normal circle) sample. */
    private val baseline = mutableMapOf<HeadDirection, Float>()

    val target: Int get() = config.samplesPerDirection.coerceAtLeast(1)
    val totalTarget: Int get() = target * HeadDirection.entries.size
    val totalCollected: Int get() = perDirection.values.sumOf { it.size }

    val nextDirection: HeadDirection?
        get() = HeadDirection.entries.firstOrNull { (perDirection[it]?.size ?: 0) < target }

    val isComplete: Boolean get() = nextDirection == null

    /** The circle for the next sample, or null when complete. */
    val nextCircle: FaceCircle?
        get() = nextDirection?.let { circleFor(collected(it)) }

    fun circleFor(sampleIndex: Int): FaceCircle =
        if (config.circleSteps) FaceCircle.entries[sampleIndex % FaceCircle.entries.size] else FaceCircle.NORMAL

    fun collected(direction: HeadDirection): Int = perDirection[direction]?.size ?: 0

    val samples: List<FloatArray> get() = HeadDirection.entries.flatMap { perDirection[it].orEmpty() }

    /**
     * [faceSize] is the face width relative to the camera frame; when it is unknown (null) the
     * circle size is not checked.
     */
    fun offer(detected: HeadDirection?, embedding: FloatArray, faceSize: Float? = null): Outcome {
        val expected = nextDirection ?: return Outcome.AlreadyComplete
        if (detected != expected) return Outcome.WrongDirection(expected, detected)
        val circle = circleFor(collected(expected))
        val base = baseline[expected]
        if (faceSize != null && base != null && base > 0f) {
            val ratio = faceSize / base
            if (circle == FaceCircle.BIG && ratio < config.minGrowth) return Outcome.WrongSize(circle, ratio)
            if (circle == FaceCircle.SMALL && ratio > config.maxShrink) return Outcome.WrongSize(circle, ratio)
        }
        val existing = samples
        val maxSimilarity = existing.maxOfOrNull { Vectors.dot(it, embedding) }
        if (maxSimilarity != null) {
            if (maxSimilarity > config.maxSimilarity) return Outcome.TooSimilar(maxSimilarity)
            if (maxSimilarity < config.minSimilarity) return Outcome.TooDifferent(maxSimilarity)
        }
        perDirection.getValue(expected) += embedding
        if (circle == FaceCircle.NORMAL && faceSize != null && expected !in baseline) baseline[expected] = faceSize
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
        baseline.clear()
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
