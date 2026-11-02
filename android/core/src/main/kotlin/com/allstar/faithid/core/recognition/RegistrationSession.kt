package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.model.Vectors
import kotlin.math.abs

/** Settings from the kiosk "등록" tab. */
data class RegistrationConfig(
    /** Samples per direction when [circleSteps] is off; with circle steps each circle gives one. */
    val samplesPerDirection: Int = 3,
    /** Collects frontal samples too (5 directions instead of 4). */
    val includeFront: Boolean = true,
    /** A new sample must be at least this similar to the samples already taken (same person). */
    val minSimilarity: Float = 0.35f,
    /** ... and at most this similar (otherwise it adds nothing new). */
    val maxSimilarity: Float = 0.85f,
    /** Finished samples this similar to an already registered person trigger the "is this you?" check. */
    val duplicateThreshold: Float = 0.45f,
    /**
     * Collects each direction with a normal, a big and a small face circle, so the person also comes
     * closer and steps back. Off: [samplesPerDirection] samples per direction at any distance.
     */
    val circleSteps: Boolean = true,
    /** Size of the big and the small circle relative to the normal one (drawn on screen). */
    val bigCircleScale: Float = 1.18f,
    val smallCircleScale: Float = 0.73f,
    /** A face fits a circle when its size is within this fraction of the circle's size. */
    val sizeTolerance: Float = 0.12f,
    /** Frames in a row the face must stay in the same cell before a sample is kept (the sharpest of them). */
    val holdFrames: Int = 2,
    val pose: PoseLimits = PoseLimits(),
)

/** Which face circle a sample is taken with. */
enum class FaceCircle { NORMAL, BIG, SMALL }

/** One sample to collect: a direction at a circle size. [slot] numbers repeats when circle steps are off. */
data class RegistrationCell(val direction: HeadDirection, val circle: FaceCircle, val slot: Int = 0)

/**
 * One analyzed frame during registration. [faceSize] is the face size relative to the normal circle
 * as drawn (1 = fits it; null = unknown, not checked). [sharpness] picks the best frame of a hold.
 */
class RegistrationFrame(
    val embedding: FloatArray,
    val pose: Pose?,
    val faceSize: Float? = null,
    val sharpness: Float = 0f,
)

/**
 * Collects the face samples of one registration: every [HeadDirection] (front optional) at every
 * circle size. The sizes come in rounds, normal, then big, then small, so the person moves forward
 * and back only twice; within a round the directions come in any order. A cell is filled after the
 * face stayed in it for [RegistrationConfig.holdFrames] frames, and only when the face fits the
 * round's circle. [suggestion] is the cell the guide asks for next. Not thread-safe; drive it from
 * one thread.
 */
class RegistrationSession(private val config: RegistrationConfig) {

    sealed class Outcome {
        data class Accepted(val cell: RegistrationCell, val maxSimilarity: Float?) : Outcome()

        /** The pose matches an empty cell; [frames] of [needed] frames held so far. */
        data class Holding(val cell: RegistrationCell, val frames: Int, val needed: Int) : Outcome()

        /** The pose matches no empty cell: [detected] is null between directions or already collected. */
        data class WrongDirection(val expected: HeadDirection, val detected: HeadDirection?) : Outcome()

        /** The face does not fit the round's circle [expected]; [ratio] = size / [expected]'s size. */
        data class WrongSize(val expected: FaceCircle, val ratio: Float) : Outcome()
        data object TooTurned : Outcome()
        data class TooDifferent(val maxSimilarity: Float) : Outcome()
        data class TooSimilar(val maxSimilarity: Float) : Outcome()
        data object AlreadyComplete : Outcome()
    }

    private val directions = HeadDirection.entries.filter { config.includeFront || it != HeadDirection.FRONT }

    /** All cells, in the order the guide suggests them. */
    val cells: List<RegistrationCell> =
        if (config.circleSteps) {
            FaceCircle.entries.flatMap { circle -> directions.map { RegistrationCell(it, circle) } }
        } else {
            (0 until config.samplesPerDirection.coerceAtLeast(1)).flatMap { slot -> directions.map { RegistrationCell(it, FaceCircle.NORMAL, slot) } }
        }

    private val collected = LinkedHashMap<RegistrationCell, FloatArray>()

    private var holdCell: RegistrationCell? = null
    private var holdFrames = 0
    private var holdBest: RegistrationFrame? = null

    val totalTarget: Int get() = cells.size
    val totalCollected: Int get() = collected.size
    val isComplete: Boolean get() = collected.size >= cells.size

    fun has(cell: RegistrationCell): Boolean = cell in collected

    /** The next cell the guide asks for, or null when complete. */
    val suggestion: RegistrationCell? get() = cells.firstOrNull { it !in collected }

    val nextDirection: HeadDirection? get() = suggestion?.direction
    val nextCircle: FaceCircle? get() = suggestion?.circle

    fun collected(direction: HeadDirection): Int = collected.keys.count { it.direction == direction }

    /** The samples in cell order. */
    val samples: List<FloatArray> get() = cells.mapNotNull { collected[it] }

    fun scaleOf(circle: FaceCircle): Float = when (circle) {
        FaceCircle.NORMAL -> 1f
        FaceCircle.BIG -> config.bigCircleScale
        FaceCircle.SMALL -> config.smallCircleScale
    }

    /** The circle a face of [size] fits best, or null when it fits none within the tolerance. */
    fun circleFor(size: Float): FaceCircle? =
        FaceCircle.entries
            .map { it to abs(size / scaleOf(it) - 1f) }
            .filter { it.second <= config.sizeTolerance }
            .minByOrNull { it.second }
            ?.first

    fun offer(frame: RegistrationFrame): Outcome {
        val next = suggestion ?: return Outcome.AlreadyComplete
        val pose = frame.pose
        if (pose == Pose.TooFar) return interrupted(Outcome.TooTurned)
        val direction = (pose as? Pose.Facing)?.direction
        // Only the current round counts: the circle size the guide asks for, any direction still empty in it.
        val open = cells.filter { it.direction == direction && it !in collected && inRound(it, next) }
        if (direction == null || open.isEmpty()) return interrupted(Outcome.WrongDirection(next.direction, direction))
        val size = frame.faceSize
        if (config.circleSteps && size != null && circleFor(size) != next.circle) {
            return interrupted(Outcome.WrongSize(next.circle, size / scaleOf(next.circle)))
        }
        val cell = open.firstOrNull { it == next } ?: open.first()

        if (cell != holdCell) {
            holdCell = cell
            holdFrames = 0
            holdBest = null
        }
        holdFrames++
        if (frame.sharpness >= (holdBest?.sharpness ?: Float.NEGATIVE_INFINITY)) holdBest = frame
        val needed = config.holdFrames.coerceAtLeast(1)
        if (holdFrames < needed) return Outcome.Holding(cell, holdFrames, needed)

        val best = holdBest!!.embedding
        resetHold()
        val maxSimilarity = collected.values.maxOfOrNull { Vectors.dot(it, best) }
        if (maxSimilarity != null) {
            if (maxSimilarity > config.maxSimilarity) return Outcome.TooSimilar(maxSimilarity)
            if (maxSimilarity < config.minSimilarity) return Outcome.TooDifferent(maxSimilarity)
        }
        collected[cell] = best
        return Outcome.Accepted(cell, maxSimilarity)
    }

    /** Something came between the frames of a hold (no face, several faces): start the hold again. */
    fun interrupt() = resetHold()

    /** With circle steps a round is one circle size; without, every open cell is in it. */
    private fun inRound(cell: RegistrationCell, next: RegistrationCell): Boolean =
        !config.circleSteps || cell.circle == next.circle

    private fun interrupted(outcome: Outcome): Outcome {
        resetHold()
        return outcome
    }

    private fun resetHold() {
        holdCell = null
        holdFrames = 0
        holdBest = null
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
        collected.clear()
        resetHold()
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
