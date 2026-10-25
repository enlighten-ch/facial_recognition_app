package com.allstar.faithid.core.recognition

/** Thresholds from the kiosk settings ("인식" tab). */
data class RecognitionThresholds(
    /** Minimum similarity for the best match to count as a registered person. */
    val minScore: Float = 0.40f,
    /** Minimum gap between the first and second match to accept the first without asking. */
    val minMargin: Float = 0.10f,
    /** Number of candidates shown when the match is ambiguous. */
    val candidateCount: Int = 3,
    /** Upper bound for adding a recognized face as a new sample (higher means it adds nothing new). */
    val adaptiveMaxScore: Float = 0.62f,
)

/** Outcome of comparing one face against the database. */
sealed class Decision {
    /** No registered person is similar enough. [best] is kept for logging. */
    data class Unregistered(val best: Match?) : Decision()

    /** One person stands out clearly. */
    data class Confirmed(val match: Match, val margin: Float) : Decision()

    /** Several people are similar; the user picks one. */
    data class Ambiguous(val candidates: List<Match>) : Decision()
}

object RecognitionPolicy {

    fun decide(rankings: List<Match>, thresholds: RecognitionThresholds): Decision {
        val top = rankings.firstOrNull() ?: return Decision.Unregistered(null)
        if (top.score < thresholds.minScore) return Decision.Unregistered(top)
        val second = rankings.getOrNull(1)
        val margin = if (second != null) top.score - second.score else top.score
        if (margin >= thresholds.minMargin || second == null) return Decision.Confirmed(top, margin)
        return Decision.Ambiguous(rankings.take(thresholds.candidateCount.coerceAtLeast(2)))
    }

    /**
     * A confirmed face is worth keeping as an extra sample only when it is recognizably the person
     * but not already well represented (score below [RecognitionThresholds.adaptiveMaxScore]).
     */
    fun isAdaptiveSample(decision: Decision, thresholds: RecognitionThresholds): Boolean =
        decision is Decision.Confirmed &&
            decision.match.score >= thresholds.minScore &&
            decision.match.score <= thresholds.adaptiveMaxScore
}
