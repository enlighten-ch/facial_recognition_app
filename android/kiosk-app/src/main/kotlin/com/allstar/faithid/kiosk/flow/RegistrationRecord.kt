package com.allstar.faithid.kiosk.flow

import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.RegistrationSession
import java.util.Locale

/**
 * Counts what happened during one registration, for the kiosk log: how long it took, why frames
 * were not kept, and each kept sample's size, head angles and similarity. Used to tune the
 * registration limits from real use; no faces or names are recorded.
 */
internal class RegistrationRecord(private val startedAt: Long) {
    private var frames = 0
    private var kept = 0
    private var holding = 0
    private val problems = LinkedHashMap<CaptureProblem, Int>()

    fun count(problem: CaptureProblem?, held: Boolean) {
        frames++
        if (problem != null) problems[problem] = (problems[problem] ?: 0) + 1 else if (held) holding++
    }

    /** One line per kept sample. */
    fun accepted(outcome: RegistrationSession.Outcome.Accepted, face: Observation.Face, total: Int, target: Int): String {
        frames++
        kept++
        val cell = outcome.cell
        val parts = buildList {
            add("${cell.direction.progressLabel.removeSuffix(" 얼굴")}·${circleName(cell.circle)}")
            face.size?.let { add("크기 ${f(it)}") }
            face.angles?.let { add("고개 좌우 ${f(it.yaw)} 위아래 ${f(it.pitch)}") }
            add("유사도 ${outcome.maxSimilarity?.let { f(it) } ?: "-"}")
        }
        return "등록 샘플 $total/$target: ${parts.joinToString(", ")}"
    }

    fun summary(now: Long, finished: Boolean): String {
        val seconds = ((now - startedAt) / 1000).coerceAtLeast(0)
        val reasons = CaptureProblem.entries.mapNotNull { p -> problems[p]?.let { "${label(p)} $it" } }
        val head = if (finished) "등록 기록" else "등록 중단"
        return "$head: ${seconds}초, 프레임 $frames — 저장 $kept, 머무는 중 $holding" +
            (if (reasons.isEmpty()) "" else ", " + reasons.joinToString(", "))
    }

    private fun label(p: CaptureProblem) = when (p) {
        CaptureProblem.NO_FACE -> "얼굴 없음"
        CaptureProblem.OUTSIDE_LEFT -> "원 왼쪽 밖"
        CaptureProblem.OUTSIDE_RIGHT -> "원 오른쪽 밖"
        CaptureProblem.SEVERAL -> "여럿"
        CaptureProblem.TOO_TURNED -> "너무 돌림"
        CaptureProblem.DIRECTION -> "방향"
        CaptureProblem.COME_CLOSER -> "더 가까이"
        CaptureProblem.STEP_BACK -> "더 멀리"
        CaptureProblem.TOO_SIMILAR -> "너무 비슷"
        CaptureProblem.OTHER_PERSON -> "다른 사람"
    }

    private fun circleName(c: FaceCircle) = when (c) {
        FaceCircle.NORMAL -> "보통"
        FaceCircle.BIG -> "가까이"
        FaceCircle.SMALL -> "멀리"
    }

    private fun f(v: Float) = String.format(Locale.US, "%.2f", v)
}
