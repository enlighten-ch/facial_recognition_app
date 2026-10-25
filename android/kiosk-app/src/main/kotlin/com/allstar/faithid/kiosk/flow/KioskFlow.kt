package com.allstar.faithid.kiosk.flow

import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.AdaptiveSampler
import com.allstar.faithid.core.recognition.CandidateStabilizer
import com.allstar.faithid.core.recognition.Decision
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.core.recognition.Match
import com.allstar.faithid.core.recognition.RecognitionPolicy
import com.allstar.faithid.core.recognition.RegistrationSession
import java.util.Locale

/**
 * What the kiosk shows and does, independent of Android: recognition decisions, registration,
 * candidate selection, check-in and the output countdown.
 *
 * Call everything from one thread (the main thread in the app). Every change is published to
 * [onState]; side effects go through [effects].
 */
class KioskFlow(
    private var config: FlowConfig,
    private val effects: FlowEffects,
    private val onState: (UiState) -> Unit,
) {
    private var index: FaceIndex = FaceIndex.EMPTY
    private var readiness: Readiness = Readiness.Ready
    private var banner: String = ""
    private var frame = 0L

    private var stabilizer = CandidateStabilizer(config.candidateSwitchFrames, config.selectedGraceFrames)
    private var sampler = AdaptiveSampler(config.adaptiveEnabled, config.adaptiveMaxPerSession, config.adaptiveMinIntervalFrames)
    private var session = RegistrationSession(config.registration)
    private var stallCount = 0

    /** Latest ranking per person key in this session, so a tapped candidate can be printed with its score. */
    private var lastMatches: Map<String, Match> = emptyMap()
    private var duplicate: Match? = null

    /** Person confirmed through "본인이 맞아요"; their check-in is recorded as such. */
    private var duplicateConfirmedKey: String? = null

    var state: UiState = UiState()
        private set

    private val messages get() = config.messages

    init {
        showIdle()
    }

    fun reconfigure(next: FlowConfig) {
        config = next
        stabilizer = CandidateStabilizer(next.candidateSwitchFrames, next.selectedGraceFrames)
        sampler = AdaptiveSampler(next.adaptiveEnabled, next.adaptiveMaxPerSession, next.adaptiveMinIntervalFrames)
        session = RegistrationSession(next.registration)
        goHome()
    }

    fun updateIndex(next: FaceIndex) {
        index = next
    }

    fun updateReadiness(next: Readiness, bannerText: String) {
        val changed = next != readiness
        readiness = next
        banner = bannerText
        if (changed && state.screen == Screen.MONITOR) showIdle() else publish(state.copy(banner = banner))
    }

    // ----- camera ---------------------------------------------------------------------------

    fun onObservation(observation: Observation) {
        frame++
        when (state.screen) {
            Screen.MONITOR -> monitor(observation)
            Screen.REGISTER_CAPTURE -> capture(observation)
            Screen.REGISTER_INFO, Screen.OUTPUT -> Unit
        }
    }

    private fun monitor(observation: Observation) {
        if (readiness != Readiness.Ready || state.duplicateOf != null) return
        when (observation) {
            Observation.NoFace -> {
                val kind = CandidateStabilizer.Observation(CandidateStabilizer.Kind.NO_FACE)
                if (stabilizer.shouldHold(kind)) return
                stabilizer.onShown(kind)
                lastMatches = emptyMap()
                showIdle()
            }
            is Observation.Face -> {
                val rankings = index.rank(observation.embedding, config.thresholds.candidateCount.coerceAtLeast(2))
                lastMatches = lastMatches + rankings.associateBy { it.personKey } // keep earlier ones: a tapped candidate may drop out of one frame
                val decision = RecognitionPolicy.decide(rankings, config.thresholds)
                val kind = when (decision) {
                    is Decision.Unregistered -> CandidateStabilizer.Observation(CandidateStabilizer.Kind.UNREGISTERED)
                    is Decision.Confirmed -> CandidateStabilizer.Observation(CandidateStabilizer.Kind.CONFIRMED, listOf(decision.match.personKey))
                    is Decision.Ambiguous -> CandidateStabilizer.Observation(CandidateStabilizer.Kind.AMBIGUOUS, decision.candidates.map { it.personKey })
                }
                if (decision is Decision.Confirmed &&
                    RecognitionPolicy.isAdaptiveSample(decision, config.thresholds) &&
                    sampler.shouldTake(frame)
                ) {
                    effects.addSamples(decision.match.personKey, listOf(observation.embedding))
                    effects.log("자동 보정 샘플: ${decision.match.label} (점수 ${score(decision.match.score)})")
                }
                if (stabilizer.shouldHold(kind)) return
                stabilizer.onShown(kind)
                logDecision(decision)
                when (decision) {
                    is Decision.Unregistered -> showUnregistered()
                    is Decision.Confirmed -> showConfirmed(decision.match)
                    is Decision.Ambiguous -> showAmbiguous(decision.candidates)
                }
            }
        }
    }

    private fun capture(observation: Observation) {
        if (state.duplicateOf != null) return
        val expected = session.nextDirection ?: return
        val message = when (observation) {
            Observation.NoFace -> {
                stallCount++
                "얼굴을 찾지 못했습니다. 화면 중앙의 타원 안으로 와주세요."
            }
            is Observation.Face -> when (val outcome = session.offer(observation.direction, observation.embedding)) {
                is RegistrationSession.Outcome.Accepted -> {
                    stallCount = 0
                    if (session.isComplete) {
                        finishCapture()
                        return
                    }
                    "좋아요! 샘플을 저장했습니다."
                }
                is RegistrationSession.Outcome.WrongDirection -> {
                    stallCount++
                    "현재 방향: ${outcome.detected?.label ?: "정면"}"
                }
                is RegistrationSession.Outcome.TooSimilar -> {
                    stallCount++
                    "조금 더 다른 각도로 움직여 주세요."
                }
                is RegistrationSession.Outcome.TooDifferent -> {
                    stallCount++
                    "같은 분이 계속 카메라 앞에 있어야 합니다."
                }
                RegistrationSession.Outcome.AlreadyComplete -> return
            }
        }
        val next = session.nextDirection ?: expected
        val hint = if (stallCount >= 5) "\n잘 안되면 카메라에 조금 더 가까이 오거나 멀어져 보세요." else ""
        publish(
            state.copy(
                message = "${next.prompt}$hint",
                detail = message,
                registration = registrationView(),
            ),
        )
    }

    private fun finishCapture() {
        val match = session.likelyDuplicateOf(index)
        if (match != null) {
            duplicate = match
            effects.log("등록 중복 의심: ${match.label} (점수 ${score(match.score)})")
            publish(
                state.copy(
                    message = messages.duplicateFound.fill("label" to match.label, "name" to match.name),
                    detail = messages.duplicateDetail,
                    registration = registrationView(),
                    duplicateOf = CandidateView(match.personKey, match.label, match.score, true),
                ),
            )
        } else {
            showRegisterInfo(labelOnly = false)
        }
    }

    private var lastDecisionLog = ""

    /** One log line whenever the shown result changes, with scores, for checking recognition in the field. */
    private fun logDecision(decision: Decision) {
        val line = when (decision) {
            is Decision.Unregistered -> "인식: 미등록" + (decision.best?.let { " (가장 비슷한 ${it.label} ${score(it.score)})" } ?: "")
            is Decision.Confirmed -> "인식: ${decision.match.label} 점수 ${score(decision.match.score)}, 2위와 차이 ${score(decision.margin)}"
            is Decision.Ambiguous -> "인식: 후보 " + decision.candidates.joinToString(", ") { "${it.label} ${score(it.score)}" }
        }
        val key = when (decision) {
            is Decision.Unregistered -> "unregistered"
            is Decision.Confirmed -> decision.match.personKey
            is Decision.Ambiguous -> decision.candidates.joinToString { it.personKey }
        }
        if (key != lastDecisionLog) effects.log(line)
        lastDecisionLog = key
    }

    // ----- user actions ----------------------------------------------------------------------

    fun startRegistration() {
        if (state.screen != Screen.MONITOR || readiness != Readiness.Ready) return
        session.reset()
        stallCount = 0
        duplicate = null
        val first = session.nextDirection ?: HeadDirection.LEFT
        publish(
            UiState(
                screen = Screen.REGISTER_CAPTURE,
                message = first.prompt,
                detail = "얼굴을 타원 안에 맞추고 안내에 따라 천천히 움직여 주세요.",
                registration = registrationView(),
                banner = banner,
            ),
        )
    }

    fun startLabelOnly() {
        if (state.screen != Screen.MONITOR) return
        showRegisterInfo(labelOnly = true)
    }

    fun selectCandidate(personKey: String) {
        if (state.screen != Screen.MONITOR) return
        val match = lastMatches[personKey] ?: return
        stabilizer.select(personKey)
        publish(
            state.copy(
                detail = messages.selectedCandidate.fill("label" to match.label),
                candidates = state.candidates.map { it.copy(selected = it.personKey == personKey) },
                printButtonEnabled = true,
            ),
        )
    }

    /** The print button: checks the person in, then prints their label. */
    fun pressPrint() {
        if (state.screen != Screen.MONITOR) return
        val key = stabilizer.selectedKey ?: return
        val match = lastMatches[key] ?: return
        val metadata = index.metadataOf(key) ?: match.metadata
        val source = when {
            key == duplicateConfirmedKey -> CheckInSource.DUPLICATE_CONFIRMED
            state.candidates.size > 1 -> CheckInSource.CANDIDATE_SELECTED
            else -> CheckInSource.RECOGNIZED
        }
        effects.checkIn(CheckIn(key, metadata, source, match.score))
        showOutput(metadata)
    }

    /** Name page: save a registration, or print a label without registering. */
    fun submitInfo(name: String, department: String, grade: String) {
        if (state.screen != Screen.REGISTER_INFO || state.saving) return
        val metadata = PersonMetadata.of(name, department, grade)
        if (metadata.name.isBlank()) {
            publish(state.copy(detail = "이름을 입력해 주세요."))
            return
        }
        if (state.labelOnly) {
            effects.checkIn(CheckIn("", metadata, CheckInSource.LABEL_ONLY, 0f))
            showOutput(metadata)
            return
        }
        val samples = session.samples
        if (samples.size < session.totalTarget) {
            publish(state.copy(detail = "얼굴 샘플이 부족합니다. 처음부터 다시 등록해 주세요."))
            return
        }
        publish(state.copy(saving = true))
        val key = effects.register(metadata, samples)
        if (key == null) {
            publish(state.copy(saving = false, detail = "저장하지 못했습니다. 다시 시도해 주세요."))
            return
        }
        effects.log("새 등록: ${metadata.label()} (샘플 ${samples.size}개)")
        goHome()
    }

    /** "본인이 맞아요" after the duplicate warning: the new samples improve the existing person. */
    fun confirmDuplicate() {
        val match = duplicate ?: return
        effects.addSamples(match.personKey, session.samples)
        duplicate = null
        duplicateConfirmedKey = match.personKey
        lastMatches = mapOf(match.personKey to match)
        stabilizer.reset()
        stabilizer.onShown(CandidateStabilizer.Observation(CandidateStabilizer.Kind.CONFIRMED, listOf(match.personKey)))
        showConfirmed(match)
    }

    /** "처음 등록이에요": a different person who only looks similar; continue to the name page. */
    fun rejectDuplicate() {
        if (duplicate == null) return
        duplicate = null
        showRegisterInfo(labelOnly = false)
    }

    fun updatePrintStatus(text: String) {
        val output = state.output ?: return
        publish(state.copy(output = output.copy(printStatus = text)))
    }

    /** Called once a second by the app. Counts down the output screen. */
    fun onSecond() {
        val output = state.output ?: return
        val left = output.secondsLeft - 1
        if (left <= 0) goHome() else publish(state.copy(output = output.copy(secondsLeft = left), message = messages.returnCountdown.fill("seconds" to left.toString())))
    }

    fun goHome() {
        session.reset()
        stabilizer.reset()
        sampler.resetSession()
        duplicate = null
        lastMatches = emptyMap()
        duplicateConfirmedKey = null
        showIdle()
    }

    // ----- states ---------------------------------------------------------------------------

    private fun showIdle() {
        val base = UiState(screen = Screen.MONITOR, banner = banner)
        publish(
            when (val r = readiness) {
                Readiness.Ready -> base.copy(message = messages.welcome)
                is Readiness.WaitingForApproval -> base.copy(
                    message = messages.awaitingApproval.fill("clientId" to r.clientId),
                    detail = messages.awaitingApprovalDetail,
                )
                is Readiness.NoServer -> base.copy(message = messages.serverDisconnected, detail = "${messages.serverDisconnectedDetail}\n${r.reason}")
            },
        )
    }

    private fun showUnregistered() {
        publish(UiState(screen = Screen.MONITOR, message = messages.unregistered, detail = messages.unregisteredDetail, showRegisterChoice = true, banner = banner))
    }

    private fun showConfirmed(match: Match) {
        publish(
            UiState(
                screen = Screen.MONITOR,
                message = messages.recognized.fill("name" to match.name, "label" to match.label, "department" to match.metadata.department, "grade" to match.metadata.grade),
                candidates = listOf(CandidateView(match.personKey, match.label, match.score, true)),
                printButtonVisible = true,
                printButtonEnabled = true,
                banner = banner,
            ),
        )
    }

    private fun showAmbiguous(candidates: List<Match>) {
        val selected = stabilizer.selectedKey?.takeIf { key -> candidates.any { it.personKey == key } }
        publish(
            UiState(
                screen = Screen.MONITOR,
                message = messages.ambiguous,
                detail = selected?.let { key -> messages.selectedCandidate.fill("label" to candidates.first { it.personKey == key }.label) } ?: messages.ambiguousDetail,
                candidates = candidates.map { CandidateView(it.personKey, it.label, it.score, it.personKey == selected) },
                printButtonVisible = true,
                printButtonEnabled = selected != null,
                banner = banner,
            ),
        )
    }

    private fun showRegisterInfo(labelOnly: Boolean) {
        publish(
            UiState(
                screen = Screen.REGISTER_INFO,
                message = if (labelOnly) messages.labelOnlyInput else messages.registerComplete,
                labelOnly = labelOnly,
                banner = banner,
            ),
        )
    }

    private fun showOutput(metadata: PersonMetadata) {
        effects.print(metadata)
        val seconds = config.outputReturnSeconds.coerceAtLeast(1)
        publish(
            UiState(
                screen = Screen.OUTPUT,
                message = messages.returnCountdown.fill("seconds" to seconds.toString()),
                output = OutputView(metadata, seconds, "출력을 준비하고 있습니다."),
                banner = banner,
            ),
        )
    }

    private fun registrationView() = RegistrationView(
        collected = HeadDirection.entries.associateWith { session.collected(it) },
        perDirection = session.target,
        total = session.totalCollected,
        target = session.totalTarget,
        next = session.nextDirection,
    )

    private fun publish(next: UiState) {
        if (next == state) return
        state = next
        onState(next)
    }

    private fun score(value: Float) = String.format(Locale.US, "%.3f", value)
}
