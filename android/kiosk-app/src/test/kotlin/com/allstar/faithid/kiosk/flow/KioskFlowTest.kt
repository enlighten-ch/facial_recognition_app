package com.allstar.faithid.kiosk.flow

import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.core.recognition.Pose
import com.allstar.faithid.core.recognition.RegistrationConfig
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class KioskFlowTest {
    private val faces = FakeFaces(seed = 12, dim = 128)
    private val hong = faces.person("홍길동", "청년부", "3학년", key = "hong")
    private val kim = faces.person("김철수", "중등부", "1학년", key = "kim")

    private class RecordingEffects : FlowEffects {
        val added = mutableListOf<Pair<String, Int>>()
        val registered = mutableListOf<Pair<PersonMetadata, Int>>()
        val checkIns = mutableListOf<CheckIn>()
        val printed = mutableListOf<PersonMetadata>()
        override fun addSamples(personKey: String, samples: List<FloatArray>) { added += personKey to samples.size }
        override fun register(metadata: PersonMetadata, samples: List<FloatArray>): String? { registered += metadata to samples.size; return "new-key" }
        override fun checkIn(checkIn: CheckIn) { checkIns += checkIn }
        override fun print(metadata: PersonMetadata) { printed += metadata }
        val logs = mutableListOf<String>()
        override fun log(message: String) { logs += message }
    }

    private fun flow(
        people: List<Person> = listOf(hong, kim),
        // One sample per direction (front included), kept on the first frame: 5 frames register someone.
        config: FlowConfig = FlowConfig(candidateSwitchFrames = 1, selectedGraceFrames = 1, registration = RegistrationConfig(circleSteps = false, samplesPerDirection = 1, holdFrames = 1)),
    ): Pair<KioskFlow, RecordingEffects> {
        val effects = RecordingEffects()
        val flow = KioskFlow(config, effects) {}
        flow.updateIndex(FaceIndex.of(people))
        return flow to effects
    }

    private fun face(identity: String, direction: HeadDirection? = null) = Observation.Face(faces.sample(identity), direction)

    @Test
    fun `a known face shows the print button and pressing it checks in and prints`() {
        val (flow, effects) = flow()
        flow.onObservation(face("hong"))
        assertEquals(Screen.MONITOR, flow.state.screen)
        assertTrue(flow.state.message.contains("홍길동"))
        assertTrue(flow.state.printButtonEnabled)
        flow.pressPrint()
        assertEquals(Screen.OUTPUT, flow.state.screen)
        assertEquals("hong", effects.checkIns.single().personKey)
        assertEquals(CheckInSource.RECOGNIZED, effects.checkIns.single().source)
        assertEquals("홍길동", effects.printed.single().name)
        repeat(3) { flow.onSecond() }
        assertEquals(Screen.MONITOR, flow.state.screen)
    }

    @Test
    fun `check-in is recorded even when nothing is printed`() {
        // The printer is the app's business; the flow records the check-in when the button is pressed.
        val (flow, effects) = flow()
        flow.onObservation(face("kim"))
        flow.pressPrint()
        assertEquals(1, effects.checkIns.size)
    }

    @Test
    fun `an unknown face offers registration and label-only printing`() {
        val (flow, effects) = flow()
        flow.onObservation(face("stranger"))
        assertTrue(flow.state.showRegisterChoice)
        flow.startLabelOnly()
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        assertTrue(flow.state.labelOnly)
        flow.submitInfo("방문자", "", "")
        assertEquals(CheckInSource.LABEL_ONLY, effects.checkIns.single().source)
        assertEquals("", effects.checkIns.single().personKey)
        assertEquals(Screen.OUTPUT, flow.state.screen)
    }

    @Test
    fun `similar faces show candidates and the chosen one is checked in`() {
        // Two people with nearly the same face: an ambiguous ranking.
        val twinA = Person("a", PersonMetadata.of("쌍둥이A", "", ""), faces.samples("twin", 6))
        val twinB = Person("b", PersonMetadata.of("쌍둥이B", "", ""), faces.samples("twin", 6))
        val (flow, effects) = flow(people = listOf(twinA, twinB))
        flow.onObservation(face("twin"))
        assertEquals(2, flow.state.candidates.size)
        assertFalse(flow.state.printButtonEnabled)
        flow.pressPrint()
        assertTrue("nothing happens before a choice", effects.checkIns.isEmpty())
        flow.selectCandidate("b")
        assertTrue(flow.state.printButtonEnabled)
        flow.pressPrint()
        assertEquals("b", effects.checkIns.single().personKey)
        assertEquals(CheckInSource.CANDIDATE_SELECTED, effects.checkIns.single().source)
    }

    @Test
    fun `registration collects every direction then saves with the entered name`() {
        val (flow, effects) = flow()
        flow.onObservation(face("newcomer"))
        flow.startRegistration()
        assertEquals(Screen.REGISTER_CAPTURE, flow.state.screen)
        flow.onObservation(face("newcomer", null)) // between two directions
        assertEquals(0, flow.state.registration!!.total)
        assertEquals(CaptureProblem.DIRECTION, flow.state.registration!!.problem)
        assertTrue(flow.state.detail.contains("현재 방향"))
        HeadDirection.entries.forEach { flow.onObservation(face("newcomer", it)) }
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        flow.submitInfo("  ", "", "")
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        flow.submitInfo("새가족", "청년부", "1학년")
        assertEquals("새가족", effects.registered.single().first.name)
        assertEquals(5, effects.registered.single().second)
        assertTrue(effects.logs.any { it.startsWith("등록 샘플 5/5: 아래쪽·보통") })
        assertTrue(effects.logs.any { it.startsWith("등록 기록: ") && it.contains("저장 5") && it.contains("방향 1") })
        assertEquals(Screen.MONITOR, flow.state.screen)
    }

    private fun sized(direction: HeadDirection?, size: Float, identity: String = "newcomer") = Observation.Face(faces.sample(identity), direction, size)

    private val circles = FlowConfig(candidateSwitchFrames = 1, selectedGraceFrames = 1, registration = RegistrationConfig(holdFrames = 1))

    @Test
    fun `with circle steps the sizes come in rounds and any direction of the round is filled`() {
        val (flow, _) = flow(config = circles)
        flow.startRegistration()
        val r = { flow.state.registration!! }
        assertEquals(15, r().target)
        assertEquals(HeadDirection.FRONT to FaceCircle.NORMAL, r().next to r().circle)
        flow.onObservation(sized(HeadDirection.FRONT, 1f))
        assertEquals(HeadDirection.LEFT to FaceCircle.NORMAL, r().next to r().circle)
        flow.onObservation(sized(HeadDirection.LEFT, 0.85f))
        assertEquals(CaptureProblem.COME_CLOSER, r().problem)
        assertTrue(flow.state.detail.contains("가까이"))
        // A big face looking left fits the next round, not this one: refused.
        flow.onObservation(sized(HeadDirection.LEFT, 1.18f))
        assertEquals(CaptureProblem.STEP_BACK, r().problem)
        assertEquals(1, r().total)
        // Looking up at the right size is kept although the guide asked for left.
        flow.onObservation(sized(HeadDirection.UP, 1f))
        assertEquals(2, r().total)
        assertEquals(HeadDirection.LEFT to FaceCircle.NORMAL, r().next to r().circle)
        listOf(HeadDirection.LEFT, HeadDirection.RIGHT, HeadDirection.DOWN).forEach { flow.onObservation(sized(it, 1f)) }
        assertEquals(HeadDirection.FRONT to FaceCircle.BIG, r().next to r().circle)
        flow.onObservation(sized(HeadDirection.FRONT, 1.5f))
        assertEquals(CaptureProblem.STEP_BACK, r().problem)
    }

    @Test
    fun `several faces, a face outside the circle or turned too far keep nothing`() {
        val config = circles.copy(registration = RegistrationConfig(holdFrames = 2))
        val (flow, effects) = flow(config = config)
        flow.startRegistration()
        val r = { flow.state.registration!! }
        flow.onObservation(sized(HeadDirection.FRONT, 1f))
        assertEquals(0.5f, r().holding)
        flow.onObservation(Observation.Several(2))
        assertEquals(CaptureProblem.SEVERAL, r().problem)
        assertEquals(0f, r().holding)
        flow.onObservation(sized(HeadDirection.FRONT, 1f))
        assertEquals("the hold starts again after the interruption", 0, r().total)
        flow.onObservation(sized(HeadDirection.FRONT, 1f))
        assertEquals(1, r().total)
        flow.onObservation(Observation.Outside(-1.4f))
        assertEquals(CaptureProblem.OUTSIDE_LEFT, r().problem)
        flow.onObservation(Observation.Face(faces.sample("newcomer"), null, 1f, Pose.TooFar))
        assertEquals(CaptureProblem.TOO_TURNED, r().problem)
        assertEquals(1, r().total)
        flow.goHome()
        val summary = effects.logs.last()
        assertTrue(summary, summary.startsWith("등록 중단: ") && summary.contains("여럿 1") && summary.contains("원 왼쪽 밖 1") && summary.contains("너무 돌림 1"))
    }

    @Test
    fun `not me pauses recognition, then looks again`() {
        val (flow, effects) = flow()
        flow.onObservation(face("hong"))
        assertTrue(flow.state.printButtonEnabled)
        flow.notMe()
        assertTrue(flow.state.retrying)
        assertTrue(flow.state.candidates.isEmpty())
        flow.onObservation(face("hong"))
        assertTrue("frames are ignored while pausing", flow.state.retrying)
        flow.resumeAfterRetry()
        assertFalse(flow.state.retrying)
        flow.onObservation(face("hong"))
        assertTrue(flow.state.printButtonEnabled)
        assertTrue(effects.checkIns.isEmpty())
    }

    @Test
    fun `a saved registration is announced once on the idle screen`() {
        val (flow, _) = flow()
        flow.startRegistration()
        HeadDirection.entries.forEach { flow.onObservation(face("newcomer", it)) }
        flow.submitInfo("새가족", "청년부", "1학년")
        assertEquals("새가족", flow.state.registered?.name)
        flow.onObservation(Observation.NoFace)
        assertNull(flow.state.registered)
    }

    @Test
    fun `registering an already registered face asks whether it is the same person`() {
        val (flow, effects) = flow()
        flow.startRegistration()
        HeadDirection.entries.forEach { flow.onObservation(face("hong", it)) }
        assertNotNull(flow.state.duplicateOf)
        assertEquals("hong", flow.state.duplicateOf!!.personKey)
        assertEquals(Screen.REGISTER_CAPTURE, flow.state.screen)

        flow.confirmDuplicate()
        assertEquals("hong" to 5, effects.added.single())
        assertTrue(effects.registered.isEmpty())
        assertTrue(flow.state.printButtonEnabled)
        flow.pressPrint()
        assertEquals(CheckInSource.DUPLICATE_CONFIRMED, effects.checkIns.single().source)
    }

    @Test
    fun `a look-alike can still register as a new person`() {
        val (flow, effects) = flow()
        flow.startRegistration()
        HeadDirection.entries.forEach { flow.onObservation(face("hong", it)) }
        flow.rejectDuplicate()
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        assertNull(flow.state.duplicateOf)
        flow.submitInfo("홍길동", "고등부", "2학년") // a namesake with a similar face
        assertEquals(1, effects.registered.size)
    }

    @Test
    fun `waiting for approval with no data blocks recognition and explains why`() {
        val (flow, _) = flow(people = emptyList())
        flow.updateIndex(FaceIndex.EMPTY)
        flow.updateReadiness(Readiness.WaitingForApproval("client-1a2b"), "")
        assertTrue(flow.state.message.contains("client-1a2b"))
        assertTrue(flow.state.detail.contains("승인"))
        flow.onObservation(face("hong"))
        assertFalse(flow.state.showRegisterChoice)
        flow.startRegistration()
        assertEquals(Screen.MONITOR, flow.state.screen)
        flow.updateReadiness(Readiness.Ready, "오프라인")
        assertEquals("오프라인", flow.state.banner)
        flow.onObservation(face("stranger"))
        assertTrue(flow.state.showRegisterChoice)
    }

    @Test
    fun `recognized faces add samples with a rate limit`() {
        val config = FlowConfig(candidateSwitchFrames = 1, selectedGraceFrames = 1, adaptiveMinIntervalFrames = 3, adaptiveMaxPerSession = 2)
        val (flow, effects) = flow(config = config.copy(thresholds = config.thresholds.copy(adaptiveMaxScore = 0.99f)))
        repeat(10) { flow.onObservation(face("hong")) }
        assertEquals(2, effects.added.size)
        flow.goHome()
        repeat(3) { flow.onObservation(face("hong")) }
        assertEquals(3, effects.added.size)
    }

    @Test
    fun `home button resets an unfinished registration`() {
        val (flow, effects) = flow()
        flow.startRegistration()
        flow.onObservation(face("x", HeadDirection.LEFT))
        flow.goHome()
        assertEquals(Screen.MONITOR, flow.state.screen)
        flow.startRegistration()
        assertEquals(0, flow.state.registration!!.total)
        assertTrue(effects.registered.isEmpty())
    }
}
