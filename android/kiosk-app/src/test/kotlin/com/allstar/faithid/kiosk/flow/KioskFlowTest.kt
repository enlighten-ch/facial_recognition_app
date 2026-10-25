package com.allstar.faithid.kiosk.flow

import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.core.recognition.HeadDirection
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
        override fun log(message: String) {}
    }

    private fun flow(
        people: List<Person> = listOf(hong, kim),
        config: FlowConfig = FlowConfig(candidateSwitchFrames = 1, selectedGraceFrames = 1, registration = RegistrationConfig(samplesPerDirection = 1)),
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
        flow.onObservation(face("newcomer", HeadDirection.UP))
        assertEquals(0, flow.state.registration!!.total)
        assertTrue(flow.state.detail.contains("현재 방향"))
        HeadDirection.entries.forEach { flow.onObservation(face("newcomer", it)) }
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        flow.submitInfo("  ", "", "")
        assertEquals(Screen.REGISTER_INFO, flow.state.screen)
        flow.submitInfo("새가족", "청년부", "1학년")
        assertEquals("새가족", effects.registered.single().first.name)
        assertEquals(4, effects.registered.single().second)
        assertEquals(Screen.MONITOR, flow.state.screen)
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
        assertEquals("hong" to 4, effects.added.single())
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
