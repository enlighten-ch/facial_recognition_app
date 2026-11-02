package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.flow.CandidateView
import com.allstar.faithid.kiosk.flow.CaptureProblem
import com.allstar.faithid.kiosk.flow.RegistrationView
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState
import org.junit.Assert.assertEquals
import org.junit.Test

class GuideCopyTest {
    private val kids = GuideCopy(kids = true)
    private val adults = GuideCopy(kids = false)
    private val haeun = PersonMetadata.of("김하은", "유년부", "2학년")

    @Test
    fun `children are called by their given name, adults politely`() {
        assertEquals("하은아", kids.callName("김하은"))
        assertEquals("준호야", kids.callName("이준호"))
        assertEquals("E2E-1", kids.callName("E2E-1"))
        assertEquals("김하은 님", adults.callName("김하은"))
    }

    @Test
    fun `a recognized person is greeted in the audience's tone`() {
        val state = UiState(candidates = listOf(CandidateView("k", haeun.label(), 0.7f, true, haeun)), printButtonEnabled = true)
        assertEquals("반가워, 하은아!", kids.bubble(state).title)
        assertEquals("김하은 님, 환영합니다", adults.bubble(state).title)
        assertEquals("유년부 · 2학년", adults.bubble(state).sub)
    }

    private fun capture(next: HeadDirection, circle: FaceCircle, problem: CaptureProblem? = null, holding: Float = 0f, stalled: Boolean = false) = UiState(
        screen = Screen.REGISTER_CAPTURE,
        registration = RegistrationView(emptyList(), 1, 15, next, circle, holding, problem, stalled),
    )

    @Test
    fun `registration says direction and circle in one line`() {
        assertEquals("왼쪽을 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL)).title)
        assertEquals("이번엔 작은 동그라미야! 위를 보면서 쏙 들어와 볼까?", kids.bubble(capture(HeadDirection.UP, FaceCircle.SMALL)).title)
        assertEquals("고개를 아래로 돌린 채 커진 원을 얼굴로 채워 주세요", adults.bubble(capture(HeadDirection.DOWN, FaceCircle.BIG)).title)
        assertEquals("앞을 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?", kids.bubble(capture(HeadDirection.FRONT, FaceCircle.NORMAL)).title)
        assertEquals("정면을 보고 원 안에 얼굴을 맞춰 주세요", adults.bubble(capture(HeadDirection.FRONT, FaceCircle.NORMAL)).title)
    }

    @Test
    fun `registration explains why nothing is collected`() {
        assertEquals(GuideCopy.Line("한 번에 한 친구만 등록할게!", "다른 친구는 뒤에서 잠깐 기다려 줄래?"), kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.SEVERAL)))
        assertEquals("한 번에 한 분씩 등록합니다", adults.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.SEVERAL)).title)
        // Left of the circle on screen: come to the right.
        assertEquals("오른쪽으로 한 걸음!", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.OUTSIDE_LEFT)).sub)
        assertEquals("왼쪽으로 조금 옮겨 주세요", adults.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.OUTSIDE_RIGHT)).sub)
        assertEquals("우와, 너무 많이 돌렸어!", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.TOO_TURNED)).title)
        assertEquals("조금 더 가까이 와 볼까?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.BIG, CaptureProblem.COME_CLOSER)).sub)
        assertEquals("조금 뒤로 물러나 주세요", adults.bubble(capture(HeadDirection.LEFT, FaceCircle.SMALL, CaptureProblem.STEP_BACK)).sub)
        assertEquals("좋아, 그대로! 하나, 둘…", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, holding = 0.5f)).sub)
        // A wrong direction is only explained once it goes on for a while.
        assertEquals("", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.DIRECTION)).sub)
        assertEquals("그림처럼 고개를 돌려 볼까?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.DIRECTION, stalled = true)).sub)
        // A face lost for one frame keeps the request; for longer, 빛이 asks where they went.
        assertEquals("왼쪽을 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.NO_FACE)).title)
        assertEquals("어디 갔어? 다시 동그라미 앞에 와 줄래?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL, CaptureProblem.NO_FACE, stalled = true)).title)
    }

    @Test
    fun `the guide introduces itself by the name set in the settings`() {
        assertEquals("안녕! 나는 빛이야. 동그라미 안에 얼굴을 보여 줄래?", kids.bubble(UiState()).title)
        assertEquals("안녕! 나는 드림이야. 동그라미 안에 얼굴을 보여 줄래?", GuideCopy(kids = true, guideName = "드림이").bubble(UiState()).title)
        assertEquals("안녕! 나는 별이야. 동그라미 안에 얼굴을 보여 줄래?", GuideCopy(kids = true, guideName = "별").bubble(UiState()).title)
        assertEquals("별이", GuideCopy.subject("별"))
        assertEquals("빛이가", GuideCopy.subject("빛이"))
    }

    @Test
    fun `another sample than the suggested one is fine too`() {
        val holding = UiState(screen = Screen.REGISTER_CAPTURE, registration = RegistrationView(emptyList(), 1, 15, HeadDirection.LEFT, FaceCircle.NORMAL, holding = 0.5f, holdingOther = true))
        assertEquals(GuideCopy.Line("오, 그것도 좋아! 그대로 있어 줘", "하나, 둘…"), kids.bubble(holding))
        val kept = UiState(screen = Screen.REGISTER_CAPTURE, registration = RegistrationView(emptyList(), 2, 15, HeadDirection.LEFT, FaceCircle.NORMAL, bonus = true))
        assertEquals(GuideCopy.Line("음… 그것도 좋아!", "별 하나 더 모았어"), kids.bubble(kept))
        assertEquals("그 자세도 좋습니다", adults.bubble(kept).title)
    }

    @Test
    fun `the celebration counts the stars of this registration`() {
        val done = UiState(screen = Screen.REGISTER_CAPTURE, registration = RegistrationView(emptyList(), 15, 15, null))
        assertEquals("별 15개 다 모았다!", kids.bubble(done).title)
        assertEquals("별 12개 다 모았다!", kids.celebration(12))
    }

    @Test
    fun `changed lines replace the defaults for their audience only`() {
        val custom = GuideCopy(kids = true, overrides = mapOf("kids.idle" to "어서 와!", "adults.idle" to "어서 오세요", "kids.hello" to "{name}, 왔구나!"))
        assertEquals("어서 와!", custom.bubble(UiState()).title)
        val state = UiState(candidates = listOf(CandidateView("k", haeun.label(), 0.7f, true, haeun)), printButtonEnabled = true)
        assertEquals("하은아, 왔구나!", custom.bubble(state).title)
        assertEquals("안녕하세요. 화면을 바라봐 주세요", GuideCopy(kids = false, overrides = mapOf("kids.idle" to "어서 와!")).bubble(UiState()).title)
    }

    @Test
    fun `not me is answered with a retry line`() {
        assertEquals(GuideCopy.Line("이런, 다시 찾아볼게!", "동그라미 안에서 잠깐만 기다려 줘"), kids.bubble(UiState(retrying = true)))
        assertEquals("다시 확인하겠습니다", adults.bubble(UiState(retrying = true)).title)
    }

    @Test
    fun `server messages are shown as the flow wrote them`() {
        val waiting = UiState(ready = false, message = "이 키오스크(client-1)는 서버 승인을 기다리고 있습니다", detail = "서버 앱에서 허락해 주세요.")
        assertEquals(GuideCopy.Line(waiting.message, waiting.detail), kids.bubble(waiting))
    }

    @Test
    fun `a problem with the name replaces the second line of the form`() {
        val form = UiState(screen = Screen.REGISTER_INFO, message = "x", detail = "이름을 입력해 주세요.")
        assertEquals(GuideCopy.Line("별을 다 모았다!", "이름을 입력해 주세요."), kids.bubble(form))
    }
}
