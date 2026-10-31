package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.flow.CandidateView
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

    @Test
    fun `registration says direction and circle in one line`() {
        fun capture(next: HeadDirection, circle: FaceCircle) = UiState(
            screen = Screen.REGISTER_CAPTURE,
            registration = RegistrationView(emptyMap(), 3, 1, 12, next, circle),
        )
        assertEquals("왼쪽을 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?", kids.bubble(capture(HeadDirection.LEFT, FaceCircle.NORMAL)).title)
        assertEquals("이번엔 작은 동그라미야! 위를 보면서 쏙 들어와 볼까?", kids.bubble(capture(HeadDirection.UP, FaceCircle.SMALL)).title)
        assertEquals("고개를 아래로 돌린 채 커진 원을 얼굴로 채워 주세요", adults.bubble(capture(HeadDirection.DOWN, FaceCircle.BIG)).title)
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
    fun `server messages are shown as the flow wrote them`() {
        val waiting = UiState(ready = false, message = "이 키오스크(client-1)는 서버 승인을 기다리고 있습니다", detail = "서버 앱에서 허락해 주세요.")
        assertEquals(GuideCopy.Line(waiting.message, waiting.detail), kids.bubble(waiting))
    }

    @Test
    fun `a problem with the name replaces the second line of the form`() {
        val form = UiState(screen = Screen.REGISTER_INFO, message = "x", detail = "이름을 입력해 주세요.")
        assertEquals(GuideCopy.Line("별 12개 다 모았다!", "이름을 입력해 주세요."), kids.bubble(form))
    }
}
