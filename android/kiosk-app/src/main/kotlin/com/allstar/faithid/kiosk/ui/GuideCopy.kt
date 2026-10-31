package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState

/**
 * What the guide (빛이) says on the operation screen: a friendly tone for children, a polite one
 * for adults. Lines changed in the settings ("안내 문구", [overrides] keyed "kids.idle") replace the
 * defaults. Messages about the server connection come from the flow as they are.
 */
class GuideCopy(
    private val kids: Boolean,
    private val overrides: Map<String, String> = emptyMap(),
) {
    data class Line(val title: String, val sub: String = "")

    /** One editable line: its key, the label in the settings and the default for each audience. */
    data class Entry(val key: String, val label: String, val kids: String, val adults: String)

    fun line(key: String, name: String? = null, direction: HeadDirection? = null): String {
        val entry = LINES.first { it.key == key }
        val text = overrides["${audienceKey(kids)}.$key"]?.takeIf { it.isNotBlank() } ?: if (kids) entry.kids else entry.adults
        return text
            .replace("{name}", name?.let { if (kids) callName(it) else it } ?: "")
            .replace("{direction}", direction?.let { directionPhrase(it) } ?: "")
    }

    // ----- speech bubble --------------------------------------------------------------------

    fun bubble(state: UiState): Line = when (state.screen) {
        Screen.MONITOR -> monitor(state)
        Screen.REGISTER_CAPTURE -> capture(state)
        Screen.REGISTER_INFO -> info(state)
        Screen.OUTPUT -> output(state)
    }

    private fun monitor(state: UiState): Line {
        if (!state.ready) return Line(state.message, state.detail)
        state.registered?.let { return Line(line("registered", it.name), line("registeredSub", it.name)) }
        if (state.showRegisterChoice) return Line(line("unknown"), line("unknownSub"))
        if (state.candidates.size > 1) {
            val chosen = state.candidates.firstOrNull { it.selected }?.metadata
            return if (chosen != null) greeting(chosen) else Line(line("twins"), line("twinsSub"))
        }
        state.candidates.singleOrNull()?.metadata?.let { return greeting(it) }
        return Line(line("idle"), line("idleSub"))
    }

    private fun greeting(person: PersonMetadata): Line {
        val where = listOf(person.department, person.grade).filter { it.isNotBlank() }.joinToString(" · ")
        return Line(line("hello", person.name), if (kids) line("helloSub") else where)
    }

    private fun capture(state: UiState): Line {
        val duplicate = state.duplicateOf
        if (duplicate != null) return Line(line("confirm", duplicate.metadata?.name ?: duplicate.label), line("confirmSub"))
        val r = state.registration ?: return Line(state.message)
        val next = r.next ?: return Line(line("celebration"))
        val key = when (r.circle ?: FaceCircle.NORMAL) {
            FaceCircle.NORMAL -> "fitNormal"
            FaceCircle.BIG -> "fitBig"
            FaceCircle.SMALL -> "fitSmall"
        }
        return Line(line(key, direction = next))
    }

    private fun info(state: UiState): Line {
        val base = if (state.labelOnly) Line(line("labelOnly"), line("labelOnlySub")) else Line(line("regDone"), line("regDoneSub"))
        // A problem with the input ("이름을 입력해 주세요.") replaces the second line.
        return if (state.detail.isNotBlank() && !state.saving) base.copy(sub = state.detail) else base
    }

    private fun output(state: UiState): Line {
        val output = state.output ?: return Line(state.message)
        return Line(line("done", output.metadata.name), countdown(output.secondsLeft))
    }

    fun countdown(seconds: Int): String = if (kids) "${seconds}초 뒤에 처음으로 돌아가요" else "${seconds}초 후 처음 화면으로 돌아갑니다"

    /** "왼쪽을" for children ("왼쪽을 보면서 …"), "왼쪽으로" for adults ("고개를 왼쪽으로 돌린 채 …"). */
    fun directionPhrase(d: HeadDirection): String {
        val word = directionWord(d)
        val vertical = d == HeadDirection.UP || d == HeadDirection.DOWN
        return if (kids) word + if (vertical) "를" else "을" else word + if (vertical) "로" else "으로"
    }

    fun directionWord(d: HeadDirection): String = when (d) {
        HeadDirection.LEFT -> "왼쪽"
        HeadDirection.RIGHT -> "오른쪽"
        HeadDirection.UP -> "위"
        HeadDirection.DOWN -> "아래"
    }

    // ----- buttons and small texts ------------------------------------------------------------

    val print: String get() = line("print")
    val register: String get() = if (kids) "좋아, 등록할래!" else "얼굴 등록하기"
    val labelOnly: String get() = if (kids) "오늘은 이름표만" else "등록 없이 이름표만"
    val notMe: String get() = if (kids) "나 아닌데?" else "본인이 아닙니다"
    fun noneOfThem(count: Int): String = if (kids) (if (count == 2) "둘 다 아니에요" else "모두 아니에요") else "해당 없음"
    val isMe: String get() = if (kids) "네, 맞아요" else "네, 본인입니다"
    val firstTime: String get() = if (kids) "아니에요, 처음이에요" else "아니요, 처음입니다"
    val save: String get() = if (kids) "다 됐어요" else "저장"
    val home: String get() = "처음으로"
    val privacy: String get() = if (kids) "사진은 저장하지 않아요" else "사진은 저장하지 않습니다"
    val stalled: String get() = if (kids) "카메라에 조금 더 가까이 오거나 멀어져 볼까?" else "카메라에 조금 더 가까이 오거나 멀어져 보세요"
    val followTitle: String get() = if (kids) "이렇게 따라 해 봐!" else "이렇게 돌려 주세요"
    val celebration: String get() = line("celebration")

    fun count(got: Int, target: Int): String = if (kids) "★ 별 $got / $target" else "$got / $target 동작"

    fun followCaption(d: HeadDirection?, circle: FaceCircle?): String {
        if (d == null) return if (kids) "최고야!" else "정면"
        val size = when (circle) {
            FaceCircle.BIG -> "큰 동그라미"
            FaceCircle.SMALL -> "작은 동그라미"
            else -> "보통 동그라미"
        }
        return "${directionWord(d)} · $size"
    }

    /** "하은아" / "준호야" for children, "김하은 님" for adults. */
    fun callName(name: String): String {
        if (!kids) return "$name 님"
        val given = givenName(name) ?: return name
        return given + if (hasFinalConsonant(given.last())) "아" else "야"
    }

    companion object {
        const val KIDS = "kids"
        const val ADULTS = "adults"

        fun audienceKey(kids: Boolean) = if (kids) KIDS else ADULTS

        /** Every line the settings can change. {name} is the person, {direction} the way to look. */
        val LINES = listOf(
            Entry("idle", "기본 안내", "안녕! 나는 빛이야. 동그라미 안에 얼굴을 보여 줄래?", "안녕하세요. 화면을 바라봐 주세요"),
            Entry("idleSub", "기본 안내 둘째 줄", "내가 이름을 찾아 줄게", "얼굴로 출석을 확인합니다"),
            Entry("hello", "인식 성공 ({name})", "반가워, {name}!", "{name} 님, 환영합니다"),
            Entry("helloSub", "인식 성공 둘째 줄 (어린이)", "오늘도 와 줘서 고마워", "오늘도 와 주셔서 감사합니다"),
            Entry("twins", "닮은 얼굴", "어? 닮은 친구가 있네!", "비슷한 분이 여러 명 있습니다"),
            Entry("twinsSub", "닮은 얼굴 둘째 줄", "누구인지 알려 줄래? 내 이름을 눌러 줘", "본인의 이름을 선택해 주세요"),
            Entry("unknown", "처음 보는 얼굴", "처음 보는 얼굴이네! 우리 친구 할까?", "처음 오셨나요?"),
            Entry("unknownSub", "처음 보는 얼굴 둘째 줄", "얼굴을 등록하면 다음부터 바로 이름표가 나와", "얼굴을 등록하면 다음부터 바로 출석됩니다"),
            Entry("fitNormal", "등록: 보통 동그라미 ({direction})", "{direction} 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?", "고개를 {direction} 돌린 채 원 안에 얼굴을 맞춰 주세요"),
            Entry("fitBig", "등록: 큰 동그라미 ({direction})", "우아, 동그라미가 커졌다! {direction} 보면서 얼굴로 꽉 채워 줄래?", "고개를 {direction} 돌린 채 커진 원을 얼굴로 채워 주세요"),
            Entry("fitSmall", "등록: 작은 동그라미 ({direction})", "이번엔 작은 동그라미야! {direction} 보면서 쏙 들어와 볼까?", "고개를 {direction} 돌린 채 작아진 원 안에 얼굴을 넣어 주세요"),
            Entry("celebration", "등록 축하", "별 12개 다 모았다!", "얼굴 등록 완료"),
            Entry("confirm", "이미 아는 얼굴 ({name})", "{name} 맞아요?", "{name} 님이 맞으신가요?"),
            Entry("confirmSub", "이미 아는 얼굴 둘째 줄", "어, 이미 아는 얼굴이야! 맞으면 네를 눌러 줘", "이미 등록된 얼굴과 같습니다"),
            Entry("regDone", "이름 입력", "별 12개 다 모았다!", "얼굴 등록이 끝났습니다"),
            Entry("regDoneSub", "이름 입력 둘째 줄", "선생님, 이름을 적어 주세요", "이름과 소속을 입력해 주세요"),
            Entry("labelOnly", "등록 없이 이름표만", "이름표에 뭐라고 쓸까?", "이름표에 넣을 내용"),
            Entry("labelOnlySub", "등록 없이 이름표만 둘째 줄", "얼굴은 저장하지 않아요", "얼굴은 저장하지 않습니다"),
            Entry("registered", "등록 완료 ({name})", "등록 완료! 반가워, {name}!", "등록되었습니다"),
            Entry("registeredSub", "등록 완료 둘째 줄", "이제 다음부터 바로 이름표가 나와", "다음부터는 바로 출석됩니다"),
            Entry("done", "이름표가 나온 뒤 ({name})", "잘 왔어, {name}! 이름표를 가져가요", "출석되었습니다. 이름표를 가져가세요"),
            Entry("print", "이름표 버튼", "이름표 받기", "이름표 출력"),
        )

        /** The given name of a Korean name of 2-4 syllables ("김하은" → "하은"); null for other names. */
        fun givenName(name: String): String? {
            val n = name.trim()
            if (n.length !in 2..4 || !n.all { it in '가'..'힣' }) return null
            return if (n.length == 2) n else n.drop(1)
        }

        private fun hasFinalConsonant(c: Char): Boolean = c in '가'..'힣' && (c - '가') % 28 != 0
    }
}
