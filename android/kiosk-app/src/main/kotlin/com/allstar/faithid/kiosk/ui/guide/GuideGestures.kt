package com.allstar.faithid.kiosk.ui.guide

/** One small movement of 빛이. */
enum class Gesture(val label: String) {
    BLINK("눈 깜빡"),
    DOUBLE_BLINK("두 번 깜빡"),
    GLANCE("두리번"),
    HOP("살짝 점프"),
    DOUBLE_HOP("콩콩 두 번 점프"),
    TILT("갸우뚱"),
    NOD("끄덕"),
    WIGGLE("좌우로 흔들흔들"),
    SQUISH("몸을 움츠렸다 펴기"),
    SLEEPY("졸린 눈"),
}

/** The moments of the kiosk screen 빛이 reacts to. */
enum class GuideMoment(val label: String) {
    IDLE("아무도 없을 때 (가끔)"),
    UNKNOWN_FACE("처음 보는 얼굴"),
    RECOGNIZED("이름을 찾았을 때"),
    LOOK_ALIKE("닮은 얼굴이 여럿일 때"),
    REGISTER_START("얼굴 등록 시작"),
    STAR_COLLECTED("별 하나를 모았을 때"),
    STALLED("등록이 잘 안될 때"),
    REGISTER_DONE("별 12개를 다 모았을 때"),
    ALREADY_KNOWN("이미 등록된 얼굴일 때"),
    NAME_FORM("이름을 적을 때"),
    SAVED("등록을 저장했을 때"),
    NAME_TAG("이름표가 나올 때"),
    OFFLINE("서버를 기다릴 때"),
}

/**
 * Which gestures 빛이 makes at each moment: positive ones (a hop, a nod) when things go well, a
 * tilt of the head when it does not know someone or something does not work. Change the table to
 * change 빛이's behaviour; the screen only reports moments.
 */
object GuideGestures {
    val TABLE: Map<GuideMoment, List<Gesture>> = linkedMapOf(
        GuideMoment.IDLE to listOf(Gesture.GLANCE, Gesture.HOP, Gesture.DOUBLE_BLINK, Gesture.WIGGLE),
        GuideMoment.UNKNOWN_FACE to listOf(Gesture.TILT),
        GuideMoment.RECOGNIZED to listOf(Gesture.HOP),
        GuideMoment.LOOK_ALIKE to listOf(Gesture.TILT, Gesture.GLANCE),
        GuideMoment.REGISTER_START to listOf(Gesture.NOD),
        GuideMoment.STAR_COLLECTED to listOf(Gesture.NOD),
        GuideMoment.STALLED to listOf(Gesture.TILT),
        GuideMoment.REGISTER_DONE to listOf(Gesture.DOUBLE_HOP),
        GuideMoment.ALREADY_KNOWN to listOf(Gesture.TILT),
        GuideMoment.NAME_FORM to listOf(Gesture.NOD),
        GuideMoment.SAVED to listOf(Gesture.DOUBLE_HOP),
        GuideMoment.NAME_TAG to listOf(Gesture.HOP, Gesture.WIGGLE),
        GuideMoment.OFFLINE to listOf(Gesture.SLEEPY),
    )

    /** For [GuideMoment.IDLE] one gesture is picked at random; for the others all play in order. */
    fun gesturesFor(moment: GuideMoment, pick: (Int) -> Int = { (0 until it).random() }): List<Gesture> {
        val list = TABLE[moment].orEmpty()
        if (list.isEmpty()) return emptyList()
        return if (moment == GuideMoment.IDLE) listOf(list[pick(list.size)]) else list
    }

    /** 빛이 blinks on its own every few seconds; an idle gesture plays now and then. */
    const val BLINK_MIN_MS = 2_500L
    const val BLINK_MAX_MS = 5_500L
    const val IDLE_GESTURE_MIN_MS = 9_000L
    const val IDLE_GESTURE_MAX_MS = 16_000L
}
