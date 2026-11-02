package com.allstar.faithid.kiosk.settings

import android.content.Context
import android.widget.ScrollView
import com.allstar.faithid.kiosk.flow.FlowMessages
import com.allstar.faithid.kiosk.ui.GuideCopy

private fun scroll(form: SettingsForm) = ScrollView(form.context).apply {
    val pad = form.dp(8)
    form.root.setPadding(pad, pad, pad, pad)
    addView(form.root)
}

/** "일반": who the kiosk talks to, the screen mode and how names are called and printed. */
class GeneralTab(context: Context, s: KioskSettings) : SettingsTab {
    override val title = "일반"
    private val form = SettingsForm(context)
    override val view = scroll(form)

    init { form.section("화면") }
    private val guideName = form.text("안내 캐릭터 이름 (예: 빛이, 별이, 드림이)", s.guideName)
    private val audience = form.choice(
        "안내 대상 (어린이: ${s.guideName}의 반말·별·꽃가루 / 어른: 같은 ${s.guideName}, 정중한 말투)",
        listOf("어린이", "어른"),
        if (s.guideAudience == GuideAudience.ADULTS) 1 else 0,
    )
    private val developerView = form.switch("개발자 화면 보기 (연결 상태, 수동 동기화 버튼)", s.developerViewEnabled)
    private val hideBars = form.switch("운영 화면에서 상단·하단 시스템 바 숨기기", s.hideSystemBars)

    init { form.section("이름") }
    private val callName = form.choice(
        "${GuideCopy.subject(s.guideName)} 부르는 이름",
        listOf("이름만 (하은아 · 하은 님)", "성과 이름 (김하은아 · 김하은 님)"),
        if (s.callWithFamilyName) 1 else 0,
    )
    private val labelFamily = form.switch("이름표에 성 넣기", s.labelFamilyName)
    private val labelGiven = form.switch("이름표에 이름 넣기", s.labelGivenName)

    init {
        form.note("이름표는 '이름' 줄에만 적용됩니다. 둘 다 끄면 성과 이름을 모두 넣습니다. 성을 따로 적지 않고 등록한 분은 이름을 보고 성을 짐작합니다(김하은 → 김 / 하은).")
    }

    override fun applyTo(settings: KioskSettings): KioskSettings = settings.copy(
        guideAudience = if (audience.selectedItemPosition == 1) GuideAudience.ADULTS else GuideAudience.KIDS,
        guideName = guideName.text.toString().trim().take(10).ifEmpty { KioskSettings.DEFAULT_GUIDE_NAME },
        developerViewEnabled = developerView.isChecked,
        hideSystemBars = hideBars.isChecked,
        callWithFamilyName = callName.selectedItemPosition == 1,
        labelFamilyName = labelFamily.isChecked,
        labelGivenName = labelGiven.isChecked,
    )
}

/**
 * "안내 문구": the guide's lines for children and adults (blank = default), then the messages of
 * the developer screen and of the server connection.
 */
class CopyTab(context: Context, s: KioskSettings) : SettingsTab {
    override val title = "안내 문구"
    private val form = SettingsForm(context)
    override val view = scroll(form)

    private val lines = mutableMapOf<String, android.widget.EditText>()

    init {
        form.note("빈칸으로 두면 ${s.guideName}의 기본 문구를 씁니다(회색 글씨). {name}은 이름으로, {direction}은 \"왼쪽을\"(어린이)·\"왼쪽으로\"(어른)처럼, {guide}는 캐릭터 이름으로 바뀝니다(\"{guide}야\" → \"${GuideCopy.introduce(s.guideName)}\").")
        listOf(true to "${GuideCopy.subject(s.guideName)} 하는 말 · 어린이", false to "${GuideCopy.subject(s.guideName)} 하는 말 · 어른").forEach { (kids, title) ->
            form.section(title)
            GuideCopy.LINES.forEach { entry ->
                val key = "${GuideCopy.audienceKey(kids)}.${entry.key}"
                lines[key] = form.text(entry.label, s.guideLines[key].orEmpty()).apply { hint = if (kids) entry.kids else entry.adults }
            }
        }
        form.section("개발자 화면 · 서버 연결 문구")
        form.note("서버 미연결과 승인 대기 문구는 운영 화면에도 그대로 나옵니다. {name}, {label}, {clientId}는 실제 값으로 바뀝니다.")
    }

    private val m = s.messages
    private val welcome = form.text("기본 안내", m.welcome)
    private val disconnected = form.text("서버 미연결", m.serverDisconnected)
    private val disconnectedDetail = form.text("서버 미연결 상세", m.serverDisconnectedDetail)
    private val approval = form.text("서버 승인 대기 ({clientId})", m.awaitingApproval)
    private val approvalDetail = form.text("서버 승인 대기 상세", m.awaitingApprovalDetail)
    private val unregistered = form.text("미등록 안내", m.unregistered)
    private val unregisteredDetail = form.text("미등록 상세", m.unregisteredDetail)
    private val recognized = form.text("인식 성공 ({name})", m.recognized)
    private val ambiguous = form.text("후보 선택", m.ambiguous)
    private val ambiguousDetail = form.text("후보 선택 상세", m.ambiguousDetail)
    private val selected = form.text("후보 선택 완료 ({label})", m.selectedCandidate)
    private val registerComplete = form.text("등록 완료", m.registerComplete)
    private val labelOnly = form.text("이름표만 출력 입력", m.labelOnlyInput)
    private val saving = form.text("저장 중", s.uiSavingMessage)
    private val outputButton = form.text("출력 버튼", s.uiOutputButtonText)

    override fun applyTo(settings: KioskSettings): KioskSettings {
        val d = FlowMessages()
        return settings.copy(
            guideLines = lines.mapValues { it.value.text.toString().trim() }.filterValues { it.isNotEmpty() },
            uiSavingMessage = saving.textOr(settings.uiSavingMessage),
            uiOutputButtonText = outputButton.textOr(settings.uiOutputButtonText),
            messages = settings.messages.copy(
                welcome = welcome.textOr(d.welcome),
                serverDisconnected = disconnected.textOr(d.serverDisconnected),
                serverDisconnectedDetail = disconnectedDetail.textOr(d.serverDisconnectedDetail),
                awaitingApproval = approval.textOr(d.awaitingApproval),
                awaitingApprovalDetail = approvalDetail.textOr(d.awaitingApprovalDetail),
                unregistered = unregistered.textOr(d.unregistered),
                unregisteredDetail = unregisteredDetail.textOr(d.unregisteredDetail),
                recognized = recognized.textOr(d.recognized),
                ambiguous = ambiguous.textOr(d.ambiguous),
                ambiguousDetail = ambiguousDetail.textOr(d.ambiguousDetail),
                selectedCandidate = selected.textOr(d.selectedCandidate),
                registerComplete = registerComplete.textOr(d.registerComplete),
                labelOnlyInput = labelOnly.textOr(d.labelOnlyInput),
            ),
        )
    }
}

/** "인식": thresholds, camera and the guide oval. */
class RecognitionTab(context: Context, s: KioskSettings) : SettingsTab {
    override val title = "인식"
    private val form = SettingsForm(context)
    override val view = scroll(form)

    init { form.section("인식 기준") }
    private val minScore = form.number("인식 최소 유사도 (0~1, 높을수록 엄격)", s.authSimThreshold)
    private val margin = form.number("1·2위 점수 차 최소값 (작으면 후보를 보여줌)", s.authMarginThreshold)
    private val topK = form.number("후보 표시 개수", s.topK)
    private val every = form.number("몇 프레임마다 분석할지", s.analyzeEveryNFrames)
    private val switchFrames = form.number("후보군 교체 확인 프레임", s.ambiguousCandidateSwitchFrames)
    private val grace = form.number("선택 후보 유지 프레임", s.selectedCandidateGraceFrames)
    private val camera = form.choice("카메라", listOf("후면 카메라 우선", "전면 카메라 우선"), s.cameraIndex.coerceIn(0, 1))
    private val alignment = form.switch("얼굴 정렬 사용 (바꾸면 모든 사람을 다시 등록해야 합니다)", s.faceAlignmentEnabled)

    init { form.section("얼굴 동그라미") }
    private val ellipse = form.pair("동그라미 너비 비율", s.registerEllipseAxisXRatio, "동그라미 높이 비율", s.registerEllipseAxisYRatio)
    private val ellipse2 = form.pair("중심 세로 오프셋 비율", s.registerEllipseCenterYOffsetRatio, "바깥 어둡게 (0~1)", s.registerEllipseOutsideDimAlpha)

    override fun applyTo(settings: KioskSettings) = settings.copy(
        authSimThreshold = minScore.floatValue(settings.authSimThreshold).coerceIn(0.05f, 0.95f),
        authMarginThreshold = margin.floatValue(settings.authMarginThreshold).coerceIn(0f, 0.9f),
        topK = topK.intValue(settings.topK).coerceIn(2, 6),
        analyzeEveryNFrames = every.intValue(settings.analyzeEveryNFrames).coerceIn(1, 30),
        ambiguousCandidateSwitchFrames = switchFrames.intValue(settings.ambiguousCandidateSwitchFrames).coerceIn(1, 30),
        selectedCandidateGraceFrames = grace.intValue(settings.selectedCandidateGraceFrames).coerceIn(1, 60),
        cameraIndex = camera.selectedItemPosition,
        faceAlignmentEnabled = alignment.isChecked,
        registerEllipseAxisXRatio = ellipse.first.floatValue(settings.registerEllipseAxisXRatio).coerceIn(0.05f, 0.5f),
        registerEllipseAxisYRatio = ellipse.second.floatValue(settings.registerEllipseAxisYRatio).coerceIn(0.05f, 0.5f),
        registerEllipseCenterYOffsetRatio = ellipse2.first.floatValue(settings.registerEllipseCenterYOffsetRatio).coerceIn(-0.3f, 0.3f),
        registerEllipseOutsideDimAlpha = ellipse2.second.floatValue(settings.registerEllipseOutsideDimAlpha).coerceIn(0f, 1f),
    )
}

/** "등록": sample collection and automatic sample updates. */
class RegistrationTab(context: Context, s: KioskSettings) : SettingsTab {
    override val title = "등록"
    private val form = SettingsForm(context)
    override val view = scroll(form)

    init { form.section("등록") }
    private val includeFront = form.switch("정면도 모으기 (정면·왼쪽·오른쪽·위·아래)", s.registerIncludeFront)
    private val circleSteps = form.switch("동그라미 3단계 (보통 → 가까이 → 멀리, 방향마다 하나씩)", s.registerCircleSteps)
    private val circleScales = form.pair("큰 동그라미 배율", s.registerBigCircleScale, "작은 동그라미 배율", s.registerSmallCircleScale)
    private val fill = form.pair("원에 맞는 얼굴 높이 (원 너비 대비)", s.registerFaceFill, "크기 허용 범위 (±)", s.registerSizeTolerance)
    private val hold = form.number("저장 전 머무는 프레임 수", s.registerHoldFrames)
    private val perDirection = form.number("방향별 수집 장수 (동그라미 3단계를 끄면 사용)", s.registerSamplesPerDirection)
    private val every = form.number("몇 프레임마다 샘플 수집", s.registerCaptureEveryNFrames)
    private val band = form.pair("샘플 유사도 하한", s.registerSimilarityLowerBoundary, "상한 (너무 비슷하면 제외)", s.registerSimilarityHigherBoundary)
    private val maxSamples = form.number("사람별 최대 샘플 수", s.maxEmbeddingsPerPerson)

    init { form.section("자동 보정") }
    private val adaptive = form.switch("인식될 때 얼굴 샘플 자동 추가", s.adaptiveUpdateEnabled)
    private val adaptiveMax = form.number("자동 추가 유사도 상한", s.adaptiveUpdateMaxSim)
    private val adaptiveInterval = form.number("자동 추가 최소 프레임 간격", s.adaptiveUpdateMinIntervalFrames)
    private val adaptivePerSession = form.number("한 번 방문에 자동 추가 최대 수", s.adaptiveUpdateMaxSamplesPerSession)
    private val idle = form.number("서버 변경 반영 전 대기 (얼굴 없음, 초)", s.adaptiveUpdateIdleSeconds)

    override fun applyTo(settings: KioskSettings) = settings.copy(
        registerSamplesPerDirection = perDirection.intValue(settings.registerSamplesPerDirection).coerceIn(1, 10),
        registerIncludeFront = includeFront.isChecked,
        registerCircleSteps = circleSteps.isChecked,
        registerFaceFill = fill.first.floatValue(settings.registerFaceFill).coerceIn(0.4f, 1.6f),
        registerSizeTolerance = fill.second.floatValue(settings.registerSizeTolerance).coerceIn(0.04f, 0.3f),
        registerHoldFrames = hold.intValue(settings.registerHoldFrames).coerceIn(1, 10),
        registerBigCircleScale = circleScales.first.floatValue(settings.registerBigCircleScale).coerceIn(1.05f, 1.6f),
        registerSmallCircleScale = circleScales.second.floatValue(settings.registerSmallCircleScale).coerceIn(0.5f, 0.95f),
        registerCaptureEveryNFrames = every.intValue(settings.registerCaptureEveryNFrames).coerceIn(1, 30),
        registerSimilarityLowerBoundary = band.first.floatValue(settings.registerSimilarityLowerBoundary).coerceIn(0f, 0.9f),
        registerSimilarityHigherBoundary = band.second.floatValue(settings.registerSimilarityHigherBoundary).coerceIn(0.1f, 1f),
        maxEmbeddingsPerPerson = maxSamples.intValue(settings.maxEmbeddingsPerPerson).coerceIn(10, 1000),
        adaptiveUpdateEnabled = adaptive.isChecked,
        adaptiveUpdateMaxSim = adaptiveMax.floatValue(settings.adaptiveUpdateMaxSim).coerceIn(0f, 1f),
        adaptiveUpdateMinIntervalFrames = adaptiveInterval.intValue(settings.adaptiveUpdateMinIntervalFrames).coerceAtLeast(1),
        adaptiveUpdateMaxSamplesPerSession = adaptivePerSession.intValue(settings.adaptiveUpdateMaxSamplesPerSession).coerceIn(0, 200),
        adaptiveUpdateIdleSeconds = idle.intValue(settings.adaptiveUpdateIdleSeconds).coerceIn(1, 60),
    )
}

/** "DB 서버": where the server is and how this kiosk identifies itself. */
class SyncTab(context: Context, s: KioskSettings, connectionSummary: String) : SettingsTab {
    override val title = "DB 서버"
    private val form = SettingsForm(context)
    override val view = scroll(form)

    init {
        form.section("서버 연결")
        form.note("현재 상태: $connectionSummary")
    }
    private val enabled = form.switch("서버와 동기화 (끄면 이 키오스크의 데이터만 사용)", s.dbSyncEnabled)
    private val url = form.text("DB 서버 주소 (서버 앱 현황 탭에 표시된 주소)", s.dbServerBaseUrl)
    private val delays = form.pair("재시도 시작 간격(초)", s.dbConnectRetryDelaySeconds, "연결 제한 시간(초)", s.dbConnectTimeoutSeconds)

    init { form.section("이 키오스크") }
    private val clientId = form.text("키오스크 ID (서버 승인 목록에 표시)", s.dbClientId)
    private val deviceName = form.text("기기 이름", s.dbDeviceName)
    private var secret = s.dbClientSecret

    init {
        form.note("키오스크 비밀키: ${if (s.dbClientSecret.isBlank()) "없음" else "설정됨 (화면에 표시하지 않습니다)"}")
        form.button("새 키오스크 ID 발급 (서버에서 다시 승인 필요)") {
            clientId.setText(com.allstar.faithid.client.sync.KioskIdentity.newClientId())
            secret = com.allstar.faithid.client.sync.KioskIdentity.newSecret()
        }
    }

    override fun applyTo(settings: KioskSettings) = settings.copy(
        dbSyncEnabled = enabled.isChecked,
        dbServerBaseUrl = url.text.toString().trim(),
        dbConnectRetryDelaySeconds = delays.first.intValue(settings.dbConnectRetryDelaySeconds).coerceIn(1, 60),
        dbConnectTimeoutSeconds = delays.second.intValue(settings.dbConnectTimeoutSeconds).coerceIn(2, 60),
        dbClientId = clientId.text.toString().trim().ifBlank { settings.dbClientId },
        dbClientSecret = secret,
        dbDeviceName = deviceName.text.toString().trim(),
    )
}
