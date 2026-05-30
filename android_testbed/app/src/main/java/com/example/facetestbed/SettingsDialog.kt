package com.example.facetestbed

import android.app.AlertDialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import java.util.Locale

object SettingsDialog {
    fun show(
        context: Context,
        current: AppSettings,
        onPickFolderRequested: ((String?) -> Unit) -> Unit,
        onPickTemplateRequested: ((String?) -> Unit) -> Unit,
        onDiscoverBrotherPrintersRequested: ((List<BrotherPairedPrinter>) -> Unit) -> Unit,
        onReadBrotherInstalledPaperRequested: (String, (BrotherDetectedMedia?) -> Unit) -> Unit,
        onRenderPrintPreview: (AppSettings) -> Bitmap?,
        onSave: (AppSettings) -> Unit,
    ) {
        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(context, 12), dp(context, 10), dp(context, 12), dp(context, 8))
        }

        val tabBar = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        root.addView(tabBar, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        val pageContainer = FrameLayout(context)
        root.addView(pageContainer, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        data class TabPage(val button: Button, val page: View)
        val tabs = mutableListOf<TabPage>()

        fun createTab(title: String): LinearLayout {
            val btn = Button(context).apply {
                text = title
                val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT)
                lp.weight = 1f
                layoutParams = lp
            }
            tabBar.addView(btn)

            val pageInner = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(context, 8), dp(context, 8), dp(context, 8), dp(context, 8))
            }
            val scroll = ScrollView(context).apply {
                addView(pageInner, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
                visibility = View.GONE
            }
            pageContainer.addView(scroll)
            tabs += TabPage(btn, scroll)
            return pageInner
        }

        fun createTemplateTab(title: String): Pair<LinearLayout, LinearLayout> {
            val btn = Button(context).apply {
                text = title
                val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT)
                lp.weight = 1f
                layoutParams = lp
            }
            tabBar.addView(btn)

            val page = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(dp(context, 8), dp(context, 8), dp(context, 8), dp(context, 8))
                visibility = View.GONE
            }
            val previewPane = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
            }
            val settingsInner = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
            }
            val settingsScroll = ScrollView(context).apply {
                isFillViewport = true
                isVerticalScrollBarEnabled = true
                isScrollbarFadingEnabled = false
                scrollBarStyle = View.SCROLLBARS_INSIDE_INSET
                addView(settingsInner, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            }

            val previewLp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                weight = 1f
                marginEnd = dp(context, 6)
            }
            val settingsLp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT).apply {
                weight = 1f
                marginStart = dp(context, 6)
            }
            page.addView(previewPane, previewLp)
            page.addView(
                settingsScroll,
                settingsLp,
            )
            pageContainer.addView(page)
            tabs += TabPage(btn, page)
            return previewPane to settingsInner
        }

        fun selectTab(index: Int) {
            tabs.forEachIndexed { i, tab ->
                tab.page.visibility = if (i == index) View.VISIBLE else View.GONE
                tab.button.isEnabled = i != index
            }
        }

        val pageGeneral = createTab("일반")
        val pageRecognition = createTab("인식")
        val pageRegister = createTab("등록")
        val pageSync = createTab("DB 동기화")
        val pagePrint = createTab("인쇄")
        val (pageTemplatePreview, pageTemplateSettings) = createTemplateTab("출력 템플릿")

        tabs.forEachIndexed { idx, tab ->
            tab.button.setOnClickListener { selectTab(idx) }
        }

        section(pageGeneral, context, "저장 위치")
        val faceDataDir = textField(pageGeneral, context, "얼굴 정보 저장 폴더", current.faceDataDir)
        faceDataDir.isFocusable = false
        faceDataDir.isClickable = false
        faceDataDir.hint = "선택된 폴더 URI가 표시됩니다"
        val pickFolderBtn = Button(context).apply {
            text = "폴더 트리에서 선택"
            setOnClickListener {
                onPickFolderRequested { selected ->
                    if (selected != null) faceDataDir.setText(selected)
                }
            }
        }
        pageGeneral.addView(pickFolderBtn)

        section(pageGeneral, context, "화면")
        val developerViewEnabled = boolField(pageGeneral, context, "개발자 화면보기", current.developerViewEnabled)

        section(pageGeneral, context, "UI 문구")
        val uiWelcomeMessage = textField(pageGeneral, context, "기본 안내 문구", current.uiWelcomeMessage)
        val uiServerDisconnectedMessage = textField(pageGeneral, context, "서버 미연결 문구", current.uiServerDisconnectedMessage)
        val uiServerDisconnectedDetail = textField(pageGeneral, context, "서버 미연결 상세 문구", current.uiServerDisconnectedDetail)
        val uiUnregisteredMessage = textField(pageGeneral, context, "미등록 안내 문구", current.uiUnregisteredMessage)
        val uiUnregisteredDetail = textField(pageGeneral, context, "미등록 상세 문구", current.uiUnregisteredDetail)
        val uiRecognizedMessageTemplate = textField(pageGeneral, context, "인식 성공 문구 ({name})", current.uiRecognizedMessageTemplate)
        val uiAmbiguousMessage = textField(pageGeneral, context, "후보 선택 문구", current.uiAmbiguousMessage)
        val uiAmbiguousDetail = textField(pageGeneral, context, "후보 선택 상세 문구", current.uiAmbiguousDetail)
        val uiSelectedCandidateTemplate = textField(pageGeneral, context, "후보 선택 완료 문구 ({label})", current.uiSelectedCandidateTemplate)
        val uiRegisterCompleteMessage = textField(pageGeneral, context, "등록 완료 문구", current.uiRegisterCompleteMessage)
        val uiLabelOnlyInputMessage = textField(pageGeneral, context, "이름표만 출력 입력 문구", current.uiLabelOnlyInputMessage)
        val uiSavingMessage = textField(pageGeneral, context, "저장 중 문구", current.uiSavingMessage)
        val uiOutputButtonText = textField(pageGeneral, context, "출력 버튼 문구", current.uiOutputButtonText)

        section(pageRecognition, context, "인식")
        val authSim = numberField(pageRecognition, context, "인식 최소 유사도 (높을수록 엄격)", current.authSimThreshold.toString(), false)
        val authMargin = numberField(pageRecognition, context, "1, 2위 점수 차 최소값", current.authMarginThreshold.toString(), false)
        val topK = numberField(pageRecognition, context, "후보 표시 개수", current.topK.toString(), true)
        val analyzeEvery = numberField(pageRecognition, context, "몇 프레임마다 분석할지", current.analyzeEveryNFrames.toString(), true)
        val candidateSwitchFrames = numberField(pageRecognition, context, "후보군 교체 확인 프레임", current.ambiguousCandidateSwitchFrames.toString(), true)
        val selectedGraceFrames = numberField(pageRecognition, context, "선택 후보 유지 프레임", current.selectedCandidateGraceFrames.toString(), true)
        val cameraIndex = numberField(pageRecognition, context, "카메라 번호 (0=후면, 1=전면 우선)", current.cameraIndex.toString(), true)
        val useEllipseMask = boolField(pageRecognition, context, "인식 시 타원 마스크 사용", current.recognitionUseEllipseMask)

        section(pageRecognition, context, "가이드 타원")
        val ellipseX = numberField(pageRecognition, context, "타원 너비 비율", current.registerEllipseAxisXRatio.toString(), false)
        val ellipseY = numberField(pageRecognition, context, "타원 높이 비율", current.registerEllipseAxisYRatio.toString(), false)
        val ellipseOffset = numberField(pageRecognition, context, "타원 중심 세로 오프셋 비율", current.registerEllipseCenterYOffsetRatio.toString(), false)
        val ellipseAlpha = numberField(pageRecognition, context, "타원 밖 어둡게 처리 강도", current.registerEllipseOutsideDimAlpha.toString(), false)

        section(pageRegister, context, "등록")
        val regSamples = numberField(pageRegister, context, "방향별 수집 장수", current.registerSamplesPerDirection.toString(), true)
        val regCaptureEvery = numberField(pageRegister, context, "몇 프레임마다 샘플 수집", current.registerCaptureEveryNFrames.toString(), true)
        val regLow = numberField(pageRegister, context, "등록 유사도 하한", current.registerSimilarityLowerBoundary.toString(), false)
        val regHigh = numberField(pageRegister, context, "등록 유사도 상한 (중복 기준)", current.registerSimilarityHigherBoundary.toString(), false)
        val maxEmb = numberField(pageRegister, context, "사람별 최대 임베딩 수", current.maxEmbeddingsPerPerson.toString(), true)

        section(pageRegister, context, "자동 보정")
        val adaptiveEnabled = boolField(pageRegister, context, "자동 보정 사용", current.adaptiveUpdateEnabled)
        val adaptiveMax = numberField(pageRegister, context, "자동 보정 유사도 상한", current.adaptiveUpdateMaxSim.toString(), false)
        val adaptiveMinInterval = numberField(pageRegister, context, "자동 보정 최소 프레임 간격", current.adaptiveUpdateMinIntervalFrames.toString(), true)
        val adaptiveMaxSamples = numberField(pageRegister, context, "세션당 자동 보정 최대 횟수", current.adaptiveUpdateMaxSamplesPerSession.toString(), true)
        val adaptiveIdleSeconds = numberField(pageRegister, context, "DB 업데이트 유휴시간(초)", current.adaptiveUpdateIdleSeconds.toString(), true)

        section(pageSync, context, "DB 서버 동기화")
        val dbSyncEnabled = boolField(pageSync, context, "서버 동기화 사용", current.dbSyncEnabled)
        val dbServerBaseUrl = textField(pageSync, context, "DB 서버 주소", current.dbServerBaseUrl)
        val dbConnectRetryCount = numberField(pageSync, context, "연결 재시도 횟수", current.dbConnectRetryCount.toString(), true)
        val dbConnectRetryDelaySeconds = numberField(pageSync, context, "연결 재시도 대기시간(초)", current.dbConnectRetryDelaySeconds.toString(), true)
        val dbConnectTimeoutSeconds = numberField(pageSync, context, "연결 시도 타임아웃(초)", current.dbConnectTimeoutSeconds.toString(), true)
        val dbClientId = textField(pageSync, context, "클라이언트 ID (비우면 자동 생성)", current.dbClientId)
        val dbDeviceName = textField(pageSync, context, "디바이스 이름 (비우면 자동 생성)", current.dbDeviceName)
        val dbAuthToken = textField(pageSync, context, "DB 인증 토큰", current.dbAuthToken)

        section(pagePrint, context, "프린터 기본")
        val printEnabled = boolField(pagePrint, context, "프린터 사용", current.printEnabled)
        val printerModel = choiceField(
            parent = pagePrint,
            context = context,
            label = "프린터 모델",
            items = listOf("기본 프린터", "Brother QL-820NWB", "Nemonic"),
            selectedIndex = when (current.printerModel) {
                PrinterModel.BROTHER_QL820NWB -> 1
                PrinterModel.NEMONIC -> 2
                else -> 0
            },
        )
        val printerName = textField(pagePrint, context, "프린터 이름", current.printerName)
        val brotherMac = textField(pagePrint, context, "Brother Bluetooth 주소", current.brotherPrinterMacAddress)
        val nemonicMac = textField(pagePrint, context, "네모닉 MAC 주소", current.nemonicPrinterMacAddress)
        val printerTimeout = numberField(pagePrint, context, "출력 대기시간(초)", current.printerTimeoutSec.toString(), true)
        val printerCopies = numberField(pagePrint, context, "출력 매수", current.printerCopies.toString(), true)
        val printerTitlePrefix = textField(pagePrint, context, "출력 작업 제목 접두어", current.printerJobTitlePrefix)
        val brotherGuide = TextView(context).apply {
            text = "Brother는 태블릿 설정에서 먼저 Bluetooth 페어링 후, 검색 버튼으로 QL-820NWB를 선택해 연결합니다."
            setPadding(0, dp(context, 4), 0, dp(context, 2))
        }
        pagePrint.addView(brotherGuide)
        val brotherSearchStatus = TextView(context).apply {
            text = "Brother 검색 대기중"
            setPadding(0, dp(context, 2), 0, dp(context, 2))
        }
        val brotherMediaStatus = TextView(context).apply {
            text = "장착 용지 정보 대기중"
            setPadding(0, dp(context, 2), 0, dp(context, 6))
        }
        val brotherSearchButton = Button(context).apply {
            text = "Brother 페어링 프린터 검색"
            setOnClickListener {
                brotherSearchStatus.text = "페어링된 Brother 프린터 검색 중..."
                onDiscoverBrotherPrintersRequested { printers ->
                    if (printers.isEmpty()) {
                        brotherSearchStatus.text = "검색 결과 없음: 태블릿 Bluetooth 페어링 상태를 확인해 주세요."
                        return@onDiscoverBrotherPrintersRequested
                    }
                    val labels = printers.map { "${it.displayName} [${it.macAddress}]" }.toTypedArray()
                    AlertDialog.Builder(context)
                        .setTitle("Brother QL-820NWB 선택")
                        .setItems(labels) { _, which ->
                            val picked = printers[which]
                            printerName.setText(picked.displayName)
                            brotherMac.setText(picked.macAddress)
                            brotherSearchStatus.text = "선택됨: ${picked.displayName} (${picked.macAddress})"
                            brotherMediaStatus.text = "프린터 선택 완료: 아래 'Brother 장착 용지 읽기'를 눌러 출력용지에 반영하세요."
                        }
                        .setNegativeButton("취소", null)
                        .show()
                }
            }
        }
        pagePrint.addView(brotherSearchButton)
        pagePrint.addView(brotherSearchStatus)
        pagePrint.addView(brotherMediaStatus)
        val printOrientation = choiceField(
            parent = pagePrint,
            context = context,
            label = "프린터 출력 방향",
            items = listOf("세로", "가로"),
            selectedIndex = if (current.printLandscape) 1 else 0,
        )

        section(pagePrint, context, "용지 설정")
        val paperProfiles = current.paperProfiles.toMutableList()
        var selectedPaperIndex = paperProfiles.indexOfFirst { it.id == current.selectedPaperId }.let { if (it >= 0) it else 0 }
        val paperSpinner = choiceField(
            parent = pagePrint,
            context = context,
            label = "출력 용지 선택",
            items = paperProfiles.map { "${it.name} (${trimZero(it.widthMm)}x${trimZero(it.heightMm)}mm)" },
            selectedIndex = selectedPaperIndex,
        )
        val paperNameInput = textField(pagePrint, context, "새 용지 이름", "")
        val paperWidthInput = numberField(pagePrint, context, "새 용지 가로(mm)", "", false)
        val paperHeightInput = numberField(pagePrint, context, "새 용지 세로(mm)", "", false)
        val paperStatus = TextView(context).apply {
            text = "용지를 추가하면 미리보기/출력 비율에 반영됩니다."
            setPadding(0, dp(context, 4), 0, dp(context, 2))
        }
        pagePrint.addView(paperStatus)
        var refreshPreview: () -> Unit = {}

        fun updatePaperSpinner() {
            val adapter = ArrayAdapter(
                context,
                android.R.layout.simple_spinner_item,
                paperProfiles.map { "${it.name} (${trimZero(it.widthMm)}x${trimZero(it.heightMm)}mm)" },
            ).apply {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
            paperSpinner.adapter = adapter
            selectedPaperIndex = selectedPaperIndex.coerceIn(0, paperProfiles.lastIndex.coerceAtLeast(0))
            paperSpinner.setSelection(selectedPaperIndex)
        }

        paperSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedPaperIndex = position.coerceIn(0, paperProfiles.lastIndex.coerceAtLeast(0))
                refreshPreview()
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        printOrientation.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {}

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        val addPaperBtn = Button(context).apply {
            text = "새 용지 추가"
            setOnClickListener {
                val name = paperNameInput.text.toString().trim()
                val w = paperWidthInput.text.toString().trim().toFloatOrNull()
                val h = paperHeightInput.text.toString().trim().toFloatOrNull()
                if (name.isBlank() || w == null || h == null || w <= 0f || h <= 0f) {
                    paperStatus.text = "용지 추가 실패: 이름/가로/세로(mm)를 올바르게 입력해 주세요."
                    return@setOnClickListener
                }
                val id = "paper_${System.currentTimeMillis()}"
                paperProfiles += PaperProfile(id = id, name = name, widthMm = w, heightMm = h)
                selectedPaperIndex = paperProfiles.lastIndex
                updatePaperSpinner()
                paperNameInput.setText("")
                paperWidthInput.setText("")
                paperHeightInput.setText("")
                paperStatus.text = "용지 추가됨: $name (${trimZero(w)}x${trimZero(h)}mm)"
                refreshPreview()
            }
        }
        pagePrint.addView(addPaperBtn)
        val detectBrotherPaperButton = Button(context).apply {
            text = "Brother 장착 용지 읽기"
            setOnClickListener {
                val mac = brotherMac.text.toString().trim()
                if (mac.isBlank()) {
                    brotherMediaStatus.text = "먼저 Brother 프린터를 검색하고 선택해 주세요."
                    return@setOnClickListener
                }
                brotherMediaStatus.text = "장착 용지 정보를 읽는 중..."
                onReadBrotherInstalledPaperRequested(mac) { media ->
                    if (media == null || media.widthMm == null) {
                        brotherMediaStatus.text = "장착 용지 읽기 실패: 출력 용지 선택을 수동으로 확인해 주세요."
                        return@onReadBrotherInstalledPaperRequested
                    }
                    val profileId = buildBrotherPaperProfileId(media)
                    val profileName = buildBrotherPaperProfileName(media)
                    val width = media.widthMm.toFloat()
                    val height = if (media.isInfinite) {
                        media.heightMm?.takeIf { it > 0 }?.toFloat() ?: 100f
                    } else {
                        media.heightMm?.takeIf { it > 0 }?.toFloat() ?: width
                    }
                    val newProfile = PaperProfile(
                        id = profileId,
                        name = profileName,
                        widthMm = width,
                        heightMm = height,
                    )
                    val existingIndex = paperProfiles.indexOfFirst { it.id == profileId }
                    if (existingIndex >= 0) {
                        paperProfiles[existingIndex] = newProfile
                        selectedPaperIndex = existingIndex
                    } else {
                        paperProfiles += newProfile
                        selectedPaperIndex = paperProfiles.lastIndex
                    }
                    updatePaperSpinner()
                    brotherMediaStatus.text = "장착 용지 반영됨: ${newProfile.name} (${trimZero(width)}x${trimZero(height)}mm)"
                    paperStatus.text = "Brother 장착 용지가 출력용지로 선택되었습니다."
                    refreshPreview()
                }
            }
        }
        pagePrint.addView(detectBrotherPaperButton, pagePrint.indexOfChild(brotherSearchButton) + 1)
        val deletePaperBtn = Button(context).apply {
            text = "선택 용지 삭제"
            setOnClickListener {
                val position = paperSpinner.selectedItemPosition.coerceIn(0, paperProfiles.lastIndex.coerceAtLeast(0))
                val target = paperProfiles.getOrNull(position)
                if (target == null) {
                    paperStatus.text = "삭제할 용지가 없습니다."
                    return@setOnClickListener
                }
                if (paperProfiles.size <= 1) {
                    paperStatus.text = "최소 1개의 용지는 남아 있어야 합니다."
                    return@setOnClickListener
                }
                paperProfiles.removeAt(position)
                selectedPaperIndex = (position - 1).coerceAtLeast(0)
                updatePaperSpinner()
                paperStatus.text = "용지 삭제됨: ${target.name}"
                refreshPreview()
            }
        }
        pagePrint.addView(deletePaperBtn)

        section(pagePrint, context, "출력 상태 안내")
        val printStatusReadyMessage = textField(pagePrint, context, "출력 준비 문구", current.printStatusReadyMessage)
        val printStatusPermissionMessage = textField(pagePrint, context, "권한 확인 문구", current.printStatusPermissionMessage)
        val printStatusConnectingMessage = textField(pagePrint, context, "연결 중 문구", current.printStatusConnectingMessage)
        val printStatusPreparingMessage = textField(pagePrint, context, "내용 준비 문구", current.printStatusPreparingMessage)
        val printStatusSendingMessage = textField(pagePrint, context, "출력 중 문구", current.printStatusSendingMessage)
        val printStatusSuccessMessage = textField(pagePrint, context, "완료 문구", current.printStatusSuccessMessage)
        val printStatusErrorMessage = textField(pagePrint, context, "오류 문구", current.printStatusErrorMessage)

        section(pagePrint, context, "복귀 안내")
        val outputReturnDelaySeconds = numberField(pagePrint, context, "출력 후 기본 화면 복귀시간(초)", current.outputReturnDelaySeconds.toString(), true)
        val outputReturnMessageTemplate = textField(pagePrint, context, "복귀 안내문구 템플릿 ({seconds} 또는 N 사용)", current.outputReturnMessageTemplate)

        val previewFrame = FrameLayout(context).apply {
            setBackgroundColor(0xFFECEFF4.toInt())
            setPadding(dp(context, 8), dp(context, 8), dp(context, 8), dp(context, 8))
        }
        val previewImage = ImageView(context).apply {
            adjustViewBounds = true
            minimumHeight = dp(context, 180)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        val guideOverlayImage = ImageView(context).apply {
            adjustViewBounds = true
            minimumHeight = dp(context, 180)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        previewFrame.addView(previewImage, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        previewFrame.addView(guideOverlayImage, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        pageTemplatePreview.addView(previewFrame, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        val previewStatus = TextView(context).apply {
            text = "미리보기를 불러오는 중..."
            setPadding(0, dp(context, 6), 0, dp(context, 4))
        }
        pageTemplatePreview.addView(previewStatus)

        section(pageTemplateSettings, context, "템플릿")
        val printerTemplateImageUri = textField(pageTemplateSettings, context, "이미지 템플릿 URI (jpg/png/webp)", current.printerTemplateImageUri)
        printerTemplateImageUri.isFocusable = false
        printerTemplateImageUri.isClickable = false
        printerTemplateImageUri.hint = "선택된 이미지 템플릿 URI가 표시됩니다"

        val pickTemplateBtn = Button(context).apply {
            text = "이미지 템플릿 불러오기"
            setOnClickListener {
                onPickTemplateRequested { selected ->
                    if (selected != null) {
                        printerTemplateImageUri.setText(selected)
                        refreshPreview()
                    }
                }
            }
        }
        pageTemplateSettings.addView(pickTemplateBtn)
        var templateRotationQuarterTurns = current.templateImageRotationQuarterTurns.mod(4)
        val templateRotationStatus = TextView(context).apply {
            text = "템플릿 회전: ${templateRotationQuarterTurns * 90}도"
            setPadding(0, dp(context, 4), 0, dp(context, 2))
        }
        pageTemplateSettings.addView(templateRotationStatus)
        val rotateTemplateBtn = Button(context).apply {
            text = "이미지 템플릿 시계방향 90도 회전"
            setOnClickListener {
                templateRotationQuarterTurns = (templateRotationQuarterTurns + 1) % 4
                templateRotationStatus.text = "템플릿 회전: ${templateRotationQuarterTurns * 90}도"
                refreshPreview()
            }
        }
        pageTemplateSettings.addView(rotateTemplateBtn)
        val paperOrientation = choiceField(
            parent = pageTemplateSettings,
            context = context,
            label = "템플릿방향",
            items = listOf("세로", "가로"),
            selectedIndex = if (current.paperLandscape) 1 else 0,
        )
        val templateShrinkToPaper = boolField(
            parent = pageTemplateSettings,
            context = context,
            label = "이미지가 용지를 초과하면 자동 축소",
            value = current.templateShrinkToPaperIfOversized,
        )
        paperOrientation.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                refreshPreview()
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        section(pageTemplateSettings, context, "텍스트 영역")
        val (regionOffsetX, regionOffsetY) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "X 오프셋(px)",
            leftValue = current.templateTextRegionOffsetX.toString(),
            rightLabel = "Y 오프셋(px)",
            rightValue = current.templateTextRegionOffsetY.toString(),
            integer = true,
        )
        val (regionWidth, regionHeight) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "영역 너비(px)",
            leftValue = current.templateTextRegionWidth.toString(),
            rightLabel = "영역 높이(px)",
            rightValue = current.templateTextRegionHeight.toString(),
            integer = true,
        )
        val regionGuideInfo = TextView(context).apply {
            text = "점선 영역은 설정 UI 가이드이며, 실제 출력물/출력 미리보기 이미지에는 포함되지 않습니다."
            setPadding(0, dp(context, 4), 0, dp(context, 2))
        }
        pageTemplateSettings.addView(regionGuideInfo)

        section(pageTemplateSettings, context, "텍스트 설정")
        val attributeColumns = listOf("name" to "이름", "department" to "부서", "grade" to "학년")
        val attributeCountSpinner = choiceField(
            parent = pageTemplateSettings,
            context = context,
            label = "출력 속성 갯수",
            items = listOf("1개", "2개", "3개"),
            selectedIndex = (current.printAttributeCount.coerceIn(1, 3) - 1),
        )
        val attribute1ColumnSpinner = choiceField(
            parent = pageTemplateSettings,
            context = context,
            label = "속성1 컬럼",
            items = attributeColumns.map { it.second },
            selectedIndex = attributeColumns.indexOfFirst { it.first == current.printAttribute1Column }.coerceAtLeast(0),
        )
        val (printBaseFont, printScale) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "기본 글자 크기",
            leftValue = current.printTextBaseFontSize.toString(),
            rightLabel = "글자 확대 배율",
            rightValue = current.printTextScaleMultiplier.toString(),
            integer = true,
        )
        val (outputNameSize, outputDepartmentSize) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "속성1 글자 크기(sp)",
            leftValue = current.printAttribute1TextSizeSp.toString(),
            rightLabel = "속성2 글자 크기(sp)",
            rightValue = current.printAttribute2TextSizeSp.toString(),
            integer = true,
        )
        val (outputNameOffsetX, outputNameOffsetY) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "속성1 X 오프셋",
            leftValue = current.printAttribute1OffsetX.toString(),
            rightLabel = "속성1 Y 오프셋",
            rightValue = current.printAttribute1OffsetY.toString(),
            integer = true,
        )
        val attribute2ColumnSpinner = choiceField(
            parent = pageTemplateSettings,
            context = context,
            label = "속성2 컬럼",
            items = attributeColumns.map { it.second },
            selectedIndex = attributeColumns.indexOfFirst { it.first == current.printAttribute2Column }.coerceAtLeast(1).coerceAtMost(attributeColumns.lastIndex),
        )
        val (outputDepartmentOffsetX, outputDepartmentOffsetY) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "속성2 X 오프셋",
            leftValue = current.printAttribute2OffsetX.toString(),
            rightLabel = "속성2 Y 오프셋",
            rightValue = current.printAttribute2OffsetY.toString(),
            integer = true,
        )
        val attribute3ColumnSpinner = choiceField(
            parent = pageTemplateSettings,
            context = context,
            label = "속성3 컬럼",
            items = attributeColumns.map { it.second },
            selectedIndex = attributeColumns.indexOfFirst { it.first == current.printAttribute3Column }.coerceAtLeast(2).coerceAtMost(attributeColumns.lastIndex),
        )
        val (outputAttribute3Size, outputAttribute3OffsetX) = compactNumberPair(
            parent = pageTemplateSettings,
            context = context,
            leftLabel = "속성3 글자 크기(sp)",
            leftValue = current.printAttribute3TextSizeSp.toString(),
            rightLabel = "속성3 X 오프셋",
            rightValue = current.printAttribute3OffsetX.toString(),
            integer = true,
        )
        val outputAttribute3OffsetY = numberField(
            parent = pageTemplateSettings,
            context = context,
            label = "속성3 Y 오프셋",
            value = current.printAttribute3OffsetY.toString(),
            integer = true,
        )

        fun selectedPaperProfile(): PaperProfile {
            return paperProfiles.getOrNull(paperSpinner.selectedItemPosition)
                ?: paperProfiles.firstOrNull()
                ?: current.paperProfiles.first()
        }

        fun paperPixelSize(profile: PaperProfile): Pair<Int, Int> {
            val pixelsPerMm = 8.0f
            val width = (profile.widthMm * pixelsPerMm).toInt().coerceIn(200, 1200)
            val height = (profile.heightMm * pixelsPerMm).toInt().coerceIn(120, 1600)
            return width to height
        }

        fun displayRectFromStored(
            storedX: Int,
            storedY: Int,
            storedWidth: Int,
            storedHeight: Int,
            landscape: Boolean,
            profile: PaperProfile,
        ): IntArray {
            val (_, portraitHeight) = paperPixelSize(profile)
            val x = storedX
            val y = storedY
            val width = storedWidth.coerceAtLeast(1)
            val height = storedHeight.coerceAtLeast(1)
            return if (!landscape) {
                intArrayOf(x, y, width, height)
            } else {
                intArrayOf(
                    portraitHeight - (y + height),
                    x,
                    height,
                    width,
                )
            }
        }

        fun storedRectFromDisplay(
            displayX: Int,
            displayY: Int,
            displayWidth: Int,
            displayHeight: Int,
            landscape: Boolean,
            profile: PaperProfile,
        ): IntArray {
            return if (!landscape) {
                intArrayOf(
                    displayX,
                    displayY,
                    displayWidth.coerceAtLeast(1),
                    displayHeight.coerceAtLeast(1),
                )
            } else {
                val (_, portraitHeight) = paperPixelSize(profile)
                val x = displayX
                val y = displayY
                val width = displayWidth.coerceAtLeast(1)
                val height = displayHeight.coerceAtLeast(1)
                val storedX = y
                val storedY = portraitHeight - (x + width)
                intArrayOf(
                    storedX,
                    storedY,
                    height,
                    width,
                )
            }
        }

        fun displayOffsetFromStored(offsetX: Int, offsetY: Int, landscape: Boolean): IntArray {
            return if (landscape) intArrayOf(-offsetY, offsetX) else intArrayOf(offsetX, offsetY)
        }

        fun storedOffsetFromDisplay(offsetX: Int, offsetY: Int, landscape: Boolean): IntArray {
            return if (landscape) intArrayOf(offsetY, -offsetX) else intArrayOf(offsetX, offsetY)
        }

        fun applyDisplayedTemplateAxisValues(
            landscape: Boolean,
            storedRegionX: Int,
            storedRegionY: Int,
            storedRegionWidth: Int,
            storedRegionHeight: Int,
            storedNameOffsetX: Int,
            storedNameOffsetY: Int,
            storedDepartmentOffsetX: Int,
            storedDepartmentOffsetY: Int,
            storedAttribute3OffsetX: Int,
            storedAttribute3OffsetY: Int,
        ) {
            val profile = selectedPaperProfile()
            val rect = displayRectFromStored(
                storedRegionX,
                storedRegionY,
                storedRegionWidth,
                storedRegionHeight,
                landscape,
                profile,
            )
            val nameOffsets = displayOffsetFromStored(storedNameOffsetX, storedNameOffsetY, landscape)
            val departmentOffsets = displayOffsetFromStored(storedDepartmentOffsetX, storedDepartmentOffsetY, landscape)
            val attribute3Offsets = displayOffsetFromStored(storedAttribute3OffsetX, storedAttribute3OffsetY, landscape)
            regionOffsetX.setText(rect[0].toString())
            regionOffsetY.setText(rect[1].toString())
            regionWidth.setText(rect[2].toString())
            regionHeight.setText(rect[3].toString())
            outputNameOffsetX.setText(nameOffsets[0].toString())
            outputNameOffsetY.setText(nameOffsets[1].toString())
            outputDepartmentOffsetX.setText(departmentOffsets[0].toString())
            outputDepartmentOffsetY.setText(departmentOffsets[1].toString())
            outputAttribute3OffsetX.setText(attribute3Offsets[0].toString())
            outputAttribute3OffsetY.setText(attribute3Offsets[1].toString())
        }

        applyDisplayedTemplateAxisValues(
            landscape = current.paperLandscape,
            storedRegionX = current.templateTextRegionOffsetX,
            storedRegionY = current.templateTextRegionOffsetY,
            storedRegionWidth = current.templateTextRegionWidth,
            storedRegionHeight = current.templateTextRegionHeight,
            storedNameOffsetX = current.outputNameOffsetX,
            storedNameOffsetY = current.outputNameOffsetY,
            storedDepartmentOffsetX = current.outputDepartmentOffsetX,
            storedDepartmentOffsetY = current.outputDepartmentOffsetY,
            storedAttribute3OffsetX = current.printAttribute3OffsetX,
            storedAttribute3OffsetY = current.printAttribute3OffsetY,
        )

        var lastTemplateLandscapeSelection = current.paperLandscape
        paperOrientation.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newLandscape = position == 1
                if (newLandscape != lastTemplateLandscapeSelection) {
                    val profile = selectedPaperProfile()
                    val storedRegion = storedRectFromDisplay(
                        regionOffsetX.asInt(current.templateTextRegionOffsetX),
                        regionOffsetY.asInt(current.templateTextRegionOffsetY),
                        regionWidth.asInt(current.templateTextRegionWidth).coerceAtLeast(1),
                        regionHeight.asInt(current.templateTextRegionHeight).coerceAtLeast(1),
                        lastTemplateLandscapeSelection,
                        profile,
                    )
                    val storedNameOffsets = storedOffsetFromDisplay(
                        outputNameOffsetX.asInt(current.outputNameOffsetX),
                        outputNameOffsetY.asInt(current.outputNameOffsetY),
                        lastTemplateLandscapeSelection,
                    )
                    val storedDepartmentOffsets = storedOffsetFromDisplay(
                        outputDepartmentOffsetX.asInt(current.outputDepartmentOffsetX),
                        outputDepartmentOffsetY.asInt(current.outputDepartmentOffsetY),
                        lastTemplateLandscapeSelection,
                    )
                    val storedAttribute3Offsets = storedOffsetFromDisplay(
                        outputAttribute3OffsetX.asInt(current.printAttribute3OffsetX),
                        outputAttribute3OffsetY.asInt(current.printAttribute3OffsetY),
                        lastTemplateLandscapeSelection,
                    )
                    applyDisplayedTemplateAxisValues(
                        landscape = newLandscape,
                        storedRegionX = storedRegion[0],
                        storedRegionY = storedRegion[1],
                        storedRegionWidth = storedRegion[2],
                        storedRegionHeight = storedRegion[3],
                        storedNameOffsetX = storedNameOffsets[0],
                        storedNameOffsetY = storedNameOffsets[1],
                        storedDepartmentOffsetX = storedDepartmentOffsets[0],
                        storedDepartmentOffsetY = storedDepartmentOffsets[1],
                        storedAttribute3OffsetX = storedAttribute3Offsets[0],
                        storedAttribute3OffsetY = storedAttribute3Offsets[1],
                    )
                    lastTemplateLandscapeSelection = newLandscape
                }
                refreshPreview()
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        fun buildUpdatedSettings(): AppSettings {
            val selectedPaper = paperProfiles.getOrNull(paperSpinner.selectedItemPosition)
                ?: paperProfiles.firstOrNull()
                ?: current.paperProfiles.first()
            val templateLandscape = paperOrientation.selectedItemPosition == 1
            val storedRegion = storedRectFromDisplay(
                regionOffsetX.asInt(current.templateTextRegionOffsetX),
                regionOffsetY.asInt(current.templateTextRegionOffsetY),
                regionWidth.asInt(current.templateTextRegionWidth).coerceAtLeast(1),
                regionHeight.asInt(current.templateTextRegionHeight).coerceAtLeast(1),
                templateLandscape,
                selectedPaper,
            )
            val storedNameOffsets = storedOffsetFromDisplay(
                outputNameOffsetX.asInt(current.outputNameOffsetX),
                outputNameOffsetY.asInt(current.outputNameOffsetY),
                templateLandscape,
            )
            val storedDepartmentOffsets = storedOffsetFromDisplay(
                outputDepartmentOffsetX.asInt(current.outputDepartmentOffsetX),
                outputDepartmentOffsetY.asInt(current.outputDepartmentOffsetY),
                templateLandscape,
            )
            val storedAttribute3Offsets = storedOffsetFromDisplay(
                outputAttribute3OffsetX.asInt(current.printAttribute3OffsetX),
                outputAttribute3OffsetY.asInt(current.printAttribute3OffsetY),
                templateLandscape,
            )
            return AppSettings(
                faceDataDir = faceDataDir.text.toString().trim(),
                developerViewEnabled = developerViewEnabled.isChecked,
                authSimThreshold = authSim.asFloat(current.authSimThreshold),
                authMarginThreshold = authMargin.asFloat(current.authMarginThreshold),
                topK = topK.asInt(current.topK).coerceAtLeast(1),
                analyzeEveryNFrames = analyzeEvery.asInt(current.analyzeEveryNFrames).coerceAtLeast(1),
                ambiguousCandidateSwitchFrames = candidateSwitchFrames.asInt(current.ambiguousCandidateSwitchFrames).coerceAtLeast(1),
                selectedCandidateGraceFrames = selectedGraceFrames.asInt(current.selectedCandidateGraceFrames).coerceAtLeast(1),
                cameraIndex = cameraIndex.asInt(current.cameraIndex).coerceIn(0, 1),
                recognitionUseEllipseMask = useEllipseMask.isChecked,
                registerSamplesPerDirection = regSamples.asInt(current.registerSamplesPerDirection).coerceAtLeast(1),
                registerCaptureEveryNFrames = regCaptureEvery.asInt(current.registerCaptureEveryNFrames).coerceAtLeast(1),
                registerSimilarityLowerBoundary = regLow.asFloat(current.registerSimilarityLowerBoundary),
                registerSimilarityHigherBoundary = regHigh.asFloat(current.registerSimilarityHigherBoundary),
                maxEmbeddingsPerPerson = maxEmb.asInt(current.maxEmbeddingsPerPerson).coerceAtLeast(1),
                registerEllipseAxisXRatio = ellipseX.asFloat(current.registerEllipseAxisXRatio),
                registerEllipseAxisYRatio = ellipseY.asFloat(current.registerEllipseAxisYRatio),
                registerEllipseCenterYOffsetRatio = ellipseOffset.asFloat(current.registerEllipseCenterYOffsetRatio),
                registerEllipseOutsideDimAlpha = ellipseAlpha.asFloat(current.registerEllipseOutsideDimAlpha),
                adaptiveUpdateEnabled = adaptiveEnabled.isChecked,
                adaptiveUpdateMaxSim = adaptiveMax.asFloat(current.adaptiveUpdateMaxSim),
                adaptiveUpdateMinIntervalFrames = adaptiveMinInterval.asInt(current.adaptiveUpdateMinIntervalFrames).coerceAtLeast(1),
                adaptiveUpdateMaxSamplesPerSession = adaptiveMaxSamples.asInt(current.adaptiveUpdateMaxSamplesPerSession).coerceAtLeast(1),
                adaptiveUpdateIdleSeconds = adaptiveIdleSeconds.asInt(current.adaptiveUpdateIdleSeconds).coerceAtLeast(1),
                dbSyncEnabled = dbSyncEnabled.isChecked,
                dbServerBaseUrl = dbServerBaseUrl.text.toString().trim(),
                dbConnectRetryCount = dbConnectRetryCount.asInt(current.dbConnectRetryCount).coerceAtLeast(1),
                dbConnectRetryDelaySeconds = dbConnectRetryDelaySeconds.asInt(current.dbConnectRetryDelaySeconds).coerceAtLeast(1),
                dbConnectTimeoutSeconds = dbConnectTimeoutSeconds.asInt(current.dbConnectTimeoutSeconds).coerceAtLeast(2),
                dbClientId = dbClientId.text.toString().trim(),
                dbAuthToken = dbAuthToken.text.toString().trim(),
                dbDeviceName = dbDeviceName.text.toString().trim(),
                printEnabled = printEnabled.isChecked,
                printerModel = when (printerModel.selectedItemPosition) {
                    1 -> PrinterModel.BROTHER_QL820NWB
                    2 -> PrinterModel.NEMONIC
                    else -> PrinterModel.SYSTEM
                },
                printerName = printerName.text.toString(),
                brotherPrinterMacAddress = brotherMac.text.toString().trim(),
                nemonicPrinterMacAddress = nemonicMac.text.toString(),
                printerTimeoutSec = printerTimeout.asInt(current.printerTimeoutSec).coerceAtLeast(1),
                printerCopies = printerCopies.asInt(current.printerCopies).coerceAtLeast(1),
                printerJobTitlePrefix = printerTitlePrefix.text.toString(),
                printerContentTemplate = current.printerContentTemplate,
                printerTemplateImageUri = printerTemplateImageUri.text.toString().trim(),
                templateImageRotationQuarterTurns = templateRotationQuarterTurns,
                templateShrinkToPaperIfOversized = templateShrinkToPaper.isChecked,
                paperProfiles = paperProfiles.toList(),
                selectedPaperId = selectedPaper.id,
                printLandscape = printOrientation.selectedItemPosition == 1,
                paperLandscape = templateLandscape,
                templateTextRegionOffsetX = storedRegion[0],
                templateTextRegionOffsetY = storedRegion[1],
                templateTextRegionWidth = storedRegion[2],
                templateTextRegionHeight = storedRegion[3],
                printTextBaseFontSize = printBaseFont.asInt(current.printTextBaseFontSize).coerceAtLeast(1),
                printTextScaleMultiplier = printScale.asInt(current.printTextScaleMultiplier).coerceAtLeast(1),
                printAttributeCount = attributeCountSpinner.selectedItemPosition + 1,
                printAttribute1Column = attributeColumns.getOrNull(attribute1ColumnSpinner.selectedItemPosition)?.first ?: "name",
                printAttribute1TextSizeSp = outputNameSize.asInt(current.printAttribute1TextSizeSp).coerceIn(12, 120),
                printAttribute1OffsetX = storedNameOffsets[0],
                printAttribute1OffsetY = storedNameOffsets[1],
                printAttribute2Column = attributeColumns.getOrNull(attribute2ColumnSpinner.selectedItemPosition)?.first ?: "department",
                printAttribute2TextSizeSp = outputDepartmentSize.asInt(current.printAttribute2TextSizeSp).coerceIn(10, 100),
                printAttribute2OffsetX = storedDepartmentOffsets[0],
                printAttribute2OffsetY = storedDepartmentOffsets[1],
                printAttribute3Column = attributeColumns.getOrNull(attribute3ColumnSpinner.selectedItemPosition)?.first ?: "grade",
                printAttribute3TextSizeSp = outputAttribute3Size.asInt(current.printAttribute3TextSizeSp).coerceIn(10, 100),
                printAttribute3OffsetX = storedAttribute3Offsets[0],
                printAttribute3OffsetY = storedAttribute3Offsets[1],
                outputNameTextSizeSp = outputNameSize.asInt(current.outputNameTextSizeSp).coerceIn(12, 120),
                outputNameOffsetX = storedNameOffsets[0],
                outputNameOffsetY = storedNameOffsets[1],
                outputDepartmentTextSizeSp = outputDepartmentSize.asInt(current.outputDepartmentTextSizeSp).coerceIn(10, 100),
                outputDepartmentOffsetX = storedDepartmentOffsets[0],
                outputDepartmentOffsetY = storedDepartmentOffsets[1],
                printStatusReadyMessage = printStatusReadyMessage.text.toString().ifBlank { current.printStatusReadyMessage },
                printStatusPermissionMessage = printStatusPermissionMessage.text.toString().ifBlank { current.printStatusPermissionMessage },
                printStatusConnectingMessage = printStatusConnectingMessage.text.toString().ifBlank { current.printStatusConnectingMessage },
                printStatusPreparingMessage = printStatusPreparingMessage.text.toString().ifBlank { current.printStatusPreparingMessage },
                printStatusSendingMessage = printStatusSendingMessage.text.toString().ifBlank { current.printStatusSendingMessage },
                printStatusSuccessMessage = printStatusSuccessMessage.text.toString().ifBlank { current.printStatusSuccessMessage },
                printStatusErrorMessage = printStatusErrorMessage.text.toString().ifBlank { current.printStatusErrorMessage },
                outputReturnDelaySeconds = outputReturnDelaySeconds.asInt(current.outputReturnDelaySeconds).coerceAtLeast(1),
                outputReturnMessageTemplate = outputReturnMessageTemplate.text.toString().ifBlank { current.outputReturnMessageTemplate },
                uiWelcomeMessage = uiWelcomeMessage.text.toString().ifBlank { current.uiWelcomeMessage },
                uiServerDisconnectedMessage = uiServerDisconnectedMessage.text.toString().ifBlank { current.uiServerDisconnectedMessage },
                uiServerDisconnectedDetail = uiServerDisconnectedDetail.text.toString().ifBlank { current.uiServerDisconnectedDetail },
                uiUnregisteredMessage = uiUnregisteredMessage.text.toString().ifBlank { current.uiUnregisteredMessage },
                uiUnregisteredDetail = uiUnregisteredDetail.text.toString().ifBlank { current.uiUnregisteredDetail },
                uiRecognizedMessageTemplate = uiRecognizedMessageTemplate.text.toString().ifBlank { current.uiRecognizedMessageTemplate },
                uiAmbiguousMessage = uiAmbiguousMessage.text.toString().ifBlank { current.uiAmbiguousMessage },
                uiAmbiguousDetail = uiAmbiguousDetail.text.toString().ifBlank { current.uiAmbiguousDetail },
                uiSelectedCandidateTemplate = uiSelectedCandidateTemplate.text.toString().ifBlank { current.uiSelectedCandidateTemplate },
                uiRegisterCompleteMessage = uiRegisterCompleteMessage.text.toString().ifBlank { current.uiRegisterCompleteMessage },
                uiLabelOnlyInputMessage = uiLabelOnlyInputMessage.text.toString().ifBlank { current.uiLabelOnlyInputMessage },
                uiSavingMessage = uiSavingMessage.text.toString().ifBlank { current.uiSavingMessage },
                uiOutputButtonText = uiOutputButtonText.text.toString().ifBlank { current.uiOutputButtonText },
            )
        }

        var previewBitmap: Bitmap? = null
        var guideBitmap: Bitmap? = null

        refreshPreview = {
            val updated = buildUpdatedSettings()
            val rendered = runCatching { onRenderPrintPreview(updated) }.getOrNull()
            previewBitmap?.recycle()
            previewBitmap = null
            guideBitmap?.recycle()
            guideBitmap = null

            if (rendered == null) {
                previewImage.setImageDrawable(null)
                guideOverlayImage.setImageDrawable(null)
                previewStatus.text = "미리보기를 생성하지 못했습니다."
            } else {
                previewBitmap = rendered
                previewImage.setImageBitmap(rendered)
                guideBitmap = buildGuideOverlayBitmap(rendered.width, rendered.height, updated)
                guideOverlayImage.setImageBitmap(guideBitmap)
                previewStatus.text = if (updated.paperLandscape) {
                    "가로 템플릿방향 기준 미리보기입니다."
                } else {
                    "세로 템플릿방향 기준 미리보기입니다."
                }
            }
        }
        val autoRefreshWatcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                refreshPreview()
            }
        }
        templateShrinkToPaper.setOnCheckedChangeListener { _, _ -> refreshPreview() }
        attributeCountSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) { refreshPreview() }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        attribute1ColumnSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) { refreshPreview() }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        attribute2ColumnSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) { refreshPreview() }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        attribute3ColumnSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) { refreshPreview() }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        listOf(
            regionOffsetX, regionOffsetY, regionWidth, regionHeight,
            printBaseFont, printScale, outputNameSize, outputDepartmentSize,
            outputNameOffsetX, outputNameOffsetY,
            outputDepartmentOffsetX, outputDepartmentOffsetY,
            outputAttribute3Size, outputAttribute3OffsetX, outputAttribute3OffsetY,
            outputReturnDelaySeconds, outputReturnMessageTemplate,
            printStatusReadyMessage, printStatusPermissionMessage, printStatusConnectingMessage,
            printStatusPreparingMessage, printStatusSendingMessage, printStatusSuccessMessage,
            printStatusErrorMessage,
        ).forEach { it.addTextChangedListener(autoRefreshWatcher) }

        val dialog = AlertDialog.Builder(context)
            .setTitle("설정")
            .setView(root)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                onSave(buildUpdatedSettings())
            }
            .show()

        dialog.setOnDismissListener {
            previewBitmap?.recycle()
            previewBitmap = null
            guideBitmap?.recycle()
            guideBitmap = null
        }

        selectTab(0)
        refreshPreview()
    }

    private fun buildBrotherPaperProfileId(media: BrotherDetectedMedia): String {
        val width = media.widthMm ?: 62
        val heightPart = if (media.isInfinite) "continuous" else (media.heightMm ?: width).toString()
        val labelPart = media.labelSize?.name ?: "unknown"
        return "brother_${labelPart.lowercase(Locale.US)}_${width}x$heightPart"
    }

    private fun buildBrotherPaperProfileName(media: BrotherDetectedMedia): String {
        val width = media.widthMm ?: 62
        val labelName = media.labelSize?.name ?: "Brother"
        return if (media.isInfinite) {
            "Brother $labelName ${width}mm 연속용지"
        } else {
            val height = media.heightMm ?: width
            "Brother $labelName ${width}x${height}"
        }
    }

    private fun section(parent: LinearLayout, context: Context, title: String) {
        val t = TextView(context).apply {
            text = title
            textSize = 18f
            setPadding(0, dp(context, 12), 0, dp(context, 8))
        }
        parent.addView(t)
    }

    private fun numberField(parent: LinearLayout, context: Context, label: String, value: String, integer: Boolean): EditText {
        val field = textField(parent, context, label, value)
        field.inputType = if (integer) {
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
        } else {
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
        }
        return field
    }

    private fun textField(parent: LinearLayout, context: Context, label: String, value: String): EditText {
        val box = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(context, 4), 0, dp(context, 4))
        }
        val tv = TextView(context).apply { text = label }
        val et = EditText(context).apply {
            setText(value)
            setSingleLine(true)
        }
        box.addView(tv)
        box.addView(et)
        parent.addView(box)
        return et
    }

    private fun compactNumberPair(
        parent: LinearLayout,
        context: Context,
        leftLabel: String,
        leftValue: String,
        rightLabel: String,
        rightValue: String,
        integer: Boolean,
    ): Pair<EditText, EditText> {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(context, 2), 0, dp(context, 2))
        }
        val left = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT)
            lp.weight = 1f
            lp.marginEnd = dp(context, 4)
            layoutParams = lp
        }
        val right = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT)
            lp.weight = 1f
            lp.marginStart = dp(context, 4)
            layoutParams = lp
        }
        val leftTitle = TextView(context).apply { text = leftLabel }
        val rightTitle = TextView(context).apply { text = rightLabel }
        val leftInput = EditText(context).apply {
            setSingleLine(true)
            setText(leftValue)
            inputType = if (integer) {
                InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
            } else {
                InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
            }
        }
        val rightInput = EditText(context).apply {
            setSingleLine(true)
            setText(rightValue)
            inputType = if (integer) {
                InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
            } else {
                InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
            }
        }
        left.addView(leftTitle)
        left.addView(leftInput)
        right.addView(rightTitle)
        right.addView(rightInput)
        row.addView(left)
        row.addView(right)
        parent.addView(row)
        return leftInput to rightInput
    }

    private fun boolField(parent: LinearLayout, context: Context, label: String, value: Boolean): Switch {
        val sw = Switch(context).apply {
            text = label
            isChecked = value
            setPadding(0, dp(context, 4), 0, dp(context, 4))
        }
        parent.addView(sw)
        return sw
    }

    private fun choiceField(
        parent: LinearLayout,
        context: Context,
        label: String,
        items: List<String>,
        selectedIndex: Int,
    ): Spinner {
        val box = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(context, 4), 0, dp(context, 4))
        }
        val tv = TextView(context).apply { text = label }
        val spinner = Spinner(context)
        val adapter = ArrayAdapter(context, android.R.layout.simple_spinner_item, items).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinner.adapter = adapter
        spinner.setSelection(selectedIndex.coerceIn(0, items.lastIndex.coerceAtLeast(0)))
        box.addView(tv)
        box.addView(spinner)
        parent.addView(box)
        return spinner
    }

    private fun buildGuideOverlayBitmap(width: Int, height: Int, settings: AppSettings): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val rect = resolveRegionRect(width, height, settings)
        val paint = Paint().apply {
            color = Color.argb(220, 50, 50, 50)
            style = Paint.Style.STROKE
            strokeWidth = 2f
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(10f, 10f), 0f)
            isAntiAlias = true
        }
        canvas.drawRect(rect, paint)
        return bitmap
    }

    private fun resolveRegionRect(width: Int, height: Int, settings: AppSettings): RectF {
        if (!settings.paperLandscape) {
            val x = settings.templateTextRegionOffsetX.coerceIn(0, (width - 1).coerceAtLeast(0))
            val y = settings.templateTextRegionOffsetY.coerceIn(0, (height - 1).coerceAtLeast(0))
            val right = (x + settings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(width)
            val bottom = (y + settings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(height)
            val safeRight = if (right <= x) (x + 1).coerceAtMost(width) else right
            val safeBottom = if (bottom <= y) (y + 1).coerceAtMost(height) else bottom
            return RectF(x.toFloat(), y.toFloat(), safeRight.toFloat(), safeBottom.toFloat())
        }
        val portraitWidth = height
        val portraitHeight = width
        val sourceX = settings.templateTextRegionOffsetX.coerceIn(0, (portraitWidth - 1).coerceAtLeast(0))
        val sourceY = settings.templateTextRegionOffsetY.coerceIn(0, (portraitHeight - 1).coerceAtLeast(0))
        val sourceRight = (sourceX + settings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(portraitWidth)
        val sourceBottom = (sourceY + settings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(portraitHeight)
        val rotatedLeft = (portraitHeight - sourceBottom).coerceIn(0, width - 1)
        val rotatedTop = sourceX.coerceIn(0, height - 1)
        val rotatedRight = (portraitHeight - sourceY).coerceIn(rotatedLeft + 1, width)
        val rotatedBottom = sourceRight.coerceIn(rotatedTop + 1, height)
        return RectF(
            rotatedLeft.toFloat(),
            rotatedTop.toFloat(),
            rotatedRight.toFloat(),
            rotatedBottom.toFloat(),
        )
    }

    private fun EditText.asInt(default: Int): Int = text.toString().trim().toIntOrNull() ?: default
    private fun EditText.asFloat(default: Float): Float = text.toString().trim().toFloatOrNull() ?: default

    private fun dp(context: Context, v: Int): Int = (v * context.resources.displayMetrics.density).toInt()

    private fun trimZero(v: Float): String {
        val roundedInt = v.toInt().toFloat()
        return if (kotlin.math.abs(v - roundedInt) < 0.0001f) {
            roundedInt.toInt().toString()
        } else {
            String.format(Locale.US, "%.1f", v)
        }
    }
}
