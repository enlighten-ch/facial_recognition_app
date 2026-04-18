package com.example.facetestbed

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Path
import android.graphics.RectF
import android.os.Build
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.ViewFlipper
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.mangoslab.nemonicsdk.INPrinterControllerCallback
import com.mangoslab.nemonicsdk.NPrintInfo
import com.mangoslab.nemonicsdk.NPrinter
import com.mangoslab.nemonicsdk.NPrinterController
import com.mangoslab.nemonicsdk.constants.NPrintQuality
import com.mangoslab.nemonicsdk.constants.NPrinterType
import com.mangoslab.nemonicsdk.constants.NResult
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    companion object {
        private const val PAGE_MONITOR = 0
        private const val PAGE_REGISTER_CAPTURE = 1
        private const val PAGE_REGISTER_NAME = 2
        private const val PAGE_OUTPUT = 3

        private val REGISTER_DIRECTIONS = listOf("left", "right", "up", "down")
        private val DIRECTION_LABELS = mapOf(
            "left" to "좌",
            "right" to "우",
            "up" to "위",
            "down" to "아래",
        )
        private val DIRECTION_PROMPTS = mapOf(
            "left" to "다음 동작: 얼굴을 화면 기준 왼쪽으로 돌려주세요.",
            "right" to "다음 동작: 얼굴을 화면 기준 오른쪽으로 돌려주세요.",
            "up" to "다음 동작: 턱을 살짝 들고 위를 봐주세요.",
            "down" to "다음 동작: 턱을 살짝 내리고 아래를 봐주세요.",
        )
    }

    private lateinit var headerTitle: TextView
    private lateinit var headerSubtitle: TextView
    private lateinit var connectionStatusIndicator: TextView
    private lateinit var dbLoadStatusIndicator: TextView
    private lateinit var dbPushStatusIndicator: TextView
    private lateinit var dbReconnectButton: Button
    private lateinit var dbDownloadButton: Button
    private lateinit var dbUploadButton: Button
    private lateinit var backButton: Button
    private lateinit var settingsButton: Button
    private var errorLogButton: Button? = null
    private lateinit var quitButton: Button
    private lateinit var previewView: PreviewView
    private lateinit var ellipseOverlay: EllipseMaskOverlayView
    private lateinit var pageFlipper: ViewFlipper

    private lateinit var monitorStatusText: TextView
    private lateinit var monitorDetailText: TextView
    private lateinit var registerStartButton: Button
    private lateinit var candidateContainer: LinearLayout
    private lateinit var outputNameButton: Button

    private lateinit var registerStatusText: TextView
    private lateinit var registerProgressBar: ProgressBar
    private lateinit var registerCountText: TextView

    private lateinit var registerCompleteText: TextView
    private lateinit var nameInput: EditText
    private lateinit var departmentInput: EditText
    private lateinit var saveNameButton: Button

    private lateinit var outputLabel: TextView
    private lateinit var outputDepartmentLabel: TextView
    private lateinit var outputSubLabel: TextView

    private lateinit var faceDb: FaceDatabase
    private lateinit var faceEngine: FaceRecognitionEngine
    private lateinit var settingsStore: AppSettingsStore
    private var settings = AppSettings()
    private lateinit var dbSyncClient: FaceDbSyncClient

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val dbUpdateExecutor = Executors.newSingleThreadExecutor()
    private val logExecutor = Executors.newSingleThreadExecutor()
    private val frameInFlight = AtomicBoolean(false)
    private var cameraProvider: ProcessCameraProvider? = null

    private var hasCameraFrame = false
    private var analyzeFrameTick = 0L
    private var registerFrameTick = 0L
    private var lastImageWidth = 0
    private var lastImageHeight = 0

    private var selectedName: String? = null
    private var selectedDepartment: String = ""
    private var latestRankings: List<FaceMatch> = emptyList()
    private var lastMonitorStateKey: String? = null
    @Volatile
    private var currentPageIndex: Int = PAGE_MONITOR

    private var adaptiveLastFrame = -1_000_000_000L
    private var adaptiveUpdateCount = 0
    private var pendingFolderPickerCallback: ((String?) -> Unit)? = null
    private var pendingTemplatePickerCallback: ((String?) -> Unit)? = null
    private var pendingNemonicPrintName: String? = null
    private var pendingNemonicPrintDepartment: String = ""
    private var connectionStatus = SyncLightStatus.IDLE
    private var dbLoadStatus = SyncLightStatus.IDLE
    private var dbPushStatus = SyncLightStatus.IDLE
    @Volatile
    private var isDbServerConnected = false
    @Volatile
    private var lastRecognitionAtMs: Long = System.currentTimeMillis()
    private val adaptivePendingLock = Any()
    private val pendingAdaptiveEmbeddings = mutableMapOf<String, MutableList<FloatArray>>()
    private val syncStatusDedupLock = Any()
    private var lastSyncStatusMessage: String = ""
    private var lastSyncStatusBusy: Boolean = false
    private val syncLogLock = Any()
    private val syncLogBuffer = ArrayDeque<String>()
    private var connectionRetryLabel: String? = null
    private lateinit var syncLogFile: File
    @Volatile
    private var adaptiveFlushScheduled: Boolean = false
    @Volatile
    private var uploadProgressText: String = ""
    private val uiHandler = Handler(Looper.getMainLooper())
    private var outputAutoReturnRunnable: Runnable? = null
    private var outputCountdownRemainingSeconds: Int = 0
    private var outputPrintStatusMessage: String = ""
    @Volatile
    private var isPreviewMirrored: Boolean = false

    private val registerDirectionEmbeddings = mutableMapOf(
        "left" to mutableListOf<FloatArray>(),
        "right" to mutableListOf<FloatArray>(),
        "up" to mutableListOf<FloatArray>(),
        "down" to mutableListOf<FloatArray>(),
    )

    private enum class SyncLightStatus {
        IDLE,
        IN_PROGRESS,
        SUCCESS,
        ERROR,
    }

    private enum class MonitorDecisionType {
        IDLE_OUTSIDE,
        UNREGISTERED,
        CLEAR,
        AMBIGUOUS,
    }

    private data class MonitorDecision(
        val stateKey: String,
        val type: MonitorDecisionType,
        val rankings: List<FaceMatch> = emptyList(),
        val topMatch: FaceMatch? = null,
        val topScore: Float = 0f,
        val margin: Float = 0f,
    )

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showError("카메라 권한이 필요합니다.")
        }

    private val folderPickerLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            val callback = pendingFolderPickerCallback
            pendingFolderPickerCallback = null
            if (uri == null) {
                callback?.invoke(null)
                return@registerForActivityResult
            }
            val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            runCatching { contentResolver.takePersistableUriPermission(uri, flags) }
            callback?.invoke(uri.toString())
        }

    private val templatePickerLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            val callback = pendingTemplatePickerCallback
            pendingTemplatePickerCallback = null
            if (uri == null) {
                callback?.invoke(null)
                return@registerForActivityResult
            }
            val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
            runCatching { contentResolver.takePersistableUriPermission(uri, flags) }
            callback?.invoke(uri.toString())
        }

    private val nemonicPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val targetName = pendingNemonicPrintName
            pendingNemonicPrintName = null
            val targetDepartment = pendingNemonicPrintDepartment
            pendingNemonicPrintDepartment = ""
            if (targetName.isNullOrBlank()) return@registerForActivityResult
            val granted = grants.entries.all { it.value }
            if (!granted) {
                setOutputPrintStatus("네모닉 출력 권한이 거부되었습니다.")
                return@registerForActivityResult
            }
            printWithNemonic(targetName, targetDepartment)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()

        settingsStore = AppSettingsStore(this)
        settings = settingsStore.load()
        syncLogFile = File(filesDir, "logs/db_sync_client.log").also { it.parentFile?.mkdirs() }
        appendSyncLog("앱 시작")
        ensureSyncIdentity()
        faceDb = createFaceDatabase(settings)
        dbSyncClient = createDbSyncClient()
        dbSyncClient.start(settings)
        faceEngine = FaceRecognitionEngine(this)
        applyEllipseSettings()

        registerStartButton.setOnClickListener { startAutoRegistration() }
        outputNameButton.setOnClickListener { outputSelectedName() }
        saveNameButton.setOnClickListener { saveRegistration() }
        backButton.setOnClickListener { goInitialState() }
        settingsButton.setOnClickListener { openSettingsDialog() }
        errorLogButton?.setOnClickListener { showSyncLogDialog() }
        dbReconnectButton.setOnClickListener { requestDbReconnect() }
        dbDownloadButton.setOnClickListener { requestDbDownload() }
        dbUploadButton.setOnClickListener { requestDbUpload() }
        quitButton.setOnClickListener { finish() }

        goInitialState()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindViews() {
        headerTitle = findViewById(R.id.headerTitle)
        headerSubtitle = findViewById(R.id.headerSubtitle)
        connectionStatusIndicator = findViewById(R.id.connectionStatusIndicator)
        dbLoadStatusIndicator = findViewById(R.id.dbLoadStatusIndicator)
        dbPushStatusIndicator = findViewById(R.id.dbPushStatusIndicator)
        dbReconnectButton = findViewById(R.id.dbReconnectButton)
        dbDownloadButton = findViewById(R.id.dbDownloadButton)
        dbUploadButton = findViewById(R.id.dbUploadButton)
        backButton = findViewById(R.id.backButton)
        settingsButton = findViewById(R.id.settingsButton)
        errorLogButton = findViewById(R.id.errorLogButton)
        quitButton = findViewById(R.id.quitButton)
        previewView = findViewById(R.id.previewView)
        ellipseOverlay = findViewById(R.id.ellipseOverlay)
        pageFlipper = findViewById(R.id.pageFlipper)

        monitorStatusText = findViewById(R.id.monitorStatusText)
        monitorDetailText = findViewById(R.id.monitorDetailText)
        registerStartButton = findViewById(R.id.registerStartButton)
        candidateContainer = findViewById(R.id.candidateContainer)
        outputNameButton = findViewById(R.id.outputNameButton)

        registerStatusText = findViewById(R.id.registerStatusText)
        registerProgressBar = findViewById(R.id.registerProgressBar)
        registerCountText = findViewById(R.id.registerCountText)

        registerCompleteText = findViewById(R.id.registerCompleteText)
        nameInput = findViewById(R.id.nameInput)
        departmentInput = findViewById(R.id.departmentInput)
        saveNameButton = findViewById(R.id.saveNameButton)

        outputLabel = findViewById(R.id.outputLabel)
        outputDepartmentLabel = findViewById(R.id.outputDepartmentLabel)
        outputSubLabel = findViewById(R.id.outputSubLabel)
    }

    private fun currentPage(): Int = currentPageIndex

    private fun goTo(page: Int) {
        currentPageIndex = page
        pageFlipper.displayedChild = page
        updateEllipseOverlay()
        when (page) {
            PAGE_MONITOR -> {
                headerTitle.text = "얼굴 인식"
                headerSubtitle.text = "카메라를 보고 있으면 자동으로 등록/인식이 진행됩니다."
                backButton.visibility = Button.GONE
            }
            PAGE_REGISTER_CAPTURE -> {
                headerTitle.text = "얼굴 등록"
                headerSubtitle.text = "랜드마크 기반으로 좌/우/위/아래 샘플을 자동 수집합니다."
                backButton.visibility = Button.VISIBLE
            }
            PAGE_REGISTER_NAME -> {
                headerTitle.text = "이름 저장"
                headerSubtitle.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
                backButton.visibility = Button.VISIBLE
            }
            PAGE_OUTPUT -> {
                headerTitle.text = "이름 출력"
                headerSubtitle.text = "선택된 이름을 출력합니다."
                backButton.visibility = Button.VISIBLE
            }
        }
    }

    private fun registrationTotalTarget(): Int {
        return settings.registerSamplesPerDirection.coerceAtLeast(1) * REGISTER_DIRECTIONS.size
    }

    private fun resetRegistrationState() {
        REGISTER_DIRECTIONS.forEach { registerDirectionEmbeddings[it]?.clear() }
        registerFrameTick = 0
        setRegisterCount()
    }

    private fun totalRegisteredSamples(): Int {
        return REGISTER_DIRECTIONS.sumOf { registerDirectionEmbeddings[it]?.size ?: 0 }
    }

    private fun setRegisterCount() {
        val total = registrationTotalTarget()
        val current = totalRegisteredSamples()
        registerProgressBar.max = total
        registerProgressBar.progress = current.coerceAtMost(total)
        registerCountText.text = "$current / $total"
    }

    private fun nextTargetDirection(): String? {
        val need = settings.registerSamplesPerDirection.coerceAtLeast(1)
        for (dir in REGISTER_DIRECTIONS) {
            if ((registerDirectionEmbeddings[dir]?.size ?: 0) < need) return dir
        }
        return null
    }

    private fun summarizeDirectionProgress(): String {
        val need = settings.registerSamplesPerDirection.coerceAtLeast(1)
        return REGISTER_DIRECTIONS.joinToString(" | ") { dir ->
            val c = registerDirectionEmbeddings[dir]?.size ?: 0
            val label = DIRECTION_LABELS[dir] ?: dir
            "$label:$c/$need"
        }
    }

    private fun allRegisteredEmbeddings(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        REGISTER_DIRECTIONS.forEach { out.addAll(registerDirectionEmbeddings[it] ?: emptyList()) }
        return out
    }

    private fun goInitialState() {
        cancelOutputAutoReturnCountdown(clearTextState = true)
        resetRegistrationState()
        selectedName = null
        selectedDepartment = ""
        latestRankings = emptyList()
        lastMonitorStateKey = null
        adaptiveLastFrame = -1_000_000_000L
        adaptiveUpdateCount = 0
        nameInput.setText("")
        departmentInput.setText("")
        outputLabel.text = ""
        outputDepartmentLabel.text = ""
        if (isDbServerConnected) {
            applyMonitorIdleState()
        } else {
            applyServerDisconnectedState()
        }
        goTo(PAGE_MONITOR)
    }

    private fun clearCandidates() {
        candidateContainer.removeAllViews()
    }

    private fun applyMonitorIdleState() {
        monitorStatusText.text = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요"
        monitorDetailText.text = ""
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        outputNameButton.isEnabled = false
        clearCandidates()
    }

    private fun applyServerDisconnectedState() {
        monitorStatusText.text = "서버 앱 실행 및 연동 후 사용해주세요"
        monitorDetailText.text = "DB 서버 연결 이후 인식 기능이 자동으로 활성화됩니다."
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        outputNameButton.isEnabled = false
        clearCandidates()
    }

    private fun ensureDbSyncEnabledForManualAction(actionName: String): Boolean {
        if (settings.dbSyncEnabled) return true
        showError("설정에서 DB 서버 동기화를 활성화한 뒤 '$actionName'을 사용해 주세요.")
        return false
    }

    private fun requestDbReconnect() {
        if (!ensureDbSyncEnabledForManualAction("DB 서버 연결 복구")) return
        monitorDetailText.text = "수동 복구 요청: DB 서버 연결을 다시 시도합니다."
        dbSyncClient.start(settings)
    }

    private fun requestDbDownload() {
        if (!ensureDbSyncEnabledForManualAction("DB 다운로드")) return
        monitorDetailText.text = "수동 다운로드 요청: 서버 DB를 가져옵니다."
        dbSyncClient.pullLatest("수동 다운로드")
    }

    private fun requestDbUpload() {
        if (!ensureDbSyncEnabledForManualAction("DB 업로드")) return
        monitorDetailText.text = "수동 업로드 요청: 로컬 DB를 서버로 전송합니다."
        dbSyncClient.pushLatest("수동 업로드")
    }

    private fun applyUnregisteredState() {
        monitorStatusText.text = "등록이 되어있지 않아요! 등록할까요?"
        monitorDetailText.text = "등록시작 버튼을 누르면 좌/우/위/아래 방향별로 ${settings.registerSamplesPerDirection}장씩 수집합니다."
        registerStartButton.visibility = Button.VISIBLE
        registerStartButton.isEnabled = true
        outputNameButton.isEnabled = false
        clearCandidates()
    }

    private fun applyRegisteredClearState(topMatch: FaceMatch, topScore: Float, margin: Float) {
        val topName = topMatch.name
        val topDepartment = topMatch.department
        val userLabel = if (topDepartment.isBlank()) topName else "$topName ($topDepartment)"
        monitorStatusText.text = "등록된 사용자입니다: $userLabel"
        monitorDetailText.text = String.format(Locale.US, "1위 점수 %.3f, 2위와 차이 %.3f", topScore, margin)
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        clearCandidates()

        val btn = Button(this).apply {
            text = "1위 $userLabel"
            isEnabled = false
        }
        candidateContainer.addView(btn)

        selectedName = topName
        selectedDepartment = topDepartment
        outputNameButton.isEnabled = true
    }

    private fun applyRegisteredAmbiguousState(rankings: List<FaceMatch>) {
        monitorStatusText.text = "유사한 얼굴이 있습니다. 이름을 선택해 주세요."
        monitorDetailText.text = "1~3위 후보 중 이름 선택 후 이름 출력하기 버튼이 활성화됩니다."
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        clearCandidates()
        selectedName = null
        selectedDepartment = ""
        outputNameButton.isEnabled = false

        rankings.forEachIndexed { idx, match ->
            val btn = Button(this).apply {
                val label = if (match.department.isBlank()) match.name else "${match.name} (${match.department})"
                text = String.format(Locale.US, "%d위 선택: %s (점수 %.3f)", idx + 1, label, match.score)
                setOnClickListener { chooseCandidateName(match) }
            }
            candidateContainer.addView(btn)
        }
    }

    private fun chooseCandidateName(match: FaceMatch) {
        selectedName = match.name
        selectedDepartment = match.department
        val label = if (match.department.isBlank()) match.name else "${match.name} (${match.department})"
        monitorDetailText.text = "선택됨: $label / 이름 출력하기 버튼을 눌러주세요."
        outputNameButton.isEnabled = true
    }

    private fun outputSelectedName() {
        val name = selectedName
        if (name.isNullOrBlank()) {
            showError("먼저 이름을 선택해 주세요.")
            return
        }
        val department = selectedDepartment.ifBlank { faceDb.departmentOf(name) }
        outputLabel.text = name
        outputDepartmentLabel.text = department
        outputLabel.textSize = settings.outputNameTextSizeSp.toFloat()
        outputDepartmentLabel.textSize = settings.outputDepartmentTextSizeSp.toFloat()
        outputDepartmentLabel.translationX = settings.outputDepartmentOffsetX.toFloat()
        outputDepartmentLabel.translationY = settings.outputDepartmentOffsetY.toFloat()
        outputDepartmentLabel.visibility = if (department.isBlank()) View.GONE else View.VISIBLE
        setOutputPrintStatus("출력 처리 중...")
        goTo(PAGE_OUTPUT)
        dispatchPrint(name, department)
        startOutputAutoReturnCountdown()
    }

    private fun dispatchPrint(name: String, department: String) {
        if (!settings.printEnabled) {
            setOutputPrintStatus(if (department.isBlank()) {
                "이름이 출력되었습니다."
            } else {
                "이름/부서가 출력되었습니다."
            })
            return
        }

        if (settings.printerModel == PrinterModel.NEMONIC) {
            if (!hasNemonicPermissions()) {
                pendingNemonicPrintName = name
                pendingNemonicPrintDepartment = department
                nemonicPermissionLauncher.launch(requiredNemonicPermissions())
                setOutputPrintStatus("네모닉 권한 요청 중...")
                return
            }
            printWithNemonic(name, department)
            return
        }

        setOutputPrintStatus(if (department.isBlank()) {
            "기본 프린터로 출력 전송됨: $name"
        } else {
            "기본 프린터로 출력 전송됨: $name / $department"
        })
    }

    private fun hasNemonicPermissions(): Boolean {
        return requiredNemonicPermissions().all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requiredNemonicPermissions(): Array<String> {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions += Manifest.permission.BLUETOOTH_SCAN
            permissions += Manifest.permission.BLUETOOTH_CONNECT
        } else {
            permissions += Manifest.permission.ACCESS_FINE_LOCATION
        }
        return permissions.toTypedArray()
    }

    private fun printWithNemonic(name: String, department: String) {
        val macAddress = settings.nemonicPrinterMacAddress.trim()
        if (macAddress.isBlank()) {
            setOutputPrintStatus("네모닉 MAC 주소를 설정해 주세요.")
            return
        }

        val printerName = settings.printerName.trim().ifBlank { "Nemonic" }
        setOutputPrintStatus("네모닉 프린터로 전송 중...")

        Thread {
            val controller = NPrinterController(
                this,
                object : INPrinterControllerCallback {
                    override fun disconnected() {}
                    override fun printProgress(current: Int, total: Int, progress: Int) {}
                    override fun printComplete(result: Int) {}
                },
            )

            try {
                controller.setPrintTimeout(true, settings.printerTimeoutSec.coerceAtLeast(1))
                val printer = NPrinter(NPrinterType.NEMONIC, printerName, macAddress)
                val connectResult = controller.connect(printer)
                if (connectResult != NResult.OK && connectResult != NResult.ALREADY_CONNECTED) {
                    runOnUiThread {
                        setOutputPrintStatus("네모닉 연결 실패: 코드 $connectResult")
                    }
                    return@Thread
                }

                val bitmap = buildNemonicPrintBitmap(name, department)
                val printInfo = NPrintInfo(printer, bitmap)
                    .setCopies(settings.printerCopies.coerceAtLeast(1))
                    .setEnableLastPageCut(true)
                    .setEnableDither(true)
                    .setPrintQuality(NPrintQuality.MIDDLE)

                val printResult = controller.print(printInfo)
                bitmap.recycle()

                runOnUiThread {
                    setOutputPrintStatus(if (printResult == NResult.OK) {
                        if (department.isBlank()) "네모닉 출력 전송 완료: $name" else "네모닉 출력 전송 완료: $name / $department"
                    } else {
                        "네모닉 출력 실패: 코드 $printResult"
                    })
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    setOutputPrintStatus("네모닉 출력 오류: ${t.javaClass.simpleName}")
                }
            } finally {
                runCatching { controller.disconnect() }
            }
        }.start()
    }

    private fun resolvePrintText(name: String, department: String, activeSettings: AppSettings = settings): String {
        val template = activeSettings.printerContentTemplate.ifBlank { "Name: {name}\nDepartment: {department}\n" }
        return template
            .replace("{name}", name)
            .replace("{department}", department)
    }

    private fun buildNemonicPrintBitmap(
        name: String,
        department: String,
        activeSettings: AppSettings = settings,
        applyLandscapeRotation: Boolean = true,
    ): Bitmap {
        val paper = resolveSelectedPaper(activeSettings)
        val pixelsPerMm = 8.0f // 203dpi thermal printer ~= 8 dot/mm
        val printWidth = (paper.widthMm * pixelsPerMm).toInt().coerceIn(200, 1200)
        val printHeight = (paper.heightMm * pixelsPerMm).toInt().coerceIn(120, 1600)
        val scale = activeSettings.printTextScaleMultiplier.coerceAtLeast(1)
        val nameFontSize = (activeSettings.outputNameTextSizeSp * scale).coerceIn(16, 240)
        val departmentFontSize = (activeSettings.outputDepartmentTextSizeSp * scale).coerceIn(12, 180)

        val namePaint = TextPaint().apply {
            color = Color.BLACK
            textSize = nameFontSize.toFloat()
            isAntiAlias = true
            isFakeBoldText = true
        }

        val departmentPaint = TextPaint().apply {
            color = Color.BLACK
            textSize = departmentFontSize.toFloat()
            isAntiAlias = true
            isFakeBoldText = true
        }

        val textRegion = resolveTemplateTextRegion(activeSettings, printWidth, printHeight)
        val baseX = textRegion.left + 8f
        val baselineY = (textRegion.top + nameFontSize + 8f).coerceAtMost(textRegion.bottom - 8f)
        val departmentX = baseX + (activeSettings.outputDepartmentOffsetX * scale)
        val departmentY = (baselineY + departmentFontSize + 10f + (activeSettings.outputDepartmentOffsetY * scale))
            .coerceAtMost(textRegion.bottom - 6f)

        val templateBitmap = loadTemplateBitmap(
            uriString = activeSettings.printerTemplateImageUri,
            printWidth = printWidth,
            printHeight = printHeight,
            shrinkToPaperIfOversized = activeSettings.templateShrinkToPaperIfOversized,
        )
        if (templateBitmap != null && templateBitmap.width == printWidth && templateBitmap.height == printHeight) {
            val canvas = Canvas(templateBitmap)
            canvas.drawText(name, baseX, baselineY, namePaint)
            if (department.isNotBlank()) {
                canvas.drawText(department, departmentX, departmentY, departmentPaint)
            }
            return rotateForLandscapeIfNeeded(templateBitmap, activeSettings, applyLandscapeRotation)
        }

        val extraTemplateText = resolvePrintText(name, department, activeSettings).trim()
        val extraPaint = TextPaint().apply {
            color = Color.DKGRAY
            textSize = (activeSettings.printTextBaseFontSize.coerceAtLeast(12) * 0.9f)
            isAntiAlias = true
        }
        val extraLayout = StaticLayout.Builder
            .obtain(extraTemplateText, 0, extraTemplateText.length, extraPaint, printWidth - 48)
            .setAlignment(Layout.Alignment.ALIGN_NORMAL)
            .setIncludePad(true)
            .setLineSpacing(0f, 1.1f)
            .build()

        val contentBottom = maxOf(
            textRegion.bottom,
            if (department.isBlank()) baselineY + nameFontSize else departmentY + departmentFontSize,
        )
        val textTop = (contentBottom + 20).toInt()
        val fallbackHeight = (textTop + extraLayout.height + 24).coerceAtLeast(180)
        val bitmapHeight = maxOf(printHeight, fallbackHeight)
        val bitmap = Bitmap.createBitmap(printWidth, bitmapHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)
        canvas.drawText(name, baseX, baselineY, namePaint)
        if (department.isNotBlank()) {
            canvas.drawText(department, departmentX, departmentY, departmentPaint)
        }
        canvas.save()
        canvas.translate(24f, textTop.toFloat())
        extraLayout.draw(canvas)
        canvas.restore()
        return rotateForLandscapeIfNeeded(bitmap, activeSettings, applyLandscapeRotation)
    }

    private fun rotateForLandscapeIfNeeded(
        bitmap: Bitmap,
        activeSettings: AppSettings,
        applyLandscapeRotation: Boolean,
    ): Bitmap {
        if (!applyLandscapeRotation || !activeSettings.printLandscape) return bitmap
        val matrix = Matrix().apply { postRotate(90f) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) {
            bitmap.recycle()
        }
        return rotated
    }

    private fun loadTemplateBitmap(
        uriString: String,
        printWidth: Int,
        printHeight: Int,
        shrinkToPaperIfOversized: Boolean,
    ): Bitmap? {
        val sourceUri = uriString.trim()
        if (sourceUri.isBlank()) return null
        return runCatching {
            val uri = Uri.parse(sourceUri)
            val decoded = openTemplateImageStream(uri.toString())?.use { BitmapFactory.decodeStream(it) } ?: return null
            if (decoded.width <= 0 || decoded.height <= 0) {
                decoded.recycle()
                return null
            }
            val target = Bitmap.createBitmap(printWidth, printHeight, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(target)
            canvas.drawColor(Color.WHITE)

            val scale = if (shrinkToPaperIfOversized) {
                minOf(
                    1f,
                    printWidth.toFloat() / decoded.width.toFloat(),
                    printHeight.toFloat() / decoded.height.toFloat(),
                )
            } else {
                1f
            }
            val drawWidth = (decoded.width * scale).toInt().coerceAtLeast(1)
            val drawHeight = (decoded.height * scale).toInt().coerceAtLeast(1)
            val left = ((printWidth - drawWidth) / 2f)
            val top = ((printHeight - drawHeight) / 2f)
            val scaled = Bitmap.createScaledBitmap(decoded, drawWidth, drawHeight, true)
            canvas.drawBitmap(scaled, left, top, null)
            if (scaled !== decoded) scaled.recycle()
            decoded.recycle()

            val mutable = target.copy(Bitmap.Config.ARGB_8888, true)
            if (mutable !== target) target.recycle()
            mutable
        }.getOrNull()
    }

    private fun openTemplateImageStream(sourceUri: String): InputStream? {
        val raw = sourceUri.trim()
        if (raw.isBlank()) return null
        return runCatching {
            when {
                raw.startsWith("content://") || raw.startsWith("file://") -> {
                    contentResolver.openInputStream(Uri.parse(raw))
                }
                else -> {
                    val file = File(raw)
                    if (file.exists() && file.isFile) {
                        FileInputStream(file)
                    } else {
                        contentResolver.openInputStream(Uri.parse(raw))
                    }
                }
            }
        }.getOrNull()
    }

    private fun resolveSelectedPaper(activeSettings: AppSettings): PaperProfile {
        val selected = activeSettings.paperProfiles.firstOrNull { it.id == activeSettings.selectedPaperId }
        if (selected != null) return selected
        return activeSettings.paperProfiles.firstOrNull()
            ?: PaperProfile(id = "fallback_72x50", name = "Fallback 72x50", widthMm = 72f, heightMm = 50f)
    }

    private fun resolveTemplateTextRegion(activeSettings: AppSettings, width: Int, height: Int): RectF {
        val x = activeSettings.templateTextRegionOffsetX.coerceIn(0, (width - 1).coerceAtLeast(0))
        val y = activeSettings.templateTextRegionOffsetY.coerceIn(0, (height - 1).coerceAtLeast(0))
        val right = (x + activeSettings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(width)
        val bottom = (y + activeSettings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(height)
        val safeRight = if (right <= x) (x + 1).coerceAtMost(width) else right
        val safeBottom = if (bottom <= y) (y + 1).coerceAtMost(height) else bottom
        return RectF(x.toFloat(), y.toFloat(), safeRight.toFloat(), safeBottom.toFloat())
    }

    private fun startAutoRegistration() {
        if (!isDbServerConnected) {
            applyServerDisconnectedState()
            return
        }
        if (!hasCameraFrame) {
            showError("카메라 프레임이 아직 없습니다.")
            return
        }
        resetRegistrationState()
        val firstDir = nextTargetDirection()
        val firstPrompt = if (firstDir != null) DIRECTION_PROMPTS[firstDir] ?: "" else ""
        registerStatusText.text = "$firstPrompt\n진행: ${summarizeDirectionProgress()}"
        goTo(PAGE_REGISTER_CAPTURE)
    }

    private fun evaluateSimilarityBand(embedding: FloatArray): Pair<Boolean, String> {
        val existing = allRegisteredEmbeddings()
        if (existing.isEmpty()) return true to ""

        val maxSim = existing.maxOf { dot(it, embedding) }
        if (maxSim > settings.registerSimilarityHigherBoundary) {
            return false to String.format(Locale.US, "중복 샘플입니다. 조금 더 다른 각도로 움직여주세요. (max=%.3f)", maxSim)
        }
        if (maxSim < settings.registerSimilarityLowerBoundary) {
            return false to String.format(Locale.US, "유사도가 낮습니다. 같은 사람이 카메라 앞에 있어야 합니다. (max=%.3f)", maxSim)
        }
        return true to String.format(Locale.US, "유사도 적합 구간 (max=%.3f)", maxSim)
    }

    private fun handleRegisterCapture(face: Face, embedding: FloatArray) {
        registerFrameTick += 1
        if (registerFrameTick % settings.registerCaptureEveryNFrames.coerceAtLeast(1) != 0L) return

        val targetDirection = nextTargetDirection()
        if (targetDirection == null) {
            registerStatusText.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
            goTo(PAGE_REGISTER_NAME)
            return
        }

        val center = face.boundingBox.centerX() to face.boundingBox.centerY()
        val ellipse = getRegisterEllipse(lastImageWidth, lastImageHeight)
        if (!isInsideEllipse(center.first, center.second, ellipse.first.first, ellipse.first.second, ellipse.second.first, ellipse.second.second)) {
            registerStatusText.text = "얼굴을 타원 가이드 안으로 맞춰주세요.\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        val detectedDirection = faceEngine.classifyDirection(face, mirroredPreview = isPreviewMirrored)
        if (detectedDirection != targetDirection) {
            val detectedLabel = DIRECTION_LABELS[detectedDirection] ?: "정면"
            registerStatusText.text =
                "현재 감지 방향: $detectedLabel\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        val (accepted, msg) = evaluateSimilarityBand(embedding)
        if (!accepted) {
            registerStatusText.text = "$msg\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        registerDirectionEmbeddings[targetDirection]?.add(embedding)
        setRegisterCount()

        val nextDirection = nextTargetDirection()
        if (nextDirection == null) {
            registerStatusText.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
            goTo(PAGE_REGISTER_NAME)
            return
        }

        registerStatusText.text = "샘플 저장 완료. $msg\n${DIRECTION_PROMPTS[nextDirection]}\n진행: ${summarizeDirectionProgress()}"
    }

    private fun saveRegistration() {
        val name = nameInput.text?.toString()?.trim().orEmpty()
        val department = departmentInput.text?.toString()?.trim().orEmpty()
        if (name.isBlank()) {
            showError("이름을 입력해 주세요.")
            return
        }
        if (department.isBlank()) {
            showError("부서를 입력해 주세요.")
            return
        }

        val embeddings = allRegisteredEmbeddings()
        if (embeddings.size < registrationTotalTarget()) {
            showError("얼굴 샘플이 아직 충분하지 않습니다.")
            return
        }

        saveNameButton.isEnabled = false
        registerCompleteText.text = "저장 중입니다. 잠시만 기다려주세요..."
        dbUpdateExecutor.execute {
            runCatching {
                faceDb.upsertPerson(name, department, embeddings, settings.maxEmbeddingsPerPerson)
                dbSyncClient.pushLatest("신규 얼굴 등록")
            }.onSuccess {
                runOnUiThread {
                    saveNameButton.isEnabled = true
                    goInitialState()
                }
            }.onFailure {
                runOnUiThread {
                    saveNameButton.isEnabled = true
                    showError("등록 저장 실패: ${it.javaClass.simpleName}")
                }
            }
        }
    }

    private fun tryAdaptiveUpdate(personName: String, topScore: Float, embedding: FloatArray): Boolean {
        if (!settings.adaptiveUpdateEnabled) return false
        if (topScore < settings.authSimThreshold || topScore > settings.adaptiveUpdateMaxSim) return false
        if (adaptiveUpdateCount >= settings.adaptiveUpdateMaxSamplesPerSession) return false
        if ((analyzeFrameTick - adaptiveLastFrame) < settings.adaptiveUpdateMinIntervalFrames) return false

        synchronized(adaptivePendingLock) {
            val bucket = pendingAdaptiveEmbeddings.getOrPut(personName) { mutableListOf() }
            bucket.add(embedding.copyOf())
        }
        adaptiveLastFrame = analyzeFrameTick
        adaptiveUpdateCount += 1
        scheduleAdaptiveFlush()
        return true
    }

    private fun scheduleAdaptiveFlush() {
        if (adaptiveFlushScheduled) return
        adaptiveFlushScheduled = true
        dbUpdateExecutor.execute {
            try {
                while (true) {
                    val snapshot = synchronized(adaptivePendingLock) {
                        if (pendingAdaptiveEmbeddings.isEmpty()) return@execute
                        pendingAdaptiveEmbeddings.mapValues { (_, list) -> list.toList() }.also { pendingAdaptiveEmbeddings.clear() }
                    }

                    while (!isRecognitionIdle(settings.adaptiveUpdateIdleSeconds.coerceAtLeast(1))) {
                        Thread.sleep(300)
                    }

                    var appliedCount = 0
                    snapshot.forEach { (personName, samples) ->
                        appliedCount += faceDb.appendSamplesToPerson(
                            name = personName,
                            embeddings = samples,
                            maxEmbeddingsPerPerson = settings.maxEmbeddingsPerPerson,
                        )
                    }
                    if (appliedCount > 0) {
                        // 이미 유휴 상태에서 DB 업데이트가 완료된 후이므로 waitForIdle=false
                        dbSyncClient.pushLatest("자동 임베딩 업데이트(유휴 적용)", waitForIdle = false)
                    }
                }
            } finally {
                adaptiveFlushScheduled = false
                val hasPending = synchronized(adaptivePendingLock) { pendingAdaptiveEmbeddings.isNotEmpty() }
                if (hasPending) {
                    scheduleAdaptiveFlush()
                }
            }
        }
    }

    private fun isRecognitionIdle(idleSeconds: Int): Boolean {
        val idleMs = idleSeconds.coerceAtLeast(1) * 1000L
        return System.currentTimeMillis() - lastRecognitionAtMs >= idleMs
    }

    private fun setOutputPrintStatus(message: String) {
        outputPrintStatusMessage = message
        refreshOutputSubLabel()
    }

    private fun refreshOutputSubLabel() {
        val countdownText = if (outputCountdownRemainingSeconds > 0) {
            buildReturnCountdownText(outputCountdownRemainingSeconds)
        } else {
            ""
        }
        outputSubLabel.text = when {
            outputPrintStatusMessage.isBlank() -> countdownText
            countdownText.isBlank() -> outputPrintStatusMessage
            else -> "$outputPrintStatusMessage\n$countdownText"
        }
    }

    private fun buildReturnCountdownText(seconds: Int): String {
        val template = settings.outputReturnMessageTemplate.ifBlank { "{seconds}초 후 기본 화면으로 복귀합니다" }
        val sec = seconds.coerceAtLeast(0).toString()
        return template
            .replace("{seconds}", sec)
            .replace("N", sec)
    }

    private fun startOutputAutoReturnCountdown() {
        cancelOutputAutoReturnCountdown(clearTextState = false)
        outputCountdownRemainingSeconds = settings.outputReturnDelaySeconds.coerceAtLeast(1)
        val runnable = object : Runnable {
            override fun run() {
                if (currentPage() != PAGE_OUTPUT) {
                    cancelOutputAutoReturnCountdown(clearTextState = false)
                    return
                }
                if (outputCountdownRemainingSeconds <= 0) {
                    goInitialState()
                    return
                }
                refreshOutputSubLabel()
                outputCountdownRemainingSeconds -= 1
                uiHandler.postDelayed(this, 1000L)
            }
        }
        outputAutoReturnRunnable = runnable
        uiHandler.post(runnable)
    }

    private fun cancelOutputAutoReturnCountdown(clearTextState: Boolean) {
        outputAutoReturnRunnable?.let { uiHandler.removeCallbacks(it) }
        outputAutoReturnRunnable = null
        outputCountdownRemainingSeconds = 0
        if (clearTextState) {
            outputPrintStatusMessage = ""
        }
    }

    private fun computeMonitorDecision(face: Face, embedding: FloatArray): MonitorDecision {
        val center = face.boundingBox.centerX() to face.boundingBox.centerY()
        val ellipse = getRegisterEllipse(lastImageWidth, lastImageHeight)
        if (!isInsideEllipse(center.first, center.second, ellipse.first.first, ellipse.first.second, ellipse.second.first, ellipse.second.second)) {
            return MonitorDecision(
                stateKey = "idle:outside",
                type = MonitorDecisionType.IDLE_OUTSIDE,
            )
        }

        val rankings = faceDb.rank(embedding, settings.topK.coerceAtLeast(1))

        if (rankings.isEmpty()) {
            return MonitorDecision(
                stateKey = "unknown:none",
                type = MonitorDecisionType.UNREGISTERED,
            )
        }

        val topMatch = rankings[0]
        val topScore = topMatch.score
        if (topScore < settings.authSimThreshold) {
            return MonitorDecision(
                stateKey = "unknown:score:${String.format(Locale.US, "%.3f", topScore)}",
                type = MonitorDecisionType.UNREGISTERED,
                rankings = rankings,
            )
        }

        val secondScore = rankings.getOrNull(1)?.score ?: -1f
        val margin = if (secondScore >= 0f) topScore - secondScore else topScore

        if (margin >= settings.authMarginThreshold) {
            tryAdaptiveUpdate(topMatch.name, topScore, embedding)
            return MonitorDecision(
                stateKey = "known:clear:${topMatch.name}:${topMatch.department}:${String.format(Locale.US, "%.3f", topScore)}:${String.format(Locale.US, "%.3f", margin)}",
                type = MonitorDecisionType.CLEAR,
                rankings = rankings,
                topMatch = topMatch,
                topScore = topScore,
                margin = margin,
            )
        }

        val top3 = rankings.take(3)
        return MonitorDecision(
            stateKey = "known:ambiguous:" + top3.joinToString("|") { "${it.name}:${it.department}:${String.format(Locale.US, "%.3f", it.score)}" },
            type = MonitorDecisionType.AMBIGUOUS,
            rankings = top3,
        )
    }

    private fun applyMonitorDecision(decision: MonitorDecision) {
        latestRankings = decision.rankings
        if (decision.stateKey == lastMonitorStateKey) return
        when (decision.type) {
            MonitorDecisionType.IDLE_OUTSIDE -> applyMonitorIdleState()
            MonitorDecisionType.UNREGISTERED -> applyUnregisteredState()
            MonitorDecisionType.CLEAR -> {
                val top = decision.topMatch ?: return
                applyRegisteredClearState(top, decision.topScore, decision.margin)
            }
            MonitorDecisionType.AMBIGUOUS -> applyRegisteredAmbiguousState(decision.rankings)
        }
        lastMonitorStateKey = decision.stateKey
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = try {
                cameraProviderFuture.get()
            } catch (t: Throwable) {
                showError("카메라 준비 실패: ${t.javaClass.simpleName}")
                return@addListener
            }

            cameraProvider = provider
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                if (frameInFlight.getAndSet(true)) {
                    image.close()
                    return@setAnalyzer
                }
                analyzeFrame(image)
            }

            val selector = pickCameraSelector(provider)
            if (selector == null) {
                frameInFlight.set(false)
                showError("카메라를 찾을 수 없습니다. 설정에서 CAMERA_INDEX를 바꿔보세요.")
                return@addListener
            }

            try {
                provider.unbindAll()
                val camera = provider.bindToLifecycle(this, selector, preview, analysis)
                runOnUiThread {
                    applyPreviewMirror(camera.cameraInfo.lensFacing)
                    val modelLabel = if (faceEngine.isBuffaloReady()) "buffalo_l ONNX" else "geometry fallback"
                    monitorDetailText.text = "모델: $modelLabel"
                }
            } catch (t: Throwable) {
                showError("카메라 바인딩 실패: ${t.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun restartCamera() {
        cameraProvider?.unbindAll()
        frameInFlight.set(false)
        startCamera()
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        hasCameraFrame = true
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            frameInFlight.set(false)
            imageProxy.close()
            return
        }

        val bitmap = ImageProxyBitmapConverter.toUprightBitmap(imageProxy)
        if (bitmap == null) {
            frameInFlight.set(false)
            imageProxy.close()
            return
        }

        val inputImage = InputImage.fromBitmap(bitmap, 0)
        lastImageWidth = inputImage.width
        lastImageHeight = inputImage.height

        faceEngine.detectLargestFace(inputImage)
            .addOnSuccessListener(cameraExecutor) { faces ->
                try {
                    val page = currentPage()
                    val face = when (page) {
                        PAGE_MONITOR, PAGE_REGISTER_CAPTURE -> pickLargestFaceInsideEllipse(faces, inputImage.width, inputImage.height)
                        else -> pickLargestFace(faces)
                    }
                    if (!isDbServerConnected) {
                        if (page == PAGE_MONITOR) {
                            val stateKey = "server:disconnected"
                            runOnUiThread {
                                if (stateKey != lastMonitorStateKey) {
                                    applyServerDisconnectedState()
                                    lastMonitorStateKey = stateKey
                                }
                            }
                        }
                        return@addOnSuccessListener
                    }
                    if (face == null) {
                        if (page == PAGE_MONITOR) {
                            val stateKey = "idle"
                            runOnUiThread {
                                if (stateKey != lastMonitorStateKey) {
                                    applyMonitorIdleState()
                                    lastMonitorStateKey = stateKey
                                }
                            }
                        } else if (page == PAGE_REGISTER_CAPTURE) {
                            runOnUiThread {
                                val t = nextTargetDirection()
                                if (t != null) {
                                    registerStatusText.text = "얼굴을 찾지 못했습니다. 화면 중앙으로 와주세요.\n${DIRECTION_PROMPTS[t]}\n진행: ${summarizeDirectionProgress()}"
                                }
                            }
                        }
                        return@addOnSuccessListener
                    }
                    lastRecognitionAtMs = System.currentTimeMillis()

                    val embeddingBitmap = buildEllipseMaskedBitmap(bitmap)
                    val embedding = try {
                        faceEngine.makeEmbedding(face, embeddingBitmap)
                    } finally {
                        embeddingBitmap.recycle()
                    }
                    when (page) {
                        PAGE_MONITOR -> {
                            analyzeFrameTick += 1
                            if (analyzeFrameTick % settings.analyzeEveryNFrames.coerceAtLeast(1) == 0L) {
                                val decision = computeMonitorDecision(face, embedding)
                                runOnUiThread { applyMonitorDecision(decision) }
                            }
                        }
                        PAGE_REGISTER_CAPTURE -> {
                            runOnUiThread { handleRegisterCapture(face, embedding) }
                        }
                    }
                } finally {
                    bitmap.recycle()
                }
            }
            .addOnFailureListener(cameraExecutor) { err ->
                bitmap.recycle()
                runOnUiThread { showError("얼굴 분석 실패: ${err.javaClass.simpleName}") }
            }
            .addOnCompleteListener(cameraExecutor) {
                frameInFlight.set(false)
                imageProxy.close()
            }
    }

    private fun pickLargestFace(faces: List<Face>): Face? {
        if (faces.isEmpty()) return null
        return faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
    }

    private fun pickLargestFaceInsideEllipse(faces: List<Face>, width: Int, height: Int): Face? {
        if (faces.isEmpty()) return null
        val ellipse = getRegisterEllipse(width, height)
        val cx = ellipse.first.first
        val cy = ellipse.first.second
        val ax = ellipse.second.first
        val ay = ellipse.second.second
        return faces
            .asSequence()
            .filter { face ->
                val centerX = face.boundingBox.centerX()
                val centerY = face.boundingBox.centerY()
                isInsideEllipse(centerX, centerY, cx, cy, ax, ay)
            }
            .maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
    }

    private fun buildEllipseMaskedBitmap(source: Bitmap): Bitmap {
        val base = source.copy(Bitmap.Config.ARGB_8888, true)
            ?: Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888).also {
                Canvas(it).drawBitmap(source, 0f, 0f, null)
            }
        val ellipse = getRegisterEllipse(base.width, base.height)
        val cx = ellipse.first.first
        val cy = ellipse.first.second
        val ax = ellipse.second.first
        val ay = ellipse.second.second
        val oval = RectF(
            (cx - ax).toFloat(),
            (cy - ay).toFloat(),
            (cx + ax).toFloat(),
            (cy + ay).toFloat(),
        )
        val outside = Path().apply {
            fillType = Path.FillType.EVEN_ODD
            addRect(0f, 0f, base.width.toFloat(), base.height.toFloat(), Path.Direction.CW)
            addOval(oval, Path.Direction.CCW)
        }
        Canvas(base).drawPath(
            outside,
            android.graphics.Paint().apply {
                color = Color.BLACK
                style = android.graphics.Paint.Style.FILL
                isAntiAlias = true
            },
        )
        return base
    }

    private fun pickCameraSelector(provider: ProcessCameraProvider): CameraSelector? {
        val preferred = if (settings.cameraIndex == 1) {
            listOf(CameraSelector.DEFAULT_FRONT_CAMERA, CameraSelector.DEFAULT_BACK_CAMERA)
        } else {
            listOf(CameraSelector.DEFAULT_BACK_CAMERA, CameraSelector.DEFAULT_FRONT_CAMERA)
        }
        for (selector in preferred) {
            try {
                if (provider.hasCamera(selector)) return selector
            } catch (_: Throwable) {
            }
        }
        return null
    }

    private fun getRegisterEllipse(width: Int, height: Int): Pair<Pair<Int, Int>, Pair<Int, Int>> {
        val w = width.coerceAtLeast(1)
        val h = height.coerceAtLeast(1)
        val cx = w / 2
        val cy = (h * (0.5 + settings.registerEllipseCenterYOffsetRatio)).toInt()
        val ax = (w * settings.registerEllipseAxisXRatio).toInt().coerceAtLeast(40)
        val ay = (h * settings.registerEllipseAxisYRatio).toInt().coerceAtLeast(60)
        return (cx to cy) to (ax to ay)
    }

    private fun isInsideEllipse(x: Int, y: Int, cx: Int, cy: Int, ax: Int, ay: Int): Boolean {
        if (ax <= 0 || ay <= 0) return false
        val dx = (x - cx) / ax.toDouble()
        val dy = (y - cy) / ay.toDouble()
        return dx * dx + dy * dy <= 1.0
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var sum = 0f
        for (i in 0 until n) sum += a[i] * b[i]
        return sum
    }

    private fun openSettingsDialog() {
        SettingsDialog.show(
            context = this,
            current = settings,
            onPickFolderRequested = { onPicked ->
                pendingFolderPickerCallback = onPicked
                val initial = resolveInitialFaceDataTreeUri()
                folderPickerLauncher.launch(initial)
            },
            onPickTemplateRequested = { onPicked ->
                pendingTemplatePickerCallback = onPicked
                templatePickerLauncher.launch(
                    arrayOf(
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/*",
                    ),
                )
            },
            onRenderPrintPreview = { previewSettings ->
                buildNemonicPrintBitmap(
                    name = "홍길동",
                    department = "올스타반",
                    activeSettings = previewSettings,
                    applyLandscapeRotation = false,
                )
            },
            onSave = { updated ->
                val requireReauth = updated.dbServerBaseUrl != settings.dbServerBaseUrl || updated.dbClientId != settings.dbClientId
                settings = if (requireReauth) updated.copy(dbAuthToken = "") else updated
                settingsStore.save(settings)
                faceDb = createFaceDatabase(settings)
                dbSyncClient.start(settings)
                applyEllipseSettings()
                setRegisterCount()
                resetRegistrationState()
                goInitialState()
                restartCamera()
            },
        )
    }

    private fun ensureSyncIdentity() {
        var next = settings
        var changed = false
        if (next.dbClientId.isBlank()) {
            next = next.copy(dbClientId = "client-" + UUID.randomUUID().toString().take(8))
            changed = true
        }
        if (next.dbDeviceName.isBlank()) {
            next = next.copy(dbDeviceName = "${Build.MANUFACTURER}-${Build.MODEL}")
            changed = true
        }
        if (changed) {
            settings = next
            settingsStore.save(next)
        }
    }

    private fun resolveInitialFaceDataTreeUri(): Uri {
        val selected = settings.faceDataDir.trim()
        if (selected.startsWith("content://")) {
            return Uri.parse(selected)
        }
        return Uri.parse("content://com.android.externalstorage.documents/tree/primary%3ADocuments")
    }

    private fun updateEllipseOverlay() {
        val visible = currentPage() == PAGE_REGISTER_CAPTURE || currentPage() == PAGE_MONITOR
        ellipseOverlay.visibility = if (visible) View.VISIBLE else View.GONE
    }

    private fun applyEllipseSettings() {
        ellipseOverlay.axisXRatio = settings.registerEllipseAxisXRatio
        ellipseOverlay.axisYRatio = settings.registerEllipseAxisYRatio
        ellipseOverlay.centerYOffsetRatio = settings.registerEllipseCenterYOffsetRatio
        ellipseOverlay.outsideDimAlpha = settings.registerEllipseOutsideDimAlpha
        updateEllipseOverlay()
    }

    private fun applyPreviewMirror(lensFacing: Int?) {
        isPreviewMirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
        previewView.scaleX = if (isPreviewMirrored) -1f else 1f
    }

    private fun createDbSyncClient(): FaceDbSyncClient {
        return FaceDbSyncClient(
            object : FaceDbSyncClient.Callbacks {
                override fun onStatus(message: String, busy: Boolean) {
                    appendSyncLog("상태: $message")
                    val shouldDispatch = synchronized(syncStatusDedupLock) {
                        if (message == lastSyncStatusMessage && busy == lastSyncStatusBusy) {
                            false
                        } else {
                            lastSyncStatusMessage = message
                            lastSyncStatusBusy = busy
                            true
                        }
                    }
                    if (!shouldDispatch) return
                    runOnUiThread {
                        updateSyncIndicators(message, busy)
                    }
                }

                override fun onDiagnosticLog(message: String) {
                    appendSyncLog("진단: $message")
                }

                override fun onClientIdIssued(clientId: String) {
                    if (clientId.isBlank() || settings.dbClientId == clientId) return
                    settings = settings.copy(dbClientId = clientId)
                    settingsStore.save(settings)
                }

                override fun exportLocalStructuredDb(): Pair<ByteArray, ByteArray>? {
                    return faceDb.exportStructuredBytes()
                }

                override fun applyRemoteStructuredDb(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
                    val applied = faceDb.replaceWithStructuredBytes(metaDbBytes, embeddingsBinBytes)
                    if (applied) {
                        lastMonitorStateKey = null
                        runOnUiThread {
                            monitorDetailText.text = "서버 구조화 DB를 반영했습니다."
                        }
                    }
                    return applied
                }

                override fun localFaceCount(): Int {
                    return faceDb.personCount()
                }

                override fun localEmbeddingCount(): Int {
                    return faceDb.totalEmbeddingCount()
                }

                override fun onUploadProgress(uploaded: Int, total: Int) {
                    val text = if (total > 0) "$uploaded/$total" else ""
                    runOnUiThread {
                        uploadProgressText = text
                        applySyncIndicatorColors()
                    }
                }

                override fun isRecognitionIdle(idleSeconds: Int): Boolean {
                    return this@MainActivity.isRecognitionIdle(idleSeconds)
                }

                override fun onAuthTokenIssued(token: String) {
                    if (settings.dbAuthToken == token) return
                    settings = settings.copy(dbAuthToken = token)
                    settingsStore.save(settings)
                }

                override fun onAuthTokenCleared() {
                    if (settings.dbAuthToken.isBlank()) return
                    settings = settings.copy(dbAuthToken = "")
                    settingsStore.save(settings)
                }
            },
        )
    }

    private fun showError(message: String) {
        AlertDialog.Builder(this)
            .setTitle("오류")
            .setMessage(message)
            .setPositiveButton("확인", null)
            .show()
    }

    private fun appendSyncLog(message: String) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.KOREA).format(Date())
        val line = "[$timestamp] $message"
        synchronized(syncLogLock) {
            syncLogBuffer.addLast(line)
            while (syncLogBuffer.size > 400) {
                syncLogBuffer.removeFirst()
            }
        }
        if (!::syncLogFile.isInitialized || logExecutor.isShutdown) return
        runCatching {
            logExecutor.execute {
                runCatching {
                    syncLogFile.parentFile?.mkdirs()
                    syncLogFile.appendText("$line\n")
                }
            }
        }
    }

    private fun showSyncLogDialog() {
        val raw = if (::syncLogFile.isInitialized && syncLogFile.exists()) {
            runCatching { syncLogFile.readText() }.getOrDefault("")
        } else {
            ""
        }
        val text = if (raw.isNotBlank()) {
            raw
        } else {
            synchronized(syncLogLock) {
                if (syncLogBuffer.isEmpty()) "저장된 로그가 없습니다." else syncLogBuffer.joinToString("\n")
            }
        }
        val scroll = ScrollView(this)
        val tv = TextView(this).apply {
            setTextIsSelectable(true)
            textSize = 13f
            setPadding(24, 24, 24, 24)
            this.text = text
        }
        scroll.addView(tv)
        AlertDialog.Builder(this)
            .setTitle("DB 서버 연결 오류 로그")
            .setView(scroll)
            .setNeutralButton("로그 지우기") { _, _ -> clearSyncLogs() }
            .setPositiveButton("닫기", null)
            .show()
    }

    private fun clearSyncLogs() {
        synchronized(syncLogLock) {
            syncLogBuffer.clear()
        }
        if (!::syncLogFile.isInitialized) return
        logExecutor.execute {
            runCatching {
                syncLogFile.parentFile?.mkdirs()
                syncLogFile.writeText("")
            }
        }
    }

    private fun resolveFaceDbFile(settings: AppSettings): File {
        val raw = settings.faceDataDir.trim()
        val dir = if (raw.isEmpty()) {
            File(filesDir, "face_data")
        } else {
            val f = File(raw)
            if (f.isAbsolute) f else File(filesDir, raw)
        }
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return File(dir, "face_db.csv")
    }

    private fun createFaceDatabase(settings: AppSettings): FaceDatabase {
        val raw = settings.faceDataDir.trim()
        if (raw.startsWith("content://")) {
            val uri = runCatching { Uri.parse(raw) }.getOrNull()
            if (uri != null) {
                return FaceDatabase(this, uri)
            }
        }
        return FaceDatabase(resolveFaceDbFile(settings))
    }

    private fun updateSyncIndicators(message: String, busy: Boolean) {
        if (message.isBlank()) return
        val text = message.lowercase(Locale.KOREA)
        val retryMatch = Regex("""(\d+)\s*/\s*(\d+)""").find(message)
        var connectionChanged = false

        when {
            text.contains("db 서버 연결됨") -> {
                connectionStatus = SyncLightStatus.SUCCESS
                connectionRetryLabel = null
                if (!isDbServerConnected) {
                    isDbServerConnected = true
                    connectionChanged = true
                }
            }
            text.contains("db 서버 연결 재시도 중") -> {
                connectionStatus = SyncLightStatus.IN_PROGRESS
                val stage =
                    when {
                        text.contains("헬스체크") -> "헬스체크"
                        text.contains("승인요청") -> "승인요청"
                        text.contains("ws") -> "WS"
                        else -> "재시도"
                    }
                connectionRetryLabel =
                    retryMatch?.let { "$stage ${it.groupValues[1]}/${it.groupValues[2]}" } ?: stage
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
            text.contains("연결 끊김") ||
                text.contains("연결 종료") ||
                text.contains("주소가 비어") ||
                text.contains("타임아웃") ||
                text.contains("인증 오류로 연결 재시도") ||
                text.contains("db 서버 연결 실패") ||
                text.contains("서버 승인 대기 타임아웃") ||
                text.contains("인증 요청 실패") -> {
                connectionStatus = SyncLightStatus.ERROR
                connectionRetryLabel = null
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
            text.contains("db 서버 헬스체크 중") -> {
                connectionStatus = SyncLightStatus.IN_PROGRESS
                connectionRetryLabel = "헬스체크"
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
            text.contains("db 서버 승인요청 중") -> {
                connectionStatus = SyncLightStatus.IN_PROGRESS
                connectionRetryLabel = "승인요청"
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
            text.contains("승인 요청") ||
                text.contains("승인 대기중") ||
                text.contains("승인대기중") ||
                text.contains("재승인 요청") ||
                text.contains("db 서버 연결 시도중") ||
                text.contains("db 서버 ws 연결 시도중") -> {
                connectionStatus = SyncLightStatus.IN_PROGRESS
                connectionRetryLabel =
                    when {
                        text.contains("승인대기중") || text.contains("승인 대기중") -> "승인대기"
                        text.contains("ws 연결 시도중") -> "WS연결"
                        else -> null
                    }
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
            text.contains("저장된 인증으로 연결됨") || text.contains("서버 승인 완료") -> {
                connectionStatus = SyncLightStatus.IN_PROGRESS
                connectionRetryLabel = null
            }
            text.contains("db 서버 동기화 비활성") -> {
                connectionStatus = SyncLightStatus.ERROR
                connectionRetryLabel = null
                if (isDbServerConnected) {
                    isDbServerConnected = false
                    connectionChanged = true
                }
            }
        }

        when {
            text.contains("db 동기화 완료") -> dbLoadStatus = SyncLightStatus.SUCCESS
            text.contains("db 동기화 실패") -> dbLoadStatus = SyncLightStatus.ERROR
            text.contains("db 동기화 대기중") || text.contains("db 동기화 중") -> dbLoadStatus = SyncLightStatus.IN_PROGRESS
        }

        when {
            text.contains("db 서버 업데이트 완료") -> {
                dbPushStatus = SyncLightStatus.SUCCESS
                uploadProgressText = ""
            }
            text.contains("db 서버 업데이트 실패") -> {
                dbPushStatus = SyncLightStatus.ERROR
                uploadProgressText = ""
            }
            text.contains("db 서버 업데이트 중") || text.contains("db 업로드 대기중") -> {
                dbPushStatus = SyncLightStatus.IN_PROGRESS
            }
        }

        if (busy) {
            if (text.contains("db 동기화")) dbLoadStatus = SyncLightStatus.IN_PROGRESS
            if (text.contains("db 서버 업데이트") || text.contains("db 업로드 대기중")) dbPushStatus = SyncLightStatus.IN_PROGRESS
        }

        if (connectionChanged) {
            lastMonitorStateKey = null
            if (isDbServerConnected) {
                if (currentPage() == PAGE_MONITOR) {
                    applyMonitorIdleState()
                }
            } else {
                if (currentPage() == PAGE_MONITOR) {
                    applyServerDisconnectedState()
                }
            }
        }

        applySyncIndicatorColors()
    }

    private fun applySyncIndicatorColors() {
        connectionStatusIndicator.text = connectionRetryLabel ?: statusTextForConnection(connectionStatus)
        dbLoadStatusIndicator.text = statusTextForDbStatus(dbLoadStatus)
        dbPushStatusIndicator.text = when {
            dbPushStatus == SyncLightStatus.IN_PROGRESS && uploadProgressText.isNotBlank() -> uploadProgressText
            else -> statusTextForDbStatus(dbPushStatus)
        }
        connectionStatusIndicator.setTextColor(colorForSyncStatus(connectionStatus))
        dbLoadStatusIndicator.setTextColor(colorForSyncStatus(dbLoadStatus))
        dbPushStatusIndicator.setTextColor(colorForSyncStatus(dbPushStatus))
    }

    private fun statusTextForConnection(status: SyncLightStatus): String {
        return when (status) {
            SyncLightStatus.IDLE -> "미연결"
            SyncLightStatus.IN_PROGRESS -> "연결중"
            SyncLightStatus.SUCCESS -> "연결됨"
            SyncLightStatus.ERROR -> "오류"
        }
    }

    private fun statusTextForDbStatus(status: SyncLightStatus): String {
        return when (status) {
            SyncLightStatus.IDLE -> "대기"
            SyncLightStatus.IN_PROGRESS -> "진행중"
            SyncLightStatus.SUCCESS -> "완료"
            SyncLightStatus.ERROR -> "실패"
        }
    }

    private fun colorForSyncStatus(status: SyncLightStatus): Int {
        return when (status) {
            SyncLightStatus.IDLE -> Color.parseColor("#9AA0A6")
            SyncLightStatus.IN_PROGRESS -> Color.parseColor("#F9AB00")
            SyncLightStatus.SUCCESS -> Color.parseColor("#1E8E3E")
            SyncLightStatus.ERROR -> Color.parseColor("#D93025")
        }
    }

    override fun onDestroy() {
        cameraProvider?.unbindAll()
        cancelOutputAutoReturnCountdown(clearTextState = true)
        dbSyncClient.stop()
        faceEngine.close()
        cameraExecutor.shutdown()
        dbUpdateExecutor.shutdown()
        logExecutor.shutdown()
        super.onDestroy()
    }
}
