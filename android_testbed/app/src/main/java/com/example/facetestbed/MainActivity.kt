package com.example.facetestbed

import android.Manifest
import android.bluetooth.BluetoothManager
import android.content.Intent
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Path
import android.graphics.RectF
import android.os.Build
import android.os.Environment
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.print.PrintAttributes
import android.print.PrintManager
import android.provider.Settings
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextPaint
import android.text.style.ForegroundColorSpan
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.ViewFlipper
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
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
        private val DIRECTION_PROGRESS_LABELS = mapOf(
            "left" to "왼쪽 얼굴",
            "right" to "오른쪽 얼굴",
            "up" to "위쪽 얼굴",
            "down" to "아래쪽 얼굴",
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
    private lateinit var dbPushProgressBar: ProgressBar
    private lateinit var printStatusIndicator: TextView
    private lateinit var printStatusDetailText: TextView
    private lateinit var dbReconnectButton: Button
    private lateinit var dbDownloadButton: Button
    private lateinit var dbUploadButton: Button
    private lateinit var backButton: Button
    private lateinit var settingsButton: Button
    private lateinit var operationSettingsButton: Button
    private var errorLogButton: Button? = null
    private lateinit var quitButton: Button
    private lateinit var developerRoot: View
    private lateinit var operationRoot: View
    private lateinit var cameraPreviewPanel: View
    private lateinit var operationCameraHost: FrameLayout
    private lateinit var operationActionPanel: FrameLayout
    private lateinit var developerPageHost: FrameLayout
    private lateinit var operationPageHost: FrameLayout
    private lateinit var operationStatusText: TextView
    private lateinit var operationDetailText: TextView
    private lateinit var operationPrintStatusText: TextView
    private lateinit var operationReturnCountdownText: TextView
    private lateinit var operationFloatingPrintButton: Button
    private lateinit var operationHomeButton: Button
    private lateinit var operationMotionHint: TextView
    private lateinit var previewView: PreviewView
    private lateinit var ellipseOverlay: EllipseMaskOverlayView
    private lateinit var pageFlipper: ViewFlipper

    private lateinit var monitorStatusText: TextView
    private lateinit var monitorDetailText: TextView
    private lateinit var registerStartButton: Button
    private lateinit var labelOnlyPrintButton: Button
    private lateinit var candidateContainer: LinearLayout
    private lateinit var outputNameButton: Button

    private lateinit var registerStatusText: TextView
    private lateinit var registerProgressBar: ProgressBar
    private lateinit var registerCountText: TextView

    private lateinit var registerCompleteText: TextView
    private lateinit var nameInput: EditText
    private lateinit var departmentInput: EditText
    private lateinit var gradeInput: EditText
    private lateinit var saveNameButton: Button

    private lateinit var outputLabel: TextView
    private lateinit var outputDepartmentLabel: TextView
    private lateinit var outputAttribute3Label: TextView
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
    private var selectedGrade: String = ""
    private var latestRankings: List<FaceMatch> = emptyList()
    private var lastMonitorStateKey: String? = null
    private var selectedCandidateKey: String? = null
    private var activeCandidateGroupKey: String? = null
    private var pendingCandidateGroupKey: String? = null
    private var pendingCandidateGroupFrames: Int = 0
    private var selectedCandidateMissingFrames: Int = 0
    private var labelOnlyPrintMode: Boolean = false
    @Volatile
    private var currentPageIndex: Int = PAGE_MONITOR

    private var adaptiveLastFrame = -1_000_000_000L
    private var adaptiveUpdateCount = 0
    private var pendingFolderPickerCallback: ((String?) -> Unit)? = null
    private var pendingTemplatePickerCallback: ((String?) -> Unit)? = null
    private var pendingBluetoothAction: (() -> Unit)? = null
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
    @Volatile
    private var uploadProgressCurrent: Int = 0
    @Volatile
    private var uploadProgressTotal: Int = 0
    private val uiHandler = Handler(Looper.getMainLooper())
    private var outputAutoReturnRunnable: Runnable? = null
    private var outputCountdownRemainingSeconds: Int = 0
    private var outputPrintStatusMessage: String = ""
    private var homePrintStatusMessage: String = "프린터 연결 및 출력 단계가 여기에 표시됩니다."
    private var homePrintStatusState: SyncLightStatus = SyncLightStatus.IDLE
    private var developerCameraParent: ViewGroup? = null
    private var developerCameraIndex: Int = -1
    private var developerCameraLayoutParams: ViewGroup.LayoutParams? = null
    private var developerPageLayoutParams: ViewGroup.LayoutParams? = null
    private var cameraPreviewOriginalPadding: IntArray? = null
    private var cameraPreviewOriginalBackground: android.graphics.drawable.Drawable? = null
    private var floatingPrintAnimationRunning: Boolean = false
    private var motionHintAnimationRunning: Boolean = false
    private var motionHintDirection: String? = null
    private var registerStallCount: Int = 0
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

    private val bluetoothPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val granted = grants.entries.all { it.value }
            if (!granted) {
                pendingBluetoothAction = null
                setOutputPrintStatus(settings.printStatusErrorMessage)
                setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
                appendPrintLog("Bluetooth 권한이 거부되었습니다.", isError = true)
                return@registerForActivityResult
            }
            val action = pendingBluetoothAction
            pendingBluetoothAction = null
            action?.invoke()
        }

    private val allFilesAccessSettingsLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            if (hasFaceDbStorageAccess()) {
                ensureFaceDbDirectoryStructure()
                configureSyncLogFile()
                appendSyncLog("faceDB 저장소 권한 확인 완료")
            } else {
                showFaceDbStoragePermissionDialog()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()

        settingsStore = AppSettingsStore(this)
        settings = settingsStore.load()
        syncLogFile = File(filesDir, "logs/db_sync_client.log").also { it.parentFile?.mkdirs() }
        requestFaceDbStorageAccessIfNeeded()
        ensureFaceDbDirectoryStructure()
        configureSyncLogFile()
        appendSyncLog("앱 시작")
        ensureSyncIdentity()
        faceDb = createFaceDatabase(settings)
        dbSyncClient = createDbSyncClient()
        dbSyncClient.start(settings)
        faceEngine = FaceRecognitionEngine(this)
        applyEllipseSettings()

        registerStartButton.setOnClickListener { startAutoRegistration() }
        labelOnlyPrintButton.setOnClickListener { startLabelOnlyPrint() }
        outputNameButton.setOnClickListener { outputSelectedName() }
        saveNameButton.setOnClickListener { saveRegistration() }
        backButton.setOnClickListener { goInitialState() }
        settingsButton.setOnClickListener { showSettingsMenu() }
        operationSettingsButton.setOnClickListener { showSettingsMenu() }
        errorLogButton?.setOnClickListener { showSyncLogDialog() }
        dbReconnectButton.setOnClickListener { requestDbReconnect() }
        dbDownloadButton.setOnClickListener { requestDbDownload() }
        dbUploadButton.setOnClickListener { requestDbUpload() }
        quitButton.setOnClickListener { finish() }
        operationFloatingPrintButton.setOnClickListener { outputSelectedName() }
        operationHomeButton.setOnClickListener { goInitialState() }

        applyScreenMode()
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
        developerRoot = findViewById(R.id.developerRoot)
        operationRoot = findViewById(R.id.operationRoot)
        cameraPreviewPanel = findViewById(R.id.cameraPreviewPanel)
        operationCameraHost = findViewById(R.id.operationCameraHost)
        operationActionPanel = findViewById(R.id.operationActionPanel)
        developerPageHost = findViewById(R.id.developerPageHost)
        operationPageHost = findViewById(R.id.operationPageHost)
        operationStatusText = findViewById(R.id.operationStatusText)
        operationDetailText = findViewById(R.id.operationDetailText)
        operationPrintStatusText = findViewById(R.id.operationPrintStatusText)
        operationReturnCountdownText = findViewById(R.id.operationReturnCountdownText)
        operationFloatingPrintButton = findViewById(R.id.operationFloatingPrintButton)
        operationHomeButton = findViewById(R.id.operationHomeButton)
        operationMotionHint = findViewById(R.id.operationMotionHint)
        connectionStatusIndicator = findViewById(R.id.connectionStatusIndicator)
        dbLoadStatusIndicator = findViewById(R.id.dbLoadStatusIndicator)
        dbPushStatusIndicator = findViewById(R.id.dbPushStatusIndicator)
        dbPushProgressBar = findViewById(R.id.dbPushProgressBar)
        printStatusIndicator = findViewById(R.id.printStatusIndicator)
        printStatusDetailText = findViewById(R.id.printStatusDetailText)
        dbReconnectButton = findViewById(R.id.dbReconnectButton)
        dbDownloadButton = findViewById(R.id.dbDownloadButton)
        dbUploadButton = findViewById(R.id.dbUploadButton)
        backButton = findViewById(R.id.backButton)
        settingsButton = findViewById(R.id.settingsButton)
        operationSettingsButton = findViewById(R.id.operationSettingsButton)
        errorLogButton = findViewById(R.id.errorLogButton)
        quitButton = findViewById(R.id.quitButton)
        previewView = findViewById(R.id.previewView)
        ellipseOverlay = findViewById(R.id.ellipseOverlay)
        pageFlipper = findViewById(R.id.pageFlipper)

        monitorStatusText = findViewById(R.id.monitorStatusText)
        monitorDetailText = findViewById(R.id.monitorDetailText)
        registerStartButton = findViewById(R.id.registerStartButton)
        labelOnlyPrintButton = findViewById(R.id.labelOnlyPrintButton)
        candidateContainer = findViewById(R.id.candidateContainer)
        outputNameButton = findViewById(R.id.outputNameButton)

        registerStatusText = findViewById(R.id.registerStatusText)
        registerProgressBar = findViewById(R.id.registerProgressBar)
        registerCountText = findViewById(R.id.registerCountText)

        registerCompleteText = findViewById(R.id.registerCompleteText)
        nameInput = findViewById(R.id.nameInput)
        departmentInput = findViewById(R.id.departmentInput)
        gradeInput = findViewById(R.id.gradeInput)
        saveNameButton = findViewById(R.id.saveNameButton)

        outputLabel = findViewById(R.id.outputLabel)
        outputDepartmentLabel = findViewById(R.id.outputDepartmentLabel)
        outputAttribute3Label = findViewById(R.id.outputAttribute3Label)
        outputSubLabel = findViewById(R.id.outputSubLabel)
        captureOriginalScreenParents()
        applyConfiguredUiText()
        applyPrintStatusPanel()
    }

    private fun applyConfiguredUiText() {
        if (!::outputNameButton.isInitialized) return
        outputNameButton.text = settings.uiOutputButtonText.ifBlank { "이름 출력하기" }
    }

    private fun formatUiMessage(template: String, values: Map<String, String>): String {
        var out = template
        values.forEach { (key, value) ->
            out = out.replace("{$key}", value)
        }
        return out
    }

    private fun captureOriginalScreenParents() {
        if (developerCameraParent != null) return
        developerCameraParent = cameraPreviewPanel.parent as? ViewGroup
        developerCameraIndex = developerCameraParent?.indexOfChild(cameraPreviewPanel) ?: -1
        developerCameraLayoutParams = cameraPreviewPanel.layoutParams
        developerPageLayoutParams = pageFlipper.layoutParams
        cameraPreviewOriginalPadding = intArrayOf(
            cameraPreviewPanel.paddingLeft,
            cameraPreviewPanel.paddingTop,
            cameraPreviewPanel.paddingRight,
            cameraPreviewPanel.paddingBottom,
        )
        cameraPreviewOriginalBackground = cameraPreviewPanel.background
        operationStatusText.visibility = View.GONE
        operationDetailText.visibility = View.GONE
        operationPrintStatusText.visibility = View.GONE
    }

    private fun applyScreenMode() {
        if (!::developerRoot.isInitialized) return
        val showDeveloper = settings.developerViewEnabled
        developerRoot.visibility = if (showDeveloper) View.VISIBLE else View.GONE
        operationRoot.visibility = if (showDeveloper) View.GONE else View.VISIBLE
        if (showDeveloper) {
            attachCameraToDeveloper()
            attachPagesToDeveloper()
        } else {
            attachCameraToOperation()
            attachPagesToOperation()
        }
        applyPrintStatusPanel()
        updateEllipseOverlay()
        updateOperationOverlay()
    }

    private fun attachCameraToDeveloper() {
        val targetParent = developerCameraParent ?: return
        if (cameraPreviewPanel.parent === targetParent) return
        (cameraPreviewPanel.parent as? ViewGroup)?.removeView(cameraPreviewPanel)
        cameraPreviewOriginalBackground?.let { cameraPreviewPanel.background = it }
        cameraPreviewOriginalPadding?.let {
            cameraPreviewPanel.setPadding(it[0], it[1], it[2], it[3])
        }
        val params = developerCameraLayoutParams ?: cameraPreviewPanel.layoutParams
        val index = developerCameraIndex.takeIf { it >= 0 } ?: targetParent.childCount
        targetParent.addView(cameraPreviewPanel, index.coerceAtMost(targetParent.childCount), params)
    }

    private fun attachCameraToOperation() {
        if (cameraPreviewPanel.parent === operationCameraHost) return
        (cameraPreviewPanel.parent as? ViewGroup)?.removeView(cameraPreviewPanel)
        cameraPreviewPanel.background = null
        cameraPreviewPanel.setPadding(0, 0, 0, 0)
        operationCameraHost.addView(
            cameraPreviewPanel,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
    }

    private fun attachPagesToDeveloper() {
        if (pageFlipper.parent === developerPageHost) return
        (pageFlipper.parent as? ViewGroup)?.removeView(pageFlipper)
        developerPageHost.addView(
            pageFlipper,
            developerPageLayoutParams ?: FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
    }

    private fun attachPagesToOperation() {
        if (pageFlipper.parent === operationPageHost) return
        (pageFlipper.parent as? ViewGroup)?.removeView(pageFlipper)
        operationPageHost.addView(
            pageFlipper,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
    }

    private fun updateOperationOverlay() {
        if (!::operationRoot.isInitialized) return
        val operationMode = !settings.developerViewEnabled
        val monitorPage = currentPageIndex == PAGE_MONITOR
        val operationRegisterCapture = operationMode && currentPageIndex == PAGE_REGISTER_CAPTURE
        monitorStatusText.visibility = if (operationMode && monitorPage) View.GONE else View.VISIBLE
        monitorDetailText.visibility = if (operationMode && monitorPage) View.GONE else View.VISIBLE
        registerStatusText.visibility = if (operationRegisterCapture) View.GONE else View.VISIBLE
        registerProgressBar.visibility = if (operationRegisterCapture) View.GONE else View.VISIBLE

        if (!operationMode) {
            operationStatusText.visibility = View.GONE
            operationDetailText.visibility = View.GONE
            operationPrintStatusText.visibility = View.GONE
            operationReturnCountdownText.visibility = View.GONE
            operationActionPanel.visibility = View.GONE
            operationHomeButton.visibility = View.GONE
            hideMotionHint()
            hideFloatingPrintButton()
            return
        }
        operationHomeButton.visibility = View.VISIBLE

        val statusText = when (currentPageIndex) {
            PAGE_MONITOR -> monitorStatusText.text?.toString().orEmpty()
            PAGE_REGISTER_CAPTURE -> registerStatusText.text?.toString().orEmpty()
            PAGE_REGISTER_NAME -> if (labelOnlyPrintMode) "" else registerCompleteText.text?.toString().orEmpty()
            PAGE_OUTPUT -> outputReturnCountdownText()
            else -> ""
        }
        val detailText = when (currentPageIndex) {
            else -> ""
        }
        operationStatusText.text = statusText
        operationDetailText.text = detailText
        operationStatusText.visibility = if (statusText.isBlank()) View.GONE else View.VISIBLE
        operationDetailText.visibility = if (detailText.isBlank()) View.GONE else View.VISIBLE
        operationReturnCountdownText.text = ""
        operationReturnCountdownText.visibility = View.GONE

        val monitorNeedsAction =
            registerStartButton.visibility == View.VISIBLE ||
                labelOnlyPrintButton.visibility == View.VISIBLE ||
                candidateContainer.childCount > 0
        val showFloatingPrint =
            monitorPage &&
                outputNameButton.visibility == View.VISIBLE &&
                outputNameButton.isEnabled &&
                candidateContainer.childCount <= 1 &&
                registerStartButton.visibility != View.VISIBLE &&
                labelOnlyPrintButton.visibility != View.VISIBLE
        operationActionPanel.visibility = if (!monitorPage || (monitorNeedsAction && !showFloatingPrint)) View.VISIBLE else View.GONE
        if (showFloatingPrint) showFloatingPrintButton() else hideFloatingPrintButton()
        updateMotionHint()
    }

    private fun updateMotionHint() {
        if (!::operationMotionHint.isInitialized) return
        val direction = if (!settings.developerViewEnabled && currentPageIndex == PAGE_REGISTER_CAPTURE) nextTargetDirection() else null
        if (direction == null) {
            hideMotionHint()
            return
        }
        val arrow = when (direction) {
            "left" -> "←"
            "right" -> "→"
            "up" -> "↑"
            "down" -> "↓"
            else -> "•"
        }
        operationMotionHint.text = arrow
        operationMotionHint.visibility = View.VISIBLE
        if (!motionHintAnimationRunning || motionHintDirection != direction) {
            motionHintDirection = direction
            startMotionHintAnimation(direction)
        }
    }

    private fun startMotionHintAnimation(direction: String) {
        if (!::operationMotionHint.isInitialized || operationMotionHint.visibility != View.VISIBLE) return
        operationMotionHint.animate().cancel()
        operationMotionHint.translationX = 0f
        operationMotionHint.translationY = 0f
        motionHintAnimationRunning = true
        val distance = 90f
        val targetX = when (direction) {
            "left" -> -distance
            "right" -> distance
            else -> 0f
        }
        val targetY = when (direction) {
            "up" -> -distance
            "down" -> distance
            else -> 0f
        }
        operationMotionHint.animate()
            .translationX(targetX)
            .translationY(targetY)
            .alpha(0.45f)
            .setDuration(560L)
            .setInterpolator(AccelerateDecelerateInterpolator())
            .withEndAction {
                if (::operationMotionHint.isInitialized && operationMotionHint.visibility == View.VISIBLE && motionHintDirection == direction) {
                    operationMotionHint.animate()
                        .translationX(0f)
                        .translationY(0f)
                        .alpha(0.90f)
                        .setDuration(560L)
                        .setInterpolator(AccelerateDecelerateInterpolator())
                        .withEndAction {
                            if (::operationMotionHint.isInitialized && operationMotionHint.visibility == View.VISIBLE && motionHintDirection == direction) {
                                startMotionHintAnimation(direction)
                            }
                        }
                        .start()
                }
            }
            .start()
    }

    private fun hideMotionHint() {
        if (!::operationMotionHint.isInitialized) return
        motionHintAnimationRunning = false
        motionHintDirection = null
        operationMotionHint.animate().cancel()
        operationMotionHint.translationX = 0f
        operationMotionHint.translationY = 0f
        operationMotionHint.alpha = 0.90f
        operationMotionHint.visibility = View.GONE
    }

    private fun showFloatingPrintButton() {
        if (!::operationFloatingPrintButton.isInitialized) return
        operationFloatingPrintButton.text = settings.uiOutputButtonText.ifBlank { "이름표 출력하기" }
        operationFloatingPrintButton.visibility = View.VISIBLE
        operationFloatingPrintButton.isEnabled = true
        if (!floatingPrintAnimationRunning) {
            floatingPrintAnimationRunning = true
            operationFloatingPrintButton.animate()
                .scaleX(1.06f)
                .scaleY(1.06f)
                .setDuration(520L)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .withEndAction {
                    if (::operationFloatingPrintButton.isInitialized && operationFloatingPrintButton.visibility == View.VISIBLE) {
                        operationFloatingPrintButton.animate()
                            .scaleX(1.0f)
                            .scaleY(1.0f)
                            .setDuration(520L)
                            .setInterpolator(AccelerateDecelerateInterpolator())
                            .withEndAction {
                                if (::operationFloatingPrintButton.isInitialized && operationFloatingPrintButton.visibility == View.VISIBLE) {
                                    floatingPrintAnimationRunning = false
                                    showFloatingPrintButton()
                                }
                            }
                            .start()
                    }
                }
                .start()
        }
    }

    private fun hideFloatingPrintButton() {
        if (!::operationFloatingPrintButton.isInitialized) return
        floatingPrintAnimationRunning = false
        operationFloatingPrintButton.animate().cancel()
        operationFloatingPrintButton.scaleX = 1.0f
        operationFloatingPrintButton.scaleY = 1.0f
        operationFloatingPrintButton.visibility = View.GONE
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
                headerTitle.text = if (labelOnlyPrintMode) "이름표 출력" else "이름 저장"
                headerSubtitle.text = if (labelOnlyPrintMode) {
                    settings.uiLabelOnlyInputMessage
                } else {
                    settings.uiRegisterCompleteMessage
                }
                backButton.visibility = Button.VISIBLE
            }
            PAGE_OUTPUT -> {
                headerTitle.text = "이름 출력"
                headerSubtitle.text = "선택된 이름을 출력합니다."
                backButton.visibility = Button.VISIBLE
            }
        }
        updateOperationOverlay()
    }

    private fun registrationTotalTarget(): Int {
        return settings.registerSamplesPerDirection.coerceAtLeast(1) * REGISTER_DIRECTIONS.size
    }

    private fun resetRegistrationState() {
        REGISTER_DIRECTIONS.forEach { registerDirectionEmbeddings[it]?.clear() }
        registerFrameTick = 0
        registerStallCount = 0
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
        registerCountText.text = buildDirectionProgressText()
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
        val target = nextTargetDirection()
        return if (target == null) {
            "모든 방향 수집 완료"
        } else {
            val current = registerDirectionEmbeddings[target]?.size ?: 0
            val label = DIRECTION_PROGRESS_LABELS[target] ?: (DIRECTION_LABELS[target] ?: target)
            "$label $current/$need 수집 중"
        }
    }

    private fun buildDirectionProgressText(): SpannableStringBuilder {
        val need = settings.registerSamplesPerDirection.coerceAtLeast(1)
        val currentTarget = nextTargetDirection()
        val total = registrationTotalTarget()
        val current = totalRegisteredSamples().coerceAtMost(total)
        val builder = SpannableStringBuilder()
        appendColoredStep(builder, "전체 $current / $total", Color.rgb(17, 17, 17))
        REGISTER_DIRECTIONS.forEach { dir ->
            builder.append('\n')
            val count = (registerDirectionEmbeddings[dir]?.size ?: 0).coerceAtMost(need)
            val label = DIRECTION_PROGRESS_LABELS[dir] ?: dir
            val completed = count >= need
            val active = dir == currentTarget
            val (marker, color, text) = when {
                completed -> Triple("✓", Color.rgb(30, 132, 73), "$label $count/$need 완료")
                active -> Triple("●", Color.rgb(245, 166, 35), "$label $count/$need")
                else -> Triple("○", Color.rgb(142, 142, 147), "$label $count/$need 대기")
            }
            appendColoredStep(builder, "$marker $text", color)
        }
        return builder
    }

    private fun appendColoredStep(builder: SpannableStringBuilder, text: String, color: Int) {
        val span = SpannableString(text)
        span.setSpan(ForegroundColorSpan(color), 0, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        builder.append(span)
    }

    private fun allRegisteredEmbeddings(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        REGISTER_DIRECTIONS.forEach { out.addAll(registerDirectionEmbeddings[it] ?: emptyList()) }
        return out
    }

    private fun registrationPrompt(base: String): String {
        val extra = if (registerStallCount >= 5) {
            "\n등록이 잘 안되면 표정을 조금 바꾸거나 얼굴을 카메라에 더 가까이/멀리 움직여주세요."
        } else {
            ""
        }
        return base + extra
    }

    private fun goInitialState() {
        cancelOutputAutoReturnCountdown(clearTextState = true)
        resetRegistrationState()
        labelOnlyPrintMode = false
        selectedName = null
        selectedDepartment = ""
        selectedGrade = ""
        selectedCandidateKey = null
        activeCandidateGroupKey = null
        pendingCandidateGroupKey = null
        pendingCandidateGroupFrames = 0
        selectedCandidateMissingFrames = 0
        latestRankings = emptyList()
        lastMonitorStateKey = null
        adaptiveLastFrame = -1_000_000_000L
        adaptiveUpdateCount = 0
        nameInput.setText("")
        departmentInput.setText("")
        gradeInput.setText("")
        outputLabel.text = ""
        outputDepartmentLabel.text = ""
        outputAttribute3Label.text = ""
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

    private fun candidateKey(match: FaceMatch): String {
        return listOf(match.name.trim(), match.department.trim(), match.grade.trim()).joinToString("\u0001")
    }

    private fun candidateGroupKey(rankings: List<FaceMatch>): String {
        return rankings.map { candidateKey(it) }.sorted().joinToString("|")
    }

    private fun resetCandidateSelectionState() {
        selectedName = null
        selectedDepartment = ""
        selectedGrade = ""
        selectedCandidateKey = null
        selectedCandidateMissingFrames = 0
        outputNameButton.isEnabled = false
    }

    private fun resetCandidateGroupState() {
        activeCandidateGroupKey = null
        pendingCandidateGroupKey = null
        pendingCandidateGroupFrames = 0
        selectedCandidateMissingFrames = 0
    }

    private fun applyMonitorIdleState() {
        monitorStatusText.text = settings.uiWelcomeMessage
        monitorDetailText.text = ""
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        labelOnlyPrintButton.visibility = Button.GONE
        labelOnlyPrintButton.isEnabled = false
        outputNameButton.visibility = Button.GONE
        resetCandidateSelectionState()
        resetCandidateGroupState()
        clearCandidates()
        updateOperationOverlay()
    }

    private fun applyServerDisconnectedState() {
        monitorStatusText.text = settings.uiServerDisconnectedMessage
        monitorDetailText.text = settings.uiServerDisconnectedDetail
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        labelOnlyPrintButton.visibility = Button.GONE
        labelOnlyPrintButton.isEnabled = false
        outputNameButton.visibility = Button.GONE
        resetCandidateSelectionState()
        resetCandidateGroupState()
        clearCandidates()
        updateOperationOverlay()
    }

    private fun ensureDbSyncEnabledForManualAction(actionName: String): Boolean {
        if (settings.dbSyncEnabled) return true
        showError("설정에서 DB 서버 동기화를 활성화한 뒤 '$actionName'을 사용해 주세요.")
        return false
    }

    private fun requestDbReconnect() {
        if (!ensureDbSyncEnabledForManualAction("DB 서버 연결 복구")) return
        monitorDetailText.text = "수동 복구 요청: DB 서버 연결을 다시 시도합니다."
        updateOperationOverlay()
        dbSyncClient.start(settings)
    }

    private fun requestDbDownload() {
        if (!ensureDbSyncEnabledForManualAction("DB 다운로드")) return
        monitorDetailText.text = "수동 다운로드 요청: 서버 DB를 가져옵니다."
        updateOperationOverlay()
        dbSyncClient.pullLatest("수동 다운로드")
    }

    private fun requestDbUpload() {
        if (!ensureDbSyncEnabledForManualAction("DB 업로드")) return
        monitorDetailText.text = "수동 업로드 요청: 로컬 DB를 서버로 전송합니다."
        updateOperationOverlay()
        dbSyncClient.pushLatest("수동 업로드")
    }

    private fun applyUnregisteredState() {
        monitorStatusText.text = settings.uiUnregisteredMessage
        monitorDetailText.text = settings.uiUnregisteredDetail
        registerStartButton.visibility = Button.VISIBLE
        registerStartButton.text = "등록 시작하기"
        registerStartButton.isEnabled = true
        labelOnlyPrintButton.visibility = Button.VISIBLE
        labelOnlyPrintButton.isEnabled = true
        outputNameButton.visibility = Button.GONE
        resetCandidateSelectionState()
        resetCandidateGroupState()
        clearCandidates()
        updateOperationOverlay()
    }

    private fun applyRegisteredClearState(topMatch: FaceMatch, topScore: Float, margin: Float) {
        val topName = topMatch.name
        val topDepartment = topMatch.department
        val topGrade = topMatch.grade
        val metaParts = listOf(topDepartment, topGrade).filter { it.isNotBlank() }
        val userLabel = if (metaParts.isEmpty()) topName else "$topName (${metaParts.joinToString(", ")})"
        monitorStatusText.text = formatUiMessage(
            settings.uiRecognizedMessageTemplate,
            mapOf(
                "name" to topName,
                "department" to topDepartment,
                "grade" to topGrade,
                "label" to userLabel,
            ),
        )
        monitorDetailText.text = ""
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        labelOnlyPrintButton.visibility = Button.GONE
        labelOnlyPrintButton.isEnabled = false
        outputNameButton.visibility = Button.VISIBLE
        resetCandidateGroupState()
        clearCandidates()

        val btn = Button(this).apply {
            text = userLabel
            isEnabled = false
        }
        candidateContainer.addView(btn)

        selectedName = topName
        selectedDepartment = topDepartment
        selectedGrade = topMatch.grade
        selectedCandidateKey = candidateKey(topMatch)
        selectedCandidateMissingFrames = 0
        outputNameButton.isEnabled = true
        updateOperationOverlay()
    }

    private fun applyRegisteredAmbiguousState(rankings: List<FaceMatch>) {
        val previousSelectedKey = selectedCandidateKey
        val selectedStillPresent = previousSelectedKey != null && rankings.any { candidateKey(it) == previousSelectedKey }
        monitorStatusText.text = settings.uiAmbiguousMessage
        if (!selectedStillPresent) {
            monitorDetailText.text = settings.uiAmbiguousDetail
        }
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        labelOnlyPrintButton.visibility = Button.GONE
        labelOnlyPrintButton.isEnabled = false
        outputNameButton.visibility = Button.VISIBLE
        clearCandidates()
        if (!selectedStillPresent) {
            resetCandidateSelectionState()
        } else {
            selectedCandidateMissingFrames = 0
            outputNameButton.isEnabled = true
        }

        rankings.forEachIndexed { idx, match ->
            val btn = Button(this).apply {
                val label = if (match.department.isBlank()) match.name else "${match.name} (${match.department})"
                val selectedMark = if (candidateKey(match) == previousSelectedKey) "선택됨: " else ""
                text = String.format(Locale.US, "%s%d위 선택: %s (점수 %.3f)", selectedMark, idx + 1, label, match.score)
                setOnClickListener { chooseCandidateName(match) }
            }
            candidateContainer.addView(btn)
        }
        updateOperationOverlay()
    }

    private fun chooseCandidateName(match: FaceMatch) {
        selectedName = match.name
        selectedDepartment = match.department
        selectedGrade = match.grade
        selectedCandidateKey = candidateKey(match)
        selectedCandidateMissingFrames = 0
        val label = if (match.department.isBlank()) match.name else "${match.name} (${match.department})"
        monitorDetailText.text = formatUiMessage(settings.uiSelectedCandidateTemplate, mapOf("label" to label))
        outputNameButton.isEnabled = true
        updateOperationOverlay()
    }

    private fun metadataValue(column: String, metadata: Map<String, String>): String {
        return metadata[column.trim().lowercase(Locale.US)].orEmpty()
    }

    private fun applyOutputAttributePreview(metadata: Map<String, String>) {
        val labels = listOf(outputLabel, outputDepartmentLabel, outputAttribute3Label)
        val specs = buildPrintAttributeSpecs(settings, scale = 1)
        labels.forEachIndexed { index, view ->
            val spec = specs.getOrNull(index)
            if (spec == null) {
                view.text = ""
                view.visibility = View.GONE
                return@forEachIndexed
            }
            val value = metadataValue(spec.column, metadata)
            view.text = value
            view.textSize = spec.textSizeSp.toFloat()
            view.translationX = spec.offsetX.toFloat()
            view.translationY = spec.offsetY.toFloat()
            view.visibility = if (value.isBlank()) View.GONE else View.VISIBLE
        }
    }

    private fun outputSelectedName() {
        val name = selectedName
        if (name.isNullOrBlank()) {
            showError("먼저 이름을 선택해 주세요.")
            return
        }
        val department = selectedDepartment.ifBlank { faceDb.departmentOf(name) }
        val grade = selectedGrade.ifBlank { faceDb.gradeOf(name) }
        applyOutputAttributePreview(
            linkedMapOf(
                "name" to name,
                "department" to department,
                "grade" to grade,
            ),
        )
        setOutputPrintStatus(settings.printStatusReadyMessage)
        setHomePrintStatus("출력 준비", settings.printStatusReadyMessage, SyncLightStatus.IN_PROGRESS)
        appendPrintLog(
            "출력 요청 시작: name=$name, department=${department.ifBlank { "-"}}, grade=${grade.ifBlank { "-" }}, model=${settings.printerModel}",
        )
        goTo(PAGE_OUTPUT)
        dispatchPrint(name, department, grade)
        startOutputAutoReturnCountdown()
    }

    private fun dispatchPrint(name: String, department: String, grade: String) {
        if (!settings.printEnabled) {
            setHomePrintStatus("출력 생략", settings.printStatusSuccessMessage, SyncLightStatus.SUCCESS)
            appendPrintLog("출력 비활성화 상태에서 화면 표시만 수행")
            setOutputPrintStatus("선택한 템플릿 속성이 화면에 표시되었습니다.")
            return
        }

        if (settings.printerModel == PrinterModel.NEMONIC) {
            if (!hasBluetoothPermissions()) {
                pendingBluetoothAction = { printWithNemonic(name, department, grade) }
                bluetoothPermissionLauncher.launch(requiredBluetoothPermissions())
                setOutputPrintStatus(settings.printStatusPermissionMessage)
                setHomePrintStatus("권한 확인", settings.printStatusPermissionMessage, SyncLightStatus.IN_PROGRESS)
                appendPrintLog("네모닉 Bluetooth 권한 요청")
                return
            }
            printWithNemonic(name, department, grade)
            return
        }

        when (settings.printerModel) {
            PrinterModel.BROTHER_QL820NWB -> {
                if (!hasBluetoothPermissions()) {
                    pendingBluetoothAction = { printWithBrotherQl820(name, department, grade) }
                    bluetoothPermissionLauncher.launch(requiredBluetoothPermissions())
                    setOutputPrintStatus(settings.printStatusPermissionMessage)
                    setHomePrintStatus("권한 확인", settings.printStatusPermissionMessage, SyncLightStatus.IN_PROGRESS)
                    appendPrintLog("Brother Bluetooth 권한 요청")
                    return
                }
                printWithBrotherQl820(name, department, grade)
            }
            else -> printWithAndroidPrintFramework(name, department, grade, printerLabel = "기본 프린터")
        }
    }

    private fun recordAttendanceOnPrintSuccess(name: String, department: String, grade: String) {
        if (!settings.dbSyncEnabled) return
        dbSyncClient.markAttendance(
            name = name,
            affiliation = grade,
            department = department,
        )
    }

    private fun hasBluetoothPermissions(): Boolean {
        return requiredBluetoothPermissions().all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requiredBluetoothPermissions(): Array<String> {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions += Manifest.permission.BLUETOOTH_SCAN
            permissions += Manifest.permission.BLUETOOTH_CONNECT
        } else {
            permissions += Manifest.permission.ACCESS_FINE_LOCATION
        }
        return permissions.toTypedArray()
    }

    private fun printWithNemonic(name: String, department: String, grade: String) {
        val macAddress = settings.nemonicPrinterMacAddress.trim()
        if (macAddress.isBlank()) {
            setOutputPrintStatus(settings.printStatusErrorMessage)
            setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
            appendPrintLog("네모닉 MAC 주소 미설정", isError = true)
            return
        }

        val printerName = settings.printerName.trim().ifBlank { "Nemonic" }
        setOutputPrintStatus(settings.printStatusConnectingMessage)
        setHomePrintStatus("연결 중", settings.printStatusConnectingMessage, SyncLightStatus.IN_PROGRESS)
        appendPrintLog("네모닉 출력 시작: printer=$printerName, mac=$macAddress")

        Thread {
            val controller = NPrinterController(
                this,
                object : INPrinterControllerCallback {
                    override fun disconnected() {
                        runOnUiThread {
                            setHomePrintStatus("대기", settings.printStatusSuccessMessage, SyncLightStatus.IDLE)
                        }
                        appendPrintLog("네모닉 프린터 연결 종료")
                    }
                    override fun printProgress(current: Int, total: Int, progress: Int) {
                        runOnUiThread {
                            setHomePrintStatus(
                                "출력 중",
                                settings.printStatusSendingMessage,
                                SyncLightStatus.IN_PROGRESS,
                            )
                        }
                    }
                    override fun printComplete(result: Int) {
                        appendPrintLog("네모닉 printComplete result=$result")
                    }
                },
            )

            try {
                controller.setPrintTimeout(true, settings.printerTimeoutSec.coerceAtLeast(1))
                val printer = NPrinter(NPrinterType.NEMONIC, printerName, macAddress)
                appendPrintLog("네모닉 연결 시도")
                val connectResult = controller.connect(printer)
                if (connectResult != NResult.OK && connectResult != NResult.ALREADY_CONNECTED) {
                    runOnUiThread {
                        setOutputPrintStatus(settings.printStatusErrorMessage)
                        setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
                    }
                    appendPrintLog("네모닉 연결 실패: code=$connectResult", isError = true)
                    return@Thread
                }
                runOnUiThread {
                    setHomePrintStatus("준비 중", settings.printStatusPreparingMessage, SyncLightStatus.IN_PROGRESS)
                }
                appendPrintLog("네모닉 연결 성공")

                val bitmap = buildFinalPrintBitmap(name, department, grade)
                runOnUiThread {
                    setOutputPrintStatus(settings.printStatusPreparingMessage)
                    setHomePrintStatus("준비 중", settings.printStatusPreparingMessage, SyncLightStatus.IN_PROGRESS)
                }
                val printInfo = NPrintInfo(printer, bitmap)
                    .setCopies(settings.printerCopies.coerceAtLeast(1))
                    .setEnableLastPageCut(true)
                    .setEnableDither(true)
                    .setPrintQuality(NPrintQuality.MIDDLE)

                appendPrintLog("네모닉 출력 명령 전송 시작")
                val printResult = controller.print(printInfo)
                bitmap.recycle()

                runOnUiThread {
                    setOutputPrintStatus(if (printResult == NResult.OK) settings.printStatusSuccessMessage else settings.printStatusErrorMessage)
                    setHomePrintStatus(
                        if (printResult == NResult.OK) "완료" else "오류",
                        if (printResult == NResult.OK) settings.printStatusSuccessMessage else settings.printStatusErrorMessage,
                        if (printResult == NResult.OK) SyncLightStatus.SUCCESS else SyncLightStatus.ERROR,
                    )
                }
                appendPrintLog(
                    if (printResult == NResult.OK) "네모닉 출력 전송 완료" else "네모닉 출력 실패: code=$printResult",
                    isError = printResult != NResult.OK,
                )
                if (printResult == NResult.OK) {
                    recordAttendanceOnPrintSuccess(name, department, grade)
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    setOutputPrintStatus(settings.printStatusErrorMessage)
                    setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
                }
                appendPrintLog("네모닉 출력 예외: ${t.javaClass.simpleName}: ${t.message ?: "no-message"}", isError = true)
            } finally {
                runCatching { controller.disconnect() }
            }
        }.start()
    }

    private fun printWithBrotherQl820(name: String, department: String, grade: String) {
        val macAddress = settings.brotherPrinterMacAddress.trim()
        if (macAddress.isBlank()) {
            setOutputPrintStatus(settings.printStatusErrorMessage)
            setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
            appendPrintLog("Brother Bluetooth 주소 미설정", isError = true)
            return
        }
        setOutputPrintStatus(settings.printStatusConnectingMessage)
        setHomePrintStatus("연결 중", settings.printStatusConnectingMessage, SyncLightStatus.IN_PROGRESS)
        appendPrintLog("Brother 출력 시작: mac=$macAddress")
        Thread {
            val bitmap = runCatching {
                buildFinalPrintBitmap(name, department, grade, settings)
            }.getOrElse { error ->
                runOnUiThread {
                    setOutputPrintStatus(settings.printStatusErrorMessage)
                    setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
                }
                appendPrintLog("Brother 출력 이미지 생성 실패: ${error.javaClass.simpleName}: ${error.message ?: "no-message"}", isError = true)
                return@Thread
            }
            try {
                runOnUiThread {
                    setOutputPrintStatus(settings.printStatusPreparingMessage)
                    setHomePrintStatus("준비 중", settings.printStatusPreparingMessage, SyncLightStatus.IN_PROGRESS)
                }
                appendPrintLog("Brother 출력 이미지 생성 완료")
                val result = BrotherPrinterManager.printBitmap(
                    this,
                    macAddress,
                    bitmap,
                    settings,
                    onStatus = { status ->
                        runOnUiThread {
                            val uiMessage = when {
                                status.contains("연결", ignoreCase = true) -> settings.printStatusConnectingMessage
                                status.contains("준비", ignoreCase = true) || status.contains("저장", ignoreCase = true) -> settings.printStatusPreparingMessage
                                else -> settings.printStatusSendingMessage
                            }
                            setOutputPrintStatus(uiMessage)
                            setHomePrintStatus("출력 중", uiMessage, SyncLightStatus.IN_PROGRESS)
                        }
                    },
                    onDiagnosticLog = { log ->
                        appendPrintLog(log)
                    },
                )
                runOnUiThread {
                    setOutputPrintStatus(if (result.success) settings.printStatusSuccessMessage else settings.printStatusErrorMessage)
                    setHomePrintStatus(
                        if (result.success) "완료" else "오류",
                        if (result.success) settings.printStatusSuccessMessage else settings.printStatusErrorMessage,
                        if (result.success) SyncLightStatus.SUCCESS else SyncLightStatus.ERROR,
                    )
                }
                appendPrintLog(result.message, isError = !result.success)
                if (result.success) {
                    recordAttendanceOnPrintSuccess(name, department, grade)
                }
            } finally {
                bitmap.recycle()
            }
        }.start()
    }

    private fun printWithAndroidPrintFramework(name: String, department: String, grade: String, printerLabel: String) {
        val bitmap = runCatching {
            buildFinalPrintBitmap(name, department, grade, settings)
        }.getOrElse { error ->
            setOutputPrintStatus(settings.printStatusErrorMessage)
            setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
            appendPrintLog("$printerLabel 출력 이미지 생성 실패: ${error.javaClass.simpleName}: ${error.message ?: "no-message"}", isError = true)
            return
        }

        val printManager = getSystemService(Context.PRINT_SERVICE) as? PrintManager
        if (printManager == null) {
            bitmap.recycle()
            setOutputPrintStatus(settings.printStatusErrorMessage)
            setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
            appendPrintLog("$printerLabel PrintManager 없음", isError = true)
            return
        }

        val paper = resolveSelectedPaper(settings)
        val mediaSize = buildCustomMediaSize(paper, settings.printLandscape)
        val attributes = PrintAttributes.Builder()
            .setMediaSize(mediaSize)
            .setColorMode(PrintAttributes.COLOR_MODE_MONOCHROME)
            .setMinMargins(PrintAttributes.Margins.NO_MARGINS)
            .setResolution(PrintAttributes.Resolution("ql820", "Brother QL-820NWB", 300, 300))
            .build()
        val jobName = buildPrintJobName(name, department, printerLabel)

        runCatching {
            setOutputPrintStatus(settings.printStatusSendingMessage)
            setHomePrintStatus("출력 중", settings.printStatusSendingMessage, SyncLightStatus.IN_PROGRESS)
            appendPrintLog("$printerLabel Android Print Framework 전송 시작")
            printManager.print(
                jobName,
                BitmapPrintDocumentAdapter(
                    context = this,
                    jobName = jobName,
                    bitmap = bitmap,
                    copies = settings.printerCopies.coerceAtLeast(1),
                ),
                attributes,
            )
            setOutputPrintStatus(settings.printStatusSuccessMessage)
            setHomePrintStatus("완료", settings.printStatusSuccessMessage, SyncLightStatus.SUCCESS)
            appendPrintLog("$printerLabel 출력 요청 전송 완료")
            recordAttendanceOnPrintSuccess(name, department, grade)
        }.onFailure { error ->
            bitmap.recycle()
            setOutputPrintStatus(settings.printStatusErrorMessage)
            setHomePrintStatus("오류", settings.printStatusErrorMessage, SyncLightStatus.ERROR)
            appendPrintLog("$printerLabel 출력 실패: ${error.javaClass.simpleName}: ${error.message ?: "no-message"}", isError = true)
        }
    }

    private fun buildPrintJobName(name: String, department: String, printerLabel: String): String {
        val prefix = settings.printerJobTitlePrefix.trim().ifBlank { "FaceRecognition" }
        val suffix = buildString {
            append(name.trim().ifBlank { "unknown" })
            if (department.isNotBlank()) {
                append("_")
                append(department.trim())
            }
        }
        return "$prefix-$printerLabel-$suffix"
    }

    private fun buildCustomMediaSize(paper: PaperProfile, landscape: Boolean): PrintAttributes.MediaSize {
        val widthMils = ((paper.widthMm / 25.4f) * 1000f).toInt().coerceAtLeast(1000)
        val heightMils = ((paper.heightMm / 25.4f) * 1000f).toInt().coerceAtLeast(1000)
        val width = if (landscape) maxOf(widthMils, heightMils) else minOf(widthMils, heightMils)
        val height = if (landscape) minOf(widthMils, heightMils) else maxOf(widthMils, heightMils)
        return PrintAttributes.MediaSize(
            "custom_${paper.id}",
            paper.name,
            width,
            height,
        )
    }

    private fun buildNemonicPrintBitmap(
        metadata: Map<String, String>,
        activeSettings: AppSettings = settings,
        applyLandscapeRotation: Boolean = true,
    ): Bitmap {
        val paper = resolveSelectedPaper(activeSettings)
        val pixelsPerMm = 8.0f // 203dpi thermal printer ~= 8 dot/mm
        val baseWidth = (paper.widthMm * pixelsPerMm).toInt().coerceIn(200, 1200)
        val baseHeight = (paper.heightMm * pixelsPerMm).toInt().coerceIn(120, 1600)
        val renderLandscape = applyLandscapeRotation && activeSettings.paperLandscape
        val printWidth = if (renderLandscape) baseHeight else baseWidth
        val printHeight = if (renderLandscape) baseWidth else baseHeight
        val scale = activeSettings.printTextScaleMultiplier.coerceAtLeast(1)

        val templateBitmap = loadTemplateBitmap(
            uriString = activeSettings.printerTemplateImageUri,
            printWidth = printWidth,
            printHeight = printHeight,
            shrinkToPaperIfOversized = activeSettings.templateShrinkToPaperIfOversized,
            rotateToLandscape = renderLandscape,
            templateQuarterTurns = activeSettings.templateImageRotationQuarterTurns,
        )
        val textRegion = resolveTemplateTextRegion(activeSettings, printWidth, printHeight)
        val regionPadding = 8f
        val attributeSpecs = buildPrintAttributeSpecs(activeSettings, scale)

        if (templateBitmap != null && templateBitmap.width == printWidth && templateBitmap.height == printHeight) {
            val canvas = Canvas(templateBitmap)
            drawTemplateAttributes(canvas, textRegion, regionPadding, activeSettings, metadata, attributeSpecs)
            return templateBitmap
        }

        val bitmap = Bitmap.createBitmap(printWidth, printHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)
        drawTemplateAttributes(canvas, textRegion, regionPadding, activeSettings, metadata, attributeSpecs)
        return bitmap
    }

    private data class PrintAttributeSpec(
        val column: String,
        val textSizeSp: Int,
        val offsetX: Int,
        val offsetY: Int,
        val fallbackBaselineY: Float,
        val bold: Boolean,
    )

    private fun buildPrintAttributeSpecs(activeSettings: AppSettings, scale: Int): List<PrintAttributeSpec> {
        val count = activeSettings.printAttributeCount.coerceIn(1, 3)
        val specs = listOf(
            PrintAttributeSpec(
                column = activeSettings.printAttribute1Column,
                textSizeSp = (activeSettings.printAttribute1TextSizeSp * scale).coerceIn(16, 240),
                offsetX = activeSettings.printAttribute1OffsetX,
                offsetY = activeSettings.printAttribute1OffsetY,
                fallbackBaselineY = 0f,
                bold = true,
            ),
            PrintAttributeSpec(
                column = activeSettings.printAttribute2Column,
                textSizeSp = (activeSettings.printAttribute2TextSizeSp * scale).coerceIn(12, 180),
                offsetX = activeSettings.printAttribute2OffsetX,
                offsetY = activeSettings.printAttribute2OffsetY,
                fallbackBaselineY = 46f,
                bold = false,
            ),
            PrintAttributeSpec(
                column = activeSettings.printAttribute3Column,
                textSizeSp = (activeSettings.printAttribute3TextSizeSp * scale).coerceIn(10, 180),
                offsetX = activeSettings.printAttribute3OffsetX,
                offsetY = activeSettings.printAttribute3OffsetY,
                fallbackBaselineY = 92f,
                bold = false,
            ),
        )
        return specs.take(count)
    }

    private fun drawTemplateAttributes(
        canvas: Canvas,
        textRegion: RectF,
        regionPadding: Float,
        activeSettings: AppSettings,
        metadata: Map<String, String>,
        specs: List<PrintAttributeSpec>,
    ) {
        specs.forEach { spec ->
            val value = metadataValue(spec.column, metadata)
            if (value.isBlank()) return@forEach
            val paint = TextPaint().apply {
                color = Color.BLACK
                textSize = spec.textSizeSp.toFloat()
                isAntiAlias = true
                isFakeBoldText = spec.bold
            }
            val (offsetX, offsetY) = mapStoredOffsetsToRender(
                spec.offsetX,
                spec.offsetY,
                activeSettings.paperLandscape,
            )
            val requestedX = textRegion.left + regionPadding + offsetX
            val requestedBaselineY = textRegion.top + regionPadding + paint.textSize + spec.fallbackBaselineY + offsetY
            val pos = resolveTextPositionInRegion(
                text = value,
                paint = paint,
                requestedX = requestedX,
                requestedBaselineY = requestedBaselineY,
                region = textRegion,
                padding = regionPadding,
            )
            canvas.drawText(value, pos.x, pos.baselineY, paint)
        }
    }

    private data class TextDrawPosition(
        val x: Float,
        val baselineY: Float,
    )

    private fun resolveTextPositionInRegion(
        text: String,
        paint: TextPaint,
        requestedX: Float,
        requestedBaselineY: Float,
        region: RectF,
        padding: Float,
    ): TextDrawPosition {
        val metrics = paint.fontMetrics
        val textWidth = paint.measureText(text).coerceAtLeast(1f)
        val minX = region.left + padding
        val maxX = (region.right - padding - textWidth).coerceAtLeast(minX)
        val minBaselineY = region.top + padding - metrics.ascent
        val maxBaselineY = (region.bottom - padding - metrics.descent).coerceAtLeast(minBaselineY)
        return TextDrawPosition(
            x = requestedX.coerceIn(minX, maxX),
            baselineY = requestedBaselineY.coerceIn(minBaselineY, maxBaselineY),
        )
    }

    private fun buildFinalPrintBitmap(
        name: String,
        department: String,
        grade: String,
        activeSettings: AppSettings = settings,
    ): Bitmap {
        val previewBitmap = buildNemonicPrintBitmap(
            metadata = linkedMapOf(
                "name" to name,
                "department" to department,
                "grade" to grade,
            ),
            activeSettings = activeSettings,
            applyLandscapeRotation = true,
        )
        if (activeSettings.paperLandscape == activeSettings.printLandscape) {
            return previewBitmap
        }
        val matrix = Matrix().apply { postRotate(90f) }
        val rotated = Bitmap.createBitmap(
            previewBitmap,
            0,
            0,
            previewBitmap.width,
            previewBitmap.height,
            matrix,
            true,
        )
        if (rotated !== previewBitmap) {
            previewBitmap.recycle()
        }
        return rotated
    }

    private fun loadTemplateBitmap(
        uriString: String,
        printWidth: Int,
        printHeight: Int,
        shrinkToPaperIfOversized: Boolean,
        rotateToLandscape: Boolean,
        templateQuarterTurns: Int,
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
            val totalRotation = ((templateQuarterTurns.mod(4) * 90) + if (rotateToLandscape) 90 else 0) % 360
            val sourceBitmap = if (totalRotation != 0) {
                val matrix = Matrix().apply { postRotate(totalRotation.toFloat()) }
                Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true).also {
                    if (it !== decoded) decoded.recycle()
                }
            } else {
                decoded
            }
            val target = Bitmap.createBitmap(printWidth, printHeight, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(target)
            canvas.drawColor(Color.WHITE)

            val scale = if (shrinkToPaperIfOversized) {
                minOf(
                    printWidth.toFloat() / sourceBitmap.width.toFloat(),
                    printHeight.toFloat() / sourceBitmap.height.toFloat(),
                )
            } else {
                if (sourceBitmap.width > printWidth || sourceBitmap.height > printHeight) {
                    minOf(
                        printWidth.toFloat() / sourceBitmap.width.toFloat(),
                        printHeight.toFloat() / sourceBitmap.height.toFloat(),
                    )
                } else {
                    1f
                }
            }
            val drawWidth = (sourceBitmap.width * scale).toInt().coerceAtLeast(1)
            val drawHeight = (sourceBitmap.height * scale).toInt().coerceAtLeast(1)
            val left = ((printWidth - drawWidth) / 2f)
            val top = ((printHeight - drawHeight) / 2f)
            val scaled = Bitmap.createScaledBitmap(sourceBitmap, drawWidth, drawHeight, true)
            canvas.drawBitmap(scaled, left, top, null)
            if (scaled !== sourceBitmap) scaled.recycle()
            sourceBitmap.recycle()

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
        if (!activeSettings.paperLandscape) {
            val x = activeSettings.templateTextRegionOffsetX.coerceIn(0, (width - 1).coerceAtLeast(0))
            val y = activeSettings.templateTextRegionOffsetY.coerceIn(0, (height - 1).coerceAtLeast(0))
            val right = (x + activeSettings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(width)
            val bottom = (y + activeSettings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(height)
            val safeRight = if (right <= x) (x + 1).coerceAtMost(width) else right
            val safeBottom = if (bottom <= y) (y + 1).coerceAtMost(height) else bottom
            return RectF(x.toFloat(), y.toFloat(), safeRight.toFloat(), safeBottom.toFloat())
        }

        val portraitWidth = height
        val portraitHeight = width
        val sourceX = activeSettings.templateTextRegionOffsetX.coerceIn(0, (portraitWidth - 1).coerceAtLeast(0))
        val sourceY = activeSettings.templateTextRegionOffsetY.coerceIn(0, (portraitHeight - 1).coerceAtLeast(0))
        val sourceRight = (sourceX + activeSettings.templateTextRegionWidth.coerceAtLeast(1)).coerceAtMost(portraitWidth)
        val sourceBottom = (sourceY + activeSettings.templateTextRegionHeight.coerceAtLeast(1)).coerceAtMost(portraitHeight)
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

    private fun mapStoredOffsetsToRender(offsetX: Int, offsetY: Int, landscape: Boolean): Pair<Float, Float> {
        return if (landscape) {
            Pair(-offsetY.toFloat(), offsetX.toFloat())
        } else {
            Pair(offsetX.toFloat(), offsetY.toFloat())
        }
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
        labelOnlyPrintMode = false
        saveNameButton.text = "이름/부서 저장"
        registerCompleteText.text = settings.uiRegisterCompleteMessage
        val firstDir = nextTargetDirection()
        val firstPrompt = if (firstDir != null) DIRECTION_PROMPTS[firstDir] ?: "" else ""
        registerStatusText.text = registrationPrompt("$firstPrompt\n진행: ${summarizeDirectionProgress()}")
        goTo(PAGE_REGISTER_CAPTURE)
        updateOperationOverlay()
    }

    private fun startLabelOnlyPrint() {
        resetRegistrationState()
        resetCandidateSelectionState()
        resetCandidateGroupState()
        clearCandidates()
        labelOnlyPrintMode = true
        nameInput.setText("")
        departmentInput.setText("")
        gradeInput.setText("")
        registerCompleteText.text = settings.uiLabelOnlyInputMessage
        saveNameButton.text = "이름표만 출력하기"
        saveNameButton.isEnabled = true
        goTo(PAGE_REGISTER_NAME)
        updateOperationOverlay()
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
            registerStatusText.text = settings.uiRegisterCompleteMessage
            goTo(PAGE_REGISTER_NAME)
            updateOperationOverlay()
            return
        }

        val center = face.boundingBox.centerX() to face.boundingBox.centerY()
        val ellipse = getRegisterEllipse(lastImageWidth, lastImageHeight)
        if (!isInsideEllipse(center.first, center.second, ellipse.first.first, ellipse.first.second, ellipse.second.first, ellipse.second.second)) {
            registerStallCount += 1
            registerStatusText.text = registrationPrompt("얼굴을 타원 가이드 안으로 맞춰주세요.\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}")
            updateOperationOverlay()
            return
        }

        val detectedDirection = faceEngine.classifyDirection(face, mirroredPreview = isPreviewMirrored)
        if (detectedDirection != targetDirection) {
            val detectedLabel = DIRECTION_LABELS[detectedDirection] ?: "정면"
            registerStallCount += 1
            registerStatusText.text = registrationPrompt(
                "현재 감지 방향: $detectedLabel\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}",
            )
            updateOperationOverlay()
            return
        }

        val (accepted, msg) = evaluateSimilarityBand(embedding)
        if (!accepted) {
            registerStallCount += 1
            registerStatusText.text = registrationPrompt("$msg\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}")
            updateOperationOverlay()
            return
        }

        registerDirectionEmbeddings[targetDirection]?.add(embedding)
        registerStallCount = 0
        setRegisterCount()

        val nextDirection = nextTargetDirection()
        if (nextDirection == null) {
            registerStatusText.text = settings.uiRegisterCompleteMessage
            goTo(PAGE_REGISTER_NAME)
            updateOperationOverlay()
            return
        }

        registerStatusText.text = registrationPrompt("샘플 저장 완료. $msg\n${DIRECTION_PROMPTS[nextDirection]}\n진행: ${summarizeDirectionProgress()}")
        updateOperationOverlay()
    }

    private fun saveRegistration() {
        val name = nameInput.text?.toString()?.trim().orEmpty()
        val department = departmentInput.text?.toString()?.trim().orEmpty()
        val grade = gradeInput.text?.toString()?.trim().orEmpty()
        if (name.isBlank()) {
            showError("이름을 입력해 주세요.")
            return
        }
        if (department.isBlank()) {
            showError("부서를 입력해 주세요.")
            return
        }
        if (grade.isBlank()) {
            showError("학년을 입력해 주세요.")
            return
        }

        if (labelOnlyPrintMode) {
            selectedName = name
            selectedDepartment = department
            selectedGrade = grade
            selectedCandidateKey = null
            selectedCandidateMissingFrames = 0
            appendPrintLog("이름표만 출력: name=$name, department=$department, grade=$grade")
            outputSelectedName()
            return
        }

        val embeddings = allRegisteredEmbeddings()
        if (embeddings.size < registrationTotalTarget()) {
            showError("얼굴 샘플이 아직 충분하지 않습니다.")
            return
        }

        saveNameButton.isEnabled = false
        registerCompleteText.text = settings.uiSavingMessage
        updateOperationOverlay()
        dbUpdateExecutor.execute {
            runCatching {
                faceDb.upsertPerson(name, department, grade, embeddings, settings.maxEmbeddingsPerPerson)
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
        appendSyncLog("진단: 적응 임베딩 수집: $personName (점수=${String.format(Locale.US, "%.3f", topScore)}, 세션=$adaptiveUpdateCount/${settings.adaptiveUpdateMaxSamplesPerSession})")
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
                    // 적응 임베딩은 즉시 로컬 DB에 반영하고 서버로 전송한다.
                    // 서버에서 수신한 DB 재로드(pull)만 유휴시간을 기다린다.
                    var appliedCount = 0
                    snapshot.forEach { (personName, samples) ->
                        appliedCount += faceDb.appendSamplesToPerson(
                            name = personName,
                            embeddings = samples,
                            maxEmbeddingsPerPerson = settings.maxEmbeddingsPerPerson,
                        )
                    }
                    if (appliedCount > 0) {
                        appendSyncLog("진단: 적응 임베딩 ${appliedCount}개 적용 완료, 서버 업로드 요청")
                        dbSyncClient.pushLatest("자동 임베딩 업데이트", waitForIdle = false)
                    } else {
                        appendSyncLog("진단: 적응 임베딩 적용 실패(이름 불일치?): ${snapshot.keys.joinToString()}")
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

    private fun setHomePrintStatus(summary: String, detail: String, state: SyncLightStatus) {
        homePrintStatusMessage = detail
        homePrintStatusState = state
        printStatusIndicator.text = summary
        printStatusDetailText.text = detail
        applyPrintStatusPanel()
        updateOperationOverlay()
    }

    private fun applyPrintStatusPanel() {
        if (!::printStatusIndicator.isInitialized || !::printStatusDetailText.isInitialized) return
        printStatusIndicator.setTextColor(colorForSyncStatus(homePrintStatusState))
        printStatusDetailText.text = homePrintStatusMessage
        if (::operationPrintStatusText.isInitialized) {
            operationPrintStatusText.text = ""
            operationPrintStatusText.visibility = View.GONE
        }
    }

    private fun refreshOutputSubLabel() {
        outputSubLabel.text = when {
            outputPrintStatusMessage.isBlank() -> ""
            else -> outputPrintStatusMessage
        }
        updateOperationOverlay()
    }

    private fun outputReturnCountdownText(): String {
        return if (outputCountdownRemainingSeconds > 0) {
            buildReturnCountdownText(outputCountdownRemainingSeconds)
        } else {
            ""
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
            stateKey = "known:ambiguous:${candidateGroupKey(top3)}",
            type = MonitorDecisionType.AMBIGUOUS,
            rankings = top3,
        )
    }

    private fun shouldHoldCurrentCandidateUi(decision: MonitorDecision): Boolean {
        if (decision.type == MonitorDecisionType.AMBIGUOUS) {
            val groupKey = candidateGroupKey(decision.rankings)
            if (activeCandidateGroupKey == null || activeCandidateGroupKey == groupKey) {
                activeCandidateGroupKey = groupKey
                pendingCandidateGroupKey = null
                pendingCandidateGroupFrames = 0
                if (selectedCandidateKey == null || decision.rankings.any { candidateKey(it) == selectedCandidateKey }) {
                    selectedCandidateMissingFrames = 0
                }
                return false
            }

            val selectedKey = selectedCandidateKey
            if (selectedKey != null) {
                if (decision.rankings.any { candidateKey(it) == selectedKey }) {
                    selectedCandidateMissingFrames = 0
                    outputNameButton.isEnabled = selectedName?.isNotBlank() == true
                    return true
                }
                selectedCandidateMissingFrames += 1
                if (selectedCandidateMissingFrames < settings.selectedCandidateGraceFrames.coerceAtLeast(1)) {
                    return true
                }
            }

            if (pendingCandidateGroupKey == groupKey) {
                pendingCandidateGroupFrames += 1
            } else {
                pendingCandidateGroupKey = groupKey
                pendingCandidateGroupFrames = 1
            }
            if (pendingCandidateGroupFrames < settings.ambiguousCandidateSwitchFrames.coerceAtLeast(1)) {
                return true
            }

            activeCandidateGroupKey = groupKey
            pendingCandidateGroupKey = null
            pendingCandidateGroupFrames = 0
            selectedCandidateMissingFrames = 0
            return false
        }

        val selectedKey = selectedCandidateKey ?: return false
        val selectedStillVisible = decision.rankings.any { candidateKey(it) == selectedKey } ||
            decision.topMatch?.let { candidateKey(it) == selectedKey } == true
        if (selectedStillVisible) {
            selectedCandidateMissingFrames = 0
            return false
        }

        selectedCandidateMissingFrames += 1
        return selectedCandidateMissingFrames < settings.selectedCandidateGraceFrames.coerceAtLeast(1)
    }

    private fun applyMonitorDecision(decision: MonitorDecision) {
        latestRankings = decision.rankings
        if (shouldHoldCurrentCandidateUi(decision)) return
        if (decision.stateKey == lastMonitorStateKey) return
        appendRecognitionDecisionLog(decision)
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

    private fun appendRecognitionDecisionLog(decision: MonitorDecision) {
        val summary =
            when (decision.type) {
                MonitorDecisionType.IDLE_OUTSIDE -> "인식 판단: 타원 영역 바깥"
                MonitorDecisionType.UNREGISTERED -> {
                    val top = decision.rankings.firstOrNull()
                    if (top == null) {
                        "인식 판단: 후보 없음"
                    } else {
                        "인식 판단: 미등록(top=${top.name}, score=${"%.3f".format(Locale.US, top.score)})"
                    }
                }
                MonitorDecisionType.CLEAR -> {
                    val top = decision.topMatch
                    if (top == null) {
                        "인식 판단: 등록 사용자(세부정보 없음)"
                    } else {
                        "인식 판단: 등록 사용자(name=${top.name}, score=${"%.3f".format(Locale.US, decision.topScore)}, margin=${"%.3f".format(Locale.US, decision.margin)})"
                    }
                }
                MonitorDecisionType.AMBIGUOUS -> {
                    val top3 = decision.rankings.joinToString(" | ") {
                        "${it.name}:${"%.3f".format(Locale.US, it.score)}"
                    }
                    "인식 판단: 후보 경합($top3)"
                }
            }
        appendSyncLog("진단: $summary")
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
                    monitorDetailText.text = if (faceEngine.isBuffaloReady()) {
                        "모델: buffalo_l ONNX"
                    } else {
                        "모델 파일을 찾을 수 없습니다: ${faceEngine.expectedModelPath()}"
                    }
                    updateOperationOverlay()
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
                                    registerStallCount += 1
                                    registerStatusText.text = registrationPrompt("얼굴을 찾지 못했습니다. 화면 중앙으로 와주세요.\n${DIRECTION_PROMPTS[t]}\n진행: ${summarizeDirectionProgress()}")
                                    updateOperationOverlay()
                                }
                            }
                        }
                        return@addOnSuccessListener
                    }
                    lastRecognitionAtMs = System.currentTimeMillis()

                    if (!faceEngine.isBuffaloReady()) {
                        runOnUiThread {
                            val message = "얼굴 인식 모델 파일을 찾을 수 없습니다.\n${faceEngine.expectedModelPath()}"
                            if (page == PAGE_MONITOR) {
                                monitorStatusText.text = "모델 파일이 필요합니다."
                                monitorDetailText.text = message
                            } else if (page == PAGE_REGISTER_CAPTURE) {
                                registerStatusText.text = message
                            }
                            updateOperationOverlay()
                        }
                        return@addOnSuccessListener
                    }

                    val embeddingBitmap = buildEllipseMaskedBitmap(bitmap)
                    val embedding = try {
                        faceEngine.makeEmbedding(face, embeddingBitmap)
                    } finally {
                        embeddingBitmap.recycle()
                    } ?: return@addOnSuccessListener
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
            onDiscoverBrotherPrintersRequested = { onFound ->
                val runSearch = {
                    Thread {
                        val printers = runCatching { BrotherPrinterManager.searchPairedBluetoothPrinters(this) }
                            .getOrDefault(emptyList())
                        runOnUiThread { onFound(printers) }
                    }.start()
                }
                if (!hasBluetoothPermissions()) {
                    pendingBluetoothAction = runSearch
                    bluetoothPermissionLauncher.launch(requiredBluetoothPermissions())
                } else {
                    runSearch()
                }
            },
            onReadBrotherInstalledPaperRequested = { macAddress, onRead ->
                val runRead = {
                    Thread {
                        val media = runCatching {
                            BrotherPrinterManager.readInstalledMedia(
                                context = this,
                                macAddress = macAddress,
                                onDiagnosticLog = { appendPrintLog(it) },
                            )
                        }.getOrNull()
                        runOnUiThread { onRead(media) }
                    }.start()
                }
                if (!hasBluetoothPermissions()) {
                    pendingBluetoothAction = runRead
                    bluetoothPermissionLauncher.launch(requiredBluetoothPermissions())
                } else {
                    runRead()
                }
            },
            onRenderPrintPreview = { previewSettings ->
                buildNemonicPrintBitmap(
                    metadata = linkedMapOf(
                        "name" to "홍길동",
                        "department" to "올스타반",
                        "grade" to "3학년",
                    ),
                    activeSettings = previewSettings,
                    applyLandscapeRotation = true,
                )
            },
            onSave = { updated ->
                val requireReauth = updated.dbServerBaseUrl != settings.dbServerBaseUrl || updated.dbClientId != settings.dbClientId
                settings = if (requireReauth) updated.copy(dbAuthToken = "") else updated
                settingsStore.save(settings)
                faceDb = createFaceDatabase(settings)
                dbSyncClient.start(settings)
                applyEllipseSettings()
                applyConfiguredUiText()
                applyScreenMode()
                setRegisterCount()
                resetRegistrationState()
                goInitialState()
                restartCamera()
            },
        )
    }

    private fun showSettingsMenu() {
        AlertDialog.Builder(this)
            .setTitle("설정")
            .setItems(arrayOf("설정 열기", "앱 종료")) { _, which ->
                when (which) {
                    0 -> openSettingsDialog()
                    1 -> finish()
                }
            }
            .show()
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
        return Uri.parse("content://com.android.externalstorage.documents/tree/primary%3AfaceDB")
    }

    private fun hasFaceDbStorageAccess(): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager()
    }

    private fun requestFaceDbStorageAccessIfNeeded() {
        if (hasFaceDbStorageAccess()) return
        showFaceDbStoragePermissionDialog()
    }

    private fun showFaceDbStoragePermissionDialog() {
        if (isFinishing || isDestroyed) return
        AlertDialog.Builder(this)
            .setTitle("faceDB 저장소 권한 필요")
            .setMessage(
                "Faith ID가 내부 저장소 root의 faceDB 폴더에서 DB, 로그, 템플릿, 출석부 파일을 읽고 쓰려면 " +
                    "'모든 파일 접근 권한'이 필요합니다.\n\n" +
                    "다음 화면에서 Faith ID의 권한을 허용해 주세요.",
            )
            .setPositiveButton("권한 설정 열기") { _, _ -> openAllFilesAccessSettings() }
            .setNegativeButton("나중에", null)
            .show()
    }

    private fun openAllFilesAccessSettings() {
        val appIntent = Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
            Uri.parse("package:$packageName"),
        )
        val fallbackIntent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
        val intent = if (appIntent.resolveActivity(packageManager) != null) appIntent else fallbackIntent
        allFilesAccessSettingsLauncher.launch(intent)
    }

    private fun configureSyncLogFile() {
        syncLogFile = if (hasFaceDbStorageAccess()) {
            File(resolveFaceDataRoot(settings), "logs/db_sync_client.log")
        } else {
            File(filesDir, "logs/db_sync_client.log")
        }.also { it.parentFile?.mkdirs() }
    }

    private fun resolveFaceDataRoot(settings: AppSettings): File {
        val raw = settings.faceDataDir.trim().ifBlank { "/sdcard/faceDB" }
        val dir = if (raw.startsWith("content://")) {
            File("/sdcard/faceDB")
        } else {
            val f = File(raw)
            if (f.isAbsolute) f else File("/sdcard", raw)
        }
        return dir
    }

    private fun ensureFaceDbDirectoryStructure() {
        val root = resolveFaceDataRoot(settings)
        val dirs = listOf(
            root,
            File(root, "models/buffalo_l"),
            File(root, "logs"),
            File(root, "templates"),
        )
        dirs.forEach { dir ->
            runCatching { dir.mkdirs() }.onFailure {
                appendSyncLog("faceDB 디렉토리 생성 실패: ${dir.absolutePath} (${it.javaClass.simpleName})")
            }
        }
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

                override fun exportLocalHashSummary(): Map<String, String> {
                    return faceDb.exportHashSummary()
                }

                override fun exportLocalDeltaBundle(serverHashes: Map<String, String>): Pair<ByteArray, Int>? {
                    return faceDb.exportDeltaBundle(serverHashes)
                }

                override fun applyRemoteDeltaBundle(bundle: ByteArray): Boolean {
                    val applied = faceDb.applyDeltaBundle(bundle)
                    if (applied) {
                        lastMonitorStateKey = null
                        runOnUiThread {
                            monitorDetailText.text = "서버 DB 변경분을 반영했습니다."
                        }
                    }
                    return applied
                }

                override fun isLocalDbSchemaCurrent(): Boolean {
                    return faceDb.isStructuredSchemaCurrent()
                }

                override fun replaceLocalStructuredDb(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
                    val applied = faceDb.replaceWithStructuredBytes(metaDbBytes, embeddingsBinBytes)
                    if (applied) {
                        lastMonitorStateKey = null
                        runOnUiThread {
                            monitorDetailText.text = "서버 DB 구조와 데이터를 반영했습니다."
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
                        uploadProgressCurrent = uploaded
                        uploadProgressTotal = total
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

    private fun appendPrintLog(message: String, isError: Boolean = false) {
        val prefix = if (isError) "출력 오류" else "출력"
        appendSyncLog("$prefix: $message")
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
            .setTitle("시스템 오류 로그")
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
        val dir = resolveFaceDataRoot(settings)
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
                uploadProgressCurrent = 0
                uploadProgressTotal = 0
            }
            text.contains("db 서버 업데이트 실패") -> {
                dbPushStatus = SyncLightStatus.ERROR
                uploadProgressText = ""
                uploadProgressCurrent = 0
                uploadProgressTotal = 0
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
        if (uploadProgressTotal > 0 && dbPushStatus == SyncLightStatus.IN_PROGRESS) {
            dbPushProgressBar.max = uploadProgressTotal
            dbPushProgressBar.progress = uploadProgressCurrent.coerceIn(0, uploadProgressTotal)
        } else {
            dbPushProgressBar.max = 100
            dbPushProgressBar.progress = when (dbPushStatus) {
                SyncLightStatus.SUCCESS -> 100
                else -> 0
            }
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
