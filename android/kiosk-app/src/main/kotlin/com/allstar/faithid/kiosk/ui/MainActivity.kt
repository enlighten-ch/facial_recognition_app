package com.allstar.faithid.kiosk.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.view.WindowManager
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.allstar.faithid.androidsupport.FileLog
import com.allstar.faithid.androidsupport.SystemBars
import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.kiosk.R
import com.allstar.faithid.kiosk.camera.CameraController
import com.allstar.faithid.kiosk.camera.GuideOverlayView
import com.allstar.faithid.kiosk.data.KioskData
import com.allstar.faithid.kiosk.data.KioskLog
import com.allstar.faithid.kiosk.data.KioskSyncService
import com.allstar.faithid.kiosk.flow.CheckIn
import com.allstar.faithid.kiosk.flow.FlowEffects
import com.allstar.faithid.kiosk.flow.KioskFlow
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.print.BrotherDetectedMedia
import com.allstar.faithid.kiosk.print.BrotherPairedPrinter
import com.allstar.faithid.kiosk.print.BrotherPrinter
import com.allstar.faithid.kiosk.print.LabelRenderer
import com.allstar.faithid.kiosk.print.PrintCoordinator
import com.allstar.faithid.kiosk.recognition.FaceEngine
import com.allstar.faithid.kiosk.recognition.FrameAnalyzer
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.settings.KioskSettingsStore
import com.allstar.faithid.kiosk.settings.PrinterTools
import com.allstar.faithid.kiosk.settings.SettingsDialog
import java.util.concurrent.Executors

/**
 * Wires the kiosk together: camera → analyzer → flow → screen, flow effects → data and printer.
 * Business rules live in [KioskFlow]; this class only connects components and Android plumbing.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var store: KioskSettingsStore
    private lateinit var settings: KioskSettings
    private lateinit var log: FileLog
    private lateinit var screen: KioskScreen
    private lateinit var flow: KioskFlow
    private lateinit var renderer: LabelRenderer
    private lateinit var printer: PrintCoordinator
    private lateinit var camera: CameraController
    private var syncService: KioskSyncService? = null

    /** Lives in [KioskSyncService], so it keeps syncing while this screen is gone. */
    private val data: KioskData? get() = syncService?.data

    private val dataHost = object : KioskData.Host {
        override fun canApplyRemoteChanges(): Boolean =
            currentScreen == Screen.MONITOR && (analyzer?.idleFor(settings.adaptiveUpdateIdleSeconds) ?: true)

        override fun onStatus(status: SyncStatus) {
            main.post { refreshReadiness() }
        }

        override fun onIndexChanged(index: FaceIndex) {
            main.post {
                flow.updateIndex(index)
                refreshReadiness()
            }
        }
    }

    /** Data actions that arrived before the service was connected (main thread only). */
    private val waitingForData = mutableListOf<(KioskData) -> Unit>()

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val service = (binder as KioskSyncService.LocalBinder).service
            syncService = service
            service.host = dataHost
            service.data?.let { d -> waitingForData.forEach { it(d) } }
            waitingForData.clear()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            syncService = null
        }
    }

    @Volatile
    private var analyzer: FrameAnalyzer? = null
    private var engine: FaceEngine? = null
    private val engineLoader = Executors.newSingleThreadExecutor { Thread(it, "faith-model") }
    private val main = Handler(Looper.getMainLooper())
    private var lastPrintStatus = ""

    @Volatile
    private var currentScreen = Screen.MONITOR

    private var pendingBluetooth: Pair<() -> Unit, () -> Unit>? = null
    private var pendingTemplatePick: ((String?) -> Unit)? = null

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) camera.start(settings.cameraIndex == 1) else showError("카메라 권한이 필요합니다. 기기 설정에서 허용해 주세요.")
        requestNotificationPermission()
    }

    /** Without it Android 13+ hides the sync service's notification; syncing itself does not need it. */
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) syncService?.showNotification()
    }

    private val bluetoothPermission = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        val (onGranted, onDenied) = pendingBluetooth ?: return@registerForActivityResult
        pendingBluetooth = null
        if (grants.values.all { it }) onGranted() else onDenied()
    }

    private val templatePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val callback = pendingTemplatePick
        pendingTemplatePick = null
        if (uri != null) runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        callback?.invoke(uri?.toString())
    }

    private val ticker = object : Runnable {
        override fun run() {
            if (::flow.isInitialized && flow.state.screen == Screen.OUTPUT) flow.onSecond()
            refreshStatus()
            main.postDelayed(this, 1_000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        SystemBars.padForSystemBars(findViewById(R.id.developerRoot))
        SystemBars.padForSystemBars(findViewById(R.id.operationOverlay))

        log = KioskLog.get(this)
        store = KioskSettingsStore(this)
        settings = KioskData.ensureIdentity(store.load()).also { store.save(it) }
        log.log("앱 시작 (키오스크 ID ${settings.dbClientId})")

        renderer = LabelRenderer(this)
        printer = PrintCoordinator(
            this,
            renderer,
            requestBluetoothPermission = { permissions, onGranted, onDenied ->
                pendingBluetooth = onGranted to onDenied
                bluetoothPermission.launch(permissions)
            },
            onStatus = { message, _ ->
                lastPrintStatus = message
                flow.updatePrintStatus(message)
            },
            log = log::log,
        )
        screen = KioskScreen(this, screenActions) { data?.index?.allMetadata().orEmpty() }
        flow = KioskFlow(settings.flowConfig(), effects) { state ->
            currentScreen = state.screen
            screen.render(state, settings.uiSavingMessage)
        }
        applyScreenMode()
        screen.render(flow.state, settings.uiSavingMessage)
        DebugBridge.observe = { observation -> main.post { flow.onObservation(observation) } }
        DebugBridge.finishScreen = { main.post { finish() } }
        DebugBridge.typeText = { text -> main.post { (currentFocus as? EditText)?.let { it.setText(text); it.setSelection(text.length) } } }

        camera = CameraController(this, findViewById<PreviewView>(R.id.previewView), ::onFrame, ::showError)
        applyGuide()
        startSyncService()
        loadModel()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            camera.start(settings.cameraIndex == 1)
            requestNotificationPermission()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    override fun onResume() {
        super.onResume()
        main.post(ticker)
        if (!settings.developerViewEnabled && settings.hideSystemBars) SystemBars.hideForKiosk(this, true)
    }

    override fun onPause() {
        main.removeCallbacks(ticker)
        super.onPause()
    }

    override fun onDestroy() {
        DebugBridge.observe = null
        DebugBridge.finishScreen = null
        camera.shutdown()
        if (syncService?.host === dataHost) syncService?.host = null
        unbindService(serviceConnection)
        printer.shutdown()
        engineLoader.execute { engine?.close() }
        engineLoader.shutdown()
        super.onDestroy()
    }

    private fun onFrame(bitmap: android.graphics.Bitmap) {
        val current = analyzer
        if (current == null || DebugBridge.cameraPaused) bitmap.recycle() else current.onFrame(bitmap)
    }

    private fun loadModel() {
        engineLoader.execute {
            val loaded = FaceEngine(this, log::log)
            engine = loaded
            if (!loaded.isReady) {
                runOnUiThread { showError(loaded.modelError ?: "얼굴 인식 모델을 불러오지 못했습니다.\n${loaded.modelPath}") }
                return@execute
            }
            analyzer = FrameAnalyzer(
                engine = loaded,
                settings = { settings },
                screen = { currentScreen },
                mirrored = { camera.mirrored },
                guideScale = { screen.guideScale },
                deliver = { observation -> main.post { flow.onObservation(observation) } },
                log = log::log,
            )
        }
    }

    /** Runs [action] now, or once the sync service is connected, so no registration or check-in is lost. */
    private fun withData(action: (KioskData) -> Unit) {
        data?.let(action) ?: waitingForData.add(action)
    }

    /** Starts (or finds the already running) sync service; it outlives this screen. */
    private fun startSyncService() {
        val intent = Intent(this, KioskSyncService::class.java)
        ContextCompat.startForegroundService(this, intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun refreshReadiness() {
        val d = data ?: return
        flow.updateReadiness(d.readiness(), d.banner())
        refreshStatus()
    }

    private fun refreshStatus() {
        val d = data ?: return
        val status = d.status
        val (ops, checkIns) = d.repository.pendingCounts()
        val connection = when (status) {
            is SyncStatus.Online -> "연결됨 · ${settings.dbServerBaseUrl} · revision ${status.serverRevision}"
            else -> KioskData.describe(status) + " · ${settings.dbServerBaseUrl}"
        } + " · ID ${settings.dbClientId}"
        val database = "${d.index.size}명 · 전송 대기 변경 ${ops}건, 출석 ${checkIns}건"
        screen.renderStatus(connection, database, lastPrintStatus)
    }

    private val effects = object : FlowEffects {
        override fun addSamples(personKey: String, samples: List<FloatArray>) {
            withData { it.addSamples(personKey, samples) }
        }

        override fun register(metadata: PersonMetadata, samples: List<FloatArray>): String? = data?.register(metadata, samples)

        override fun checkIn(checkIn: CheckIn) {
            withData { it.checkIn(checkIn) }
        }

        override fun print(metadata: PersonMetadata) = printer.print(metadata, settings)

        override fun log(message: String) = log.log(message)
    }

    private val screenActions = object : KioskScreen.Actions {
        override fun startRegistration() = flow.startRegistration()
        override fun startLabelOnly() = flow.startLabelOnly()
        override fun selectCandidate(personKey: String) = flow.selectCandidate(personKey)
        override fun print() = flow.pressPrint()
        override fun submitInfo(name: String, department: String, grade: String) = flow.submitInfo(name, department, grade)
        override fun duplicateIsMe() = flow.confirmDuplicate()
        override fun duplicateNotMe() = flow.rejectDuplicate()
        override fun home() = flow.goHome()
        override fun openSettings() = showSettings()
        override fun openLog() = showLog()
        override fun syncNow() {
            data?.syncNow()
            log.log("수동 동기화 요청")
        }
    }

    private fun showSettings() {
        SettingsDialog.show(
            context = this,
            current = settings,
            connectionSummary = data?.let { KioskData.describe(it.status) } ?: "-",
            renderer = renderer,
            printerTools = printerTools,
            pickTemplate = { callback ->
                pendingTemplatePick = callback
                templatePicker.launch(arrayOf("image/*"))
            },
            onSave = ::applySettings,
        )
    }

    private fun applySettings(next: KioskSettings) {
        val previous = settings
        settings = KioskData.ensureIdentity(next)
        store.save(settings)
        log.log("설정 저장")
        val dataChanged = previous.dbSyncEnabled != settings.dbSyncEnabled ||
            previous.dbServerBaseUrl != settings.dbServerBaseUrl ||
            previous.dbClientId != settings.dbClientId ||
            previous.dbClientSecret != settings.dbClientSecret ||
            previous.maxEmbeddingsPerPerson != settings.maxEmbeddingsPerPerson ||
            previous.dbConnectTimeoutSeconds != settings.dbConnectTimeoutSeconds ||
            previous.dbConnectRetryDelaySeconds != settings.dbConnectRetryDelaySeconds
        flow.reconfigure(settings.flowConfig())
        if (dataChanged) syncService?.open(settings)
        if (previous.cameraIndex != settings.cameraIndex) camera.restart(settings.cameraIndex == 1)
        applyGuide()
        applyScreenMode()
        refreshReadiness()
    }

    private fun applyScreenMode() {
        screen.configure(settings)
        SystemBars.hideForKiosk(this, !settings.developerViewEnabled && settings.hideSystemBars)
        screen.render(flow.state, settings.uiSavingMessage)
    }

    private fun applyGuide() {
        findViewById<GuideOverlayView>(R.id.guideOverlay).apply {
            axisXRatio = settings.registerEllipseAxisXRatio
            axisYRatio = settings.registerEllipseAxisYRatio
            centerYOffsetRatio = settings.registerEllipseCenterYOffsetRatio
            outsideDimAlpha = settings.registerEllipseOutsideDimAlpha
        }
    }

    private val printerTools = object : PrinterTools {
        override fun findBrotherPrinters(onFound: (List<BrotherPairedPrinter>) -> Unit) = withBluetooth {
            Thread {
                val printers = runCatching { BrotherPrinter.searchPairedBluetoothPrinters(this@MainActivity) }.getOrDefault(emptyList())
                runOnUiThread { onFound(printers) }
            }.start()
        }

        override fun readBrotherMedia(macAddress: String, onRead: (BrotherDetectedMedia?) -> Unit) = withBluetooth {
            Thread {
                val media = runCatching { BrotherPrinter.readInstalledMedia(this@MainActivity, macAddress) { log.log(it) } }.getOrNull()
                runOnUiThread { onRead(media) }
            }.start()
        }
    }

    private fun withBluetooth(action: () -> Unit) {
        val missing = PrintCoordinator.bluetoothPermissions().any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (!missing) {
            action()
            return
        }
        pendingBluetooth = action to { showError("Bluetooth 권한이 없어 프린터를 찾을 수 없습니다.") }
        bluetoothPermission.launch(PrintCoordinator.bluetoothPermissions())
    }

    private fun showLog() {
        val text = TextView(this).apply {
            setTextIsSelectable(true)
            textSize = 13f
            setPadding(24, 24, 24, 24)
            this.text = log.readAll().ifBlank { "기록이 없습니다." }
        }
        AlertDialog.Builder(this)
            .setTitle("로그 (${log.path})")
            .setView(ScrollView(this).apply { addView(text) })
            .setNeutralButton("지우기") { _, _ -> log.clear() }
            .setPositiveButton("닫기", null)
            .show()
    }

    private fun showError(message: String) {
        log.log("오류: $message")
        AlertDialog.Builder(this).setTitle("확인이 필요합니다").setMessage(message).setPositiveButton("확인", null).show()
    }
}
