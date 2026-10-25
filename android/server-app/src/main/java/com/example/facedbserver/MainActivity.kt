package com.example.facedbserver

import android.app.AlertDialog
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.text.InputType
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TableLayout
import android.widget.TableRow
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.net.NetworkInterface
import java.util.Collections
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var loginContainer: LinearLayout
    private lateinit var loginIdInput: EditText
    private lateinit var loginPasswordInput: EditText
    private lateinit var loginButton: Button
    private lateinit var loginMessageText: TextView

    private lateinit var mainContainer: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var endpointText: TextView
    private lateinit var pendingText: TextView
    private lateinit var approvedText: TextView
    private lateinit var clientStatusText: TextView
    private lateinit var logText: TextView
    private lateinit var portInput: EditText
    private lateinit var approveClientInput: EditText
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var reloadDbButton: Button
    private lateinit var reloadDbCsvButton: Button
    private lateinit var approveClientButton: Button
    private lateinit var approveAllButton: Button
    private lateinit var settingsButton: Button
    private lateinit var quitButton: Button
    private lateinit var dbTableScroll: ScrollView
    private lateinit var dbTableScrollBar: FrameLayout
    private lateinit var dbTableScrollThumb: View
    private lateinit var dbTable: TableLayout

    private lateinit var configStore: ServerConfigStore
    private var config = ServerUiConfig()
    private val logs = mutableListOf<String>()
    private var pendingStoragePickerCallback: ((String?) -> Unit)? = null
    private val uiHandler = Handler(Looper.getMainLooper())
    private var refreshScheduled = false
    private var lastRenderedRows: List<DbPersonRow> = emptyList()
    private var syncingDbTableScrollBar = false
    private var dbTableScrollRange = 0
    private var dbTableThumbDragOffset = 0f

    @Volatile
    private var isDbLoading: Boolean = false

    private var serverService: FaceDbServerService? = null
    private var isServiceBound = false
    private val pendingServiceTasks = mutableListOf<(FaceDbServerService) -> Unit>()

    private val serviceListener =
        object : FaceDbServerService.Listener {
            override fun onLog(line: String) {
                runOnUiThread { appendLog(line) }
            }

            override fun onStateChanged() {
                runOnUiThread { scheduleRefreshStatus(0L) }
            }
        }

    private val serviceConnection =
        object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                val local = binder as? FaceDbServerService.LocalBinder ?: return
                serverService = local.getService()
                isServiceBound = true
                serverService?.registerListener(serviceListener)
                serverService?.applyStorageDir(config.dbStorageDir)
                if (logs.isEmpty()) {
                    serverService?.snapshotLogs()?.forEach { appendLog(it) }
                }
                flushPendingServiceTasks()
                scheduleRefreshStatus(0L)
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                serverService?.unregisterListener(serviceListener)
                serverService = null
                isServiceBound = false
                scheduleRefreshStatus(0L)
            }
        }

    private val storagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            val callback = pendingStoragePickerCallback
            pendingStoragePickerCallback = null
            if (uri == null) {
                callback?.invoke(null)
                return@registerForActivityResult
            }
            val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            runCatching { contentResolver.takePersistableUriPermission(uri, flags) }
            callback?.invoke(uri.toString())
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()

        configStore = ServerConfigStore(this)
        config = configStore.load()
        portInput.setText(config.defaultPort.toString())

        loginButton.setOnClickListener { tryLogin() }
        startButton.setOnClickListener { startServer() }
        stopButton.setOnClickListener { stopServer() }
        reloadDbButton.setOnClickListener { reloadDb() }
        reloadDbCsvButton.setOnClickListener { reloadDbFromCsv() }
        approveClientButton.setOnClickListener { approveSingleClient() }
        approveAllButton.setOnClickListener { approveAllClients() }
        settingsButton.setOnClickListener { openSettingsDialog() }
        quitButton.setOnClickListener { quitApp() }

        bindServerService()
        showLoginView()
        autoLoadDbOnAppLaunch()
    }

    override fun onStart() {
        super.onStart()
        bindServerService()
    }

    override fun onStop() {
        unbindServerService()
        super.onStop()
    }

    private fun bindViews() {
        loginContainer = findViewById(R.id.loginContainer)
        loginIdInput = findViewById(R.id.loginIdInput)
        loginPasswordInput = findViewById(R.id.loginPasswordInput)
        loginButton = findViewById(R.id.loginButton)
        loginMessageText = findViewById(R.id.loginMessageText)

        mainContainer = findViewById(R.id.mainContainer)
        statusText = findViewById(R.id.statusText)
        endpointText = findViewById(R.id.endpointText)
        pendingText = findViewById(R.id.pendingText)
        approvedText = findViewById(R.id.approvedText)
        clientStatusText = findViewById(R.id.clientStatusText)
        logText = findViewById(R.id.logText)
        portInput = findViewById(R.id.portInput)
        approveClientInput = findViewById(R.id.approveClientInput)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        reloadDbButton = findViewById(R.id.reloadDbButton)
        reloadDbCsvButton = findViewById(R.id.reloadDbCsvButton)
        approveClientButton = findViewById(R.id.approveClientButton)
        approveAllButton = findViewById(R.id.approveAllButton)
        settingsButton = findViewById(R.id.settingsButton)
        quitButton = findViewById(R.id.quitButton)
        dbTableScroll = findViewById(R.id.dbTableScroll)
        dbTableScrollBar = findViewById(R.id.dbTableScrollBar)
        dbTableScrollThumb = findViewById(R.id.dbTableScrollThumb)
        dbTable = findViewById(R.id.dbTable)
        setupDbTableFastScroll()
    }

    private fun setupDbTableFastScroll() {
        dbTableScrollBar.setOnTouchListener { _, event ->
            if (dbTableScrollRange <= 0) return@setOnTouchListener false
            val thumbHeight = dbTableScrollThumb.height.takeIf { it > 0 } ?: return@setOnTouchListener true
            val trackRange = (dbTableScrollBar.height - thumbHeight).coerceAtLeast(1)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    dbTableThumbDragOffset = (event.y - dbTableScrollThumb.y).coerceIn(0f, thumbHeight.toFloat())
                    true
                }
                MotionEvent.ACTION_MOVE, MotionEvent.ACTION_UP -> {
                    val nextThumbY = (event.y - dbTableThumbDragOffset).coerceIn(0f, trackRange.toFloat())
                    val nextScrollY = ((nextThumbY / trackRange.toFloat()) * dbTableScrollRange).toInt()
                    dbTableScroll.scrollTo(dbTableScroll.scrollX, nextScrollY)
                    true
                }
                else -> false
            }
        }
        dbTableScroll.setOnScrollChangeListener { _, _, scrollY, _, _ ->
            updateDbTableThumbPosition(scrollY)
        }
    }

    private fun bindServerService() {
        if (isServiceBound) return
        bindService(
            Intent(this, FaceDbServerService::class.java),
            serviceConnection,
            Context.BIND_AUTO_CREATE,
        )
    }

    private fun unbindServerService() {
        if (!isServiceBound) return
        runCatching { serverService?.unregisterListener(serviceListener) }
        runCatching { unbindService(serviceConnection) }
        serverService = null
        isServiceBound = false
    }

    private fun flushPendingServiceTasks() {
        val service = serverService ?: return
        if (pendingServiceTasks.isEmpty()) return
        val tasks = pendingServiceTasks.toList()
        pendingServiceTasks.clear()
        tasks.forEach { task ->
            runCatching { task(service) }
        }
    }

    private fun withService(task: (FaceDbServerService) -> Unit) {
        val service = serverService
        if (service != null) {
            task(service)
            return
        }
        pendingServiceTasks += task
        bindServerService()
    }

    private fun withServiceAsync(name: String, task: (FaceDbServerService) -> Unit) {
        withService { service ->
            thread(name = name) { task(service) }
        }
    }

    private fun ensureServiceStarted() {
        runCatching {
            startService(Intent(this, FaceDbServerService::class.java))
        }
    }

    private fun quitApp() {
        val shutdownIntent = Intent(this, FaceDbServerService::class.java)
            .setAction(FaceDbServerService.ACTION_SHUTDOWN_SERVICE)
        runCatching { startService(shutdownIntent) }
        runCatching { stopService(Intent(this, FaceDbServerService::class.java)) }
        finish()
    }

    private fun showLoginView() {
        loginContainer.visibility = View.VISIBLE
        mainContainer.visibility = View.GONE
        loginIdInput.setText("")
        loginPasswordInput.setText("")
        loginMessageText.text = ""
    }

    private fun showMainView() {
        loginContainer.visibility = View.GONE
        mainContainer.visibility = View.VISIBLE
        scheduleRefreshStatus(0L)
    }

    private fun autoLoadDbOnAppLaunch() {
        if (isDbLoading) return
        if (config.dbStorageDir.isBlank()) {
            appendLog("경고: DB 경로 설정이 없어 자동 로드를 건너뜁니다. 설정에서 DB 디렉토리를 지정해 주세요.")
            return
        }
        ensureServiceStarted()
        setDbLoading(true)
        withServiceAsync("dbserver-auto-load") { service ->
            service.applyStorageDir(config.dbStorageDir)
            val ok = service.reloadDbFromStorage(config.dbStorageDir)
            val shouldAutoStart = ok && !service.isRunning()
            runOnUiThread {
                setDbLoading(false)
                appendLog(if (ok) "앱 시작시 DB 자동 로드 완료" else "앱 시작시 DB 자동 로드 실패")
                if (shouldAutoStart) {
                    appendLog("앱이 화면에 떠 있는 동안 서버 자동 시작 시도")
                    startServer()
                }
            }
        }
    }

    private fun tryLogin() {
        val id = loginIdInput.text.toString().trim()
        val password = loginPasswordInput.text.toString()
        if (id == config.adminId && password == config.adminPassword) {
            appendLog("관리자 로그인 성공: $id")
            showMainView()
            appendLog("로그인 후 서버 자동 시작 시도")
            startServer()
        } else {
            loginMessageText.text = "ID 또는 PW가 올바르지 않습니다."
        }
    }

    private fun openSettingsDialog() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 24, 40, 8)
        }

        val idInput = EditText(this).apply {
            hint = "관리자 ID"
            setText(config.adminId)
            setSingleLine(true)
        }
        val pwInput = EditText(this).apply {
            hint = "관리자 PW"
            setText(config.adminPassword)
            setSingleLine(true)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val portSetting = EditText(this).apply {
            hint = "기본 포트"
            setText(config.defaultPort.toString())
            setSingleLine(true)
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        val dbStorageInput = EditText(this).apply {
            hint = "DB 저장 디렉토리(face_db.csv 생성)"
            setText(config.dbStorageDir)
            setSingleLine(true)
            isFocusable = false
            isClickable = false
        }
        val pickFolderButton = Button(this).apply {
            text = "폴더 트리에서 선택"
            setOnClickListener {
                pendingStoragePickerCallback = { selected ->
                    if (!selected.isNullOrBlank()) {
                        dbStorageInput.setText(selected)
                    }
                }
                storagePickerLauncher.launch(null)
            }
        }

        root.addView(idInput)
        root.addView(pwInput)
        root.addView(portSetting)
        root.addView(dbStorageInput)
        root.addView(pickFolderButton)

        AlertDialog.Builder(this)
            .setTitle("서버 설정")
            .setView(root)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                val next = ServerUiConfig(
                    adminId = idInput.text.toString().trim().ifBlank { "admin" },
                    adminPassword = pwInput.text.toString().ifBlank { "1234" },
                    defaultPort = portSetting.text.toString().trim().toIntOrNull()?.coerceIn(1, 65535) ?: 18080,
                    dbStorageDir = dbStorageInput.text.toString().trim(),
                )
                val ok = configStore.save(next)
                config = configStore.load()
                withService { service ->
                    service.applyStorageDir(config.dbStorageDir)
                }
                if (serverService?.isRunning() != true) {
                    portInput.setText(config.defaultPort.toString())
                }
                appendLog(if (ok) "설정 저장 완료(config.yaml, DB 디렉토리 포함)" else "설정 저장 실패")
            }
            .show()
    }

    private fun startServer() {
        if (isDbLoading) return
        val port = portInput.text.toString().trim().toIntOrNull()?.coerceIn(1, 65535) ?: config.defaultPort
        ensureServiceStarted()
        setDbLoading(true)
        withServiceAsync("dbserver-start") { service ->
            service.applyStorageDir(config.dbStorageDir)
            val started = service.startServer(port, config.dbStorageDir)
            runOnUiThread {
                setDbLoading(false)
                if (started) {
                    appendLog("서버 앱 시작 완료 (DB 경로: ${config.dbStorageDir.ifBlank { "기본 내부저장소" }})")
                } else {
                    appendLog("서버 시작 요청 처리됨(이미 실행중이거나 시작 실패)")
                    scheduleRefreshStatus(0L)
                }
            }
        }
        scheduleRefreshStatus(0L)
    }

    private fun stopServer() {
        setDbLoading(true)
        withServiceAsync("dbserver-stop") { service ->
            val wasRunning = service.stopServer()
            runOnUiThread {
                setDbLoading(false)
                appendLog(if (wasRunning) "서버 중지 완료" else "서버가 이미 중지 상태입니다.")
            }
        }
    }

    private fun reloadDb() {
        if (isDbLoading) return
        setDbLoading(true)
        withServiceAsync("dbserver-reload") { service ->
            service.applyStorageDir(config.dbStorageDir)
            val ok = service.reloadDbFromStorage(config.dbStorageDir)
            runOnUiThread {
                setDbLoading(false)
                if (ok) {
                    appendLog("DB 재로드 완료")
                    lastRenderedRows = emptyList()
                    scheduleRefreshStatus(0L)
                } else {
                    appendLog("DB 재로드 실패")
                    scheduleRefreshStatus(0L)
                }
            }
        }
    }

    private fun reloadDbFromCsv() {
        if (isDbLoading) return
        setDbLoading(true)
        withServiceAsync("dbserver-reload-csv") { service ->
            service.applyStorageDir(config.dbStorageDir)
            val ok = service.reloadDbMetadataFromCsv(config.dbStorageDir)
            runOnUiThread {
                setDbLoading(false)
                if (ok) {
                    appendLog("DB 재로드(CSV) 완료")
                    lastRenderedRows = emptyList()
                    scheduleRefreshStatus(0L)
                } else {
                    appendLog("DB 재로드(CSV) 실패")
                    scheduleRefreshStatus(0L)
                }
            }
        }
    }

    private fun setDbLoading(loading: Boolean) {
        isDbLoading = loading
        if (loading) {
            statusText.text = "DB 로드중..."
            endpointText.text = "엔드포인트: -"
            startButton.isEnabled = false
            stopButton.isEnabled = false
            reloadDbButton.isEnabled = false
            reloadDbCsvButton.isEnabled = false
        } else {
            scheduleRefreshStatus(0L)
        }
    }

    private fun approveSingleClient() {
        val target = approveClientInput.text.toString().trim()
        if (target.isBlank()) {
            appendLog("승인할 clientId를 입력해 주세요.")
            return
        }
        withServiceAsync("dbserver-approve-one") { service ->
            val ok = service.approveClient(target)
            runOnUiThread {
                appendLog(if (ok) "클라이언트 승인: $target" else "승인 실패(대기중 아님): $target")
                scheduleRefreshStatus(0L)
            }
        }
    }

    private fun approveAllClients() {
        withServiceAsync("dbserver-approve-all") { service ->
            val count = service.approveAllPending()
            runOnUiThread {
                appendLog("전체 승인 완료: ${count}건")
                scheduleRefreshStatus(0L)
            }
        }
    }

    private fun scheduleRefreshStatus(delayMs: Long = 120L) {
        if (refreshScheduled) return
        refreshScheduled = true
        uiHandler.postDelayed(
            {
                refreshScheduled = false
                refreshStatus()
            },
            delayMs.coerceAtLeast(0L),
        )
    }

    private fun refreshStatus() {
        if (isDbLoading) {
            statusText.text = "DB 로드중..."
            endpointText.text = "엔드포인트: -"
            startButton.isEnabled = false
            stopButton.isEnabled = false
            reloadDbButton.isEnabled = false
            reloadDbCsvButton.isEnabled = false
            return
        }

        val service = serverService
        if (service == null) {
            statusText.text = "서버 서비스 연결중..."
            endpointText.text = "엔드포인트: -"
            startButton.isEnabled = true
            stopButton.isEnabled = false
            reloadDbButton.isEnabled = false
            reloadDbCsvButton.isEnabled = false
            pendingText.text = "승인 대기: -"
            approvedText.text = "승인 완료: -"
            clientStatusText.text = "연결된 클라이언트 없음"
            if (lastRenderedRows.isNotEmpty()) {
                renderDbTable(emptyList())
                lastRenderedRows = emptyList()
            }
            return
        }

        val running = service.isRunning()
        if (running) {
            val port = service.endpointPort()
            val ip = detectLocalIpAddress()
            statusText.text = "서버 실행중"
            endpointText.text = "엔드포인트: http://$ip:$port"
        } else {
            statusText.text = "서버 중지"
            endpointText.text = "엔드포인트: -"
        }

        startButton.isEnabled = true
        stopButton.isEnabled = running
        reloadDbButton.isEnabled = true
        reloadDbCsvButton.isEnabled = true

        val pending = service.pendingClientIds()
        val approved = service.approvedClientIds()
        pendingText.text = "승인 대기: " + if (pending.isEmpty()) "없음" else pending.joinToString(", ")
        approvedText.text = "승인 완료: " + if (approved.isEmpty()) "없음" else approved.joinToString(", ")

        val statuses = service.clientStatuses()
        clientStatusText.text = if (statuses.isEmpty()) {
            "연결된 클라이언트 없음"
        } else {
            statuses.joinToString("\n") { status ->
                "${status.clientId} | ${status.status} | 세션 ${status.onlineSessionCount}"
            }
        }

        val rows = service.dbRowsSnapshot()
        if (rows != lastRenderedRows) {
            renderDbTable(rows)
            lastRenderedRows = rows.toList()
        }
    }

    private fun renderDbTable(rows: List<DbPersonRow>) {
        dbTable.removeAllViews()

        val metadataHeaders = rows
            .flatMap { it.metadata.keys }
            .distinctBy { it.lowercase(Locale.US) }
            .ifEmpty { listOf("이름", "부서", "학년") }

        val header = TableRow(this)
        metadataHeaders.forEach { headerLabel ->
            addCell(header, headerLabel, isHeader = true)
        }
        addCell(header, "임베딩 수", isHeader = true)
        addCell(header, "수정", isHeader = true)
        addCell(header, "삭제", isHeader = true)
        dbTable.addView(header)

        if (rows.isEmpty()) {
            val row = TableRow(this)
            metadataHeaders.forEachIndexed { index, _ ->
                addCell(row, if (index == 0) "데이터 없음" else "-", isHeader = false)
            }
            addCell(row, "0", isHeader = false)
            addCell(row, "-", isHeader = false)
            addCell(row, "-", isHeader = false)
            dbTable.addView(row)
            updateDbTableFastScroll()
            return
        }

        rows.forEach { person ->
            val row = TableRow(this)
            metadataHeaders.forEach { key ->
                addCell(row, person.metadata[key].orEmpty().ifBlank { "-" }, isHeader = false)
            }
            addCell(row, person.embeddingCount.toString(), isHeader = false)
            addActionCell(row, "수정") {
                openEditPersonDialog(person)
            }
            addActionCell(row, "삭제") {
                openDeletePersonDialog(person)
            }
            dbTable.addView(row)
        }
        updateDbTableFastScroll()
    }

    private fun updateDbTableFastScroll() {
        dbTableScroll.post {
            val childHeight = dbTableScroll.getChildAt(0)?.height ?: 0
            val scrollRange = (childHeight - dbTableScroll.height).coerceAtLeast(0)
            dbTableScrollRange = scrollRange
            if (scrollRange <= 0) {
                dbTableScrollBar.visibility = View.GONE
                dbTableScrollThumb.translationY = 0f
                return@post
            }
            dbTableScrollBar.visibility = View.VISIBLE
            val trackHeight = dbTableScrollBar.height
            if (trackHeight <= 1) {
                dbTableScrollBar.post { updateDbTableFastScroll() }
                return@post
            }
            val minThumbHeight = minOf(dp(36), trackHeight)
            val thumbHeight = ((dbTableScroll.height.toFloat() / childHeight.toFloat()) * trackHeight)
                .toInt()
                .coerceIn(minThumbHeight, trackHeight)
            val params = dbTableScrollThumb.layoutParams
            if (params.height != thumbHeight) {
                params.height = thumbHeight
                dbTableScrollThumb.layoutParams = params
            }
            updateDbTableThumbPosition(dbTableScroll.scrollY)
        }
    }

    private fun updateDbTableThumbPosition(scrollY: Int) {
        if (dbTableScrollRange <= 0 || dbTableScrollBar.visibility != View.VISIBLE) return
        dbTableScrollBar.post {
            val thumbHeight = dbTableScrollThumb.height.takeIf { it > 0 } ?: return@post
            val trackRange = (dbTableScrollBar.height - thumbHeight).coerceAtLeast(0)
            val ratio = scrollY.coerceIn(0, dbTableScrollRange).toFloat() / dbTableScrollRange.toFloat()
            dbTableScrollThumb.translationY = trackRange * ratio
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    private fun addCell(row: TableRow, text: String, isHeader: Boolean) {
        val tv = TextView(this).apply {
            this.text = text
            setPadding(20, 12, 20, 12)
            setTextColor(if (isHeader) Color.WHITE else Color.parseColor("#1E1E1E"))
            setBackgroundColor(if (isHeader) Color.parseColor("#1A3D8F") else Color.parseColor("#EEF3FF"))
            textSize = if (isHeader) 14f else 13f
        }
        val lp = TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT)
        lp.setMargins(2, 2, 2, 2)
        row.addView(tv, lp)
    }

    private fun addActionCell(row: TableRow, buttonText: String, onClick: () -> Unit) {
        val actionButton = Button(this).apply {
            text = buttonText
            textSize = 12f
            setOnClickListener { onClick() }
        }
        val lp = TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT, TableRow.LayoutParams.WRAP_CONTENT)
        lp.setMargins(2, 2, 2, 2)
        row.addView(actionButton, lp)
    }

    private fun openEditPersonDialog(person: DbPersonRow) {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 24, 40, 8)
        }

        val nameInput = EditText(this).apply {
            hint = "이름"
            setText(person.name)
            setSingleLine(true)
        }

        val departmentInput = EditText(this).apply {
            hint = "부서"
            setText(person.department)
            setSingleLine(true)
        }
        val gradeInput = EditText(this).apply {
            hint = "학년"
            setText(person.grade)
            setSingleLine(true)
        }

        val embeddingCountView = TextView(this).apply {
            text = "임베딩 수(읽기 전용): ${person.embeddingCount}"
            setTextColor(Color.parseColor("#333333"))
            textSize = 13f
            setPadding(0, 12, 0, 0)
        }

        root.addView(nameInput)
        root.addView(departmentInput)
        root.addView(gradeInput)
        root.addView(embeddingCountView)

        AlertDialog.Builder(this)
            .setTitle("DB 항목 수정")
            .setView(root)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                setDbLoading(true)
                withServiceAsync("dbserver-edit-row") { service ->
                    val ok = service.updatePersonRow(
                        personKey = person.personKey,
                        name = nameInput.text.toString(),
                        department = departmentInput.text.toString(),
                        grade = gradeInput.text.toString(),
                    )
                    runOnUiThread {
                        setDbLoading(false)
                        appendLog(if (ok) "DB 항목 수정 완료: ${person.name}" else "DB 항목 수정 실패")
                    }
                }
            }
            .show()
    }

    private fun openDeletePersonDialog(person: DbPersonRow) {
        val desc = buildString {
            append("아래 DB 항목을 삭제할까요?\n\n")
            append("이름: ${person.name}\n")
            append("부서: ${person.department.ifBlank { "-" }}\n")
            append("학년: ${person.grade.ifBlank { "-" }}\n")
            append("임베딩 수: ${person.embeddingCount}")
        }
        AlertDialog.Builder(this)
            .setTitle("DB 항목 삭제")
            .setMessage(desc)
            .setNegativeButton("취소", null)
            .setPositiveButton("삭제") { _, _ ->
                setDbLoading(true)
                withServiceAsync("dbserver-delete-row") { service ->
                    val ok = service.deletePersonRow(person.personKey)
                    runOnUiThread {
                        setDbLoading(false)
                        appendLog(if (ok) "DB 항목 삭제 완료: ${person.name}" else "DB 항목 삭제 실패")
                    }
                }
            }
            .show()
    }

    private fun appendLog(line: String) {
        logs += line
        while (logs.size > 200) logs.removeAt(0)
        logText.text = logs.joinToString("\n")
        if (mainContainer.visibility == View.VISIBLE) {
            scheduleRefreshStatus()
        }
    }

    private fun detectLocalIpAddress(): String {
        return runCatching {
            val interfaces = Collections.list(NetworkInterface.getNetworkInterfaces())
            for (networkInterface in interfaces) {
                val addresses = Collections.list(networkInterface.inetAddresses)
                for (address in addresses) {
                    val host = address.hostAddress ?: continue
                    if (!address.isLoopbackAddress && host.indexOf(':') < 0) {
                        return host
                    }
                }
            }
            "127.0.0.1"
        }.getOrDefault("127.0.0.1")
    }

    override fun onDestroy() {
        uiHandler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }
}
