package com.allstar.faithid.serverapp.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.IBinder
import android.provider.Settings
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.allstar.faithid.androidsupport.SystemBars
import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.server.ServerSettings
import com.allstar.faithid.server.ServerSettingsStore
import com.allstar.faithid.server.StoragePaths
import com.allstar.faithid.serverapp.R
import com.allstar.faithid.serverapp.ServerService
import java.io.File
import java.time.LocalDate
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Login, the four dashboard tabs (오늘의 순서, 명단, 출석, 키오스크) and the developer screen. All
 * work happens in [ServerService]; this class only shows its state and forwards button presses.
 */
class MainActivity : AppCompatActivity(), ServerService.Listener {

    private enum class Tab { TODAY, PEOPLE, ATTENDANCE, KIOSKS }

    private lateinit var settingsStore: ServerSettingsStore
    private lateinit var settingsFile: File
    private lateinit var settings: ServerSettings
    private var service: ServerService? = null
    private var loggedIn = false

    private lateinit var loginContainer: View
    private lateinit var dashboard: View
    private lateinit var stateChip: TextView
    private lateinit var kioskBadge: View
    private lateinit var tabs: Map<Tab, TextView>
    private lateinit var pages: Map<Tab, View>
    private lateinit var todayPage: TodayPage
    private lateinit var peoplePage: PeoplePage
    private lateinit var attendancePage: AttendancePage
    private lateinit var kiosksPage: KiosksPage
    private lateinit var devPage: DevPage
    private var tab = Tab.TODAY
    private var attendanceDate: LocalDate = LocalDate.now()
    private var pendingFolderPick: ((String) -> Unit)? = null
    private val ticker = android.os.Handler(android.os.Looper.getMainLooper())

    /** Today's attendance, read off the main thread on every refresh; null until the first read. */
    private var todayRows: List<AttendanceRow>? = null
    private val reader = Executors.newSingleThreadExecutor { Thread(it, "faith-ui-reader").apply { isDaemon = true } }
    private val reading = AtomicBoolean(false)

    /** Kiosk requests and last-seen times change without a state event; refresh the dashboard every 2 s. */
    private val tick = object : Runnable {
        override fun run() {
            if (loggedIn) refresh()
            ticker.postDelayed(this, 2_000)
        }
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val bound = (binder as ServerService.LocalBinder).service
            service = bound
            bound.addListener(this@MainActivity)
            if (settings.autoStart && !bound.isRunning) {
                bound.start(settings)
            } else {
                bound.open(settings)
            }
            refresh()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    private val folderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        val callback = pendingFolderPick
        pendingFolderPick = null
        if (uri != null) callback?.invoke(StoragePaths.normalize(uri.toString()))
    }

    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        SystemBars.padForSystemBars(findViewById(R.id.root))
        settingsFile = File(filesDir, "face_db_server/config.yaml")
        settingsStore = ServerSettingsStore(settingsFile)
        settings = settingsStore.load()
        bindViews()
        showLogin()
        requestPermissionsIfNeeded()
        startService(Intent(this, ServerService::class.java))
        bindService(Intent(this, ServerService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    override fun onResume() {
        super.onResume()
        ticker.post(tick)
    }

    override fun onPause() {
        ticker.removeCallbacks(tick)
        super.onPause()
    }

    override fun onDestroy() {
        service?.removeListener(this)
        unbindService(connection)
        reader.shutdown()
        super.onDestroy()
    }

    override fun onStateChanged() = runOnUiThread { refresh() }

    override fun onLog(line: String) = runOnUiThread { if (loggedIn) devPage.appendLog(line) }

    private fun bindViews() {
        loginContainer = findViewById(R.id.loginContainer)
        dashboard = findViewById(R.id.dashboardContainer)
        stateChip = findViewById(R.id.serverStateChip)
        kioskBadge = findViewById(R.id.kioskBadge)

        val passwordInput = findViewById<EditText>(R.id.loginPasswordInput)
        findViewById<Button>(R.id.loginButton).setOnClickListener { tryLogin() }
        passwordInput.setOnEditorActionListener { _, action, _ -> (action == EditorInfo.IME_ACTION_DONE).also { if (it) tryLogin() } }

        tabs = mapOf(
            Tab.TODAY to findViewById(R.id.tabToday),
            Tab.PEOPLE to findViewById(R.id.tabPeople),
            Tab.ATTENDANCE to findViewById(R.id.tabAttendance),
            Tab.KIOSKS to findViewById(R.id.tabKiosks),
        )
        pages = mapOf(
            Tab.TODAY to findViewById(R.id.todayPage),
            Tab.PEOPLE to findViewById(R.id.peoplePage),
            Tab.ATTENDANCE to findViewById(R.id.attendancePage),
            Tab.KIOSKS to findViewById(R.id.kiosksPage),
        )
        tabs.forEach { (t, view) -> view.setOnClickListener { showTab(t) } }

        todayPage = TodayPage(findViewById(R.id.todayPage), todayActions)
        peoplePage = PeoplePage(findViewById(R.id.peoplePage), peopleActions)
        attendancePage = AttendancePage(findViewById(R.id.attendancePage), attendanceActions)
        kiosksPage = KiosksPage(findViewById(R.id.kiosksPage), kioskActions)
        devPage = DevPage(findViewById(R.id.devPage), devActions)

        findViewById<View>(R.id.devScreenButton).setOnClickListener { openDevScreen() }
        findViewById<View>(R.id.logoutButton).setOnClickListener { showLogin() }
        // Back closes the developer screen before it leaves the app.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (devPage.isShown) {
                    devPage.close()
                    return
                }
                isEnabled = false
                onBackPressedDispatcher.onBackPressed()
                isEnabled = true
            }
        })
        showTab(Tab.TODAY)
    }

    private fun tryLogin() {
        val id = findViewById<EditText>(R.id.loginIdInput).text.toString()
        val passwordInput = findViewById<EditText>(R.id.loginPasswordInput)
        val message = findViewById<TextView>(R.id.loginMessageText)
        if (!settings.credentials.verify(id, passwordInput.text.toString())) {
            message.text = "ID 또는 비밀번호가 올바르지 않습니다."
            passwordInput.setText("")
            return
        }
        message.text = ""
        loggedIn = true
        loginContainer.visibility = View.GONE
        dashboard.visibility = View.VISIBLE
        if (settings.credentials.mustChangePassword) {
            Dialogs.changePassword(this, forced = true) { savePassword(it) }
        }
        refresh()
    }

    private fun showLogin() {
        loggedIn = false
        devPage.close()
        dashboard.visibility = View.GONE
        loginContainer.visibility = View.VISIBLE
        findViewById<EditText>(R.id.loginPasswordInput).setText("")
    }

    private fun showTab(next: Tab) {
        tab = next
        tabs.forEach { (t, view) -> view.isSelected = t == next }
        pages.forEach { (t, page) -> page.visibility = if (t == next) View.VISIBLE else View.GONE }
        if (next == Tab.ATTENDANCE) loadAttendance(attendanceDate)
        refresh()
    }

    private fun openDevScreen() {
        val current = service
        devPage.open(settings, settingsFile, current?.log?.path.orEmpty(), current?.log?.recent().orEmpty())
        refresh()
    }

    private fun refresh() {
        val current = service
        val running = current?.isRunning == true
        stateChip.text = getString(if (running) R.string.state_running else R.string.state_stopped)
        stateChip.setTextColor(ContextCompat.getColor(this, if (running) R.color.ok else R.color.ink_muted))
        if (!loggedIn) return
        val server = current?.server
        val people = server?.people?.list().orEmpty()
        val online = server?.onlineClients().orEmpty()
        val kiosks = server?.clients?.list().orEmpty().map { KiosksPage.KioskRow(it, online[it.clientId] ?: 0) }
        kioskBadge.visibility = if (kiosks.any { !it.record.approved }) View.VISIBLE else View.GONE
        readToday()
        val today = todayRows
        val samples = people.sumOf { it.embeddings.size }
        when (tab) {
            Tab.TODAY -> todayPage.render(
                TodayPage.State(
                    running = running,
                    startedAt = current?.startedAt,
                    autoStart = settings.autoStart,
                    pending = kiosks.filter { !it.record.approved }.map { it.record },
                    approvedKiosks = kiosks.count { it.record.approved },
                    connectedKiosks = kiosks.count { it.record.approved && it.connections > 0 },
                    rosterCount = people.size,
                    sampleCount = samples,
                    today = today,
                ),
            )
            Tab.PEOPLE -> peoplePage.render(people, today.orEmpty().filter { it.present && it.onRoster }.map { it.personKey }.toSet())
            Tab.ATTENDANCE -> if (attendanceDate == LocalDate.now() && today != null) {
                server?.attendance?.let { ledger -> attendancePage.render(attendanceDate, today, ledger::formatTime) }
            }
            Tab.KIOSKS -> kiosksPage.render(kiosks, settings.port)
        }
        if (devPage.isShown) {
            devPage.render(
                DevPage.State(
                    running = running,
                    startedAt = current?.startedAt,
                    lastError = current?.lastError,
                    port = settings.port,
                    dataDir = settings.dataDir,
                    revision = server?.people?.revision ?: 0,
                    rosterCount = people.size,
                    sampleCount = samples,
                    deletedCount = server?.people?.deletedCount ?: 0,
                    todayCount = today.orEmpty().count { it.present },
                    todayOffRoster = today.orEmpty().count { it.present && !it.onRoster },
                    kiosks = kiosks,
                    appVersion = appVersion(),
                ),
            )
        }
    }

    /** Reads today's sheet in the background; the next refresh shows it. */
    private fun readToday() {
        val ledger = service?.server?.attendance ?: return
        if (!reading.compareAndSet(false, true)) return
        reader.execute {
            val rows = runCatching { ledger.dailyRows(ledger.today()) }.getOrNull()
            reading.set(false)
            runOnUiThread {
                if (rows != null && rows != todayRows) {
                    todayRows = rows
                    if (loggedIn) refresh()
                }
            }
        }
    }

    private fun appVersion(): String = runCatching { packageManager.getPackageInfo(packageName, 0).versionName }.getOrNull() ?: "?"

    private fun startServer() {
        if (!hasStorageAccess()) {
            requestStorageAccess()
            return
        }
        service?.start(settings) { ok -> runOnUiThread { if (!ok) toast(service?.lastError ?: "서버를 시작하지 못했습니다.") } }
    }

    private fun approve(clientId: String) {
        service?.run({ it.clients.approve(clientId) })
    }

    private fun remove(clientId: String) {
        val waiting = service?.server?.clients?.isPending(clientId) == true
        val (title, message, button) = if (waiting) {
            Triple("요청 거절", "$clientId 의 연결 요청을 거절합니다. 우리 교회 태블릿이면 키오스크에서 다시 요청할 수 있습니다.", "거절")
        } else {
            Triple("연결 끊기", "$clientId 의 접속 허락을 없앱니다. 다시 쓰려면 새로 허락해야 합니다.", "끊기")
        }
        Dialogs.confirm(this, title, message, button) {
            service?.run({ it.revokeClient(clientId) })
        }
    }

    private fun saveSheet(date: LocalDate) {
        service?.run({ it.attendance.writeDailyCsv(date) }) { file -> runOnUiThread { toast("저장했습니다: ${file?.absolutePath}") } }
    }

    private fun exportRange(endDate: LocalDate) {
        Dialogs.pickRange(this, endDate) { from, to ->
            service?.run({ it.attendance.exportRange(from, to) }) { file -> runOnUiThread { toast("저장했습니다: ${file?.absolutePath}") } }
        }
    }

    private val todayActions = object : TodayPage.Actions {
        override fun startServer() = this@MainActivity.startServer()
        override fun approve(clientId: String) = this@MainActivity.approve(clientId)
        override fun saveTodaySheet() = saveSheet(LocalDate.now())
        override fun showAttendance() = showTab(Tab.ATTENDANCE)
    }

    private val kioskActions = object : KiosksPage.Actions {
        override fun approve(clientId: String) = this@MainActivity.approve(clientId)

        override fun approveAll() {
            val pending = service?.server?.clients?.list().orEmpty().filter { !it.approved }
            Dialogs.confirm(
                this@MainActivity,
                "전체 허락",
                "다음 키오스크를 모두 허락합니다.\n\n" + pending.joinToString("\n") { "• ${it.clientId} (${it.deviceName})" },
                "허락",
            ) { service?.run({ it.clients.approveAllPending() }) }
        }

        override fun remove(clientId: String) = this@MainActivity.remove(clientId)
    }

    private val peopleActions = object : PeoplePage.Actions {
        override fun save(person: Person, name: String, department: String, grade: String) {
            if (name.isEmpty()) {
                Dialogs.info(this@MainActivity, "저장하지 않았습니다", "이름은 비워 둘 수 없습니다.")
                return
            }
            service?.run({ it.people.updateMetadata(person.key, name, department, grade) }) {
                runOnUiThread {
                    peoplePage.closeEditor()
                    toast("$name 정보를 저장했습니다. 모든 키오스크에 반영됩니다.")
                }
            }
        }

        override fun delete(person: Person) {
            Dialogs.confirm(
                this@MainActivity,
                "명단에서 삭제",
                "${person.metadata.label()}\n얼굴 샘플 ${person.embeddings.size}개\n\n모든 키오스크에서도 지워지고, 되돌릴 수 없습니다.",
                "삭제",
            ) { service?.run({ it.people.delete(person.key) }) }
        }
    }

    private val attendanceActions = object : AttendancePage.Actions {
        override fun showDate(date: LocalDate) = loadAttendance(date)
        override fun saveSheet(date: LocalDate) = this@MainActivity.saveSheet(date)
        override fun exportRange(endDate: LocalDate) = this@MainActivity.exportRange(endDate)
    }

    private val devActions = object : DevPage.Actions {
        override fun back() = devPage.close()

        override fun startServer() = this@MainActivity.startServer()

        override fun stopServer() {
            Dialogs.confirm(this@MainActivity, "서버 중지", "중지하는 동안 키오스크는 오프라인으로 동작하고, 다시 시작하면 자동으로 전송합니다.", "중지") {
                service?.stop()
            }
        }

        override fun quit() {
            Dialogs.confirm(this@MainActivity, "서버 종료", "서버를 끄면 키오스크가 오프라인으로 동작합니다. 종료할까요?", "종료") {
                service?.shutdown()
                finish()
            }
        }

        override fun pickFolder(onPicked: (String) -> Unit) {
            pendingFolderPick = onPicked
            folderPicker.launch(null)
        }

        override fun changePassword() = Dialogs.changePassword(this@MainActivity, forced = false) { savePassword(it) }

        override fun saveSettings(next: ServerSettings) {
            val restart = service?.isRunning == true && (next.port != settings.port || next.dataDir != settings.dataDir || next.maxEmbeddingsPerPerson != settings.maxEmbeddingsPerPerson)
            settings = next
            settingsStore.save(next)
            devPage.settingsChanged(next, settingsFile)
            if (restart) service?.start(next)
            toast(if (restart) "설정을 저장하고 서버를 다시 시작했습니다." else "설정을 저장했습니다.")
            refresh()
        }

        override fun importCsv() {
            service?.run({ server ->
                val file = File(server.config.dataDir, "face_db.csv")
                if (!file.exists()) null else server.people.importCsvEdits(file.readText())
            }) { result ->
                runOnUiThread {
                    when {
                        result == null -> toast("face_db.csv 파일이 없습니다.")
                        else -> Dialogs.info(
                            this@MainActivity,
                            "CSV 편집 반영",
                            "바뀐 사람 ${result.changed.size}명" +
                                if (result.unmatched.isEmpty()) "" else "\n찾지 못한 줄 ${result.unmatched.size}개: " + result.unmatched.take(5).joinToString { it.metadata.name },
                        )
                    }
                }
            }
        }

        override fun exportCsv() {
            service?.run({ it.people.exportCsv() }) { file -> runOnUiThread { toast("저장했습니다: ${file?.absolutePath}") } }
        }

        override fun reloadDatabase() {
            Dialogs.confirm(
                this@MainActivity,
                "DB 다시 읽기",
                "데이터 폴더의 face_meta.db를 다시 읽습니다. 백업 파일로 바꿀 때는 먼저 서버를 중지하고 파일을 바꾼 뒤(같은 이름의 -wal, -shm 파일은 지우기), 서버를 시작하고 이 버튼을 누르세요. 모든 키오스크가 전체 DB를 새로 받습니다.",
                "다시 읽기",
            ) { service?.run({ it.people.reload() }) }
        }

        override fun saveTodaySheet() = saveSheet(LocalDate.now())

        override fun exportRange() = this@MainActivity.exportRange(LocalDate.now())

        override fun approve(clientId: String) = this@MainActivity.approve(clientId)

        override fun remove(clientId: String) = this@MainActivity.remove(clientId)

        override fun clearLog() {
            Dialogs.confirm(this@MainActivity, "로그 지우기", "server.log의 내용을 지웁니다.", "지우기") {
                service?.log?.clear()
                devPage.clearLog()
            }
        }
    }

    private fun loadAttendance(date: LocalDate) {
        attendanceDate = date
        service?.run({ server -> server.attendance.dailyRows(date) to server.attendance }) { result ->
            val (rows, ledger) = result ?: return@run
            runOnUiThread { attendancePage.render(date, rows, ledger::formatTime) }
        }
    }

    private fun savePassword(password: String) {
        settings = settings.copy(credentials = settings.credentials.withPassword(password))
        settingsStore.save(settings)
        devPage.settingsChanged(settings, settingsFile)
        toast("비밀번호를 바꿨습니다.")
    }

    private fun hasStorageAccess(): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) Environment.isExternalStorageManager() else true

    private fun requestStorageAccess() {
        AlertDialog.Builder(this)
            .setTitle("저장소 권한이 필요합니다")
            .setMessage("얼굴 DB와 출석부를 ${settings.dataDir} 폴더에 저장하려면 '모든 파일 접근' 권한이 필요합니다. 다음 화면에서 허용해 주세요.")
            .setPositiveButton("권한 화면 열기") { _, _ ->
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName"))
                runCatching { startActivity(intent) }.onFailure { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
            }
            .setNegativeButton("나중에", null)
            .show()
    }

    private fun requestPermissionsIfNeeded() {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (!hasStorageAccess()) requestStorageAccess()
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()
}
