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
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.allstar.faithid.androidsupport.SystemBars
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.server.ServerSettings
import com.allstar.faithid.server.ServerSettingsStore
import com.allstar.faithid.server.StoragePaths
import com.allstar.faithid.serverapp.R
import com.allstar.faithid.serverapp.ServerService
import com.google.android.material.tabs.TabLayout
import java.io.File
import java.time.LocalDate

/**
 * Login and the administrator dashboard. All work happens in [ServerService]; this class only
 * shows its state and forwards button presses.
 */
class MainActivity : AppCompatActivity(), ServerService.Listener {

    private lateinit var settingsStore: ServerSettingsStore
    private lateinit var settings: ServerSettings
    private var service: ServerService? = null
    private var loggedIn = false

    private lateinit var loginContainer: View
    private lateinit var dashboard: View
    private lateinit var stateChip: TextView
    private lateinit var pages: List<View>
    private lateinit var statusPage: StatusPage
    private lateinit var peoplePage: PeoplePage
    private lateinit var attendancePage: AttendancePage
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView
    private var attendanceDate: LocalDate = LocalDate.now()
    private var pendingFolderPick: ((String) -> Unit)? = null
    private val ticker = android.os.Handler(android.os.Looper.getMainLooper())

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
        settingsStore = ServerSettingsStore(File(filesDir, "face_db_server/config.yaml"))
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
        super.onDestroy()
    }

    override fun onStateChanged() = runOnUiThread { refresh() }

    override fun onLog(line: String) = runOnUiThread { if (loggedIn) appendLog(line) }

    private fun bindViews() {
        loginContainer = findViewById(R.id.loginContainer)
        dashboard = findViewById(R.id.dashboardContainer)
        stateChip = findViewById(R.id.serverStateChip)
        logText = findViewById(R.id.logText)
        logScroll = findViewById(R.id.logScroll)

        val passwordInput = findViewById<EditText>(R.id.loginPasswordInput)
        findViewById<Button>(R.id.loginButton).setOnClickListener { tryLogin() }
        passwordInput.setOnEditorActionListener { _, action, _ -> (action == EditorInfo.IME_ACTION_DONE).also { if (it) tryLogin() } }

        pages = listOf(R.id.statusPage, R.id.peoplePage, R.id.attendancePage, R.id.logPage).map { findViewById(it) }
        val tabs = findViewById<TabLayout>(R.id.tabs)
        listOf(R.string.tab_status, R.string.tab_people, R.string.tab_attendance, R.string.tab_log).forEach { tabs.addTab(tabs.newTab().setText(it)) }
        tabs.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) = showPage(tab.position)
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) = showPage(tab.position)
        })

        statusPage = StatusPage(findViewById(R.id.statusPage), statusActions)
        peoplePage = PeoplePage(findViewById(R.id.peoplePage), peopleActions)
        attendancePage = AttendancePage(findViewById(R.id.attendancePage), attendanceActions)

        findViewById<Button>(R.id.settingsButton).setOnClickListener { openSettings() }
        findViewById<Button>(R.id.logoutButton).setOnClickListener { showLogin() }
        findViewById<Button>(R.id.quitButton).setOnClickListener {
            Dialogs.confirm(this, "서버 종료", "서버를 끄면 키오스크가 오프라인으로 동작합니다. 종료할까요?", "종료") {
                service?.shutdown()
                finish()
            }
        }
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
        logText.text = service?.log?.recent()?.joinToString("\n").orEmpty()
        findViewById<TextView>(R.id.logPathText).text = "로그 파일: ${service?.log?.path.orEmpty()}"
        if (settings.credentials.mustChangePassword) {
            Dialogs.changePassword(this, forced = true) { savePassword(it) }
        }
        refresh()
    }

    private fun showLogin() {
        loggedIn = false
        dashboard.visibility = View.GONE
        loginContainer.visibility = View.VISIBLE
        findViewById<EditText>(R.id.loginPasswordInput).setText("")
    }

    private fun showPage(index: Int) {
        pages.forEachIndexed { i, page -> page.visibility = if (i == index) View.VISIBLE else View.GONE }
        if (index == 2) loadAttendance(attendanceDate)
        refresh()
    }

    private fun refresh() {
        val current = service
        val running = current?.isRunning == true
        stateChip.text = getString(if (running) R.string.state_running else R.string.state_stopped)
        stateChip.setTextColor(ContextCompat.getColor(this, if (running) R.color.ok else R.color.ink_muted))
        if (!loggedIn) return
        statusPage.render(current, settings.port, settings.dataDir)
        current?.server?.let { peoplePage.render(it.people.list(), it.people.displayNumbers()) }
    }

    private fun appendLog(line: String) {
        logText.append(if (logText.text.isEmpty()) line else "\n$line")
        if (logText.lineCount > 600) logText.text = logText.text.lines().takeLast(500).joinToString("\n")
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private val statusActions = object : StatusPage.Actions {
        override fun startServer() {
            if (!hasStorageAccess()) {
                requestStorageAccess()
                return
            }
            service?.start(settings) { ok -> runOnUiThread { if (!ok) toast(service?.lastError ?: "서버를 시작하지 못했습니다.") } }
        }

        override fun stopServer() {
            Dialogs.confirm(this@MainActivity, "서버 중지", "중지하는 동안 키오스크는 오프라인으로 동작하고, 다시 시작하면 자동으로 전송합니다.", "중지") {
                service?.stop()
            }
        }

        override fun approve(clientId: String) {
            service?.run({ it.clients.approve(clientId) })
        }

        override fun approveAll() {
            val pending = service?.server?.clients?.list().orEmpty().filter { !it.approved }
            Dialogs.confirm(
                this@MainActivity,
                "전체 승인",
                "다음 키오스크를 모두 승인합니다.\n\n" + pending.joinToString("\n") { "• ${it.clientId} (${it.deviceName})" },
                "승인",
            ) { service?.run({ it.clients.approveAllPending() }) }
        }

        override fun remove(clientId: String) {
            Dialogs.confirm(this@MainActivity, "승인 해제", "$clientId 의 접속 권한을 없앱니다. 다시 쓰려면 새로 승인해야 합니다.", "해제") {
                service?.run({ it.revokeClient(clientId) })
            }
        }
    }

    private val peopleActions = object : PeoplePage.Actions {
        override fun edit(person: Person) {
            Dialogs.editPerson(this@MainActivity, person) { name, department, grade ->
                service?.run({ it.people.updateMetadata(person.key, name, department, grade) })
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
                "데이터 폴더의 face_meta.db를 다시 읽습니다. 백업 파일로 바꿔 넣었을 때 사용하세요. 모든 키오스크가 전체 DB를 새로 받습니다.",
                "다시 읽기",
            ) { service?.run({ it.people.reload() }) }
        }
    }

    private val attendanceActions = object : AttendancePage.Actions {
        override fun showDate(date: LocalDate) = loadAttendance(date)

        override fun saveSheet(date: LocalDate) {
            service?.run({ it.attendance.writeDailyCsv(date) }) { file -> runOnUiThread { toast("저장했습니다: ${file?.absolutePath}") } }
        }

        override fun exportRange(endDate: LocalDate) {
            Dialogs.pickRange(this@MainActivity, endDate) { from, to ->
                service?.run({ it.attendance.exportRange(from, to) }) { file -> runOnUiThread { toast("저장했습니다: ${file?.absolutePath}") } }
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

    private fun openSettings() {
        Dialogs.settings(
            this,
            settings,
            onPickFolder = { callback ->
                pendingFolderPick = callback
                folderPicker.launch(null)
            },
            onChangePassword = { Dialogs.changePassword(this, forced = false) { savePassword(it) } },
            onSave = { next ->
                val restart = service?.isRunning == true && (next.port != settings.port || next.dataDir != settings.dataDir || next.maxEmbeddingsPerPerson != settings.maxEmbeddingsPerPerson)
                settings = next
                settingsStore.save(next)
                if (restart) service?.start(next)
                toast(if (restart) "설정을 저장하고 서버를 다시 시작했습니다." else "설정을 저장했습니다.")
                refresh()
            },
        )
    }

    private fun savePassword(password: String) {
        settings = settings.copy(credentials = settings.credentials.withPassword(password))
        settingsStore.save(settings)
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
