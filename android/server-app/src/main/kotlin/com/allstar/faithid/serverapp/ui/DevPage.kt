package com.allstar.faithid.serverapp.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Typeface
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.allstar.faithid.server.ServerSettings
import com.allstar.faithid.server.auth.AdminCredentials
import com.allstar.faithid.serverapp.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The plain developer screen: server state, every setting, the data folder, all kiosks and the
 * whole log with filters. Everything the dashboard tabs do not need day to day lives here.
 */
internal class DevPage(
    private val root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun back()
        fun startServer()
        fun stopServer()
        fun quit()
        fun pickFolder(onPicked: (String) -> Unit)
        fun changePassword()
        fun saveSettings(next: ServerSettings)
        fun importCsv()
        fun exportCsv()
        fun reloadDatabase()
        fun saveTodaySheet()
        fun exportRange()
        fun approve(clientId: String)
        fun remove(clientId: String)
        fun clearLog()
    }

    data class State(
        val running: Boolean,
        val startedAt: Long?,
        val lastError: String?,
        val port: Int,
        val dataDir: String,
        val revision: Long,
        val rosterCount: Int,
        val sampleCount: Int,
        val deletedCount: Int,
        val todayCount: Int,
        val todayOffRoster: Int,
        val kiosks: List<KiosksPage.KioskRow>,
        val appVersion: String,
    )

    enum class Section { STATUS, SETTINGS, DATA, KIOSKS, LOG }

    private val context: Context = root.context
    private val stateText: TextView = root.findViewById(R.id.devStateText)
    private val nav = mapOf(
        Section.STATUS to root.findViewById<TextView>(R.id.devNavStatus),
        Section.SETTINGS to root.findViewById(R.id.devNavSettings),
        Section.DATA to root.findViewById(R.id.devNavData),
        Section.KIOSKS to root.findViewById(R.id.devNavKiosks),
        Section.LOG to root.findViewById(R.id.devNavLog),
    )
    private val sections = mapOf(
        Section.STATUS to root.findViewById<View>(R.id.devStatusSection),
        Section.SETTINGS to root.findViewById(R.id.devSettingsSection),
        Section.DATA to root.findViewById(R.id.devDataSection),
        Section.KIOSKS to root.findViewById(R.id.devKiosksSection),
        Section.LOG to root.findViewById(R.id.devLogSection),
    )
    private val statusList: LinearLayout = root.findViewById(R.id.devStatusList)
    private val dataDirLine: TextView = root.findViewById(R.id.devDataDirLine)
    private val startButton: Button = root.findViewById(R.id.startButton)
    private val stopButton: Button = root.findViewById(R.id.stopButton)

    private val adminId: EditText = root.findViewById(R.id.devAdminIdInput)
    private val port: EditText = root.findViewById(R.id.devPortInput)
    private val dataDir: EditText = root.findViewById(R.id.devDataDirInput)
    private val maxSamples: EditText = root.findViewById(R.id.devMaxSamplesInput)
    private val autoStart: MaterialSwitch = root.findViewById(R.id.devAutoStartSwitch)
    private val configPath: TextView = root.findViewById(R.id.devConfigPathText)
    private val configText: TextView = root.findViewById(R.id.devConfigText)

    private val dataList: LinearLayout = root.findViewById(R.id.devDataList)
    private val fileList: LinearLayout = root.findViewById(R.id.devFileList)
    private val clientTable: LinearLayout = root.findViewById(R.id.devClientTable)

    private val logPath: TextView = root.findViewById(R.id.logPathText)
    private val logChips: LinearLayout = root.findViewById(R.id.devLogChips)
    private val logSearch: EditText = root.findViewById(R.id.devLogSearchInput)
    private val logScroll: ScrollView = root.findViewById(R.id.logScroll)
    private val logText: TextView = root.findViewById(R.id.logText)
    private val logCount: TextView = root.findViewById(R.id.devLogCountText)

    var section = Section.STATUS
        private set
    private var settings: ServerSettings? = null
    private var last: State? = null
    private val logLines = ArrayDeque<String>()
    private var logFilter: LogKind? = null
    private val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.KOREA)

    val isShown: Boolean get() = root.visibility == View.VISIBLE

    init {
        root.findViewById<Button>(R.id.devBackButton).setOnClickListener { actions.back() }
        nav.forEach { (s, view) -> view.setOnClickListener { show(s) } }
        startButton.setOnClickListener { actions.startServer() }
        stopButton.setOnClickListener { actions.stopServer() }
        root.findViewById<Button>(R.id.quitButton).setOnClickListener { actions.quit() }
        root.findViewById<Button>(R.id.devPickFolderButton).setOnClickListener { actions.pickFolder { dataDir.setText(it) } }
        root.findViewById<Button>(R.id.devPasswordButton).setOnClickListener { actions.changePassword() }
        root.findViewById<Button>(R.id.devRevertButton).setOnClickListener { settings?.let(::fillSettings) }
        root.findViewById<Button>(R.id.devSaveButton).setOnClickListener { saveSettings() }
        root.findViewById<Button>(R.id.importCsvButton).setOnClickListener { actions.importCsv() }
        root.findViewById<Button>(R.id.exportCsvButton).setOnClickListener { actions.exportCsv() }
        root.findViewById<Button>(R.id.reloadDbButton).setOnClickListener { actions.reloadDatabase() }
        root.findViewById<Button>(R.id.devSheetButton).setOnClickListener { actions.saveTodaySheet() }
        root.findViewById<Button>(R.id.devRangeButton).setOnClickListener { actions.exportRange() }
        root.findViewById<Button>(R.id.devLogCopyButton).setOnClickListener { copyLog() }
        root.findViewById<Button>(R.id.devLogClearButton).setOnClickListener { actions.clearLog() }
        logSearch.onTextChanged { renderLog(scrollToEnd = false) }
    }

    fun open(settings: ServerSettings, configFile: File, logFile: String, lines: List<String>) {
        this.settings = settings
        fillSettings(settings)
        showConfig(configFile)
        logPath.text = "$logFile · 2 MB가 넘으면 server.log.1로 넘김 · 화면에는 최근 ${MAX_LOG_LINES}줄"
        logLines.clear()
        lines.takeLast(MAX_LOG_LINES).forEach { logLines.addLast(it) }
        renderLog(scrollToEnd = true)
        root.visibility = View.VISIBLE
        show(section)
    }

    fun close() {
        root.visibility = View.GONE
    }

    /** Settings were saved elsewhere (for example a new password); refill the form and config view. */
    fun settingsChanged(settings: ServerSettings, configFile: File) {
        this.settings = settings
        fillSettings(settings)
        showConfig(configFile)
    }

    fun show(next: Section) {
        section = next
        nav.forEach { (s, view) -> view.isSelected = s == next }
        sections.forEach { (s, view) -> view.visibility = if (s == next) View.VISIBLE else View.GONE }
        last?.let { renderSection(it) }
        if (next == Section.LOG) renderLog(scrollToEnd = true)
    }

    fun render(state: State) {
        stateText.text = (if (state.running) "● 실행 중" else "○ 중지됨") + " · 포트 ${state.port}"
        // Redrawn on every refresh so the uptime and "마지막 접속" stay current.
        last = state
        renderSection(state)
    }

    /** A live line from the server, without the file's timestamp. */
    fun appendLog(message: String) {
        logLines.addLast("[${synchronized(stamp) { stamp.format(Date()) }}] $message")
        while (logLines.size > MAX_LOG_LINES) logLines.removeFirst()
        if (isShown && section == Section.LOG) renderLog(scrollToEnd = isNearEnd())
    }

    fun clearLog() {
        logLines.clear()
        renderLog(scrollToEnd = false)
    }

    private fun renderSection(state: State) {
        when (section) {
            Section.STATUS -> renderStatus(state)
            Section.DATA -> renderData(state)
            Section.KIOSKS -> renderKiosks(state)
            Section.SETTINGS, Section.LOG -> Unit
        }
    }

    private fun renderStatus(state: State) {
        startButton.isEnabled = !state.running
        stopButton.isEnabled = state.running
        val online = state.kiosks.count { it.record.approved && it.connections > 0 }
        val pending = state.kiosks.count { !it.record.approved }
        val addresses = Format.localAddresses().map { "http://$it:${state.port}" }
        keyValues(
            statusList,
            listOf(
                "서버" to when {
                    state.running -> "실행 중 · " + (state.startedAt?.let { "${Format.dateTime(it)}부터 (${Format.duration(System.currentTimeMillis() - it)})" } ?: "")
                    state.lastError != null -> "중지됨 · ${state.lastError}"
                    else -> "중지됨"
                },
                "키오스크 접속 주소" to addresses.joinToString("  ").ifBlank { "Wi-Fi 주소 없음" },
                "DB revision" to state.revision.toString(),
                "명단" to "${state.rosterCount}명 · 얼굴 샘플 ${state.sampleCount}개 · 삭제 기록 ${state.deletedCount}건",
                "오늘 출석" to "${state.todayCount}명" + if (state.todayOffRoster > 0) " (명단 외 ${state.todayOffRoster}명)" else "",
                "키오스크" to "${online}대 연결 · ${pending}대 허락 대기 · 전체 ${state.kiosks.size}대",
                "앱" to "Faith DB Server ${state.appVersion} (${context.packageName}) · Android ${android.os.Build.VERSION.RELEASE}",
            ),
        )
        dataDirLine.text = "데이터 폴더: ${state.dataDir}"
    }

    private fun renderData(state: State) {
        keyValues(
            dataList,
            listOf(
                "위치" to state.dataDir,
                "DB revision" to state.revision.toString(),
                "명단" to "${state.rosterCount}명 · 샘플 ${state.sampleCount}개",
            ),
        )
        fileList.removeAllViews()
        val weights = floatArrayOf(2f, 1f, 3f)
        fileList.addRow(tableRow(listOf("파일", "크기", "내용"), header = true, weights = weights))
        val files = File(state.dataDir).listFiles()?.filter { it.isFile }?.sortedBy { it.name }.orEmpty()
        if (files.isEmpty()) fileList.addView(plain("폴더가 비어 있거나 읽을 수 없습니다.", R.color.dev_muted))
        files.forEach { f -> fileList.addRow(tableRow(listOf(f.name, Format.size(f.length()), describe(f.name)), weights = weights)) }
    }

    private fun describe(name: String): String = when {
        name == "face_meta.db" -> "명단·얼굴 샘플 (SQLite)"
        name.startsWith("face_meta.db-") -> "쓰기 중인 변경 (DB 교체 시 함께 지움)"
        name == "face_db.csv" -> "명단 (엑셀로 고친 뒤 'CSV 편집 반영')"
        name.startsWith("attendance.db") -> "출석 기록 (SQLite)"
        Regex("""attendance_\d{4}-\d{2}-\d{2}\.csv""").matches(name) -> "그날의 출석부"
        name.startsWith("attendance_") && name.endsWith(".csv") -> "기간 출석 내보내기"
        else -> ""
    }

    private fun renderKiosks(state: State) {
        clientTable.removeAllViews()
        val weights = floatArrayOf(1.1f, 1.5f, 1.8f, 1.2f, 1.2f, 1.1f, 2.1f)
        clientTable.addRow(tableRow(listOf("상태", "ID", "기기 이름", "요청", "허락", "마지막 접속", "작업"), header = true, weights = weights))
        if (state.kiosks.isEmpty()) clientTable.addView(plain("아직 연결을 요청한 키오스크가 없습니다.", R.color.dev_muted))
        state.kiosks.sortedWith(compareBy<KiosksPage.KioskRow> { it.record.approved }.thenBy { it.record.clientId }).forEach { row ->
            val r = row.record
            val status = when {
                !r.approved -> "허락 대기"
                row.connections > 0 -> "연결됨"
                else -> "오프라인"
            }
            val line = tableRow(
                listOf(status, r.clientId, r.deviceName, Format.dateTime(r.requestedAt), Format.dateTime(r.approvedAt), Format.ago(r.lastSeenAt), ""),
                weights = weights,
            )
            (line.getChildAt(0) as TextView).apply {
                setTypeface(typeface, Typeface.BOLD)
                setTextColor(ContextCompat.getColor(context, if (!r.approved) R.color.warn else if (row.connections > 0) R.color.ok else R.color.dev_muted))
            }
            val buttons = line.getChildAt(line.childCount - 1) as LinearLayout
            if (!r.approved) buttons.addView(devButton("허락", primary = true) { actions.approve(r.clientId) })
            buttons.addView(devButton(if (r.approved) "허락 해제" else "거절") { actions.remove(r.clientId) })
            clientTable.addRow(line)
        }
    }

    private fun fillSettings(s: ServerSettings) {
        adminId.setText(s.credentials.adminId)
        port.setText(s.port.toString())
        dataDir.setText(s.dataDir)
        maxSamples.setText(s.maxEmbeddingsPerPerson.toString())
        autoStart.isChecked = s.autoStart
    }

    private fun saveSettings() {
        val current = settings ?: return
        actions.saveSettings(
            current.copy(
                credentials = current.credentials.copy(adminId = adminId.text.toString().trim().ifBlank { AdminCredentials.DEFAULT_ID }),
                port = port.text.toString().toIntOrNull()?.coerceIn(1024, 65535) ?: current.port,
                dataDir = dataDir.text.toString().trim().ifBlank { ServerSettings.DEFAULT_DATA_DIR },
                maxEmbeddingsPerPerson = maxSamples.text.toString().toIntOrNull()?.coerceIn(10, 1000) ?: current.maxEmbeddingsPerPerson,
                autoStart = autoStart.isChecked,
            ),
        )
    }

    private fun showConfig(file: File) {
        configPath.text = "${file.absolutePath} · 비밀번호는 해시로만 저장됩니다."
        configText.text = runCatching { ConfigView.masked(file.readText()) }.getOrDefault("(읽을 수 없습니다)")
    }

    private fun renderLog(scrollToEnd: Boolean) {
        val query = logSearch.text?.toString()?.trim().orEmpty()
        val counts = logLines.groupingBy { LogKind.of(it) }.eachCount()
        logChips.removeAllViews()
        logChips.addView(logChip("전체", logLines.size, logFilter == null) { logFilter = null })
        LogKind.entries.forEach { kind -> logChips.addView(logChip(kind.label, counts[kind] ?: 0, logFilter == kind) { logFilter = kind }) }
        val shown = logLines.filter { (logFilter == null || LogKind.of(it) == logFilter) && (query.isEmpty() || it.contains(query, ignoreCase = true)) }
        val text = SpannableStringBuilder()
        shown.forEach { line ->
            if (text.isNotEmpty()) text.append('\n')
            val start = text.length
            text.append(line)
            val stampEnd = line.indexOf("] ").let { if (it < 0) 0 else it + 1 }
            text.setSpan(ForegroundColorSpan(0xFF8F8597.toInt()), start, start + stampEnd, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            LOG_COLORS[LogKind.of(line)]?.let { text.setSpan(ForegroundColorSpan(it), start + stampEnd, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE) }
        }
        logText.text = if (shown.isEmpty()) "맞는 줄이 없습니다." else text
        logCount.text = "${shown.size}줄 표시"
        if (scrollToEnd) logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun isNearEnd(): Boolean {
        val child = logScroll.getChildAt(0) ?: return true
        return child.bottom - (logScroll.height + logScroll.scrollY) < context.dp(80)
    }

    private fun copyLog() {
        val clipboard = context.getSystemService(ClipboardManager::class.java)
        clipboard.setPrimaryClip(ClipData.newPlainText("server.log", logText.text.toString()))
        Toast.makeText(context, "로그를 복사했습니다.", Toast.LENGTH_SHORT).show()
    }

    private fun logChip(label: String, count: Int, selected: Boolean, onClick: () -> Unit) = TextView(context).apply {
        text = "$label  $count"
        textSize = 14f
        setTypeface(typeface, Typeface.BOLD)
        gravity = Gravity.CENTER
        setPadding(context.dp(12), 0, context.dp(12), 0)
        setBackgroundResource(R.drawable.bg_dev_chip)
        setTextColor(ContextCompat.getColorStateList(context, R.color.dev_chip_text))
        isSelected = selected
        setOnClickListener {
            onClick()
            renderLog(scrollToEnd = true)
        }
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, context.dp(40)).apply { marginEnd = context.dp(6) }
    }

    private fun keyValues(target: LinearLayout, rows: List<Pair<String, String>>) {
        target.removeAllViews()
        rows.forEach { (k, v) ->
            val row = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, context.dp(9), 0, context.dp(9))
            }
            row.addView(TextView(context).apply {
                text = k
                textSize = 15f
                setTextColor(ContextCompat.getColor(context, R.color.dev_muted))
            }, LinearLayout.LayoutParams(context.dp(170), LinearLayout.LayoutParams.WRAP_CONTENT))
            row.addView(TextView(context).apply {
                text = v
                textSize = 15f
                fontFeatureSettings = "tnum"
                setTextColor(ContextCompat.getColor(context, R.color.dev_ink))
                setTextIsSelectable(true)
            }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            target.addView(row)
            target.addView(divider())
        }
    }

    private fun tableRow(cells: List<String>, header: Boolean = false, weights: FloatArray): LinearLayout {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            minimumHeight = context.dp(if (header) 36 else 52)
        }
        cells.forEachIndexed { i, text ->
            val isActions = !header && i == cells.lastIndex && text.isEmpty()
            val cell: View = if (isActions) {
                LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.END or Gravity.CENTER_VERTICAL }
            } else {
                TextView(context).apply {
                    this.text = text
                    textSize = if (header) 13.5f else 14.5f
                    fontFeatureSettings = "tnum"
                    setPadding(0, 0, context.dp(8), 0)
                    setTextColor(ContextCompat.getColor(context, if (header) R.color.dev_muted else R.color.dev_ink))
                    if (header) setTypeface(typeface, Typeface.BOLD)
                }
            }
            row.addView(cell, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, weights.getOrElse(i) { 1f }))
        }
        return row
    }

    private fun LinearLayout.addRow(row: View) {
        addView(row)
        addView(divider())
    }

    private fun divider() = View(context).apply {
        setBackgroundColor(ContextCompat.getColor(context, R.color.dev_line))
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1)
    }

    private fun plain(text: String, color: Int) = TextView(context).apply {
        this.text = text
        textSize = 14.5f
        setPadding(0, context.dp(10), 0, context.dp(10))
        setTextColor(ContextCompat.getColor(context, color))
    }

    private fun devButton(label: String, primary: Boolean = false, onClick: () -> Unit) =
        MaterialButton(context, null, if (primary) com.google.android.material.R.attr.materialButtonStyle else com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = label
            textSize = 14f
            cornerRadius = context.dp(8)
            if (!primary) {
                strokeColor = ContextCompat.getColorStateList(context, R.color.dev_line)
                setTextColor(ContextCompat.getColor(context, R.color.dev_ink))
            }
            setOnClickListener { onClick() }
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = context.dp(6) }
        }

    private companion object {
        const val MAX_LOG_LINES = 500
        val LOG_COLORS = mapOf(
            LogKind.ERROR to 0xFFFF9F8F.toInt(),
            LogKind.KIOSK to 0xFFB9D7FF.toInt(),
            LogKind.PEOPLE to 0xFFFFD9A0.toInt(),
            LogKind.ATTENDANCE to 0xFFBFE6CC.toInt(),
        )
    }
}
