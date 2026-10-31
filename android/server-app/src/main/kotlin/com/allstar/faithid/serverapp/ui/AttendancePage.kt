package com.allstar.faithid.serverapp.ui

import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.serverapp.R
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

/** "출석": one day's attendance as a table, with department summary, filters and export buttons. */
internal class AttendancePage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun showDate(date: LocalDate)
        fun saveSheet(date: LocalDate)
        fun exportRange(endDate: LocalDate)
    }

    private val context = root.context
    private val title: TextView = root.findViewById(R.id.attendanceTitleText)
    private val summary: TextView = root.findViewById(R.id.attendanceSummaryText)
    private val footer: TextView = root.findViewById(R.id.attendanceFooterText)
    private val search: EditText = root.findViewById(R.id.attendanceSearchInput)
    private val departments: LinearLayout = root.findViewById(R.id.attendanceDepartmentSummary)
    private val shows = mapOf(
        AttendanceShow.ALL to root.findViewById<TextView>(R.id.showAllButton),
        AttendanceShow.PRESENT to root.findViewById(R.id.showPresentButton),
        AttendanceShow.ABSENT to root.findViewById(R.id.showAbsentButton),
    )
    private val headers = mapOf(
        AttendanceTable.STATUS to root.findViewById<TextView>(R.id.headAttStatus),
        AttendanceTable.NAME to root.findViewById(R.id.headAttName),
        AttendanceTable.DEPARTMENT to root.findViewById(R.id.headAttDepartment),
        AttendanceTable.TIME to root.findViewById(R.id.headAttTime),
        AttendanceTable.KIOSK to root.findViewById(R.id.headAttKiosk),
    )
    private val headerLabels = headers.mapValues { it.value.text.toString() }
    private val adapter = RowAdapter()

    var date: LocalDate = LocalDate.now()
        private set
    private var rows: List<AttendanceRow> = emptyList()
    private var formatTime: (Long) -> String = { it.toString() }
    private var department: String? = null
    private var show = AttendanceShow.ALL
    private var sort = SortState(AttendanceTable.STATUS)

    init {
        root.findViewById<RecyclerView>(R.id.attendanceList).apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@AttendancePage.adapter
            addItemDecoration(DividerItemDecoration(context, DividerItemDecoration.VERTICAL))
        }
        root.findViewById<Button>(R.id.previousDayButton).setOnClickListener { actions.showDate(date.minusDays(1)) }
        root.findViewById<Button>(R.id.nextDayButton).setOnClickListener { actions.showDate(date.plusDays(1)) }
        root.findViewById<Button>(R.id.todayButton).setOnClickListener { actions.showDate(LocalDate.now()) }
        root.findViewById<Button>(R.id.saveSheetButton).setOnClickListener { actions.saveSheet(date) }
        root.findViewById<Button>(R.id.exportRangeButton).setOnClickListener { actions.exportRange(date) }
        search.onTextChanged { applyFilter() }
        shows.forEach { (value, view) ->
            view.setOnClickListener {
                show = value
                applyFilter()
            }
        }
        headers.forEach { (key, view) ->
            view.setOnClickListener {
                sort = sort.tap(key)
                applyFilter()
            }
        }
    }

    fun render(date: LocalDate, rows: List<AttendanceRow>, formatTime: (Long) -> String) {
        this.date = date
        this.rows = rows
        this.formatTime = formatTime
        title.text = if (date == LocalDate.now()) "오늘 출석" else date.format(DateTimeFormatter.ofPattern("M월 d일 (E) 출석", Locale.KOREAN))
        val present = rows.count { it.present }
        summary.text = buildString {
            append(date.format(DateTimeFormatter.ofPattern("M월 d일", Locale.KOREAN)))
            append(" · ${present}명 왔어요 · 명단 ${rows.count { it.onRoster }}명")
            val offRoster = rows.count { !it.onRoster }
            if (offRoster > 0) append(" · 명단 외 ${offRoster}명")
            val repeats = rows.count { it.checkIns > 1 }
            if (repeats > 0) append(" · 두 번 이상 체크인 ${repeats}명")
        }
        applyFilter()
    }

    private fun applyFilter() {
        val summaries = AttendanceTable.summary(rows)
        if (department != null && summaries.none { it.department == department }) department = null
        departments.removeAllViews()
        summaries.forEach { departments.addView(summaryButton(it)) }
        shows.forEach { (value, view) -> view.isSelected = value == show }
        headers.forEach { (key, view) -> view.text = headerLabels.getValue(key) + sort.arrow(key) }
        val visible = AttendanceTable.rows(rows, search.text?.toString().orEmpty(), department, show, sort)
        adapter.submit(visible)
        footer.text = "${visible.size}명 표시 중 · 시각은 처음 키오스크에 온 시각"
    }

    private fun summaryButton(s: DepartmentSummary): View = LinearLayout(context).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        minimumHeight = context.dp(48)
        setPadding(context.dp(12), 0, context.dp(12), 0)
        setBackgroundResource(R.drawable.bg_summary_button)
        isSelected = s.department == department
        contentDescription = "${s.department} ${s.present}명 출석, 명단 ${s.roster}명. 눌러서 이 부서만 보기"
        addView(TextView(context).apply {
            val label = SpannableString("●  ${s.department.ifBlank { "부서 없음" }}")
            label.setSpan(ForegroundColorSpan(Departments.color(s.department)), 0, 1, 0)
            text = label
            textSize = 15f
            setTextColor(ContextCompat.getColor(context, R.color.ink))
        })
        addView(
            MeterView(context, ContextCompat.getColor(context, R.color.accent_soft), Departments.color(s.department)).apply {
                fraction = if (s.roster == 0) 0f else s.present.toFloat() / s.roster
            },
            LinearLayout.LayoutParams(context.dp(46), context.dp(8)).apply { marginStart = context.dp(10) },
        )
        addView(TextView(context).apply {
            text = "${s.present}/${s.roster}"
            textSize = 14f
            fontFeatureSettings = "tnum"
            setTextColor(ContextCompat.getColor(context, R.color.ink_muted))
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = context.dp(8) })
        setOnClickListener {
            department = if (department == s.department) null else s.department
            applyFilter()
        }
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, context.dp(48)).apply { marginEnd = context.dp(8) }
    }

    private inner class RowAdapter : RecyclerView.Adapter<RowAdapter.Holder>() {
        private var rows: List<AttendanceRow> = emptyList()

        fun submit(next: List<AttendanceRow>) {
            rows = next
            notifyDataSetChanged()
        }

        inner class Holder(view: View) : RecyclerView.ViewHolder(view) {
            val status: TextView = view.findViewById(R.id.attStatus)
            val initial: TextView = view.findViewById(R.id.attInitial)
            val name: TextView = view.findViewById(R.id.attName)
            val dot: View = view.findViewById(R.id.attDepartmentDot)
            val guest: TextView = view.findViewById(R.id.attGuest)
            val department: TextView = view.findViewById(R.id.attDepartment)
            val grade: TextView = view.findViewById(R.id.attGrade)
            val first: TextView = view.findViewById(R.id.attFirst)
            val kiosk: TextView = view.findViewById(R.id.attKiosk)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_attendance, parent, false))

        override fun getItemCount() = rows.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val row = rows[position]
            val color = if (row.onRoster) Departments.color(row.department) else Departments.NONE_COLOR
            if (row.present) holder.status.pill(if (row.checkIns > 1) "✓ 출석 ${row.checkIns}회" else "✓ 출석", PillKind.IN) else holder.status.pill("아직", PillKind.OUT)
            holder.initial.text = initialOf(row.name)
            holder.initial.tint(color)
            holder.initial.alpha = if (row.present) 1f else 0.45f
            holder.name.text = row.name
            holder.name.setTextColor(ContextCompat.getColor(holder.itemView.context, if (row.present) R.color.ink else R.color.ink_muted))
            holder.dot.visibility = if (row.onRoster) View.VISIBLE else View.GONE
            holder.dot.tint(color)
            holder.guest.visibility = if (row.onRoster) View.GONE else View.VISIBLE
            holder.department.text = row.department
            holder.grade.text = row.grade
            holder.first.text = row.firstAt?.let(formatTime) ?: "–"
            holder.kiosk.text = row.kiosks.joinToString(" ").ifBlank { "–" }
        }
    }
}
