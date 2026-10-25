package com.allstar.faithid.serverapp.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
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

/** "출석" tab: the attendance sheet of one day, with export buttons. */
internal class AttendancePage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun showDate(date: LocalDate)
        fun saveSheet(date: LocalDate)
        fun exportRange(endDate: LocalDate)
    }

    private val dateText: TextView = root.findViewById(R.id.attendanceDateText)
    private val summary: TextView = root.findViewById(R.id.attendanceSummaryText)
    private val adapter = RowAdapter()
    var date: LocalDate = LocalDate.now()
        private set

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
    }

    fun render(date: LocalDate, rows: List<AttendanceRow>, formatTime: (Long) -> String) {
        this.date = date
        dateText.text = date.format(DateTimeFormatter.ofPattern("yyyy년 M월 d일 (E)", Locale.KOREAN))
        val present = rows.count { it.present }
        val roster = rows.count { it.onRoster }
        val offRoster = rows.count { !it.onRoster }
        summary.text = buildString {
            append("출석 $present 명 / 명단 $roster 명")
            if (offRoster > 0) append(" · 명단 외 $offRoster 명")
            val repeats = rows.count { it.checkIns > 1 }
            if (repeats > 0) append(" · 두 번 이상 체크인 $repeats 명")
        }
        adapter.submit(rows.sortedWith(compareByDescending<AttendanceRow> { it.present }.then(com.allstar.faithid.core.attendance.AttendanceReport.ROW_ORDER)), formatTime)
    }

    private inner class RowAdapter : RecyclerView.Adapter<RowAdapter.Holder>() {
        private var rows: List<AttendanceRow> = emptyList()
        private var formatTime: (Long) -> String = { it.toString() }

        fun submit(next: List<AttendanceRow>, format: (Long) -> String) {
            rows = next
            formatTime = format
            notifyDataSetChanged()
        }

        inner class Holder(view: View) : RecyclerView.ViewHolder(view) {
            val name: TextView = view.findViewById(R.id.attName)
            val department: TextView = view.findViewById(R.id.attDepartment)
            val grade: TextView = view.findViewById(R.id.attGrade)
            val status: TextView = view.findViewById(R.id.attStatus)
            val first: TextView = view.findViewById(R.id.attFirst)
            val last: TextView = view.findViewById(R.id.attLast)
            val count: TextView = view.findViewById(R.id.attCount)
            val kiosk: TextView = view.findViewById(R.id.attKiosk)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_attendance, parent, false))

        override fun getItemCount() = rows.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val row = rows[position]
            val context = holder.itemView.context
            holder.name.text = if (row.onRoster) row.name else "${row.name} (명단 외)"
            holder.department.text = row.department
            holder.grade.text = row.grade
            holder.status.text = if (row.present) "출석" else "미출석"
            holder.status.setTextColor(ContextCompat.getColor(context, if (row.present) R.color.ok else R.color.ink_muted))
            holder.first.text = row.firstAt?.let(formatTime).orEmpty()
            holder.last.text = row.lastAt?.let(formatTime).orEmpty()
            holder.count.text = if (row.checkIns > 0) row.checkIns.toString() else ""
            holder.kiosk.text = row.kiosks.joinToString(" ")
        }
    }
}
