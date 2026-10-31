package com.allstar.faithid.serverapp.ui

import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.server.auth.ClientRecord
import com.allstar.faithid.serverapp.R
import com.google.android.material.button.MaterialButton
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

/** "오늘의 순서": what to check on a Sunday morning, in order, and today's attendance so far. */
internal class TodayPage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun startServer()
        fun approve(clientId: String)
        fun saveTodaySheet()
        fun showAttendance()
    }

    data class State(
        val running: Boolean,
        val startedAt: Long?,
        val autoStart: Boolean,
        val pending: List<ClientRecord>,
        val approvedKiosks: Int,
        val rosterCount: Int,
        val sampleCount: Int,
        val today: List<AttendanceRow>?,
    )

    private val context = root.context
    private val date: TextView = root.findViewById(R.id.todayDateText)
    private val steps: LinearLayout = root.findViewById(R.id.todaySteps)
    private val count: TextView = root.findViewById(R.id.todayCountText)
    private val departments: LinearLayout = root.findViewById(R.id.todayDepartments)
    private var last: State? = null

    init {
        root.findViewById<Button>(R.id.todayAttendanceButton).setOnClickListener { actions.showAttendance() }
    }

    fun render(state: State) {
        if (state == last) return
        last = state
        date.text = LocalDate.now().format(DateTimeFormatter.ofPattern("M월 d일 EEEE · 위에서부터 차례로 확인하세요", Locale.KOREAN))
        steps.removeAllViews()
        val firstPending = state.pending.firstOrNull()
        step(
            1, done = state.running, title = "서버 켜기",
            detail = if (state.running) {
                (state.startedAt?.let { Format.time(it) + "에 켰어요" } ?: "켜져 있어요") + if (state.autoStart) " · 자동 시작" else ""
            } else {
                "서버가 꺼져 있어 키오스크가 오프라인으로 동작합니다"
            },
            button = if (state.running) null else "서버 시작" to { actions.startServer() },
        )
        step(
            2, done = firstPending == null, title = "키오스크 연결",
            detail = if (firstPending != null) "${firstPending.deviceName.ifBlank { firstPending.clientId }}이(가) 허락을 기다려요" else "${state.approvedKiosks}대 모두 연결됨",
            button = firstPending?.let { p -> "허락하기" to { actions.approve(p.clientId) } },
        )
        step(3, done = state.rosterCount > 0, title = "명단", detail = "등록 ${state.rosterCount}명 · 얼굴 사진 ${state.sampleCount}장")
        step(
            4, done = false, later = true, title = "예배 후 출석부 저장", detail = "예배가 끝나면 눌러 주세요",
            button = "출석부 저장" to { actions.saveTodaySheet() }, ghost = true,
        )

        val rows = state.today.orEmpty()
        count.text = "${rows.count { it.present }}명"
        departments.removeAllViews()
        AttendanceTable.summary(rows).forEach { departments.addView(departmentRow(it)) }
    }

    private fun step(n: Int, done: Boolean, title: String, detail: String, button: Pair<String, () -> Unit>? = null, later: Boolean = false, ghost: Boolean = false) {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, context.dp(14), 0, context.dp(14))
        }
        val badge = TextView(context).apply {
            text = if (done) "✓" else n.toString()
            gravity = Gravity.CENTER
            textSize = 17f
            setTypeface(typeface, Typeface.BOLD)
            setBackgroundResource(R.drawable.bg_dot)
            val (bg, fg) = when {
                done -> R.color.ok to android.R.color.white
                later -> R.color.accent_soft to R.color.accent
                else -> R.color.apricot to android.R.color.white
            }
            tint(ContextCompat.getColor(context, bg))
            setTextColor(ContextCompat.getColor(context, fg))
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        row.addView(badge, LinearLayout.LayoutParams(context.dp(44), context.dp(44)))
        val texts = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        texts.addView(TextView(context).apply {
            text = title
            textSize = 19f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(ContextCompat.getColor(context, R.color.ink))
        })
        texts.addView(TextView(context).apply {
            text = detail
            textSize = 15f
            setTextColor(ContextCompat.getColor(context, R.color.ink_muted))
        })
        row.addView(texts, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = context.dp(16) })
        if (button != null) {
            val b = MaterialButton(context, null, if (ghost) com.google.android.material.R.attr.materialButtonOutlinedStyle else com.google.android.material.R.attr.materialButtonStyle).apply {
                text = button.first
                setOnClickListener { button.second() }
                if (ghost) {
                    strokeColor = ContextCompat.getColorStateList(context, R.color.line)
                    setTextColor(ContextCompat.getColor(context, R.color.ink))
                }
            }
            row.addView(b, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        if (steps.childCount > 0) {
            steps.addView(View(context).apply { setBackgroundColor(ContextCompat.getColor(context, R.color.line_soft)) }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1))
        }
        steps.addView(row)
    }

    private fun departmentRow(summary: DepartmentSummary): View {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, context.dp(5), 0, context.dp(5))
        }
        row.addView(TextView(context).apply {
            text = summary.department.ifBlank { "부서 없음" }
            textSize = 15f
            setTextColor(0xFFFFFFFF.toInt())
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(
            MeterView(context, 0x40FFFFFF, Departments.color(summary.department)).apply {
                fraction = if (summary.roster == 0) 0f else summary.present.toFloat() / summary.roster
            },
            LinearLayout.LayoutParams(context.dp(120), context.dp(9)),
        )
        row.addView(TextView(context).apply {
            text = "${summary.present}/${summary.roster}"
            textSize = 15f
            gravity = Gravity.END
            fontFeatureSettings = "tnum"
            setTextColor(0xFFFFFFFF.toInt())
        }, LinearLayout.LayoutParams(context.dp(56), LinearLayout.LayoutParams.WRAP_CONTENT))
        return row
    }
}
