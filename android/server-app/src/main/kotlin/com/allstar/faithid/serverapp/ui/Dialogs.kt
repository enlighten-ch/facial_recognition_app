package com.allstar.faithid.serverapp.ui

import android.content.Context
import android.text.InputType
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import com.allstar.faithid.server.auth.AdminCredentials
import java.time.LocalDate

internal object Dialogs {

    fun confirm(context: Context, title: String, message: String, positive: String, onConfirm: () -> Unit) {
        AlertDialog.Builder(context)
            .setTitle(title)
            .setMessage(message)
            .setNegativeButton("취소", null)
            .setPositiveButton(positive) { _, _ -> onConfirm() }
            .show()
    }

    fun info(context: Context, title: String, message: String) {
        AlertDialog.Builder(context).setTitle(title).setMessage(message).setPositiveButton("확인", null).show()
    }

    /** Asks for a new administrator password. [forced] dialogs cannot be dismissed. */
    fun changePassword(context: Context, forced: Boolean, onSave: (String) -> Unit) {
        val form = form(context)
        if (forced) {
            form.addView(TextView(context).apply {
                text = "처음 설정된 비밀번호(1234)는 누구나 알 수 있습니다. 새 비밀번호를 정해 주세요."
                setPadding(0, 0, 0, 16)
            })
        }
        val first = field(form, "새 비밀번호 (${AdminCredentials.MIN_PASSWORD_LENGTH}자 이상)", "", password = true)
        val second = field(form, "새 비밀번호 확인", "", password = true)
        val dialog = AlertDialog.Builder(context)
            .setTitle("관리자 비밀번호 변경")
            .setView(form)
            .setCancelable(!forced)
            .setPositiveButton("변경", null)
            .apply { if (!forced) setNegativeButton("취소", null) }
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val password = first.text.toString()
                when {
                    !AdminCredentials.isAcceptablePassword(password) ->
                        first.error = "${AdminCredentials.MIN_PASSWORD_LENGTH}자 이상, 1234가 아닌 비밀번호를 입력해 주세요."
                    password != second.text.toString() -> second.error = "두 비밀번호가 다릅니다."
                    else -> {
                        onSave(password)
                        dialog.dismiss()
                    }
                }
            }
        }
        dialog.show()
    }

    fun pickRange(context: Context, endDate: LocalDate, onPick: (LocalDate, LocalDate) -> Unit) {
        val options = listOf(
            "최근 7일" to (endDate.minusDays(6) to endDate),
            "최근 4주" to (endDate.minusDays(27) to endDate),
            "이번 달" to (endDate.withDayOfMonth(1) to endDate.withDayOfMonth(endDate.lengthOfMonth())),
            "지난 달" to endDate.minusMonths(1).let { it.withDayOfMonth(1) to it.withDayOfMonth(it.lengthOfMonth()) },
            "최근 3개월" to (endDate.minusMonths(3).plusDays(1) to endDate),
        )
        AlertDialog.Builder(context)
            .setTitle("출석 기록 내보내기 기간")
            .setItems(options.map { "${it.first}  (${it.second.first} ~ ${it.second.second})" }.toTypedArray()) { _, which ->
                val (from, to) = options[which].second
                onPick(from, to)
            }
            .setNegativeButton("취소", null)
            .show()
    }

    private fun form(context: Context) = LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL
        val pad = (20 * resources.displayMetrics.density).toInt()
        setPadding(pad, pad / 2, pad, 0)
    }

    private fun field(parent: LinearLayout, label: String, value: String, password: Boolean = false, number: Boolean = false): EditText {
        parent.addView(TextView(parent.context).apply { text = label })
        return EditText(parent.context).apply {
            setText(value)
            setSingleLine(true)
            inputType = when {
                password -> InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
                number -> InputType.TYPE_CLASS_NUMBER
                else -> InputType.TYPE_CLASS_TEXT
            }
            parent.addView(this)
        }
    }
}
