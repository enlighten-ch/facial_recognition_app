package com.allstar.faithid.serverapp.ui

import android.content.Context
import android.text.InputType
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.server.ServerSettings
import com.allstar.faithid.server.auth.AdminCredentials
import com.google.android.material.materialswitch.MaterialSwitch
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

    fun editPerson(context: Context, person: Person, onSave: (name: String, department: String, grade: String) -> Unit) {
        val form = form(context)
        val name = field(form, "이름", person.name)
        val department = field(form, "부서", person.department)
        val grade = field(form, "학년", person.grade)
        AlertDialog.Builder(context)
            .setTitle("명단 수정")
            .setView(form)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                val newName = name.text.toString().trim()
                if (newName.isEmpty()) {
                    info(context, "저장하지 않았습니다", "이름은 비워 둘 수 없습니다.")
                } else {
                    onSave(newName, department.text.toString().trim(), grade.text.toString().trim())
                }
            }
            .show()
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

    fun settings(
        context: Context,
        current: ServerSettings,
        onPickFolder: (onPicked: (String) -> Unit) -> Unit,
        onChangePassword: () -> Unit,
        onSave: (ServerSettings) -> Unit,
    ) {
        val form = form(context)
        val adminId = field(form, "관리자 ID", current.credentials.adminId)
        val port = field(form, "포트 (키오스크 설정과 같아야 합니다)", current.port.toString(), number = true)
        val folder = field(form, "데이터 폴더 (얼굴 DB, 출석부)", current.dataDir)
        val pick = com.google.android.material.button.MaterialButton(context).apply {
            text = "폴더 선택"
            setOnClickListener { onPickFolder { folder.setText(it) } }
        }
        form.addView(pick)
        val maxSamples = field(form, "사람별 최대 얼굴 샘플 수", current.maxEmbeddingsPerPerson.toString(), number = true)
        val autoStart = MaterialSwitch(context).apply {
            text = "앱 실행 시 로그인 없이 서버 자동 시작"
            isChecked = current.autoStart
        }
        form.addView(autoStart)
        val password = com.google.android.material.button.MaterialButton(context).apply {
            text = "관리자 비밀번호 변경"
            setOnClickListener { onChangePassword() }
        }
        form.addView(password)
        AlertDialog.Builder(context)
            .setTitle("서버 설정")
            .setView(android.widget.ScrollView(context).apply { addView(form) })
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ ->
                onSave(
                    current.copy(
                        credentials = current.credentials.copy(adminId = adminId.text.toString().trim().ifBlank { AdminCredentials.DEFAULT_ID }),
                        port = port.text.toString().toIntOrNull()?.coerceIn(1024, 65535) ?: current.port,
                        dataDir = folder.text.toString().trim().ifBlank { ServerSettings.DEFAULT_DATA_DIR },
                        maxEmbeddingsPerPerson = maxSamples.text.toString().toIntOrNull()?.coerceIn(10, 1000) ?: current.maxEmbeddingsPerPerson,
                        autoStart = autoStart.isChecked,
                    ),
                )
            }
            .show()
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
