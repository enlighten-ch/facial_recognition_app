package com.allstar.faithid.kiosk.ui

import android.app.Activity
import android.graphics.Color
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.ViewFlipper
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.R
import com.allstar.faithid.kiosk.flow.RegistrationView
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState
import com.allstar.faithid.kiosk.settings.KioskSettings
import java.util.Locale

/**
 * Renders [UiState] into the views of activity_main.xml: the developer layout here, the full-screen
 * operation layout through [OperationScreen]. Holds no state of its own beyond animations.
 */
class KioskScreen(activity: Activity, private val actions: Actions, people: () -> List<PersonMetadata>) {

    interface Actions {
        fun startRegistration()
        fun startLabelOnly()
        fun selectCandidate(personKey: String)
        fun print()
        fun submitInfo(name: String, department: String, grade: String)
        fun duplicateIsMe()
        fun duplicateNotMe()
        fun home()
        fun openSettings()
        fun openLog()
        fun syncNow()
    }

    private fun <T : View> Activity.v(id: Int): T = findViewById(id)

    private val developerRoot: View = activity.v(R.id.developerRoot)
    private val operationRoot: View = activity.v(R.id.operationRoot)
    private val cameraPanel: View = activity.v(R.id.cameraPreviewPanel)
    private val operationCameraHost: FrameLayout = activity.v(R.id.operationCameraHost)
    private val developerPageHost: FrameLayout = activity.v(R.id.developerPageHost)
    private val pageFlipper: ViewFlipper = activity.v(R.id.pageFlipper)
    private val headerTitle: TextView = activity.v(R.id.headerTitle)
    private val headerSubtitle: TextView = activity.v(R.id.headerSubtitle)
    private val backButton: Button = activity.v(R.id.backButton)
    private val connectionStatus: TextView = activity.v(R.id.connectionStatusText)
    private val dbStatus: TextView = activity.v(R.id.dbStatusText)
    private val printStatus: TextView = activity.v(R.id.printStatusText)

    private val monitorMessage: TextView = activity.v(R.id.monitorStatusText)
    private val monitorDetail: TextView = activity.v(R.id.monitorDetailText)
    private val registerButton: Button = activity.v(R.id.registerStartButton)
    private val labelOnlyButton: Button = activity.v(R.id.labelOnlyPrintButton)
    private val candidates: LinearLayout = activity.v(R.id.candidateContainer)
    private val printButton: Button = activity.v(R.id.printButton)

    private val registerMessage: TextView = activity.v(R.id.registerStatusText)
    private val registerDetail: TextView = activity.v(R.id.registerDetailText)
    private val registerProgress: ProgressBar = activity.v(R.id.registerProgressBar)
    private val registerCount: TextView = activity.v(R.id.registerCountText)
    private val duplicateIsMe: Button = activity.v(R.id.duplicateIsMeButton)
    private val duplicateNotMe: Button = activity.v(R.id.duplicateNotMeButton)

    private val infoTitle: TextView = activity.v(R.id.infoTitleText)
    private val infoDetail: TextView = activity.v(R.id.infoDetailText)
    private val nameInput: EditText = activity.v(R.id.nameInput)
    private val departmentInput: EditText = activity.v(R.id.departmentInput)
    private val gradeInput: EditText = activity.v(R.id.gradeInput)
    private val submitButton: Button = activity.v(R.id.submitInfoButton)

    private val outputName: TextView = activity.v(R.id.outputNameText)
    private val outputDepartment: TextView = activity.v(R.id.outputDepartmentText)
    private val outputGrade: TextView = activity.v(R.id.outputGradeText)
    private val outputStatus: TextView = activity.v(R.id.outputStatusText)

    private val operation = OperationScreen(activity, actions, people)

    private var developerMode = true
    private var lastScreen: Screen? = null

    /** Size of the face circle the camera analysis should use (registration steps). */
    val guideScale: Float get() = if (developerMode) 1f else operation.guideScale

    init {
        registerButton.setOnClickListener { actions.startRegistration() }
        labelOnlyButton.setOnClickListener { actions.startLabelOnly() }
        printButton.setOnClickListener { actions.print() }
        duplicateIsMe.setOnClickListener { actions.duplicateIsMe() }
        duplicateNotMe.setOnClickListener { actions.duplicateNotMe() }
        submitButton.setOnClickListener {
            actions.submitInfo(nameInput.text.toString(), departmentInput.text.toString(), gradeInput.text.toString())
        }
        backButton.setOnClickListener { actions.home() }
        activity.v<Button>(R.id.settingsButton).setOnClickListener { actions.openSettings() }
        activity.v<Button>(R.id.logButton).setOnClickListener { actions.openLog() }
        activity.v<Button>(R.id.syncNowButton).setOnClickListener { actions.syncNow() }
    }

    fun configure(settings: KioskSettings) {
        developerMode = settings.developerViewEnabled
        printButton.text = settings.uiOutputButtonText
        developerRoot.visibility = if (developerMode) View.VISIBLE else View.GONE
        operationRoot.visibility = if (developerMode) View.GONE else View.VISIBLE
        if (developerMode) {
            operation.reset()
            move(cameraPanel, (developerPageHost.parent as ViewGroup), 0, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f).apply { marginEnd = 20 })
            cameraPanel.setPadding(24, 24, 24, 24)
        } else {
            move(cameraPanel, operationCameraHost, 0, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            cameraPanel.setPadding(0, 0, 0, 0)
            operation.configure(settings)
        }
        lastScreen = null
    }

    fun render(state: UiState, savingText: String) {
        if (!developerMode) {
            operation.render(state)
            return
        }
        if (state.screen != lastScreen) onScreenChanged(state)
        lastScreen = state.screen
        when (state.screen) {
            Screen.MONITOR -> renderMonitor(state)
            Screen.REGISTER_CAPTURE -> renderCapture(state)
            Screen.REGISTER_INFO -> renderInfo(state, savingText)
            Screen.OUTPUT -> renderOutput(state)
        }
    }

    fun renderStatus(connection: String, database: String, print: String) {
        connectionStatus.text = connection
        dbStatus.text = database
        if (print.isNotEmpty()) printStatus.text = print
    }

    private fun onScreenChanged(state: UiState) {
        pageFlipper.displayedChild = state.screen.ordinal
        val (title, subtitle) = when (state.screen) {
            Screen.MONITOR -> "얼굴 인식" to "카메라를 보면 자동으로 인식합니다."
            Screen.REGISTER_CAPTURE -> "얼굴 등록" to "좌·우·위·아래 방향 샘플을 모읍니다."
            Screen.REGISTER_INFO -> (if (state.labelOnly) "이름표 출력" else "이름 입력") to ""
            Screen.OUTPUT -> "이름표 출력" to ""
        }
        headerTitle.text = title
        headerSubtitle.text = subtitle
        backButton.visibility = if (state.screen == Screen.MONITOR) View.GONE else View.VISIBLE
        if (state.screen == Screen.REGISTER_INFO) {
            listOf(nameInput, departmentInput, gradeInput).forEach { it.setText("") }
            submitButton.text = if (state.labelOnly) "이름표 출력하기" else "저장하기"
        }
    }

    private fun renderMonitor(state: UiState) {
        monitorMessage.text = state.message
        monitorDetail.text = state.detail
        registerButton.visibility = if (state.showRegisterChoice) View.VISIBLE else View.GONE
        labelOnlyButton.visibility = if (state.showRegisterChoice) View.VISIBLE else View.GONE
        printButton.visibility = if (state.printButtonVisible) View.VISIBLE else View.GONE
        printButton.isEnabled = state.printButtonEnabled
        candidates.removeAllViews()
        if (state.candidates.size > 1) {
            state.candidates.forEachIndexed { i, c ->
                candidates.addView(Button(candidates.context).apply {
                    text = String.format(Locale.US, "%s%d. %s", if (c.selected) "✓ " else "", i + 1, c.label)
                    isAllCaps = false
                    textSize = 18f
                    setOnClickListener { actions.selectCandidate(c.personKey) }
                })
            }
        }
    }

    private fun renderCapture(state: UiState) {
        registerMessage.text = state.message
        registerDetail.text = state.detail
        state.registration?.let { r ->
            registerProgress.max = r.target
            registerProgress.progress = r.total
            registerCount.text = progressText(r)
        }
        val duplicate = state.duplicateOf != null
        duplicateIsMe.visibility = if (duplicate) View.VISIBLE else View.GONE
        duplicateNotMe.visibility = if (duplicate) View.VISIBLE else View.GONE
    }

    private fun renderInfo(state: UiState, savingText: String) {
        infoTitle.text = state.message
        infoDetail.text = if (state.saving) savingText else state.detail
        submitButton.isEnabled = !state.saving
    }

    private fun renderOutput(state: UiState) {
        val output = state.output ?: return
        outputName.text = output.metadata.name
        outputDepartment.text = output.metadata.department
        outputGrade.text = output.metadata.grade
        outputStatus.text = "${output.printStatus}\n${state.message}"
    }

    private fun progressText(r: RegistrationView): CharSequence {
        val builder = SpannableStringBuilder()
        append(builder, "전체 ${r.total} / ${r.target}", Color.rgb(17, 17, 17))
        HeadDirection.entries.forEach { direction ->
            val count = (r.collected[direction] ?: 0).coerceAtMost(r.perDirection)
            builder.append('\n')
            when {
                count >= r.perDirection -> append(builder, "✓ ${direction.progressLabel} $count/${r.perDirection} 완료", Color.rgb(30, 132, 73))
                direction == r.next -> append(builder, "● ${direction.progressLabel} $count/${r.perDirection}", Color.rgb(245, 166, 35))
                else -> append(builder, "○ ${direction.progressLabel} $count/${r.perDirection} 대기", Color.rgb(142, 142, 147))
            }
        }
        return builder
    }

    private fun append(builder: SpannableStringBuilder, text: String, color: Int) {
        builder.append(SpannableString(text).apply { setSpan(ForegroundColorSpan(color), 0, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE) })
    }

    private fun move(view: View, target: ViewGroup, index: Int, params: ViewGroup.LayoutParams) {
        if (view.parent === target) return
        (view.parent as? ViewGroup)?.removeView(view)
        target.addView(view, index.coerceAtMost(target.childCount), params)
    }
}
