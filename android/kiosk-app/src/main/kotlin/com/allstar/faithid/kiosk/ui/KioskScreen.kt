package com.allstar.faithid.kiosk.ui

import android.app.Activity
import android.graphics.Color
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.ViewFlipper
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.R
import com.allstar.faithid.kiosk.flow.RegistrationView
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState
import java.util.Locale

/**
 * Renders [UiState] into the views of activity_main.xml, in either the developer layout or the
 * full-screen operation layout. Holds no state of its own beyond animations.
 */
class KioskScreen(activity: Activity, private val actions: Actions) {

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
    private val operationPageHost: FrameLayout = activity.v(R.id.operationPageHost)
    private val operationActionPanel: View = activity.v(R.id.operationActionPanel)
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

    private val operationMessage: TextView = activity.v(R.id.operationMessageText)
    private val operationDetail: TextView = activity.v(R.id.operationDetailText)
    private val operationPrintButton: Button = activity.v(R.id.operationPrintButton)
    private val motionHint: TextView = activity.v(R.id.operationMotionHint)
    private val banner: TextView = activity.v(R.id.operationBanner)

    private var developerMode = true
    private var lastScreen: Screen? = null
    private var hintDirection: HeadDirection? = null
    private var pulsing = false

    init {
        registerButton.setOnClickListener { actions.startRegistration() }
        labelOnlyButton.setOnClickListener { actions.startLabelOnly() }
        printButton.setOnClickListener { actions.print() }
        operationPrintButton.setOnClickListener { actions.print() }
        duplicateIsMe.setOnClickListener { actions.duplicateIsMe() }
        duplicateNotMe.setOnClickListener { actions.duplicateNotMe() }
        submitButton.setOnClickListener {
            actions.submitInfo(nameInput.text.toString(), departmentInput.text.toString(), gradeInput.text.toString())
        }
        backButton.setOnClickListener { actions.home() }
        activity.v<Button>(R.id.operationHomeButton).setOnClickListener { actions.home() }
        activity.v<Button>(R.id.operationSettingsButton).setOnClickListener { actions.openSettings() }
        activity.v<Button>(R.id.settingsButton).setOnClickListener { actions.openSettings() }
        activity.v<Button>(R.id.logButton).setOnClickListener { actions.openLog() }
        activity.v<Button>(R.id.syncNowButton).setOnClickListener { actions.syncNow() }
    }

    fun setDeveloperMode(enabled: Boolean, printButtonText: String) {
        developerMode = enabled
        printButton.text = printButtonText
        operationPrintButton.text = printButtonText
        developerRoot.visibility = if (enabled) View.VISIBLE else View.GONE
        operationRoot.visibility = if (enabled) View.GONE else View.VISIBLE
        if (enabled) {
            move(cameraPanel, (developerPageHost.parent as ViewGroup), 0, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f).apply { marginEnd = 20 })
            cameraPanel.setPadding(24, 24, 24, 24)
            move(pageFlipper, developerPageHost, 0, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        } else {
            move(cameraPanel, operationCameraHost, 0, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            cameraPanel.setPadding(0, 0, 0, 0)
            move(pageFlipper, operationPageHost, 0, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        lastScreen = null
    }

    fun render(state: UiState, savingText: String) {
        if (state.screen != lastScreen) onScreenChanged(state)
        lastScreen = state.screen
        when (state.screen) {
            Screen.MONITOR -> renderMonitor(state)
            Screen.REGISTER_CAPTURE -> renderCapture(state)
            Screen.REGISTER_INFO -> renderInfo(state, savingText)
            Screen.OUTPUT -> renderOutput(state)
        }
        renderOperationOverlay(state)
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

    private fun renderOperationOverlay(state: UiState) {
        banner.text = state.banner
        banner.visibility = if (!developerMode && state.banner.isNotEmpty()) View.VISIBLE else View.GONE
        if (developerMode) {
            listOf(monitorMessage, monitorDetail, registerMessage, registerDetail).forEach { it.visibility = View.VISIBLE }
            stopPulse()
            showHint(null)
            return
        }
        val (message, detail) = when (state.screen) {
            Screen.MONITOR -> state.message to state.detail
            Screen.REGISTER_CAPTURE -> state.message to state.detail
            Screen.REGISTER_INFO -> "" to ""
            Screen.OUTPUT -> (state.output?.printStatus ?: "") to state.message
        }
        operationMessage.text = message
        operationDetail.text = detail
        operationMessage.visibility = if (message.isEmpty()) View.GONE else View.VISIBLE
        operationDetail.visibility = if (detail.isEmpty()) View.GONE else View.VISIBLE

        // The floating button replaces the panel when exactly one person is recognized.
        val floatingPrint = state.screen == Screen.MONITOR && state.printButtonEnabled && state.candidates.size <= 1
        val panelNeeded = when (state.screen) {
            Screen.MONITOR -> state.showRegisterChoice || state.candidates.size > 1
            Screen.REGISTER_CAPTURE -> state.duplicateOf != null
            Screen.REGISTER_INFO, Screen.OUTPUT -> true
        }
        operationActionPanel.visibility = if (panelNeeded && !floatingPrint) View.VISIBLE else View.GONE
        monitorMessage.visibility = if (state.screen == Screen.MONITOR) View.GONE else View.VISIBLE
        monitorDetail.visibility = monitorMessage.visibility
        registerMessage.visibility = View.GONE
        registerDetail.visibility = View.GONE
        if (floatingPrint) startPulse() else stopPulse()
        showHint(if (state.screen == Screen.REGISTER_CAPTURE && state.duplicateOf == null) state.registration?.next else null)
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

    private fun showHint(direction: HeadDirection?) {
        if (direction == hintDirection) return
        hintDirection = direction
        motionHint.animate().cancel()
        motionHint.translationX = 0f
        motionHint.translationY = 0f
        if (direction == null) {
            motionHint.visibility = View.GONE
            return
        }
        motionHint.text = direction.arrow
        motionHint.visibility = View.VISIBLE
        animateHint(direction)
    }

    private fun animateHint(direction: HeadDirection) {
        if (hintDirection != direction) return
        val distance = 90f
        val dx = when (direction) { HeadDirection.LEFT -> -distance; HeadDirection.RIGHT -> distance; else -> 0f }
        val dy = when (direction) { HeadDirection.UP -> -distance; HeadDirection.DOWN -> distance; else -> 0f }
        motionHint.animate().translationX(dx).translationY(dy).alpha(0.45f).setDuration(560).setInterpolator(AccelerateDecelerateInterpolator())
            .withEndAction {
                motionHint.animate().translationX(0f).translationY(0f).alpha(0.9f).setDuration(560)
                    .withEndAction { animateHint(direction) }.start()
            }.start()
    }

    private fun startPulse() {
        operationPrintButton.visibility = View.VISIBLE
        if (pulsing) return
        pulsing = true
        pulse()
    }

    private fun pulse() {
        if (!pulsing) return
        operationPrintButton.animate().scaleX(1.06f).scaleY(1.06f).setDuration(520)
            .withEndAction { operationPrintButton.animate().scaleX(1f).scaleY(1f).setDuration(520).withEndAction { pulse() }.start() }
            .start()
    }

    private fun stopPulse() {
        pulsing = false
        operationPrintButton.animate().cancel()
        operationPrintButton.scaleX = 1f
        operationPrintButton.scaleY = 1f
        operationPrintButton.visibility = View.GONE
    }

    private fun move(view: View, target: ViewGroup, index: Int, params: ViewGroup.LayoutParams) {
        if (view.parent === target) return
        (view.parent as? ViewGroup)?.removeView(view)
        target.addView(view, index.coerceAtMost(target.childCount), params)
    }
}
