package com.allstar.faithid.kiosk.ui

import android.annotation.SuppressLint
import android.app.Activity
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.res.ResourcesCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.core.widget.doAfterTextChanged
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.PersonNames
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.core.recognition.RegistrationCell
import com.allstar.faithid.kiosk.R
import com.allstar.faithid.kiosk.camera.GuideOverlayView
import com.allstar.faithid.kiosk.flow.CaptureProblem
import com.allstar.faithid.kiosk.flow.RegistrationView
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState
import com.allstar.faithid.kiosk.settings.GuideAudience
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.ui.guide.CelebrationView
import com.allstar.faithid.kiosk.ui.guide.GhostFaceView
import com.allstar.faithid.kiosk.ui.guide.GuideAvatarView
import com.allstar.faithid.kiosk.ui.guide.GuideGestures
import com.allstar.faithid.kiosk.ui.guide.GuideMoment
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.allstar.faithid.androidsupport.R as SupportR

/**
 * The operation screen (따뜻한 안내자): the camera fills the screen, 빛이 talks from the top and the
 * replies float at the bottom. Registration shows a white face over the camera that turns and grows
 * to what to copy, 빛이 turning the same way, and the star count. Draws [UiState]; holds only
 * presentation state.
 */
class OperationScreen(
    private val activity: Activity,
    private val actions: KioskScreen.Actions,
    /** Everyone on this kiosk, for the department and grade chips of the name form. */
    private val people: () -> List<PersonMetadata>,
) {
    private fun <T : View> v(id: Int): T = activity.findViewById(id)

    private val density = activity.resources.displayMetrics.density
    private val main = Handler(Looper.getMainLooper())
    private val jua: Typeface? = runCatching { ResourcesCompat.getFont(activity, SupportR.font.jua) }.getOrNull()

    private val root: View = v(R.id.operationRoot)
    private val overlay: View = v(R.id.operationOverlay)
    private val guideOverlay: GuideOverlayView = v(R.id.guideOverlay)
    private val ghost: GhostFaceView = v(R.id.opGhost)
    private val celebration: CelebrationView = v(R.id.opCelebration)
    private val guide: View = v(R.id.opGuide)
    private val title: TextView = v(R.id.opTitle)
    private val sub: TextView = v(R.id.opSub)
    private val banner: TextView = v(R.id.opBanner)
    private val replies: LinearLayout = v(R.id.opReplies)
    private val privacy: TextView = v(R.id.opPrivacy)
    private val count: TextView = v(R.id.opCount)
    private val home: ImageButton = v(R.id.opHomeButton)
    private val gear: ImageButton = v(R.id.opSettingsButton)
    private val avatar: GuideAvatarView = v(R.id.opAvatar)
    private val formAvatar: GuideAvatarView = v(R.id.opFormAvatar)
    private val outputAvatar: GuideAvatarView = v(R.id.opOutputAvatar)

    private val form: View = v(R.id.opForm)
    private val formTitle: TextView = v(R.id.opFormTitle)
    private val formSub: TextView = v(R.id.opFormSub)
    private val familyNameInput: EditText = v(R.id.opFamilyNameInput)
    private val nameInput: EditText = v(R.id.opNameInput)
    private val departmentChips: ChipGroup = v(R.id.opDepartmentChips)
    private val gradeLabel: TextView = v(R.id.opGradeLabel)
    private val familyNameLabel: TextView = v(R.id.opFamilyNameLabel)
    private val nameLabel: TextView = v(R.id.opNameLabel)
    private val departmentLabel: TextView = v(R.id.opDepartmentLabel)
    private val gradeChips: ChipGroup = v(R.id.opGradeChips)
    private val departmentInput: EditText = v(R.id.opDepartmentInput)
    private val gradeInput: EditText = v(R.id.opGradeInput)
    private val submit: Button = v(R.id.opSubmitButton)

    private val output: View = v(R.id.opOutput)
    private val tag: View = v(R.id.opTag)
    private val tagStripe: View = v(R.id.opTagStripe)
    private val tagName: TextView = v(R.id.opTagName)
    private val tagSub: TextView = v(R.id.opTagSub)
    private val outputTitle: TextView = v(R.id.opOutputTitle)
    private val outputSub: TextView = v(R.id.opOutputSub)

    private var kids = true
    private var copy = GuideCopy(kids = true)
    private var labelFamilyName = true
    private var labelGivenName = true

    /** Taps on 빛이 in a row; the seventh shows the look-alikes. */
    private var taps = 0
    private var lastTapAt = 0L
    private var bigScale = 1.18f
    private var smallScale = 0.73f

    /** The last state drawn; while the celebration plays, newer states wait in [pending]. */
    private var shown: UiState? = null
    private var pending: UiState? = null
    private var registered: PersonMetadata? = null
    private var registeredUntil = 0L
    private var greetedKey: String? = null
    private var chipsFromCode = false

    /** Size of the face circle the camera analysis should use (registration steps). */
    @Volatile
    var guideScale: Float = 1f
        private set

    /** The last moment 빛이 reacted to, so each one plays once. */
    private var moment: GuideMoment? = null

    /** Now and then, while nobody is in front of the kiosk, 빛이 does something small. */
    private val idleGesture = object : Runnable {
        override fun run() {
            val s = shown
            if (s != null && s.screen == Screen.MONITOR && s.ready && s.candidates.isEmpty() && !s.showRegisterChoice && !celebration.isPlaying) {
                react(GuideMoment.IDLE, force = true)
            }
            main.postDelayed(this, kotlin.random.Random.nextLong(GuideGestures.IDLE_GESTURE_MIN_MS, GuideGestures.IDLE_GESTURE_MAX_MS))
        }
    }

    private val openSettings = Runnable {
        gearIdle()
        actions.openSettings()
    }

    init {
        listOf(title, formTitle, outputTitle, tagName).forEach { it.typeface = jua }
        celebration.typeface = jua
        home.setOnClickListener { actions.home() }
        v<Button>(R.id.opOutputHome).setOnClickListener { actions.home() }
        submit.setOnClickListener {
            actions.submitInfo(familyNameInput.text.toString(), nameInput.text.toString(), departmentInput.text.toString(), gradeInput.text.toString())
        }
        avatar.setOnClickListener { tapGuide() }
        setUpGear()
        departmentInput.doAfterTextChanged { if (!chipsFromCode) selectChip(departmentChips, it.toString()) }
        gradeInput.doAfterTextChanged { if (!chipsFromCode) selectChip(gradeChips, it.toString()) }
        // Android 15 draws behind the keyboard; keep the form (and its save button) above it.
        ViewCompat.setOnApplyWindowInsetsListener(form) { view, insets ->
            view.updatePadding(bottom = insets.getInsets(WindowInsetsCompat.Type.ime()).bottom)
            insets
        }
        gradeInput.setOnEditorActionListener { _, action, _ ->
            (action == EditorInfo.IME_ACTION_DONE).also { if (it) submit.performClick() }
        }
        // Let 빛이 hop above its own row.
        listOf(avatar, formAvatar, outputAvatar).forEach { a ->
            (a.parent as? ViewGroup)?.let { it.clipChildren = false; (it.parent as? ViewGroup)?.clipChildren = false }
        }
        main.postDelayed(idleGesture, GuideGestures.IDLE_GESTURE_MIN_MS)
        guideOverlay.onOvalChanged = { placeAroundOval() }
        root.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> placeAroundOval() }
    }

    fun setColumnLabels(names: FormLabels) {
        familyNameLabel.text = names.familyName
        nameLabel.text = names.name
        departmentLabel.text = names.department
        gradeLabel.text = names.grade
        departmentInput.hint = "${names.department} 직접 입력"
        gradeInput.hint = "${names.grade} 직접 입력"
    }

    fun configure(settings: KioskSettings) {
        kids = settings.guideAudience != GuideAudience.ADULTS
        copy = GuideCopy(kids, settings.guideLines, settings.guideName, settings.callWithFamilyName)
        labelFamilyName = settings.labelFamilyName
        labelGivenName = settings.labelGivenName
        bigScale = settings.registerBigCircleScale
        smallScale = settings.registerSmallCircleScale
        guideOverlay.axisXRatio = settings.registerEllipseAxisXRatio
        guideOverlay.axisYRatio = settings.registerEllipseAxisYRatio
        guideOverlay.centerYOffsetRatio = settings.registerEllipseCenterYOffsetRatio
        guideOverlay.outsideDimAlpha = settings.registerEllipseOutsideDimAlpha
        ghost.bigScale = bigScale
        ghost.smallScale = smallScale
        val titleFace = if (kids) jua else Typeface.DEFAULT_BOLD
        listOf(title, formTitle, outputTitle).forEach { it.typeface = titleFace }
        shown = null
        placeAroundOval()
    }

    /** Called when the operation screen is hidden (developer view). */
    fun reset() {
        celebration.stop()
        ghost.hide()
        main.removeCallbacks(captureTick)
        pending = null
        shown = null
        guideOverlay.animateScale(1f)
        guideScale = 1f
    }

    fun render(state: UiState) {
        if (state.registered != null) {
            registered = state.registered
            registeredUntil = SystemClock.uptimeMillis() + REGISTERED_NOTICE_MS
            main.postDelayed({ shown?.let { if (!celebration.isPlaying && it.screen == Screen.MONITOR) draw(it) } }, REGISTERED_NOTICE_MS + 100)
        }
        if (celebration.isPlaying && pending != null) {
            pending = state
            return
        }
        val before = shown
        val finishedCapture = before?.screen == Screen.REGISTER_CAPTURE && state.screen == Screen.REGISTER_INFO && !state.labelOnly
        if (finishedCapture) {
            celebrate(before!!, state)
            return
        }
        draw(state)
    }

    /** Plays the celebration over the finished registration, then shows what came next. */
    private fun celebrate(capture: UiState, next: UiState) {
        pending = next
        ghost.collected()
        ghost.hide()
        lookStraight()
        title.text = copy.celebration(capture.registration?.target ?: 0)
        sub.visibility = View.GONE
        count.text = capture.registration?.let { copy.count(it.target, it.target) } ?: ""
        guideOverlay.animateScale(1f)
        react(GuideMoment.REGISTER_DONE)
        val oval = ovalIn(celebration, 1f)
        celebration.celebrate(oval[0], oval[1], kids, copy.celebration(capture.registration?.target ?: 0)) {
            val waiting = pending ?: next
            pending = null
            draw(waiting)
        }
    }

    // ----- drawing --------------------------------------------------------------------------

    private fun draw(state: UiState) {
        val previous = shown
        shown = state
        val screen = state.screen
        if (previous?.screen == Screen.REGISTER_INFO && screen != Screen.REGISTER_INFO) {
            hideKeyboard()
            listOf(familyNameInput, nameInput, departmentInput, gradeInput).forEach { it.clearFocus() }
        }
        val camera = screen == Screen.MONITOR || screen == Screen.REGISTER_CAPTURE
        val capture = screen == Screen.REGISTER_CAPTURE

        guide.visibility = if (camera) View.VISIBLE else View.GONE
        form.visibility = if (screen == Screen.REGISTER_INFO) View.VISIBLE else View.GONE
        output.visibility = if (screen == Screen.OUTPUT) View.VISIBLE else View.GONE
        home.visibility = if (screen == Screen.OUTPUT) View.GONE else View.VISIBLE
        gear.visibility = if (camera) View.VISIBLE else View.GONE
        privacy.visibility = if (camera && !capture) View.VISIBLE else View.GONE
        privacy.text = copy.privacy
        count.visibility = if (capture) View.VISIBLE else View.GONE
        if (!capture) {
            ghost.hide()
            lookStraight()
            main.removeCallbacks(captureTick)
            lastTotal = -1
        }
        // The look-alikes' three lines need the room under the bubble.
        banner.visibility = if (camera && state.banner.isNotEmpty() && state.lookAlike == null) View.VISIBLE else View.GONE
        banner.text = "● ${state.banner}"
        banner.setTextColor(Color.parseColor("#33263F"))

        momentFor(previous, state)?.let { react(it) }

        val found = screen == Screen.REGISTER_CAPTURE || state.showRegisterChoice || state.candidates.isNotEmpty()
        // While registering, the white face shows the target; the circle stays only as a faint line.
        guideOverlay.outlineColor = if (capture) FAINT_WHITE else if (found) APRICOT else Color.WHITE
        val scale = if (capture) circleScale(state.registration?.let { it.held?.circle ?: it.circle }) else 1f
        guideScale = scale
        guideOverlay.animateScale(scale)

        when (screen) {
            Screen.MONITOR -> drawMonitor(state)
            Screen.REGISTER_CAPTURE -> drawCapture(state)
            Screen.REGISTER_INFO -> drawForm(state, entering = previous?.screen != Screen.REGISTER_INFO)
            Screen.OUTPUT -> drawOutput(state, entering = previous?.screen != Screen.OUTPUT)
        }
    }

    private fun drawMonitor(state: UiState) {
        val idle = state.ready && state.candidates.isEmpty() && !state.showRegisterChoice
        val notice = registered?.takeIf { idle && SystemClock.uptimeMillis() < registeredUntil }
        setBubble(copy.bubble(if (notice != null) state.copy(registered = notice) else state))
        replies.removeAllViews()
        when {
            !state.ready -> Unit
            state.showRegisterChoice -> {
                addReply(copy.register, main = true, icon = R.drawable.ic_face) { actions.startRegistration() }
                addReply(copy.labelOnly, icon = R.drawable.ic_tag) { actions.startLabelOnly() }
            }
            state.candidates.size > 1 -> {
                val selected = state.candidates.firstOrNull { it.selected }
                if (selected != null) addReply(copy.print, main = true, icon = R.drawable.ic_print) { actions.print() }
                state.candidates.filter { !it.selected }.forEach { c ->
                    addReply(candidateLabel(c.metadata?.name ?: c.label, c.metadata)) { actions.selectCandidate(c.personKey) }
                }
                if (selected == null) addReply(copy.noneOfThem(state.candidates.size), small = true) { actions.notMe() }
            }
            state.candidates.size == 1 && state.printButtonEnabled -> {
                addReply(copy.print, main = true, icon = R.drawable.ic_print) { actions.print() }
                addReply(copy.notMe, small = true) { actions.notMe() }
            }
        }
        val greeted = state.candidates.singleOrNull()?.personKey ?: state.candidates.firstOrNull { it.selected }?.personKey
        if (greeted != null && greeted != greetedKey && kids) celebration.confetti()
        greetedKey = greeted
    }

    /** "그것도 좋아!" stays a moment after a sample other than the suggested one was kept. */
    private var bonusUntil = 0L

    private var lastTotal = -1

    /** The last time a sample was held or kept; after a while without, the white face shows the turn. */
    private var progressAt = 0L

    /** Since when the distance has been wrong; after a while, footprints show the step. */
    private var sizeOffSince = 0L

    /** Redraws the registration now and then, so the turn demo and the footprints start on time. */
    private val captureTick = object : Runnable {
        override fun run() {
            val s = shown
            if (s == null || s.screen != Screen.REGISTER_CAPTURE || celebration.isPlaying) return
            drawCapture(s)
            main.postDelayed(this, CAPTURE_TICK_MS)
        }
    }

    private fun drawCapture(state: UiState) {
        val r = state.registration ?: return
        val now = SystemClock.uptimeMillis()
        if (lastTotal < 0) {
            progressAt = now
            main.removeCallbacks(captureTick)
            main.postDelayed(captureTick, CAPTURE_TICK_MS)
        }
        if (r.bonus) bonusUntil = now + BONUS_MS
        setBubble(if (now < bonusUntil && r.next != null) copy.bonus() else copy.bubble(state))
        replies.removeAllViews()
        count.text = copy.count(r.total, r.target)
        if (lastTotal in 0 until r.total) ghost.collected()
        if (r.total != lastTotal || r.holding > 0f) progressAt = now
        lastTotal = r.total

        // The face shows the cell being held, else the suggested one.
        val cell = r.held ?: r.next?.let { RegistrationCell(it, r.circle ?: FaceCircle.NORMAL) } ?: return ghost.hide()
        val sizeProblem = r.problem == CaptureProblem.COME_CLOSER || r.problem == CaptureProblem.STEP_BACK
        if (!sizeProblem) sizeOffSince = 0L else if (sizeOffSince == 0L) sizeOffSince = now
        val steps = when {
            !sizeProblem || now - sizeOffSince < HINT_AFTER_MS -> 0
            r.problem == CaptureProblem.COME_CLOSER -> 1
            else -> -1
        }
        val stuck = r.held == null && now - progressAt > HINT_AFTER_MS &&
            (r.problem == null || r.problem == CaptureProblem.DIRECTION || r.problem == CaptureProblem.TOO_SIMILAR)
        val hint = if (sizeProblem) r.faceSize?.let { it / circleSize(cell.circle) } else null
        ghost.show(cell.direction, cell.circle, r.holding, stuck, hint, steps, dim = r.problem == CaptureProblem.SEVERAL)
        // 빛이 turns the same way and grows or shrinks with the circle.
        if (avatar.maxScale != AVATAR_MAX) {
            avatar.maxScale = AVATAR_MAX
            avatar.layoutParams = avatar.layoutParams.apply { width = (AVATAR_DP * AVATAR_MAX * density).toInt(); height = width }
        }
        avatar.turnTo(yawOf(cell.direction), pitchOf(cell.direction), circleSize(cell.circle))
    }

    private fun circleSize(circle: FaceCircle) = when (circle) {
        FaceCircle.BIG -> bigScale
        FaceCircle.SMALL -> smallScale
        else -> 1f
    }

    private fun yawOf(d: HeadDirection) = when (d) {
        HeadDirection.LEFT -> -AVATAR_TURN
        HeadDirection.RIGHT -> AVATAR_TURN
        else -> 0f
    }

    private fun pitchOf(d: HeadDirection) = when (d) {
        HeadDirection.UP -> -AVATAR_NOD
        HeadDirection.DOWN -> AVATAR_NOD
        else -> 0f
    }

    /** 빛이 back to its usual size, looking straight ahead. */
    private fun lookStraight() {
        avatar.turnTo(0f, 0f, 1f)
        if (avatar.maxScale != 1f) {
            avatar.maxScale = 1f
            avatar.layoutParams = avatar.layoutParams.apply { width = (AVATAR_DP * density).toInt(); height = width }
        }
    }

    /** Seven quick taps on 빛이 while it waits: who looks most like the face in front of the camera. */
    private fun tapGuide() {
        val s = shown ?: return
        if (s.screen != Screen.MONITOR || !s.ready) return
        val now = SystemClock.uptimeMillis()
        taps = if (now - lastTapAt <= TAP_GAP_MS) taps + 1 else 1
        lastTapAt = now
        if (taps < SECRET_TAPS) return
        taps = 0
        actions.lookAlikes()
    }

    private fun drawForm(state: UiState, entering: Boolean) {
        val line = copy.bubble(state)
        formTitle.text = line.title
        formSub.text = line.sub
        formSub.setTextColor(if (state.detail.isNotBlank() && !state.saving) Color.parseColor("#C0461B") else Color.parseColor("#6E6078"))
        submit.text = if (state.labelOnly) copy.print else copy.save
        submit.setCompoundDrawablesRelativeWithIntrinsicBounds(if (state.labelOnly) R.drawable.ic_print else R.drawable.ic_check, 0, 0, 0)
        submit.typeface = if (kids) jua else Typeface.DEFAULT_BOLD
        submit.isEnabled = !state.saving
        familyNameInput.hint = if (state.labelOnly) "예: 이" else "예: 김"
        nameInput.hint = if (state.labelOnly) "예: 준호" else "예: 하은"
        if (entering) {
            chipsFromCode = true
            listOf(familyNameInput, nameInput, departmentInput, gradeInput).forEach { it.setText("") }
            chipsFromCode = false
            buildDepartmentChips()
            showGrades(null)
        }
    }

    private fun drawOutput(state: UiState, entering: Boolean) {
        val out = state.output ?: return
        val line = copy.bubble(state)
        outputTitle.text = line.title
        outputSub.text = line.sub
        if (!entering) return
        val m = out.metadata
        tagName.text = PersonNames.display(m, labelFamilyName, labelGivenName)
        tagName.typeface = jua
        tagSub.text = listOf(m.department, m.grade).filter { it.isNotBlank() }.joinToString(" · ")
        tagStripe.setBackgroundColor(if (m.department.isBlank()) Color.parseColor("#C9BCD2") else DepartmentColors.of(m.department))
        tag.translationY = -260 * density
        tag.animate().translationY(0f).setDuration(1500).setInterpolator(android.view.animation.DecelerateInterpolator()).start()
        if (kids) celebration.confetti()
    }

    /** What just happened, in 빛이's terms (see [GuideGestures.TABLE]). */
    private fun momentFor(previous: UiState?, state: UiState): GuideMoment? {
        val sameScreen = previous?.screen == state.screen
        return when (state.screen) {
            Screen.MONITOR -> when {
                !state.ready -> GuideMoment.OFFLINE
                state.lookAlike != null -> GuideMoment.SECRET
                state.retrying -> GuideMoment.RETRY
                state.registered != null -> GuideMoment.SAVED
                state.showRegisterChoice -> GuideMoment.UNKNOWN_FACE
                state.candidates.size > 1 && state.candidates.none { it.selected } -> GuideMoment.LOOK_ALIKE
                state.candidates.isNotEmpty() -> GuideMoment.RECOGNIZED
                else -> {
                    moment = null
                    null
                }
            }
            Screen.REGISTER_CAPTURE -> {
                val r = state.registration
                val before = previous?.registration
                when {
                    !sameScreen -> GuideMoment.REGISTER_START
                    r != null && before != null && r.total > before.total -> GuideMoment.STAR_COLLECTED
                    r?.problem == CaptureProblem.SEVERAL -> GuideMoment.SEVERAL_FACES
                    r?.stalled == true -> GuideMoment.STALLED
                    else -> moment
                }
            }
            Screen.REGISTER_INFO -> GuideMoment.NAME_FORM
            Screen.OUTPUT -> GuideMoment.NAME_TAG
        }
    }

    /** Plays the gestures for [next] once (again only after the moment changed, unless [force]). */
    private fun react(next: GuideMoment, force: Boolean = false) {
        // A collected star is worth a nod every time; other moments play once.
        if (!force && next == moment && next != GuideMoment.STAR_COLLECTED) return
        moment = next
        val who = when {
            output.visibility == View.VISIBLE -> outputAvatar
            form.visibility == View.VISIBLE -> formAvatar
            else -> avatar
        }
        who.play(GuideGestures.gesturesFor(next))
    }

    private fun setBubble(line: GuideCopy.Line) {
        title.text = line.title
        sub.text = line.sub
        sub.visibility = if (line.sub.isBlank()) View.GONE else View.VISIBLE
        title.textSize = if (kids) 30f else 26f
    }

    // ----- replies --------------------------------------------------------------------------

    private fun addReply(text: CharSequence, main: Boolean = false, small: Boolean = false, icon: Int = 0, onClick: () -> Unit) {
        val button = Button(activity).apply {
            this.text = text
            isAllCaps = false
            stateListAnimator = null
            backgroundTintList = null
            setBackgroundResource(if (main) R.drawable.bg_reply_main else R.drawable.bg_reply)
            setTextColor(if (main) Color.WHITE else Color.parseColor("#33263F"))
            typeface = if (kids) jua else Typeface.DEFAULT_BOLD
            textSize = if (small) 22f else if (kids) 32f else 28f
            gravity = if (small) Gravity.CENTER else Gravity.CENTER_VERTICAL or Gravity.START
            setPadding((28 * density).toInt(), 0, (28 * density).toInt(), 0)
            if (icon != 0) {
                setCompoundDrawablesRelativeWithIntrinsicBounds(icon, 0, 0, 0)
                compoundDrawablePadding = (16 * density).toInt()
                compoundDrawableTintList = if (main) ColorStateList.valueOf(Color.WHITE) else null
            }
            setOnClickListener { onClick() }
        }
        val height = ((if (small) 68 else 100) * density).toInt()
        val params = if (small) LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, height)
        else LinearLayout.LayoutParams(0, height, 1f)
        if (replies.childCount > 0) params.marginStart = (16 * density).toInt()
        replies.addView(button, params)
    }

    private fun candidateLabel(name: String, metadata: PersonMetadata?): CharSequence {
        val where = metadata?.let { listOf(it.department, it.grade).filter { s -> s.isNotBlank() }.joinToString(" ") }.orEmpty()
        if (where.isEmpty()) return name
        return SpannableStringBuilder(name).apply {
            val start = length
            append("  ").append(where)
            setSpan(RelativeSizeSpan(0.62f), start, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            setSpan(ForegroundColorSpan(Color.parseColor("#806E6078")), start, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
    }

    // ----- department and grade chips -------------------------------------------------------

    private fun buildDepartmentChips() {
        departmentChips.removeAllViews()
        DepartmentChoices.departments(people()).forEach { department ->
            departmentChips.addView(chip(department, DepartmentColors.of(department)) { checked ->
                if (!checked) return@chip
                // A chip the person tapped ends typing; one selected because the typed text matches does not.
                if (!chipsFromCode) {
                    hideKeyboard()
                    setFromChip(departmentInput, department)
                    setFromChip(gradeInput, "")
                }
                showGrades(department)
            })
        }
        departmentChips.visibility = if (departmentChips.childCount == 0) View.GONE else View.VISIBLE
    }

    private fun showGrades(department: String?) {
        gradeChips.removeAllViews()
        val grades = department?.let { DepartmentChoices.grades(people(), it) }.orEmpty()
        grades.forEach { grade ->
            gradeChips.addView(chip(grade, null) { checked ->
                if (checked && !chipsFromCode) {
                    hideKeyboard()
                    setFromChip(gradeInput, grade)
                }
            })
        }
        val visible = if (grades.isEmpty()) View.GONE else View.VISIBLE
        gradeChips.visibility = visible
        gradeLabel.visibility = visible
    }

    private fun chip(text: String, dot: Int?, onChecked: (Boolean) -> Unit): Chip = Chip(activity).apply {
        this.text = text
        isCheckable = true
        isCheckedIconVisible = false
        textSize = 20f
        typeface = Typeface.DEFAULT_BOLD
        chipMinHeight = 52 * density
        chipStrokeWidth = 2 * density
        chipStrokeColor = ColorStateList.valueOf(Color.parseColor("#F0DFD4"))
        val states = arrayOf(intArrayOf(android.R.attr.state_checked), intArrayOf())
        chipBackgroundColor = ColorStateList(states, intArrayOf(Color.parseColor("#5B3F8C"), Color.WHITE))
        setTextColor(ColorStateList(states, intArrayOf(Color.WHITE, Color.parseColor("#33263F"))))
        if (dot != null) {
            chipIcon = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(dot) }
            chipIconSize = 16 * density
            isChipIconVisible = true
        }
        setOnCheckedChangeListener { _, checked -> onChecked(checked) }
    }

    /** Choosing a chip means typing is done; the keyboard would otherwise cover the save button. */
    private fun hideKeyboard() {
        activity.getSystemService(android.view.inputmethod.InputMethodManager::class.java)?.hideSoftInputFromWindow(root.windowToken, 0)
    }

    private fun setFromChip(input: EditText, text: String) {
        chipsFromCode = true
        input.setText(text)
        input.setSelection(text.length)
        chipsFromCode = false
    }

    /** Typing a value that matches a chip selects it; anything else clears the selection. */
    private fun selectChip(group: ChipGroup, text: String) {
        val match = (0 until group.childCount).map { group.getChildAt(it) as Chip }.firstOrNull { it.text.toString() == text.trim() }
        chipsFromCode = true
        if (match != null) group.check(match.id) else group.clearCheck()
        chipsFromCode = false
    }

    // ----- layout around the face circle ----------------------------------------------------

    /**
     * The registration circles shrink together (keeping their ratio) when the biggest one and its
     * stars would not fit between the guide's bubble and the bottom edge, e.g. with a large oval
     * someone set for the old screen.
     */
    private var fit = 1f

    /** Size of the normal face circle on screen relative to the configured one; faces are measured against it. */
    val normalScale: Float get() = fit
    private var guideBottom = 0f

    private fun circleScale(circle: FaceCircle?): Float = fit * when (circle) {
        FaceCircle.BIG -> bigScale
        FaceCircle.SMALL -> smallScale
        else -> 1f
    }

    private fun updateFit() {
        val base = guideOverlay.ovalRect(1f)
        if (base.height() <= 0f) return
        val origin = IntArray(2).also { guideOverlay.getLocationInWindow(it) }
        val rootAt = IntArray(2).also { root.getLocationInWindow(it) }
        val centerY = base.centerY() + origin[1] - rootAt[1]
        if (guide.isShown) {
            val g = IntArray(2).also { guide.getLocationInWindow(it) }
            guideBottom = maxOf(guideBottom, (g[1] - rootAt[1] + guide.height).toFloat())
        }
        val radius = base.height() / 2
        val margin = 12 * density
        val reach = RING_REACH_DP * density
        val room = minOf(centerY - guideBottom - margin - reach, root.height - margin - reach - centerY)
        fit = (room / radius / maxOf(bigScale, 1f)).coerceIn(0.5f, 1f)
    }

    /** Centre of the face circle in [target]'s coordinates. */
    private fun ovalIn(target: View, scale: Float): FloatArray {
        val rect = guideOverlay.ovalRect(scale)
        val from = IntArray(2).also { guideOverlay.getLocationInWindow(it) }
        val to = IntArray(2).also { target.getLocationInWindow(it) }
        return floatArrayOf(rect.centerX() + from[0] - to[0], rect.centerY() + from[1] - to[1])
    }

    /** The white face is centred on the normal circle. */
    private fun placeAroundOval() {
        if (guideOverlay.width == 0 || root.width == 0) return
        updateFit()
        val normal = guideOverlay.ovalRect(fit)
        val from = IntArray(2).also { guideOverlay.getLocationInWindow(it) }
        val to = IntArray(2).also { ghost.getLocationInWindow(it) }
        normal.offset((from[0] - to[0]).toFloat(), (from[1] - to[1]).toFloat())
        ghost.normalOval = normal
    }

    // ----- settings gear --------------------------------------------------------------------

    /** Nearly invisible; a 1.2 s press opens the settings, a tap does nothing. */
    @SuppressLint("ClickableViewAccessibility")
    private fun setUpGear() {
        gear.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    view.alpha = 1f
                    view.setBackgroundResource(R.drawable.bg_glass_circle)
                    gear.setColorFilter(Color.parseColor("#33263F"))
                    main.postDelayed(openSettings, SETTINGS_HOLD_MS)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    main.removeCallbacks(openSettings)
                    gearIdle()
                }
            }
            true
        }
        // Accessibility services use a long click instead of a held touch.
        gear.setOnLongClickListener {
            actions.openSettings()
            true
        }
    }

    private fun gearIdle() {
        gear.alpha = 0.16f
        gear.setBackgroundColor(Color.TRANSPARENT)
        gear.clearColorFilter()
    }

    companion object {
        private const val APRICOT = 0xFFFFB183.toInt()
        private const val SETTINGS_HOLD_MS = 1_200L

        /** Taps on 빛이 for the look-alikes, each within this long of the one before. */
        private const val SECRET_TAPS = 7
        private const val TAP_GAP_MS = 700L
        private const val BONUS_MS = 1_700L
        private const val CAPTURE_TICK_MS = 500L

        /** After this long without progress the face shows the turn; with the wrong distance, footprints. */
        private const val HINT_AFTER_MS = 3_000L

        /** Room the hold ring takes around the biggest circle. */
        private const val RING_REACH_DP = 28f
        private const val FAINT_WHITE = 0x66FFFFFF
        private const val AVATAR_DP = 76f
        private const val AVATAR_MAX = 1.25f
        private const val AVATAR_TURN = 0.6f
        private const val AVATAR_NOD = 0.55f
        private const val REGISTERED_NOTICE_MS = 3_000L
    }
}
