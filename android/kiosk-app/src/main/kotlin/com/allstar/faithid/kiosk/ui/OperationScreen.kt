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
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.res.ResourcesCompat
import androidx.core.widget.doAfterTextChanged
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceCircle
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.R
import com.allstar.faithid.kiosk.camera.GuideOverlayView
import com.allstar.faithid.kiosk.flow.Screen
import com.allstar.faithid.kiosk.flow.UiState
import com.allstar.faithid.kiosk.settings.GuideAudience
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.ui.guide.CelebrationView
import com.allstar.faithid.kiosk.ui.guide.HeadView
import com.allstar.faithid.kiosk.ui.guide.StarFieldView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.allstar.faithid.androidsupport.R as SupportR

/**
 * The operation screen (따뜻한 안내자): the camera fills the screen, 빛이 talks from the top and the
 * replies float at the bottom. Registration shows stars where the person looks, the face circle in
 * three sizes and a head that shows the move. Draws [UiState]; holds only presentation state.
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
    private val stars: StarFieldView = v(R.id.opStars)
    private val celebration: CelebrationView = v(R.id.opCelebration)
    private val guide: View = v(R.id.opGuide)
    private val title: TextView = v(R.id.opTitle)
    private val sub: TextView = v(R.id.opSub)
    private val banner: TextView = v(R.id.opBanner)
    private val replies: LinearLayout = v(R.id.opReplies)
    private val privacy: TextView = v(R.id.opPrivacy)
    private val headCard: View = v(R.id.opHeadCard)
    private val head: HeadView = v(R.id.opHead)
    private val headTitle: TextView = v(R.id.opHeadTitle)
    private val headCaption: TextView = v(R.id.opHeadCaption)
    private val count: TextView = v(R.id.opCount)
    private val home: ImageButton = v(R.id.opHomeButton)
    private val gear: ImageButton = v(R.id.opSettingsButton)

    private val form: View = v(R.id.opForm)
    private val formTitle: TextView = v(R.id.opFormTitle)
    private val formSub: TextView = v(R.id.opFormSub)
    private val nameInput: EditText = v(R.id.opNameInput)
    private val departmentChips: ChipGroup = v(R.id.opDepartmentChips)
    private val gradeLabel: View = v(R.id.opGradeLabel)
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

    private val openSettings = Runnable {
        gearIdle()
        actions.openSettings()
    }

    init {
        listOf(title, formTitle, outputTitle, headTitle, tagName).forEach { it.typeface = jua }
        celebration.typeface = jua
        home.setOnClickListener { actions.home() }
        v<Button>(R.id.opOutputHome).setOnClickListener { actions.home() }
        submit.setOnClickListener {
            actions.submitInfo(nameInput.text.toString(), departmentInput.text.toString(), gradeInput.text.toString())
        }
        setUpGear()
        departmentInput.doAfterTextChanged { if (!chipsFromCode) selectChip(departmentChips, it.toString()) }
        gradeInput.doAfterTextChanged { if (!chipsFromCode) selectChip(gradeChips, it.toString()) }
        guideOverlay.onOvalChanged = { placeAroundOval() }
        root.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> placeAroundOval() }
    }

    fun configure(settings: KioskSettings) {
        kids = settings.guideAudience != GuideAudience.ADULTS
        copy = GuideCopy(kids, settings.guideLines)
        bigScale = settings.registerBigCircleScale
        smallScale = settings.registerSmallCircleScale
        guideOverlay.axisXRatio = settings.registerEllipseAxisXRatio
        guideOverlay.axisYRatio = settings.registerEllipseAxisYRatio
        guideOverlay.centerYOffsetRatio = settings.registerEllipseCenterYOffsetRatio
        guideOverlay.outsideDimAlpha = settings.registerEllipseOutsideDimAlpha
        head.kids = kids
        val titleFace = if (kids) jua else Typeface.DEFAULT_BOLD
        listOf(title, formTitle, outputTitle, headTitle).forEach { it.typeface = titleFace }
        shown = null
        placeAroundOval()
    }

    /** Called when the operation screen is hidden (developer view). */
    fun reset() {
        celebration.stop()
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
        val finishedCapture = before?.screen == Screen.REGISTER_CAPTURE && before.duplicateOf == null &&
            ((state.screen == Screen.REGISTER_INFO && !state.labelOnly) || (state.screen == Screen.REGISTER_CAPTURE && state.duplicateOf != null))
        if (finishedCapture) {
            celebrate(before!!, state)
            return
        }
        draw(state)
    }

    /** Plays the celebration over the finished registration, then shows what came next. */
    private fun celebrate(capture: UiState, next: UiState) {
        pending = next
        capture.registration?.let { r ->
            stars.show(IntArray(4) { r.perDirection }, r.perDirection, -1, kids)
        }
        head.done = true
        headCaption.text = copy.followCaption(null, null)
        title.text = copy.celebration
        sub.visibility = View.GONE
        count.text = capture.registration?.let { copy.count(it.target, it.target) } ?: ""
        guideOverlay.animateScale(1f)
        val oval = ovalIn(celebration, 1f)
        celebration.celebrate(oval[0], oval[1], kids, copy.celebration) {
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
        val camera = screen == Screen.MONITOR || screen == Screen.REGISTER_CAPTURE
        val capture = screen == Screen.REGISTER_CAPTURE && state.duplicateOf == null

        guide.visibility = if (camera) View.VISIBLE else View.GONE
        form.visibility = if (screen == Screen.REGISTER_INFO) View.VISIBLE else View.GONE
        output.visibility = if (screen == Screen.OUTPUT) View.VISIBLE else View.GONE
        home.visibility = if (screen == Screen.OUTPUT) View.GONE else View.VISIBLE
        gear.visibility = if (camera) View.VISIBLE else View.GONE
        privacy.visibility = if (camera && !capture) View.VISIBLE else View.GONE
        privacy.text = copy.privacy
        listOf(stars, headCard, count).forEach { it.visibility = if (capture) View.VISIBLE else View.GONE }
        banner.visibility = if (camera && state.banner.isNotEmpty()) View.VISIBLE else View.GONE
        banner.text = "● ${state.banner}"
        banner.setTextColor(Color.parseColor("#33263F"))

        val found = screen == Screen.REGISTER_CAPTURE || state.showRegisterChoice || state.candidates.isNotEmpty()
        guideOverlay.outlineColor = if (found) APRICOT else Color.WHITE
        val scale = if (capture) circleScale(state.registration?.circle) else 1f
        guideScale = scale
        guideOverlay.animateScale(scale)

        when (screen) {
            Screen.MONITOR -> drawMonitor(state)
            Screen.REGISTER_CAPTURE -> if (capture) drawCapture(state) else drawDuplicate(state)
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
                if (selected == null) addReply(copy.noneOfThem(state.candidates.size), small = true) { actions.home() }
            }
            state.candidates.size == 1 && state.printButtonEnabled -> {
                addReply(copy.print, main = true, icon = R.drawable.ic_print) { actions.print() }
                addReply(copy.notMe, small = true) { actions.home() }
            }
        }
        val greeted = state.candidates.singleOrNull()?.personKey ?: state.candidates.firstOrNull { it.selected }?.personKey
        if (greeted != null && greeted != greetedKey && kids) celebration.confetti()
        greetedKey = greeted
    }

    private fun drawCapture(state: UiState) {
        val r = state.registration ?: return
        setBubble(copy.bubble(state).let { if (r.stalled) it.copy(sub = copy.stalled) else it })
        replies.removeAllViews()
        val collected = IntArray(4) { i -> (r.collected[HeadDirection.entries[i]] ?: 0).coerceAtMost(r.perDirection) }
        stars.show(collected, r.perDirection, r.next?.ordinal ?: -1, kids)
        head.done = r.next == null
        head.direction = r.next
        head.circle = r.circle
        headTitle.text = copy.followTitle
        headCaption.text = copy.followCaption(r.next, r.circle)
        count.text = copy.count(r.total, r.target)
    }

    private fun drawDuplicate(state: UiState) {
        setBubble(copy.bubble(state))
        replies.removeAllViews()
        addReply(copy.isMe, main = true, icon = R.drawable.ic_check) { actions.duplicateIsMe() }
        addReply(copy.firstTime) { actions.duplicateNotMe() }
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
        nameInput.hint = if (state.labelOnly) "예: 이준호" else "예: 김하은"
        if (entering) {
            chipsFromCode = true
            listOf(nameInput, departmentInput, gradeInput).forEach { it.setText("") }
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
        tagName.text = m.name
        tagName.typeface = jua
        tagSub.text = listOf(m.department, m.grade).filter { it.isNotBlank() }.joinToString(" · ")
        tagStripe.setBackgroundColor(if (m.department.isBlank()) Color.parseColor("#C9BCD2") else DepartmentColors.of(m.department))
        tag.translationY = -260 * density
        tag.animate().translationY(0f).setDuration(1500).setInterpolator(android.view.animation.DecelerateInterpolator()).start()
        if (kids) celebration.confetti()
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
                setFromChip(departmentInput, department)
                setFromChip(gradeInput, "")
                showGrades(department)
            })
        }
        departmentChips.visibility = if (departmentChips.childCount == 0) View.GONE else View.VISIBLE
    }

    private fun showGrades(department: String?) {
        gradeChips.removeAllViews()
        val grades = department?.let { DepartmentChoices.grades(people(), it) }.orEmpty()
        grades.forEach { grade ->
            gradeChips.addView(chip(grade, null) { checked -> if (checked) setFromChip(gradeInput, grade) })
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

    private fun circleScale(circle: FaceCircle?): Float = when (circle) {
        FaceCircle.BIG -> bigScale
        FaceCircle.SMALL -> smallScale
        else -> 1f
    }

    /** Centre of the face circle in [target]'s coordinates. */
    private fun ovalIn(target: View, scale: Float): FloatArray {
        val rect = guideOverlay.ovalRect(scale)
        val from = IntArray(2).also { guideOverlay.getLocationInWindow(it) }
        val to = IntArray(2).also { target.getLocationInWindow(it) }
        return floatArrayOf(rect.centerX() + from[0] - to[0], rect.centerY() + from[1] - to[1])
    }

    /** Stars hug the biggest circle; the head's centre sits level with the circle's centre. */
    private fun placeAroundOval() {
        if (guideOverlay.width == 0 || root.width == 0) return
        val big = guideOverlay.ovalRect(maxOf(bigScale, 1f))
        val from = IntArray(2).also { guideOverlay.getLocationInWindow(it) }
        val to = IntArray(2).also { stars.getLocationInWindow(it) }
        big.offset((from[0] - to[0]).toFloat(), (from[1] - to[1]).toFloat())
        stars.circle = big
        val centerY = ovalIn(overlay, 1f)[1] - overlay.paddingTop
        val headCenter = headCard.top - overlay.paddingTop + head.top + head.height * HeadView.HEAD_CENTER_FRACTION
        val wanted = centerY - headCenter
        val max = (overlay.height - overlay.paddingBottom - headCard.bottom).toFloat()
        headCard.translationY = wanted.coerceIn(0f, maxOf(0f, max))
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
        private const val REGISTERED_NOTICE_MS = 3_000L
    }
}
