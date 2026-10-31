package com.allstar.faithid.serverapp.ui

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.text.Editable
import android.text.TextWatcher
import android.util.TypedValue
import android.view.View
import android.widget.EditText
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.allstar.faithid.serverapp.R

internal fun Context.dp(value: Int): Int =
    TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics).toInt()

internal fun View.tint(color: Int) {
    backgroundTintList = ColorStateList.valueOf(color)
}

internal fun EditText.onTextChanged(action: (String) -> Unit) = addTextChangedListener(object : TextWatcher {
    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
    override fun afterTextChanged(s: Editable?) = action(s?.toString().orEmpty())
})

/** The coloured status labels used in the tables. */
internal enum class PillKind(val background: Int, val text: Int) {
    IN(R.color.ok_soft, R.color.ok),
    OUT(R.color.muted_soft, R.color.ink_muted),
    WAIT(R.color.line, R.color.warn),
    GUEST(R.color.guest_soft, R.color.guest_ink),
}

internal fun TextView.pill(text: String, kind: PillKind) {
    this.text = text
    tint(ContextCompat.getColor(context, kind.background))
    setTextColor(ContextCompat.getColor(context, kind.text))
}

/** The first letter shown in a name's coloured circle ("김하은" → "하"). */
internal fun initialOf(name: String): String {
    val n = name.trim()
    return when {
        n.isEmpty() -> "?"
        n.length in 2..4 && n.all { it in '가'..'힣' } -> n.substring(1, 2)
        else -> n.substring(0, 1)
    }
}

/** A thin rounded bar: [fraction] of the track filled with [fill]. */
internal class MeterView(context: Context, private val track: Int, private val fill: Int) : View(context) {
    var fraction: Float = 0f
        set(value) {
            field = value.coerceIn(0f, 1f)
            invalidate()
        }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()

    init {
        // The numbers next to the bar say the same thing.
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    override fun onDraw(canvas: Canvas) {
        val r = height / 2f
        rect.set(0f, 0f, width.toFloat(), height.toFloat())
        paint.color = track
        canvas.drawRoundRect(rect, r, r, paint)
        if (fraction > 0f) {
            rect.right = (width * fraction).coerceAtLeast(height.toFloat())
            paint.color = fill
            canvas.drawRoundRect(rect, r, r, paint)
        }
    }
}
