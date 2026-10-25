package com.allstar.faithid.kiosk.settings

import android.content.Context
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import com.google.android.material.materialswitch.MaterialSwitch
import java.util.Locale

/** One tab of the settings dialog. */
interface SettingsTab {
    val title: String
    val view: View

    /** Returns [settings] with this tab's values. */
    fun applyTo(settings: KioskSettings): KioskSettings
}

/** Builds simple labelled form rows into a vertical [LinearLayout]. */
class SettingsForm(val context: Context, val root: LinearLayout = verticalLayout(context)) {

    fun section(title: String) {
        root.addView(TextView(context).apply {
            text = title
            textSize = 18f
            setPadding(0, dp(14), 0, dp(6))
        })
    }

    fun note(text: String): TextView = TextView(context).apply {
        this.text = text
        textSize = 13f
        setPadding(0, dp(2), 0, dp(6))
        root.addView(this)
    }

    fun text(label: String, value: String): EditText = labelled(label, EditText(context).apply {
        setText(value)
        setSingleLine(true)
    })

    fun number(label: String, value: Number, decimal: Boolean = value is Float || value is Double): EditText = labelled(label, numberInput(value, decimal))

    fun switch(label: String, value: Boolean): MaterialSwitch = MaterialSwitch(context).apply {
        text = label
        isChecked = value
        setPadding(0, dp(4), 0, dp(4))
        root.addView(this)
    }

    fun choice(label: String, items: List<String>, selected: Int, onChange: (Int) -> Unit = {}): Spinner {
        val spinner = Spinner(context)
        spinner.adapter = ArrayAdapter(context, android.R.layout.simple_spinner_item, items).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinner.setSelection(selected.coerceIn(0, (items.size - 1).coerceAtLeast(0)))
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) = onChange(position)
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        return labelled(label, spinner)
    }

    fun button(label: String, onClick: () -> Unit): Button = Button(context).apply {
        text = label
        setOnClickListener { onClick() }
        root.addView(this)
    }

    /** Two number inputs side by side. */
    fun pair(leftLabel: String, left: Number, rightLabel: String, right: Number): Pair<EditText, EditText> {
        val row = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
        fun column(label: String, value: Number): EditText {
            val box = verticalLayout(context)
            box.addView(TextView(context).apply { text = label })
            val input = numberInput(value, value is Float)
            box.addView(input)
            row.addView(box, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) })
            return input
        }
        val l = column(leftLabel, left)
        val r = column(rightLabel, right)
        root.addView(row)
        return l to r
    }

    private fun numberInput(value: Number, decimal: Boolean) = EditText(context).apply {
        setText(if (decimal) trimZero(value.toFloat()) else value.toString())
        setSingleLine(true)
        inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED or (if (decimal) InputType.TYPE_NUMBER_FLAG_DECIMAL else 0)
    }

    private fun <T : View> labelled(label: String, input: T): T {
        val box = verticalLayout(context).apply { setPadding(0, dp(4), 0, dp(4)) }
        box.addView(TextView(context).apply { text = label })
        box.addView(input)
        root.addView(box)
        return input
    }

    fun dp(value: Int): Int = (value * context.resources.displayMetrics.density).toInt()

    companion object {
        fun verticalLayout(context: Context) = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }

        fun trimZero(v: Float): String =
            if (kotlin.math.abs(v - v.toInt()) < 0.0001f) v.toInt().toString() else String.format(Locale.US, "%.2f", v).trimEnd('0').trimEnd('.')
    }
}

fun EditText.intValue(default: Int): Int = text.toString().trim().toIntOrNull() ?: default
fun EditText.floatValue(default: Float): Float = text.toString().trim().toFloatOrNull() ?: default
fun EditText.textOr(default: String): String = text.toString().ifBlank { default }

fun EditText.onChanged(action: () -> Unit) = addTextChangedListener(object : TextWatcher {
    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
    override fun afterTextChanged(s: Editable?) = action()
})
