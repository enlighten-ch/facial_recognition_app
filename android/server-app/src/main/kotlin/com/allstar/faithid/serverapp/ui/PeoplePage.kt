package com.allstar.faithid.serverapp.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.serverapp.R

/** "명단": every registered person as a table; tap a row to edit it in place. */
internal class PeoplePage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun save(person: Person, name: String, department: String, grade: String)
        fun delete(person: Person)
    }

    private val context = root.context
    private val summary: TextView = root.findViewById(R.id.peopleSummaryText)
    private val footer: TextView = root.findViewById(R.id.peopleFooterText)
    private val search: EditText = root.findViewById(R.id.peopleSearchInput)
    private val chips: LinearLayout = root.findViewById(R.id.peopleDepartmentChips)
    private val headers = mapOf(
        PeopleTable.NAME to root.findViewById<TextView>(R.id.headPersonName),
        PeopleTable.DEPARTMENT to root.findViewById(R.id.headPersonDepartment),
        PeopleTable.GRADE to root.findViewById(R.id.headPersonGrade),
        PeopleTable.SAMPLES to root.findViewById(R.id.headPersonSamples),
        PeopleTable.TODAY to root.findViewById(R.id.headPersonToday),
    )
    private val headerLabels = headers.mapValues { it.value.text.toString() }
    private val adapter = PersonAdapter()

    private var people: List<Person> = emptyList()
    private var signature: List<String>? = null
    private var present: Set<String> = emptySet()
    private var department: String? = null
    private var sort = SortState(PeopleTable.DEPARTMENT)
    private var editing: String? = null
    private var chipNames: List<String> = emptyList()

    init {
        root.findViewById<RecyclerView>(R.id.peopleList).apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@PeoplePage.adapter
            addItemDecoration(DividerItemDecoration(context, DividerItemDecoration.VERTICAL))
        }
        search.onTextChanged { applyFilter() }
        headers.forEach { (key, view) ->
            view.setOnClickListener {
                sort = sort.tap(key)
                applyFilter()
            }
        }
    }

    /** [present] holds the keys of people who checked in today. */
    fun render(people: List<Person>, present: Set<String>) {
        // The dashboard refreshes every 2 s; redraw only when something changed, so an open editor keeps its text.
        val signature = people.map { "${it.key}:${it.revision}:${it.updatedAt}:${it.embeddings.size}" }
        if (signature == this.signature && present == this.present) return
        this.signature = signature
        this.people = people
        this.present = present
        if (editing != null && people.none { it.key == editing }) editing = null
        applyFilter()
    }

    fun closeEditor() {
        editing = null
        applyFilter()
    }

    private fun applyFilter() {
        renderChips()
        val rows = PeopleTable.rows(people, present, search.text?.toString().orEmpty(), department, sort)
        adapter.submit(rows)
        headers.forEach { (key, view) -> view.text = headerLabels.getValue(key) + sort.arrow(key) }
        val namesakes = people.groupBy { it.name }.count { it.value.size > 1 }
        summary.text = buildString {
            append("${people.size}명 · 줄을 누르면 그 자리에서 고칠 수 있어요")
            if (namesakes > 0) append(" · 동명이인 ${namesakes}쌍 (부서·학년으로 구분)")
        }
        footer.text = "${rows.size}명 표시 중"
    }

    private fun renderChips() {
        val names = Departments.sorted(people.map { it.department })
        if (department != null && department !in names) department = null
        if (names == chipNames && chips.childCount == names.size) {
            for (i in 0 until chips.childCount) chips.getChildAt(i).isSelected = names[i] == department
            return
        }
        chipNames = names
        chips.removeAllViews()
        names.forEach { name -> chips.addView(chip(name)) }
    }

    private fun chip(name: String): View = TextView(context).apply {
        text = "●  $name"
        textSize = 15f
        minHeight = context.dp(44)
        gravity = android.view.Gravity.CENTER
        setPadding(context.dp(14), 0, context.dp(16), 0)
        setBackgroundResource(R.drawable.bg_filter_chip)
        setTextColor(ContextCompat.getColorStateList(context, R.color.chip_text))
        isSelected = name == department
        // The dot keeps the department colour on both backgrounds.
        val span = android.text.SpannableString(text)
        span.setSpan(android.text.style.ForegroundColorSpan(Departments.color(name)), 0, 1, 0)
        text = span
        contentDescription = "$name 만 보기"
        setOnClickListener {
            department = if (department == name) null else name
            applyFilter()
        }
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, context.dp(44)).apply { marginEnd = context.dp(8) }
    }

    private inner class PersonAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
        private var rows: List<PersonRow> = emptyList()

        fun submit(next: List<PersonRow>) {
            rows = next
            notifyDataSetChanged()
        }

        inner class RowHolder(view: View) : RecyclerView.ViewHolder(view) {
            val initial: TextView = view.findViewById(R.id.personInitial)
            val name: TextView = view.findViewById(R.id.personName)
            val dot: View = view.findViewById(R.id.personDepartmentDot)
            val department: TextView = view.findViewById(R.id.personDepartment)
            val grade: TextView = view.findViewById(R.id.personGrade)
            val samples: TextView = view.findViewById(R.id.personSamples)
            val today: TextView = view.findViewById(R.id.personToday)
        }

        inner class EditHolder(view: View) : RecyclerView.ViewHolder(view) {
            val name: EditText = view.findViewById(R.id.personNameInput)
            val department: EditText = view.findViewById(R.id.personDepartmentInput)
            val grade: EditText = view.findViewById(R.id.personGradeInput)
            val samples: TextView = view.findViewById(R.id.personEditSamples)
            val save: Button = view.findViewById(R.id.personSaveButton)
            val close: Button = view.findViewById(R.id.personCloseButton)
            val delete: Button = view.findViewById(R.id.personDeleteButton)
        }

        override fun getItemViewType(position: Int) = if (rows[position].person.key == editing) EDIT else ROW

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
            val inflater = LayoutInflater.from(parent.context)
            return if (viewType == EDIT) EditHolder(inflater.inflate(R.layout.item_person_edit, parent, false))
            else RowHolder(inflater.inflate(R.layout.item_person, parent, false))
        }

        override fun getItemCount() = rows.size

        override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
            val row = rows[position]
            val person = row.person
            val color = Departments.color(person.department)
            when (holder) {
                is RowHolder -> {
                    holder.initial.text = initialOf(person.name)
                    holder.initial.tint(color)
                    holder.name.text = person.name
                    holder.dot.tint(color)
                    holder.department.text = person.department
                    holder.grade.text = person.grade
                    holder.samples.text = "${person.embeddings.size}장"
                    if (row.presentToday) holder.today.pill("✓ 왔어요", PillKind.IN) else holder.today.pill("아직", PillKind.OUT)
                    holder.itemView.contentDescription = "${person.name}, ${person.department} ${person.grade}. 눌러서 고치기"
                    holder.itemView.setOnClickListener {
                        editing = person.key
                        applyFilter()
                    }
                }
                is EditHolder -> {
                    holder.name.setText(person.name)
                    holder.department.setText(person.department)
                    holder.grade.setText(person.grade)
                    holder.samples.text = "사진 ${person.embeddings.size}장"
                    holder.save.setOnClickListener {
                        actions.save(person, holder.name.text.toString().trim(), holder.department.text.toString().trim(), holder.grade.text.toString().trim())
                    }
                    holder.close.setOnClickListener { closeEditor() }
                    holder.delete.setOnClickListener { actions.delete(person) }
                }
            }
        }
    }

    private companion object {
        const val ROW = 0
        const val EDIT = 1
    }
}
