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
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.serverapp.R

/**
 * "명단": every registered person as a table; tap a row to edit it in place. Besides the name, the
 * columns are the people's own CSV columns ([RosterColumns]); the first of them gives the filter chips.
 */
internal class PeoplePage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        /** [values] maps column keys to the edited values; columns not shown are left as they are. */
        fun save(person: Person, name: String, values: Map<String, String>)
        fun delete(person: Person)
    }

    private val context = root.context
    private val summary: TextView = root.findViewById(R.id.peopleSummaryText)
    private val footer: TextView = root.findViewById(R.id.peopleFooterText)
    private val search: EditText = root.findViewById(R.id.peopleSearchInput)
    private val chips: LinearLayout = root.findViewById(R.id.peopleDepartmentChips)
    private val nameHeader: TextView = root.findViewById(R.id.headPersonName)
    private val columnHeaders: LinearLayout = root.findViewById(R.id.headPersonColumns)
    private val fixedHeaders = mapOf(
        PeopleTable.SAMPLES to root.findViewById<TextView>(R.id.headPersonSamples),
        PeopleTable.TODAY to root.findViewById(R.id.headPersonToday),
    )
    private val fixedLabels = fixedHeaders.mapValues { it.value.text.toString() }
    private val adapter = PersonAdapter()
    private val inflater = LayoutInflater.from(context)

    private var people: List<Person> = emptyList()
    private var signature: List<String>? = null
    private var present: Set<String> = emptySet()
    private var labels: Map<String, String> = emptyMap()
    private var hidden: Set<String> = emptySet()
    private var columns: List<RosterColumn> = emptyList()
    /** The chosen filter chip: a value of the first column. */
    private var chosen: String? = null
    private var sort = SortState(PersonMetadata.CANONICAL_DEPARTMENT)
    private var editing: String? = null
    private var chipNames: List<String> = emptyList()

    init {
        root.findViewById<RecyclerView>(R.id.peopleList).apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@PeoplePage.adapter
            addItemDecoration(DividerItemDecoration(context, DividerItemDecoration.VERTICAL))
        }
        search.onTextChanged { applyFilter() }
        nameHeader.setOnClickListener { tapHeader(PeopleTable.NAME) }
        fixedHeaders.forEach { (key, view) -> view.setOnClickListener { tapHeader(key) } }
    }

    /**
     * [present] holds the keys of people who checked in today; [labels] and [hidden] are the
     * column headers and hidden columns from the developer screen.
     */
    fun render(people: List<Person>, present: Set<String>, labels: Map<String, String>, hidden: Set<String>) {
        // The dashboard refreshes every 2 s; redraw only when something changed, so an open editor keeps its text.
        val signature = people.map { "${it.key}:${it.revision}:${it.updatedAt}:${it.embeddings.size}" }
        if (signature == this.signature && present == this.present && labels == this.labels && hidden == this.hidden) return
        this.signature = signature
        this.people = people
        this.present = present
        this.labels = labels
        this.hidden = hidden
        columns = RosterColumns.shown(people, labels, hidden)
        if (editing != null && people.none { it.key == editing }) editing = null
        if (sort.key !in setOf(PeopleTable.NAME, PeopleTable.SAMPLES, PeopleTable.TODAY) && columns.none { it.key == sort.key }) {
            sort = SortState(columns.firstOrNull()?.key ?: PeopleTable.NAME)
        }
        buildHeaders()
        search.hint = "찾기: " + (listOf(RosterColumns.label(RosterColumns.NAME, labels)) + columns.take(2).map { it.label }).joinToString(", ")
        applyFilter()
    }

    private fun tapHeader(key: String) {
        sort = sort.tap(key)
        applyFilter()
    }

    private fun buildHeaders() {
        columnHeaders.removeAllViews()
        columns.forEach { column ->
            val cell = inflater.inflate(R.layout.cell_roster_header, columnHeaders, false) as TextView
            cell.tag = column.key
            cell.setOnClickListener { tapHeader(column.key) }
            columnHeaders.addView(cell)
        }
    }

    fun closeEditor() {
        editing = null
        applyFilter()
    }

    private fun applyFilter() {
        renderChips()
        val rows = PeopleTable.rows(people, present, search.text?.toString().orEmpty(), columns, chosen, sort)
        adapter.submit(rows)
        nameHeader.text = RosterColumns.label(RosterColumns.NAME, labels) + sort.arrow(PeopleTable.NAME)
        fixedHeaders.forEach { (key, view) -> view.text = fixedLabels.getValue(key) + sort.arrow(key) }
        columns.forEachIndexed { i, column -> (columnHeaders.getChildAt(i) as? TextView)?.text = column.label + sort.arrow(column.key) }
        val namesakes = people.groupBy { it.name }.count { it.value.size > 1 }
        summary.text = buildString {
            append("${people.size}명 · 줄을 누르면 그 자리에서 고칠 수 있어요")
            if (namesakes > 0) append(" · 동명이인 ${namesakes}쌍 (${columns.take(2).joinToString("·") { it.label }}으로 구분)")
        }
        footer.text = "${rows.size}명 표시 중"
    }

    private fun renderChips() {
        val group = columns.firstOrNull()
        val names = if (group == null) emptyList() else Departments.sorted(people.map { group.valueOf(it) })
        if (chosen != null && chosen !in names) chosen = null
        if (names == chipNames && chips.childCount == names.size) {
            for (i in 0 until chips.childCount) chips.getChildAt(i).isSelected = names[i] == chosen
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
        isSelected = name == chosen
        // The dot keeps the colour of its value on both backgrounds.
        val span = android.text.SpannableString(text)
        span.setSpan(android.text.style.ForegroundColorSpan(Departments.color(name)), 0, 1, 0)
        text = span
        contentDescription = "$name 만 보기"
        setOnClickListener {
            chosen = if (chosen == name) null else name
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
            val cells: LinearLayout = view.findViewById(R.id.personColumns)
            val samples: TextView = view.findViewById(R.id.personSamples)
            val today: TextView = view.findViewById(R.id.personToday)
        }

        inner class EditHolder(view: View) : RecyclerView.ViewHolder(view) {
            val name: EditText = view.findViewById(R.id.personNameInput)
            val inputs: LinearLayout = view.findViewById(R.id.personColumnInputs)
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
            val color = Departments.color(columns.firstOrNull()?.valueOf(person).orEmpty())
            when (holder) {
                is RowHolder -> {
                    holder.initial.text = initialOf(person.name)
                    holder.initial.tint(color)
                    holder.name.text = person.name
                    fillCells(holder.cells, person, color)
                    holder.samples.text = "${person.embeddings.size}장"
                    if (row.presentToday) holder.today.pill("✓ 왔어요", PillKind.IN) else holder.today.pill("아직", PillKind.OUT)
                    holder.itemView.contentDescription = "${person.name}, ${columns.joinToString(" ") { it.valueOf(person) }}. 눌러서 고치기"
                    holder.itemView.setOnClickListener {
                        editing = person.key
                        applyFilter()
                    }
                }
                is EditHolder -> {
                    holder.name.setText(person.name)
                    holder.name.hint = RosterColumns.label(RosterColumns.NAME, labels)
                    val inputs = columns.map { column ->
                        (inflater.inflate(R.layout.cell_roster_input, holder.inputs, false) as EditText).apply {
                            hint = column.label
                            setText(column.valueOf(person))
                        }
                    }
                    holder.inputs.removeAllViews()
                    inputs.forEach { holder.inputs.addView(it) }
                    holder.samples.text = "사진 ${person.embeddings.size}장"
                    holder.save.setOnClickListener {
                        val values = columns.zip(inputs).associate { (column, input) -> column.key to input.text.toString().trim() }
                        actions.save(person, holder.name.text.toString().trim(), values)
                    }
                    holder.close.setOnClickListener { closeEditor() }
                    holder.delete.setOnClickListener { actions.delete(person) }
                }
            }
        }
    }

    /** One cell per shown column; the first one carries the colour dot. Cells are reused while the columns stay. */
    private fun fillCells(container: LinearLayout, person: Person, color: Int) {
        if (container.childCount != columns.size) {
            container.removeAllViews()
            repeat(columns.size) { container.addView(inflater.inflate(R.layout.cell_roster_value, container, false)) }
        }
        columns.forEachIndexed { i, column ->
            val cell = container.getChildAt(i)
            val value = column.valueOf(person)
            cell.findViewById<TextView>(R.id.cellText).text = value
            cell.findViewById<View>(R.id.cellDot).apply {
                visibility = if (i == 0 && value.isNotBlank()) View.VISIBLE else View.GONE
                tint(color)
            }
        }
    }

    private companion object {
        const val ROW = 0
        const val EDIT = 1
    }
}
