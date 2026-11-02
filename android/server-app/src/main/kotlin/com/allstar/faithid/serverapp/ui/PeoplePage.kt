package com.allstar.faithid.serverapp.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.storage.PeopleCsv
import com.allstar.faithid.serverapp.R

/**
 * "명단": every registered person as a table; tap a row to edit it in place, long-press a cell to
 * change that one value, long-press a header to rename or hide the column. Besides the name, the
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

        /** New name of a column (null = its own name again); the data stays in the column. */
        fun renameColumn(column: String, label: String?)
        fun hideColumn(column: String)
        fun openCsv()

        /** The column the filter chips and colours use was chosen in the 분류 menu. */
        fun groupPeople(column: String)
    }

    private val context = root.context
    private val summary: TextView = root.findViewById(R.id.peopleSummaryText)
    private val footer: TextView = root.findViewById(R.id.peopleFooterText)
    private val search: EditText = root.findViewById(R.id.peopleSearchInput)
    private val chips: LinearLayout = root.findViewById(R.id.peopleDepartmentChips)
    private val groupButton: TextView = root.findViewById(R.id.peopleGroupButton)
    private var groupColumn: String? = null
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
    private var csvPending = false
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
        nameHeader.setOnLongClickListener { editHeader(RosterColumns.NAME); true }
        root.findViewById<View>(R.id.peopleOpenCsvButton).setOnClickListener { actions.openCsv() }
        fixedHeaders.forEach { (key, view) -> view.setOnClickListener { tapHeader(key) } }
    }

    /**
     * [present] holds the keys of people who checked in today; [labels] and [hidden] are the
     * column headers and hidden columns from the developer screen.
     */
    fun render(people: List<Person>, present: Set<String>, labels: Map<String, String>, hidden: Set<String>, csvPending: Boolean, group: String) {
        // The dashboard refreshes every 2 s; redraw only when something changed, so an open editor keeps its text.
        val signature = people.map { "${it.key}:${it.revision}:${it.updatedAt}:${it.embeddings.size}" }
        if (signature == this.signature && present == this.present && labels == this.labels && hidden == this.hidden && csvPending == this.csvPending && group == this.groupColumn) return
        if (group != this.groupColumn) chosen = null
        this.groupColumn = group
        this.csvPending = csvPending
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
        val choices = Grouping.choices(RosterColumns.all(people, labels, emptySet()))
        groupButton.groupMenu(choices, choices.firstOrNull { it.key == group }) { actions.groupPeople(it) }
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
            cell.setOnLongClickListener { editHeader(column.key); true }
            columnHeaders.addView(cell)
        }
    }

    /** Long press on a header: rename the column (only its name; kiosks and the CSV follow) or hide it. */
    private fun editHeader(key: String) {
        val input = (inflater.inflate(R.layout.cell_roster_input, null, false) as EditText).apply {
            setText(RosterColumns.label(key, labels))
            hint = RosterColumns.DEFAULT_LABELS[key] ?: key
            contentDescription = "열 이름"
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, context.dp(52))
        }
        val dialog = AlertDialog.Builder(context)
            .setTitle("열 이름 바꾸기")
            .setMessage("명단, CSV 머리글, 키오스크 입력 칸의 이름이 바뀝니다. 내용은 그대로입니다. 비우면 기본 설정값(${RosterColumns.DEFAULT_LABELS[key] ?: key})으로 돌아갑니다.")
            .setView(padded(input))
            .setPositiveButton("저장") { _, _ ->
                val label = input.text.toString().trim()
                actions.renameColumn(key, label.takeIf { it.isNotEmpty() && it != (RosterColumns.DEFAULT_LABELS[key] ?: key) })
            }
            .setNegativeButton("취소", null)
        if (key != RosterColumns.NAME) dialog.setNeutralButton("열 숨기기") { _, _ -> actions.hideColumn(key) }
        dialog.show()
    }

    /** Long press on a cell: change that one value (saved, written to the CSV and sent to the kiosks). */
    private fun editCell(person: Person, key: String) {
        val input = (inflater.inflate(R.layout.cell_roster_input, null, false) as EditText).apply {
            setText(person.metadata.valueOf(key))
            hint = RosterColumns.label(key, labels)
            contentDescription = "${RosterColumns.label(key, labels)} 값"
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, context.dp(52))
        }
        AlertDialog.Builder(context)
            .setTitle("${person.name} · ${RosterColumns.label(key, labels)}")
            .setView(padded(input))
            .setPositiveButton("저장") { _, _ ->
                val value = input.text.toString().trim()
                if (key == RosterColumns.NAME) actions.save(person, value, emptyMap())
                else actions.save(person, person.name, mapOf(key to value))
            }
            .setNegativeButton("취소", null)
            .show()
    }

    private fun padded(view: View) = FrameLayout(context).apply {
        setPadding(context.dp(22), context.dp(8), context.dp(22), 0)
        addView(view)
    }

    private fun openEditor(person: Person) {
        editing = person.key
        applyFilter()
    }

    fun closeEditor() {
        editing = null
        applyFilter()
    }

    private fun applyFilter() {
        renderChips()
        val rows = PeopleTable.rows(people, present, search.text?.toString().orEmpty(), columns, chosen, sort, groupColumn)
        adapter.submit(rows)
        nameHeader.text = RosterColumns.label(RosterColumns.NAME, labels) + sort.arrow(PeopleTable.NAME)
        fixedHeaders.forEach { (key, view) -> view.text = fixedLabels.getValue(key) + sort.arrow(key) }
        columns.forEachIndexed { i, column -> (columnHeaders.getChildAt(i) as? TextView)?.text = column.label + sort.arrow(column.key) }
        val namesakes = people.groupBy { it.name }.count { it.value.size > 1 }
        summary.text = buildString {
            append("${people.size}명 · 줄을 누르면 그 자리에서, 칸을 길게 누르면 그 칸만 고쳐요")
            if (csvPending) append(" · CSV 편집 대기 중(서버를 다시 시작하면 반영)")
            if (namesakes > 0) append(" · 동명이인 ${namesakes}쌍 (${columns.take(2).joinToString("·") { it.label }}으로 구분)")
        }
        footer.text = "${rows.size}명 표시 중"
    }

    private fun renderChips() {
        val names = Departments.sorted(people.map { groupOf(it) })
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
            val color = Departments.color(groupOf(person))
            when (holder) {
                is RowHolder -> {
                    holder.initial.text = initialOf(person.name)
                    holder.initial.tint(color)
                    holder.name.text = person.name
                    // A long-press listener makes a view take taps too: a tap still opens the row's editor.
                    holder.name.setOnClickListener { openEditor(person) }
                    holder.name.setOnLongClickListener { editCell(person, RosterColumns.NAME); true }
                    fillCells(holder.cells, person, color)
                    holder.samples.text = "${person.embeddings.size}장"
                    if (row.presentToday) holder.today.pill("✓ 왔어요", PillKind.IN) else holder.today.pill("아직", PillKind.OUT)
                    holder.itemView.contentDescription = "${person.name}, ${columns.joinToString(" ") { it.valueOf(person) }}. 눌러서 고치기"
                    holder.itemView.setOnClickListener { openEditor(person) }
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

    /** The person's value in the grouping column (filter chips and colours). */
    private fun groupOf(person: Person): String = groupColumn?.let { PeopleCsv.cellOf(person.metadata, it) } ?: person.department

    /** One cell per shown column; the grouping column carries the colour dot. Cells are reused while the columns stay. */
    private fun fillCells(container: LinearLayout, person: Person, color: Int) {
        if (container.childCount != columns.size) {
            container.removeAllViews()
            repeat(columns.size) { container.addView(inflater.inflate(R.layout.cell_roster_value, container, false)) }
        }
        columns.forEachIndexed { i, column ->
            val cell = container.getChildAt(i)
            val value = column.valueOf(person)
            cell.setOnClickListener { openEditor(person) }
            cell.setOnLongClickListener { editCell(person, column.key); true }
            cell.findViewById<TextView>(R.id.cellText).text = value
            cell.findViewById<View>(R.id.cellDot).apply {
                visibility = if (column.key == groupColumn && value.isNotBlank()) View.VISIBLE else View.GONE
                tint(color)
            }
        }
    }

    private companion object {
        const val ROW = 0
        const val EDIT = 1
    }
}
