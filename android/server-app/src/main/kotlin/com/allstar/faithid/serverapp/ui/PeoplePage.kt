package com.allstar.faithid.serverapp.ui

import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.serverapp.R

/** "명단" tab: every registered person, searchable, with edit and delete. */
internal class PeoplePage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun edit(person: Person)
        fun delete(person: Person)
        fun importCsv()
        fun exportCsv()
        fun reloadDatabase()
    }

    private val summary: TextView = root.findViewById(R.id.peopleSummaryText)
    private val search: EditText = root.findViewById(R.id.peopleSearchInput)
    private val adapter = PersonAdapter()
    private var people: List<Person> = emptyList()
    private var numbers: Map<String, Long> = emptyMap()

    init {
        root.findViewById<View>(R.id.peopleHeader).apply {
            findViewById<View>(R.id.personEditButton).visibility = View.INVISIBLE
            findViewById<View>(R.id.personDeleteButton).visibility = View.INVISIBLE
        }
        root.findViewById<RecyclerView>(R.id.peopleList).apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@PeoplePage.adapter
            addItemDecoration(DividerItemDecoration(context, DividerItemDecoration.VERTICAL))
        }
        search.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) = applyFilter()
        })
        root.findViewById<Button>(R.id.importCsvButton).setOnClickListener { actions.importCsv() }
        root.findViewById<Button>(R.id.exportCsvButton).setOnClickListener { actions.exportCsv() }
        root.findViewById<Button>(R.id.reloadDbButton).setOnClickListener { actions.reloadDatabase() }
    }

    fun render(people: List<Person>, displayNumbers: Map<String, Long>) {
        if (people === this.people && displayNumbers == numbers) return
        this.people = people.sortedWith(compareBy<Person> { displayNumbers[it.key] ?: Long.MAX_VALUE }.thenBy { it.name })
        numbers = displayNumbers
        applyFilter()
    }

    private fun applyFilter() {
        val query = search.text?.toString()?.trim().orEmpty()
        val visible = if (query.isEmpty()) people else people.filter { p ->
            listOf(p.name, p.department, p.grade).any { it.contains(query, ignoreCase = true) }
        }
        adapter.submit(visible)
        val namesakes = people.groupBy { it.name }.count { it.value.size > 1 }
        summary.text = buildString {
            append("등록 ${people.size}명")
            if (query.isNotEmpty()) append(" · 검색 결과 ${visible.size}명")
            if (namesakes > 0) append(" · 동명이인 ${namesakes}쌍 (부서·학년으로 구분)")
            append(" · face_db.csv를 엑셀로 고친 뒤 'CSV 편집 반영'을 누르면 이름·부서·학년이 바뀝니다.")
        }
    }

    private inner class PersonAdapter : RecyclerView.Adapter<PersonAdapter.Holder>() {
        private var rows: List<Person> = emptyList()

        fun submit(next: List<Person>) {
            rows = next
            notifyDataSetChanged()
        }

        inner class Holder(view: View) : RecyclerView.ViewHolder(view) {
            val number: TextView = view.findViewById(R.id.personNumber)
            val name: TextView = view.findViewById(R.id.personName)
            val department: TextView = view.findViewById(R.id.personDepartment)
            val grade: TextView = view.findViewById(R.id.personGrade)
            val samples: TextView = view.findViewById(R.id.personSamples)
            val updated: TextView = view.findViewById(R.id.personUpdated)
            val edit: Button = view.findViewById(R.id.personEditButton)
            val delete: Button = view.findViewById(R.id.personDeleteButton)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_person, parent, false))

        override fun getItemCount() = rows.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val person = rows[position]
            holder.number.text = (numbers[person.key] ?: 0L).toString()
            holder.name.text = person.name
            holder.department.text = person.department
            holder.grade.text = person.grade
            holder.samples.text = person.embeddings.size.toString()
            holder.updated.text = Format.dateTime(person.updatedAt)
            holder.edit.setOnClickListener { actions.edit(person) }
            holder.delete.setOnClickListener { actions.delete(person) }
        }
    }
}
