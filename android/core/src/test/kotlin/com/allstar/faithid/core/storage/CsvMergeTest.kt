package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CsvMergeTest {
    private val faces = FakeFaces(seed = 3, dim = 8)
    private val people = (1..20).map { i -> faces.person("사람$i", if (i % 2 == 0) "유년부" else "초등부", "${i % 6 + 1}학년", key = "k$i") }
    private val numbers = people.mapIndexed { i, p -> p.key to (i + 1).toLong() }.toMap()
    private val exported = PeopleCsv.exportTable(people, numbers)
    private val base = CsvMerge.Base(exported.text, exported.columns)

    /** The exported table with [edit] applied to its rows (header first). */
    private fun edited(edit: (MutableList<MutableList<String>>) -> Unit): String {
        val rows = Csv.parse(exported.text).map { it.toMutableList() }.toMutableList()
        edit(rows)
        return Csv.write(rows)
    }

    private val header = Csv.parse(exported.text)[0]
    private fun col(name: String) = header.indexOf(name)

    private fun diff(text: String, current: List<Person> = people, labels: Map<String, String> = emptyMap()) =
        CsvMerge.diff(base, text, current, numbers, labels)

    @Test
    fun `an unchanged file changes nothing`() {
        assertTrue(diff(exported.text).isEmpty)
    }

    @Test
    fun `a changed cell is applied, a value changed in the app since then is kept`() {
        val text = edited { it[1][col("grade")] = "6학년" } // row of 사람1
        // Meanwhile the app changed 사람2's department: the file still has the old value there, which is no edit.
        val current = people.map { if (it.key == "k2") it.copy(metadata = it.metadata.withValue("department", "청년부")) else it }
        val changes = diff(text, current)
        assertEquals(mapOf("k1" to mapOf("grade" to "6학년")), changes.values)
    }

    @Test
    fun `a header renamed in place renames the column and keeps its data, edits in it still count`() {
        val text = edited {
            it[0][col("department")] = "반"
            it[3][col("department")] = "새반"
        }
        val changes = diff(text)
        assertEquals(mapOf("department" to "반"), changes.renamed)
        assertEquals(mapOf("k3" to mapOf("department" to "새반")), changes.values)
        assertTrue(changes.removedColumns.isEmpty())
        // Written next time with the name, the same header maps back to the column.
        val labelled = PeopleCsv.exportTable(people, numbers, mapOf("department" to "반"))
        assertEquals("반", Csv.parse(labelled.text)[0][col("department")])
        assertTrue(CsvMerge.diff(CsvMerge.Base(labelled.text, labelled.columns), labelled.text, people, numbers, mapOf("department" to "반")).isEmpty)
    }

    @Test
    fun `a renamed column is found by its values even when a new column moved it`() {
        val text = edited { rows ->
            val given = col("given_name")
            rows[0][col("department")] = "반"
            rows.forEachIndexed { i, row -> row.add(given + 1, if (i == 0) "메모" else if (i == 1) "새가족" else "") }
        }
        val changes = diff(text)
        assertEquals(mapOf("department" to "반"), changes.renamed)
        assertTrue(changes.removedColumns.isEmpty())
        assertEquals(mapOf("k1" to mapOf("메모" to "새가족")), changes.values)
    }

    @Test
    fun `a new column is added, a removed column is removed`() {
        val text = edited { rows ->
            val grade = col("grade")
            rows.forEachIndexed { i, row ->
                row.removeAt(grade)
                row.add(grade + 1, if (i == 0) "메모" else if (i == 1) "새가족" else "")
            }
        }
        val changes = diff(text)
        assertEquals(setOf("grade"), changes.removedColumns)
        assertEquals(mapOf("k1" to mapOf("메모" to "새가족")), changes.values)
    }

    @Test
    fun `a removed row deletes that person`() {
        val text = edited { it.removeAt(5) } // 사람5
        val changes = diff(text)
        assertEquals(setOf("k5"), changes.deleted)
        assertEquals(0, changes.heldDeletions)
    }

    @Test
    fun `a file without rows or with most rows gone deletes no one`() {
        val headerOnly = edited { rows -> while (rows.size > 1) rows.removeAt(1) }
        assertEquals(20, diff(headerOnly).heldDeletions)
        assertTrue(diff(headerOnly).deleted.isEmpty())
        val mostGone = edited { rows -> repeat(12) { rows.removeAt(1) } }
        assertTrue(diff(mostGone).deleted.isEmpty())
        assertEquals(12, diff(mostGone).heldDeletions)
    }

    @Test
    fun `rows match by number when the key cell is damaged, unknown rows are reported`() {
        val text = edited { rows ->
            rows[2][rows[2].lastIndex] = ""
            rows[2][col("given_name")] = "사람2바뀜"
            rows += MutableList(header.size) { "" }.also { it[col("family_name")] = "새"; it[col("given_name")] = "사람" }
        }
        val changes = diff(text)
        assertEquals(mapOf("k2" to mapOf("given_name" to "사람2바뀜")), changes.values)
        assertTrue(changes.deleted.isEmpty())
        assertEquals(listOf("새사람"), changes.unmatched)
    }

    @Test
    fun `the name is written as family and given name, and taking those columns out changes nothing`() {
        assertEquals(listOf("person_idx", "family_name", "given_name", "department", "grade"), header.take(5))
        assertEquals(listOf("", "사람1"), listOf(Csv.parse(exported.text)[1][1], Csv.parse(exported.text)[1][2]))
        val noFamily = edited { rows -> rows.forEach { it.removeAt(col("family_name")) } }
        assertTrue(diff(noFamily).removedColumns.isEmpty())
        assertTrue(diff(noFamily).values.isEmpty())
    }

    @Test
    fun `without a column in the metadata a removal empties the well-known fields and drops the others`() {
        val m = PersonMetadata.of(linkedMapOf("name" to "a", "department" to "b", "grade" to "c", "family_name" to "김", "메모" to "x"))
        assertEquals("", m.without("부서").department)
        assertEquals("a", m.without("name").name.ifEmpty { "a" })
        assertTrue(m.without("메모").entries.keys.none { it == "메모" })
        assertTrue(m.without("성").entries.keys.none { it == "family_name" })
    }
}
