package com.allstar.faithid.server

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.model.EmbeddingSets
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.server.people.PeopleRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.atomic.AtomicLong

class PeopleRepositoryTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val faces = FakeFaces(seed = 21, dim = 32)
    private val time = AtomicLong(1_000)
    private val clock = Clock { time.incrementAndGet() }
    private lateinit var dir: File

    private fun open(maxEmbeddings: Int = 100): Pair<PeopleStore, PeopleRepository> {
        if (!::dir.isInitialized) dir = temp.newFolder("data")
        val store = PeopleStore.open(dir, JdbcSqlDatabase.FACTORY)
        return store to PeopleRepository(store, maxEmbeddings, clock, ServerLog.NONE)
    }

    private fun register(key: String, name: String, samples: List<FloatArray>, opId: String = "r-$key") =
        PersonOp.Register(opId, key, PersonMetadata.of(name, "청년부", "1학년"), samples, 1)

    @Test
    fun `register then add samples bumps revisions and is idempotent`() {
        val (_, repo) = open()
        val first = repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 4))), "k1")
        assertEquals(OpResult.APPLIED, first.results["r-a"])
        assertEquals(1, repo.revision)
        val more = faces.samples("a", 2)
        repo.applyOps(listOf(PersonOp.AddEmbeddings("add1", "a", more)), "k1")
        assertEquals(2, repo.revision)
        assertEquals(6, repo.get("a")!!.embeddings.size)
        // Retrying both operations (lost responses) changes nothing.
        val replay = repo.applyOps(listOf(register("a", "홍길동", faces.samples("x", 1).take(0) + more), PersonOp.AddEmbeddings("add1", "a", more)), "k1")
        assertEquals(2, replay.serverRevision)
        assertEquals(6, repo.get("a")!!.embeddings.size)
    }

    @Test
    fun `samples from two kiosks merge instead of overwriting each other`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 3))), "k1")
        val fromK1 = faces.samples("a", 2)
        val fromK2 = faces.samples("a", 2)
        repo.applyOps(listOf(PersonOp.AddEmbeddings("k2-op", "a", fromK2)), "k2")
        repo.applyOps(listOf(PersonOp.AddEmbeddings("k1-op", "a", fromK1)), "k1")
        val merged = repo.get("a")!!.embeddings
        assertEquals(7, merged.size)
        assertTrue(EmbeddingSets.containsAll(merged, fromK1 + fromK2))
    }

    @Test
    fun `server metadata wins over a kiosk re-registration`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 3))), "k1")
        repo.updateMetadata("a", "홍길순", "중등부", "2학년")
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 1), opId = "again")), "k1")
        assertEquals("홍길순", repo.get("a")!!.name)
        assertEquals(4, repo.get("a")!!.embeddings.size)
    }

    @Test
    fun `deleted people stay deleted and unknown or invalid ops are rejected`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 3)), register("b", "김철수", faces.samples("b", 3))), "k1")
        assertTrue(repo.delete("a"))
        assertFalse(repo.delete("a"))
        val results = repo.applyOps(
            listOf(
                PersonOp.AddEmbeddings("1", "a", faces.samples("a", 1)),
                register("a", "홍길동", faces.samples("a", 1), opId = "2"),
                PersonOp.AddEmbeddings("3", "ghost", faces.samples("g", 1)),
                PersonOp.AddEmbeddings("4", "b", emptyList()),
                PersonOp.AddEmbeddings("5", "b", listOf(FloatArray(5))),
                register("c", "", faces.samples("c", 1), opId = "6"),
            ),
            "k2",
        ).results
        assertEquals(OpResult.REJECTED_DELETED, results["1"])
        assertEquals(OpResult.REJECTED_DELETED, results["2"])
        assertEquals(OpResult.REJECTED_UNKNOWN, results["3"])
        assertEquals(OpResult.REJECTED_INVALID, results["4"])
        assertEquals(OpResult.REJECTED_INVALID, results["5"])
        assertEquals(OpResult.REJECTED_INVALID, results["6"])
        assertNull(repo.get("a"))
        assertTrue(repo.isDeleted("a"))
    }

    @Test
    fun `changes since a revision include only newer people and tombstones`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "A", faces.samples("a", 2))), "k")
        repo.applyOps(listOf(register("b", "B", faces.samples("b", 2))), "k")
        val epoch = repo.epoch
        repo.delete("a")
        val delta = repo.changesSince(2, epoch)
        assertFalse(delta.fullSnapshot)
        assertTrue(delta.persons.isEmpty())
        assertEquals(listOf("a"), delta.tombstones.map { it.key })
        val fromOne = repo.changesSince(1, epoch)
        assertEquals(listOf("b"), fromOne.persons.map { it.key })
        assertTrue(repo.changesSince(0, epoch).fullSnapshot)
        assertTrue(repo.changesSince(2, "other-epoch").fullSnapshot)
        assertTrue(repo.changesSince(99, epoch).fullSnapshot)
        assertTrue(repo.changesSince(repo.revision, epoch).isEmpty())
    }

    @Test
    fun `state survives a restart and reload starts a new epoch`() {
        val (store, repo) = open()
        repo.applyOps(listOf(register("a", "A", faces.samples("a", 2))), "k")
        repo.delete("a")
        repo.applyOps(listOf(register("b", "B", faces.samples("b", 2))), "k")
        val epoch = repo.epoch
        store.close()
        val (store2, reopened) = open()
        assertEquals(epoch, reopened.epoch)
        assertEquals(3, reopened.revision)
        assertTrue(reopened.isDeleted("a"))
        assertEquals(listOf("b"), reopened.list().map { it.key })
        var notified = -1L
        reopened.onChanged = { notified = it }
        reopened.reload()
        assertNotEquals(epoch, reopened.epoch)
        assertEquals(3, notified)
        store2.close()
    }

    @Test
    fun `samples beyond the limit are pruned on the server`() {
        val (_, repo) = open(maxEmbeddings = 5)
        repo.applyOps(listOf(register("a", "A", faces.samples("a", 4))), "k")
        repo.applyOps(listOf(PersonOp.AddEmbeddings("x", "a", faces.samples("a", 4))), "k")
        assertEquals(5, repo.get("a")!!.embeddings.size)
    }

    private fun csv(repo: PeopleRepository) = File(dir, PeopleStore.CSV_FILE)

    /** Edits face_db.csv like a spreadsheet would (the file's time changes as well). */
    private fun editCsv(repo: PeopleRepository, edit: (MutableList<MutableList<String>>) -> Unit) {
        val file = csv(repo)
        val rows = Csv.parse(file.readText()).map { it.toMutableList() }.toMutableList()
        edit(rows)
        file.writeText(Csv.write(rows))
        file.setLastModified(file.lastModified() + 2_000)
    }

    @Test
    fun `csv edits wait for the next start and are not overwritten meanwhile`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2)), register("b", "김철수", faces.samples("b", 2))), "k")
        repo.applyCsvFile()
        val nameColumn = Csv.parse(csv(repo).readText())[0].indexOf("given_name")
        editCsv(repo) { rows -> rows.first { "a" in it }[nameColumn] = "길순" }
        // A kiosk change while the edit waits: the file is left alone.
        repo.applyOps(listOf(PersonOp.AddEmbeddings("x", "b", faces.samples("b", 1))), "k")
        assertNull(repo.exportCsv())
        assertTrue(repo.csvEditPending)
        assertTrue(csv(repo).readText().contains("길순"))
        // At the next start (here: applying the file) the edit is applied and the kiosk change kept.
        val before = repo.revision
        val result = repo.applyCsvFile()
        assertEquals(1, result.changedPeople)
        assertEquals("홍길순", repo.get("a")!!.name)
        assertEquals(3, repo.get("b")!!.embeddings.size)
        assertEquals(before + 1, repo.revision)
        assertFalse(repo.csvEditPending)
        // Applying the same file again changes nothing.
        assertEquals(0, repo.applyCsvFile().changedPeople)
        assertEquals(before + 1, repo.revision)
    }

    @Test
    fun `a new time on an unchanged file is not an edit`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2))), "k")
        repo.applyCsvFile()
        csv(repo).setLastModified(csv(repo).lastModified() + 5_000) // shared storage reporting another time
        assertFalse(repo.csvEditPending)
        repo.updateMetadata("a", mapOf("grade" to "2학년"))
        assertTrue(repo.exportCsv() != null)
        assertTrue(csv(repo).readText().contains("2학년"))
    }

    @Test
    fun `a value changed in the app after the csv was written is not reverted by an edit elsewhere`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2)), register("b", "김철수", faces.samples("b", 2))), "k")
        repo.applyCsvFile()
        val gradeColumn = Csv.parse(csv(repo).readText())[0].indexOf("grade")
        editCsv(repo) { rows -> rows.first { "b" in it }[gradeColumn] = "2학년" }
        repo.updateMetadata("a", mapOf("department" to "중등부")) // in the app, while the edit waits
        repo.applyCsvFile()
        assertEquals("중등부", repo.get("a")!!.department)
        assertEquals("2학년", repo.get("b")!!.grade)
    }

    @Test
    fun `a renamed header names the column, a removed row deletes the person after a backup`() {
        val (_, repo) = open()
        repo.applyOps((1..5).map { register("p$it", "사람$it", faces.samples("p$it", 1)) }, "k")
        repo.applyCsvFile()
        val reasons = mutableListOf<String>()
        repo.beforeBulkChange = { reasons += it }
        val changed = mutableListOf<Long>()
        repo.onChanged = { changed += it }
        editCsv(repo) { rows ->
            rows[0][rows[0].indexOf("department")] = "반"
            rows.removeAll { "p3" in it }
        }
        val result = repo.applyCsvFile()
        assertEquals(mapOf("department" to "반"), repo.columnLabels)
        assertEquals("청년부", repo.get("p1")!!.department)
        assertNull(repo.get("p3"))
        assertTrue(repo.isDeleted("p3"))
        assertEquals(setOf("p3"), result.changes.deleted)
        assertEquals(listOf("csv"), reasons)
        assertEquals(1, changed.size)
        // The written CSV uses the new name, and reading it back changes nothing.
        assertTrue(Csv.parse(csv(repo).readText())[0].contains("반"))
        assertTrue(repo.applyCsvFile().changes.isEmpty)
        // Written back under its own name, the column loses the name.
        editCsv(repo) { rows -> rows[0][rows[0].indexOf("반")] = "department" }
        repo.applyCsvFile()
        assertEquals(emptyMap<String, String>(), repo.columnLabels)
        assertEquals("청년부", repo.get("p1")!!.department)
    }

    @Test
    fun `someone registered while the csv was open in a spreadsheet is not taken as deleted`() {
        val (_, repo) = open()
        repo.applyOps((1..3).map { register("p$it", "사람$it", faces.samples("p$it", 1)) }, "k")
        repo.applyCsvFile()
        val opened = Csv.parse(csv(repo).readText()).map { it.toMutableList() }.toMutableList() // the spreadsheet's copy
        repo.applyOps(listOf(register("new", "새사람", faces.samples("new", 1))), "k")
        repo.exportCsv() // the server writes the newer list meanwhile
        val nameColumn = opened[0].indexOf("given_name")
        opened.first { "p1" in it }[nameColumn] = "사람1바뀜"
        opened.removeAll { "p2" in it }
        csv(repo).writeText(Csv.write(opened)) // the spreadsheet saves its older copy
        val result = repo.applyCsvFile()
        assertEquals("새사람", repo.get("new")?.name)
        assertEquals("사람1바뀜", repo.get("p1")!!.name)
        assertEquals(setOf("p2"), result.changes.deleted)
    }

    @Test
    fun `family and given name in the csv set the name and where it splits`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "강율", faces.samples("a", 2))), "k") // two syllables: no split known
        repo.applyCsvFile()
        val header = Csv.parse(csv(repo).readText())[0]
        editCsv(repo) { rows ->
            val row = rows.first { "a" in it }
            assertEquals(listOf("", "강율"), listOf(row[header.indexOf("family_name")], row[header.indexOf("given_name")]))
            row[header.indexOf("family_name")] = "강"
            row[header.indexOf("given_name")] = "율"
        }
        repo.applyCsvFile()
        assertEquals("강율", repo.get("a")!!.name)
        assertEquals("강", repo.get("a")!!.metadata.familyName)
        // Now the given name alone can change.
        editCsv(repo) { rows -> rows.first { "a" in it }[header.indexOf("given_name")] = "한결" }
        repo.applyCsvFile()
        assertEquals("강한결", repo.get("a")!!.name)
    }

    @Test
    fun `a column added in the csv can be taken out again`() {
        val (_, repo) = open()
        repo.applyOps((1..3).map { register("p$it", "사람$it", faces.samples("p$it", 1)) }, "k")
        repo.applyCsvFile()
        editCsv(repo) { rows -> rows.forEachIndexed { i, row -> row.add(3, if (i == 0) "메모" else if (i == 1) "새가족" else "") } }
        repo.applyCsvFile()
        assertEquals("새가족", repo.get("p1")!!.metadata["메모"])
        editCsv(repo) { rows -> val memo = rows[0].indexOf("메모"); rows.forEach { it.removeAt(memo) } }
        repo.applyCsvFile()
        assertTrue(repo.list().all { p -> p.metadata.entries.keys.none { it == "메모" } })
    }

    @Test
    fun `a deleted or emptied csv deletes no one`() {
        val (_, repo) = open()
        repo.applyOps((1..5).map { register("p$it", "사람$it", faces.samples("p$it", 1)) }, "k")
        repo.applyCsvFile()
        csv(repo).delete()
        assertFalse(repo.applyCsvFile().found)
        assertEquals(5, repo.list().size)
        editCsv(repo) { rows -> while (rows.size > 1) rows.removeAt(1) }
        assertEquals(5, repo.applyCsvFile().changes.heldDeletions)
        assertEquals(5, repo.list().size)
    }

    @Test
    fun `korean excel's plain csv (cp949) is read`() {
        val text = "name,department\n홍길동,청년부\n"
        assertEquals(text, PeopleRepository.readCsv(text.toByteArray(charset("x-windows-949"))))
        assertEquals(text, PeopleRepository.readCsv(text.toByteArray()))
    }

    @Test
    fun `column names bump the revision and travel with a restore`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2))), "k")
        val rev = repo.revision
        repo.setColumnLabel("부서", "반")
        assertEquals(mapOf("department" to "반"), repo.columnLabels)
        assertEquals(rev + 1, repo.revision)
        repo.setColumnLabel("department", "반")
        assertEquals(rev + 1, repo.revision)
        val backup = File(temp.root, "backup.db")
        repo.backupTo(backup)
        repo.setColumnLabel("department", null)
        repo.delete("a")
        val restoredDir = temp.newFolder("restore")
        backup.copyTo(File(restoredDir, PeopleStore.DB_FILE))
        val copy = PeopleStore.open(restoredDir, JdbcSqlDatabase.FACTORY)
        val epoch = repo.epoch
        repo.restore(copy.loadPeople(), copy.loadTombstones(), com.allstar.faithid.core.sync.ColumnLabels.decode(copy.meta(PeopleRepository.META_COLUMN_LABELS)))
        copy.close()
        assertEquals("홍길동", repo.get("a")!!.name)
        assertFalse(repo.isDeleted("a"))
        assertEquals(mapOf("department" to "반"), repo.columnLabels)
        assertNotEquals(epoch, repo.epoch)
    }

    @Test
    fun `the name can be edited in its parts, a guessed split is not stored unchanged`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2))), "k")
        val rev = repo.revision
        assertTrue(repo.updateMetadata("a", mapOf("family_name" to "홍", "given_name" to "길동")))
        assertEquals(rev, repo.revision)
        assertEquals("", repo.get("a")!!.metadata.familyName)
        repo.updateMetadata("a", mapOf("family_name" to "홍", "given_name" to "길순"))
        assertEquals("홍길순", repo.get("a")!!.name)
        assertEquals("홍", repo.get("a")!!.metadata.familyName)
        assertFalse(repo.updateMetadata("a", mapOf("family_name" to "", "given_name" to "")))
    }

    @Test
    fun `editing to the same values does not create a revision`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2))), "k")
        val rev = repo.revision
        assertTrue(repo.updateMetadata("a", "홍길동", "청년부", "1학년"))
        assertEquals(rev, repo.revision)
        assertFalse(repo.updateMetadata("missing", "x", "", ""))
    }

    @Test
    fun `concurrent csv exports always leave a complete file`() {
        val (_, repo) = open()
        repo.applyOps((1..50).map { register("p$it", "사람$it", faces.samples("p$it", 1)) }, "k1")
        val threads = (1..4).map {
            Thread { repeat(50) { assertEquals(51, Csv.parse(repo.exportCsv(force = true)!!.readText()).size) } }
        }
        val errors = java.util.concurrent.CopyOnWriteArrayList<Throwable>()
        threads.forEach { t -> t.setUncaughtExceptionHandler { _, e -> errors += e }; t.start() }
        threads.forEach { it.join() }
        assertTrue(errors.toString(), errors.isEmpty())
    }
}
