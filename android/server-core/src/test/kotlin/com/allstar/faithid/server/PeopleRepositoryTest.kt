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

    @Test
    fun `csv export and spreadsheet edits round trip`() {
        val (_, repo) = open()
        repo.applyOps(listOf(register("a", "홍길동", faces.samples("a", 2)), register("b", "김철수", faces.samples("b", 2))), "k")
        val file = repo.exportCsv()
        val table = Csv.parse(file.readText()).toMutableList()
        val nameColumn = table[0].indexOf("name")
        val row = table.indexOfFirst { it.contains("a") && it[nameColumn] == "홍길동" }
        table[row] = table[row].toMutableList().also { it[nameColumn] = "홍길순" }
        val before = repo.revision
        val result = repo.importCsvEdits(Csv.write(table))
        assertEquals(1, result.changed.size)
        assertEquals("홍길순", repo.get("a")!!.name)
        assertEquals(before + 1, repo.revision)
        assertEquals(before + 1, repo.get("a")!!.revision)
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
            Thread { repeat(50) { assertEquals(51, Csv.parse(repo.exportCsv().readText()).size) } }
        }
        val errors = java.util.concurrent.CopyOnWriteArrayList<Throwable>()
        threads.forEach { t -> t.setUncaughtExceptionHandler { _, e -> errors += e }; t.start() }
        threads.forEach { it.join() }
        assertTrue(errors.toString(), errors.isEmpty())
    }
}
