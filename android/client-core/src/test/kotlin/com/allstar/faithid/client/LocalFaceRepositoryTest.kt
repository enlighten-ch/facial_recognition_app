package com.allstar.faithid.client

import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.EmbeddingSets
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.ChangeSet
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.Tombstone
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class LocalFaceRepositoryTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val faces = FakeFaces(seed = 31, dim = 32)
    private lateinit var dir: File

    private fun open(): LocalFaceRepository {
        if (!::dir.isInitialized) dir = temp.newFolder("kiosk")
        return LocalFaceRepository(PeopleStore.open(dir, JdbcSqlDatabase.FACTORY))
    }

    @Test
    fun `registration is usable immediately and queued for the server`() {
        val repo = open()
        var indexSize = -1
        repo.addIndexListener { indexSize = it.size }
        val person = repo.register(PersonMetadata.of("홍길동", "청년부", "1학년"), faces.samples("hong", 4))
        assertEquals(1, indexSize)
        assertEquals(person.key, repo.index.best(faces.sample("hong"))!!.personKey)
        val op = repo.pendingOps().single() as PersonOp.Register
        assertEquals(person.key, op.personKey)
        repo.addSamples(person.key, faces.samples("hong", 2))
        assertEquals(6, repo.get(person.key)!!.embeddings.size)
        assertEquals(2, repo.pendingOps().size)
    }

    @Test
    fun `queued changes and check-ins survive a restart`() {
        val first = open()
        val person = first.register(PersonMetadata.of("A", "", ""), faces.samples("a", 2))
        first.recordAttendance(AttendanceEvent("e1", person.key, "A", "", "", "k", 1, CheckInSource.RECOGNIZED, 0.7f))
        first.close()
        val second = open()
        assertEquals(1, second.pendingOps().size)
        assertEquals("e1", second.pendingAttendance().single().eventId)
        assertNotNull(second.get(person.key))
    }

    @Test
    fun `server changes are applied and pending changes stay on top`() {
        val repo = open()
        val local = repo.register(PersonMetadata.of("Local", "", ""), faces.samples("local", 2))
        val serverPerson = faces.person("Server", key = "srv").copy(revision = 5)
        val changes = ChangeSet("e", 5, fullSnapshot = true, persons = listOf(serverPerson), tombstones = emptyList())
        repo.applyServerChanges(changes)
        // The full snapshot did not contain the unsent registration, but it is still here.
        assertNotNull(repo.get(local.key))
        assertNotNull(repo.get("srv"))
        assertEquals(LocalFaceRepository.SyncPosition("e", 5), repo.syncPosition())

        val extra = faces.samples("srv", 2)
        repo.addSamples("srv", extra)
        val serverUpdate = serverPerson.copy(metadata = serverPerson.metadata.with(name = "Renamed"), revision = 6)
        repo.applyServerChanges(ChangeSet("e", 6, false, listOf(serverUpdate), emptyList()))
        val merged = repo.get("srv")!!
        assertEquals("Renamed", merged.name)
        assertTrue(EmbeddingSets.containsAll(merged.embeddings, extra))
    }

    @Test
    fun `tombstones remove people even with pending samples`() {
        val repo = open()
        repo.applyServerChanges(ChangeSet("e", 1, true, listOf(faces.person("A", key = "a").copy(revision = 1)), emptyList()))
        repo.addSamples("a", faces.samples("a", 1))
        repo.applyServerChanges(ChangeSet("e", 2, false, emptyList(), listOf(Tombstone("a", 2))))
        assertNull(repo.get("a"))
        assertTrue(repo.index.isEmpty())
    }

    @Test
    fun `rejected registration is dropped locally, applied one stays`() {
        val repo = open()
        val rejected = repo.register(PersonMetadata.of("A", "", ""), faces.samples("a", 2))
        val accepted = repo.register(PersonMetadata.of("B", "", ""), faces.samples("b", 2))
        val ops = repo.pendingOps().associateBy { it.personKey }
        repo.acknowledgeOps(mapOf(ops.getValue(rejected.key).opId to OpResult.REJECTED_INVALID, ops.getValue(accepted.key).opId to OpResult.APPLIED))
        assertNull(repo.get(rejected.key))
        assertNotNull(repo.get(accepted.key))
        assertTrue(repo.pendingOps().isEmpty())
    }
}
