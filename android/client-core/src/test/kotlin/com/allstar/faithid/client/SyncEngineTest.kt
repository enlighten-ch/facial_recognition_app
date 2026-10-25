package com.allstar.faithid.client

import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.client.sync.KioskIdentity
import com.allstar.faithid.client.sync.SyncConfig
import com.allstar.faithid.client.sync.SyncEngine
import com.allstar.faithid.client.sync.SyncListener
import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.server.ServerConfig
import com.allstar.faithid.server.http.FaithServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.net.ServerSocket
import java.time.ZoneId
import java.util.concurrent.CopyOnWriteArrayList

class SyncEngineTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val faces = FakeFaces(seed = 41, dim = 32)
    private var port = 0
    private lateinit var server: FaithServer
    private val engines = mutableListOf<SyncEngine>()

    @Before
    fun setUp() {
        port = ServerSocket(0).use { it.localPort }
        server = FaithServer.create(
            ServerConfig(port = port, dataDir = temp.newFolder("server-data"), stateDir = temp.newFolder("server-state"), zone = ZoneId.of("Asia/Seoul")),
            JdbcSqlDatabase.FACTORY,
        )
        assertTrue(server.start())
    }

    @After
    fun tearDown() {
        engines.forEach { it.stop() }
        server.close()
    }

    private inner class Kiosk(
        id: String,
        url: String = "http://127.0.0.1:$port",
        applyIdleWaitMillis: Long = 0,
        canApplyNow: () -> Boolean = { true },
    ) {
        val statuses = CopyOnWriteArrayList<SyncStatus>()
        val logs = CopyOnWriteArrayList<String>()
        val repo = LocalFaceRepository(PeopleStore.open(temp.newFolder(id), JdbcSqlDatabase.FACTORY))
        val engine = SyncEngine(
            SyncConfig(url, KioskIdentity(id, "secret-for-$id-0123456789", "Tab $id"), connectTimeoutMillis = 1_000, applyIdleWaitMillis = applyIdleWaitMillis),
            repo,
            object : SyncListener {
                override fun onStatus(status: SyncStatus) { statuses += status }
                override fun onLog(message: String) { logs += message }
            },
            canApplyNow,
        ).also { engines += it }

        fun sync() = engine.syncOnce()
    }

    @Test
    fun `waits for approval, then downloads, uploads and converges`() {
        val kiosk = Kiosk("kiosk-a")
        val pending = kiosk.sync()
        assertTrue(pending is SyncStatus.AwaitingApproval)
        assertEquals("kiosk-a", (pending as SyncStatus.AwaitingApproval).clientId)

        server.clients.approve("kiosk-a")
        val person = kiosk.repo.register(PersonMetadata.of("홍길동", "청년부", "3학년"), faces.samples("hong", 4))
        kiosk.repo.recordAttendance(AttendanceEvent("e1", person.key, "홍길동", "청년부", "3학년", "kiosk-a", System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.8f))
        val online = kiosk.sync()
        assertTrue(online is SyncStatus.Online)
        assertEquals(0, (online as SyncStatus.Online).pendingChanges)
        assertEquals(0, online.pendingCheckIns)
        assertNotNull(server.people.get(person.key))
        assertEquals(1, server.attendance.eventsBetween(server.attendance.today(), server.attendance.today()).size)
        assertEquals(server.people.revision, kiosk.repo.syncPosition().revision)
        assertTrue(kiosk.repo.get(person.key)!!.revision > 0)
    }

    @Test
    fun `changes from one kiosk and the administrator reach the other kiosk`() {
        val a = Kiosk("kiosk-a")
        val b = Kiosk("kiosk-b")
        a.sync(); b.sync()
        server.clients.approveAllPending()
        val person = a.repo.register(PersonMetadata.of("김철수", "중등부", "1학년"), faces.samples("kim", 3))
        a.sync()
        b.sync()
        assertEquals("김철수", b.repo.get(person.key)!!.name)

        server.people.updateMetadata(person.key, "김철수", "고등부", "1학년")
        b.sync()
        assertEquals("고등부", b.repo.get(person.key)!!.department)

        server.people.delete(person.key)
        a.repo.addSamples(person.key, faces.samples("kim", 1)) // kiosk a has not heard of the deletion yet
        a.sync()
        b.sync()
        assertNull(a.repo.get(person.key))
        assertNull(b.repo.get(person.key))
        assertNull(server.people.get(person.key))
    }

    @Test
    fun `offline changes are kept and sent after the server comes back`() {
        val kiosk = Kiosk("kiosk-a")
        kiosk.sync()
        server.clients.approve("kiosk-a")
        kiosk.sync()
        server.stop()
        val person = kiosk.repo.register(PersonMetadata.of("오프라인", "", ""), faces.samples("off", 3))
        val offline = kiosk.sync()
        assertTrue(offline is SyncStatus.Offline)
        assertNotNull("still recognizable while offline", kiosk.repo.index.best(faces.sample("off")))
        assertTrue(server.start())
        assertTrue(kiosk.sync() is SyncStatus.Online)
        assertNotNull(server.people.get(person.key))
    }

    @Test
    fun `revoked kiosk goes back to waiting for approval and a stolen id is rejected`() {
        val kiosk = Kiosk("kiosk-a")
        kiosk.sync()
        server.clients.approve("kiosk-a")
        assertTrue(kiosk.sync() is SyncStatus.Online)
        server.revokeClient("kiosk-a")
        assertTrue(kiosk.sync() is SyncStatus.AwaitingApproval)

        val impostor = SyncEngine(
            SyncConfig("http://127.0.0.1:$port", KioskIdentity("kiosk-a", "another-secret-0123456789", "laptop")),
            LocalFaceRepository(PeopleStore.open(temp.newFolder("impostor"), JdbcSqlDatabase.FACTORY)),
            object : SyncListener {},
        )
        assertTrue(impostor.syncOnce() is SyncStatus.Rejected)
    }

    @Test
    fun `unreachable server reports offline instead of failing`() {
        val dead = ServerSocket(0).use { it.localPort }
        val kiosk = Kiosk("kiosk-x", url = "127.0.0.1:$dead")
        val status = kiosk.sync()
        assertTrue(status is SyncStatus.Offline)
        assertTrue(Kiosk("kiosk-y", url = "").sync() is SyncStatus.Offline)
    }

    @Test
    fun `background loop reacts to server notifications`() {
        val a = Kiosk("kiosk-a")
        a.sync()
        server.clients.approve("kiosk-a")
        a.engine.start()
        waitUntil { a.engine.status is SyncStatus.Online }
        val person = server.people.applyOps(
            listOf(com.allstar.faithid.core.sync.PersonOp.Register("op", "p_admin", PersonMetadata.of("관리자등록", "", ""), faces.samples("adm", 2), 1)),
            "test",
        )
        assertEquals(1, person.results.size)
        waitUntil { a.repo.get("p_admin") != null }
    }

    @Test
    fun `check-ins are sent while applying server changes waits for the camera to be free`() {
        val faceInFront = java.util.concurrent.atomic.AtomicBoolean(false)
        val kiosk = Kiosk("kiosk-a", applyIdleWaitMillis = 20_000, canApplyNow = { !faceInFront.get() })
        kiosk.sync()
        server.clients.approve("kiosk-a")
        kiosk.engine.start()
        waitUntil { kiosk.engine.status is SyncStatus.Online }

        faceInFront.set(true)
        server.people.applyOps(
            listOf(com.allstar.faithid.core.sync.PersonOp.Register("op", "p_admin", PersonMetadata.of("관리자등록", "", ""), faces.samples("adm", 2), 1)),
            "test",
        )
        Thread.sleep(1_000) // the notified cycle is now waiting to apply the download
        kiosk.repo.recordAttendance(AttendanceEvent("e1", "p_admin", "관리자등록", "", "", "kiosk-a", System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.8f))
        waitUntil(5_000) { server.attendance.eventsBetween(server.attendance.today(), server.attendance.today()).size == 1 }
        assertNull("not applied while a face is in front", kiosk.repo.get("p_admin"))

        faceInFront.set(false)
        waitUntil(5_000) { kiosk.repo.get("p_admin") != null }
    }

    @Test
    fun `a reloaded server database reaches running kiosks at once`() {
        val a = Kiosk("kiosk-a")
        a.sync()
        server.clients.approve("kiosk-a")
        a.engine.start()
        waitUntil { a.engine.status is SyncStatus.Online }
        Thread.sleep(500) // update socket open
        val before = a.repo.syncPosition()
        server.people.reload() // new epoch, same revision (e.g. a backup was put back)
        assertEquals(before.revision, server.people.revision)
        waitUntil(5_000) { a.repo.syncPosition().epoch == server.people.epoch } // the fallback poll is 30 s
    }

    @Test
    fun `an update socket that keeps closing does not make the kiosk sync in a tight loop`() {
        val a = Kiosk("kiosk-a")
        a.sync()
        server.clients.approve("kiosk-a")
        a.engine.start()
        waitUntil { a.engine.status is SyncStatus.Online }
        Thread.sleep(500)
        // A server going down closes the socket while HTTP still answers for a moment.
        val before = a.statuses.size
        val end = System.currentTimeMillis() + 3_000
        while (System.currentTimeMillis() < end) {
            server.updates.disconnect("kiosk-a")
            Thread.sleep(20)
        }
        val cycles = a.statuses.size - before
        println("socket churn: $cycles sync cycles in 3 s")
        assertTrue("$cycles sync cycles in 3 s", cycles <= 10)
    }

    @Test
    fun `an empty server database is not downloaded again on every sync`() {
        val a = Kiosk("kiosk-a")
        a.sync()
        server.clients.approve("kiosk-a")
        a.sync()
        a.sync()
        a.sync()
        assertEquals(a.logs.toString(), 1, a.logs.count { it.startsWith("서버 DB 전체 수신") })
    }

    private fun waitUntil(timeoutMs: Long = 10_000, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (!condition()) {
            if (System.currentTimeMillis() > deadline) throw AssertionError("condition not met within $timeoutMs ms")
            Thread.sleep(50)
        }
    }
}
