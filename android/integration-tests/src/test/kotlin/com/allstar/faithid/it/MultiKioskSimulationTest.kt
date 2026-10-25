package com.allstar.faithid.it

import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.server.ServerConfig
import com.allstar.faithid.server.ServerLog
import com.allstar.faithid.server.http.FaithServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.net.ServerSocket
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

/**
 * Many kiosks working against one server at the same time, in no particular order, over an
 * unreliable network, while the administrator edits, deletes, revokes and restarts the server.
 * Afterwards every kiosk must hold exactly the server's data, and nothing may be lost, duplicated
 * or resurrected.
 */
class MultiKioskSimulationTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val zone = ZoneId.of("Asia/Seoul")
    private val serverLog = StringBuffer()
    private lateinit var server: FaithServer
    private val kiosks = mutableListOf<SimKiosk>()
    private var port = 0

    private fun startServer(maxEmbeddings: Int) {
        port = ServerSocket(0).use { it.localPort }
        server = FaithServer.create(
            ServerConfig(port = port, dataDir = temp.newFolder("faceDB"), stateDir = temp.newFolder("state"), maxEmbeddingsPerPerson = maxEmbeddings, zone = zone),
            JdbcSqlDatabase.FACTORY,
            log = ServerLog { serverLog.append(it).append('\n') },
        )
        check(server.start())
    }

    @After
    fun tearDown() {
        kiosks.forEach { runCatching { it.close() } }
        runCatching { server.close() }
    }

    /** What the kiosks did, recorded independently of the system under test. */
    private class Truth {
        val registered = ConcurrentHashMap<String, String>() // key -> kiosk
        val samples = ConcurrentHashMap<String, MutableSet<Long>>() // key -> fingerprints sent
        val events = ConcurrentHashMap.newKeySet<String>()
        val deleted = ConcurrentHashMap.newKeySet<String>()
        val lastEdit = ConcurrentHashMap<String, PersonMetadata>()

        fun addSamples(key: String, list: List<FloatArray>) {
            samples.computeIfAbsent(key) { ConcurrentHashMap.newKeySet() }.addAll(list.map { Vectors.fingerprint(it) })
        }
    }

    private fun runScenario(
        kioskCount: Int,
        maxEmbeddings: Int,
        durationMs: Long,
        seed: Long,
    ): Truth {
        startServer(maxEmbeddings)
        val faces = FakeFaces(seed, dim = 64)
        val truth = Truth()
        repeat(kioskCount) { i ->
            kiosks += SimKiosk("kiosk-$i", temp.newFolder("kiosk-$i"), "http://127.0.0.1:$port", seed + i, maxEmbeddings)
        }
        kiosks.forEach { it.engine.start() }

        val stop = AtomicBoolean(false)
        val failures = ConcurrentHashMap.newKeySet<Throwable>()
        val actions = AtomicInteger()
        val identities = ConcurrentHashMap<String, String>() // person key -> fake identity
        val adminIterations = AtomicInteger()

        val kioskThreads = kiosks.mapIndexed { i, kiosk ->
            Thread {
                val random = Random(seed * 31 + i)
                try {
                    while (!stop.get()) {
                        val people = kiosk.repo.people()
                        when (random.nextInt(100)) {
                            in 0..17 -> {
                                val identity = "id-${UUID.randomUUID()}"
                                val samples = faces.samples(identity, 4 + random.nextInt(9))
                                val name = listOf("김철수", "이영희", "박민준", "홍길동")[random.nextInt(4)] // namesakes on purpose
                                val person = kiosk.repo.register(PersonMetadata.of(name, "부서${random.nextInt(3)}", "${random.nextInt(6) + 1}학년"), samples)
                                identities[person.key] = identity
                                truth.registered[person.key] = kiosk.clientId
                                truth.addSamples(person.key, samples)
                            }
                            in 18..54 -> if (people.isNotEmpty()) {
                                val target = people[random.nextInt(people.size)]
                                val samples = faces.samples(identities[target.key] ?: target.key, 1 + random.nextInt(3))
                                if (kiosk.repo.addSamples(target.key, samples)) truth.addSamples(target.key, samples)
                            }
                            in 55..84 -> {
                                val target = people.randomOrNull(random)
                                val event = if (target == null || random.nextInt(8) == 0) {
                                    AttendanceEvent(UUID.randomUUID().toString(), "", "방문자${random.nextInt(5)}", "", "", kiosk.clientId, System.currentTimeMillis(), CheckInSource.LABEL_ONLY, 0f)
                                } else {
                                    AttendanceEvent(UUID.randomUUID().toString(), target.key, target.name, target.department, target.grade, kiosk.clientId, System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.7f)
                                }
                                kiosk.repo.recordAttendance(event)
                                truth.events += event.eventId
                            }
                            in 85..92 -> kiosk.network.offline = !kiosk.network.offline
                            else -> {
                                kiosk.network.dropRequestRate = random.nextDouble(0.0, 0.25)
                                kiosk.network.dropResponseRate = random.nextDouble(0.0, 0.25)
                            }
                        }
                        kiosk.engine.requestSync()
                        actions.incrementAndGet()
                        Thread.sleep(random.nextLong(5, 60))
                    }
                } catch (t: Throwable) {
                    failures += t
                }
            }.apply { name = "actor-${kiosk.clientId}"; start() }
        }

        val admin = Thread {
            val random = Random(seed * 17)
            var restarted = false
            var revoked = false
            val started = System.currentTimeMillis()
            try {
                while (!stop.get()) {
                    val elapsed = System.currentTimeMillis() - started
                    // Approve gradually, like an administrator walking from kiosk to kiosk.
                    server.clients.list().filter { !it.approved }.firstOrNull()?.let { server.clients.approve(it.clientId) }
                    val people = server.people.list()
                    when (random.nextInt(10)) {
                        0, 1 -> people.randomOrNull(random)?.let { p ->
                            val edited = p.metadata.with(name = p.name + "*", department = "편집부${random.nextInt(3)}")
                            if (server.people.updateMetadata(p.key, edited.name, edited.department, edited.grade)) truth.lastEdit[p.key] = server.people.get(p.key)?.metadata ?: edited
                        }
                        2 -> people.randomOrNull(random)?.let { p ->
                            if (server.people.delete(p.key)) truth.deleted += p.key
                        }
                        3 -> {
                            // Spreadsheet round trip: change one grade in face_db.csv and re-import it.
                            val table = Csv.parse(server.people.exportCsv().readText()).toMutableList()
                            if (table.size > 1) {
                                val gradeColumn = table[0].indexOf("grade")
                                val row = 1 + random.nextInt(table.size - 1)
                                table[row] = table[row].toMutableList().also { it[gradeColumn] = "CSV${random.nextInt(9)}" }
                                val result = server.people.importCsvEdits(Csv.write(table))
                                result.changed.forEach { truth.lastEdit[it.key] = it.metadata }
                            }
                        }
                        else -> Unit
                    }
                    if (!revoked && elapsed > durationMs / 3) {
                        revoked = true
                        server.revokeClient("kiosk-0")
                    }
                    if (!restarted && elapsed > durationMs / 2) {
                        restarted = true
                        server.stop()
                        Thread.sleep(400)
                        check(server.start())
                    }
                    adminIterations.incrementAndGet()
                    Thread.sleep(random.nextLong(30, 150))
                }
            } catch (t: Throwable) {
                failures += t
            }
        }.apply { name = "admin"; start() }

        Thread.sleep(durationMs)
        stop.set(true)
        (kioskThreads + admin).forEach { it.join(10_000) }
        if (failures.isNotEmpty()) throw AssertionError("actor failed", failures.first())

        // Quiesce: network healed, everyone approved, let all kiosks catch up.
        kiosks.forEach { it.network.heal() }
        server.clients.approveAllPending()
        waitUntil(90_000, "all kiosks caught up with the server") {
            kiosks.forEach { it.engine.requestSync() }
            kiosks.all { it.isCaughtUp(server.people.epoch, server.people.revision) }
        }
        kiosks.forEach { it.engine.stop() }

        val dropped = kiosks.sumOf { it.network.droppedRequests.get() } to kiosks.sumOf { it.network.droppedResponses.get() }
        println(
            "scenario kiosks=$kioskCount actions=${actions.get()} registered=${truth.registered.size} " +
                "deleted=${truth.deleted.size} edits=${truth.lastEdit.size} checkIns=${truth.events.size} " +
                "adminIterations=${adminIterations.get()} droppedRequests=${dropped.first} lostResponses=${dropped.second} revision=${server.people.revision}",
        )
        return truth
    }

    private fun verifyConvergence(truth: Truth, everySampleKept: Boolean) {
        val serverPeople = server.people.list()
        // 1. Every kiosk holds exactly the server's data.
        kiosks.forEach { kiosk ->
            val diff = Compare.diff(serverPeople, kiosk.repo.people())
            assertNull("kiosk ${kiosk.clientId} differs from the server:\n$diff", diff)
        }
        val serverKeys = serverPeople.map { it.key }.toSet()
        // 2. Every registration that was not deleted reached the server.
        val lost = truth.registered.keys - truth.deleted - serverKeys
        assertTrue("registrations lost: $lost", lost.isEmpty())
        // 3. Deleted people never come back.
        val resurrected = truth.deleted.intersect(serverKeys)
        assertTrue("deleted people came back: $resurrected", resurrected.isEmpty())
        // 4. Nothing appears that no kiosk registered.
        assertTrue(serverKeys.all { truth.registered.containsKey(it) })
        // 5. No sample is lost (when the per-person limit is not reached).
        if (everySampleKept) {
            serverPeople.forEach { person ->
                val stored = person.embeddings.map { Vectors.fingerprint(it) }.toSet()
                val sent = truth.samples[person.key].orEmpty()
                val missing = sent - stored
                assertTrue("samples lost for ${person.key}: ${missing.size} of ${sent.size}", missing.isEmpty())
            }
        }
        // 6. The administrator's last edit wins over kiosk uploads.
        truth.lastEdit.forEach { (key, metadata) ->
            val person = serverPeople.firstOrNull { it.key == key } ?: return@forEach
            assertEquals("edit of $key", metadata, person.metadata)
        }
        // 7. Every check-in is stored exactly once.
        val stored = server.attendance.eventsBetween(LocalDate.now(zone).minusDays(1), LocalDate.now(zone).plusDays(1))
        assertEquals("check-ins stored", truth.events.size, stored.size)
        assertEquals(truth.events, stored.map { it.eventId }.toSet())
        // 8. The daily sheet agrees with the stored events.
        val sheet = Csv.parse(server.attendance.writeDailyCsv(LocalDate.now(zone)).readText())
        val counted = sheet.drop(2).sumOf { it[6].toIntOrNull() ?: 0 }
        val today = stored.count { java.time.Instant.ofEpochMilli(it.occurredAt).atZone(zone).toLocalDate() == LocalDate.now(zone) }
        assertEquals("check-ins counted in today's sheet", today, counted)
    }

    @Test
    fun `eight kiosks, flaky network, admin activity and a server restart converge`() {
        val seed = System.getProperty("faith.it.seed")?.toLongOrNull() ?: 1L
        val truth = runScenario(kioskCount = 8, maxEmbeddings = 10_000, durationMs = 20_000, seed = seed)
        verifyConvergence(truth, everySampleKept = true)
        val revokedKiosk = kiosks.first { it.clientId == "kiosk-0" }
        assertTrue("revoked kiosk had to wait for approval again", revokedKiosk.statuses.any { it is SyncStatus.AwaitingApproval })
        assertTrue(truth.deleted.isNotEmpty())
        assertTrue(truth.lastEdit.isNotEmpty())
    }

    @Test
    fun `kiosks agree with the server when samples are pruned to a small limit`() {
        val truth = runScenario(kioskCount = 6, maxEmbeddings = 8, durationMs = 12_000, seed = 2)
        verifyConvergence(truth, everySampleKept = false)
        assertTrue(server.people.list().all { it.embeddings.size <= 8 })
    }

    @Test
    fun `twenty kiosks start at once, download a large database and register simultaneously`() {
        startServer(maxEmbeddings = 100)
        val faces = FakeFaces(seed = 3, dim = 512)
        // Seed the server with 200 people x 20 samples (realistic Buffalo-L dimension).
        val seedOps = (1..200).map { n ->
            com.allstar.faithid.core.sync.PersonOp.Register("seed-$n", "seed-$n", PersonMetadata.of("교인$n", "부서${n % 7}", "${n % 6 + 1}학년"), faces.samples("seed-$n", 20), 1)
        }
        seedOps.chunked(20).forEach { server.people.applyOps(it, "seed") }

        repeat(20) { i -> kiosks += SimKiosk("herd-$i", temp.newFolder("herd-$i"), "http://127.0.0.1:$port", 100L + i, 100) }
        val ready = CountDownLatch(1)
        val registeredKeys = ConcurrentHashMap.newKeySet<String>()
        val threads = kiosks.map { kiosk ->
            Thread {
                ready.await()
                kiosk.engine.syncOnce() // asks for approval
            }.apply { start() }
        }
        ready.countDown()
        threads.forEach { it.join() }
        assertEquals(20, server.clients.approveAllPending())

        val start = CountDownLatch(1)
        val workers = kiosks.map { kiosk ->
            Thread {
                start.await()
                assertTrue(kiosk.engine.syncOnce() is SyncStatus.Online) // full download, all at once
                val person = kiosk.repo.register(PersonMetadata.of("동시등록", kiosk.clientId, ""), faces.samples("new-${kiosk.clientId}", 12))
                registeredKeys += person.key
                kiosk.engine.syncOnce()
            }.apply { start() }
        }
        val began = System.currentTimeMillis()
        start.countDown()
        workers.forEach { it.join(120_000) }
        println("herd: 20 full downloads of 200 people + 20 registrations in ${System.currentTimeMillis() - began} ms")
        kiosks.forEach { it.engine.syncOnce() }
        assertEquals(220, server.people.list().size)
        assertTrue(registeredKeys.all { server.people.get(it) != null })
        kiosks.forEach { kiosk -> assertNull(Compare.diff(server.people.list(), kiosk.repo.people())) }
    }

    @Test
    fun `lost responses never duplicate registrations or check-ins`() {
        startServer(maxEmbeddings = 100)
        val faces = FakeFaces(seed = 4, dim = 64)
        repeat(4) { i -> kiosks += SimKiosk("lossy-$i", temp.newFolder("lossy-$i"), "http://127.0.0.1:$port", 200L + i, 100) }
        kiosks.forEach { it.engine.syncOnce() }
        server.clients.approveAllPending()
        val events = mutableSetOf<String>()
        kiosks.forEach { kiosk ->
            kiosk.network.dropResponseRate = 0.4
            repeat(10) { n ->
                val person = kiosk.repo.register(PersonMetadata.of("p$n", kiosk.clientId, ""), faces.samples("${kiosk.clientId}-$n", 5))
                val event = AttendanceEvent(UUID.randomUUID().toString(), person.key, person.name, person.department, "", kiosk.clientId, System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.9f)
                kiosk.repo.recordAttendance(event)
                events += event.eventId
            }
        }
        // Keep syncing through the losses until everything is acknowledged.
        waitUntil(60_000, "outboxes drained despite lost responses") {
            kiosks.forEach { it.engine.syncOnce() }
            kiosks.all { it.repo.pendingCounts() == (0 to 0) }
        }
        kiosks.forEach { it.network.heal(); it.engine.syncOnce() }
        assertTrue(kiosks.sumOf { it.network.droppedResponses.get() } > 0)
        assertEquals(40, server.people.list().size)
        val stored = server.attendance.eventsBetween(LocalDate.now(zone).minusDays(1), LocalDate.now(zone).plusDays(1))
        assertEquals(events, stored.map { it.eventId }.toSet())
        assertEquals(40, stored.size)
        kiosks.forEach { kiosk -> assertNull(Compare.diff(server.people.list(), kiosk.repo.people())) }
    }
}

private fun <T> List<T>.randomOrNull(random: Random): T? = if (isEmpty()) null else this[random.nextInt(size)]
