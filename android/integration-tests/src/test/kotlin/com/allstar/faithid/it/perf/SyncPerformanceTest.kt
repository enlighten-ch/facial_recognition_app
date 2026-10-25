package com.allstar.faithid.it.perf

import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.ChangeSetCodec
import com.allstar.faithid.core.sync.Protocol
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.it.Compare
import com.allstar.faithid.it.SimKiosk
import com.allstar.faithid.it.waitUntil
import com.allstar.faithid.server.ServerConfig
import com.allstar.faithid.server.http.FaithServer
import okhttp3.OkHttpClient
import okhttp3.Request
import org.junit.After
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.net.ServerSocket
import java.time.ZoneId
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random

/**
 * Measures how fast changes travel through the system, with production sync timers (changes are
 * announced over the WebSocket; the 30 s poll is only a fallback):
 *
 * 1. a new kiosk downloading the whole database (download, decode, local apply), and several
 *    kiosks joining at the same time
 * 2. a registration on one kiosk until the server and every other kiosk have it
 * 3. an administrator edit or delete on the server until every kiosk has it (in-process server only;
 *    the Android server allows administration only from its screen)
 * 4. a check-in until the server has stored it
 * 5. server response time per API, idle and under load
 * 6. with -Pfaith.it.adbSerial: a registration until the real kiosk app on that device applied it
 *
 * Runs against an in-process server by default, or the Android server app with
 * -Pfaith.it.perfUrl=http://host:18080 (approve the kiosks in the app; tools/perf_android_server.sh
 * does that and restores the server data afterwards). Results go to build/perf/<target>.json.
 */
class SyncPerformanceTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val remoteUrl = System.getProperty("faith.it.perfUrl")?.takeIf { it.isNotBlank() }
    private val adbSerial = System.getProperty("faith.it.adbSerial")?.takeIf { it.isNotBlank() }
    private val peopleCount = System.getProperty("faith.it.perfPeople")?.toInt() ?: 1000
    private val samplesPerPerson = 12 // one registration
    private val kioskCount = 8
    private val loadSeconds = System.getProperty("faith.it.perfLoadSeconds")?.toLong() ?: 45L

    private val faces = FakeFaces(seed = 5, dim = 512)
    private val metrics = Metrics()
    private val notes = linkedMapOf<String, Any>()
    private val kiosks = mutableListOf<SimKiosk>()
    private var server: FaithServer? = null

    @Volatile
    private var phase = "setup"

    @After
    fun tearDown() {
        kiosks.forEach { runCatching { it.close() } }
        server?.close()
    }

    @Test
    fun `sync timings`() {
        val url = remoteUrl ?: startLocalServer()
        val runId = UUID.randomUUID().toString().take(6)
        notes["target"] = if (remoteUrl != null) "android-server $remoteUrl" else "in-process server"
        notes["people"] = peopleCount
        notes["samplesPerPerson"] = samplesPerPerson
        notes["kiosks"] = kioskCount

        // --- seed: one kiosk registers the whole congregation and uploads it -------------------
        val seeder = newKiosk("perf-$runId-seed", 1)
        approve(listOf(seeder))
        repeat(peopleCount) { i -> seeder.repo.register(PersonMetadata.of("성도$i", "부서${i % 12}", runId), faces.samples("seed-$i", samplesPerPerson)) }
        val seedStart = System.nanoTime()
        while (seeder.repo.pendingCounts().first > 0) seeder.engine.syncOnce()
        notes["seedUploadMs"] = Math.round(millisSince(seedStart))
        seeder.close()
        kiosks.remove(seeder)

        // --- join: approve the identities, then measure fresh kiosks with those identities ------
        val seeds = (0 until kioskCount).map { 100L + it }
        val ids = seeds.map { "perf-$runId-k${it - 100}" }
        val approved = ids.zip(seeds).map { (id, seed) -> newKiosk(id, seed) }
        approve(approved)
        approved.forEach { it.close() }
        kiosks.clear()

        phase = "join"
        measureDownloadBreakdown(url, ids[0], seeds[0])
        val fleet = ids.zip(seeds).map { (id, seed) -> newKiosk(id, seed) }
        fleet.map { kiosk ->
            Thread {
                val began = System.nanoTime()
                check(kiosk.engine.syncOnce() is SyncStatus.Online)
                metrics.add("join: $kioskCount kiosks download at once (each)", millisSince(began))
            }.apply { start() }
        }.forEach { it.join() }
        fleet.forEach { it.engine.start() }
        waitUntil(30_000, "fleet online") { fleet.all { it.engine.status is SyncStatus.Online } }
        Thread.sleep(1_000) // sockets open

        // --- idle: one change at a time --------------------------------------------------------
        phase = "idle"
        val random = Random(7)
        repeat(20) {
            val origin = fleet[random.nextInt(fleet.size)]
            measureRegistration(origin, fleet, "idle", runId)
            Thread.sleep(200)
        }
        repeat(20) {
            measureCheckIn(fleet[random.nextInt(fleet.size)], "idle")
            Thread.sleep(200)
        }
        server?.let { local ->
            repeat(10) { n ->
                val target = local.people.list().random(random)
                val name = "수정$n-$runId"
                val began = System.nanoTime()
                local.people.updateMetadata(target.key, name, target.department, target.grade)
                awaitAll(fleet, "idle: admin edit → all kiosks", began) { it.repo.get(target.key)?.name == name }
                Thread.sleep(200)
            }
            repeat(5) {
                val target = local.people.list().random(random)
                val began = System.nanoTime()
                local.people.delete(target.key)
                awaitAll(fleet, "idle: admin delete → all kiosks", began) { it.repo.get(target.key) == null }
                Thread.sleep(200)
            }
        }

        // --- load: every other kiosk busy, the first one probes --------------------------------
        phase = "load"
        val probe = fleet[0]
        val busy = fleet.drop(1)
        val stop = AtomicBoolean(false)
        val actions = java.util.concurrent.atomic.AtomicInteger()
        val workers = busy.mapIndexed { i, kiosk ->
            Thread {
                val r = Random(1000 + i)
                while (!stop.get()) {
                    Thread.sleep(r.nextLong(500, 1500))
                    val people = kiosk.repo.people()
                    when (r.nextInt(10)) {
                        in 0..6 -> people.randomOrNull(r)?.let { checkIn(kiosk, it.key, it.name) }
                        in 7..8 -> people.randomOrNull(r)?.let { kiosk.repo.addSamples(it.key, faces.samples(it.key, 2)) }
                        else -> kiosk.repo.register(PersonMetadata.of("부하${r.nextInt(1000)}", "부하", runId), faces.samples("load-$i-${r.nextInt()}", samplesPerPerson))
                    }
                    kiosk.engine.requestSync()
                    actions.incrementAndGet()
                }
            }.apply { start() }
        }
        val loadEnd = System.currentTimeMillis() + loadSeconds * 1000
        while (System.currentTimeMillis() < loadEnd) {
            measureRegistration(probe, fleet, "load", runId)
            measureCheckIn(probe, "load")
            Thread.sleep(300)
        }
        stop.set(true)
        workers.forEach { it.join() }
        notes["loadActionsPerSecond"] = Math.round(actions.get() * 10.0 / loadSeconds) / 10.0

        // --- the real kiosk app on the device ---------------------------------------------------
        phase = "device"
        adbSerial?.let { measureDeviceKiosk(DeviceKiosk(it), probe, runId) }

        // --- everyone converged ------------------------------------------------------------------
        phase = "settle"
        waitUntil(120_000, "fleet caught up") {
            fleet.forEach { it.engine.requestSync() }
            fleet.all { it.repo.pendingCounts() == (0 to 0) } &&
                fleet.map { it.repo.syncPosition().revision }.distinct().size == 1
        }
        val reference = fleet[0].repo.people()
        fleet.drop(1).forEach { assertNull("${it.clientId} differs", Compare.diff(reference, it.repo.people())) }
        server?.let { assertNull(Compare.diff(it.people.list(), reference)) }

        report()
        if (remoteUrl == null) {
            // Generous limits: a regression guard, not a benchmark target.
            assertTrue(metrics.get("idle: registration → all kiosks").percentile(95.0) < 2_000)
            assertTrue(metrics.get("idle: check-in → server stored").percentile(95.0) < 2_000)
            assertTrue(metrics.get("load: registration → all kiosks").percentile(95.0) < 5_000)
        }
    }

    // -----------------------------------------------------------------------------------------

    private fun startLocalServer(): String {
        val port = ServerSocket(0).use { it.localPort }
        val local = FaithServer.create(
            ServerConfig(port = port, dataDir = temp.newFolder("server-data"), stateDir = temp.newFolder("server-state"), zone = ZoneId.of("Asia/Seoul")),
            JdbcSqlDatabase.FACTORY,
        )
        check(local.start())
        server = local
        return "http://127.0.0.1:$port"
    }

    private fun newKiosk(id: String, seed: Long): SimKiosk =
        SimKiosk(
            id, temp.newFolder(), remoteUrl ?: "http://127.0.0.1:${server!!.config.port}", seed, 100,
            fastTimers = false,
            canApplyNow = { true },
            extraInterceptor = ApiTimer(metrics) { phase },
        ).also { kiosks += it }

    private fun approve(list: List<SimKiosk>) {
        list.forEach { it.engine.syncOnce() }
        server?.clients?.approveAllPending()
        if (remoteUrl != null) println("PERF: approve ${list.size} kiosks in the server app")
        waitUntil(180_000, "kiosks approved") { list.all { it.engine.syncOnce() is SyncStatus.Online } }
    }

    /** Download, decode and local apply of the full database, measured separately (median of 3). */
    private fun measureDownloadBreakdown(url: String, clientId: String, seed: Long) {
        val identity = SimKiosk(clientId, temp.newFolder(), url, seed, 100).also { it.close() }.identity
        val http = OkHttpClient.Builder().readTimeout(60, TimeUnit.SECONDS).build()
        repeat(3) {
            val request = Request.Builder().url("$url${Protocol.CHANGES}?${Protocol.PARAM_SINCE}=0")
                .header(Protocol.HEADER_CLIENT_ID, identity.clientId)
                .header(Protocol.HEADER_AUTHORIZATION, Protocol.BEARER + identity.secret)
                .build()
            var began = System.nanoTime()
            val bytes = http.newCall(request).execute().use { it.body!!.bytes() }
            metrics.add("join: full download over HTTP", millisSince(began))
            notes["fullDownloadBytes"] = bytes.size
            began = System.nanoTime()
            val changes = ChangeSetCodec.decode(bytes)
            metrics.add("join: decode", millisSince(began))
            LocalFaceRepository(PeopleStore.open(temp.newFolder(), JdbcSqlDatabase.FACTORY), 100).use { repo ->
                began = System.nanoTime()
                repo.applyServerChanges(changes)
                metrics.add("join: apply to local DB (SQLite)", millisSince(began))
            }
        }
    }

    private fun measureRegistration(origin: SimKiosk, fleet: List<SimKiosk>, phase: String, runId: String) {
        val began = System.nanoTime()
        val person = origin.repo.register(PersonMetadata.of("측정${Random.nextInt(100000)}", "측정", runId), faces.samples("probe-${UUID.randomUUID()}", samplesPerPerson))
        origin.engine.requestSync()
        awaitAll(fleet - origin, "$phase: registration → all kiosks", began) { it.repo.get(person.key) != null }
    }

    private fun measureCheckIn(origin: SimKiosk, phase: String) {
        val target = origin.repo.people().random()
        val began = System.nanoTime()
        val eventId = checkIn(origin, target.key, target.name)
        origin.engine.requestSync()
        val local = server
        waitUntil(30_000, "check-in stored") {
            if (local != null) local.attendance.eventsBetween(local.attendance.today(), local.attendance.today()).any { it.eventId == eventId }
            else origin.repo.pendingCounts().second == 0 // acknowledged = stored on the server
        }
        metrics.add("$phase: check-in → server stored", millisSince(began))
    }

    private fun checkIn(kiosk: SimKiosk, key: String, name: String): String {
        val event = AttendanceEvent(UUID.randomUUID().toString(), key, name, "", "", kiosk.clientId, System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.8f)
        kiosk.repo.recordAttendance(event)
        return event.eventId
    }

    /** Records each kiosk's arrival time and the time until the last one had the change. */
    private fun awaitAll(targets: List<SimKiosk>, name: String, began: Long, arrived: (SimKiosk) -> Boolean) {
        val pending = targets.toMutableSet()
        val deadline = System.nanoTime() + 60_000_000_000
        while (pending.isNotEmpty()) {
            check(System.nanoTime() < deadline) { "$name: ${pending.size} kiosks never got the change" }
            pending.removeAll { kiosk -> arrived(kiosk).also { if (it) metrics.add("$name (each)", millisSince(began)) } }
            if (pending.isNotEmpty()) Thread.sleep(5)
        }
        metrics.add(name, millisSince(began))
    }

    private fun measureDeviceKiosk(device: DeviceKiosk, origin: SimKiosk, runId: String) {
        notes["deviceClockOffsetMs"] = device.clockOffsetMillis
        device.bringToFront()
        device.pauseCamera(true)
        try {
            Thread.sleep(2_000)
            repeat(10) {
                val began = System.currentTimeMillis()
                origin.repo.register(PersonMetadata.of("기기측정$it", "측정", runId), faces.samples("device-$runId-$it", samplesPerPerson))
                origin.engine.requestSync()
                val applied = device.waitForLog("서버 변경", began)
                if (applied != null) metrics.add("device: registration → real kiosk app applied", (applied - began).toDouble())
                else metrics.add("device: not applied within 30 s", 30_000.0)
                Thread.sleep(1_500)
            }
        } finally {
            device.pauseCamera(false)
        }
    }

    private fun report() {
        val lines = mutableListOf<String>()
        notes.forEach { (k, v) -> lines += "  \"$k\": ${json(v)}" }
        metrics.names().forEach { name -> lines += "  ${json(name)}: ${json(metrics.get(name).summary())}" }
        val target = if (remoteUrl != null) "android" else "local"
        val file = File("build/perf/perf-$target.json").apply { parentFile.mkdirs() }
        file.writeText("{\n" + lines.joinToString(",\n") + "\n}\n")
        println("PERF results → ${file.absolutePath}")
        notes.forEach { (k, v) -> println("PERF  %-28s %s".format(k, v)) }
        metrics.names().forEach { name ->
            val s = metrics.get(name).summary()
            println("PERF  %-52s n=%-4s p50=%-8s p95=%-8s max=%s".format(name, s["n"], s["p50"], s["p95"], s["max"]))
        }
    }

    private fun json(v: Any?): String = when (v) {
        is String -> "\"" + v.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        is Map<*, *> -> v.entries.joinToString(", ", "{", "}") { "${json(it.key.toString())}: ${json(it.value)}" }
        null -> "null"
        else -> v.toString()
    }
}
