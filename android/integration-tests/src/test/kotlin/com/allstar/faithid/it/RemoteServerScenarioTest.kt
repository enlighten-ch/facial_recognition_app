package com.allstar.faithid.it

import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.After
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random

/**
 * The multi-kiosk scenario against a real Faith DB Server app running on an Android device.
 *
 * Run with `-Pfaith.it.serverUrl=http://127.0.0.1:18080` (adb forward) while the server app is
 * running. The kiosks wait until they are approved in the server app (tools/e2e_server_scenario.sh
 * taps "대기 중 전체 승인" through adb). Check-ins are verified by the script afterwards from the
 * attendance sheet on the device; the expected count is written to build/remote-scenario.properties.
 */
class RemoteServerScenarioTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val serverUrl = System.getProperty("faith.it.serverUrl")
    private val kiosks = mutableListOf<SimKiosk>()

    @After
    fun tearDown() = kiosks.forEach { runCatching { it.close() } }

    @Test
    fun `six kiosks against the Android server converge`() {
        assumeTrue("set -Pfaith.it.serverUrl to run against a device", !serverUrl.isNullOrBlank())
        val runId = UUID.randomUUID().toString().take(6)
        val faces = FakeFaces(seed = 77, dim = 512)
        repeat(6) { i -> kiosks += SimKiosk("remote-$runId-$i", temp.newFolder("k$i"), serverUrl!!, 300L + i, 100, fastTimers = true) }

        println("REMOTE: waiting for approval of ${kiosks.size} kiosks (run $runId)")
        kiosks.forEach { it.engine.syncOnce() }
        waitUntil(180_000, "kiosks approved in the server app") {
            kiosks.all { it.engine.syncOnce() is SyncStatus.Online }
        }
        println("REMOTE: approved, starting activity")
        kiosks.forEach { it.engine.start() }

        val registered = ConcurrentHashMap<String, MutableSet<Long>>()
        val events = ConcurrentHashMap.newKeySet<String>()
        val stop = AtomicBoolean(false)
        val threads = kiosks.mapIndexed { i, kiosk ->
            Thread {
                val random = Random(900 + i)
                while (!stop.get()) {
                    val mine = kiosk.repo.people().filter { registered.containsKey(it.key) }
                    when (random.nextInt(10)) {
                        0, 1 -> {
                            val samples = faces.samples("r-$runId-$i-${random.nextInt()}", 8)
                            val person = kiosk.repo.register(PersonMetadata.of("원격${random.nextInt(50)}", "테스트부", "$runId"), samples)
                            registered[person.key] = samples.mapTo(ConcurrentHashMap.newKeySet()) { Vectors.fingerprint(it) }
                        }
                        2, 3, 4 -> mine.randomOrNull(random)?.let { target ->
                            val samples = faces.samples(target.key, 2)
                            if (kiosk.repo.addSamples(target.key, samples)) registered[target.key]?.addAll(samples.map { Vectors.fingerprint(it) })
                        }
                        5, 6, 7 -> mine.randomOrNull(random)?.let { target ->
                            val event = AttendanceEvent(UUID.randomUUID().toString(), target.key, target.name, target.department, target.grade, kiosk.clientId, System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.7f)
                            kiosk.repo.recordAttendance(event)
                            events += event.eventId
                        }
                        8 -> kiosk.network.offline = !kiosk.network.offline
                        else -> kiosk.network.dropResponseRate = random.nextDouble(0.0, 0.2)
                    }
                    kiosk.engine.requestSync()
                    Thread.sleep(random.nextLong(50, 200))
                }
            }.apply { start() }
        }
        Thread.sleep(25_000)
        stop.set(true)
        threads.forEach { it.join() }
        kiosks.forEach { it.network.heal() }

        // Converged when every outbox is empty and all kiosks report the same revision twice in a row.
        var lastRevision = -1L
        waitUntil(120_000, "kiosks converge on the device server") {
            kiosks.forEach { it.engine.syncOnce() }
            val pendingEmpty = kiosks.all { it.repo.pendingCounts() == (0 to 0) }
            val revisions = kiosks.map { it.repo.syncPosition().revision }.toSet()
            val stable = pendingEmpty && revisions.size == 1 && revisions.single() == lastRevision
            lastRevision = revisions.singleOrNull() ?: -1
            stable
        }

        // A brand-new kiosk downloads the server's full database as the reference.
        val observer = SimKiosk("remote-$runId-observer", temp.newFolder("observer"), serverUrl!!, 999, 100)
        kiosks += observer
        observer.engine.syncOnce()
        println("REMOTE: approve the observer kiosk")
        waitUntil(120_000, "observer approved") { observer.engine.syncOnce() is SyncStatus.Online }
        val reference = observer.repo.people()

        kiosks.filter { it !== observer }.forEach { kiosk ->
            assertNull("kiosk ${kiosk.clientId} differs from the server", Compare.diff(reference, kiosk.repo.people()))
        }
        val referenceByKey = reference.associateBy { it.key }
        registered.forEach { (key, sent) ->
            val person = referenceByKey[key]
            assertTrue("registration $key missing on the server", person != null)
            val stored = person!!.embeddings.map { Vectors.fingerprint(it) }.toSet()
            assertTrue("samples lost for $key", stored.containsAll(sent))
        }
        val summary = "runId=$runId\nregistered=${registered.size}\ncheckIns=${events.size}\ngrade=$runId\n"
        File("build/remote-scenario.properties").writeText(summary)
        println("REMOTE: OK ${summary.replace('\n', ' ')}")
    }
}

private fun <T> List<T>.randomOrNull(random: Random): T? = if (isEmpty()) null else this[random.nextInt(size)]
