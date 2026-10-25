package com.allstar.faithid.it

import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.client.sync.KioskIdentity
import com.allstar.faithid.client.sync.SyncConfig
import com.allstar.faithid.client.sync.SyncEngine
import com.allstar.faithid.client.sync.SyncListener
import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Response
import java.io.File
import java.io.IOException
import java.net.ConnectException
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

/**
 * Network between one kiosk and the server, with injectable faults:
 * - [offline]: every request fails to connect (Wi-Fi down)
 * - [dropRequestRate]: the request is lost before it reaches the server
 * - [dropResponseRate]: the server processes the request but the answer is lost, so the kiosk
 *   retries something that already happened (the hardest case for exactly-once behavior)
 */
class FlakyNetwork(seed: Long) : Interceptor {
    private val random = Random(seed)

    @Volatile var offline = false
    @Volatile var dropRequestRate = 0.0
    @Volatile var dropResponseRate = 0.0
    val droppedRequests = AtomicInteger()
    val droppedResponses = AtomicInteger()

    fun heal() {
        offline = false
        dropRequestRate = 0.0
        dropResponseRate = 0.0
    }

    override fun intercept(chain: Interceptor.Chain): Response {
        if (offline) throw ConnectException("simulated: Wi-Fi down")
        if (roll() < dropRequestRate) {
            droppedRequests.incrementAndGet()
            throw IOException("simulated: request lost")
        }
        val response = chain.proceed(chain.request())
        if (!chain.request().url.encodedPath.startsWith("/ws") && roll() < dropResponseRate) {
            response.close()
            droppedResponses.incrementAndGet()
            throw IOException("simulated: response lost after the server processed the request")
        }
        return response
    }

    @Synchronized
    private fun roll(): Double = random.nextDouble()
}

/** One simulated kiosk: its own database, network and sync engine with a background loop. */
class SimKiosk(
    val clientId: String,
    dir: File,
    serverUrl: String,
    seed: Long,
    maxEmbeddings: Int,
    fastTimers: Boolean = true,
    /** Defaults to "sometimes a face is in front of the camera". */
    canApplyNow: () -> Boolean = { Random.nextInt(4) != 0 },
    extraInterceptor: Interceptor? = null,
) {
    val network = FlakyNetwork(seed)
    val identity = KioskIdentity(clientId, "secret-$clientId-${seed}abcdef0123", "Sim $clientId")
    val repo = LocalFaceRepository(PeopleStore.open(dir, JdbcSqlDatabase.FACTORY), maxEmbeddings)
    val statuses = CopyOnWriteArrayList<SyncStatus>()
    val engine = SyncEngine(
        SyncConfig(
            serverUrl = serverUrl,
            identity = identity,
            connectTimeoutMillis = 2_000,
            readTimeoutMillis = 20_000,
            retryDelayMillis = if (fastTimers) 100 else 3_000,
            maxRetryDelayMillis = if (fastTimers) 800 else 60_000,
            approvalPollMillis = if (fastTimers) 150 else 3_000,
            pollIntervalMillis = if (fastTimers) 700 else 30_000,
            applyIdleWaitMillis = 300,
        ),
        repo,
        object : SyncListener {
            override fun onStatus(status: SyncStatus) {
                statuses += status
            }
        },
        canApplyNow = canApplyNow,
        httpClient = OkHttpClient.Builder().addInterceptor(network).apply { extraInterceptor?.let { addInterceptor(it) } }.build(),
    )

    fun isCaughtUp(serverEpoch: String, serverRevision: Long): Boolean {
        val (ops, checkIns) = repo.pendingCounts()
        val position = repo.syncPosition()
        return ops == 0 && checkIns == 0 && position.revision == serverRevision && position.epoch == serverEpoch
    }

    fun close() {
        engine.stop()
        repo.close()
    }
}

object Compare {
    /** Describes differences between two people lists, or returns null if they are equal. */
    fun diff(expected: List<Person>, actual: List<Person>): String? {
        val e = expected.associateBy { it.key }
        val a = actual.associateBy { it.key }
        val problems = mutableListOf<String>()
        (e.keys - a.keys).forEach { problems += "missing ${e.getValue(it)}" }
        (a.keys - e.keys).forEach { problems += "unexpected ${a.getValue(it)}" }
        e.keys.intersect(a.keys).forEach { key ->
            val x = e.getValue(key)
            val y = a.getValue(key)
            if (x.metadata != y.metadata) problems += "metadata differs for $key: ${x.metadata} vs ${y.metadata}"
            val xs = x.embeddings.map { Vectors.fingerprint(it) }.toSet()
            val ys = y.embeddings.map { Vectors.fingerprint(it) }.toSet()
            if (xs != ys) problems += "samples differ for $key: ${xs.size} vs ${ys.size} (only expected ${(xs - ys).size}, only actual ${(ys - xs).size})"
        }
        return problems.takeIf { it.isNotEmpty() }?.joinToString("\n")
    }
}

fun waitUntil(timeoutMs: Long, what: String, condition: () -> Boolean) {
    val deadline = System.currentTimeMillis() + timeoutMs
    while (!condition()) {
        if (System.currentTimeMillis() > deadline) throw AssertionError("timed out after $timeoutMs ms: $what")
        Thread.sleep(100)
    }
}
