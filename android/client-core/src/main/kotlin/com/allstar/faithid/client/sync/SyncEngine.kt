package com.allstar.faithid.client.sync

import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.core.attendance.AttendanceCodec
import com.allstar.faithid.core.sync.ChangeSetCodec
import com.allstar.faithid.core.sync.KeyValueText
import com.allstar.faithid.core.sync.OpsResponse
import com.allstar.faithid.core.sync.PersonOpCodec
import com.allstar.faithid.core.sync.Protocol
import com.allstar.faithid.core.sync.UpdateMessage
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Keeps one kiosk in sync with the DB server.
 *
 * Each cycle: ask for access (also the health check) → upload queued changes → upload queued
 * check-ins → download changes since the last known revision. Uploading first means a download
 * never hides an unsent local change. A WebSocket from the server triggers a cycle as soon as
 * another kiosk or the administrator changes something; a timer covers a broken socket.
 *
 * Failures never lose data: queued entries stay in the outbox until the server has answered, and
 * every server operation is safe to repeat.
 */
class SyncEngine(
    private val config: SyncConfig,
    private val repository: LocalFaceRepository,
    private val listener: SyncListener,
    /** True when downloaded changes may be applied now (no face in front of the camera). */
    private val canApplyNow: () -> Boolean = { true },
    httpClient: OkHttpClient = OkHttpClient(),
    private val now: () -> Long = System::currentTimeMillis,
) {
    private val http = httpClient.newBuilder()
        .connectTimeout(config.connectTimeoutMillis, TimeUnit.MILLISECONDS)
        .readTimeout(config.readTimeoutMillis, TimeUnit.MILLISECONDS)
        .writeTimeout(config.readTimeoutMillis, TimeUnit.MILLISECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
    private val binary = "application/octet-stream".toMediaType()
    private val wakeLock = Object()
    private val cycleLock = Any()
    private var wakeRequested = false

    @Volatile
    private var running = false
    private var thread: Thread? = null

    @Volatile
    private var socket: WebSocket? = null

    @Volatile
    private var socketOpen = false

    @Volatile
    var status: SyncStatus = SyncStatus.Connecting(config.baseUrl)
        private set

    private class AuthLostException(message: String) : IOException(message)

    fun start() {
        if (running) return
        running = true
        thread = Thread(::runLoop, "faith-sync").apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        running = false
        wake()
        closeSocket()
        thread?.join(3_000)
        thread = null
    }

    /** Asks the engine to sync soon (after a local change, or a manual retry). */
    fun requestSync() = wake()

    /**
     * Runs one complete cycle on the calling thread and returns the resulting status. Used by the
     * background loop and directly by tests.
     */
    fun syncOnce(): SyncStatus {
        val result = synchronized(cycleLock) {
            runCatching { cycle() }.getOrElse { error ->
                when (error) {
                    is AuthLostException -> SyncStatus.Connecting(config.baseUrl)
                    else -> SyncStatus.Offline(describe(error), 0)
                }
            }
        }
        publish(result)
        return result
    }

    private fun runLoop() {
        var failures = 0
        while (running) {
            val result = syncOnce()
            val delay = when (result) {
                is SyncStatus.Online -> {
                    failures = 0
                    config.pollIntervalMillis
                }
                is SyncStatus.AwaitingApproval -> config.approvalPollMillis
                is SyncStatus.Rejected -> config.maxRetryDelayMillis
                is SyncStatus.Connecting -> 200L
                is SyncStatus.Offline -> {
                    val backoff = (config.retryDelayMillis shl failures.coerceAtMost(5)).coerceAtMost(config.maxRetryDelayMillis)
                    failures++
                    publish(result.copy(retryInMillis = backoff))
                    backoff
                }
                SyncStatus.Disabled -> Long.MAX_VALUE
            }
            waitForWake(delay)
        }
    }

    private fun cycle(): SyncStatus {
        if (config.baseUrl.isEmpty()) return SyncStatus.Offline("서버 주소가 설정되지 않았습니다.", 0)
        when (val access = requestAccess()) {
            Protocol.AUTH_PENDING -> {
                closeSocket()
                return SyncStatus.AwaitingApproval(config.identity.clientId, config.identity.deviceName)
            }
            Protocol.AUTH_APPROVED -> Unit
            else -> return SyncStatus.Rejected(access)
        }
        ensureSocket()
        pushOps()
        pushCheckIns()
        val revision = pull()
        val (ops, checkIns) = repository.pendingCounts()
        return SyncStatus.Online(now(), ops, checkIns, revision)
    }

    /** Returns [Protocol.AUTH_APPROVED], [Protocol.AUTH_PENDING] or a rejection message. */
    private fun requestAccess(): String {
        val request = authorized(Protocol.AUTH_REQUEST).post(ByteArray(0).toRequestBody()).build()
        http.newCall(request).execute().use { response ->
            return when (response.code) {
                200 -> KeyValueText.decode(response.body?.string().orEmpty())["status"].orEmpty().ifBlank { "unknown response" }
                409 -> "다른 기기가 같은 키오스크 ID(${config.identity.clientId})를 쓰고 있습니다. 설정에서 ID를 바꿔 주세요."
                400 -> "키오스크 ID 형식이 올바르지 않습니다: ${config.identity.clientId}"
                else -> throw IOException("HTTP ${response.code}")
            }
        }
    }

    private fun pushOps() {
        while (true) {
            val batch = repository.pendingOps(config.maxOpsPerRequest)
            if (batch.isEmpty()) return
            val request = authorized(Protocol.OPS).post(PersonOpCodec.encode(batch).toRequestBody(binary)).build()
            val response = execute(request) { OpsResponse.decode(it.body?.string().orEmpty()) }
            repository.acknowledgeOps(response.results)
            listener.onLog("변경 ${batch.size}건 전송, 서버 revision=${response.serverRevision}")
            if (response.results.keys.none { id -> batch.any { it.opId == id } }) {
                throw IOException("server answered for none of the uploaded changes")
            }
        }
    }

    private fun pushCheckIns() {
        while (true) {
            val batch = repository.pendingAttendance(config.maxCheckInsPerRequest)
            if (batch.isEmpty()) return
            val request = authorized(Protocol.ATTENDANCE).post(AttendanceCodec.encode(batch).toRequestBody(binary)).build()
            execute(request) { it.body?.string() }
            repository.acknowledgeAttendance(batch.map { it.eventId })
            listener.onLog("출석 ${batch.size}건 전송")
        }
    }

    /** Downloads and applies server changes; returns the server revision. */
    private fun pull(): Long {
        val position = repository.syncPosition()
        val url = (config.baseUrl + Protocol.CHANGES).toHttpUrl().newBuilder()
            .addQueryParameter(Protocol.PARAM_SINCE, position.revision.toString())
            .apply { position.epoch?.let { addQueryParameter(Protocol.PARAM_EPOCH, it) } }
            .build()
        val changes = execute(authorized(Protocol.CHANGES).url(url).get().build()) { ChangeSetCodec.decode(it.body!!.bytes()) }
        if (changes.isEmpty() && changes.serverRevision == position.revision && changes.serverEpoch == position.epoch) {
            return changes.serverRevision
        }
        waitUntilApplyAllowed()
        val changed = repository.applyServerChanges(changes)
        listener.onRemoteChangesApplied(changed, changes.fullSnapshot)
        listener.onLog(
            if (changes.fullSnapshot) "서버 DB 전체 수신: ${changes.persons.size}명, revision=${changes.serverRevision}"
            else "서버 변경 ${changed}건 반영, revision=${changes.serverRevision}",
        )
        return changes.serverRevision
    }

    /**
     * Waits (bounded) until no one is in front of the camera. Check-ins recorded meanwhile are
     * still sent, so the server's attendance view does not lag behind while someone stands there.
     */
    private fun waitUntilApplyAllowed() {
        val deadline = now() + config.applyIdleWaitMillis
        while (running && !canApplyNow() && now() < deadline) {
            Thread.sleep(200)
            pushCheckIns()
        }
    }

    private fun ensureSocket() {
        if (socketOpen && socket != null) return
        closeSocket()
        val wsUrl = config.baseUrl.replaceFirst("http", "ws") + Protocol.UPDATES_WS
        socket = http.newWebSocket(
            authorized(Protocol.UPDATES_WS).url(wsUrl).build(),
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    socketOpen = true
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    val announced = UpdateMessage.decodeRevision(text) ?: return
                    if (announced != repository.syncPosition().revision) wake()
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    socketOpen = false
                    webSocket.close(1000, null)
                    if (running) wake()
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    socketOpen = false
                    if (running) wake()
                }
            },
        )
    }

    private fun closeSocket() {
        socketOpen = false
        socket?.cancel()
        socket = null
    }

    private fun authorized(path: String): Request.Builder =
        Request.Builder()
            .url(config.baseUrl + path)
            .header(Protocol.HEADER_CLIENT_ID, config.identity.clientId)
            .header(Protocol.HEADER_DEVICE_NAME, config.identity.deviceName.toAsciiHeader())
            .header(Protocol.HEADER_AUTHORIZATION, Protocol.BEARER + config.identity.secret)

    private fun <T> execute(request: Request, read: (Response) -> T): T {
        http.newCall(request).execute().use { response ->
            when (response.code) {
                200 -> return read(response)
                401, 403 -> {
                    closeSocket()
                    throw AuthLostException("HTTP ${response.code}")
                }
                else -> throw IOException("HTTP ${response.code} ${response.body?.string()?.take(100).orEmpty()}")
            }
        }
    }

    private fun publish(next: SyncStatus) {
        status = next
        listener.onStatus(next)
    }

    private fun wake() {
        synchronized(wakeLock) {
            wakeRequested = true
            wakeLock.notifyAll()
        }
    }

    private fun waitForWake(millis: Long) {
        synchronized(wakeLock) {
            val deadline = System.currentTimeMillis() + millis.coerceAtMost(Long.MAX_VALUE / 2)
            while (running && !wakeRequested) {
                val remaining = deadline - System.currentTimeMillis()
                if (remaining <= 0) break
                wakeLock.wait(remaining)
            }
            wakeRequested = false
        }
    }

    private fun describe(error: Throwable): String = when (error) {
        is java.net.ConnectException -> "서버에 연결할 수 없습니다."
        is java.net.SocketTimeoutException -> "서버 응답 시간이 초과되었습니다."
        is java.net.UnknownHostException -> "서버 주소를 찾을 수 없습니다."
        else -> "${error.javaClass.simpleName}: ${error.message.orEmpty()}"
    }

    /** OkHttp only accepts ASCII header values; device names like "Galaxy Tab S7+" pass, others are escaped. */
    private fun String.toAsciiHeader(): String =
        buildString { this@toAsciiHeader.forEach { c -> if (c.code in 0x20..0x7e) append(c) else append('?') } }
}
