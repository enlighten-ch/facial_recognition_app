package com.allstar.faithid.server

import com.allstar.faithid.core.attendance.AttendanceCodec
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.sync.ChangeSetCodec
import com.allstar.faithid.core.sync.KeyValueText
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.OpsResponse
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.PersonOpCodec
import com.allstar.faithid.core.sync.Protocol
import com.allstar.faithid.core.sync.UpdateMessage
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.server.http.FaithServer
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.net.ServerSocket
import java.time.ZoneId
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

class HttpApiTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val faces = FakeFaces(seed = 8, dim = 16)
    private val http = OkHttpClient.Builder().readTimeout(10, TimeUnit.SECONDS).build()
    private val binary = "application/octet-stream".toMediaType()
    private val secret = "kiosk-secret-0123456789abcdef"
    private lateinit var server: FaithServer
    private var port = 0

    @Before
    fun setUp() {
        port = ServerSocket(0).use { it.localPort }
        val config = ServerConfig(port = port, dataDir = temp.newFolder("faceDB"), stateDir = temp.newFolder("state"), zone = ZoneId.of("Asia/Seoul"))
        server = FaithServer.create(config, JdbcSqlDatabase.FACTORY)
        assertTrue(server.start())
    }

    @After
    fun tearDown() = server.close()

    private fun url(path: String) = "http://127.0.0.1:$port$path"

    private fun request(path: String, clientId: String = "kiosk-1", secret: String = this.secret) =
        Request.Builder().url(url(path)).header(Protocol.HEADER_CLIENT_ID, clientId).header("Authorization", "Bearer $secret")

    private fun call(request: Request): Response = http.newCall(request).execute()

    private fun approveKiosk(clientId: String = "kiosk-1") {
        call(request(Protocol.AUTH_REQUEST, clientId).post(ByteArray(0).toRequestBody()).build()).close()
        assertTrue(server.clients.approve(clientId))
    }

    @Test
    fun `health check needs no credentials`() {
        call(Request.Builder().url(url(Protocol.HEALTH)).build()).use {
            assertEquals(200, it.code)
            assertEquals("ok", KeyValueText.decode(it.body!!.string())["status"])
        }
    }

    @Test
    fun `approval flow and status codes`() {
        call(request(Protocol.AUTH_REQUEST).post(ByteArray(0).toRequestBody()).build()).use {
            assertEquals(Protocol.AUTH_PENDING, KeyValueText.decode(it.body!!.string())["status"])
        }
        call(request(Protocol.CHANGES).get().build()).use { assertEquals(403, it.code) }
        call(request(Protocol.CHANGES, clientId = "nobody").get().build()).use { assertEquals(401, it.code) }
        server.clients.approve("kiosk-1")
        call(request(Protocol.AUTH_REQUEST).post(ByteArray(0).toRequestBody()).build()).use {
            val body = it.body!!.string()
            assertEquals(Protocol.AUTH_APPROVED, KeyValueText.decode(body)["status"])
            assertTrue("no secret or token in the response", !body.contains(secret))
        }
        call(request(Protocol.CHANGES).get().build()).use { assertEquals(200, it.code) }
        call(request(Protocol.CHANGES, secret = "wrong-secret-0123456789").get().build()).use { assertEquals(401, it.code) }
        call(request(Protocol.AUTH_REQUEST, secret = "other-secret-0123456789").post(ByteArray(0).toRequestBody()).build()).use {
            assertEquals(409, it.code)
        }
    }

    @Test
    fun `administration is not reachable over HTTP`() {
        call(request(Protocol.AUTH_REQUEST, clientId = "intruder").post(ByteArray(0).toRequestBody()).build()).close()
        listOf("/api/auth/approve-all", "/api/auth/approve", "/api/auth/pending", "/api/auth/approved", "/api/auth/status").forEach { path ->
            val response = call(Request.Builder().url(url(path)).post(ByteArray(0).toRequestBody()).build())
            response.use { assertEquals("$path must not exist", 404, it.code) }
        }
        call(request(Protocol.CHANGES, clientId = "intruder").get().build()).use { assertEquals(403, it.code) }
    }

    @Test
    fun `ops, changes and attendance end to end`() {
        approveKiosk()
        val ops = listOf(PersonOp.Register("op1", "p_1", PersonMetadata.of("홍길동", "청년부", "3학년"), faces.samples("p1", 3), 1))
        val opsResponse = call(request(Protocol.OPS).post(PersonOpCodec.encode(ops).toRequestBody(binary)).build()).use {
            OpsResponse.decode(it.body!!.string())
        }
        assertEquals(OpResult.APPLIED, opsResponse.results["op1"])
        val changes = call(request("${Protocol.CHANGES}?since=0").get().build()).use { ChangeSetCodec.decode(it.body!!.bytes()) }
        assertTrue(changes.fullSnapshot)
        assertEquals("홍길동", changes.persons.single().name)
        val incremental = call(request("${Protocol.CHANGES}?since=${changes.serverRevision}&epoch=${changes.serverEpoch}").get().build()).use {
            ChangeSetCodec.decode(it.body!!.bytes())
        }
        assertTrue(incremental.isEmpty())

        val event = AttendanceEvent("e1", "p_1", "홍길동", "청년부", "3학년", "kiosk-1", System.currentTimeMillis(), CheckInSource.RECOGNIZED, 0.8f)
        repeat(2) { attempt ->
            call(request(Protocol.ATTENDANCE).post(AttendanceCodec.encode(listOf(event)).toRequestBody(binary)).build()).use {
                assertEquals(if (attempt == 0) "1" else "0", KeyValueText.decode(it.body!!.string())["stored"])
            }
        }
    }

    @Test
    fun `malformed and oversized bodies are refused`() {
        approveKiosk()
        call(request(Protocol.OPS).post("garbage".toByteArray().toRequestBody(binary)).build()).use { assertEquals(400, it.code) }
        call(request(Protocol.ATTENDANCE).post(ByteArray(0).toRequestBody(binary)).build()).use { assertEquals(400, it.code) }
        val huge = ByteArray((Protocol.MAX_BODY_BYTES + 10).toInt())
        val code = runCatching { call(request(Protocol.OPS).post(huge.toRequestBody(binary)).build()).use { it.code } }.getOrDefault(413)
        assertEquals(413, code)
    }

    @Test
    fun `update socket announces new revisions and refuses strangers`() {
        approveKiosk()
        val messages = LinkedBlockingQueue<String>()
        val closed = LinkedBlockingQueue<Int>()
        val listener = object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) { messages += text }
            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) { closed += code }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) { closed += -1 }
        }
        val socket = http.newWebSocket(request(Protocol.UPDATES_WS).url(url(Protocol.UPDATES_WS).replace("http", "ws")).build(), listener)
        assertEquals(0L, UpdateMessage.decodeRevision(messages.poll(5, TimeUnit.SECONDS)!!))
        server.people.applyOps(listOf(PersonOp.Register("x", "p_x", PersonMetadata.of("A", "", ""), faces.samples("a", 1), 1)), "admin")
        assertEquals(1L, UpdateMessage.decodeRevision(messages.poll(5, TimeUnit.SECONDS)!!))
        assertEquals(mapOf("kiosk-1" to 1), server.onlineClients())
        server.revokeClient("kiosk-1")
        assertTrue(closed.poll(5, TimeUnit.SECONDS) != null)
        socket.cancel()

        val strangerClosed = LinkedBlockingQueue<Int>()
        http.newWebSocket(
            request(Protocol.UPDATES_WS, clientId = "stranger").url(url(Protocol.UPDATES_WS).replace("http", "ws")).build(),
            object : WebSocketListener() {
                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) { strangerClosed += code }
                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) { strangerClosed += -1 }
            },
        )
        assertTrue(strangerClosed.poll(5, TimeUnit.SECONDS) != null)
    }
}
