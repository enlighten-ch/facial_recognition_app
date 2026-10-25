package com.allstar.faithid.server.http

import com.allstar.faithid.core.attendance.AttendanceCodec
import com.allstar.faithid.core.sync.ChangeSetCodec
import com.allstar.faithid.core.sync.KeyValueText
import com.allstar.faithid.core.sync.PersonOpCodec
import com.allstar.faithid.core.sync.Protocol
import com.allstar.faithid.core.sync.UpdateMessage
import com.allstar.faithid.server.ServerLog
import com.allstar.faithid.server.attendance.AttendanceLedger
import com.allstar.faithid.server.auth.ClientRegistry
import com.allstar.faithid.server.people.PeopleRepository
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.request.header
import io.ktor.server.request.receiveChannel
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.pingPeriod
import io.ktor.server.websocket.timeout
import io.ktor.server.websocket.webSocket
import io.ktor.utils.io.core.readBytes
import io.ktor.utils.io.readRemaining
import io.ktor.websocket.CloseReason
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import io.ktor.websocket.readText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.time.Duration

private val TEXT = ContentType.Text.Plain.withParameter("charset", "utf-8")

/** Routes of protocol version 2 (see [Protocol]). */
internal fun Application.faithModule(
    people: PeopleRepository,
    clients: ClientRegistry,
    attendance: AttendanceLedger,
    updates: UpdateBroadcaster,
    log: ServerLog,
) {
    install(WebSockets) {
        pingPeriod = Duration.ofSeconds(20)
        timeout = Duration.ofSeconds(45)
    }

    routing {
        get(Protocol.HEALTH) {
            call.respondText(KeyValueText.encode("status" to "ok", "protocol" to Protocol.VERSION), TEXT)
        }

        post(Protocol.AUTH_REQUEST) {
            val clientId = call.request.header(Protocol.HEADER_CLIENT_ID).orEmpty().trim()
            val secret = call.bearerSecret()
            val deviceName = call.request.header(Protocol.HEADER_DEVICE_NAME).orEmpty().trim().ifBlank { "unknown" }
            val status = withContext(Dispatchers.IO) { clients.requestAccess(clientId, deviceName, secret) }
            when (status) {
                ClientRegistry.AccessStatus.APPROVED -> call.respondText(KeyValueText.encode("status" to Protocol.AUTH_APPROVED, "clientId" to clientId), TEXT)
                ClientRegistry.AccessStatus.PENDING -> call.respondText(KeyValueText.encode("status" to Protocol.AUTH_PENDING, "clientId" to clientId), TEXT)
                ClientRegistry.AccessStatus.SECRET_MISMATCH -> call.respondText("error=client_id_in_use", TEXT, HttpStatusCode.Conflict)
                ClientRegistry.AccessStatus.INVALID_REQUEST -> call.respondText("error=invalid_request", TEXT, HttpStatusCode.BadRequest)
            }
        }

        get(Protocol.CHANGES) {
            val clientId = call.authenticate(clients) ?: return@get
            val since = call.request.queryParameters[Protocol.PARAM_SINCE]?.toLongOrNull() ?: 0L
            val epoch = call.request.queryParameters[Protocol.PARAM_EPOCH]
            val bytes = withContext(Dispatchers.IO) { ChangeSetCodec.encode(people.changesSince(since, epoch)) }
            call.respondBytes(bytes, ContentType.Application.OctetStream)
            if (since <= 0) log.log("full download: client=$clientId, bytes=${bytes.size}")
        }

        post(Protocol.OPS) {
            val clientId = call.authenticate(clients) ?: return@post
            val body = call.receiveLimited() ?: return@post
            val ops = runCatching { PersonOpCodec.decode(body) }.getOrElse {
                call.respondText("error=invalid_ops", TEXT, HttpStatusCode.BadRequest)
                return@post
            }
            val response = withContext(Dispatchers.IO) { people.applyOps(ops, clientId) }
            call.respondText(response.encode(), TEXT)
        }

        post(Protocol.ATTENDANCE) {
            val clientId = call.authenticate(clients) ?: return@post
            val body = call.receiveLimited() ?: return@post
            val events = runCatching { AttendanceCodec.decode(body) }.getOrElse {
                call.respondText("error=invalid_attendance", TEXT, HttpStatusCode.BadRequest)
                return@post
            }
            val stored = withContext(Dispatchers.IO) { attendance.record(events, clientId) }
            call.respondText(KeyValueText.encode("received" to events.size, "stored" to stored), TEXT)
        }

        webSocket(Protocol.UPDATES_WS) {
            val clientId = call.request.header(Protocol.HEADER_CLIENT_ID).orEmpty()
            val authorized = withContext(Dispatchers.IO) { clients.authenticate(clientId, call.bearerSecret()) }
            if (!authorized) {
                close(CloseReason(CloseReason.Codes.VIOLATED_POLICY, "unauthorized"))
                return@webSocket
            }
            updates.add(this, clientId)
            try {
                send(Frame.Text(UpdateMessage.encode(people.revision, people.epoch)))
                for (frame in incoming) {
                    if (frame is Frame.Text && frame.readText() == "ping") send(Frame.Text("pong"))
                }
            } finally {
                updates.remove(this)
            }
        }
    }
}

private fun ApplicationCall.bearerSecret(): String =
    request.header(Protocol.HEADER_AUTHORIZATION)
        ?.takeIf { it.startsWith(Protocol.BEARER, ignoreCase = true) }
        ?.substring(Protocol.BEARER.length)
        ?.trim()
        .orEmpty()

/** Returns the authenticated client id, or responds 401/403 and returns null. */
private suspend fun ApplicationCall.authenticate(clients: ClientRegistry): String? {
    val clientId = request.header(Protocol.HEADER_CLIENT_ID).orEmpty().trim()
    val secret = bearerSecret()
    val ok = withContext(Dispatchers.IO) { clients.authenticate(clientId, secret) }
    if (ok) return clientId
    val pending = withContext(Dispatchers.IO) { clientId.isNotBlank() && clients.isPending(clientId) }
    if (pending) {
        respondText("error=pending_approval", TEXT, HttpStatusCode.Forbidden)
    } else {
        respondText("error=unauthorized", TEXT, HttpStatusCode.Unauthorized)
    }
    return null
}

/** Reads the request body up to [Protocol.MAX_BODY_BYTES]; responds 413/400 and returns null otherwise. */
private suspend fun ApplicationCall.receiveLimited(): ByteArray? {
    val declared = request.header(HttpHeaders.ContentLength)?.toLongOrNull()
    if (declared != null && declared > Protocol.MAX_BODY_BYTES) {
        respondText("error=too_large", TEXT, HttpStatusCode.PayloadTooLarge)
        return null
    }
    val packet = receiveChannel().readRemaining(Protocol.MAX_BODY_BYTES + 1)
    if (packet.remaining > Protocol.MAX_BODY_BYTES) {
        packet.release()
        respondText("error=too_large", TEXT, HttpStatusCode.PayloadTooLarge)
        return null
    }
    val bytes = packet.readBytes()
    if (bytes.isEmpty()) {
        respondText("error=empty_body", TEXT, HttpStatusCode.BadRequest)
        return null
    }
    return bytes
}
