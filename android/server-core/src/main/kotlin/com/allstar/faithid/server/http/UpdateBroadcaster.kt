package com.allstar.faithid.server.http

import com.allstar.faithid.core.sync.UpdateMessage
import com.allstar.faithid.server.ServerLog
import io.ktor.websocket.CloseReason
import io.ktor.websocket.DefaultWebSocketSession
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

/** Pushes "revision changed" messages to every connected kiosk. */
class UpdateBroadcaster(private val log: ServerLog) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val sessions = ConcurrentHashMap<DefaultWebSocketSession, String>()

    fun add(session: DefaultWebSocketSession, clientId: String) {
        sessions[session] = clientId
    }

    fun remove(session: DefaultWebSocketSession) {
        sessions.remove(session)
    }

    fun broadcast(revision: Long, epoch: String) {
        val message = UpdateMessage.encode(revision, epoch)
        sessions.keys.forEach { session ->
            scope.launch { runCatching { session.send(Frame.Text(message)) } }
        }
    }

    /** Disconnects every session of [clientId] (used when its approval is revoked). */
    fun disconnect(clientId: String) {
        sessions.filterValues { it == clientId }.keys.forEach { session ->
            scope.launch { runCatching { session.close(CloseReason(CloseReason.Codes.VIOLATED_POLICY, "revoked")) } }
            sessions.remove(session)
        }
    }

    fun closeAll() {
        sessions.keys.forEach { session -> scope.launch { runCatching { session.close() } } }
        sessions.clear()
    }

    fun onlineClients(): Map<String, Int> = sessions.values.groupingBy { it }.eachCount()
}
