package com.allstar.faithid.server.http

import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.core.storage.SqlDatabaseFactory
import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerConfig
import com.allstar.faithid.server.ServerLog
import com.allstar.faithid.server.attendance.AttendanceLedger
import com.allstar.faithid.server.auth.ClientRegistry
import com.allstar.faithid.server.people.PeopleRepository
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import java.io.Closeable
import java.io.File
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The DB server: storage, approvals, attendance and the HTTP/WebSocket endpoint, wired together.
 * The Android server app creates one instance and calls the admin methods on [clients], [people]
 * and [attendance] from its UI.
 */
class FaithServer private constructor(
    val config: ServerConfig,
    val people: PeopleRepository,
    val clients: ClientRegistry,
    val attendance: AttendanceLedger,
    private val peopleStore: PeopleStore,
    private val stateDb: SqlDatabase,
    private val attendanceDb: SqlDatabase,
    private val log: ServerLog,
) : Closeable {

    val updates = UpdateBroadcaster(log)
    private var engine: ApplicationEngine? = null
    private val csvExecutor = Executors.newSingleThreadExecutor { Thread(it, "faith-csv-export").apply { isDaemon = true } }
    private val csvExportQueued = AtomicBoolean(false)

    init {
        people.onChanged = { revision ->
            updates.broadcast(revision, people.epoch)
            scheduleCsvExport()
        }
        scheduleCsvExport()
    }

    val isRunning: Boolean get() = engine != null

    @Synchronized
    fun start(): Boolean {
        if (engine != null) return true
        return runCatching {
            val server = embeddedServer(CIO, port = config.port, host = "0.0.0.0") {
                faithModule(people, clients, attendance, updates, log)
            }
            server.start(wait = false)
            engine = server
            if (!waitUntilListening(config.port, timeoutMs = 10_000)) error("port ${config.port} did not open")
            log.log("server started on port ${config.port}")
            true
        }.getOrElse {
            log.log("server start failed: ${it.javaClass.simpleName}: ${it.message}")
            engine?.stop(0, 500)
            engine = null
            false
        }
    }

    @Synchronized
    fun stop() {
        val server = engine ?: return
        updates.closeAll()
        server.stop(500, 2000)
        engine = null
        log.log("server stopped")
    }

    /** Removes a kiosk's approval and closes its open connections. */
    fun revokeClient(clientId: String): Boolean {
        val removed = clients.revoke(clientId)
        updates.disconnect(clientId)
        return removed
    }

    /** Kiosks that currently hold an update connection, with their session counts. */
    fun onlineClients(): Map<String, Int> = updates.onlineClients()

    override fun close() {
        stop()
        csvExecutor.shutdown()
        peopleStore.close()
        stateDb.close()
        attendanceDb.close()
    }

    private fun scheduleCsvExport() {
        if (!csvExportQueued.compareAndSet(false, true)) return
        csvExecutor.execute {
            csvExportQueued.set(false)
            runCatching { people.exportCsv() }.onFailure { log.log("face_db.csv export failed: ${it.message}") }
        }
    }

    private fun waitUntilListening(port: Int, timeoutMs: Long): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val connected = runCatching { Socket().use { it.connect(InetSocketAddress("127.0.0.1", port), 300) } }.isSuccess
            if (connected) return true
            Thread.sleep(50)
        }
        return false
    }

    companion object {
        const val STATE_DB = "server_state.db"
        const val ATTENDANCE_DB = "attendance.db"
        const val LEGACY_AUTH_FILE = "auth_state.json"

        fun create(
            config: ServerConfig,
            sqlFactory: SqlDatabaseFactory,
            clock: Clock = Clock.SYSTEM,
            log: ServerLog = ServerLog.NONE,
        ): FaithServer {
            config.dataDir.mkdirs()
            config.stateDir.mkdirs()
            val store = PeopleStore.open(config.dataDir, sqlFactory) { log.log(it) }
            val people = PeopleRepository(store, config.maxEmbeddingsPerPerson, clock, log)
            val stateDb = sqlFactory.open(File(config.stateDir, STATE_DB))
            val clients = ClientRegistry(stateDb, clock, log)
            clients.migrateLegacyAuthFile(File(config.stateDir, LEGACY_AUTH_FILE))
            val attendanceDb = sqlFactory.open(File(config.dataDir, ATTENDANCE_DB))
            val attendance = AttendanceLedger(attendanceDb, config.dataDir, config.zone, clock, log) { people.list() }
            return FaithServer(config, people, clients, attendance, store, stateDb, attendanceDb, log)
        }
    }
}
