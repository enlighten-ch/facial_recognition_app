package com.allstar.faithid.server

import java.io.File
import java.time.ZoneId

fun interface Clock {
    fun now(): Long

    companion object {
        val SYSTEM = Clock { System.currentTimeMillis() }
    }
}

/** Receives human-readable server log lines (shown in the server app's log panel). */
fun interface ServerLog {
    fun log(message: String)

    companion object {
        val NONE = ServerLog { }
    }
}

data class ServerConfig(
    val port: Int = 18080,
    /** Shared folder with the face DB, the CSV export and attendance sheets (default /sdcard/faceDB). */
    val dataDir: File,
    /** App-private folder for client approvals; never shared. */
    val stateDir: File,
    val maxEmbeddingsPerPerson: Int = 100,
    val zone: ZoneId = ZoneId.systemDefault(),
)
