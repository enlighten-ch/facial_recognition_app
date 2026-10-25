package com.allstar.faithid.core.storage

import java.io.Closeable
import java.io.File

/**
 * Minimal SQL access shared by Android (platform SQLite) and JVM tests (sqlite-jdbc).
 *
 * Bind arguments may be `null`, [String], [Int], [Long], [Double], [Float] or [ByteArray].
 */
interface SqlDatabase : Closeable {
    fun execute(sql: String, vararg args: Any?)

    fun <T> query(sql: String, vararg args: Any?, mapper: (SqlRow) -> T): List<T>

    /** Runs [block] in a transaction; nested calls join the outer transaction. */
    fun <T> transaction(block: () -> T): T

    fun columns(table: String): Set<String> =
        query("PRAGMA table_info($table)") { it.string("name").orEmpty().lowercase() }.toSet()

    fun tables(): Set<String> =
        query("SELECT name FROM sqlite_master WHERE type='table'") { it.string("name").orEmpty().lowercase() }.toSet()
}

interface SqlRow {
    fun string(column: String): String?
    fun long(column: String): Long?
    fun bytes(column: String): ByteArray?
}

fun interface SqlDatabaseFactory {
    /** Opens (creating if needed) the database file at [file]. */
    fun open(file: File): SqlDatabase
}
