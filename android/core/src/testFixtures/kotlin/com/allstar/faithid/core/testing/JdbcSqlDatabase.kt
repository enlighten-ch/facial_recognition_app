package com.allstar.faithid.core.testing

import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.core.storage.SqlDatabaseFactory
import com.allstar.faithid.core.storage.SqlRow
import java.io.File
import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet

/** [SqlDatabase] on sqlite-jdbc, used by JVM unit and integration tests in place of Android's SQLite. */
class JdbcSqlDatabase(file: File) : SqlDatabase {
    private val connection: Connection = DriverManager.getConnection("jdbc:sqlite:${file.absolutePath}")
    private var depth = 0
    private val lock = Any()

    override fun execute(sql: String, vararg args: Any?) {
        synchronized(lock) {
            connection.prepareStatement(sql).use { statement ->
                bind(statement, args)
                statement.execute()
            }
        }
    }

    override fun <T> query(sql: String, vararg args: Any?, mapper: (SqlRow) -> T): List<T> {
        synchronized(lock) {
            connection.prepareStatement(sql).use { statement ->
                bind(statement, args)
                statement.executeQuery().use { rs ->
                    val row = ResultSetRow(rs)
                    val out = mutableListOf<T>()
                    while (rs.next()) out += mapper(row)
                    return out
                }
            }
        }
    }

    override fun <T> transaction(block: () -> T): T {
        synchronized(lock) {
            if (depth++ == 0) connection.autoCommit = false
            try {
                val result = block()
                if (depth == 1) connection.commit()
                return result
            } catch (t: Throwable) {
                if (depth == 1) connection.rollback()
                throw t
            } finally {
                if (--depth == 0) connection.autoCommit = true
            }
        }
    }

    override fun close() = connection.close()

    private fun bind(statement: java.sql.PreparedStatement, args: Array<out Any?>) {
        args.forEachIndexed { i, value ->
            val index = i + 1
            when (value) {
                null -> statement.setObject(index, null)
                is String -> statement.setString(index, value)
                is Int -> statement.setLong(index, value.toLong())
                is Long -> statement.setLong(index, value)
                is Double -> statement.setDouble(index, value)
                is Float -> statement.setDouble(index, value.toDouble())
                is ByteArray -> statement.setBytes(index, value)
                is Boolean -> statement.setLong(index, if (value) 1 else 0)
                else -> error("unsupported bind type ${value.javaClass}")
            }
        }
    }

    private class ResultSetRow(private val rs: ResultSet) : SqlRow {
        override fun string(column: String): String? = rs.getString(column)
        override fun long(column: String): Long? = rs.getLong(column).takeUnless { rs.wasNull() }
        override fun bytes(column: String): ByteArray? = rs.getBytes(column)
    }

    companion object {
        val FACTORY = SqlDatabaseFactory { JdbcSqlDatabase(it) }
    }
}
