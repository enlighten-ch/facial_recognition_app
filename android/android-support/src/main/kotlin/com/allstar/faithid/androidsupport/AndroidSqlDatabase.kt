package com.allstar.faithid.androidsupport

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.core.storage.SqlDatabaseFactory
import com.allstar.faithid.core.storage.SqlRow
import java.io.File

/** [SqlDatabase] on the platform SQLite. */
class AndroidSqlDatabase(file: File) : SqlDatabase {
    private val db: SQLiteDatabase = run {
        file.parentFile?.mkdirs()
        SQLiteDatabase.openOrCreateDatabase(file, null)
    }

    override fun execute(sql: String, vararg args: Any?) {
        if (args.isEmpty()) db.execSQL(sql) else db.execSQL(sql, args.map(::bindable).toTypedArray())
    }

    override fun <T> query(sql: String, vararg args: Any?, mapper: (SqlRow) -> T): List<T> {
        // rawQuery binds strings only; SQLite's type affinity compares them correctly with INTEGER columns.
        val stringArgs = args.map { it?.toString() }.toTypedArray()
        db.rawQuery(sql, stringArgs).use { cursor ->
            val row = CursorRow(cursor)
            val out = ArrayList<T>(cursor.count)
            while (cursor.moveToNext()) out += mapper(row)
            return out
        }
    }

    override fun <T> transaction(block: () -> T): T {
        db.beginTransaction()
        try {
            val result = block()
            db.setTransactionSuccessful()
            return result
        } finally {
            db.endTransaction()
        }
    }

    override fun close() = db.close()

    private fun bindable(value: Any?): Any? = when (value) {
        is Boolean -> if (value) 1L else 0L
        is Int -> value.toLong()
        is Float -> value.toDouble()
        else -> value
    }

    private class CursorRow(private val cursor: Cursor) : SqlRow {
        override fun string(column: String): String? = index(column)?.let { if (cursor.isNull(it)) null else cursor.getString(it) }
        override fun long(column: String): Long? = index(column)?.let { if (cursor.isNull(it)) null else cursor.getLong(it) }
        override fun bytes(column: String): ByteArray? = index(column)?.let { if (cursor.isNull(it)) null else cursor.getBlob(it) }
        private fun index(column: String): Int? = cursor.getColumnIndex(column).takeIf { it >= 0 }
    }

    companion object {
        val FACTORY = SqlDatabaseFactory { AndroidSqlDatabase(it) }
    }
}
