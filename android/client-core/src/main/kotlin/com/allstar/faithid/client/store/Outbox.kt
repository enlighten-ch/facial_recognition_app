package com.allstar.faithid.client.store

import com.allstar.faithid.core.attendance.AttendanceCodec
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.PersonOpCodec

/**
 * Changes made on this kiosk that the server has not confirmed yet. Stored in the kiosk database,
 * so they survive restarts and network outages. Entries leave the outbox only when the server has
 * answered for them.
 */
internal class Outbox(private val db: SqlDatabase) {

    init {
        db.execute("CREATE TABLE IF NOT EXISTS outbox_ops(seq INTEGER PRIMARY KEY AUTOINCREMENT, op_id TEXT UNIQUE, payload BLOB, created_at INTEGER)")
        db.execute("CREATE TABLE IF NOT EXISTS outbox_attendance(seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, payload BLOB, created_at INTEGER)")
    }

    fun addOp(op: PersonOp, now: Long) {
        db.execute("INSERT OR IGNORE INTO outbox_ops(op_id, payload, created_at) VALUES(?,?,?)", op.opId, PersonOpCodec.encodeOne(op), now)
    }

    fun addAttendance(event: AttendanceEvent, now: Long) {
        db.execute(
            "INSERT OR IGNORE INTO outbox_attendance(event_id, payload, created_at) VALUES(?,?,?)",
            event.eventId,
            AttendanceCodec.encodeOne(event),
            now,
        )
    }

    /** Oldest first, so the server sees a registration before samples added to that person. */
    fun ops(limit: Int = Int.MAX_VALUE): List<PersonOp> =
        db.query("SELECT payload FROM outbox_ops ORDER BY seq LIMIT ?", limit.toLong()) { row ->
            row.bytes("payload")?.let { runCatching { PersonOpCodec.decodeOne(it) }.getOrNull() }
        }.filterNotNull()

    fun attendance(limit: Int = Int.MAX_VALUE): List<AttendanceEvent> =
        db.query("SELECT payload FROM outbox_attendance ORDER BY seq LIMIT ?", limit.toLong()) { row ->
            row.bytes("payload")?.let { runCatching { AttendanceCodec.decodeOne(it) }.getOrNull() }
        }.filterNotNull()

    fun removeOps(opIds: Collection<String>) = opIds.forEach { db.execute("DELETE FROM outbox_ops WHERE op_id = ?", it) }

    fun removeAttendance(eventIds: Collection<String>) = eventIds.forEach { db.execute("DELETE FROM outbox_attendance WHERE event_id = ?", it) }

    fun opCount(): Int = db.query("SELECT COUNT(*) AS n FROM outbox_ops") { it.long("n") ?: 0L }.first().toInt()

    fun attendanceCount(): Int = db.query("SELECT COUNT(*) AS n FROM outbox_attendance") { it.long("n") ?: 0L }.first().toInt()
}
