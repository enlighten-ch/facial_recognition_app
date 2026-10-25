package com.allstar.faithid.server.attendance

import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.AttendanceReport
import com.allstar.faithid.core.attendance.AttendanceRow
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerLog
import java.io.File
import java.time.LocalDate
import java.time.ZoneId
import java.util.concurrent.ConcurrentSkipListSet
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Stores check-in events (`attendance.db`) and keeps the daily sheet
 * `attendance_YYYY-MM-DD.csv` up to date in the data folder.
 *
 * Events are deduplicated by event id, so a kiosk may resend after a lost response. The sheet is
 * rewritten in the background: it covers the whole roster, and writing it inside the request made
 * every check-in wait for it.
 */
class AttendanceLedger(
    private val db: SqlDatabase,
    private val outputDir: File,
    private val zone: ZoneId,
    private val clock: Clock,
    private val log: ServerLog,
    private val roster: () -> List<Person>,
) {
    private val report = AttendanceReport(zone)
    private val sheetWriter = Executors.newSingleThreadExecutor { Thread(it, "faith-attendance-sheet").apply { isDaemon = true } }
    private val staleSheets = ConcurrentSkipListSet<LocalDate>()
    private val fileLock = Any()

    init {
        db.execute(
            "CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, person_key TEXT, name TEXT, department TEXT, " +
                "grade TEXT, kiosk_id TEXT, occurred_at INTEGER, source TEXT, score REAL, received_at INTEGER, client_id TEXT)",
        )
        db.execute("CREATE INDEX IF NOT EXISTS events_occurred ON events(occurred_at)")
    }

    /** Stores new events and refreshes the affected daily sheets. Returns how many were new. */
    fun record(events: List<AttendanceEvent>, clientId: String): Int {
        val (inserted, dates) = synchronized(this) {
            var inserted = 0
            val dates = sortedSetOf<LocalDate>()
            db.transaction {
                events.forEach { event ->
                    val exists = db.query("SELECT 1 AS x FROM events WHERE event_id = ?", event.eventId) { true }.isNotEmpty()
                    if (!exists) {
                        db.execute(
                            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            event.eventId, event.personKey, event.name, event.department, event.grade, event.kioskId,
                            event.occurredAt, event.source.name, event.score.toDouble(), clock.now(), clientId,
                        )
                        inserted++
                        dates += report.dateOf(event)
                    }
                }
            }
            inserted to dates
        }
        if (dates.isNotEmpty()) {
            staleSheets += dates
            sheetWriter.execute(::writeStaleSheets)
        }
        if (inserted > 0) log.log("check-ins stored: client=$clientId, new=$inserted, received=${events.size}")
        return inserted
    }

    @Synchronized
    fun eventsBetween(from: LocalDate, to: LocalDate): List<AttendanceEvent> {
        val start = from.atStartOfDay(zone).toInstant().toEpochMilli()
        val end = to.plusDays(1).atStartOfDay(zone).toInstant().toEpochMilli()
        return db.query(
            "SELECT * FROM events WHERE occurred_at >= ? AND occurred_at < ? ORDER BY occurred_at",
            start,
            end,
        ) { row ->
            AttendanceEvent(
                eventId = row.string("event_id").orEmpty(),
                personKey = row.string("person_key").orEmpty(),
                name = row.string("name").orEmpty(),
                department = row.string("department").orEmpty(),
                grade = row.string("grade").orEmpty(),
                kioskId = row.string("kiosk_id").orEmpty(),
                occurredAt = row.long("occurred_at") ?: 0L,
                source = runCatching { CheckInSource.valueOf(row.string("source").orEmpty()) }.getOrDefault(CheckInSource.RECOGNIZED),
                score = row.string("score")?.toFloatOrNull() ?: 0f,
            )
        }
    }

    fun today(): LocalDate = java.time.Instant.ofEpochMilli(clock.now()).atZone(zone).toLocalDate()

    fun dailyRows(date: LocalDate): List<AttendanceRow> = report.daily(date, roster(), eventsBetween(date, date))

    /** "HH:mm:ss" in the server's time zone. */
    fun formatTime(epochMillis: Long): String = report.formatTime(epochMillis)

    fun writeDailyCsv(date: LocalDate): File {
        val file = File(outputDir, AttendanceReport.fileName(date))
        val text = report.dailyCsv(date, dailyRows(date))
        synchronized(fileLock) { writeAtomically(file, text) }
        return file
    }

    /** Waits until the daily sheets include every check-in recorded so far. */
    fun flushSheets() {
        sheetWriter.submit {}.get()
    }

    fun close() {
        sheetWriter.shutdown()
        sheetWriter.awaitTermination(10, TimeUnit.SECONDS)
    }

    /** Several queued requests collapse into one rewrite per date. */
    private fun writeStaleSheets() {
        while (true) {
            val date = staleSheets.pollFirst() ?: return
            runCatching { writeDailyCsv(date) }.onFailure { log.log("attendance sheet not written: ${it.message}") }
        }
    }

    /** Writes `attendance_<from>_<to>.csv` with one column per day. */
    fun exportRange(from: LocalDate, to: LocalDate): File {
        val file = File(outputDir, "attendance_${from}_$to.csv")
        val text = report.rangeCsv(from, to, roster(), eventsBetween(from, to))
        synchronized(fileLock) { writeAtomically(file, text) }
        log.log("attendance exported: ${file.name}")
        return file
    }

    private fun writeAtomically(file: File, text: String) {
        outputDir.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(text)
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
    }
}
