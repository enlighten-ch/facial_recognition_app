package com.allstar.faithid.server.people

import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerLog
import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Copies of the face database in `<data folder>/backups`: one a day, and one before every change
 * that touches many people at once (applying face_db.csv, restoring). The newest [DAILY_KEEP]
 * daily and [OTHER_KEEP] other copies are kept, so a CSV deleted or emptied by mistake can be undone.
 */
class DatabaseBackups(
    val directory: File,
    private val write: (File) -> Unit,
    private val clock: Clock,
    private val zone: ZoneId,
    private val log: ServerLog,
) {
    data class Backup(val file: File, val reason: String, val at: Long) {
        val sizeBytes: Long get() = file.length()
    }

    @Volatile
    private var lastDaily: LocalDate? = null

    /** Makes a copy; returns it, or null if it could not be written (the change goes ahead regardless). */
    fun make(reason: String): File? {
        val stamp = STAMP.format(Instant.ofEpochMilli(clock.now()).atZone(zone))
        var file = File(directory, "$PREFIX${stamp}_$reason.db")
        var n = 2
        while (file.exists()) file = File(directory, "$PREFIX${stamp}-${n++}_$reason.db")
        return runCatching {
            write(file)
            prune()
            log.log("database backup: ${file.name}")
            file
        }.getOrElse {
            file.delete()
            log.log("database backup failed: ${it.javaClass.simpleName}: ${it.message}")
            null
        }
    }

    /** Makes today's copy if there is none yet. Cheap to call often. */
    fun dailyIfDue() {
        val today = Instant.ofEpochMilli(clock.now()).atZone(zone).toLocalDate()
        if (lastDaily == today) return
        val stamp = DAY.format(today)
        if (list().none { it.reason == DAILY && it.file.name.startsWith(PREFIX + stamp) }) make(DAILY)
        lastDaily = today
    }

    /** Newest first. */
    fun list(): List<Backup> =
        directory.listFiles()?.mapNotNull { file ->
            val match = NAME.matchEntire(file.name) ?: return@mapNotNull null
            Backup(file, match.groupValues[2], file.lastModified())
        }?.sortedByDescending { it.file.name }.orEmpty()

    private fun prune() {
        val (daily, other) = list().partition { it.reason == DAILY }
        (daily.drop(DAILY_KEEP) + other.drop(OTHER_KEEP)).forEach { it.file.delete() }
    }

    companion object {
        const val DAILY = "daily"
        const val DAILY_KEEP = 14
        const val OTHER_KEEP = 20
        const val PREFIX = "face_meta_"
        private val STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")
        private val DAY = DateTimeFormatter.ofPattern("yyyyMMdd")
        private val NAME = Regex("""face_meta_(\d{8}-\d{6}(?:-\d+)?)_([a-z]+)\.db""")
    }
}
