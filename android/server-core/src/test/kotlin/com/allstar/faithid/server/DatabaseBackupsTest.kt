package com.allstar.faithid.server

import com.allstar.faithid.server.people.DatabaseBackups
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.ZoneOffset
import java.util.concurrent.atomic.AtomicLong

class DatabaseBackupsTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val day = 86_400_000L
    private val time = AtomicLong(1_762_000_000_000L)

    private fun backups() = DatabaseBackups(temp.newFolder("backups"), { it.writeText("db") }, Clock { time.get() }, ZoneOffset.UTC, ServerLog.NONE)

    @Test
    fun `one daily copy a day, the newest kept, other copies kept apart`() {
        val b = backups()
        b.dailyIfDue()
        b.dailyIfDue()
        assertEquals(1, b.list().size)
        repeat(20) {
            time.addAndGet(day)
            b.dailyIfDue()
        }
        repeat(25) { b.make("csv") }
        val all = b.list()
        assertEquals(DatabaseBackups.DAILY_KEEP, all.count { it.reason == DatabaseBackups.DAILY })
        assertEquals(DatabaseBackups.OTHER_KEEP, all.count { it.reason == "csv" })
        assertTrue(all.zipWithNext().all { (a, b) -> a.file.name >= b.file.name })
    }
}
