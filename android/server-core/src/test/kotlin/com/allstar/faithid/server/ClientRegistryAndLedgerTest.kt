package com.allstar.faithid.server

import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import com.allstar.faithid.server.attendance.AttendanceLedger
import com.allstar.faithid.server.auth.ClientRegistry
import com.allstar.faithid.server.auth.ClientRegistry.AccessStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

class ClientRegistryTest {
    @get:Rule
    val temp = TemporaryFolder()

    private var now = 0L
    private fun registry(maxPending: Int = 50) =
        ClientRegistry(JdbcSqlDatabase(temp.newFile("state.db")), Clock { ++now }, ServerLog.NONE, maxPending)

    private val secret = "s".repeat(32)

    @Test
    fun `pending until approved by the administrator`() {
        val r = registry()
        assertEquals(AccessStatus.PENDING, r.requestAccess("kiosk-1", "Tab S7+", secret))
        assertFalse(r.authenticate("kiosk-1", secret))
        assertTrue(r.isPending("kiosk-1"))
        assertTrue(r.approve("kiosk-1"))
        assertEquals(AccessStatus.APPROVED, r.requestAccess("kiosk-1", "Tab S7+", secret))
        assertTrue(r.authenticate("kiosk-1", secret))
        assertFalse(r.authenticate("kiosk-1", "x".repeat(32)))
        assertTrue(r.list().single().lastSeenAt!! > 0)
    }

    @Test
    fun `a known id with another secret is refused, approved or not`() {
        val r = registry()
        r.requestAccess("kiosk-1", "a", secret)
        assertEquals(AccessStatus.SECRET_MISMATCH, r.requestAccess("kiosk-1", "b", "t".repeat(32)))
        r.approve("kiosk-1")
        assertEquals(AccessStatus.SECRET_MISMATCH, r.requestAccess("kiosk-1", "b", "t".repeat(32)))
        assertTrue(r.authenticate("kiosk-1", secret))
    }

    @Test
    fun `malformed ids and short secrets are rejected`() {
        val r = registry()
        assertEquals(AccessStatus.INVALID_REQUEST, r.requestAccess("", "a", secret))
        assertEquals(AccessStatus.INVALID_REQUEST, r.requestAccess("bad id!", "a", secret))
        assertEquals(AccessStatus.INVALID_REQUEST, r.requestAccess("k".repeat(65), "a", secret))
        assertEquals(AccessStatus.INVALID_REQUEST, r.requestAccess("kiosk", "a", "short"))
        assertTrue(r.list().isEmpty())
    }

    @Test
    fun `approve all, revoke and the pending limit`() {
        val r = registry(maxPending = 3)
        (1..5).forEach { r.requestAccess("k$it", "d", secret) }
        assertEquals(3, r.list().size)
        assertEquals(listOf("k3", "k4", "k5"), r.list().map { it.clientId })
        assertEquals(3, r.approveAllPending())
        assertTrue(r.revoke("k4"))
        assertFalse(r.authenticate("k4", secret))
        assertFalse(r.revoke("k4"))
        assertFalse(r.approve("unknown"))
    }

    @Test
    fun `approvals from the old server stay valid with the old token as secret`() {
        val r = registry()
        val legacy = File(temp.root, "auth_state.json")
        legacy.writeText("""{"approvedTokensByClient": {"client-3065d7cf": "1de600af228b41deb78d5f4988b9eb0e", "bad id": "x"}}""")
        r.migrateLegacyAuthFile(legacy)
        assertTrue(r.authenticate("client-3065d7cf", "1de600af228b41deb78d5f4988b9eb0e"))
        assertEquals(1, r.list().size)
        assertFalse(legacy.exists())
        assertTrue(File(temp.root, "auth_state.json.migrated").exists())
    }
}

class AttendanceLedgerTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val zone = ZoneId.of("Asia/Seoul")
    private val day = LocalDate.of(2026, 10, 25)
    private val faces = FakeFaces(seed = 4, dim = 8)
    private val roster = listOf(faces.person("홍길동", key = "hong"), faces.person("김철수", key = "kim"))

    private fun at(hour: Int, date: LocalDate = day) = ZonedDateTime.of(date.atTime(hour, 0), zone).toInstant().toEpochMilli()

    private fun event(id: String, key: String, time: Long) =
        AttendanceEvent(id, key, if (key == "hong") "홍길동" else "김철수", "청년부", "1학년", "kiosk-1", time, CheckInSource.RECOGNIZED, 0.7f)

    @Test
    fun `stores each event once and keeps the daily sheet current`() {
        val out = temp.newFolder("faceDB")
        val ledger = AttendanceLedger(JdbcSqlDatabase(temp.newFile("a.db")), out, zone, Clock { at(12) }, ServerLog.NONE) { roster }
        assertEquals(2, ledger.record(listOf(event("e1", "hong", at(9)), event("e2", "kim", at(10))), "kiosk-1"))
        assertEquals(0, ledger.record(listOf(event("e1", "hong", at(9))), "kiosk-1"))
        assertEquals(1, ledger.record(listOf(event("e3", "hong", at(11))), "kiosk-2"))
        assertEquals(3, ledger.eventsBetween(day, day).size)
        val sheet = Csv.parse(File(out, "attendance_2026-10-25.csv").readText())
        val hong = sheet.first { it[0] == "홍길동" }
        assertEquals("출석", hong[3])
        assertEquals("2", hong[6])
        assertEquals(day, ledger.today())
        val rows = ledger.dailyRows(day)
        assertEquals(2, rows.count { it.present })
    }

    @Test
    fun `range export covers every day of the period`() {
        val out = temp.newFolder("faceDB")
        val ledger = AttendanceLedger(JdbcSqlDatabase(temp.newFile("a.db")), out, zone, Clock { 0 }, ServerLog.NONE) { roster }
        ledger.record(listOf(event("e1", "hong", at(9)), event("e2", "hong", at(9, day.plusDays(6)))), "k")
        val file = ledger.exportRange(day, day.plusDays(6))
        val table = Csv.parse(file.readText())
        assertEquals(4 + 7, table[1].size)
        assertEquals("2", table.first { it[0] == "홍길동" }[3])
        assertTrue(File(out, "attendance_2026-10-31.csv").exists())
    }
}
