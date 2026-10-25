package com.allstar.faithid.server.auth

import com.allstar.faithid.core.storage.SqlDatabase
import com.allstar.faithid.server.Clock
import com.allstar.faithid.server.ServerLog
import java.io.File
import java.security.MessageDigest

/** A kiosk known to the server. */
data class ClientRecord(
    val clientId: String,
    val deviceName: String,
    val approved: Boolean,
    val requestedAt: Long,
    val approvedAt: Long?,
    val lastSeenAt: Long?,
)

/**
 * Kiosk approvals.
 *
 * A kiosk proves its identity with a secret it generated itself; the server keeps only the
 * secret's SHA-256 and never returns secrets or tokens. Approving and revoking happen only through
 * this class's methods, which the server app calls from its UI after the administrator logs in;
 * there is no HTTP endpoint for them.
 *
 * Every kiosk request is authenticated, so the records are kept in memory and the database is
 * written only when something changes (the last-seen time at most once a minute per kiosk).
 */
class ClientRegistry(
    private val db: SqlDatabase,
    private val clock: Clock,
    private val log: ServerLog,
    private val maxPending: Int = 50,
) {
    enum class AccessStatus { APPROVED, PENDING, SECRET_MISMATCH, INVALID_REQUEST }

    private class Entry(var record: ClientRecord, val secretHash: String, var persistedLastSeen: Long?)

    private val entries = LinkedHashMap<String, Entry>()

    init {
        db.execute(
            "CREATE TABLE IF NOT EXISTS clients(client_id TEXT PRIMARY KEY, device_name TEXT, secret_hash TEXT, " +
                "approved INTEGER, requested_at INTEGER, approved_at INTEGER, last_seen_at INTEGER)",
        )
        db.query("SELECT * FROM clients ORDER BY client_id") { row ->
            val record = ClientRecord(
                clientId = row.string("client_id").orEmpty(),
                deviceName = row.string("device_name").orEmpty(),
                approved = row.long("approved") == 1L,
                requestedAt = row.long("requested_at") ?: 0L,
                approvedAt = row.long("approved_at"),
                lastSeenAt = row.long("last_seen_at"),
            )
            Entry(record, row.string("secret_hash").orEmpty(), record.lastSeenAt)
        }.forEach { entries[it.record.clientId] = it }
    }

    /** A kiosk asks for access, or checks whether it has been approved. */
    @Synchronized
    fun requestAccess(clientId: String, deviceName: String, secret: String): AccessStatus {
        if (!isValidClientId(clientId) || secret.length < MIN_SECRET_LENGTH) return AccessStatus.INVALID_REQUEST
        val hash = hash(secret)
        val existing = entries[clientId]
        if (existing != null) {
            // Another device is using this id. A reinstalled kiosk generates a new id, so never hand
            // an existing entry (approved or pending) to a different secret.
            if (!constantTimeEquals(existing.secretHash, hash)) return AccessStatus.SECRET_MISMATCH
            touch(existing)
            return if (existing.record.approved) AccessStatus.APPROVED else AccessStatus.PENDING
        }
        dropOldestPendingIfFull()
        val now = clock.now()
        val record = ClientRecord(clientId, deviceName.take(80), approved = false, requestedAt = now, approvedAt = null, lastSeenAt = now)
        db.execute(
            "INSERT INTO clients(client_id, device_name, secret_hash, approved, requested_at, approved_at, last_seen_at) VALUES(?,?,?,?,?,?,?)",
            clientId, record.deviceName, hash, 0, now, null, now,
        )
        entries[clientId] = Entry(record, hash, now)
        log.log("approval requested: client=$clientId, device=$deviceName")
        return AccessStatus.PENDING
    }

    /** True if [clientId] is approved and [secret] matches. Updates the last-seen time. */
    @Synchronized
    fun authenticate(clientId: String, secret: String): Boolean {
        if (clientId.isBlank() || secret.isBlank()) return false
        val entry = entries[clientId] ?: return false
        if (!entry.record.approved || !constantTimeEquals(entry.secretHash, hash(secret))) return false
        touch(entry)
        return true
    }

    @Synchronized
    fun isPending(clientId: String): Boolean = entries[clientId]?.record?.approved == false

    @Synchronized
    fun approve(clientId: String): Boolean {
        val entry = entries[clientId] ?: return false
        if (!entry.record.approved) {
            val now = clock.now()
            db.execute("UPDATE clients SET approved = 1, approved_at = ? WHERE client_id = ?", now, clientId)
            entry.record = entry.record.copy(approved = true, approvedAt = now)
            log.log("client approved: $clientId")
        }
        return true
    }

    @Synchronized
    fun approveAllPending(): Int {
        val pending = entries.values.filter { !it.record.approved }.map { it.record.clientId }
        pending.forEach { approve(it) }
        return pending.size
    }

    /** Removes the kiosk entirely; it has to request approval again. */
    @Synchronized
    fun revoke(clientId: String): Boolean {
        entries.remove(clientId) ?: return false
        db.execute("DELETE FROM clients WHERE client_id = ?", clientId)
        log.log("client removed: $clientId")
        return true
    }

    @Synchronized
    fun list(): List<ClientRecord> = entries.values.map { it.record }.sortedBy { it.clientId }

    /**
     * Imports approvals from the 2026-05 server (`auth_state.json`). Old kiosks keep their stored
     * token and now use it as their secret, so they stay approved after the upgrade.
     */
    @Synchronized
    fun migrateLegacyAuthFile(file: File) {
        if (!file.exists()) return
        val text = runCatching { file.readText() }.getOrNull() ?: return
        val block = Regex("\"approvedTokensByClient\"\\s*:\\s*\\{([^}]*)\\}").find(text)?.groupValues?.get(1).orEmpty()
        var imported = 0
        Regex("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"").findAll(block).forEach { match ->
            val (clientId, token) = match.destructured
            if (isValidClientId(clientId) && clientId !in entries) {
                val now = clock.now()
                val record = ClientRecord(clientId, LEGACY_DEVICE_NAME, approved = true, requestedAt = now, approvedAt = now, lastSeenAt = null)
                db.execute(
                    "INSERT INTO clients(client_id, device_name, secret_hash, approved, requested_at, approved_at, last_seen_at) VALUES(?,?,?,?,?,?,?)",
                    clientId, record.deviceName, hash(token), 1, now, now, null,
                )
                entries[clientId] = Entry(record, hash(token), null)
                imported++
            }
        }
        file.renameTo(File(file.parentFile, file.name + ".migrated"))
        log.log("legacy approvals imported: $imported")
    }

    private fun touch(entry: Entry) {
        val now = clock.now()
        entry.record = entry.record.copy(lastSeenAt = now)
        val persisted = entry.persistedLastSeen
        if (persisted == null || now - persisted >= LAST_SEEN_PERSIST_INTERVAL_MS) {
            db.execute("UPDATE clients SET last_seen_at = ? WHERE client_id = ?", now, entry.record.clientId)
            entry.persistedLastSeen = now
        }
    }

    private fun dropOldestPendingIfFull() {
        val pending = entries.values.filter { !it.record.approved }.sortedBy { it.record.requestedAt }
        if (pending.size >= maxPending) {
            pending.take(pending.size - maxPending + 1).forEach {
                entries.remove(it.record.clientId)
                db.execute("DELETE FROM clients WHERE client_id = ?", it.record.clientId)
                log.log("pending request dropped (too many pending): ${it.record.clientId}")
            }
        }
    }

    companion object {
        const val MIN_SECRET_LENGTH = 16
        const val LAST_SEEN_PERSIST_INTERVAL_MS = 60_000L
        const val LEGACY_DEVICE_NAME = "(이전 버전에서 승인)"
        private val CLIENT_ID = Regex("[A-Za-z0-9._-]{1,64}")

        fun isValidClientId(clientId: String): Boolean = CLIENT_ID.matches(clientId)

        fun hash(secret: String): String =
            MessageDigest.getInstance("SHA-256").digest(secret.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

        private fun constantTimeEquals(a: String, b: String): Boolean =
            MessageDigest.isEqual(a.toByteArray(Charsets.US_ASCII), b.toByteArray(Charsets.US_ASCII))
    }
}
