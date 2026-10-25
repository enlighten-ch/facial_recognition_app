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
 */
class ClientRegistry(
    private val db: SqlDatabase,
    private val clock: Clock,
    private val log: ServerLog,
    private val maxPending: Int = 50,
) {
    enum class AccessStatus { APPROVED, PENDING, SECRET_MISMATCH, INVALID_REQUEST }

    init {
        db.execute(
            "CREATE TABLE IF NOT EXISTS clients(client_id TEXT PRIMARY KEY, device_name TEXT, secret_hash TEXT, " +
                "approved INTEGER, requested_at INTEGER, approved_at INTEGER, last_seen_at INTEGER)",
        )
    }

    /** A kiosk asks for access, or checks whether it has been approved. */
    @Synchronized
    fun requestAccess(clientId: String, deviceName: String, secret: String): AccessStatus {
        if (!isValidClientId(clientId) || secret.length < MIN_SECRET_LENGTH) return AccessStatus.INVALID_REQUEST
        val hash = hash(secret)
        val existing = record(clientId)
        if (existing != null) {
            val storedHash = storedHash(clientId)
            // Another device is using this id. A reinstalled kiosk generates a new id, so never hand
            // an existing entry (approved or pending) to a different secret.
            if (!constantTimeEquals(storedHash, hash)) return AccessStatus.SECRET_MISMATCH
            touch(clientId)
            return if (existing.approved) AccessStatus.APPROVED else AccessStatus.PENDING
        }
        dropOldestPendingIfFull()
        db.execute(
            "INSERT INTO clients(client_id, device_name, secret_hash, approved, requested_at, approved_at, last_seen_at) VALUES(?,?,?,?,?,?,?)",
            clientId,
            deviceName.take(80),
            hash,
            0,
            clock.now(),
            null,
            clock.now(),
        )
        log.log("approval requested: client=$clientId, device=$deviceName")
        return AccessStatus.PENDING
    }

    /** True if [clientId] is approved and [secret] matches. Updates the last-seen time. */
    @Synchronized
    fun authenticate(clientId: String, secret: String): Boolean {
        if (clientId.isBlank() || secret.isBlank()) return false
        val row = db.query("SELECT secret_hash, approved FROM clients WHERE client_id = ?", clientId) {
            it.string("secret_hash").orEmpty() to (it.long("approved") == 1L)
        }.firstOrNull() ?: return false
        if (!row.second || !constantTimeEquals(row.first, hash(secret))) return false
        touch(clientId)
        return true
    }

    @Synchronized
    fun isPending(clientId: String): Boolean = record(clientId)?.approved == false

    @Synchronized
    fun approve(clientId: String): Boolean {
        val existing = record(clientId) ?: return false
        if (!existing.approved) {
            db.execute("UPDATE clients SET approved = 1, approved_at = ? WHERE client_id = ?", clock.now(), clientId)
            log.log("client approved: $clientId")
        }
        return true
    }

    @Synchronized
    fun approveAllPending(): Int {
        val pending = list().filter { !it.approved }
        pending.forEach { approve(it.clientId) }
        return pending.size
    }

    /** Removes the kiosk entirely; it has to request approval again. */
    @Synchronized
    fun revoke(clientId: String): Boolean {
        val existed = record(clientId) != null
        db.execute("DELETE FROM clients WHERE client_id = ?", clientId)
        if (existed) log.log("client removed: $clientId")
        return existed
    }

    @Synchronized
    fun list(): List<ClientRecord> =
        db.query("SELECT client_id, device_name, approved, requested_at, approved_at, last_seen_at FROM clients ORDER BY client_id") { row ->
            ClientRecord(
                clientId = row.string("client_id").orEmpty(),
                deviceName = row.string("device_name").orEmpty(),
                approved = row.long("approved") == 1L,
                requestedAt = row.long("requested_at") ?: 0L,
                approvedAt = row.long("approved_at"),
                lastSeenAt = row.long("last_seen_at"),
            )
        }

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
            if (isValidClientId(clientId) && record(clientId) == null) {
                db.execute(
                    "INSERT INTO clients(client_id, device_name, secret_hash, approved, requested_at, approved_at, last_seen_at) VALUES(?,?,?,?,?,?,?)",
                    clientId, "(이전 버전에서 승인)", hash(token), 1, clock.now(), clock.now(), null,
                )
                imported++
            }
        }
        file.renameTo(File(file.parentFile, file.name + ".migrated"))
        log.log("legacy approvals imported: $imported")
    }

    private fun record(clientId: String): ClientRecord? = list().firstOrNull { it.clientId == clientId }

    private fun storedHash(clientId: String): String =
        db.query("SELECT secret_hash FROM clients WHERE client_id = ?", clientId) { it.string("secret_hash").orEmpty() }.firstOrNull().orEmpty()

    private fun touch(clientId: String) {
        db.execute("UPDATE clients SET last_seen_at = ? WHERE client_id = ?", clock.now(), clientId)
    }

    private fun dropOldestPendingIfFull() {
        val pending = list().filter { !it.approved }.sortedBy { it.requestedAt }
        if (pending.size >= maxPending) {
            pending.take(pending.size - maxPending + 1).forEach {
                db.execute("DELETE FROM clients WHERE client_id = ?", it.clientId)
                log.log("pending request dropped (too many pending): ${it.clientId}")
            }
        }
    }

    companion object {
        const val MIN_SECRET_LENGTH = 16
        private val CLIENT_ID = Regex("[A-Za-z0-9._-]{1,64}")

        fun isValidClientId(clientId: String): Boolean = CLIENT_ID.matches(clientId)

        fun hash(secret: String): String =
            MessageDigest.getInstance("SHA-256").digest(secret.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

        private fun constantTimeEquals(a: String, b: String): Boolean =
            MessageDigest.isEqual(a.toByteArray(Charsets.US_ASCII), b.toByteArray(Charsets.US_ASCII))
    }
}
