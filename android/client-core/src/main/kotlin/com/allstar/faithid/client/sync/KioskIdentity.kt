package com.allstar.faithid.client.sync

import java.security.SecureRandom

/**
 * How this kiosk identifies itself to the server. [secret] is generated once on the kiosk and kept
 * in its settings; the server stores only a hash of it.
 */
data class KioskIdentity(
    val clientId: String,
    val secret: String,
    val deviceName: String,
) {
    companion object {
        private val random = SecureRandom()

        fun newClientId(): String = "client-" + hex(4)

        fun newSecret(): String = hex(24)

        private fun hex(bytes: Int): String = ByteArray(bytes).also { random.nextBytes(it) }.joinToString("") { "%02x".format(it) }
    }
}

data class SyncConfig(
    val serverUrl: String,
    val identity: KioskIdentity,
    val connectTimeoutMillis: Long = 6_000,
    val readTimeoutMillis: Long = 60_000,
    /** First retry delay after a failure; doubles up to [maxRetryDelayMillis]. */
    val retryDelayMillis: Long = 3_000,
    val maxRetryDelayMillis: Long = 60_000,
    /** How often to ask whether the kiosk has been approved. */
    val approvalPollMillis: Long = 3_000,
    /** Pull even without a notification at this interval, in case the update socket is down. */
    val pollIntervalMillis: Long = 30_000,
    /** Longest wait for the camera to be idle before applying downloaded changes anyway. */
    val applyIdleWaitMillis: Long = 15_000,
    val maxOpsPerRequest: Int = 40,
    val maxCheckInsPerRequest: Int = 200,
) {
    /** Server URL with scheme and without trailing slash; empty if not configured. */
    val baseUrl: String
        get() {
            val trimmed = serverUrl.trim().trimEnd('/')
            if (trimmed.isEmpty()) return ""
            return if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) trimmed else "http://$trimmed"
        }
}
