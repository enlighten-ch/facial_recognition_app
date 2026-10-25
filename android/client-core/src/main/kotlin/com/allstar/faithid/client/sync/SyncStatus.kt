package com.allstar.faithid.client.sync

/** Connection state of a kiosk, shown on the kiosk screen and in the developer view. */
sealed class SyncStatus {
    /** Synchronization is turned off in the settings; the kiosk works with its local data only. */
    data object Disabled : SyncStatus()

    data class Connecting(val serverUrl: String) : SyncStatus()

    /** The server did not answer. The kiosk keeps working offline and retries. */
    data class Offline(val reason: String, val retryInMillis: Long) : SyncStatus()

    /** Connected, but the administrator has not approved this kiosk yet. */
    data class AwaitingApproval(val clientId: String, val deviceName: String) : SyncStatus()

    /** Another device already uses this kiosk's id; a new id is needed (settings). */
    data class Rejected(val reason: String) : SyncStatus()

    /** Approved and connected. */
    data class Online(
        val lastSyncAt: Long,
        val pendingChanges: Int,
        val pendingCheckIns: Int,
        val serverRevision: Long,
    ) : SyncStatus()

    /** True when the kiosk has a working, approved connection. */
    val isOnline: Boolean get() = this is Online
}

/** Callbacks from [SyncEngine]; called on the engine's thread. */
interface SyncListener {
    fun onStatus(status: SyncStatus) {}

    /** Server changes were applied to the local database ([changedPeople] people added, changed or removed). */
    fun onRemoteChangesApplied(changedPeople: Int, fullSnapshot: Boolean) {}

    fun onLog(message: String) {}
}
