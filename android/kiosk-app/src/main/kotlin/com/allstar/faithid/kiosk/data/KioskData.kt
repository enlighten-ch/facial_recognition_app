package com.allstar.faithid.kiosk.data

import android.content.Context
import android.os.Build
import com.allstar.faithid.androidsupport.AndroidSqlDatabase
import com.allstar.faithid.androidsupport.FileLog
import com.allstar.faithid.client.store.LocalFaceRepository
import com.allstar.faithid.client.sync.KioskIdentity
import com.allstar.faithid.client.sync.SyncConfig
import com.allstar.faithid.client.sync.SyncEngine
import com.allstar.faithid.client.sync.SyncListener
import com.allstar.faithid.client.sync.SyncStatus
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.FaceIndex
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.kiosk.flow.CheckIn
import com.allstar.faithid.kiosk.flow.Readiness
import com.allstar.faithid.kiosk.settings.KioskSettings
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

/**
 * The kiosk's data: the local face database (app-private storage), the connection to the DB
 * server and the check-in queue. One instance per settings version; [close] before replacing it.
 *
 * Owned by [KioskSyncService], so it keeps syncing while the kiosk screen is gone. The screen
 * attaches itself as [host] while it exists.
 */
class KioskData(
    context: Context,
    val settings: KioskSettings,
    val identity: KioskIdentity,
    private val log: FileLog,
) : AutoCloseable {

    /** The kiosk screen. Without one, server changes are applied at once. */
    interface Host {
        /** False while someone is in front of the camera, so the database does not change mid-recognition. */
        fun canApplyRemoteChanges(): Boolean
        fun onStatus(status: SyncStatus)
        fun onIndexChanged(index: FaceIndex)
    }

    /** Set by the screen while it exists; it is told the current index and status right away. */
    @Volatile
    var host: Host? = null
        set(value) {
            field = value
            value?.onIndexChanged(repository.index)
            value?.onStatus(status)
        }

    private val background = Executors.newSingleThreadExecutor { Thread(it, "faith-data") }
    val repository: LocalFaceRepository
    private val sync: SyncEngine?

    @Volatile
    var status: SyncStatus = if (settings.dbSyncEnabled) SyncStatus.Connecting(settings.dbServerBaseUrl) else SyncStatus.Disabled
        private set

    init {
        val dir = File(context.filesDir, DB_DIR)
        val store = PeopleStore.open(dir, AndroidSqlDatabase.FACTORY) { log.log("DB: $it") }
        if (!settings.dbSyncEnabled) importLegacyDatabaseIfEmpty(store, context)
        repository = LocalFaceRepository(store, settings.maxEmbeddingsPerPerson)
        repository.addIndexListener { index -> host?.onIndexChanged(index) }
        log.log("로컬 DB: ${repository.people().size}명, 미전송 변경 ${repository.pendingCounts().first}건, 미전송 출석 ${repository.pendingCounts().second}건")
        sync = if (settings.dbSyncEnabled) {
            SyncEngine(
                SyncConfig(
                    serverUrl = settings.dbServerBaseUrl,
                    identity = identity,
                    connectTimeoutMillis = settings.dbConnectTimeoutSeconds.coerceAtLeast(2) * 1000L,
                    retryDelayMillis = settings.dbConnectRetryDelaySeconds.coerceAtLeast(1) * 1000L,
                    applyIdleWaitMillis = (settings.adaptiveUpdateIdleSeconds.coerceAtLeast(1) * 1000L * 6),
                ),
                repository,
                object : SyncListener {
                    override fun onStatus(status: SyncStatus) {
                        if (status::class != this@KioskData.status::class) log.log("서버 연결 상태: ${describe(status)}")
                        this@KioskData.status = status
                        host?.onStatus(status)
                    }

                    override fun onRemoteChangesApplied(changedPeople: Int, fullSnapshot: Boolean) = Unit

                    override fun onLog(message: String) = log.log("동기화: $message")
                },
                canApplyNow = { host?.canApplyRemoteChanges() ?: true },
            )
        } else {
            null
        }
    }

    fun start() {
        sync?.start()
    }

    fun syncNow() = sync?.requestSync()

    val index: FaceIndex get() = repository.index

    /** Saves a registration locally (usable at once) and queues it for the server. */
    fun register(metadata: PersonMetadata, samples: List<FloatArray>): String? =
        runCatching { repository.register(metadata, samples).key }
            .onFailure { log.log("등록 저장 실패: ${it.javaClass.simpleName}: ${it.message}") }
            .getOrNull()
            ?.also { sync?.requestSync() }

    fun addSamples(personKey: String, samples: List<FloatArray>) {
        background.execute {
            runCatching { repository.addSamples(personKey, samples) }.onFailure { log.log("샘플 저장 실패: ${it.message}") }
            sync?.requestSync()
        }
    }

    fun checkIn(checkIn: CheckIn) {
        val event = AttendanceEvent(
            eventId = UUID.randomUUID().toString(),
            personKey = checkIn.personKey,
            name = checkIn.metadata.name,
            department = checkIn.metadata.department,
            grade = checkIn.metadata.grade,
            kioskId = identity.clientId,
            occurredAt = System.currentTimeMillis(),
            source = checkIn.source,
            score = checkIn.score,
        )
        background.execute {
            runCatching { repository.recordAttendance(event) }.onFailure { log.log("출석 저장 실패: ${it.message}") }
            log.log("출석: ${checkIn.metadata.label()} (${checkIn.source.label})")
            sync?.requestSync()
        }
    }

    /** Whether recognition can run: with local data it always can, even offline. */
    fun readiness(): Readiness {
        if (!repository.index.isEmpty() || !settings.dbSyncEnabled) return Readiness.Ready
        return when (val s = status) {
            is SyncStatus.Online -> Readiness.Ready
            is SyncStatus.AwaitingApproval -> Readiness.WaitingForApproval(s.clientId)
            is SyncStatus.Rejected -> Readiness.NoServer(s.reason)
            is SyncStatus.Offline -> Readiness.NoServer(s.reason)
            is SyncStatus.Connecting -> Readiness.NoServer("서버(${settings.dbServerBaseUrl})에 연결하는 중입니다.")
            SyncStatus.Disabled -> Readiness.Ready
        }
    }

    /** Short status for the operation screen when the kiosk works without a live connection. */
    fun banner(): String {
        val (ops, checkIns) = repository.pendingCounts()
        val pending = if (ops + checkIns > 0) " · 전송 대기 ${ops + checkIns}건" else ""
        return when (val s = status) {
            is SyncStatus.Online -> if (pending.isNotEmpty()) "서버 연결됨$pending" else ""
            is SyncStatus.Offline -> "오프라인으로 동작 중$pending"
            is SyncStatus.AwaitingApproval -> "서버 승인 대기 중 (${s.clientId})$pending"
            is SyncStatus.Rejected -> "서버 연결 거부: ${s.reason}"
            is SyncStatus.Connecting -> "서버 연결 중$pending"
            SyncStatus.Disabled -> ""
        }
    }

    override fun close() {
        sync?.stop()
        background.shutdown()
        background.awaitTermination(3, java.util.concurrent.TimeUnit.SECONDS)
        repository.close()
    }

    /**
     * Without a server, a kiosk upgraded from the 2026-05 version keeps its old database: it is
     * copied from the shared folder and migrated once. With a server, the kiosk downloads everything.
     */
    private fun importLegacyDatabaseIfEmpty(store: PeopleStore, context: Context) {
        if (store.loadPeople().isNotEmpty() || store.meta(META_LEGACY_IMPORTED) != null) return
        store.setMeta(META_LEGACY_IMPORTED, System.currentTimeMillis().toString())
        val legacyDir = File(settings.legacyFaceDataDir)
        val files = listOf(PeopleStore.DB_FILE, PeopleStore.LEGACY_BIN_FILE, PeopleStore.CSV_FILE).map { File(legacyDir, it) }
        if (files.none { it.exists() }) return
        val work = File(context.cacheDir, "legacy-import").apply { deleteRecursively(); mkdirs() }
        runCatching {
            files.filter { it.exists() }.forEach { it.copyTo(File(work, it.name), overwrite = true) }
            PeopleStore.open(work, AndroidSqlDatabase.FACTORY) { log.log("이전 DB: $it") }.use { legacy ->
                val people = legacy.loadPeople()
                store.transaction { people.forEach { store.upsert(it) } }
                log.log("이전 버전 DB를 가져왔습니다: ${people.size}명 (${legacyDir.absolutePath})")
            }
        }.onFailure { log.log("이전 DB 가져오기 실패: ${it.message}") }
        work.deleteRecursively()
    }

    companion object {
        const val DB_DIR = "faceDB"
        private const val META_LEGACY_IMPORTED = "legacy_imported_at"

        fun describe(status: SyncStatus): String = when (status) {
            is SyncStatus.Online -> "연결됨"
            is SyncStatus.Offline -> "오프라인 (${status.reason})"
            is SyncStatus.AwaitingApproval -> "승인 대기 (${status.clientId})"
            is SyncStatus.Rejected -> "거부됨 (${status.reason})"
            is SyncStatus.Connecting -> "연결 중"
            SyncStatus.Disabled -> "동기화 꺼짐"
        }

        /** Makes sure the kiosk has an id, a secret and a device name; returns settings to save if any were missing. */
        fun ensureIdentity(settings: KioskSettings): KioskSettings {
            var next = settings
            if (next.dbClientId.isBlank()) next = next.copy(dbClientId = KioskIdentity.newClientId())
            if (next.dbClientSecret.length < 16) next = next.copy(dbClientSecret = KioskIdentity.newSecret())
            if (next.dbDeviceName.isBlank()) next = next.copy(dbDeviceName = "${Build.MANUFACTURER} ${Build.MODEL}")
            return next
        }
    }
}
