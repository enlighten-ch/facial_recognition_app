package com.example.facedbserver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlin.concurrent.thread
import java.util.concurrent.CopyOnWriteArraySet

class FaceDbServerService : Service() {
    interface Listener {
        fun onLog(line: String)
        fun onStateChanged()
    }

    inner class LocalBinder : Binder() {
        fun getService(): FaceDbServerService = this@FaceDbServerService
    }

    private val binder = LocalBinder()
    private val listeners = CopyOnWriteArraySet<Listener>()
    private val logBuffer = ArrayDeque<String>()

    private lateinit var serverEngine: FaceDbServerEngine
    private var activePort: Int = 18080
    @Volatile
    private var autoStartTriggered = false

    override fun onCreate() {
        super.onCreate()
        serverEngine = FaceDbServerEngine(this) { line ->
            appendLog(line)
        }
        ensureNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_SHUTDOWN_SERVICE) {
            shutdownService()
            return START_NOT_STICKY
        }
        return START_STICKY
    }

    override fun onDestroy() {
        listeners.clear()
        serverEngine.release()
        super.onDestroy()
    }

    fun registerListener(listener: Listener) {
        listeners.add(listener)
    }

    fun unregisterListener(listener: Listener) {
        listeners.remove(listener)
    }

    fun snapshotLogs(): List<String> {
        synchronized(logBuffer) {
            return logBuffer.toList()
        }
    }

    @Synchronized
    fun startServer(port: Int, dbStorageDir: String): Boolean {
        serverEngine.setDbStorageDir(dbStorageDir)
        val started = serverEngine.start(port)
        if (started) {
            activePort = port
            startForeground(NOTIFICATION_ID, buildNotification(running = true))
        }
        notifyStateChanged()
        return started
    }

    @Synchronized
    fun stopServer(): Boolean {
        val wasRunning = serverEngine.isRunning()
        if (wasRunning) {
            serverEngine.stop()
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        notifyStateChanged()
        return wasRunning
    }

    @Synchronized
    fun reloadDbFromStorage(dbStorageDir: String): Boolean {
        serverEngine.setDbStorageDir(dbStorageDir)
        val ok = serverEngine.reloadDbFromStorage()
        notifyStateChanged()
        return ok
    }

    @Synchronized
    fun reloadDbMetadataFromCsv(dbStorageDir: String): Boolean {
        serverEngine.setDbStorageDir(dbStorageDir)
        val ok = serverEngine.reloadDbMetadataFromCsv()
        notifyStateChanged()
        return ok
    }

    @Synchronized
    fun applyStorageDir(dbStorageDir: String) {
        serverEngine.setDbStorageDir(dbStorageDir)
    }

    fun isRunning(): Boolean = serverEngine.isRunning()

    fun endpointPort(): Int = if (serverEngine.isRunning()) serverEngine.endpointPort() else activePort

    fun pendingClientIds(): List<String> = serverEngine.pendingClientIds()

    fun approvedClientIds(): List<String> = serverEngine.approvedClientIds()

    fun clientStatuses(): List<ClientConnectionStatus> = serverEngine.clientStatuses()

    fun dbRowsSnapshot(): List<DbPersonRow> = serverEngine.dbRowsSnapshot()

    @Synchronized
    fun updatePersonRow(personKey: String, name: String, department: String, grade: String): Boolean {
        val ok = serverEngine.updatePersonRow(personKey, name, department, grade)
        notifyStateChanged()
        return ok
    }

    @Synchronized
    fun deletePersonRow(personKey: String): Boolean {
        val ok = serverEngine.deletePersonRow(personKey)
        notifyStateChanged()
        return ok
    }

    @Synchronized
    fun approveClient(clientId: String): Boolean {
        val ok = serverEngine.approveClient(clientId)
        notifyStateChanged()
        return ok
    }

    @Synchronized
    fun approveAllPending(): Int {
        val count = serverEngine.approveAllPending()
        notifyStateChanged()
        return count
    }

    @Synchronized
    fun shutdownService() {
        runCatching { stopServer() }
        stopSelf()
    }

    private fun appendLog(line: String) {
        synchronized(logBuffer) {
            logBuffer.addLast(line)
            while (logBuffer.size > 300) {
                logBuffer.removeFirst()
            }
        }
        listeners.forEach { listener ->
            runCatching { listener.onLog(line) }
        }
        notifyStateChanged()
    }

    private fun notifyStateChanged() {
        if (serverEngine.isRunning()) {
            runCatching {
                val manager = getSystemService(NotificationManager::class.java)
                manager.notify(NOTIFICATION_ID, buildNotification(running = true))
            }
        }
        listeners.forEach { listener ->
            runCatching { listener.onStateChanged() }
        }
    }

    private fun buildNotification(running: Boolean): Notification {
        val intent = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val pendingIntent = PendingIntent.getActivity(
            this,
            1001,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val contentText = if (running) {
            "실행중: 포트 $activePort"
        } else {
            "중지됨"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Faith DB 서버")
            .setContentText(contentText)
            .setOngoing(running)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(pendingIntent)
            .build()
    }

    private fun ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Faith DB 서버",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "DB 서버 실행 상태 알림"
            setShowBadge(false)
        }
        manager.createNotificationChannel(channel)
    }

    private fun triggerAutoStartIfNeeded() {
        if (autoStartTriggered || serverEngine.isRunning()) return
        autoStartTriggered = true
        val config = ServerConfigStore(this).load()
        if (config.dbStorageDir.isBlank()) {
            appendLog("경고: DB 경로 설정이 없어 서비스 자동 시작을 건너뜁니다.")
            return
        }
        thread(name = "dbserver-service-autostart") {
            val started = runCatching { startServer(config.defaultPort, config.dbStorageDir) }.getOrDefault(false)
            if (!started) {
                autoStartTriggered = false
                appendLog("서비스 자동 시작 보류: 이미 실행중이거나 시작 실패")
            } else {
                appendLog("서비스 자동 시작 완료")
            }
        }
    }

    companion object {
        private const val CHANNEL_ID = "faith_db_server_channel"
        private const val NOTIFICATION_ID = 18080
        const val ACTION_SHUTDOWN_SERVICE = "com.example.facedbserver.action.SHUTDOWN_SERVICE"
    }
}
