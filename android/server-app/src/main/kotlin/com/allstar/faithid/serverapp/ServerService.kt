package com.allstar.faithid.serverapp

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.allstar.faithid.androidsupport.AndroidSqlDatabase
import com.allstar.faithid.androidsupport.FileLog
import com.allstar.faithid.server.ServerConfig
import com.allstar.faithid.server.ServerLog
import com.allstar.faithid.server.ServerSettings
import com.allstar.faithid.server.http.FaithServer
import com.allstar.faithid.serverapp.ui.MainActivity
import java.io.File
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.Executors

/**
 * Keeps the DB server running in a foreground service, so it survives the screen turning off and
 * the app going to the background.
 */
class ServerService : Service() {

    interface Listener {
        fun onStateChanged()
        fun onLog(line: String)
    }

    inner class LocalBinder : Binder() {
        val service: ServerService get() = this@ServerService
    }

    private val binder = LocalBinder()
    private val listeners = CopyOnWriteArraySet<Listener>()
    private val worker = Executors.newSingleThreadExecutor { Thread(it, "faith-server-control") }
    private var wakeLock: PowerManager.WakeLock? = null

    lateinit var log: FileLog
        private set

    @Volatile
    var server: FaithServer? = null
        private set

    @Volatile
    var lastError: String? = null
        private set

    override fun onCreate() {
        super.onCreate()
        log = FileLog(File(filesDir, "logs/server.log"), "FaithServer")
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        server?.close()
        server = null
        wakeLock?.release()
        worker.shutdown()
        super.onDestroy()
    }

    fun addListener(listener: Listener) = listeners.add(listener)

    fun removeListener(listener: Listener) = listeners.remove(listener)

    val isRunning: Boolean get() = server?.isRunning == true

    /** Opens the database in [settings] and starts listening. Runs off the main thread. */
    fun start(settings: ServerSettings, onDone: (Boolean) -> Unit = {}) {
        worker.execute {
            val ok = runCatching {
                val current = server
                val needsNewInstance = current == null ||
                    current.config.dataDir.absolutePath != File(settings.dataDir).absolutePath ||
                    current.config.port != settings.port ||
                    current.config.maxEmbeddingsPerPerson != settings.maxEmbeddingsPerPerson
                if (needsNewInstance) {
                    current?.close()
                    server = FaithServer.create(
                        ServerConfig(
                            port = settings.port,
                            dataDir = File(settings.dataDir),
                            stateDir = stateDir(),
                            maxEmbeddingsPerPerson = settings.maxEmbeddingsPerPerson,
                        ),
                        AndroidSqlDatabase.FACTORY,
                        log = ServerLog { line ->
                            log.log(line)
                            listeners.forEach { it.onLog(line) }
                        },
                    )
                }
                val started = server!!.start()
                if (started) goForeground(settings.port)
                lastError = if (started) null else "포트 ${settings.port}에서 서버를 시작하지 못했습니다."
                started
            }.getOrElse {
                lastError = "${it.javaClass.simpleName}: ${it.message}"
                log.log("server start failed: $lastError")
                false
            }
            notifyChanged()
            onDone(ok)
        }
    }

    fun stop(onDone: () -> Unit = {}) {
        worker.execute {
            server?.stop()
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            wakeLock?.release()
            wakeLock = null
            notifyChanged()
            onDone()
        }
    }

    /** Runs an admin action on the control thread and refreshes listeners afterwards. */
    fun <T> run(action: (FaithServer) -> T, onDone: (T?) -> Unit = {}) {
        worker.execute {
            val result = server?.let { runCatching { action(it) }.onFailure { e -> log.log("action failed: ${e.message}") }.getOrNull() }
            notifyChanged()
            onDone(result)
        }
    }

    /**
     * Opens the database without starting the HTTP server, so the administrator can look at it
     * before deciding to start.
     */
    fun open(settings: ServerSettings, onDone: () -> Unit = {}) {
        worker.execute {
            if (server == null) {
                runCatching {
                    server = FaithServer.create(
                        ServerConfig(settings.port, File(settings.dataDir), stateDir(), settings.maxEmbeddingsPerPerson),
                        AndroidSqlDatabase.FACTORY,
                        log = ServerLog { line ->
                            log.log(line)
                            listeners.forEach { it.onLog(line) }
                        },
                    )
                }.onFailure { lastError = "DB를 열 수 없습니다: ${it.message}" }
            }
            notifyChanged()
            onDone()
        }
    }

    fun shutdown() {
        stop { stopSelf() }
    }

    private fun stateDir(): File {
        val dir = File(filesDir, "face_db_server")
        dir.mkdirs()
        return dir
    }

    private fun notifyChanged() = listeners.forEach { it.onStateChanged() }

    private fun goForeground(port: Int) {
        val type = if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0
        ServiceCompat.startForeground(this, NOTIFICATION_ID, buildNotification(port), type)
        if (wakeLock == null) {
            wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FaithServer:running")
                .apply { setReferenceCounted(false); acquire() }
        }
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "DB 서버 상태", NotificationManager.IMPORTANCE_LOW))
    }

    private fun buildNotification(port: Int): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Faith DB Server 실행 중")
            .setContentText("키오스크 연결 대기 중 (포트 $port)")
            .setSmallIcon(android.R.drawable.stat_sys_upload_done)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "faith_server"
        private const val NOTIFICATION_ID = 1801
    }
}
