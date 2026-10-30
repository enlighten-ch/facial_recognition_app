package com.allstar.faithid.kiosk.data

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.allstar.faithid.client.sync.KioskIdentity
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.settings.KioskSettingsStore
import com.allstar.faithid.kiosk.ui.MainActivity

/**
 * Keeps the kiosk's data and server connection alive in a foreground service. When the DB server
 * app runs on the same tablet and is brought to the front, Android may destroy the kiosk screen
 * or kill a background process; the service keeps receiving changes and sending check-ins, and a
 * returning screen finds everything up to date.
 *
 * Camera and recognition stay with the screen: they are only needed while it is shown.
 */
class KioskSyncService : Service() {

    inner class LocalBinder : Binder() {
        val service: KioskSyncService get() = this@KioskSyncService
    }

    private val binder = LocalBinder()

    @Volatile
    var data: KioskData? = null
        private set

    /** The screen, handed to every [KioskData] this service opens. */
    var host: KioskData.Host? = null
        set(value) {
            field = value
            data?.host = value
        }

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(NotificationChannel(CHANNEL_ID, "서버 동기화", NotificationManager.IMPORTANCE_LOW))
        showNotification()
        // The screen saves its settings before starting the service; after a restart by the system there is no screen.
        val store = KioskSettingsStore(this)
        open(KioskData.ensureIdentity(store.load()).also { store.save(it) })
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        data?.close()
        data = null
        super.onDestroy()
    }

    /** Also called once notifications are allowed: Android 13+ drops one posted before that. */
    fun showNotification() {
        val type = if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0
        runCatching { ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(), type) }
            .onFailure { KioskLog.get(this).log("동기화 서비스를 포그라운드로 올리지 못했습니다: ${it.message}") }
    }

    /** Opens (or reopens after a settings change) the local database and the server connection. */
    fun open(settings: KioskSettings) {
        val log = KioskLog.get(this)
        data?.close()
        data = KioskData(this, settings, KioskIdentity(settings.dbClientId, settings.dbClientSecret, settings.dbDeviceName), log)
            .also {
                it.host = host
                it.start()
            }
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Faith ID 키오스크")
            .setContentText("서버와 동기화를 유지하고 있습니다")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "faith_kiosk_sync"
        private const val NOTIFICATION_ID = 1802
    }
}
