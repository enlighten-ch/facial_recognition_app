package com.allstar.faithid.kiosk.data

import android.content.Context
import com.allstar.faithid.androidsupport.FileLog
import java.io.File

/** One log per process, shared by the screen and [KioskSyncService] (the log viewer shows both). */
object KioskLog {
    @Volatile
    private var instance: FileLog? = null

    fun get(context: Context): FileLog =
        instance ?: synchronized(this) {
            instance ?: FileLog(File(context.applicationContext.filesDir, "logs/kiosk.log"), "FaithKiosk").also { instance = it }
        }
}
