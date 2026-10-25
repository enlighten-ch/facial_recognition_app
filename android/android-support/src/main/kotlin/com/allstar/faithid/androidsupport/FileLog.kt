package com.allstar.faithid.androidsupport

import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Diagnostic log kept in memory (for the in-app log viewer) and appended to a file. The file is
 * rotated at [maxBytes] so a kiosk running for months does not fill the storage.
 */
class FileLog(
    private val file: File,
    private val tag: String,
    private val memoryLines: Int = 500,
    private val maxBytes: Long = 2L * 1024 * 1024,
) {
    private val buffer = ArrayDeque<String>()
    private val writer = Executors.newSingleThreadExecutor { Thread(it, "$tag-log").apply { isDaemon = true } }
    private val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.KOREA)

    fun log(message: String) {
        val line = "[${synchronized(timestamp) { timestamp.format(Date()) }}] $message"
        Log.i(tag, message)
        synchronized(buffer) {
            buffer.addLast(line)
            while (buffer.size > memoryLines) buffer.removeFirst()
        }
        writer.execute {
            runCatching {
                file.parentFile?.mkdirs()
                if (file.length() > maxBytes) {
                    val previous = File(file.parentFile, file.name + ".1")
                    previous.delete()
                    file.renameTo(previous)
                }
                file.appendText(line + "\n")
            }
        }
    }

    /** Recent lines, oldest first. */
    fun recent(): List<String> = synchronized(buffer) { buffer.toList() }

    /** The file contents if readable, otherwise the in-memory lines. */
    fun readAll(): String = runCatching { file.readText() }.getOrNull()?.takeIf { it.isNotBlank() } ?: recent().joinToString("\n")

    fun clear() {
        synchronized(buffer) { buffer.clear() }
        writer.execute { runCatching { file.writeText("") } }
    }

    val path: String get() = file.absolutePath
}
