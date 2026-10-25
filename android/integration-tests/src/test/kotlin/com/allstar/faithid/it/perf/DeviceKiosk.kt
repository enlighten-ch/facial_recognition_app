package com.allstar.faithid.it.perf

import java.util.concurrent.TimeUnit

/**
 * The real Faith ID kiosk app on an adb-connected device. Its sync log lines (logcat, epoch time)
 * show when it applied a server change; host and device clocks are aligned with a measured offset.
 */
class DeviceKiosk(private val serial: String) {
    private val receiver = "com.example.facetestbed/com.allstar.faithid.kiosk.debug.DebugObservationReceiver"

    /** Device clock minus host clock, in ms (median of a few round trips). */
    val clockOffsetMillis: Long = List(5) {
        val before = System.currentTimeMillis()
        val device = adb("shell", "date", "+%s%3N").trim().toLong()
        val after = System.currentTimeMillis()
        device - (before + after) / 2
    }.sorted()[2]

    /**
     * Brings the kiosk app to the front. Android may destroy it while another app is in front, and
     * it only syncs while its screen exists. Waits for it to report the server connection.
     */
    fun bringToFront() {
        val since = System.currentTimeMillis()
        adb("shell", "am", "start", "-n", "com.example.facetestbed/com.allstar.faithid.kiosk.ui.MainActivity")
        waitForLog("서버 연결 상태: 연결됨", since, timeoutMillis = 20_000)
        Thread.sleep(2_000)
    }

    /** No face in front of the camera, so downloaded changes are applied immediately. */
    fun pauseCamera(paused: Boolean) {
        adb("shell", "am", "broadcast", "-n", receiver, "-a", "com.allstar.faithid.kiosk.DEBUG_CAMERA", "--ez", "paused", "$paused")
        if (paused) adb("shell", "am", "broadcast", "-n", receiver, "-a", "com.allstar.faithid.kiosk.DEBUG_NO_FACE")
    }

    /**
     * Host time (ms) of the first kiosk log line after [afterHostMillis] that contains [text], or
     * null if none appears within [timeoutMillis].
     */
    fun waitForLog(text: String, afterHostMillis: Long, timeoutMillis: Long = 30_000): Long? {
        val deadline = System.currentTimeMillis() + timeoutMillis
        while (System.currentTimeMillis() < deadline) {
            adb("logcat", "-d", "-v", "epoch", "-s", "FaithKiosk:I").lineSequence()
                .filter { text in it }
                .mapNotNull { line -> line.trim().substringBefore(' ').toDoubleOrNull()?.let { (it * 1000).toLong() - clockOffsetMillis } }
                .firstOrNull { it > afterHostMillis }
                ?.let { return it }
            Thread.sleep(150)
        }
        return null
    }

    private fun adb(vararg args: String): String {
        val process = ProcessBuilder(listOf("adb", "-s", serial) + args).redirectErrorStream(true).start()
        val output = process.inputStream.bufferedReader().readText()
        process.waitFor(20, TimeUnit.SECONDS)
        return output
    }
}
