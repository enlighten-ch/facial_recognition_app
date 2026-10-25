package com.allstar.faithid.serverapp.ui

import java.net.Inet4Address
import java.net.NetworkInterface
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal object Format {
    private val dateTime = SimpleDateFormat("MM-dd HH:mm", Locale.KOREA)

    fun dateTime(epochMillis: Long?): String =
        if (epochMillis == null || epochMillis <= 0) "-" else synchronized(dateTime) { dateTime.format(Date(epochMillis)) }

    /** "3분 전" style text for the last time a kiosk was seen. */
    fun ago(epochMillis: Long?, now: Long = System.currentTimeMillis()): String {
        if (epochMillis == null || epochMillis <= 0) return "기록 없음"
        val seconds = (now - epochMillis) / 1000
        return when {
            seconds < 60 -> "방금"
            seconds < 3600 -> "${seconds / 60}분 전"
            seconds < 86_400 -> "${seconds / 3600}시간 전"
            else -> dateTime(epochMillis)
        }
    }

    /** IPv4 addresses kiosks can use to reach this tablet (Wi-Fi first). */
    fun localAddresses(): List<String> = runCatching {
        NetworkInterface.getNetworkInterfaces().toList()
            .filter { it.isUp && !it.isLoopback }
            .sortedBy { if (it.name.startsWith("wlan")) 0 else 1 }
            .flatMap { nif -> nif.inetAddresses.toList().filterIsInstance<Inet4Address>().map { it.hostAddress.orEmpty() } }
            .filter { it.isNotBlank() }
    }.getOrDefault(emptyList())
}
