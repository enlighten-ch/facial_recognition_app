package com.allstar.faithid.it.perf

import okhttp3.Interceptor
import okhttp3.Response
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

/** A set of durations in milliseconds with the percentiles the report uses. */
class Samples {
    private val values = CopyOnWriteArrayList<Double>()

    fun add(ms: Double) {
        values += ms
    }

    val count: Int get() = values.size

    fun percentile(p: Double): Double {
        val sorted = values.sorted()
        if (sorted.isEmpty()) return Double.NaN
        val index = ((p / 100.0) * (sorted.size - 1)).let { Math.round(it).toInt() }
        return sorted[index.coerceIn(0, sorted.lastIndex)]
    }

    fun summary(): Map<String, Any> = mapOf(
        "n" to count,
        "p50" to round(percentile(50.0)),
        "p95" to round(percentile(95.0)),
        "max" to round(values.maxOrNull() ?: Double.NaN),
    )

    private fun round(v: Double) = if (v.isNaN()) "-" else Math.round(v * 10) / 10.0
}

/** Named [Samples], filled from many threads. */
class Metrics {
    private val byName = ConcurrentHashMap<String, Samples>()

    fun add(name: String, ms: Double) = byName.computeIfAbsent(name) { Samples() }.add(ms)

    fun get(name: String): Samples = byName.computeIfAbsent(name) { Samples() }

    fun names(): List<String> = byName.keys.sorted()
}

/**
 * Measures every HTTP call a kiosk makes (request sent → response body read), grouped by API path
 * and by the phase currently running, so idle and load numbers can be compared.
 */
class ApiTimer(private val metrics: Metrics, private val phase: () -> String) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val path = chain.request().url.encodedPath
        if (path.startsWith("/ws")) return chain.proceed(chain.request())
        val began = System.nanoTime()
        val response = chain.proceed(chain.request())
        // Buffer the body so the time includes the transfer of large downloads.
        val body = response.peekBody(Long.MAX_VALUE)
        metrics.add("api ${phase()} ${path.substringAfterLast("/")}", (System.nanoTime() - began) / 1e6)
        response.body?.close()
        return response.newBuilder().body(body).build()
    }
}

fun millisSince(nanos: Long): Double = (System.nanoTime() - nanos) / 1e6
