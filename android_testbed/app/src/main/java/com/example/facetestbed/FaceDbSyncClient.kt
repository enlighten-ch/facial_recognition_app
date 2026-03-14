package com.example.facetestbed

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URI
import java.util.Collections
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class FaceDbSyncClient(
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        fun onStatus(message: String, busy: Boolean)
        fun onDiagnosticLog(message: String)
        fun onClientIdIssued(clientId: String)
        fun exportLocalDbJson(): String
        fun exportLocalDbSummaryJson(): String
        fun applyRemoteDbJson(rawJson: String): Boolean
        fun applyRemoteDbDiff(upsertsRaw: String, deletesRaw: String): Boolean
        fun isRecognitionIdle(idleSeconds: Int): Boolean
        fun localFaceCount(): Int
        fun onAuthTokenIssued(token: String)
        fun onAuthTokenCleared()
    }

    @Volatile
    private var httpClient: OkHttpClient = buildHttpClient(6)
    @Volatile
    private var webSocketClient: OkHttpClient = buildWebSocketClient(6)

    private val executor = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)

    private var baseUrl: String = ""
    private var clientId: String = ""
    private var deviceName: String = "android-client"
    private var authToken: String = ""
    private var webSocket: WebSocket? = null
    private var dbLoadIdleSeconds: Int = 5
    private var connectRetryMaxAttempts: Int = 10
    private var connectRetryDelayMillis: Long = 3000L
    private var connectTimeoutSeconds: Int = 6
    private var wsReconnectAttempt: Int = 0
    @Volatile
    private var initialSyncCompleted: Boolean = false

    fun start(settings: AppSettings) {
        stop()
        if (!settings.dbSyncEnabled) {
            callbacks.onStatus("DB 서버 동기화 비활성", false)
            callbacks.onDiagnosticLog("동기화 비활성: 설정에서 DB_SYNC_ENABLED=false")
            return
        }

        baseUrl = normalizeBaseUrl(settings.dbServerBaseUrl)
        if (baseUrl.isBlank()) {
            callbacks.onStatus("DB 서버 주소가 비어 있습니다.", false)
            callbacks.onDiagnosticLog("서버 주소 누락: DB_SERVER_BASE_URL 빈값")
            return
        }

        clientId = settings.dbClientId.trim().ifBlank { generateClientId() }
        if (settings.dbClientId.trim().isBlank()) {
            callbacks.onClientIdIssued(clientId)
            callbacks.onDiagnosticLog("클라이언트 ID 자동 생성: $clientId")
        }
        deviceName = settings.dbDeviceName.trim().ifBlank { "android-client" }
        authToken = settings.dbAuthToken.trim()
        dbLoadIdleSeconds = settings.adaptiveUpdateIdleSeconds.coerceAtLeast(1)
        connectRetryMaxAttempts = settings.dbConnectRetryCount.coerceAtLeast(1)
        connectRetryDelayMillis = settings.dbConnectRetryDelaySeconds.coerceAtLeast(1).toLong() * 1000L
        connectTimeoutSeconds = settings.dbConnectTimeoutSeconds.coerceAtLeast(2)
        replaceHttpClients(connectTimeoutSeconds)
        wsReconnectAttempt = 0
        initialSyncCompleted = false
        running.set(true)

        executor.execute {
            callbacks.onDiagnosticLog(
                "동기화 시작: baseUrl=$baseUrl, clientId=$clientId, connectTimeout=${connectTimeoutSeconds}s, wsReadTimeout=무제한, retry=${connectRetryMaxAttempts}회/${retryDelaySeconds()}초",
            )
            callbacks.onStatus("DB 서버 헬스체크 중...", true)
            if (!probeServerHealth()) return@execute
            if (authToken.isBlank()) {
                val approved = awaitAuthorization()
                if (!approved || !running.get()) return@execute
            } else {
                callbacks.onStatus("저장된 인증으로 연결됨", false)
                callbacks.onDiagnosticLog("저장된 인증 토큰 사용")
            }

            connectWebSocket()
            scheduleInitialSyncFallback()
        }
    }

    fun stop() {
        running.set(false)
        wsReconnectAttempt = 0
        webSocket?.close(1000, "stop")
        webSocket = null
    }

    fun pushLatest(reason: String) {
        if (!running.get()) return
        executor.execute { pushLatestInternal(reason) }
    }

    fun pullLatest(reason: String) {
        if (!running.get()) return
        executor.execute { pullLatestInternal(reason) }
    }

    private fun ensureAuthorized(): Boolean {
        if (authToken.isNotBlank()) return true
        callbacks.onStatus("DB 서버 승인요청 중...", true)
        callbacks.onDiagnosticLog("인증 토큰 없음: 승인 요청 시작")
        return awaitAuthorization()
    }

    private fun awaitAuthorization(): Boolean {
        var errorAttempt = 0
        while (running.get()) {
            callbacks.onStatus("DB 서버 승인요청 중...", true)
            val result = requestAuthorization()
            when (result.state) {
                AuthState.APPROVED -> return true
                AuthState.PENDING -> {
                    callbacks.onStatus("DB 서버 승인대기중(clientId=$clientId)", true)
                    callbacks.onDiagnosticLog("승인 대기중: clientId=$clientId, detail=${result.detail}")
                }
                AuthState.ERROR -> {
                    errorAttempt += 1
                    callbacks.onDiagnosticLog("인증 요청 실패($errorAttempt/$connectRetryMaxAttempts): ${result.detail}")
                    if (errorAttempt >= connectRetryMaxAttempts) {
                        callbacks.onStatus("DB 서버 연결 실패(인증 재시도 초과)", false)
                        return false
                    }
                    callbacks.onStatus(
                        "DB 서버 연결 재시도 중 (승인요청 $errorAttempt/$connectRetryMaxAttempts, ${retryDelaySeconds()}초 후)",
                        true,
                    )
                }
            }
            Thread.sleep(connectRetryDelayMillis)
        }
        callbacks.onDiagnosticLog("승인 요청 중단: running=false")
        return false
    }

    private fun requestAuthorization(): AuthRequestResult {
        val payload = JSONObject()
            .put("clientId", clientId)
            .put("deviceName", deviceName)

        val req = Request.Builder()
            .url("$baseUrl/api/auth/request")
            .post(payload.toString().toRequestBody(JSON_MEDIA))
            .build()

        callbacks.onDiagnosticLog("인증 요청: ${req.url}")
        return runCatching {
            httpClient.newCall(req).execute().use { res ->
                if (!res.isSuccessful) {
                    return AuthRequestResult(AuthState.ERROR, "HTTP ${res.code}")
                }
                val body = res.body?.string().orEmpty()
                val root = runCatching { JSONObject(body) }.getOrNull()
                    ?: return AuthRequestResult(AuthState.ERROR, "응답 JSON 파싱 실패")
                val status = root.optString("status")
                if (status == "approved") {
                    val token = root.optString("token", "").trim()
                    if (token.isBlank()) {
                        return AuthRequestResult(AuthState.ERROR, "승인 응답에 토큰 없음")
                    }
                    authToken = token
                    callbacks.onAuthTokenIssued(token)
                    callbacks.onStatus("서버 승인 완료", false)
                    callbacks.onDiagnosticLog("승인 완료: clientId=$clientId")
                    return AuthRequestResult(AuthState.APPROVED, "approved")
                }
                return if (status == "pending") {
                    AuthRequestResult(AuthState.PENDING, "pending")
                } else {
                    AuthRequestResult(AuthState.ERROR, "알 수 없는 status=$status")
                }
            }
        }.getOrElse { err ->
            AuthRequestResult(AuthState.ERROR, "${err.javaClass.simpleName}: ${err.message.orEmpty()}")
        }
    }

    private fun pullLatestInternal(reason: String, allowReauthRetry: Boolean = true) {
        if (!ensureAuthorized()) return
        callbacks.onStatus("DB 동기화 중: $reason", true)
        val summaryObj = runCatching { JSONObject(callbacks.exportLocalDbSummaryJson()) }
            .getOrDefault(JSONObject().put("people", JSONArray()))
        val payload = JSONObject().put("summary", summaryObj)
        val request = Request.Builder()
            .url("$baseUrl/api/db/diff")
            .header("Authorization", "Bearer $authToken")
            .post(payload.toString().toRequestBody(JSON_MEDIA))
            .build()

        runCatching {
            httpClient.newCall(request).execute().use { res ->
                if (isAuthError(res.code) && allowReauthRetry) {
                    callbacks.onStatus("인증 만료, 재승인 요청 중...", true)
                    invalidateAuthToken()
                    if (ensureAuthorized()) {
                        pullLatestInternal(reason, allowReauthRetry = false)
                    } else {
                        callbacks.onStatus("DB 동기화 실패(승인 필요)", false)
                    }
                    return
                }
                if (res.code == 404 || res.code == 405) {
                    pullLatestFullInternal(reason, allowReauthRetry)
                    return
                }
                if (!res.isSuccessful) {
                    callbacks.onDiagnosticLog("DB diff 실패: HTTP ${res.code}")
                    callbacks.onStatus("DB 동기화 실패(HTTP ${res.code})", false)
                    return
                }
                val body = res.body?.string().orEmpty()
                val parsed = runCatching { JSONObject(body) }.getOrNull() ?: run {
                    callbacks.onStatus("DB 동기화 실패(응답 파싱 오류)", false)
                    return
                }
                val upserts = parsed.optJSONArray("upserts") ?: JSONArray()
                val deletes = parsed.optJSONArray("deletes") ?: JSONArray()
                val hasChanges = parsed.optBoolean("hasChanges", upserts.length() > 0 || deletes.length() > 0)
                val applied = if (hasChanges) {
                    if (!waitForRecognitionIdleBeforeDbApply()) {
                        callbacks.onStatus("DB 동기화 취소(앱 중지)", false)
                        return
                    }
                    callbacks.applyRemoteDbDiff(upserts.toString(), deletes.toString())
                } else {
                    true
                }
                if (applied) {
                    initialSyncCompleted = true
                    val count = callbacks.localFaceCount()
                    if (hasChanges) {
                        callbacks.onStatus("DB 동기화 완료 (변경 ${upserts.length() + deletes.length()}건, 얼굴 ${count}명)", false)
                    } else {
                        callbacks.onStatus("DB 동기화 완료 (변경 없음, 얼굴 ${count}명)", false)
                    }
                } else {
                    callbacks.onStatus("DB 동기화 실패(DB 반영 오류)", false)
                }
            }
        }.onFailure {
            callbacks.onDiagnosticLog("DB diff 예외: ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            callbacks.onStatus("DB 동기화 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun pullLatestFullInternal(reason: String, allowReauthRetry: Boolean = true) {
        val request = Request.Builder()
            .url("$baseUrl/api/db")
            .header("Authorization", "Bearer $authToken")
            .get()
            .build()

        runCatching {
            httpClient.newCall(request).execute().use { res ->
                if (isAuthError(res.code) && allowReauthRetry) {
                    callbacks.onStatus("인증 만료, 재승인 요청 중...", true)
                    invalidateAuthToken()
                    if (ensureAuthorized()) {
                        pullLatestFullInternal(reason, allowReauthRetry = false)
                    } else {
                        callbacks.onStatus("DB 동기화 실패(승인 필요)", false)
                    }
                    return
                }
                if (!res.isSuccessful) {
                    callbacks.onDiagnosticLog("DB 전체 조회 실패: HTTP ${res.code}")
                    callbacks.onStatus("DB 동기화 실패(HTTP ${res.code})", false)
                    return
                }
                val body = res.body?.string().orEmpty()
                val dbJson = extractDbJson(body) ?: run {
                    callbacks.onStatus("DB 동기화 실패(응답 파싱 오류)", false)
                    return
                }
                if (!waitForRecognitionIdleBeforeDbApply()) {
                    callbacks.onStatus("DB 동기화 취소(앱 중지)", false)
                    return
                }
                val applied = callbacks.applyRemoteDbJson(dbJson)
                if (applied) {
                    initialSyncCompleted = true
                    callbacks.onStatus("DB 동기화 완료 (전체 다운로드, 얼굴 ${callbacks.localFaceCount()}명)", false)
                } else {
                    callbacks.onStatus("DB 동기화 실패(DB 반영 오류)", false)
                }
            }
        }.onFailure {
            callbacks.onDiagnosticLog("DB 전체 조회 예외: ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            callbacks.onStatus("DB 동기화 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun pushLatestInternal(
        reason: String,
        allowReauthRetry: Boolean = true,
        retryCount: Int = 0,
    ) {
        if (!ensureAuthorized()) return
        callbacks.onStatus("DB 서버 업데이트 중: $reason", true)
        val dbRaw = callbacks.exportLocalDbJson()
        val dbObj = runCatching { JSONObject(dbRaw) }.getOrNull()
        if (dbObj == null) {
            callbacks.onStatus("DB 서버 업데이트 실패(로컬 DB 직렬화 오류)", false)
            return
        }
        val payload = JSONObject().put("db", dbObj)

        val req = Request.Builder()
            .url("$baseUrl/api/db/push")
            .header("Authorization", "Bearer $authToken")
            .post(payload.toString().toRequestBody(JSON_MEDIA))
            .build()

        runCatching {
            httpClient.newCall(req).execute().use { res ->
                if (isAuthError(res.code) && allowReauthRetry) {
                    callbacks.onStatus("인증 만료, 재승인 요청 중...", true)
                    invalidateAuthToken()
                    if (ensureAuthorized()) {
                        pushLatestInternal(reason, allowReauthRetry = false)
                    } else {
                        callbacks.onStatus("DB 서버 업데이트 실패(승인 필요)", false)
                    }
                    return
                }
                if (!res.isSuccessful) {
                    val body = runCatching { res.body?.string().orEmpty() }.getOrDefault("")
                    if (shouldRetryPush(res.code, retryCount)) {
                        val next = retryCount + 1
                        callbacks.onStatus("DB 서버 업데이트 재시도(${next}/2): HTTP ${res.code}", true)
                        Thread.sleep((1000L * next).coerceAtMost(2500L))
                        pushLatestInternal(reason, allowReauthRetry = false, retryCount = next)
                        return
                    }
                    val detail = body.take(120).replace('\n', ' ').trim()
                    val suffix = if (detail.isBlank()) "" else " - $detail"
                    callbacks.onDiagnosticLog("DB push 실패: HTTP ${res.code}$suffix")
                    callbacks.onStatus("DB 서버 업데이트 실패(HTTP ${res.code}$suffix)", false)
                    return
                }
                callbacks.onStatus("DB 서버 업데이트 완료", false)
            }
        }.onFailure {
            callbacks.onDiagnosticLog("DB push 예외: ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            if (retryCount < 2 && running.get()) {
                val next = retryCount + 1
                callbacks.onStatus("DB 서버 업데이트 재시도(${next}/2): ${it.javaClass.simpleName}", true)
                Thread.sleep((1000L * next).coerceAtMost(2500L))
                pushLatestInternal(reason, allowReauthRetry = allowReauthRetry, retryCount = next)
                return@onFailure
            }
            callbacks.onStatus("DB 서버 업데이트 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun connectWebSocket() {
        if (!running.get()) return
        if (!ensureAuthorized()) {
            scheduleWebSocketReconnect("인증 실패")
            return
        }

        val token = authToken
        if (token.isBlank()) {
            callbacks.onDiagnosticLog("WS 연결 생략: 토큰 비어 있음")
            scheduleWebSocketReconnect("토큰 비어 있음")
            return
        }

        val wsUrl = toWebSocketUrl(baseUrl) + "/ws/db-updates?token=$token"
        val req = Request.Builder().url(wsUrl).build()
        callbacks.onStatus("DB 서버 WS 연결 시도중...", true)
        callbacks.onDiagnosticLog("WS 연결 시도: $wsUrl")

        webSocket = webSocketClient.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                wsReconnectAttempt = 0
                callbacks.onStatus("DB 서버 연결됨", false)
                callbacks.onDiagnosticLog("WS 연결 성공: code=${response.code}")
                if (!initialSyncCompleted) {
                    callbacks.onStatus("DB 동기화 대기중", true)
                    executor.execute { pullLatestInternal("초기 동기화") }
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val parsed = runCatching { JSONObject(text) }.getOrNull() ?: return
                if (parsed.optString("type") != "db_update") return
                val source = parsed.optString("sourceClientId")
                if (source == clientId) return
                if (source == "server_bootstrap" && !initialSyncCompleted) {
                    executor.execute { pullLatestInternal("초기 동기화") }
                    return
                }
                if (source == "server_bootstrap") return
                executor.execute { pullLatestInternal("서버 변경 수신") }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (!running.get() || code == 1000) return
                if (code == 1008) {
                    callbacks.onStatus("인증 만료로 연결 종료, 재승인 요청", false)
                    invalidateAuthToken()
                } else {
                    callbacks.onStatus("DB 서버 연결 종료(code=$code)", false)
                }
                callbacks.onDiagnosticLog("WS 종료: code=$code, reason=$reason")
                scheduleWebSocketReconnect("code=$code, reason=$reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (!running.get()) return
                val code = response?.code ?: -1
                if (isAuthError(code)) {
                    callbacks.onStatus("인증 오류로 연결 재시도", false)
                    invalidateAuthToken()
                } else if (t is java.net.SocketTimeoutException) {
                    callbacks.onStatus("DB 서버 연결 타임아웃(WS)", false)
                } else {
                    callbacks.onStatus("DB 서버 연결 끊김(${t.javaClass.simpleName})", false)
                }
                callbacks.onDiagnosticLog("WS 실패: code=$code, err=${t.javaClass.simpleName}: ${t.message.orEmpty()}")
                scheduleWebSocketReconnect("code=$code, err=${t.javaClass.simpleName}")
            }
        })
    }

    private fun scheduleWebSocketReconnect(reason: String) {
        executor.execute {
            if (!running.get()) return@execute
            wsReconnectAttempt += 1
            val attempt = wsReconnectAttempt
            if (attempt > connectRetryMaxAttempts) {
                callbacks.onStatus("DB 서버 연결 실패(재시도 초과)", false)
                callbacks.onDiagnosticLog("WS 재연결 중단: 재시도 초과, reason=$reason")
                return@execute
            }
            callbacks.onStatus(
                "DB 서버 연결 재시도 중 (WS $attempt/$connectRetryMaxAttempts, ${retryDelaySeconds()}초 후)",
                true,
            )
            callbacks.onDiagnosticLog("WS 재연결 예약: attempt=$attempt, reason=$reason")
            Thread.sleep(connectRetryDelayMillis)
            if (!running.get()) return@execute
            connectWebSocket()
        }
    }

    private fun scheduleInitialSyncFallback() {
        executor.execute {
            Thread.sleep(2500)
            if (!running.get() || initialSyncCompleted) return@execute
            callbacks.onStatus("DB 동기화 대기중", true)
            callbacks.onDiagnosticLog("초기 동기화 fallback 실행")
            pullLatestInternal("초기 동기화(재시도)")
        }
    }

    private fun invalidateAuthToken() {
        if (authToken.isBlank()) return
        callbacks.onDiagnosticLog("인증 토큰 무효화")
        authToken = ""
        callbacks.onAuthTokenCleared()
    }

    private fun isAuthError(code: Int): Boolean {
        return code == 401 || code == 403
    }

    private fun extractDbJson(responseBody: String): String? {
        val root = runCatching { JSONObject(responseBody) }.getOrNull() ?: return null
        val db = root.optJSONObject("db") ?: return null
        return db.toString()
    }

    private fun toWebSocketUrl(httpBase: String): String {
        return when {
            httpBase.startsWith("https://") -> "wss://" + httpBase.removePrefix("https://")
            httpBase.startsWith("http://") -> "ws://" + httpBase.removePrefix("http://")
            else -> "ws://$httpBase"
        }
    }

    private fun generateClientId(): String {
        return "client-" + UUID.randomUUID().toString().take(8)
    }

    private fun shouldRetryPush(code: Int, retryCount: Int): Boolean {
        if (retryCount >= 2) return false
        return code == 408 || code == 429 || code in 500..599
    }

    private fun waitForRecognitionIdleBeforeDbApply(): Boolean {
        val idleSeconds = dbLoadIdleSeconds.coerceAtLeast(1)
        var lastStatusAt = 0L
        while (running.get() && !callbacks.isRecognitionIdle(idleSeconds)) {
            val now = System.currentTimeMillis()
            if (now - lastStatusAt >= 1000L) {
                callbacks.onStatus("DB 동기화 대기중(유휴시간 ${idleSeconds}초)", true)
                lastStatusAt = now
            }
            Thread.sleep(300)
        }
        return running.get()
    }

    private fun probeServerHealth(): Boolean {
        val candidates = buildHealthCheckCandidates(baseUrl)
        callbacks.onDiagnosticLog("헬스체크 대상: ${candidates.joinToString()}")
        for (attempt in 1..connectRetryMaxAttempts) {
            if (!running.get()) return false
            callbacks.onStatus("DB 서버 헬스체크 중 ($attempt/$connectRetryMaxAttempts)", true)
            var connected: String? = null
            for (candidate in candidates) {
                if (probeHealthOnce(candidate)) {
                    connected = candidate
                    break
                }
            }
            if (connected != null) {
                if (baseUrl != connected) {
                    if (shouldAutoSwitchBaseUrl(baseUrl, connected)) {
                        callbacks.onDiagnosticLog("서버 주소 자동전환: $baseUrl -> $connected")
                        baseUrl = connected
                    } else {
                        callbacks.onDiagnosticLog("서버 주소 유지: 입력값=$baseUrl, 연결확인=$connected")
                    }
                } else {
                    callbacks.onDiagnosticLog("헬스체크 성공: $baseUrl")
                }
                return true
            }
            if (attempt >= connectRetryMaxAttempts) break
            callbacks.onStatus(
                "DB 서버 연결 재시도 중 (헬스체크 $attempt/$connectRetryMaxAttempts, ${retryDelaySeconds()}초 후)",
                true,
            )
            Thread.sleep(connectRetryDelayMillis)
        }
        callbacks.onStatus("DB 서버 연결 실패(헬스체크 실패: 서버앱 로그인 후 서버 시작 확인)", false)
        return false
    }

    private fun probeHealthOnce(targetBaseUrl: String): Boolean {
        val req = Request.Builder()
            .url("$targetBaseUrl/api/health")
            .get()
            .build()
        return runCatching {
            httpClient.newCall(req).execute().use { res ->
                if (!res.isSuccessful) {
                    callbacks.onDiagnosticLog("헬스체크 실패($targetBaseUrl): HTTP ${res.code}")
                    return@use false
                }
                val body = res.body?.string().orEmpty().trim()
                if (body.equals("ok", ignoreCase = true)) {
                    callbacks.onDiagnosticLog("헬스체크 성공($targetBaseUrl)")
                    return@use true
                }
                callbacks.onDiagnosticLog("헬스체크 실패($targetBaseUrl): body=$body")
                return@use false
            }
        }.getOrElse {
            callbacks.onDiagnosticLog("헬스체크 예외($targetBaseUrl): ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            false
        }
    }

    private fun buildHealthCheckCandidates(initialBaseUrl: String): List<String> {
        val base = normalizeBaseUrl(initialBaseUrl)
        val uri = runCatching { URI(base) }.getOrNull() ?: return listOf(base)
        val host = uri.host?.lowercase() ?: return listOf(base)
        if (!isLocalHostAlias(host)) {
            return listOf(base)
        }
        val scheme = uri.scheme ?: "http"
        val port = if (uri.port > 0) uri.port else 80
        val out = linkedSetOf<String>()

        // Loopback 입력은 같은 기기 서버를 명시한 것으로 보고 LAN 자동 전환을 금지한다.
        if (isLoopbackHost(host)) {
            out.add("$scheme://127.0.0.1:$port")
            out.add("$scheme://localhost:$port")
            out.add("$scheme://[::1]:$port")
            out.add(base)
            return out.toList().distinct()
        }

        // Wildcard(0.0.0.0)일 때만 LAN 후보를 함께 탐색한다.
        localIpv4Candidates().forEach { ip ->
            out.add("$scheme://$ip:$port")
        }
        out.add("$scheme://127.0.0.1:$port")
        out.add("$scheme://localhost:$port")
        out.add("$scheme://[::1]:$port")
        out.add(base)
        return out.toList()
    }

    private fun isLocalHostAlias(host: String): Boolean {
        return host == "127.0.0.1" ||
            host == "localhost" ||
            host == "::1" ||
            host == "0.0.0.0"
    }

    private fun isLoopbackHost(host: String): Boolean {
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    private fun shouldAutoSwitchBaseUrl(currentBaseUrl: String, candidateBaseUrl: String): Boolean {
        val currentHost = runCatching { URI(normalizeBaseUrl(currentBaseUrl)).host?.lowercase().orEmpty() }
            .getOrDefault("")
        if (candidateBaseUrl.isBlank()) return false
        return currentHost == "0.0.0.0"
    }

    private fun localIpv4Candidates(): List<String> {
        return runCatching {
            val out = mutableListOf<String>()
            val interfaces = NetworkInterface.getNetworkInterfaces() ?: return@runCatching emptyList()
            for (netIf in Collections.list(interfaces)) {
                if (!netIf.isUp || netIf.isLoopback || netIf.isVirtual) continue
                for (addr in Collections.list(netIf.inetAddresses)) {
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        val ip = addr.hostAddress?.trim().orEmpty()
                        if (ip.isNotBlank()) out += ip
                    }
                }
            }
            out.distinct()
        }.getOrDefault(emptyList())
    }

    private fun normalizeBaseUrl(raw: String): String {
        var trimmed = raw.trim().trimEnd('/')
        if (trimmed.isBlank()) return ""
        trimmed = when {
            trimmed.startsWith("ws://", ignoreCase = true) -> "http://" + trimmed.removePrefix("ws://")
            trimmed.startsWith("wss://", ignoreCase = true) -> "https://" + trimmed.removePrefix("wss://")
            trimmed.startsWith("http://", ignoreCase = true) -> trimmed
            trimmed.startsWith("https://", ignoreCase = true) -> trimmed
            else -> "http://$trimmed"
        }
        if (trimmed.endsWith("/api", ignoreCase = true)) {
            trimmed = trimmed.dropLast(4).trimEnd('/')
        }
        return trimmed
    }

    private fun retryDelaySeconds(): Int {
        return (connectRetryDelayMillis / 1000L).toInt().coerceAtLeast(1)
    }

    private fun replaceHttpClients(timeoutSeconds: Int) {
        val previousRest = httpClient
        val previousWs = webSocketClient
        httpClient = buildHttpClient(timeoutSeconds)
        webSocketClient = buildWebSocketClient(timeoutSeconds)
        closeHttpClient(previousRest)
        closeHttpClient(previousWs)
    }

    private fun closeHttpClient(client: OkHttpClient) {
        runCatching {
            client.dispatcher.cancelAll()
            client.connectionPool.evictAll()
            client.dispatcher.executorService.shutdown()
        }
    }

    private fun buildHttpClient(timeoutSeconds: Int): OkHttpClient {
        val safeConnectTimeout = timeoutSeconds.coerceAtLeast(2).toLong()
        val safeIoTimeout = maxOf(safeConnectTimeout * 2, 20L)
        val safeCallTimeout = maxOf(safeIoTimeout + 5L, 30L)
        return OkHttpClient.Builder()
            .connectTimeout(safeConnectTimeout, TimeUnit.SECONDS)
            .readTimeout(safeIoTimeout, TimeUnit.SECONDS)
            .writeTimeout(safeIoTimeout, TimeUnit.SECONDS)
            .callTimeout(safeCallTimeout, TimeUnit.SECONDS)
            .build()
    }

    private fun buildWebSocketClient(timeoutSeconds: Int): OkHttpClient {
        val safeConnectTimeout = timeoutSeconds.coerceAtLeast(2).toLong()
        val safeWriteTimeout = maxOf(safeConnectTimeout * 2, 20L)
        return OkHttpClient.Builder()
            .connectTimeout(safeConnectTimeout, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .writeTimeout(safeWriteTimeout, TimeUnit.SECONDS)
            .pingInterval(20, TimeUnit.SECONDS)
            .callTimeout(0, TimeUnit.MILLISECONDS)
            .build()
    }

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }

    private enum class AuthState {
        APPROVED,
        PENDING,
        ERROR,
    }

    private data class AuthRequestResult(
        val state: AuthState,
        val detail: String,
    )
}
