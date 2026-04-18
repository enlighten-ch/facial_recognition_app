package com.example.facetestbed

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.Buffer
import okio.BufferedSink
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
        fun exportLocalStructuredDb(): Pair<ByteArray, ByteArray>?
        fun applyRemoteStructuredDb(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean
        fun isRecognitionIdle(idleSeconds: Int): Boolean
        fun localFaceCount(): Int
        fun localEmbeddingCount(): Int
        fun onAuthTokenIssued(token: String)
        fun onAuthTokenCleared()
        fun onUploadProgress(uploaded: Int, total: Int)
    }

    @Volatile
    private var httpClient: OkHttpClient = buildHttpClient(6)
    @Volatile
    private var webSocketClient: OkHttpClient = buildWebSocketClient(6)

    private val executor = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)
    private val syncInFlight = AtomicBoolean(false)

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
        syncInFlight.set(false)
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

    fun pushLatest(reason: String, waitForIdle: Boolean = true) {
        if (!running.get()) return
        executor.execute {
            if (!syncInFlight.compareAndSet(false, true)) {
                callbacks.onDiagnosticLog("업로드 요청 병합: 이미 진행 중인 작업이 있어 건너뜀 ($reason)")
                return@execute
            }
            try {
                pushLatestCore(reason, waitForIdle = waitForIdle)
            } finally {
                syncInFlight.set(false)
            }
        }
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
        val payload = buildKeyValueBody(
            "clientId" to clientId,
            "deviceName" to deviceName,
        )
        val req = Request.Builder()
            .url("$baseUrl/api/auth/request")
            .post(payload.toByteArray().toRequestBody(TEXT_MEDIA))
            .build()

        callbacks.onDiagnosticLog("인증 요청: ${req.url}")
        return runCatching {
            httpClient.newCall(req).execute().use { res ->
                if (!res.isSuccessful) {
                    return AuthRequestResult(AuthState.ERROR, "HTTP ${res.code}")
                }
                val body = parseKeyValueBody(res.body?.string().orEmpty())
                val status = body["status"].orEmpty()
                if (status == "approved") {
                    val token = body["token"].orEmpty().trim()
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
        if (!syncInFlight.compareAndSet(false, true)) {
            callbacks.onDiagnosticLog("동기화 요청 병합: 이미 진행 중인 작업에 합류 ($reason)")
            return
        }
        try {
            pullLatestStructuredInternal(reason, allowReauthRetry)
        } finally {
            syncInFlight.set(false)
        }
    }

    private fun pullLatestStructuredInternal(reason: String, allowReauthRetry: Boolean = true) {
        if (!ensureAuthorized()) return
        callbacks.onStatus("DB 동기화 중: $reason", true)
        val startedAt = System.currentTimeMillis()
        val metaRequest = Request.Builder()
            .url("$baseUrl/api/db/structured/meta")
            .header("Authorization", "Bearer $authToken")
            .get()
            .build()
        val binRequest = Request.Builder()
            .url("$baseUrl/api/db/structured/bin")
            .header("Authorization", "Bearer $authToken")
            .get()
            .build()

        runCatching {
            val metaBytes = downloadStructuredPart(metaRequest, "meta", reason, allowReauthRetry) ?: return
            val binBytes = downloadStructuredPart(binRequest, "bin", reason, allowReauthRetry) ?: return
            if (!waitForRecognitionIdleBeforeDbApply()) {
                callbacks.onStatus("DB 동기화 취소(앱 중지)", false)
                return
            }
            val applied = callbacks.applyRemoteStructuredDb(metaBytes, binBytes)
            if (applied) {
                initialSyncCompleted = true
                val elapsedMs = System.currentTimeMillis() - startedAt
                callbacks.onDiagnosticLog(
                    "구조화 DB 다운로드 완료: meta=${metaBytes.size}B, bin=${binBytes.size}B, total=${elapsedMs}ms",
                )
                callbacks.onStatus("DB 동기화 완료 (구조화 다운로드, 얼굴 ${callbacks.localFaceCount()}명)", false)
            } else {
                callbacks.onStatus("DB 동기화 실패(DB 반영 오류)", false)
            }
        }.onFailure {
            callbacks.onDiagnosticLog("구조화 DB 다운로드 예외: ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            callbacks.onStatus("DB 동기화 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun downloadStructuredPart(request: Request, label: String, reason: String, allowReauthRetry: Boolean): ByteArray? {
        httpClient.newCall(request).execute().use { res ->
            if (isAuthError(res.code) && allowReauthRetry) {
                callbacks.onStatus("인증 만료, 재승인 요청 중...", true)
                invalidateAuthToken()
                if (ensureAuthorized()) {
                    pullLatestStructuredInternal(reason, allowReauthRetry = false)
                } else {
                    callbacks.onStatus("DB 동기화 실패(승인 필요)", false)
                }
                return null
            }
            if (!res.isSuccessful) {
                callbacks.onDiagnosticLog("구조화 $label 다운로드 실패: HTTP ${res.code}")
                callbacks.onStatus("DB 동기화 실패(HTTP ${res.code})", false)
                return null
            }
            return res.body?.bytes() ?: ByteArray(0)
        }
    }

    // 실제 push 로직 - syncInFlight는 호출자가 관리
    private fun pushLatestCore(
        reason: String,
        waitForIdle: Boolean = true,
        allowReauthRetry: Boolean = true,
        retryCount: Int = 0,
    ) {
        if (waitForIdle) {
            if (!waitForRecognitionIdleBeforeUpload()) {
                callbacks.onStatus("DB 업로드 취소(앱 중지)", false)
                return
            }
        }
        if (!ensureAuthorized()) return

        val totalEmbeddings = callbacks.localEmbeddingCount()
        callbacks.onUploadProgress(0, totalEmbeddings)
        callbacks.onStatus("DB 서버 업데이트 중: $reason (0/$totalEmbeddings)", true)

        val snapshot = callbacks.exportLocalStructuredDb()
        if (snapshot == null) {
            callbacks.onStatus("DB 서버 업데이트 실패(구조화 DB 읽기 오류)", false)
            return
        }
        val payload = buildStructuredBundle(snapshot.first, snapshot.second)

        // 1개 embedding당 500ms 기준 동적 callTimeout (최소 30초, 여유 15초 추가)
        val dynamicCallTimeoutSeconds = maxOf(30L, totalEmbeddings * 500L / 1000L + 15L)

        val progressBody = buildProgressTrackingBody(payload, totalEmbeddings) { uploaded, total ->
            callbacks.onUploadProgress(uploaded, total)
            callbacks.onStatus("DB 서버 업데이트 중: $reason ($uploaded/$total)", true)
        }

        val req = Request.Builder()
            .url("$baseUrl/api/db/structured/push")
            .header("Authorization", "Bearer $authToken")
            .post(progressBody)
            .build()

        // 업로드용 동적 타임아웃 클라이언트 (connection pool 공유)
        val uploadClient = httpClient.newBuilder()
            .callTimeout(dynamicCallTimeoutSeconds, TimeUnit.SECONDS)
            .build()

        runCatching {
            uploadClient.newCall(req).execute().use { res ->
                if (isAuthError(res.code) && allowReauthRetry) {
                    callbacks.onStatus("인증 만료, 재승인 요청 중...", true)
                    invalidateAuthToken()
                    if (ensureAuthorized()) {
                        pushLatestCore(reason, waitForIdle = false, allowReauthRetry = false, retryCount = retryCount)
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
                        pushLatestCore(reason, waitForIdle = false, allowReauthRetry = false, retryCount = next)
                        return
                    }
                    val detail = body.take(120).replace('\n', ' ').trim()
                    val suffix = if (detail.isBlank()) "" else " - $detail"
                    callbacks.onDiagnosticLog("DB push 실패: HTTP ${res.code}$suffix")
                    callbacks.onStatus("DB 서버 업데이트 실패(HTTP ${res.code}$suffix)", false)
                    return
                }
                callbacks.onUploadProgress(totalEmbeddings, totalEmbeddings)
                callbacks.onStatus("DB 서버 업데이트 완료 ($totalEmbeddings/$totalEmbeddings)", false)
            }
        }.onFailure {
            callbacks.onDiagnosticLog("DB push 예외: ${it.javaClass.simpleName}: ${it.message.orEmpty()}")
            if (retryCount < 2 && running.get()) {
                val next = retryCount + 1
                callbacks.onStatus("DB 서버 업데이트 재시도(${next}/2): ${it.javaClass.simpleName}", true)
                Thread.sleep((1000L * next).coerceAtMost(2500L))
                pushLatestCore(reason, waitForIdle = false, allowReauthRetry = allowReauthRetry, retryCount = next)
                return@onFailure
            }
            callbacks.onStatus("DB 서버 업데이트 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun buildProgressTrackingBody(
        bytes: ByteArray,
        totalEmbeddings: Int,
        onProgress: (Int, Int) -> Unit,
    ): RequestBody {
        return object : RequestBody() {
            override fun contentType() = BINARY_MEDIA
            override fun contentLength() = bytes.size.toLong()
            override fun writeTo(sink: BufferedSink) {
                val source = Buffer().also { it.write(bytes) }
                var totalWritten = 0L
                val buffer = Buffer()
                while (true) {
                    val read = source.read(buffer, 8192L)
                    if (read == -1L) break
                    sink.write(buffer, read)
                    totalWritten += read
                    val progress = if (bytes.isEmpty()) totalEmbeddings
                    else (totalWritten.toDouble() / bytes.size * totalEmbeddings).toInt().coerceIn(0, totalEmbeddings)
                    onProgress(progress, totalEmbeddings)
                }
            }
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
                val parsed = parseKeyValueBody(text)
                if (parsed["type"].orEmpty() != "db_update") return
                val source = parsed["sourceClientId"].orEmpty()
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

    private fun waitForRecognitionIdleBeforeUpload(): Boolean {
        val idleSeconds = dbLoadIdleSeconds.coerceAtLeast(1)
        var lastStatusAt = 0L
        while (running.get() && !callbacks.isRecognitionIdle(idleSeconds)) {
            val now = System.currentTimeMillis()
            if (now - lastStatusAt >= 1000L) {
                callbacks.onStatus("DB 업로드 대기중(유휴시간 ${idleSeconds}초)", true)
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

        if (isLoopbackHost(host)) {
            out.add("$scheme://127.0.0.1:$port")
            out.add("$scheme://localhost:$port")
            out.add("$scheme://[::1]:$port")
            out.add(base)
            return out.toList().distinct()
        }

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

    private fun buildKeyValueBody(vararg pairs: Pair<String, String>): String {
        return pairs.joinToString("\n") { (key, value) -> "$key=${value.replace("\n", " ").trim()}" }
    }

    private fun ByteArray.toRequestBody(mediaType: okhttp3.MediaType?) = object : RequestBody() {
        override fun contentType() = mediaType
        override fun contentLength() = this@toRequestBody.size.toLong()
        override fun writeTo(sink: BufferedSink) { sink.write(this@toRequestBody) }
    }

    private fun parseKeyValueBody(raw: String): Map<String, String> {
        return raw
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() && it.contains('=') }
            .associate { line ->
                val idx = line.indexOf('=')
                line.substring(0, idx).trim() to line.substring(idx + 1).trim()
            }
    }

    private fun buildStructuredBundle(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): ByteArray {
        val out = java.io.ByteArrayOutputStream(metaDbBytes.size + embeddingsBinBytes.size + 16)
        out.write(byteArrayOf('F'.code.toByte(), 'D'.code.toByte(), 'B'.code.toByte(), '1'.code.toByte()))
        writeIntLe(out, metaDbBytes.size)
        writeIntLe(out, embeddingsBinBytes.size)
        out.write(metaDbBytes)
        out.write(embeddingsBinBytes)
        return out.toByteArray()
    }

    private fun writeIntLe(out: java.io.ByteArrayOutputStream, value: Int) {
        out.write(value and 0xFF)
        out.write((value ushr 8) and 0xFF)
        out.write((value ushr 16) and 0xFF)
        out.write((value ushr 24) and 0xFF)
    }

    companion object {
        private val TEXT_MEDIA = "text/plain; charset=utf-8".toMediaType()
        private val BINARY_MEDIA = "application/octet-stream".toMediaType()
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
