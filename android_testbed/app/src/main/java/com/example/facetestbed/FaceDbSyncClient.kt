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
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class FaceDbSyncClient(
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        fun onStatus(message: String, busy: Boolean)
        fun exportLocalDbJson(): String
        fun exportLocalDbSummaryJson(): String
        fun applyRemoteDbJson(rawJson: String): Boolean
        fun applyRemoteDbDiff(upsertsRaw: String, deletesRaw: String): Boolean
        fun localFaceCount(): Int
        fun onAuthTokenIssued(token: String)
        fun onAuthTokenCleared()
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(4, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    private val executor = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)

    private var baseUrl: String = ""
    private var clientId: String = ""
    private var deviceName: String = "android-client"
    private var authToken: String = ""
    private var webSocket: WebSocket? = null
    @Volatile
    private var initialSyncCompleted: Boolean = false

    fun start(settings: AppSettings) {
        stop()
        if (!settings.dbSyncEnabled) {
            callbacks.onStatus("DB 서버 동기화 비활성", false)
            return
        }

        baseUrl = settings.dbServerBaseUrl.trim().trimEnd('/')
        if (baseUrl.isBlank()) {
            callbacks.onStatus("DB 서버 주소가 비어 있습니다.", false)
            return
        }

        clientId = settings.dbClientId.trim().ifBlank { generateClientId() }
        deviceName = settings.dbDeviceName.trim().ifBlank { "android-client" }
        authToken = settings.dbAuthToken.trim()
        initialSyncCompleted = false
        running.set(true)

        executor.execute {
            if (authToken.isBlank()) {
                val approved = awaitAuthorization()
                if (!approved || !running.get()) return@execute
            } else {
                callbacks.onStatus("저장된 인증으로 연결됨", false)
            }

            connectWebSocket()
            scheduleInitialSyncFallback()
        }
    }

    fun stop() {
        running.set(false)
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
        callbacks.onStatus("서버 승인 요청 중...", true)
        return awaitAuthorization()
    }

    private fun awaitAuthorization(): Boolean {
        repeat(1200) {
            if (!running.get()) return false
            when (requestAuthorization()) {
                AuthState.APPROVED -> return true
                AuthState.PENDING -> {
                    callbacks.onStatus("서버 승인 대기중(clientId=$clientId)", true)
                    Thread.sleep(3000)
                }
                AuthState.ERROR -> {
                    callbacks.onStatus("인증 요청 실패, 재시도 중...", false)
                    Thread.sleep(3000)
                }
            }
        }
        callbacks.onStatus("서버 승인 대기 타임아웃", false)
        return false
    }

    private fun requestAuthorization(): AuthState {
        val payload = JSONObject()
            .put("clientId", clientId)
            .put("deviceName", deviceName)

        val req = Request.Builder()
            .url("$baseUrl/api/auth/request")
            .post(payload.toString().toRequestBody(JSON_MEDIA))
            .build()

        return runCatching {
            httpClient.newCall(req).execute().use { res ->
                if (!res.isSuccessful) return AuthState.ERROR
                val body = res.body?.string().orEmpty()
                val root = runCatching { JSONObject(body) }.getOrNull() ?: return AuthState.ERROR
                val status = root.optString("status")
                if (status == "approved") {
                    val token = root.optString("token", "").trim()
                    if (token.isBlank()) return AuthState.ERROR
                    authToken = token
                    callbacks.onAuthTokenIssued(token)
                    callbacks.onStatus("서버 승인 완료", false)
                    return AuthState.APPROVED
                }
                return if (status == "pending") AuthState.PENDING else AuthState.ERROR
            }
        }.getOrDefault(AuthState.ERROR)
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
                    callbacks.onStatus("DB 동기화 실패(HTTP ${res.code})", false)
                    return
                }
                val body = res.body?.string().orEmpty()
                val dbJson = extractDbJson(body) ?: run {
                    callbacks.onStatus("DB 동기화 실패(응답 파싱 오류)", false)
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
            callbacks.onStatus("DB 동기화 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun pushLatestInternal(reason: String, allowReauthRetry: Boolean = true) {
        if (!ensureAuthorized()) return
        callbacks.onStatus("DB 서버 업데이트 중: $reason", true)
        val dbRaw = callbacks.exportLocalDbJson()
        val payload = JSONObject().put("db", JSONObject(dbRaw))

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
                    callbacks.onStatus("DB 서버 업데이트 실패(HTTP ${res.code})", false)
                    return
                }
                callbacks.onStatus("DB 서버 업데이트 완료", false)
            }
        }.onFailure {
            callbacks.onStatus("DB 서버 업데이트 실패(${it.javaClass.simpleName})", false)
        }
    }

    private fun connectWebSocket() {
        if (!running.get()) return
        if (!ensureAuthorized()) return

        val token = authToken
        if (token.isBlank()) return

        val wsUrl = toWebSocketUrl(baseUrl) + "/ws/db-updates?token=$token"
        val req = Request.Builder().url(wsUrl).build()

        webSocket = httpClient.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                callbacks.onStatus("DB 서버 연결됨", false)
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
                scheduleWebSocketReconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (!running.get()) return
                val code = response?.code ?: -1
                if (isAuthError(code)) {
                    callbacks.onStatus("인증 오류로 연결 재시도", false)
                    invalidateAuthToken()
                } else {
                    callbacks.onStatus("DB 서버 연결 끊김(${t.javaClass.simpleName})", false)
                }
                scheduleWebSocketReconnect()
            }
        })
    }

    private fun scheduleWebSocketReconnect() {
        executor.execute {
            Thread.sleep(3000)
            if (!running.get()) return@execute
            connectWebSocket()
        }
    }

    private fun scheduleInitialSyncFallback() {
        executor.execute {
            Thread.sleep(2500)
            if (!running.get() || initialSyncCompleted) return@execute
            callbacks.onStatus("DB 동기화 대기중", true)
            pullLatestInternal("초기 동기화(재시도)")
        }
    }

    private fun invalidateAuthToken() {
        if (authToken.isBlank()) return
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

    companion object {
        private val JSON_MEDIA = "application/json; charset=utf-8".toMediaType()
    }

    private enum class AuthState {
        APPROVED,
        PENDING,
        ERROR,
    }
}
