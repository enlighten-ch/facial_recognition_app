package com.example.facedbserver

import android.content.Context
import android.net.Uri
import android.database.sqlite.SQLiteDatabase
import android.content.ContentValues
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import com.example.faithid.common.decodeMetadataBlob
import com.example.faithid.common.defaultMetadataMap
import com.example.faithid.common.encodeMetadataBlob
import com.example.faithid.common.metadataHash
import com.example.faithid.common.metadataIdentityKey
import com.example.faithid.common.metadataName
import com.example.faithid.common.normalizeMetadataMap
import com.example.faithid.common.parsePersonId
import com.example.faithid.common.parsePersonKey
import com.example.faithid.common.personIdFromIdentity
import com.example.faithid.common.personKeyFromIdentity
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.request.receiveStream
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondText
import io.ktor.server.response.respondTextWriter
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.server.websocket.DefaultWebSocketServerSession
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.CloseReason
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import io.ktor.websocket.readText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.LinkedHashMap
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

data class DbPersonRow(
    val personKey: String,
    val name: String,
    val department: String,
    val grade: String,
    val embeddingCount: Int,
    val metadata: Map<String, String> = linkedMapOf(
        "이름" to name,
        "부서" to department,
        "학년" to grade,
    ),
)

data class ServerPersonRecord(
    val name: String,
    val department: String,
    val grade: String = "",
    val embeddings: MutableList<FloatArray>,
    val hash: String,
    val updatedAt: Long,
    val personId: Long = personIdFromIdentity(name, department, grade),
    val personKey: String = personKeyFromIdentity(name, department, grade),
    val metadata: LinkedHashMap<String, String> = defaultMetadataMap(name, department, grade),
    val metadataHash: String = metadataHash(metadata),
)

data class ClientConnectionStatus(
    val clientId: String,
    val status: String,
    val onlineSessionCount: Int,
)

class FaceDbServerEngine(
    private val context: Context,
    private val onLog: (String) -> Unit,
) {
    private var server: ApplicationEngine? = null
    private var port: Int = 18080

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val sessions = ConcurrentHashMap<String, DefaultWebSocketServerSession>()
    private val revision = AtomicInteger(0)

    private val pendingClients = ConcurrentHashMap<String, String>()
    private val approvedTokensByClient = ConcurrentHashMap<String, String>()
    private val clientByToken = ConcurrentHashMap<String, String>()
    private val onlineSessionsByClient = ConcurrentHashMap<String, AtomicInteger>()

    private val serverDataDir: File by lazy {
        val dir = File(context.filesDir, "face_db_server")
        if (!dir.exists()) dir.mkdirs()
        dir
    }

    private val authStateFile: File by lazy { File(serverDataDir, "auth_state.json") }
    private var dbStorageDirSpec: String = ""
    private val csvCodec = FaceDbCsvCodec()
    private var dbStorageBackend: DbStorageBackend =
        FileDbStorageBackend(
            csvCodec = csvCodec,
            csvFile = File(serverDataDir, "face_db.csv"),
            metaDbFile = File(serverDataDir, "face_meta.db"),
            embeddingsBinFile = File(serverDataDir, "face_embeddings.bin"),
        )

    @Volatile
    private var currentPeople: MutableList<ServerPersonRecord> = mutableListOf()
    @Volatile
    private var dbLoadedOnce: Boolean = false

    fun setDbStorageDir(spec: String) {
        val normalized = DbStorageLocation.normalize(spec)
        ensureUserAccessibleDbDirectory(normalized)
        if (dbStorageDirSpec != normalized) {
            dbStorageDirSpec = normalized
            dbLoadedOnce = false
        }
    }

    private fun ensureUserAccessibleDbDirectory(spec: String) {
        if (spec.startsWith("content://") || spec.isBlank()) return
        val root = File(spec)
        val dir = if (root.isAbsolute) root else File(context.filesDir, spec)
        listOf(
            dir,
            File(dir, "logs"),
            File(dir, "models/buffalo_l"),
            File(dir, "templates"),
        ).forEach { folder ->
            runCatching { folder.mkdirs() }
                .onFailure { log("faceDB 디렉토리 생성 실패: ${folder.absolutePath} (${it.javaClass.simpleName})") }
        }
    }

    fun start(serverPort: Int): Boolean {
        if (server != null) {
            log("이미 서버가 실행 중입니다.")
            return false
        }
        port = serverPort
        dbStorageBackend = resolveDbStorageBackend()
        if (!dbLoadedOnce) {
            loadDbFromDisk()
            dbLoadedOnce = true
        } else {
            log("DB 재사용: 기존 로드 데이터를 사용합니다.")
        }
        loadAuthStateFromDisk()

        val engine = embeddedServer(CIO, host = "0.0.0.0", port = port) {
            install(WebSockets)
            routing {
                get("/api/health") {
                    call.respondText("ok", ContentType.Text.Plain)
                }

                post("/api/auth/request") {
                    val req = parsePlainFields(call.receiveText())
                    val clientId = req["clientId"].orEmpty().trim()
                    val deviceName = req["deviceName"].orEmpty().trim().ifBlank { "unknown" }
                    if (clientId.isBlank()) {
                        call.respondText("error=missing_client_id", ContentType.Text.Plain, HttpStatusCode.BadRequest)
                        return@post
                    }

                    val token = approvedTokensByClient[clientId]
                    if (!token.isNullOrBlank()) {
                        call.respondText(
                            buildPlainFields(
                                "status" to "approved",
                                "clientId" to clientId,
                                "token" to token,
                            ),
                            ContentType.Text.Plain,
                        )
                        return@post
                    }

                    pendingClients[clientId] = deviceName
                    log("승인 요청 수신: clientId=$clientId, device=$deviceName")
                    call.respondText(
                        buildPlainFields(
                            "status" to "pending",
                            "clientId" to clientId,
                        ),
                        ContentType.Text.Plain,
                    )
                }

                get("/api/auth/status") {
                    val clientId = call.request.queryParameters["clientId"].orEmpty().trim()
                    if (clientId.isBlank()) {
                        call.respondText("error=missing_client_id", ContentType.Text.Plain, HttpStatusCode.BadRequest)
                        return@get
                    }

                    val token = approvedTokensByClient[clientId]
                    val status = when {
                        !token.isNullOrBlank() -> "approved"
                        pendingClients.containsKey(clientId) -> "pending"
                        else -> "unknown"
                    }
                    call.respondText(
                        buildPlainFields(
                            "status" to status,
                            "clientId" to clientId,
                            "token" to (token ?: ""),
                        ),
                        ContentType.Text.Plain,
                    )
                }

                post("/api/auth/approve") {
                    val req = parseRequestBody(call.receiveText())
                    val clientId = req.optString("clientId", "").trim()
                    if (clientId.isBlank()) {
                        call.respondText(
                            "{\"error\":\"missing_client_id\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.BadRequest,
                        )
                        return@post
                    }
                    val token = approveClientInternal(clientId)
                    if (token == null) {
                        call.respondText(
                            "{\"error\":\"not_pending\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.NotFound,
                        )
                        return@post
                    }
                    call.respondText(
                        JSONObject().put("status", "approved").put("clientId", clientId).put("token", token).toString(),
                        ContentType.Application.Json,
                    )
                }

                post("/api/auth/approve-all") {
                    val approvedCount = approveAllPendingInternal()
                    call.respondText(
                        JSONObject().put("approvedCount", approvedCount).toString(),
                        ContentType.Application.Json,
                    )
                }

                get("/api/auth/pending") {
                    val arr = JSONArray()
                    pendingClients.toSortedMap().forEach { (clientId, deviceName) ->
                        arr.put(JSONObject().put("clientId", clientId).put("deviceName", deviceName))
                    }
                    call.respondText(JSONObject().put("pending", arr).toString(), ContentType.Application.Json)
                }

                get("/api/auth/approved") {
                    val arr = JSONArray()
                    approvedTokensByClient.toSortedMap().forEach { (clientId, _) ->
                        arr.put(JSONObject().put("clientId", clientId))
                    }
                    call.respondText(JSONObject().put("approved", arr).toString(), ContentType.Application.Json)
                }

                get("/api/db") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText(
                            "{\"error\":\"unauthorized\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.Unauthorized,
                        )
                        return@get
                    }
                    call.respondText(
                        "{\"error\":\"legacy_json_db_endpoint_disabled\"}",
                        ContentType.Application.Json,
                        HttpStatusCode.Gone,
                    )
                }

                get("/api/db/structured/meta") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@get
                    }
                    dbStorageBackend.readPeople()
                    val metaBytes = dbStorageBackend.readStructuredMetaBytes()
                    if (metaBytes == null) {
                        call.respondText("error=structured_meta_missing", ContentType.Text.Plain, HttpStatusCode.NotFound)
                        return@get
                    }
                    log("구조화 meta 다운로드: client=$authClient, bytes=${metaBytes.size}")
                    call.respondBytes(metaBytes, ContentType.Application.OctetStream)
                }

                get("/api/db/structured/bin") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@get
                    }
                    dbStorageBackend.readPeople()
                    val binBytes = dbStorageBackend.readStructuredEmbeddingsBytes()
                    if (binBytes == null) {
                        call.respondText("error=structured_bin_missing", ContentType.Text.Plain, HttpStatusCode.NotFound)
                        return@get
                    }
                    log("구조화 bin 다운로드: client=$authClient, bytes=${binBytes.size}")
                    call.respondBytes(binBytes, ContentType.Application.OctetStream)
                }

                get("/api/db/hash-summary") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@get
                    }
                    call.respondText(buildHashSummaryBody(currentPeople), ContentType.Text.Plain)
                }

                post("/api/db/delta/pull") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@post
                    }
                    val clientHashes = parseHashSummaryBody(call.receiveText())
                    val startedAt = System.currentTimeMillis()
                    val deltaBundle = buildDeltaBundleForClient(clientHashes)
                    log(
                        "DB 델타 전송: client=$authClient, clientPeople=${clientHashes.size}, " +
                            "bytes=${deltaBundle.size}, total=${System.currentTimeMillis() - startedAt}ms",
                    )
                    call.respondBytes(deltaBundle, ContentType.Application.OctetStream)
                }

                post("/api/db/delta/push") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@post
                    }
                    val payload = runCatching { call.receiveStream().readBytes() }.getOrNull()
                    val delta = payload?.let { parseDeltaBundle(it) }
                    if (delta == null) {
                        call.respondText("error=invalid_delta", ContentType.Text.Plain, HttpStatusCode.BadRequest)
                        return@post
                    }
                    val updatedCount = applyDeltaToCurrentDb(delta)
                    if (updatedCount == null) {
                        call.respondText("error=write_failed", ContentType.Text.Plain, HttpStatusCode.InternalServerError)
                        return@post
                    }
                    val nextRevision = revision.incrementAndGet()
                    log(
                        "임베딩 업데이트 수신: client=$authClient, upserts=${delta.upserts.size}명, " +
                        "embeddings=${updatedCount}개 → DB 반영(revision=$nextRevision) → " +
                        "${sessions.size}개 세션에 동기화 신호 전송",
                    )
                    broadcastDbUpdate(nextRevision, authClient)
                    call.respondText(
                        buildPlainFields(
                            "status" to "ok",
                            "revision" to nextRevision.toString(),
                            "updatedEmbeddings" to updatedCount.toString(),
                        ),
                        ContentType.Text.Plain,
                    )
                }

                post("/api/db/structured/push") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@post
                    }
                    val payload = runCatching { call.receiveStream().readBytes() }.getOrNull()
                    val bundle = payload?.let { parseStructuredBundle(it) }
                    if (bundle == null) {
                        call.respondText("error=invalid_bundle", ContentType.Text.Plain, HttpStatusCode.BadRequest)
                        return@post
                    }
                    if (!dbStorageBackend.writeStructured(bundle.first, bundle.second)) {
                        call.respondText("error=write_failed", ContentType.Text.Plain, HttpStatusCode.InternalServerError)
                        return@post
                    }
                    loadDbFromDisk()
                    val nextRevision = revision.incrementAndGet()
                    log(
                        "구조화 DB 업로드 반영: client=$authClient, revision=$nextRevision, " +
                            "meta=${bundle.first.size}B, bin=${bundle.second.size}B → " +
                            "${sessions.size}개 세션에 동기화 신호 전송",
                    )
                    broadcastDbUpdate(nextRevision, authClient)
                    call.respondText(
                        buildPlainFields(
                            "status" to "ok",
                            "revision" to nextRevision.toString(),
                        ),
                        ContentType.Text.Plain,
                    )
                }

                post("/api/db/diff") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText(
                            "{\"error\":\"unauthorized\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.Unauthorized,
                        )
                        return@post
                    }

                    call.respondText(
                        "{\"error\":\"legacy_json_diff_endpoint_disabled\"}",
                        ContentType.Application.Json,
                        HttpStatusCode.Gone,
                    )
                }

                post("/api/db/push") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText(
                            "{\"error\":\"unauthorized\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.Unauthorized,
                        )
                        return@post
                    }

                    call.respondText(
                        "{\"error\":\"legacy_json_push_endpoint_disabled\"}",
                        ContentType.Application.Json,
                        HttpStatusCode.Gone,
                    )
                }

                post("/api/attendance/mark") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@post
                    }
                    val req = parsePlainFields(call.receiveText())
                    val name = req["name"].orEmpty().trim()
                    if (name.isBlank()) {
                        call.respondText("error=missing_name", ContentType.Text.Plain, HttpStatusCode.BadRequest)
                        return@post
                    }
                    val affiliation = req["affiliation"].orEmpty().trim()
                    val department = req["department"].orEmpty().trim()
                    val fileName = markAttendance(authClient, name, affiliation, department)
                    if (fileName == null) {
                        call.respondText("error=attendance_write_failed", ContentType.Text.Plain, HttpStatusCode.InternalServerError)
                        return@post
                    }
                    call.respondText(
                        buildPlainFields(
                            "status" to "ok",
                            "file" to fileName,
                            "name" to name,
                        ),
                        ContentType.Text.Plain,
                    )
                }

                webSocket("/ws/db-updates") {
                    val token = call.request.queryParameters["token"].orEmpty().trim()
                    val clientId = clientByToken[token]
                    if (clientId.isNullOrBlank()) {
                        close(CloseReason(CloseReason.Codes.VIOLATED_POLICY, "unauthorized"))
                        return@webSocket
                    }

                    val sessionKey = "$clientId#${hashCode()}"
                    sessions[sessionKey] = this
                    val onlineCounter = onlineSessionsByClient.getOrPut(clientId) { AtomicInteger(0) }
                    onlineCounter.incrementAndGet()
                    log("WS 연결: $clientId (총 ${sessions.size})")

                    send(
                        Frame.Text(
                            buildPlainFields(
                                "type" to "db_update",
                                "revision" to revision.get().toString(),
                                "sourceClientId" to "server_bootstrap",
                            ),
                        ),
                    )

                    try {
                        for (frame in incoming) {
                            if (frame is Frame.Text) {
                                val text = frame.readText()
                                if (text == "ping") send(Frame.Text("pong"))
                            }
                        }
                    } finally {
                        sessions.remove(sessionKey)
                        val remaining = onlineSessionsByClient[clientId]?.decrementAndGet() ?: 0
                        if (remaining <= 0) {
                            onlineSessionsByClient.remove(clientId)
                        }
                        log("WS 종료: $clientId (총 ${sessions.size})")
                    }
                }
            }
        }

        return runCatching {
            engine.start(wait = false)
            server = engine
            if (awaitHealthReady(port, 20000L)) {
                log("서버 시작: 0.0.0.0:$port")
            } else {
                log("서버 시작 검증 지연: /api/health 응답 대기 지속 (port=$port)")
            }
            true
        }.getOrElse {
            log("서버 시작 실패: ${it.javaClass.simpleName}")
            false
        }
    }

    private fun awaitHealthReady(port: Int, timeoutMs: Long): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs.coerceAtLeast(1000L)
        while (System.currentTimeMillis() < deadline) {
            if (probeHealthOnce(port)) return true
            Thread.sleep(120)
        }
        return false
    }

    private fun probeHealthOnce(port: Int): Boolean {
        val conn = runCatching {
            (URL("http://127.0.0.1:$port/api/health").openConnection() as HttpURLConnection).apply {
                connectTimeout = 800
                readTimeout = 800
                requestMethod = "GET"
                useCaches = false
            }
        }.getOrNull() ?: return false
        return runCatching {
            conn.connect()
            if (conn.responseCode !in 200..299) return@runCatching false
            val body = conn.inputStream.bufferedReader().use { it.readText() }.trim()
            body.equals("ok", ignoreCase = true)
        }.getOrDefault(false).also {
            runCatching { conn.disconnect() }
        }
    }

    fun stop() {
        val running = server ?: return
        runCatching {
            running.stop(1000, 2000)
            sessions.clear()
            onlineSessionsByClient.clear()
            log("서버 중지")
        }
        server = null
    }

    fun isRunning(): Boolean = server != null

    fun endpointPort(): Int = port

    fun pendingClientIds(): List<String> {
        return pendingClients.keys.sorted()
    }

    fun approvedClientIds(): List<String> {
        return approvedTokensByClient.keys.sorted()
    }

    fun approveClient(clientId: String): Boolean {
        return approveClientInternal(clientId) != null
    }

    fun approveAllPending(): Int {
        return approveAllPendingInternal()
    }

    fun reloadDbFromStorage(): Boolean {
        return runCatching {
            dbStorageBackend = resolveDbStorageBackend()
            loadDbFromDisk()
            dbLoadedOnce = true
            val nextRevision = revision.incrementAndGet()
            if (isRunning()) {
                broadcastDbUpdate(nextRevision, "server_reload")
                log("DB 재로드 브로드캐스트 전송: rev=$nextRevision")
            }
            true
        }.getOrElse {
            log("DB 재로드 실패: ${it.javaClass.simpleName}")
            false
        }
    }

    fun reloadDbMetadataFromCsv(): Boolean {
        return runCatching {
            dbStorageBackend = resolveDbStorageBackend()
            val csvRaw = dbStorageBackend.readCsvText()
            if (csvRaw.isNullOrBlank()) {
                log("DB 재로드(CSV) 실패: CSV 파일을 읽을 수 없습니다.")
                return false
            }
            if (currentPeople.isEmpty()) {
                loadDbFromDisk()
            }
            if (currentPeople.isEmpty()) {
                log("DB 재로드(CSV) 실패: 기준이 되는 메모리/구조화 DB가 비어 있습니다.")
                return false
            }
            val merged = StructuredDbCodec.mergeMetadataIntoPeople(currentPeople, csvRaw, csvCodec)
            if (merged.isEmpty()) {
                log("DB 재로드(CSV) 실패: CSV 메타데이터와 일치하는 항목이 없습니다.")
                return false
            }
            if (!replaceCurrentPeople(merged)) {
                log("DB 재로드(CSV) 실패: face_meta.db 저장 중 오류가 발생했습니다.")
                return false
            }
            val nextRevision = revision.incrementAndGet()
            if (isRunning()) {
                broadcastDbUpdate(nextRevision, "server_reload_csv")
                log("DB 재로드(CSV) 브로드캐스트 전송: rev=$nextRevision")
            }
            log("DB 재로드(CSV) 완료: 메타데이터 ${merged.size}건 반영")
            true
        }.getOrElse {
            log("DB 재로드(CSV) 실패: ${it.javaClass.simpleName}")
            false
        }
    }

    @Synchronized
    fun dbRowsSnapshot(): List<DbPersonRow> {
        return currentPeople
            .asSequence()
            .sortedBy { it.name }
            .map { person ->
                DbPersonRow(
                    personKey = person.personKey,
                    name = person.name,
                    department = person.department,
                    grade = person.grade,
                    embeddingCount = person.embeddings.size,
                    metadata = LinkedHashMap(person.metadata),
                )
            }
            .toList()
    }

    private fun updateDigestInt(digest: MessageDigest, value: Int) {
        digest.update(((value ushr 24) and 0xFF).toByte())
        digest.update(((value ushr 16) and 0xFF).toByte())
        digest.update(((value ushr 8) and 0xFF).toByte())
        digest.update((value and 0xFF).toByte())
    }

    @Synchronized
    fun updatePersonRow(personKey: String, name: String, department: String, grade: String): Boolean {
        val nextName = name.trim()
        if (nextName.isBlank()) {
            log("DB 수정 실패: 이름이 비어 있습니다.")
            return false
        }
        val normalizedKey = personKey.trim()
        val targetIndex = currentPeople.indexOfFirst { it.personKey == normalizedKey }
        if (targetIndex < 0) {
            log("DB 수정 실패: 대상을 찾지 못했습니다. key=$normalizedKey")
            return false
        }
        if (currentPeople.any { it.personKey != normalizedKey && it.name == nextName }) {
            log("DB 수정 실패: 동일한 이름이 이미 존재합니다. name=$nextName")
            return false
        }
        val target = currentPeople[targetIndex]
        val metadata = LinkedHashMap(target.metadata)
        metadata.entries.firstOrNull { it.key.equals("name", ignoreCase = true) }?.setValue(nextName)
        if (metadata.keys.none { it.equals("name", ignoreCase = true) }) metadata["name"] = nextName
        metadata.entries.firstOrNull { it.key.equals("department", ignoreCase = true) }?.setValue(department.trim())
        if (metadata.keys.none { it.equals("department", ignoreCase = true) }) metadata["department"] = department.trim()
        metadata.entries.firstOrNull { it.key.equals("grade", ignoreCase = true) }?.setValue(grade.trim())
        if (metadata.keys.none { it.equals("grade", ignoreCase = true) }) metadata["grade"] = grade.trim()
        val normalizedMetadata = normalizeMetadataMap(metadata, nextName, department.trim(), grade.trim())
        val nextPeople = currentPeople.toMutableList()
        nextPeople[targetIndex] = target.copy(
            name = nextName,
            department = department.trim(),
            grade = grade.trim(),
            hash = personHash(nextName, department.trim(), grade.trim(), target.embeddings),
            updatedAt = System.currentTimeMillis(),
            personId = target.personId,
            personKey = target.personKey,
            metadata = normalizedMetadata,
            metadataHash = metadataHash(normalizedMetadata),
        )
        if (!replaceCurrentPeople(nextPeople)) {
            log("DB 수정 실패: 저장 중 오류가 발생했습니다. key=$normalizedKey")
            return false
        }
        val nextRevision = revision.incrementAndGet()
        if (isRunning()) {
            broadcastDbUpdate(nextRevision, "server_ui_edit")
        }
        log("DB 수정 완료: key=$normalizedKey, name=$nextName, rev=$nextRevision")
        return true
    }

    @Synchronized
    fun deletePersonRow(personKey: String): Boolean {
        val normalizedKey = personKey.trim()
        val nextPeople = currentPeople.filterNot { it.personKey == normalizedKey }.toMutableList()
        if (nextPeople.size == currentPeople.size) {
            log("DB 삭제 실패: 대상을 찾지 못했습니다. key=$normalizedKey")
            return false
        }
        if (!replaceCurrentPeople(nextPeople)) {
            log("DB 삭제 실패: 저장 중 오류가 발생했습니다. key=$normalizedKey")
            return false
        }
        val nextRevision = revision.incrementAndGet()
        if (isRunning()) {
            broadcastDbUpdate(nextRevision, "server_ui_delete")
        }
        log("DB 삭제 완료: key=${normalizedKey.ifBlank { "-" }}, rev=$nextRevision")
        return true
    }

    fun clientStatuses(): List<ClientConnectionStatus> {
        val allClientIds = mutableSetOf<String>()
        allClientIds += pendingClients.keys
        allClientIds += approvedTokensByClient.keys
        allClientIds += onlineSessionsByClient.keys

        return allClientIds.sorted().map { clientId ->
            val isPending = pendingClients.containsKey(clientId)
            val isApproved = approvedTokensByClient.containsKey(clientId)
            val onlineCount = onlineSessionsByClient[clientId]?.get() ?: 0
            val status = when {
                isPending -> "승인 대기"
                isApproved && onlineCount > 0 -> "연결됨"
                isApproved -> "승인됨(오프라인)"
                onlineCount > 0 -> "연결됨(미승인)"
                else -> "알 수 없음"
            }
            ClientConnectionStatus(clientId, status, onlineCount)
        }
    }

    private fun approveClientInternal(clientId: String): String? {
        val isAlreadyApproved = approvedTokensByClient.containsKey(clientId)
        if (!isAlreadyApproved && !pendingClients.containsKey(clientId)) {
            return null
        }
        val token = approvedTokensByClient[clientId] ?: UUID.randomUUID().toString().replace("-", "")
        approvedTokensByClient[clientId] = token
        clientByToken[token] = clientId
        pendingClients.remove(clientId)
        saveAuthStateToDisk()
        log("클라이언트 승인: clientId=$clientId")
        return token
    }

    private fun approveAllPendingInternal(): Int {
        val targets = pendingClients.keys.toList()
        targets.forEach { approveClientInternal(it) }
        if (targets.isNotEmpty()) {
            log("대기중 클라이언트 전체 승인: ${targets.size}건")
        }
        return targets.size
    }

    private fun authorizedClientId(authHeader: String?, tokenQuery: String?): String? {
        val headerToken = authHeader
            ?.takeIf { it.startsWith("Bearer ", ignoreCase = true) }
            ?.removePrefix("Bearer ")
            ?.trim()
            .orEmpty()

        val token = when {
            headerToken.isNotBlank() -> headerToken
            !tokenQuery.isNullOrBlank() -> tokenQuery.trim()
            else -> ""
        }
        if (token.isBlank()) return null
        return clientByToken[token]
    }

    private fun broadcastDbUpdate(nextRevision: Int, sourceClientId: String) {
        val msg = buildPlainFields(
            "type" to "db_update",
            "revision" to nextRevision.toString(),
            "sourceClientId" to sourceClientId,
        )

        log("DB 동기화 신호 전송: source=$sourceClientId, revision=$nextRevision, targets=${sessions.size}")

        sessions.values.forEach { session ->
            scope.launch {
                runCatching { session.send(Frame.Text(msg)) }
            }
        }
    }

    private fun loadDbFromDisk() {
        val startedAt = System.currentTimeMillis()
        log("DB 로드 시작: ${dbStorageBackend.description()}")
        if (!dbStorageBackend.hasStructuredDb()) {
            log("경고: DB 파일(.db/.bin)이 없습니다. (${dbStorageBackend.structuredDbDescription()})")
        }
        val readStartedAt = System.currentTimeMillis()
        val loaded = dbStorageBackend.readPeople()
        val readElapsedMs = System.currentTimeMillis() - readStartedAt
        if (loaded == null) {
            currentPeople = mutableListOf()
            revision.set(0)
            val totalElapsedMs = System.currentTimeMillis() - startedAt
            log("DB 초기화: ${dbStorageBackend.description()} (read=${readElapsedMs}ms, total=${totalElapsedMs}ms)")
            return
        }
        if (dbStorageBackend.needsStructuredSchemaUpgrade()) {
            if (dbStorageBackend.writePeople(loaded)) {
                log("Structured DB schema upgraded to person_key + person_idx + metadata_blob")
            } else {
                log("Structured DB schema upgrade failed")
            }
        }
        val normalized = assignSequentialPersonIds(loaded)
        currentPeople = normalized.toMutableList()
        if (loaded.map { it.personId } != normalized.map { it.personId }) {
            dbStorageBackend.writePeople(normalized)
        }
        revision.set(1)
        val totalElapsedMs = System.currentTimeMillis() - startedAt
        log(
            "DB 로드 완료: ${dbStorageBackend.description()} " +
                "(read=${readElapsedMs}ms, total=${totalElapsedMs}ms)",
        )
    }

    private fun saveDbToDisk(people: List<ServerPersonRecord> = currentPeople): Boolean {
        val ok = dbStorageBackend.writePeople(people)
        if (!ok) {
            log("DB 저장 실패: ${dbStorageBackend.description()}")
        }
        return ok
    }

    @Synchronized
    private fun replaceCurrentPeople(nextPeople: List<ServerPersonRecord>): Boolean {
        val snapshot = assignSequentialPersonIds(nextPeople)
        if (!saveDbToDisk(snapshot)) {
            return false
        }
        currentPeople = snapshot.toMutableList()
        return true
    }

    private fun assignSequentialPersonIds(people: List<ServerPersonRecord>): List<ServerPersonRecord> {
        return people
            .asSequence()
            .filter { it.name.isNotBlank() }
            .sortedWith(compareBy<ServerPersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
            .mapIndexed { index, person ->
                person.copy(
                    personId = (index + 1).toLong(),
                    embeddings = person.embeddings.map { it.copyOf() }.toMutableList(),
                )
            }
            .toList()
    }

    private fun buildHashSummaryBody(people: List<ServerPersonRecord>): String {
        return people
            .asSequence()
            .filter { it.personKey.isNotBlank() && it.hash.isNotBlank() }
            .sortedBy { it.personKey }
            .joinToString("\n") { "${it.personKey}\t${it.hash}" }
    }

    private fun parseHashSummaryBody(raw: String): Map<String, String> {
        val out = linkedMapOf<String, String>()
        raw.lineSequence().forEach { line ->
            val idx = line.indexOf('\t')
            if (idx <= 0) return@forEach
            val personKey = line.substring(0, idx).trim()
            val hash = line.substring(idx + 1).trim()
            if (personKey.isNotBlank() && hash.isNotBlank()) {
                out[personKey] = hash
            }
        }
        return out
    }

    private fun buildDeltaBundleForClient(clientHashes: Map<String, String>): ByteArray {
        val serverKeys = currentPeople.mapTo(mutableSetOf()) { it.personKey }
        val upserts = currentPeople
            .asSequence()
            .filter { it.personKey.isNotBlank() && clientHashes[it.personKey] != it.hash }
            .sortedBy { it.personKey }
            .toList()
        val deletes = clientHashes.keys.filter { it !in serverKeys }
        return encodeDeltaBundle(upserts, deletes)
    }

    private fun applyDeltaToCurrentDb(delta: DeltaPayload): Int? {
        val nextByPersonKey = linkedMapOf<String, ServerPersonRecord>()
        val nextByPersonId = linkedMapOf<Long, ServerPersonRecord>()
        val nextByIdentity = linkedMapOf<String, ServerPersonRecord>()
        currentPeople.forEach { person ->
            nextByPersonKey[person.personKey] = person
            nextByPersonId[person.personId] = person
            nextByIdentity[metadataIdentityKey(person.metadata)] = person
        }
        delta.deletes.forEach { personKey ->
            val removed = nextByPersonKey.remove(personKey)
            if (removed != null) {
                nextByIdentity.remove(metadataIdentityKey(removed.metadata))
            }
        }
        delta.upserts.forEach { incoming ->
            val existing = nextByPersonKey[incoming.personKey]
                ?: nextByIdentity[metadataIdentityKey(incoming.metadata)]
            val merged =
                if (existing != null) {
                    // 클라이언트 업로드는 임베딩만 반영하고, 서버 메타데이터(name/department/grade)가 정본이다.
                    existing.copy(
                        embeddings = incoming.embeddings.map { it.copyOf() }.toMutableList(),
                        hash = personHash(existing.name, existing.department, existing.grade, incoming.embeddings),
                        updatedAt = incoming.updatedAt,
                        metadataHash = metadataHash(existing.metadata),
                    )
                } else {
                    incoming
                }
            if (existing != null && existing.personId != merged.personId) {
                nextByPersonKey.remove(existing.personKey)
                nextByIdentity.remove(metadataIdentityKey(existing.metadata))
            }
            nextByPersonKey[merged.personKey] = merged
            nextByIdentity[metadataIdentityKey(merged.metadata)] = merged
        }
        val nextPeople = nextByPersonKey.values.sortedBy { it.name }
        if (!replaceCurrentPeople(nextPeople)) {
            return null
        }
        return delta.upserts.sumOf { it.embeddings.size }
    }

    @Synchronized
    private fun markAttendance(clientId: String, name: String, affiliationHint: String, departmentHint: String): String? {
        val date = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        val fileName = "attendance_$date.csv"
        val previousRaw = dbStorageBackend.readTextFile(fileName).orEmpty()
        val existingMarked = previousRaw
            .ifBlank { null }
            ?.let { csvCodec.parseAttendanceMarkedNames(it) }
            ?.toMutableSet()
            ?: linkedSetOf()
        existingMarked += name

        val baseRows = loadAttendanceBaseRows()
        val byName = linkedMapOf<String, AttendanceBaseRow>()
        baseRows.sortedBy { it.name }.forEach { byName[it.name] = it }
        if (!byName.containsKey(name)) {
            byName[name] = AttendanceBaseRow(
                name = name,
                affiliation = affiliationHint,
                department = departmentHint,
            )
        }
        val csv = buildAttendanceCsv(date, byName.values.toList(), existingMarked)
        val storageDescription = dbStorageBackend.description()
        log(
            "출석부 생성 준비: client=$clientId, name=$name, baseRows=${byName.size}, " +
                "marked=${existingMarked.size}, prevBytes=${previousRaw.toByteArray().size}, newBytes=${csv.toByteArray().size}, " +
                "storage=$storageDescription, file=$fileName",
        )
        if (!dbStorageBackend.writeTextFile(fileName, csv)) {
            log("출석부 저장 실패: client=$clientId, name=$name, file=$fileName, storage=$storageDescription")
            return null
        }
        val changed = previousRaw != csv
        val changeType = when {
            previousRaw.isBlank() -> "생성"
            changed -> "내용변경"
            else -> "변경없음"
        }
        val firstDataLine = csv.lineSequence().drop(2).firstOrNull().orEmpty()
        log(
            "출석부 $changeType 완료: file=$fileName, client=$clientId, name=$name, " +
                "rows=${byName.size}, marked=${existingMarked.size}, sample=${firstDataLine.ifBlank { "-" }}, " +
                "storage=$storageDescription",
        )
        return fileName
    }

    private fun loadAttendanceBaseRows(): List<AttendanceBaseRow> {
        val byName = linkedMapOf<String, AttendanceBaseRow>()
        val fromCsv = dbStorageBackend.readCsvText()
            ?.let { csvCodec.parseAttendanceRows(it) }
            .orEmpty()
            .filter { it.name.isNotBlank() }
        fromCsv.forEach {
            byName[it.name] = AttendanceBaseRow(
                name = it.name,
                affiliation = it.affiliation,
                department = it.department,
            )
        }
        currentPeople
            .asSequence()
            .filter { it.name.isNotBlank() }
            .forEach { person ->
                val existing = byName[person.name]
                byName[person.name] =
                    AttendanceBaseRow(
                        name = person.name,
                        affiliation = existing?.affiliation?.ifBlank { person.grade } ?: person.grade,
                        department = existing?.department?.ifBlank { person.department } ?: person.department,
                    )
            }
        log("출석부 기준 명단 로드: csv=${fromCsv.size}, memory=${currentPeople.size}, merged=${byName.size}")
        return byName.values.toList()
    }

    private fun buildAttendanceCsv(date: String, rows: List<AttendanceBaseRow>, attendedNames: Set<String>): String {
        val out = StringBuilder()
        out.append("출석부 기록일자,").append(csvEscape(date)).append('\n')
        out.append("이름,소속,부서,출석여부\n")
        rows.sortedBy { it.name }.forEach { row ->
            out.append(csvEscape(row.name))
                .append(',')
                .append(csvEscape(row.affiliation))
                .append(',')
                .append(csvEscape(row.department))
                .append(',')
                .append(csvEscape(if (row.name in attendedNames) "출석" else "미출석"))
                .append('\n')
        }
        return out.toString()
    }

    private fun csvEscape(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        val requiresQuote = escaped.any { ch -> ch == ',' || ch == '"' || ch == '\n' || ch == '\r' }
        return if (requiresQuote) "\"$escaped\"" else escaped
    }

    private fun resolveDbStorageBackend(): DbStorageBackend {
        val spec = dbStorageDirSpec.trim()
        if (spec.startsWith("content://")) {
            val uri = runCatching { Uri.parse(spec) }.getOrNull()
            if (uri != null) {
                val uriKey = Integer.toHexString(uri.toString().hashCode())
                return TreeUriDbStorageBackend(
                    csvCodec = csvCodec,
                    context = context,
                    treeUri = uri,
                    csvFileName = "face_db.csv",
                    metaDbFileName = "face_meta.db",
                    embeddingsBinFileName = "face_embeddings.bin",
                    metaDbFile = File(serverDataDir, "meta/face_meta_$uriKey.db"),
                    embeddingsBinFile = File(serverDataDir, "meta/face_embeddings_$uriKey.bin"),
                )
            }
        }
        if (spec.isNotBlank()) {
            val f = File(spec)
            val dir = if (f.isAbsolute) f else File(context.filesDir, spec)
            return FileDbStorageBackend(
                csvCodec = csvCodec,
                csvFile = File(dir, "face_db.csv"),
                metaDbFile = File(dir, "face_meta.db"),
                embeddingsBinFile = File(dir, "face_embeddings.bin"),
            )
        }
        return FileDbStorageBackend(
            csvCodec = csvCodec,
            csvFile = File(serverDataDir, "face_db.csv"),
            metaDbFile = File(serverDataDir, "face_meta.db"),
            embeddingsBinFile = File(serverDataDir, "face_embeddings.bin"),
        )
    }

    private fun personHash(name: String, department: String, grade: String, embeddings: List<FloatArray>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(name.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(department.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(grade.toByteArray(Charsets.UTF_8))
        digest.update(0)
        updateDigestInt(digest, embeddings.size)
        embeddings.forEach { emb ->
            updateDigestInt(digest, emb.size)
            emb.forEach { value ->
                updateDigestInt(digest, java.lang.Float.floatToIntBits(value))
            }
        }
        return digest.digest().joinToString("") { b -> "%02x".format(b) }
    }

    private fun loadAuthStateFromDisk() {
        approvedTokensByClient.clear()
        clientByToken.clear()
        if (!authStateFile.exists()) return

        val raw = runCatching { authStateFile.readText() }.getOrNull() ?: return
        val root = runCatching { JSONObject(raw) }.getOrNull() ?: return
        val approved = root.optJSONObject("approvedTokensByClient") ?: JSONObject()

        var loadedCount = 0
        val keys = approved.keys()
        while (keys.hasNext()) {
            val clientId = keys.next()
            val token = approved.optString(clientId, "").trim()
            if (clientId.isBlank() || token.isBlank()) continue
            approvedTokensByClient[clientId] = token
            clientByToken[token] = clientId
            loadedCount += 1
        }
        if (loadedCount > 0) {
            log("승인 토큰 로드 완료: ${loadedCount}건")
        }
    }

    private fun saveAuthStateToDisk() {
        runCatching {
            val approved = JSONObject()
            approvedTokensByClient.toSortedMap().forEach { (clientId, token) ->
                approved.put(clientId, token)
            }
            val out = JSONObject().put("approvedTokensByClient", approved)
            authStateFile.parentFile?.mkdirs()
            authStateFile.writeText(out.toString(2))
        }.onFailure {
            log("승인 토큰 저장 실패: ${it.javaClass.simpleName}")
        }
    }

    private fun parseRequestBody(raw: String): JSONObject {
        return runCatching { JSONObject(raw) }.getOrDefault(JSONObject())
    }

    private fun parsePlainFields(raw: String): Map<String, String> {
        return raw
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() && it.contains('=') }
            .associate { line ->
                val idx = line.indexOf('=')
                line.substring(0, idx).trim() to line.substring(idx + 1).trim()
            }
    }

    private fun buildPlainFields(vararg pairs: Pair<String, String>): String {
        return pairs.joinToString("\n") { (key, value) -> "$key=${value.replace("\n", " ").trim()}" }
    }

    private fun encodeDeltaBundle(upserts: List<ServerPersonRecord>, deletes: List<String>): ByteArray {
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf('F'.code.toByte(), 'D'.code.toByte(), 'D'.code.toByte(), '1'.code.toByte()))
        writeIntLe(out, deletes.size)
        deletes.sorted().forEach { writeString(out, it) }
        writeIntLe(out, upserts.size)
        upserts.sortedBy { it.personKey }.forEach { person ->
            val embCount = person.embeddings.size
            val dim = person.embeddings.firstOrNull()?.size ?: 0
            writeString(out, person.personKey)
            writeString(out, person.personId.toString())
            writeString(out, person.name)
            writeString(out, person.department)
            writeString(out, person.grade)
            writeString(out, person.hash)
            writeLongLe(out, person.updatedAt)
            val metadataBlob = encodeMetadataBlob(person.metadata)
            writeIntLe(out, metadataBlob.size)
            out.write(metadataBlob)
            writeString(out, person.metadataHash)
            writeIntLe(out, embCount)
            writeIntLe(out, dim)
            person.embeddings.forEach { emb ->
                for (j in 0 until dim) {
                    writeFloatLe(out, emb.getOrElse(j) { 0f })
                }
            }
        }
        return out.toByteArray()
    }

    private fun parseDeltaBundle(bytes: ByteArray): DeltaPayload? {
        if (bytes.size < 4) return null
        if (bytes[0] != 'F'.code.toByte() || bytes[1] != 'D'.code.toByte() || bytes[2] != 'D'.code.toByte() || bytes[3] != '1'.code.toByte()) return null
        val cursor = DeltaCursor(bytes, 4)
        val deleteCount = cursor.readInt() ?: return null
        val deletes = MutableList(deleteCount.coerceAtLeast(0)) { cursor.readString().orEmpty() }.filter { it.isNotBlank() }
        val upsertCount = cursor.readInt() ?: return null
        val upserts = mutableListOf<ServerPersonRecord>()
        repeat(upsertCount.coerceAtLeast(0)) {
            val personKeyRaw = cursor.readString()
            val personId = parsePersonId(cursor.readString())
            val name = cursor.readString() ?: return@repeat
            val department = cursor.readString() ?: return@repeat
            val grade = cursor.readString() ?: return@repeat
            val hash = cursor.readString() ?: return@repeat
            val updatedAt = cursor.readLong() ?: return@repeat
            val metadataBlobLength = cursor.readInt() ?: return@repeat
            val metadataBlob = cursor.readBytes(metadataBlobLength) ?: return@repeat
            val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
            val incomingMetadataHash = cursor.readString() ?: metadataHash(metadata)
            val embCount = cursor.readInt() ?: return@repeat
            val dim = cursor.readInt() ?: return@repeat
            val embeddings = mutableListOf<FloatArray>()
            repeat(embCount.coerceAtLeast(0)) {
                val emb = FloatArray(dim.coerceAtLeast(0))
                for (i in 0 until dim.coerceAtLeast(0)) {
                    emb[i] = cursor.readFloat() ?: return@repeat
                }
                embeddings += emb
            }
            if (name.isBlank() || embeddings.isEmpty()) return@repeat
            upserts += ServerPersonRecord(
                name = name,
                department = department,
                grade = grade,
                embeddings = embeddings,
                hash = hash,
                updatedAt = updatedAt,
                personId = if (personId > 0L) personId else personIdFromIdentity(name, department, grade),
                personKey = parsePersonKey(personKeyRaw, name, department, grade),
                metadata = metadata,
                metadataHash = incomingMetadataHash.ifBlank { metadataHash(metadata) },
            )
        }
        return DeltaPayload(upserts, deletes)
    }

    private fun writeString(out: ByteArrayOutputStream, value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeIntLe(out, bytes.size)
        out.write(bytes)
    }

    private fun writeIntLe(out: ByteArrayOutputStream, value: Int) {
        out.write(value and 0xFF)
        out.write((value ushr 8) and 0xFF)
        out.write((value ushr 16) and 0xFF)
        out.write((value ushr 24) and 0xFF)
    }

    private fun writeLongLe(out: ByteArrayOutputStream, value: Long) {
        for (shift in 0 until 64 step 8) {
            out.write(((value ushr shift) and 0xFF).toInt())
        }
    }

    private fun writeFloatLe(out: ByteArrayOutputStream, value: Float) {
        writeIntLe(out, java.lang.Float.floatToIntBits(value))
    }

    private data class DeltaPayload(
        val upserts: List<ServerPersonRecord>,
        val deletes: List<String>,
    )

    private class DeltaCursor(private val bytes: ByteArray, private var offset: Int) {
        fun readInt(): Int? {
            if (offset + 4 > bytes.size) return null
            val value =
                (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            offset += 4
            return value
        }

        fun readLong(): Long? {
            if (offset + 8 > bytes.size) return null
            var value = 0L
            for (i in 0 until 8) {
                value = value or ((bytes[offset + i].toLong() and 0xFFL) shl (8 * i))
            }
            offset += 8
            return value
        }

        fun readFloat(): Float? = readInt()?.let { Float.fromBits(it) }

        fun readString(): String? {
            val size = readInt() ?: return null
            if (size < 0 || offset + size > bytes.size) return null
            val value = bytes.copyOfRange(offset, offset + size).toString(Charsets.UTF_8)
            offset += size
            return value
        }

        fun readBytes(size: Int): ByteArray? {
            if (size < 0 || offset + size > bytes.size) return null
            val value = bytes.copyOfRange(offset, offset + size)
            offset += size
            return value
        }
    }

    private fun parseStructuredBundle(payload: ByteArray): Pair<ByteArray, ByteArray>? {
        if (payload.size < 12) return null
        if (payload[0] != 'F'.code.toByte() || payload[1] != 'D'.code.toByte() || payload[2] != 'B'.code.toByte() || payload[3] != '1'.code.toByte()) {
            return null
        }
        val metaLength = readIntLe(payload, 4)
        val binLength = readIntLe(payload, 8)
        if (metaLength < 0 || binLength < 0) return null
        val totalLength = 12L + metaLength.toLong() + binLength.toLong()
        if (totalLength != payload.size.toLong()) return null
        val metaStart = 12
        val metaEnd = metaStart + metaLength
        val binEnd = metaEnd + binLength
        return payload.copyOfRange(metaStart, metaEnd) to payload.copyOfRange(metaEnd, binEnd)
    }

    private fun readIntLe(bytes: ByteArray, offset: Int): Int {
        if (offset < 0 || offset + 4 > bytes.size) return -1
        return (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 3].toInt() and 0xFF) shl 24)
    }

    private fun log(message: String) {
        Log.i(LOGCAT_TAG, message)
        val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        onLog("[$ts] $message")
    }

    fun release() {
        stop()
        scope.cancel()
    }

    private interface DbStorageBackend {
        fun readPeople(): MutableList<ServerPersonRecord>?
        fun writePeople(people: List<ServerPersonRecord>): Boolean
        fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean
        fun readCsvText(): String?
        fun readTextFile(fileName: String): String?
        fun writeTextFile(fileName: String, raw: String): Boolean
        fun description(): String
        fun hasStructuredDb(): Boolean
        fun structuredDbDescription(): String
        fun readStructuredMetaBytes(): ByteArray?
        fun readStructuredEmbeddingsBytes(): ByteArray?
        fun needsStructuredSchemaUpgrade(): Boolean
    }

    private class FileDbStorageBackend(
        private val csvCodec: FaceDbCsvCodec,
        private val csvFile: File,
        private val metaDbFile: File,
        private val embeddingsBinFile: File,
    ) : DbStorageBackend {
        override fun hasStructuredDb(): Boolean {
            return metaDbFile.exists() && embeddingsBinFile.exists()
        }

        override fun structuredDbDescription(): String {
            return "${metaDbFile.absolutePath}, ${embeddingsBinFile.absolutePath}"
        }

        override fun readPeople(): MutableList<ServerPersonRecord>? {
            val structuredExists = hasStructuredDb()
            val csvNeedsImport = csvFile.exists() &&
                (
                    !structuredExists ||
                        csvFile.lastModified() > maxOf(metaDbFile.lastModified(), embeddingsBinFile.lastModified())
                    )

            if (csvNeedsImport) {
                val raw = runCatching { csvFile.readText() }.getOrNull()
                if (raw != null) {
                    val people =
                        if (structuredExists) {
                            val base = StructuredDbCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredDbCodec.mergeMetadataIntoPeople(base, raw, csvCodec) else null
                        } else {
                            StructuredDbCodec.parseCsvDbToPeople(raw, csvCodec)
                        }
                    if (people != null && StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)) {
                        writeCsvSnapshot(people)
                        return people
                    }
                }
            }

            if (structuredExists) {
                StructuredDbCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { people ->
                    if (!csvFile.exists()) {
                        writeCsvSnapshot(people)
                    }
                    if (StructuredDbCodec.needsSchemaUpgrade(metaDbFile)) {
                        StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)
                    }
                    return people
                }
            }
            return null
        }

        override fun writePeople(people: List<ServerPersonRecord>): Boolean {
            if (!StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)) return false
            return writeCsvSnapshot(people)
        }

        override fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaDbBytes)
                embeddingsBinFile.writeBytes(embeddingsBinBytes)
                val people = StructuredDbCodec.loadPeopleMetadataOnly(metaDbFile) ?: return@runCatching false
                writeCsvSnapshot(people)
            }.getOrDefault(false)
        }

        override fun readCsvText(): String? {
            return runCatching { csvFile.readText() }.getOrNull()
        }

        override fun readTextFile(fileName: String): String? {
            val file = File(csvFile.parentFile ?: metaDbFile.parentFile ?: embeddingsBinFile.parentFile ?: File("."), fileName)
            return runCatching { if (file.exists()) file.readText() else null }.getOrNull()
        }

        override fun writeTextFile(fileName: String, raw: String): Boolean {
            val file = File(csvFile.parentFile ?: metaDbFile.parentFile ?: embeddingsBinFile.parentFile ?: File("."), fileName)
            return runCatching {
                file.parentFile?.mkdirs()
                file.writeText(raw)
                true
            }.getOrDefault(false)
        }

        override fun description(): String {
            return "${csvFile.absolutePath} (meta=${metaDbFile.absolutePath}, bin=${embeddingsBinFile.absolutePath})"
        }

        override fun readStructuredMetaBytes(): ByteArray? {
            return runCatching { metaDbFile.readBytes() }.getOrNull()
        }

        override fun readStructuredEmbeddingsBytes(): ByteArray? {
            return runCatching { embeddingsBinFile.readBytes() }.getOrNull()
        }

        override fun needsStructuredSchemaUpgrade(): Boolean = StructuredDbCodec.needsSchemaUpgrade(metaDbFile)

        private fun writeCsvSnapshot(people: List<ServerPersonRecord>): Boolean {
            return runCatching {
                csvFile.parentFile?.mkdirs()
                csvFile.writeText(csvCodec.encodePeople(people))
                true
            }.getOrDefault(false)
        }
    }

    private class TreeUriDbStorageBackend(
        private val csvCodec: FaceDbCsvCodec,
        private val context: Context,
        private val treeUri: Uri,
        private val csvFileName: String,
        private val metaDbFileName: String,
        private val embeddingsBinFileName: String,
        private val metaDbFile: File,
        private val embeddingsBinFile: File,
    ) : DbStorageBackend {
        override fun hasStructuredDb(): Boolean {
            val metaDoc = resolveDocumentFile(metaDbFileName, createIfMissing = false)
            val binDoc = resolveDocumentFile(embeddingsBinFileName, createIfMissing = false)
            return metaDoc != null && binDoc != null
        }

        override fun structuredDbDescription(): String {
            return "$treeUri/$metaDbFileName, $treeUri/$embeddingsBinFileName"
        }

        override fun readPeople(): MutableList<ServerPersonRecord>? {
            val metaDoc = resolveDocumentFile(metaDbFileName, createIfMissing = false)
            val binDoc = resolveDocumentFile(embeddingsBinFileName, createIfMissing = false)
            val structuredExists = (metaDoc != null && binDoc != null) && syncStructuredFromDocuments(metaDoc, binDoc)
            val csvDoc = resolveDocumentFile(csvFileName, createIfMissing = false)
            val csvNeedsImport = csvDoc != null &&
                (
                    !structuredExists ||
                        csvDoc.lastModified() > maxOf(metaDoc?.lastModified() ?: 0L, binDoc?.lastModified() ?: 0L)
                    )

            if (csvNeedsImport) {
                val csvRaw = runCatching {
                    context.contentResolver.openInputStream(csvDoc!!.uri)?.bufferedReader()?.use { it.readText() }
                }.getOrNull()
                if (csvRaw != null) {
                    val people =
                        if (structuredExists) {
                            val base = StructuredDbCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredDbCodec.mergeMetadataIntoPeople(base, csvRaw, csvCodec) else null
                        } else {
                            StructuredDbCodec.parseCsvDbToPeople(csvRaw, csvCodec)
                        }
                    if (people != null && StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)) {
                        syncStructuredToDocuments()
                        writeDocumentText(csvFileName, csvCodec.encodePeople(people))
                        return people
                    }
                }
            }

            if (structuredExists) {
                StructuredDbCodec.loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { people ->
                    if (csvDoc == null) {
                        writeDocumentText(csvFileName, csvCodec.encodePeople(people))
                    }
                    if (StructuredDbCodec.needsSchemaUpgrade(metaDbFile)) {
                        StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)
                        syncStructuredToDocuments()
                    }
                    return people
                }
            }
            return null
        }

        override fun writePeople(people: List<ServerPersonRecord>): Boolean {
            if (!StructuredDbCodec.savePeopleToStructured(people, metaDbFile, embeddingsBinFile)) return false
            if (!syncStructuredToDocuments()) return false
            return writeDocumentText(csvFileName, csvCodec.encodePeople(people))
        }

        override fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
            val localOk = runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaDbBytes)
                embeddingsBinFile.writeBytes(embeddingsBinBytes)
                true
            }.getOrDefault(false)
            if (!localOk) return false
            val meta = StructuredDbCodec.loadPeopleMetadataOnly(metaDbFile) ?: return false
            syncStructuredToDocuments()
            return writeDocumentText(csvFileName, csvCodec.encodePeople(meta))
        }

        override fun readCsvText(): String? = readDocumentText(csvFileName)

        override fun readTextFile(fileName: String): String? = readDocumentText(fileName)

        override fun writeTextFile(fileName: String, raw: String): Boolean = writeDocumentText(fileName, raw)

        override fun description(): String {
            return "$treeUri (csv=$csvFileName, meta=$metaDbFileName, bin=$embeddingsBinFileName)"
        }

        override fun readStructuredMetaBytes(): ByteArray? {
            if (metaDbFile.exists()) {
                runCatching { metaDbFile.readBytes() }.getOrNull()?.let { return it }
            }
            val metaDoc = resolveDocumentFile(metaDbFileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(metaDoc.uri)?.use { it.readBytes() }
            }.getOrNull()
        }

        override fun readStructuredEmbeddingsBytes(): ByteArray? {
            if (embeddingsBinFile.exists()) {
                runCatching { embeddingsBinFile.readBytes() }.getOrNull()?.let { return it }
            }
            val binDoc = resolveDocumentFile(embeddingsBinFileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(binDoc.uri)?.use { it.readBytes() }
            }.getOrNull()
        }

        override fun needsStructuredSchemaUpgrade(): Boolean = StructuredDbCodec.needsSchemaUpgrade(metaDbFile)

        private fun readDocumentText(fileName: String): String? {
            val file = resolveDocumentFile(fileName, createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(file.uri)?.bufferedReader()?.use { it.readText() }
            }.getOrNull()
        }

        private fun writeDocumentText(fileName: String, raw: String): Boolean {
            val file = resolveDocumentFile(fileName, createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "rwt")?.bufferedWriter()?.use { it.write(raw) } != null
            }.getOrDefault(false)
        }

        private fun writeDocumentBytes(fileName: String, bytes: ByteArray): Boolean {
            val file = resolveDocumentFile(fileName, createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "w")?.use { it.write(bytes) } != null
            }.getOrDefault(false)
        }

        private fun syncStructuredFromDocuments(metaDoc: DocumentFile?, binDoc: DocumentFile?): Boolean {
            if (metaDoc == null || binDoc == null) return false
            val metaBytes = runCatching {
                context.contentResolver.openInputStream(metaDoc.uri)?.use { it.readBytes() }
            }.getOrNull() ?: return false
            val binBytes = runCatching {
                context.contentResolver.openInputStream(binDoc.uri)?.use { it.readBytes() }
            }.getOrNull() ?: return false

            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaBytes)
                embeddingsBinFile.writeBytes(binBytes)
                true
            }.getOrDefault(false)
        }

        private fun syncStructuredToDocuments(): Boolean {
            val metaBytes = runCatching { metaDbFile.readBytes() }.getOrNull() ?: return false
            val binBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return false
            val metaOk = writeDocumentBytes(metaDbFileName, metaBytes)
            val binOk = writeDocumentBytes(embeddingsBinFileName, binBytes)
            return metaOk && binOk
        }

        private fun resolveDocumentFile(fileName: String, createIfMissing: Boolean): DocumentFile? {
            val root = DocumentFile.fromTreeUri(context, treeUri) ?: return null
            val existing = root.findFile(fileName)
            if (existing != null) return existing
            if (!createIfMissing) return null
            val lowerName = fileName.lowercase(Locale.US)
            val mimeType = when {
                lowerName.endsWith(".csv") -> "text/csv"
                lowerName.endsWith(".db") || lowerName.endsWith(".bin") -> "application/octet-stream"
                else -> "application/octet-stream"
            }
            return root.createFile(mimeType, fileName)
        }
    }

    private object StructuredDbCodec {
        private const val TABLE = "people"

        fun parseCsvDbToPeople(raw: String, csvCodec: FaceDbCsvCodec): MutableList<ServerPersonRecord>? {
            val root = parseCsvDb(raw, csvCodec) ?: return null
            return jsonRootToPeople(root)
        }

        fun needsSchemaUpgrade(metaDbFile: File): Boolean {
            if (!metaDbFile.exists()) return false
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return false
            return runCatching {
                !(hasColumn(db, TABLE, "person_key") && hasColumn(db, TABLE, "person_idx") && hasColumn(db, TABLE, "metadata_blob") && hasColumn(db, TABLE, "metadata_hash"))
            }.getOrDefault(false).also { db.close() }
        }

        fun mergeMetadataIntoPeople(
            currentPeople: List<ServerPersonRecord>,
            csvRaw: String,
            csvCodec: FaceDbCsvCodec,
        ): MutableList<ServerPersonRecord> {
            val rows = csvCodec.parseMetadataRows(csvRaw)
            if (rows.isEmpty()) return currentPeople.map { it.copy(embeddings = it.embeddings.map { emb -> emb.copyOf() }.toMutableList()) }.toMutableList()
            val byPersonKey = currentPeople.associateBy { it.personKey }
            val byPersonId = currentPeople.associateBy { it.personId }
            val byIdentity = currentPeople.associateBy { metadataIdentityKey(it.metadata) }
            val byUniqueName = currentPeople
                .groupBy { metadataName(it.metadata, it.name).trim().lowercase(Locale.US) }
                .filterValues { it.size == 1 }
                .mapValues { it.value.first() }
            return rows
                .mapNotNull { row ->
                    val baseByPersonKey = byPersonKey[row.personKey]
                    val baseByPersonId = byPersonId[row.personId]
                    val baseByIdentity = byIdentity[metadataIdentityKey(row.metadata)]
                    val baseByName = byUniqueName[row.name.trim().lowercase(Locale.US)]
                    val base = when {
                        baseByPersonKey != null -> baseByPersonKey
                        baseByPersonId != null && baseByName != null &&
                            !metadataName(baseByPersonId.metadata, baseByPersonId.name).equals(row.name, ignoreCase = true) -> baseByName
                        baseByPersonId != null && metadataIdentityKey(baseByPersonId.metadata) == metadataIdentityKey(row.metadata) -> baseByPersonId
                        else -> baseByIdentity ?: baseByName
                    } ?: return@mapNotNull null
                    base.copy(
                        name = row.name,
                        department = row.department,
                        grade = row.grade,
                        embeddings = base.embeddings.map { emb -> emb.copyOf() }.toMutableList(),
                        hash = computeHash(row.name, row.department, row.grade, base.embeddings),
                        updatedAt = row.updatedAt,
                        personId = row.personId,
                        personKey = row.personKey,
                        metadata = row.metadata,
                        metadataHash = row.metadataHash,
                    )
                }
                .toMutableList()
        }

        fun loadPeopleMetadataOnly(metaDbFile: File): MutableList<ServerPersonRecord>? {
            if (!metaDbFile.exists()) return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val out = mutableListOf<ServerPersonRecord>()
                val hasGrade = hasColumn(db, TABLE, "grade")
                val hasPersonId = hasColumn(db, TABLE, "person_idx")
                val hasMetadataBlob = hasColumn(db, TABLE, "metadata_blob")
                val hasMetadataHash = hasColumn(db, TABLE, "metadata_hash")
                val projection = if (hasGrade) {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("grade")
                        add("embedding_count")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                } else {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("embedding_count")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                }
                db.query(
                    TABLE,
                    projection,
                    null, null, null, null, "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val rawPersonId = cursor.getString(0)
                        val name = cursor.getString(1)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(2)?.trim().orEmpty()
                        val grade = if (hasGrade) cursor.getString(3)?.trim().orEmpty() else ""
                        val personId = parsePersonId(rawPersonId, name, department, grade)
                        val base = if (hasGrade) 0 else -1
                        val count = cursor.getInt(4 + base).coerceAtLeast(0)
                        val hash = cursor.getString(5 + base)?.trim().orEmpty()
                        val updatedAt = cursor.getLong(6 + base)
                        val metadataBlob = if (hasMetadataBlob) cursor.getBlob(7 + base) else null
                        val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
                        val placeholders = MutableList(count) { FloatArray(0) }
                        out += ServerPersonRecord(
                            name = name,
                            department = department,
                            grade = grade,
                            embeddings = placeholders,
                            hash = hash,
                            updatedAt = updatedAt,
                            personId = personId,
                            metadata = metadata,
                            metadataHash = if (hasMetadataHash) {
                                cursor.getString(8 + base)?.trim().orEmpty().ifBlank { metadataHash(metadata) }
                            } else {
                                metadataHash(metadata)
                            },
                        )
                    }
                }
                out
            }.getOrNull().also { db.close() }
        }

        fun loadPeopleFromStructured(metaDbFile: File, embeddingsBinFile: File): MutableList<ServerPersonRecord>? {
            if (!metaDbFile.exists() || !embeddingsBinFile.exists()) return null
            val embeddingsBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val out = mutableListOf<ServerPersonRecord>()
                val hasGrade = hasColumn(db, TABLE, "grade")
                val hasPersonId = hasColumn(db, TABLE, "person_idx")
                val hasMetadataBlob = hasColumn(db, TABLE, "metadata_blob")
                val hasMetadataHash = hasColumn(db, TABLE, "metadata_hash")
                val projection = if (hasGrade) {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("grade")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                } else {
                    buildList {
                        add(if (hasPersonId) "person_idx" else "name")
                        add("name")
                        add("department")
                        add("offset_bytes")
                        add("length_bytes")
                        add("embedding_count")
                        add("embedding_dim")
                        add("hash")
                        add("updated_at")
                        if (hasMetadataBlob) add("metadata_blob")
                        if (hasMetadataHash) add("metadata_hash")
                    }.toTypedArray()
                }
                db.query(
                    TABLE,
                    projection,
                    null, null, null, null, "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val rawPersonId = cursor.getString(0)
                        val name = cursor.getString(1)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(2)?.trim().orEmpty()
                        val grade = if (hasGrade) cursor.getString(3)?.trim().orEmpty() else ""
                        val personId = parsePersonId(rawPersonId, name, department, grade)
                        val base = if (hasGrade) 0 else -1
                        val offset = cursor.getLong(4 + base).toInt()
                        val length = cursor.getLong(5 + base).toInt()
                        val count = cursor.getInt(6 + base).coerceAtLeast(0)
                        val dim = cursor.getInt(7 + base).coerceAtLeast(0)
                        val hash = cursor.getString(8 + base)?.trim().orEmpty()
                        val updatedAt = cursor.getLong(9 + base)
                        val metadataBlob = if (hasMetadataBlob) cursor.getBlob(10 + base) else null
                        val metadata = normalizeMetadataMap(decodeMetadataBlob(metadataBlob), name, department, grade)
                        val embeddings = decodeEmbeddingsList(embeddingsBytes, offset, length, count, dim)
                        if (embeddings.isEmpty()) continue
                        out += ServerPersonRecord(
                            name = name,
                            department = department,
                            grade = grade,
                            embeddings = embeddings,
                            hash = hash.ifBlank { computeHash(name, department, grade, embeddings) },
                            updatedAt = if (updatedAt > 0L) updatedAt else System.currentTimeMillis(),
                            personId = personId,
                            metadata = metadata,
                            metadataHash = if (hasMetadataHash) {
                                cursor.getString(11 + base)?.trim().orEmpty().ifBlank { metadataHash(metadata) }
                            } else {
                                metadataHash(metadata)
                            },
                        )
                    }
                }
                out
            }.getOrNull().also { db.close() }
        }

        fun savePeopleToStructured(people: List<ServerPersonRecord>, metaDbFile: File, embeddingsBinFile: File): Boolean {
            val records = people
                .asSequence()
                .filter { it.name.isNotBlank() && it.embeddings.isNotEmpty() }
                .sortedWith(compareBy<ServerPersonRecord> { it.name }.thenBy { it.grade }.thenBy { it.department })
                .mapIndexed { index, person -> person.copy(personId = (index + 1).toLong()) }
                .map { person ->
                    Record(
                        personId = person.personId,
                        personKey = person.personKey,
                        name = person.name,
                        department = person.department,
                        grade = person.grade,
                        embeddings = person.embeddings.map { it.copyOf() },
                        hash = person.hash.ifBlank { computeHash(person.name, person.department, person.grade, person.embeddings) },
                        updatedAt = person.updatedAt,
                        metadataBlob = encodeMetadataBlob(person.metadata),
                        metadataHash = person.metadataHash,
                    )
                }
                .toList()
            return saveRecordsToStructured(records, metaDbFile, embeddingsBinFile)
        }

        fun parseCsvDb(raw: String, csvCodec: FaceDbCsvCodec): JSONObject? {
            val fromCsv = csvCodec.decode(raw) ?: return null
            return normalizeRoot(fromCsv)
        }

        fun mergeWithCsvMetadata(currentRoot: JSONObject, csvRaw: String, csvCodec: FaceDbCsvCodec): JSONObject {
            val base = normalizeRoot(currentRoot)
            val rows = csvCodec.parseMetadataRows(csvRaw)
            if (rows.isEmpty()) {
                return JSONObject().put("people", JSONArray())
            }
            val people = base.optJSONArray("people") ?: JSONArray()
            val byName = mutableMapOf<String, JSONObject>()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                byName[name] = person
            }
            val out = JSONArray()
            rows.forEach { row ->
                val basePerson = byName[row.name] ?: return@forEach
                val obj = JSONObject(basePerson.toString())
                obj.put("name", row.name)
                obj.put("department", row.department)
                obj.put("grade", row.grade)
                obj.put("updatedAt", row.updatedAt)
                obj.put("personId", row.personId)
                obj.put("metadata", JSONObject(row.metadata as Map<*, *>))
                val embeddings = normalizeEmbeddingsList(obj.optJSONArray("embeddings") ?: JSONArray())
                obj.put("embeddings", peopleToJsonEmbeddings(embeddings))
                obj.put("hash", computeHash(row.name, row.department, row.grade, embeddings))
                out.put(obj)
            }
            return JSONObject().put("people", out)
        }

        fun loadMetadataOnly(metaDbFile: File): JSONObject? {
            return loadPeopleMetadataOnly(metaDbFile)?.let { peopleToJsonRoot(it, includeEmbeddings = false) }
        }

        fun loadFromStructured(metaDbFile: File, embeddingsBinFile: File): JSONObject? {
            return loadPeopleFromStructured(metaDbFile, embeddingsBinFile)?.let { peopleToJsonRoot(it, includeEmbeddings = true) }
        }

        fun saveToStructured(root: JSONObject, metaDbFile: File, embeddingsBinFile: File): Boolean {
            val normalized = normalizeRoot(root)
            val people = normalized.optJSONArray("people") ?: JSONArray()
            val records = mutableListOf<Record>()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                val department = person.optString("department", person.optString("folder", "")).trim()
                val grade = person.optString("grade", "").trim()
                val embeddings = normalizeEmbeddingsList(person.optJSONArray("embeddings") ?: JSONArray())
                if (embeddings.isEmpty()) continue
                val updatedAt = person.optLong("updatedAt", System.currentTimeMillis())
                val hash = person.optString("hash", "").trim().ifBlank { computeHash(name, department, grade, embeddings) }
                val metadataJson = person.optJSONObject("metadata")
                val metadata = linkedMapOf<String, String>()
                if (metadataJson != null) {
                    val keys = metadataJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        metadata[key] = metadataJson.optString(key, "")
                    }
                }
                val normalizedMetadata = normalizeMetadataMap(metadata, name, department, grade)
                records.add(
                    Record(
                        personId = parsePersonId(person.optString("personIdx", person.optString("personId", "")), name, department, grade),
                        personKey = parsePersonKey(person.optString("personKey", ""), name, department, grade),
                        name = name,
                        department = department,
                        grade = grade,
                        embeddings = embeddings,
                        hash = hash,
                        updatedAt = updatedAt,
                        metadataBlob = encodeMetadataBlob(normalizedMetadata),
                        metadataHash = metadataHash(normalizedMetadata),
                    ),
                )
            }
            return saveRecordsToStructured(records, metaDbFile, embeddingsBinFile)
        }

        private fun normalizeRoot(root: JSONObject): JSONObject {
            root.optJSONArray("people")?.let { return root }
            root.optJSONArray("persons")?.let { return JSONObject().put("people", it) }
            root.optJSONArray("faces")?.let { return JSONObject().put("people", it) }
            val map = root.optJSONObject("people")
                ?: root.optJSONObject("persons")
                ?: root.optJSONObject("faces")
            if (map != null) {
                val arr = JSONArray()
                val keys = map.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    val person = map.optJSONObject(key) ?: JSONObject()
                    if (person.optString("name", "").isBlank()) {
                        person.put("name", key)
                    }
                    arr.put(person)
                }
                return JSONObject().put("people", arr)
            }
            return JSONObject().put("people", JSONArray())
        }

        private fun normalizeEmbeddings(arr: JSONArray): JSONArray = peopleToJsonEmbeddings(normalizeEmbeddingsList(arr))

        private fun normalizeEmbeddingsList(arr: JSONArray): MutableList<FloatArray> {
            val out = mutableListOf<FloatArray>()
            for (i in 0 until arr.length()) {
                val emb = arr.optJSONArray(i) ?: continue
                val vec = FloatArray(emb.length())
                for (j in 0 until emb.length()) {
                    vec[j] = emb.optDouble(j, 0.0).toFloat()
                }
                out += vec
            }
            return out
        }

        private fun encodeEmbeddings(records: List<Record>): EncodedData {
            val rows = mutableListOf<RowMeta>()
            val out = ByteArrayOutputStream()
            records.sortedBy { it.name }.forEach { record ->
                val offset = out.size()
                val count = record.embeddings.size
                val dim = record.embeddings.firstOrNull()?.size ?: 0
                record.embeddings.forEach { emb ->
                    for (j in 0 until dim) {
                        writeFloatLe(out, emb.getOrElse(j) { 0f })
                    }
                }
                val length = out.size() - offset
                rows.add(
                    RowMeta(
                        personId = record.personId,
                        personKey = record.personKey,
                        name = record.name,
                        department = record.department,
                        grade = record.grade,
                        offsetBytes = offset,
                        lengthBytes = length,
                        embeddingCount = count,
                        embeddingDim = dim,
                        hash = record.hash,
                        updatedAt = record.updatedAt,
                        metadataBlob = record.metadataBlob,
                        metadataHash = record.metadataHash,
                    ),
                )
            }
            return EncodedData(rows, out.toByteArray())
        }

        private fun decodeEmbeddings(bytes: ByteArray, offset: Int, length: Int, count: Int, dim: Int): JSONArray {
            return peopleToJsonEmbeddings(decodeEmbeddingsList(bytes, offset, length, count, dim))
        }

        private fun ensurePeopleTableSchema(db: SQLiteDatabase) {
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN grade TEXT DEFAULT ''") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_key TEXT") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_idx INTEGER") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN person_id INTEGER") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN metadata_blob BLOB") }
            runCatching { db.execSQL("ALTER TABLE $TABLE ADD COLUMN metadata_hash TEXT") }
        }

        private fun recreatePeopleTable(db: SQLiteDatabase) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE")
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS $TABLE(" +
                    "person_key TEXT PRIMARY KEY,person_idx INTEGER," +
                    "name TEXT," +
                    "department TEXT," +
                    "grade TEXT," +
                    "offset_bytes INTEGER," +
                    "length_bytes INTEGER," +
                    "embedding_count INTEGER," +
                    "embedding_dim INTEGER," +
                    "hash TEXT," +
                    "updated_at INTEGER," +
                    "metadata_blob BLOB," +
                    "metadata_hash TEXT" +
                    ")",
            )
        }

        private fun hasColumn(db: SQLiteDatabase, table: String, column: String): Boolean {
            return runCatching {
                db.rawQuery("PRAGMA table_info($table)", null).use { cursor ->
                    val nameIndex = cursor.getColumnIndex("name")
                    while (cursor.moveToNext()) {
                        if (nameIndex >= 0 && cursor.getString(nameIndex).equals(column, ignoreCase = true)) {
                            return@use true
                        }
                    }
                    false
                }
            }.getOrDefault(false)
        }

        private fun writeFloatLe(out: ByteArrayOutputStream, value: Float) {
            val bits = java.lang.Float.floatToIntBits(value)
            out.write(bits and 0xFF)
            out.write((bits ushr 8) and 0xFF)
            out.write((bits ushr 16) and 0xFF)
            out.write((bits ushr 24) and 0xFF)
        }

        private fun computeHash(name: String, department: String, grade: String, embeddings: List<FloatArray>): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(grade.toByteArray(Charsets.UTF_8))
            digest.update(0)
            updateDigestInt(digest, embeddings.size)
            embeddings.forEach { emb ->
                updateDigestInt(digest, emb.size)
                emb.forEach { value ->
                    updateDigestInt(digest, java.lang.Float.floatToIntBits(value))
                }
            }
            return digest.digest().joinToString("") { b -> "%02x".format(b) }
        }

        private fun saveRecordsToStructured(records: List<Record>, metaDbFile: File, embeddingsBinFile: File): Boolean {
            val encoded = encodeEmbeddings(records)
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                embeddingsBinFile.writeBytes(encoded.bytes)
                val db = SQLiteDatabase.openOrCreateDatabase(metaDbFile, null)
                recreatePeopleTable(db)
                db.beginTransaction()
                try {
                    db.delete(TABLE, null, null)
                    encoded.rows.forEach { row ->
                        val values = ContentValues().apply {
                            put("person_key", row.personKey)
                            put("person_idx", row.personId)
                            put("name", row.name)
                            put("department", row.department)
                            put("grade", row.grade)
                            put("offset_bytes", row.offsetBytes)
                            put("length_bytes", row.lengthBytes)
                            put("embedding_count", row.embeddingCount)
                            put("embedding_dim", row.embeddingDim)
                            put("hash", row.hash)
                            put("updated_at", row.updatedAt)
                            put("metadata_blob", row.metadataBlob)
                            put("metadata_hash", row.metadataHash)
                        }
                        db.insert(TABLE, null, values)
                    }
                    db.setTransactionSuccessful()
                } finally {
                    db.endTransaction()
                    db.close()
                }
                true
            }.getOrDefault(false)
        }

        private fun decodeEmbeddingsList(bytes: ByteArray, offset: Int, length: Int, count: Int, dim: Int): MutableList<FloatArray> {
            if (offset < 0 || length < 0 || offset + length > bytes.size || count <= 0 || dim <= 0) return mutableListOf()
            val out = mutableListOf<FloatArray>()
            var p = offset
            val limit = offset + length
            for (i in 0 until count) {
                if (p + dim * 4 > limit) break
                val emb = FloatArray(dim)
                for (j in 0 until dim) {
                    val bits =
                        (bytes[p].toInt() and 0xFF) or
                            ((bytes[p + 1].toInt() and 0xFF) shl 8) or
                            ((bytes[p + 2].toInt() and 0xFF) shl 16) or
                            ((bytes[p + 3].toInt() and 0xFF) shl 24)
                    p += 4
                    emb[j] = Float.fromBits(bits)
                }
                out += emb
            }
            return out
        }

        private fun jsonRootToPeople(root: JSONObject): MutableList<ServerPersonRecord> {
            val normalized = normalizeRoot(root)
            val people = normalized.optJSONArray("people") ?: JSONArray()
            val out = mutableListOf<ServerPersonRecord>()
            for (i in 0 until people.length()) {
                val person = people.optJSONObject(i) ?: continue
                val name = person.optString("name", "").trim()
                if (name.isBlank()) continue
                val department = person.optString("department", person.optString("folder", "")).trim()
                val grade = person.optString("grade", "").trim()
                val embeddings = normalizeEmbeddingsList(person.optJSONArray("embeddings") ?: JSONArray())
                if (embeddings.isEmpty()) continue
                val updatedAt = person.optLong("updatedAt", System.currentTimeMillis())
                val hash = person.optString("hash", "").trim().ifBlank { computeHash(name, department, grade, embeddings) }
                val metadataJson = person.optJSONObject("metadata")
                val metadata = linkedMapOf<String, String>()
                if (metadataJson != null) {
                    val keys = metadataJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        metadata[key] = metadataJson.optString(key, "")
                    }
                }
                val normalizedMetadata = normalizeMetadataMap(metadata, name, department, grade)
                out += ServerPersonRecord(
                    name = name,
                    department = department,
                    grade = grade,
                    embeddings = embeddings,
                    hash = hash,
                    updatedAt = updatedAt,
                    personId = parsePersonId(person.optString("personIdx", person.optString("personId", "")), name, department, grade),
                    personKey = parsePersonKey(person.optString("personKey", ""), name, department, grade),
                    metadata = normalizedMetadata,
                    metadataHash = metadataHash(normalizedMetadata),
                )
            }
            return out
        }

        private fun peopleToJsonRoot(people: List<ServerPersonRecord>, includeEmbeddings: Boolean): JSONObject {
            val arr = JSONArray()
            people.sortedBy { it.name }.forEach { person ->
                val obj = JSONObject()
                    .put("name", person.name)
                    .put("department", person.department)
                    .put("grade", person.grade)
                    .put("personIdx", person.personId)
                    .put("personKey", person.personKey)
                    .put("metadata", JSONObject(person.metadata as Map<*, *>))
                    .put("hash", person.hash)
                    .put("updatedAt", person.updatedAt)
                if (includeEmbeddings) {
                    obj.put("embeddings", peopleToJsonEmbeddings(person.embeddings))
                } else {
                    val stubs = JSONArray()
                    repeat(person.embeddings.size) { stubs.put(JSONArray()) }
                    obj.put("embeddings", stubs)
                }
                arr.put(obj)
            }
            return JSONObject().put("people", arr)
        }

        private fun peopleToJsonEmbeddings(embeddings: List<FloatArray>): JSONArray {
            val out = JSONArray()
            embeddings.forEach { emb ->
                val row = JSONArray()
                emb.forEach { row.put(it.toDouble()) }
                out.put(row)
            }
            return out
        }

        private fun updateDigestInt(digest: MessageDigest, value: Int) {
            digest.update(((value ushr 24) and 0xFF).toByte())
            digest.update(((value ushr 16) and 0xFF).toByte())
            digest.update(((value ushr 8) and 0xFF).toByte())
            digest.update((value and 0xFF).toByte())
        }

        private data class Record(
            val personId: Long,
            val personKey: String,
            val name: String,
            val department: String,
            val grade: String,
            val embeddings: List<FloatArray>,
            val hash: String,
            val updatedAt: Long,
            val metadataBlob: ByteArray,
            val metadataHash: String,
        )

        private data class RowMeta(
            val personId: Long,
            val personKey: String,
            val name: String,
            val department: String,
            val grade: String,
            val offsetBytes: Int,
            val lengthBytes: Int,
            val embeddingCount: Int,
            val embeddingDim: Int,
            val hash: String,
            val updatedAt: Long,
            val metadataBlob: ByteArray,
            val metadataHash: String,
        )

        private data class EncodedData(
            val rows: List<RowMeta>,
            val bytes: ByteArray,
        )
    }

    private companion object {
        const val LOGCAT_TAG = "FaceDbServer"
    }

    private data class AttendanceBaseRow(
        val name: String,
        val affiliation: String,
        val department: String,
    )
}
