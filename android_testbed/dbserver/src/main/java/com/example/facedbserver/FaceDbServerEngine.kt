package com.example.facedbserver

import android.content.Context
import android.net.Uri
import android.database.sqlite.SQLiteDatabase
import android.content.ContentValues
import android.util.Log
import androidx.documentfile.provider.DocumentFile
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
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

data class DbPersonRow(
    val personIndex: Int,
    val name: String,
    val department: String,
    val embeddingCount: Int,
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
    private var currentDb: JSONObject = JSONObject().put("people", JSONArray())
    @Volatile
    private var dbLoadedOnce: Boolean = false

    fun setDbStorageDir(spec: String) {
        val normalized = spec.trim()
        if (dbStorageDirSpec != normalized) {
            dbStorageDirSpec = normalized
            dbLoadedOnce = false
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
                    val payload = JSONObject().apply {
                        put("revision", revision.get())
                        put("db", currentDb)
                    }
                    call.respondText(payload.toString(), ContentType.Application.Json)
                }

                get("/api/db/structured/meta") {
                    val authClient = authorizedClientId(call.request.headers["Authorization"], call.request.queryParameters["token"])
                    if (authClient == null) {
                        call.respondText("error=unauthorized", ContentType.Text.Plain, HttpStatusCode.Unauthorized)
                        return@get
                    }
                    dbStorageBackend.readDb()
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
                    dbStorageBackend.readDb()
                    val binBytes = dbStorageBackend.readStructuredEmbeddingsBytes()
                    if (binBytes == null) {
                        call.respondText("error=structured_bin_missing", ContentType.Text.Plain, HttpStatusCode.NotFound)
                        return@get
                    }
                    log("구조화 bin 다운로드: client=$authClient, bytes=${binBytes.size}")
                    call.respondBytes(binBytes, ContentType.Application.OctetStream)
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
                    broadcastDbUpdate(nextRevision, authClient)
                    log(
                        "구조화 DB 업로드 반영: client=$authClient, revision=$nextRevision, " +
                            "meta=${bundle.first.size}B, bin=${bundle.second.size}B",
                    )
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

                    val diffStartedAt = System.currentTimeMillis()
                    val req = parseRequestBody(call.receiveText())
                    val summary = req.optJSONObject("summary") ?: JSONObject().put("people", JSONArray())
                    val clientHashes = parseClientHashes(summary)
                    val people = extractPeopleArray(currentDb)

                    val serverHashByName = mutableMapOf<String, String>()
                    val upserts = JSONArray()
                    for (i in 0 until people.length()) {
                        val person = people.optJSONObject(i) ?: continue
                        val name = person.optString("name", "").trim()
                        if (name.isBlank()) continue
                        val cachedHash = person.optString("hash", "").trim()
                        val serverHash = if (cachedHash.isNotBlank()) cachedHash else personHash(person)
                        serverHashByName[name] = serverHash
                        val clientHash = clientHashes[name]
                        if (serverHash != clientHash) {
                            upserts.put(person)
                        }
                    }

                    val deletes = JSONArray()
                    clientHashes.keys.forEach { name ->
                        if (!serverHashByName.containsKey(name)) {
                            deletes.put(name)
                        }
                    }

                    val hasChanges = upserts.length() > 0 || deletes.length() > 0
                    val currentRevision = revision.get()
                    val diffElapsedMs = System.currentTimeMillis() - diffStartedAt
                    log(
                        "DB diff 응답: client=$authClient, clientPeople=${clientHashes.size}, " +
                            "serverPeople=${serverHashByName.size}, upserts=${upserts.length()}, " +
                            "deletes=${deletes.length()}, total=${diffElapsedMs}ms",
                    )
                    call.respondTextWriter(ContentType.Application.Json) {
                        write("{")
                        write("\"revision\":")
                        write(currentRevision.toString())
                        write(",\"hasChanges\":")
                        write(if (hasChanges) "true" else "false")
                        write(",\"upserts\":[")
                        for (i in 0 until upserts.length()) {
                            if (i > 0) write(",")
                            write(upserts.optJSONObject(i)?.toString() ?: "null")
                        }
                        write("],\"deletes\":[")
                        for (i in 0 until deletes.length()) {
                            if (i > 0) write(",")
                            write(JSONObject.quote(deletes.optString(i, "")))
                        }
                        write("],\"serverPeopleCount\":")
                        write(serverHashByName.size.toString())
                        write("}")
                    }
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

                    val req = parseRequestBody(call.receiveText())
                    val db = req.optJSONObject("db")
                    if (db == null) {
                        call.respondText(
                            "{\"error\":\"missing_db\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.BadRequest,
                        )
                        return@post
                    }

                    currentDb = normalizeDbRoot(db)
                    normalizePeopleMetadata(currentDb, preserveUpdatedAt = false)
                    saveDbToDisk()
                    val nextRevision = revision.incrementAndGet()
                    broadcastDbUpdate(nextRevision, authClient)
                    log("DB 업데이트 수신: source=$authClient, rev=$nextRevision")

                    call.respondText(
                        JSONObject().put("revision", nextRevision).toString(),
                        ContentType.Application.Json,
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

    @Synchronized
    fun dbRowsSnapshot(): List<DbPersonRow> {
        val arr = extractPeopleArray(currentDb)
        if (arr.length() <= 0) return emptyList()
        val rows = ArrayList<DbPersonRow>(arr.length())
        for (i in 0 until arr.length()) {
            val person = arr.optJSONObject(i) ?: continue
            val name = person.optString("name", "").trim()
            if (name.isBlank()) continue
            rows.add(
                DbPersonRow(
                    personIndex = i,
                    name = name,
                    department = person.optString("department", person.optString("folder", "")).trim(),
                    embeddingCount = extractEmbeddingCount(person),
                ),
            )
        }
        return rows.sortedBy { it.name }
    }

    private fun extractPeopleArray(root: JSONObject): JSONArray {
        root.optJSONArray("people")?.let { return it }
        root.optJSONArray("persons")?.let { return it }
        root.optJSONArray("faces")?.let { return it }

        val map = root.optJSONObject("people")
            ?: root.optJSONObject("persons")
            ?: root.optJSONObject("faces")
        if (map != null) {
            val arr = JSONArray()
            val keys = map.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val item = map.optJSONObject(key) ?: JSONObject()
                if (item.optString("name", "").isBlank()) {
                    item.put("name", key)
                }
                arr.put(item)
            }
            return arr
        }
        return JSONArray()
    }

    private fun extractEmbeddingCount(person: JSONObject): Int {
        person.optJSONArray("embeddings")?.let { return it.length() }
        person.optInt("num_samples", -1).takeIf { it >= 0 }?.let { return it }
        person.optJSONArray("samples")?.let { return it.length() }
        val byDirection = person.optJSONObject("embeddings_by_direction")
        if (byDirection != null) {
            var count = 0
            val keys = byDirection.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                count += byDirection.optJSONArray(key)?.length() ?: 0
            }
            return count
        }
        return 0
    }

    private fun parseClientHashes(summary: JSONObject): Map<String, String> {
        val result = mutableMapOf<String, String>()
        val arr = summary.optJSONArray("people") ?: JSONArray()
        for (i in 0 until arr.length()) {
            val item = arr.optJSONObject(i) ?: continue
            val name = item.optString("name", "").trim()
            val hash = item.optString("hash", "").trim()
            if (name.isNotBlank() && hash.isNotBlank()) {
                result[name] = hash
            }
        }
        return result
    }

    private fun personHash(person: JSONObject): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val name = person.optString("name", "").trim()
        val department = person.optString("department", person.optString("folder", "")).trim()
        val embeddings = person.optJSONArray("embeddings") ?: JSONArray()

        digest.update(name.toByteArray(Charsets.UTF_8))
        digest.update(0)
        digest.update(department.toByteArray(Charsets.UTF_8))
        digest.update(0)
        updateDigestInt(digest, embeddings.length())

        for (i in 0 until embeddings.length()) {
            val emb = embeddings.optJSONArray(i) ?: JSONArray()
            updateDigestInt(digest, emb.length())
            for (j in 0 until emb.length()) {
                val v = emb.optDouble(j, 0.0).toFloat()
                updateDigestInt(digest, java.lang.Float.floatToIntBits(v))
            }
        }

        return digest.digest().joinToString("") { b -> "%02x".format(b) }
    }

    private fun updateDigestInt(digest: MessageDigest, value: Int) {
        digest.update(((value ushr 24) and 0xFF).toByte())
        digest.update(((value ushr 16) and 0xFF).toByte())
        digest.update(((value ushr 8) and 0xFF).toByte())
        digest.update((value and 0xFF).toByte())
    }

    @Synchronized
    fun updatePersonRow(personIndex: Int, name: String, department: String): Boolean {
        val nextName = name.trim()
        if (nextName.isBlank()) {
            log("DB 수정 실패: 이름이 비어 있습니다.")
            return false
        }
        val people = currentDb.optJSONArray("people") ?: return false
        if (personIndex !in 0 until people.length()) return false
        val person = people.optJSONObject(personIndex) ?: return false
        person.put("name", nextName)
        person.put("department", department.trim())
        person.put("updatedAt", System.currentTimeMillis())
        person.put("hash", personHash(person))
        saveDbToDisk()
        val nextRevision = revision.incrementAndGet()
        if (isRunning()) {
            broadcastDbUpdate(nextRevision, "server_ui_edit")
        }
        log("DB 수정 완료: index=$personIndex, name=$nextName, rev=$nextRevision")
        return true
    }

    @Synchronized
    fun deletePersonRow(personIndex: Int): Boolean {
        val people = currentDb.optJSONArray("people") ?: return false
        if (personIndex !in 0 until people.length()) return false
        val deletedName = people.optJSONObject(personIndex)?.optString("name", "")?.trim().orEmpty()
        val nextPeople = JSONArray()
        for (i in 0 until people.length()) {
            if (i == personIndex) continue
            nextPeople.put(people.opt(i))
        }
        currentDb.put("people", nextPeople)
        saveDbToDisk()
        val nextRevision = revision.incrementAndGet()
        if (isRunning()) {
            broadcastDbUpdate(nextRevision, "server_ui_delete")
        }
        log("DB 삭제 완료: index=$personIndex, name=${deletedName.ifBlank { "-" }}, rev=$nextRevision")
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
        val loaded = dbStorageBackend.readDb()
        val readElapsedMs = System.currentTimeMillis() - readStartedAt
        if (loaded == null) {
            currentDb = JSONObject().put("people", JSONArray())
            revision.set(0)
            val totalElapsedMs = System.currentTimeMillis() - startedAt
            log("DB 초기화: ${dbStorageBackend.description()} (read=${readElapsedMs}ms, total=${totalElapsedMs}ms)")
            return
        }
        val normalizeStartedAt = System.currentTimeMillis()
        val normalized = normalizeDbRoot(loaded)
        normalizePeopleMetadata(normalized, preserveUpdatedAt = true)
        val normalizeElapsedMs = System.currentTimeMillis() - normalizeStartedAt
        currentDb = normalized
        revision.set(1)
        val totalElapsedMs = System.currentTimeMillis() - startedAt
        log(
            "DB 로드 완료: ${dbStorageBackend.description()} " +
                "(read=${readElapsedMs}ms, normalize=${normalizeElapsedMs}ms, total=${totalElapsedMs}ms)",
        )
    }

    private fun saveDbToDisk() {
        val ok = dbStorageBackend.writeDb(currentDb)
        if (!ok) {
            log("DB 저장 실패: ${dbStorageBackend.description()}")
        }
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

    private fun normalizeDbRoot(root: JSONObject): JSONObject {
        root.optJSONArray("people")?.let { return root }
        root.optJSONArray("persons")?.let { return JSONObject().put("people", it) }
        root.optJSONArray("faces")?.let { return JSONObject().put("people", it) }

        val arr = extractPeopleArray(root)
        return JSONObject().put("people", arr)
    }

    private fun normalizePeopleMetadata(root: JSONObject, preserveUpdatedAt: Boolean) {
        val people = root.optJSONArray("people") ?: return
        val now = System.currentTimeMillis()
        for (i in 0 until people.length()) {
            val person = people.optJSONObject(i) ?: continue
            val name = person.optString("name", "").trim()
            if (name.isBlank()) continue
            val department = person.optString("department", person.optString("folder", "")).trim()
            val embeddings = person.optJSONArray("embeddings") ?: JSONArray()
            person.put("name", name)
            person.put("department", department)
            person.put("embeddings", embeddings)
            if (!preserveUpdatedAt || person.optLong("updatedAt", 0L) <= 0L) {
                person.put("updatedAt", now)
            }
            val existingHash = person.optString("hash", "").trim()
            if (existingHash.isBlank()) {
                person.put("hash", personHash(person))
            } else {
                person.put("hash", existingHash)
            }
        }
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
        fun readDb(): JSONObject?
        fun writeDb(root: JSONObject): Boolean
        fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean
        fun description(): String
        fun hasStructuredDb(): Boolean
        fun structuredDbDescription(): String
        fun readStructuredMetaBytes(): ByteArray?
        fun readStructuredEmbeddingsBytes(): ByteArray?
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

        override fun readDb(): JSONObject? {
            val structuredExists = hasStructuredDb()
            val csvNeedsImport = csvFile.exists() &&
                (
                    !structuredExists ||
                        csvFile.lastModified() > maxOf(metaDbFile.lastModified(), embeddingsBinFile.lastModified())
                    )

            if (csvNeedsImport) {
                val raw = runCatching { csvFile.readText() }.getOrNull()
                if (raw != null) {
                    if (structuredExists && !csvCodec.hasEmbeddingsColumn(raw)) {
                        StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                    val root =
                        if (structuredExists) {
                            val base = StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredDbCodec.mergeWithCsvMetadata(base, raw, csvCodec) else null
                        } else {
                            StructuredDbCodec.parseCsvDb(raw, csvCodec)
                        }
                    if (root != null && StructuredDbCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) {
                        return root
                    }
                }
            }

            if (structuredExists) {
                StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)?.let { root ->
                    if (!csvFile.exists()) {
                        writeCsvSnapshot(root)
                    }
                    return root
                }
            }
            return null
        }

        override fun writeDb(root: JSONObject): Boolean {
            if (!StructuredDbCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) return false
            return writeCsvSnapshot(root)
        }

        override fun writeStructured(metaDbBytes: ByteArray, embeddingsBinBytes: ByteArray): Boolean {
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                metaDbFile.writeBytes(metaDbBytes)
                embeddingsBinFile.writeBytes(embeddingsBinBytes)
                val meta = StructuredDbCodec.loadMetadataOnly(metaDbFile) ?: return@runCatching false
                writeCsvSnapshot(meta)
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

        private fun writeCsvSnapshot(root: JSONObject): Boolean {
            return runCatching {
                csvFile.parentFile?.mkdirs()
                csvFile.writeText(csvCodec.encode(root))
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

        override fun readDb(): JSONObject? {
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
                    if (structuredExists && !csvCodec.hasEmbeddingsColumn(csvRaw)) {
                        StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)?.let { return it }
                    }
                    val root =
                        if (structuredExists) {
                            val base = StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)
                            if (base != null) StructuredDbCodec.mergeWithCsvMetadata(base, csvRaw, csvCodec) else null
                        } else {
                            StructuredDbCodec.parseCsvDb(csvRaw, csvCodec)
                        }
                    if (root != null && StructuredDbCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) {
                        syncStructuredToDocuments()
                        return root
                    }
                }
            }

            if (structuredExists) {
                StructuredDbCodec.loadFromStructured(metaDbFile, embeddingsBinFile)?.let { root ->
                    if (csvDoc == null) {
                        writeDocumentText(csvFileName, csvCodec.encode(root))
                    }
                    return root
                }
            }
            return null
        }

        override fun writeDb(root: JSONObject): Boolean {
            if (!StructuredDbCodec.saveToStructured(root, metaDbFile, embeddingsBinFile)) return false
            if (!syncStructuredToDocuments()) return false
            return writeDocumentText(csvFileName, csvCodec.encode(root))
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
            val meta = StructuredDbCodec.loadMetadataOnly(metaDbFile) ?: return false
            syncStructuredToDocuments()
            return writeDocumentText(csvFileName, csvCodec.encode(meta))
        }

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
                obj.put("updatedAt", row.updatedAt)
                val embeddings = normalizeEmbeddings(obj.optJSONArray("embeddings") ?: JSONArray())
                obj.put("embeddings", embeddings)
                obj.put("hash", computeHash(row.name, row.department, embeddings))
                out.put(obj)
            }
            return JSONObject().put("people", out)
        }

        fun loadMetadataOnly(metaDbFile: File): JSONObject? {
            if (!metaDbFile.exists()) return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val arr = JSONArray()
                db.query(
                    TABLE,
                    arrayOf("name", "department", "embedding_count", "updated_at"),
                    null, null, null, null, "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val name = cursor.getString(0)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(1)?.trim().orEmpty()
                        val count = cursor.getInt(2).coerceAtLeast(0)
                        val updatedAt = cursor.getLong(3)
                        val stubs = JSONArray()
                        repeat(count) { stubs.put(JSONArray()) }
                        val obj = JSONObject()
                            .put("name", name)
                            .put("department", department)
                            .put("embeddings", stubs)
                        if (updatedAt > 0L) obj.put("updatedAt", updatedAt)
                        arr.put(obj)
                    }
                }
                JSONObject().put("people", arr)
            }.getOrNull().also { db.close() }
        }

        fun loadFromStructured(metaDbFile: File, embeddingsBinFile: File): JSONObject? {
            if (!metaDbFile.exists() || !embeddingsBinFile.exists()) return null
            val embeddingsBytes = runCatching { embeddingsBinFile.readBytes() }.getOrNull() ?: return null
            val db = runCatching { SQLiteDatabase.openDatabase(metaDbFile.absolutePath, null, SQLiteDatabase.OPEN_READONLY) }
                .getOrNull() ?: return null
            return runCatching {
                val arr = JSONArray()
                db.query(
                    TABLE,
                    arrayOf("name", "department", "offset_bytes", "length_bytes", "embedding_count", "embedding_dim", "hash", "updated_at"),
                    null,
                    null,
                    null,
                    null,
                    "name ASC",
                ).use { cursor ->
                    while (cursor.moveToNext()) {
                        val name = cursor.getString(0)?.trim().orEmpty()
                        if (name.isBlank()) continue
                        val department = cursor.getString(1)?.trim().orEmpty()
                        val offset = cursor.getLong(2).toInt()
                        val length = cursor.getLong(3).toInt()
                        val count = cursor.getInt(4).coerceAtLeast(0)
                        val dim = cursor.getInt(5).coerceAtLeast(0)
                        val hash = cursor.getString(6)?.trim().orEmpty()
                        val updatedAt = cursor.getLong(7)
                        val embeddings = decodeEmbeddings(embeddingsBytes, offset, length, count, dim)
                        val obj = JSONObject()
                            .put("name", name)
                            .put("department", department)
                            .put("embeddings", embeddings)
                        if (updatedAt > 0L) obj.put("updatedAt", updatedAt)
                        obj.put("hash", if (hash.isNotBlank()) hash else computeHash(name, department, embeddings))
                        arr.put(obj)
                    }
                }
                JSONObject().put("people", arr)
            }.getOrNull().also { db.close() }
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
                val embeddings = normalizeEmbeddings(person.optJSONArray("embeddings") ?: JSONArray())
                if (embeddings.length() <= 0) continue
                val updatedAt = person.optLong("updatedAt", System.currentTimeMillis())
                val hash = person.optString("hash", "").trim().ifBlank { computeHash(name, department, embeddings) }
                records.add(Record(name, department, embeddings, hash, updatedAt))
            }

            val encoded = encodeEmbeddings(records)
            return runCatching {
                metaDbFile.parentFile?.mkdirs()
                embeddingsBinFile.parentFile?.mkdirs()
                embeddingsBinFile.writeBytes(encoded.bytes)
                val db = SQLiteDatabase.openOrCreateDatabase(metaDbFile, null)
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS $TABLE(" +
                        "name TEXT PRIMARY KEY," +
                        "department TEXT," +
                        "offset_bytes INTEGER," +
                        "length_bytes INTEGER," +
                        "embedding_count INTEGER," +
                        "embedding_dim INTEGER," +
                        "hash TEXT," +
                        "updated_at INTEGER" +
                        ")",
                )
                db.beginTransaction()
                try {
                    db.delete(TABLE, null, null)
                    encoded.rows.forEach { row ->
                        val values = ContentValues().apply {
                            put("name", row.name)
                            put("department", row.department)
                            put("offset_bytes", row.offsetBytes)
                            put("length_bytes", row.lengthBytes)
                            put("embedding_count", row.embeddingCount)
                            put("embedding_dim", row.embeddingDim)
                            put("hash", row.hash)
                            put("updated_at", row.updatedAt)
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

        private fun normalizeRoot(root: JSONObject): JSONObject {
            root.optJSONArray("people")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
            root.optJSONArray("persons")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
            root.optJSONArray("faces")?.let { return JSONObject().put("people", JSONArray(it.toString())) }
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

        private fun normalizeEmbeddings(arr: JSONArray): JSONArray {
            val out = JSONArray()
            for (i in 0 until arr.length()) {
                val emb = arr.optJSONArray(i) ?: continue
                val vec = JSONArray()
                for (j in 0 until emb.length()) {
                    vec.put(emb.optDouble(j, 0.0))
                }
                out.put(vec)
            }
            return out
        }

        private fun encodeEmbeddings(records: List<Record>): EncodedData {
            val rows = mutableListOf<RowMeta>()
            val out = ByteArrayOutputStream()
            records.sortedBy { it.name }.forEach { record ->
                val offset = out.size()
                val count = record.embeddings.length()
                val dim = if (count > 0) (record.embeddings.optJSONArray(0)?.length() ?: 0) else 0
                for (i in 0 until count) {
                    val emb = record.embeddings.optJSONArray(i) ?: JSONArray()
                    for (j in 0 until dim) {
                        val v = emb.optDouble(j, 0.0).toFloat()
                        writeFloatLe(out, v)
                    }
                }
                val length = out.size() - offset
                rows.add(RowMeta(record.name, record.department, offset, length, count, dim, record.hash, record.updatedAt))
            }
            return EncodedData(rows, out.toByteArray())
        }

        private fun decodeEmbeddings(bytes: ByteArray, offset: Int, length: Int, count: Int, dim: Int): JSONArray {
            if (offset < 0 || length < 0 || offset + length > bytes.size || count <= 0 || dim <= 0) return JSONArray()
            val arr = JSONArray()
            var p = offset
            val limit = offset + length
            for (i in 0 until count) {
                if (p + dim * 4 > limit) break
                val emb = JSONArray()
                for (j in 0 until dim) {
                    val bits =
                        (bytes[p].toInt() and 0xFF) or
                            ((bytes[p + 1].toInt() and 0xFF) shl 8) or
                            ((bytes[p + 2].toInt() and 0xFF) shl 16) or
                            ((bytes[p + 3].toInt() and 0xFF) shl 24)
                    p += 4
                    emb.put(Float.fromBits(bits).toDouble())
                }
                arr.put(emb)
            }
            return arr
        }

        private fun writeFloatLe(out: ByteArrayOutputStream, value: Float) {
            val bits = java.lang.Float.floatToIntBits(value)
            out.write(bits and 0xFF)
            out.write((bits ushr 8) and 0xFF)
            out.write((bits ushr 16) and 0xFF)
            out.write((bits ushr 24) and 0xFF)
        }

        private fun computeHash(name: String, department: String, embeddings: JSONArray): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(name.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(department.toByteArray(Charsets.UTF_8))
            digest.update(0)
            updateDigestInt(digest, embeddings.length())
            for (i in 0 until embeddings.length()) {
                val emb = embeddings.optJSONArray(i) ?: JSONArray()
                updateDigestInt(digest, emb.length())
                for (j in 0 until emb.length()) {
                    val v = emb.optDouble(j, 0.0).toFloat()
                    updateDigestInt(digest, java.lang.Float.floatToIntBits(v))
                }
            }
            return digest.digest().joinToString("") { b -> "%02x".format(b) }
        }

        private fun updateDigestInt(digest: MessageDigest, value: Int) {
            digest.update(((value ushr 24) and 0xFF).toByte())
            digest.update(((value ushr 16) and 0xFF).toByte())
            digest.update(((value ushr 8) and 0xFF).toByte())
            digest.update((value and 0xFF).toByte())
        }

        private data class Record(
            val name: String,
            val department: String,
            val embeddings: JSONArray,
            val hash: String,
            val updatedAt: Long,
        )

        private data class RowMeta(
            val name: String,
            val department: String,
            val offsetBytes: Int,
            val lengthBytes: Int,
            val embeddingCount: Int,
            val embeddingDim: Int,
            val hash: String,
            val updatedAt: Long,
        )

        private data class EncodedData(
            val rows: List<RowMeta>,
            val bytes: ByteArray,
        )
    }

    private companion object {
        const val LOGCAT_TAG = "FaceDbServer"
    }
}
