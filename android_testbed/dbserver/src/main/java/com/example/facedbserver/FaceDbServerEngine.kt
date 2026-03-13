package com.example.facedbserver

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondText
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
import java.io.File
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
    private var dbStorageBackend: DbStorageBackend = FileDbStorageBackend(File(serverDataDir, "face_db.json"))

    @Volatile
    private var currentDb: JSONObject = JSONObject().put("people", JSONArray())

    fun setDbStorageDir(spec: String) {
        dbStorageDirSpec = spec.trim()
    }

    fun start(serverPort: Int): Boolean {
        if (server != null) {
            log("이미 서버가 실행 중입니다.")
            return false
        }
        port = serverPort
        dbStorageBackend = resolveDbStorageBackend()
        loadDbFromDisk()
        loadAuthStateFromDisk()

        val engine = embeddedServer(CIO, port = port) {
            install(WebSockets)
            routing {
                get("/api/health") {
                    call.respondText("ok", ContentType.Text.Plain)
                }

                post("/api/auth/request") {
                    val req = parseRequestBody(call.receiveText())
                    val clientId = req.optString("clientId", "").trim()
                    val deviceName = req.optString("deviceName", "unknown").trim()
                    if (clientId.isBlank()) {
                        call.respondText(
                            "{\"error\":\"missing_client_id\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.BadRequest,
                        )
                        return@post
                    }

                    val token = approvedTokensByClient[clientId]
                    if (!token.isNullOrBlank()) {
                        call.respondText(
                            JSONObject()
                                .put("status", "approved")
                                .put("clientId", clientId)
                                .put("token", token)
                                .toString(),
                            ContentType.Application.Json,
                        )
                        return@post
                    }

                    pendingClients[clientId] = deviceName
                    log("승인 요청 수신: clientId=$clientId, device=$deviceName")
                    call.respondText(
                        JSONObject()
                            .put("status", "pending")
                            .put("clientId", clientId)
                            .toString(),
                        ContentType.Application.Json,
                    )
                }

                get("/api/auth/status") {
                    val clientId = call.request.queryParameters["clientId"].orEmpty().trim()
                    if (clientId.isBlank()) {
                        call.respondText(
                            "{\"error\":\"missing_client_id\"}",
                            ContentType.Application.Json,
                            HttpStatusCode.BadRequest,
                        )
                        return@get
                    }

                    val token = approvedTokensByClient[clientId]
                    val status = when {
                        !token.isNullOrBlank() -> "approved"
                        pendingClients.containsKey(clientId) -> "pending"
                        else -> "unknown"
                    }
                    val out = JSONObject().put("status", status).put("clientId", clientId)
                    if (!token.isNullOrBlank()) out.put("token", token)
                    call.respondText(out.toString(), ContentType.Application.Json)
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

                    val req = parseRequestBody(call.receiveText())
                    val summary = req.optJSONObject("summary") ?: JSONObject().put("people", JSONArray())
                    val clientHashes = parseClientHashes(summary)
                    val people = extractPeopleArray(currentDb)

                    val serverPeople = mutableListOf<Pair<String, JSONObject>>()
                    val serverHashByName = mutableMapOf<String, String>()
                    for (i in 0 until people.length()) {
                        val person = people.optJSONObject(i) ?: continue
                        val name = person.optString("name", "").trim()
                        if (name.isBlank()) continue
                        val normalized = JSONObject(person.toString())
                        normalized.put("name", name)
                        serverPeople.add(name to normalized)
                        serverHashByName[name] = personHash(normalized)
                    }

                    val upserts = JSONArray()
                    serverPeople.sortedBy { it.first }.forEach { (name, obj) ->
                        val serverHash = serverHashByName[name]
                        val clientHash = clientHashes[name]
                        if (serverHash != clientHash) {
                            upserts.put(obj)
                        }
                    }

                    val deletes = JSONArray()
                    clientHashes.keys.sorted().forEach { name ->
                        if (!serverHashByName.containsKey(name)) {
                            deletes.put(name)
                        }
                    }

                    val hasChanges = upserts.length() > 0 || deletes.length() > 0
                    val payload = JSONObject()
                        .put("revision", revision.get())
                        .put("hasChanges", hasChanges)
                        .put("upserts", upserts)
                        .put("deletes", deletes)
                        .put("serverPeopleCount", serverPeople.size)

                    call.respondText(payload.toString(), ContentType.Application.Json)
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

                    currentDb = db
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
                            JSONObject()
                                .put("type", "db_update")
                                .put("revision", revision.get())
                                .put("sourceClientId", "server_bootstrap")
                                .toString(),
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
            log("서버 시작: 0.0.0.0:$port")
            true
        }.getOrElse {
            log("서버 시작 실패: ${it.javaClass.simpleName}")
            false
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

    fun dbRowsSnapshot(): List<DbPersonRow> {
        val root = runCatching { JSONObject(currentDb.toString()) }.getOrNull() ?: return emptyList()
        val arr = extractPeopleArray(root)
        val rows = mutableListOf<DbPersonRow>()
        for (i in 0 until arr.length()) {
            val person = arr.optJSONObject(i) ?: continue
            val name = person.optString("name", "").trim()
            if (name.isBlank()) continue
            val department = person.optString("department", person.optString("folder", "")).trim()
            val embeddingCount = extractEmbeddingCount(person)
            rows.add(
                DbPersonRow(
                    personIndex = i,
                    name = name,
                    department = department,
                    embeddingCount = embeddingCount,
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
        saveDbToDisk()
        val nextRevision = revision.incrementAndGet()
        if (isRunning()) {
            broadcastDbUpdate(nextRevision, "server_ui_edit")
        }
        log("DB 수정 완료: index=$personIndex, name=$nextName, rev=$nextRevision")
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
        val msg = JSONObject()
            .put("type", "db_update")
            .put("revision", nextRevision)
            .put("sourceClientId", sourceClientId)
            .toString()

        sessions.values.forEach { session ->
            scope.launch {
                runCatching { session.send(Frame.Text(msg)) }
            }
        }
    }

    private fun loadDbFromDisk() {
        val raw = dbStorageBackend.readText()
        if (raw.isNullOrBlank()) {
            currentDb = JSONObject().put("people", JSONArray())
            revision.set(0)
            log("DB 초기화: ${dbStorageBackend.description()}")
            return
        }
        val loaded = runCatching { JSONObject(raw) }.getOrNull()
        currentDb = loaded ?: JSONObject().put("people", JSONArray())
        revision.set(1)
        log("DB 로드 완료: ${dbStorageBackend.description()}")
    }

    private fun saveDbToDisk() {
        val ok = dbStorageBackend.writeText(currentDb.toString(2))
        if (!ok) {
            log("DB 저장 실패: ${dbStorageBackend.description()}")
        }
    }

    private fun resolveDbStorageBackend(): DbStorageBackend {
        val spec = dbStorageDirSpec.trim()
        if (spec.startsWith("content://")) {
            val uri = runCatching { Uri.parse(spec) }.getOrNull()
            if (uri != null) {
                return TreeUriDbStorageBackend(context, uri, "face_db.json")
            }
        }
        if (spec.isNotBlank()) {
            val f = File(spec)
            val dir = if (f.isAbsolute) f else File(context.filesDir, spec)
            return FileDbStorageBackend(File(dir, "face_db.json"))
        }
        return FileDbStorageBackend(File(serverDataDir, "face_db.json"))
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

    private fun log(message: String) {
        val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        onLog("[$ts] $message")
    }

    fun release() {
        stop()
        scope.cancel()
    }

    private interface DbStorageBackend {
        fun readText(): String?
        fun writeText(raw: String): Boolean
        fun description(): String
    }

    private class FileDbStorageBackend(private val dbFile: File) : DbStorageBackend {
        override fun readText(): String? {
            if (!dbFile.exists()) return null
            return runCatching { dbFile.readText() }.getOrNull()
        }

        override fun writeText(raw: String): Boolean {
            return runCatching {
                dbFile.parentFile?.mkdirs()
                dbFile.writeText(raw)
                true
            }.getOrDefault(false)
        }

        override fun description(): String {
            return dbFile.absolutePath
        }
    }

    private class TreeUriDbStorageBackend(
        private val context: Context,
        private val treeUri: Uri,
        private val fileName: String,
    ) : DbStorageBackend {
        override fun readText(): String? {
            val file = resolveDocumentFile(createIfMissing = false) ?: return null
            return runCatching {
                context.contentResolver.openInputStream(file.uri)?.bufferedReader()?.use { it.readText() }
            }.getOrNull()
        }

        override fun writeText(raw: String): Boolean {
            val file = resolveDocumentFile(createIfMissing = true) ?: return false
            return runCatching {
                context.contentResolver.openOutputStream(file.uri, "wt")?.bufferedWriter()?.use { it.write(raw) } != null
            }.getOrDefault(false)
        }

        override fun description(): String {
            return treeUri.toString()
        }

        private fun resolveDocumentFile(createIfMissing: Boolean): DocumentFile? {
            val root = DocumentFile.fromTreeUri(context, treeUri) ?: return null
            val existing = root.findFile(fileName)
            if (existing != null) return existing
            if (!createIfMissing) return null
            return root.createFile("application/json", fileName)
        }
    }
}
