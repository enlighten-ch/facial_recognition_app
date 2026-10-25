package com.allstar.faithid.server

import com.allstar.faithid.server.auth.AdminCredentials
import java.io.File

/** Settings of the server app, stored as `config.yaml` in the app's private folder. */
data class ServerSettings(
    val credentials: AdminCredentials,
    val port: Int = DEFAULT_PORT,
    /** Folder with the face DB and attendance sheets, normally /sdcard/faceDB. */
    val dataDir: String = DEFAULT_DATA_DIR,
    /** Start the server when the app starts, without waiting for an administrator login. */
    val autoStart: Boolean = false,
    val maxEmbeddingsPerPerson: Int = 100,
) {
    companion object {
        const val DEFAULT_PORT = 18080
        const val DEFAULT_DATA_DIR = "/sdcard/faceDB"
    }
}

/**
 * Reads and writes `config.yaml` (flat `key: "value"` lines). Old files with `admin_password` in
 * plain text are converted to a hash on the first load.
 */
class ServerSettingsStore(private val file: File) {

    fun load(): ServerSettings {
        val values = parse()
        val credentials = when {
            values["admin_password_hash"].isNullOrBlank() || values["admin_password_salt"].isNullOrBlank() ->
                AdminCredentials.fromPlainText(values["admin_id"].orEmpty(), values["admin_password"] ?: AdminCredentials.DEFAULT_PASSWORD)
            else -> AdminCredentials(
                adminId = values["admin_id"].orEmpty().ifBlank { AdminCredentials.DEFAULT_ID },
                passwordHash = values.getValue("admin_password_hash"),
                salt = values.getValue("admin_password_salt"),
                mustChangePassword = values["must_change_password"] == "true",
            )
        }
        val settings = ServerSettings(
            credentials = credentials,
            port = values["default_port"]?.toIntOrNull()?.takeIf { it in 1..65535 } ?: ServerSettings.DEFAULT_PORT,
            dataDir = StoragePaths.normalize(values["db_storage_dir"].orEmpty()),
            autoStart = values["auto_start"] == "true",
            maxEmbeddingsPerPerson = values["max_embeddings_per_person"]?.toIntOrNull()?.coerceIn(10, 1000) ?: 100,
        )
        if (!file.exists() || values.containsKey("admin_password")) save(settings)
        return settings
    }

    fun save(settings: ServerSettings) {
        file.parentFile?.mkdirs()
        val lines = listOf(
            "admin_id" to settings.credentials.adminId,
            "admin_password_hash" to settings.credentials.passwordHash,
            "admin_password_salt" to settings.credentials.salt,
            "must_change_password" to settings.credentials.mustChangePassword.toString(),
            "default_port" to settings.port.toString(),
            "db_storage_dir" to settings.dataDir,
            "auto_start" to settings.autoStart.toString(),
            "max_embeddings_per_person" to settings.maxEmbeddingsPerPerson.toString(),
        )
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(lines.joinToString("\n", postfix = "\n") { (key, value) -> "$key: \"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\"" })
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
    }

    private fun parse(): Map<String, String> {
        if (!file.exists()) return emptyMap()
        return file.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") && it.contains(':') }
            .associate { line ->
                val key = line.substringBefore(':').trim()
                var value = line.substringAfter(':').trim()
                if (value.length >= 2 && value.first() == '"' && value.last() == '"') {
                    value = value.substring(1, value.length - 1).replace("\\\"", "\"").replace("\\\\", "\\")
                }
                key to value
            }
    }
}

object StoragePaths {
    private const val PRIMARY_TREE_PREFIX = "content://com.android.externalstorage.documents/tree/primary%3A"

    /**
     * Turns the stored folder into a file path. Folders picked with the system folder picker are
     * stored as `content://…/tree/primary%3A<path>`; on internal storage they map to /sdcard/<path>.
     */
    fun normalize(spec: String): String {
        val trimmed = spec.trim()
        if (trimmed.isEmpty()) return ServerSettings.DEFAULT_DATA_DIR
        if (trimmed.startsWith(PRIMARY_TREE_PREFIX)) {
            val relative = java.net.URLDecoder.decode(trimmed.removePrefix(PRIMARY_TREE_PREFIX), "UTF-8")
            return "/sdcard/" + relative.trim('/')
        }
        if (trimmed.startsWith("content://")) return ServerSettings.DEFAULT_DATA_DIR
        return trimmed.removePrefix("file://").trimEnd('/').ifEmpty { ServerSettings.DEFAULT_DATA_DIR }
    }
}
