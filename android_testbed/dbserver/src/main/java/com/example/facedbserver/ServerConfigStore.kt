package com.example.facedbserver

import android.content.Context
import java.io.File

data class ServerUiConfig(
    val adminId: String = "admin",
    val adminPassword: String = "1234",
    val defaultPort: Int = 18080,
    val dbStorageDir: String = "",
)

class ServerConfigStore(context: Context) {
    private val configFile: File by lazy {
        val dir = File(context.filesDir, "face_db_server")
        if (!dir.exists()) dir.mkdirs()
        File(dir, "config.yaml")
    }

    fun load(): ServerUiConfig {
        val defaults = ServerUiConfig()
        if (!configFile.exists()) {
            save(defaults)
            return defaults
        }

        val raw = runCatching { configFile.readText() }.getOrNull().orEmpty()
        if (raw.isBlank()) {
            save(defaults)
            return defaults
        }

        val kv = mutableMapOf<String, String>()
        raw.lineSequence().forEach { line ->
            val trimmed = line.trim()
            if (trimmed.isBlank() || trimmed.startsWith("#")) return@forEach
            val idx = trimmed.indexOf(':')
            if (idx <= 0) return@forEach
            val key = trimmed.substring(0, idx).trim()
            val value = unquote(trimmed.substring(idx + 1).trim())
            kv[key] = value
        }

        return ServerUiConfig(
            adminId = kv["admin_id"].orEmpty().ifBlank { defaults.adminId },
            adminPassword = kv["admin_password"].orEmpty().ifBlank { defaults.adminPassword },
            defaultPort = kv["default_port"]?.toIntOrNull()?.coerceIn(1, 65535) ?: defaults.defaultPort,
            dbStorageDir = kv["db_storage_dir"].orEmpty(),
        )
    }

    fun save(config: ServerUiConfig): Boolean {
        val normalized = config.copy(
            adminId = config.adminId.trim().ifBlank { "admin" },
            adminPassword = config.adminPassword.ifBlank { "1234" },
            defaultPort = config.defaultPort.coerceIn(1, 65535),
            dbStorageDir = config.dbStorageDir.trim(),
        )

        val content = buildString {
            appendLine("admin_id: \"${escape(normalized.adminId)}\"")
            appendLine("admin_password: \"${escape(normalized.adminPassword)}\"")
            appendLine("default_port: ${normalized.defaultPort}")
            appendLine("db_storage_dir: \"${escape(normalized.dbStorageDir)}\"")
        }

        return runCatching {
            configFile.parentFile?.mkdirs()
            configFile.writeText(content)
            true
        }.getOrDefault(false)
    }

    private fun unquote(value: String): String {
        if (value.length < 2) return value
        val first = value.first()
        val last = value.last()
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return value.substring(1, value.length - 1)
        }
        return value
    }

    private fun escape(value: String): String {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
    }
}
