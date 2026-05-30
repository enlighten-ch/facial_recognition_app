package com.example.facetestbed

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.util.LinkedHashMap
import java.util.Locale

private val NAME_CANDIDATES = listOf("name", "이름")
private val DEPARTMENT_CANDIDATES = listOf("department", "dept", "folder", "부서")
private val GRADE_CANDIDATES = listOf("grade", "year", "학년")
private val RESERVED_METADATA_COLUMNS =
    setOf(
        "person_id",
        "person_idx",
        "person_key",
        "metadata_hash",
        "embeddings_json",
        "embeddings",
        "embedding_json",
        "embedding_count",
        "hash",
        "record_hash",
        "updated_at",
        "updatedat",
    )

fun defaultMetadataMap(name: String, department: String, grade: String): LinkedHashMap<String, String> {
    return linkedMapOf(
        "name" to name.trim(),
        "department" to department.trim(),
        "grade" to grade.trim(),
    )
}

fun metadataName(metadata: Map<String, String>, fallback: String = ""): String =
    metadataLookup(metadata, NAME_CANDIDATES).ifBlank { fallback.trim() }

fun metadataDepartment(metadata: Map<String, String>, fallback: String = ""): String =
    metadataLookup(metadata, DEPARTMENT_CANDIDATES).ifBlank { fallback.trim() }

fun metadataGrade(metadata: Map<String, String>, fallback: String = ""): String =
    metadataLookup(metadata, GRADE_CANDIDATES).ifBlank { fallback.trim() }

fun personIdFromIdentity(name: String, department: String, grade: String): Long = 0L

fun parsePersonId(raw: String?, name: String = "", department: String = "", grade: String = ""): Long {
    val trimmed = raw?.trim().orEmpty()
    return trimmed.toLongOrNull()?.takeIf { it > 0L } ?: 0L
}

fun personKeyFromIdentity(name: String, department: String, grade: String): String {
    val digest = MessageDigest.getInstance("SHA-256")
    listOf(name.trim(), department.trim(), grade.trim()).forEach { value ->
        digest.update(value.lowercase(Locale.US).toByteArray(Charsets.UTF_8))
        digest.update(0)
    }
    return digest.digest().joinToString("") { b -> "%02x".format(b) }
}

fun parsePersonKey(raw: String?, name: String = "", department: String = "", grade: String = ""): String {
    return raw?.trim()?.takeIf { it.isNotBlank() } ?: personKeyFromIdentity(name, department, grade)
}

fun metadataIdentityKey(name: String, department: String, grade: String): String {
    return listOf(name.trim(), grade.trim(), department.trim()).joinToString("\u0001")
}

fun metadataIdentityKey(metadata: Map<String, String>): String {
    return metadataIdentityKey(
        metadataName(metadata),
        metadataDepartment(metadata),
        metadataGrade(metadata),
    )
}

fun normalizeMetadataMap(
    metadata: Map<String, String>,
    fallbackName: String = "",
    fallbackDepartment: String = "",
    fallbackGrade: String = "",
): LinkedHashMap<String, String> {
    val out = linkedMapOf<String, String>()
    metadata.forEach { (key, value) ->
        val trimmedKey = key.trim()
        if (trimmedKey.isBlank()) return@forEach
        out[trimmedKey] = value.trim()
    }
    val name = metadataName(out, fallbackName)
    val department = metadataDepartment(out, fallbackDepartment)
    val grade = metadataGrade(out, fallbackGrade)
    if (out.keys.none { it.equals("name", ignoreCase = true) }) out["name"] = name
    if (out.keys.none { it.equals("department", ignoreCase = true) }) out["department"] = department
    if (out.keys.none { it.equals("grade", ignoreCase = true) }) out["grade"] = grade
    return out
}

fun metadataHash(metadata: Map<String, String>): String {
    val digest = MessageDigest.getInstance("SHA-256")
    metadata.entries
        .sortedBy { it.key.lowercase(Locale.US) }
        .forEach { (key, value) ->
            digest.update(key.trim().lowercase(Locale.US).toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(value.trim().toByteArray(Charsets.UTF_8))
            digest.update(0)
        }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

fun encodeMetadataBlob(metadata: Map<String, String>): ByteArray {
    val normalized = normalizeMetadataMap(metadata)
    val out = ByteArrayOutputStream()
    out.write(byteArrayOf('M'.code.toByte(), 'D'.code.toByte(), 'T'.code.toByte(), '1'.code.toByte()))
    writeIntLe(out, normalized.size)
    normalized.forEach { (key, value) ->
        writeStringLe(out, key)
        writeStringLe(out, value)
    }
    return out.toByteArray()
}

fun decodeMetadataBlob(bytes: ByteArray?): LinkedHashMap<String, String> {
    if (bytes == null || bytes.size < 8) return linkedMapOf()
    if (bytes[0] != 'M'.code.toByte() || bytes[1] != 'D'.code.toByte() || bytes[2] != 'T'.code.toByte() || bytes[3] != '1'.code.toByte()) {
        return linkedMapOf()
    }
    val input = ByteArrayInputStream(bytes, 4, bytes.size - 4)
    val count = readIntLe(input) ?: return linkedMapOf()
    val out = linkedMapOf<String, String>()
    repeat(count.coerceAtLeast(0)) {
        val key = readStringLe(input) ?: return@repeat
        val value = readStringLe(input) ?: return@repeat
        if (key.isNotBlank()) {
            out[key] = value
        }
    }
    return normalizeMetadataMap(out)
}

fun csvMetadataFromRow(header: List<String>, row: List<String>): LinkedHashMap<String, String> {
    val metadata = linkedMapOf<String, String>()
    header.forEachIndexed { index, rawKey ->
        val trimmedKey = rawKey.trim()
        if (trimmedKey.isBlank()) return@forEachIndexed
        val normalizedKey = trimmedKey.lowercase(Locale.US)
        if (normalizedKey in RESERVED_METADATA_COLUMNS) return@forEachIndexed
        metadata[trimmedKey] = row.getOrNull(index)?.trim().orEmpty()
    }
    return normalizeMetadataMap(metadata)
}

private fun metadataLookup(metadata: Map<String, String>, candidates: List<String>): String {
    candidates.forEach { candidate ->
        metadata.entries.firstOrNull { it.key.equals(candidate, ignoreCase = true) }?.value?.trim()?.let { value ->
            if (value.isNotBlank()) return value
        }
    }
    return ""
}

private fun writeIntLe(out: ByteArrayOutputStream, value: Int) {
    out.write(value and 0xFF)
    out.write((value ushr 8) and 0xFF)
    out.write((value ushr 16) and 0xFF)
    out.write((value ushr 24) and 0xFF)
}

private fun writeStringLe(out: ByteArrayOutputStream, value: String) {
    val bytes = value.toByteArray(Charsets.UTF_8)
    writeIntLe(out, bytes.size)
    out.write(bytes)
}

private fun readIntLe(input: ByteArrayInputStream): Int? {
    val b0 = input.read()
    val b1 = input.read()
    val b2 = input.read()
    val b3 = input.read()
    if (b3 < 0) return null
    return (b0 and 0xFF) or ((b1 and 0xFF) shl 8) or ((b2 and 0xFF) shl 16) or ((b3 and 0xFF) shl 24)
}

private fun readStringLe(input: ByteArrayInputStream): String? {
    val length = readIntLe(input) ?: return null
    if (length < 0 || length > input.available()) return null
    val bytes = ByteArray(length)
    val read = input.read(bytes, 0, length)
    if (read != length) return null
    return bytes.toString(Charsets.UTF_8)
}
