package com.allstar.faithid.core.model

import java.security.MessageDigest
import java.util.Locale

/**
 * Display attributes of a person (name, department, grade and any extra CSV columns).
 *
 * Keys keep their original spelling and order so that a CSV edited in Excel round-trips unchanged.
 * Lookups of the three well-known fields accept the Korean and legacy column names as aliases.
 */
class PersonMetadata private constructor(private val fields: LinkedHashMap<String, String>) {

    val name: String get() = lookup(NAME_ALIASES)
    val department: String get() = lookup(DEPARTMENT_ALIASES)
    val grade: String get() = lookup(GRADE_ALIASES)

    /** Fields in insertion order, including name/department/grade. */
    val entries: Map<String, String> get() = fields

    operator fun get(key: String): String =
        fields.entries.firstOrNull { it.key.equals(key.trim(), ignoreCase = true) }?.value.orEmpty()

    /** Value for a print/display column; the well-known columns resolve through their aliases. */
    fun valueOf(column: String): String {
        val normalized = column.trim().lowercase(Locale.US)
        return when (normalized) {
            in NAME_ALIASES -> name
            in DEPARTMENT_ALIASES -> department
            in GRADE_ALIASES -> grade
            else -> get(column)
        }
    }

    fun with(name: String = this.name, department: String = this.department, grade: String = this.grade): PersonMetadata {
        val next = LinkedHashMap(fields)
        replace(next, NAME_ALIASES, CANONICAL_NAME, name)
        replace(next, DEPARTMENT_ALIASES, CANONICAL_DEPARTMENT, department)
        replace(next, GRADE_ALIASES, CANONICAL_GRADE, grade)
        return of(next)
    }

    /** Human-readable label such as "홍길동 (청년부, 3학년)". */
    fun label(): String {
        val extra = listOf(department, grade).filter { it.isNotBlank() }
        return if (extra.isEmpty()) name else "$name (${extra.joinToString(", ")})"
    }

    /** Identity used to detect "same person" when no stable key is available (legacy data, CSV rows). */
    fun identityKey(): String = listOf(name, department, grade).joinToString("\u0001") { normalizeForIdentity(it) }

    /** Order-independent hash of all fields. */
    fun contentHash(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        fields.entries.sortedBy { it.key.lowercase(Locale.US) }.forEach { (key, value) ->
            digest.update(key.lowercase(Locale.US).toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(value.toByteArray(Charsets.UTF_8))
            digest.update(0)
        }
        return digest.digest().toHex()
    }

    override fun equals(other: Any?): Boolean = other is PersonMetadata && other.fields == fields
    override fun hashCode(): Int = fields.hashCode()
    override fun toString(): String = "PersonMetadata($fields)"

    private fun lookup(aliases: Set<String>): String =
        fields.entries.firstOrNull { it.key.lowercase(Locale.US) in aliases && it.value.isNotBlank() }?.value.orEmpty()

    companion object {
        const val CANONICAL_NAME = "name"
        const val CANONICAL_DEPARTMENT = "department"
        const val CANONICAL_GRADE = "grade"

        val NAME_ALIASES = setOf("name", "이름")
        val DEPARTMENT_ALIASES = setOf("department", "dept", "folder", "부서")
        val GRADE_ALIASES = setOf("grade", "year", "학년")

        fun of(name: String, department: String, grade: String): PersonMetadata =
            of(linkedMapOf(CANONICAL_NAME to name, CANONICAL_DEPARTMENT to department, CANONICAL_GRADE to grade))

        /**
         * Normalizes arbitrary fields: trims keys and values, drops blank keys and guarantees that
         * name/department/grade exist (empty if unknown).
         */
        fun of(fields: Map<String, String>): PersonMetadata {
            val out = LinkedHashMap<String, String>()
            fields.forEach { (rawKey, rawValue) ->
                val key = rawKey.trim()
                if (key.isNotEmpty() && out.keys.none { it.equals(key, ignoreCase = true) }) {
                    out[key] = collapseSpaces(rawValue)
                }
            }
            ensure(out, NAME_ALIASES, CANONICAL_NAME)
            ensure(out, DEPARTMENT_ALIASES, CANONICAL_DEPARTMENT)
            ensure(out, GRADE_ALIASES, CANONICAL_GRADE)
            return PersonMetadata(out)
        }

        fun normalizeForIdentity(value: String): String = collapseSpaces(value).lowercase(Locale.ROOT)

        private fun collapseSpaces(value: String): String = value.trim().replace(WHITESPACE, " ")

        private fun ensure(fields: LinkedHashMap<String, String>, aliases: Set<String>, canonical: String) {
            if (fields.keys.none { it.lowercase(Locale.US) in aliases }) fields[canonical] = ""
        }

        private fun replace(fields: LinkedHashMap<String, String>, aliases: Set<String>, canonical: String, value: String) {
            val keys = fields.keys.filter { it.lowercase(Locale.US) in aliases }
            if (keys.isEmpty()) {
                fields[canonical] = collapseSpaces(value)
            } else {
                keys.forEach { fields[it] = collapseSpaces(value) }
            }
        }

        private val WHITESPACE = Regex("\\s+")
    }
}

internal fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
