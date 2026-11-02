package com.allstar.faithid.core.model

import java.security.MessageDigest
import java.util.Locale

/**
 * Display attributes of a person (name, department, grade, an optional family name and any extra
 * CSV columns). [name] is always the full name; the family name only tells where it splits.
 *
 * Keys keep their original spelling and order so that a CSV edited in Excel round-trips unchanged.
 * Lookups of the three well-known fields accept the Korean and legacy column names as aliases.
 */
class PersonMetadata private constructor(private val fields: LinkedHashMap<String, String>) {

    val name: String get() = lookup(NAME_ALIASES)
    val department: String get() = lookup(DEPARTMENT_ALIASES)
    val grade: String get() = lookup(GRADE_ALIASES)

    /** The family name entered at registration ("김"), or "" for people registered before it was asked. */
    val familyName: String get() = lookup(FAMILY_NAME_ALIASES)

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
            in FAMILY_NAME_ALIASES -> familyName
            else -> get(column)
        }
    }

    /**
     * Sets the name from its parts: the full name is both joined ([PersonNames.join]); the family
     * name is kept when both parts are given, otherwise dropped.
     */
    fun withName(familyName: String, givenName: String): PersonMetadata {
        val named = withValue(CANONICAL_NAME, PersonNames.join(familyName, givenName))
        return if (familyName.isNotBlank() && givenName.isNotBlank()) named.withValue(CANONICAL_FAMILY_NAME, familyName) else named.without(CANONICAL_FAMILY_NAME)
    }

    /** Sets one column, keeping the spelling already used for it (also for the well-known aliases). */
    fun withValue(column: String, value: String): PersonMetadata {
        val next = LinkedHashMap(fields)
        when (val aliases = aliasesOf(column)) {
            null -> {
                val existing = next.keys.firstOrNull { it.equals(column.trim(), ignoreCase = true) }
                next[existing ?: column.trim()] = collapseSpaces(value)
            }
            else -> replace(next, aliases, canonicalOf(aliases), value)
        }
        return of(next)
    }

    /**
     * Removes a column. Name, department and grade always exist, so they are emptied instead; the
     * other columns (also family_name) disappear.
     */
    fun without(column: String): PersonMetadata {
        val aliases = aliasesOf(column)
        if (aliases != null && aliases !== FAMILY_NAME_ALIASES) return withValue(column, "")
        val next = LinkedHashMap(fields)
        next.keys.filter { key -> if (aliases != null) key.lowercase(Locale.US) in aliases else key.equals(column.trim(), ignoreCase = true) }
            .forEach { next.remove(it) }
        return of(next)
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
        const val CANONICAL_FAMILY_NAME = "family_name"

        /** Only in face_db.csv: the given name, next to family_name, in place of the full name. */
        const val CSV_GIVEN_NAME = "given_name"

        val NAME_ALIASES = setOf("name", "이름")
        val DEPARTMENT_ALIASES = setOf("department", "dept", "folder", "부서")
        val GRADE_ALIASES = setOf("grade", "year", "학년")
        val FAMILY_NAME_ALIASES = setOf("family_name", "성")

        /** The canonical key of a column: "부서" → "department"; other columns are kept as they are. */
        fun canonicalColumn(column: String): String = aliasesOf(column)?.let { canonicalOf(it) } ?: column.trim()

        private fun aliasesOf(column: String): Set<String>? {
            val normalized = column.trim().lowercase(Locale.US)
            return listOf(NAME_ALIASES, DEPARTMENT_ALIASES, GRADE_ALIASES, FAMILY_NAME_ALIASES).firstOrNull { normalized in it }
        }

        private fun canonicalOf(aliases: Set<String>): String = when (aliases) {
            NAME_ALIASES -> CANONICAL_NAME
            DEPARTMENT_ALIASES -> CANONICAL_DEPARTMENT
            GRADE_ALIASES -> CANONICAL_GRADE
            else -> CANONICAL_FAMILY_NAME
        }

        fun of(name: String, department: String, grade: String): PersonMetadata =
            of(linkedMapOf(CANONICAL_NAME to name, CANONICAL_DEPARTMENT to department, CANONICAL_GRADE to grade))

        /** A new registration: the full name is [familyName] and [givenName] joined (see [PersonNames.join]). */
        fun of(familyName: String, givenName: String, department: String, grade: String): PersonMetadata {
            val fields = linkedMapOf(CANONICAL_NAME to PersonNames.join(familyName, givenName), CANONICAL_DEPARTMENT to department, CANONICAL_GRADE to grade)
            if (familyName.isNotBlank() && givenName.isNotBlank()) fields[CANONICAL_FAMILY_NAME] = familyName
            return of(fields)
        }

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
