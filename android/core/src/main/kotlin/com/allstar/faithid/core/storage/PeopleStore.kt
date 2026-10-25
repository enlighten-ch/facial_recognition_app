package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.FloatPacking
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.sync.Tombstone
import java.io.Closeable
import java.io.File

/**
 * The face database file (`face_meta.db`, schema version 3).
 *
 * One row per person with the samples in a BLOB, so changing one person rewrites one row instead of
 * the whole database. Also holds tombstones of deleted people and a small key/value table for
 * store-level values (server epoch and revision, kiosk sync position).
 *
 * Databases written by earlier versions (people table + separate `face_embeddings.bin`, or a CSV with
 * an embeddings column) are migrated on [open]; the original files are kept with a `.v2-backup`
 * suffix.
 */
class PeopleStore private constructor(
    val directory: File,
    private val db: SqlDatabase,
) : Closeable {

    /** Stable display number per person (first insert order); never renumbered. */
    fun displayNumbers(): Map<String, Long> =
        db.query("SELECT person_key, person_idx FROM people") { it.string("person_key").orEmpty() to (it.long("person_idx") ?: 0L) }.toMap()

    fun loadPeople(): List<Person> =
        db.query(
            "SELECT person_key, name, department, grade, embedding_count, embedding_dim, embeddings, " +
                "updated_at, metadata_blob, revision FROM people ORDER BY person_idx",
        ) { row -> readPerson(row) }.filterNotNull()

    fun loadTombstones(): List<Tombstone> =
        db.query("SELECT person_key, revision FROM tombstones ORDER BY revision") {
            Tombstone(it.string("person_key").orEmpty(), it.long("revision") ?: 0L)
        }

    fun upsert(person: Person) {
        val existingIdx = db.query("SELECT person_idx FROM people WHERE person_key = ?", person.key) { it.long("person_idx") }
            .firstOrNull()
        val idx = existingIdx ?: nextDisplayNumber()
        db.execute(
            "INSERT OR REPLACE INTO people(person_key, person_idx, name, department, grade, embedding_count, " +
                "embedding_dim, embeddings, hash, updated_at, metadata_blob, metadata_hash, revision) " +
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            person.key,
            idx,
            person.name,
            person.department,
            person.grade,
            person.embeddings.size,
            person.embeddingDim,
            FloatPacking.pack(person.embeddings),
            person.contentHash(),
            person.updatedAt,
            PersonCodec.encodeMetadata(person.metadata),
            person.metadata.contentHash(),
            person.revision,
        )
        db.execute("DELETE FROM tombstones WHERE person_key = ?", person.key)
    }

    fun delete(personKey: String) {
        db.execute("DELETE FROM people WHERE person_key = ?", personKey)
    }

    fun putTombstone(tombstone: Tombstone, deletedAt: Long) {
        db.execute("DELETE FROM people WHERE person_key = ?", tombstone.key)
        db.execute(
            "INSERT OR REPLACE INTO tombstones(person_key, revision, deleted_at) VALUES(?,?,?)",
            tombstone.key,
            tombstone.revision,
            deletedAt,
        )
    }

    /** Replaces all people and tombstones (full snapshot from the server). */
    fun replaceAll(people: List<Person>, tombstones: List<Tombstone>) {
        transaction {
            db.execute("DELETE FROM people")
            db.execute("DELETE FROM tombstones")
            people.forEach { upsert(it) }
            tombstones.forEach { putTombstone(it, 0L) }
        }
    }

    fun meta(key: String): String? = db.query("SELECT value FROM store_meta WHERE key = ?", key) { it.string("value") }.firstOrNull()

    fun setMeta(key: String, value: String?) {
        if (value == null) {
            db.execute("DELETE FROM store_meta WHERE key = ?", key)
        } else {
            db.execute("INSERT OR REPLACE INTO store_meta(key, value) VALUES(?,?)", key, value)
        }
    }

    fun <T> transaction(block: () -> T): T = db.transaction(block)

    /** Direct access for components that keep their own tables in the same file (kiosk outbox). */
    val database: SqlDatabase get() = db

    override fun close() = db.close()

    private fun nextDisplayNumber(): Long =
        (db.query("SELECT MAX(person_idx) AS m FROM people") { it.long("m") }.firstOrNull() ?: 0L) + 1

    private fun readPerson(row: SqlRow): Person? {
        val key = row.string("person_key")?.takeIf { it.isNotBlank() } ?: return null
        val count = (row.long("embedding_count") ?: 0L).toInt()
        val dim = (row.long("embedding_dim") ?: 0L).toInt()
        val embeddings = runCatching { FloatPacking.unpack(row.bytes("embeddings") ?: ByteArray(0), count, dim) }.getOrNull()
            ?: return null
        val metadata = PersonCodec.decodeMetadata(row.bytes("metadata_blob"))
            ?: PersonMetadata.of(row.string("name").orEmpty(), row.string("department").orEmpty(), row.string("grade").orEmpty())
        return Person(key, metadata, embeddings, row.long("revision") ?: 0L, row.long("updated_at") ?: 0L)
    }

    companion object {
        const val DB_FILE = "face_meta.db"
        const val LEGACY_BIN_FILE = "face_embeddings.bin"
        const val CSV_FILE = "face_db.csv"
        const val SCHEMA_VERSION = 3
        const val BACKUP_SUFFIX = ".v2-backup"

        fun open(directory: File, factory: SqlDatabaseFactory, log: (String) -> Unit = {}): PeopleStore {
            directory.mkdirs()
            val dbFile = File(directory, DB_FILE)
            val legacyPeople = LegacyImport.readIfLegacy(directory, factory, log)
            if (legacyPeople != null) {
                LegacyImport.backupLegacyFiles(directory, log)
            }
            val db = factory.open(dbFile)
            createSchema(db)
            val store = PeopleStore(directory, db)
            if (legacyPeople != null) {
                store.transaction { legacyPeople.forEach { store.upsert(it) } }
                log("legacy database migrated: ${legacyPeople.size} people")
            }
            return store
        }

        private fun createSchema(db: SqlDatabase) {
            db.transaction {
                db.execute(
                    "CREATE TABLE IF NOT EXISTS people(" +
                        "person_key TEXT PRIMARY KEY, person_idx INTEGER, name TEXT, department TEXT, grade TEXT, " +
                        "embedding_count INTEGER, embedding_dim INTEGER, embeddings BLOB, hash TEXT, updated_at INTEGER, " +
                        "metadata_blob BLOB, metadata_hash TEXT, revision INTEGER)",
                )
                db.execute("CREATE TABLE IF NOT EXISTS tombstones(person_key TEXT PRIMARY KEY, revision INTEGER, deleted_at INTEGER)")
                db.execute("CREATE TABLE IF NOT EXISTS store_meta(key TEXT PRIMARY KEY, value TEXT)")
                db.execute("INSERT OR REPLACE INTO store_meta(key, value) VALUES('schema_version', ?)", SCHEMA_VERSION.toString())
            }
        }
    }
}
