package com.allstar.faithid.androidsupport

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.allstar.faithid.core.codec.FloatPacking
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonKeys
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.storage.PeopleStore
import com.allstar.faithid.core.sync.Tombstone
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.random.Random

/** Runs the storage layer on the device's SQLite (the JVM tests use sqlite-jdbc). */
@RunWith(AndroidJUnit4::class)
class AndroidStorageTest {
    private lateinit var dir: File
    private val random = Random(5)

    private fun samples(n: Int) = List(n) { Vectors.l2Normalize(FloatArray(512) { random.nextFloat() - 0.5f }) }

    @Before
    fun setUp() {
        dir = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "storage-test-${System.nanoTime()}")
        dir.mkdirs()
    }

    @After
    fun tearDown() {
        dir.deleteRecursively()
    }

    @Test
    fun peopleStoreRoundTripOnPlatformSqlite() {
        val person = Person("p_1", PersonMetadata.of("홍길동", "청년부", "3학년"), samples(20), revision = 4, updatedAt = 99)
        PeopleStore.open(dir, AndroidSqlDatabase.FACTORY).use { store ->
            store.transaction { store.upsert(person) }
            store.upsert(Person("p_2", PersonMetadata.of("김철수", "", ""), samples(3)))
            store.putTombstone(Tombstone("p_2", 5), 1)
            store.setMeta("k", "v")
        }
        PeopleStore.open(dir, AndroidSqlDatabase.FACTORY).use { store ->
            val loaded = store.loadPeople().single()
            assertTrue(loaded.sameContentAs(person))
            assertEquals(4, loaded.revision)
            assertEquals(listOf(Tombstone("p_2", 5)), store.loadTombstones())
            assertEquals("v", store.meta("k"))
            assertEquals(mapOf("p_1" to 1L), store.displayNumbers())
        }
    }

    @Test
    fun legacyDatabaseIsMigratedOnPlatformSqlite() {
        val legacy = samples(4)
        val packed = FloatPacking.pack(legacy)
        AndroidSqlDatabase(File(dir, PeopleStore.DB_FILE)).use { db ->
            db.execute(
                "CREATE TABLE people(person_key TEXT PRIMARY KEY,person_idx INTEGER,name TEXT,department TEXT,grade TEXT," +
                    "offset_bytes INTEGER,length_bytes INTEGER,embedding_count INTEGER,embedding_dim INTEGER,hash TEXT," +
                    "updated_at INTEGER,metadata_blob BLOB,metadata_hash TEXT)",
            )
            db.execute(
                "INSERT INTO people VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                PersonKeys.legacyKey("홍길동", "청년부", ""), 1, "홍길동", "청년부", "", 0, packed.size, 4, 512, "h", 1L,
                PersonCodec.encodeMetadata(PersonMetadata.of("홍길동", "청년부", "")), "m",
            )
        }
        File(dir, PeopleStore.LEGACY_BIN_FILE).writeBytes(packed)
        PeopleStore.open(dir, AndroidSqlDatabase.FACTORY).use { store ->
            val person = store.loadPeople().single()
            assertEquals("홍길동", person.name)
            assertTrue(person.embeddings.indices.all { person.embeddings[it].contentEquals(legacy[it]) })
        }
        assertTrue(File(dir, PeopleStore.DB_FILE + PeopleStore.BACKUP_SUFFIX).exists())
    }
}
