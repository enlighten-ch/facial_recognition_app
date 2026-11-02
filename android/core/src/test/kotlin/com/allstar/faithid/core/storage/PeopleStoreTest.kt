package com.allstar.faithid.core.storage

import com.allstar.faithid.core.codec.Csv
import com.allstar.faithid.core.codec.FloatPacking
import com.allstar.faithid.core.codec.PersonCodec
import com.allstar.faithid.core.model.Person
import com.allstar.faithid.core.model.PersonKeys
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.sync.Tombstone
import com.allstar.faithid.core.testing.FakeFaces
import com.allstar.faithid.core.testing.JdbcSqlDatabase
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class PeopleStoreTest {
    @get:Rule
    val temp = TemporaryFolder()

    private val faces = FakeFaces(seed = 9, dim = 32)
    private val factory = JdbcSqlDatabase.FACTORY

    @Test
    fun `stores people, keeps display numbers stable and reloads them`() {
        val dir = temp.newFolder("db")
        val a = faces.person("A", key = "a").copy(revision = 3, updatedAt = 10)
        val b = faces.person("B", key = "b")
        PeopleStore.open(dir, factory).use { store ->
            store.upsert(a)
            store.upsert(b)
            store.upsert(a.copy(metadata = a.metadata.with(grade = "4학년")))
            assertEquals(mapOf("a" to 1L, "b" to 2L), store.displayNumbers())
        }
        PeopleStore.open(dir, factory).use { store ->
            val loaded = store.loadPeople().associateBy { it.key }
            assertEquals(2, loaded.size)
            assertEquals("4학년", loaded.getValue("a").grade)
            assertEquals(3, loaded.getValue("a").revision)
            assertTrue(loaded.getValue("b").sameContentAs(b))
        }
    }

    @Test
    fun `tombstones remove the person and a later upsert clears the tombstone`() {
        PeopleStore.open(temp.newFolder(), factory).use { store ->
            val a = faces.person("A", key = "a")
            store.upsert(a)
            store.putTombstone(Tombstone("a", 5), deletedAt = 1)
            assertTrue(store.loadPeople().isEmpty())
            assertEquals(listOf(Tombstone("a", 5)), store.loadTombstones())
            store.upsert(a)
            assertTrue(store.loadTombstones().isEmpty())
        }
    }

    @Test
    fun `replaceAll, meta values and rollback on failure`() {
        PeopleStore.open(temp.newFolder(), factory).use { store ->
            store.upsert(faces.person("old", key = "old"))
            store.replaceAll(listOf(faces.person("new", key = "new")), listOf(Tombstone("old", 2)))
            assertEquals(listOf("new"), store.loadPeople().map { it.key })
            store.setMeta("epoch", "e1")
            assertEquals("e1", store.meta("epoch"))
            store.setMeta("epoch", null)
            assertNull(store.meta("epoch"))
            runCatching {
                store.transaction {
                    store.upsert(faces.person("temp", key = "temp"))
                    error("boom")
                }
            }
            assertEquals(listOf("new"), store.loadPeople().map { it.key })
        }
    }

    @Test
    fun `migrates the 2026-05 layout and keeps the old files`() {
        val dir = temp.newFolder("legacy")
        val people = listOf(
            faces.person("홍길동", "청년부", "3학년", samples = 3, key = "hk"),
            faces.person("김철수", "중등부", "", samples = 2, key = "kk"),
        )
        writeLegacyV2(dir, people)
        PeopleStore.open(dir, factory).use { store ->
            val loaded = store.loadPeople().associateBy { it.name }
            assertEquals(2, loaded.size)
            val hong = loaded.getValue("홍길동")
            assertEquals(PersonKeys.legacyKey("홍길동", "청년부", "3학년"), hong.key)
            assertTrue(EmbeddingsMatch(people[0].embeddings, hong.embeddings))
            assertEquals("중등부", loaded.getValue("김철수").department)
        }
        assertTrue(File(dir, "face_meta.db.v2-backup").exists())
        assertTrue(File(dir, "face_embeddings.bin.v2-backup").exists())
        // Opening again must not migrate twice.
        PeopleStore.open(dir, factory).use { assertEquals(2, it.loadPeople().size) }
        assertTrue(!File(dir, "face_meta.db.v2-backup.1").exists())
    }

    @Test
    fun `imports a prototype CSV that still has an embeddings column`() {
        val dir = temp.newFolder("csv")
        val samples = faces.samples("x", 2)
        val json = samples.joinToString(",", "[", "]") { v -> v.joinToString(",", "[", "]") }
        File(dir, "face_db.csv").writeText(Csv.write(listOf(listOf("name", "department", "grade", "embeddings_json"), listOf("홍길동", "청년부", "1학년", json))))
        PeopleStore.open(dir, factory).use { store ->
            val person = store.loadPeople().single()
            assertEquals("홍길동", person.name)
            assertEquals(2, person.embeddings.size)
            assertEquals(samples[1][5], person.embeddings[1][5], 1e-6f)
        }
    }

    @Test
    fun `float json parser rejects malformed input`() {
        assertEquals(2, FloatJson.parseMatrix("[[1, 2.5], [3, -4e-1]]").size)
        assertTrue(FloatJson.parseMatrix("[[1, x]]").isEmpty())
        assertTrue(FloatJson.parseMatrix("not json").isEmpty())
        assertEquals(1, FloatJson.parseMatrix("[[1,2],[3]]").size)
    }

    private fun EmbeddingsMatch(a: List<FloatArray>, b: List<FloatArray>) =
        a.size == b.size && a.indices.all { a[it].contentEquals(b[it]) }

    /** Writes the people table + face_embeddings.bin exactly like the 2026-05 apps did. */
    private fun writeLegacyV2(dir: File, people: List<com.allstar.faithid.core.model.Person>) {
        val bin = java.io.ByteArrayOutputStream()
        JdbcSqlDatabase(File(dir, "face_meta.db")).use { db ->
            db.execute(
                "CREATE TABLE people(person_key TEXT PRIMARY KEY,person_idx INTEGER,name TEXT,department TEXT,grade TEXT," +
                    "offset_bytes INTEGER,length_bytes INTEGER,embedding_count INTEGER,embedding_dim INTEGER,hash TEXT," +
                    "updated_at INTEGER,metadata_blob BLOB,metadata_hash TEXT)",
            )
            people.forEachIndexed { index, person ->
                val offset = bin.size()
                val packed = FloatPacking.pack(person.embeddings)
                bin.write(packed)
                db.execute(
                    "INSERT INTO people VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    PersonKeys.legacyKey(person.name, person.department, person.grade),
                    index + 1,
                    person.name,
                    person.department,
                    person.grade,
                    offset,
                    packed.size,
                    person.embeddings.size,
                    person.embeddingDim,
                    "h",
                    1L,
                    PersonCodec.encodeMetadata(PersonMetadata.of(person.name, person.department, person.grade)),
                    "m",
                )
            }
        }
        File(dir, "face_embeddings.bin").writeBytes(bin.toByteArray())
    }
}

class PeopleCsvTest {
    private val faces = FakeFaces(seed = 2, dim = 8)

    @Test
    fun `export lists people by number with one column per metadata column`() {
        val hong = faces.person("홍길동", "청년부", "3학년", key = "hong")
        val kim = faces.person("김철수", "중등부", "1학년", key = "kim")
        val numbers = mapOf("hong" to 1L, "kim" to 2L)
        val table = Csv.parse(PeopleCsv.export(listOf(kim, hong), numbers))
        assertEquals(listOf("person_idx", "family_name", "given_name", "department", "grade", "embedding_count", "updated_at", "person_key"), table[0])
        assertEquals(listOf("홍", "길동"), table[1].subList(1, 3))
    }

    @Test
    fun `aliases fold into one column, names from the app become headers, the name is split`() {
        val korean = Person("a", PersonMetadata.of(linkedMapOf("이름" to "홍길동", "부서" to "청년부")), faces.samples("a", 1))
        val split = Person("b", PersonMetadata.of("강", "율", "찬양팀", ""), faces.samples("b", 1))
        val twoSyllables = Person("c", PersonMetadata.of("하은", "", ""), faces.samples("c", 1))
        val export = PeopleCsv.exportTable(listOf(korean, split, twoSyllables), mapOf("a" to 1L, "b" to 2L, "c" to 3L), mapOf("department" to "반"))
        assertEquals(listOf("family_name", "given_name", "department", "grade"), export.columns)
        val table = Csv.parse(export.text)
        assertEquals(listOf("person_idx", "family_name", "given_name", "반", "grade", "embedding_count", "updated_at", "person_key"), table[0])
        assertEquals(listOf("홍", "길동", "청년부"), table[1].subList(1, 4))
        assertEquals(listOf("강", "율"), table[2].subList(1, 3))
        assertEquals(listOf("", "하은"), table[3].subList(1, 3))
    }
}
