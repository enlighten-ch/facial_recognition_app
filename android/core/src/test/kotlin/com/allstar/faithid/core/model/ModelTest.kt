package com.allstar.faithid.core.model

import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PersonMetadataTest {

    @Test
    fun `korean and legacy column names resolve to the well-known fields`() {
        val metadata = PersonMetadata.of(linkedMapOf("이름" to " 홍길동 ", "부서" to "청년부", "학년" to "3학년", "비고" to "새가족"))
        assertEquals("홍길동", metadata.name)
        assertEquals("청년부", metadata.department)
        assertEquals("3학년", metadata.grade)
        assertEquals("새가족", metadata["비고"])
        assertEquals("홍길동", metadata.valueOf("name"))
        assertEquals("새가족", metadata.valueOf("비고"))
    }

    @Test
    fun `missing well-known fields are added empty and duplicate keys keep the first`() {
        val metadata = PersonMetadata.of(linkedMapOf("Name" to "A", "name" to "B"))
        assertEquals("A", metadata.name)
        assertEquals("", metadata.department)
        assertTrue(metadata.entries.keys.any { it == "department" })
        assertEquals(3, metadata.entries.size)
    }

    @Test
    fun `with replaces values under the original column names`() {
        val metadata = PersonMetadata.of(linkedMapOf("이름" to "홍길동", "부서" to "청년부", "학년" to "3학년"))
        val edited = metadata.with(name = "홍길순", department = "  중등부  ")
        assertEquals("홍길순", edited.name)
        assertEquals("중등부", edited.department)
        assertEquals(listOf("이름", "부서", "학년"), edited.entries.keys.toList())
    }

    @Test
    fun `identity ignores case and extra spaces and label shows department and grade`() {
        val a = PersonMetadata.of("Kim  Chul", "청년부", "1학년")
        val b = PersonMetadata.of("kim chul", "청년부 ", "1학년")
        assertEquals(a.identityKey(), b.identityKey())
        assertEquals("Kim Chul (청년부, 1학년)", a.label())
        assertEquals("이름만", PersonMetadata.of("이름만", "", "").label())
    }

    @Test
    fun `content hash ignores field order but not values`() {
        val a = PersonMetadata.of(linkedMapOf("name" to "A", "department" to "D", "grade" to "G"))
        val b = PersonMetadata.of(linkedMapOf("grade" to "G", "name" to "A", "department" to "D"))
        assertEquals(a.contentHash(), b.contentHash())
        assertNotEquals(a.contentHash(), a.with(grade = "H").contentHash())
    }
}

class VectorsAndEmbeddingSetsTest {
    private val faces = FakeFaces(seed = 7, dim = 64)

    @Test
    fun `centroid is unit length and close to the identity base`() {
        val samples = faces.samples("a", 10)
        val centroid = Vectors.centroid(samples)
        assertEquals(1f, Vectors.dot(centroid, centroid), 1e-4f)
        assertTrue(Vectors.dot(centroid, faces.base("a")) > 0.9f)
        assertTrue(Vectors.dot(centroid, faces.base("b")) < 0.5f)
    }

    @Test
    fun `fingerprint identifies equal vectors only`() {
        val v = faces.sample("a")
        assertEquals(Vectors.fingerprint(v), Vectors.fingerprint(v.copyOf()))
        val changed = v.copyOf().also { it[3] = it[3] + 1e-6f }
        assertNotEquals(Vectors.fingerprint(v), Vectors.fingerprint(changed))
    }

    @Test
    fun `union is idempotent and order independent as a set`() {
        val a = faces.samples("a", 5)
        val b = faces.samples("a", 4)
        val ab = EmbeddingSets.union(a, b)
        val ba = EmbeddingSets.union(b, a)
        assertEquals(9, ab.size)
        assertTrue(EmbeddingSets.sameSet(ab, ba))
        assertTrue(EmbeddingSets.sameSet(ab, EmbeddingSets.union(ab, b)))
        assertTrue(EmbeddingSets.containsAll(ab, b))
        assertFalse(EmbeddingSets.containsAll(a, b))
    }

    @Test
    fun `union drops samples with a different dimension`() {
        val a = faces.samples("a", 2)
        val merged = EmbeddingSets.union(a, listOf(FloatArray(3) { 1f }))
        assertEquals(2, merged.size)
    }

    @Test
    fun `prune keeps the limit, drops the outlier first and is deterministic`() {
        val samples = faces.samples("a", 10) + listOf(faces.sample("intruder"))
        val pruned = EmbeddingSets.prune(samples, 10)
        assertEquals(10, pruned.size)
        assertFalse(pruned.any { Vectors.fingerprint(it) == Vectors.fingerprint(samples.last()) })
        assertTrue(EmbeddingSets.sameSet(pruned, EmbeddingSets.prune(samples.reversed(), 10)))
        assertEquals(3, EmbeddingSets.prune(samples, 3).size)
        assertEquals(samples.size, EmbeddingSets.prune(samples, 100).size)
    }

    @Test
    fun `person content hash covers metadata and the sample set`() {
        val p = faces.person("홍길동")
        val reordered = p.copy(embeddings = p.embeddings.reversed())
        assertEquals(p.contentHash(), reordered.contentHash())
        assertTrue(p.sameContentAs(reordered))
        assertNotEquals(p.contentHash(), p.copy(metadata = p.metadata.with(grade = "2학년")).contentHash())
        assertNotEquals(p.contentHash(), p.copy(embeddings = p.embeddings.drop(1)).contentHash())
    }

    @Test
    fun `new keys are unique and legacy keys match the old formula`() {
        assertNotEquals(PersonKeys.newKey(), PersonKeys.newKey())
        assertTrue(PersonKeys.newKey().startsWith("p_"))
        // Value produced by the 2026-05 app for ("홍길동", "청년부", "3학년").
        assertEquals(PersonKeys.legacyKey("홍길동", "청년부", "3학년"), PersonKeys.legacyKey(" 홍길동", "청년부 ", "3학년"))
        assertEquals(64, PersonKeys.legacyKey("a", "b", "c").length)
    }
}

class PersonNamesTest {

    @Test
    fun `a registration with family and given name keeps the full name and where it splits`() {
        val metadata = PersonMetadata.of("김", "하은", "유치부", "5세")
        assertEquals("김하은", metadata.name)
        assertEquals("김", metadata.familyName)
        assertEquals("김", metadata.valueOf("성"))
        assertEquals("하은", PersonNames.given(metadata))
        assertEquals("김", PersonNames.family(metadata))
        assertEquals("Kim Haeun", PersonNames.join("Kim", "Haeun"))
    }

    @Test
    fun `the stored family name decides the split of two-syllable names`() {
        val metadata = PersonMetadata.of("강", "율", "찬양팀", "")
        assertEquals("강율", metadata.name)
        assertEquals("율", PersonNames.given(metadata))
        // Without the field a two-syllable name is taken as a given name.
        assertEquals("강율", PersonNames.given(PersonMetadata.of("강율", "찬양팀", "")))
    }

    @Test
    fun `older names are split by guessing, double family names included`() {
        assertEquals("하은", PersonNames.given(PersonMetadata.of("김하은", "", "")))
        assertEquals("민수", PersonNames.given(PersonMetadata.of("남궁민수", "", "")))
        assertEquals("E2E-1", PersonNames.given(PersonMetadata.of("E2E-1", "", "")))
        // A renamed person whose stored family name no longer fits falls back to the guess.
        val renamed = PersonMetadata.of("김", "하은", "", "").with(name = "이서연")
        assertEquals("서연", PersonNames.given(renamed))
    }

    @Test
    fun `display picks the chosen parts and falls back to the full name`() {
        val metadata = PersonMetadata.of("김", "하은", "", "")
        assertEquals("김하은", PersonNames.display(metadata, family = true, given = true))
        assertEquals("하은", PersonNames.display(metadata, family = false, given = true))
        assertEquals("김", PersonNames.display(metadata, family = true, given = false))
        assertEquals("김하은", PersonNames.display(metadata, family = false, given = false))
        assertEquals("하은", PersonNames.display(PersonMetadata.of("", "하은", "", ""), family = true, given = false))
    }

    @Test
    fun `withValue sets a column under the spelling already in use`() {
        val metadata = PersonMetadata.of(linkedMapOf("이름" to "홍길동", "부서" to "청년부", "반" to "1반"))
        val next = metadata.withValue("department", "중고등부").withValue("반", "2반").withValue("메모", "새가족")
        assertEquals("중고등부", next["부서"])
        assertEquals("2반", next["반"])
        assertEquals("새가족", next["메모"])
        assertEquals("department", PersonMetadata.canonicalColumn("부서"))
        assertEquals("반", PersonMetadata.canonicalColumn("반"))
    }
}
