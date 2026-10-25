package com.allstar.faithid.core.codec

import com.allstar.faithid.core.attendance.AttendanceCodec
import com.allstar.faithid.core.attendance.AttendanceEvent
import com.allstar.faithid.core.attendance.CheckInSource
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.sync.ChangeSet
import com.allstar.faithid.core.sync.ChangeSetCodec
import com.allstar.faithid.core.sync.KeyValueText
import com.allstar.faithid.core.sync.OpResult
import com.allstar.faithid.core.sync.OpsResponse
import com.allstar.faithid.core.sync.PersonOp
import com.allstar.faithid.core.sync.PersonOpCodec
import com.allstar.faithid.core.sync.Tombstone
import com.allstar.faithid.core.sync.UpdateMessage
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BinaryTest {
    @Test
    fun `round trips primitives and strings`() {
        val bytes = BinaryWriter().magic("TEST").byte(200).int(-5).long(Long.MIN_VALUE + 3).float(1.5f).string("한글 ✓").bytes(byteArrayOf(1, 2)).toByteArray()
        val r = BinaryReader(bytes)
        r.expectMagic("TEST")
        assertEquals(200, r.byte())
        assertEquals(-5, r.int())
        assertEquals(Long.MIN_VALUE + 3, r.long())
        assertEquals(1.5f, r.float())
        assertEquals("한글 ✓", r.string())
        assertArrayEquals(byteArrayOf(1, 2), r.bytes())
        assertEquals(0, r.remaining)
    }

    @Test(expected = CodecException::class)
    fun `truncated input fails instead of returning garbage`() {
        val bytes = BinaryWriter().string("hello").toByteArray()
        BinaryReader(bytes.copyOf(bytes.size - 1)).string()
    }

    @Test(expected = CodecException::class)
    fun `wrong magic is rejected`() {
        BinaryReader("ABCD".toByteArray()).expectMagic("FCS1")
    }

    @Test(expected = CodecException::class)
    fun `implausible counts are rejected before allocating`() {
        BinaryReader(BinaryWriter().int(Int.MAX_VALUE).toByteArray()).count(minBytesPerItem = 4)
    }

    @Test
    fun `float packing round trips and validates bounds`() {
        val vectors = listOf(floatArrayOf(1f, -2f, 3.25f), floatArrayOf(0f, 0.5f, -0f))
        val packed = FloatPacking.pack(vectors)
        val back = FloatPacking.unpack(packed, 2, 3)
        assertArrayEquals(vectors[0], back[0], 0f)
        assertArrayEquals(vectors[1], back[1], 0f)
        assertTrue(runCatching { FloatPacking.unpack(packed, 3, 3) }.isFailure)
        assertEquals(0, FloatPacking.unpack(packed, 0, 3).size)
    }
}

class CsvTest {
    @Test
    fun `parses quotes, embedded commas, newlines and a byte order mark`() {
        val text = "﻿이름,비고\r\n\"홍, 길동\",\"줄1\n줄2\"\n\"따옴표 \"\"x\"\"\",\n"
        val rows = Csv.parse(text)
        assertEquals(listOf("이름", "비고"), rows[0])
        assertEquals(listOf("홍, 길동", "줄1\n줄2"), rows[1])
        assertEquals(listOf("따옴표 \"x\"", ""), rows[2])
        assertEquals(3, rows.size)
    }

    @Test
    fun `write then parse is lossless`() {
        val table = listOf(listOf("a", "b,c", "d\"e"), listOf(" lead", "", "줄\n바꿈"))
        val text = Csv.write(table)
        assertTrue(text.startsWith("﻿"))
        assertEquals(table, Csv.parse(text))
    }

    @Test
    fun `header lookup uses aliases case-insensitively`() {
        val header = CsvHeader(listOf("﻿Name", "부서"))
        assertEquals(0, header.indexOf("이름", "name"))
        assertEquals(1, header.indexOf("department", "부서"))
        assertEquals(-1, header.indexOf("grade"))
        assertEquals("", listOf("x").cell(-1))
    }
}

class ProtocolCodecTest {
    private val faces = FakeFaces(seed = 3, dim = 16)

    @Test
    fun `metadata blob keeps order and extra columns`() {
        val metadata = PersonMetadata.of(linkedMapOf("이름" to "홍길동", "부서" to "청년부", "학년" to "1", "비고" to "x"))
        val back = PersonCodec.decodeMetadata(PersonCodec.encodeMetadata(metadata))
        assertEquals(metadata, back)
        assertNull(PersonCodec.decodeMetadata(byteArrayOf(1, 2, 3)))
        assertNull(PersonCodec.decodeMetadata(null))
    }

    @Test
    fun `change set round trip`() {
        val p1 = faces.person("A").copy(revision = 7, updatedAt = 123)
        val p2 = faces.person("B", samples = 1).copy(revision = 9)
        val set = ChangeSet("epoch-1", 9, fullSnapshot = true, persons = listOf(p1, p2), tombstones = listOf(Tombstone("gone", 8)))
        val back = ChangeSetCodec.decode(ChangeSetCodec.encode(set))
        assertEquals("epoch-1", back.serverEpoch)
        assertEquals(9, back.serverRevision)
        assertTrue(back.fullSnapshot)
        assertEquals(listOf(Tombstone("gone", 8)), back.tombstones)
        assertTrue(back.persons[0].sameContentAs(p1))
        assertEquals(7, back.persons[0].revision)
        assertEquals(123, back.persons[0].updatedAt)
        assertTrue(back.persons[1].sameContentAs(p2))
    }

    @Test
    fun `person ops round trip in batches and one by one`() {
        val register = PersonOp.Register("op1", "p_1", PersonMetadata.of("A", "D", "G"), faces.samples("a", 3), 55)
        val add = PersonOp.AddEmbeddings("op2", "p_1", faces.samples("a", 2))
        val back = PersonOpCodec.decode(PersonOpCodec.encode(listOf(register, add)))
        assertEquals(2, back.size)
        val r = back[0] as PersonOp.Register
        assertEquals("op1", r.opId)
        assertEquals("p_1", r.personKey)
        assertEquals("A", r.metadata.name)
        assertEquals(55, r.createdAt)
        assertEquals(3, r.embeddings.size)
        val a = PersonOpCodec.decodeOne(PersonOpCodec.encodeOne(add)) as PersonOp.AddEmbeddings
        assertArrayEquals(add.embeddings[1], a.embeddings[1], 0f)
    }

    @Test
    fun `ops response and update message round trip`() {
        val response = OpsResponse(42, mapOf("a" to OpResult.APPLIED, "b" to OpResult.REJECTED_DELETED))
        assertEquals(response, OpsResponse.decode(response.encode()))
        assertEquals(17L, UpdateMessage.decodeRevision(UpdateMessage.encode(17, "e")))
        assertNull(UpdateMessage.decodeRevision("pong"))
        assertEquals(UpdateMessage.Update(17, "e"), UpdateMessage.decode(UpdateMessage.encode(17, "e")))
        assertEquals(mapOf("k" to "v w"), KeyValueText.decode(KeyValueText.encode("k" to "v\nw")))
    }

    @Test
    fun `attendance events round trip`() {
        val events = listOf(
            AttendanceEvent("e1", "p_1", "홍길동", "청년부", "3학년", "client-1", 1_700_000_000_000, CheckInSource.RECOGNIZED, 0.71f),
            AttendanceEvent("e2", "", "손님", "", "", "client-2", 1_700_000_100_000, CheckInSource.LABEL_ONLY, 0f),
        )
        assertEquals(events, AttendanceCodec.decode(AttendanceCodec.encode(events)))
        assertEquals(events[1], AttendanceCodec.decodeOne(AttendanceCodec.encodeOne(events[1])))
    }
}
