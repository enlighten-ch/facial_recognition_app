package com.allstar.faithid.core.attendance

import com.allstar.faithid.core.codec.BinaryReader
import com.allstar.faithid.core.codec.BinaryWriter
import com.allstar.faithid.core.codec.CodecException

/** How the person was identified when they checked in. */
enum class CheckInSource(val label: String) {
    /** Recognized with a clear match, then confirmed by pressing the print button. */
    RECOGNIZED("얼굴 인식"),

    /** Picked from the candidate list when several people looked similar. */
    CANDIDATE_SELECTED("후보 선택"),

    /** "This is me" after the registration duplicate warning. */
    DUPLICATE_CONFIRMED("본인 확인"),

    /** Name typed in for a label without face registration. Not linked to a registered person. */
    LABEL_ONLY("이름표만 출력"),
}

/**
 * One check-in at a kiosk. Created when the person presses the print button and kept in the
 * kiosk outbox until the server stores it. [eventId] makes resending safe.
 */
data class AttendanceEvent(
    val eventId: String,
    /** Registered person, or empty for [CheckInSource.LABEL_ONLY]. */
    val personKey: String,
    val name: String,
    val department: String,
    val grade: String,
    val kioskId: String,
    /** Epoch milliseconds on the kiosk clock. */
    val occurredAt: Long,
    val source: CheckInSource,
    /** Recognition score, or 0 when not recognized. */
    val score: Float,
)

object AttendanceCodec {
    private const val MAGIC = "FAT1"

    fun encode(events: List<AttendanceEvent>): ByteArray {
        val w = BinaryWriter().magic(MAGIC).int(events.size)
        events.forEach { writeEvent(w, it) }
        return w.toByteArray()
    }

    fun decode(bytes: ByteArray): List<AttendanceEvent> {
        val r = BinaryReader(bytes)
        r.expectMagic(MAGIC)
        return List(r.count(minBytesPerItem = 40)) { readEvent(r) }
    }

    fun encodeOne(event: AttendanceEvent): ByteArray = BinaryWriter(256).also { writeEvent(it, event) }.toByteArray()

    fun decodeOne(bytes: ByteArray): AttendanceEvent = readEvent(BinaryReader(bytes))

    private fun writeEvent(w: BinaryWriter, e: AttendanceEvent) {
        w.string(e.eventId).string(e.personKey).string(e.name).string(e.department).string(e.grade)
            .string(e.kioskId).long(e.occurredAt).string(e.source.name).float(e.score)
    }

    private fun readEvent(r: BinaryReader): AttendanceEvent {
        val eventId = r.string()
        val personKey = r.string()
        val name = r.string()
        val department = r.string()
        val grade = r.string()
        val kioskId = r.string()
        val occurredAt = r.long()
        val sourceName = r.string()
        val score = r.float()
        val source = runCatching { CheckInSource.valueOf(sourceName) }.getOrElse { throw CodecException("unknown source $sourceName") }
        if (eventId.isBlank()) throw CodecException("blank event id")
        return AttendanceEvent(eventId, personKey, name, department, grade, kioskId, occurredAt, source, score)
    }
}
