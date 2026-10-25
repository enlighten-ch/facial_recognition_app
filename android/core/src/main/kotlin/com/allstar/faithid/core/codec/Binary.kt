package com.allstar.faithid.core.codec

import java.io.ByteArrayOutputStream

/** Thrown when a binary payload is truncated or malformed. */
class CodecException(message: String) : RuntimeException(message)

/** Little-endian binary writer. Strings are written as int32 byte length + UTF-8 bytes. */
class BinaryWriter(initialCapacity: Int = 1024) {
    private val out = ByteArrayOutputStream(initialCapacity)

    fun magic(tag: String): BinaryWriter {
        require(tag.length == 4) { "magic must be 4 characters" }
        out.write(tag.toByteArray(Charsets.US_ASCII))
        return this
    }

    fun byte(value: Int): BinaryWriter {
        out.write(value and 0xFF)
        return this
    }

    fun int(value: Int): BinaryWriter {
        out.write(value and 0xFF)
        out.write((value ushr 8) and 0xFF)
        out.write((value ushr 16) and 0xFF)
        out.write((value ushr 24) and 0xFF)
        return this
    }

    fun long(value: Long): BinaryWriter {
        for (shift in 0 until 64 step 8) out.write(((value ushr shift) and 0xFF).toInt())
        return this
    }

    fun float(value: Float): BinaryWriter = int(java.lang.Float.floatToIntBits(value))

    fun string(value: String): BinaryWriter {
        val bytes = value.toByteArray(Charsets.UTF_8)
        int(bytes.size)
        out.write(bytes)
        return this
    }

    fun bytes(value: ByteArray): BinaryWriter {
        int(value.size)
        out.write(value)
        return this
    }

    fun floats(values: FloatArray): BinaryWriter {
        values.forEach { float(it) }
        return this
    }

    fun toByteArray(): ByteArray = out.toByteArray()
}

/** Little-endian binary reader matching [BinaryWriter]. Every read is bounds-checked. */
class BinaryReader(private val data: ByteArray, private var offset: Int = 0) {

    val remaining: Int get() = data.size - offset

    fun expectMagic(tag: String) {
        val actual = readRaw(4).toString(Charsets.US_ASCII)
        if (actual != tag) throw CodecException("expected $tag but found $actual")
    }

    fun byte(): Int = readRaw(1)[0].toInt() and 0xFF

    fun int(): Int {
        ensure(4)
        val value = (data[offset].toInt() and 0xFF) or
            ((data[offset + 1].toInt() and 0xFF) shl 8) or
            ((data[offset + 2].toInt() and 0xFF) shl 16) or
            ((data[offset + 3].toInt() and 0xFF) shl 24)
        offset += 4
        return value
    }

    fun long(): Long {
        ensure(8)
        var value = 0L
        for (i in 0 until 8) value = value or ((data[offset + i].toLong() and 0xFF) shl (8 * i))
        offset += 8
        return value
    }

    fun float(): Float = java.lang.Float.intBitsToFloat(int())

    fun string(): String = readRaw(length()).toString(Charsets.UTF_8)

    fun bytes(): ByteArray = readRaw(length())

    fun floats(count: Int): FloatArray {
        if (count < 0) throw CodecException("negative float count $count")
        ensure(count.toLong() * 4)
        return FloatArray(count) { float() }
    }

    /** Count prefix that must be non-negative and plausible for the remaining payload. */
    fun count(minBytesPerItem: Int = 1): Int {
        val count = int()
        if (count < 0 || count.toLong() * minBytesPerItem > remaining) {
            throw CodecException("implausible item count $count (remaining=$remaining)")
        }
        return count
    }

    private fun length(): Int {
        val size = int()
        if (size < 0) throw CodecException("negative length $size")
        return size
    }

    private fun readRaw(size: Int): ByteArray {
        ensure(size.toLong())
        val out = data.copyOfRange(offset, offset + size)
        offset += size
        return out
    }

    private fun ensure(size: Long) {
        if (size < 0 || offset + size > data.size) {
            throw CodecException("payload truncated: need $size bytes at offset $offset of ${data.size}")
        }
    }
}

/** float32 little-endian packing used for embeddings in storage and on the wire. */
object FloatPacking {
    fun pack(vectors: List<FloatArray>): ByteArray {
        val dim = vectors.firstOrNull()?.size ?: 0
        val out = ByteArray(vectors.size * dim * 4)
        var p = 0
        vectors.forEach { v ->
            require(v.size == dim) { "embeddings must share one dimension" }
            v.forEach { x ->
                val bits = java.lang.Float.floatToIntBits(x)
                out[p++] = bits.toByte()
                out[p++] = (bits ushr 8).toByte()
                out[p++] = (bits ushr 16).toByte()
                out[p++] = (bits ushr 24).toByte()
            }
        }
        return out
    }

    fun unpack(bytes: ByteArray, count: Int, dim: Int, offset: Int = 0): List<FloatArray> {
        if (count <= 0 || dim <= 0) return emptyList()
        val needed = count.toLong() * dim * 4
        if (offset < 0 || offset + needed > bytes.size) {
            throw CodecException("embedding block out of range: offset=$offset count=$count dim=$dim size=${bytes.size}")
        }
        var p = offset
        return List(count) {
            FloatArray(dim) {
                val bits = (bytes[p].toInt() and 0xFF) or
                    ((bytes[p + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[p + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[p + 3].toInt() and 0xFF) shl 24)
                p += 4
                java.lang.Float.intBitsToFloat(bits)
            }
        }
    }
}
