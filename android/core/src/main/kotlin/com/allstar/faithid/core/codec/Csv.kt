package com.allstar.faithid.core.codec

import java.util.Locale

/**
 * RFC 4180 CSV reading and writing.
 *
 * Files written for people to open in Excel start with a UTF-8 byte order mark; without it Excel
 * shows Korean text as mojibake. Reading strips the mark if present.
 */
object Csv {
    private const val BOM = '﻿'

    fun parse(text: String): List<List<String>> {
        val raw = text.removePrefix(BOM.toString())
        val rows = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val cell = StringBuilder()
        var inQuotes = false
        var i = 0
        fun endCell() {
            row.add(cell.toString())
            cell.setLength(0)
        }
        fun endRow() {
            endCell()
            rows.add(row)
            row = mutableListOf()
        }
        while (i < raw.length) {
            val c = raw[i]
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < raw.length && raw[i + 1] == '"') {
                        cell.append('"')
                        i++
                    } else {
                        inQuotes = false
                    }
                } else {
                    cell.append(c)
                }
            } else {
                when (c) {
                    '"' -> inQuotes = true
                    ',' -> endCell()
                    '\n' -> endRow()
                    '\r' -> {
                        endRow()
                        if (i + 1 < raw.length && raw[i + 1] == '\n') i++
                    }
                    else -> cell.append(c)
                }
            }
            i++
        }
        if (cell.isNotEmpty() || row.isNotEmpty()) endRow()
        return rows.filterNot { it.size == 1 && it[0].isBlank() }
    }

    fun write(rows: List<List<String>>, withBom: Boolean = true): String {
        val out = StringBuilder()
        if (withBom) out.append(BOM)
        rows.forEach { row ->
            out.append(row.joinToString(",") { escape(it) }).append("\r\n")
        }
        return out.toString()
    }

    fun escape(value: String): String {
        val needsQuote = value.any { it == ',' || it == '"' || it == '\n' || it == '\r' } ||
            value.startsWith(" ") || value.endsWith(" ")
        return if (needsQuote) "\"" + value.replace("\"", "\"\"") + "\"" else value
    }
}

/** Column lookup in a CSV header with case-insensitive aliases. */
class CsvHeader(raw: List<String>) {
    val columns: List<String> = raw.map { it.trim().removePrefix("﻿") }
    private val lowered = columns.map { it.lowercase(Locale.US) }

    fun indexOf(vararg aliases: String): Int {
        aliases.forEach { alias ->
            val index = lowered.indexOf(alias.lowercase(Locale.US))
            if (index >= 0) return index
        }
        return -1
    }

    fun indexOf(aliases: Collection<String>): Int = indexOf(*aliases.toTypedArray())
}

fun List<String>.cell(index: Int): String = if (index < 0) "" else getOrNull(index)?.trim().orEmpty()
