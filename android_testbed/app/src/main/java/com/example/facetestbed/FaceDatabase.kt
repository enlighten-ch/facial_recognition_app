package com.example.facetestbed

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt

data class PersonRecord(
    val name: String,
    val centroid: FloatArray,
    val embeddings: MutableList<FloatArray>,
)

class FaceDatabase(private val dbFile: File) {
    private val people = mutableListOf<PersonRecord>()

    init {
        load()
    }

    fun isEmpty(): Boolean = people.isEmpty()

    fun upsertPerson(name: String, samples: List<FloatArray>, maxEmbeddingsPerPerson: Int = 100) {
        if (samples.isEmpty()) return

        val cleanedName = sanitizeName(name)
        val pruned = pruneToLimit(samples.map { it.copyOf() }.toMutableList(), maxEmbeddingsPerPerson)
        val centroid = l2Normalize(meanVector(pruned))

        people.removeAll { it.name == cleanedName }
        people.add(PersonRecord(cleanedName, centroid, pruned.toMutableList()))
        save()
    }

    fun rank(embedding: FloatArray, topK: Int = 3): List<Pair<String, Float>> {
        if (people.isEmpty()) return emptyList()

        return people
            .map { it.name to dot(it.centroid, embedding) }
            .sortedByDescending { it.second }
            .take(topK.coerceAtLeast(1))
    }

    fun appendSampleToPerson(name: String, embedding: FloatArray, maxEmbeddingsPerPerson: Int = 100): Boolean {
        val person = people.firstOrNull { it.name == name } ?: return false
        person.embeddings.add(embedding.copyOf())
        val pruned = pruneToLimit(person.embeddings, maxEmbeddingsPerPerson).toMutableList()
        person.embeddings.clear()
        person.embeddings.addAll(pruned)
        val newCentroid = l2Normalize(meanVector(person.embeddings))
        for (i in person.centroid.indices) {
            person.centroid[i] = if (i < newCentroid.size) newCentroid[i] else 0f
        }
        save()
        return true
    }

    private fun load() {
        if (!dbFile.exists()) return
        val raw = runCatching { dbFile.readText() }.getOrNull() ?: return
        val root = runCatching { JSONObject(raw) }.getOrNull() ?: return
        val arr = root.optJSONArray("people") ?: return

        people.clear()
        for (i in 0 until arr.length()) {
            val item = arr.optJSONObject(i) ?: continue
            val name = item.optString("name", "").trim()
            if (name.isEmpty()) continue

            val embeddingsJson = item.optJSONArray("embeddings") ?: JSONArray()
            val embeddings = mutableListOf<FloatArray>()
            for (j in 0 until embeddingsJson.length()) {
                val embJson = embeddingsJson.optJSONArray(j) ?: continue
                embeddings.add(jsonArrayToFloatArray(embJson))
            }
            if (embeddings.isEmpty()) continue

            val centroid = l2Normalize(meanVector(embeddings))
            people.add(PersonRecord(name, centroid, embeddings))
        }
    }

    private fun save() {
        val root = JSONObject()
        val arr = JSONArray()
        for (p in people) {
            val obj = JSONObject()
            obj.put("name", p.name)
            obj.put("centroid", floatArrayToJsonArray(p.centroid))
            val embs = JSONArray()
            for (e in p.embeddings) {
                embs.put(floatArrayToJsonArray(e))
            }
            obj.put("embeddings", embs)
            arr.put(obj)
        }
        root.put("people", arr)
        dbFile.parentFile?.mkdirs()
        dbFile.writeText(root.toString(2))
    }

    private fun pruneToLimit(embeddings: MutableList<FloatArray>, limit: Int): List<FloatArray> {
        val safeLimit = limit.coerceAtLeast(1)
        while (embeddings.size > safeLimit) {
            val centroid = l2Normalize(meanVector(embeddings))
            var dropIdx = 0
            var minScore = Float.MAX_VALUE
            for (i in embeddings.indices) {
                val score = dot(embeddings[i], centroid)
                if (score < minScore) {
                    minScore = score
                    dropIdx = i
                }
            }
            embeddings.removeAt(dropIdx)
        }
        return embeddings
    }

    private fun sanitizeName(raw: String): String {
        val cleaned = buildString {
            raw.trim().forEach { ch ->
                if (ch.isLetterOrDigit() || ch == '_' || ch == '-' || ch.code in 0xAC00..0xD7A3) {
                    append(ch)
                } else {
                    append('_')
                }
            }
        }
        return if (cleaned.isBlank()) "unknown" else cleaned
    }

    private fun meanVector(vectors: List<FloatArray>): FloatArray {
        val dim = vectors.firstOrNull()?.size ?: 0
        val out = FloatArray(dim)
        if (dim == 0) return out
        for (v in vectors) {
            for (i in 0 until dim) out[i] += v[i]
        }
        for (i in 0 until dim) out[i] /= vectors.size.toFloat()
        return out
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += (x * x).toDouble()
        val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(v.size) { i -> v[i] / norm }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var s = 0f
        for (i in 0 until n) s += a[i] * b[i]
        return s
    }

    private fun floatArrayToJsonArray(v: FloatArray): JSONArray {
        val arr = JSONArray()
        for (x in v) arr.put(x.toDouble())
        return arr
    }

    private fun jsonArrayToFloatArray(arr: JSONArray): FloatArray {
        val out = FloatArray(arr.length())
        for (i in 0 until arr.length()) {
            out[i] = arr.optDouble(i, 0.0).toFloat()
        }
        return out
    }
}
