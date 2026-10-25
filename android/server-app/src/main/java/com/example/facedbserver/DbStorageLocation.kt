package com.example.facedbserver

object DbStorageLocation {
    const val FACE_DB_DIR = "/sdcard/faceDB"
    const val FACE_DB_TREE_URI = "content://com.android.externalstorage.documents/tree/primary%3AfaceDB"

    fun normalize(spec: String): String {
        val trimmed = spec.trim()
        if (trimmed.isBlank()) return FACE_DB_DIR

        val normalizedSlashes = trimmed.replace('\\', '/')
        val lowered = normalizedSlashes.lowercase()
        return when {
            lowered == "/sdcard/facedb" ||
                lowered == "/storage/emulated/0/facedb" ||
                lowered == "/storage/self/primary/facedb" ||
                lowered == "file:///sdcard/facedb" ||
                lowered == "file:///storage/emulated/0/facedb" ||
                lowered == "file:///storage/self/primary/facedb" ||
                lowered == FACE_DB_TREE_URI.lowercase() ->
                FACE_DB_DIR
            else -> trimmed
        }
    }
}
