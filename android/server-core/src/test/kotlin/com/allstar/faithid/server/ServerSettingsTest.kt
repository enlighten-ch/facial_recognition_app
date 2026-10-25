package com.allstar.faithid.server

import com.allstar.faithid.server.auth.AdminCredentials
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ServerSettingsTest {
    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun `old plain-text config is converted to a hash and the default password must change`() {
        val file = File(temp.root, "config.yaml")
        file.writeText("admin_id: \"admin\"\nadmin_password: \"1234\"\ndefault_port: 18081\ndb_storage_dir: \"content://com.android.externalstorage.documents/tree/primary%3AfaceDB\"\n")
        val settings = ServerSettingsStore(file).load()
        assertTrue(settings.credentials.verify("admin", "1234"))
        assertTrue(settings.credentials.mustChangePassword)
        assertEquals(18081, settings.port)
        assertEquals("/sdcard/faceDB", settings.dataDir)
        val text = file.readText()
        assertFalse("plain password removed", text.contains("admin_password:"))
        assertFalse(text.contains("\"1234\""))
        // Reloading keeps the same hash.
        assertTrue(ServerSettingsStore(file).load().credentials.verify("admin", "1234"))
    }

    @Test
    fun `changed password, wrong attempts and round trip`() {
        val file = File(temp.root, "config.yaml")
        val store = ServerSettingsStore(file)
        val changed = store.load().copy(autoStart = true, credentials = AdminCredentials.defaults().withPassword("church-2026"))
        store.save(changed)
        val loaded = store.load()
        assertTrue(loaded.credentials.verify("admin", "church-2026"))
        assertFalse(loaded.credentials.verify("admin", "1234"))
        assertFalse(loaded.credentials.verify("root", "church-2026"))
        assertFalse(loaded.credentials.mustChangePassword)
        assertTrue(loaded.autoStart)
    }

    @Test
    fun `weak passwords are refused`() {
        assertFalse(AdminCredentials.isAcceptablePassword("1234"))
        assertFalse(AdminCredentials.isAcceptablePassword("abc"))
        assertTrue(AdminCredentials.isAcceptablePassword("abcdef"))
        assertTrue(runCatching { AdminCredentials.defaults().withPassword("12") }.isFailure)
    }

    @Test
    fun `storage paths`() {
        assertEquals("/sdcard/faceDB", StoragePaths.normalize(""))
        assertEquals("/sdcard/교회/faceDB", StoragePaths.normalize("content://com.android.externalstorage.documents/tree/primary%3A%EA%B5%90%ED%9A%8C%2FfaceDB"))
        assertEquals("/sdcard/faceDB", StoragePaths.normalize("content://com.android.externalstorage.documents/tree/1234-5678%3Ax"))
        assertEquals("/storage/emulated/0/x", StoragePaths.normalize("file:///storage/emulated/0/x/"))
    }
}
