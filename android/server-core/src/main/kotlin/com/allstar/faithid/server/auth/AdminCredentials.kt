package com.allstar.faithid.server.auth

import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

/**
 * The server app's administrator login. The password is stored as a salted PBKDF2 hash, never in
 * plain text. [mustChangePassword] is set while the factory default is in use, so the first login
 * asks for a new password.
 */
data class AdminCredentials(
    val adminId: String,
    val passwordHash: String,
    val salt: String,
    val mustChangePassword: Boolean,
) {
    fun verify(id: String, password: String): Boolean {
        val idMatches = MessageDigest.isEqual(id.trim().toByteArray(), adminId.toByteArray())
        val hashMatches = MessageDigest.isEqual(hash(password, salt).toByteArray(), passwordHash.toByteArray())
        return idMatches && hashMatches
    }

    fun withPassword(newPassword: String): AdminCredentials {
        require(isAcceptablePassword(newPassword)) { "password too short" }
        val newSalt = newSalt()
        return copy(passwordHash = hash(newPassword, newSalt), salt = newSalt, mustChangePassword = false)
    }

    companion object {
        const val DEFAULT_ID = "admin"
        const val DEFAULT_PASSWORD = "1234"
        const val MIN_PASSWORD_LENGTH = 6
        private const val ITERATIONS = 60_000

        fun defaults(): AdminCredentials = fromPlainText(DEFAULT_ID, DEFAULT_PASSWORD)

        /** Converts the plain-text password of an old `config.yaml`. The default password must be changed. */
        fun fromPlainText(id: String, password: String): AdminCredentials {
            val salt = newSalt()
            return AdminCredentials(
                adminId = id.trim().ifBlank { DEFAULT_ID },
                passwordHash = hash(password, salt),
                salt = salt,
                mustChangePassword = password == DEFAULT_PASSWORD || !isAcceptablePassword(password),
            )
        }

        fun isAcceptablePassword(password: String): Boolean = password.length >= MIN_PASSWORD_LENGTH && password != DEFAULT_PASSWORD

        fun hash(password: String, salt: String): String {
            val spec = PBEKeySpec(password.toCharArray(), Base64.getDecoder().decode(salt), ITERATIONS, 256)
            val bytes = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded
            return Base64.getEncoder().encodeToString(bytes)
        }

        private fun newSalt(): String = Base64.getEncoder().encodeToString(ByteArray(16).also { SecureRandom().nextBytes(it) })
    }
}
