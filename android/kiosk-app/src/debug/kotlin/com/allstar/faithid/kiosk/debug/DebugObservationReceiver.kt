package com.allstar.faithid.kiosk.debug

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.allstar.faithid.core.model.Vectors
import com.allstar.faithid.core.recognition.HeadDirection
import com.allstar.faithid.kiosk.flow.Observation
import com.allstar.faithid.kiosk.ui.DebugBridge
import kotlin.random.Random

/**
 * Debug builds only. Lets E2E scripts drive the kiosk without a camera:
 *
 *   adb shell am broadcast -n com.example.facetestbed/com.allstar.faithid.kiosk.debug.DebugObservationReceiver \
 *     -a com.allstar.faithid.kiosk.DEBUG_FACE --es identity alice --es direction LEFT --ei count 3
 *   ... -a com.allstar.faithid.kiosk.DEBUG_NO_FACE
 *   ... -a com.allstar.faithid.kiosk.DEBUG_CAMERA --ez paused true
 *   ... -a com.allstar.faithid.kiosk.DEBUG_FINISH   (closes the kiosk screen; syncing goes on)
 *
 * The same identity always yields samples of the same synthetic face (512-d, ~0.74 apart).
 */
class DebugObservationReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == ACTION_FINISH) {
            DebugBridge.finishScreen?.invoke()
            return
        }
        val observe = DebugBridge.observe ?: return
        when (intent.action) {
            ACTION_CAMERA -> DebugBridge.cameraPaused = intent.getBooleanExtra("paused", true)
            ACTION_NO_FACE -> observe(Observation.NoFace)
            ACTION_FACE -> {
                val identity = intent.getStringExtra("identity") ?: "anonymous"
                val direction = intent.getStringExtra("direction")?.let { name -> HeadDirection.entries.firstOrNull { it.name == name } }
                repeat(intent.getIntExtra("count", 1).coerceIn(1, 50)) {
                    observe(Observation.Face(sample(identity), direction))
                }
            }
        }
    }

    private fun sample(identity: String): FloatArray {
        val baseRandom = Random(identity.hashCode().toLong())
        val base = Vectors.l2Normalize(FloatArray(DIM) { gaussian(baseRandom) })
        val scale = 0.6f / kotlin.math.sqrt(DIM.toFloat())
        return Vectors.l2Normalize(FloatArray(DIM) { base[it] + scale * gaussian(noise) })
    }

    private fun gaussian(random: Random): Float {
        val u = random.nextDouble().coerceAtLeast(1e-12)
        val v = random.nextDouble()
        return (kotlin.math.sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * Math.PI * v)).toFloat()
    }

    companion object {
        const val ACTION_FACE = "com.allstar.faithid.kiosk.DEBUG_FACE"
        const val ACTION_NO_FACE = "com.allstar.faithid.kiosk.DEBUG_NO_FACE"
        const val ACTION_CAMERA = "com.allstar.faithid.kiosk.DEBUG_CAMERA"
        const val ACTION_FINISH = "com.allstar.faithid.kiosk.DEBUG_FINISH"
        private const val DIM = 512
        private val noise = Random(System.nanoTime())
    }
}
