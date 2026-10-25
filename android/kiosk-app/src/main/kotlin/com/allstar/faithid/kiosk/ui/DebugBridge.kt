package com.allstar.faithid.kiosk.ui

import com.allstar.faithid.kiosk.flow.Observation

/**
 * Entry points for automated end-to-end tests on emulators, which have no real face in front of
 * the camera. Only the debug build registers a receiver that uses them (src/debug); in release
 * builds nothing sets or calls these.
 */
object DebugBridge {
    /** Delivers an observation to the running kiosk flow on the main thread. */
    @Volatile
    var observe: ((Observation) -> Unit)? = null

    /** Closes the kiosk screen, as Android may do while another app is in front. */
    @Volatile
    var finishScreen: (() -> Unit)? = null

    /** While true, real camera frames are dropped so injected observations are not overwritten. */
    @Volatile
    var cameraPaused: Boolean = false
}
