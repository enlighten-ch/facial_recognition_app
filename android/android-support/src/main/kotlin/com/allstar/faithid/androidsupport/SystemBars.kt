package com.allstar.faithid.androidsupport

import android.app.Activity
import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updatePadding

/**
 * Android 15 draws apps behind the status and navigation bars (edge-to-edge is enforced for
 * targetSdk 35). These helpers keep controls out of the bar areas, where touches do not reach the app.
 */
object SystemBars {

    /** Pads [view] by the system bar and display cutout insets, on top of its original padding. */
    fun padForSystemBars(view: View) {
        val start = view.paddingLeft
        val top = view.paddingTop
        val end = view.paddingRight
        val bottom = view.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(view) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.updatePadding(left = start + bars.left, top = top + bars.top, right = end + bars.right, bottom = bottom + bars.bottom)
            insets
        }
        ViewCompat.requestApplyInsets(view)
    }

    /**
     * Hides status and navigation bars for an unattended kiosk screen. A swipe from the edge shows
     * them briefly; they hide again on their own.
     */
    fun hideForKiosk(activity: Activity, hide: Boolean) {
        val controller = WindowCompat.getInsetsController(activity.window, activity.window.decorView)
        if (hide) {
            controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            controller.hide(WindowInsetsCompat.Type.systemBars())
        } else {
            controller.show(WindowInsetsCompat.Type.systemBars())
        }
    }
}
