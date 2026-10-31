package com.allstar.faithid.kiosk.settings

import android.app.AlertDialog
import android.content.Context
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import com.allstar.faithid.kiosk.print.LabelRenderer
import com.google.android.material.tabs.TabLayout

/** The kiosk settings dialog: seven tabs, saved together. */
object SettingsDialog {

    fun show(
        context: Context,
        current: KioskSettings,
        connectionSummary: String,
        renderer: LabelRenderer,
        printerTools: PrinterTools,
        pickTemplate: ((String?) -> Unit) -> Unit,
        onSave: (KioskSettings) -> Unit,
    ) {
        val tabs = mutableListOf<SettingsTab>()
        val draft: () -> KioskSettings = { tabs.fold(current) { settings, tab -> tab.applyTo(settings) } }
        var template: TemplateTab? = null

        tabs += GeneralTab(context, current)
        tabs += CopyTab(context, current)
        tabs += RecognitionTab(context, current)
        tabs += RegistrationTab(context, current)
        tabs += SyncTab(context, current, connectionSummary)
        tabs += PrintTab(context, current, printerTools) { template?.refresh() }
        template = TemplateTab(context, current, renderer, draft, pickTemplate).also { tabs += it }

        val pages = FrameLayout(context)
        tabs.forEachIndexed { i, tab ->
            tab.view.visibility = if (i == 0) View.VISIBLE else View.GONE
            pages.addView(tab.view, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
        val tabBar = TabLayout(context).apply {
            tabMode = TabLayout.MODE_FIXED
            tabGravity = TabLayout.GRAVITY_FILL
            tabs.forEach { addTab(newTab().setText(it.title)) }
            addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
                override fun onTabSelected(tab: TabLayout.Tab) {
                    tabs.forEachIndexed { i, page -> page.view.visibility = if (i == tab.position) View.VISIBLE else View.GONE }
                    if (tabs[tab.position] === template) template?.refresh()
                }
                override fun onTabUnselected(tab: TabLayout.Tab) = Unit
                override fun onTabReselected(tab: TabLayout.Tab) = Unit
            })
        }
        val height = (context.resources.displayMetrics.heightPixels * 0.62f).toInt()
        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            addView(tabBar)
            addView(pages, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, height))
        }
        AlertDialog.Builder(context)
            .setTitle("설정")
            .setView(root)
            .setNegativeButton("취소", null)
            .setPositiveButton("저장") { _, _ -> onSave(draft()) }
            .create()
            .apply { setOnDismissListener { template?.release() } }
            .also { dialog ->
                dialog.show()
                // Use most of a tablet screen; the template preview and long texts need the room.
                val metrics = context.resources.displayMetrics
                dialog.window?.setLayout((metrics.widthPixels * 0.86f).toInt(), ViewGroup.LayoutParams.WRAP_CONTENT)
            }
    }
}
