package com.allstar.faithid.serverapp.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.server.auth.ClientRecord
import com.allstar.faithid.serverapp.R

/** "키오스크": allow or disconnect kiosks, and the address a new kiosk needs. */
internal class KiosksPage(
    root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun approve(clientId: String)
        fun approveAll()
        fun remove(clientId: String)
    }

    data class KioskRow(val record: ClientRecord, val connections: Int)

    private val endpointText: TextView = root.findViewById(R.id.endpointText)
    private val approveAllButton: Button = root.findViewById(R.id.approveAllButton)
    private val emptyText: TextView = root.findViewById(R.id.clientEmptyText)
    private val adapter = ClientAdapter()

    init {
        root.findViewById<RecyclerView>(R.id.clientList).apply {
            layoutManager = GridLayoutManager(context, 3)
            adapter = this@KiosksPage.adapter
        }
        approveAllButton.setOnClickListener { actions.approveAll() }
    }

    fun render(rows: List<KioskRow>, port: Int) {
        val addresses = Format.localAddresses()
        endpointText.text = if (addresses.isEmpty()) {
            "Wi-Fi 주소를 찾을 수 없습니다. 네트워크 연결을 확인하세요."
        } else {
            addresses.joinToString("  또는  ") { "http://$it:$port" }
        }
        adapter.submit(rows.sortedWith(compareBy<KioskRow> { it.record.approved }.thenBy { it.record.clientId }))
        emptyText.visibility = if (rows.isEmpty()) View.VISIBLE else View.GONE
        approveAllButton.visibility = if (rows.count { !it.record.approved } >= 2) View.VISIBLE else View.GONE
    }

    private inner class ClientAdapter : RecyclerView.Adapter<ClientAdapter.Holder>() {
        private var rows: List<KioskRow> = emptyList()

        fun submit(next: List<KioskRow>) {
            if (next == rows) return
            rows = next
            notifyDataSetChanged()
        }

        inner class Holder(view: View) : RecyclerView.ViewHolder(view) {
            val status: TextView = view.findViewById(R.id.clientStatus)
            val name: TextView = view.findViewById(R.id.clientName)
            val detail: TextView = view.findViewById(R.id.clientDetail)
            val approve: Button = view.findViewById(R.id.clientApproveButton)
            val remove: Button = view.findViewById(R.id.clientRemoveButton)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_client, parent, false))

        override fun getItemCount() = rows.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val row = rows[position]
            val record = row.record
            when {
                !record.approved -> holder.status.pill("허락 대기", PillKind.WAIT)
                row.connections > 0 -> holder.status.pill("● 연결됨", PillKind.IN)
                else -> holder.status.pill("오프라인", PillKind.OUT)
            }
            holder.name.text = record.deviceName.ifBlank { record.clientId }
            holder.detail.text = buildString {
                append(record.clientId)
                append(if (record.approved) " · 마지막 접속 ${Format.ago(record.lastSeenAt)}" else " · ${Format.ago(record.requestedAt)} 요청")
            }
            holder.approve.visibility = if (record.approved) View.GONE else View.VISIBLE
            holder.approve.setOnClickListener { actions.approve(record.clientId) }
            holder.remove.text = holder.itemView.context.getString(if (record.approved) R.string.disconnect else R.string.reject)
            holder.remove.setOnClickListener { actions.remove(record.clientId) }
        }
    }
}
