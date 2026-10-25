package com.allstar.faithid.serverapp.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.allstar.faithid.server.auth.ClientRecord
import com.allstar.faithid.serverapp.R
import com.allstar.faithid.serverapp.ServerService

/** "현황" tab: server start/stop and the list of kiosks with approve/remove buttons. */
internal class StatusPage(
    private val root: View,
    private val actions: Actions,
) {
    interface Actions {
        fun startServer()
        fun stopServer()
        fun approve(clientId: String)
        fun approveAll()
        fun remove(clientId: String)
    }

    data class KioskRow(val record: ClientRecord, val connections: Int)

    private val statusText: TextView = root.findViewById(R.id.serverStatusText)
    private val endpointText: TextView = root.findViewById(R.id.endpointText)
    private val startButton: Button = root.findViewById(R.id.startButton)
    private val stopButton: Button = root.findViewById(R.id.stopButton)
    private val approveAllButton: Button = root.findViewById(R.id.approveAllButton)
    private val emptyText: TextView = root.findViewById(R.id.clientEmptyText)
    private val adapter = ClientAdapter()

    init {
        root.findViewById<RecyclerView>(R.id.clientList).apply {
            layoutManager = LinearLayoutManager(context)
            adapter = this@StatusPage.adapter
        }
        startButton.setOnClickListener { actions.startServer() }
        stopButton.setOnClickListener { actions.stopServer() }
        approveAllButton.setOnClickListener { actions.approveAll() }
    }

    fun render(service: ServerService?, port: Int, dataDir: String) {
        val server = service?.server
        val running = service?.isRunning == true
        startButton.isEnabled = service != null && !running
        stopButton.isEnabled = running
        statusText.text = when {
            service == null -> "서비스 연결 중..."
            running -> "실행 중 · 등록 ${server?.people?.list()?.size ?: 0}명 · DB revision ${server?.people?.revision ?: 0}\n데이터 폴더: $dataDir"
            service.lastError != null -> "중지됨 · ${service.lastError}"
            else -> "중지됨 · '서버 시작'을 누르면 키오스크가 접속할 수 있습니다.\n데이터 폴더: $dataDir"
        }
        val addresses = Format.localAddresses()
        endpointText.text = if (addresses.isEmpty()) {
            "Wi-Fi 주소를 찾을 수 없습니다. 네트워크 연결을 확인하세요."
        } else {
            "키오스크 설정 > DB 서버 주소: " + addresses.joinToString("  또는  ") { "http://$it:$port" }
        }
        val online = server?.onlineClients().orEmpty()
        val rows = server?.clients?.list().orEmpty()
            .map { KioskRow(it, online[it.clientId] ?: 0) }
            .sortedWith(compareBy<KioskRow> { it.record.approved }.thenBy { it.record.clientId })
        adapter.submit(rows)
        emptyText.visibility = if (rows.isEmpty()) View.VISIBLE else View.GONE
        approveAllButton.isEnabled = rows.any { !it.record.approved }
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
            val context = holder.itemView.context
            val (label, color) = when {
                !record.approved -> "승인 대기" to R.color.warn
                row.connections > 0 -> "연결됨" to R.color.ok
                else -> "오프라인" to R.color.ink_muted
            }
            holder.status.text = label
            holder.status.setTextColor(ContextCompat.getColor(context, color))
            holder.name.text = "${record.clientId}  ·  ${record.deviceName}"
            holder.detail.text = buildString {
                append("요청 ").append(Format.dateTime(record.requestedAt))
                if (record.approved) append(" · 승인 ").append(Format.dateTime(record.approvedAt))
                append(" · 마지막 접속 ").append(Format.ago(record.lastSeenAt))
            }
            holder.approve.visibility = if (record.approved) View.GONE else View.VISIBLE
            holder.approve.setOnClickListener { actions.approve(record.clientId) }
            holder.remove.text = if (record.approved) "승인 해제" else "거절"
            holder.remove.setOnClickListener { actions.remove(record.clientId) }
        }
    }
}
