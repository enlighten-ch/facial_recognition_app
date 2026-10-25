package com.allstar.faithid.kiosk.settings

import android.app.AlertDialog
import android.content.Context
import android.widget.ArrayAdapter
import android.widget.ScrollView
import android.widget.Spinner
import com.allstar.faithid.kiosk.print.BrotherDetectedMedia
import com.allstar.faithid.kiosk.print.BrotherPairedPrinter
import java.util.Locale

/** Actions of the print tab that need the activity (Bluetooth permission, background work). */
interface PrinterTools {
    fun findBrotherPrinters(onFound: (List<BrotherPairedPrinter>) -> Unit)
    fun readBrotherMedia(macAddress: String, onRead: (BrotherDetectedMedia?) -> Unit)
}

/** "인쇄": printer model and address, paper sizes and printing messages. */
class PrintTab(
    context: Context,
    s: KioskSettings,
    private val tools: PrinterTools,
    private val onPaperChanged: () -> Unit,
) : SettingsTab {
    override val title = "인쇄"
    private val form = SettingsForm(context)
    override val view = ScrollView(context).apply {
        val pad = form.dp(8)
        form.root.setPadding(pad, pad, pad, pad)
        addView(form.root)
    }

    private val papers = s.paperProfiles.toMutableList()
    private var paperIndex = papers.indexOfFirst { it.id == s.selectedPaperId }.coerceAtLeast(0)

    init { form.section("프린터") }
    private val enabled = form.switch("이름표 출력 (꺼도 출석은 기록됩니다)", s.printEnabled)
    private val model = form.choice(
        "프린터 종류",
        listOf("Android 기본 인쇄", "Brother QL-820NWB", "Nemonic"),
        when (s.printerModel) {
            PrinterModel.BROTHER_QL820NWB -> 1
            PrinterModel.NEMONIC -> 2
            else -> 0
        },
    )
    private val name = form.text("프린터 이름", s.printerName)
    private val brotherMac = form.text("Brother Bluetooth 주소", s.brotherPrinterMacAddress)
    private val nemonicMac = form.text("Nemonic Bluetooth 주소", s.nemonicPrinterMacAddress)
    private val brotherStatus = form.note("Brother는 태블릿 설정에서 먼저 Bluetooth로 페어링한 뒤 아래에서 찾습니다.")

    init {
        form.button("페어링된 Brother 프린터 찾기") {
            brotherStatus.text = "찾는 중..."
            tools.findBrotherPrinters { printers ->
                if (printers.isEmpty()) {
                    brotherStatus.text = "찾지 못했습니다. 태블릿 Bluetooth 페어링을 확인해 주세요."
                    return@findBrotherPrinters
                }
                AlertDialog.Builder(context)
                    .setTitle("Brother 프린터 선택")
                    .setItems(printers.map { "${it.displayName} [${it.macAddress}]" }.toTypedArray()) { _, which ->
                        val picked = printers[which]
                        name.setText(picked.displayName)
                        brotherMac.setText(picked.macAddress)
                        brotherStatus.text = "선택됨: ${picked.displayName}. '장착 용지 읽기'로 용지를 맞추세요."
                    }
                    .setNegativeButton("취소", null)
                    .show()
            }
        }
        form.button("Brother 장착 용지 읽기") {
            val mac = brotherMac.text.toString().trim()
            if (mac.isEmpty()) {
                brotherStatus.text = "먼저 Brother 프린터를 선택해 주세요."
                return@button
            }
            brotherStatus.text = "장착 용지를 읽는 중..."
            tools.readBrotherMedia(mac) { media ->
                val width = media?.widthMm
                if (media == null || width == null) {
                    brotherStatus.text = "용지를 읽지 못했습니다. 용지를 직접 선택해 주세요."
                    return@readBrotherMedia
                }
                val height = (media.heightMm?.takeIf { it > 0 } ?: if (media.isInfinite) 100 else width).toFloat()
                val label = media.labelSize?.name ?: "Brother"
                val id = "brother_${label.lowercase(Locale.US)}_${width}x${if (media.isInfinite) "continuous" else height.toInt().toString()}"
                val profile = PaperProfile(id, if (media.isInfinite) "Brother $label ${width}mm 연속용지" else "Brother $label ${width}x${height.toInt()}", width.toFloat(), height)
                val existing = papers.indexOfFirst { it.id == id }
                if (existing >= 0) papers[existing] = profile else papers += profile
                paperIndex = papers.indexOfFirst { it.id == id }
                refreshPaperSpinner()
                brotherStatus.text = "장착 용지 반영: ${profile.name}"
            }
        }
    }

    private val timing = form.pair("출력 제한 시간(초)", s.printerTimeoutSec, "출력 매수", s.printerCopies)
    private val jobPrefix = form.text("인쇄 작업 이름 앞부분", s.printerJobTitlePrefix)
    private val orientation = form.choice("프린터 출력 방향", listOf("세로", "가로"), if (s.printLandscape) 1 else 0)

    init { form.section("용지") }
    private val paperSpinner: Spinner = form.choice("출력 용지", paperLabels(), paperIndex) { position ->
        paperIndex = position
        onPaperChanged()
    }
    private val newPaperName = form.text("새 용지 이름", "")
    private val newPaperSize = form.pair("가로(mm)", 0, "세로(mm)", 0)
    private val paperStatus = form.note("")

    init {
        form.button("새 용지 추가") {
            val paperName = newPaperName.text.toString().trim()
            val w = newPaperSize.first.text.toString().toFloatOrNull()
            val h = newPaperSize.second.text.toString().toFloatOrNull()
            if (paperName.isEmpty() || w == null || h == null || w <= 0f || h <= 0f) {
                paperStatus.text = "이름과 가로·세로(mm)를 입력해 주세요."
                return@button
            }
            papers += PaperProfile("paper_${System.currentTimeMillis()}", paperName, w, h)
            paperIndex = papers.lastIndex
            refreshPaperSpinner()
            paperStatus.text = "추가했습니다: $paperName"
        }
        form.button("선택한 용지 삭제") {
            if (papers.size <= 1) {
                paperStatus.text = "용지는 하나 이상 있어야 합니다."
                return@button
            }
            val removed = papers.removeAt(paperSpinner.selectedItemPosition.coerceIn(0, papers.lastIndex))
            paperIndex = 0
            refreshPaperSpinner()
            paperStatus.text = "삭제했습니다: ${removed.name}"
        }
        form.section("출력 안내 문구")
    }

    private val ready = form.text("준비", s.printStatusReadyMessage)
    private val permission = form.text("권한 확인", s.printStatusPermissionMessage)
    private val connecting = form.text("연결 중", s.printStatusConnectingMessage)
    private val preparing = form.text("내용 준비", s.printStatusPreparingMessage)
    private val sending = form.text("출력 중", s.printStatusSendingMessage)
    private val success = form.text("완료", s.printStatusSuccessMessage)
    private val error = form.text("오류", s.printStatusErrorMessage)
    private val returnDelay = form.number("출력 후 처음 화면으로 돌아가기까지(초)", s.outputReturnDelaySeconds)
    private val returnMessage = form.text("복귀 안내 ({seconds})", s.messages.returnCountdown)

    private fun paperLabels() = papers.map { "${it.name} (${SettingsForm.trimZero(it.widthMm)}x${SettingsForm.trimZero(it.heightMm)}mm)" }

    private fun refreshPaperSpinner() {
        paperSpinner.adapter = ArrayAdapter(form.context, android.R.layout.simple_spinner_item, paperLabels()).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        paperSpinner.setSelection(paperIndex.coerceIn(0, papers.lastIndex))
        onPaperChanged()
    }

    override fun applyTo(settings: KioskSettings) = settings.copy(
        printEnabled = enabled.isChecked,
        printerModel = when (model.selectedItemPosition) {
            1 -> PrinterModel.BROTHER_QL820NWB
            2 -> PrinterModel.NEMONIC
            else -> PrinterModel.SYSTEM
        },
        printerName = name.text.toString().trim(),
        brotherPrinterMacAddress = brotherMac.text.toString().trim(),
        nemonicPrinterMacAddress = nemonicMac.text.toString().trim(),
        printerTimeoutSec = timing.first.intValue(settings.printerTimeoutSec).coerceIn(1, 120),
        printerCopies = timing.second.intValue(settings.printerCopies).coerceIn(1, 10),
        printerJobTitlePrefix = jobPrefix.text.toString().trim(),
        printLandscape = orientation.selectedItemPosition == 1,
        paperProfiles = papers.toList(),
        selectedPaperId = papers.getOrNull(paperIndex)?.id ?: settings.selectedPaperId,
        printStatusReadyMessage = ready.textOr(settings.printStatusReadyMessage),
        printStatusPermissionMessage = permission.textOr(settings.printStatusPermissionMessage),
        printStatusConnectingMessage = connecting.textOr(settings.printStatusConnectingMessage),
        printStatusPreparingMessage = preparing.textOr(settings.printStatusPreparingMessage),
        printStatusSendingMessage = sending.textOr(settings.printStatusSendingMessage),
        printStatusSuccessMessage = success.textOr(settings.printStatusSuccessMessage),
        printStatusErrorMessage = error.textOr(settings.printStatusErrorMessage),
        outputReturnDelaySeconds = returnDelay.intValue(settings.outputReturnDelaySeconds).coerceIn(1, 60),
        messages = settings.messages.copy(returnCountdown = returnMessage.textOr(settings.messages.returnCountdown)),
    )
}
