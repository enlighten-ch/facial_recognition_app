package com.allstar.faithid.kiosk.print

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.print.PrintAttributes
import android.print.PrintManager
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.mangoslab.nemonicsdk.INPrinterControllerCallback
import com.mangoslab.nemonicsdk.NPrintInfo
import com.mangoslab.nemonicsdk.NPrinter
import com.mangoslab.nemonicsdk.NPrinterController
import com.mangoslab.nemonicsdk.constants.NPrintQuality
import com.mangoslab.nemonicsdk.constants.NPrinterType
import com.mangoslab.nemonicsdk.constants.NResult

/** Progress steps reported while printing; the coordinator maps them to the configured messages. */
enum class PrintStep { CONNECTING, PREPARING, SENDING }

data class PrintOutcome(val success: Boolean, val message: String)

/** One printer type. [print] runs on a background thread unless [needsMainThread]. */
interface LabelPrinter {
    val needsBluetooth: Boolean
    val needsMainThread: Boolean get() = false
    fun print(bitmap: Bitmap, settings: KioskSettings, onStep: (PrintStep) -> Unit, log: (String) -> Unit): PrintOutcome
}

class NemonicLabelPrinter(private val context: Context) : LabelPrinter {
    override val needsBluetooth = true

    override fun print(bitmap: Bitmap, settings: KioskSettings, onStep: (PrintStep) -> Unit, log: (String) -> Unit): PrintOutcome {
        val mac = settings.nemonicPrinterMacAddress.trim()
        if (mac.isEmpty()) return PrintOutcome(false, "네모닉 MAC 주소가 설정되지 않았습니다.")
        val name = settings.printerName.trim().ifBlank { "Nemonic" }
        val controller = NPrinterController(
            context,
            object : INPrinterControllerCallback {
                override fun disconnected() = log("네모닉 연결 종료")
                override fun printProgress(current: Int, total: Int, progress: Int) = onStep(PrintStep.SENDING)
                override fun printComplete(result: Int) = log("네모닉 printComplete result=$result")
            },
        )
        return try {
            onStep(PrintStep.CONNECTING)
            controller.setPrintTimeout(true, settings.printerTimeoutSec.coerceAtLeast(1))
            val printer = NPrinter(NPrinterType.NEMONIC, name, mac)
            val connect = controller.connect(printer)
            if (connect != NResult.OK && connect != NResult.ALREADY_CONNECTED) return PrintOutcome(false, "네모닉 연결 실패: code=$connect")
            onStep(PrintStep.PREPARING)
            val info = NPrintInfo(printer, bitmap)
                .setCopies(settings.printerCopies.coerceAtLeast(1))
                .setEnableLastPageCut(true)
                .setEnableDither(true)
                .setPrintQuality(NPrintQuality.MIDDLE)
            onStep(PrintStep.SENDING)
            val result = controller.print(info)
            if (result == NResult.OK) PrintOutcome(true, "네모닉 출력 전송 완료") else PrintOutcome(false, "네모닉 출력 실패: code=$result")
        } catch (t: Throwable) {
            PrintOutcome(false, "네모닉 출력 오류: ${t.javaClass.simpleName}: ${t.message.orEmpty()}")
        } finally {
            runCatching { controller.disconnect() }
        }
    }
}

class BrotherLabelPrinter(private val context: Context) : LabelPrinter {
    override val needsBluetooth = true

    override fun print(bitmap: Bitmap, settings: KioskSettings, onStep: (PrintStep) -> Unit, log: (String) -> Unit): PrintOutcome {
        val result = BrotherPrinter.printBitmap(
            context,
            settings.brotherPrinterMacAddress,
            bitmap,
            settings,
            onStatus = { status ->
                onStep(
                    when {
                        status.contains("연결") -> PrintStep.CONNECTING
                        status.contains("준비") || status.contains("저장") -> PrintStep.PREPARING
                        else -> PrintStep.SENDING
                    },
                )
            },
            onDiagnosticLog = log,
        )
        return PrintOutcome(result.success, result.message)
    }
}

/**
 * Android's print dialog. Success means the job was handed to the dialog; whether the person then
 * prints or cancels is not reported back, which is why check-ins no longer depend on printing.
 */
class SystemLabelPrinter(private val activity: Activity) : LabelPrinter {
    override val needsBluetooth = false
    override val needsMainThread = true

    override fun print(bitmap: Bitmap, settings: KioskSettings, onStep: (PrintStep) -> Unit, log: (String) -> Unit): PrintOutcome {
        val manager = activity.getSystemService(Context.PRINT_SERVICE) as? PrintManager
            ?: return PrintOutcome(false, "이 기기에서 인쇄 서비스를 찾을 수 없습니다.")
        val paper = settings.selectedPaper
        val widthMils = ((paper.widthMm / 25.4f) * 1000f).toInt().coerceAtLeast(1000)
        val heightMils = ((paper.heightMm / 25.4f) * 1000f).toInt().coerceAtLeast(1000)
        val landscape = settings.printLandscape
        val media = PrintAttributes.MediaSize(
            "custom_${paper.id}",
            paper.name,
            if (landscape) maxOf(widthMils, heightMils) else minOf(widthMils, heightMils),
            if (landscape) minOf(widthMils, heightMils) else maxOf(widthMils, heightMils),
        )
        val attributes = PrintAttributes.Builder()
            .setMediaSize(media)
            .setColorMode(PrintAttributes.COLOR_MODE_MONOCHROME)
            .setMinMargins(PrintAttributes.Margins.NO_MARGINS)
            .setResolution(PrintAttributes.Resolution("label", "Label printer", 300, 300))
            .build()
        val jobName = "${settings.printerJobTitlePrefix.ifBlank { "FaceRecognition" }}-${System.currentTimeMillis()}"
        onStep(PrintStep.SENDING)
        manager.print(jobName, BitmapPrintDocumentAdapter(activity, jobName, bitmap, settings.printerCopies.coerceAtLeast(1)), attributes)
        return PrintOutcome(true, "인쇄 화면으로 보냈습니다.")
    }
}
