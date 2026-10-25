package com.allstar.faithid.kiosk.print

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.core.content.ContextCompat
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.settings.PrinterModel
import java.util.concurrent.Executors

/**
 * Prints labels with the configured printer, asking for Bluetooth permission first when needed,
 * and reports progress with the messages from the settings.
 */
class PrintCoordinator(
    private val activity: Activity,
    private val renderer: LabelRenderer,
    private val requestBluetoothPermission: (Array<String>, onGranted: () -> Unit, onDenied: () -> Unit) -> Unit,
    private val onStatus: (message: String, success: Boolean?) -> Unit,
    private val log: (String) -> Unit,
) {
    private val worker = Executors.newSingleThreadExecutor { Thread(it, "faith-print") }
    private val main = Handler(Looper.getMainLooper())

    fun print(metadata: PersonMetadata, settings: KioskSettings) {
        if (!settings.printEnabled) {
            onStatus("출력을 사용하지 않도록 설정되어 있습니다.", true)
            return
        }
        val printer = printerFor(settings)
        onStatus(settings.printStatusReadyMessage, null)
        val run = { runPrint(printer, metadata, settings) }
        if (printer.needsBluetooth && !hasBluetoothPermission()) {
            onStatus(settings.printStatusPermissionMessage, null)
            requestBluetoothPermission(bluetoothPermissions(), run) {
                onStatus(settings.printStatusErrorMessage, false)
                log("Bluetooth 권한이 거부되어 출력하지 못했습니다.")
            }
        } else {
            run()
        }
    }

    fun shutdown() = worker.shutdown()

    private fun runPrint(printer: LabelPrinter, metadata: PersonMetadata, settings: KioskSettings) {
        val task = Runnable {
            val outcome = runCatching {
                val bitmap = renderer.renderForPrinter(metadata, settings)
                try {
                    printer.print(bitmap, settings, { step -> post { onStatus(messageFor(step, settings), null) } }, log)
                } finally {
                    if (!printer.needsMainThread) bitmap.recycle()
                }
            }.getOrElse { PrintOutcome(false, "출력 이미지 생성 실패: ${it.javaClass.simpleName}: ${it.message.orEmpty()}") }
            log((if (outcome.success) "출력: " else "출력 오류: ") + outcome.message)
            post { onStatus(if (outcome.success) settings.printStatusSuccessMessage else settings.printStatusErrorMessage, outcome.success) }
        }
        if (printer.needsMainThread) main.post(task) else worker.execute(task)
    }

    private fun printerFor(settings: KioskSettings): LabelPrinter = when (settings.printerModel) {
        PrinterModel.NEMONIC -> NemonicLabelPrinter(activity)
        PrinterModel.BROTHER_QL820NWB -> BrotherLabelPrinter(activity)
        else -> SystemLabelPrinter(activity)
    }

    private fun messageFor(step: PrintStep, settings: KioskSettings) = when (step) {
        PrintStep.CONNECTING -> settings.printStatusConnectingMessage
        PrintStep.PREPARING -> settings.printStatusPreparingMessage
        PrintStep.SENDING -> settings.printStatusSendingMessage
    }

    private fun hasBluetoothPermission() = bluetoothPermissions().all {
        ContextCompat.checkSelfPermission(activity, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun post(block: () -> Unit) {
        main.post(block)
    }

    companion object {
        fun bluetoothPermissions(): Array<String> =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
            } else {
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
            }
    }
}
