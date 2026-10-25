package com.allstar.faithid.kiosk.print

import com.allstar.faithid.kiosk.settings.KioskSettings
import com.allstar.faithid.kiosk.settings.PaperProfile

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.graphics.Bitmap
import com.brother.sdk.lmprinter.Channel
import com.brother.sdk.lmprinter.MediaInfo
import com.brother.sdk.lmprinter.OpenChannelError
import com.brother.sdk.lmprinter.PrintError
import com.brother.sdk.lmprinter.PrinterDriverGenerator
import com.brother.sdk.lmprinter.PrinterModel
import com.brother.sdk.lmprinter.PrinterSearcher
import com.brother.sdk.lmprinter.setting.PrintImageSettings
import com.brother.sdk.lmprinter.setting.QLPrintSettings
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs

data class BrotherPairedPrinter(
    val modelName: String,
    val displayName: String,
    val macAddress: String,
)

data class BrotherPrintResult(
    val success: Boolean,
    val message: String,
)

data class BrotherDetectedMedia(
    val labelSize: QLPrintSettings.LabelSize?,
    val description: String,
    val widthMm: Int?,
    val heightMm: Int?,
    val isInfinite: Boolean,
)

/** Brother QL-820NWB over Bluetooth (Brother Print SDK). */
object BrotherPrinter {
    fun searchPairedBluetoothPrinters(context: Context): List<BrotherPairedPrinter> {
        val adapter = getBluetoothAdapter(context) ?: return emptyList()
        if (!adapter.isEnabled) return emptyList()
        val result = PrinterSearcher.startBluetoothSearch(context)
        return result.channels
            .mapNotNull { channel ->
                val modelName = channel.extraInfo[Channel.ExtraInfoKey.ModelName]?.trim().orEmpty()
                if (!modelName.contains("QL-820NWB", ignoreCase = true)) return@mapNotNull null
                val macAddress = channel.channelInfo?.trim().orEmpty()
                if (macAddress.isBlank()) return@mapNotNull null
                val alias = channel.extraInfo[Channel.ExtraInfoKey.BluetoothAlias]?.trim().orEmpty()
                val displayName = alias.ifBlank { modelName }
                BrotherPairedPrinter(
                    modelName = modelName,
                    displayName = displayName,
                    macAddress = macAddress,
                )
            }
            .distinctBy { it.macAddress }
            .sortedWith(compareBy<BrotherPairedPrinter> { it.displayName }.thenBy { it.macAddress })
    }

    fun printBitmap(
        context: Context,
        macAddress: String,
        bitmap: Bitmap,
        settings: KioskSettings,
        onStatus: ((String) -> Unit)? = null,
        onDiagnosticLog: ((String) -> Unit)? = null,
    ): BrotherPrintResult {
        val cleanMac = macAddress.trim()
        if (cleanMac.isBlank()) {
            return BrotherPrintResult(false, "Brother Bluetooth 주소를 설정해 주세요.")
        }
        val adapter = getBluetoothAdapter(context)
            ?: return BrotherPrintResult(false, "Bluetooth 어댑터를 찾을 수 없습니다.")
        if (!adapter.isEnabled) {
            return BrotherPrintResult(false, "태블릿 Bluetooth가 꺼져 있습니다.")
        }

        onStatus?.invoke("Brother Bluetooth 채널 준비 중...")
        onDiagnosticLog?.invoke("Brother: Bluetooth adapter enabled, target=$cleanMac")
        val channel = Channel.newBluetoothChannel(cleanMac, adapter)
        onStatus?.invoke("Brother Bluetooth 채널 연결 중...")
        val openResult = PrinterDriverGenerator.openChannel(channel)
        if (openResult.error.code != OpenChannelError.ErrorCode.NoError || openResult.driver == null) {
            onDiagnosticLog?.invoke("Brother: openChannel failed code=${openResult.error.code}")
            return BrotherPrintResult(false, "Brother 연결 실패: ${openResult.error.code}")
        }
        onStatus?.invoke("Brother 채널 연결 완료, 출력 데이터 준비 중...")
        onDiagnosticLog?.invoke("Brother: channel connected successfully")

        val tempFile = File.createTempFile("brother_print_", ".png", context.cacheDir)
        return try {
            onStatus?.invoke("Brother 장착 라벨 정보를 확인 중...")
            val detectedMedia = detectInstalledMedia(openResult.driver, onDiagnosticLog)
            onDiagnosticLog?.invoke("Brother: selected media=${detectedMedia.description}")

            onStatus?.invoke("Brother 출력 이미지를 임시 파일로 저장 중...")
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            onDiagnosticLog?.invoke("Brother: temp image saved path=${tempFile.absolutePath}, size=${tempFile.length()}")

            val fallbackLabelSize = resolveLabelSize(resolveSelectedPaper(settings))
            val effectiveLabelSize = detectedMedia.labelSize ?: fallbackLabelSize
            val qlSettings = QLPrintSettings(PrinterModel.QL_820NWB).apply {
                labelSize = effectiveLabelSize
                workPath = context.filesDir.absolutePath
                numCopies = settings.printerCopies.coerceAtLeast(1)
                scaleMode = PrintImageSettings.ScaleMode.FitPaperAspect
                halftone = PrintImageSettings.Halftone.PatternDither
                printQuality = PrintImageSettings.PrintQuality.Best
                printOrientation = PrintImageSettings.Orientation.Portrait
                imageRotation = PrintImageSettings.Rotation.Rotate0
            }
            qlSettings.setAutoCut(true)
            qlSettings.setCutAtEnd(true)

            onDiagnosticLog?.invoke(
                "Brother: print settings labelSize=${qlSettings.labelSize}, detectedMedia=${detectedMedia.description}, copies=${qlSettings.numCopies}, " +
                    "orientation=${qlSettings.printOrientation}, rotation=${qlSettings.imageRotation}",
            )
            onStatus?.invoke("Brother 프린터로 출력 명령 전송 중... (${detectedMedia.description})")
            val printError = openResult.driver.printImage(tempFile.absolutePath, qlSettings)
            if (printError.code == PrintError.ErrorCode.NoError) {
                onStatus?.invoke("Brother 출력 명령 전송 완료")
                onDiagnosticLog?.invoke("Brother: printImage completed successfully")
                BrotherPrintResult(true, "Brother QL-820NWB 출력 전송 완료")
            } else {
                onDiagnosticLog?.invoke("Brother: printImage failed code=${printError.code}")
                BrotherPrintResult(false, "Brother 출력 실패: ${printError.code}")
            }
        } catch (t: Throwable) {
            onDiagnosticLog?.invoke("Brother: exception=${t.javaClass.simpleName}, message=${t.message ?: "no-message"}")
            BrotherPrintResult(false, "Brother 출력 오류: ${t.javaClass.simpleName}")
        } finally {
            onStatus?.invoke("Brother 채널 종료 중...")
            runCatching { openResult.driver.closeChannel() }
                .onFailure { error ->
                    onDiagnosticLog?.invoke("Brother: closeChannel failed=${error.javaClass.simpleName}:${error.message}")
                }
            runCatching { tempFile.delete() }
                .onFailure { error ->
                    onDiagnosticLog?.invoke("Brother: temp file delete failed=${error.javaClass.simpleName}:${error.message}")
                }
        }
    }

    private fun getBluetoothAdapter(context: Context): BluetoothAdapter? {
        val manager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
        return manager?.adapter
    }

    private fun detectInstalledMedia(
        driver: com.brother.sdk.lmprinter.PrinterDriver,
        onDiagnosticLog: ((String) -> Unit)?,
    ): BrotherDetectedMedia {
        val statusResult = driver.getPrinterStatus()
        val statusError = statusResult.error
        if (statusError.code != com.brother.sdk.lmprinter.GetStatusError.ErrorCode.NoError) {
            onDiagnosticLog?.invoke(
                "Brother: getPrinterStatus failed code=${statusError.code}, suggestion=${statusError.errorRecoverySuggestion}",
            )
            return BrotherDetectedMedia(
                labelSize = null,
                description = "라벨 정보 읽기 실패(${statusError.code})",
                widthMm = null,
                heightMm = null,
                isInfinite = false,
            )
        }

        val mediaInfo = statusResult.printerStatus?.mediaInfo
        if (mediaInfo == null) {
            onDiagnosticLog?.invoke("Brother: mediaInfo is null")
            return BrotherDetectedMedia(
                labelSize = null,
                description = "장착 라벨 정보 없음",
                widthMm = null,
                heightMm = null,
                isInfinite = false,
            )
        }

        val detectedLabelSize = mediaInfo.getQLLabelSize()
        val description = buildMediaDescription(mediaInfo, detectedLabelSize)
        onDiagnosticLog?.invoke("Brother: media detected $description")
        return BrotherDetectedMedia(
            labelSize = detectedLabelSize,
            description = description,
            widthMm = mediaInfo.width_mm,
            heightMm = mediaInfo.height_mm,
            isInfinite = mediaInfo.isHeightInfinite,
        )
    }

    fun readInstalledMedia(
        context: Context,
        macAddress: String,
        onDiagnosticLog: ((String) -> Unit)? = null,
    ): BrotherDetectedMedia? {
        val cleanMac = macAddress.trim()
        if (cleanMac.isBlank()) return null
        val adapter = getBluetoothAdapter(context) ?: return null
        if (!adapter.isEnabled) return null

        val channel = Channel.newBluetoothChannel(cleanMac, adapter)
        val openResult = PrinterDriverGenerator.openChannel(channel)
        if (openResult.error.code != OpenChannelError.ErrorCode.NoError || openResult.driver == null) {
            onDiagnosticLog?.invoke("Brother: media read openChannel failed code=${openResult.error.code}")
            return null
        }
        return try {
            detectInstalledMedia(openResult.driver, onDiagnosticLog)
        } finally {
            runCatching { openResult.driver.closeChannel() }
                .onFailure { error ->
                    onDiagnosticLog?.invoke("Brother: media read closeChannel failed=${error.javaClass.simpleName}:${error.message}")
                }
        }
    }

    private fun buildMediaDescription(
        mediaInfo: MediaInfo,
        detectedLabelSize: QLPrintSettings.LabelSize?,
    ): String {
        val width = mediaInfo.width_mm
        val height = mediaInfo.height_mm
        val shape = if (mediaInfo.isHeightInfinite) {
            "${width}mm continuous"
        } else {
            "${width}x${height}mm"
        }
        val mediaType = mediaInfo.mediaType?.name ?: "UnknownType"
        val labelName = detectedLabelSize?.name ?: "UnknownLabel"
        return "$shape / $mediaType / $labelName"
    }

    private fun resolveSelectedPaper(settings: KioskSettings): PaperProfile = settings.selectedPaper

    private fun resolveLabelSize(paper: PaperProfile): QLPrintSettings.LabelSize {
        val width = paper.widthMm.toInt()
        val height = paper.heightMm.toInt()
        return when {
            matches(width, 17) && matches(height, 54) -> QLPrintSettings.LabelSize.DieCutW17H54
            matches(width, 17) && matches(height, 87) -> QLPrintSettings.LabelSize.DieCutW17H87
            matches(width, 23) && matches(height, 23) -> QLPrintSettings.LabelSize.DieCutW23H23
            matches(width, 29) && matches(height, 42) -> QLPrintSettings.LabelSize.DieCutW29H42
            matches(width, 29) && matches(height, 90) -> QLPrintSettings.LabelSize.DieCutW29H90
            matches(width, 38) && matches(height, 90) -> QLPrintSettings.LabelSize.DieCutW38H90
            matches(width, 39) && matches(height, 48) -> QLPrintSettings.LabelSize.DieCutW39H48
            matches(width, 52) && matches(height, 29) -> QLPrintSettings.LabelSize.DieCutW52H29
            matches(width, 54) && matches(height, 29) -> QLPrintSettings.LabelSize.DieCutW54H29
            matches(width, 60) && matches(height, 86) -> QLPrintSettings.LabelSize.DieCutW60H86
            matches(width, 62) && matches(height, 29) -> QLPrintSettings.LabelSize.DieCutW62H29
            matches(width, 62) && matches(height, 60) -> QLPrintSettings.LabelSize.DieCutW62H60
            matches(width, 62) && matches(height, 75) -> QLPrintSettings.LabelSize.DieCutW62H75
            matches(width, 62) && matches(height, 100) -> QLPrintSettings.LabelSize.DieCutW62H100
            matches(width, 102) && matches(height, 51) -> QLPrintSettings.LabelSize.DieCutW102H51
            matches(width, 102) && matches(height, 152) -> QLPrintSettings.LabelSize.DieCutW102H152
            matches(width, 103) && matches(height, 164) -> QLPrintSettings.LabelSize.DieCutW103H164
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 12 -> QLPrintSettings.LabelSize.RollW12
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 29 -> QLPrintSettings.LabelSize.RollW29
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 38 -> QLPrintSettings.LabelSize.RollW38
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 50 -> QLPrintSettings.LabelSize.RollW50
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 54 -> QLPrintSettings.LabelSize.RollW54
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 102 -> QLPrintSettings.LabelSize.RollW102
            closest(width, 12, 29, 38, 50, 54, 62, 102, 103) == 103 -> QLPrintSettings.LabelSize.RollW103
            else -> QLPrintSettings.LabelSize.RollW62
        }
    }

    private fun matches(value: Int, target: Int): Boolean = abs(value - target) <= 2

    private fun closest(value: Int, vararg candidates: Int): Int {
        return candidates.minByOrNull { abs(value - it) } ?: candidates.first()
    }
}
