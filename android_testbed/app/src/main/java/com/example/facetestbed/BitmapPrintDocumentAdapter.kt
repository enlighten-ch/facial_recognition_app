package com.example.facetestbed

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import android.os.Bundle
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.print.PageRange
import android.print.PrintAttributes
import android.print.PrintDocumentAdapter
import android.print.PrintDocumentInfo
import android.print.pdf.PrintedPdfDocument
import java.io.FileOutputStream

class BitmapPrintDocumentAdapter(
    private val context: Context,
    private val jobName: String,
    private val bitmap: Bitmap,
    private val copies: Int,
) : PrintDocumentAdapter() {
    private var pdfDocument: PrintedPdfDocument? = null
    private var pageCount: Int = copies.coerceAtLeast(1)

    override fun onLayout(
        oldAttributes: PrintAttributes?,
        newAttributes: PrintAttributes,
        cancellationSignal: CancellationSignal,
        callback: LayoutResultCallback,
        extras: Bundle?,
    ) {
        if (cancellationSignal.isCanceled) {
            callback.onLayoutCancelled()
            return
        }
        pdfDocument?.close()
        pdfDocument = PrintedPdfDocument(context, newAttributes)
        val info = PrintDocumentInfo.Builder(jobName)
            .setContentType(PrintDocumentInfo.CONTENT_TYPE_DOCUMENT)
            .setPageCount(pageCount)
            .build()
        callback.onLayoutFinished(info, true)
    }

    override fun onWrite(
        pages: Array<out PageRange>,
        destination: ParcelFileDescriptor,
        cancellationSignal: CancellationSignal,
        callback: WriteResultCallback,
    ) {
        val document = pdfDocument ?: run {
            callback.onWriteFailed("PDF document is not ready.")
            return
        }

        try {
            for (pageIndex in 0 until pageCount) {
                if (!containsPage(pages, pageIndex)) continue
                if (cancellationSignal.isCanceled) {
                    callback.onWriteCancelled()
                    return
                }
                val page = document.startPage(pageIndex)
                val content = RectF(page.info.contentRect)
                drawBitmapFitCenter(page, content)
                document.finishPage(page)
            }
            FileOutputStream(destination.fileDescriptor).use { out ->
                document.writeTo(out)
            }
            callback.onWriteFinished(arrayOf(PageRange.ALL_PAGES))
        } catch (_: Throwable) {
            callback.onWriteFailed("출력 문서를 생성하지 못했습니다.")
        } finally {
            document.close()
            pdfDocument = null
        }
    }

    private fun drawBitmapFitCenter(page: PdfDocument.Page, content: RectF) {
        val scale = minOf(
            content.width() / bitmap.width.toFloat(),
            content.height() / bitmap.height.toFloat(),
        )
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val left = content.left + (content.width() - drawWidth) / 2f
        val top = content.top + (content.height() - drawHeight) / 2f
        page.canvas.drawBitmap(bitmap, null, RectF(left, top, left + drawWidth, top + drawHeight), null)
    }

    private fun containsPage(ranges: Array<out PageRange>, pageIndex: Int): Boolean {
        return ranges.any { range -> pageIndex in range.start..range.end }
    }
}
