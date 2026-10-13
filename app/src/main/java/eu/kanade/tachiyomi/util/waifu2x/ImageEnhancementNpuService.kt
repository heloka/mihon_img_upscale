package eu.kanade.tachiyomi.util.waifu2x

import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Owns HTP runtime state in a killable process. QNN calls can block inside vendor
 * DSP code, so cancellation in the reader process is not sufficient on its own.
 */
class ImageEnhancementNpuService : Service() {
    private val requests = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "mihon-qnn-requests")
    }
    private val watchdog = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable, "mihon-qnn-watchdog")
    }

    @Volatile
    private var initializedModelKey: String? = null

    private val binder = object : IImageEnhancementNpuService.Stub() {
        override fun getProcessId(): Int = Process.myPid()

        override fun initialize(
            modelPath: String,
            nativeLibraryDir: String,
            padding: Int,
            inputBgr: Boolean,
            requestId: Int,
            callback: IImageEnhancementNpuCallback,
        ) {
            val modelKey = "$modelPath|$padding|$inputBgr"
            if (initializedModelKey == modelKey && Waifu2x.isLocalQnnActive()) {
                report(callback, requestId, success = true)
                return
            }
            requests.execute {
                val timeout = watchdog.schedule(
                    { killStuckProcess("QNN model initialization exceeded 25 seconds") },
                    INITIALIZE_TIMEOUT_MS,
                    TimeUnit.MILLISECONDS,
                )
                try {
                    val active = Waifu2x.initializeQnnModelForService(
                        this@ImageEnhancementNpuService,
                        modelPath,
                        padding,
                        inputBgr,
                    )
                    if (active) initializedModelKey = modelKey
                    report(
                        callback,
                        requestId,
                        success = active,
                        error = if (active) null else "QNN rejected the model context",
                    )
                } catch (error: Throwable) {
                    Log.e(TAG, "QNN service initialization failed", error)
                    report(callback, requestId, success = false, error = error.message ?: error.javaClass.simpleName)
                } finally {
                    timeout.cancel(false)
                }
            }
        }

        override fun process(
            input: ParcelFileDescriptor,
            output: ParcelFileDescriptor,
            requestId: Int,
            imageId: Int,
            outputScale: Int,
            callback: IImageEnhancementNpuCallback,
        ) {
            Log.i(TAG, "Binder received QNN page request $requestId")
            requests.execute {
                var inputBitmap: Bitmap? = null
                var outputBitmap: Bitmap? = null
                var progressWatch: ScheduledFuture<*>? = null
                try {
                    require(Waifu2x.isLocalQnnActive()) { "QNN service has no active graph" }
                    Log.i(TAG, "Decoding input for QNN page request $requestId")
                    inputBitmap = input.use { descriptor ->
                        BitmapFactory.decodeFileDescriptor(descriptor.fileDescriptor)
                    } ?: error("Could not decode the input image descriptor")
                    Log.i(TAG, "Decoded QNN page request $requestId (${inputBitmap!!.width}x${inputBitmap!!.height})")

                    val startedAt = SystemClock.elapsedRealtime()
                    runCatching { callback.onProgress(requestId, imageId, 0) }
                    val lastProgress = AtomicInteger(-1)
                    val lastProgressAt = AtomicLong(startedAt)
                    progressWatch = watchdog.scheduleAtFixedRate(
                        {
                            val packed = runCatching { Waifu2x.getProgress() }.getOrDefault(-1L)
                            val observed = if (packed >= 0L) (packed and 0xFFFFFFFFL).toInt() else -1
                            if (observed >= 0 && lastProgress.getAndSet(observed) != observed) {
                                lastProgressAt.set(SystemClock.elapsedRealtime())
                                Log.v(TAG, "Service request $requestId progress=$observed%")
                                runCatching { callback.onProgress(requestId, imageId, observed.coerceIn(0, 100)) }
                            }
                            val now = SystemClock.elapsedRealtime()
                            if (now - startedAt >= PAGE_TIMEOUT_MS) {
                                killStuckProcess("QNN page execution exceeded 60 seconds")
                            } else if (now - lastProgressAt.get() >= GRAPH_STALL_TIMEOUT_MS) {
                                killStuckProcess("QNN graph execution stopped reporting progress for 5 seconds")
                            }
                        },
                        WATCHDOG_INTERVAL_MS,
                        WATCHDOG_INTERVAL_MS,
                        TimeUnit.MILLISECONDS,
                    )

                    Log.i(TAG, "Starting HTP graph execution for request $requestId")
                    outputBitmap = Waifu2x.processQnnOnly(inputBitmap, imageId, outputScale)
                        ?: error("QNN graph execution failed")
                    output.use { descriptor ->
                        FileOutputStream(descriptor.fileDescriptor).use { stream ->
                            check(outputBitmap!!.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                                "Could not encode the QNN output image"
                            }
                            stream.flush()
                        }
                    }
                    report(
                        callback,
                        requestId,
                        success = true,
                        backend = Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
                        width = outputBitmap!!.width,
                        height = outputBitmap!!.height,
                    )
                } catch (error: Throwable) {
                    Log.e(TAG, "QNN service inference failed for request $requestId", error)
                    runCatching { input.close() }
                    runCatching { output.close() }
                    report(callback, requestId, success = false, error = error.message ?: error.javaClass.simpleName)
                } finally {
                    progressWatch?.cancel(false)
                    inputBitmap?.recycle()
                    outputBitmap?.recycle()
                }
            }
        }

        override fun abort(imageId: Int) {
            if (Waifu2x.getProgressId() == imageId) Waifu2x.abortQnnServiceProcessing()
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        requests.shutdownNow()
        watchdog.shutdownNow()
        Waifu2x.shutdownQnnService()
        super.onDestroy()
    }

    private fun killStuckProcess(reason: String) {
        Log.e(TAG, "$reason; terminating isolated QNN process")
        Process.killProcess(Process.myPid())
    }

    private fun report(
        callback: IImageEnhancementNpuCallback,
        requestId: Int,
        success: Boolean,
        backend: Int = Waifu2x.INFERENCE_BACKEND_NONE,
        width: Int = 0,
        height: Int = 0,
        error: String? = null,
    ) {
        runCatching { callback.onResult(requestId, success, backend, width, height, error.orEmpty()) }
    }

    companion object {
        private const val TAG = "ImageEnhancementNpu"
        private const val INITIALIZE_TIMEOUT_MS = 25_000L
        // Mirrors the client value: final tiles can outpace 1%-granularity
    // progress updates, so 5s mis-killed healthy pages (denoise3x at 60%).
    private const val GRAPH_STALL_TIMEOUT_MS = 15_000L
        private const val PAGE_TIMEOUT_MS = 60_000L
        private const val WATCHDOG_INTERVAL_MS = 250L
    }
}
