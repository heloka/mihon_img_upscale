package eu.kanade.tachiyomi.util.waifu2x

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.IBinder
import android.os.Handler
import android.os.HandlerThread
import android.os.ParcelFileDescriptor
import android.os.Process
import android.os.RemoteException
import android.util.Log
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/** Binder client for the isolated, watchdog-protected HTP process. */
internal object ImageEnhancementNpuClient {
    data class Model(
        val path: String,
        val nativeLibraryDir: String,
        val padding: Int,
        val inputBgr: Boolean,
        val scale: Int,
    ) {
        val key: String get() = "$path|$padding|$inputBgr|$scale"
    }

    private data class Result(
        val success: Boolean,
        val backend: Int,
        val width: Int,
        val height: Int,
        val error: String,
    )

    private data class RequestWatch(
        val model: Model,
        val startedAt: Long,
        val idleTimeoutMs: Long,
        val totalTimeoutMs: Long,
        val label: String,
        val lastProgressAt: AtomicLong = AtomicLong(startedAt),
    )

    private val lock = Any()
    private val requestIds = AtomicInteger(1)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<Result>>()
    private val requestImageIds = ConcurrentHashMap<Int, Int>()
    private val requestWatches = ConcurrentHashMap<Int, RequestWatch>()
    private val watchdogThread = HandlerThread("mihon-qnn-client-watchdog").apply { start() }
    private val watchdogHandler = Handler(watchdogThread.looper)
    private val watchdogTasks = ConcurrentHashMap<Int, Runnable>()
    private val failedModels = ConcurrentHashMap.newKeySet<String>()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile private var remote: IImageEnhancementNpuService? = null
    @Volatile private var activeModelKey: String? = null
    @Volatile private var remotePid: Int = -1
    @Volatile private var applicationContext: Context? = null
    @Volatile private var connectionWaiter: CompletableDeferred<IImageEnhancementNpuService>? = null
    @Volatile private var serviceConnection: ServiceConnection? = null
    @Volatile var lastFailureReason: String? = null
        private set

    private val callback = object : IImageEnhancementNpuCallback.Stub() {
        override fun onProgress(requestId: Int, imageId: Int, percent: Int) {
            requestWatches[requestId]?.lastProgressAt?.set(android.os.SystemClock.elapsedRealtime())
            Log.v(TAG, "Request $requestId progress=$percent%")
            if (requestImageIds[requestId] == imageId) Waifu2x.updateRemoteProgress(imageId, percent)
        }

        override fun onResult(
            requestId: Int,
            success: Boolean,
            backend: Int,
            width: Int,
            height: Int,
            error: String?,
        ) {
            Log.i(TAG, "Request $requestId completed: success=$success backend=$backend error=${error.orEmpty()}")
            pending.remove(requestId)?.complete(
                Result(success, backend, width, height, error.orEmpty()),
            )
        }
    }

    suspend fun initialize(context: Context, model: Model, allowRetry: Boolean = false): Boolean {
        applicationContext = context.applicationContext
        if (allowRetry) failedModels.remove(model.key)
        if (model.key in failedModels) return false
        if (activeModelKey == model.key && remote?.asBinder()?.isBinderAlive == true) return true

        var watchedRequestId = 0
        return try {
            val service = connect(context)
            val requestId = requestIds.getAndIncrement()
            watchedRequestId = requestId
            val completion = CompletableDeferred<Result>()
            pending[requestId] = completion
            startWatchdog(
                requestId,
                RequestWatch(
                    model = model,
                    startedAt = android.os.SystemClock.elapsedRealtime(),
                    idleTimeoutMs = INITIALIZE_TIMEOUT_MS,
                    totalTimeoutMs = INITIALIZE_TIMEOUT_MS,
                    label = "QNN model initialization",
                ),
            )
            Log.i(TAG, "Initializing isolated QNN model for request $requestId")
            service.initialize(
                model.path,
                model.nativeLibraryDir,
                model.padding,
                model.inputBgr,
                requestId,
                callback,
            )
            val result = withTimeout(INITIALIZE_TIMEOUT_MS) { completion.await() }
            clearWatchdog(requestId)
            Log.i(TAG, "Isolated QNN initialization await returned for request $requestId")
            if (result.success) {
                activeModelKey = model.key
                lastFailureReason = null
                true
            } else {
                failModel(model, result.error.ifBlank { "QNN rejected the model" })
                false
            }
        } catch (error: Exception) {
            if (watchedRequestId != 0) clearWatchdog(watchedRequestId)
            if (error is CancellationException && error !is TimeoutCancellationException) throw error
            pending.entries.removeIf { it.value.isCancelled || it.value.isCompleted }
            failModel(model, error.message ?: error.javaClass.simpleName)
            terminateRemoteProcess()
            false
        }
    }

    suspend fun process(context: Context, model: Model, input: Bitmap, imageId: Int): Bitmap? = withContext(Dispatchers.IO) {
        if (!initialize(context, model)) return@withContext null

        val directory = File(context.cacheDir, "qnn-requests")
        if (!directory.isDirectory && !directory.mkdirs()) {
            failModel(model, "Could not create QNN request cache")
            return@withContext null
        }
        val requestId = requestIds.getAndIncrement()
        val inputFile = File(directory, "$requestId-input.png")
        val outputFile = File(directory, "$requestId-output.png")
        try {
            Log.i(TAG, "Preparing isolated QNN page request $requestId (${input.width}x${input.height})")
            check(inputFile.outputStream().use { input.compress(Bitmap.CompressFormat.PNG, 100, it) }) {
                "Could not encode the QNN input image"
            }
            val inputDescriptor = ParcelFileDescriptor.open(inputFile, ParcelFileDescriptor.MODE_READ_ONLY)
            val outputDescriptor = ParcelFileDescriptor.open(
                outputFile,
                ParcelFileDescriptor.MODE_CREATE or ParcelFileDescriptor.MODE_READ_WRITE or ParcelFileDescriptor.MODE_TRUNCATE,
            )
            val service = connect(context)
            Log.i(TAG, "Connected to isolated QNN service for request $requestId")
            val completion = CompletableDeferred<Result>()
            pending[requestId] = completion
            requestImageIds[requestId] = imageId
            val startedAt = android.os.SystemClock.elapsedRealtime()
            startWatchdog(
                requestId,
                RequestWatch(
                    model = model,
                    startedAt = startedAt,
                    idleTimeoutMs = GRAPH_STALL_TIMEOUT_MS,
                    totalTimeoutMs = PAGE_TIMEOUT_MS,
                    label = "QNN page inference",
                ),
            )
            Waifu2x.updateRemoteProgress(imageId, 0)
            try {
                Log.i(TAG, "Dispatching isolated QNN page request $requestId")
                service.process(inputDescriptor, outputDescriptor, requestId, imageId, model.scale, callback)
                Log.i(TAG, "Binder accepted isolated QNN page request $requestId")
            } finally {
                inputDescriptor.close()
                outputDescriptor.close()
            }
            val result = withTimeout(PAGE_TIMEOUT_MS) { completion.await() }
            clearWatchdog(requestId)
            Log.i(TAG, "Isolated QNN page await returned for request $requestId")
            if (!result.success || result.backend != Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU) {
                failModel(model, result.error.ifBlank { "QNN page inference did not complete on HTP" })
                return@withContext null
            }
            check(result.width > 0 && result.height > 0) { "QNN service returned invalid output dimensions" }
            BitmapFactory.decodeFile(outputFile.absolutePath).also { bitmap ->
                if (bitmap == null) failModel(model, "Could not decode the QNN output image")
            }
        } catch (error: Exception) {
            if (error is CancellationException) {
                // Page cancellation (fast page turns, pruning) is routine and says
                // nothing about model health. Abort just this request and keep the
                // warm service process; killing it here would failModel the model
                // and push every later page onto the Vulkan path.
                runCatching { remote?.abort(imageId) }
                throw error
            }
            failModel(model, error.message ?: error.javaClass.simpleName)
            terminateRemoteProcess()
            null
        } finally {
            pending.remove(requestId)
            clearWatchdog(requestId)
            requestImageIds.remove(requestId)
            Waifu2x.clearRemoteProgress(imageId)
            inputFile.delete()
            outputFile.delete()
        }
    }

    fun isActive(): Boolean = activeModelKey != null && remote?.asBinder()?.isBinderAlive == true

    fun abort(imageId: Int) {
        runCatching { remote?.abort(imageId) }
    }

    fun retryLastModel() {
        val model = Waifu2x.currentQnnModel() ?: return
        failedModels.remove(model.key)
        activeModelKey = null
        val context = applicationContext ?: return
        scope.launch {
            initialize(context, model, allowRetry = true)
        }
    }

    fun shutdown() {
        activeModelKey = null
        runCatching { remote?.abort(-1) }
        terminateRemoteProcess()
    }

    private suspend fun connect(context: Context): IImageEnhancementNpuService {
        val cached = remote
        if (cached != null && cached.asBinder().isBinderAlive) return cached

        val waiter: CompletableDeferred<IImageEnhancementNpuService>
        synchronized(lock) {
            val connected = remote
            if (connected != null && connected.asBinder().isBinderAlive) return connected
            waiter = connectionWaiter ?: CompletableDeferred<IImageEnhancementNpuService>().also { pendingConnection ->
                val appContext = context.applicationContext
                val connection = object : ServiceConnection {
                    override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                        val service = IImageEnhancementNpuService.Stub.asInterface(binder)
                        remote = service
                        remotePid = runCatching { service.processId }.getOrDefault(-1)
                        connectionWaiter?.complete(service)
                    }

                    override fun onServiceDisconnected(name: ComponentName) {
                        onRemoteDisconnected("QNN service process died")
                    }

                    override fun onBindingDied(name: ComponentName) {
                        onRemoteDisconnected("QNN service binding died")
                    }

                    override fun onNullBinding(name: ComponentName) {
                        onRemoteDisconnected("QNN service returned an empty binding")
                    }
                }
                serviceConnection = connection
                connectionWaiter = pendingConnection
                val intent = Intent(appContext, ImageEnhancementNpuService::class.java)
                if (!appContext.bindService(intent, connection, Context.BIND_AUTO_CREATE)) {
                    pendingConnection.completeExceptionally(IllegalStateException("Could not bind QNN service"))
                }
            }
        }
        return withTimeout(SERVICE_BIND_TIMEOUT_MS) { waiter.await() }
    }

    private fun onRemoteDisconnected(reason: String) {
        Log.e(TAG, "Isolated QNN Binder disconnected: $reason")
        remote = null
        activeModelKey = null
        remotePid = -1
        connectionWaiter?.completeExceptionally(RemoteException(reason))
        connectionWaiter = null
        pending.values.forEach { it.completeExceptionally(RemoteException(reason)) }
        watchdogTasks.keys.toList().forEach(::clearWatchdog)
        requestWatches.clear()
        pending.clear()
        requestImageIds.values.forEach(Waifu2x::clearRemoteProgress)
        requestImageIds.clear()
    }

    private fun failModel(model: Model, reason: String) {
        failedModels.add(model.key)
        activeModelKey = null
        lastFailureReason = reason
        Log.w(TAG, "QNN model ${model.path} disabled for this session: $reason")
    }

    /**
     * This timer deliberately lives in the reader process. A QNN graph call can
     * wedge every thread in the isolated service, including its own watchdog.
     */
    private fun startWatchdog(requestId: Int, watch: RequestWatch) {
        clearWatchdog(requestId)
        requestWatches[requestId] = watch
        Log.i(TAG, "Arming reader-process watchdog for ${watch.label} request $requestId")
        var lastLoggedSecond = -1L
        val task = object : Runnable {
            override fun run() {
                if (requestWatches[requestId] !== watch) {
                    Log.d(TAG, "Watchdog request $requestId stopped because its request state was cleared")
                    return
                }
                val now = android.os.SystemClock.elapsedRealtime()
                val elapsedSecond = (now - watch.startedAt) / 1_000L
                if (elapsedSecond != lastLoggedSecond) {
                    lastLoggedSecond = elapsedSecond
                    Log.d(
                        TAG,
                        "Main-loop watchdog tick request=$requestId elapsed=${now - watch.startedAt}ms " +
                            "idle=${now - watch.lastProgressAt.get()}ms",
                    )
                }
                val reason = when {
                    now - watch.startedAt >= watch.totalTimeoutMs ->
                        "${watch.label} exceeded ${watch.totalTimeoutMs} ms"
                    now - watch.lastProgressAt.get() >= watch.idleTimeoutMs ->
                        "${watch.label} made no progress for ${watch.idleTimeoutMs} ms"
                    else -> null
                }
                if (reason == null) {
                    watchdogHandler.postDelayed(this, WATCHDOG_INTERVAL_MS)
                    return
                }

                if (requestWatches.remove(requestId, watch)) {
                    Log.e(TAG, "$reason; terminating isolated QNN process from reader process")
                    pending[requestId]?.completeExceptionally(TimeoutException(reason))
                    failModel(watch.model, reason)
                    terminateRemoteProcess()
                }
            }
        }
        watchdogTasks[requestId] = task
        watchdogHandler.postDelayed(task, WATCHDOG_INTERVAL_MS)
    }

    private fun clearWatchdog(requestId: Int) {
        requestWatches.remove(requestId)
        watchdogTasks.remove(requestId)?.let(watchdogHandler::removeCallbacks)
    }

    private fun terminateRemoteProcess() {
        val service = remote
        val pid = remotePid.takeIf { it > 0 } ?: runCatching { service?.processId }.getOrNull() ?: -1
        if (pid > 0 && pid != Process.myPid()) runCatching { Process.killProcess(pid) }
        val context = applicationContext
        val connection = serviceConnection
        if (context != null && connection != null) runCatching { context.unbindService(connection) }
        remote = null
        remotePid = -1
        connectionWaiter = null
        serviceConnection = null
        pending.values.forEach { it.completeExceptionally(RemoteException("QNN service was terminated")) }
        pending.clear()
        requestImageIds.values.forEach(Waifu2x::clearRemoteProgress)
        requestImageIds.clear()
    }

    private const val TAG = "ImageEnhancementNpu"
    private const val SERVICE_BIND_TIMEOUT_MS = 10_000L
    private const val INITIALIZE_TIMEOUT_MS = 25_000L
    private const val PAGE_TIMEOUT_MS = 60_000L
    // The final tiles can finish faster than the 1%-granularity progress
    // counter updates, so a page can legitimately sit at the same percentage
    // for a while before completion; 5s mis-killed healthy denoise3x pages at
    // 60%. 15s still bounds a truly hung graph far below the 60s page limit.
    private const val GRAPH_STALL_TIMEOUT_MS = 15_000L
    private const val WATCHDOG_INTERVAL_MS = 250L
}
