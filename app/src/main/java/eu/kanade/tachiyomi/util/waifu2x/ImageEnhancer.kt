package eu.kanade.tachiyomi.util.waifu2x

import android.content.Context
import coil3.SingletonImageLoader
import coil3.request.ImageRequest
import eu.kanade.tachiyomi.data.coil.enhanced
import eu.kanade.tachiyomi.data.coil.mangaId
import eu.kanade.tachiyomi.data.coil.chapterId
import eu.kanade.tachiyomi.data.coil.pageIndex
import eu.kanade.tachiyomi.data.coil.pageVariant
import eu.kanade.tachiyomi.ui.reader.model.ReaderPage
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import tachiyomi.core.common.util.system.logcat
import eu.kanade.tachiyomi.data.coil.customDecoder
import logcat.LogPriority
import java.util.concurrent.PriorityBlockingQueue
import kotlinx.coroutines.runInterruptible
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ConcurrentHashMap
import coil3.request.CachePolicy
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get

object ImageEnhancer {
    // Watchdog tuning: a request whose native progress stops advancing for
    // WATCHDOG_STALL_MS, or that simply runs longer than WATCHDOG_TOTAL_MS, is
    // aborted so the single worker never wedges the whole queue behind it.
    private const val WATCHDOG_POLL_MS = 2_000L
    private const val WATCHDOG_STALL_MS = 15_000L
    private const val WATCHDOG_TOTAL_MS = 60_000L
    private const val MAX_WATCHDOG_DROPS_PER_PAGE = 2

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val pendingRequests = ConcurrentHashMap<String, Int>()
    private val watchdogDropCounts = ConcurrentHashMap<String, Int>()

    @Volatile
    private var activeRequestStartedAt = 0L

    @Volatile
    private var watchdogLastProgress = -1

    @Volatile
    private var watchdogLastProgressAt = 0L

    @Volatile
    private var watchdogDroppedCurrent = false

    @Volatile
    private var lastWarmUpKey: String? = null
    
    // Priority Queue order:
    // 1. Current visible primary page
    // 2. Current visible secondary page in double-page mode
    // 3. Other promoted/high-priority requests
    // 4. Normal preload requests
    // Then Distance from Target ASC, Seq ASC
    private val queue = PriorityBlockingQueue<EnhanceRequest>()
    private val seqGenerator = AtomicInteger(0)
    private val generation = AtomicInteger(0)
    private val runtimeInitialized = AtomicBoolean(false)
    private val preferences: ReaderPreferences by lazy { Injekt.get() }

    @Volatile
    private var lastResetTime = 0L

    @Volatile
    private var isFirstRequestAfterReset = false

    @Volatile
    private var initialTargetEnqueued = false

    @Volatile
    private var activeMangaId = -1L

    @Volatile
    private var activeChapterId = -1L

    @Volatile
    private var activePageIndex = -1

    @Volatile
    private var activePageVariant = ""

    @Volatile
    private var activeJob: Job? = null

    @Volatile
    private var activeRequest: EnhanceRequest? = null

    @Volatile
    private var nativeResetJob: Job? = null

    // Current page the user is viewing. Used to prioritize requests closest to this page.
    @Volatile
    var targetPageIndex: Int = 0

    @Volatile
    private var targetPageVariant: String = ""

    @Volatile
    private var targetSecondaryPageIndex: Int = -1

    @Volatile
    private var targetSecondaryPageVariant: String = ""

    data class EnhanceRequest(
        val context: Context,
        val mangaId: Long,
        val chapterId: Long,
        val pageIndex: Int,
        val pageVariant: String,
        val dataProvider: () -> Any?,
        val priority: Int, // 1 = promoted/high priority, 0 = preload
        val generation: Int,
        val seq: Int = 0
    ) : Comparable<EnhanceRequest> {
        val cancelled = AtomicBoolean(false)
        val requeueOnCancel = AtomicBoolean(false)

        private fun effectivePriority(): Int {
            return when {
                pageIndex == targetPageIndex && pageVariant == targetPageVariant -> 3
                pageIndex == targetSecondaryPageIndex && pageVariant == targetSecondaryPageVariant -> 2
                priority > 0 -> 1
                else -> 0
            }
        }

        override fun compareTo(other: EnhanceRequest): Int {
            // 1. Effective priority based on current visible spread and promotion state.
            val p = other.effectivePriority().compareTo(effectivePriority()) // Descending
            if (p != 0) return p
            
            // 2. Distance from Target Page (Closer > Farther)
            // Even if multiple pages are "High Priority", the one closest to user focus wins.
            val currentTarget = targetPageIndex
            val dist1 = kotlin.math.abs(pageIndex - currentTarget)
            val dist2 = kotlin.math.abs(other.pageIndex - currentTarget)
            
            val d = dist1.compareTo(dist2) // Ascending (0 distance is best)
            if (d != 0) return d

            // 3. Fallback: FIFO (Older seq first)
            return seq.compareTo(other.seq)
        }
    }

    init {
        // Worker Loop
        scope.launch {
            while (true) {
                try {
                    if (isFirstRequestAfterReset) {
                        val elapsed = System.currentTimeMillis() - lastResetTime
                        if (elapsed < 700) {
                            kotlinx.coroutines.delay(700 - elapsed)
                        }
                        isFirstRequestAfterReset = false
                    }

                    val req = runInterruptible { queue.take() }
                    if (req.generation == generation.get()) {
                        processRequest(req)
                    } else {
                        pendingRequests.remove(req.key, req.generation)
                    }
                } catch (e: Exception) {
                    if (e !is InterruptedException && e !is CancellationException) {
                        logcat(LogPriority.ERROR, e) { "ImageEnhancer: Worker loop error" }
                    }
                }
            }
        }

        // Watchdog Loop: unblocks the worker when a request stops making
        // measurable progress (wedged decode/network path, or native stall).
        scope.launch {
            while (true) {
                delay(WATCHDOG_POLL_MS)
                try {
                    val req = activeRequest ?: continue
                    val startedAt = activeRequestStartedAt
                    if (startedAt == 0L) continue
                    val now = System.currentTimeMillis()

                    val packed = try {
                        Waifu2x.getProgress()
                    } catch (_: Throwable) {
                        -1L
                    }
                    val nativeId = if (packed >= 0) (packed shr 32).toInt() else -1
                    val nativePercent = if (packed >= 0) (packed and 0xFFFFFFFFL).toInt() else -1

                    if (nativeId == req.pageIndex && nativePercent >= 0 && nativePercent != watchdogLastProgress) {
                        watchdogLastProgress = nativePercent
                        watchdogLastProgressAt = now
                    }

                    val lastActivity = maxOf(startedAt, watchdogLastProgressAt)
                    val stalled = now - lastActivity >= WATCHDOG_STALL_MS
                    val overdue = now - startedAt >= WATCHDOG_TOTAL_MS
                    if ((stalled || overdue) && !watchdogDroppedCurrent) {
                        watchdogDroppedCurrent = true
                        val key = requestKey(req.mangaId, req.chapterId, req.pageIndex, req.pageVariant)
                        watchdogDropCounts[key] = (watchdogDropCounts[key] ?: 0) + 1
                        logcat(LogPriority.WARN) {
                            "ImageEnhancer: Watchdog aborting page ${req.pageIndex}/${req.pageVariant} " +
                                "(stalled=${stalled}, overdue=${overdue}, " +
                                "nativeProgress=$nativeId/$nativePercent, dropCount=${watchdogDropCounts[key]})"
                        }
                        try {
                            Waifu2x.abortProcessing()
                        } catch (t: Throwable) {
                            logcat(LogPriority.WARN, t) { "ImageEnhancer: Watchdog failed to signal native abort" }
                        }
                        activeJob?.cancel(
                            CancellationException(
                                "Image enhancement watchdog timeout (page ${req.pageIndex}/${req.pageVariant})",
                            ),
                        )
                    }
                } catch (e: Exception) {
                    if (e !is CancellationException) {
                        logcat(LogPriority.ERROR, e) { "ImageEnhancer: Watchdog loop error" }
                    }
                }
            }
        }
    }

    fun enhance(context: Context, page: ReaderPage, highPriority: Boolean = false) {
        val mangaId = page.chapter.chapter.manga_id ?: -1L
        val chapterId = page.chapter.chapter.id ?: -1L
        
        if (mangaId == -1L || chapterId == -1L) return

        enhanceLazy(
            context = context,
            mangaId = mangaId,
            chapterId = chapterId,
            pageIndex = page.index,
            highPriority = highPriority,
            pageVariant = page.enhancementKeySuffix,
        ) {
            // Streams are opened only after queue de-duplication and on the worker dispatcher.
            page.enhancementStream?.let(::bufferStream)
                ?: page.stream?.let(::bufferStream)
                ?: page.imageUrl
        }
    }

    fun enhance(context: Context, mangaId: Long, chapterId: Long, pageIndex: Int, data: Any, highPriority: Boolean, pageVariant: String = "") {
        enhanceLazy(context, mangaId, chapterId, pageIndex, highPriority, pageVariant) { data }
    }

    fun enhanceLazy(
        context: Context,
        mangaId: Long,
        chapterId: Long,
        pageIndex: Int,
        highPriority: Boolean,
        pageVariant: String = "",
        dataProvider: () -> Any?,
    ) {
        initializeRuntime(context.applicationContext)
        val configHash = currentConfigHash()
        if (
            ImageEnhancementCache.isCached(mangaId, chapterId, pageIndex, configHash, pageVariant) ||
            ImageEnhancementCache.isSkipped(mangaId, chapterId, pageIndex, configHash, pageVariant)
        ) {
            return
        }
        if (ImageEnhancementCache.isSavePending(mangaId, chapterId, pageIndex, pageVariant)) return

        val requestKey = requestKey(mangaId, chapterId, pageIndex, pageVariant)
        if ((watchdogDropCounts[requestKey] ?: 0) >= MAX_WATCHDOG_DROPS_PER_PAGE) {
            // This page repeatedly stalled the worker; leave it as a raw image
            // instead of letting it starve the rest of the queue again.
            return
        }

        val isInitialTargetRequest = !initialTargetEnqueued && pageIndex == targetPageIndex
        val effectiveHighPriority = highPriority || isInitialTargetRequest
        val requestGeneration = generation.get()
        
        val existingGeneration = pendingRequests[requestKey]
        if (existingGeneration != null) {
            if (effectiveHighPriority) {
                 val existingRequest = queue.firstOrNull {
                     it.key == requestKey && it.generation == existingGeneration
                 }
                 if (existingRequest?.priority == 1) return
                 // Upgrade priority: Remove existing (likely Low) and re-add as High
                 val removed = queue.removeIf { 
                     it.mangaId == mangaId &&
                         it.chapterId == chapterId &&
                         it.pageIndex == pageIndex &&
                         it.pageVariant == pageVariant &&
                         it.generation == existingGeneration
                 }
                 if (removed) {
                     logcat(LogPriority.DEBUG) { "ImageEnhancer: Upgrading page $pageIndex/$pageVariant to High Priority" }
                     pendingRequests.remove(requestKey, existingGeneration)
                     // Proceed to add below
                 } else {
                     // Already processing or failed to remove, skip
                     return
                 }
            } else {
                // Already pending and we are Low priority, so skip
                return
            }
        }

        if (pendingRequests.putIfAbsent(requestKey, requestGeneration) != null) return

        if (isInitialTargetRequest) {
            initialTargetEnqueued = true
        }

        val priorityLevel = if (effectiveHighPriority) 1 else 0
        val req = EnhanceRequest(
            context,
            mangaId,
            chapterId,
            pageIndex,
            pageVariant,
            dataProvider,
            priorityLevel,
            requestGeneration,
            seqGenerator.getAndIncrement(),
        )
        queue.offer(req)

        // A visible page may interrupt background work, but another preloaded page should not be
        // discarded just because the reader advanced within the preload window.
        if (
            effectiveHighPriority &&
            activePageIndex >= 0 &&
            (activePageIndex != pageIndex || activePageVariant != pageVariant) &&
            !isFocusedTarget(activePageIndex, activePageVariant)
        ) {
            preemptActiveRequest(
                reason = "visible page requested",
                requeue = activePageIndex > targetPageIndex,
            )
        }
        
        logcat(LogPriority.DEBUG) { "ImageEnhancer: Enqueued page $pageIndex/$pageVariant (priority=$priorityLevel)" }
    }

    fun reset(initialPageIndex: Int = 0) {
        cancelAll(reason = "reset")
        queue.clear()
        pendingRequests.clear()
        targetPageIndex = initialPageIndex
        targetPageVariant = ""
        targetSecondaryPageIndex = -1
        targetSecondaryPageVariant = ""
        seqGenerator.set(0)
        lastResetTime = System.currentTimeMillis()
        isFirstRequestAfterReset = true
        initialTargetEnqueued = false
        logcat(LogPriority.DEBUG) { "ImageEnhancer: Resetting state to page $initialPageIndex" }
    }

    /**
     * Prepare the native upscaler before the first page needs it: extract model
     * assets, initialize the model for the current settings, and run one tiny
     * inference so all GPU pipelines are compiled ahead of real work. Called when
     * the reader opens; the first page then skips the multi-second cold start.
     */
    fun warmUp(context: Context) {
        val appContext = context.applicationContext
        if (!preferences.realCuganEnabled().get()) return
        initializeRuntime(appContext)

        val warmUpKey = "${currentConfigHash()}:model${preferences.realCuganModel().get()}"
        if (lastWarmUpKey == warmUpKey) return
        lastWarmUpKey = warmUpKey

        scope.launch {
            try {
                // A concurrent cancelAll() may still be tearing down the previous
                // native model; wait it out so warm-up is not immediately undone.
                nativeResetJob?.let { job ->
                    runCatching { job.join() }
                }
                val model = preferences.realCuganModel().get()
                val initialized = initModelForCurrentConfig(appContext, model)
                if (!initialized) {
                    logcat(LogPriority.WARN) { "ImageEnhancer: Warm-up could not initialize model $model" }
                    return@launch
                }
                val warmed = runWarmUpInference(model)
                logcat(LogPriority.DEBUG) {
                    "ImageEnhancer: Warm-up complete for model $model (inference=${if (warmed) "ok" else "skipped"})"
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                logcat(LogPriority.WARN, e) { "ImageEnhancer: Warm-up failed" }
            }
        }
    }

    private fun initModelForCurrentConfig(context: Context, model: Int): Boolean {
        val noise = preferences.realCuganNoiseLevel().get()
        val scale = ImageEnhancementCache.getEffectiveScale(
            model,
            preferences.realCuganScale().get(),
            preferences.realEsrganStyle().get(),
        )
        val tileSleepMs = when (preferences.realCuganPerformanceMode().get()) {
            1 -> 5
            2 -> 15
            else -> 0
        }
        val tileSize = preferences.realCuganTileSize().get().coerceAtLeast(32)
        val precision = preferences.realCuganPrecision().get()
        // INT8 is an NPU-only selection: the ncnn fallback always runs FP16 so a
        // QNN failure lands on the fast Vulkan pipeline instead of the slow
        // int8 CPU path.
        val requestedBackend = preferences.realCuganProcessingBackend().get()
        val processingBackend = Waifu2x.resolveProcessingBackend(
            requestedBackend,
            model,
            scale,
        )
        val ncnnPrecision = Waifu2x.resolveVulkanFallbackPrecision(precision, requestedBackend, model, scale)
        val fp16Arithmetic = Waifu2x.resolveVulkanFallbackFp16Arithmetic(
            preferences.realCuganFp16Arithmetic().get(),
            requestedBackend,
            model,
            scale,
        )
        return when (model) {
            0 -> Waifu2x.initRealCugan(context, noise, scale, isPro = false, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic, processingBackend = processingBackend, npuPrecision = precision)
            1 -> Waifu2x.initRealCugan(context, noise, scale, isPro = true, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic, processingBackend = processingBackend, npuPrecision = precision)
            Waifu2x.MODEL_REAL_ESRGAN_ANIME -> Waifu2x.initRealESRGAN(context, scale, style = preferences.realEsrganStyle().get(), tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic, processingBackend = processingBackend, npuPrecision = precision)
            3 -> Waifu2x.initNose(context, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic)
            4 -> Waifu2x.initWaifu2x(context, noise, scale, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic)
            5 -> Waifu2x.initWaifu2xUpconv7(context, noise, scale, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic)
            else -> if (Waifu2x.isW2xExModel(model)) {
                Waifu2x.initW2xEx(context, model, scale = scale, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic, processingBackend = processingBackend, npuPrecision = precision)
            } else {
                Waifu2x.initRealCugan(context, noise, scale, isPro = false, tileSleepMs = tileSleepMs, tileSize = tileSize, precision = ncnnPrecision, fp16Arithmetic = fp16Arithmetic, processingBackend = processingBackend, npuPrecision = precision)
            }
        }
    }

    private fun runWarmUpInference(model: Int): Boolean {
        val dummy = android.graphics.Bitmap.createBitmap(64, 64, android.graphics.Bitmap.Config.ARGB_8888)
        return try {
            dummy.eraseColor(0xFF808080.toInt())
            val result = when (model) {
                0, 1 -> Waifu2x.processRealCugan(dummy, -1)
                Waifu2x.MODEL_REAL_ESRGAN_ANIME -> Waifu2x.processRealESRGAN(dummy, -1)
                3 -> Waifu2x.processNose(dummy, -1)
                4, 5 -> Waifu2x.processWaifu2x(dummy, -1)
                else -> if (Waifu2x.isW2xExModel(model)) Waifu2x.processW2xEx(dummy, -1) else Waifu2x.processRealCugan(dummy, -1)
            }
            result?.recycle()
            result != null
        } finally {
            dummy.recycle()
        }
    }

    fun cancelAll(reason: String = "cancelAll", resetNative: Boolean = true) {
        generation.incrementAndGet()
        queue.clear()
        pendingRequests.clear()
        activeJob?.cancel(CancellationException("Image enhancement cancelled: $reason"))
        activeRequest?.cancelled?.set(true)
        activeJob = null
        activeRequest = null
        activeRequestStartedAt = 0L
        watchdogLastProgress = -1
        watchdogLastProgressAt = 0L
        watchdogDroppedCurrent = false
        watchdogDropCounts.clear()
        // cancelAll tears down the native model; the next reader session must
        // warm it up again.
        lastWarmUpKey = null
        activeMangaId = -1L
        activeChapterId = -1L
        activePageIndex = -1
        activePageVariant = ""
        initialTargetEnqueued = false
        if (resetNative) {
            try {
                Waifu2x.abortProcessing()
            } catch (t: Throwable) {
                logcat(LogPriority.WARN, t) { "ImageEnhancer: Failed to signal native abort" }
            }
            if (nativeResetJob?.isActive != true) {
                nativeResetJob = scope.launch {
                    try {
                        Waifu2x.resetRealCugan()
                    } catch (t: Throwable) {
                        logcat(LogPriority.ERROR, t) { "ImageEnhancer: Failed to reset native upscaler" }
                    }
                }
            }
        }
        logcat(LogPriority.DEBUG) { "ImageEnhancer: Cancelled all enhancement work (reason=$reason)" }
    }

    fun reprioritizeAround(
        pageIndex: Int,
        pageVariant: String = "",
        secondaryPageIndex: Int? = null,
        secondaryPageVariant: String = "",
    ) {
        targetPageIndex = pageIndex
        targetPageVariant = pageVariant
        targetSecondaryPageIndex = secondaryPageIndex ?: -1
        targetSecondaryPageVariant = if (secondaryPageIndex != null) secondaryPageVariant else ""
        preemptActiveRequestIfBehindTarget()
        val snapshot = mutableListOf<EnhanceRequest>()
        queue.drainTo(snapshot)
        if (snapshot.isNotEmpty()) {
            queue.addAll(snapshot)
            logcat(LogPriority.DEBUG) {
                "ImageEnhancer: Reprioritized ${snapshot.size} queued pages around target=$pageIndex/$pageVariant secondary=${targetSecondaryPageIndex}/${targetSecondaryPageVariant}"
            }
        }
    }


    fun hasRequest(mangaId: Long, chapterId: Long, pageIndex: Int, pageVariant: String = ""): Boolean {
        return pendingRequests.containsKey(requestKey(mangaId, chapterId, pageIndex, pageVariant)) ||
            ImageEnhancementCache.isSavePending(mangaId, chapterId, pageIndex, pageVariant)
    }

    fun isFocusedTarget(pageIndex: Int, pageVariant: String = ""): Boolean {
        return (pageIndex == targetPageIndex && pageVariant == targetPageVariant) ||
            (pageIndex == targetSecondaryPageIndex && pageVariant == targetSecondaryPageVariant)
    }

    fun isActivelyProcessing(mangaId: Long, chapterId: Long, pageIndex: Int, pageVariant: String = ""): Boolean {
        return activeMangaId == mangaId &&
            activeChapterId == chapterId &&
            activePageIndex == pageIndex &&
            activePageVariant == pageVariant
    }

    fun cancel(mangaId: Long, chapterId: Long, pageIndex: Int, pageVariant: String = "") {
        val requestKey = requestKey(mangaId, chapterId, pageIndex, pageVariant)
        if (pendingRequests.remove(requestKey) != null) {
             val removed = queue.removeIf { 
                 it.mangaId == mangaId && it.chapterId == chapterId && it.pageIndex == pageIndex && it.pageVariant == pageVariant
             }
             if (removed) {
                 logcat(LogPriority.DEBUG) { "ImageEnhancer: Cancelled page $pageIndex/$pageVariant" }
             }
        }
        if (
            activeMangaId == mangaId &&
            activeChapterId == chapterId &&
            activePageIndex == pageIndex &&
            activePageVariant == pageVariant
        ) {
            preemptActiveRequest("page cancelled")
        }
    }

    fun cancelRequestsLessThan(context: Context, mangaId: Long, chapterId: Long, thresholdPageIndex: Int) {
        queue.removeIf { req ->
            if (req.mangaId == mangaId && req.chapterId == chapterId && req.pageIndex < thresholdPageIndex) {
                pendingRequests.remove(req.key, req.generation)
                logcat(LogPriority.DEBUG) { "ImageEnhancer: Pruned page ${req.pageIndex}/${req.pageVariant} (reason: < $thresholdPageIndex)" }
                true
            } else {
                false
            }
        }
    }

    fun cancelRequestsGreaterThan(context: Context, mangaId: Long, chapterId: Long, thresholdPageIndex: Int) {
        queue.removeIf { req ->
            if (req.mangaId == mangaId && req.chapterId == chapterId && req.pageIndex > thresholdPageIndex) {
                pendingRequests.remove(req.key, req.generation)
                logcat(LogPriority.DEBUG) { "ImageEnhancer: Pruned page ${req.pageIndex}/${req.pageVariant} (reason: > $thresholdPageIndex)" }
                true
            } else {
                false
            }
        }
    }

    private suspend fun processRequest(req: EnhanceRequest) {
        try {
            if (req.generation != generation.get()) return
            activeMangaId = req.mangaId
            activeChapterId = req.chapterId
            activePageIndex = req.pageIndex
            activePageVariant = req.pageVariant
            activeRequest = req
            activeRequestStartedAt = System.currentTimeMillis()
            watchdogLastProgress = -1
            watchdogLastProgressAt = 0L
            watchdogDroppedCurrent = false
            logcat(LogPriority.DEBUG) { "ImageEnhancer: Processing page ${req.pageIndex}/${req.pageVariant} (priority=${req.priority})" }
            val data = req.dataProvider() ?: return
            if (req.generation != generation.get() || req.cancelled.get()) return
            Waifu2x.prepareProcessing()
            if (req.cancelled.get()) return
            val request = ImageRequest.Builder(req.context)
                .data(data)
                .memoryCachePolicy(CachePolicy.DISABLED)
                .customDecoder(true)
                .enhanced(true)
                .mangaId(req.mangaId)
                .chapterId(req.chapterId)
                .pageIndex(req.pageIndex)
                .pageVariant(req.pageVariant)
                .build()
            
            val disposable = SingletonImageLoader.get(req.context).enqueue(request)
            val job = disposable.job
            activeJob = job
            if (req.generation != generation.get() || req.cancelled.get()) {
                Waifu2x.abortProcessing()
                job.cancel(CancellationException("Image enhancement request became obsolete"))
            }
            job.await()
        } finally {
            val shouldRequeue = req.requeueOnCancel.get() && req.generation == generation.get()
            activeMangaId = -1L
            activeChapterId = -1L
            activePageIndex = -1
            activePageVariant = ""
            activeJob = null
            activeRequest = null
            activeRequestStartedAt = 0L
            watchdogLastProgress = -1
            watchdogLastProgressAt = 0L
            watchdogDroppedCurrent = false
            pendingRequests.remove(req.key, req.generation)

            if (shouldRequeue) {
                logcat(LogPriority.DEBUG) {
                    "ImageEnhancer: Re-queueing preempted preload page ${req.pageIndex}/${req.pageVariant}"
                }
                enhanceLazy(
                    context = req.context,
                    mangaId = req.mangaId,
                    chapterId = req.chapterId,
                    pageIndex = req.pageIndex,
                    highPriority = false,
                    pageVariant = req.pageVariant,
                    dataProvider = req.dataProvider,
                )
            }
        }
    }

    private val EnhanceRequest.key: String
        get() = requestKey(mangaId, chapterId, pageIndex, pageVariant)

    private fun requestKey(mangaId: Long, chapterId: Long, pageIndex: Int, pageVariant: String): String {
        return "${mangaId}_${chapterId}_${pageIndex}_${pageVariant}"
    }

    private fun bufferStream(streamFactory: () -> java.io.InputStream): Any? {
        return try {
            streamFactory().use { okio.Buffer().readFrom(it) }
        } catch (e: Exception) {
            logcat(LogPriority.WARN, e) { "ImageEnhancer: Failed to read enhancement source" }
            null
        }
    }

    private fun initializeRuntime(context: Context) {
        ImageEnhancementCache.init(context)
        if (!runtimeInitialized.compareAndSet(false, true)) return

        scope.launch {
            ImageEnhancementCache.checkAndTrim(context)
            updatePerformanceSettings()
        }
        // Keep the enhanced-image cache trimmed during long reading sessions
        // instead of only checking once per process.
        scope.launch {
            while (true) {
                delay(10 * 60 * 1000)
                ImageEnhancementCache.checkAndTrim(context)
            }
        }
        scope.launch {
            preferences.realCuganPerformanceMode().changes().collect {
                updatePerformanceSettings()
            }
        }
        scope.launch {
            preferences.realCuganTileSize().changes().collect {
                updatePerformanceSettings()
            }
        }
    }

    private fun updatePerformanceSettings() {
        val sleepMs = when (preferences.realCuganPerformanceMode().get()) {
            0 -> 0
            1 -> 5
            2 -> 15
            else -> 0
        }
        Waifu2x.updatePerformance(sleepMs, preferences.realCuganTileSize().get().coerceAtLeast(32))
    }

    private fun currentConfigHash(): String {
        return ImageEnhancementCache.getConfigHash(
            noise = preferences.realCuganNoiseLevel().get(),
            scale = preferences.realCuganScale().get(),
            model = preferences.realCuganModel().get(),
            realEsrganStyle = preferences.realEsrganStyle().get(),
            maxWidth = preferences.realCuganMaxSizeWidth().get(),
            maxHeight = preferences.realCuganMaxSizeHeight().get(),
            skipMaxWidth = preferences.realCuganSkipMaxSizeWidth().get(),
            skipMaxHeight = preferences.realCuganSkipMaxSizeHeight().get(),
            tileSize = preferences.realCuganTileSize().get(),
            precision = preferences.realCuganPrecision().get(),
            fp16Arithmetic = preferences.realCuganFp16Arithmetic().get(),
            processingBackend = preferences.realCuganProcessingBackend().get(),
        )
    }

    private fun preemptActiveRequestIfBehindTarget() {
        if (activePageIndex >= 0 && activePageIndex < targetPageIndex) {
            preemptActiveRequest("active page is behind visible target")
        }
    }

    private fun preemptActiveRequest(reason: String, requeue: Boolean = false) {
        val request = activeRequest ?: return
        if (requeue) {
            request.requeueOnCancel.set(true)
        }
        if (!request.cancelled.compareAndSet(false, true)) return

        logcat(LogPriority.DEBUG) {
            "ImageEnhancer: Preempting active page $activePageIndex/$activePageVariant ($reason)"
        }
        Waifu2x.abortProcessing()
        activeJob?.cancel(CancellationException("Image enhancement preempted: $reason"))
    }
}
