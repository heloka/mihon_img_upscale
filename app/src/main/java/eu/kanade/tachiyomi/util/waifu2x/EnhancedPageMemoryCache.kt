package eu.kanade.tachiyomi.util.waifu2x

import android.content.Context
import android.graphics.Bitmap
import android.util.LruCache

/**
 * Memory ring for decoded enhanced pages so page revisits and prefetched neighbors skip the
 * enhanced-file decode entirely. Entries are owned by this cache: eviction and removal recycle
 * them, so views must never recycle a bitmap obtained from [get].
 *
 * The budget grows to `preload pages + visible spread` frames once a real frame size is known,
 * capped by device memory class, mirroring a reader-side ring the same way the disk cache
 * mirrors the model output.
 */
object EnhancedPageMemoryCache {
    private const val LARGE_CAP_BYTES = 512L * 1024 * 1024
    private const val SMALL_CAP_BYTES = 256L * 1024 * 1024
    private const val INITIAL_MAX_BYTES = 32L * 1024 * 1024

    private val lock = Any()
    private var cache = object : LruCache<String, Bitmap>(INITIAL_MAX_BYTES.toInt()) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount

        override fun entryRemoved(evicted: Boolean, key: String, oldValue: Bitmap, newValue: Bitmap?) {
            // A put() replacing a key with the same instance must not recycle the live bitmap.
            if (oldValue !== newValue && !oldValue.isRecycled) oldValue.recycle()
        }
    }

    fun key(mangaId: Long, chapterId: Long, pageIndex: Int, configHash: String, pageVariant: String): String =
        "$mangaId/$chapterId/$pageIndex/$configHash/$pageVariant"

    /** Grows the ring to hold the preload window; never shrinks, so live entries survive. */
    fun configure(context: Context, preloadPages: Int, frameBytes: Int) {
        if (preloadPages <= 0 || frameBytes <= 0) return
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
        val memoryClass = manager?.memoryClass ?: 128
        val largeMemoryClass = manager?.largeMemoryClass ?: 192
        val cap = if (memoryClass >= 256 || largeMemoryClass >= 512) LARGE_CAP_BYTES else SMALL_CAP_BYTES
        val wanted = minOf(cap, frameBytes.toLong() * (preloadPages + 2L)).toInt()
        synchronized(lock) {
            if (wanted > cache.maxSize()) cache.resize(wanted)
        }
    }

    fun get(key: String): Bitmap? = synchronized(lock) { cache.get(key) }

    fun put(key: String, bitmap: Bitmap) {
        if (bitmap.isRecycled || bitmap.byteCount <= 0) return
        synchronized(lock) { cache.put(key, bitmap) }
    }

    fun remove(key: String) {
        synchronized(lock) { cache.remove(key) }
    }

    /** Drops every variant of a page, used when its disk entry is invalidated. */
    fun removePage(mangaId: Long, chapterId: Long, pageIndex: Int) {
        val prefix = "$mangaId/$chapterId/$pageIndex/"
        synchronized(lock) {
            cache.snapshot().keys.filter { it.startsWith(prefix) }.forEach { cache.remove(it) }
        }
    }

    fun removeChapter(mangaId: Long, chapterId: Long) {
        val prefix = "$mangaId/$chapterId/"
        synchronized(lock) {
            cache.snapshot().keys.filter { it.startsWith(prefix) }.forEach { cache.remove(it) }
        }
    }

    fun clear() {
        synchronized(lock) { cache.evictAll() }
    }
}
