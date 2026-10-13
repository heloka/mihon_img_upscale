package eu.kanade.tachiyomi.util.qnn

import android.content.Context
import android.util.Log

/** Runtime HTP capability detection shared by image enhancement and spatial depth. */
object QualcommHtp {
    // Runtime capability detection is independent of model availability. A model
    // is selected only when the packaged manifest declares this exact HTP target.
    private val supportedArchitectures = setOf(69, 73, 75, 79, 81, 85)

    data class PlatformInfo(val htpArchitecture: Int, val socModel: Int)

    @Volatile
    private var cachedPlatformInfo: PlatformInfo? = null

    init {
        runCatching { System.loadLibrary("waifu2x-jni") }
    }

    fun architecture(context: Context): Int? = platformInfo(context)?.htpArchitecture?.takeIf(supportedArchitectures::contains)

    fun architecture(): Int? = platformInfo()?.htpArchitecture?.takeIf(supportedArchitectures::contains)

    fun platformInfo(context: Context? = null): PlatformInfo? {
        cachedPlatformInfo?.let { return it }
        val nativeLibraryDir = context?.applicationInfo?.nativeLibraryDir.orEmpty()
        var info = cachedPlatformInfo
        if (info == null) {
            synchronized(this) {
                info = cachedPlatformInfo
                if (info == null) {
                    val values = runCatching { nativePlatformInfo(nativeLibraryDir) }.getOrNull()
                    info = values?.takeIf { it.size >= 2 }?.let { PlatformInfo(it[0], it[1]) }
                    if (info != null) {
                        cachedPlatformInfo = info
                        Log.d(
                            TAG,
                            "Detected Qualcomm HTP architecture v${info!!.htpArchitecture}, " +
                                "QNN SoC model ${info!!.socModel}",
                        )
                    } else {
                        Log.w(TAG, "Unable to query Qualcomm HTP platform information")
                    }
                }
            }
        }
        return info
    }

    private external fun nativePlatformInfo(nativeLibraryDir: String): IntArray

    private const val TAG = "QualcommHtp"
}
