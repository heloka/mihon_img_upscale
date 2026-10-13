package eu.kanade.tachiyomi.util.waifu2x

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class Waifu2xNpuConfigTest {
    @Test
    fun temporarilyDisabledNpuAlwaysResolvesToVulkan() {
        assertTrue(Waifu2x.QUALCOMM_NPU_TEMPORARILY_DISABLED)
        assertEquals(
            Waifu2x.PROCESSING_BACKEND_VULKAN,
            Waifu2x.resolveProcessingBackend(
                Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
                model = 1,
                scale = 2,
            ),
        )
    }

    @Test
    fun npuPrecisionSelectsOnlyPackagedFp16OrW8a16Modes() {
        assertEquals(0, Waifu2x.resolveNpuPrecision(0))
        assertEquals(0, Waifu2x.resolveNpuPrecision(1))
        assertEquals(2, Waifu2x.resolveNpuPrecision(2))
        assertEquals(0, Waifu2x.resolveNpuPrecision(3))
    }

    @Test
    fun npuSupportMatchesPackagedX2ModelFamilies() {
        val supported = listOf(
            0,
            1,
            Waifu2x.MODEL_REAL_ESRGAN_ANIME,
            Waifu2x.MODEL_W2XEX_PHOTO_SMALL,
            Waifu2x.MODEL_SPAN_NOMOSUNI_PHOTO,
        )
        supported.forEach { model ->
            assertTrue(Waifu2x.isQualcommNpuModelSupported(model, 2), "model $model")
            assertFalse(Waifu2x.isQualcommNpuModelSupported(model, 3), "model $model at x3")
        }
        assertFalse(Waifu2x.isQualcommNpuModelSupported(4, 2), "unpackaged Waifu2x model")
    }

    @Test
    fun qnnArtifactFallbackStaysOnTheReportedArchitecture() {
        val assets = setOf(
            "realcugan-pro-x2-no-denoise.v81.model.so",
            "realcugan-pro-x2-no-denoise.v81.bin",
            "realcugan-pro-x2-no-denoise.v85.bin",
        )

        assertEquals(
            listOf(
                "realcugan-pro-x2-no-denoise.v81.model.so",
                "realcugan-pro-x2-no-denoise.v81.bin",
            ),
            Waifu2x.qnnAssetCandidateNames("realcugan-pro-x2-no-denoise", 81, assets),
        )
        assertEquals(
            listOf("realcugan-pro-x2-no-denoise.v85.bin"),
            Waifu2x.qnnAssetCandidateNames("realcugan-pro-x2-no-denoise", 85, assets),
        )
    }

    @Test
    fun serializedQnnContextsMustMatchTheReportedSocModel() {
        assertTrue(Waifu2x.isQnnAssetSocCompatible("context", "97", 97))
        assertFalse(Waifu2x.isQnnAssetSocCompatible("context", "87", 97))
        assertFalse(Waifu2x.isQnnAssetSocCompatible("context", null, 97))
        assertTrue(Waifu2x.isQnnAssetSocCompatible("model-library", "runtime", 97))
        assertFalse(Waifu2x.isQnnAssetSocCompatible("model-library", "87", 97))
    }

    @Test
    fun npuFailureUsesVulkanFp16StorageAndFp32Arithmetic() {
        assertEquals(
            0,
            Waifu2x.resolveVulkanFallbackPrecision(
                requestedPrecision = 1,
                requestedBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
                model = 1,
                scale = 2,
            ),
        )
        assertEquals(
            0,
            Waifu2x.resolveVulkanFallbackPrecision(
                requestedPrecision = 2,
                requestedBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
                model = 1,
                scale = 2,
            ),
        )
        assertFalse(
            Waifu2x.resolveVulkanFallbackFp16Arithmetic(
                requestedFp16Arithmetic = true,
                requestedBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
                model = 1,
                scale = 2,
            ),
        )
        assertTrue(
            Waifu2x.resolveVulkanFallbackFp16Arithmetic(
                requestedFp16Arithmetic = true,
                requestedBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
                model = 1,
                scale = 2,
            ),
        )
    }
}
