package eu.kanade.tachiyomi.util.qnn

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import eu.kanade.tachiyomi.util.waifu2x.Waifu2x
import eu.kanade.tachiyomi.util.waifu2x.ImageEnhancementNpuClient
import java.io.File
import java.io.FileOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlinx.coroutines.runBlocking

@RunWith(AndroidJUnit4::class)
class QualcommHtpDeviceTest {
    @Test
    fun detectsArchitectureWithoutSocModelAndPackagesMatchingContext() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val architecture = QualcommHtp.architecture()

        assertNotNull("No supported HTP architecture was detected", architecture)
        context.assets.open("qnn-contexts/realesrgan-animevideov3-x2.v$architecture.bin").use { input ->
            assertTrue(input.available() > 0)
        }
        Log.i("QualcommHtpDeviceTest", "Detected and verified packaged HTP v$architecture context")
    }

    @Test
    fun initializesEnhancementFromVersionedContext() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        assertTrue(
            Waifu2x.initRealESRGAN(
                context = context,
                scale = 2,
                processingBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
            ),
        )
        assertTrue("Versioned QNN enhancement context did not initialize", Waifu2x.isQnnActive())
    }

    @Test
    fun executesRealEsrganOnHtpWithoutGpuFallback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue(
            "Real-ESRGAN NPU model failed to initialize",
            Waifu2x.initRealESRGAN(
                context = context,
                scale = 2,
                precision = 0,
                npuPrecision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
            ),
        )
        assertTrue("No active QNN graph after model initialization", Waifu2x.isQnnActive())

        val input = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        for (y in 0 until input.height) {
            for (x in 0 until input.width) {
                val checker = ((x / 8 + y / 8) and 1) == 0
                val red = if (checker) 235 else 36
                val green = 40 + (x * 150 / input.width)
                val blue = 35 + (y * 170 / input.height)
                input.setPixel(x, y, Color.rgb(red, green, blue))
            }
        }

        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealESRGAN(input, id = 81001)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Enhancement returned no bitmap", output)
        assertEquals("Output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Output height was not 2x", input.height * 2, output.height)
        assertTrue("QNN graph was replaced or shut down during inference", Waifu2x.isQnnActive())
        saveEvidence(context, "realesrgan-anime-x2-fp16", input, output)
        Log.i(
            "QualcommHtpDeviceTest",
            "Inference result: model=realesrgan-anime-x2 precision=FP16 " +
                "backend=${Waifu2x.lastInferenceBackend()} elapsed=${elapsedMs}ms",
        )
        assertEquals(
            "Image completed on a fallback backend instead of QNN/HTP",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
        Log.i(
            "QualcommHtpDeviceTest",
            "HTP inference passed: model=realesrgan-anime-x2 precision=FP16 " +
                "input=${input.width}x${input.height} output=${output.width}x${output.height} elapsed=${elapsedMs}ms",
        )
    }

    @Test
    fun executesExportedRealEsrganNcnnWeightsOnHtp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-ESRGAN Vulkan reference model failed to initialize",
            Waifu2x.initRealESRGAN(
                context = context,
                scale = 2,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val asset = "realesrgan-animevideov3-x2-ncnn-export.v81.model.so"
        val modelFile = File(context.cacheDir, "qnn-diagnostics/$asset")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        InstrumentationRegistry.getInstrumentation().context.assets.open(asset).use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Exported Real-ESRGAN model failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 18,
                inputBgr = true,
            ),
        )

        val input = createMangaQualityFixture()
        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealESRGAN(input, id = 81101)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Exported Real-ESRGAN HTP graph returned no bitmap", output)
        assertEquals(input.width * 2, output!!.width)
        assertEquals(input.height * 2, output.height)
        saveEvidence(context, "realesrgan-anime-x2-exported-htp", input, output)
        assertEquals(Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU, Waifu2x.lastInferenceBackend())
        Log.i("QualcommHtpDeviceTest", "Exported Real-ESRGAN HTP fixture completed in ${elapsedMs}ms")
    }

    @Test
    fun executesExportedRealEsrgan128TileAcrossMultipleTilesOnHtp() {
        assertIsolatedMultiTileNpuRequest(
            asset = "realesrgan-animevideov3-x2-ncnn-export.v81.model.so",
            label = "realesrgan-anime-x2-128tile-multitile-htp",
        )
    }

    @Test
    fun executesExportedRealEsrgan256TileOnHtp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-ESRGAN Vulkan reference model failed to initialize",
            Waifu2x.initRealESRGAN(
                context = context,
                scale = 2,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val asset = "realesrgan-animevideov3-x2-256.v81.model.so"
        val modelFile = File(context.cacheDir, "qnn-diagnostics/$asset")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        InstrumentationRegistry.getInstrumentation().context.assets.open(asset).use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "256-tile Real-ESRGAN model failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 18,
                inputBgr = true,
            ),
        )

        val input = createMangaQualityFixture()
        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealESRGAN(input, id = 81103)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("256-tile Real-ESRGAN HTP graph returned no bitmap", output)
        assertEquals(input.width * 2, output!!.width)
        assertEquals(input.height * 2, output.height)
        saveEvidence(context, "realesrgan-anime-x2-256tile-htp", input, output)
        assertEquals(Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU, Waifu2x.lastInferenceBackend())
        Log.i("QualcommHtpDeviceTest", "256-tile Real-ESRGAN HTP fixture completed in ${elapsedMs}ms")
    }

    @Test
    fun executesExportedRealEsrgan256TileAcrossMultipleTilesOnHtp() {
        assertIsolatedMultiTileNpuRequest(
            asset = "realesrgan-animevideov3-x2-256.v81.model.so",
            label = "realesrgan-anime-x2-256tile-multitile-htp",
        )
    }

    private fun assertIsolatedMultiTileNpuRequest(asset: String, label: String) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelFile = File(context.cacheDir, "qnn-diagnostics/$asset")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        InstrumentationRegistry.getInstrumentation().context.assets.open(asset).use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        val model = ImageEnhancementNpuClient.Model(
            path = modelFile.absolutePath,
            nativeLibraryDir = context.applicationInfo.nativeLibraryDir,
            padding = 18,
            inputBgr = true,
            scale = 2,
        )
        assertTrue("Real-ESRGAN model failed to initialize in the isolated HTP service", runBlocking {
            ImageEnhancementNpuClient.initialize(context, model)
        })

        val input = createMangaQualityFixture(size = 512)
        val start = SystemClock.elapsedRealtime()
        val output = runBlocking { ImageEnhancementNpuClient.process(context, model, input, imageId = 81104) }
        val elapsedMs = SystemClock.elapsedRealtime() - start
        if (output != null) {
            assertEquals(input.width * 2, output.width)
            assertEquals(input.height * 2, output.height)
            saveEvidence(context, label, input, output)
            output.recycle()
            assertTrue("HTP service should stay alive after a successful page", ImageEnhancementNpuClient.isActive())
        } else {
            saveInputEvidence(context, "$label-failed", input)
            assertTrue("Stalled HTP service was not terminated promptly", elapsedMs < 15_000)
            assertNull("A timed-out HTP request must not return a partial bitmap", output)
            assertTrue("Timed-out HTP model should be disabled until retry", !ImageEnhancementNpuClient.isActive())
        }
        Log.i("QualcommHtpDeviceTest", "Isolated multi-tile HTP request finished in ${elapsedMs}ms, returnedOutput=${output != null}")
    }

    @Test
    fun executesRealEsrganVulkanReferenceForNpuQualityComparison() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-ESRGAN Vulkan reference model failed to initialize",
            Waifu2x.initRealESRGAN(
                context = context,
                scale = 2,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val input = createMangaQualityFixture()
        val output = Waifu2x.processRealESRGAN(input, id = 81102)
        assertNotNull("Real-ESRGAN Vulkan reference returned no bitmap", output)
        saveEvidence(context, "realesrgan-anime-x2-vulkan-fixture", input, output!!)
        assertEquals(Waifu2x.INFERENCE_BACKEND_VULKAN, Waifu2x.lastInferenceBackend())
    }

    @Test
    fun executesRealCuganSeOnHtpWithoutGpuFallback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue(
            "Real-CUGAN SE NPU model failed to initialize",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = false,
                precision = 0,
                npuPrecision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
            ),
        )
        assertTrue("No active QNN graph after model initialization", Waifu2x.isQnnActive())

        val input = createFixedInput()
        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealCugan(input, id = 81002)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Enhancement returned no bitmap", output)
        assertEquals("Output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "realcugan-se-x2-fp16", input, output)
        Log.i(
            "QualcommHtpDeviceTest",
            "Inference result: model=realcugan-se-x2-no-denoise precision=FP16 " +
                "backend=${Waifu2x.lastInferenceBackend()} elapsed=${elapsedMs}ms",
        )
        assertEquals(
            "Image completed on a fallback backend instead of QNN/HTP",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProFp16OnHtpWithoutGpuFallback() {
        runRealCuganProInference(precision = 0, label = "fp16")
    }

    @Test
    fun executesNearestNeighborDiagnosticGraphOnHtpWithExactPixelOutput() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-nearest-x2.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-nearest-x2.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Nearest-neighbor diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(context, modelFile.absolutePath),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81003)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        var maximumChannelError = 0
        for (y in 0 until output.height) {
            for (x in 0 until output.width) {
                val expected = input.getPixel(x / 2, y / 2)
                val actual = output.getPixel(x, y)
                maximumChannelError = maxOf(
                    maximumChannelError,
                    kotlin.math.abs(Color.red(expected) - Color.red(actual)),
                    kotlin.math.abs(Color.green(expected) - Color.green(actual)),
                    kotlin.math.abs(Color.blue(expected) - Color.blue(actual)),
                )
            }
        }
        saveEvidence(context, "diagnostic-nearest-x2-htp", input, output)
        assertEquals("QNN transport changed nearest-neighbor pixels", 0, maximumChannelError)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesGlobalMeanGateDiagnosticGraphOnHtp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-global-mean-gate-x2.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-global-mean-gate-x2.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Global-mean gate diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(context, modelFile.absolutePath),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81004)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        val channelMeans = DoubleArray(3)
        for (y in 0 until input.height) {
            for (x in 0 until input.width) {
                val color = input.getPixel(x, y)
                channelMeans[0] += Color.red(color) / 255.0
                channelMeans[1] += Color.green(color) / 255.0
                channelMeans[2] += Color.blue(color) / 255.0
            }
        }
        for (channel in 0..2) {
            channelMeans[channel] /= input.width * input.height
            channelMeans[channel] = 1.0 / (1.0 + kotlin.math.exp(-channelMeans[channel]))
        }
        var maximumChannelError = 0
        for (y in 0 until output.height) {
            for (x in 0 until output.width) {
                val source = input.getPixel(x / 2, y / 2)
                val expectedRed = (Color.red(source) * channelMeans[0]).toInt()
                val expectedGreen = (Color.green(source) * channelMeans[1]).toInt()
                val expectedBlue = (Color.blue(source) * channelMeans[2]).toInt()
                val actual = output.getPixel(x, y)
                maximumChannelError = maxOf(
                    maximumChannelError,
                    kotlin.math.abs(expectedRed - Color.red(actual)),
                    kotlin.math.abs(expectedGreen - Color.green(actual)),
                    kotlin.math.abs(expectedBlue - Color.blue(actual)),
                )
            }
        }
        saveEvidence(context, "diagnostic-global-mean-gate-x2-htp", input, output)
        Log.i("QualcommHtpDeviceTest", "GlobalMeanGate maximumChannelError=$maximumChannelError")
        assertTrue("QNN global-mean/broadcast output differs from reference: $maximumChannelError", maximumChannelError <= 2)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProFirstBlockDiagnosticOnHtp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-pro-first-leaky-x2.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-pro-first-leaky-x2.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro first-block diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = true,
            ),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81005)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "diagnostic-pro-first-leaky-x2-htp", input, output)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProDownsampleDiagnosticOnHtp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-pro-downsample.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-pro-downsample.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro downsample diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = true,
            ),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81006)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "diagnostic-pro-downsample-htp", input, output)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProConv2BlockDiagnosticOnHtp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-pro-conv2-block.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-pro-conv2-block.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro conv2-block diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = true,
            ),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81007)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "diagnostic-pro-conv2-block-htp", input, output)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProConv2UpDiagnosticOnHtp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/diagnostic-pro-conv2-up.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open("diagnostic-pro-conv2-up.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro conv2-up diagnostic graph failed to initialize on HTP",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = true,
            ),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = 81008)
        assertNotNull("HTP diagnostic graph returned no bitmap", output)
        assertEquals("Diagnostic graph output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "diagnostic-pro-conv2-up-htp", input, output)
        assertEquals(
            "Diagnostic image completed on a fallback backend",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProConv3DiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-conv3.v81.model.so",
            label = "diagnostic-pro-conv3-htp",
            id = 81009,
        )
    }

    @Test
    fun executesRealCuganProSecondPassDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-conv1.v81.model.so",
            label = "diagnostic-pro-unet2-conv1-htp",
            id = 81010,
        )
    }

    @Test
    fun executesRealCuganProSecondPassDownsampleDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-downsample.v81.model.so",
            label = "diagnostic-pro-unet2-downsample-htp",
            id = 81011,
        )
    }

    @Test
    fun executesRealCuganProDeepDownsampleDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-deep-downsample.v81.model.so",
            label = "diagnostic-pro-unet2-deep-downsample-htp",
            id = 81012,
        )
    }

    @Test
    fun executesRealCuganProSecondPassConv2PreSeDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-conv2-prese.v81.model.so",
            label = "diagnostic-pro-unet2-conv2-prese-htp",
            id = 81013,
        )
    }

    @Test
    fun executesRealCuganProSecondPassConv2LinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-conv2-prese-linear.v81.model.so",
            label = "diagnostic-pro-unet2-conv2-prese-linear-htp",
            id = 81014,
        )
    }

    @Test
    fun executesRealCuganProSecondPassSeGateDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-se-gate.v81.model.so",
            label = "diagnostic-pro-unet2-se-gate-htp",
            id = 81015,
        )
    }

    @Test
    fun executesRealCuganProDeepDownsampleLinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-deep-downsample-linear.v81.model.so",
            label = "diagnostic-pro-unet2-deep-downsample-linear-htp",
            id = 81016,
        )
    }

    @Test
    fun executesRealCuganProSecondPassSeMultiplyLinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-se-mul-linear.v81.model.so",
            label = "diagnostic-pro-unet2-se-mul-linear-htp",
            id = 81017,
        )
    }

    @Test
    fun executesRealCuganProFinalBlockLinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-unet2-final-block-linear.v81.model.so",
            label = "diagnostic-pro-unet2-final-block-linear-htp",
            id = 81018,
        )
    }

    @Test
    fun executesRealCuganProFinalResidualAddLinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-final-add-linear.v81.model.so",
            label = "diagnostic-pro-final-add-linear-htp",
            id = 81019,
        )
    }

    @Test
    fun executesRealCuganProFinalConvolutionLinearDiagnosticOnHtp() {
        runRealCuganProDiagnostic(
            asset = "diagnostic-pro-final-conv-linear.v81.model.so",
            label = "diagnostic-pro-final-conv-linear-htp",
            id = 81020,
        )
    }

    @Test
    fun executesRealCuganProVulkanBaselineForNpuComparison() {
        runRealCuganProVulkanBaseline(label = "color", input = createFixedInput())
    }

    @Test
    fun executesRealCuganProFp16OnHtpForNeutralGrayComparison() {
        runRealCuganProInference(
            precision = 0,
            label = "fp16-gray",
            input = createUniformGrayInput(),
        )
    }

    @Test
    fun executesRealCuganProVulkanBaselineForNeutralGrayComparison() {
        runRealCuganProVulkanBaseline(label = "gray", input = createUniformGrayInput())
    }

    @Test
    fun executesRealCuganProFp16OnHtpForMonochromeHalftoneComparison() {
        runRealCuganProInference(
            precision = 0,
            label = "fp16-halftone",
            input = createMonochromeHalftoneInput(),
        )
    }

    @Test
    fun executesRealCuganProFp16OnHtpWithRgbTensorOrder() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/realcugan-pro-x2-no-denoise.v81.model.so")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        context.assets.open("qnn-contexts/realcugan-pro-x2-no-denoise.v81.model.so").use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro diagnostic model failed to initialize on HTP with RGB tensor order",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = false,
            ),
        )

        val input = createFixedInput()
        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealCugan(input, id = 81020)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Enhancement returned no bitmap", output)
        assertEquals("Output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "realcugan-pro-x2-fp16-rgbfeed", input, output)
        Log.i("QualcommHtpDeviceTest", "RGB tensor order HTP inference completed in ${elapsedMs}ms")
        assertEquals(
            "Image completed on a fallback backend instead of QNN/HTP",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    @Test
    fun executesRealCuganProVulkanBaselineForMonochromeHalftoneComparison() {
        runRealCuganProVulkanBaseline(label = "halftone", input = createMonochromeHalftoneInput())
    }

    @Test
    fun executesRealCuganProVulkanFp32BaselineForMonochromeHalftoneComparison() {
        runRealCuganProVulkanBaseline(
            label = "halftone-fp32",
            input = createMonochromeHalftoneInput(),
            precision = 1,
        )
    }

    @Test
    fun executesRealCuganProFp16OnHtpForUniformColorComparison() {
        runRealCuganProInference(
            precision = 0,
            label = "fp16-uniform-color",
            input = createUniformColorInput(),
        )
    }

    private fun runRealCuganProVulkanBaseline(label: String, input: Bitmap, precision: Int = 0) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue(
            "Real-CUGAN Pro Vulkan model failed to initialize",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = precision,
                npuPrecision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealCugan(input, id = 81020)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Vulkan enhancement returned no bitmap", output)
        assertEquals("Output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "realcugan-pro-x2-vulkan-baseline-$label", input, output)
        assertTrue(
            "Expected a Vulkan inference backend, got ${Waifu2x.lastInferenceBackend()}",
            Waifu2x.lastInferenceBackend() == Waifu2x.INFERENCE_BACKEND_VULKAN ||
                Waifu2x.lastInferenceBackend() == Waifu2x.INFERENCE_BACKEND_NCNN,
        )
        Log.i(
            "QualcommHtpDeviceTest",
            "Vulkan comparison baseline: model=realcugan-pro-x2-no-denoise " +
                "backend=${Waifu2x.lastInferenceBackend()} elapsed=${elapsedMs}ms",
        )
    }

    @Test
    fun executesRealCuganProW8A16OnHtpWithoutGpuFallback() {
        runRealCuganProInference(precision = 2, label = "w8a16")
    }

    private fun runRealCuganProInference(precision: Int, label: String) {
        runRealCuganProInference(precision, label, createFixedInput())
    }

    private fun runRealCuganProInference(precision: Int, label: String, input: Bitmap) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue(
            "Real-CUGAN Pro $label NPU model failed to initialize",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = precision,
                npuPrecision = precision,
                processingBackend = Waifu2x.PROCESSING_BACKEND_QUALCOMM_NPU,
            ),
        )
        assertTrue("No active QNN graph after model initialization", Waifu2x.isQnnActive())

        val start = SystemClock.elapsedRealtime()
        val output = Waifu2x.processRealCugan(input, id = 81010 + precision)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        assertNotNull("Enhancement returned no bitmap", output)
        assertEquals("Output width was not 2x", input.width * 2, output!!.width)
        assertEquals("Output height was not 2x", input.height * 2, output.height)
        saveEvidence(context, "realcugan-pro-x2-$label", input, output)
        Log.i(
            "QualcommHtpDeviceTest",
            "Inference result: model=realcugan-pro-x2-no-denoise precision=$label " +
                "backend=${Waifu2x.lastInferenceBackend()} elapsed=${elapsedMs}ms",
        )
        assertEquals(
            "Image completed on a fallback backend instead of QNN/HTP",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    private fun runRealCuganProDiagnostic(asset: String, label: String, id: Int) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        Waifu2x.resetRealCugan()
        assertTrue(
            "Real-CUGAN model failed to initialize the shared bitmap pipeline",
            Waifu2x.initRealCugan(
                context = context,
                noiseLevel = 0,
                scale = 2,
                isPro = true,
                precision = 0,
                processingBackend = Waifu2x.PROCESSING_BACKEND_VULKAN,
            ),
        )

        val modelFile = File(context.cacheDir, "qnn-diagnostics/$asset")
        assertTrue("Could not create diagnostic model directory", modelFile.parentFile?.mkdirs() == true || modelFile.parentFile?.isDirectory == true)
        instrumentation.context.assets.open(asset).use { input ->
            modelFile.outputStream().use(input::copyTo)
        }
        assertTrue(
            "Real-CUGAN Pro diagnostic graph failed to initialize on HTP: $asset",
            Waifu2x.initializeQnnModelForInstrumentation(
                context = context,
                modelLibraryPath = modelFile.absolutePath,
                padding = 0,
                inputBgr = true,
            ),
        )

        val input = createFixedInput()
        val output = Waifu2x.processRealCugan(input, id = id)
        assertNotNull("HTP diagnostic graph returned no bitmap: $asset", output)
        assertEquals("Diagnostic graph output width was not 2x: $asset", input.width * 2, output!!.width)
        assertEquals("Diagnostic graph output height was not 2x: $asset", input.height * 2, output.height)
        saveEvidence(context, label, input, output)
        assertEquals(
            "Diagnostic image completed on a fallback backend: $asset",
            Waifu2x.INFERENCE_BACKEND_QUALCOMM_NPU,
            Waifu2x.lastInferenceBackend(),
        )
    }

    private fun createFixedInput(): Bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).apply {
        for (y in 0 until height) {
            for (x in 0 until width) {
                val checker = ((x / 8 + y / 8) and 1) == 0
                val red = if (checker) 235 else 36
                val green = 40 + (x * 150 / width)
                val blue = 35 + (y * 170 / height)
                setPixel(x, y, Color.rgb(red, green, blue))
            }
        }
    }

    private fun createMangaQualityFixture(size: Int = 128): Bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888).apply {
        eraseColor(Color.rgb(248, 246, 239))
        val canvas = Canvas(this)
        canvas.scale(size / 128f, size / 128f)
        val ink = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(24, 25, 29) }
        val gray = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(116, 118, 121) }
        val accent = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(174, 45, 52) }
        canvas.drawRect(5f, 5f, 61f, 69f, ink)
        canvas.drawRect(7f, 7f, 59f, 67f, Paint().apply { color = Color.WHITE })
        canvas.drawOval(15f, 13f, 49f, 49f, ink)
        canvas.drawOval(18f, 16f, 46f, 46f, Paint().apply { color = Color.rgb(227, 220, 202) })
        canvas.drawLine(9f, 53f, 57f, 18f, ink.apply { strokeWidth = 1.5f })
        for (y in 73 until 121 step 3) {
            for (x in 6 until 57 step 3) {
                if ((x * 7 + y * 11) % 5 < 2) canvas.drawCircle(x.toFloat(), y.toFloat(), 0.65f, gray)
            }
        }
        canvas.drawRect(68f, 6f, 122f, 52f, accent)
        canvas.drawRect(70f, 8f, 120f, 50f, Paint().apply { color = Color.rgb(255, 232, 212) })
        canvas.drawRect(76f, 14f, 114f, 42f, Paint().apply { color = Color.rgb(53, 107, 141) })
        canvas.drawLine(70f, 45f, 120f, 12f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; strokeWidth = 2f })
        canvas.drawRect(66f, 59f, 123f, 122f, ink)
        canvas.drawRect(68f, 61f, 121f, 120f, Paint().apply { color = Color.WHITE })
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(24, 25, 29)
            textSize = 11f
            typeface = android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.BOLD)
        }
        canvas.drawText("NPU 8G", 73f, 76f, text)
        canvas.drawText("Manga 2x", 73f, 92f, text)
        canvas.drawText("A5 8B", 73f, 108f, text)
        canvas.drawLine(72f, 112f, 117f, 112f, gray.apply { strokeWidth = 0.8f })
    }

    private fun createUniformGrayInput(): Bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).apply {
        eraseColor(Color.rgb(144, 144, 144))
    }

    private fun createMonochromeHalftoneInput(): Bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).apply {
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = if (((x / 4) + (y / 4)) % 2 == 0) 232 else 24
                setPixel(x, y, Color.rgb(value, value, value))
            }
        }
    }

    private fun createUniformColorInput(): Bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).apply {
        eraseColor(Color.rgb(196, 82, 34))
    }

    private fun saveEvidence(context: android.content.Context, name: String, input: Bitmap, output: Bitmap) {
        val evidenceDirectory = File(context.getExternalFilesDir(null), "npu-validation")
        assertTrue("Could not create evidence directory", evidenceDirectory.mkdirs() || evidenceDirectory.isDirectory)
        FileOutputStream(File(evidenceDirectory, "$name-input.png")).use {
            assertTrue("Could not save fixed NPU input", input.compress(Bitmap.CompressFormat.PNG, 100, it))
        }
        FileOutputStream(File(evidenceDirectory, "$name-output.png")).use {
            assertTrue("Could not save NPU output", output.compress(Bitmap.CompressFormat.PNG, 100, it))
        }
    }

    private fun saveInputEvidence(context: android.content.Context, name: String, input: Bitmap) {
        val evidenceDirectory = File(context.getExternalFilesDir(null), "npu-validation")
        assertTrue("Could not create evidence directory", evidenceDirectory.mkdirs() || evidenceDirectory.isDirectory)
        FileOutputStream(File(evidenceDirectory, "$name-input.png")).use {
            assertTrue("Could not save fixed NPU input", input.compress(Bitmap.CompressFormat.PNG, 100, it))
        }
    }
}
