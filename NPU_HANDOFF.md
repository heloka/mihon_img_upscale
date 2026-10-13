# Mihon image-enhancement / Qualcomm HTP handoff

> Historical NPU snapshot (2026-10-03). For current user priorities, GPU/preload work, the newer isolated service source, and the APK installed on device, read `PROJECT_HANDOFF.md` first. This file and its APK version details are not the latest project status.

Updated 2026-10-03 for the OPPO Pad mini. This file replaces the earlier handoff's
unverified chip inference and claims that non-SE model loading was fixed. Historical notes
from that handoff are useful clues, not current device-validation evidence.

## Target and verified environment

- Device: OPPO Pad mini, model `OPD2515`, Android 16, build-reported SoC `SM8845P`, platform
  `canoe`. Qualcomm's product brief identifies SM8845 as Snapdragon 8 Gen 5. Do not substitute
  SM8850 based on a product-generation guess.
- QAIRT 2.51 platform query on the connected device reported HTP v81. The connected-device
  instrumentation result is in
  `app/build/outputs/androidTest-results/connected/debug/TEST-OPD2515 - 16-_app-.xml`.
- Windows build tools: JDK 17 at
  `D:\02_Software_Repo\Dev_Tools\jdk-17.0.20.1+1`, Android SDK at `D:\Android\Sdk`, NDK
  `28.2.13676358`, QAIRT at `D:\Android\qnn-sdk`.
- Use the Gradle wrapper. `app/build.gradle.kts` stages the QNN runtime from one configured
  SDK tree. The checked-in payload and default staged HTP target are v81.
- Release credentials are configured in ignored `local.properties`; never print or check
  them in. The existing `app.mihon` signing certificate SHA-256 is
  `7baeda92b86f50f09e05339a777352f487af53177dc266a6bc7b4e5a7d12e7d1`.

## Implemented in this working tree

- NPU precision is carried independently from NCNN/Vulkan precision. W8A16 chooses the
  quantized asset while Vulkan fallback stays FP16; other supported NPU precision requests
  map to FP16.
- Runtime selects assets for the exact HTP version reported by QNN. The manifest pins hashes,
  model family, scale, precision, tensor layout, and target HTP. Asset extraction validates
  both cached and newly copied files and commits through a temporary file.
- `llvm-nm` audit finds unresolved converter weight-blob symbols in 24 of 29 model libraries.
  Those entries are marked `runtimeEligible: false`, skipped by runtime, and an exact-HTP
  packaged context is tried when present. Five Real-CUGAN SE FP16 model libraries pass this
  static linker-symbol audit. This audit does not prove graph execution.
- Failed `.model.so` initialization can continue to its same-architecture `.bin`/`.dlc`
  candidate. Invalid generated context files are removed before a rebuild attempt.
- Model-library context generation no longer uses `std::async` as a false 25-second timeout;
  the old future destructor could still block. The JNI call now runs synchronously on the
  enhancement IO worker. **There is still no hard-kill worker process or enforceable NPU
  deadline**, so the service isolation requirement remains open.
- Reader toolbar has a separate original/enhanced comparison control. Original image appears
  first while enhancement runs; comparison swaps retained views without clearing cache or
  scheduling inference. Pager and webtoon holders receive the session state, and zoom center
  is mapped between source and enhanced dimensions. Session state survives activity recreation.
- Pro export now applies the upstream output-range inverse `(x - 0.15) / 0.7`, clamps, and
  rounds. NCNN/PyTorch/ONNX numerical parity has not yet been run, so this is not a validated
  quality fix.
- Runtime staging is automated in Gradle and no APK ZIP surgery is required.

## Device evidence and present limits

The architecture-only connected test passed and reported v81. A real release-app attempt
with Real-CUGAN Pro x2 W8A16 on this tablet selected its `.model.so`, then `dlopen` failed on
the unresolved symbol `_binary_obj_binary_model_unet1_conv1_conv_0_weight_raw_start`. The
manifest audit reproduces the missing-weight-symbol class across the other affected model
libraries. This proves that the former Pro NPU load path was broken.

The exact-v81 `.bin` fallback was added after that observed failure. OPPO's install guide
interrupted the follow-up isolated `.dev` instrumentation run, so no claim is made that the
fallback contexts execute successfully. There is no complete SE/Pro/ESRGAN/W2xEX/SPAN ×
FP16/W8A16 inference matrix, fixed-image color/quality report, P95 comparison measurement, or
50-page stability run. See `NPU_DEVICE_TEST_REPORT.md`.

Also still open: QNN work in an independently killable `:image_enhancement` service, Binder
request/progress protocol using file descriptors, per-model circuit breaking/retry controls,
unified cached backend/precision/timing/failure results, source-checkpoint version pinning,
the complete export parity suite, and automated reader UI/cache/queue timeout tests. Keep
failed NPU combinations labelled unavailable and report Vulkan as the actual fallback until
those checks are implemented.

## Build, test, and install

```powershell
$env:JAVA_HOME = 'D:\02_Software_Repo\Dev_Tools\jdk-17.0.20.1+1'
$env:ANDROID_HOME = 'D:\Android\Sdk'
./gradlew.bat :app:testReleaseUnitTest :app:assembleRelease --console=plain
```

Use `app/build/outputs/apk/release/app-arm64-v8a-release.apk` for OPPO Pad mini. Check its
manifest, `testOnly`, signature, QNN libraries, and asset hashes before installing. Install
updates with `adb -s 27ee113 install -r <apk>`; do not uninstall `app.mihon`. The delivered
build is version `1.3.10.13` / versionCode 54, signed with the installed app's certificate,
and was installed with `-r` without clearing app data. It is not published or pushed.

The release unit suite passed, including NPU precision/family/architecture-candidate tests.
The manifest generator tests passed. The device architecture test passed. Model execution and
reader comparison still need the device report's unfinished checks before describing this as
a completed NPU support fix.
