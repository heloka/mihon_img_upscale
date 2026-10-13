# Qualcomm QNN / HTP image-enhancement path

## Current device and build status

The attached OPPO Pad mini reports `SM8845P` and `canoe` on Android 16. The app's
instrumentation test queried QAIRT's platform API on that device and detected HTP v81.
Model selection uses that runtime result and an exact-architecture asset manifest; it does
not infer an HTP version from a Snapdragon product name or try v81/v85 speculatively.

The checked-in QNN payloads currently target v81. Their presence means that an artifact is
packaged, not that its graph has passed HTP execution or quality checks. See
[`NPU_DEVICE_TEST_REPORT.md`](../../NPU_DEVICE_TEST_REPORT.md) for the observed model-library
linker failures and the limits of the device testing completed so far.

When `local.properties` contains `qnn.sdk.dir`, Gradle's `stageQnnRuntime` task copies the
QNN HTP, System, HTP Prepare, Model DLC, v81 Stub, and v81 Skel from that single QAIRT SDK
tree. The default `qnn.htp.archs` is `81`, matching the only checked-in payload generation.
Add another target only after its assets and matching runtime Stub/Skel are available. The
release APK is signed by the configured Gradle signing config; no post-build APK edits are
part of the workflow.

Build the arm64 release APK with JDK 17 and the Android SDK configured:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:ANDROID_HOME = 'C:\path\to\Android\Sdk'
./gradlew.bat :app:testReleaseUnitTest :app:assembleRelease --console=plain
```

The NPU asset manifest pins each payload SHA-256, model family, precision, output scale,
NCHW RGB input/output layout, and HTP architecture. `build_model_manifest.py` can also audit
Android model-library dynamic symbols. Use the same NDK that builds the app:

```powershell
python tools/qnn/build_model_manifest.py app/src/main/assets/qnn-contexts `
  --llvm-nm C:\path\to\Android\Sdk\ndk\28.2.13676358\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-nm.exe
python tools/qnn/test_build_model_manifest.py
```

The audit records model libraries with unlinked QNN-converter weight-blob symbols as
`runtimeEligible: false`. Runtime code skips those `.model.so` files and tries a packaged
`.bin`/`.dlc` only for the architecture reported by QNN. A missing or failing context falls
through to the existing Vulkan path. A successful graph load alone is not evidence of a
successful inference.

## Export and context generation

`export_models.py`, `export_span_ncnn.py`, `export_w2xex_onnx.py`, and the converter/container
scripts are development tools for regenerating payloads. Their converter versions and
weights must be recorded alongside each generated artifact before results can be compared.
The currently checked-in payloads do not pin source-checkpoint revisions, and the existing
container recipe uses an older QAIRT toolchain than the runtime packaged in the app. Do not
mix outputs from those paths and describe them as reproducible QAIRT 2.51 artifacts without
rebuilding and recording the tool versions.

For any model change, validate the export against its source implementation first: layer
mapping, transposed-convolution weights, RGB/BGR conversion, normalization, padding/cropping,
and residual/output-range transforms. Then compare QNN output to the validated export on
fixed monochrome, color, halftone, and text samples. Do not mark a model available because
conversion or graph initialization succeeded. Pro's `(x - 0.15) / 0.7` output mapping is now
present in the ONNX exporter but has not yet passed the planned NCNN/PyTorch output-parity
suite.
