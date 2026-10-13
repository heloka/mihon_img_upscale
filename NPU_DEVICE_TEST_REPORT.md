# OPPO Pad mini NPU device report

> Historical snapshot from 2026-10-03. The workspace contains later PNG diagnostics and a newer installed APK; see `PROJECT_HANDOFF.md` for current status and evidence boundaries.

Date: 2026-10-03. Device serial: `27ee113`.

## Device and APK

| Check | Result |
| --- | --- |
| Device properties | `OPD2515`, `SM8845P`, platform `canoe`, Android 16 / API 36 |
| QAIRT platform query | HTP v81; connected instrumentation architecture test passed |
| Delivered APK | `app/build/outputs/apk/release/app-arm64-v8a-release.apk` |
| APK SHA-256 | `F4D32B7502CE948709C7A3C530A59C8B61A89897C9721959C326437FB5033B38` |
| Package/version | `app.mihon`, versionName `1.3.10.13`, versionCode 54 |
| Signature | SHA-256 `7baeda92b86f50f09e05339a777352f487af53177dc266a6bc7b4e5a7d12e7d1`; matches the installed app |
| `testOnly` | false; manifest attribute absent |
| Runtime contents | QAIRT 2.51 HTP/System/Prepare/ModelDlc plus matching v81 Stub and Skel |
| Manifest payload integrity | 49 entries; all declared assets were present in the APK and matched their SHA-256 |
| Install | `adb install -r` succeeded; the app was updated in place and its data was not cleared |

## Model payload matrix

The table reports what is packaged, not what has passed HTP inference. All listed artifacts
declare HTP v81. FP16 and W8A16 refer to manifest precision. “Context” means an exact-v81
`.bin` candidate is packaged as a fallback.

| Family | Scale / variants | FP16 payload | W8A16 payload | Model-library linker audit |
| --- | --- | --- | --- | --- |
| Real-CUGAN SE | x2, five denoise modes | `.model.so`; no `.bin` | `.model.so`; no `.bin` | FP16: 5/5 eligible; W8A16: weight symbols unresolved |
| Real-CUGAN Pro | x2 and x3, three modes per scale | `.model.so` + `.bin` | `.model.so` + `.bin` | All Pro `.model.so` files have 162 unresolved weight symbols; runtime skips them and tries `.bin` |
| Real-ESRGAN anime | x2 | `.model.so` + `.bin` | `.model.so` + `.bin` | 106 unresolved symbols in each `.model.so` |
| Real-ESRGAN photo/general | x2 output | `.model.so` + `.bin` | `.model.so` + `.bin` | 202 unresolved symbols in each `.model.so` |
| W2xEX Photo Small | x2 | `.model.so` + `.bin` | `.model.so` + `.bin` | 106 unresolved symbols in each `.model.so` |
| SPAN NomosUni | x2 | `.model.so` + `.bin` | `.model.so` only | 104 unresolved symbols; W8A16 has no `.bin` fallback |

The NDK `llvm-nm -D -u` audit found five linker-clean model libraries out of 29: the five SE
FP16 variants. The other 24 reference QNN converter weight-blob symbols that are absent from
their shared libraries. This is a static-linkage finding, not an inference-quality test.

## Direct inference evidence

The connected-device instrumentation test confirmed HTP v81 and that the test package
contains a matching v81 context. It did not load or execute each model.

In the previously installed release build (version 1.3.10.10), the active reader attempted
Real-CUGAN Pro x2 W8A16. QNN failed to load
`realcugan-pro-x2-no-denoise-int8.v81.model.so` because the shared library did not define
`_binary_obj_binary_model_unet1_conv1_conv_0_weight_raw_start`. The runtime then fell back to
the existing image-enhancement path. This failure motivated the manifest symbol audit and
same-HTP `.bin` fallback, included in version 1.3.10.11 and the delivered 1.3.10.12.

The final version's `.bin` fallback has not yet been observed executing on the tablet. OPPO's
install guide appeared during the follow-up isolated `.dev` test run, so that run was stopped
to avoid further changing the reader screen. No model in the matrix is marked “NPU inference
passed” from the evidence available here.

No fixed black/white, color, halftone, or text image was processed and saved on device. There
are no measured graph-build/load/first-execute/steady-state durations, quality scores, current
page comparison P95, 50-page stability result, or preload hit rate. Those acceptance tests
remain outstanding. The architecture test log is available under
`app/build/outputs/androidTest-results/connected/debug/OPD2515 - 16/`.

## Software tests completed

- `:app:testReleaseUnitTest`: passed, including FP16/W8A16 selection, NPU model/scale
  eligibility, and exact-architecture artifact fallback ordering.
- `python tools/qnn/test_build_model_manifest.py`: 3 tests passed.
- APK static validation: expected package/version/signature, non-test APK, complete HTP v81
  runtime set, all model assets present, and all asset hashes matched the manifest.

The device report should be extended only with results that include the actual backend and
precision used. A Vulkan fallback must not be counted as NPU success.
