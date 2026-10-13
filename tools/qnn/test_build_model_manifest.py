import tempfile
import unittest
from pathlib import Path

from build_model_manifest import build, unresolved_weight_symbol_count


class BuildModelManifestTest(unittest.TestCase):
    def test_real_cugan_declares_ncnn_bgr_tensor_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir)
            asset = asset_dir / "realcugan-pro-x2.v81.model.so"
            asset.write_bytes(b"model-library")

            entry = build(asset_dir)["models"][0]

            self.assertEqual(entry["inputLayout"], "NCHW_BGR")
            self.assertEqual(entry["outputLayout"], "NCHW_BGR")
            self.assertEqual(entry["tilePadding"], 0)

    def test_asset_metadata_overrides_layout_and_records_source_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir)
            asset = asset_dir / "realesrgan-animevideov3-x2.v81.model.so"
            asset.write_bytes(b"model-library")
            (asset_dir / "model-metadata.json").write_text(
                '{"assets":{"realesrgan-animevideov3-x2.v81.model.so":'
                '{"inputLayout":"NCHW_BGR","outputLayout":"NCHW_BGR",'
                '"tileSize":128,"tilePadding":18,"modelVersion":"anime-v3-x2"}}}',
                encoding="utf-8",
            )

            entry = build(asset_dir)["models"][0]

            self.assertEqual(entry["inputLayout"], "NCHW_BGR")
            self.assertEqual(entry["outputLayout"], "NCHW_BGR")
            self.assertEqual(entry["tileSize"], 128)
            self.assertEqual(entry["tilePadding"], 18)
            self.assertEqual(entry["modelVersion"], "anime-v3-x2")

    def test_records_architecture_precision_layout_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir)
            fp16 = asset_dir / "realesrgan-anime-x2.v81.model.so"
            int8 = asset_dir / "realesrgan-anime-x2-int8.v81.bin"
            fp16.write_bytes(b"model-library")
            int8.write_bytes(b"w8a16-context")

            manifest = build(asset_dir, context_soc_model=87)

            self.assertEqual(manifest["formatVersion"], 1)
            self.assertEqual(len(manifest["models"]), 2)
            entries = {entry["asset"]: entry for entry in manifest["models"]}
            model_library = entries[fp16.name]
            context = entries[int8.name]
            self.assertEqual(model_library["model"], "realesrgan-anime-x2")
            self.assertEqual(model_library["precision"], "FP16")
            self.assertEqual(model_library["artifact"], "model-library")
            self.assertEqual(model_library["htpArchitecture"], 81)
            self.assertEqual(model_library["scale"], 2)
            self.assertEqual(model_library["inputLayout"], "NCHW_RGB")
            self.assertEqual(model_library["qnnSocModel"], "runtime")
            self.assertEqual(context["precision"], "W8A16")
            self.assertEqual(context["artifact"], "context")
            self.assertEqual(context["qnnSocModel"], 87)
            self.assertEqual(
                context["sha256"],
                "a83493b41c74c71f760b467039bf667b7e6097176e22c80a1802e0a5ce9154a5",
            )

    def test_rejects_assets_without_declared_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir)
            (asset_dir / "unexpected.bin").write_bytes(b"payload")
            with self.assertRaises(ValueError):
                build(asset_dir)

    def test_requires_explicit_soc_model_for_precompiled_contexts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir)
            (asset_dir / "realcugan-pro-x2.v81.bin").write_bytes(b"context")
            with self.assertRaisesRegex(ValueError, "QNN SoC model ID"):
                build(asset_dir)

    def test_counts_unlinked_qnn_weight_blob_symbols(self):
        output = """                 U _binary_obj_binary_conv_weight_raw_start
                 U _binary_obj_binary_conv_weight_raw_end
                 U __cxa_finalize@LIBC
"""
        self.assertEqual(unresolved_weight_symbol_count(output), 2)


if __name__ == "__main__":
    unittest.main()
