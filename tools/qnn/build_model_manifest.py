#!/usr/bin/env python3
"""Generate a checksum-pinned manifest for packaged QNN model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ARCH_RE = re.compile(r"\.v(\d+)\.(model\.so|bin|dlc)$")
WEIGHT_SYMBOL_RE = re.compile(r"_binary_obj_binary_.*_raw_(?:start|end)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unresolved_weight_symbol_count(output: str) -> int:
    return sum(1 for line in output.splitlines() if WEIGHT_SYMBOL_RE.search(line.split()[-1]))


def build(
    directory: Path,
    llvm_nm: str | None = None,
    context_soc_model: int | None = None,
) -> dict[str, object]:
    metadata_path = directory / "model-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    asset_metadata = metadata.get("assets", {})
    models = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name in ("manifest.json", "model-metadata.json"):
            continue
        match = ARCH_RE.search(path.name)
        if not match:
            raise ValueError(f"Unrecognized QNN asset filename: {path.name}")
        model = path.name[: match.start()]
        base_model = model.removesuffix("-int8")
        # Model libraries relinked in this workspace were converted with RGB encoding;
        # the packaged context .bin payloads were converted with BGR encoding.
        channel_order = "BGR" if (base_model.startswith("realcugan-") and match.group(2) != "model.so") else "RGB"
        scales = re.findall(r"x(\d+)(?=-|$)", model.removesuffix("-int8"))
        if not scales:
            raise ValueError(f"QNN model name does not declare an output scale: {path.name}")
        entry: dict[str, object] = {
                "asset": path.name,
                "model": base_model,
                "precision": "W8A16" if model.endswith("-int8") else "FP16",
                "artifact": "model-library" if match.group(2) == "model.so" else "context",
                "htpArchitecture": int(match.group(1)),
                "scale": int(scales[-1]),
                "inputLayout": f"NCHW_{channel_order}",
                "outputLayout": f"NCHW_{channel_order}",
                "sha256": sha256(path),
            }
        if match.group(2) == "model.so":
            entry["qnnSocModel"] = "runtime"
            # The Real-CUGAN ONNX export itself applies the model's reflect pad.
            # The native tiled adapter therefore passes raw tile cores to QNN.
            if base_model.startswith("realcugan-"):
                entry["tilePadding"] = 0
        else:
            if context_soc_model is None:
                raise ValueError(
                    "A QNN SoC model ID detected from the target device is required "
                    f"for context binary: {path.name}",
                )
            entry["qnnSocModel"] = context_soc_model
        if llvm_nm and match.group(2) == "model.so":
            result = subprocess.run(
                [llvm_nm, "-D", "-u", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            unresolved = unresolved_weight_symbol_count(result.stdout)
            entry["unresolvedWeightSymbolCount"] = unresolved
            entry["runtimeEligible"] = unresolved == 0
        entry.update(asset_metadata.get(path.name, {}))
        models.append(entry)
    if not models:
        raise ValueError(f"No QNN model assets in {directory}")
    return {"formatVersion": 1, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("--llvm-nm", help="Audit model-library ELF files for unlinked converter weight blobs")
    parser.add_argument(
        "--context-soc-model",
        type=int,
        help="QNN SoC model ID reported by deviceGetPlatformInfo for context binaries",
    )
    args = parser.parse_args()
    output = args.asset_dir / "manifest.json"
    output.write_text(
        json.dumps(build(args.asset_dir, args.llvm_nm, args.context_soc_model), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
