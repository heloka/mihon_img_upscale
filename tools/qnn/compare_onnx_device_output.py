#!/usr/bin/env python3
"""Compare one saved QNN tile output with the validated ONNX export."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def reflect101(indices: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(indices)
    reflected = indices.copy()
    while np.any((reflected < 0) | (reflected >= size)):
        reflected = np.where(reflected < 0, -reflected, reflected)
        reflected = np.where(reflected >= size, 2 * size - 2 - reflected, reflected)
    return reflected


def compare(
    onnx_path: Path,
    input_path: Path,
    output_path: Path,
    padding: int = 18,
    expected_output_path: Path | None = None,
    channel_order: str = "RGB",
) -> dict[str, float]:
    source = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    device = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
    height, width = source.shape[:2]
    if channel_order not in ("RGB", "BGR"):
        raise ValueError(f"Unsupported channel order: {channel_order}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    if len(model_input.shape) != 4 or not isinstance(model_input.shape[-1], int):
        raise ValueError(f"Expected a fixed square NCHW input, got {model_input.shape}")
    tile_size = model_input.shape[-1]
    core_size = tile_size - 2 * padding
    if core_size <= 0:
        raise ValueError(f"Padding {padding} leaves no tile core in {tile_size}px input")
    if device.shape[0] % height or device.shape[1] % width:
        raise ValueError(f"Device output {device.shape[1]}x{device.shape[0]} is not an integer upscale of the input")
    scale_y = device.shape[0] // height
    scale_x = device.shape[1] // width
    if scale_x != scale_y:
        raise ValueError("Device output has different horizontal and vertical scales")
    scale = scale_x
    expected_u8 = np.empty_like(device)
    input_name = model_input.name
    for tile_y in range(0, height, core_size):
        for tile_x in range(0, width, core_size):
            y = reflect101(np.arange(tile_size) + tile_y - padding, height)
            x = reflect101(np.arange(tile_size) + tile_x - padding, width)
            tile = source[y[:, None], x[None, :], :].astype(np.float32) / 255.0
            if channel_order == "BGR":
                tile = tile[:, :, ::-1]
            result = session.run(None, {input_name: tile.transpose(2, 0, 1)[None, ...]})[0]
            output = np.clip(result[0].transpose(1, 2, 0), 0.0, 1.0)
            if channel_order == "BGR":
                output = output[:, :, ::-1]
            offset = padding * scale
            copy_width = min(core_size, width - tile_x) * scale
            copy_height = min(core_size, height - tile_y) * scale
            destination_x = tile_x * scale
            destination_y = tile_y * scale
            expected_u8[
                destination_y:destination_y + copy_height,
                destination_x:destination_x + copy_width,
            ] = np.floor(
                output[offset:offset + copy_height, offset:offset + copy_width] * 255.0 + 0.5,
            ).astype(np.uint8)
    if expected_u8.shape != device.shape:
        raise ValueError(f"Expected {expected_u8.shape[1]}x{expected_u8.shape[0]} device output, got {device.shape[1]}x{device.shape[0]}")
    if expected_output_path:
        expected_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(expected_u8, mode="RGB").save(expected_output_path)
    difference = np.abs(expected_u8.astype(np.int16) - device.astype(np.int16))
    mse = float(np.mean(np.square(difference.astype(np.float64))))
    best_shift = (0, 0, float(difference.mean()))
    for shift_y in range(-8, 9):
        for shift_x in range(-8, 9):
            expected_crop = expected_u8[
                max(0, shift_y):expected_u8.shape[0] + min(0, shift_y),
                max(0, shift_x):expected_u8.shape[1] + min(0, shift_x),
            ]
            device_crop = device[
                max(0, -shift_y):device.shape[0] - max(0, shift_y),
                max(0, -shift_x):device.shape[1] - max(0, shift_x),
            ]
            mean_error = float(np.abs(expected_crop.astype(np.int16) - device_crop.astype(np.int16)).mean())
            if mean_error < best_shift[2]:
                best_shift = (shift_x, shift_y, mean_error)
    return {
        "max_abs_channel_error": float(difference.max()),
        "mean_abs_channel_error": float(difference.mean()),
        "p99_abs_channel_error": float(np.percentile(difference, 99)),
        "psnr_db": float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse))),
        "best_alignment_shift_xy": [best_shift[0], best_shift[1]],
        "best_aligned_mean_abs_channel_error": best_shift[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=18)
    parser.add_argument("--channel-order", choices=("RGB", "BGR"), default="RGB")
    parser.add_argument("--expected-output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.onnx,
                args.input,
                args.device_output,
                args.padding,
                args.expected_output,
                args.channel_order,
            ),
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
