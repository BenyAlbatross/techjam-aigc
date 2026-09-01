#!/usr/bin/env python3
"""Run TRACE-RX-M v2 inference for local website uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import center_crop
from techjam_aigc.trace_rx_m.config import TraceRXMConfig
from techjam_aigc.trace_rx_m.training import load_detector_checkpoint


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


def preprocess(image: Image.Image, config) -> np.ndarray:
    config.validate()
    prepared = image.convert("RGB")
    width, height = prepared.size
    short_side = min(width, height)
    if short_side > config.max_short_side:
        scale = config.max_short_side / short_side
        prepared = prepared.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
    cropped = center_crop(prepared, [config.image_size, config.image_size])
    values = np.asarray(cropped, dtype=np.float32) / 255.0
    values = (values - np.asarray(config.image_mean)) / np.asarray(config.image_std)
    return np.ascontiguousarray(values.transpose(2, 0, 1), dtype=np.float32)


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {directory}.")
    return paths


@torch.inference_mode()
def predict_images(model, preprocessing, paths: list[Path], device: torch.device):
    rows = []
    for path in paths:
        with Image.open(path) as opened:
            values = preprocess(opened, preprocessing)
        pixels = torch.from_numpy(values).unsqueeze(0).to(device)
        probability = torch.sigmoid(model(pixels).logit.float()).item()
        rows.append({"image_path": str(path), "pred": probability})
    return rows


def write_predictions(rows, output: Path) -> None:
    if any(set(row) != {"image_path", "pred"} for row in rows):
        raise ValueError("Every inference record must contain exactly image_path and pred.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, artifact = load_detector_checkpoint(
        args.artifacts / "s4_detector.pt",
        args.artifacts / "s3_memory.pt",
        device=device,
    )
    config = TraceRXMConfig.from_dict(artifact["config"])
    write_predictions(
        predict_images(model, config.preprocessing, image_paths(args.input), device),
        args.output,
    )


if __name__ == "__main__":
    main()
