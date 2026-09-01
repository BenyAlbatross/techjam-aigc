"""Artifact reconstruction and directory inference for the three-branch model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch

from .config import ThreeBranchConfig
from .data import _global_view, _native_crops
from .memory import DualPrototypeMemory
from .training import build_model, file_sha256


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_final_detector(
    config_path: Path,
    memory_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
):
    config = ThreeBranchConfig.load(config_path)
    config.validate(require_backbone_access=True)
    memory_artifact = torch.load(memory_path, map_location="cpu", weights_only=True)
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if memory_artifact.get("stage") != "three-branch-memory":
        raise ValueError("Invalid three-branch memory artifact.")
    if final.get("stage") != "three-branch-final" or int(final.get("epoch", 0)) != 10:
        raise ValueError("Inference requires the final epoch-10 three-branch artifact.")
    if final.get("memory_sha256") != file_sha256(memory_path):
        raise ValueError("Final checkpoint and dual memory artifact do not match.")
    if final.get("config") != config.to_dict():
        raise ValueError("Final checkpoint and inference config do not match.")
    memory = DualPrototypeMemory(
        memory_artifact["authentic_prototypes"],
        memory_artifact["synthetic_prototypes"],
        topk=int(memory_artifact["topk"]),
        temperature=float(memory_artifact["temperature"]),
    )
    model = build_model(config, memory)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    supplied = set(final["trainable_state"])
    if supplied != expected:
        raise ValueError(
            "Final trainable state is incomplete or incompatible: "
            f"missing={sorted(expected - supplied)}, unexpected={sorted(supplied - expected)}"
        )
    incompatible = model.load_state_dict(final["trainable_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.eval().to(device)
    return model, config


def image_paths(directory: Path) -> list[Path]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {root}.")
    return paths


def _batches(values: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def predict_directory(
    model,
    config: ThreeBranchConfig,
    directory: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    records: list[dict[str, object]] = []
    for paths in _batches(image_paths(directory), batch_size):
        global_values = []
        native_values = []
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                global_values.append(_global_view(image, config.preprocessing))
                native_values.append(_native_crops(image, config.preprocessing))
        global_pixels = torch.from_numpy(np.stack(global_values)).to(device)
        native_crops = torch.from_numpy(np.stack(native_values)).to(device)
        output = model(global_pixels, native_crops)
        probabilities = torch.sigmoid(output.logit.float()).cpu().tolist()
        records.extend(
            {"image_path": str(path), "pred": float(probability)}
            for path, probability in zip(paths, probabilities, strict=True)
        )
    return records


def write_predictions(records: list[dict[str, object]], path: Path) -> None:
    if any(set(record) != {"image_path", "pred"} for record in records):
        raise ValueError("Every inference record must contain exactly image_path and pred.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
