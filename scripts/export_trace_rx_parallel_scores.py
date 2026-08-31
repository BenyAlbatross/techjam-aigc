#!/usr/bin/env python3
"""Export frozen TRACE-RX-Parallel logits, endpoint quality, and artifact provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from techjam_aigc.trace_rx_parallel.config import TraceRXParallelConfig
from techjam_aigc.trace_rx_m.data import TraceRXMDataset, load_training_manifest
from techjam_aigc.trace_rx_m.quality import quality_vector
from techjam_aigc.trace_rx_m.reliability import PassiveQualityStacker, ReliabilityTable
from techjam_aigc.trace_rx_m.training import file_sha256
from techjam_aigc.trace_rx_parallel.training import load_detector_checkpoint


QUALITY_COLUMNS = (
    "log_min_dimension", "noise_sigma", "blockiness", "structural_hf_energy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--roles", nargs="+", default=["authentic_null", "development", "calibration"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _endpoint_quality(frame: pd.DataFrame, repo_root: Path, hf_cutoff: float) -> np.ndarray:
    descriptors = []
    for path_value in frame["local_path"]:
        path = Path(str(path_value))
        resolved = path if path.is_absolute() else repo_root / path
        with Image.open(resolved) as image:
            descriptors.append(quality_vector(image.convert("RGB"), hf_cutoff=hf_cutoff))
    return np.asarray(descriptors)


def _apply_reliability(path: Path, logits: np.ndarray, quality: np.ndarray) -> np.ndarray:
    artifact = json.loads(path.read_text())
    if artifact.get("mode") == "availability":
        return ReliabilityTable.from_dict(artifact["table"]).fuse(logits, quality)
    if artifact.get("mode") == "passive_quality_fallback":
        return PassiveQualityStacker.from_dict(artifact["stacker"]).fused_logits(logits, quality)
    raise ValueError("Unknown S5 reliability artifact mode.")


def main() -> None:
    args = parse_args()
    config = TraceRXParallelConfig.load(args.config)
    frame = load_training_manifest(args.manifest, roles=args.roles)
    missing_endpoint = {"condition", "transform_family"} - set(frame.columns)
    if missing_endpoint:
        raise ValueError(
            f"Score-export manifests require endpoint fields: {sorted(missing_endpoint)}"
        )
    dataset = TraceRXMDataset(frame, args.repo_root, image_size=config.backbone.image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
    )
    device = torch.device(args.device)
    detector_path = args.artifacts / "s4_detector.pt"
    model, detector_artifact = load_detector_checkpoint(
        detector_path,
        args.artifacts / "s3_memory.pt",
        device=device,
    )
    if config.to_dict() != detector_artifact["config"]:
        raise ValueError("Exporter config does not match the frozen S4 checkpoint config.")
    if detector_artifact.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("Exporter manifest does not match the manifest used for S4 training.")
    logits = []
    global_logits = []
    memory_logits = []
    fusion_weights = None
    with torch.inference_mode():
        for batch in loader:
            pixels = torch.as_tensor(batch["pixel_values"], device=device)
            output = model(pixels)
            logits.extend(output.logit.cpu().tolist())
            global_logits.extend(output.global_logit.cpu().tolist())
            memory_logits.extend(output.memory_logit.cpu().tolist())
            fusion_weights = output.fusion_weights.detach().cpu().tolist()
    quality = _endpoint_quality(frame, args.repo_root, config.reliability.hf_cutoff)
    result = frame.copy()
    result["logit"] = np.asarray(logits)
    result["global_logit"] = np.asarray(global_logits)
    result["memory_logit"] = np.asarray(memory_logits)
    if fusion_weights is None:
        raise ValueError("Cannot export scores from an empty manifest.")
    result["fusion_global_weight"] = float(fusion_weights[0])
    result["fusion_memory_weight"] = float(fusion_weights[1])
    for index, name in enumerate(QUALITY_COLUMNS):
        result[name] = quality[:, index]
    result["detector_sha256"] = file_sha256(detector_path)
    reliability_path = args.artifacts / "s5_reliability.json"
    if reliability_path.exists():
        result["fused_logit"] = _apply_reliability(
            reliability_path, result["logit"].to_numpy(), quality
        )
        result["reliability_sha256"] = file_sha256(reliability_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
