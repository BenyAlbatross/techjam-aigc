"""Detection-stage helpers specialized for TRACE-RX-Parallel checkpoints."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

import torch

from techjam_aigc.trace_rx_m.training import (
    EpochMetrics,
    build_detection_optimizer,
    cosine_warmup_scheduler,
    file_sha256,
    load_memory_artifact,
    train_detection_epoch,
)

from .model import TraceRXParallel


def save_detector_checkpoint(
    model: TraceRXParallel,
    path: Path,
    *,
    config: Mapping[str, Any],
    memory_artifact_sha256: str,
    manifest_sha256: str,
    history: Iterable[EpochMetrics],
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    selection_metric: Mapping[str, Any] | None = None,
) -> None:
    """Save a portable, architecture-tagged parallel detector checkpoint."""

    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("encoder.") or name.endswith(("lora_A", "lora_B"))
    }
    encoder_mode = "lora" if any(
        name.startswith("encoder.") and name.endswith(("lora_A", "lora_B"))
        for name in state
    ) else "frozen"
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "stage": "S4",
        "architecture": "trace-rx-parallel",
        "encoder_mode": encoder_mode,
        "model_state": state,
        "config": dict(config),
        "source_memory_sha256": memory_artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "history": [asdict(item) for item in history],
        "epoch": epoch,
        "selection_metric": None if selection_metric is None else dict(selection_metric),
    }
    if optimizer is not None:
        artifact["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        artifact["scheduler_state"] = scheduler.state_dict()
    torch.save(artifact, path)


def load_detector_checkpoint(
    checkpoint_path: Path,
    memory_path: Path,
    *,
    device: torch.device,
) -> tuple[TraceRXParallel, dict[str, Any]]:
    """Reconstruct a LoRA or frozen TRACE-RX-Parallel detector."""

    from techjam_aigc.trace_rx_m.backbone import DinoV3PatchEncoder

    from .config import TraceRXParallelConfig

    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if artifact.get("stage") != "S4" or artifact.get("architecture") != "trace-rx-parallel":
        raise ValueError("Checkpoint is not a TRACE-RX-Parallel S4 artifact.")
    if artifact.get("encoder_mode") not in {"lora", "frozen"}:
        raise ValueError("Invalid TRACE-RX-Parallel encoder mode.")
    if artifact.get("source_memory_sha256") != file_sha256(memory_path):
        raise ValueError("S4 detector and S3 memory artifact hashes disagree.")
    config = TraceRXParallelConfig.from_dict(artifact["config"])
    backbone_config = config.backbone
    if artifact["encoder_mode"] == "frozen":
        backbone_config = replace(backbone_config, lora_rank=0)
    memory = load_memory_artifact(memory_path)
    model = TraceRXParallel(
        DinoV3PatchEncoder(backbone_config),
        memory,
        config.head,
    )
    incompatible = model.load_state_dict(artifact["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith("encoder.")
    ]
    if unexpected or invalid_missing:
        raise ValueError(
            f"Checkpoint state mismatch; unexpected={unexpected}, missing={invalid_missing}"
        )
    model.requires_grad_(False)
    model.eval().to(device)
    return model, artifact


__all__ = (
    "build_detection_optimizer",
    "cosine_warmup_scheduler",
    "load_detector_checkpoint",
    "save_detector_checkpoint",
    "train_detection_epoch",
)
