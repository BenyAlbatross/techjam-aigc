"""Memory fitting, exact-coverage training, and resumable artifacts."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from techjam_aigc.trace_rx_m.backbone import DinoV3PatchEncoder

from .config import ThreeBranchConfig
from .data import parent_digest
from .memory import DualPrototypeMemory
from .model import ThreeBranchDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _autocast(device: torch.device, config: ThreeBranchConfig):
    if device.type == "cuda" and config.optimizer.mixed_precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _select_memory_tokens(tokens: Tensor, parent_ids: Iterable[str], count: int) -> Tensor:
    """Select spatially spread tokens with a deterministic per-image rotation."""

    batch, patches, _ = tokens.shape
    count = min(count, patches)
    base = torch.linspace(0, patches - 1, steps=count, device=tokens.device).round().long()
    rows = []
    for index, parent_id in enumerate(parent_ids):
        offset = int.from_bytes(sha256(str(parent_id).encode()).digest()[:4], "big") % patches
        rows.append(tokens[index, (base + offset) % patches])
    if len(rows) != batch:
        raise RuntimeError("Parent identifiers and token batches are misaligned.")
    return torch.stack(rows)


class _StreamingKMeans:
    def __init__(self, clusters: int, batch_size: int, seed: int) -> None:
        self.clusters = clusters
        self.model = MiniBatchKMeans(
            n_clusters=clusters,
            batch_size=batch_size,
            random_state=seed,
            n_init=1,
            reassignment_ratio=0.01,
        )
        self.pending_values: list[np.ndarray] = []
        self.pending_weights: list[np.ndarray] = []
        self.initialized = False
        self.samples_seen = 0

    def update(self, values: np.ndarray, weights: np.ndarray) -> None:
        if values.size == 0:
            return
        self.samples_seen += int(values.shape[0])
        if not self.initialized:
            self.pending_values.append(values)
            self.pending_weights.append(weights)
            if sum(chunk.shape[0] for chunk in self.pending_values) < self.clusters:
                return
            values = np.concatenate(self.pending_values)
            weights = np.concatenate(self.pending_weights)
            self.pending_values.clear()
            self.pending_weights.clear()
            self.initialized = True
        self.model.partial_fit(values, sample_weight=weights)

    def centers(self) -> Tensor:
        if not self.initialized:
            raise ValueError("Insufficient samples to initialize prototype memory.")
        return F.normalize(torch.from_numpy(self.model.cluster_centers_).float(), dim=-1)


@torch.inference_mode()
def fit_dual_memories(
    encoder: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    config: ThreeBranchConfig,
    device: torch.device,
    expected_parent_ids: set[str],
    provenance: Mapping[str, str],
    output_path: Path,
) -> dict[str, Any]:
    """Fit both class dictionaries from every train image exactly once."""

    encoder.requires_grad_(False)
    encoder.eval().to(device)
    clusters = config.memory.prototypes_per_class
    estimators = {
        0: _StreamingKMeans(clusters, config.memory.kmeans_batch_size, config.data.seed),
        1: _StreamingKMeans(clusters, config.memory.kmeans_batch_size, config.data.seed + 1),
    }
    seen: list[str] = []
    class_rows = defaultdict(int)
    started = time.monotonic()
    for batch_index, batch in enumerate(loader, start=1):
        pixels = torch.as_tensor(batch["global_pixels"], device=device)
        with _autocast(device, config):
            tokens = F.normalize(encoder(pixels).float(), dim=-1)
        selected = _select_memory_tokens(
            tokens,
            list(map(str, batch["parent_id"])),
            config.memory.tokens_per_image,
        ).cpu().numpy()
        labels = torch.as_tensor(batch["target"]).numpy()
        image_weights = (
            torch.as_tensor(batch["group_weight"]).numpy()
            if config.memory.group_balanced_fit
            else np.ones_like(labels, dtype=np.float32)
        )
        for target in (0, 1):
            mask = labels == target
            if not np.any(mask):
                continue
            values = selected[mask].reshape(-1, selected.shape[-1])
            weights = np.repeat(image_weights[mask], selected.shape[1]).astype(np.float64)
            estimators[target].update(values, weights)
            class_rows[target] += int(mask.sum())
        seen.extend(map(str, batch["parent_id"]))
        if batch_index % 100 == 0:
            print(
                json.dumps({
                    "stage": "fit-memories",
                    "batches": batch_index,
                    "rows": len(seen),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }),
                flush=True,
            )
    _assert_exact_coverage(seen, expected_parent_ids, stage="memory fitting")
    artifact = {
        "stage": "three-branch-memory",
        "authentic_prototypes": estimators[0].centers(),
        "synthetic_prototypes": estimators[1].centers(),
        "topk": config.memory.topk,
        "temperature": config.memory.retrieval_temperature,
        "dimension": int(estimators[0].centers().shape[1]),
        "rows_seen": len(seen),
        "class_rows": {"authentic": class_rows[0], "synthetic": class_rows[1]},
        "tokens_seen": {
            "authentic": estimators[0].samples_seen,
            "synthetic": estimators[1].samples_seen,
        },
        "parent_digest": parent_digest(seen),
        "config": config.to_dict(),
        **dict(provenance),
    }
    _atomic_torch_save(artifact, output_path)
    return {key: value for key, value in artifact.items() if not isinstance(value, Tensor)}


def load_dual_memory(
    path: Path,
    *,
    config: ThreeBranchConfig,
    provenance: Mapping[str, str],
) -> DualPrototypeMemory:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("stage") != "three-branch-memory":
        raise ValueError("Expected a three-branch memory artifact.")
    for field, expected in provenance.items():
        if artifact.get(field) != expected:
            raise ValueError(f"Memory artifact provenance mismatch for {field}.")
    if int(artifact["rows_seen"]) <= 0:
        raise ValueError("Memory artifact contains no training rows.")
    return DualPrototypeMemory(
        artifact["authentic_prototypes"],
        artifact["synthetic_prototypes"],
        topk=config.memory.topk,
        temperature=config.memory.retrieval_temperature,
    )


def build_model(
    config: ThreeBranchConfig,
    memory: DualPrototypeMemory,
) -> ThreeBranchDetector:
    encoder = DinoV3PatchEncoder(config.backbone)
    model = ThreeBranchDetector(encoder, memory, config.head)
    model.configure_for_training()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 2_000_000_000:
        raise ValueError("Three-branch model violates the challenge's <2B parameter rule.")
    return model


def _assert_exact_coverage(seen: list[str], expected: set[str], *, stage: str) -> None:
    observed = set(seen)
    if len(seen) != len(expected) or len(observed) != len(seen) or observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        duplicates = len(seen) - len(observed)
        raise RuntimeError(
            f"{stage} did not use every train row exactly once: "
            f"missing={missing}, extra={extra}, duplicates={duplicates}."
        )


def _weighted_bce(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def _pauc_surrogate(logits: Tensor, labels: Tensor, config: ThreeBranchConfig) -> Tensor:
    positives = logits[labels == 1]
    negatives = logits[labels == 0]
    if not positives.numel() or not negatives.numel():
        return logits.sum() * 0
    count = max(1, math.ceil(config.loss.pauc_alpha * negatives.numel()))
    hardest = torch.topk(negatives, k=count).values
    return F.softplus(
        config.loss.ranking_margin - positives[:, None] + hardest[None, :]
    ).mean()


def _objective(output, labels: Tensor, weights: Tensor, config: ThreeBranchConfig):
    bce = _weighted_bce(output.logit, labels, weights)
    pauc = _pauc_surrogate(output.logit, labels, config)
    auxiliary = torch.stack([
        _weighted_bce(output.global_logit, labels, weights),
        _weighted_bce(output.memory_logit, labels, weights),
        _weighted_bce(output.forensic_logit, labels, weights),
    ]).mean()
    total = bce + config.loss.pauc_weight * pauc + config.loss.auxiliary_branch_weight * auxiliary
    return total, bce, pauc, auxiliary


def build_optimizer(model: ThreeBranchDetector, config: ThreeBranchConfig):
    adapters = list(model.adapter_parameters())
    heads = list(model.head_parameters())
    if not adapters or not heads:
        raise ValueError("Expected trainable LoRA adapters and three-branch heads.")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters, "lr": config.optimizer.adapter_lr, "name": "lora"},
            {"params": heads, "lr": config.optimizer.head_lr, "name": "heads"},
        ],
        weight_decay=config.optimizer.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, *, steps: int, warmup_fraction: float):
    warmup = max(1, round(steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _epoch_metrics(
    values: Mapping[str, float],
    logits: list[float],
    labels: list[int],
    branch_logits: Mapping[str, list[float]],
    rows: int,
    seconds: float,
) -> dict[str, Any]:
    targets = np.asarray(labels)
    scores = np.asarray(logits)
    metrics: dict[str, Any] = {
        **{key: value / rows for key, value in values.items()},
        "rows": rows,
        "roc_auc": float(roc_auc_score(targets, scores)),
        "average_precision": float(average_precision_score(targets, scores)),
        "accuracy_at_zero": float(np.mean((scores >= 0) == targets)),
        "seconds": seconds,
    }
    for name, branch in branch_logits.items():
        metrics[f"{name}_roc_auc"] = float(roc_auc_score(targets, branch))
    return metrics


def _checkpoint(
    model: ThreeBranchDetector,
    optimizer,
    scheduler,
    *,
    epoch: int,
    history: list[dict[str, Any]],
    config: ThreeBranchConfig,
    provenance: Mapping[str, str],
    memory_sha256: str,
) -> dict[str, Any]:
    return {
        "stage": "three-branch-training",
        "epoch": epoch,
        "trainable_state": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "config": config.to_dict(),
        "memory_sha256": memory_sha256,
        **dict(provenance),
    }


def train_ten_epochs(
    model: ThreeBranchDetector,
    loader,
    *,
    config: ThreeBranchConfig,
    device: torch.device,
    expected_parent_ids: set[str],
    provenance: Mapping[str, str],
    memory_path: Path,
    output_directory: Path,
    resume_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Train the final epoch-10 model without validation or holdout selection."""

    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        steps=config.optimizer.epochs * len(loader),
        warmup_fraction=config.optimizer.warmup_fraction,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    memory_hash = file_sha256(memory_path)
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved.get("memory_sha256") != memory_hash:
            raise ValueError("Resume checkpoint uses a different memory artifact.")
        for field, expected in provenance.items():
            if saved.get(field) != expected:
                raise ValueError(f"Resume provenance mismatch for {field}.")
        incompatible = model.load_state_dict(saved["trainable_state"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected trainable checkpoint keys: {incompatible.unexpected_keys}")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1

    output_directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, config.optimizer.epochs + 1):
        model.train()
        totals = defaultdict(float)
        logits: list[float] = []
        labels: list[int] = []
        branch_logits = {"global": [], "memory": [], "forensic": []}
        seen: list[str] = []
        started = time.monotonic()
        for step, batch in enumerate(loader, start=1):
            global_pixels = torch.as_tensor(batch["global_pixels"], device=device)
            native_crops = torch.as_tensor(batch["native_crops"], device=device)
            target = torch.as_tensor(batch["target"], device=device)
            weights = torch.as_tensor(batch["sample_weight"], device=device).float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config):
                output = model(global_pixels, native_crops)
                total, bce, pauc, auxiliary = _objective(output, target, weights, config)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.optimizer.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            batch_rows = int(target.numel())
            for name, value in (
                ("loss", total),
                ("bce", bce),
                ("pauc", pauc),
                ("auxiliary", auxiliary),
            ):
                totals[name] += float(value.detach()) * batch_rows
            totals["gradient_norm"] += float(gradient_norm) * batch_rows
            logits.extend(output.logit.detach().float().cpu().tolist())
            labels.extend(target.cpu().tolist())
            branch_logits["global"].extend(output.global_logit.detach().float().cpu().tolist())
            branch_logits["memory"].extend(output.memory_logit.detach().float().cpu().tolist())
            branch_logits["forensic"].extend(output.forensic_logit.detach().float().cpu().tolist())
            seen.extend(map(str, batch["parent_id"]))
            if step % 100 == 0:
                print(json.dumps({
                    "stage": "train",
                    "epoch": epoch,
                    "step": step,
                    "steps": len(loader),
                    "rows": len(seen),
                    "loss": round(totals["loss"] / len(seen), 6),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }), flush=True)
        _assert_exact_coverage(seen, expected_parent_ids, stage=f"epoch {epoch}")
        metrics = _epoch_metrics(
            totals,
            logits,
            labels,
            branch_logits,
            len(seen),
            time.monotonic() - started,
        )
        metrics.update({
            "epoch": epoch,
            "parent_digest": parent_digest(seen),
            "checkpoint_selection": "final_epoch_only",
            "holdout_used": False,
        })
        history.append(metrics)
        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            history=history,
            config=config,
            provenance=provenance,
            memory_sha256=memory_hash,
        )
        epoch_path = output_directory / "checkpoints" / f"epoch-{epoch:04d}.pt"
        _atomic_torch_save(checkpoint, epoch_path)
        _atomic_torch_save(checkpoint, output_directory / "latest.pt")
        _atomic_json(history, output_directory / "training-history.json")
        print(json.dumps({"stage": "epoch-complete", **metrics}), flush=True)

    final = {
        "stage": "three-branch-final",
        "epoch": config.optimizer.epochs,
        "trainable_state": model.trainable_state_dict(),
        "history": history,
        "config": config.to_dict(),
        "memory_sha256": memory_hash,
        "selection": {
            "method": "final_epoch",
            "epoch": config.optimizer.epochs,
            "validation_used": False,
            "generator_holdout_used": False,
        },
        **dict(provenance),
    }
    _atomic_torch_save(final, output_directory / "three-branch-final.pt")
    return history
