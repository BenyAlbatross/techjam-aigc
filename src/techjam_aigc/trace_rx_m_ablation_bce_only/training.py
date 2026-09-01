"""Strictly ordered S1--S4 training utilities for TRACE-RX-M v2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LossConfig, MemoryConfig, OptimizerConfig
from .losses import detection_objective
from .memory import AuthenticMemory
from .model import TraceRXM


@dataclass(frozen=True)
class CoverageMetrics:
    size: int
    topk: int
    mean_error: float
    tail_threshold: float
    tail_fraction: float
    worst_subtype_tail_fraction: float
    subtype_tail_fractions: dict[str, float]


@dataclass(frozen=True)
class EpochMetrics:
    total: float
    bce: float
    pauc: float
    pair: float
    mean_authentic_loss: float
    worst_authentic_subtype_loss: float
    subtype_losses: dict[str, float]
    gradient_conflicts: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def cache_patch_features(
    encoder: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    output_path: Path,
    *,
    device: torch.device,
    backbone_model_id: str,
    backbone_revision: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """S1: cache patch tokens from the frozen, unadapted encoder."""

    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("S1 requires a fully frozen, unadapted encoder.")
    encoder.eval().to(device)
    token_batches: list[Tensor] = []
    parent_ids: list[str] = []
    subtypes: list[str] = []
    audit_pixels: Tensor | None = None
    for batch in loader:
        pixels = torch.as_tensor(batch["pixel_values"], device=device)
        if audit_pixels is None:
            audit_pixels = pixels[:1].detach().clone()
        tokens = F.normalize(encoder(pixels).float(), dim=-1)
        token_batches.append(tokens.to(device="cpu", dtype=torch.bfloat16))
        parent_ids.extend(map(str, batch["parent_id"]))
        subtype = batch.get("authentic_subtype", batch.get("source_dataset"))
        if subtype is None:
            subtype = ["unknown"] * tokens.shape[0]
        subtypes.extend(map(str, subtype))
    if not token_batches:
        raise ValueError("Cannot cache an empty loader.")
    cached = torch.cat(token_batches)
    if audit_pixels is None:  # pragma: no cover - guarded by token_batches
        raise RuntimeError("Missing cache audit sample.")
    direct = F.normalize(encoder(audit_pixels).float(), dim=-1).cpu()
    cache_error = float((direct - cached[:1].float()).abs().max())
    if cache_error > 0.01:
        raise RuntimeError(
            "S1 cache gate failed: BF16 cache does not match a direct encoder forward "
            f"(max absolute error {cache_error:.6g})."
        )
    artifact = {
        "stage": "S1",
        "tokens": cached,
        "parent_ids": parent_ids,
        "authentic_subtypes": subtypes,
        "backbone_model_id": backbone_model_id,
        "backbone_revision": backbone_revision,
        "manifest_sha256": manifest_sha256,
        "direct_forward_max_abs_error": cache_error,
        "direct_forward_gate_passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return {key: value for key, value in artifact.items() if key != "tokens"}


def load_feature_cache(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "stage",
        "tokens",
        "parent_ids",
        "authentic_subtypes",
        "backbone_model_id",
        "backbone_revision",
        "manifest_sha256",
        "direct_forward_gate_passed",
    }
    if not isinstance(artifact, dict) or not required <= set(artifact):
        raise ValueError(f"Invalid S1 feature cache: {path}")
    if artifact["stage"] != "S1":
        raise ValueError("Memory fitting accepts S1 caches only.")
    if artifact["direct_forward_gate_passed"] is not True:
        raise ValueError("S1 direct-forward cache-equivalence gate did not pass.")
    return artifact


def fit_authentic_memory(
    tokens: Tensor,
    *,
    size: int,
    topk: int,
    config: MemoryConfig,
    optimizer_config: OptimizerConfig,
    device: torch.device,
    seed: int,
    images_per_step: int = 8,
) -> tuple[AuthenticMemory, list[dict[str, float]]]:
    """S2: initialize with k-means and fit only authentic prototypes M."""

    if tokens.ndim != 3:
        raise ValueError("Cached tokens must have shape [images, patches, dimension].")
    seed_everything(seed)
    memory = AuthenticMemory(
        size,
        tokens.shape[-1],
        topk,
        score_chunk_size=config.score_chunk_size,
    ).to(device)
    memory.initialize_kmeans(tokens, seed=seed)
    optimizer = torch.optim.AdamW(
        [memory.prototypes],
        lr=optimizer_config.memory_lr,
        weight_decay=optimizer_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(optimizer_config.memory_epochs):
        totals = defaultdict(float)
        steps = 0
        for indices in torch.randperm(tokens.shape[0], generator=generator).split(images_per_step):
            batch = tokens[indices].to(device=device, dtype=torch.float32)
            loss = memory.phase1_loss(batch, config.diversity_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()
            with torch.no_grad():
                memory.prototypes.copy_(F.normalize(memory.prototypes, dim=-1))
            for name in ("total", "reconstruction", "diversity"):
                totals[name] += float(getattr(loss, name).detach())
            steps += 1
        history.append({"epoch": float(epoch), **{key: value / steps for key, value in totals.items()}})
    return memory, history


@torch.inference_mode()
def evaluate_memory_coverage(
    memory: AuthenticMemory,
    tokens: Tensor,
    authentic_subtypes: Iterable[str],
    *,
    tail_threshold: float | None,
    tail_quantile: float,
    device: torch.device,
) -> CoverageMetrics:
    """S3: measure the authentic patch-error tail and worst subtype."""

    memory.eval().to(device)
    errors: list[Tensor] = []
    for batch in tokens.split(8):
        normalized = F.normalize(batch.to(device=device, dtype=torch.float32), dim=-1)
        errors.append((normalized - memory(normalized).reference).square().sum(-1).cpu())
    error = torch.cat(errors)
    threshold = float(torch.quantile(error.flatten(), tail_quantile)) if tail_threshold is None else tail_threshold
    subtypes = np.asarray(list(map(str, authentic_subtypes)))
    if subtypes.size != error.shape[0]:
        raise ValueError("One authentic subtype is required per cached image.")
    fractions = {
        str(subtype): float(
            (error[torch.from_numpy(subtypes == subtype)] > threshold).float().mean()
        )
        for subtype in sorted(set(subtypes))
    }
    return CoverageMetrics(
        size=memory.size,
        topk=memory.topk,
        mean_error=float(error.mean()),
        tail_threshold=threshold,
        tail_fraction=float((error > threshold).float().mean()),
        worst_subtype_tail_fraction=max(fractions.values()),
        subtype_tail_fractions=fractions,
    )


def select_capacity(
    candidates: Iterable[CoverageMetrics],
    *,
    relative_tolerance: float,
) -> CoverageMetrics:
    """Choose the smallest candidate within tolerance of the best tail."""

    ordered = sorted(candidates, key=lambda value: (value.size, value.topk))
    if not ordered:
        raise ValueError("At least one capacity candidate is required.")
    best = min(item.worst_subtype_tail_fraction for item in ordered)
    eligible = [
        item for item in ordered
        if item.worst_subtype_tail_fraction <= best * (1 + relative_tolerance) + 1e-12
    ]
    return min(eligible, key=lambda value: (value.size, value.topk))


@torch.inference_mode()
def prototype_usage_histogram(
    memory: AuthenticMemory,
    tokens: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Accumulate sparse-selection usage without materializing all scores."""

    usage = torch.zeros(memory.size, dtype=torch.int64)
    memory.eval().to(device)
    for batch in tokens.split(8):
        usage += memory.usage_histogram(batch.to(device=device, dtype=torch.float32)).cpu()
    return usage


def save_memory_artifact(
    memory: AuthenticMemory,
    path: Path,
    *,
    source_cache_sha256: str,
    backbone_model_id: str,
    backbone_revision: str,
    manifest_sha256: str,
    history: list[dict[str, float]],
    coverage: CoverageMetrics,
) -> None:
    def safe_builtin(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): safe_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_builtin(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "S3",
            "prototypes": F.normalize(memory.prototypes.detach().cpu(), dim=-1),
            "size": memory.size,
            "dimension": memory.dimension,
            "topk": memory.topk,
            "score_chunk_size": memory.score_chunk_size,
            "source_cache_sha256": source_cache_sha256,
            "backbone_model_id": backbone_model_id,
            "backbone_revision": backbone_revision,
            "manifest_sha256": manifest_sha256,
            "history": safe_builtin(history),
            "coverage": safe_builtin(asdict(coverage)),
        },
        path,
    )


def load_memory_artifact(
    path: Path,
    *,
    expected_cache_sha256: str | None = None,
    expected_backbone_model_id: str | None = None,
    expected_backbone_revision: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> AuthenticMemory:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("stage") != "S3":
        raise ValueError("S4 requires a capacity-selected S3 memory artifact.")
    if expected_cache_sha256 and artifact.get("source_cache_sha256") != expected_cache_sha256:
        raise ValueError("Memory artifact was fitted from a different feature cache.")
    expectations = {
        "backbone_model_id": expected_backbone_model_id,
        "backbone_revision": expected_backbone_revision,
        "manifest_sha256": expected_manifest_sha256,
    }
    for field, expected in expectations.items():
        if expected is not None and artifact.get(field) != expected:
            raise ValueError(f"Memory artifact {field} does not match the S4 configuration.")
    memory = AuthenticMemory(
        int(artifact["size"]),
        int(artifact["dimension"]),
        int(artifact["topk"]),
        score_chunk_size=int(artifact["score_chunk_size"]),
        prototypes=artifact["prototypes"],
    )
    memory.requires_grad_(False)
    return memory


def build_detection_optimizer(model: TraceRXM, config: OptimizerConfig) -> torch.optim.AdamW:
    """Use separate low-LR LoRA and high-LR head parameter groups."""
    adapters = [
        parameter for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and name.endswith(("lora_A", "lora_B"))
    ]
    heads = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    if not heads:
        raise ValueError("No trainable detector heads found.")
    groups: list[dict[str, Any]] = []
    if adapters:
        groups.append({"params": adapters, "lr": config.adapter_lr, "name": "lora"})
    groups.append({"params": heads, "lr": config.head_lr, "name": "heads"})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Short linear warm-up followed by cosine decay."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def paired_dda_logits(batch: Mapping[str, Any], logits: Tensor) -> tuple[Tensor, Tensor]:
    """Resolve DDA/source pairs co-located by the batch sampler."""
    kinds = list(map(str, batch["sample_kind"]))
    parent_ids = list(map(str, batch["parent_id"]))
    source_ids = list(map(str, batch.get("source_parent_id", [""] * len(kinds))))
    lookup = {parent_id: index for index, parent_id in enumerate(parent_ids)}
    dda_indices: list[int] = []
    real_indices: list[int] = []
    for index, (kind, source_id) in enumerate(zip(kinds, source_ids, strict=True)):
        if kind != "dda":
            continue
        if source_id not in lookup:
            raise ValueError("Every DDA sample's authentic source must be in its batch.")
        dda_indices.append(index)
        real_indices.append(lookup[source_id])
    return logits[dda_indices], logits[real_indices]


def primary_objective_mask(
    batch: Mapping[str, Any],
    *,
    include_dda: bool,
) -> list[bool]:
    """Select native samples for BCE/pAUC without double-counting DDA pairs."""

    kinds = list(map(str, batch["sample_kind"]))
    parent_ids = list(map(str, batch["parent_id"]))
    source_ids = list(map(str, batch.get("source_parent_id", [""] * len(kinds))))
    paired_sources = {
        source_id
        for kind, source_id in zip(kinds, source_ids, strict=True)
        if kind == "dda"
    }
    return [
        include_dda or (kind != "dda" and parent_id not in paired_sources)
        for kind, parent_id in zip(kinds, parent_ids, strict=True)
    ]


def train_detection_epoch(
    model: TraceRXM,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    device: torch.device,
) -> EpochMetrics:
    """S4: one BF16 epoch with BCE, partial-AUC, and DDA pair terms."""
    if model.memory.prototypes.requires_grad:
        raise ValueError("The authentic memory must remain frozen throughout S4.")
    model.train().to(device)
    sums = defaultdict(float)
    subtype_sums = defaultdict(float)
    subtype_counts = defaultdict(int)
    conflict_diagnostics: dict[str, float] = {}
    steps = 0
    for batch in loader:
        pixels = torch.as_tensor(batch["pixel_values"], device=device)
        labels = torch.as_tensor(batch["target"], device=device).float()
        default_weights = torch.ones_like(labels)
        weights = torch.as_tensor(batch.get("balance_weight", default_weights), device=device).float()
        primary_mask = torch.tensor(
            primary_objective_mask(
                batch,
                include_dda=loss_config.dda_in_primary_objective,
            ),
            dtype=torch.bool,
            device=device,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and optimizer_config.mixed_precision == "bf16"
            else nullcontext()
        )
        with autocast:
            logits = model(pixels).logit
            dda_logits, real_logits = paired_dda_logits(batch, logits)
            losses = detection_objective(
                logits,
                labels,
                weights,
                dda_logits=dda_logits,
                source_real_logits=real_logits,
                config=loss_config,
                primary_mask=primary_mask,
            )
        if steps == 0:
            conflict_diagnostics = gradient_conflict_cosines(
                {"bce": losses.bce, "pauc": losses.pauc, "pair": losses.pair},
                model.parameters(),
            )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            optimizer_config.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        for name in ("total", "bce", "pauc", "pair"):
            sums[name] += float(getattr(losses, name).detach())

        per_sample = F.binary_cross_entropy_with_logits(logits.detach(), labels, reduction="none")
        subtype_values = list(
            map(
                str,
                batch.get(
                    "authentic_subtype",
                    batch.get("source_dataset", ["unknown"] * len(labels)),
                ),
            )
        )
        for index, (label, subtype) in enumerate(zip(labels, subtype_values, strict=True)):
            if not bool(label):
                subtype_sums[subtype] += float(per_sample[index])
                subtype_counts[subtype] += 1
        steps += 1
    if not steps:
        raise ValueError("Cannot train from an empty loader.")
    subtype_losses = {
        name: subtype_sums[name] / subtype_counts[name]
        for name in subtype_sums
    }
    authentic_count = sum(subtype_counts.values())
    mean_authentic_loss = (
        sum(subtype_sums.values()) / authentic_count
        if authentic_count
        else float("nan")
    )
    return EpochMetrics(
        total=sums["total"] / steps,
        bce=sums["bce"] / steps,
        pauc=sums["pauc"] / steps,
        pair=sums["pair"] / steps,
        mean_authentic_loss=mean_authentic_loss,
        worst_authentic_subtype_loss=max(subtype_losses.values(), default=float("nan")),
        subtype_losses=subtype_losses,
        gradient_conflicts=conflict_diagnostics,
    )


def gradient_conflict_cosines(
    losses: Mapping[str, Tensor],
    parameters: Iterable[nn.Parameter],
) -> dict[str, float]:
    """Compute the proposal's C_ij gradient-conflict diagnostics."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    gradients: dict[str, Tensor] = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat([
            (torch.zeros_like(parameter) if value is None else value).flatten()
            for parameter, value in zip(trainable, values, strict=True)
        ])
    result: dict[str, float] = {}
    names = list(gradients)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            numerator = torch.dot(gradients[left], gradients[right])
            denominator = gradients[left].norm() * gradients[right].norm()
            result[f"{left}:{right}"] = float(
                (numerator / denominator.clamp_min(1e-12)).detach()
            )
    return result


def save_detector_checkpoint(
    model: TraceRXM,
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
    """Save portable weights plus optional state needed to resume an S4 run."""
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
) -> tuple[TraceRXM, dict[str, Any]]:
    """Reconstruct a LoRA or frozen-fallback detector from S3/S4 artifacts."""

    from .backbone import DinoV3PatchEncoder
    from .config import TraceRXMConfig

    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if artifact.get("stage") != "S4" or artifact.get("encoder_mode") not in {"lora", "frozen"}:
        raise ValueError("Invalid or legacy S4 detector checkpoint.")
    if artifact.get("source_memory_sha256") != file_sha256(memory_path):
        raise ValueError("S4 detector and S3 memory artifact hashes disagree.")
    config = TraceRXMConfig.from_dict(artifact["config"])
    backbone_config = config.backbone
    if artifact["encoder_mode"] == "frozen":
        backbone_config = replace(backbone_config, lora_rank=0)
    memory = load_memory_artifact(memory_path)
    model = TraceRXM(DinoV3PatchEncoder(backbone_config), memory, config.head)
    incompatible = model.load_state_dict(artifact["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith("encoder.")
    ]
    if unexpected or invalid_missing:
        raise ValueError(
            f"Checkpoint state mismatch; unexpected={unexpected}, missing={invalid_missing}"
        )
    model.requires_grad_(False)
    model.eval().to(device)
    return model, artifact
