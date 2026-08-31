#!/usr/bin/env python3
"""Run the strictly ordered TRACE-RX-Parallel training stages."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from techjam_aigc.trace_rx_m.calibration import PositiveTemperatureScaler
from techjam_aigc.trace_rx_parallel.config import TraceRXParallelConfig
from techjam_aigc.trace_rx_m.data import (
    BalancedTraceBatchSampler,
    TraceRXMDataset,
    add_balance_weights,
    load_training_manifest,
)
from techjam_aigc.trace_rx_m.quality import QUALITY_DIMENSION
from techjam_aigc.trace_rx_m.reliability import (
    PassiveQualityStacker,
    ReliabilityTable,
    audit_heldout_availability,
    audit_quality_cell_occupancy,
    normalized_partial_auc,
)


QUALITY_COLUMNS = (
    "log_min_dimension",
    "noise_sigma",
    "blockiness",
    "structural_hf_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "protocol", "cache", "capacity", "detection", "reliability", "calibration"
    ))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scores", type=Path, help="S5/S6 CSV containing logits and quality fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume-detection",
        action="store_true",
        help="Resume S4 from the latest validated local periodic checkpoint.",
    )
    return parser.parse_args()


def _require_manifest(path: Path | None) -> Path:
    if path is None:
        raise ValueError("This stage requires --manifest.")
    return path.resolve()


def _require_scores(path: Path | None) -> Path:
    if path is None:
        raise ValueError("This stage requires --scores.")
    return path.resolve()


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install training dependencies with `uv sync --group train`.") from error
    return torch


def _require_passed_protocol(output: Path) -> dict[str, Any]:
    path = output / "s0_protocol.json"
    if not path.exists():
        raise FileNotFoundError("S1 requires a passed s0_protocol.json in --output.")
    audit = json.loads(path.read_text())
    if audit.get("passed") is not True:
        raise RuntimeError("S0 protocol gate did not pass; fix data before caching features.")
    return audit


def _loader(frame: pd.DataFrame, repo_root: Path, config: TraceRXParallelConfig, *, augment: bool):
    torch = _torch()
    transform_sampler = None
    if augment:
        from techjam_aigc.trace_rx_m.augment import SymmetricTransformSampler
        transform_sampler = SymmetricTransformSampler(
            held_out_family=config.data.held_out_transform_family,
            base_seed=config.data.seed,
            clean_probability=config.data.clean_probability,
        )
    dataset = TraceRXMDataset(
        frame,
        repo_root,
        transform_sampler=transform_sampler,
        image_size=config.backbone.image_size,
    )
    if augment:
        sampler = BalancedTraceBatchSampler(
            frame,
            batch_size=config.data.batch_size,
            dda_positive_share=config.data.dda_positive_share,
            seed=config.data.seed,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.data.workers,
            pin_memory=True,
        ), dataset, sampler
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=True,
    ), dataset, None


def run_protocol(args: argparse.Namespace, config: TraceRXParallelConfig) -> None:
    from techjam_aigc.trace_rx_m.protocol import run_nuisance_probes
    from techjam_aigc.trace_rx_m.training import file_sha256

    config.validate(require_backbone_access=True)
    manifest_path = _require_manifest(args.manifest)
    manifest = load_training_manifest(manifest_path)
    generated = manifest.loc[manifest["sample_kind"].eq("native_aigc"), "generator_family"]
    probe_frame = manifest[manifest["role"].eq("supervised")]
    nuisance = run_nuisance_probes(
        probe_frame,
        folds=config.protocol.nuisance_folds,
        seed=config.data.seed,
    )
    summary = {
        "passed": max(nuisance.values()) <= config.protocol.nuisance_max_auc,
        "manifest_sha256": file_sha256(manifest_path),
        "rows": len(manifest),
        "roles": manifest["role"].value_counts().to_dict(),
        "generator_family_count": int(generated.nunique()),
        "generator_families": sorted(generated.unique().tolist()),
        "nuisance_auc": nuisance,
        "nuisance_max_auc": config.protocol.nuisance_max_auc,
        "resampler_parity": "training and evaluation use feature_lab.transforms",
        "lowest_resolution_behavior": (
            f"endpoints are resized to {config.backbone.image_size} square; "
            "the pinned backbone's patch size determines the token grid"
        ),
        "warning": "Fewer than eight generator families" if generated.nunique() < 8 else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s0_protocol.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_cache(args: argparse.Namespace, config: TraceRXParallelConfig) -> None:
    torch = _torch()
    from techjam_aigc.trace_rx_m.backbone import DinoV3PatchEncoder
    from techjam_aigc.trace_rx_m.training import cache_patch_features, file_sha256

    config.validate(require_backbone_access=True)
    protocol = _require_passed_protocol(args.output)
    manifest_path = _require_manifest(args.manifest)
    if protocol.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("S0 protocol artifact belongs to a different manifest.")
    manifest = load_training_manifest(manifest_path)
    unadapted = replace(config.backbone, lora_rank=0)
    encoder = DinoV3PatchEncoder(unadapted)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for role in ("memory_pool", "capacity_validation"):
        frame = manifest[manifest["role"].eq(role)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Manifest has no {role} rows.")
        loader, _, _ = _loader(frame, args.repo_root, config, augment=False)
        cache_patch_features(
            encoder,
            loader,
            args.output / f"s1_{role}.pt",
            device=device,
            backbone_model_id=config.backbone.model_id,
            backbone_revision=str(config.backbone.revision),
            manifest_sha256=file_sha256(manifest_path),
        )


def run_capacity(args: argparse.Namespace, config: TraceRXParallelConfig) -> None:
    torch = _torch()
    from techjam_aigc.trace_rx_parallel.integrations import WandbTracker
    from techjam_aigc.trace_rx_m.training import (
        evaluate_memory_coverage, file_sha256, fit_authentic_memory,
        load_feature_cache, prototype_usage_histogram, save_memory_artifact,
        select_capacity,
    )

    device = torch.device(args.device)
    memory_cache_path = args.output / "s1_memory_pool.pt"
    capacity_cache_path = args.output / "s1_capacity_validation.pt"
    memory_cache = load_feature_cache(memory_cache_path)
    capacity_cache = load_feature_cache(capacity_cache_path)
    if memory_cache["backbone_revision"] != capacity_cache["backbone_revision"]:
        raise ValueError("S1 caches use different backbone revisions.")
    for field in ("backbone_model_id", "manifest_sha256"):
        if memory_cache[field] != capacity_cache[field]:
            raise ValueError(f"S1 caches use different {field} values.")

    tracker = WandbTracker(
        config.tracking,
        stage="s2-s3-memory-capacity",
        run_config=config.to_dict(),
    )
    candidates = []
    fitted = {}
    usage_reports = []
    threshold = config.memory.tail_error_threshold
    tracking_step = 0
    for size in config.memory.candidate_sizes:
        for topk in config.memory.candidate_topk:
            if topk > size:
                continue
            model, history = fit_authentic_memory(
                memory_cache["tokens"], size=size, topk=topk,
                config=config.memory, optimizer_config=config.optimizer,
                device=device, seed=config.data.seed,
            )
            for epoch_metrics in history:
                tracking_step += 1
                tracker.log({
                    f"memory/{size}x{topk}/{name}": value
                    for name, value in epoch_metrics.items()
                }, step=tracking_step)
            usage = prototype_usage_histogram(
                model, memory_cache["tokens"], device=device
            )
            dead = int((usage == 0).sum())
            maximum_share = float(usage.max() / usage.sum().clamp_min(1))
            usage_gate = (
                dead == 0 and maximum_share <= config.memory.max_prototype_usage_share
            )
            usage_reports.append({
                "size": size,
                "topk": topk,
                "dead_prototypes": dead,
                "maximum_usage_share": maximum_share,
                "passed": usage_gate,
                "histogram": usage.tolist(),
            })
            if not usage_gate:
                continue
            coverage = evaluate_memory_coverage(
                model, capacity_cache["tokens"], capacity_cache["authentic_subtypes"],
                tail_threshold=threshold, tail_quantile=config.memory.tail_quantile,
                device=device,
            )
            if threshold is None:
                threshold = coverage.tail_threshold
            candidates.append(coverage)
            fitted[(size, topk)] = (model, history)
    selected = select_capacity(
        candidates, relative_tolerance=config.memory.capacity_relative_tolerance
    )
    memory, history = fitted[(selected.size, selected.topk)]
    save_memory_artifact(
        memory, args.output / "s3_memory.pt",
        source_cache_sha256=file_sha256(memory_cache_path),
        backbone_model_id=str(memory_cache["backbone_model_id"]),
        backbone_revision=str(memory_cache["backbone_revision"]),
        manifest_sha256=str(memory_cache["manifest_sha256"]),
        history=history, coverage=selected,
    )
    best_tail_by_size = {
        size: min(
            item.worst_subtype_tail_fraction
            for item in candidates
            if item.size == size
        )
        for size in sorted({item.size for item in candidates})
    }
    curve_sizes = sorted(best_tail_by_size)
    curve_still_rising = False
    if len(curve_sizes) >= 2:
        previous = best_tail_by_size[curve_sizes[-2]]
        latest = best_tail_by_size[curve_sizes[-1]]
        relative_improvement = (previous - latest) / max(previous, 1e-12)
        curve_still_rising = (
            curve_sizes[-1] == max(config.memory.candidate_sizes)
            and relative_improvement > config.memory.capacity_relative_tolerance
        )
    report = {
        "common_tail_threshold": threshold,
        "selected": selected.__dict__,
        "candidates": [candidate.__dict__ for candidate in candidates],
        "prototype_usage": usage_reports,
        "best_worst_subtype_tail_by_size": best_tail_by_size,
        "curve_still_rising_at_maximum": curve_still_rising,
        "undersized": curve_still_rising,
        "undersizing_failure_direction": (
            "elevated worst-authentic-subtype false-positive rate"
            if curve_still_rising else None
        ),
    }
    (args.output / "s3_capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    tracker.summarize({
        "capacity/selected_size": selected.size,
        "capacity/selected_topk": selected.topk,
        "capacity/worst_subtype_tail_fraction": selected.worst_subtype_tail_fraction,
        "capacity/undersized": curve_still_rising,
    })
    tracker.finish()


def _evaluate_aucs(model, loader, device) -> dict[str, float]:
    torch = _torch()
    from sklearn.metrics import roc_auc_score
    fused_logits, global_logits, memory_logits, labels = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(torch.as_tensor(batch["pixel_values"], device=device))
            fused_logits.extend(output.logit.cpu().tolist())
            global_logits.extend(output.global_logit.cpu().tolist())
            memory_logits.extend(output.memory_logit.cpu().tolist())
            labels.extend(torch.as_tensor(batch["target"]).tolist())
    if len(set(labels)) != 2:
        raise ValueError("Development evaluation needs both authentic and AIGC rows.")
    return {
        "fused": float(roc_auc_score(labels, fused_logits)),
        "global": float(roc_auc_score(labels, global_logits)),
        "memory": float(roc_auc_score(labels, memory_logits)),
    }


def _epoch_log_values(metrics, *, variant: str, optimizer, model) -> dict[str, float]:
    values = {
        f"{variant}/train/{name}": float(getattr(metrics, name))
        for name in (
            "total", "bce", "pauc", "pair", "mean_authentic_loss",
            "worst_authentic_subtype_loss",
        )
    }
    values.update({
        f"{variant}/authentic_subtype/{name}": float(value)
        for name, value in metrics.subtype_losses.items()
    })
    values.update({
        f"{variant}/gradient_conflict/{name.replace(':', '_')}": float(value)
        for name, value in metrics.gradient_conflicts.items()
    })
    values.update({
        f"{variant}/learning_rate/{group.get('name', index)}": float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    })
    values.update({
        f"{variant}/fusion/global_weight": float(model.fusion.weight[0, 0].detach()),
        f"{variant}/fusion/memory_weight": float(model.fusion.weight[0, 1].detach()),
        f"{variant}/fusion/bias": float(model.fusion.bias[0].detach()),
    })
    return values


def _train_detection_variant(
    *,
    model,
    loader,
    selection_loader,
    dataset,
    sampler,
    config: TraceRXParallelConfig,
    device,
    variant: str,
    output: Path,
    memory_sha256: str,
    manifest_sha256: str,
    tracker,
    publisher,
    step_offset: int,
    resume: bool,
):
    from techjam_aigc.trace_rx_parallel.training import (
        authentic_subtype_regressed,
        build_detection_optimizer,
        cosine_warmup_scheduler,
        save_detector_checkpoint,
        train_detection_epoch,
    )
    from techjam_aigc.trace_rx_m.training import seed_everything

    model.to(device)
    optimizer = build_detection_optimizer(model, config.optimizer)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps=config.optimizer.detection_epochs * len(loader),
        warmup_fraction=config.optimizer.warmup_fraction,
    )
    history = []
    best_selection_auc = float("-inf")
    best_epoch = 0
    checkpoint_directory = output / "checkpoints" / variant
    best_path = checkpoint_directory / "best_detector.pt"
    final_path = checkpoint_directory / "final_detector.pt"
    start_epoch = 0
    if resume:
        periodic_paths = sorted(checkpoint_directory.glob("epoch-*.pt"))
        if periodic_paths:
            resume_path = periodic_paths[-1]
            artifact = torch.load(resume_path, map_location="cpu", weights_only=True)
            expected_mode = "frozen" if variant == "frozen" else "lora"
            expected = {
                "stage": "S4",
                "architecture": "trace-rx-parallel",
                "encoder_mode": expected_mode,
                "source_memory_sha256": memory_sha256,
                "manifest_sha256": manifest_sha256,
                "config": config.to_dict(),
            }
            mismatches = [
                field for field, value in expected.items() if artifact.get(field) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Cannot resume {variant} from {resume_path}: mismatched {mismatches}."
                )
            incompatible = model.load_state_dict(artifact["model_state"], strict=False)
            invalid_missing = [
                name for name in incompatible.missing_keys if not name.startswith("encoder.")
            ]
            if incompatible.unexpected_keys or invalid_missing:
                raise ValueError(
                    "Resume checkpoint state mismatch; "
                    f"unexpected={list(incompatible.unexpected_keys)}, "
                    f"missing={invalid_missing}."
                )
            optimizer.load_state_dict(artifact["optimizer_state"])
            scheduler.load_state_dict(artifact["scheduler_state"])
            history = [EpochMetrics(**row) for row in artifact.get("history", [])]
            start_epoch = int(artifact["epoch"])
            if start_epoch != len(history):
                raise ValueError("Resume checkpoint epoch does not match its metric history.")
            if not best_path.is_file():
                raise FileNotFoundError("Resume requires the matching local best checkpoint.")
            best_artifact = torch.load(best_path, map_location="cpu", weights_only=True)
            best_selection = best_artifact.get("selection_metric") or {}
            best_epoch = int(best_artifact["epoch"])
            best_selection_auc = float(best_selection["value"])
            publisher.upload_best(best_path, epoch=best_epoch, variant=variant)
            tracker.summarize({
                f"{variant}/resumed_from_epoch": start_epoch,
                f"{variant}/resumed_from_checkpoint": resume_path.name,
            })

    stopped_early = False
    rejected_epoch = None
    for epoch_index in range(start_epoch, config.optimizer.detection_epochs):
        epoch = epoch_index + 1
        seed_everything(config.data.seed + epoch_index)
        dataset.set_epoch(epoch_index)
        sampler.set_epoch(epoch_index)
        metrics = train_detection_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            loss_config=config.loss,
            optimizer_config=config.optimizer,
            device=device,
        )
        if history and authentic_subtype_regressed(history[-1], metrics):
            stopped_early = True
            rejected_epoch = epoch
            tracker.log(
                {
                    f"{variant}/early_stop/triggered": 1,
                    f"{variant}/early_stop/rejected_epoch": epoch,
                    f"{variant}/early_stop/candidate_mean_authentic_loss": (
                        metrics.mean_authentic_loss
                    ),
                    f"{variant}/early_stop/candidate_worst_authentic_subtype_loss": (
                        metrics.worst_authentic_subtype_loss
                    ),
                },
                step=step_offset + epoch,
            )
            rollback_path = checkpoint_directory / f"epoch-{epoch - 1:04d}.pt"
            rollback = torch.load(rollback_path, map_location="cpu", weights_only=True)
            model.load_state_dict(rollback["model_state"], strict=False)
            break
        history.append(metrics)
        selection_aucs = _evaluate_aucs(model, selection_loader, device)
        log_values = _epoch_log_values(
            metrics,
            variant=variant,
            optimizer=optimizer,
            model=model,
        )
        log_values.update(
            {
                f"{variant}/model_selection/{branch}_roc_auc": value
                for branch, value in selection_aucs.items()
            }
        )
        tracker.log(log_values, step=step_offset + epoch)
        selection = {
            "name": "development_fused_roc_auc",
            "mode": "max",
            "value": selection_aucs["fused"],
            "epoch": epoch,
            "branch_roc_auc": selection_aucs,
            "reason": (
                "non-held-out development groups select epochs; the configured "
                "generator family remains reserved for the one-time S4 validity gate"
            ),
        }
        if selection_aucs["fused"] > best_selection_auc:
            best_selection_auc = selection_aucs["fused"]
            best_epoch = epoch
            save_detector_checkpoint(
                model,
                best_path,
                config=config.to_dict(),
                memory_artifact_sha256=memory_sha256,
                manifest_sha256=manifest_sha256,
                history=history,
                epoch=epoch,
                selection_metric=selection,
            )
            publisher.upload_best(best_path, epoch=epoch, variant=variant)
        if epoch % config.hub.checkpoint_every_epochs == 0:
            periodic_path = checkpoint_directory / f"epoch-{epoch:04d}.pt"
            save_detector_checkpoint(
                model,
                periodic_path,
                config=config.to_dict(),
                memory_artifact_sha256=memory_sha256,
                manifest_sha256=manifest_sha256,
                history=history,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                selection_metric=selection,
            )
            publisher.upload_periodic(periodic_path, epoch=epoch, variant=variant)

    save_detector_checkpoint(
        model,
        final_path,
        config=config.to_dict(),
        memory_artifact_sha256=memory_sha256,
        manifest_sha256=manifest_sha256,
        history=history,
        epoch=len(history),
        selection_metric={
            "name": "final_epoch",
            "mode": "last",
            "value": len(history),
            "epoch": len(history),
        },
    )
    return (
        best_path,
        final_path,
        best_epoch,
        best_selection_auc,
        {
            "triggered": stopped_early,
            "rejected_epoch": rejected_epoch,
            "accepted_epochs": len(history),
        },
    )


def run_detection(args: argparse.Namespace, config: TraceRXParallelConfig) -> None:
    torch = _torch()
    from techjam_aigc.trace_rx_m.backbone import DinoV3PatchEncoder
    from techjam_aigc.trace_rx_parallel.integrations import (
        HubCheckpointPublisher,
        WandbTracker,
    )
    from techjam_aigc.trace_rx_parallel.model import TraceRXParallel
    from techjam_aigc.trace_rx_m.training import (
        file_sha256, load_memory_artifact, seed_everything,
    )

    config.validate(require_backbone_access=True, require_remote_artifacts=True)
    family = config.data.held_out_generator_family
    if not family:
        raise ValueError("Set data.held_out_generator_family before S4.")
    manifest_path = _require_manifest(args.manifest)
    manifest_hash = file_sha256(manifest_path)
    manifest = load_training_manifest(manifest_path)
    train = manifest[manifest["role"].eq("supervised")].copy()
    heldout_in_train = train["generator_family"].astype(str).str.casefold().eq(family.casefold())
    train = add_balance_weights(train[~heldout_in_train].reset_index(drop=True))
    development = manifest[manifest["role"].eq("development")].copy()
    if "development_purpose" in development:
        selection_development = development[
            development["development_purpose"].eq("model_selection")
        ].reset_index(drop=True)
        gate_development = development[
            development["development_purpose"].eq("generator_gate")
        ].reset_index(drop=True)
    else:
        heldout_dev = development["generator_family"].astype(str).str.casefold().eq(
            family.casefold()
        )
        selection_development = development[
            development["target"].eq(0) | (~heldout_dev)
        ].reset_index(drop=True)
        gate_development = development[
            development["target"].eq(0) | heldout_dev
        ].reset_index(drop=True)
    if heldout_in_train.sum() == 0:
        raise ValueError("The configured held-out generator family is absent from supervised rows.")
    gate_positive_families = set(
        gate_development.loc[
            gate_development["target"].eq(1), "generator_family"
        ].astype(str).str.casefold()
    )
    if gate_positive_families != {family.casefold()}:
        raise ValueError(
            "Generator-gate positives must contain exactly the held-out family; "
            f"found {sorted(gate_positive_families)}."
        )
    if set(selection_development["target"]) != {0, 1} or set(
        gate_development["target"]
    ) != {0, 1}:
        raise ValueError("Both development purposes require authentic and AIGC rows.")

    device = torch.device(args.device)
    memory_path = args.output / "s3_memory.pt"
    memory_kwargs = {
        "expected_backbone_model_id": config.backbone.model_id,
        "expected_backbone_revision": str(config.backbone.revision),
        "expected_manifest_sha256": manifest_hash,
    }
    memory = load_memory_artifact(memory_path, **memory_kwargs)
    memory_hash = file_sha256(memory_path)
    seed_everything(config.data.seed)
    encoder = DinoV3PatchEncoder(config.backbone)
    model = TraceRXParallel(encoder, memory, config.head)
    model.configure_for_detection()
    loader, dataset, sampler = _loader(train, args.repo_root, config, augment=True)
    selection_loader, _, _ = _loader(
        selection_development,
        args.repo_root,
        config,
        augment=False,
    )
    gate_loader, _, _ = _loader(
        gate_development,
        args.repo_root,
        config,
        augment=False,
    )
    tracker = WandbTracker(
        config.tracking,
        stage="s4-detection",
        run_config=config.to_dict(),
    )
    exit_code = 1
    try:
        tracker.watch(model)
        summary_path = manifest_path.with_suffix(".summary.json")
        dataset_paths = [manifest_path]
        if summary_path.is_file():
            dataset_paths.append(summary_path)
        dataset_metadata = {
            "dataset_id": str(manifest.get("dataset_id", pd.Series(["unknown"])).iloc[0]),
            "dataset_revision": str(
                manifest.get("dataset_revision", pd.Series(["unknown"])).iloc[0]
            ),
            "manifest_sha256": manifest_hash,
            "training_rows": len(train),
            "model_selection_rows": len(selection_development),
            "generator_gate_rows": len(gate_development),
            "held_out_generator_family": family,
        }
        tracker.log_dataset_artifact(dataset_paths, metadata=dataset_metadata)
        tracker.summarize({
            f"dataset/{key}": value for key, value in dataset_metadata.items()
        })
        publisher = HubCheckpointPublisher(config.hub, run_id=tracker.run_id)
        best_variant_path, final_variant_path, best_epoch, best_selection_auc, lora_stop = (
            _train_detection_variant(
                model=model,
                loader=loader,
                selection_loader=selection_loader,
                dataset=dataset,
                sampler=sampler,
                config=config,
                device=device,
                variant="lora",
                output=args.output,
                memory_sha256=memory_hash,
                manifest_sha256=manifest_hash,
                tracker=tracker,
                publisher=publisher,
                step_offset=0,
                resume=args.resume_detection,
            )
        )
        gate_aucs = _evaluate_aucs(model, gate_loader, device)
        auc = gate_aucs["fused"]
        lora_gate_aucs = gate_aucs
        tracker.log(
            {
                f"lora/held_out_generator/{branch}_roc_auc": value
                for branch, value in gate_aucs.items()
            },
            step=config.optimizer.detection_epochs,
        )
        fallback_used = False
        selected_variant = "lora"
        frozen_gate_aucs = None
        frozen_stop = None
        if auc < config.data.held_out_min_roc_auc:
            # Pre-declared failure path: discard LoRA and retrain only the heads
            # against the same memory. Never retune adapters on this held-out result.
            fallback_used = True
            selected_variant = "frozen"
            memory = load_memory_artifact(memory_path, **memory_kwargs)
            seed_everything(config.data.seed)
            frozen_config = replace(config.backbone, lora_rank=0)
            model = TraceRXParallel(DinoV3PatchEncoder(frozen_config), memory, config.head)
            model.configure_for_detection(frozen_encoder_fallback=True)
            tracker.watch(model)
            best_variant_path, final_variant_path, best_epoch, best_selection_auc, frozen_stop = (
                _train_detection_variant(
                    model=model,
                    loader=loader,
                    selection_loader=selection_loader,
                    dataset=dataset,
                    sampler=sampler,
                    config=config,
                    device=device,
                    variant="frozen",
                    output=args.output,
                    memory_sha256=memory_hash,
                    manifest_sha256=manifest_hash,
                    tracker=tracker,
                    publisher=publisher,
                    step_offset=config.optimizer.detection_epochs,
                    resume=args.resume_detection,
                )
            )
            gate_aucs = _evaluate_aucs(model, gate_loader, device)
            frozen_gate_aucs = gate_aucs
            auc = gate_aucs["fused"]
            tracker.log(
                {
                    f"frozen/held_out_generator/{branch}_roc_auc": value
                    for branch, value in gate_aucs.items()
                },
                step=2 * config.optimizer.detection_epochs,
            )

        args.output.mkdir(parents=True, exist_ok=True)
        best_path = args.output / "s4_best_detector.pt"
        final_path = args.output / "s4_final_detector.pt"
        shipping_path = args.output / "s4_detector.pt"
        shutil.copy2(best_variant_path, best_path)
        shutil.copy2(final_variant_path, final_path)
        # The shipping detector remains the final epoch. The independently named
        # best checkpoint is selected on non-held-out dev groups and never on the
        # one-time generator gate.
        shutil.copy2(final_path, shipping_path)
        validity_path = args.output / "s4_validity.json"
        validity = {
            "held_out_generator_family": family,
            "roc_auc": auc,
            "held_out_branch_roc_auc": gate_aucs,
            "variant_held_out_branch_roc_auc": {
                "lora": lora_gate_aucs,
                "frozen": frozen_gate_aucs,
            },
            "threshold": config.data.held_out_min_roc_auc,
            "frozen_encoder_fallback": fallback_used,
            "selected_variant": selected_variant,
            "robustness_early_stop": {
                "lora": lora_stop,
                "frozen": frozen_stop,
            },
            "best_checkpoint": {
                "path": best_path.name,
                "epoch": best_epoch,
                "metric": "development_fused_roc_auc",
                "mode": "max",
                "value": best_selection_auc,
                "held_out_family_used_for_selection": False,
            },
            "final_checkpoint": final_path.name,
            "shipping_checkpoint": shipping_path.name,
            "wandb_run_id": tracker.run_id,
            "huggingface_repo_id": config.hub.repo_id,
        }
        validity_path.write_text(json.dumps(validity, indent=2) + "\n")
        final_metadata_paths = [
            memory_path,
            args.config,
            validity_path,
            manifest_path,
        ]
        if summary_path.is_file():
            final_metadata_paths.append(summary_path)
        hub_commit_url = publisher.upload_final_bundle(
            best_path=best_path,
            final_path=final_path,
            metadata_paths=final_metadata_paths,
        )
        validity["huggingface_final_commit"] = hub_commit_url
        validity_path.write_text(json.dumps(validity, indent=2) + "\n")
        tracker.summarize({
            "held_out_generator/roc_auc": auc,
            "held_out_generator/family": family,
            "frozen_encoder_fallback": fallback_used,
            "selected_variant": selected_variant,
            "best/epoch": best_epoch,
            "best/development_fused_roc_auc": best_selection_auc,
            "robustness_early_stop": bool(
                (frozen_stop if selected_variant == "frozen" else lora_stop)["triggered"]
            ),
            **{
                f"held_out_generator/{branch}_roc_auc": value
                for branch, value in gate_aucs.items()
            },
            "huggingface/final_commit": hub_commit_url,
        })
        tracker.log_model_artifact(
            (best_path, final_path, memory_path, args.config, validity_path),
            metadata={
                "selected_variant": selected_variant,
                "best_epoch": best_epoch,
                "best_development_fused_roc_auc": best_selection_auc,
                "held_out_generator_roc_auc": auc,
                "source_memory_sha256": memory_hash,
                "manifest_sha256": manifest_hash,
            },
        )
        exit_code = 0
    finally:
        tracker.finish(exit_code=exit_code)


def _scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "role", "lineage_id", "logit", "target", "detector_sha256", *QUALITY_COLUMNS
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")
    return frame


def _verify_score_provenance(
    frame: pd.DataFrame,
    *,
    artifact: Path,
    column: str,
) -> None:
    from techjam_aigc.trace_rx_m.training import file_sha256

    if not artifact.exists():
        raise FileNotFoundError(f"Required prior-stage artifact is missing: {artifact}")
    if column not in frame:
        raise ValueError(f"Score table must include {column} for stage provenance.")
    values = set(frame[column].astype(str))
    expected = file_sha256(artifact)
    if values != {expected}:
        raise ValueError(f"Score table {column} does not match {artifact.name}.")


def run_reliability(args: argparse.Namespace, config: TraceRXParallelConfig) -> None:
    from techjam_aigc.trace_rx_m.protocol import run_nuisance_probes

    frame = _scores(_require_scores(args.scores))
    _verify_score_provenance(
        frame,
        artifact=args.output / "s4_detector.pt",
        column="detector_sha256",
    )
    null = frame[frame["role"].eq("authentic_null")]
    development = frame[frame["role"].eq("development")]
    if null.empty or development.empty:
        raise ValueError("S5 needs authentic_null and development score rows.")
    if not null["target"].eq(0).all():
        raise ValueError("S5 authentic_null rows must all be authentic negatives.")
    if set(null["lineage_id"].astype(str)) & set(development["lineage_id"].astype(str)):
        raise ValueError("S5 authentic-null and development lineages must be disjoint.")
    post_probe = development.copy()
    post_probe["target"] = (
        post_probe["logit"].rank(method="average", pct=True) > 0.5
    ).astype(int)
    prediction_nuisance = run_nuisance_probes(
        post_probe,
        folds=config.protocol.nuisance_folds,
        seed=config.data.seed,
    )
    if max(prediction_nuisance.values()) > config.protocol.nuisance_max_auc:
        raise RuntimeError(
            "Post-training nuisance probe predicts the adapted model ranking above the S0 gate."
        )
    if "transform_family" not in development:
        raise ValueError("S5 development scores require transform_family.")
    heldout_mask = development["transform_family"].eq(config.data.held_out_transform_family)
    heldout = development[heldout_mask]
    fit_development = development[~heldout_mask]
    if heldout.empty:
        raise ValueError("S5 scores do not contain the configured held-out transform family.")
    table = ReliabilityTable.fit(
        authentic_null_logits=null["logit"].to_numpy(),
        authentic_null_quality=null[list(QUALITY_COLUMNS)].to_numpy(),
        development_logits=fit_development["logit"].to_numpy(),
        development_labels=fit_development["target"].to_numpy(),
        development_quality=fit_development[list(QUALITY_COLUMNS)].to_numpy(),
        clean_mask=fit_development["condition"].eq("clean").to_numpy(),
        n_bins=config.reliability.quality_bins,
        prior_strength=config.reliability.prior_strength,
        variance_floor=config.reliability.variance_floor,
    )
    audit = audit_heldout_availability(
        table,
        logits=heldout["logit"].to_numpy(),
        labels=heldout["target"].to_numpy(),
        quality=heldout[list(QUALITY_COLUMNS)].to_numpy(),
        min_samples_per_class=config.reliability.heldout_min_samples_per_class,
        min_spearman=config.reliability.heldout_min_spearman,
    )
    occupancy = audit_quality_cell_occupancy(
        table,
        quality=development[list(QUALITY_COLUMNS)].to_numpy(),
        conditions=development["condition"],
        transform_families=development["transform_family"],
        max_clean_noise_overlap=config.reliability.max_clean_noise_cell_overlap,
    )
    passive = PassiveQualityStacker().fit(
        fit_development["logit"].to_numpy(),
        fit_development[list(QUALITY_COLUMNS)].to_numpy(),
        fit_development["target"].to_numpy(),
    )
    heldout_labels = heldout["target"].to_numpy()
    availability_pauc = normalized_partial_auc(
        heldout_labels,
        table.fuse(
            heldout["logit"].to_numpy(),
            heldout[list(QUALITY_COLUMNS)].to_numpy(),
        ),
        max_fpr=config.loss.pauc_alpha,
    )
    passive_pauc = normalized_partial_auc(
        heldout_labels,
        passive.fused_logits(
            heldout["logit"].to_numpy(),
            heldout[list(QUALITY_COLUMNS)].to_numpy(),
        ),
        max_fpr=config.loss.pauc_alpha,
    )
    pauc_gain = availability_pauc - passive_pauc
    comparison_passed = (
        pauc_gain >= config.reliability.availability_min_normalized_pauc_gain
    )
    audit_json = {
        "cell_count": audit.cell_count,
        "spearman": audit.spearman if np.isfinite(audit.spearman) else None,
        "passed": audit.passed,
        "measured_d_prime": {str(cell): value for cell, value in audit.measured_d_prime.items()},
        "predicted_d_prime": {str(cell): value for cell, value in audit.predicted_d_prime.items()},
        "occupancy": {
            "clean_count": occupancy.clean_count,
            "noise_count": occupancy.noise_count,
            "distribution_overlap": occupancy.distribution_overlap,
            "cell_counts": {str(cell): count for cell, count in occupancy.cell_counts.items()},
            "passed": occupancy.passed,
        },
        "passive_comparison": {
            "max_fpr": config.loss.pauc_alpha,
            "availability_normalized_pauc": availability_pauc,
            "passive_normalized_pauc": passive_pauc,
            "gain": pauc_gain,
            "minimum_gain": config.reliability.availability_min_normalized_pauc_gain,
            "passed": comparison_passed,
        },
    }
    detector_hash = str(frame["detector_sha256"].iloc[0])
    if audit.passed and occupancy.passed and comparison_passed:
        artifact = {
            "mode": "availability",
            "table": table.to_dict(),
            "audit": audit_json,
            "prediction_nuisance_auc": prediction_nuisance,
            "source_detector_sha256": detector_hash,
        }
    else:
        artifact = {
            "mode": "passive_quality_fallback",
            "stacker": passive.to_dict(),
            "audit": audit_json,
            "prediction_nuisance_auc": prediction_nuisance,
            "source_detector_sha256": detector_hash,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s5_reliability.json").write_text(json.dumps(artifact, indent=2) + "\n")


def run_calibration(args: argparse.Namespace) -> None:
    frame = _scores(_require_scores(args.scores))
    _verify_score_provenance(
        frame,
        artifact=args.output / "s5_reliability.json",
        column="reliability_sha256",
    )
    _verify_score_provenance(
        frame,
        artifact=args.output / "s4_detector.pt",
        column="detector_sha256",
    )
    reliability = json.loads((args.output / "s5_reliability.json").read_text())
    detector_hashes = set(frame["detector_sha256"].astype(str))
    if detector_hashes != {str(reliability.get("source_detector_sha256"))}:
        raise ValueError("S6 score table and S5 reliability use different detectors.")
    calibration = frame[frame["role"].eq("calibration")]
    if calibration.empty or "fused_logit" not in calibration:
        raise ValueError("S6 needs calibration rows with fused_logit.")
    scaler = PositiveTemperatureScaler().fit(
        calibration["fused_logit"].to_numpy(), calibration["target"].to_numpy()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s6_calibration.json").write_text(json.dumps(scaler.to_dict(), indent=2) + "\n")


def main() -> None:
    args = parse_args()
    config = TraceRXParallelConfig.load(args.config)
    dispatch = {
        "protocol": lambda: run_protocol(args, config),
        "cache": lambda: run_cache(args, config),
        "capacity": lambda: run_capacity(args, config),
        "detection": lambda: run_detection(args, config),
        "reliability": lambda: run_reliability(args, config),
        "calibration": lambda: run_calibration(args),
    }
    dispatch[args.stage]()


if __name__ == "__main__":
    main()
