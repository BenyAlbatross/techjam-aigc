#!/usr/bin/env python3
"""Evaluate a frozen TRACE-RX-M detector on modular transformed datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from techjam_aigc.trace_rx_m.config import TraceRXMConfig
from techjam_aigc.trace_rx_m.evaluation import (
    ALWAYS_REPORT_METRICS,
    ROBUSTNESS_METRICS,
    clean_to_condition_drops,
    comparison_slice_metrics,
    metric_slices,
    paired_endpoint_drift,
    robustness_summary,
)
from techjam_aigc.trace_rx_m.evaluation_data import (
    AssignedChainEndpointDataset,
    TransformEndpointDataset,
    assigned_chain_banks,
    assigned_transform_specs,
    load_evaluation_datasets,
    resolve_transform_specs,
    transform_registry,
)
from techjam_aigc.trace_rx_m.training import file_sha256, load_detector_checkpoint


PREDICTION_COLUMNS = (
    "dataset_id",
    "parent_id",
    "lineage_id",
    "image_path",
    "target",
    "generator_family",
    "generation_model",
    "source_dataset",
    "authentic_subtype",
    "condition",
    "transform_family",
    "severity",
    "official_transform",
    "transform_design",
    "transform_step_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each JSON-configured CSV or class-folder dataset.",
    )
    parser.add_argument(
        "--transform-profile",
        action="append",
        default=None,
        help="Repeat to union transform profiles; defaults to core.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help="Optional exact condition allowlist; clean remains mandatory.",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Exclude evaluation-only compositions from the selected profiles.",
    )
    parser.add_argument(
        "--assigned-chain-length",
        action="append",
        type=int,
        choices=(1, 2, 3),
        help=(
            "Assign one deterministic recipe of this length to every parent. "
            "Repeat for multiple lengths; incompatible with --condition and --official-only."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--max-images-per-dataset",
        type=int,
        help="Deterministic smoke-test limit; never use for headline results.",
    )
    return parser.parse_args()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _limit_records(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None:
        return frame
    if maximum < 2:
        raise ValueError("--max-images-per-dataset must be at least two.")
    pieces = []
    for _, dataset in frame.groupby("dataset_id", sort=True):
        if len(dataset) <= maximum:
            pieces.append(dataset)
            continue
        pieces.append(dataset.sample(n=maximum, random_state=seed))
    return pd.concat(pieces, ignore_index=True)


def _batch_rows(batch, logits, probabilities) -> list[dict[str, object]]:
    rows = []
    for index in range(len(logits)):
        row = {}
        for column in PREDICTION_COLUMNS:
            value = batch[column][index]
            if hasattr(value, "item"):
                value = value.item()
            row[column] = value
        row["logit"] = float(logits[index])
        row["pred"] = float(probabilities[index])
        rows.append(row)
    return rows


def _run_inference(model, loader, device) -> pd.DataFrame:
    import torch

    rows: list[dict[str, object]] = []
    model.eval()
    started = time.monotonic()
    total_batches = len(loader)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            pixels = torch.as_tensor(batch["pixel_values"], device=device)
            logits = model(pixels).logit.float().cpu().numpy()
            probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
            rows.extend(_batch_rows(batch, logits, probabilities))
            if batch_index % 50 == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - started
                rate = len(rows) / elapsed if elapsed else float("nan")
                print(
                    f"Inference {batch_index}/{total_batches} batches; "
                    f"{len(rows)} endpoints; {rate:.2f} endpoints/s",
                    flush=True,
                )
    return pd.DataFrame(rows)


def _prediction_json_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    columns = (
        "dataset_id",
        "parent_id",
        "image_path",
        "condition",
        "transform_family",
        "target",
        "pred",
    )
    return predictions[list(columns)].to_dict("records")


def main() -> None:
    args = parse_args()
    if not 0 < args.max_fpr <= 1:
        raise ValueError("--max-fpr must lie in (0, 1].")
    config = TraceRXMConfig.load(args.config)
    repo_root = args.repo_root.resolve()
    records, dataset_specs = load_evaluation_datasets(
        args.dataset_spec,
        repo_root=repo_root,
        verify_paths=True,
    )
    records = _limit_records(records, args.max_images_per_dataset, args.seed)
    if args.assigned_chain_length:
        if args.condition or args.official_only:
            raise ValueError(
                "Assigned chain evaluation is incompatible with --condition and --official-only."
            )
        banks = assigned_chain_banks(args.assigned_chain_length)
        transforms = assigned_transform_specs(banks)
        registry = transform_registry(transforms)
        endpoint_dataset = AssignedChainEndpointDataset(
            records,
            banks,
            image_size=config.backbone.image_size,
            base_seed=args.seed,
        )
        assignment_policy = "one_sha256_assigned_recipe_per_parent_and_chain_length"
    else:
        transforms = resolve_transform_specs(
            args.transform_profile or ("core",),
            conditions=args.condition,
            official_only=args.official_only,
        )
        registry = transform_registry(transforms)
        endpoint_dataset = TransformEndpointDataset(
            records,
            transforms,
            image_size=config.backbone.image_size,
            base_seed=args.seed,
        )
        assignment_policy = "exhaustive_selected_condition_matrix"

    import torch

    device = torch.device(args.device)
    model, checkpoint = load_detector_checkpoint(args.checkpoint, args.memory, device=device)
    if config.to_dict() != checkpoint["config"]:
        raise ValueError("Evaluator config does not match the frozen checkpoint config.")
    loader = torch.utils.data.DataLoader(
        endpoint_dataset,
        batch_size=args.batch_size or config.data.batch_size,
        shuffle=False,
        num_workers=config.data.workers if args.workers is None else args.workers,
        pin_memory=device.type == "cuda",
    )
    predictions = _run_inference(model, loader, device)
    if len(predictions) != len(endpoint_dataset):
        raise RuntimeError("Inference did not produce one prediction per transformed endpoint.")

    condition_metrics = metric_slices(
        predictions,
        ["dataset_id", "condition", "transform_family"],
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    dataset_metrics = metric_slices(
        predictions,
        ["dataset_id"],
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    family_metrics = metric_slices(
        predictions,
        ["dataset_id", "transform_family"],
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    chain_length_metrics = metric_slices(
        predictions,
        ["dataset_id", "transform_step_count"],
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    generator_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group=None,
        output_group="comparison_generator_family",
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    generation_model_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generation_model",
        negative_group=None,
        output_group="comparison_generation_model",
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    authentic_source_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group="authentic_subtype",
        output_group="comparison_authentic_subtype",
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    drops = clean_to_condition_drops(condition_metrics)
    drift = paired_endpoint_drift(predictions, threshold=args.threshold)
    summary = robustness_summary(condition_metrics, registry)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    _write_json(output / "predictions.json", _prediction_json_rows(predictions))
    registry.to_csv(output / "transform_registry.csv", index=False)
    condition_metrics.to_csv(output / "metrics_by_condition.csv", index=False)
    dataset_metrics.to_csv(output / "metrics_by_dataset.csv", index=False)
    family_metrics.to_csv(output / "metrics_by_transform_family.csv", index=False)
    chain_length_metrics.to_csv(output / "metrics_by_chain_length.csv", index=False)
    generator_metrics.to_csv(output / "metrics_by_generator.csv", index=False)
    generation_model_metrics.to_csv(output / "metrics_by_generation_model.csv", index=False)
    authentic_source_metrics.to_csv(output / "metrics_by_authentic_source.csv", index=False)
    drops.to_csv(output / "clean_to_condition_drop.csv", index=False)
    drift.to_csv(output / "paired_endpoint_drift.csv", index=False)
    _write_json(output / "summary.json", summary)
    _write_json(output / "run_metadata.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "memory": str(args.memory.resolve()),
        "memory_sha256": file_sha256(args.memory),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "dataset_specs": [
            {**asdict(spec), "path": str(spec.path), "sha256": file_sha256(spec.path)}
            for spec in dataset_specs
        ],
        "source_images": int(len(records)),
        "transformed_endpoints": int(len(endpoint_dataset)),
        "transform_profiles": args.transform_profile or ["core"],
        "conditions": [spec.name for spec in transforms],
        "assigned_chain_lengths": args.assigned_chain_length,
        "condition_assignment_policy": assignment_policy,
        "official_only": bool(args.official_only),
        "seed": args.seed,
        "threshold": args.threshold,
        "threshold_scale": "logit",
        "max_fpr": args.max_fpr,
        "always_report_metrics": ALWAYS_REPORT_METRICS,
        "robustness_metrics": ROBUSTNESS_METRICS,
        "smoke_test_limit_per_dataset": args.max_images_per_dataset,
    })


if __name__ == "__main__":
    main()
