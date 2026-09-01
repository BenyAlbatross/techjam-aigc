#!/usr/bin/env python3
"""Evaluate the frozen TRACE-RX-M v2 detector on configured image datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from techjam_aigc.trace_rx_m.config import TraceRXMConfig
from techjam_aigc.trace_rx_m.evaluation import (
    ALWAYS_REPORT_METRICS,
    REPORT_METRICS,
    ROBUSTNESS_METRICS,
    clean_to_condition_drops,
    comparison_slice_metrics,
    metric_slices,
    paired_endpoint_drift,
    paired_lineage_drift,
    robustness_summary,
    score_distribution_slices,
)
from techjam_aigc.trace_rx_m.evaluation_data import (
    AsIsEndpointDataset,
    AssignedChainEndpointDataset,
    TransformEndpointDataset,
    UniformSequentialChainEndpointDataset,
    assigned_chain_banks,
    assigned_transform_specs,
    build_uniform_chain_assignments,
    load_evaluation_datasets,
    public_uniform_chain_assignments,
    resolve_transform_specs,
    transform_registry,
    uniform_chain_assignment_steps,
)
from techjam_aigc.trace_rx_m.training import file_sha256, load_detector_checkpoint


CURRENT_V2_BACKBONE_PREFIX = "facebook/dinov3-"
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
    "transform_variant",
    "severity",
    "official_transform",
    "transform_design",
    "transform_step_count",
    "endpoint_policy",
    "endpoint_class",
    "sampling_stratum",
    "ordered_operations",
    "ordered_recipe_json",
    "recipe_sha256",
    "ordered_families_json",
    "repeated_transform_family_count",
)
PREDICTION_DEFAULTS: dict[str, object] = {
    "transform_variant": "unknown",
    "endpoint_policy": "legacy",
    "endpoint_class": "unknown",
    "sampling_stratum": "not_applicable",
    "ordered_operations": "unknown",
    "ordered_recipe_json": "[]",
    "recipe_sha256": "",
    "ordered_families_json": "[]",
    "repeated_transform_family_count": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument(
        "--as-is-dataset-spec",
        type=Path,
        action="append",
        help="Repeat for datasets whose supplied assets are each evaluated once.",
    )
    parser.add_argument(
        "--uniform-chain-dataset-spec",
        type=Path,
        action="append",
        help="Repeat for datasets assigned exactly one counterbalanced 1--6-step chain.",
    )
    parser.add_argument(
        "--chain-length",
        action="append",
        type=int,
        choices=range(1, 7),
        help="Uniform-chain length to include; repeat as needed (default: all 1--6).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inventories, assignments, and representative 224 inputs without inference.",
    )
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        action="append",
        help="Legacy exhaustive-evaluation dataset specification.",
    )
    parser.add_argument(
        "--transform-profile",
        action="append",
        default=None,
        help="Legacy mode: repeat to union transform profiles; defaults to core.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help="Legacy mode: optional exact condition allowlist; clean remains mandatory.",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Legacy mode: exclude evaluation-only transform compositions.",
    )
    parser.add_argument(
        "--assigned-chain-length",
        action="append",
        type=int,
        choices=(1, 2, 3),
        help="Legacy mode: evaluate clean plus one bank recipe of this length.",
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
        help=(
            "Deterministic class/source-stratified sample size for every dataset; "
            "omit to evaluate full inventories."
        ),
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _proportional_quotas(counts: pd.Series, total: int) -> dict[object, int]:
    """Allocate an exact total proportionally with stable largest remainders."""

    if total < 0 or total > int(counts.sum()):
        raise ValueError("Sample quota must lie between zero and the available rows.")
    if counts.empty:
        return {}
    ideal = counts.astype(float) * total / int(counts.sum())
    quotas = np.floor(ideal).astype(int)
    remaining = total - int(quotas.sum())
    if remaining:
        order = sorted(
            counts.index,
            key=lambda key: (-(float(ideal.loc[key]) - int(quotas.loc[key])), str(key)),
        )
        for key in order[:remaining]:
            quotas.loc[key] += 1
    return {key: int(value) for key, value in quotas.items()}


def _source_sampling_stratum(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            frame["target"].eq(1),
            "aigc:" + frame["generation_model"].astype(str),
            "authentic:" + frame["authentic_subtype"].astype(str),
        ),
        index=frame.index,
        dtype="object",
    )


def _sample_order_key(seed: int, dataset_id: object, parent_id: object) -> str:
    return sha256(f"{seed}\0{dataset_id}\0{parent_id}".encode()).hexdigest()


def _limit_records(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    """Take an exact deterministic sample while preserving class/source mix."""

    if maximum is None:
        return frame.reset_index(drop=True)
    if maximum < 2:
        raise ValueError("--max-images-per-dataset must be at least two.")
    pieces = []
    for dataset_id, raw_dataset in frame.groupby("dataset_id", sort=True):
        dataset = raw_dataset.copy()
        if len(dataset) <= maximum:
            pieces.append(dataset)
            continue
        dataset["_sample_stratum"] = _source_sampling_stratum(dataset)
        target_counts = dataset["target"].value_counts().sort_index()
        target_quotas = _proportional_quotas(target_counts, maximum)
        selected = []
        for target, target_rows in dataset.groupby("target", sort=True):
            target_quota = target_quotas[target]
            stratum_counts = target_rows["_sample_stratum"].value_counts().sort_index()
            stratum_quotas = _proportional_quotas(stratum_counts, target_quota)
            for stratum, stratum_rows in target_rows.groupby(
                "_sample_stratum", sort=True
            ):
                quota = stratum_quotas[stratum]
                if not quota:
                    continue
                ordered = stratum_rows.assign(
                    _sample_order=[
                        _sample_order_key(seed, dataset_id, parent_id)
                        for parent_id in stratum_rows["parent_id"]
                    ]
                ).sort_values(["_sample_order", "parent_id"])
                selected.append(ordered.head(quota))
        sampled = pd.concat(selected).sort_index()
        if len(sampled) != maximum:
            raise RuntimeError(
                f"Deterministic sampling selected {len(sampled)} rows for "
                f"{dataset_id!r}, expected {maximum}."
            )
        pieces.append(sampled.drop(columns=["_sample_stratum", "_sample_order"]))
    return pd.concat(pieces, ignore_index=True)


def _strict_v2_config(path: Path) -> TraceRXMConfig:
    raw = json.loads(path.read_text())
    if "preprocessing" not in raw:
        raise ValueError(
            "The evaluator requires explicit v2 preprocessing metadata; legacy configs are rejected."
        )
    config = TraceRXMConfig.from_dict(raw)
    if not config.backbone.model_id.startswith(CURRENT_V2_BACKBONE_PREFIX):
        raise ValueError(
            "This evaluator targets TRACE-RX-M v2 DINOv3 checkpoints and rejects "
            "historical DINOv2 artifacts."
        )
    config.preprocessing.validate()
    return config


def _validate_v2_checkpoint(config: TraceRXMConfig, checkpoint: dict[str, Any]) -> None:
    frozen = checkpoint.get("config")
    if not isinstance(frozen, dict) or "preprocessing" not in frozen:
        raise ValueError(
            "Checkpoint has no explicit v2 preprocessing metadata and is incompatible."
        )
    if frozen != config.to_dict():
        raise ValueError("Evaluator config does not match the frozen checkpoint config.")


def _preflight_v2_checkpoint(path: Path, config: TraceRXMConfig) -> None:
    """Reject legacy checkpoints before constructing or downloading a backbone."""

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Detector checkpoint must contain an artifact dictionary.")
    _validate_v2_checkpoint(config, checkpoint)


def _batch_value(batch: dict[str, Any], column: str, index: int) -> object:
    if column not in batch:
        return PREDICTION_DEFAULTS.get(column, "unknown")
    value = batch[column][index]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _batch_rows(batch, logits, probabilities) -> list[dict[str, object]]:
    rows = []
    for index in range(len(logits)):
        row = {
            column: _batch_value(batch, column, index)
            for column in PREDICTION_COLUMNS
        }
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
        "image_path",
        "pred",
        "dataset_id",
        "parent_id",
        "condition",
        "transform_family",
        "target",
    )
    return predictions[list(columns)].to_dict("records")


def _spec_metadata(specs) -> list[dict[str, object]]:
    return [
        {**asdict(spec), "path": str(spec.path), "sha256": file_sha256(spec.path)}
        for spec in specs
    ]


def _inventory(records: pd.DataFrame) -> list[dict[str, object]]:
    counts = (
        records.groupby(["dataset_id", "target"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = records.groupby("dataset_id").size().to_dict()
    result = []
    for dataset_id, frame in counts.groupby("dataset_id", sort=True):
        by_target = {str(int(row.target)): int(row.rows) for row in frame.itertuples()}
        result.append({
            "dataset_id": str(dataset_id),
            "rows": int(totals[dataset_id]),
            "target_counts": by_target,
            "unique_parents": int(
                records.loc[records["dataset_id"].eq(dataset_id), "parent_id"].nunique()
            ),
        })
    return result


def _source_inventory_sha256(records: pd.DataFrame) -> str:
    digest = sha256()
    columns = ["dataset_id", "parent_id", "image_path", "target"]
    ordered = records.sort_values(["dataset_id", "parent_id"])[columns]
    for row in ordered.itertuples(index=False, name=None):
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _assignment_artifacts(assignments: pd.DataFrame, output: Path) -> dict[str, int]:
    public = public_uniform_chain_assignments(assignments)
    steps = uniform_chain_assignment_steps(assignments)
    public.to_csv(output / "transform_assignments.csv", index=False)
    steps.to_csv(output / "transform_assignment_steps.csv", index=False)

    base = ["dataset_id", "target", "sampling_stratum"]
    length_audit = (
        public.groupby([*base, "transform_step_count"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    step_audit = (
        steps.groupby(
            [
                *base,
                "transform_step_count",
                "step_position",
                "transform_family",
                "transform_variant",
                "severity",
            ],
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    repeated_audit = (
        public.groupby(
            [
                *base,
                "transform_step_count",
                "repeated_transform_family_count",
            ],
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    length_audit.to_csv(output / "assignment_audit_chain_length.csv", index=False)
    step_audit.to_csv(output / "assignment_audit_steps.csv", index=False)
    repeated_audit.to_csv(output / "assignment_audit_repeated_families.csv", index=False)
    return {
        "assignment_rows": int(len(public)),
        "assignment_step_rows": int(len(steps)),
        "assignments_with_repeated_families": int(
            public["repeated_transform_family_count"].gt(0).sum()
        ),
    }


def _representative_preprocessing_checks(
    named_datasets: Iterable[tuple[str, object]],
) -> pd.DataFrame:
    rows = []
    for dataset_id, dataset in named_datasets:
        indices = sorted({0, len(dataset) - 1})
        for index in indices:
            item = dataset[index]
            pixels = np.asarray(item["pixel_values"])
            if pixels.shape != (3, 224, 224) or pixels.dtype != np.float32:
                raise RuntimeError(
                    f"Preprocessing contract failed for {dataset_id} index {index}: "
                    f"shape={pixels.shape}, dtype={pixels.dtype}."
                )
            if not np.isfinite(pixels).all():
                raise RuntimeError(f"Preprocessing produced non-finite pixels for {dataset_id}.")
            rows.append({
                "dataset_id": dataset_id,
                "endpoint_index": index,
                "parent_id": item["parent_id"],
                "endpoint_policy": item["endpoint_policy"],
                "transform_step_count": int(item["transform_step_count"]),
                "shape": "3x224x224",
                "dtype": str(pixels.dtype),
                "minimum": float(pixels.min()),
                "maximum": float(pixels.max()),
                "finite": True,
            })
    return pd.DataFrame(rows)


def _score_distribution_tables(predictions: pd.DataFrame) -> pd.DataFrame:
    tables = []

    def append(name: str, frame: pd.DataFrame, groups: list[str]) -> None:
        if frame.empty:
            return
        table = score_distribution_slices(frame, groups)
        table.insert(0, "slice_type", name)
        tables.append(table)

    append("dataset", predictions, ["dataset_id"])
    append("endpoint_class", predictions, ["dataset_id", "endpoint_class"])
    append("chain_length", predictions, ["dataset_id", "transform_step_count"])
    as_is = predictions[predictions["endpoint_policy"].eq("as_is")]
    append("supplied_transform_family", as_is, ["dataset_id", "transform_family"])
    append("supplied_transform_variant", as_is, ["dataset_id", "transform_variant"])
    append(
        "generator_family",
        predictions,
        ["dataset_id", "generator_family"],
    )
    append(
        "generator_family_by_chain_length",
        predictions,
        ["dataset_id", "transform_step_count", "generator_family"],
    )
    append(
        "generation_model",
        predictions,
        ["dataset_id", "generation_model"],
    )
    append(
        "generation_model_by_chain_length",
        predictions,
        ["dataset_id", "transform_step_count", "generation_model"],
    )
    append(
        "authentic_source",
        predictions[predictions["target"].eq(0)],
        ["dataset_id", "authentic_subtype"],
    )
    append(
        "authentic_source_by_chain_length",
        predictions[predictions["target"].eq(0)],
        ["dataset_id", "transform_step_count", "authentic_subtype"],
    )
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _write_suite_metrics(
    predictions: pd.DataFrame,
    output: Path,
    *,
    threshold: float,
    max_fpr: float,
) -> dict[str, object]:
    kwargs = {"threshold": threshold, "max_fpr": max_fpr}
    dataset_metrics = metric_slices(predictions, ["dataset_id"], **kwargs)
    endpoint_metrics = metric_slices(
        predictions, ["dataset_id", "endpoint_class"], **kwargs
    )
    chain_metrics = metric_slices(
        predictions, ["dataset_id", "transform_step_count"], **kwargs
    )
    as_is = predictions[predictions["endpoint_policy"].eq("as_is")]
    transform_family_metrics = metric_slices(
        as_is, ["dataset_id", "transform_family"], **kwargs
    )
    transform_variant_metrics = metric_slices(
        as_is, ["dataset_id", "transform_family", "transform_variant"], **kwargs
    )
    generator_family_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group=None,
        output_group="comparison_generator_family",
        base_groups=("dataset_id",),
        **kwargs,
    )
    generator_family_chain_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group=None,
        output_group="comparison_generator_family",
        base_groups=("dataset_id", "transform_step_count"),
        **kwargs,
    )
    generation_model_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generation_model",
        negative_group=None,
        output_group="comparison_generation_model",
        base_groups=("dataset_id",),
        **kwargs,
    )
    generation_model_chain_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generation_model",
        negative_group=None,
        output_group="comparison_generation_model",
        base_groups=("dataset_id", "transform_step_count"),
        **kwargs,
    )
    authentic_source_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group="authentic_subtype",
        output_group="comparison_authentic_subtype",
        base_groups=("dataset_id",),
        **kwargs,
    )
    authentic_source_chain_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group="authentic_subtype",
        output_group="comparison_authentic_subtype",
        base_groups=("dataset_id", "transform_step_count"),
        **kwargs,
    )
    positive_generator_family = metric_slices(
        predictions[predictions["target"].eq(1)],
        ["dataset_id", "generator_family"],
        **kwargs,
    )
    positive_generator_family_chain = metric_slices(
        predictions[predictions["target"].eq(1)],
        ["dataset_id", "transform_step_count", "generator_family"],
        **kwargs,
    )
    positive_generation_model = metric_slices(
        predictions[predictions["target"].eq(1)],
        ["dataset_id", "generation_model"],
        **kwargs,
    )
    positive_generation_model_chain = metric_slices(
        predictions[predictions["target"].eq(1)],
        ["dataset_id", "transform_step_count", "generation_model"],
        **kwargs,
    )
    score_distributions = _score_distribution_tables(predictions)
    lineage_drift = paired_lineage_drift(as_is, threshold=threshold)

    predicted_target = predictions["logit"].ge(threshold).astype(int)
    errors = predictions.assign(predicted_target=predicted_target)
    false_positives = errors[errors["target"].eq(0) & errors["predicted_target"].eq(1)]
    false_negatives = errors[errors["target"].eq(1) & errors["predicted_target"].eq(0)]

    dataset_metrics.to_csv(output / "metrics_by_dataset.csv", index=False)
    endpoint_metrics.to_csv(output / "metrics_by_endpoint_class.csv", index=False)
    chain_metrics.to_csv(output / "metrics_by_chain_length.csv", index=False)
    transform_family_metrics.to_csv(output / "metrics_by_supplied_transform_family.csv", index=False)
    transform_variant_metrics.to_csv(output / "metrics_by_supplied_transform_variant.csv", index=False)
    generator_family_metrics.to_csv(output / "metrics_by_generator_family.csv", index=False)
    generator_family_chain_metrics.to_csv(
        output / "metrics_by_generator_family_and_chain_length.csv", index=False
    )
    generation_model_metrics.to_csv(output / "metrics_by_generation_model.csv", index=False)
    generation_model_chain_metrics.to_csv(
        output / "metrics_by_generation_model_and_chain_length.csv", index=False
    )
    authentic_source_metrics.to_csv(output / "metrics_by_authentic_source.csv", index=False)
    authentic_source_chain_metrics.to_csv(
        output / "metrics_by_authentic_source_and_chain_length.csv", index=False
    )
    positive_generator_family.to_csv(
        output / "positive_metrics_by_generator_family.csv", index=False
    )
    positive_generator_family_chain.to_csv(
        output / "positive_metrics_by_generator_family_and_chain_length.csv",
        index=False,
    )
    positive_generation_model.to_csv(
        output / "positive_metrics_by_generation_model.csv", index=False
    )
    positive_generation_model_chain.to_csv(
        output / "positive_metrics_by_generation_model_and_chain_length.csv",
        index=False,
    )
    score_distributions.to_csv(output / "score_distributions.csv", index=False)
    lineage_drift.to_csv(output / "techjam_lineage_paired_drift.csv", index=False)
    false_positives.to_csv(output / "false_positives.csv", index=False)
    false_negatives.to_csv(output / "false_negatives.csv", index=False)
    return {
        "datasets": dataset_metrics.to_dict("records"),
        "false_positive_rows": int(len(false_positives)),
        "false_negative_rows": int(len(false_negatives)),
    }


def _markdown_value(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return "N/A"
        return f"{number:.4f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    available = [(column, label) for column, label in columns if column in frame]
    if frame.empty or not available:
        return "_No rows available._"
    header = "| " + " | ".join(label for _, label in available) + " |"
    divider = "| " + " | ".join("---" for _ in available) + " |"
    rows = [header, divider]
    for row in frame.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(_markdown_value(row.get(column)) for column, _ in available)
            + " |"
        )
    return "\n".join(rows)


def _write_markdown_report(
    output: Path,
    *,
    summary: dict[str, object],
    metadata: dict[str, object],
) -> None:
    """Write a compact human-readable companion to the complete CSV artifacts."""

    dataset_metrics = pd.read_csv(output / "metrics_by_dataset.csv")
    chain_metrics = pd.read_csv(output / "metrics_by_chain_length.csv")
    transform_metrics = pd.read_csv(
        output / "metrics_by_supplied_transform_family.csv"
    )
    generator_family_metrics = pd.read_csv(output / "metrics_by_generator_family.csv")
    lineage_drift = pd.read_csv(output / "techjam_lineage_paired_drift.csv")
    inventory_rows = []
    sampled_by_id = {
        str(row["dataset_id"]): row for row in metadata["sampled_inventory"]
    }
    for full in metadata["full_inventory"]:
        sampled = sampled_by_id[str(full["dataset_id"])]
        inventory_rows.append({
            "dataset_id": full["dataset_id"],
            "full_rows": full["rows"],
            "sampled_rows": sampled["rows"],
            "sampled_positive": sampled["target_counts"].get("1", 0),
            "sampled_negative": sampled["target_counts"].get("0", 0),
        })
    inventory = pd.DataFrame(inventory_rows)
    uniform_dataset_ids = set(
        metadata["endpoint_policies"]["uniform_sequential_chain"]
    )
    external_chain_metrics = chain_metrics[
        chain_metrics["dataset_id"].isin(uniform_dataset_ids)
    ]

    overall_columns = [
        ("dataset_id", "Dataset"),
        ("rows", "N"),
        ("positives", "Pos"),
        ("negatives", "Neg"),
        ("positive_prevalence", "AIGC prevalence"),
        ("true_positive", "TP"),
        ("true_negative", "TN"),
        ("false_positive", "FP"),
        ("false_negative", "FN"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced acc."),
        ("roc_auc", "AUROC"),
        ("average_precision", "AUPRC/AP"),
        ("normalized_pauc", "pAUROC@5%"),
        ("precision", "Precision"),
        ("recall", "Recall/TPR"),
        ("specificity", "Specificity/TNR"),
        ("f1", "F1"),
        ("predicted_positive_rate", "Predicted-positive rate"),
        ("false_positive_rate", "FPR"),
        ("false_negative_rate", "FNR"),
        ("matthews_correlation_coefficient", "MCC"),
        ("status", "Status"),
    ]
    slice_columns = [
        ("dataset_id", "Dataset"),
        ("transform_step_count", "Chain length"),
        ("transform_family", "Transform family"),
        ("rows", "N"),
        ("roc_auc", "AUROC"),
        ("average_precision", "AUPRC/AP"),
        ("accuracy", "Accuracy"),
        ("recall", "Recall/TPR"),
        ("false_positive_rate", "FPR"),
        ("false_negative_rate", "FNR"),
        ("status", "Status"),
    ]
    report = f"""# TRACE-RX-M v2 evaluation results

Generated at `{metadata['created_at']}` from checkpoint epoch `{metadata.get('checkpoint_epoch')}`.

- Evaluated endpoints: **{int(metadata['source_images']):,}**
- Sampling seed: `{metadata['seed']}`
- Decision threshold: `{metadata['threshold']}` on the `{metadata['threshold_scale']}` scale
- Inference time: {_markdown_value(metadata.get('inference_seconds'))} seconds
- Throughput: {_markdown_value(metadata.get('inference_throughput_endpoints_per_second'))} endpoints/second
- False positives: **{int(summary['false_positive_rows']):,}**
- False negatives: **{int(summary['false_negative_rows']):,}**

## Inventory

{_markdown_table(inventory, [('dataset_id', 'Dataset'), ('full_rows', 'Full inventory'), ('sampled_rows', 'Evaluated'), ('sampled_positive', 'AIGC'), ('sampled_negative', 'Authentic')])}

Each external image has exactly one deterministic sequential transform chain and no clean duplicate. EvalGEN is positive-only, so its two-class metrics are reported as `N/A` rather than inferred from a nonexistent negative class.

## Overall metrics

{_markdown_table(dataset_metrics, overall_columns)}

## External performance by chain length

{_markdown_table(external_chain_metrics, slice_columns)}

## TechJam supplied transform families

{_markdown_table(transform_metrics, slice_columns)}

## Generator-family slices

{_markdown_table(generator_family_metrics, [('dataset_id', 'Dataset'), ('comparison_generator_family', 'Generator family'), ('rows', 'N'), ('roc_auc', 'AUROC'), ('average_precision', 'AUPRC/AP'), ('accuracy', 'Accuracy'), ('recall', 'Recall/TPR'), ('false_positive_rate', 'FPR'), ('false_negative_rate', 'FNR'), ('status', 'Status')])}

## TechJam lineage-paired drift

{_markdown_table(lineage_drift, [('condition', 'Condition'), ('transform_family', 'Family'), ('pairs', 'Pairs'), ('mean_score_drift', 'Mean drift'), ('mean_absolute_score_drift', 'Mean abs. drift'), ('prediction_flip_rate', 'Flip rate'), ('correct_to_incorrect_rate', 'Correct -> incorrect'), ('incorrect_to_correct_rate', 'Incorrect -> correct')])}

## Complete artifacts

- [`predictions.csv`](predictions.csv) and [`predictions.json`](predictions.json)
- [`metrics_by_dataset.csv`](metrics_by_dataset.csv), [`metrics_by_chain_length.csv`](metrics_by_chain_length.csv), and all generator/source/transform slice tables
- [`score_distributions.csv`](score_distributions.csv) with per-class logit and probability mean, standard deviation, median, p05, and p95
- [`transform_assignments.csv`](transform_assignments.csv), expanded steps, and assignment audit tables
- [`false_positives.csv`](false_positives.csv) and [`false_negatives.csv`](false_negatives.csv)
- [`run_metadata.json`](run_metadata.json) and [`summary.json`](summary.json)
"""
    (output / "evaluation_report.md").write_text(report, encoding="utf-8")


def _load_suite(args, repo_root: Path, config: TraceRXMConfig, output: Path):
    as_is_paths = args.as_is_dataset_spec or []
    uniform_paths = args.uniform_chain_dataset_spec or []
    if not as_is_paths and not uniform_paths:
        raise ValueError(
            "Suite mode needs --as-is-dataset-spec and/or --uniform-chain-dataset-spec."
        )
    as_is_records = pd.DataFrame()
    uniform_records = pd.DataFrame()
    full_as_is_records = pd.DataFrame()
    full_uniform_records = pd.DataFrame()
    as_is_specs = []
    uniform_specs = []
    if as_is_paths:
        full_as_is_records, as_is_specs = load_evaluation_datasets(
            as_is_paths, repo_root=repo_root, verify_paths=True
        )
        as_is_records = _limit_records(
            full_as_is_records, args.max_images_per_dataset, args.seed
        )
    if uniform_paths:
        full_uniform_records, uniform_specs = load_evaluation_datasets(
            uniform_paths, repo_root=repo_root, verify_paths=True
        )
        uniform_records = _limit_records(
            full_uniform_records, args.max_images_per_dataset, args.seed
        )
    dataset_ids = [spec.dataset_id for spec in [*as_is_specs, *uniform_specs]]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Dataset IDs must be unique across endpoint policies.")

    named_datasets: list[tuple[str, object]] = []
    endpoint_datasets: list[object] = []
    if not as_is_records.empty:
        for dataset_id, records in as_is_records.groupby("dataset_id", sort=True):
            dataset = AsIsEndpointDataset(records, preprocessing=config.preprocessing)
            named_datasets.append((str(dataset_id), dataset))
            endpoint_datasets.append(dataset)

    assignments = pd.DataFrame()
    if not uniform_records.empty:
        assignments = build_uniform_chain_assignments(
            uniform_records,
            base_seed=args.seed,
            chain_lengths=args.chain_length or range(1, 7),
        )
        for dataset_id, assigned in assignments.groupby("dataset_id", sort=True):
            dataset = UniformSequentialChainEndpointDataset(
                assigned,
                preprocessing=config.preprocessing,
                base_seed=args.seed,
            )
            named_datasets.append((str(dataset_id), dataset))
            endpoint_datasets.append(dataset)
        _assignment_artifacts(assignments, output)
    else:
        pd.DataFrame().to_csv(output / "transform_assignments.csv", index=False)

    records = pd.concat(
        [frame for frame in (as_is_records, uniform_records) if not frame.empty],
        ignore_index=True,
    )
    full_records = pd.concat(
        [
            frame
            for frame in (full_as_is_records, full_uniform_records)
            if not frame.empty
        ],
        ignore_index=True,
    )
    return (
        records,
        full_records,
        [*as_is_specs, *uniform_specs],
        named_datasets,
        endpoint_datasets,
        assignments,
    )


def _run_suite(args, config: TraceRXMConfig, repo_root: Path, output: Path) -> None:
    started = time.monotonic()
    (
        records,
        full_records,
        specs,
        named_datasets,
        endpoint_datasets,
        assignments,
    ) = _load_suite(
        args, repo_root, config, output
    )
    preprocessing_checks = _representative_preprocessing_checks(named_datasets)
    preprocessing_checks.to_csv(output / "preprocessing_validation.csv", index=False)
    inventory = _inventory(records)
    full_inventory = _inventory(full_records)
    assignment_summary = {
        "assignment_rows": int(len(assignments)),
        "assignment_step_rows": int(assignments["transform_step_count"].sum())
        if not assignments.empty
        else 0,
        "assignments_with_repeated_families": int(
            assignments["repeated_transform_family_count"].gt(0).sum()
        )
        if not assignments.empty
        else 0,
    }

    common_metadata: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "mode": "v2_benchmark_suite",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "dataset_specs": _spec_metadata(specs),
        "inventory": inventory,
        "sampled_inventory": inventory,
        "full_inventory": full_inventory,
        "source_inventory_sha256": _source_inventory_sha256(records),
        "full_source_inventory_sha256": _source_inventory_sha256(full_records),
        "preprocessing": asdict(config.preprocessing),
        "endpoint_policies": {
            "as_is": [spec.dataset_id for spec in as_is_specs_from(specs, args)],
            "uniform_sequential_chain": [
                spec.dataset_id for spec in uniform_specs_from(specs, args)
            ],
        },
        "uniform_chain_lengths": args.chain_length or list(range(1, 7)),
        "condition_assignment_policy": (
            "counterbalanced_within_dataset_target_and_generator_or_authentic_source_strata"
        ),
        "source_images": int(len(records)),
        "transformed_endpoints": int(sum(len(dataset) for dataset in endpoint_datasets)),
        "seed": args.seed,
        "threshold": args.threshold,
        "threshold_scale": "logit",
        "max_fpr": args.max_fpr,
        "always_report_metrics": ALWAYS_REPORT_METRICS,
        "robustness_metrics": ROBUSTNESS_METRICS,
        "threshold_report_metrics": REPORT_METRICS,
        "sample_size_per_dataset": args.max_images_per_dataset,
        **assignment_summary,
    }
    if not assignments.empty:
        common_metadata["transform_assignments_sha256"] = file_sha256(
            output / "transform_assignments.csv"
        )
        common_metadata["transform_assignment_steps_sha256"] = file_sha256(
            output / "transform_assignment_steps.csv"
        )
    if args.validate_only:
        common_metadata.update({
            "validation_only": True,
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_json(output / "validation_summary.json", common_metadata)
        _write_json(output / "run_metadata.json", common_metadata)
        print(
            f"Validated {len(records)} sources and {len(assignments)} external assignments.",
            flush=True,
        )
        return

    if args.checkpoint is None or args.memory is None:
        raise ValueError("Full inference requires both --checkpoint and --memory.")
    import torch

    device = torch.device(args.device)
    _preflight_v2_checkpoint(args.checkpoint, config)
    model, checkpoint = load_detector_checkpoint(args.checkpoint, args.memory, device=device)
    _validate_v2_checkpoint(config, checkpoint)
    endpoint_dataset = torch.utils.data.ConcatDataset(endpoint_datasets)
    loader = torch.utils.data.DataLoader(
        endpoint_dataset,
        batch_size=args.batch_size or config.data.batch_size,
        shuffle=False,
        num_workers=config.data.workers if args.workers is None else args.workers,
        pin_memory=device.type == "cuda",
    )
    inference_started = time.monotonic()
    predictions = _run_inference(model, loader, device)
    inference_seconds = time.monotonic() - inference_started
    if len(predictions) != len(endpoint_dataset):
        raise RuntimeError("Inference did not produce one prediction per endpoint.")
    if predictions.duplicated(["dataset_id", "parent_id"]).any():
        raise RuntimeError("Benchmark suite produced more than one endpoint per source.")
    if set(predictions["dataset_id"]) != set(records["dataset_id"]):
        raise RuntimeError("Inference output omitted a configured dataset.")

    predictions.to_csv(output / "predictions.csv", index=False)
    _write_json(output / "predictions.json", _prediction_json_rows(predictions))
    summary = _write_suite_metrics(
        predictions,
        output,
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    summary.update({
        "inventory": inventory,
        "assignment_summary": assignment_summary,
        "threshold": args.threshold,
        "threshold_scale": "logit",
    })
    _write_json(output / "summary.json", summary)
    elapsed = time.monotonic() - started
    common_metadata.update({
        "validation_only": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "memory": str(args.memory.resolve()),
        "memory_sha256": file_sha256(args.memory),
        "elapsed_seconds": elapsed,
        "inference_seconds": inference_seconds,
        "inference_throughput_endpoints_per_second": (
            len(predictions) / inference_seconds if inference_seconds else None
        ),
    })
    _write_json(output / "run_metadata.json", common_metadata)
    _write_markdown_report(output, summary=summary, metadata=common_metadata)


def as_is_specs_from(specs, args):
    paths = {path.resolve() for path in (args.as_is_dataset_spec or [])}
    return [spec for spec in specs if spec.path.resolve() in paths]


def uniform_specs_from(specs, args):
    paths = {path.resolve() for path in (args.uniform_chain_dataset_spec or [])}
    return [spec for spec in specs if spec.path.resolve() in paths]


def _run_legacy(args, config: TraceRXMConfig, repo_root: Path, output: Path) -> None:
    if not args.dataset_spec:
        raise ValueError("Legacy mode requires --dataset-spec.")
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
            preprocessing=config.preprocessing,
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
            preprocessing=config.preprocessing,
        )
        assignment_policy = "exhaustive_selected_condition_matrix"
    if args.validate_only:
        _representative_preprocessing_checks([("legacy", endpoint_dataset)]).to_csv(
            output / "preprocessing_validation.csv", index=False
        )
        _write_json(output / "run_metadata.json", {
            "validation_only": True,
            "mode": "legacy",
            "inventory": _inventory(records),
            "preprocessing": asdict(config.preprocessing),
        })
        return
    if args.checkpoint is None or args.memory is None:
        raise ValueError("Full inference requires both --checkpoint and --memory.")

    import torch

    device = torch.device(args.device)
    _preflight_v2_checkpoint(args.checkpoint, config)
    model, checkpoint = load_detector_checkpoint(args.checkpoint, args.memory, device=device)
    _validate_v2_checkpoint(config, checkpoint)
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

    kwargs = {"threshold": args.threshold, "max_fpr": args.max_fpr}
    condition_metrics = metric_slices(
        predictions, ["dataset_id", "condition", "transform_family"], **kwargs
    )
    dataset_metrics = metric_slices(predictions, ["dataset_id"], **kwargs)
    family_metrics = metric_slices(
        predictions, ["dataset_id", "transform_family"], **kwargs
    )
    chain_length_metrics = metric_slices(
        predictions, ["dataset_id", "transform_step_count"], **kwargs
    )
    generator_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group=None,
        output_group="comparison_generator_family",
        **kwargs,
    )
    generation_model_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generation_model",
        negative_group=None,
        output_group="comparison_generation_model",
        **kwargs,
    )
    authentic_source_metrics = comparison_slice_metrics(
        predictions,
        positive_group="generator_family",
        negative_group="authentic_subtype",
        output_group="comparison_authentic_subtype",
        **kwargs,
    )
    drops = clean_to_condition_drops(condition_metrics)
    drift = paired_endpoint_drift(predictions, threshold=args.threshold)
    summary = robustness_summary(condition_metrics, registry)

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
        "mode": "legacy",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "memory": str(args.memory.resolve()),
        "memory_sha256": file_sha256(args.memory),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "dataset_specs": _spec_metadata(dataset_specs),
        "source_images": int(len(records)),
        "transformed_endpoints": int(len(endpoint_dataset)),
        "transform_profiles": args.transform_profile or ["core"],
        "conditions": [spec.name for spec in transforms],
        "assigned_chain_lengths": args.assigned_chain_length,
        "condition_assignment_policy": assignment_policy,
        "official_only": bool(args.official_only),
        "preprocessing": asdict(config.preprocessing),
        "seed": args.seed,
        "threshold": args.threshold,
        "threshold_scale": "logit",
        "max_fpr": args.max_fpr,
        "always_report_metrics": ALWAYS_REPORT_METRICS,
        "robustness_metrics": ROBUSTNESS_METRICS,
        "smoke_test_limit_per_dataset": args.max_images_per_dataset,
    })


def main() -> None:
    args = parse_args()
    if not 0 < args.max_fpr <= 1:
        raise ValueError("--max-fpr must lie in (0, 1].")
    suite_mode = bool(args.as_is_dataset_spec or args.uniform_chain_dataset_spec)
    if suite_mode and args.dataset_spec:
        raise ValueError("Do not mix benchmark-suite and legacy dataset arguments.")
    if args.chain_length and not args.uniform_chain_dataset_spec:
        raise ValueError("--chain-length requires --uniform-chain-dataset-spec.")
    config = _strict_v2_config(args.config)
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if suite_mode:
        _run_suite(args, config, repo_root, output)
    else:
        _run_legacy(args, config, repo_root, output)


if __name__ == "__main__":
    main()
