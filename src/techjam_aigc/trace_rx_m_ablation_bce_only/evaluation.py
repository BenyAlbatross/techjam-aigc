"""Canonical metrics and slices for frozen TRACE-RX-M evaluations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ALWAYS_REPORT_METRICS = (
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
)
ROBUSTNESS_METRICS = (*ALWAYS_REPORT_METRICS, "normalized_pauc")


def binary_detection_metrics(
    labels: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate ranking and thresholded metrics with an explicit threshold.

    ``scores`` may be logits or probabilities as long as ``threshold`` uses the
    same scale. AIGC is the positive class (1) and authentic is class 0.
    """

    targets = np.asarray(labels).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.size == 0 or targets.size != values.size:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("evaluation requires both binary classes (0 authentic, 1 AIGC)")

    predictions = values >= threshold
    return {
        "rows": int(targets.size),
        "positives": int(np.sum(targets == 1)),
        "negatives": int(np.sum(targets == 0)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(targets, values)),
        "average_precision": float(average_precision_score(targets, values)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
    }


def normalized_partial_auc(
    labels: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> float:
    """Return partial ROC area on ``[0, max_fpr]`` normalized to ``[0, 1]``."""

    from sklearn.metrics import roc_curve

    targets = np.asarray(labels).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.size == 0 or targets.size != values.size:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("partial AUC requires both binary classes")
    if not np.isfinite(values).all() or not 0 < max_fpr <= 1:
        raise ValueError("scores must be finite and max_fpr must lie in (0, 1]")
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, values)
    boundary = float(np.interp(max_fpr, false_positive_rate, true_positive_rate))
    keep = false_positive_rate < max_fpr
    clipped_fpr = np.concatenate((false_positive_rate[keep], [max_fpr]))
    clipped_tpr = np.concatenate((true_positive_rate[keep], [boundary]))
    return float(np.trapezoid(clipped_tpr, clipped_fpr) / max_fpr)


def robustness_detection_metrics(
    labels: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    threshold: float,
    max_fpr: float = 0.05,
) -> dict[str, float | int]:
    """Always-report metrics plus the challenge-relevant low-FPR ranking metric."""

    metrics = binary_detection_metrics(labels, scores, threshold=threshold)
    metrics["normalized_pauc"] = normalized_partial_auc(
        labels,
        scores,
        max_fpr=max_fpr,
    )
    return metrics


def _unscorable_metrics(
    labels: np.ndarray,
    *,
    threshold: float,
    status: str,
) -> dict[str, float | int | str]:
    return {
        "rows": int(labels.size),
        "positives": int(np.sum(labels == 1)),
        "negatives": int(np.sum(labels == 0)),
        "threshold": float(threshold),
        **{metric: float("nan") for metric in ROBUSTNESS_METRICS},
        "status": status,
    }


def metric_slices(
    predictions: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    score_column: str = "logit",
    threshold: float = 0.0,
    max_fpr: float = 0.05,
) -> pd.DataFrame:
    """Compute schema-stable metrics without crashing on one-class slices."""

    groups = list(group_columns)
    required = {"target", score_column, *groups}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(groups, dropna=False, sort=True) if groups else [((), predictions)]
    for key, frame in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(groups, keys, strict=True))
        labels = frame["target"].to_numpy(dtype=int)
        scores = frame[score_column].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            metrics = _unscorable_metrics(
                labels,
                threshold=threshold,
                status="unscorable_one_class",
            )
        elif not np.isfinite(scores).all():
            metrics = _unscorable_metrics(
                labels,
                threshold=threshold,
                status="unscorable_non_finite_scores",
            )
        else:
            metrics = {
                **robustness_detection_metrics(
                    labels,
                    scores,
                    threshold=threshold,
                    max_fpr=max_fpr,
                ),
                "status": "ok",
            }
        metrics["pauc_max_fpr"] = float(max_fpr)
        rows.append({**group_values, **metrics})
    return pd.DataFrame(rows)


def comparison_slice_metrics(
    predictions: pd.DataFrame,
    *,
    positive_group: str,
    negative_group: str | None,
    output_group: str,
    score_column: str = "logit",
    threshold: float = 0.0,
    max_fpr: float = 0.05,
) -> pd.DataFrame:
    """Compare each positive or negative subgroup against the opposite class.

    Generator metrics use ``positive_group`` and all authentic images. Authentic
    source metrics additionally provide ``negative_group`` and compare one real
    source at a time against all generated images.
    """

    required = {
        "dataset_id",
        "condition",
        "transform_family",
        "target",
        positive_group,
        score_column,
    }
    if negative_group is not None:
        required.add(negative_group)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    comparisons: list[pd.DataFrame] = []
    base_groups = ["dataset_id", "condition", "transform_family"]
    for _, endpoint in predictions.groupby(base_groups, sort=True):
        positives = endpoint[endpoint["target"].eq(1)]
        negatives = endpoint[endpoint["target"].eq(0)]
        if negative_group is None:
            for name, positive_rows in positives.groupby(positive_group, sort=True):
                comparisons.append(
                    pd.concat((negatives, positive_rows), ignore_index=True).assign(
                        **{output_group: str(name)}
                    )
                )
        else:
            for name, negative_rows in negatives.groupby(negative_group, sort=True):
                comparisons.append(
                    pd.concat((negative_rows, positives), ignore_index=True).assign(
                        **{output_group: str(name)}
                    )
                )
    if not comparisons:
        return pd.DataFrame(columns=[*base_groups, output_group, "status", *ROBUSTNESS_METRICS])
    return metric_slices(
        pd.concat(comparisons, ignore_index=True),
        [*base_groups, output_group],
        score_column=score_column,
        threshold=threshold,
        max_fpr=max_fpr,
    )


def clean_to_condition_drops(metrics: pd.DataFrame) -> pd.DataFrame:
    """Join each transformed metric row to its dataset's clean baseline."""

    required = {"dataset_id", "condition", *ROBUSTNESS_METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Condition metrics are missing columns: {sorted(missing)}")
    clean = metrics[metrics["condition"].eq("clean")]
    if clean["dataset_id"].duplicated().any():
        raise ValueError("Condition metrics need exactly one clean row per dataset.")
    baseline = clean.set_index("dataset_id")
    rows = []
    for row in metrics[~metrics["condition"].eq("clean")].to_dict("records"):
        dataset_id = row["dataset_id"]
        if dataset_id not in baseline.index:
            raise ValueError(f"Dataset {dataset_id!r} has no clean baseline.")
        clean_row = baseline.loc[dataset_id]
        result = dict(row)
        for metric in ROBUSTNESS_METRICS:
            result[f"clean_{metric}"] = clean_row[metric]
            result[f"{metric}_drop"] = clean_row[metric] - row[metric]
        rows.append(result)
    return pd.DataFrame(rows)


def paired_endpoint_drift(
    predictions: pd.DataFrame,
    *,
    score_column: str = "logit",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Measure paired score and correctness changes from clean source images."""

    required = {
        "dataset_id",
        "parent_id",
        "condition",
        "transform_family",
        "target",
        score_column,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    key = ["dataset_id", "parent_id"]
    clean = predictions[predictions["condition"].eq("clean")][
        [*key, "target", score_column]
    ].rename(columns={score_column: "clean_score", "target": "clean_target"})
    if clean.duplicated(key).any():
        raise ValueError("Predictions need exactly one clean row per source image.")
    transformed = predictions[~predictions["condition"].eq("clean")]
    joined = transformed.merge(clean, on=key, how="left", validate="many_to_one")
    if joined["clean_score"].isna().any() or not joined["target"].eq(joined["clean_target"]).all():
        raise ValueError("Every transformed endpoint must match one clean endpoint and target.")
    joined["score_drift"] = joined[score_column] - joined["clean_score"]
    joined["clean_correct"] = (joined["clean_score"].ge(threshold).astype(int) == joined["target"])
    joined["condition_correct"] = (joined[score_column].ge(threshold).astype(int) == joined["target"])
    rows = []
    for (dataset_id, condition, family), frame in joined.groupby(
        ["dataset_id", "condition", "transform_family"], sort=True
    ):
        rows.append({
            "dataset_id": dataset_id,
            "condition": condition,
            "transform_family": family,
            "parents": int(len(frame)),
            "mean_score_drift": float(frame["score_drift"].mean()),
            "mean_absolute_score_drift": float(frame["score_drift"].abs().mean()),
            "prediction_flip_rate": float(
                frame["clean_score"].ge(threshold).ne(frame[score_column].ge(threshold)).mean()
            ),
            "correct_to_incorrect_rate": float(
                (frame["clean_correct"] & ~frame["condition_correct"]).mean()
            ),
            "incorrect_to_correct_rate": float(
                (~frame["clean_correct"] & frame["condition_correct"]).mean()
            ),
        })
    return pd.DataFrame(rows)


def robustness_summary(
    condition_metrics: pd.DataFrame,
    transform_registry: pd.DataFrame,
) -> dict[str, object]:
    """Return compact per-dataset clean, macro, and worst-condition results."""

    official = transform_registry.set_index("name")["official"].to_dict()
    reports: dict[str, object] = {}
    for dataset_id, frame in condition_metrics.groupby("dataset_id", sort=True):
        clean_rows = frame[frame["condition"].eq("clean")]
        transformed = frame[
            frame["condition"].map(official).fillna(False)
            & ~frame["condition"].eq("clean")
            & frame["status"].eq("ok")
        ]
        clean = clean_rows.iloc[0].to_dict() if len(clean_rows) else None
        if transformed.empty:
            worst = None
            macro = {metric: float("nan") for metric in ROBUSTNESS_METRICS}
        else:
            worst_row = transformed.loc[transformed["roc_auc"].idxmin()]
            worst = {
                "condition": worst_row["condition"],
                "transform_family": worst_row["transform_family"],
                **{metric: float(worst_row[metric]) for metric in ROBUSTNESS_METRICS},
            }
            macro = {
                metric: float(transformed[metric].mean()) for metric in ROBUSTNESS_METRICS
            }
        reports[str(dataset_id)] = {
            "clean": None if clean is None else {
                metric: float(clean[metric]) for metric in ROBUSTNESS_METRICS
            },
            "official_transformed_condition_count": int(len(transformed)),
            "macro_official_transformed": macro,
            "worst_official_transform_by_roc_auc": worst,
        }
    return reports
