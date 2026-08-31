#!/usr/bin/env python3
"""Summarize external assigned-chain detector predictions by model and chain length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from techjam_aigc.trace_rx_m.evaluation import robustness_detection_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-fpr", type=float, default=0.05)
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


def _comparison_metrics(
    predictions: pd.DataFrame,
    *,
    positive_group: str,
    negative_group: str | None = None,
    threshold: float,
    max_fpr: float,
) -> pd.DataFrame:
    rows = []
    for (dataset_id, chain_length), endpoint in predictions.groupby(
        ["dataset_id", "transform_step_count"], sort=True
    ):
        positives = endpoint[endpoint["target"].eq(1)]
        negatives = endpoint[endpoint["target"].eq(0)]
        if negative_group is None:
            comparisons = (
                (str(name), pd.concat((negatives, group), ignore_index=True), group, negatives)
                for name, group in positives.groupby(positive_group, sort=True)
            )
            group_name = positive_group
        else:
            comparisons = (
                (str(name), pd.concat((group, positives), ignore_index=True), positives, group)
                for name, group in negatives.groupby(negative_group, sort=True)
            )
            group_name = negative_group
        for name, frame, positive_rows, negative_rows in comparisons:
            metrics = robustness_detection_metrics(
                frame["target"],
                frame["logit"],
                threshold=threshold,
                max_fpr=max_fpr,
            )
            rows.append({
                "dataset_id": dataset_id,
                "transform_step_count": int(chain_length),
                group_name: name,
                **metrics,
                "positive_recall": float(positive_rows["logit"].ge(threshold).mean()),
                "negative_recall": float(negative_rows["logit"].lt(threshold).mean()),
                "positive_mean_logit": float(positive_rows["logit"].mean()),
                "negative_mean_logit": float(negative_rows["logit"].mean()),
            })
    return pd.DataFrame(rows)


def _paired_chain_drift(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    clean = predictions[predictions["transform_step_count"].eq(0)][
        ["dataset_id", "parent_id", "target", "logit"]
    ].rename(columns={"logit": "clean_logit"})
    transformed = predictions[predictions["transform_step_count"].gt(0)].merge(
        clean,
        on=["dataset_id", "parent_id", "target"],
        how="left",
        validate="many_to_one",
    )
    transformed["score_drift"] = transformed["logit"] - transformed["clean_logit"]
    transformed["clean_positive"] = transformed["clean_logit"].ge(threshold)
    transformed["condition_positive"] = transformed["logit"].ge(threshold)
    transformed["clean_correct"] = transformed["clean_positive"].astype(int).eq(
        transformed["target"]
    )
    transformed["condition_correct"] = transformed["condition_positive"].astype(int).eq(
        transformed["target"]
    )
    rows = []
    for (dataset_id, chain_length), frame in transformed.groupby(
        ["dataset_id", "transform_step_count"], sort=True
    ):
        rows.append({
            "dataset_id": dataset_id,
            "transform_step_count": int(chain_length),
            "parents": int(len(frame)),
            "mean_score_drift": float(frame["score_drift"].mean()),
            "mean_absolute_score_drift": float(frame["score_drift"].abs().mean()),
            "clean_condition_pearson": float(
                frame[["clean_logit", "logit"]].corr().iloc[0, 1]
            ),
            "prediction_flip_rate": float(
                frame["clean_positive"].ne(frame["condition_positive"]).mean()
            ),
            "correct_to_incorrect_rate": float(
                (frame["clean_correct"] & ~frame["condition_correct"]).mean()
            ),
            "incorrect_to_correct_rate": float(
                (~frame["clean_correct"] & frame["condition_correct"]).mean()
            ),
        })
    return pd.DataFrame(rows)


def _dataset_chain_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    max_fpr: float,
) -> pd.DataFrame:
    rows = []
    for (dataset_id, chain_length), frame in predictions.groupby(
        ["dataset_id", "transform_step_count"], sort=True
    ):
        metrics = robustness_detection_metrics(
            frame["target"],
            frame["logit"],
            threshold=threshold,
            max_fpr=max_fpr,
        )
        positives = frame[frame["target"].eq(1)]
        negatives = frame[frame["target"].eq(0)]
        rows.append({
            "dataset_id": dataset_id,
            "transform_step_count": int(chain_length),
            **metrics,
            "positive_recall": float(positives["logit"].ge(threshold).mean()),
            "negative_recall": float(negatives["logit"].lt(threshold).mean()),
            "positive_mean_logit": float(positives["logit"].mean()),
            "negative_mean_logit": float(negatives["logit"].mean()),
        })
    return pd.DataFrame(rows)


def _compact_summary(
    chain_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    drift: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dataset_id, dataset in chain_metrics.groupby("dataset_id", sort=True):
        by_length = dataset.set_index("transform_step_count")
        model_clean = model_metrics[
            model_metrics["dataset_id"].eq(dataset_id)
            & model_metrics["transform_step_count"].eq(0)
        ].sort_values("roc_auc")
        model_triple = model_metrics[
            model_metrics["dataset_id"].eq(dataset_id)
            & model_metrics["transform_step_count"].eq(3)
        ][["generation_model", "roc_auc", "positive_recall"]].rename(columns={
            "roc_auc": "triple_roc_auc",
            "positive_recall": "triple_positive_recall",
        })
        model_changes = model_clean.merge(model_triple, on="generation_model")
        model_changes["triple_roc_auc_change"] = (
            model_changes["triple_roc_auc"] - model_changes["roc_auc"]
        )
        triple_drift = drift[
            drift["dataset_id"].eq(dataset_id)
            & drift["transform_step_count"].eq(3)
        ].iloc[0]
        result[str(dataset_id)] = {
            "by_chain_length": {
                str(int(length)): {
                    key: float(row[key])
                    for key in (
                        "roc_auc", "average_precision", "normalized_pauc",
                        "balanced_accuracy", "positive_recall", "negative_recall",
                    )
                }
                for length, row in by_length.iterrows()
            },
            "clean_models_below_chance_roc_auc": int(model_clean["roc_auc"].lt(0.5).sum()),
            "clean_model_count": int(len(model_clean)),
            "weakest_clean_models": model_clean.head(5)[
                ["generation_model", "roc_auc", "positive_recall"]
            ].to_dict("records"),
            "strongest_clean_models": model_clean.tail(5).sort_values("roc_auc", ascending=False)[
                ["generation_model", "roc_auc", "positive_recall"]
            ].to_dict("records"),
            "largest_triple_roc_auc_drops": model_changes.sort_values(
                "triple_roc_auc_change"
            ).head(5)[
                ["generation_model", "roc_auc", "triple_roc_auc", "triple_roc_auc_change"]
            ].to_dict("records"),
            "triple_chain_paired_drift": {
                key: float(triple_drift[key])
                for key in (
                    "mean_score_drift", "mean_absolute_score_drift",
                    "clean_condition_pearson", "prediction_flip_rate",
                    "correct_to_incorrect_rate", "incorrect_to_correct_rate",
                )
            },
        }
    return result


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    required = {
        "dataset_id", "parent_id", "target", "logit", "transform_step_count",
        "generation_model", "authentic_subtype",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if not np.isfinite(predictions["logit"]).all():
        raise ValueError("Predictions contain non-finite logits.")

    chain_metrics = _dataset_chain_metrics(
        predictions,
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    model_metrics = _comparison_metrics(
        predictions,
        positive_group="generation_model",
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    source_metrics = _comparison_metrics(
        predictions,
        positive_group="generation_model",
        negative_group="authentic_subtype",
        threshold=args.threshold,
        max_fpr=args.max_fpr,
    )
    drift = _paired_chain_drift(predictions, threshold=args.threshold)
    summary = _compact_summary(chain_metrics, model_metrics, drift)

    args.output.mkdir(parents=True, exist_ok=True)
    chain_metrics.to_csv(args.output / "external_metrics_by_chain_length.csv", index=False)
    model_metrics.to_csv(
        args.output / "external_metrics_by_chain_length_and_generation_model.csv",
        index=False,
    )
    source_metrics.to_csv(
        args.output / "external_metrics_by_chain_length_and_authentic_source.csv",
        index=False,
    )
    drift.to_csv(args.output / "external_paired_drift_by_chain_length.csv", index=False)
    (args.output / "external_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
