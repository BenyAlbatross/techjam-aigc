#!/usr/bin/env python3
"""Summarize five validation-only TRACE-RX-M ablations with paired intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


RUNS = {
    "holdout-gemini": "techjam-aigc/trace-rx-m-v2-ablation-holdout-gemini",
    "holdout-flux": "techjam-aigc/trace-rx-m-v2-ablation-holdout-flux",
    "frozen-encoder": "techjam-aigc/trace-rx-m-v2-ablation-frozen-encoder",
    "no-memory": "techjam-aigc/trace-rx-m-v2-ablation-no-memory",
    "bce-only": "techjam-aigc/trace-rx-m-v2-ablation-bce-only",
}
COMPONENT_RUNS = ("frozen-encoder", "no-memory", "bce-only")
MAX_FPR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def normalized_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    x = fpr[:stop].tolist()
    y = tpr[:stop].tolist()
    if not x or x[-1] < max_fpr:
        index = min(stop, len(fpr) - 1)
        left = max(0, index - 1)
        value = float(np.interp(max_fpr, fpr[[left, index]], tpr[[left, index]]))
        x.append(max_fpr)
        y.append(value)
    return float(np.trapezoid(y, x) / max_fpr)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = tpr[fpr <= limit]
    return float(valid.max()) if len(valid) else 0.0


def binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["target"].to_numpy(dtype=int)
    scores = frame["logit"].to_numpy(dtype=float)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "normalized_pauc_0_05": normalized_pauc(labels, scores),
        "tpr_at_fpr_0_05": tpr_at_fpr(labels, scores, 0.05),
        "tpr_at_fpr_0_01": tpr_at_fpr(labels, scores, 0.01),
    }


def sliced_metrics(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"overall": binary_metrics(frame)}
    transforms: dict[str, dict[str, float]] = {}
    for name, group in frame.groupby("transform_family", dropna=False):
        if group["target"].nunique() == 2:
            transforms[str(name)] = binary_metrics(group)
    result["by_transform"] = transforms
    result["worst_transform_roc_auc"] = min(
        (values["roc_auc"] for values in transforms.values()), default=None
    )

    negatives = frame[frame["target"].eq(0)]
    threshold = float(np.quantile(negatives["logit"], 0.95, method="higher"))
    subtype_fpr = {
        str(name): float((group["logit"] >= threshold).mean())
        for name, group in negatives.groupby("authentic_subtype", dropna=False)
    }
    result["threshold_at_overall_authentic_fpr_0_05"] = threshold
    result["authentic_subtype_fpr"] = subtype_fpr
    result["worst_authentic_subtype_fpr"] = max(subtype_fpr.values(), default=None)
    return result


def paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260831,
) -> dict[str, object]:
    keys = ["lineage_id", "parent_id", "condition"]
    left = reference[keys + ["target", "logit"]].rename(columns={"logit": "reference"})
    right = candidate[keys + ["target", "logit"]].rename(columns={"logit": "candidate"})
    paired = left.merge(right, on=keys + ["target"], how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Component ablations do not contain identical validation endpoints.")
    groups = {name: indices.to_numpy() for name, indices in paired.groupby("lineage_id").groups.items()}
    lineages = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    observed = (
        normalized_pauc(paired["target"].to_numpy(), paired["reference"].to_numpy())
        - normalized_pauc(paired["target"].to_numpy(), paired["candidate"].to_numpy())
    )
    deltas: list[float] = []
    for _ in range(samples):
        drawn = rng.choice(lineages, size=len(lineages), replace=True)
        indices = np.concatenate([groups[item] for item in drawn])
        labels = paired["target"].to_numpy()[indices]
        if len(np.unique(labels)) < 2:
            continue
        deltas.append(
            normalized_pauc(labels, paired["reference"].to_numpy()[indices])
            - normalized_pauc(labels, paired["candidate"].to_numpy()[indices])
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "metric": "reference_minus_candidate_normalized_pauc_0_05",
        "observed_delta": observed,
        "ci_95": [float(low), float(high)],
        "bootstrap_samples": len(deltas),
        "lineages": len(lineages),
        "supports_full_component": bool(low > 0),
    }


def make_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# TRACE-RX-M ablation results",
        "",
        "All metrics use the validation split only; the organizer test split was not opened.",
        "",
        "| Run | Held-out ROC-AUC | Overall ROC-AUC | AP | n-pAUC@5% | TPR@1% FPR | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for slug in RUNS:
        run = summary["runs"][slug]
        overall = run["metrics"]["overall"]
        lines.append(
            f"| {slug} | {run['validity']['roc_auc']:.4f} | {overall['roc_auc']:.4f} | "
            f"{overall['average_precision']:.4f} | {overall['normalized_pauc_0_05']:.4f} | "
            f"{overall['tpr_at_fpr_0_01']:.4f} | "
            f"{'pass' if run['validity']['gate_passed'] else 'fail'} |"
        )
    lines.extend(("", "## Paired component deltas", ""))
    for slug, delta in summary["paired_deltas"].items():
        low, high = delta["ci_95"]
        lines.append(
            f"- `{slug}`: full-minus-ablation n-pAUC delta {delta['observed_delta']:.4f} "
            f"(95% lineage-bootstrap CI {low:.4f} to {high:.4f}); "
            f"component supported: **{'yes' if delta['supports_full_component'] else 'no'}**."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.artifacts_root.resolve()
    frames: dict[str, pd.DataFrame] = {}
    summary: dict[str, object] = {"test_split_opened": False, "runs": {}, "paired_deltas": {}}
    for slug, repo_id in RUNS.items():
        output = root / slug
        frame = pd.read_csv(output / "scores.csv")
        if set(frame["split"]) - {"val", "train"}:
            raise ValueError(f"Unexpected non-validation score split in {slug}.")
        validation = frame[frame["split"].eq("val")].reset_index(drop=True)
        frames[slug] = validation
        validity = json.loads((output / "s4_validity.json").read_text())
        metadata = json.loads((output / "run-metadata.json").read_text())
        summary["runs"][slug] = {
            "repo_id": repo_id,
            "validity": validity,
            "runtime": metadata,
            "metrics": sliced_metrics(validation),
        }

    reference = frames["holdout-gemini"]
    for slug in COMPONENT_RUNS:
        summary["paired_deltas"][slug] = paired_bootstrap(
            reference,
            frames[slug],
            samples=args.bootstrap_samples,
        )

    summary_path = root / "summary.json"
    markdown_path = root / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(make_markdown(summary))

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        for repo_id in RUNS.values():
            for path in (summary_path, markdown_path):
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"ablation-results/{path.name}",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Upload five-run TRACE-RX-M ablation comparison",
                )


if __name__ == "__main__":
    main()
