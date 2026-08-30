"""Deterministic reporting over frozen prediction shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
import tomllib

import numpy as np

if __package__:
    from scripts.benchmark import CONDITIONS, SEED
else:
    from benchmark import CONDITIONS, SEED


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_DATASET = "sid_set"
METRIC_NAMES = ("error_rate", "fpr", "fnr", "balanced_accuracy", "roc_auc")
RANKING_KEYS = (
    "worst_condition_balanced_accuracy",
    "worst_real_source_fpr",
    "worst_ai_source_fnr",
    "aggregate_roc_auc",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rank_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(labels[index] for index in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(rows: list[dict], threshold: float | None = None) -> dict:
    if not rows:
        return {
            "n": 0,
            "error_rate": None,
            "fpr": None,
            "fnr": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }
    cutoff = float(rows[0]["threshold"] if threshold is None else threshold)
    labels = [int(item["label"]) for item in rows]
    scores = [float(item["probability_ai"]) for item in rows]
    predicted = [score >= cutoff for score in scores]
    tp = sum(label == 1 and decision for label, decision in zip(labels, predicted))
    tn = sum(label == 0 and not decision for label, decision in zip(labels, predicted))
    fp = sum(label == 0 and decision for label, decision in zip(labels, predicted))
    fn = sum(label == 1 and not decision for label, decision in zip(labels, predicted))
    fpr = _ratio(fp, fp + tn)
    fnr = _ratio(fn, fn + tp)
    balanced = None if fpr is None or fnr is None else ((1 - fpr) + (1 - fnr)) / 2
    return {
        "n": len(rows),
        "error_rate": (fp + fn) / len(rows),
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": balanced,
        "roc_auc": _rank_auc(labels, scores),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def grouped_bootstrap(
    rows: list[dict],
    statistic: Callable[[list[dict]], float | None],
    seed: int = SEED,
    replicates: int = 2000,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    if replicates < 1:
        raise ValueError("replicates must be positive")
    values = []
    for sampled in _grouped_samples(rows, seed, replicates):
        value = statistic(sampled)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return _percentile_interval(values)


def _grouped_samples(rows: list[dict], seed: int, replicates: int):
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        group_id = item.get("base_id")
        if not group_id:
            if item.get("dataset") == CONTROLLED_DATASET:
                raise ValueError("controlled rows require base_id lineage")
            group_id = item.get("sample_id")
        if not group_id:
            raise ValueError("rows require base_id or sample_id")
        groups[str(group_id)].append(item)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = []
        for index in rng.integers(0, len(group_ids), size=len(group_ids)):
            sampled.extend(groups[group_ids[int(index)]])
        yield sampled


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _grouped_intervals(
    rows: list[dict],
    statistic: Callable[[list[dict]], dict],
    keys: Iterable[str],
    replicates: int,
) -> dict[str, list[float | None]]:
    values = {key: [] for key in keys}
    for sampled in _grouped_samples(rows, SEED, replicates):
        result = statistic(sampled)
        for key in keys:
            value = result[key]
            if value is not None and math.isfinite(float(value)):
                values[key].append(float(value))
    return {key: list(_percentile_interval(items)) for key, items in values.items()}


def paired_deltas(rows: list[dict]) -> list[dict]:
    clean = {}
    for item in rows:
        if item["condition"] != "clean":
            continue
        key = (item["model"], item["dataset"], item["sample_id"])
        if key in clean:
            raise ValueError(f"duplicate clean row: {item['identity']}")
        clean[key] = item
    grouped: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for item in rows:
        condition = item["condition"]
        if condition == "clean":
            continue
        key = (item["model"], item["dataset"], item["sample_id"])
        baseline = clean.get(key)
        if baseline is None:
            continue
        grouped[(item["model"], item["dataset"], condition)].append((
            float(item["probability_ai"]) - float(baseline["probability_ai"]),
            int(item["decision"] != baseline["decision"]),
        ))
    output = []
    for (model, dataset, condition), values in sorted(grouped.items()):
        output.append({
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "paired_count": len(values),
            "mean_score_delta": sum(value[0] for value in values) / len(values),
            "decision_flip_rate": sum(value[1] for value in values) / len(values),
        })
    return output


def _sample_ids(value: Iterable[str] | int) -> set[str] | None:
    if isinstance(value, int):
        return None
    return {str(item) for item in value}


def validate_coverage(
    rows: list[dict],
    selected_samples: dict[str, Iterable[str] | int],
    expected_models: Iterable[str] | None = None,
) -> list[str]:
    errors = []
    identities = {}
    prediction_keys = {}
    for item in rows:
        identity = item.get("identity")
        if identity in identities:
            kind = "duplicate identity" if item == identities[identity] else "conflicting duplicate identity"
            errors.append(f"{kind}: {identity}")
        else:
            identities[identity] = item
        key = (
            item.get("model"), item.get("dataset"), item.get("sample_id"),
            item.get("condition"),
        )
        if key in prediction_keys and prediction_keys[key] != identity:
            errors.append(f"conflicting prediction tuple: {key}")
        else:
            prediction_keys[key] = identity

    models = sorted(expected_models or {item.get("model") for item in rows})
    expected_keys = set()
    count_expectations = {}
    for model in models:
        for dataset, selected in selected_samples.items():
            conditions = CONDITIONS if dataset == CONTROLLED_DATASET else ("clean",)
            sample_ids = _sample_ids(selected)
            if sample_ids is None:
                count_expectations[(model, dataset)] = (selected, conditions)
            else:
                expected_keys.update(
                    (model, dataset, sample_id, condition)
                    for sample_id in sample_ids
                    for condition in conditions
                )

    actual_keys = set(prediction_keys)
    for key in sorted(expected_keys - actual_keys):
        errors.append(f"missing expected row: {key}")
    for key in sorted(actual_keys - expected_keys) if expected_keys else ():
        errors.append(f"unexpected row: {key}")
    for pair, (selected_count, conditions) in count_expectations.items():
        pair_rows = [
            item for item in rows
            if item["model"] == pair[0] and item["dataset"] == pair[1]
        ]
        actual_conditions = {item["condition"] for item in pair_rows}
        if actual_conditions != set(conditions):
            errors.append(
                f"condition set mismatch for {pair}: expected {sorted(conditions)}, "
                f"got {sorted(actual_conditions)}"
            )
        sample_sets = []
        for condition in conditions:
            condition_rows = [item for item in pair_rows if item["condition"] == condition]
            if len(condition_rows) != selected_count:
                errors.append(
                    f"condition count mismatch for {pair}/{condition}: "
                    f"expected {selected_count}, got {len(condition_rows)}"
                )
            sample_sets.append({item["sample_id"] for item in condition_rows})
        if sample_sets and any(items != sample_sets[0] for items in sample_sets[1:]):
            errors.append(f"sample identities differ across conditions for {pair}")
    return errors


def rank_models(summary: list[dict], contamination: dict) -> list[str]:
    eligible = []
    for item in summary:
        reasons = []
        model_entry = contamination.get(item["model"], {})
        if model_entry.get("status") != "approved":
            reasons.append(f"model status is {model_entry.get('status')}")
        if item.get("dataset_status") != "approved":
            reasons.append(f"dataset status is {item.get('dataset_status')}")
        contaminated = model_entry.get("contaminated_datasets", [])
        if item["dataset"] in contaminated:
            reasons.append(f"contaminated model/dataset pair: {item['dataset']}")
        if item.get("ranking_eligible") is False:
            reasons.append(item.get("exclusion_reason", "marked ineligible"))
        if any(item.get(key) is None for key in RANKING_KEYS):
            reasons.append("ranking metrics are incomplete")
        item["ranking_eligible"] = not reasons
        item["exclusion_reason"] = "; ".join(dict.fromkeys(reasons))
        if not reasons:
            eligible.append(item)
    eligible.sort(key=lambda item: (
        -item["worst_condition_balanced_accuracy"],
        item["worst_real_source_fpr"],
        item["worst_ai_source_fnr"],
        -item["aggregate_roc_auc"],
        item["model"],
    ))
    return [item["model"] for item in eligible]


def _worst_cohort(rows: list[dict], label: int, metric_name: str) -> tuple[str | None, float | None]:
    cohorts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in rows:
        if item["label"] == label:
            cohorts[(item["cohort"], item["condition"])].append(item)
    values = [(key, metrics(items)[metric_name]) for key, items in cohorts.items()]
    values = [(key, value) for key, value in values if value is not None]
    if not values:
        return None, None
    (cohort, condition), value = max(values, key=lambda pair: (pair[1], pair[0]))
    return f"{cohort}/{condition}", value


def _ranking_statistics(rows: list[dict]) -> dict:
    condition_rows: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        condition_rows[item["condition"]].append(item)
    balanced = [metrics(items)["balanced_accuracy"] for items in condition_rows.values()]
    _, worst_fpr = _worst_cohort(rows, 0, "fpr")
    _, worst_fnr = _worst_cohort(rows, 1, "fnr")
    return {
        "worst_condition_balanced_accuracy": min(
            (value for value in balanced if value is not None), default=None
        ),
        "worst_real_source_fpr": worst_fpr,
        "worst_ai_source_fnr": worst_fnr,
        "aggregate_roc_auc": metrics(rows)["roc_auc"],
    }


def _comparison_supported(first: dict, second: dict) -> bool:
    intervals_first = first.get("ranking_confidence_intervals", {})
    intervals_second = second.get("ranking_confidence_intervals", {})
    for index, key in enumerate(RANKING_KEYS):
        if first[key] == second[key]:
            continue
        first_ci = intervals_first.get(key, [None, None])
        second_ci = intervals_second.get(key, [None, None])
        if None in first_ci or None in second_ci:
            return False
        if index in (0, 3):
            return first_ci[0] > second_ci[1]
        return first_ci[1] < second_ci[0]
    return False


def _ranking_resolution(pairs: list[dict], ranking: list[str]) -> dict:
    by_model = {item["model"]: item for item in pairs}
    ordered = [by_model[model] for model in ranking]
    unresolved_groups = []
    if ordered:
        group = [ordered[0]["model"]]
        for first, second in zip(ordered, ordered[1:]):
            if _comparison_supported(first, second):
                if len(group) > 1:
                    unresolved_groups.append(group)
                group = [second["model"]]
            else:
                group.append(second["model"])
        if len(group) > 1:
            unresolved_groups.append(group)
    winner_supported = (
        None if not ordered
        else len(ordered) == 1 or _comparison_supported(ordered[0], ordered[1])
    )
    boundary_supported = (
        _comparison_supported(ordered[2], ordered[3]) if len(ordered) >= 4 else None
    )
    return {
        "unresolved_groups": unresolved_groups,
        "winner_supported": winner_supported,
        "selected_winner": ordered[0]["model"] if ordered and winner_supported else None,
        "top_three_boundary_supported": boundary_supported,
        "selected_top_three": ranking[:3] if boundary_supported else [],
    }


def build_summary(
    rows: list[dict],
    model_registry: dict,
    dataset_registry: dict,
    replicates: int = 2000,
) -> dict:
    condition_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in rows:
        condition_groups[(item["model"], item["dataset"], item["condition"])].append(item)
        pair_groups[(item["model"], item["dataset"])].append(item)

    conditions = []
    for (model, dataset, condition), items in sorted(condition_groups.items()):
        result = metrics(items)
        intervals = _grouped_intervals(
            items, metrics, METRIC_NAMES, replicates,
        )
        conditions.append({
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "metrics": result,
            "metric_confidence_intervals": intervals,
            "balanced_accuracy_ci": intervals["balanced_accuracy"],
        })

    pairs = []
    for (model, dataset), items in sorted(pair_groups.items()):
        ranking_statistics = _ranking_statistics(items)
        real_cohort, _ = _worst_cohort(items, 0, "fpr")
        ai_cohort, _ = _worst_cohort(items, 1, "fnr")
        shard_rows = {}
        for item in items:
            shard_rows.setdefault(item["condition"], item)
        elapsed = sum(float(item["elapsed_seconds"]) for item in shard_rows.values())
        invalid = sum(int(item["excluded_count"]) for item in shard_rows.values())
        pairs.append({
            "model": model,
            "dataset": dataset,
            "model_status": model_registry.get(model, {}).get("status", "unknown"),
            "dataset_status": dataset_registry.get(dataset, {}).get("status", "unknown"),
            **ranking_statistics,
            "ranking_confidence_intervals": _grouped_intervals(
                items, _ranking_statistics, RANKING_KEYS, replicates,
            ),
            "worst_cohorts": {"real": real_cohort, "ai": ai_cohort},
            "throughput_rows_per_second": len(items) / elapsed if elapsed else None,
            "invalid_count": invalid,
            "ranking_eligible": True,
            "exclusion_reason": "",
        })
    ranking = rank_models(pairs, model_registry)
    return {
        "conditions": conditions,
        "pairs": pairs,
        "deltas": paired_deltas(rows),
        "ranking": ranking,
        "ranking_resolution": _ranking_resolution(pairs, ranking),
        "limitations": [
            "No thresholds were tuned or calibrated.",
            "Only approved uncontaminated model/dataset pairs enter ranking.",
            "Controlled confidence intervals resample base_id; no-lineage wild data uses sample_id.",
        ],
    }


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_report(summary: dict) -> str:
    lines = [
        "# Public baseline robustness report",
        "",
        "## Per-condition metrics and 95% confidence intervals",
        "",
        "| Model | Dataset | Condition | N | TP | TN | FP | FN | Error (95% CI) | FPR (95% CI) | FNR (95% CI) | Balanced accuracy (95% CI) | AUROC (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for item in summary.get("conditions", []):
        metric = item["metrics"]
        intervals = item["metric_confidence_intervals"]
        confusion = metric["confusion"]
        lines.append(
            f"| {item['model']} | {item['dataset']} | {item['condition']} | "
            f"{metric['n']} | {confusion['tp']} | {confusion['tn']} | {confusion['fp']} | "
            f"{confusion['fn']} | {_fmt(metric['error_rate'])} "
            f"[{_fmt(intervals['error_rate'][0])}, {_fmt(intervals['error_rate'][1])}] | "
            f"{_fmt(metric['fpr'])} [{_fmt(intervals['fpr'][0])}, {_fmt(intervals['fpr'][1])}] | "
            f"{_fmt(metric['fnr'])} [{_fmt(intervals['fnr'][0])}, {_fmt(intervals['fnr'][1])}] | "
            f"{_fmt(metric['balanced_accuracy'])} [{_fmt(intervals['balanced_accuracy'][0])}, "
            f"{_fmt(intervals['balanced_accuracy'][1])}] | {_fmt(metric['roc_auc'])} "
            f"[{_fmt(intervals['roc_auc'][0])}, {_fmt(intervals['roc_auc'][1])}] |"
        )
    lines.extend([
        "",
        "## Worst cohorts",
        "",
        "| Model | Dataset | Worst real cohort | Worst FPR | Worst AI cohort | Worst FNR |",
        "|---|---|---|---:|---|---:|",
    ])
    for item in summary.get("pairs", []):
        cohorts = item.get("worst_cohorts", {})
        lines.append(
            f"| {item['model']} | {item['dataset']} | {_fmt(cohorts.get('real'))} | "
            f"{_fmt(item.get('worst_real_source_fpr'))} | {_fmt(cohorts.get('ai'))} | "
            f"{_fmt(item.get('worst_ai_source_fnr'))} |"
        )
    lines.extend([
        "",
        "## Ranking uncertainty",
        "",
        "| Model | Dataset | Worst BA (95% CI) | Worst FPR (95% CI) | Worst FNR (95% CI) | Aggregate AUROC (95% CI) |",
        "|---|---|---|---|---|---|",
    ])
    for item in summary.get("pairs", []):
        intervals = item.get("ranking_confidence_intervals", {})
        values = [
            f"{_fmt(item[key])} [{_fmt(intervals.get(key, [None, None])[0])}, "
            f"{_fmt(intervals.get(key, [None, None])[1])}]"
            for key in RANKING_KEYS
        ]
        lines.append(
            f"| {item['model']} | {item['dataset']} | " + " | ".join(values) + " |"
        )
    lines.extend([
        "",
        "## Score deltas and decision flips",
        "",
        "| Model | Dataset | Condition | Paired N | Mean score delta | Decision flip rate |",
        "|---|---|---|---:|---:|---:|",
    ])
    for item in summary.get("deltas", []):
        lines.append(
            f"| {item['model']} | {item['dataset']} | {item['condition']} | "
            f"{item['paired_count']} | {_fmt(item['mean_score_delta'])} | "
            f"{_fmt(item['decision_flip_rate'])} |"
        )
    lines.extend([
        "",
        "## Throughput and invalid counts",
        "",
        "| Model | Dataset | Rows/second | Invalid count |",
        "|---|---|---:|---:|",
    ])
    for item in summary.get("pairs", []):
        lines.append(
            f"| {item['model']} | {item['dataset']} | "
            f"{_fmt(item.get('throughput_rows_per_second'))} | {item.get('invalid_count', 0)} |"
        )
    lines.extend([
        "",
        "## Contamination and exclusions",
        "",
        "| Model | Dataset | Ranking eligible | Reason |",
        "|---|---|---|---|",
    ])
    for item in summary.get("pairs", []):
        lines.append(
            f"| {item['model']} | {item['dataset']} | "
            f"{str(item.get('ranking_eligible', False)).lower()} | "
            f"{item.get('exclusion_reason') or 'none'} |"
        )
    lines.extend([
        "",
        "## Top-three rule",
        "",
        "Rank eligible uncontaminated pairs by worst-condition balanced accuracy (descending), "
        "worst real-source FPR (ascending), worst AI-source FNR (ascending), aggregate AUROC "
        "(descending), then model name. Deterministic display order: "
        + (", ".join(summary.get("ranking", [])) or "none"),
        "",
    ])
    resolution = summary.get("ranking_resolution", {})
    groups = resolution.get("unresolved_groups", [])
    lines.extend([
        "Statistically unresolved groups: "
        + ("; ".join(", ".join(group) for group in groups) or "none"),
        "",
        "Selected winner: " + (
            resolution.get("selected_winner") or "unresolved"
        ),
        "",
        "Selected top three: " + (
            ", ".join(resolution.get("selected_top_three", []))
            if resolution.get("selected_top_three") else "unresolved"
        ),
        "",
        "## Scope limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in summary.get("limitations", []))
    return "\n".join(lines) + "\n"


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for shard in sorted(path.rglob("*.jsonl")):
        with shard.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL in {shard.name}:{line_number}") from error
                if not isinstance(item, dict):
                    raise ValueError(f"prediction row must be an object: {shard.name}:{line_number}")
                rows.append(item)
    if not rows:
        raise ValueError("no prediction rows found")
    return rows


def _load_toml(path: Path, section: str) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle).get(section, {})


def _write_csv(path: Path, summary: dict) -> None:
    fields = [
        "record_type", "model", "dataset", "condition", "n", "tp", "tn", "fp", "fn",
        "error_rate", "fpr", "fnr", "balanced_accuracy", "roc_auc",
        *(f"{metric}_ci_{bound}" for metric in METRIC_NAMES for bound in ("low", "high")),
        "paired_count", "mean_score_delta", "decision_flip_rate",
        *RANKING_KEYS,
        *(f"{key}_ci_{bound}" for key in RANKING_KEYS for bound in ("low", "high")),
        "throughput_rows_per_second", "invalid_count", "model_status", "dataset_status",
        "ranking_eligible", "exclusion_reason", "rank", "unresolved_group",
        "winner_supported", "top_three_boundary_supported", "selected_winner",
        "selected_top_three",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary["conditions"]:
            metric = item["metrics"]
            intervals = item["metric_confidence_intervals"]
            output = {
                "record_type": "condition",
                "model": item["model"],
                "dataset": item["dataset"],
                "condition": item["condition"],
                "n": metric["n"],
                **metric["confusion"],
                "error_rate": metric["error_rate"],
                "fpr": metric["fpr"],
                "fnr": metric["fnr"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "roc_auc": metric["roc_auc"],
            }
            for metric_name, interval in intervals.items():
                output[f"{metric_name}_ci_low"], output[f"{metric_name}_ci_high"] = interval
            writer.writerow(output)
        for item in summary["deltas"]:
            writer.writerow({"record_type": "delta", **item})
        resolution = summary["ranking_resolution"]
        ranking = summary["ranking"]
        for item in summary["pairs"]:
            output = {
                "record_type": "pair",
                **{key: item.get(key) for key in (
                    "model", "dataset", *RANKING_KEYS, "throughput_rows_per_second",
                    "invalid_count", "model_status", "dataset_status", "ranking_eligible",
                    "exclusion_reason",
                )},
                "rank": ranking.index(item["model"]) + 1 if item["model"] in ranking else "",
                "unresolved_group": next((
                    ",".join(group) for group in resolution["unresolved_groups"]
                    if item["model"] in group
                ), ""),
                "winner_supported": resolution["winner_supported"],
                "top_three_boundary_supported": resolution["top_three_boundary_supported"],
                "selected_winner": resolution["selected_winner"],
                "selected_top_three": ",".join(resolution["selected_top_three"]),
            }
            for key, interval in item["ranking_confidence_intervals"].items():
                output[f"{key}_ci_low"], output[f"{key}_ci_high"] = interval
            writer.writerow(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets.toml")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = _load_rows(args.predictions)
    manifest = json.loads(args.manifest.read_text())
    selected = {manifest["dataset"]: {item["sample_id"] for item in manifest["samples"]}}
    errors = validate_coverage(rows, selected, expected_models=sorted({item["model"] for item in rows}))
    if errors:
        raise ValueError("invalid report coverage:\n" + "\n".join(errors))
    summary = build_summary(
        rows,
        _load_toml(args.models, "models"),
        _load_toml(args.datasets, "datasets"),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(args.csv, summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
