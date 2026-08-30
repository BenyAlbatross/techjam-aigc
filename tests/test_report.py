import csv
import json
from pathlib import Path

import pytest

from scripts.benchmark import CONDITIONS
from scripts.report import build_summary
from scripts.report import grouped_bootstrap
from scripts.report import main
from scripts.report import metrics
from scripts.report import paired_deltas
from scripts.report import rank_models
from scripts.report import render_report
from scripts.report import validate_coverage


def row(
    sample_id: str,
    label: int,
    score: float,
    *,
    model: str = "model-a",
    dataset: str = "sid_set",
    condition: str = "clean",
    base_id: str | None = None,
    identity: str | None = None,
    cohort: str | None = None,
) -> dict:
    return {
        "identity": identity or f"{model}:{dataset}:{sample_id}:{condition}",
        "model": model,
        "dataset": dataset,
        "sample_id": sample_id,
        "base_id": sample_id if base_id is None else base_id,
        "label": label,
        "cohort": cohort or ("real-source" if label == 0 else "ai-source"),
        "condition": condition,
        "raw_score": score,
        "probability_ai": score,
        "threshold": 0.5,
        "decision": int(score >= 0.5),
        "elapsed_seconds": 1.0,
        "attempted_count": 2,
        "valid_count": 2,
        "excluded_count": 0,
    }


def test_metrics_are_exact_and_auc_is_tie_aware():
    rows = [
        row("r1", 0, 0.1),
        row("r2", 0, 0.9),
        row("a1", 1, 0.8),
        row("a2", 1, 0.2),
    ]

    result = metrics(rows, threshold=0.5)

    assert result == {
        "n": 4,
        "error_rate": 0.5,
        "fpr": 0.5,
        "fnr": 0.5,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5,
        "confusion": {"tp": 1, "tn": 1, "fp": 1, "fn": 1},
    }
    tied = [row("r1", 0, 0.5), row("r2", 0, 0.5), row("a1", 1, 0.5)]
    assert metrics(tied)["roc_auc"] == 0.5


def test_metrics_cover_perfect_reversed_empty_and_single_class_inputs():
    perfect = [row("r", 0, 0.1), row("a", 1, 0.9)]
    reversed_rows = [row("r", 0, 0.9), row("a", 1, 0.1)]

    assert metrics(perfect)["roc_auc"] == 1.0
    assert metrics(reversed_rows)["roc_auc"] == 0.0
    assert metrics([]) == {
        "n": 0,
        "error_rate": None,
        "fpr": None,
        "fnr": None,
        "balanced_accuracy": None,
        "roc_auc": None,
        "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }
    real_only = metrics([row("r", 0, 0.9)])
    assert real_only["error_rate"] == 1.0
    assert real_only["fpr"] == 1.0
    assert real_only["fnr"] is None
    assert real_only["balanced_accuracy"] is None
    assert real_only["roc_auc"] is None


def test_grouped_bootstrap_resamples_whole_base_ids_and_is_deterministic():
    rows = [
        row("a-clean", 0, 0.0, base_id="a"),
        row("b-clean", 1, 1.0, base_id="b"),
        row("b-jpeg", 1, 1.0, base_id="b", condition="jpeg_q90"),
        row("b-blur", 1, 1.0, base_id="b", condition="blur_sigma1"),
    ]

    first = grouped_bootstrap(rows, lambda sampled: len(sampled), replicates=200)

    assert first == (2.0, 6.0)
    assert first == grouped_bootstrap(
        rows, lambda sampled: len(sampled), replicates=200
    )


def test_grouped_bootstrap_uses_sample_id_only_when_lineage_is_absent():
    rows = [
        {**row("x", 0, 0.0, dataset="wild"), "base_id": ""},
        {**row("y", 1, 1.0, dataset="wild"), "base_id": ""},
    ]

    assert grouped_bootstrap(rows, lambda sampled: len(sampled), replicates=50) == (
        2.0,
        2.0,
    )


def test_paired_deltas_join_clean_and_transformed_by_exact_sample_identity():
    rows = [
        row("s1", 0, 0.2),
        row("s2", 1, 0.8),
        row("s1", 0, 0.7, condition="jpeg_q90"),
        row("s2", 1, 0.6, condition="jpeg_q90"),
        row("unpaired", 1, 0.1, condition="jpeg_q90"),
    ]

    assert paired_deltas(rows) == [
        {
            "model": "model-a",
            "dataset": "sid_set",
            "condition": "jpeg_q90",
            "paired_count": 2,
            "mean_score_delta": pytest.approx(0.15),
            "decision_flip_rate": 0.5,
        }
    ]


def test_validate_coverage_rejects_duplicate_conflict_and_missing_rows():
    complete = [
        row(sample, label, 0.1 if label == 0 else 0.9, condition=condition)
        for condition in CONDITIONS
        for sample, label in (("r", 0), ("a", 1))
    ]
    selected = {"sid_set": {"r", "a"}}

    assert validate_coverage(complete, selected, expected_models=["model-a"]) == []

    duplicate = complete + [dict(complete[0])]
    assert any("duplicate identity" in error for error in validate_coverage(
        duplicate, selected, expected_models=["model-a"]
    ))

    conflict = complete + [{**complete[0], "probability_ai": 0.8}]
    assert any("conflicting duplicate identity" in error for error in validate_coverage(
        conflict, selected, expected_models=["model-a"]
    ))

    missing = complete[:-1]
    assert any("missing expected row" in error for error in validate_coverage(
        missing, selected, expected_models=["model-a"]
    ))


def test_validate_coverage_requires_clean_only_for_wild_panels():
    wild = [row("r", 0, 0.1, dataset="wild")]
    selected = {"wild": {"r"}}

    assert validate_coverage(wild, selected, expected_models=["model-a"]) == []
    transformed = wild + [row("r", 0, 0.2, dataset="wild", condition="jpeg_q90")]
    assert any("unexpected row" in error for error in validate_coverage(
        transformed, selected, expected_models=["model-a"]
    ))


def test_count_coverage_rejects_replaced_controlled_condition():
    rows = [
        row(sample, label, 0.1 if label == 0 else 0.9, condition=condition)
        for condition in CONDITIONS
        for sample, label in (("r", 0), ("a", 1))
    ]
    for item in rows[-2:]:
        item["condition"] = "replacement"

    errors = validate_coverage(
        rows, {"sid_set": 2}, expected_models=["model-a"]
    )

    assert any("condition set" in error for error in errors)


def test_count_coverage_rejects_wrong_per_condition_count():
    rows = [
        row(sample, label, 0.1 if label == 0 else 0.9, condition=condition)
        for condition in CONDITIONS
        for sample, label in (("r", 0), ("a", 1))
    ]
    rows[-1]["condition"] = CONDITIONS[0]
    rows[-1]["sample_id"] = "third"

    errors = validate_coverage(
        rows, {"sid_set": 2}, expected_models=["model-a"]
    )

    assert any("condition count" in error for error in errors)


def test_count_coverage_requires_identical_sample_sets_across_conditions():
    rows = [
        row(sample, label, 0.1 if label == 0 else 0.9, condition=condition)
        for condition in CONDITIONS
        for sample, label in (("r", 0), ("a", 1))
    ]
    rows[-1]["sample_id"] = "replacement"

    errors = validate_coverage(
        rows, {"sid_set": 2}, expected_models=["model-a"]
    )

    assert any("sample identities differ" in error for error in errors)


def summary_row(model: str, **overrides) -> dict:
    result = {
        "model": model,
        "dataset": "sid_set",
        "dataset_status": "approved",
        "worst_condition_balanced_accuracy": 0.7,
        "worst_real_source_fpr": 0.2,
        "worst_ai_source_fnr": 0.3,
        "aggregate_roc_auc": 0.8,
        "ranking_eligible": True,
    }
    result.update(overrides)
    return result


def approved_registry(*models: str, **overrides) -> dict:
    registry = {model: {"status": "approved"} for model in models}
    registry.update(overrides)
    return registry


def test_rank_models_excludes_contaminated_and_unapproved_pairs():
    summary = [
        summary_row("clean"),
        summary_row("contaminated", dataset="leaked"),
        summary_row("review", dataset="review_set", dataset_status="review"),
        summary_row("blocked", dataset="blocked_set", dataset_status="blocked"),
    ]
    contamination = approved_registry(
        "clean", "review", "blocked",
        contaminated={
            "status": "approved",
            "contaminated_datasets": ["leaked"],
        },
    )

    assert rank_models(summary, contamination) == ["clean"]
    assert summary[0]["ranking_eligible"] is True
    assert summary[1]["ranking_eligible"] is False
    assert "contaminated" in summary[1]["exclusion_reason"]
    assert summary[2]["ranking_eligible"] is False
    assert summary[3]["ranking_eligible"] is False


def test_rank_models_excludes_unapproved_model_registry_entries():
    summary = [summary_row("approved"), summary_row("review")]

    assert rank_models(summary, {
        "approved": {"status": "approved"},
        "review": {"status": "review"},
    }) == ["approved"]
    assert summary[1]["ranking_eligible"] is False
    assert "model status is review" in summary[1]["exclusion_reason"]


def test_rank_models_uses_every_preregistered_tie_break_in_order():
    summary = [
        summary_row("z-name"),
        summary_row("a-name"),
        summary_row("auc", aggregate_roc_auc=0.81),
        summary_row("fnr", worst_ai_source_fnr=0.29),
        summary_row("fpr", worst_real_source_fpr=0.19),
        summary_row("balanced", worst_condition_balanced_accuracy=0.71),
    ]

    assert rank_models(summary, approved_registry(*(item["model"] for item in summary))) == [
        "balanced",
        "fpr",
        "fnr",
        "auc",
        "a-name",
        "z-name",
    ]


def test_summary_uses_worst_source_condition_not_cross_condition_average():
    rows = [
        row("r", 0, 0.1, condition="clean", cohort="opaque-real"),
        row("a", 1, 0.9, condition="clean", cohort="opaque-ai"),
        row("r", 0, 0.9, condition="jpeg_q90", cohort="opaque-real"),
        row("a", 1, 0.9, condition="jpeg_q90", cohort="opaque-ai"),
    ]

    summary = build_summary(
        rows,
        {"model-a": {}},
        {"sid_set": {"status": "approved"}},
        replicates=20,
    )

    assert summary["pairs"][0]["worst_real_source_fpr"] == 1.0
    assert summary["pairs"][0]["worst_cohorts"]["real"] == (
        "opaque-real/jpeg_q90"
    )


def test_summary_retains_all_condition_and_pair_ranking_intervals():
    rows = [
        row("r1", 0, 0.1, base_id="pair-1"),
        row("a1", 1, 0.9, base_id="pair-1"),
        row("r2", 0, 0.2, base_id="pair-2"),
        row("a2", 1, 0.8, base_id="pair-2"),
    ]

    summary = build_summary(
        rows,
        {"model-a": {"status": "approved"}},
        {"sid_set": {"status": "approved"}},
        replicates=20,
    )

    assert summary["conditions"][0]["metric_confidence_intervals"] == {
        "error_rate": [0.0, 0.0],
        "fpr": [0.0, 0.0],
        "fnr": [0.0, 0.0],
        "balanced_accuracy": [1.0, 1.0],
        "roc_auc": [1.0, 1.0],
    }
    assert summary["pairs"][0]["ranking_confidence_intervals"] == {
        "worst_condition_balanced_accuracy": [1.0, 1.0],
        "worst_real_source_fpr": [0.0, 0.0],
        "worst_ai_source_fnr": [0.0, 0.0],
        "aggregate_roc_auc": [1.0, 1.0],
    }


def test_overlapping_ranking_intervals_make_winner_and_boundary_unresolved():
    rows = []
    models = ("a", "b", "c", "d")
    for model in models:
        rows.extend([
            row("r", 0, 0.1, model=model, base_id="paired"),
            row("a", 1, 0.9, model=model, base_id="paired"),
        ])

    summary = build_summary(
        rows,
        approved_registry(*models),
        {"sid_set": {"status": "approved"}},
        replicates=20,
    )

    assert summary["ranking"] == ["a", "b", "c", "d"]
    assert summary["ranking_resolution"] == {
        "unresolved_groups": [["a", "b", "c", "d"]],
        "winner_supported": False,
        "selected_winner": None,
        "top_three_boundary_supported": False,
        "selected_top_three": [],
    }


def test_ranking_resolution_has_no_winner_when_no_pair_is_eligible():
    rows = [
        row("r", 0, 0.1, base_id="paired"),
        row("a", 1, 0.9, base_id="paired"),
    ]

    summary = build_summary(
        rows,
        {"model-a": {"status": "review"}},
        {"sid_set": {"status": "approved"}},
        replicates=20,
    )

    assert summary["ranking"] == []
    assert summary["ranking_resolution"]["winner_supported"] is None
    assert summary["ranking_resolution"]["selected_winner"] is None

def _report_summary() -> dict:
    return {
        "conditions": [{
            "model": "model-a",
            "dataset": "sid_set",
            "condition": "clean",
            "metrics": {
                "n": 2, "error_rate": 0.0, "fpr": 0.0, "fnr": 0.0,
                "balanced_accuracy": 1.0, "roc_auc": 1.0,
                "confusion": {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
            },
            "metric_confidence_intervals": {
                name: [value, value] for name, value in {
                    "error_rate": 0.0, "fpr": 0.0, "fnr": 0.0,
                    "balanced_accuracy": 1.0, "roc_auc": 1.0,
                }.items()
            },
        }],
        "pairs": [summary_row(
            "model-a",
            throughput_rows_per_second=2.0,
            invalid_count=0,
            worst_cohorts={"real": "opaque-real", "ai": "opaque-ai"},
            ranking_confidence_intervals={
                "worst_condition_balanced_accuracy": [0.5, 1.0],
                "worst_real_source_fpr": [0.0, 0.5],
                "worst_ai_source_fnr": [0.0, 0.5],
                "aggregate_roc_auc": [0.5, 1.0],
            },
        )],
        "deltas": [{
            "model": "model-a",
            "dataset": "sid_set",
            "condition": "jpeg_q90",
            "paired_count": 2,
            "mean_score_delta": -0.1,
            "decision_flip_rate": 0.5,
        }],
        "ranking": ["model-a"],
        "ranking_resolution": {
            "unresolved_groups": [["model-a", "model-b"]],
            "winner_supported": False,
            "selected_winner": None,
            "top_three_boundary_supported": False,
            "selected_top_three": [],
        },
        "limitations": ["No thresholds were tuned or calibrated."],
    }


def test_render_report_covers_required_sections_without_paths():
    markdown = render_report(_report_summary())

    for text in (
        "Per-condition metrics and 95% confidence intervals",
        "Worst cohorts",
        "Score deltas and decision flips",
        "Throughput and invalid counts",
        "Contamination and exclusions",
        "Top-three rule",
        "Scope limitations",
        "opaque-real",
        "TP | TN | FP | FN",
        "Statistically unresolved groups",
        "Selected winner: unresolved",
        "Selected top three: unresolved",
    ):
        assert text in markdown
    assert "image_path" not in markdown
    assert "Top three: model-a" not in markdown


def test_cli_reads_existing_rows_and_writes_json_csv_and_markdown(tmp_path: Path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    prediction_rows = [
        row(sample, label, 0.1 if label == 0 else 0.9, condition=condition)
        for condition in CONDITIONS
        for sample, label in (("r", 0), ("a", 1))
    ]
    (predictions / "rows.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in prediction_rows)
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "sid_set",
        "samples": [{"sample_id": "r"}, {"sample_id": "a"}],
    }))
    models = tmp_path / "models.toml"
    models.write_text('[models.model-a]\nstatus = "approved"\n')
    datasets = tmp_path / "datasets.toml"
    datasets.write_text('[datasets.sid_set]\nstatus = "approved"\n')
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "metrics.csv"
    markdown_path = tmp_path / "report.md"

    exit_code = main([
        "--predictions", str(predictions),
        "--manifest", str(manifest),
        "--models", str(models),
        "--datasets", str(datasets),
        "--json", str(json_path),
        "--csv", str(csv_path),
        "--markdown", str(markdown_path),
    ])

    assert exit_code == 0
    summary = json.loads(json_path.read_text())
    assert summary["ranking"] == ["model-a"]
    assert len(summary["conditions"]) == 15
    assert set(summary["conditions"][0]["metric_confidence_intervals"]) == {
        "error_rate", "fpr", "fnr", "balanced_accuracy", "roc_auc",
    }
    assert set(summary["pairs"][0]["ranking_confidence_intervals"]) == {
        "worst_condition_balanced_accuracy", "worst_real_source_fpr",
        "worst_ai_source_fnr", "aggregate_roc_auc",
    }
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {item["record_type"] for item in csv_rows} == {
        "condition", "delta", "pair",
    }
    condition = next(item for item in csv_rows if item["record_type"] == "condition")
    pair = next(item for item in csv_rows if item["record_type"] == "pair")
    delta = next(item for item in csv_rows if item["record_type"] == "delta")
    assert {"tp", "tn", "fp", "fn"} <= condition.keys()
    for metric in ("error_rate", "fpr", "fnr", "balanced_accuracy", "roc_auc"):
        assert f"{metric}_ci_low" in condition
        assert f"{metric}_ci_high" in condition
    assert delta["mean_score_delta"] != ""
    assert delta["decision_flip_rate"] != ""
    for field in (
        "throughput_rows_per_second", "invalid_count", "ranking_eligible",
        "exclusion_reason", "rank", "winner_supported",
        "top_three_boundary_supported",
    ):
        assert field in pair
    assert "Top-three rule" in markdown_path.read_text()

    with pytest.raises(SystemExit):
        main([
            "--predictions", str(predictions),
            "--manifest", str(manifest),
            "--models", str(models),
            "--datasets", str(datasets),
            "--json", str(json_path),
            "--csv", str(csv_path),
            "--markdown", str(markdown_path),
            "--bootstrap-replicates", "20",
        ])
