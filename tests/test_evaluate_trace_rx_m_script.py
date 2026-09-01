from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
from PIL import Image
import pytest

from techjam_aigc.trace_rx_m.config import TraceRXMConfig


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_trace_rx_m.py"
SPEC = importlib.util.spec_from_file_location("evaluate_trace_rx_m", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_spec(path: Path, values: dict[str, object]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, **values}))
    return path


def test_validation_only_suite_smokes_csv_and_zip_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.json"
    TraceRXMConfig().write(config)
    images = tmp_path / "images"
    images.mkdir()
    for name, color in (("real.png", "navy"), ("fake.png", "orange")):
        Image.new("RGB", (301, 229), color).save(images / name)
    pd.DataFrame([
        {"path": "images/real.png", "label": "real", "id": "real"},
        {"path": "images/fake.png", "label": "fake", "id": "fake"},
    ]).to_csv(tmp_path / "as-is.csv", index=False)
    as_is = _write_spec(tmp_path / "as-is.json", {
        "dataset_id": "as-is",
        "adapter": "csv",
        "manifest": "as-is.csv",
        "root": ".",
        "columns": {"image_path": "path", "target": "label", "parent_id": "id"},
        "label_map": {"real": 0, "fake": 1},
        "expected_rows": 2,
        "expected_target_counts": {"0": 1, "1": 1},
    })

    archive = tmp_path / "positive.zip"
    with ZipFile(archive, "w") as handle:
        handle.write(images / "fake.png", "generated/fake.png")
    zipped = _write_spec(tmp_path / "zip.json", {
        "dataset_id": "zip-positive",
        "adapter": "zip_class_folders",
        "archive": "positive.zip",
        "classes": {
            "generated": {
                "target": 1,
                "generator_family": "toy",
                "generation_model": "toy-v1",
            }
        },
        "expected_counts": {"generated": 1},
        "expected_rows": 1,
        "expected_target_counts": {"1": 1},
    })
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--config", str(config),
        "--as-is-dataset-spec", str(as_is),
        "--uniform-chain-dataset-spec", str(zipped),
        "--validate-only",
        "--output", str(output),
        "--repo-root", str(tmp_path),
    ])

    MODULE.main()

    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["source_images"] == metadata["transformed_endpoints"] == 3
    assert metadata["assignment_rows"] == 1
    assert len(metadata["source_inventory_sha256"]) == 64
    assignments = pd.read_csv(output / "transform_assignments.csv")
    assert len(assignments) == assignments["parent_id"].nunique() == 1
    checks = pd.read_csv(output / "preprocessing_validation.csv")
    assert checks["shape"].eq("3x224x224").all()


def test_dataset_limit_is_exact_class_source_stratified_and_replayable() -> None:
    rows = []
    for dataset_id in ("first", "second"):
        for target in (0, 1):
            for source in ("a", "b"):
                for index in range(5):
                    rows.append({
                        "dataset_id": dataset_id,
                        "parent_id": f"{dataset_id}-{target}-{source}-{index}",
                        "target": target,
                        "generation_model": source if target else "not_applicable",
                        "authentic_subtype": source if not target else "not_applicable",
                    })
    records = pd.DataFrame(rows)

    first = MODULE._limit_records(records, 12, 17)
    replay = MODULE._limit_records(records, 12, 17)
    changed = MODULE._limit_records(records, 12, 18)

    assert first.groupby("dataset_id").size().eq(12).all()
    assert first.groupby(["dataset_id", "target"]).size().eq(6).all()
    assert first.groupby(
        ["dataset_id", "target", "generation_model", "authentic_subtype"]
    ).size().eq(3).all()
    pd.testing.assert_frame_equal(first, replay)
    assert set(first["parent_id"]) != set(changed["parent_id"])


def test_v2_config_accepts_dinov3_variants_and_rejects_dinov2(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    TraceRXMConfig().write(current)
    assert MODULE._strict_v2_config(current).backbone.model_id.startswith(
        "facebook/dinov3-"
    )

    historical_values = json.loads(current.read_text())
    historical_values["backbone"]["model_id"] = "facebook/dinov2-small"
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps(historical_values))
    with pytest.raises(ValueError, match="rejects historical DINOv2"):
        MODULE._strict_v2_config(historical)


def test_markdown_report_renders_metrics_and_unavailable_values(tmp_path: Path) -> None:
    metrics = pd.DataFrame([{
        "dataset_id": "evalgen-positive-only",
        "rows": 2,
        "positives": 2,
        "negatives": 0,
        "true_positive": 1,
        "true_negative": float("nan"),
        "false_positive": float("nan"),
        "false_negative": 1,
        "roc_auc": float("nan"),
        "recall": 0.5,
        "false_negative_rate": 0.5,
        "status": "positive_only",
    }])
    metrics.to_csv(tmp_path / "metrics_by_dataset.csv", index=False)
    metrics.assign(transform_step_count=1).to_csv(
        tmp_path / "metrics_by_chain_length.csv", index=False
    )
    metrics.assign(transform_family="clean").to_csv(
        tmp_path / "metrics_by_supplied_transform_family.csv", index=False
    )
    metrics.assign(comparison_generator_family="toy-generator").to_csv(
        tmp_path / "metrics_by_generator_family.csv", index=False
    )
    pd.DataFrame(columns=["condition", "transform_family", "pairs"]).to_csv(
        tmp_path / "techjam_lineage_paired_drift.csv", index=False
    )
    metadata = {
        "created_at": "2026-09-01T00:00:00+00:00",
        "checkpoint_epoch": 5,
        "source_images": 2,
        "seed": 17,
        "threshold": 0.0,
        "threshold_scale": "logit",
        "inference_seconds": 1.0,
        "inference_throughput_endpoints_per_second": 2.0,
        "full_inventory": [{
            "dataset_id": "evalgen-positive-only",
            "rows": 10,
            "target_counts": {"1": 10},
        }],
        "sampled_inventory": [{
            "dataset_id": "evalgen-positive-only",
            "rows": 2,
            "target_counts": {"1": 2},
        }],
        "endpoint_policies": {
            "uniform_sequential_chain": ["evalgen-positive-only"],
        },
    }

    MODULE._write_markdown_report(
        tmp_path,
        summary={"false_positive_rows": 0, "false_negative_rows": 1},
        metadata=metadata,
    )

    report = (tmp_path / "evaluation_report.md").read_text()
    assert "# TRACE-RX-M v2 evaluation results" in report
    assert "evalgen-positive-only" in report
    assert "positive_only" in report
    assert "N/A" in report
