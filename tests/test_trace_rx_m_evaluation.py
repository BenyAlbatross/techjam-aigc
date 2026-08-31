from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from techjam_aigc.trace_rx_m.evaluation import (
    ALWAYS_REPORT_METRICS,
    binary_detection_metrics,
    clean_to_condition_drops,
    metric_slices,
    paired_endpoint_drift,
)
from techjam_aigc.trace_rx_m.evaluation_data import (
    AssignedChainEndpointDataset,
    TransformEndpointDataset,
    assigned_chain_banks,
    assigned_transform_specs,
    load_evaluation_datasets,
    resolve_transform_specs,
)


def test_binary_detection_metrics_always_include_balanced_accuracy() -> None:
    metrics = binary_detection_metrics(
        np.array([0, 0, 0, 1]),
        np.array([-2.0, -1.0, 1.0, 2.0]),
        threshold=0.0,
    )

    assert ALWAYS_REPORT_METRICS == (
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1.0) / 2)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 0], [0.1, 0.2], "both binary classes"),
        ([0, 1], [0.1], "equal length"),
        ([0, 1], [0.1, np.nan], "finite"),
    ],
)
def test_binary_detection_metrics_reject_invalid_inputs(labels, scores, message) -> None:
    with pytest.raises(ValueError, match=message):
        binary_detection_metrics(labels, scores, threshold=0.0)


def _write_spec(path: Path, values: dict[str, object]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, **values}), encoding="utf-8")
    return path


def test_csv_dataset_spec_maps_filters_and_resolves_paths(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    for name, color in (
        ("real.png", "navy"),
        ("fake.png", "orange"),
        ("ignored.png", "gray"),
    ):
        Image.new("RGB", (12, 8), color).save(image_directory / name)
    pd.DataFrame([
        {"path": "images/real.png", "label": "real", "id": "real", "split": "test"},
        {"path": "images/fake.png", "label": "fake", "id": "fake", "split": "test"},
        {"path": "images/ignored.png", "label": "real", "id": "ignored", "split": "train"},
    ]).to_csv(tmp_path / "labels.csv", index=False)
    spec = _write_spec(tmp_path / "dataset.json", {
        "dataset_id": "mapped",
        "adapter": "csv",
        "manifest": "labels.csv",
        "root": ".",
        "columns": {"image_path": "path", "target": "label", "parent_id": "id"},
        "filters": {"split": ["test"]},
        "label_map": {"real": 0, "fake": 1},
    })

    records, specs = load_evaluation_datasets([spec], repo_root=tmp_path)

    assert [item.dataset_id for item in specs] == ["mapped"]
    assert records[["parent_id", "target"]].to_dict("records") == [
        {"parent_id": "real", "target": 0},
        {"parent_id": "fake", "target": 1},
    ]
    assert records["resolved_path"].map(lambda value: Path(value).is_file()).all()
    assert records.loc[records["target"].eq(1), "generator_family"].item() == "unknown_aigc"


def test_class_folder_adapter_discovers_supported_images(tmp_path: Path) -> None:
    for directory, color in (("real", "navy"), ("generated", "orange")):
        class_directory = tmp_path / directory
        class_directory.mkdir()
        Image.new("RGB", (10, 10), color).save(class_directory / "one.png")
        (class_directory / "ignore.txt").write_text("not an image")
    spec = _write_spec(tmp_path / "folders.json", {
        "dataset_id": "folders",
        "adapter": "class_folders",
        "root": ".",
        "classes": {"real": 0, "generated": 1},
        "class_metadata": {
            "generated": {
                "generator_family": "test-generator",
                "generation_model": "test-model"
            }
        }
    })

    records, _ = load_evaluation_datasets([spec], repo_root=tmp_path)

    assert len(records) == 2
    assert set(records["target"]) == {0, 1}
    assert records.loc[records["target"].eq(1), "generator_family"].item() == "test-generator"


def test_transform_endpoint_dataset_is_exhaustive_and_deterministic(tmp_path: Path) -> None:
    image = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
    path = tmp_path / "source.png"
    Image.fromarray(image, "RGB").save(path)
    records = pd.DataFrame([{
        "dataset_id": "one",
        "parent_id": "parent",
        "lineage_id": "lineage",
        "image_path": "source.png",
        "resolved_path": str(path),
        "target": 1,
        "generator_family": "generator",
        "generation_model": "model",
        "source_dataset": "source",
        "authentic_subtype": "not_applicable",
    }])
    transforms = resolve_transform_specs(
        ("core",), conditions=("clean", "noise_sigma0.02", "jpeg_q70")
    )
    dataset = TransformEndpointDataset(records, transforms, image_size=8, base_seed=17)

    assert len(dataset) == 3
    assert [dataset[index]["condition"] for index in range(3)] == [
        "clean",
        "noise_sigma0.02",
        "jpeg_q70",
    ]
    np.testing.assert_array_equal(dataset[1]["pixel_values"], dataset[1]["pixel_values"])
    assert dataset[0]["pixel_values"].shape == (3, 8, 8)
    assert [dataset[index]["transform_step_count"] for index in range(3)] == [0, 1, 1]


def test_assigned_chain_endpoint_dataset_gives_each_parent_clean_and_lengths_1_to_3(
    tmp_path: Path,
) -> None:
    for index, color in enumerate(("navy", "orange")):
        Image.new("RGB", (16, 12), color).save(tmp_path / f"{index}.png")
    records = pd.DataFrame([
        {
            "dataset_id": "assigned",
            "parent_id": f"parent-{index}",
            "lineage_id": f"lineage-{index}",
            "image_path": f"{index}.png",
            "resolved_path": str(tmp_path / f"{index}.png"),
            "target": index,
            "generator_family": "generator" if index else "authentic",
            "generation_model": "model" if index else "none",
            "source_dataset": "source",
            "authentic_subtype": "not_applicable" if index else "real-source",
        }
        for index in range(2)
    ])
    banks = assigned_chain_banks((1, 2, 3))
    dataset = AssignedChainEndpointDataset(records, banks, image_size=8, base_seed=17)
    same_seed_dataset = AssignedChainEndpointDataset(
        records,
        banks,
        image_size=8,
        base_seed=17,
    )

    assert len(dataset) == 8
    assert [dataset[index]["transform_step_count"] for index in range(4)] == [0, 1, 2, 3]
    assert [dataset[index]["condition"] for index in range(4)] == [
        same_seed_dataset[index]["condition"] for index in range(4)
    ]
    assert dataset[0]["condition"] == "clean"
    assert len(assigned_transform_specs(banks)) == 1 + sum(map(len, banks.values()))


def _prediction_rows() -> pd.DataFrame:
    rows = []
    clean_scores = {"real-a": -2.0, "real-b": -1.0, "fake-a": 2.0, "fake-b": 1.0}
    transformed_scores = {"real-a": 1.0, "real-b": -0.5, "fake-a": 0.2, "fake-b": -1.0}
    for condition, scores in (("clean", clean_scores), ("jpeg_q30", transformed_scores)):
        for parent_id, score in scores.items():
            target = int(parent_id.startswith("fake"))
            rows.append({
                "dataset_id": "dataset",
                "parent_id": parent_id,
                "target": target,
                "condition": condition,
                "transform_family": "clean" if condition == "clean" else "jpeg",
                "generator_family": "generator" if target else "authentic",
                "authentic_subtype": "real-source" if not target else "not_applicable",
                "logit": score,
            })
    return pd.DataFrame(rows)


def test_condition_metrics_drops_and_paired_drift() -> None:
    predictions = _prediction_rows()

    metrics = metric_slices(
        predictions,
        ["dataset_id", "condition", "transform_family"],
        threshold=0.0,
    )
    drops = clean_to_condition_drops(metrics)
    drift = paired_endpoint_drift(predictions, threshold=0.0)

    assert metrics["status"].eq("ok").all()
    assert metrics.loc[metrics["condition"].eq("clean"), "roc_auc"].item() == pytest.approx(1.0)
    assert drops.loc[0, "roc_auc_drop"] > 0
    assert drift.loc[0, "prediction_flip_rate"] == pytest.approx(0.5)
    assert drift.loc[0, "correct_to_incorrect_rate"] == pytest.approx(0.5)


def test_metric_slices_records_one_class_groups_instead_of_crashing() -> None:
    predictions = _prediction_rows()

    metrics = metric_slices(predictions, ["target"], threshold=0.0)

    assert set(metrics["status"]) == {"unscorable_one_class"}
    assert metrics["roc_auc"].isna().all()
