from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from techjam_aigc.feature_lab.evaluation import (
    auprc_degradation_tables,
    chronological_confirmation_metrics,
    directed_pair_interactions,
    directed_pair_order_sensitivity,
    discovery_directions,
    parent_paired_feature_drift,
    write_evaluation_cache,
)
from techjam_aigc.feature_lab.registry import FEATURE_REGISTRY
from techjam_aigc.feature_lab.transforms import transform_frame


def _feature_row(
    parent_id: str,
    phase: str,
    condition: str,
    target: int,
    value: float,
    *,
    view: str = "canonical_128",
    window: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "parent_id": parent_id,
        "local_path": f"{parent_id}.png",
        "phase": phase,
        "condition": condition,
        "transform_family": "clean" if condition == "clean" else "jpeg",
        "official_transform": True,
        "view": view,
        "dataset": "toy",
        "generation_model": "toygen" if target else "none",
        "source_dataset": "toy-source",
        "target": target,
        "luma_mean": value,
    }
    if window is not None:
        row["chronological_window"] = window
    return row


def _metric(condition: str, auprc: float, prevalence: float = 0.5) -> dict[str, object]:
    return {
        "view": "canonical_128",
        "feature": "luma_mean",
        "condition": condition,
        "oriented_auprc": auprc,
        "positive_prevalence": prevalence,
        "auprc_gain": auprc - prevalence,
        "normalized_auprc": (auprc - prevalence) / (1 - prevalence),
        "direction": 1,
        "parents": 20,
        "low_power": False,
        "direction_reversal": auprc < prevalence,
    }


def test_parent_paired_drift_is_oriented_and_uses_only_registered_present_features() -> None:
    rows = [
        _feature_row("d-real", "discovery", "clean", 0, 0.0),
        _feature_row("d-fake", "discovery", "clean", 1, 1.0),
    ]
    for target, clean in ((0, 0.1), (1, 1.1)):
        parent = f"c-{target}"
        rows.append(_feature_row(parent, "confirmation", "clean", target, clean))
        rows.append(_feature_row(parent, "confirmation", "jpeg_q70", target, clean + 0.2))
    features = pd.DataFrame(rows)

    directions = discovery_directions(features)
    assert directions["feature"].tolist() == ["luma_mean"]
    drift = parent_paired_feature_drift(features, directions)

    assert set(drift["target"]) == {0, 1}
    assert drift["signed_drift_mean"].to_numpy() == pytest.approx([0.2, 0.2])
    assert drift["absolute_drift_mean"].to_numpy() == pytest.approx([0.2, 0.2])
    assert drift["within_parent_transform_variance_mean"].to_numpy() == pytest.approx([0.01, 0.01])


def test_auprc_drop_and_severity_area_use_official_ordered_severities() -> None:
    metrics = pd.DataFrame([
        _metric("clean", 0.9),
        _metric("blur_sigma0.5", 0.8),
        _metric("blur_sigma1", 0.7),
        _metric("blur_sigma2", 0.6),
        _metric("color_jitter_minus20", 0.5),
        _metric("color_jitter_plus20", 0.5),
    ])

    drops, areas = auprc_degradation_tables(metrics)

    blur_drops = drops[drops["transform_family"] == "gaussian_blur"]
    assert blur_drops["auprc_drop"].to_numpy() == pytest.approx([0.1, 0.2, 0.3])
    blur_area = areas[areas["transform_family"] == "gaussian_blur"].iloc[0]
    assert blur_area["normalized_severity_auprc_area"] == pytest.approx(0.725)
    assert blur_area["normalized_severity_auprc_drop_area"] == pytest.approx(0.175)
    assert "color_jitter" not in set(areas["transform_family"])


def test_directed_pair_interaction_and_exact_reverse_order_sensitivity() -> None:
    metadata = transform_frame("all")
    pairs = metadata[metadata["design"] == "directed_medium_pair"].set_index("ordered_operations")
    jpeg_then_blur = pairs.loc["jpeg>gaussian_blur", "name"]
    blur_then_jpeg = pairs.loc["gaussian_blur>jpeg", "name"]
    metrics = pd.DataFrame([
        _metric("clean", 0.9),
        _metric("jpeg_q70", 0.8),
        _metric("blur_sigma1", 0.85),
        _metric(jpeg_then_blur, 0.65),
        _metric(blur_then_jpeg, 0.75),
    ])

    interactions = directed_pair_interactions(metrics)
    assert set(interactions["pair_condition"]) == {jpeg_then_blur, blur_then_jpeg}
    by_condition = interactions.set_index("pair_condition")
    assert by_condition.loc[jpeg_then_blur, "interaction_excess_auprc_drop"] == pytest.approx(0.1)
    assert by_condition.loc[blur_then_jpeg, "interaction_excess_auprc_drop"] == pytest.approx(0.0)

    order = directed_pair_order_sensitivity(interactions)
    assert len(order) == 1
    assert order.iloc[0]["absolute_order_sensitivity"] == pytest.approx(0.1)
    assert {order.iloc[0]["condition_a_then_b"], order.iloc[0]["condition_b_then_a"]} == {
        jpeg_then_blur,
        blur_then_jpeg,
    }


def test_core_metrics_return_empty_interaction_schemas() -> None:
    interactions = directed_pair_interactions(pd.DataFrame([
        _metric("clean", 0.9),
        _metric("jpeg_q70", 0.8),
    ]))
    order = directed_pair_order_sensitivity(interactions)

    assert interactions.empty and "interaction_excess_auprc_drop" in interactions
    assert order.empty and "absolute_order_sensitivity" in order


def test_chronological_metrics_require_a_window_and_keep_discovery_direction() -> None:
    without_window = pd.DataFrame([
        _feature_row("d-real", "discovery", "clean", 0, 0.0),
        _feature_row("d-fake", "discovery", "clean", 1, 1.0),
        _feature_row("f-real", "final_confirmation", "clean", 0, 1.0),
        _feature_row("f-fake", "final_confirmation", "clean", 1, 0.0),
    ])
    empty = chronological_confirmation_metrics(
        without_window,
        repetitions=5,
        base_seed=1,
        min_group_images=1,
    )
    assert empty.empty and "chronological_window" in empty

    with_window = without_window.copy()
    with_window["chronological_window"] = [None, None, "future", "future"]
    result = chronological_confirmation_metrics(
        with_window,
        repetitions=5,
        base_seed=1,
        min_group_images=1,
    )
    assert set(result["phase"]) == {"final_confirmation"}
    assert set(result["chronological_window"]) == {"future"}
    assert result.iloc[0]["direction"] == 1
    assert result.iloc[0]["oriented_auprc"] == pytest.approx(0.5)
    assert not bool(result.iloc[0]["direction_reversal"])


def test_evaluation_cache_writes_extension_tables_for_core_profile(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for view in ("native_capped", "canonical_128"):
        for phase in ("discovery", "confirmation"):
            conditions = ("clean",) if phase == "discovery" else ("clean", "jpeg_q70")
            for condition in conditions:
                for target in (0, 1):
                    for index in range(2):
                        value = float(target) + index / 100
                        row = _feature_row(
                            f"{phase}-{target}-{index}", phase, condition, target, value, view=view
                        )
                        for spec in FEATURE_REGISTRY:
                            row[spec.name] = value
                        rows.append(row)
    tables = write_evaluation_cache(
        tmp_path,
        pd.DataFrame(rows),
        repetitions=2,
        seed=1,
        min_group_images=1,
    )

    expected = {
        "parent_paired_feature_drift",
        "clean_to_condition_auprc_drop",
        "severity_area",
        "directed_pair_interactions",
        "directed_pair_order_sensitivity",
        "chronological_confirmation_metrics",
    }
    assert expected <= set(tables)
    assert tables["directed_pair_interactions"].empty
    for name in expected:
        assert (tmp_path / f"{name}.csv.gz").is_file()
    assert (tmp_path / "evaluation_metadata.json").is_file()
