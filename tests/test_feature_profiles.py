from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import techjam_aigc.feature_lab.features as feature_module
from techjam_aigc.feature_lab.features import extract_features
from techjam_aigc.feature_lab.pipeline import ExperimentConfig, write_extraction_cache
from techjam_aigc.feature_lab.registry import (
    DEFAULT_FEATURE_PROFILE,
    EXPANDED_V2_ONLY_NAMES,
    FEATURE_PROFILES,
    feature_names,
    feature_schema_sha256,
    registry_frame,
)


def test_versioned_feature_profiles_preserve_original_schema() -> None:
    frozen = feature_names(profile="frozen_v1")
    expanded = feature_names(profile="expanded_v2")

    assert tuple(FEATURE_PROFILES) == ("frozen_v1", "expanded_v2")
    assert len(frozen) == 53
    assert len(expanded) == 82
    assert set(expanded) - set(frozen) == set(EXPANDED_V2_ONLY_NAMES)
    assert len(EXPANDED_V2_ONLY_NAMES) == 29
    assert {
        "self_jpeg70_mse",
        "self_blur1_mse",
        "self_resize05_mse",
        "self_jpeg70_gradient_drop",
    } <= set(frozen)
    assert feature_schema_sha256("frozen_v1") != feature_schema_sha256("expanded_v2")


def test_default_pipeline_profile_is_frozen_and_skips_extensions(monkeypatch) -> None:
    assert DEFAULT_FEATURE_PROFILE == "frozen_v1"
    assert ExperimentConfig().feature_profile == "frozen_v1"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("expanded feature helper ran under frozen_v1")

    monkeypatch.setattr(feature_module, "_bitplane_features", fail_if_called)
    values = extract_features(Image.new("RGB", (24, 24), "gray"), profile="frozen_v1")
    assert set(values) == set(feature_names(profile="frozen_v1"))


def test_both_extraction_profiles_are_deterministic_and_finite() -> None:
    image = Image.new("RGB", (1, 1), (127, 127, 127))
    for profile, expected_count in (("frozen_v1", 53), ("expanded_v2", 82)):
        first = extract_features(image, profile=profile)
        second = extract_features(image, profile=profile)
        assert first == second
        assert len(first) == expected_count
        assert np.isfinite(list(first.values())).all()

    assert extract_features(image) == extract_features(image, profile="expanded_v2")


def test_cache_records_resolved_feature_profile_and_schema(tmp_path: Path) -> None:
    binary_index = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "phase": "discovery",
                "binary_label": "authentic",
                "generator_family": "authentic",
                "generation_model": "none",
                "source_dataset": "toy-source",
            }
        ]
    )
    index_path = tmp_path / "index.csv"
    binary_index.to_csv(index_path, index=False)
    config = ExperimentConfig(
        views=("canonical_128",),
        conditions=("clean",),
        feature_profile="frozen_v1",
    )

    metadata = write_extraction_cache(
        tmp_path / "cache",
        pd.DataFrame([{"feature_profile": "frozen_v1"}]),
        binary_index,
        index_path,
        config,
    )

    registry = pd.read_csv(tmp_path / "cache/feature_registry.csv")
    disk_metadata = json.loads((tmp_path / "cache/run_metadata.json").read_text())
    expected_hash = feature_schema_sha256("frozen_v1")

    assert len(registry) == len(registry_frame("frozen_v1")) == 53
    assert registry["feature_profile"].eq("frozen_v1").all()
    assert registry["feature_schema_sha256"].eq(expected_hash).all()
    assert metadata["feature_schema"] == disk_metadata["feature_schema"]
    assert metadata["feature_schema"]["profile"] == "frozen_v1"
    assert metadata["feature_schema"]["sha256"] == expected_hash
    assert metadata["feature_schema"]["feature_count"] == 53


def test_every_feature_has_complete_hypothesis_and_cost_metadata() -> None:
    registry = registry_frame("expanded_v2")
    for column in ("measurement", "hypothesis", "expected_failure", "family", "role", "cost"):
        assert registry[column].astype(str).str.strip().ne("").all()
    assert set(registry["cost"]) <= {"low", "medium", "high"}
