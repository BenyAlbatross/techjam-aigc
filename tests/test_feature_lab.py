from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from techjam_aigc.feature_lab.data import load_binary_index
from techjam_aigc.feature_lab.evaluation import discovery_directions, univariate_tables
from techjam_aigc.feature_lab.features import extract_features
from techjam_aigc.feature_lab.registry import FEATURE_REGISTRY
from techjam_aigc.feature_lab.transforms import TRANSFORM_SPECS, analysis_view, apply_transform


def test_all_brief_transform_parameters_are_registered() -> None:
    names = {spec.name for spec in TRANSFORM_SPECS}
    expected = {
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_sigma0.5", "blur_sigma1", "blur_sigma2",
        "resize_0.5", "resize_0.25",
        "noise_sigma0.02", "noise_sigma0.05", "noise_sigma0.10",
        "color_jitter_minus20", "color_jitter_plus20",
        "center_crop_80",
    }
    assert expected <= names


def test_transforms_are_deterministic_and_preserve_geometry() -> None:
    pixels = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
    image = Image.fromarray(pixels, "RGB")
    for spec in TRANSFORM_SPECS:
        first = apply_transform(image, spec.name, parent_id="parent-1")
        second = apply_transform(image, spec.name, parent_id="parent-1")
        assert first.mode == "RGB"
        assert first.size == image.size
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_analysis_views_make_preprocessing_explicit() -> None:
    image = Image.new("RGB", (640, 320), "gray")
    assert analysis_view(image, "native_capped").size == (256, 128)
    assert analysis_view(image, "canonical_128").size == (128, 128)


def test_feature_extractor_matches_registry_and_is_finite() -> None:
    y, x = np.mgrid[0:64, 0:64]
    rgb = np.stack([x * 4, y * 4, ((x + y) * 2)], axis=2).astype(np.uint8)
    values = extract_features(
        Image.fromarray(rgb, "RGB"),
        {"width": 64, "height": 64, "bytes": 4096, "format": "PNG"},
    )
    assert set(values) == {spec.name for spec in FEATURE_REGISTRY}
    assert np.isfinite(list(values.values())).all()


def test_binary_index_excludes_tampered_and_groups_variants(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        _index_row("a.jpg", "train", "real"),
        _index_row("b.jpg", "train", "full_synthetic"),
        _index_row("c.jpg", "validation", "real"),
        _index_row("d.jpg", "validation", "full_synthetic"),
        _index_row("e.jpg", "validation", "tampered"),
    ])
    path = tmp_path / "index.csv"
    frame.to_csv(path, index=False)
    binary = load_binary_index(path)
    assert len(binary) == 4
    assert set(binary["target"]) == {0, 1}
    assert set(binary["phase"]) == {"discovery", "confirmation"}
    assert binary["parent_id"].is_unique


def test_direction_is_learned_only_on_discovery_and_reversal_is_visible() -> None:
    records = []
    for view in ("canonical_128",):
        for phase, split in (("discovery", "train"), ("confirmation", "test")):
            for target in (0, 1):
                for index in range(6):
                    value = target + index / 100 if phase == "discovery" else (1 - target) + index / 100
                    record = {
                        "parent_id": f"{phase}-{target}-{index}",
                        "phase": phase,
                        "condition": "clean",
                        "transform_family": "clean",
                        "official_transform": True,
                        "view": view,
                        "dataset": "toy",
                        "generation_model": "toygen" if target else "none",
                        "target": target,
                    }
                    for spec in FEATURE_REGISTRY:
                        record[spec.name] = value
                    records.append(record)
    frame = pd.DataFrame(records)
    directions = discovery_directions(frame)
    assert (directions["direction"] == 1).all()
    _, transforms, _, generators = univariate_tables(
        frame,
        repetitions=20,
        base_seed=1,
        min_group_images=5,
    )
    assert transforms["direction_reversal"].all()
    assert generators["direction_reversal"].all()


def _index_row(path: str, split: str, label: str) -> dict[str, object]:
    return {
        "dataset": "SID Set",
        "split": split,
        "label": label,
        "source": "test",
        "generator_family": "diffusion" if label != "real" else "authentic",
        "generation_model": "toy" if label != "real" else "none",
        "source_dataset": "OpenImages",
        "split_method": "official",
        "stratum": "toy",
        "source_member": path,
        "local_path": path,
        "width": 64,
        "height": 64,
        "mode": "RGB",
        "format": "JPEG",
        "bytes": 1000,
    }
