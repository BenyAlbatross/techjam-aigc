from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from techjam_aigc.trace_rx_m.augment import canonical_preprocess

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_techjam2026_training.py"
SPEC = importlib.util.spec_from_file_location("prepare_techjam2026_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
prepare = MODULE.prepare


def test_preparation_preserves_v2_splits_rows_and_transform_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = []
    specifications = [
        ("train-real-1", "train", "real", False, "NIL", "NIL", "real_source"),
        ("train-real-2", "train", "real", False, "NIL", "NIL", "real_source"),
        ("train-real-3", "train", "real", False, "NIL", "NIL", "real_source"),
        ("train-ai", "train", "ai_full", False, "NIL", "NIL", "synthetic_text_to_image"),
        (
            "train-dda",
            "train",
            "ai_full",
            False,
            "NIL",
            "NIL",
            "synthetic_dual_alignment",
        ),
        ("val-ai", "val", "ai_full", False, "NIL", "NIL", "synthetic_text_to_image"),
        (
            "val-real-aug",
            "val",
            "real",
            True,
            "gaussian_noise",
            "noise_s0_05",
            "real_source",
        ),
        ("test-real", "test", "real", False, "NIL", "NIL", "real_source"),
        (
            "test-dda",
            "test",
            "ai_full",
            True,
            "jpeg",
            "jpeg_q70",
            "synthetic_dual_alignment",
        ),
    ]
    for asset_id, split, label, transformed, family, variant, origin in specifications:
        relative = Path("images") / f"{asset_id}.png"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (31, 19), "navy").save(path)
        rows.append({
            "asset_id": asset_id,
            "image_path": str(relative),
            "split": split,
            "label": label,
            "content_group_id": f"group-{asset_id}",
            "lineage_id": f"lineage-{asset_id}",
            "model_family": "generator" if label == "ai_full" else None,
            "generator_model": "model" if label == "ai_full" else None,
            "source_dataset": "source",
            "source_family": "camera" if label == "real" else "generator",
            "data_source_segment": "source-segment",
            "is_transformed": transformed,
            "transform_family": family,
            "transform_variant_id": variant,
            "image_origin": origin,
            "pair_role": "dda_reconstruction" if "dual_alignment" in origin else "unpaired",
        })
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    output = tmp_path / "normalized"
    args = SimpleNamespace(
        labels=labels,
        source_root=source,
        output_root=output,
        manifest=output / "manifest.csv",
        repo_root=tmp_path,
        image_size=16,
        memory_images=1,
        capacity_images=1,
        authentic_null_images=1,
        seed=9,
        workers=2,
        manifest_only=False,
    )

    manifest, summary = prepare(args)

    assert summary["training_pools"] == {
        "authentic_null": 1,
        "capacity": 1,
        "detector": 2,
        "memory": 1,
        "none": 4,
    }
    assert len(manifest) == len(specifications)
    assert "role" not in manifest.columns
    assert "phase" not in manifest.columns
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert summary["source_rows_retained"] == len(specifications)
    assert summary["splits"] == {"test": 2, "train": 5, "val": 2}
    assert summary["image_origins"]["synthetic_dual_alignment"] == 2
    rows = manifest.set_index("parent_id")
    assert rows.loc["val-ai", ["split", "training_pool"]].tolist() == ["val", "none"]
    assert rows.loc["val-real-aug", ["split", "training_pool"]].tolist() == [
        "val",
        "none",
    ]
    assert rows.loc["test-real", ["split", "training_pool"]].tolist() == [
        "test",
        "none",
    ]
    assert rows.loc["test-dda", ["split", "training_pool"]].tolist() == [
        "test",
        "none",
    ]
    kinds = manifest.set_index("parent_id")["sample_kind"]
    assert kinds["train-dda"] == "native_aigc"
    assert kinds["test-dda"] == "native_aigc"
    origins = manifest.set_index("parent_id")["image_origin"]
    assert origins["train-dda"] == "synthetic_dual_alignment"
    assert origins["test-dda"] == "synthetic_dual_alignment"
    endpoints = manifest.set_index("parent_id")[["condition", "transform_family"]]
    assert endpoints.loc["train-ai"].tolist() == ["clean", "clean"]
    assert endpoints.loc["val-real-aug"].tolist() == ["noise_s0_05", "gaussian_noise"]
    assert endpoints.loc["test-dda"].tolist() == ["jpeg_q70", "jpeg"]
    assert set(manifest["format"]) == {"BMP"}
    assert set(manifest["width"]) == {16}
    assert set(manifest["height"]) == {16}
    assert manifest["bytes"].nunique() == 1
    assert summary["preprocessing_version"] == "center-crop-v1"
    assert summary["normalization"] == (
        "RGB bicubic 512px short-side limit, center crop with zero-padding, "
        "then uncompressed BMP"
    )
    for value in manifest["local_path"]:
        assert value.endswith(".center-crop-v1.bmp")
        with Image.open(tmp_path / value) as image:
            assert image.mode == "RGB"
            assert image.size == (16, 16)

    source_path = source / "images" / f"{manifest.iloc[0]['parent_id']}.png"
    normalized_path = tmp_path / manifest.iloc[0]["local_path"]
    with Image.open(source_path) as source_image, Image.open(normalized_path) as normalized:
        np.testing.assert_array_equal(
            canonical_preprocess(source_image, image_size=16),
            canonical_preprocess(normalized, image_size=16),
        )
