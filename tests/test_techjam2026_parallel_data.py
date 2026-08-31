from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_techjam2026_parallel.py"
SPEC = importlib.util.spec_from_file_location("prepare_techjam2026_parallel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


def _row(
    asset: str,
    *,
    split: str,
    label: str,
    group: str,
    family: str = "authentic",
) -> dict[str, object]:
    is_real = label == "real"
    return {
        "image_path": f"images/{split}/{label}/{asset}.png",
        "asset_id": asset,
        "label": label,
        "split": split,
        "source_dataset": "camera" if is_real else "generation",
        "source_family": "camera" if is_real else family,
        "lineage_id": f"lineage-{asset}",
        "content_group_id": group,
        "generator_model": None if is_real else family,
        "model_family": None if is_real else family,
        "width": 20,
        "height": 12,
        "file_format": "PNG",
        "file_size_bytes": 100,
        "sha256": asset.ljust(64, "0")[:64],
        "licence": "CC BY 4.0",
    }


def _source(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = [
        *[
            _row(f"train-real-{index}", split="train", label="real", group=f"tr-{index}")
            for index in range(7)
        ],
        _row("train-flux", split="train", label="ai_full", group="ta-0", family="flux_1_schnell"),
        _row("train-gemini", split="train", label="ai_full", group="ta-1", family="gemini_flash_image"),
        _row("dev-gemini", split="dev", label="ai_full", group="dg-0", family="gemini_flash_image"),
        _row("dev-flux", split="dev", label="ai_full", group="ds-0", family="flux_1_schnell"),
        _row("dev-real-gate", split="dev", label="real", group="dr-0"),
        _row("dev-real-selection", split="dev", label="real", group="dr-1"),
        _row("cal-real", split="calibration", label="real", group="cr-0"),
        _row("cal-ai", split="calibration", label="ai_full", group="ca-0", family="flux_1_schnell"),
        _row("locked-real", split="own_locked", label="real", group="lr-0"),
        _row("locked-ai", split="own_locked", label="ai_full", group="la-0", family="gemini_flash_image"),
    ]
    frame = pd.DataFrame(rows)
    source_root = tmp_path / "source"
    source_root.mkdir()
    frame.to_csv(source_root / "labels.csv", index=False)
    (source_root / "verification.json").write_text(
        json.dumps({"asset_count": len(frame), "release_id": "test-release"})
    )
    for row in frame.itertuples():
        if row.split == "own_locked":
            continue
        path = source_root / row.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 12), "navy").save(path)
    return source_root, frame


def _args(tmp_path: Path, source_root: Path, *, acknowledged: bool = True):
    return argparse.Namespace(
        labels=None,
        source_root=source_root,
        output_root=tmp_path / "normalized",
        manifest=tmp_path / "training-manifest.csv",
        repo_root=tmp_path,
        image_size=16,
        memory_pool_images=1,
        capacity_validation_images=1,
        authentic_null_images=1,
        generator_gate_real_images=1,
        held_out_generator_family="gemini_flash_image",
        seed=7,
        workers=2,
        acknowledge_dataset_terms=acknowledged,
        no_download=True,
        include_locked_images=False,
        manifest_only=False,
    )


def test_preparation_preserves_fixed_splits_and_disjoint_dev_purposes(tmp_path: Path) -> None:
    source_root, _ = _source(tmp_path)
    manifest, summary = prepare_module.prepare(_args(tmp_path, source_root))

    assert set(manifest.loc[manifest["dataset_split"].eq("calibration"), "role"]) == {
        "calibration"
    }
    assert set(manifest.loc[manifest["dataset_split"].eq("own_locked"), "role"]) == {
        "locked_evaluation"
    }
    train_roles = manifest.loc[
        manifest["dataset_split"].eq("train") & manifest["sample_kind"].eq("authentic"),
        "role",
    ]
    assert {"memory_pool", "capacity_validation", "authentic_null", "supervised"} <= set(
        train_roles
    )

    development = manifest[manifest["role"].eq("development")]
    gemini = development["generator_family"].eq("gemini_flash_image")
    assert set(development.loc[gemini, "development_purpose"]) == {"generator_gate"}
    assert set(
        development.loc[
            development["generator_family"].eq("flux_1_schnell"),
            "development_purpose",
        ]
    ) == {"model_selection"}
    assert development.groupby("duplicate_group_id")["development_purpose"].nunique().max() == 1
    assert summary["dataset_revision"] == prepare_module.DATASET_REVISION
    assert summary["development_purposes"] == {
        "generator_gate": 2,
        "model_selection": 2,
    }

    unlocked = manifest[manifest["role"].ne("locked_evaluation")]
    assert set(unlocked["format"]) == {"BMP"}
    assert set(unlocked["width"]) == {16}
    assert set(unlocked["height"]) == {16}
    assert not (tmp_path / "normalized" / "images" / "own_locked").exists()


def test_preparation_fails_closed_until_dataset_terms_are_acknowledged(tmp_path: Path) -> None:
    source_root, _ = _source(tmp_path)
    with pytest.raises(PermissionError, match="acknowledge-dataset-terms"):
        prepare_module.prepare(_args(tmp_path, source_root, acknowledged=False))
