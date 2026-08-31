from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_techjam2026_training.py"
SPEC = importlib.util.spec_from_file_location("prepare_techjam2026_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
prepare = MODULE.prepare


def test_preparation_preserves_fixed_splits_and_normalizes_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = []
    specifications = [
        ("train-real-1", "train", "real"),
        ("train-real-2", "train", "real"),
        ("train-real-3", "train", "real"),
        ("train-ai", "train", "ai_full"),
        ("dev-ai", "dev", "ai_full"),
        ("cal-real", "calibration", "real"),
        ("locked-real", "own_locked", "real"),
    ]
    for asset_id, split, label in specifications:
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
        memory_pool_images=1,
        capacity_validation_images=1,
        authentic_null_images=1,
        seed=9,
        workers=2,
        manifest_only=False,
    )

    manifest, summary = prepare(args)

    assert summary["roles"] == {
        "authentic_null": 1,
        "calibration": 1,
        "capacity_validation": 1,
        "development": 1,
        "locked_evaluation": 1,
        "memory_pool": 1,
        "supervised": 1,
    }
    assert set(manifest["format"]) == {"BMP"}
    assert set(manifest["width"]) == {16}
    assert set(manifest["height"]) == {16}
    assert manifest["bytes"].nunique() == 1
    for value in manifest["local_path"]:
        with Image.open(tmp_path / value) as image:
            assert image.mode == "RGB"
            assert image.size == (16, 16)
