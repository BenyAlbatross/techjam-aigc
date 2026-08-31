from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from techjam_aigc.trace_rx_m.augment import (
    SymmetricTransformSampler,
    canonical_preprocess,
)
from techjam_aigc.trace_rx_m.data import (
    BalancedTraceBatchSampler,
    TraceRXMDataset,
    load_training_manifest,
    validate_training_manifest,
)


def _row(
    parent_id: str,
    *,
    role: str = "supervised",
    label: str = "real",
    sample_kind: str = "authentic",
    lineage_id: str | None = None,
    generator_family: str = "authentic",
    generation_model: str = "none",
    source_parent_id: str = "",
    phase: str = "discovery",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "lineage_id": lineage_id or parent_id,
        "role": role,
        "phase": phase,
        "label": label,
        "sample_kind": sample_kind,
        "generator_family": generator_family,
        "generation_model": generation_model,
        "source_dataset": "camera-source",
        "local_path": f"images/{parent_id}.png",
        "source_parent_id": source_parent_id,
    }


def _supervised_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(_row(f"real-{index}"))
    for family in ("diffusion", "dit", "gan"):
        for index in range(3):
            rows.append(
                _row(
                    f"native-{family}-{index}",
                    label="aigc",
                    sample_kind="native_aigc",
                    generator_family=family,
                    generation_model=f"{family}-{index}",
                )
            )
    for index, family in enumerate(("diffusion", "dit")):
        source = f"pair-real-{index}"
        lineage = f"pair-{index}"
        rows.append(_row(source, lineage_id=lineage))
        rows.append(
            _row(
                f"dda-{index}",
                label="dda_aligned",
                sample_kind="dda",
                lineage_id=lineage,
                generator_family=family,
                generation_model="aligned-vae",
                source_parent_id=source,
            )
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("label", ["AI-edited", "partial_composite", "inpainted"])
def test_manifest_rejects_out_of_scope_labels(label: str) -> None:
    with pytest.raises(ValueError, match="outside the binary task"):
        validate_training_manifest(pd.DataFrame([_row("bad", label=label)]))


def test_manifest_rejects_demo_rows_role_leakage_and_bad_dda_pair() -> None:
    demo = _row("demo", label="aigc", sample_kind="native_aigc")
    demo.update(generation_model="DALL-E Advanced", source_dataset="COCO val2017")
    with pytest.raises(ValueError, match="demonstration-only"):
        validate_training_manifest(pd.DataFrame([demo]))

    leaked = pd.DataFrame(
        [
            _row("one", role="supervised", lineage_id="shared"),
            _row("two", role="calibration", lineage_id="shared"),
        ]
    )
    with pytest.raises(ValueError, match="lineage_id crosses"):
        validate_training_manifest(leaked)

    paired = _supervised_frame()
    first_dda = paired.index[paired["sample_kind"].eq("dda")][0]
    paired.loc[first_dda, "source_parent_id"] = "missing"
    with pytest.raises(ValueError, match="source_parent_id rows are missing"):
        validate_training_manifest(paired)


def test_final_confirmation_is_locked_and_authentic_roles_are_strict() -> None:
    unlocked = pd.DataFrame([_row("final", phase="final_confirmation")])
    with pytest.raises(ValueError, match="correspond exactly"):
        validate_training_manifest(unlocked)
    fake_memory = pd.DataFrame(
        [
            _row(
                "fake-memory",
                role="memory_pool",
                label="aigc",
                sample_kind="native_aigc",
            )
        ]
    )
    with pytest.raises(ValueError, match="authentic rows only"):
        validate_training_manifest(fake_memory)


def test_source_manifest_cannot_bypass_existing_acquisition_audit(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    frame = pd.DataFrame([{**_row("real"), "source_id": "licensed-source"}])
    frame.to_csv(path, index=False)
    with pytest.raises(PermissionError, match="license audit"):
        load_training_manifest(path)
    audit = {
        "acquisition_allowed": True,
        "selection_sha256": sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    assert load_training_manifest(path)["parent_id"].tolist() == ["real"]
    audit["selection_sha256"] = "0" * 64
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not match"):
        load_training_manifest(path)


def test_symmetric_transform_is_deterministic_and_excludes_heldout_family() -> None:
    sampler = SymmetricTransformSampler(held_out_family="jpeg", base_seed=9)
    assert not any(condition.startswith("jpeg_") for condition in sampler.conditions)
    assert "resize_0.25" in sampler.conditions
    assert "noise_sigma0.10" in sampler.conditions
    assert sampler.sample_condition(parent_id="same", epoch=2) == sampler.sample_condition(
        parent_id="same", epoch=2
    )
    image = Image.new("RGB", (31, 17), "navy")
    first_name, first = sampler.apply(image, parent_id="same", epoch=2)
    second_name, second = sampler.apply(image, parent_id="same", epoch=2)
    assert first_name == second_name
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_canonical_preprocess_and_dataset_are_torch_collation_ready(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (40, 20), (10, 20, 30)).save(image_dir / "real.png")
    array = canonical_preprocess(Image.new("RGB", (40, 20), "gray"), image_size=16)
    assert array.shape == (3, 16, 16)
    assert array.dtype == np.float32
    assert array.flags.c_contiguous

    frame = pd.DataFrame([_row("real")])
    dataset = TraceRXMDataset(frame, tmp_path, image_size=16)
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 16, 16)
    assert item["target"] == 0
    assert item["condition"] == "clean"


def test_batch_sampler_has_exact_mix_pairs_and_generator_rotation() -> None:
    frame = validate_training_manifest(_supervised_frame())
    sampler = BalancedTraceBatchSampler(frame, batch_size=10, seed=4, batches_per_epoch=2)
    first_epoch = list(sampler)
    assert first_epoch == list(sampler)
    for batch in first_epoch:
        rows = frame.iloc[batch]
        assert rows["sample_kind"].value_counts().to_dict() == {
            "authentic": 5,
            "native_aigc": 4,
            "dda": 1,
        }
        dda = rows.loc[rows["sample_kind"].eq("dda")].iloc[0]
        assert dda["source_parent_id"] in set(rows["parent_id"])
        assert rows.loc[rows["sample_kind"].eq("native_aigc"), "generator_family"].nunique() == 3
    sampler.set_epoch(1)
    assert first_epoch != list(sampler)


def test_batch_sampler_supports_dataset_without_dda_rows() -> None:
    frame = _supervised_frame()
    frame = frame[frame["sample_kind"].ne("dda")]
    paired_sources = frame["parent_id"].str.startswith("pair-real-")
    frame = validate_training_manifest(frame[~paired_sources].reset_index(drop=True))
    sampler = BalancedTraceBatchSampler(
        frame,
        batch_size=10,
        seed=4,
        batches_per_epoch=2,
    )

    for batch in sampler:
        rows = frame.iloc[batch]
        assert rows["sample_kind"].value_counts().to_dict() == {
            "authentic": 5,
            "native_aigc": 5,
        }
