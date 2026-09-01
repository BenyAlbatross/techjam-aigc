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
    canonical_center_crop,
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
    split: str = "train",
    training_pool: str = "detector",
    label: str = "real",
    sample_kind: str = "authentic",
    lineage_id: str | None = None,
    generator_family: str = "authentic",
    generation_model: str = "none",
    source_parent_id: str = "",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "lineage_id": lineage_id or parent_id,
        "split": split,
        "training_pool": training_pool,
        "label": label,
        "sample_kind": sample_kind,
        "generator_family": generator_family,
        "generation_model": generation_model,
        "source_dataset": "camera-source",
        "local_path": f"images/{parent_id}.png",
        "source_parent_id": source_parent_id,
    }


def _detector_frame() -> pd.DataFrame:
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


def test_manifest_rejects_demo_rows_partition_leakage_and_bad_dda_pair() -> None:
    demo = _row("demo", label="aigc", sample_kind="native_aigc")
    demo.update(generation_model="DALL-E Advanced", source_dataset="COCO val2017")
    with pytest.raises(ValueError, match="demonstration-only"):
        validate_training_manifest(pd.DataFrame([demo]))

    leaked = pd.DataFrame(
        [
            _row("one", lineage_id="shared"),
            _row("two", split="val", training_pool="none", lineage_id="shared"),
        ]
    )
    with pytest.raises(ValueError, match="lineage_id crosses"):
        validate_training_manifest(leaked)

    paired = _detector_frame()
    first_dda = paired.index[paired["sample_kind"].eq("dda")][0]
    paired.loc[first_dda, "source_parent_id"] = "missing"
    with pytest.raises(ValueError, match="source_parent_id rows are missing"):
        validate_training_manifest(paired)


def test_only_train_val_test_and_training_pools_are_strict() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset splits"):
        validate_training_manifest(pd.DataFrame([_row("old", split="holdout")]))
    with pytest.raises(ValueError, match="val/test rows require training_pool=none"):
        validate_training_manifest(pd.DataFrame([_row("val", split="val")]))
    fake_memory = pd.DataFrame(
        [
            _row(
                "fake-memory",
                training_pool="memory",
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


def test_canonical_center_crop_limits_large_images_with_bicubic_resize() -> None:
    pixels = np.zeros((600, 900, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(900, dtype=np.uint16) % 256
    pixels[:, :, 1] = np.arange(600, dtype=np.uint16)[:, None] % 256
    image = Image.fromarray(pixels)

    resized = image.resize((768, 512), Image.Resampling.BICUBIC)
    expected = np.asarray(resized.crop((272, 144, 496, 368)))
    actual = np.asarray(canonical_center_crop(image))

    np.testing.assert_array_equal(actual, expected)


def test_canonical_center_crop_handles_large_portrait_and_odd_native_sizes() -> None:
    portrait_pixels = np.zeros((901, 600, 3), dtype=np.uint8)
    portrait_pixels[:, :, 0] = np.arange(600, dtype=np.uint16) % 256
    portrait_pixels[:, :, 1] = np.arange(901, dtype=np.uint16)[:, None] % 256
    portrait = Image.fromarray(portrait_pixels)
    resized = portrait.resize((512, 769), Image.Resampling.BICUBIC)
    expected_portrait = np.asarray(resized.crop((144, 272, 368, 496)))
    np.testing.assert_array_equal(
        np.asarray(canonical_center_crop(portrait)), expected_portrait
    )

    odd_pixels = np.arange(227 * 225 * 3, dtype=np.uint8).reshape(227, 225, 3)
    expected_odd = odd_pixels[2:226, 0:224]
    np.testing.assert_array_equal(
        np.asarray(canonical_center_crop(Image.fromarray(odd_pixels))), expected_odd
    )


def test_canonical_center_crop_keeps_native_scale_at_or_below_512() -> None:
    pixels = np.zeros((300, 400, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(400, dtype=np.uint16) % 256
    pixels[:, :, 1] = np.arange(300, dtype=np.uint16)[:, None] % 256

    actual = np.asarray(canonical_center_crop(Image.fromarray(pixels)))

    np.testing.assert_array_equal(actual, pixels[38:262, 88:312])


def test_canonical_center_crop_zero_pads_undersized_dimensions() -> None:
    image = Image.new("RGB", (4, 2), (255, 0, 0))

    actual = np.asarray(canonical_center_crop(image, image_size=6))

    expected = np.zeros((6, 6, 3), dtype=np.uint8)
    expected[2:4, 1:5, 0] = 255
    np.testing.assert_array_equal(actual, expected)


def test_batch_sampler_has_exact_mix_pairs_and_generator_rotation() -> None:
    frame = validate_training_manifest(_detector_frame())
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


def test_batch_sampler_supports_native_only_positive_data() -> None:
    frame = _detector_frame()
    frame = frame[frame["sample_kind"].ne("dda")]
    paired_sources = frame["parent_id"].str.startswith("pair-real-")
    frame = validate_training_manifest(frame[~paired_sources].reset_index(drop=True))
    sampler = BalancedTraceBatchSampler(frame, batch_size=10, seed=4, batches_per_epoch=2)

    for batch in sampler:
        rows = frame.iloc[batch]
        assert rows["sample_kind"].value_counts().to_dict() == {
            "authentic": 5,
            "native_aigc": 5,
        }
        uniform_weights = pd.Series(1.0, index=rows.index)
        class_weight_mass = uniform_weights.groupby(rows["target"]).sum()
        assert class_weight_mass.to_dict() == {0: 5.0, 1: 5.0}
