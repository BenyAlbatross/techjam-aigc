#!/usr/bin/env python3
"""Prepare techjam2026_v2 for TRACE-RX-M training.

The source dataset provides lineage-safe train/val/test splits. This script
preserves every row in those splits, subdivides only authentic training groups
for the detector's internal stages, and writes uniformly decoded BMP inputs so
file format, dimensions, and encoded byte count cannot identify the class.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random

import pandas as pd
from PIL import Image

from techjam_aigc.trace_rx_m.augment import canonical_center_crop
from techjam_aigc.trace_rx_m.config import PreprocessingConfig, TraceRXMConfig
from techjam_aigc.trace_rx_m.data import validate_training_manifest


DATASET_ID = "techjam2026_v2_augmented"
DATASET_REVISION = "v2"
PREPROCESSING_VERSION = "center-crop-v1"
EXPECTED_SPLITS = {"train", "val", "test"}
REQUIRED_SOURCE_COLUMNS = {
    "asset_id",
    "content_group_id",
    "data_source_segment",
    "generator_model",
    "image_path",
    "image_origin",
    "is_transformed",
    "label",
    "lineage_id",
    "model_family",
    "pair_role",
    "source_dataset",
    "source_family",
    "split",
    "transform_family",
    "transform_variant_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/techjam2026_v2/labels.csv"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/techjam2026_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/techjam2026_v2-normalized"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/techjam2026_v2-normalized/training-manifest.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trace-rx-m-v2.json"),
        help="V2 config whose preprocessing contract is embedded in checkpoints.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--memory-images", type=int, default=256)
    parser.add_argument("--capacity-images", type=int, default=256)
    parser.add_argument("--authentic-null-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the deterministic manifest before images are available (for staged downloads).",
    )
    return parser.parse_args()


def _assign_training_pools(
    frame: pd.DataFrame,
    *,
    requested: dict[str, int],
    seed: int,
) -> pd.Series:
    """Assign whole, real-only train groups to model-internal pools."""

    pools = pd.Series("none", index=frame.index, dtype="string")
    train = frame[frame["split"].eq("train")]
    pools.loc[train.index] = "detector"
    group_labels = train.groupby("content_group_id")["label"].agg(set)
    eligible = sorted(group_labels[group_labels.map(lambda labels: labels == {"real"})].index)
    random.Random(seed).shuffle(eligible)
    cursor = 0
    assignments: dict[str, str] = {}
    for pool, target_rows in requested.items():
        assigned = 0
        while cursor < len(eligible) and assigned < target_rows:
            group_id = eligible[cursor]
            cursor += 1
            assignments[group_id] = pool
            assigned += int(train["content_group_id"].eq(group_id).sum())
        if assigned < target_rows:
            raise ValueError(
                f"Only assigned {assigned} of {target_rows} requested rows to {pool}."
            )
    mapped = frame["content_group_id"].map(assignments)
    pools.loc[mapped.notna()] = mapped[mapped.notna()]
    return pools


def _normalized_path(output_root: Path, row: pd.Series, version: str) -> Path:
    filename = f"{row['asset_id']}.{version}.bmp"
    return output_root / "images" / str(row["split"]) / str(row["label"]) / filename


def _normalize_one(
    source: Path,
    destination: Path,
    preprocessing: PreprocessingConfig,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with Image.open(source) as opened:
            image = canonical_center_crop(
                opened,
                image_size=preprocessing.image_size,
                max_short_side=preprocessing.max_short_side,
            )
            image.save(destination, format="BMP")
    with Image.open(destination) as verified:
        if verified.mode != "RGB" or verified.size != (
            preprocessing.image_size,
            preprocessing.image_size,
        ):
            raise ValueError(f"Invalid normalized image: {destination}")
    return destination.stat().st_size


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _endpoint_metadata(labels: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Map v2's NIL sentinel to TRACE-RX-M's explicit clean endpoint."""

    families = labels["transform_family"].astype("string").fillna("NIL").str.strip()
    variants = labels["transform_variant_id"].astype("string").fillna("NIL").str.strip()
    clean = ~labels["is_transformed"].astype(bool)
    invalid_clean = clean & families.str.casefold().ne("nil")
    invalid_transformed = ~clean & families.str.casefold().eq("nil")
    if invalid_clean.any() or invalid_transformed.any():
        raise ValueError("is_transformed and transform_family disagree in labels.csv.")
    condition = variants.where(~clean, "clean")
    transform_family = families.where(~clean, "clean")
    return condition, transform_family


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    if args.image_size < 1 or args.workers < 1:
        raise ValueError("image-size and workers must be positive.")
    config_path = getattr(args, "config", None)
    if config_path is None:
        preprocessing = PreprocessingConfig(image_size=args.image_size)
    else:
        config = TraceRXMConfig.load(Path(config_path))
        preprocessing = config.preprocessing
        if args.image_size != preprocessing.image_size:
            raise ValueError(
                "--image-size must match preprocessing.image_size in the v2 config."
            )
    labels_path = args.labels.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    labels = pd.read_csv(labels_path, low_memory=False)
    missing = REQUIRED_SOURCE_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"techjam2026_v2 labels are missing fields: {sorted(missing)}")
    if set(labels["split"]) != EXPECTED_SPLITS:
        raise ValueError(f"Expected fixed splits {sorted(EXPECTED_SPLITS)}.")
    if set(labels["label"]) != {"real", "ai_full"}:
        raise ValueError("Expected binary labels real and ai_full.")

    training_pools = _assign_training_pools(
        labels,
        requested={
            "memory": args.memory_images,
            "capacity": args.capacity_images,
            "authentic_null": args.authentic_null_images,
        },
        seed=args.seed,
    )
    destinations = [
        _normalized_path(output_root, row, preprocessing.version)
        for _, row in labels.iterrows()
    ]
    sources = [source_root / str(value) for value in labels["image_path"]]
    if args.manifest_only:
        buffer = BytesIO()
        Image.new(
            "RGB",
            (preprocessing.image_size, preprocessing.image_size),
        ).save(buffer, format="BMP")
        byte_counts = [buffer.tell()] * len(sources)
    else:
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} source images are missing; first missing path: {missing[0]}"
            )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            byte_counts = list(
                executor.map(
                    _normalize_one,
                    sources,
                    destinations,
                    [preprocessing] * len(sources),
                )
            )

    def portable(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    is_real = labels["label"].eq("real")
    condition, transform_family = _endpoint_metadata(labels)
    manifest = pd.DataFrame({
        "parent_id": labels["asset_id"].astype(str),
        "lineage_id": labels["lineage_id"].astype(str),
        "duplicate_group_id": labels["content_group_id"].astype(str),
        "split": labels["split"].astype(str),
        "training_pool": training_pools,
        "label": labels["label"].map({"real": "real", "ai_full": "aigc"}),
        "sample_kind": labels["label"].map({"real": "authentic", "ai_full": "native_aigc"}),
        "generator_family": labels["model_family"].where(~is_real, "authentic"),
        "generation_model": labels["generator_model"].where(~is_real, "none"),
        "source_dataset": labels["source_dataset"].astype(str),
        "authentic_subtype": labels["source_family"].astype(str),
        "image_origin": labels["image_origin"].astype(str),
        "pair_role": labels["pair_role"].astype(str),
        "data_source_segment": labels["data_source_segment"].astype(str),
        "is_transformed": labels["is_transformed"].astype(bool),
        "transform_variant_id": labels["transform_variant_id"].astype(str),
        "local_path": [portable(path) for path in destinations],
        "source_parent_id": "",
        "width": preprocessing.image_size,
        "height": preprocessing.image_size,
        "bytes": byte_counts,
        "format": "BMP",
        "condition": condition,
        "transform_family": transform_family,
    })
    manifest = validate_training_manifest(manifest)
    summary: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "labels_sha256": _file_sha256(labels_path),
        "preprocessing_version": preprocessing.version,
        "preprocessing": asdict(preprocessing),
        "source_rows_retained": len(labels),
        "splits": manifest["split"].value_counts().sort_index().to_dict(),
        "image_origins": manifest["image_origin"].value_counts().sort_index().to_dict(),
        "image_size": preprocessing.image_size,
        "normalization": (
            "RGB bicubic 512px short-side limit, center crop with zero-padding, "
            "then uncompressed BMP"
        ),
        "seed": args.seed,
        "rows": len(manifest),
        "training_pools": manifest["training_pool"].value_counts().sort_index().to_dict(),
        "training_pool_labels": {
            f"{pool}/{label}": int(count)
            for (pool, label), count in manifest.groupby(["training_pool", "label"]).size().items()
        },
        "generator_families": manifest.loc[
            manifest["sample_kind"].eq("native_aigc"), "generator_family"
        ].value_counts().sort_index().to_dict(),
        "split_policy": {
            "train": "model fitting and train-only internal pools",
            "val": "validation only",
            "test": "test only",
        },
        "transform_families": manifest["transform_family"].value_counts().sort_index().to_dict(),
    }
    return manifest, summary


def main() -> None:
    args = parse_args()
    manifest, summary = prepare(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
