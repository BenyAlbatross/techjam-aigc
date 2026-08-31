#!/usr/bin/env python3
"""Prepare the pinned TechJam 2026 dataset for TRACE-RX-M training.

The source dataset already provides lineage-safe train/dev/calibration/locked
splits. This script preserves them, subdivides only authentic training groups
for the detector's internal stages, and writes uniformly decoded BMP inputs so
file format, dimensions, and encoded byte count cannot identify the class.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import random

import pandas as pd
from PIL import Image

from techjam_aigc.trace_rx_m.data import validate_training_manifest


DATASET_ID = "Joshyxwa/techjam2026"
DATASET_REVISION = "fd6ff453e8214359423c8ab8e44150b2660ce36c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--memory-pool-images", type=int, default=256)
    parser.add_argument("--capacity-validation-images", type=int, default=256)
    parser.add_argument("--authentic-null-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the deterministic manifest before images are available (for staged downloads).",
    )
    return parser.parse_args()


def _assign_authentic_roles(
    frame: pd.DataFrame,
    *,
    requested: dict[str, int],
    seed: int,
) -> pd.Series:
    """Assign whole, real-only content groups to authentic-only roles."""

    roles = pd.Series("supervised", index=frame.index, dtype="string")
    train = frame[frame["split"].eq("train")]
    group_labels = train.groupby("content_group_id")["label"].agg(set)
    eligible = sorted(group_labels[group_labels.map(lambda labels: labels == {"real"})].index)
    random.Random(seed).shuffle(eligible)
    cursor = 0
    assignments: dict[str, str] = {}
    for role, target_rows in requested.items():
        assigned = 0
        while cursor < len(eligible) and assigned < target_rows:
            group_id = eligible[cursor]
            cursor += 1
            assignments[group_id] = role
            assigned += int(train["content_group_id"].eq(group_id).sum())
        if assigned < target_rows:
            raise ValueError(
                f"Only assigned {assigned} of {target_rows} requested rows to {role}."
            )
    mapped = frame["content_group_id"].map(assignments)
    roles.loc[mapped.notna()] = mapped[mapped.notna()]
    roles.loc[frame["split"].eq("dev")] = "development"
    roles.loc[frame["split"].eq("calibration")] = "calibration"
    roles.loc[frame["split"].eq("own_locked")] = "locked_evaluation"
    return roles


def _normalized_path(output_root: Path, row: pd.Series) -> Path:
    return output_root / "images" / str(row["split"]) / str(row["label"]) / f"{row['asset_id']}.bmp"


def _normalize_one(source: Path, destination: Path, image_size: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with Image.open(source) as opened:
            image = opened.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            image.save(destination, format="BMP")
    with Image.open(destination) as verified:
        if verified.mode != "RGB" or verified.size != (image_size, image_size):
            raise ValueError(f"Invalid normalized image: {destination}")
    return destination.stat().st_size


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    if args.image_size < 1 or args.workers < 1:
        raise ValueError("image-size and workers must be positive.")
    labels_path = args.labels.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    labels = pd.read_csv(labels_path)
    expected_splits = {"train", "dev", "calibration", "own_locked"}
    if set(labels["split"]) != expected_splits:
        raise ValueError(f"Expected fixed splits {sorted(expected_splits)}.")
    if set(labels["label"]) != {"real", "ai_full"}:
        raise ValueError("Expected binary labels real and ai_full.")

    roles = _assign_authentic_roles(
        labels,
        requested={
            "memory_pool": args.memory_pool_images,
            "capacity_validation": args.capacity_validation_images,
            "authentic_null": args.authentic_null_images,
        },
        seed=args.seed,
    )
    destinations = [_normalized_path(output_root, row) for _, row in labels.iterrows()]
    sources = [source_root / str(value) for value in labels["image_path"]]
    if args.manifest_only:
        buffer = BytesIO()
        Image.new("RGB", (args.image_size, args.image_size)).save(buffer, format="BMP")
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
                    [args.image_size] * len(sources),
                )
            )

    def portable(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    is_real = labels["label"].eq("real")
    manifest = pd.DataFrame({
        "parent_id": labels["asset_id"].astype(str),
        "lineage_id": labels["lineage_id"].astype(str),
        "duplicate_group_id": labels["content_group_id"].astype(str),
        "role": roles,
        "phase": roles.map(lambda role: "final_confirmation" if role == "locked_evaluation" else "discovery"),
        "label": labels["label"].map({"real": "real", "ai_full": "aigc"}),
        "sample_kind": labels["label"].map({"real": "authentic", "ai_full": "native_aigc"}),
        "generator_family": labels["model_family"].where(~is_real, "authentic"),
        "generation_model": labels["generator_model"].where(~is_real, "none"),
        "source_dataset": labels["source_dataset"].astype(str),
        "authentic_subtype": labels["source_family"].astype(str),
        "local_path": [portable(path) for path in destinations],
        "source_parent_id": "",
        "width": args.image_size,
        "height": args.image_size,
        "bytes": byte_counts,
        "format": "BMP",
        "condition": "clean",
        "transform_family": "clean",
    })
    manifest = validate_training_manifest(manifest)
    summary: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "labels_sha256_note": "Dataset verification.json is retained beside labels.csv.",
        "image_size": args.image_size,
        "normalization": "RGB bilinear square resize followed by uncompressed BMP",
        "seed": args.seed,
        "rows": len(manifest),
        "roles": manifest["role"].value_counts().sort_index().to_dict(),
        "role_labels": {
            f"{role}/{label}": int(count)
            for (role, label), count in manifest.groupby(["role", "label"]).size().items()
        },
        "generator_families": manifest.loc[
            manifest["sample_kind"].eq("native_aigc"), "generator_family"
        ].value_counts().sort_index().to_dict(),
        "fixed_source_split_policy": {
            "train": "internal training roles only",
            "dev": "development/validation only",
            "calibration": "calibration only",
            "own_locked": "locked evaluation only",
        },
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
