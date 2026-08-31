#!/usr/bin/env python3
"""Prepare the pinned TechJam 2026 dataset for TRACE-RX-Parallel training.

The source dataset already provides lineage-safe train/dev/calibration/locked
splits. This script preserves them, subdivides only authentic training groups
for the detector's internal stages, and writes uniformly decoded BMP inputs so
file format, dimensions, and encoded byte count cannot identify the class.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
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
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/techjam2026"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/techjam2026-parallel"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/techjam2026-parallel/training-manifest.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--memory-pool-images", type=int, default=256)
    parser.add_argument("--capacity-validation-images", type=int, default=256)
    parser.add_argument("--authentic-null-images", type=int, default=512)
    parser.add_argument("--generator-gate-real-images", type=int, default=512)
    parser.add_argument("--held-out-generator-family", default="gemini_flash_image")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--acknowledge-dataset-terms",
        action="store_true",
        help=(
            "Confirm that the dataset card and row-level licence fields were reviewed "
            "for the intended use. Preparation fails closed without this flag."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use an existing pinned snapshot under --source-root.",
    )
    parser.add_argument(
        "--include-locked-images",
        action="store_true",
        help="Download locked images. They remain forbidden from all training stages.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the deterministic manifest before images are available (for staged downloads).",
    )
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> None:
    """Download the pinned public snapshot without locked pixels by default."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - declared training dependency
        raise RuntimeError("Install dependencies with `uv sync --group train`.") from error
    patterns = ["labels.csv", "verification.json", "README.md"]
    if not args.manifest_only:
        patterns.extend(("images/train/**", "images/dev/**", "images/calibration/**"))
        if args.include_locked_images:
            patterns.append("images/own_locked/**")
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=args.source_root,
        allow_patterns=patterns,
        max_workers=args.workers,
    )


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
    eligible = set(group_labels[group_labels.map(lambda labels: labels == {"real"})].index)
    group_sources = train.groupby("content_group_id")["source_family"].first()
    groups_by_source: dict[str, list[object]] = {}
    for group_id in eligible:
        groups_by_source.setdefault(str(group_sources.loc[group_id]), []).append(group_id)
    for source, group_ids in groups_by_source.items():
        random.Random(f"{seed}:{source}").shuffle(group_ids)
    eligible = []
    while any(groups_by_source.values()):
        for source in sorted(groups_by_source):
            if groups_by_source[source]:
                eligible.append(groups_by_source[source].pop())
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


def _assign_development_purposes(
    frame: pd.DataFrame,
    *,
    held_out_generator_family: str,
    requested_gate_reals: int,
    seed: int,
) -> pd.Series:
    """Split dev groups into recurring selection and one-time generator gate sets."""

    purposes = pd.Series("not_development", index=frame.index, dtype="string")
    dev = frame[frame["split"].eq("dev")]
    by_group = dev.groupby("content_group_id", sort=True)
    heldout_groups = {
        str(group_id)
        for group_id, group in by_group
        if group["model_family"].astype(str).eq(held_out_generator_family).any()
    }
    purposes.loc[
        frame["split"].eq("dev")
        & frame["content_group_id"].astype(str).isin(heldout_groups)
    ] = "generator_gate"

    group_labels = dev.groupby("content_group_id")["label"].agg(set)
    real_only = sorted(group_labels[group_labels.map(lambda values: values == {"real"})].index)
    random.Random(f"{seed}:generator-gate").shuffle(real_only)
    assigned = 0
    selected_real_groups: set[str] = set()
    for group_id in real_only:
        if assigned >= requested_gate_reals:
            break
        selected_real_groups.add(str(group_id))
        assigned += int(dev["content_group_id"].eq(group_id).sum())
    if assigned < requested_gate_reals:
        raise ValueError(
            f"Only assigned {assigned} of {requested_gate_reals} requested gate reals."
        )
    purposes.loc[
        frame["split"].eq("dev")
        & frame["content_group_id"].astype(str).isin(selected_real_groups)
    ] = "generator_gate"
    purposes.loc[frame["split"].eq("dev") & purposes.eq("not_development")] = (
        "model_selection"
    )
    dev_purposes = purposes[frame["split"].eq("dev")]
    if set(dev_purposes) != {"model_selection", "generator_gate"}:
        raise ValueError("Development rows must cover both selection and generator-gate sets.")
    conflicts = (
        frame[frame["split"].eq("dev")]
        .assign(development_purpose=dev_purposes)
        .groupby("content_group_id")["development_purpose"]
        .nunique()
    )
    if (conflicts > 1).any():  # pragma: no cover - assignments are group based
        raise RuntimeError("Development content groups cross purposes.")
    return purposes


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
    if not args.acknowledge_dataset_terms:
        raise PermissionError(
            "Review the dataset card and row-level licence fields, then pass "
            "--acknowledge-dataset-terms."
        )
    source_root = args.source_root.resolve()
    labels_path = (args.labels or source_root / "labels.csv").resolve()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    labels = pd.read_csv(labels_path)
    verification_path = source_root / "verification.json"
    if not verification_path.is_file():
        raise FileNotFoundError(f"Missing pinned release verification: {verification_path}")
    verification = json.loads(verification_path.read_text())
    if verification.get("asset_count") != len(labels):
        raise ValueError("labels.csv does not match verification.json asset_count.")
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
    development_purposes = _assign_development_purposes(
        labels,
        held_out_generator_family=args.held_out_generator_family,
        requested_gate_reals=args.generator_gate_real_images,
        seed=args.seed,
    )
    destinations = [_normalized_path(output_root, row) for _, row in labels.iterrows()]
    sources = [source_root / str(value) for value in labels["image_path"]]
    usable = labels["split"].ne("own_locked").to_numpy()
    if args.manifest_only:
        buffer = BytesIO()
        Image.new("RGB", (args.image_size, args.image_size)).save(buffer, format="BMP")
        byte_counts = labels["file_size_bytes"].astype(int).tolist()
        for index, use_row in enumerate(usable):
            if use_row:
                byte_counts[index] = buffer.tell()
    else:
        missing = [
            str(source)
            for source, destination, use_row in zip(
                sources,
                destinations,
                usable,
                strict=True,
            )
            if use_row and not source.is_file() and not destination.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} source images are missing; first missing path: {missing[0]}"
            )
        byte_counts = labels["file_size_bytes"].astype(int).tolist()
        selected_indices = [index for index, use_row in enumerate(usable) if use_row]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            normalized_counts = list(
                executor.map(
                    _normalize_one,
                    [sources[index] for index in selected_indices],
                    [destinations[index] for index in selected_indices],
                    [args.image_size] * len(selected_indices),
                )
            )
        for index, byte_count in zip(selected_indices, normalized_counts, strict=True):
            byte_counts[index] = byte_count

    def portable(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    is_real = labels["label"].eq("real")
    local_paths = [
        portable(destination if use_row else source)
        for destination, source, use_row in zip(destinations, sources, usable, strict=True)
    ]
    widths = labels["width"].astype(int).copy()
    heights = labels["height"].astype(int).copy()
    formats = labels["file_format"].astype(str).copy()
    widths.loc[usable] = args.image_size
    heights.loc[usable] = args.image_size
    formats.loc[usable] = "BMP"
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
        "local_path": local_paths,
        "source_parent_id": "",
        "width": widths,
        "height": heights,
        "bytes": byte_counts,
        "format": formats,
        "condition": "clean",
        "transform_family": "clean",
        "development_purpose": development_purposes,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": labels["split"].astype(str),
        "source_sha256": labels["sha256"].astype(str),
        "licence": labels["licence"].astype(str),
    })
    manifest = validate_training_manifest(manifest)
    labels_sha256 = sha256(labels_path.read_bytes()).hexdigest()
    pending_licence_rows = int(
        labels["licence"].astype(str).str.contains("pending", case=False).sum()
    )
    summary: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "release_id": verification.get("release_id"),
        "labels_sha256": labels_sha256,
        "dataset_terms_acknowledged": True,
        "pending_licence_metadata_rows": pending_licence_rows,
        "licence_review_note": (
            "The source is public and row-level terms are retained; re-review rows "
            "whose licence metadata contains 'pending' before competition submission."
        ),
        "image_size": args.image_size,
        "normalization": "RGB bilinear square resize followed by uncompressed BMP",
        "seed": args.seed,
        "rows": len(manifest),
        "roles": manifest["role"].value_counts().sort_index().to_dict(),
        "development_purposes": manifest.loc[
            manifest["role"].eq("development"), "development_purpose"
        ].value_counts().sort_index().to_dict(),
        "role_labels": {
            f"{role}/{label}": int(count)
            for (role, label), count in manifest.groupby(["role", "label"]).size().items()
        },
        "authentic_role_sources": {
            role: manifest.loc[
                manifest["role"].eq(role), "authentic_subtype"
            ].value_counts().sort_index().to_dict()
            for role in ("memory_pool", "capacity_validation", "authentic_null")
        },
        "generator_families": manifest.loc[
            manifest["sample_kind"].eq("native_aigc"), "generator_family"
        ].value_counts().sort_index().to_dict(),
        "fixed_source_split_policy": {
            "train": "internal training roles only",
            "dev": "group-disjoint recurring model selection and one-time Gemini gate",
            "calibration": "calibration only",
            "own_locked": "locked evaluation only; pixels excluded from default download",
        },
    }
    return manifest, summary


def main() -> None:
    args = parse_args()
    if not args.no_download:
        download_dataset(args)
    manifest, summary = prepare(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
