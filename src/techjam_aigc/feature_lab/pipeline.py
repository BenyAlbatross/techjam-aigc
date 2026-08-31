"""Feature extraction and cache orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
from PIL import Image

from .data import coverage_table, load_binary_index
from .features import extract_features
from .registry import (
    DEFAULT_FEATURE_PROFILE,
    feature_schema_sha256,
    registry_frame,
)
from .transforms import (
    TransformSpec,
    analysis_view,
    apply_transform,
    get_transform_specs,
    transform_frame,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260830
    views: tuple[str, ...] = ("native_capped", "canonical_128")
    transform_profile: str = "core"
    conditions: tuple[str, ...] | None = None
    feature_profile: str = DEFAULT_FEATURE_PROFILE
    evaluate_final_confirmation: bool = False
    min_group_images: int = 20
    bootstrap_repetitions: int = 200


def _resolved_transform_specs(config: ExperimentConfig) -> tuple[TransformSpec, ...]:
    specs = get_transform_specs(config.transform_profile)
    if config.conditions is None:
        return specs
    lookup = {spec.name: spec for spec in specs}
    missing = set(config.conditions) - set(lookup)
    if missing:
        raise KeyError(
            f"Conditions are not in profile {config.transform_profile!r}: {sorted(missing)}"
        )
    return tuple(lookup[name] for name in config.conditions)


def _extract_parent(task: tuple[dict[str, Any], str, ExperimentConfig]) -> list[dict[str, Any]]:
    row, repo_root_text, config = task
    repo_root = Path(repo_root_text)
    path = repo_root / str(row["local_path"])
    records: list[dict[str, Any]] = []
    transform_specs = _resolved_transform_specs(config)
    transform_lookup = {spec.name: spec for spec in transform_specs}
    transform_metadata = transform_frame(
        config.transform_profile,
        conditions=tuple(transform_lookup),
    ).set_index("name")
    with Image.open(path) as opened:
        source = opened.convert("RGB")
    for condition in transform_lookup:
        transformed = apply_transform(
            source,
            condition,
            parent_id=str(row["parent_id"]),
            base_seed=config.seed,
        )
        transform_spec = transform_lookup[condition]
        for view in config.views:
            prepared = analysis_view(transformed, view)
            values = extract_features(prepared, row, profile=config.feature_profile)
            optional_metadata = {
                key: row[key]
                for key in (
                    "chronological_window",
                    "generator_window",
                    "release_window",
                    "time_window",
                    "window",
                    "prompt_id",
                    "source_id",
                    "revision",
                )
                if key in row and pd.notna(row[key])
            }
            records.append(
                {
                    "parent_id": row["parent_id"],
                    "local_path": row["local_path"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "phase": row["phase"],
                    "label": row["label"],
                    "binary_label": row["binary_label"],
                    "target": int(row["target"]),
                    "generator_family": row["generator_family"],
                    "generation_model": row["generation_model"],
                    "source_dataset": row["source_dataset"],
                    "format": row["format"],
                    "native_width": int(row["width"]),
                    "native_height": int(row["height"]),
                    "condition": condition,
                    "transform_family": transform_spec.family,
                    "official_transform": transform_spec.official,
                    "transform_profile": config.transform_profile,
                    "transform_design": transform_spec.design,
                    "feature_profile": config.feature_profile,
                    "transform_step_count": int(transform_metadata.loc[condition, "step_count"]),
                    "transform_order": transform_metadata.loc[condition, "ordered_operations"],
                    "transform_recipe_sha256": transform_metadata.loc[condition, "recipe_sha256"],
                    "view": view,
                    **optional_metadata,
                    **values,
                }
            )
    return records


def extract_feature_table(
    binary_index: pd.DataFrame,
    repo_root: Path,
    config: ExperimentConfig,
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Extract all parent/condition/view rows, optionally across processes."""

    tasks = [(row, str(repo_root), config) for row in binary_index.to_dict("records")]
    nested: list[list[dict[str, Any]]]
    if workers <= 1:
        nested = [_extract_parent(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_extract_parent, tasks, chunksize=1))
    return pd.DataFrame(record for parent_records in nested for record in parent_records)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_extraction_cache(
    output_dir: Path,
    feature_table: pd.DataFrame,
    binary_index: pd.DataFrame,
    index_path: Path,
    config: ExperimentConfig,
    *,
    final_confirmation_available_parents: int = 0,
    final_confirmation_evaluated_at: str | None = None,
    extraction_wall_seconds: float | None = None,
    workers: int = 1,
    eligible_binary_parent_images: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.csv.gz"
    feature_table.to_csv(feature_path, index=False, compression="gzip")
    resolved_feature_registry = registry_frame(config.feature_profile)
    resolved_feature_schema_sha256 = feature_schema_sha256(config.feature_profile)
    resolved_feature_registry.assign(
        feature_profile=config.feature_profile,
        feature_schema_sha256=resolved_feature_schema_sha256,
    ).to_csv(output_dir / "feature_registry.csv", index=False)
    resolved_specs = _resolved_transform_specs(config)
    transform_frame(
        config.transform_profile,
        conditions=tuple(spec.name for spec in resolved_specs),
    ).to_csv(output_dir / "transform_registry.csv", index=False)
    coverage_table(binary_index).to_csv(output_dir / "coverage.csv", index=False)

    metadata = {
        "schema_version": 2,
        "config": {
            **asdict(config),
            "resolved_conditions": [spec.name for spec in resolved_specs],
            "resolved_condition_count": len(resolved_specs),
        },
        "input_index": str(index_path),
        "feature_schema": {
            "profile": config.feature_profile,
            "sha256": resolved_feature_schema_sha256,
            "feature_count": int(len(resolved_feature_registry)),
            "names": resolved_feature_registry["name"].tolist(),
        },
        "input_index_sha256": _sha256(index_path),
        "binary_parent_images": int(len(binary_index)),
        "eligible_binary_parent_images": int(
            eligible_binary_parent_images
            if eligible_binary_parent_images is not None
            else len(binary_index)
        ),
        "withheld_final_confirmation_images": int(
            max(0, (eligible_binary_parent_images or len(binary_index)) - len(binary_index))
        ),
        "excluded_nonbinary_images": int(
            len(pd.read_csv(index_path))
            - (
                eligible_binary_parent_images
                if eligible_binary_parent_images is not None
                else len(binary_index)
            )
        ),
        "feature_rows": int(len(feature_table)),
        "features": int(len(resolved_feature_registry)),
        "extraction_cost": {
            "wall_seconds": extraction_wall_seconds,
            "workers": int(workers),
            "seconds_per_parent": (
                extraction_wall_seconds / len(binary_index)
                if extraction_wall_seconds is not None and len(binary_index)
                else None
            ),
            "rows_per_second": (
                len(feature_table) / extraction_wall_seconds
                if extraction_wall_seconds is not None and extraction_wall_seconds > 0
                else None
            ),
        },
        "semantic_control": {
            "status": "not_run",
            "reason": "No generic pretrained backbone or cached weights are provisioned; the laboratory does not silently download model weights.",
            "recommended_optional_control": "Frozen public DINOv2/CLIP embedding plus a discovery-only linear probe.",
        },
        "final_confirmation": {
            "available_parents": int(final_confirmation_available_parents),
            "evaluated": final_confirmation_evaluated_at is not None,
            "evaluated_at": final_confirmation_evaluated_at,
            "policy": "Explicit --evaluate-final-confirmation opt-in is required; otherwise sealed rows are withheld from extraction.",
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_extraction(
    repo_root: Path,
    output_dir: Path,
    *,
    index_path: Path | None = None,
    config: ExperimentConfig | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = config or ExperimentConfig()
    resolved_index_path = index_path or repo_root / "data/samples/index.csv"
    if not resolved_index_path.is_absolute():
        resolved_index_path = repo_root / resolved_index_path
    resolved_index_path = resolved_index_path.resolve()
    _assert_planned_index_acquisition_allowed(resolved_index_path)
    binary_index = load_binary_index(resolved_index_path)
    eligible_binary_parent_images = len(binary_index)
    final_mask = binary_index["phase"].eq("final_confirmation")
    final_count = int(final_mask.sum())
    if config.evaluate_final_confirmation:
        if final_count == 0:
            raise ValueError(
                "--evaluate-final-confirmation was requested, but the index contains no "
                "final_confirmation rows."
            )
        final_receipt_path = resolved_index_path.with_suffix(
            resolved_index_path.suffix + ".final-confirmation-evaluated.json"
        )
        if final_receipt_path.exists():
            receipt = json.loads(final_receipt_path.read_text(encoding="utf-8"))
            raise PermissionError(
                "This final-confirmation index already has a first-evaluation receipt "
                f"from {receipt.get('first_evaluated_at', 'an unknown time')}: "
                f"{final_receipt_path}"
            )
        final_evaluated_at = datetime.now(timezone.utc).isoformat()
    else:
        final_receipt_path = None
        binary_index = binary_index[~final_mask].reset_index(drop=True)
        final_evaluated_at = None
        if final_count and "confirmation" not in set(binary_index["phase"]):
            raise PermissionError(
                "The only evaluation rows are sealed final_confirmation rows. Keep them "
                "untouched, or rerun once with --evaluate-final-confirmation after freezing "
                "features, transforms, thresholds, fusion, and hyperparameters."
            )
    extraction_started = time.perf_counter()
    features = extract_feature_table(binary_index, repo_root, config, workers=workers)
    extraction_wall_seconds = time.perf_counter() - extraction_started
    metadata = write_extraction_cache(
        output_dir,
        features,
        binary_index,
        resolved_index_path,
        config,
        final_confirmation_available_parents=final_count,
        final_confirmation_evaluated_at=final_evaluated_at,
        extraction_wall_seconds=extraction_wall_seconds,
        workers=workers,
        eligible_binary_parent_images=eligible_binary_parent_images,
    )
    if final_receipt_path is not None:
        receipt = {
            "schema_version": 1,
            "first_evaluated_at": final_evaluated_at,
            "input_index": str(resolved_index_path),
            "input_index_sha256": metadata["input_index_sha256"],
            "output_dir": str(output_dir.resolve()),
            "feature_schema_sha256": metadata["feature_schema"]["sha256"],
            "transform_conditions": metadata["config"]["resolved_conditions"],
        }
        final_receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return features, binary_index, metadata


def _assert_planned_index_acquisition_allowed(index_path: Path) -> None:
    """Require the planner's positive license audit before reading selected images."""

    columns = set(pd.read_csv(index_path, nrows=0).columns)
    if "source_id" not in columns:
        return
    audit_path = index_path.parent / "audit.json"
    if not audit_path.is_file():
        raise PermissionError(
            f"Planned expansion index requires its sibling license audit: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("acquisition_allowed") is not True:
        raise PermissionError(
            "Feature extraction is blocked because the expansion source/license audit "
            "is not fully allowlisted."
        )
    audited_hash = audit.get("selection_sha256")
    actual_hash = _sha256(index_path)
    if audited_hash != actual_hash:
        raise PermissionError(
            "Feature extraction is blocked because selection.csv does not match "
            "the SHA-256 recorded by its license audit."
        )
