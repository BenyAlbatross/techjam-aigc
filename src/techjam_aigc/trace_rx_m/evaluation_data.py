"""Dataset adapters and lazy transformed endpoints for detector evaluation.

Evaluation datasets intentionally use a smaller contract than training
manifests.  A JSON specification maps an external dataset into one canonical
row per source image; :class:`TransformEndpointDataset` then expands each row
into deterministic clean and transformed endpoints without writing image
copies to disk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from techjam_aigc.feature_lab.transforms import (
    TRANSFORM_PROFILES,
    TransformSpec,
    apply_transform,
    get_transform_specs,
    transform_frame,
)

from .augment import canonical_preprocess


REQUIRED_CANONICAL_COLUMNS = {
    "dataset_id",
    "parent_id",
    "image_path",
    "resolved_path",
    "target",
}
OPTIONAL_CANONICAL_COLUMNS = (
    "lineage_id",
    "generator_family",
    "generation_model",
    "source_dataset",
    "authentic_subtype",
)
DEFAULT_IMAGE_EXTENSIONS = (
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Validated dataset-adapter configuration loaded from JSON."""

    path: Path
    dataset_id: str
    adapter: str
    values: Mapping[str, Any]


DatasetAdapter = Callable[[DatasetSpec, Path], pd.DataFrame]
_DATASET_ADAPTERS: dict[str, DatasetAdapter] = {}


def register_dataset_adapter(name: str) -> Callable[[DatasetAdapter], DatasetAdapter]:
    """Register an adapter while keeping the public plug-in boundary explicit."""

    normalized = name.strip().casefold()
    if not normalized:
        raise ValueError("Dataset adapter names cannot be empty.")

    def decorator(adapter: DatasetAdapter) -> DatasetAdapter:
        if normalized in _DATASET_ADAPTERS:
            raise ValueError(f"Dataset adapter {normalized!r} is already registered.")
        _DATASET_ADAPTERS[normalized] = adapter
        return adapter

    return decorator


def available_dataset_adapters() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_ADAPTERS))


def _repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else repo_root / path).resolve()


def load_dataset_spec(path: Path) -> DatasetSpec:
    """Load one dataset specification without touching the dataset itself."""

    resolved = Path(path).resolve()
    values = json.loads(resolved.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Dataset specification {resolved} must contain a JSON object.")
    if values.get("schema_version") != 1:
        raise ValueError(f"Dataset specification {resolved} requires schema_version 1.")
    dataset_id = str(values.get("dataset_id", "")).strip()
    adapter = str(values.get("adapter", "")).strip().casefold()
    if not dataset_id:
        raise ValueError(f"Dataset specification {resolved} needs a non-empty dataset_id.")
    if adapter not in _DATASET_ADAPTERS:
        raise ValueError(
            f"Unknown dataset adapter {adapter!r}; choose from {available_dataset_adapters()}."
        )
    return DatasetSpec(resolved, dataset_id, adapter, values)


def _mapped_targets(values: pd.Series, label_map: Mapping[str, Any] | None) -> pd.Series:
    if label_map:
        normalized_map = {str(key): int(value) for key, value in label_map.items()}
        result = values.astype(str).map(normalized_map)
        if result.isna().any():
            missing = sorted(values[result.isna()].astype(str).unique().tolist())
            raise ValueError(f"label_map has no entries for labels: {missing}")
        return result.astype(np.int64)
    result = pd.to_numeric(values, errors="coerce")
    if result.isna().any():
        raise ValueError("Targets must be numeric 0/1 values or use label_map.")
    return result.astype(np.int64)


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    result = frame
    for column, allowed in filters.items():
        if column not in result:
            raise ValueError(f"Dataset filter references missing column {column!r}.")
        values = allowed if isinstance(allowed, list) else [allowed]
        result = result[result[column].isin(values)]
    if result.empty:
        raise ValueError("Dataset filters selected zero rows.")
    return result.copy()


def _derived_parent_id(dataset_id: str, image_path: str) -> str:
    digest = sha256(f"{dataset_id}:{image_path}".encode()).hexdigest()[:20]
    return f"{dataset_id}-{digest}"


def _canonicalize(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    root: Path,
    source_path: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "image_path" not in result or "target" not in result:
        raise ValueError("Dataset adapters must provide image_path and target.")
    result["image_path"] = result["image_path"].astype(str)
    if "parent_id" not in result:
        result["parent_id"] = [
            _derived_parent_id(dataset_id, value) for value in result["image_path"]
        ]
    result["parent_id"] = result["parent_id"].astype(str)
    result.insert(0, "dataset_id", dataset_id)
    result["resolved_path"] = [str(_repo_path(value, root)) for value in result["image_path"]]

    defaults = {
        "lineage_id": result["parent_id"],
        "generator_family": np.where(result["target"].eq(0), "authentic", "unknown_aigc"),
        "generation_model": np.where(result["target"].eq(0), "none", "unknown"),
        "source_dataset": dataset_id,
        "authentic_subtype": np.where(result["target"].eq(0), dataset_id, "not_applicable"),
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
        result[column] = result[column].fillna("unknown").astype(str)
    result["dataset_spec"] = source_path
    return validate_evaluation_records(result)


@register_dataset_adapter("csv")
def _load_csv(spec: DatasetSpec, repo_root: Path) -> pd.DataFrame:
    values = spec.values
    if "manifest" not in values:
        raise ValueError(f"CSV dataset {spec.dataset_id!r} requires manifest.")
    manifest = _repo_path(str(values["manifest"]), repo_root)
    frame = pd.read_csv(manifest)
    frame = _apply_filters(frame, values.get("filters", {}))

    columns = values.get("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("CSV dataset columns must be a canonical-to-source mapping.")
    required_mapping = {"image_path", "target"}
    missing_mapping = required_mapping - set(columns)
    if missing_mapping:
        raise ValueError(f"CSV dataset columns are missing: {sorted(missing_mapping)}")
    missing_source = set(columns.values()) - set(frame.columns)
    if missing_source:
        raise ValueError(f"CSV manifest is missing mapped columns: {sorted(missing_source)}")

    canonical = pd.DataFrame({name: frame[source] for name, source in columns.items()})
    canonical["target"] = _mapped_targets(canonical["target"], values.get("label_map"))
    root = _repo_path(str(values.get("root", ".")), repo_root)
    return _canonicalize(
        canonical,
        dataset_id=spec.dataset_id,
        root=root,
        source_path=str(spec.path),
    )


@register_dataset_adapter("class_folders")
def _load_class_folders(spec: DatasetSpec, repo_root: Path) -> pd.DataFrame:
    values = spec.values
    if "root" not in values or "classes" not in values:
        raise ValueError(f"Class-folder dataset {spec.dataset_id!r} requires root and classes.")
    root = _repo_path(str(values["root"]), repo_root)
    classes = values["classes"]
    if not isinstance(classes, dict) or not classes:
        raise ValueError("classes must map relative directories to binary targets.")
    extensions = tuple(
        str(item).casefold() for item in values.get("extensions", DEFAULT_IMAGE_EXTENSIONS)
    )
    rows: list[dict[str, Any]] = []
    class_metadata = values.get("class_metadata", {})
    for relative_directory, raw_target in classes.items():
        target = int(raw_target)
        directory = root / str(relative_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset class directory does not exist: {directory}")
        metadata = class_metadata.get(relative_directory, {})
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.suffix.casefold() not in extensions:
                continue
            relative_path = str(path.relative_to(root))
            rows.append({
                "image_path": relative_path,
                "target": target,
                "parent_id": _derived_parent_id(spec.dataset_id, relative_path),
                **metadata,
            })
    if not rows:
        raise ValueError(f"Class-folder dataset {spec.dataset_id!r} contains no supported images.")
    frame = pd.DataFrame(rows)
    return _canonicalize(
        frame,
        dataset_id=spec.dataset_id,
        root=root,
        source_path=str(spec.path),
    )


def validate_evaluation_records(frame: pd.DataFrame, *, verify_paths: bool = False) -> pd.DataFrame:
    """Validate one canonical row per immutable source image."""

    missing = REQUIRED_CANONICAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation records are missing columns: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("Evaluation requires at least one source image.")
    targets = set(pd.to_numeric(result["target"], errors="coerce").dropna().astype(int))
    if not targets <= {0, 1} or result["target"].isna().any():
        raise ValueError("Evaluation targets must contain only 0 (authentic) and 1 (AIGC).")
    result["target"] = result["target"].astype(np.int64)
    duplicate = result.duplicated(["dataset_id", "parent_id"], keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, ["dataset_id", "parent_id"]].head().to_dict("records")
        raise ValueError(f"Evaluation parent IDs must be unique within a dataset: {examples}")
    if verify_paths:
        missing_paths = [value for value in result["resolved_path"] if not Path(str(value)).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Evaluation image paths do not exist: {missing_paths[:5]}")
    return result


def load_evaluation_datasets(
    paths: Iterable[Path],
    *,
    repo_root: Path,
    verify_paths: bool = True,
) -> tuple[pd.DataFrame, list[DatasetSpec]]:
    """Load multiple independently configured datasets into one canonical table."""

    specs = [load_dataset_spec(path) for path in paths]
    if not specs:
        raise ValueError("At least one dataset specification is required.")
    dataset_ids = [spec.dataset_id for spec in specs]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Each dataset specification must have a unique dataset_id.")
    frames = [_DATASET_ADAPTERS[spec.adapter](spec, repo_root.resolve()) for spec in specs]
    combined = pd.concat(frames, ignore_index=True)
    return validate_evaluation_records(combined, verify_paths=verify_paths), specs


def resolve_transform_specs(
    profiles: Iterable[str],
    *,
    conditions: Iterable[str] | None = None,
    official_only: bool = False,
) -> tuple[TransformSpec, ...]:
    """Resolve a stable de-duplicated transform matrix from named profiles."""

    requested_profiles = tuple(profiles)
    if not requested_profiles:
        requested_profiles = ("core",)
    unknown = set(requested_profiles) - set(TRANSFORM_PROFILES)
    if unknown:
        raise ValueError(f"Unknown transform profiles: {sorted(unknown)}")
    by_name: dict[str, TransformSpec] = {}
    for profile in requested_profiles:
        for spec in get_transform_specs(profile):
            if official_only and not spec.official:
                continue
            by_name.setdefault(spec.name, spec)
    if conditions is not None:
        condition_tuple = tuple(conditions)
        missing = set(condition_tuple) - set(by_name)
        if missing:
            raise ValueError(
                f"Requested conditions are absent from the selected profiles: {sorted(missing)}"
            )
        by_name = {name: by_name[name] for name in condition_tuple}
    if "clean" not in by_name:
        raise ValueError("Robustness evaluation must include the clean condition.")
    return tuple(by_name.values())


def transform_registry(specs: Iterable[TransformSpec]) -> pd.DataFrame:
    names = [spec.name for spec in specs]
    registry = transform_frame("all")
    return registry.set_index("name").loc[names].reset_index()


def assigned_chain_banks(
    lengths: Iterable[int] = (1, 2, 3),
) -> dict[int, tuple[TransformSpec, ...]]:
    """Return predeclared recipe banks for per-parent chain assignment.

    Singles cover every official challenge endpoint.  Two-step recipes cover
    every ordered pair of distinct medium-severity official operations.
    Three-step recipes use the preregistered realistic journeys, excluding the
    two four-step journeys in that profile.
    """

    requested = tuple(dict.fromkeys(int(length) for length in lengths))
    if not requested or not set(requested) <= {1, 2, 3}:
        raise ValueError("Assigned chain lengths must be a non-empty subset of {1, 2, 3}.")
    candidates = get_transform_specs("all")
    banks = {
        1: tuple(
            spec
            for spec in candidates
            if spec.official and len(spec.steps) == 1
        ),
        2: tuple(
            spec
            for spec in candidates
            if spec.design == "directed_medium_pair" and len(spec.steps) == 2
        ),
        3: tuple(
            spec
            for spec in candidates
            if spec.design == "preregistered_realistic_chain" and len(spec.steps) == 3
        ),
    }
    selected = {length: banks[length] for length in requested}
    empty = [length for length, specs in selected.items() if not specs]
    if empty:
        raise RuntimeError(f"No predeclared recipes exist for chain lengths: {empty}")
    return selected


def assigned_transform_specs(
    banks: Mapping[int, Iterable[TransformSpec]],
) -> tuple[TransformSpec, ...]:
    """Flatten assigned recipe banks into a stable registry including clean."""

    clean = next(spec for spec in get_transform_specs("core") if spec.name == "clean")
    by_name = {clean.name: clean}
    for length in sorted(banks):
        for spec in banks[length]:
            by_name.setdefault(spec.name, spec)
    return tuple(by_name.values())


def _assigned_spec_index(
    *,
    dataset_id: str,
    parent_id: str,
    chain_length: int,
    base_seed: int,
    bank_size: int,
) -> int:
    material = (
        f"{base_seed}:{dataset_id}:{parent_id}:assigned-chain-length-{chain_length}"
    )
    return int.from_bytes(sha256(material.encode()).digest()[:8], "big") % bank_size


class TransformEndpointDataset:
    """Lazily evaluate every source image under every selected condition."""

    def __init__(
        self,
        records: pd.DataFrame,
        transforms: Iterable[TransformSpec],
        *,
        image_size: int,
        base_seed: int,
    ) -> None:
        self.records = validate_evaluation_records(records, verify_paths=True)
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError("At least one transform is required.")
        if image_size < 1:
            raise ValueError("image_size must be positive.")
        self.image_size = int(image_size)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.records) * len(self.transforms)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, transform_index = divmod(index, len(self.transforms))
        row = self.records.iloc[record_index]
        spec = self.transforms[transform_index]
        with Image.open(str(row["resolved_path"])) as opened:
            source = opened.convert("RGB")
            transformed = apply_transform(
                source,
                spec.name,
                parent_id=f"{row['dataset_id']}:{row['parent_id']}",
                base_seed=self.base_seed,
            )
        return {
            "pixel_values": canonical_preprocess(transformed, image_size=self.image_size),
            "dataset_id": str(row["dataset_id"]),
            "parent_id": str(row["parent_id"]),
            "lineage_id": str(row["lineage_id"]),
            "image_path": str(row["image_path"]),
            "target": np.int64(row["target"]),
            "generator_family": str(row["generator_family"]),
            "generation_model": str(row["generation_model"]),
            "source_dataset": str(row["source_dataset"]),
            "authentic_subtype": str(row["authentic_subtype"]),
            "condition": spec.name,
            "transform_family": spec.family,
            "severity": np.float64(spec.severity),
            "official_transform": bool(spec.official),
            "transform_design": spec.design,
            "transform_step_count": np.int64(len(spec.steps)),
        }


class AssignedChainEndpointDataset:
    """Evaluate clean plus one deterministic recipe from each chain-length bank."""

    def __init__(
        self,
        records: pd.DataFrame,
        banks: Mapping[int, Iterable[TransformSpec]],
        *,
        image_size: int,
        base_seed: int,
    ) -> None:
        self.records = validate_evaluation_records(records, verify_paths=True)
        self.banks = {
            int(length): tuple(specs)
            for length, specs in sorted(banks.items())
        }
        if not self.banks or any(not specs for specs in self.banks.values()):
            raise ValueError("Every assigned chain-length bank must be non-empty.")
        if set(self.banks) - {1, 2, 3}:
            raise ValueError("Assigned chain lengths must be in {1, 2, 3}.")
        for length, specs in self.banks.items():
            if any(len(spec.steps) != length for spec in specs):
                raise ValueError(f"Chain bank {length} contains a recipe of another length.")
        if image_size < 1:
            raise ValueError("image_size must be positive.")
        self.image_size = int(image_size)
        self.base_seed = int(base_seed)
        self.chain_lengths = tuple(self.banks)

    def __len__(self) -> int:
        return len(self.records) * (1 + len(self.chain_lengths))

    def _spec(self, row: pd.Series, endpoint_index: int) -> TransformSpec:
        if endpoint_index == 0:
            return next(spec for spec in get_transform_specs("core") if spec.name == "clean")
        length = self.chain_lengths[endpoint_index - 1]
        bank = self.banks[length]
        index = _assigned_spec_index(
            dataset_id=str(row["dataset_id"]),
            parent_id=str(row["parent_id"]),
            chain_length=length,
            base_seed=self.base_seed,
            bank_size=len(bank),
        )
        return bank[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        endpoints_per_parent = 1 + len(self.chain_lengths)
        record_index, endpoint_index = divmod(index, endpoints_per_parent)
        row = self.records.iloc[record_index]
        spec = self._spec(row, endpoint_index)
        with Image.open(str(row["resolved_path"])) as opened:
            source = opened.convert("RGB")
            transformed = apply_transform(
                source,
                spec.name,
                parent_id=f"{row['dataset_id']}:{row['parent_id']}",
                base_seed=self.base_seed,
            )
        return {
            "pixel_values": canonical_preprocess(transformed, image_size=self.image_size),
            "dataset_id": str(row["dataset_id"]),
            "parent_id": str(row["parent_id"]),
            "lineage_id": str(row["lineage_id"]),
            "image_path": str(row["image_path"]),
            "target": np.int64(row["target"]),
            "generator_family": str(row["generator_family"]),
            "generation_model": str(row["generation_model"]),
            "source_dataset": str(row["source_dataset"]),
            "authentic_subtype": str(row["authentic_subtype"]),
            "condition": spec.name,
            "transform_family": spec.family,
            "severity": np.float64(spec.severity),
            "official_transform": bool(spec.official),
            "transform_design": spec.design,
            "transform_step_count": np.int64(len(spec.steps)),
        }
