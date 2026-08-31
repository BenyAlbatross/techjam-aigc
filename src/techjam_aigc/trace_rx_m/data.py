"""Leakage-safe manifests, datasets, and balanced batches for TRACE-RX-M.

The model has several strictly ordered data roles.  This module validates the
whole manifest before selecting a role so a caller cannot hide leakage by
filtering first.  Dataset objects deliberately return NumPy arrays: PyTorch's
default collator converts them to tensors, while manifest tests remain usable
without importing or downloading the deep-learning stack.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from techjam_aigc.feature_lab.pipeline import _assert_planned_index_acquisition_allowed

from .augment import SymmetricTransformSampler, canonical_preprocess


AUTHENTIC_LABELS = {"authentic", "real"}
NATIVE_AIGC_LABELS = {"aigc", "fake", "full_synthetic"}
DDA_LABELS = {"dda", "dda_aligned", "dual_data_aligned"}
LABEL_TARGETS = {
    **{label: 0 for label in AUTHENTIC_LABELS},
    **{label: 1 for label in NATIVE_AIGC_LABELS | DDA_LABELS},
}
SAMPLE_KINDS = {"authentic", "native_aigc", "dda"}
TRAINING_ROLES = {
    "memory_pool",
    "capacity_validation",
    "supervised",
    "authentic_null",
    "development",
    "calibration",
    "locked_evaluation",
}
AUTHENTIC_ONLY_ROLES = {"memory_pool", "capacity_validation", "authentic_null"}
FORBIDDEN_LABEL_TOKENS = (
    "tamper",
    "edited",
    "composite",
    "partial",
    "inpaint",
    "photoshop",
)
REQUIRED_COLUMNS = {
    "parent_id",
    "lineage_id",
    "role",
    "phase",
    "label",
    "sample_kind",
    "generator_family",
    "generation_model",
    "source_dataset",
    "local_path",
}
OPTIONAL_LEAKAGE_GROUPS = ("master_id", "prompt_group_id", "duplicate_group_id")


def load_training_manifest(
    path: Path,
    *,
    roles: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and validate a TRACE-RX-M manifest.

    When the manifest carries ``source_id``, the exact existing expansion
    audit gate is applied before any row is read for training.  Validation is
    performed before optional role selection.
    """

    resolved = Path(path).resolve()
    _assert_planned_index_acquisition_allowed(resolved)
    frame = pd.read_csv(resolved)
    validated = validate_training_manifest(frame)
    if roles is None:
        return validated
    requested = {str(role).strip().casefold() for role in roles}
    invalid = requested - TRAINING_ROLES
    if invalid:
        raise ValueError(f"Unsupported requested roles: {sorted(invalid)}")
    return validated[validated["role"].isin(requested)].reset_index(drop=True)


def validate_training_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce binary scope, role isolation, and DDA pairing."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Training manifest is missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Training manifest cannot be empty.")
    result = frame.copy()
    required = sorted(REQUIRED_COLUMNS)
    if result[required].isna().any().any():
        columns = result[required].columns[result[required].isna().any()].tolist()
        raise ValueError(f"Training manifest has missing required values: {columns}")

    for column in ("parent_id", "lineage_id", "role", "phase", "label", "sample_kind"):
        result[column] = result[column].astype(str).str.strip()
    if result["parent_id"].eq("").any() or result["lineage_id"].eq("").any():
        raise ValueError("parent_id and lineage_id must be non-empty.")
    if result["parent_id"].duplicated().any():
        duplicates = sorted(result.loc[result["parent_id"].duplicated(False), "parent_id"].unique())
        raise ValueError(f"parent_id must be globally unique: {duplicates}")

    result["role"] = result["role"].str.casefold()
    invalid_roles = sorted(set(result["role"]) - TRAINING_ROLES)
    if invalid_roles:
        raise ValueError(f"Unsupported training roles: {invalid_roles}")
    result["sample_kind"] = result["sample_kind"].str.casefold()
    invalid_kinds = sorted(set(result["sample_kind"]) - SAMPLE_KINDS)
    if invalid_kinds:
        raise ValueError(f"Unsupported sample kinds: {invalid_kinds}")

    labels = result["label"].str.casefold()
    forbidden = labels.apply(lambda value: any(token in value for token in FORBIDDEN_LABEL_TOKENS))
    if forbidden.any():
        raise ValueError(
            "Tampered, edited, or composited samples are outside the binary task: "
            f"{sorted(set(labels[forbidden]))}"
        )
    invalid_labels = sorted(set(labels) - set(LABEL_TARGETS))
    if invalid_labels:
        raise ValueError(f"Unsupported binary/DDA labels: {invalid_labels}")
    result["label"] = labels
    result["target"] = labels.map(LABEL_TARGETS).astype(int)

    expected_kind = labels.map(
        lambda label: (
            "authentic"
            if label in AUTHENTIC_LABELS
            else "dda"
            if label in DDA_LABELS
            else "native_aigc"
        )
    )
    mismatched_kind = result["sample_kind"] != expected_kind
    if mismatched_kind.any():
        raise ValueError("label and sample_kind assignments disagree.")

    demo = (
        result["generation_model"].astype(str).str.contains(
            "DALL-E Advanced", case=False, na=False
        )
        & result["source_dataset"].astype(str).str.contains(
            r"COCO.*val2017", case=False, regex=True, na=False
        )
    ) | result["phase"].str.contains(
        r"organizer.*demo|demo.*only", case=False, regex=True, na=False
    )
    if demo.any():
        raise ValueError("Organizer demonstration-only rows are forbidden.")

    final = result["phase"].str.casefold().eq("final_confirmation")
    locked = result["role"].eq("locked_evaluation")
    if (final != locked).any():
        raise ValueError(
            "final_confirmation rows and locked_evaluation role must correspond exactly."
        )

    authentic_only = result["role"].isin(AUTHENTIC_ONLY_ROLES)
    if (authentic_only & result["sample_kind"].ne("authentic")).any():
        raise ValueError(
            "memory_pool, capacity_validation, and authentic_null accept authentic rows only."
        )
    dda_outside_supervised = result["sample_kind"].eq("dda") & result["role"].ne("supervised")
    if dda_outside_supervised.any():
        raise ValueError("DDA samples are permitted only in the supervised role.")

    _assert_groups_do_not_cross_roles(result)
    _validate_dda_pairs(result)
    return result.reset_index(drop=True)


def _assert_groups_do_not_cross_roles(frame: pd.DataFrame) -> None:
    for column in ("lineage_id", *OPTIONAL_LEAKAGE_GROUPS):
        if column not in frame:
            continue
        values = frame[column].astype("string").str.strip()
        present = values.notna() & values.ne("")
        grouped = frame.loc[present].assign(_group=values[present]).groupby("_group")["role"].nunique()
        offenders = sorted(grouped[grouped > 1].index.astype(str))
        if offenders:
            raise ValueError(f"{column} crosses data roles: {offenders}")


def _validate_dda_pairs(frame: pd.DataFrame) -> None:
    dda = frame[frame["sample_kind"].eq("dda")]
    if dda.empty:
        return
    if "source_parent_id" not in frame:
        raise ValueError("DDA rows require source_parent_id.")
    source_ids = dda["source_parent_id"].astype("string").str.strip()
    if source_ids.isna().any() or source_ids.eq("").any():
        raise ValueError("Every DDA row requires a non-empty source_parent_id.")
    if source_ids.duplicated().any():
        raise ValueError("Each authentic source may have at most one DDA derivative.")
    by_parent = frame.set_index("parent_id", drop=False)
    missing = sorted(set(source_ids.astype(str)) - set(by_parent.index.astype(str)))
    if missing:
        raise ValueError(f"DDA source_parent_id rows are missing: {missing}")
    for dda_row, source_id in zip(dda.itertuples(), source_ids.astype(str), strict=True):
        source = by_parent.loc[source_id]
        if source["sample_kind"] != "authentic" or int(source["target"]) != 0:
            raise ValueError(f"DDA source {source_id!r} is not authentic.")
        if source["role"] != "supervised" or source["role"] != dda_row.role:
            raise ValueError("DDA samples and authentic sources must share the supervised role.")
        if source["lineage_id"] != dda_row.lineage_id:
            raise ValueError("DDA samples and authentic sources must share lineage_id.")


class TraceRXMDataset:
    """Image sequence compatible with ``torch.utils.data.DataLoader``."""

    def __init__(
        self,
        frame: pd.DataFrame,
        repo_root: Path,
        *,
        transform_sampler: SymmetricTransformSampler | None = None,
        image_size: int = 224,
    ) -> None:
        self.frame = validate_training_manifest(frame).reset_index(drop=True)
        self.repo_root = Path(repo_root).resolve()
        self.transform_sampler = transform_sampler
        self.image_size = image_size
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row["local_path"]))
        resolved = path if path.is_absolute() else self.repo_root / path
        with Image.open(resolved) as opened:
            image = opened.convert("RGB")
        condition = "clean"
        if self.transform_sampler is not None:
            condition, image = self.transform_sampler.apply(
                image,
                parent_id=str(row["parent_id"]),
                epoch=self.epoch,
            )
        return {
            "pixel_values": canonical_preprocess(image, image_size=self.image_size),
            "target": np.int64(row["target"]),
            "parent_id": str(row["parent_id"]),
            "sample_kind": str(row["sample_kind"]),
            "generator_family": str(row["generator_family"]),
            "source_dataset": str(row["source_dataset"]),
            "authentic_subtype": str(row.get("authentic_subtype", row["source_dataset"])),
            "source_parent_id": str(row.get("source_parent_id", "")),
            "balance_weight": np.float32(row.get("balance_weight", 1.0)),
            "condition": condition,
        }


@dataclass
class _BalancedCycle:
    groups: dict[str, list[int]]
    rng: random.Random

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("Balanced sampling group cannot be empty.")
        self.names = sorted(self.groups)
        self.rng.shuffle(self.names)
        self.group_position = 0
        self.positions = {name: 0 for name in self.names}
        for values in self.groups.values():
            self.rng.shuffle(values)

    def take(self, *, excluding: set[int] | None = None) -> int:
        excluded = excluding or set()
        attempts = max(1, sum(len(values) for values in self.groups.values()) * 2)
        for _ in range(attempts):
            name = self.names[self.group_position % len(self.names)]
            self.group_position += 1
            values = self.groups[name]
            position = self.positions[name]
            if position and position % len(values) == 0:
                self.rng.shuffle(values)
            candidate = values[position % len(values)]
            self.positions[name] = position + 1
            if candidate not in excluded:
                return candidate
        raise ValueError("Could not draw a distinct balanced sample for this batch.")


class BalancedTraceBatchSampler:
    """Yield class-balanced batches with a configurable DDA-positive share.

    Every DDA row is emitted with its authentic source, which consumes one of
    the real slots. Native and DDA rows rotate across generator families before
    repeating a family. Call ``set_epoch`` alongside the dataset.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        batch_size: int = 10,
        dda_positive_share: float = 0.20,
        seed: int = 20260830,
        batches_per_epoch: int | None = None,
    ) -> None:
        self.frame = validate_training_manifest(frame).reset_index(drop=True)
        if set(self.frame["role"]) != {"supervised"}:
            raise ValueError("BalancedTraceBatchSampler accepts supervised rows only.")
        if batch_size < 10 or batch_size % 2:
            raise ValueError("batch_size must be even and at least ten.")
        if not 0.10 <= dda_positive_share <= 0.20:
            raise ValueError("dda_positive_share must lie in the proposal's 10--20% range.")
        self.batch_size = int(batch_size)
        self.real_count = batch_size // 2
        has_dda = self.frame["sample_kind"].eq("dda").any()
        self.dda_count = (
            max(1, round((batch_size // 2) * dda_positive_share)) if has_dda else 0
        )
        self.native_count = batch_size // 2 - self.dda_count
        self.dda_positive_share = dda_positive_share
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(frame) / batch_size)
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive.")

        kinds = self.frame["sample_kind"]
        required_kinds = {"authentic", "native_aigc"}
        if has_dda:
            required_kinds.add("dda")
        for kind in required_kinds:
            if not kinds.eq(kind).any():
                raise ValueError(f"Supervised sampling requires {kind!r} rows.")
        self._parent_to_index = {
            str(parent_id): int(index)
            for index, parent_id in enumerate(self.frame["parent_id"])
        }

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        real = self._cycle("authentic", "source_dataset", rng)
        native = self._cycle("native_aigc", "generator_family", rng)
        dda = self._cycle("dda", "generator_family", rng) if self.dda_count else None
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            used: set[int] = set()
            for _ in range(self.dda_count):
                assert dda is not None
                dda_index = dda.take(excluding=used)
                source_id = str(self.frame.iloc[dda_index]["source_parent_id"])
                source_index = self._parent_to_index[source_id]
                if source_index in used:
                    raise ValueError("A DDA source pair was duplicated within a batch.")
                batch.extend((source_index, dda_index))
                used.update((source_index, dda_index))
            for _ in range(self.native_count):
                index = native.take(excluding=used)
                batch.append(index)
                used.add(index)
            remaining_real = self.real_count - self.dda_count
            for _ in range(remaining_real):
                index = real.take(excluding=used)
                batch.append(index)
                used.add(index)
            rng.shuffle(batch)
            yield batch

    def _cycle(self, kind: str, group_column: str, rng: random.Random) -> _BalancedCycle:
        rows = self.frame[self.frame["sample_kind"].eq(kind)]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in zip(rows.index, rows[group_column], strict=True):
            key = " ".join(str(value).strip().casefold().split())
            if not key:
                raise ValueError(f"{kind} rows require non-empty {group_column}.")
            groups[key].append(int(index))
        return _BalancedCycle(dict(groups), rng)


def add_balance_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Equalize class and within-class authentic/generator groups."""

    result = validate_training_manifest(frame)
    authentic_groups = result.get("authentic_subtype", result["source_dataset"]).astype(str)
    groups = np.where(
        result["target"].eq(0),
        authentic_groups,
        result["generator_family"].astype(str),
    )
    keys = pd.Series(
        [f"{target}:{group}" for target, group in zip(result["target"], groups, strict=True)],
        index=result.index,
    )
    group_counts = keys.map(keys.value_counts())
    groups_per_class = keys.groupby(result["target"]).transform("nunique")
    weights = 1.0 / (group_counts * groups_per_class)
    result["balance_weight"] = weights / weights.mean()
    return result
