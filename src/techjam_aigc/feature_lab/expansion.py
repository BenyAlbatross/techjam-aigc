"""License-gated and leakage-audited dataset expansion planning.

This module deliberately plans selections from an existing metadata manifest.
It performs no network access and contains no downloader.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_LABELS = {"authentic": 0, "real": 0, "aigc": 1, "fake": 1, "full_synthetic": 1}
FORBIDDEN_LABEL_TOKENS = ("tamper", "edited", "composite", "partial", "inpaint", "photoshop")
REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "dataset",
    "purpose",
    "selected",
    "source_url",
    "revision",
    "file_list",
    "file_list_sha256",
    "content_sha256",
    "license_name",
    "license_url",
    "license_decision",
    "underlying_image_license_name",
    "underlying_image_license_url",
    "underlying_image_license_decision",
    "reviewer",
    "review_date",
}
REQUIRED_MANIFEST_FIELDS = {
    "source_id",
    "parent_id",
    "dataset",
    "split",
    "label",
    "generation_model",
    "generator_family",
    "source_dataset",
    "source_member",
    "local_path",
    "width",
    "height",
    "format",
    "bytes",
    "revision",
    "phase",
}
HEX_DIGITS = set("0123456789abcdef")


def load_expansion_config(path: Path) -> dict[str, Any]:
    """Load and structurally validate the expansion configuration."""

    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_expansion_config(config)
    return config


def validate_expansion_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete source records and unsafe acquisition defaults."""

    if config.get("schema_version") != 1:
        raise ValueError("Unsupported data-expansion schema_version; expected 1.")
    if not str(config.get("selection_seed", "")).strip():
        raise ValueError("selection_seed must be non-empty.")
    acquisition = config.get("acquisition", {})
    if acquisition.get("default_mode") != "dry_run":
        raise ValueError("Data acquisition must default to dry_run.")
    if acquisition.get("require_all_selected_sources_allowlisted") is not True:
        raise ValueError("Every selected source must be license-gated.")

    confirmation = config.get("final_confirmation", {})
    if confirmation.get("phase") != "final_confirmation" or confirmation.get("sealed") is not True:
        raise ValueError("The final_confirmation phase must be explicitly sealed.")
    if not isinstance(confirmation.get("generator_ids"), list):
        raise ValueError("final_confirmation.generator_ids must be a list.")

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("At least one source policy is required.")
    source_ids: list[str] = []
    for source in sources:
        missing = REQUIRED_SOURCE_FIELDS - set(source)
        if missing:
            raise ValueError(f"Source policy is missing fields: {sorted(missing)}")
        source_id = str(source["source_id"]).strip()
        if not source_id:
            raise ValueError("source_id must be non-empty.")
        source_ids.append(source_id)
        if source["license_decision"] not in {"pending", "allowlisted", "rejected"}:
            raise ValueError(f"Invalid license_decision for {source_id}.")
        if source["underlying_image_license_decision"] not in {
            "pending",
            "allowlisted",
            "rejected",
        }:
            raise ValueError(
                f"Invalid underlying_image_license_decision for {source_id}."
            )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique.")


def acquisition_audit(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return one auditable license-gate decision per configured source."""

    validate_expansion_config(config)
    records: list[dict[str, Any]] = []
    for source in config["sources"]:
        required_values = {
            name: source.get(name)
            for name in (
                "source_url",
                "revision",
                "file_list",
                "file_list_sha256",
                "content_sha256",
                "license_name",
                "license_url",
                "underlying_image_license_name",
                "underlying_image_license_url",
                "reviewer",
                "review_date",
            )
        }
        missing = [name for name, value in required_values.items() if value in {None, ""}]
        invalid_hashes = [
            name
            for name in ("file_list_sha256", "content_sha256")
            if source.get(name) not in {None, ""} and not _is_sha256(str(source[name]))
        ]
        allowlisted = (
            source["license_decision"] == "allowlisted"
            and source["underlying_image_license_decision"] == "allowlisted"
            and not missing
            and not invalid_hashes
        )
        records.append(
            {
                "source_id": source["source_id"],
                "dataset": source["dataset"],
                "selected": bool(source["selected"]),
                "license_decision": source["license_decision"],
                "underlying_image_license_decision": source[
                    "underlying_image_license_decision"
                ],
                "metadata_complete": not missing and not invalid_hashes,
                "missing_fields": ",".join(missing),
                "invalid_hash_fields": ",".join(invalid_hashes),
                "acquisition_allowed": allowlisted,
            }
        )
    return pd.DataFrame.from_records(records)


def assert_acquisition_allowed(config: Mapping[str, Any]) -> None:
    """Block acquisition unless every selected source is explicitly allowlisted."""

    audit = acquisition_audit(config)
    blocked = audit[audit["selected"] & ~audit["acquisition_allowed"]]
    if not blocked.empty:
        details = ", ".join(
            f"{row.source_id} ({row.license_decision})" for row in blocked.itertuples()
        )
        raise PermissionError(f"Acquisition blocked by source/license audit: {details}")


def select_expansion_rows(manifest: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Validate and deterministically sample configured source/generator strata."""

    validate_expansion_config(config)
    audited = audit_expansion_manifest(manifest, config)
    source_policies = {source["source_id"]: source for source in config["sources"]}
    selected_source_ids = {key for key, source in source_policies.items() if source["selected"]}
    selected = audited[audited["source_id"].isin(selected_source_ids)].copy()
    if selected.empty:
        return selected

    selected["selection_stratum"] = selected.apply(_selection_stratum, axis=1)
    selected["selection_hash"] = selected.apply(
        lambda row: stable_hash(
            str(config["selection_seed"]),
            row["source_id"],
            row["phase"],
            row["selection_stratum"],
            row["parent_id"],
        ),
        axis=1,
    )
    selected = selected.sort_values(
        ["source_id", "phase", "selection_stratum", "selection_hash", "parent_id"],
        kind="stable",
    )

    pieces: list[pd.DataFrame] = []
    for source_id, source_rows in selected.groupby("source_id", sort=True):
        policy = source_policies[source_id]
        limit = _selection_limit(policy)
        if limit is None:
            pieces.append(source_rows)
        else:
            pieces.append(
                source_rows.groupby(["phase", "selection_stratum"], sort=True, group_keys=False)
                .head(limit)
            )
    result = pd.concat(pieces, ignore_index=True)
    result["selection_rank"] = (
        result.groupby(["source_id", "phase", "selection_stratum"]).cumcount() + 1
    )
    return result.sort_values(
        ["source_id", "phase", "selection_stratum", "selection_rank"], kind="stable"
    ).reset_index(drop=True)


def audit_expansion_manifest(manifest: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Enforce binary scope, provenance, phase sealing, and generator isolation."""

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.columns)
    if missing:
        raise ValueError(f"Expansion manifest is missing fields: {sorted(missing)}")
    frame = manifest.copy()
    if frame.empty:
        return frame.assign(target=pd.Series(dtype="int64"))
    if frame[list(REQUIRED_MANIFEST_FIELDS)].isna().any().any():
        columns = frame[list(REQUIRED_MANIFEST_FIELDS)].columns[
            frame[list(REQUIRED_MANIFEST_FIELDS)].isna().any()
        ].tolist()
        raise ValueError(f"Expansion manifest has missing provenance values: {columns}")
    if frame["parent_id"].astype(str).duplicated().any():
        raise ValueError("parent_id must be globally unique to preserve parent pairing.")

    configured_sources = {str(source["source_id"]) for source in config["sources"]}
    unknown_sources = sorted(set(frame["source_id"].astype(str)) - configured_sources)
    if unknown_sources:
        raise ValueError(f"Manifest contains unconfigured source_id values: {unknown_sources}")
    source_policies = {
        str(source["source_id"]): source for source in config["sources"]
    }
    for source_id, rows in frame.groupby("source_id", sort=False):
        configured_revision = source_policies[str(source_id)].get("revision")
        if configured_revision not in {None, ""}:
            observed_revisions = set(rows["revision"].astype(str))
            if observed_revisions != {str(configured_revision)}:
                raise ValueError(
                    f"Manifest revision for {source_id} does not match the configured "
                    f"revision {configured_revision!r}: {sorted(observed_revisions)}"
                )

    labels = frame["label"].astype(str).str.strip().str.lower()
    forbidden_labels = labels.apply(lambda value: any(token in value for token in FORBIDDEN_LABEL_TOKENS))
    if forbidden_labels.any():
        values = sorted(set(labels[forbidden_labels]))
        raise ValueError(f"Tampered, edited, or composited labels are forbidden: {values}")
    invalid_labels = sorted(set(labels) - set(ALLOWED_LABELS))
    if invalid_labels:
        raise ValueError(f"Only authentic or fully synthetic labels are allowed: {invalid_labels}")
    frame["target"] = labels.map(ALLOWED_LABELS).astype(int)

    demo_rows = (
        frame["generation_model"].astype(str).str.contains("DALL-E Advanced", case=False, na=False)
        & frame["source_dataset"].astype(str).str.contains("COCO.*val2017", case=False, regex=True, na=False)
    ) | frame["phase"].astype(str).str.contains("organizer.*demo|demo.*only", case=False, regex=True, na=False)
    if demo_rows.any():
        raise ValueError("Organizer demonstration-only rows are forbidden.")

    allowed_phases = {"discovery", "confirmation", "final_confirmation"}
    phases = set(frame["phase"].astype(str))
    if not phases <= allowed_phases:
        raise ValueError(f"Unsupported phases: {sorted(phases - allowed_phases)}")

    confirmation = config["final_confirmation"]
    final_rows = frame[frame["phase"] == "final_confirmation"]
    assigned = {_normalize_generator(value) for value in confirmation["generator_ids"]}
    if not final_rows.empty:
        if confirmation.get("sealed") is not True:
            raise ValueError("final_confirmation rows require a sealed assignment.")
        final_generators = _generated_generators(final_rows)
        if confirmation.get("assignment_required_before_selection") and not assigned:
            raise ValueError("Assign final_confirmation.generator_ids before selecting final rows.")
        unassigned = sorted(final_generators - assigned)
        if unassigned:
            raise ValueError(f"Unassigned generators appear in final_confirmation: {unassigned}")

    discovery_generators = _generated_generators(frame[frame["phase"] == "discovery"])
    final_generators = _generated_generators(final_rows)
    overlap = sorted(discovery_generators & final_generators)
    if overlap:
        raise ValueError(f"Discovery/final-confirmation generator overlap is forbidden: {overlap}")
    if assigned & discovery_generators:
        raise ValueError(
            "A sealed final-confirmation generator appears in discovery: "
            f"{sorted(assigned & discovery_generators)}"
        )

    return frame


def plan_diversity_cohorts(
    selected: pd.DataFrame,
    *,
    source_id: str = "community_forensics_diversity",
    few_generators: int = 10,
    many_generators: int = 100,
    total_images: int | None = None,
    seed: str = "techjam-aigc-diversity-v1",
) -> pd.DataFrame:
    """Plan disjoint few/many-generator cohorts with exactly equal image counts."""

    if few_generators < 1 or many_generators <= few_generators:
        raise ValueError("Require 1 <= few_generators < many_generators.")
    required = {"source_id", "phase", "target", "generation_model", "parent_id"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Cohort input is missing fields: {sorted(missing)}")
    candidates = selected[
        (selected["source_id"] == source_id)
        & (selected["phase"] == "discovery")
        & (selected["target"] == 1)
    ].copy()
    candidates["generator_id"] = candidates["generation_model"].map(_normalize_generator)
    generators = sorted(
        candidates["generator_id"].unique(),
        key=lambda generator: stable_hash(seed, "generator", generator),
    )
    needed = few_generators + many_generators
    if len(generators) < needed:
        raise ValueError(f"Need {needed} disjoint generators; found {len(generators)}.")
    cohort_generators = {
        "few_generators": generators[:few_generators],
        "many_generators": generators[few_generators:needed],
    }
    pools = {
        name: candidates[candidates["generator_id"].isin(ids)].copy()
        for name, ids in cohort_generators.items()
    }
    maximum_equal_count = min(len(pool) for pool in pools.values())
    requested = maximum_equal_count if total_images is None else total_images
    if requested < 1 or requested > maximum_equal_count:
        raise ValueError(
            f"total_images must be between 1 and {maximum_equal_count} for both cohorts."
        )

    planned: list[pd.DataFrame] = []
    for name, pool in pools.items():
        sample = _round_robin_sample(pool, requested, seed=seed, cohort=name)
        sample["diversity_cohort"] = name
        sample["cohort_generator_count"] = len(cohort_generators[name])
        sample["cohort_image_count"] = requested
        planned.append(sample)
    result = pd.concat(planned, ignore_index=True)
    if result.groupby("diversity_cohort").size().nunique() != 1:
        raise AssertionError("Diversity cohorts must have equal image counts.")
    return result.sort_values(["diversity_cohort", "cohort_rank"], kind="stable").reset_index(drop=True)


def expansion_coverage(selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize phase/class/generator/source coverage and power warnings."""

    if selected.empty:
        return pd.DataFrame(
            columns=["source_id", "phase", "target", "generation_model", "source_dataset", "images", "low_power"]
        )
    return (
        selected.groupby(
            ["source_id", "phase", "target", "generation_model", "source_dataset"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "images"})
        .assign(low_power=lambda table: table["images"] < 200)
    )


def build_audit_summary(
    manifest: pd.DataFrame,
    selected: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable audit artifact for a planned selection."""

    final_rows = selected[selected.get("phase", pd.Series(dtype=str)) == "final_confirmation"]
    return {
        "schema_version": 1,
        "selection_seed": config["selection_seed"],
        "manifest_rows": int(len(manifest)),
        "selected_rows": int(len(selected)),
        "selected_parent_ids_unique": bool(selected.get("parent_id", pd.Series(dtype=str)).is_unique),
        "selection_sha256": selection_sha256,
        "selected_sources": sorted(selected.get("source_id", pd.Series(dtype=str)).unique().tolist()),
        "final_confirmation": {
            "sealed": bool(config["final_confirmation"]["sealed"]),
            "configured_generators": sorted(config["final_confirmation"]["generator_ids"]),
            "selected_rows": int(len(final_rows)),
            "first_evaluated_at": config["final_confirmation"].get("first_evaluated_at"),
        },
        "acquisition_allowed": bool(
            acquisition_audit(config).query("selected")["acquisition_allowed"].all()
        ),
        "network_actions_performed": False,
    }


def stable_hash(*parts: object) -> str:
    """Return a platform-independent hash used for deterministic ordering."""

    serialized = json.dumps([str(part) for part in parts], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _selection_limit(policy: Mapping[str, Any]) -> int | None:
    if policy["source_id"] == "aigenbench_targeted":
        return int(policy["target_images_per_generator_pilot"])
    if policy["source_id"] == "community_forensics_diversity":
        return int(policy["target_images_per_generator_max"])
    value = policy.get("target_images_per_stratum")
    return None if value is None else int(value)


def _selection_stratum(row: pd.Series) -> str:
    if int(row["target"]) == 1:
        return f"generated:{_normalize_generator(row['generation_model'])}"
    return f"authentic:{str(row['source_dataset']).strip().casefold()}"


def _generated_generators(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "target" not in frame:
        return set()
    return {
        _normalize_generator(value)
        for value in frame.loc[frame["target"] == 1, "generation_model"].tolist()
    }


def _normalize_generator(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _round_robin_sample(pool: pd.DataFrame, count: int, *, seed: str, cohort: str) -> pd.DataFrame:
    groups: dict[str, list[int]] = {}
    for generator, rows in pool.groupby("generator_id", sort=True):
        ordered = rows.assign(
            _hash=rows["parent_id"].map(lambda parent: stable_hash(seed, cohort, generator, parent))
        ).sort_values(["_hash", "parent_id"], kind="stable")
        groups[str(generator)] = ordered.index.tolist()
    generator_order = sorted(groups, key=lambda generator: stable_hash(seed, cohort, generator))
    chosen: list[int] = []
    offset = 0
    while len(chosen) < count:
        progressed = False
        for generator in generator_order:
            indices = groups[generator]
            if offset < len(indices):
                chosen.append(indices[offset])
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise ValueError("Insufficient rows to complete deterministic cohort sample.")
        offset += 1
    result = pool.loc[chosen].copy()
    result["cohort_rank"] = range(1, len(result) + 1)
    return result


def _is_sha256(value: str) -> bool:
    normalized = value.strip().lower()
    return len(normalized) == 64 and set(normalized) <= HEX_DIGITS


def write_expansion_artifacts(
    output_dir: Path,
    manifest: pd.DataFrame,
    selected: pd.DataFrame,
    config: Mapping[str, Any],
    diversity_cohorts: pd.DataFrame | None = None,
) -> None:
    """Write deterministic selection, source audit, coverage, and run audit files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selection.csv", index=False)
    selection_path = output_dir / "selection.csv"
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    expansion_coverage(selected).to_csv(output_dir / "coverage.csv", index=False)
    acquisition_audit(config).to_csv(output_dir / "source_audit.csv", index=False)
    if diversity_cohorts is not None:
        diversity_cohorts.to_csv(output_dir / "diversity_cohorts.csv", index=False)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(
            build_audit_summary(
                manifest,
                selected,
                config,
                selection_sha256=selection_sha256,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
