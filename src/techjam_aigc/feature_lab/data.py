"""Dataset indexing and guardrails for the binary experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BINARY_LABELS = {"real": 0, "fake": 1, "full_synthetic": 1}
EXPANSION_BINARY_LABELS = {**BINARY_LABELS, "authentic": 0, "aigc": 1}
EXPANSION_PHASES = {"discovery", "confirmation", "final_confirmation"}
EXPANSION_REQUIRED_COLUMNS = {
    "parent_id",
    "target",
    "phase",
    "dataset",
    "split",
    "label",
    "generator_family",
    "generation_model",
    "source_dataset",
    "source_member",
    "local_path",
    "width",
    "height",
    "format",
    "bytes",
}
FORBIDDEN_EXPANSION_LABEL_TOKENS = ("tamper", "edited", "composite", "partial", "inpaint")


def load_binary_index(index_path: Path) -> pd.DataFrame:
    """Load the local visual slice and enforce the challenge's binary scope."""

    frame = pd.read_csv(index_path)
    if {"parent_id", "target", "phase"} <= set(frame.columns):
        return _load_expansion_index(frame)

    forbidden = (
        frame["generation_model"].astype(str).str.contains("DALL-E Advanced", case=False, na=False)
        & frame["source_dataset"].astype(str).str.contains("COCO.*val2017", case=False, regex=True, na=False)
    )
    if forbidden.any():
        raise ValueError("Organizer demonstration-only validation rows are forbidden in feature selection.")

    binary = frame[frame["label"].isin(BINARY_LABELS)].copy()
    binary["target"] = binary["label"].map(BINARY_LABELS).astype(int)
    binary["parent_id"] = binary["local_path"].astype(str)
    binary["phase"] = binary["split"].map(_phase_from_split)
    binary["binary_label"] = binary["target"].map({0: "authentic", 1: "AIGC"})
    if binary["parent_id"].duplicated().any():
        raise ValueError("Every source image must have a unique parent_id.")
    if set(binary["phase"]) != {"discovery", "confirmation"}:
        raise ValueError("Both discovery and confirmation phases are required.")
    return binary.reset_index(drop=True)


def _load_expansion_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a dry-run expansion selection without rewriting its assignments."""

    missing = EXPANSION_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Expansion selection is missing required image metadata: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Expansion selection cannot be empty.")
    required = sorted(EXPANSION_REQUIRED_COLUMNS)
    if frame[required].isna().any().any():
        columns = frame[required].columns[frame[required].isna().any()].tolist()
        raise ValueError(f"Expansion selection has missing required values: {columns}")

    labels = frame["label"].astype(str).str.strip().str.casefold()
    forbidden_labels = labels.apply(
        lambda value: any(token in value for token in FORBIDDEN_EXPANSION_LABEL_TOKENS)
    )
    if forbidden_labels.any():
        raise ValueError("Tampered, edited, or composited rows are forbidden in expansion selections.")
    invalid_labels = sorted(set(labels) - set(EXPANSION_BINARY_LABELS))
    if invalid_labels:
        raise ValueError(f"Only authentic or fully synthetic labels are allowed: {invalid_labels}")
    target = pd.to_numeric(frame["target"], errors="coerce")
    integer_target = target.dropna().astype(int)
    if target.isna().any() or not set(integer_target) <= {0, 1} or not (target == target.astype(int)).all():
        raise ValueError("Expansion target must contain only integer binary values 0 and 1.")
    expected_target = labels.map(EXPANSION_BINARY_LABELS).astype(int)
    if not target.astype(int).equals(expected_target):
        raise ValueError("Expansion label and target assignments disagree.")

    demo_rows = (
        frame["generation_model"].astype(str).str.contains("DALL-E Advanced", case=False, na=False)
        & frame["source_dataset"].astype(str).str.contains(
            "COCO.*val2017", case=False, regex=True, na=False
        )
    ) | frame["phase"].astype(str).str.contains(
        "organizer.*demo|demo.*only", case=False, regex=True, na=False
    )
    if demo_rows.any():
        raise ValueError("Organizer demonstration-only validation rows are forbidden in feature selection.")

    phase = frame["phase"].astype(str).str.strip().str.casefold()
    invalid_phases = sorted(set(phase) - EXPANSION_PHASES)
    if invalid_phases:
        raise ValueError(f"Unsupported expansion phases: {invalid_phases}")
    if "discovery" not in set(phase) or not ({"confirmation", "final_confirmation"} & set(phase)):
        raise ValueError("Expansion selection requires discovery and at least one evaluation phase.")

    binary = frame.copy()
    binary["target"] = target.astype(int)
    binary["phase"] = phase
    binary["parent_id"] = binary["parent_id"].astype(str)
    binary["binary_label"] = binary["target"].map({0: "authentic", 1: "AIGC"})
    if binary["parent_id"].duplicated().any():
        raise ValueError("Every source image must have a unique parent_id.")

    discovery_generators = _generated_models(binary, "discovery")
    final_generators = _generated_models(binary, "final_confirmation")
    overlap = sorted(discovery_generators & final_generators)
    if overlap:
        raise ValueError(
            "Generated model overlap between discovery and final_confirmation is forbidden: "
            f"{overlap}"
        )
    return binary.reset_index(drop=True)


def _generated_models(frame: pd.DataFrame, phase: str) -> set[str]:
    values = frame.loc[
        (frame["phase"] == phase) & (frame["target"] == 1), "generation_model"
    ]
    return {" ".join(str(value).strip().casefold().split()) for value in values}


def _phase_from_split(split: str) -> str:
    normalized = str(split).lower()
    if "train" in normalized:
        return "discovery"
    if "test" in normalized or "validation" in normalized:
        return "confirmation"
    raise ValueError(f"Cannot map split to discovery/confirmation: {split}")


def coverage_table(binary: pd.DataFrame) -> pd.DataFrame:
    return (
        binary.groupby(
            [
                "dataset",
                "phase",
                "binary_label",
                "generator_family",
                "generation_model",
                "source_dataset",
            ],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "images"})
        .assign(low_power=lambda table: table["images"] < 20)
    )


def resolve_repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "pyproject.toml").exists() and (path / "docs/problem-statement.md").exists():
            return path
    raise FileNotFoundError("Could not locate repository root.")
