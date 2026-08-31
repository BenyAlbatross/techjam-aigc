"""Merge baseline and training-candidate dataset registries without mixing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _repository_identity(value: str) -> str:
    marker = "/datasets/"
    repository = value.split(marker, 1)[-1] if marker in value else value
    return _key(repository.removesuffix("/summary"))


def merge_registries(baseline: Path, expansion: Path) -> dict:
    baseline_rows = tomllib.loads(baseline.read_text(encoding="utf-8"))["datasets"]
    expansion_rows = json.loads(expansion.read_text(encoding="utf-8"))["sources"]
    records: dict[str, dict] = {}

    for source_id, row in baseline_rows.items():
        record = {
            "id": source_id,
            "name": source_id.replace("_", " ").title(),
            "repository": row.get("repository"),
            "revision": row.get("revision"),
            "status": row["status"],
            "roles": [row.get("use", "unassigned")],
            "selected": row["status"] == "approved",
            "license": row.get("license", "UNDECLARED"),
            "reason": row.get("reason"),
            "sources": ["public-baseline-robustness"],
        }
        records[_repository_identity(row.get("repository", source_id))] = record

    for row in expansion_rows:
        identity = _repository_identity(row.get("source_url", "") or row["source_id"])
        existing_key = identity if identity in records else None
        if existing_key:
            record = records[existing_key]
            role = row.get("purpose", "unassigned")
            if role not in record["roles"]:
                record["roles"].append(role)
            record["selected"] = record["selected"] and bool(row.get("selected"))
            record["sources"].append("trace-rx-parallel")
            if row.get("license_decision") == "pending" and record["status"] == "approved":
                record["integration_note"] = (
                    "Approved for the existing evaluation role only; training expansion "
                    "remains pending its separate audit."
                )
            continue

        decision = row.get("license_decision", "pending")
        status = "review" if decision == "pending" else decision
        records[identity or _key(row["source_id"])] = {
            "id": row["source_id"],
            "name": row["dataset"],
            "repository": row.get("source_url"),
            "revision": row.get("revision"),
            "status": status,
            "roles": [row.get("purpose", "unassigned")],
            "selected": bool(row.get("selected")) and status == "approved",
            "license": row.get("license_name") or "UNDECLARED",
            "reason": row.get("notes"),
            "sources": ["trace-rx-parallel"],
        }

    datasets = sorted(records.values(), key=lambda item: (item["status"], item["name"]))
    groups = {
        status: [item["id"] for item in datasets if item["status"] == status]
        for status in ("approved", "review", "blocked")
    }
    groups["training_candidates"] = [
        item["id"]
        for item in datasets
        if any("training" in role or "discovery" in role for role in item["roles"])
    ]
    groups["evaluation_only"] = [
        item["id"]
        for item in datasets
        if any("gate" in role or "evaluation" in role or "stress" in role for role in item["roles"])
    ]
    return {
        "schema_version": 1,
        "merge_policy": "virtual_registry_only",
        "datasets": datasets,
        "groups": groups,
        "guardrail": (
            "Grouping does not grant permission. Only approved datasets may be opened, "
            "copied, transformed, trained on, or benchmarked."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "datasets.toml")
    parser.add_argument(
        "--expansion", type=Path, default=ROOT / "configs/data-expansion.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "configs/datasets.grouped.json"
    )
    args = parser.parse_args()
    payload = merge_registries(args.baseline, args.expansion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
