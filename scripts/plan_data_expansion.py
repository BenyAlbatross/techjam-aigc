#!/usr/bin/env -S uv run
"""Plan a deterministic, license-gated dataset expansion without downloading data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from techjam_aigc.feature_lab.data import resolve_repo_root
from techjam_aigc.feature_lab.expansion import (
    acquisition_audit,
    load_expansion_config,
    plan_diversity_cohorts,
    select_expansion_rows,
    write_expansion_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV metadata manifest; no image bytes are read.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data-expansion.json"),
        help="Expansion policy relative to the repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/data_expansion"),
        help="Audit artifact directory relative to the repository root.",
    )
    parser.add_argument("--few-generators", type=int, default=10)
    parser.add_argument("--many-generators", type=int, default=100)
    parser.add_argument("--diversity-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_repo_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output if args.output.is_absolute() else root / args.output
    config = load_expansion_config(config_path)
    manifest = pd.read_csv(manifest_path)
    selected = select_expansion_rows(manifest, config)

    diversity = None
    if (selected.get("source_id") == "community_forensics_diversity").any():
        diversity = plan_diversity_cohorts(
            selected,
            few_generators=args.few_generators,
            many_generators=args.many_generators,
            total_images=args.diversity_images,
            seed=str(config["selection_seed"]),
        )
    write_expansion_artifacts(output_dir, manifest, selected, config, diversity)
    gate = acquisition_audit(config)
    blocked = gate[gate["selected"] & ~gate["acquisition_allowed"]]["source_id"].tolist()
    print(f"Planned {len(selected):,} rows and wrote dry-run artifacts to {output_dir}.")
    if blocked:
        print(f"Acquisition remains blocked pending complete allowlisting: {', '.join(blocked)}")
    else:
        print("All selected sources pass the acquisition audit; this planner still performs no downloads.")


if __name__ == "__main__":
    main()
