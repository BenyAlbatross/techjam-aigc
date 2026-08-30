#!/usr/bin/env -S uv run
"""Run the reproducible AIGC feature-robustness experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from techjam_aigc.feature_lab.data import resolve_repo_root
from techjam_aigc.feature_lab.evaluation import write_evaluation_cache
from techjam_aigc.feature_lab.pipeline import ExperimentConfig, run_extraction
from techjam_aigc.feature_lab.registry import DEFAULT_FEATURE_PROFILE, FEATURE_PROFILES
from techjam_aigc.feature_lab.transforms import TRANSFORM_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/feature_lab"),
        help="Cache directory relative to the repository root.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help=(
            "Optional local index or expansion selection CSV, relative to the "
            "repository root. Defaults to data/samples/index.csv."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--transform-profile",
        choices=TRANSFORM_PROFILES,
        default="core",
        help=(
            "Core is the unchanged default. Other profiles add only their bounded "
            "factorial, directed-pair, realistic, or covering-bank conditions."
        ),
    )
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Feature schema: frozen_v1 reproduces the original 53-feature registry; expanded_v2 adds 29 candidates.",
    )
    parser.add_argument(
        "--evaluate-final-confirmation",
        action="store_true",
        help=(
            "Explicit one-time opt-in to touch sealed final_confirmation rows. "
            "Use only after features, transforms, thresholds, fusion, and hyperparameters are frozen."
        ),
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse output/features.csv.gz; only recompute evaluation tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    output_dir = args.output if args.output.is_absolute() else repo_root / args.output
    feature_path = output_dir / "features.csv.gz"
    started = time.perf_counter()
    if args.reuse_features:
        if not feature_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing cache: {feature_path}")
        features = pd.read_csv(feature_path)
        print(f"Reused {len(features):,} cached feature rows.")
    else:
        config = ExperimentConfig(
            bootstrap_repetitions=args.bootstrap,
            transform_profile=args.transform_profile,
            feature_profile=args.feature_profile,
            evaluate_final_confirmation=args.evaluate_final_confirmation,
        )
        features, binary_index, metadata = run_extraction(
            repo_root,
            output_dir,
            index_path=args.index,
            config=config,
            workers=max(1, args.workers),
        )
        print(
            f"Extracted {len(features):,} rows from "
            f"{len(binary_index):,} binary parent images with transform profile "
            f"{args.transform_profile!r}, feature profile "
            f"{args.feature_profile!r}, and wrote {output_dir}."
        )
        print(f"Semantic control: {metadata['semantic_control']['status']}.")

    tables = write_evaluation_cache(
        output_dir,
        features,
        repetitions=args.bootstrap,
        seed=20260830,
        min_group_images=20,
    )
    decisions = tables["decision_ledger"]["decision"].value_counts().to_dict()
    elapsed = time.perf_counter() - started
    print(f"Evaluation complete in {elapsed:.1f}s. Decisions: {decisions}")
    print("Important: WildFake generator confirmation groups are underpowered; see coverage.csv.")


if __name__ == "__main__":
    main()
