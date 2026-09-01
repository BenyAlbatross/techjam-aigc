#!/usr/bin/env python3
"""Evaluate only validation rows from an ablation score table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from summarize_trace_rx_m_ablations import sliced_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    if "split" not in frame or "test" in set(frame["split"].astype(str)):
        raise ValueError("Ablation score table is missing split provenance or contains test rows.")
    validation = frame[frame["split"].eq("val")].reset_index(drop=True)
    report = {
        "selection_split": "val",
        "test_split_opened": False,
        "rows": len(validation),
        "metrics": sliced_metrics(validation),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
