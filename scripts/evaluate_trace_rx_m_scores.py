#!/usr/bin/env python3
"""Evaluate a frozen TRACE-RX-M score CSV with the always-report metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from techjam_aigc.trace_rx_m.evaluation import (
    ALWAYS_REPORT_METRICS,
    binary_detection_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-column", default="logit")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Decision threshold in the score column's scale (default: 0.0 for logits).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scores)
    required = {"target", args.score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")

    report = {
        "always_report_metrics": list(ALWAYS_REPORT_METRICS),
        "positive_class": "AIGC",
        "score_column": args.score_column,
        "metrics": binary_detection_metrics(
            frame["target"].to_numpy(),
            frame[args.score_column].to_numpy(),
            threshold=args.threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
