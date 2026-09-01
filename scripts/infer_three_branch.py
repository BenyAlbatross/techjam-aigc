#!/usr/bin/env python3
"""Run final three-branch inference for an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from techjam_aigc.three_branch.inference import (
    load_final_detector,
    predict_directory,
    write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/three-branch.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/three-branch-techjam2026-v2/three-branch-final.pt"),
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path("artifacts/three-branch-techjam2026-v2/dual-memory.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_final_detector(
        args.config.resolve(),
        args.memory.resolve(),
        args.checkpoint.resolve(),
        device=device,
    )
    records = predict_directory(
        model,
        config,
        args.image_dir,
        device=device,
        batch_size=args.batch_size,
    )
    write_predictions(records, args.output)
    print(f"Wrote {len(records)} predictions to {args.output}")


if __name__ == "__main__":
    main()
