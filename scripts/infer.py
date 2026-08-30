"""Exact competition submission inference from a local image directory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image, ImageOps

if __package__:
    from scripts.model_adapters import load_model
else:
    from model_adapters import load_model


BATCH_SIZE = 32


def infer_directory(adapter, input_dir: Path) -> tuple[list[dict], list[str]]:
    try:
        root = input_dir.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"input directory does not exist: {input_dir}") from error
    if not root.is_dir():
        raise ValueError(f"input directory is not a directory: {input_dir}")

    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    decoded: list[tuple[Path, Image.Image]] = []
    invalid: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        try:
            with Image.open(path) as image:
                decoded.append(
                    (relative, ImageOps.exif_transpose(image).convert("RGB").copy())
                )
        except (OSError, ValueError):
            opaque_path = relative.as_posix()
            invalid.append(opaque_path)
            print(f"corrupt image: {opaque_path}", file=sys.stderr)

    rows: list[dict] = []
    for start in range(0, len(decoded), BATCH_SIZE):
        batch = decoded[start : start + BATCH_SIZE]
        scores = adapter.score_batch([image for _, image in batch])
        if len(scores) != len(batch):
            raise ValueError("adapter returned the wrong number of scores")
        for (relative, _), score in zip(batch, scores, strict=True):
            try:
                _, probability_ai = score
                probability_ai = float(probability_ai)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid adapter score for {relative.as_posix()}"
                ) from error
            if not math.isfinite(probability_ai) or not 0.0 <= probability_ai <= 1.0:
                raise ValueError(
                    f"AI probability for {relative.as_posix()} must be finite and in [0, 1]"
                )
            rows.append({"image_path": relative.as_posix(), "pred": probability_ai})
    return rows, invalid


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", type=Path, default=Path("work/hf-cache"))
    args = parser.parse_args()

    adapter = load_model(args.model, args.device, args.cache)
    rows, invalid = infer_directory(adapter, args.input)
    _write_json(args.output, rows)
    if args.diagnostics:
        _write_json(
            args.diagnostics,
            {
                "attempted_count": len(rows) + len(invalid),
                "valid_count": len(rows),
                "invalid_image_paths": invalid,
            },
        )


if __name__ == "__main__":
    main()
