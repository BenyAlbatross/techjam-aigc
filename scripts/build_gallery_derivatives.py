"""Materialize a small ignored gallery cache with benchmark-exact transforms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

CONDITIONS = (
    "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "blur_sigma0.5", "blur_sigma1", "blur_sigma2", "resize_0.5", "resize_0.25",
    "noise_sigma0.02", "noise_sigma0.05", "noise_sigma0.10", "color_jitter_20",
    "center_crop_80",
)
SEED = 20260829


def _sample_seed(sample_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(sample_id: str) -> np.random.Generator:
    return np.random.default_rng(_sample_seed(sample_id))


def apply_condition(image: Image.Image, condition: str, sample_id: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if condition.startswith("jpeg_q"):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(condition.removeprefix("jpeg_q")))
        buffer.seek(0)
        with Image.open(buffer) as result:
            return result.convert("RGB").copy()
    if condition.startswith("blur_sigma"):
        return image.filter(ImageFilter.GaussianBlur(float(condition.removeprefix("blur_sigma"))))
    if condition.startswith("resize_"):
        width, height = image.size
        scale = float(condition.removeprefix("resize_"))
        small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if condition.startswith("noise_sigma"):
        values = np.asarray(image, dtype=np.uint8)
        noise = rng_for(sample_id).normal(0.0, float(condition.removeprefix("noise_sigma")) * 255, values.shape)
        return Image.fromarray(np.clip(values.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    if condition == "color_jitter_20":
        factors = rng_for(sample_id).choice((0.8, 1.2), size=3)
        result = ImageEnhance.Brightness(image).enhance(float(factors[0]))
        result = ImageEnhance.Contrast(result).enhance(float(factors[1]))
        return ImageEnhance.Color(result).enhance(float(factors[2]))
    if condition == "center_crop_80":
        width, height = image.size
        crop_width, crop_height = round(width * 0.8), round(height * 0.8)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))
    return image.copy()

ROOT = Path(__file__).resolve().parents[1]


def build(manifest_path: Path, output: Path, limit: int) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest_path.parent.parent
    written = 0
    for sample in payload["samples"][:limit]:
        source = (data_root / sample["path"]).resolve()
        sample_root = output / sample["sample_id"]
        sample_root.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            deterministic_id = sample.get("source_sample_id", sample["sample_id"])
            for condition in CONDITIONS:
                if condition == "clean":
                    continue
                destination = sample_root / f"{condition}.png"
                if destination.exists():
                    continue
                transformed = apply_condition(image, condition, deterministic_id)
                transformed.save(destination, format="PNG", compress_level=3)
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "work/manifests/sid_set_1000x2_canonical.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work/app-gallery")
    parser.add_argument("--limit", type=int, default=72)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    print(f"wrote {build(args.manifest, args.output, args.limit)} derivatives")


if __name__ == "__main__":
    main()
