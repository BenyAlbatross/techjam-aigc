from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from aidetector.evaluation import compute_metrics
from aidetector.model import create_detector


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sid_validation"
OUTPUT_DIR = ROOT / "results"
MANIFEST_PATH = DATA_DIR / "manifest.json"
TARGET_PER_CLASS = 12
SEED = 20260829

MODELS = {
    "univfd": {"backend": "univfd", "parameters": 428_000_000, "license": "MIT"},
    "steganograph": {
        "backend": "hf",
        "hf_model": "delpot/steganograph-ia-detector",
        "parameters": 85_800_000,
        "license": "MIT",
    },
    "capcheck": {
        "backend": "hf",
        "hf_model": "capcheck/ai-image-detection",
        "parameters": 85_800_000,
        "license": "Apache-2.0",
    },
}

CONDITIONS = (
    "clean",
    "jpeg_q30",
    "blur_sigma2",
    "resize_0.25",
    "noise_sigma0.10",
    "color_jitter_20",
    "center_crop_80",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def suffix_for(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")


def prepare_sample() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["samples"]

    from datasets import Image as HFImage
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("saberzl/SID_Set", split="validation", streaming=True)
    dataset = dataset.cast_column("image", HFImage(decode=False))
    counts = {0: 0, 1: 0}
    samples: list[dict] = []

    for row_index, row in enumerate(dataset):
        label = int(row["label"])
        if label not in counts or counts[label] >= TARGET_PER_CLASS:
            continue
        payload = row["image"]
        data = payload.get("bytes")
        if data is None:
            source_path = payload.get("path")
            if not source_path:
                raise RuntimeError(f"No image bytes or path for row {row_index}")
            data = Path(source_path).read_bytes()

        image_id = str(row["img_id"])
        filename = f"{label}_{counts[label]:02d}_{safe_name(image_id)}{suffix_for(data)}"
        path = DATA_DIR / filename
        path.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = image.format
        samples.append(
            {
                "sample_id": image_id,
                "source_row": row_index,
                "label": label,
                "truth": "ai" if label else "real",
                "path": str(path.resolve()),
                "width": width,
                "height": height,
                "format": image_format,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        counts[label] += 1
        if all(count == TARGET_PER_CLASS for count in counts.values()):
            break

    if counts != {0: TARGET_PER_CLASS, 1: TARGET_PER_CLASS}:
        raise RuntimeError(f"Could not build balanced sample: {counts}")

    manifest = {
        "dataset": "saberzl/SID_Set",
        "dataset_revision_observed": "dc03ead57929879319ce30a82bfcfb8d317b10bd",
        "split": "validation",
        "license": "CC BY 4.0",
        "selection": "first 12 label-0 and first 12 label-1 rows; label-2 excluded",
        "seed": SEED,
        "samples": samples,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return samples


def stable_rng(sample_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def transform(image: Image.Image, condition: str, sample_id: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if condition == "clean":
        return image.copy()
    if condition == "jpeg_q30":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=30)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if condition == "blur_sigma2":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "resize_0.25":
        width, height = image.size
        small = image.resize(
            (max(1, round(width * 0.25)), max(1, round(height * 0.25))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if condition == "noise_sigma0.10":
        values = np.asarray(image, dtype=np.uint8)
        noise = stable_rng(sample_id).normal(0.0, 25.5, size=values.shape)
        return Image.fromarray(np.clip(values.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    if condition == "color_jitter_20":
        rng = stable_rng(sample_id)
        factors = rng.choice((0.8, 1.2), size=3)
        result = ImageEnhance.Brightness(image).enhance(float(factors[0]))
        result = ImageEnhance.Contrast(result).enhance(float(factors[1]))
        return ImageEnhance.Color(result).enhance(float(factors[2]))
    if condition == "center_crop_80":
        width, height = image.size
        crop_width, crop_height = max(1, round(width * 0.8)), max(1, round(height * 0.8))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))
    raise ValueError(f"Unknown condition: {condition}")


def load_image(sample: dict) -> Image.Image:
    with Image.open(sample["path"]) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def jpeg_blockiness(image: Image.Image) -> float:
    gray = ImageOps.grayscale(image)
    gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    diffs = []
    for axis in (0, 1):
        delta = np.abs(np.diff(values, axis=axis))
        indices = np.arange(delta.shape[axis])
        boundary = (indices + 1) % 8 == 0
        if not boundary.any() or boundary.all():
            continue
        boundary_values = np.take(delta, np.flatnonzero(boundary), axis=axis)
        inner_values = np.take(delta, np.flatnonzero(~boundary), axis=axis)
        diffs.append(float(boundary_values.mean() - inner_values.mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def high_frequency_ratio(image: Image.Image) -> float:
    gray = ImageOps.grayscale(image).resize((256, 256), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32) / 255.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    mask = np.ones_like(spectrum, dtype=bool)
    mask[96:160, 96:160] = False
    return float(spectrum[mask].sum() / max(spectrum.sum(), 1e-12))


def metadata_keyword(image: Image.Image) -> float:
    fields = [str(image.info)]
    try:
        fields.append(str(dict(image.getexif())))
    except Exception:
        pass
    text = " ".join(fields).lower()
    tokens = ("stable diffusion", "midjourney", "dall-e", "comfyui", "automatic1111", "ai generated")
    return float(any(token in text for token in tokens))


def run_heuristics(samples: list[dict]) -> dict:
    rows = []
    for sample in samples:
        with Image.open(sample["path"]) as native:
            width, height = native.size
            row = {
                **sample,
                "scores": {
                    "square_aspect": -abs(math.log(width / height)),
                    "exact_1024_square": float(width == height == 1024),
                    "metadata_ai_keyword": metadata_keyword(native),
                    "low_jpeg_blockiness": -jpeg_blockiness(native.convert("RGB")),
                    "high_frequency_energy": high_frequency_ratio(native.convert("RGB")),
                },
            }
        rows.append(row)

    metrics = {}
    labels = [row["label"] for row in rows]
    for name in rows[0]["scores"]:
        scores = [row["scores"][name] for row in rows]
        metric = compute_metrics(labels, scores, threshold=0.5, dataset=f"heuristic:{name}")
        metrics[name] = {
            "roc_auc": metric.roc_auc,
            "note": "AUROC only is decision-relevant; threshold 0.5 is not calibrated for continuous heuristics.",
        }
    result = {"metrics": metrics, "predictions": rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "heuristics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def detector_for(name: str):
    config = MODELS[name]
    kwargs = {"device": "cpu", "threshold": 0.5}
    if "hf_model" in config:
        kwargs["hf_model"] = config["hf_model"]
    return create_detector(config["backend"], **kwargs)


def run_model(name: str, samples: list[dict]) -> dict:
    started = time.perf_counter()
    detector = detector_for(name)
    model_info = detector.model_info()
    if hasattr(detector, "model"):
        commit = getattr(detector.model.config, "_commit_hash", None)
        if commit:
            model_info["resolved_commit"] = commit

    metrics_by_condition = {}
    predictions = []
    for condition in CONDITIONS:
        condition_started = time.perf_counter()
        labels, scores, predicted = [], [], []
        for sample in samples:
            clean = load_image(sample)
            altered = transform(clean, condition, sample["sample_id"])
            result = detector.predict_image(altered)
            labels.append(sample["label"])
            scores.append(result.probability_ai)
            predicted.append(1 if result.probability_ai >= 0.5 else 0)
            predictions.append(
                {
                    "model": name,
                    "condition": condition,
                    "sample_id": sample["sample_id"],
                    "truth": sample["truth"],
                    "probability_ai": result.probability_ai,
                    "pred": "ai" if predicted[-1] else "real",
                }
            )
        elapsed = time.perf_counter() - condition_started
        metric = compute_metrics(
            labels,
            scores,
            predicted,
            threshold=0.5,
            dataset=f"SID_Set validation spike:{condition}",
            seconds=elapsed,
        )
        metrics_by_condition[condition] = metric.as_dict()
        print(
            f"{name:14} {condition:18} "
            f"BA={metric.balanced_accuracy:.3f} AUC={metric.roc_auc:.3f} "
            f"FP={metric.false_positive} FN={metric.false_negative}",
            flush=True,
        )

    result = {
        "model": name,
        "declared": MODELS[name],
        "resolved": model_info,
        "sample_count": len(samples),
        "threshold": 0.5,
        "fine_tuning": False,
        "conditions": metrics_by_condition,
        "predictions": predictions,
        "seconds_total": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def self_test() -> None:
    image = Image.new("RGB", (100, 80), (120, 130, 140))
    assert transform(image, "center_crop_80", "x").size == (80, 64)
    assert transform(image, "resize_0.25", "x").size == image.size
    first = np.asarray(transform(image, "noise_sigma0.10", "x"))
    second = np.asarray(transform(image, "noise_sigma0.10", "x"))
    assert np.array_equal(first, second)
    perfect = compute_metrics([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
    assert perfect.roc_auc == 1.0 and perfect.balanced_accuracy == 1.0
    assert all(MODELS[name]["parameters"] < 2_000_000_000 for name in MODELS)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disposable zero-training TechJam baseline spike")
    parser.add_argument("action", choices=("self-test", "prepare", "heuristics", "model"))
    parser.add_argument("--model", choices=tuple(MODELS))
    args = parser.parse_args()

    if args.action == "self-test":
        self_test()
        return
    samples = prepare_sample()
    if args.action == "prepare":
        print(f"prepared {len(samples)} samples at {DATA_DIR}")
    elif args.action == "heuristics":
        run_heuristics(samples)
    elif args.action == "model":
        if not args.model:
            parser.error("--model is required for action=model")
        run_model(args.model, samples)


if __name__ == "__main__":
    main()
