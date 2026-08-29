from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import expanded
from aidetector.evaluation import compute_metrics


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sid_validation_40"
MANIFEST = DATA_DIR / "manifest.json"
RESULTS = ROOT / "expanded-results"
EXTERNAL = ROOT.parent / "dalle3-eval-samples" / "dalle3-eval"
TARGET_PER_CLASS = 40
SEED = 20260829
CONDITIONS = (
    "clean",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_sigma0.5",
    "blur_sigma1",
    "blur_sigma2",
    "resize_0.5",
    "resize_0.25",
    "noise_sigma0.02",
    "noise_sigma0.05",
    "noise_sigma0.10",
    "color_jitter_20",
    "center_crop_80",
)


def rng_for(sample_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def transformed(image: Image.Image, condition: str, sample_id: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if condition == "clean":
        return image.copy()
    if condition.startswith("jpeg_q"):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(condition.removeprefix("jpeg_q")))
        buffer.seek(0)
        with Image.open(buffer) as result:
            return result.convert("RGB").copy()
    if condition.startswith("blur_sigma"):
        radius = float(condition.removeprefix("blur_sigma"))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if condition.startswith("resize_"):
        scale = float(condition.removeprefix("resize_"))
        width, height = image.size
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if condition.startswith("noise_sigma"):
        sigma = float(condition.removeprefix("noise_sigma")) * 255
        values = np.asarray(image, dtype=np.uint8)
        noise = rng_for(sample_id).normal(0.0, sigma, values.shape)
        return Image.fromarray(
            np.clip(values.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        )
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
    raise ValueError(condition)


def prepare_sid() -> list[dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]

    from datasets import Image as HFImage
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("saberzl/SID_Set", split="validation", streaming=True)
    dataset = dataset.cast_column("image", HFImage(decode=False))
    counts = {0: 0, 1: 0}
    samples = []
    for row_index, row in enumerate(dataset):
        label = int(row["label"])
        if label not in counts or counts[label] >= TARGET_PER_CLASS:
            continue
        payload = row["image"]
        data = payload.get("bytes")
        if data is None:
            data = Path(payload["path"]).read_bytes()
        image_id = str(row["img_id"])
        filename = f"{label}_{counts[label]:02d}_{image_id}.img"
        path = DATA_DIR / filename
        path.write_bytes(data)
        samples.append(
            {
                "sample_id": image_id,
                "source_row": row_index,
                "label": label,
                "truth": "ai" if label else "real",
                "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        counts[label] += 1
        if counts == {0: TARGET_PER_CLASS, 1: TARGET_PER_CLASS}:
            break
    if counts != {0: TARGET_PER_CLASS, 1: TARGET_PER_CLASS}:
        raise RuntimeError(f"Incomplete SID sample: {counts}")
    payload = {
        "dataset": "saberzl/SID_Set",
        "revision": "dc03ead57929879319ce30a82bfcfb8d317b10bd",
        "split": "validation",
        "license": "CC BY 4.0",
        "selection": "first 40 label-0 and first 40 label-1 rows; label-2 excluded",
        "samples": samples,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return samples


def run_sid() -> dict:
    samples = prepare_sid()
    detector = expanded.CandidateDetector("ateeqq_siglip")
    metrics, predictions = {}, []
    started = time.perf_counter()
    for condition in CONDITIONS:
        labels, scores = [], []
        condition_started = time.perf_counter()
        for sample in samples:
            with Image.open(DATA_DIR / sample["filename"]) as source:
                image = transformed(source, condition, sample["sample_id"])
            score, raw = detector.predict(image)
            labels.append(sample["label"])
            scores.append(score)
            predictions.append(
                {
                    "condition": condition,
                    "sample_id": sample["sample_id"],
                    "truth": sample["truth"],
                    "probability_ai": score,
                    "raw_score": raw,
                    "pred": "ai" if score >= detector.threshold else "real",
                }
            )
        metric = compute_metrics(
            labels,
            scores,
            threshold=detector.threshold,
            dataset=f"SID_Set validation 40+40:{condition}",
        ).as_dict()
        metric["seconds"] = time.perf_counter() - condition_started
        metrics[condition] = metric
        print(
            f"{condition:18s} BA={metric['balanced_accuracy']:.3f} "
            f"AUC={metric['roc_auc']:.3f} "
            f"FP={metric['confusion']['false_positive']} "
            f"FN={metric['confusion']['false_negative']}",
            flush=True,
        )
    result = {
        "model": "ateeqq_siglip",
        "model_info": detector.info(),
        "dataset": "saberzl/SID_Set validation",
        "sample_count": len(samples),
        "fine_tuning": False,
        "conditions": metrics,
        "predictions": predictions,
        "seconds_total": time.perf_counter() - started,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "confirmation_sid_40x2.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_external() -> dict:
    detector = expanded.CandidateDetector("ateeqq_siglip")
    generators = {"3": "DALL-E 3", "mj": "Midjourney", "sdxl": "SDXL"}
    results, predictions = {}, []
    for folder, generator in generators.items():
        paths = sorted((EXTERNAL / folder).glob("[0-9]_0.png"))
        if len(paths) != 10:
            raise RuntimeError(f"Expected 10 {generator} files, found {len(paths)}")
        scores = []
        for path in paths:
            with Image.open(path) as image:
                score, raw = detector.predict(image.convert("RGB"))
            scores.append(score)
            predictions.append(
                {
                    "generator": generator,
                    "sample_id": path.stem,
                    "probability_ai": score,
                    "raw_score": raw,
                    "pred": "ai" if score >= detector.threshold else "real",
                }
            )
        results[generator] = {
            "n": len(scores),
            "ai_recall_at_published_threshold": sum(
                score >= detector.threshold for score in scores
            )
            / len(scores),
            "mean_probability_ai": float(np.mean(scores)),
            "min_probability_ai": min(scores),
            "max_probability_ai": max(scores),
        }
        print(generator, results[generator], flush=True)
    payload = {
        "model": "ateeqq_siglip",
        "source": "openai/dalle3-eval-samples",
        "source_license": "MIT repository license",
        "scope": "synthetic recall only; no real comparator",
        "fine_tuning": False,
        "results": results,
        "predictions": predictions,
    }
    (RESULTS / "external_generator_recall.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def self_test() -> None:
    image = Image.new("RGB", (100, 80), (128, 64, 32))
    assert len(CONDITIONS) == 15
    assert transformed(image, "resize_0.25", "a").size == image.size
    assert transformed(image, "center_crop_80", "a").size == (80, 64)
    first = np.asarray(transformed(image, "noise_sigma0.05", "a"))
    second = np.asarray(transformed(image, "noise_sigma0.05", "a"))
    assert np.array_equal(first, second)
    assert compute_metrics(
        [0, 1], [0.1, 0.9], threshold=0.5, dataset="self-test"
    ).balanced_accuracy == 1.0
    print("confirmation self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("self-test", "sid", "external"))
    args = parser.parse_args()
    {"self-test": self_test, "sid": run_sid, "external": run_external}[args.action]()


if __name__ == "__main__":
    main()
