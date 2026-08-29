from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SID_DIR = ROOT / "data" / "sid"
EXTERNAL_DIR = ROOT / "data" / "external"
RESULTS_DIR = ROOT / "results"
THRESHOLD = 0.5
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
        return image.filter(
            ImageFilter.GaussianBlur(float(condition.removeprefix("blur_sigma")))
        )
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


class Detector:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        ).to(self.device)
        self.model.eval()
        labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        self.ai_index = next(index for index, label in labels.items() if label == "ai")

    def score(self, image: Image.Image) -> float:
        values = self.processor(images=image, return_tensors="pt")
        values = {key: value.to(self.device) for key, value in values.items()}
        with torch.inference_mode():
            logits = self.model(**values).logits[0]
        return float(torch.softmax(logits, dim=-1)[self.ai_index].cpu())


def metrics(labels: list[int], scores: list[float]) -> dict:
    predictions = [int(score >= THRESHOLD) for score in scores]
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    real_specificity = tn / (tn + fp)
    ai_recall = tp / (tp + fn)

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    order = np.argsort(scores_array, kind="stable")
    ranks = np.empty(len(scores_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores_array[order[end]] == scores_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = int(labels_array.sum())
    negatives = len(labels) - positives
    auc = (ranks[labels_array == 1].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return {
        "n": len(labels),
        "balanced_accuracy": float((real_specificity + ai_recall) / 2),
        "roc_auc": float(auc),
        "confusion": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        },
    }


def run_sid(detector: Detector) -> dict:
    manifest = json.loads((SID_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    condition_metrics, predictions = {}, []
    for condition in CONDITIONS:
        labels, scores = [], []
        started = time.perf_counter()
        for sample in samples:
            with Image.open(SID_DIR / sample["filename"]) as source:
                image = transformed(source, condition, sample["sample_id"])
            score = detector.score(image)
            labels.append(sample["label"])
            scores.append(score)
            predictions.append(
                {
                    "condition": condition,
                    "sample_id": sample["sample_id"],
                    "truth": sample["truth"],
                    "probability_ai": score,
                    "pred": "ai" if score >= THRESHOLD else "real",
                }
            )
        result = metrics(labels, scores)
        result["seconds"] = time.perf_counter() - started
        condition_metrics[condition] = result
        confusion = result["confusion"]
        print(
            f"{condition:18s} BA={result['balanced_accuracy']:.3f} "
            f"AUC={result['roc_auc']:.3f} FP={confusion['false_positive']} "
            f"FN={confusion['false_negative']}",
            flush=True,
        )
    return {
        "dataset": "saberzl/SID_Set validation",
        "license": "CC BY 4.0",
        "sample_count": len(samples),
        "conditions": condition_metrics,
        "predictions": predictions,
    }


def run_external(detector: Detector) -> dict:
    generators = {
        "dalle3": "DALL-E 3",
        "midjourney": "Midjourney",
        "sdxl": "SDXL",
    }
    results, predictions = {}, []
    for folder, generator in generators.items():
        paths = sorted((EXTERNAL_DIR / folder).glob("*.png"))
        if len(paths) != 10:
            raise RuntimeError(f"Expected 10 {generator} images, found {len(paths)}")
        scores = []
        for path in paths:
            with Image.open(path) as image:
                score = detector.score(image.convert("RGB"))
            scores.append(score)
            predictions.append(
                {
                    "generator": generator,
                    "sample_id": path.stem,
                    "probability_ai": score,
                    "pred": "ai" if score >= THRESHOLD else "real",
                }
            )
        results[generator] = {
            "n": len(scores),
            "ai_recall": sum(score >= THRESHOLD for score in scores) / len(scores),
            "mean_probability_ai": float(np.mean(scores)),
            "min_probability_ai": min(scores),
            "max_probability_ai": max(scores),
        }
        print(generator, results[generator], flush=True)
    return {
        "source": "openai/dalle3-eval-samples",
        "source_revision": "d7c88c07b492ad7b9fd3003126d00719a2edabb1",
        "source_license": "MIT repository license",
        "scope": "synthetic recall only; no real comparator",
        "results": results,
        "predictions": predictions,
    }


def environment(device: str) -> dict:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "transformers": __import__("transformers").__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
    }


def run(device: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    detector = Detector(device)
    started = time.perf_counter()
    sid = run_sid(detector)
    external = run_external(detector)
    metadata = environment(device)
    metadata["seconds_total"] = time.perf_counter() - started
    (RESULTS_DIR / "sid.json").write_text(json.dumps(sid, indent=2), encoding="utf-8")
    (RESULTS_DIR / "external.json").write_text(
        json.dumps(external, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def self_test() -> None:
    image = Image.new("RGB", (100, 80), (128, 64, 32))
    assert len(CONDITIONS) == 15
    assert transformed(image, "resize_0.25", "a").size == image.size
    assert transformed(image, "center_crop_80", "a").size == (80, 64)
    assert np.array_equal(
        transformed(image, "noise_sigma0.05", "a"),
        transformed(image, "noise_sigma0.05", "a"),
    )
    perfect = metrics([0, 1], [0.1, 0.9])
    assert perfect["balanced_accuracy"] == perfect["roc_auc"] == 1.0
    assert metrics([0, 1], [0.9, 0.1])["roc_auc"] == 0.0
    metadata = environment("cpu")
    assert "torchvision" in metadata
    assert "cuda_capability" in metadata
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("self-test", "run"))
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    self_test() if args.action == "self-test" else run(args.device)


if __name__ == "__main__":
    main()
