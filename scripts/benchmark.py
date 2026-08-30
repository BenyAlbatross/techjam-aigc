"""Deterministic, compliance-gated robustness inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

if __package__:
    from scripts.compliance import check_datasets, check_models, load_registry
    from scripts.data_manifest import validate_manifest
    from scripts.model_adapters import load_model
else:
    from compliance import check_datasets, check_models, load_registry
    from data_manifest import validate_manifest
    from model_adapters import load_model


ROOT = Path(__file__).resolve().parents[1]
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
REQUIRED_ROW_FIELDS = {
    "identity", "model", "model_revision", "model_hash", "dataset",
    "dataset_revision", "sample_id", "base_id", "content_hash", "label",
    "cohort", "condition", "condition_parameters", "raw_score",
    "probability_ai", "threshold", "decision", "device",
    "effective_batch_size", "elapsed_seconds", "git_commit", "config_hash",
    "seed", "attempted_count", "valid_count", "excluded_count",
    "per_class_counts",
}


def _sample_seed(sample_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(sample_id: str) -> np.random.Generator:
    return np.random.default_rng(_sample_seed(sample_id))


def _condition_parameters(condition: str, sample_id: str) -> dict:
    if condition == "clean":
        return {}
    if condition.startswith("jpeg_q"):
        return {"quality": int(condition.removeprefix("jpeg_q"))}
    if condition.startswith("blur_sigma"):
        return {"sigma": float(condition.removeprefix("blur_sigma"))}
    if condition.startswith("resize_"):
        return {
            "scale": float(condition.removeprefix("resize_")),
            "resampling": "bicubic",
        }
    if condition.startswith("noise_sigma"):
        return {
            "sigma": float(condition.removeprefix("noise_sigma")),
            "sample_seed": _sample_seed(sample_id),
        }
    if condition == "color_jitter_20":
        factors = rng_for(sample_id).choice((0.8, 1.2), size=3)
        return {
            "brightness": float(factors[0]),
            "contrast": float(factors[1]),
            "color": float(factors[2]),
            "sample_seed": _sample_seed(sample_id),
        }
    if condition == "center_crop_80":
        return {"fraction": 0.8}
    raise ValueError(f"unknown condition: {condition}")


def apply_condition(
    image: Image.Image, condition: str, sample_id: str
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    parameters = _condition_parameters(condition, sample_id)
    if condition == "clean":
        return image.copy()
    if condition.startswith("jpeg_q"):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=parameters["quality"])
        buffer.seek(0)
        with Image.open(buffer) as result:
            return result.convert("RGB").copy()
    if condition.startswith("blur_sigma"):
        return image.filter(ImageFilter.GaussianBlur(parameters["sigma"]))
    if condition.startswith("resize_"):
        width, height = image.size
        scale = parameters["scale"]
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if condition.startswith("noise_sigma"):
        values = np.asarray(image, dtype=np.uint8)
        noise = rng_for(sample_id).normal(
            0.0, parameters["sigma"] * 255, values.shape
        )
        return Image.fromarray(
            np.clip(values.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        )
    if condition == "color_jitter_20":
        result = ImageEnhance.Brightness(image).enhance(parameters["brightness"])
        result = ImageEnhance.Contrast(result).enhance(parameters["contrast"])
        return ImageEnhance.Color(result).enhance(parameters["color"])
    if condition == "center_crop_80":
        width, height = image.size
        crop_width, crop_height = round(width * 0.8), round(height * 0.8)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))
    raise AssertionError("validated condition was not applied")


def score_with_backoff(adapter, images, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not images:
        return [], 0
    size = min(batch_size, len(images))
    while size >= 1:
        try:
            scored = []
            for start in range(0, len(images), size):
                scored.extend(adapter.score_batch(images[start : start + size]))
            return scored, size
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            torch.cuda.empty_cache()
            size = max(1, size // 2)
    raise AssertionError("unreachable")


def write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def validate_shard(path: Path, expected: set[str]) -> list[str]:
    if not path.is_file():
        return [f"missing shard: {path}"]
    errors, rows = [], []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    errors.append(f"line {number}: empty row")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"line {number}: invalid JSON: {error.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"line {number}: row must be an object")
                    continue
                rows.append(row)
                errors.extend(
                    f"line {number}: {error}"
                    for error in _prediction_row_errors(row)
                )
    except OSError as error:
        return [f"cannot read shard: {error}"]

    if len(rows) != len(expected):
        errors.append(f"row count {len(rows)} does not match expected {len(expected)}")
    if rows:
        for field in (
            "model", "model_revision", "model_hash", "dataset",
            "dataset_revision", "condition", "threshold", "device",
            "effective_batch_size", "elapsed_seconds", "git_commit",
            "config_hash", "seed", "attempted_count", "valid_count",
            "excluded_count", "per_class_counts",
        ):
            if any(row.get(field) != rows[0].get(field) for row in rows[1:]):
                errors.append(f"conflicting {field} metadata")
        for number, row in enumerate(rows, 1):
            if row.get("valid_count") != len(expected):
                errors.append(
                    f"line {number}: valid_count does not match expected row count"
                )
    identities: dict[str, dict] = {}
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, str) or not identity:
            errors.append("missing identity")
            continue
        if identity in identities:
            kind = (
                "conflicting duplicate identity"
                if row != identities[identity]
                else "duplicate identity"
            )
            errors.append(f"{kind}: {identity}")
        else:
            identities[identity] = row
    actual = set(identities)
    missing, unexpected = expected - actual, actual - expected
    if missing:
        errors.append(f"missing identities: {sorted(missing)}")
    if unexpected:
        errors.append(f"unexpected identities: {sorted(unexpected)}")
    return errors


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value: object, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _prediction_row_errors(row: dict) -> list[str]:
    missing = REQUIRED_ROW_FIELDS - row.keys()
    errors = [f"missing required field: {field}" for field in sorted(missing)]

    for field in (
        "model", "model_revision", "dataset", "dataset_revision", "sample_id",
        "base_id", "cohort", "device",
    ):
        if field in row and (not isinstance(row[field], str) or not row[field]):
            errors.append(f"{field} must be a non-empty string")
    for field in ("identity", "model_hash", "content_hash", "config_hash"):
        if field in row and not _is_hex(row[field], {64}):
            errors.append(f"{field} must be a SHA-256 hex string")
    if "git_commit" in row and not _is_hex(row["git_commit"], {40, 64}):
        errors.append("git_commit must be a Git object hex string")

    condition = row.get("condition")
    sample_id = row.get("sample_id")
    if condition not in CONDITIONS:
        errors.append("condition must be canonical")
    if "condition_parameters" in row:
        if not isinstance(row["condition_parameters"], dict):
            errors.append("condition_parameters must be an object")
        elif condition in CONDITIONS and isinstance(sample_id, str) and sample_id:
            if row["condition_parameters"] != _condition_parameters(condition, sample_id):
                errors.append("condition_parameters do not match condition")

    label = row.get("label")
    if "label" in row and (not _is_int(label) or label not in (0, 1)):
        errors.append("label must be 0 or 1")
    raw_score = row.get("raw_score")
    if "raw_score" in row and (
        not _is_number(raw_score) or not math.isfinite(raw_score)
    ):
        errors.append("raw_score must be finite")
    probability = row.get("probability_ai")
    if "probability_ai" in row and (
        not _is_number(probability)
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        errors.append("probability_ai must be finite and within [0, 1]")
    threshold = row.get("threshold")
    if "threshold" in row and (
        not _is_number(threshold)
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        errors.append("threshold must be finite and within [0, 1]")
    decision = row.get("decision")
    if "decision" in row and (not _is_int(decision) or decision not in (0, 1)):
        errors.append("decision must be 0 or 1")
    if (
        _is_number(probability)
        and math.isfinite(probability)
        and _is_number(threshold)
        and math.isfinite(threshold)
        and _is_int(decision)
        and decision != int(probability >= threshold)
    ):
        errors.append("decision does not match probability_ai and threshold")

    effective_size = row.get("effective_batch_size")
    if "effective_batch_size" in row and (
        not _is_int(effective_size) or effective_size < 1
    ):
        errors.append("effective_batch_size must be positive")
    elapsed = row.get("elapsed_seconds")
    if "elapsed_seconds" in row and (
        not _is_number(elapsed) or not math.isfinite(elapsed) or elapsed < 0
    ):
        errors.append("elapsed_seconds must be finite and non-negative")
    if "seed" in row and (not _is_int(row["seed"]) or row["seed"] != SEED):
        errors.append(f"seed must equal {SEED}")

    for field, minimum in (
        ("attempted_count", 1), ("valid_count", 1), ("excluded_count", 0),
    ):
        if field in row and (not _is_int(row[field]) or row[field] < minimum):
            errors.append(f"{field} must be an integer >= {minimum}")
    if all(field in row and _is_int(row[field]) for field in (
        "attempted_count", "valid_count", "excluded_count"
    )) and row["valid_count"] + row["excluded_count"] != row["attempted_count"]:
        errors.append("attempted_count must equal valid_count plus excluded_count")
    counts = row.get("per_class_counts")
    if "per_class_counts" in row:
        valid_counts = (
            isinstance(counts, dict)
            and set(counts) == {"0", "1"}
            and all(_is_int(value) and value >= 0 for value in counts.values())
        )
        if not valid_counts:
            errors.append("per_class_counts must contain non-negative 0 and 1 counts")
        elif _is_int(row.get("valid_count")) and sum(counts.values()) != row["valid_count"]:
            errors.append("per_class_counts must sum to valid_count")

    identity_fields = (
        "model_revision", "model_hash", "dataset_revision", "content_hash",
        "condition", "git_commit", "config_hash",
    )
    if "identity" in row and all(field in row for field in identity_fields):
        recomputed = _hash_json([row[field] for field in identity_fields])
        if row["identity"] != recomputed:
            errors.append("identity does not match row metadata")
    return errors


def _config_hash(model_entry: dict, dataset_entry: dict) -> str:
    return _hash_json({
        "model": model_entry,
        "dataset": dataset_entry,
        "conditions": CONDITIONS,
        "seed": SEED,
    })


def _identity(
    model_entry: dict,
    dataset_revision: str,
    content_hash: str,
    condition: str,
    git_commit: str,
    config_hash: str,
) -> str:
    return _hash_json([
        model_entry["revision"],
        model_entry["sha256"],
        dataset_revision,
        content_hash,
        condition,
        git_commit,
        config_hash,
    ])


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _approved_entries(
    model_names: list[str], dataset_name: str
) -> tuple[dict, dict]:
    models = load_registry(ROOT / "models.toml", "models")
    datasets = load_registry(ROOT / "datasets.toml", "datasets")
    errors = check_models(models, model_names, "benchmark")
    errors.extend(check_datasets(datasets, [dataset_name]))
    if errors:
        raise RuntimeError("\n".join(errors))
    return models, datasets


def _load_manifest(manifest_path: Path, dataset_name: str, dataset_entry: dict):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset_name:
        raise RuntimeError("manifest dataset does not match selected dataset")
    if payload.get("revision") != dataset_entry["revision"]:
        raise RuntimeError("manifest revision does not match approved dataset")
    errors = validate_manifest(payload, manifest_path.parent.parent)
    if errors:
        raise RuntimeError("invalid manifest:\n" + "\n".join(errors))
    return payload


def _decode_samples(payload: dict, data_root: Path) -> tuple[list[dict], list[Image.Image]]:
    valid_rows, images, invalid_ids = [], [], []
    for row in payload["samples"]:
        try:
            with Image.open(data_root / row["path"]) as source:
                source.load()
                image = source.copy()
        except (OSError, ValueError):
            invalid_ids.append(str(row["sample_id"]))
            print(f"{row['sample_id']}: corrupt or unsupported image", file=sys.stderr)
            continue
        valid_rows.append(row)
        images.append(image)
    limit = math.floor(0.001 * len(payload["samples"]))
    if len(invalid_ids) > limit:
        raise RuntimeError(
            f"{len(invalid_ids)} invalid images exceed limit {limit}"
        )
    return valid_rows, images


def run_panel(
    model_names: list[str],
    dataset_name: str,
    manifest_path: Path,
    conditions: list[str],
    device: str,
    batch_size: int,
    output: Path,
    cache: Path = Path("work/hf-cache"),
    *,
    adapter_loader: Callable = load_model,
) -> list[Path]:
    models, datasets = _approved_entries(model_names, dataset_name)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if not conditions:
        raise ValueError("at least one condition is required")
    payload = _load_manifest(Path(manifest_path), dataset_name, datasets[dataset_name])
    sample_rows, base_images = _decode_samples(payload, Path(manifest_path).parent.parent)
    attempted = len(payload["samples"])
    valid = len(sample_rows)
    counts = {
        str(label): sum(row["label"] == label for row in sample_rows)
        for label in (0, 1)
    }
    git_commit = _git_commit()
    paths = []

    for model_name in model_names:
        model_entry = models[model_name]
        config_hash = _config_hash(model_entry, datasets[dataset_name])
        expected_by_condition = {
            condition: {
                _identity(
                    model_entry,
                    payload["revision"],
                    row["sha256"],
                    condition,
                    git_commit,
                    config_hash,
                )
                for row in sample_rows
            }
            for condition in conditions
        }
        if any(len(expected) != valid for expected in expected_by_condition.values()):
            raise RuntimeError("manifest contains duplicate prediction identity")
        pending = []
        for condition in conditions:
            path = Path(output) / model_name / dataset_name / f"{condition}.jsonl"
            paths.append(path)
            if path.exists():
                errors = validate_shard(path, expected_by_condition[condition])
                if errors:
                    raise RuntimeError(
                        f"invalid existing shard {path}: " + "; ".join(errors)
                    )
            else:
                pending.append((condition, path))
        if not pending:
            continue

        adapter = adapter_loader(model_name, device, Path(cache))
        if (
            adapter.revision != model_entry["revision"]
            or adapter.weight_sha256 != model_entry["sha256"]
            or float(adapter.threshold) != float(model_entry["threshold"])
        ):
            raise RuntimeError(f"{model_name}: adapter metadata conflicts with registry")

        for condition, path in pending:
            transformed = []
            for image, sample in zip(base_images, sample_rows, strict=True):
                try:
                    transformed.append(
                        apply_condition(image, condition, sample["sample_id"])
                    )
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as error:
                    raise RuntimeError(
                        f"{model_name}/{dataset_name}/{condition}: "
                        f"transformation failed for sample ID {sample['sample_id']}"
                    ) from error
            started = time.perf_counter()
            try:
                scores, effective_size = score_with_backoff(
                    adapter, transformed, batch_size
                )
                if len(scores) != valid:
                    raise ValueError(
                        f"returned {len(scores)} scores for {valid} images"
                    )
                normalized_scores = []
                for score in scores:
                    if not isinstance(score, (tuple, list)) or len(score) != 2:
                        raise ValueError("adapter score must be a (raw_score, p_ai) pair")
                    raw_score, probability_ai = map(float, score)
                    if not math.isfinite(raw_score) or not 0.0 <= probability_ai <= 1.0:
                        raise ValueError("adapter returned an invalid score")
                    normalized_scores.append((raw_score, probability_ai))
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                sample_ids = ", ".join(str(row["sample_id"]) for row in sample_rows)
                raise RuntimeError(
                    f"{model_name}/{dataset_name}/{condition}: "
                    f"scoring failed for sample IDs [{sample_ids}]"
                ) from error
            elapsed = time.perf_counter() - started
            rows = []
            for sample, (raw_score, probability_ai) in zip(
                sample_rows, normalized_scores, strict=True
            ):
                cohort = sample.get("generator_family") or sample.get("source_family", "")
                rows.append({
                    "identity": _identity(
                        model_entry,
                        payload["revision"],
                        sample["sha256"],
                        condition,
                        git_commit,
                        config_hash,
                    ),
                    "model": model_name,
                    "model_revision": adapter.revision,
                    "model_hash": adapter.weight_sha256,
                    "dataset": dataset_name,
                    "dataset_revision": payload["revision"],
                    "sample_id": sample["sample_id"],
                    "base_id": sample.get("base_id", sample["sample_id"]),
                    "content_hash": sample["sha256"],
                    "label": sample["label"],
                    "cohort": cohort,
                    "condition": condition,
                    "condition_parameters": _condition_parameters(
                        condition, sample["sample_id"]
                    ),
                    "raw_score": raw_score,
                    "probability_ai": probability_ai,
                    "threshold": float(adapter.threshold),
                    "decision": int(probability_ai >= adapter.threshold),
                    "device": device,
                    "effective_batch_size": effective_size,
                    "elapsed_seconds": elapsed,
                    "git_commit": git_commit,
                    "config_hash": config_hash,
                    "seed": SEED,
                    "attempted_count": attempted,
                    "valid_count": valid,
                    "excluded_count": attempted - valid,
                    "per_class_counts": counts,
                })
            write_shard(path, rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("work/predictions"))
    parser.add_argument("--cache", type=Path, default=Path("work/hf-cache"))
    args = parser.parse_args()
    conditions = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")
    paths = run_panel(
        args.models.split(","),
        args.dataset,
        args.manifest,
        conditions,
        args.device,
        args.batch_size,
        args.output,
        args.cache,
    )
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
