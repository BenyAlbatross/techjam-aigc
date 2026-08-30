import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import scripts.benchmark as benchmark
from scripts.benchmark import CONDITIONS
from scripts.benchmark import SEED
from scripts.benchmark import apply_condition
from scripts.benchmark import run_panel
from scripts.benchmark import score_with_backoff
from scripts.benchmark import validate_shard
from scripts.benchmark import write_shard


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
MODEL_REVISION = "60e82406916921b823616bee33397baab38af3f0"
MODEL_HASH = "ab0b8cad7462a047ff4e2888cb4f11b1abe568d73ce07a2649a3f1541f73675f"
DATASET_REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
REQUIRED_ROW_FIELDS = {
    "identity", "model", "model_revision", "model_hash", "dataset",
    "dataset_revision", "sample_id", "base_id", "content_hash", "label",
    "cohort", "condition", "condition_parameters", "raw_score",
    "probability_ai", "threshold", "decision", "device",
    "effective_batch_size", "elapsed_seconds", "git_commit", "config_hash",
    "seed", "attempted_count", "valid_count", "excluded_count",
    "per_class_counts",
}


def test_conditions_preserve_canonical_geometry_and_randomness():
    image = Image.new("RGB", (100, 80), (10, 20, 30))

    assert len(CONDITIONS) == 15
    assert apply_condition(image, "resize_0.25", "x").size == image.size
    assert apply_condition(image, "center_crop_80", "x").size == (80, 64)
    first = np.asarray(apply_condition(image, "noise_sigma0.05", "x"))
    second = np.asarray(apply_condition(image, "noise_sigma0.05", "x"))
    different_sample = np.asarray(
        apply_condition(image, "noise_sigma0.05", "different")
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_sample)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition_output_is_class_symmetric_for_shared_sample_id(condition: str):
    real_row_image = Image.new("RGB", (20, 16), (40, 80, 120))
    ai_row_image = real_row_image.copy()

    real = np.asarray(apply_condition(real_row_image, condition, "shared"))
    ai = np.asarray(apply_condition(ai_row_image, condition, "shared"))

    assert np.array_equal(real, ai)


def test_unknown_condition_fails():
    with pytest.raises(ValueError, match="unknown"):
        apply_condition(Image.new("RGB", (2, 2)), "unknown", "x")


class OOMAdapter:
    def score_batch(self, images):
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("fixture limit")
        return [(float(index), index / 10) for index, _ in enumerate(images)]


def test_score_with_backoff_retries_whole_input_at_smaller_batches():
    images = [Image.new("RGB", (1, 1)) for _ in range(5)]

    scores, effective_size = score_with_backoff(OOMAdapter(), images, 8)

    assert effective_size == 2
    assert scores == [(0.0, 0.0), (1.0, 0.1), (0.0, 0.0), (1.0, 0.1), (0.0, 0.0)]


def _valid_shard_row(content_hash: str = "b" * 64, panel_count: int = 1) -> dict:
    row = {
        "model": "detector",
        "model_revision": MODEL_REVISION,
        "model_hash": MODEL_HASH,
        "dataset": "sid_set",
        "dataset_revision": DATASET_REVISION,
        "sample_id": "opaque-sample",
        "base_id": "opaque-base",
        "content_hash": content_hash,
        "label": 0,
        "cohort": "fixture-real",
        "condition": "clean",
        "condition_parameters": {},
        "raw_score": -0.5,
        "probability_ai": 0.25,
        "threshold": 0.5,
        "decision": 0,
        "device": "cpu",
        "effective_batch_size": 2,
        "elapsed_seconds": 0.01,
        "git_commit": "c" * 40,
        "config_hash": "d" * 64,
        "seed": SEED,
        "attempted_count": panel_count,
        "valid_count": panel_count,
        "excluded_count": 0,
        "per_class_counts": {"0": panel_count, "1": 0},
    }
    identity_values = [
        row["model_revision"], row["model_hash"], row["dataset_revision"],
        row["content_hash"], row["condition"], row["git_commit"],
        row["config_hash"],
    ]
    encoded = json.dumps(
        identity_values, sort_keys=True, separators=(",", ":")
    ).encode()
    row["identity"] = hashlib.sha256(encoded).hexdigest()
    return row


def test_write_and_validate_shard_are_atomic_and_exact(tmp_path: Path):
    path = tmp_path / "nested/predictions.jsonl"
    rows = [
        _valid_shard_row("a" * 64, 2),
        _valid_shard_row("b" * 64, 2),
    ]
    expected = {row["identity"] for row in rows}

    write_shard(path, rows)

    assert not path.with_suffix(".jsonl.tmp").exists()
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    assert validate_shard(path, expected) == []


@pytest.mark.parametrize(
    ("rows", "expected", "message"),
    [
        ([_valid_shard_row(panel_count=2)], {"a", "b"}, "row count"),
        (
            [
                _valid_shard_row(panel_count=2),
                {**_valid_shard_row(panel_count=2), "raw_score": 0.5},
            ],
            {"a", "b"},
            "conflicting duplicate identity",
        ),
    ],
)
def test_validate_shard_rejects_partial_or_conflicting_rows(
    tmp_path: Path, rows: list[dict], expected: set[str], message: str
):
    path = tmp_path / "predictions.jsonl"
    write_shard(path, rows)

    assert any(message in error for error in validate_shard(path, expected))


def test_validate_shard_rejects_every_missing_required_field(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    for field in REQUIRED_ROW_FIELDS:
        row = _valid_shard_row()
        expected = {row["identity"]}
        del row[field]
        write_shard(path, [row])

        errors = validate_shard(path, expected)

        assert any(field in error for error in errors), (field, errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "forged-revision"),
        ("content_hash", "e" * 64),
        ("condition_parameters", {"quality": 90}),
        ("raw_score", float("inf")),
        ("probability_ai", float("nan")),
        ("probability_ai", 1.01),
        ("threshold", -0.01),
        ("decision", 1),
        ("effective_batch_size", 0),
        ("elapsed_seconds", -0.01),
        ("seed", 0),
        ("excluded_count", 1),
        ("per_class_counts", {"0": 2, "1": 1}),
    ],
)
def test_validate_shard_rejects_forged_identity_metadata_and_invalid_values(
    tmp_path: Path, field: str, value
):
    path = tmp_path / "predictions.jsonl"
    row = _valid_shard_row()
    expected = {row["identity"]}
    row[field] = value
    write_shard(path, [row])

    assert validate_shard(path, expected), field


def test_validate_shard_rejects_conflicting_shard_metadata(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    first = _valid_shard_row("a" * 64, 2)
    second = _valid_shard_row("b" * 64, 2)
    second["model"] = "different-detector"
    rows = [first, second]
    write_shard(path, rows)

    errors = validate_shard(path, {row["identity"] for row in rows})

    assert any("conflicting model" in error for error in errors)


def test_validate_shard_requires_valid_count_to_match_expected_rows(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    row = _valid_shard_row(panel_count=2)
    write_shard(path, [row])

    errors = validate_shard(path, {row["identity"]})

    assert any("valid_count" in error for error in errors)


class DummyAdapter:
    name = "ateeqq_siglip"
    revision = MODEL_REVISION
    weight_sha256 = MODEL_HASH
    threshold = 0.5

    def __init__(self):
        self.calls = 0

    def score_batch(self, images):
        self.calls += 1
        results = []
        for image in images:
            probability = float(np.asarray(image, dtype=float)[:, :, 0].mean() / 255)
            results.append((probability * 2 - 1, probability))
        return results


class FailingAdapter(DummyAdapter):
    def score_batch(self, images):
        raise ValueError("adapter exploded")


class MalformedAdapter(DummyAdapter):
    def score_batch(self, images):
        return [(0.1,) for image in images]


def _manifest(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    images = work / "data/sid_set/images"
    images.mkdir(parents=True)
    samples = []
    for label, fixture in enumerate(("real.ppm", "ai.ppm")):
        data = (FIXTURES / fixture).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        path = images / fixture
        path.write_bytes(data)
        samples.append({
            "sample_id": f"sample-{label}",
            "base_id": f"base-{label}",
            "label": label,
            "truth": "ai" if label else "real",
            "path": f"data/sid_set/images/{fixture}",
            "sha256": digest,
            "source_family": "fixture-real" if not label else "fixture-ai",
            "generator_family": "" if not label else "fixture-generator",
            "license": "CC0-1.0",
        })
    manifest = work / "manifests/panel.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "dataset": "sid_set",
        "revision": DATASET_REVISION,
        "samples": samples,
    }))
    return manifest


def test_run_panel_writes_full_metadata_and_resumes_without_rescoring(tmp_path: Path):
    manifest = _manifest(tmp_path)
    adapter = DummyAdapter()
    output = tmp_path / "predictions"
    kwargs = {
        "model_names": [adapter.name],
        "dataset_name": "sid_set",
        "manifest_path": manifest,
        "conditions": ["clean", "jpeg_q90"],
        "device": "cpu",
        "batch_size": 2,
        "output": output,
        "adapter_loader": lambda name, device, cache: adapter,
    }

    paths = run_panel(**kwargs)
    calls_after_first_run = adapter.calls
    resumed_paths = run_panel(**kwargs)

    assert resumed_paths == paths
    assert adapter.calls == calls_after_first_run
    assert len(paths) == 2
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 2
    assert REQUIRED_ROW_FIELDS <= rows[0].keys()
    assert rows[0]["seed"] == SEED
    assert rows[0]["attempted_count"] == rows[0]["valid_count"] == 2
    assert rows[0]["excluded_count"] == 0
    assert rows[0]["per_class_counts"] == {"0": 1, "1": 1}
    assert rows[0]["decision"] == int(
        rows[0]["probability_ai"] >= rows[0]["threshold"]
    )

    paths[0].write_text(paths[0].read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(RuntimeError, match="invalid existing shard"):
        run_panel(**kwargs)


def test_run_panel_wraps_transform_failure_with_opaque_context(
    tmp_path: Path, monkeypatch
):
    manifest = _manifest(tmp_path)

    def fail_second_transform(image, condition, sample_id):
        if sample_id == "sample-1":
            raise ValueError("transform exploded")
        return image.copy()

    monkeypatch.setattr(benchmark, "apply_condition", fail_second_transform)
    with pytest.raises(RuntimeError) as captured:
        run_panel(
            model_names=["ateeqq_siglip"],
            dataset_name="sid_set",
            manifest_path=manifest,
            conditions=["clean"],
            device="cpu",
            batch_size=2,
            output=tmp_path / "predictions",
            adapter_loader=lambda *args: DummyAdapter(),
        )

    message = str(captured.value)
    assert all(value in message for value in (
        "ateeqq_siglip", "sid_set", "clean", "sample-1",
    ))
    assert str(manifest) not in message
    assert isinstance(captured.value.__cause__, ValueError)


@pytest.mark.parametrize("adapter", [FailingAdapter(), MalformedAdapter()])
def test_run_panel_wraps_adapter_and_malformed_score_failures_with_context(
    tmp_path: Path, adapter
):
    manifest = _manifest(tmp_path)

    with pytest.raises(RuntimeError) as captured:
        run_panel(
            model_names=["ateeqq_siglip"],
            dataset_name="sid_set",
            manifest_path=manifest,
            conditions=["clean"],
            device="cpu",
            batch_size=2,
            output=tmp_path / "predictions",
            adapter_loader=lambda *args: adapter,
        )

    message = str(captured.value)
    assert all(value in message for value in (
        "ateeqq_siglip", "sid_set", "clean", "sample-0", "sample-1",
    ))
    assert str(manifest) not in message
    assert isinstance(captured.value.__cause__, ValueError)


def test_run_panel_blocks_unapproved_dataset_before_reading_manifest(tmp_path: Path):
    with pytest.raises(RuntimeError, match="status is review"):
        run_panel(
            model_names=["ateeqq_siglip"],
            dataset_name="ntire_2026_train",
            manifest_path=tmp_path / "absent.json",
            conditions=["clean"],
            device="cpu",
            batch_size=1,
            output=tmp_path / "predictions",
            adapter_loader=lambda *args: pytest.fail("model loaded before compliance"),
        )


def test_run_panel_rejects_duplicate_prediction_identity_before_scoring(tmp_path: Path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    duplicate = dict(payload["samples"][0])
    duplicate["sample_id"] = "duplicate-bytes"
    duplicate["base_id"] = "duplicate-base"
    payload["samples"].append(duplicate)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="duplicate prediction identity"):
        run_panel(
            model_names=["ateeqq_siglip"],
            dataset_name="sid_set",
            manifest_path=manifest,
            conditions=["clean"],
            device="cpu",
            batch_size=2,
            output=tmp_path / "predictions",
            adapter_loader=lambda *args: pytest.fail("duplicate reached scoring"),
        )


def test_run_panel_fails_when_corrupt_images_exceed_point_one_percent(
    tmp_path: Path, capsys
):
    work = tmp_path / "work"
    images = work / "data/sid_set/images"
    images.mkdir(parents=True)
    good = (FIXTURES / "real.ppm").read_bytes()
    bad = b"not an image"
    (images / "good.ppm").write_bytes(good)
    (images / "bad.bin").write_bytes(bad)
    rows = []
    for index in range(1000):
        corrupt = index < 2
        data = bad if corrupt else good
        rows.append({
            "sample_id": f"opaque-{index}",
            "base_id": f"base-{index}",
            "label": 0,
            "path": f"data/sid_set/images/{'bad.bin' if corrupt else 'good.ppm'}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_family": "fixture",
            "generator_family": "",
            "license": "CC0-1.0",
        })
    manifest = work / "manifests/panel.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "dataset": "sid_set", "revision": DATASET_REVISION, "samples": rows,
    }))

    with pytest.raises(RuntimeError, match="2 invalid images exceed limit 1"):
        run_panel(
            model_names=["ateeqq_siglip"],
            dataset_name="sid_set",
            manifest_path=manifest,
            conditions=["clean"],
            device="cpu",
            batch_size=32,
            output=tmp_path / "predictions",
            adapter_loader=lambda *args: pytest.fail("model loaded after invalid panel"),
        )

    stderr = capsys.readouterr().err
    assert "opaque-0" in stderr and "opaque-1" in stderr
    assert "bad.bin" not in stderr
