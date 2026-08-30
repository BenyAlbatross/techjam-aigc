import csv
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import scripts.data_manifest as data_manifest
from scripts.data_manifest import build_ntire, validate_manifest


FIXTURES = Path("tests/fixtures")


def fixture_payload(tmp_path: Path) -> dict:
    samples = []
    for label, name in enumerate(("real.ppm", "ai.ppm")):
        shutil.copy(FIXTURES / name, tmp_path / name)
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        samples.append({
            "sample_id": name, "base_id": name, "label": label,
            "truth": "ai" if label else "real", "path": name,
            "sha256": digest, "source_family": "fixture",
            "generator_family": "fixture", "license": "CC0-1.0",
        })
    return {"dataset": "fixture", "revision": "1", "samples": samples}


def test_fixture_manifest_validates(tmp_path: Path):
    payload = fixture_payload(tmp_path)
    assert validate_manifest(payload, tmp_path) == []
    payload["samples"][1]["sha256"] = "0" * 64
    assert "hash mismatch" in " ".join(validate_manifest(payload, tmp_path))


def test_manifest_validation_rejects_invalid_rows(tmp_path: Path):
    payload = fixture_payload(tmp_path)
    payload["samples"][1]["sample_id"] = "real.ppm"
    assert "duplicate or empty sample_id" in " ".join(validate_manifest(payload, tmp_path))

    payload = fixture_payload(tmp_path)
    payload["samples"][1]["label"] = 2
    assert "invalid label" in " ".join(validate_manifest(payload, tmp_path))

    payload = fixture_payload(tmp_path)
    payload["samples"][1]["path"] = "/tmp/ai.ppm"
    assert "path must stay relative" in " ".join(validate_manifest(payload, tmp_path))

    payload = fixture_payload(tmp_path)
    payload["samples"][1]["license"] = ""
    assert "missing rights" in " ".join(validate_manifest(payload, tmp_path))


def test_manifest_validation_rejects_absent_file_and_cross_label_hash(tmp_path: Path):
    payload = fixture_payload(tmp_path)
    payload["samples"][1]["path"] = "real.ppm"
    payload["samples"][1]["sha256"] = hashlib.sha256((tmp_path / "real.ppm").read_bytes()).hexdigest()
    payload["samples"].append({
        **payload["samples"][0], "sample_id": "missing.ppm", "path": "missing.ppm",
    })
    errors = " ".join(validate_manifest(payload, tmp_path))
    assert "absent file" in errors
    assert "same bytes have conflicting labels" in errors


def test_ntire_builder_blocks_review_dataset_before_shard_access(tmp_path: Path):
    with pytest.raises(ValueError, match="ntire_2026_train: status is review"):
        build_ntire(tmp_path / "absent", tmp_path / "manifest.json")


def test_approved_ntire_builder_sorts_rows_and_hashes_local_files(tmp_path: Path):
    shard = tmp_path / "shard"
    images = shard / "images"
    images.mkdir(parents=True)
    shutil.copy(FIXTURES / "real.ppm", images / "z.ppm")
    shutil.copy(FIXTURES / "ai.ppm", images / "a.ppm")
    with (shard / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows([{"image": "z.ppm", "label": "0"}, {"image": "a.ppm", "label": "1"}])

    output = tmp_path / "manifest.json"
    assert data_manifest._build_ntire(shard, output) == output
    payload = json.loads(output.read_text())
    assert [row["sample_id"] for row in payload["samples"]] == ["a.ppm", "z.ppm"]
    assert [row["path"] for row in payload["samples"]] == ["images/a.ppm", "images/z.ppm"]
    assert validate_manifest(payload, shard) == []


@pytest.mark.parametrize("name", ["/tmp/escape.ppm", "../escape.ppm", "link.ppm"])
def test_ntire_builder_rejects_paths_outside_images(tmp_path: Path, name: str):
    shard = tmp_path / "shard"
    images = shard / "images"
    images.mkdir(parents=True)
    outside = tmp_path / "escape.ppm"
    shutil.copy(FIXTURES / "real.ppm", outside)
    if name == "link.ppm":
        (images / name).symlink_to(outside)
    with (shard / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerow({"image": name, "label": "0"})
    with pytest.raises(ValueError, match="path must stay under images"):
        data_manifest._build_ntire(shard, tmp_path / "manifest.json")


def test_sid_duplicate_bytes_do_not_fill_class_quota(tmp_path: Path, monkeypatch):
    rows = [
        {"id": "real-1", "label": 0, "image": {"bytes": b"real", "path": "one.jpg"}},
        {"id": "real-2", "label": 0, "image": {"bytes": b"real", "path": "two.jpg"}},
        {"id": "ai-1", "label": 1, "image": {"bytes": b"ai", "path": "three.jpg"}},
        {"id": "ai-2", "label": 1, "image": {"bytes": b"other-ai", "path": "four.jpg"}},
    ]
    monkeypatch.setattr(data_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(data_manifest, "_dataset", lambda name: {
        "repository": "fixture", "revision": "1", "split": "validation",
        "license": "CC0-1.0", "excluded_labels": [2],
    })
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        Image=lambda decode: object(), load_dataset=lambda *args, **kwargs: FakeRows(rows),
    ))
    with pytest.raises(ValueError, match="SID_Set lacks 2 samples for each class"):
        data_manifest.build_sid(tmp_path / "manifest.json", 2)


def test_sid_conflicting_label_bytes_are_rejected(tmp_path: Path, monkeypatch):
    rows = [
        {"id": "real", "label": 0, "image": {"bytes": b"same", "path": "one.jpg"}},
        {"id": "ai", "label": 1, "image": {"bytes": b"same", "path": "two.jpg"}},
    ]
    monkeypatch.setattr(data_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(data_manifest, "_dataset", lambda name: {
        "repository": "fixture", "revision": "1", "split": "validation",
        "license": "CC0-1.0", "excluded_labels": [2],
    })
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        Image=lambda decode: object(), load_dataset=lambda *args, **kwargs: FakeRows(rows),
    ))
    with pytest.raises(ValueError, match="conflicting labels"):
        data_manifest.build_sid(tmp_path / "manifest.json", 1)


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def cast_column(self, name, image):
        return self.rows


def test_ntire_cli_blocks_unapproved_dataset_before_reading_files(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "scripts/data_manifest.py", "build-ntire", str(tmp_path), "--output", str(tmp_path / "out.json")],
        text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "ntire_2026_train: status is review" in result.stderr + result.stdout
