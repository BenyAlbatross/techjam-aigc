import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

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


def test_ntire_builder_sorts_rows_and_hashes_local_files(tmp_path: Path):
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
    assert build_ntire(shard, output) == output
    payload = json.loads(output.read_text())
    assert [row["sample_id"] for row in payload["samples"]] == ["a.ppm", "z.ppm"]
    assert [row["path"] for row in payload["samples"]] == ["images/a.ppm", "images/z.ppm"]
    assert validate_manifest(payload, shard) == []


def test_ntire_cli_blocks_unapproved_dataset_before_reading_files(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "scripts/data_manifest.py", "build-ntire", str(tmp_path), "--output", str(tmp_path / "out.json")],
        text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "ntire_2026_train: status is review" in result.stderr + result.stdout
