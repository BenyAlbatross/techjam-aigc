from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.cleanup import SAFE_TARGETS, cleanup_targets, delete_targets
from scripts.infer import infer_directory


ROOT = Path(__file__).resolve().parents[1]


class DummyAdapter:
    def score_batch(self, images):
        return [(float(index), 0.25 + index / 10) for index in range(len(images))]


class InvalidScoreAdapter:
    def __init__(self, probability):
        self.probability = probability

    def score_batch(self, images):
        return [(0.0, self.probability) for _ in images]


def copy_fixture(path: Path) -> None:
    shutil.copy(ROOT / "tests/fixtures/real.ppm", path)


def test_submission_rows_have_exact_schema_and_order(tmp_path):
    copy_fixture(tmp_path / "image.ppm")

    rows, invalid = infer_directory(DummyAdapter(), tmp_path)

    assert invalid == []
    assert list(rows[0]) == ["image_path", "pred"]
    assert rows == [{"image_path": "image.ppm", "pred": 0.25}]


def test_inference_recursively_sorts_relative_posix_paths(tmp_path):
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    copy_fixture(tmp_path / "z/first.ppm")
    copy_fixture(tmp_path / "a/second.ppm")
    copy_fixture(tmp_path / "root.ppm")

    rows, invalid = infer_directory(DummyAdapter(), tmp_path)

    assert invalid == []
    assert [row["image_path"] for row in rows] == [
        "a/second.ppm",
        "root.ppm",
        "z/first.ppm",
    ]
    assert [row["pred"] for row in rows] == [0.25, 0.35, 0.45]


def test_corrupt_file_is_reported_by_opaque_relative_path(tmp_path, capsys):
    copy_fixture(tmp_path / "good.ppm")
    (tmp_path / "bad.bin").write_bytes(b"not an image")

    rows, invalid = infer_directory(DummyAdapter(), tmp_path)

    assert rows == [{"image_path": "good.ppm", "pred": 0.25}]
    assert invalid == ["bad.bin"]
    stderr = capsys.readouterr().err
    assert "bad.bin" in stderr
    assert str(tmp_path) not in stderr


@pytest.mark.parametrize("probability", [math.nan, math.inf, -0.01, 1.01])
def test_inference_rejects_nonfinite_or_out_of_range_scores(tmp_path, probability):
    copy_fixture(tmp_path / "image.ppm")

    with pytest.raises(ValueError, match="AI probability"):
        infer_directory(InvalidScoreAdapter(probability), tmp_path)


def test_cleanup_targets_are_exact_and_preview_does_not_delete(tmp_path):
    for relative in SAFE_TARGETS:
        target = tmp_path / relative
        target.mkdir(parents=True)
        (target / "x").write_text("x", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/cleanup.py"),
            "preview",
            "--root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    planned = cleanup_targets(tmp_path)
    assert planned == [tmp_path / relative for relative in SAFE_TARGETS]
    assert all(target.exists() for target in planned)
    inventory = json.loads((tmp_path / "work/cleanup-inventory.json").read_text())
    assert [item["path"] for item in inventory["targets"]] == [
        relative.as_posix() for relative in SAFE_TARGETS
    ]
    assert all(item["file_count"] == 1 for item in inventory["targets"])
    assert "preview" in result.stdout.lower()


def test_delete_targets_is_confined_and_attestation_survives(tmp_path):
    target = tmp_path / "work/data"
    target.mkdir(parents=True)
    (target / "x").write_text("x", encoding="utf-8")

    delete_targets(tmp_path, cleanup_targets(tmp_path))

    assert not target.exists()
    attestation = tmp_path / "outputs/data-deletion-attestation.json"
    assert attestation.is_file()
    record = json.loads(attestation.read_text())
    assert [item["path"] for item in record["deleted_targets"]] == ["work/data"]


@pytest.mark.parametrize(
    "forged",
    [
        Path("work"),
        Path("work/data/child"),
        Path("outputs"),
    ],
)
def test_delete_rejects_forged_non_safe_or_broad_targets(tmp_path, forged):
    target = tmp_path / forged
    target.mkdir(parents=True)

    with pytest.raises(ValueError):
        delete_targets(tmp_path, [target])

    assert target.exists()


def test_cleanup_rejects_absent_root(tmp_path):
    absent = tmp_path / "absent"

    with pytest.raises(ValueError, match="root"):
        cleanup_targets(absent)


def test_delete_rejects_unresolved_safe_target(tmp_path):
    with pytest.raises(ValueError, match="exist"):
        delete_targets(tmp_path, [tmp_path / "work/data"])


def test_cleanup_and_delete_reject_safe_target_symlink_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep", encoding="utf-8")
    (root / "work").mkdir()
    (root / "work/data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|escape"):
        cleanup_targets(root)
    with pytest.raises(ValueError, match="symlink|escape"):
        delete_targets(root, [root / "work/data"])

    assert (outside / "keep").read_text() == "keep"


def test_delete_is_anchored_when_safe_target_parent_is_swapped(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    target = root / "work/data"
    target.mkdir(parents=True)
    (target / "inside").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    (outside / "data").mkdir(parents=True)
    (outside / "data/keep").write_text("keep", encoding="utf-8")
    real_rmtree = shutil.rmtree

    def swap_parent_then_delete(path, *args, **kwargs):
        (root / "work").rename(root / "original-work")
        (root / "work").symlink_to(outside, target_is_directory=True)
        return real_rmtree(path, *args, **kwargs)

    swap_parent_then_delete.avoids_symlink_attacks = True
    monkeypatch.setattr(shutil, "rmtree", swap_parent_then_delete)

    delete_targets(root, [target])

    assert (outside / "data/keep").read_text() == "keep"
    assert not (root / "original-work/data").exists()


def test_delete_fails_closed_when_target_is_swapped_to_symlink(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    target = root / "work/data"
    target.mkdir(parents=True)
    (target / "inside").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep", encoding="utf-8")
    real_rmtree = shutil.rmtree

    def swap_target_then_delete(path, *args, **kwargs):
        target.rename(root / "work/original-data")
        target.symlink_to(outside, target_is_directory=True)
        return real_rmtree(path, *args, **kwargs)

    swap_target_then_delete.avoids_symlink_attacks = True
    monkeypatch.setattr(shutil, "rmtree", swap_target_then_delete)

    with pytest.raises(OSError):
        delete_targets(root, [target])

    assert (outside / "keep").read_text() == "keep"
    assert (root / "work/original-data/inside").read_text() == "inside"


def test_delete_fails_closed_without_symlink_safe_rmtree(tmp_path, monkeypatch):
    target = tmp_path / "work/data"
    target.mkdir(parents=True)
    (target / "inside").write_text("inside", encoding="utf-8")
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)

    with pytest.raises(RuntimeError, match="symlink-safe"):
        delete_targets(tmp_path, [target])

    assert (target / "inside").read_text() == "inside"


def test_preview_never_follows_predictable_temp_or_final_symlink(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside_temp = tmp_path / "outside-temp"
    outside_temp.write_text("temp sentinel", encoding="utf-8")
    outside_final = tmp_path / "outside-final"
    outside_final.write_text("final sentinel", encoding="utf-8")
    (work / "cleanup-inventory.json.tmp").symlink_to(outside_temp)
    inventory = work / "cleanup-inventory.json"
    inventory.symlink_to(outside_final)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/cleanup.py"),
            "preview",
            "--root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "deleted 0" in result.stdout
    assert outside_temp.read_text() == "temp sentinel"
    assert outside_final.read_text() == "final sentinel"
    assert inventory.is_file() and not inventory.is_symlink()


def test_attestation_never_follows_predictable_temp_or_final_symlink(tmp_path):
    target = tmp_path / "work/data"
    target.mkdir(parents=True)
    (target / "inside").write_text("inside", encoding="utf-8")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    outside_temp = tmp_path / "outside-temp"
    outside_temp.write_text("temp sentinel", encoding="utf-8")
    outside_final = tmp_path / "outside-final"
    outside_final.write_text("final sentinel", encoding="utf-8")
    (outputs / "data-deletion-attestation.json.tmp").symlink_to(outside_temp)
    attestation = outputs / "data-deletion-attestation.json"
    attestation.symlink_to(outside_final)

    delete_targets(tmp_path, [target])

    assert not target.exists()
    assert outside_temp.read_text() == "temp sentinel"
    assert outside_final.read_text() == "final sentinel"
    assert attestation.is_file() and not attestation.is_symlink()


def test_cli_rejects_output_diagnostics_collision_before_model_load(tmp_path):
    (tmp_path / "nested").mkdir()
    output = tmp_path / "submission.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer.py"),
            "--model",
            "no-such-model",
            "--input",
            str(ROOT / "tests/fixtures"),
            "--output",
            str(output),
            "--diagnostics",
            str(tmp_path / "nested/../submission.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "output and diagnostics must differ" in result.stderr
    assert "KeyError" not in result.stderr
    assert not output.exists()
