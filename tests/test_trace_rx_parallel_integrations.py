from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from techjam_aigc.trace_rx_m.config import HubConfig, TrackingConfig
from techjam_aigc.trace_rx_parallel.integrations import (
    HubCheckpointPublisher,
    WandbTracker,
)


class _FakeRun:
    id = "run-123"

    def __init__(self) -> None:
        self.summary = {}
        self.artifacts = []

    def log_artifact(self, artifact, aliases) -> None:
        self.artifacts.append((artifact, aliases))


class _FakeArtifact:
    def __init__(self, name, type, metadata) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name) -> None:
        self.files.append((path, name))


class _FakeHubApi:
    def __init__(self) -> None:
        self.created = []
        self.uploaded = []
        self.commits = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs)
        return SimpleNamespace(commit_url="https://huggingface.co/commit/periodic")

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://huggingface.co/commit/final")


def _write(path: Path) -> Path:
    path.write_bytes(b"checkpoint")
    return path


def test_wandb_records_parallel_model_and_exact_dataset_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: run,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        TrackingConfig(wandb_project="parallel", wandb_mode="offline"),
        stage="s4-detection",
        run_config={"seed": 3},
    )
    best = _write(tmp_path / "best.pt")
    final = _write(tmp_path / "final.pt")
    manifest = _write(tmp_path / "training-manifest.csv")

    tracker.log_model_artifact((best, final), metadata={"architecture": "parallel"})
    tracker.log_dataset_artifact((manifest,), metadata={"dataset_revision": "pinned"})

    assert run.artifacts[0][0].name == "trace-rx-parallel-run-123"
    assert run.artifacts[0][1] == ["best", "final"]
    assert run.artifacts[1][0].type == "dataset"
    assert run.artifacts[1][1] == ["training-input"]


def test_hub_uploads_nonempty_periodic_best_and_final_to_canonical_paths(
    tmp_path: Path,
) -> None:
    api = _FakeHubApi()
    publisher = HubCheckpointPublisher(
        HubConfig(repo_id="owner/model", path_prefix="parallel"),
        run_id="run-123",
        api=api,
    )
    periodic = _write(tmp_path / "epoch.pt")
    best = _write(tmp_path / "best.pt")
    final = _write(tmp_path / "final.pt")
    metadata = _write(tmp_path / "validity.json")

    publisher.upload_periodic(periodic, epoch=1, variant="lora")
    publisher.upload_best(best, epoch=1, variant="lora")
    publisher.upload_final_bundle(
        best_path=best,
        final_path=final,
        metadata_paths=(metadata,),
    )

    assert api.uploaded[0]["path_in_repo"].endswith("checkpoints/lora/epoch-0001.pt")
    best_destinations = {item.path_in_repo for item in api.commits[0]["operations"]}
    assert "parallel/best_detector.pt" in best_destinations
    assert "parallel/runs/run-123/checkpoints/lora/best_detector.pt" in best_destinations
    destinations = {item.path_in_repo for item in api.commits[1]["operations"]}
    assert "parallel/best_detector.pt" in destinations
    assert "parallel/final_detector.pt" in destinations
    assert "parallel/runs/run-123/best_detector.pt" in destinations
    assert "parallel/runs/run-123/final_detector.pt" in destinations

    empty = tmp_path / "empty.pt"
    empty.touch()
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        publisher.upload_periodic(empty, epoch=2, variant="lora")
