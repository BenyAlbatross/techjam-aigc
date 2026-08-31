from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from techjam_aigc.trace_rx_m.config import HubConfig, TrackingConfig
from techjam_aigc.trace_rx_m.integrations import HubCheckpointPublisher, WandbTracker


class _FakeRun:
    id = "run-123"

    def __init__(self) -> None:
        self.summary = {}
        self.logged = []
        self.artifacts = []
        self.finished = None

    def watch(self, *args, **kwargs) -> None:
        self.watched = (args, kwargs)

    def log(self, values, step=None) -> None:
        self.logged.append((values, step))

    def log_artifact(self, artifact, aliases) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code=0) -> None:
        self.finished = exit_code


class _FakeArtifact:
    def __init__(self, name, type, metadata) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name) -> None:
        self.files.append((path, name))


def test_wandb_tracker_logs_metrics_summary_and_model_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    run = _FakeRun()
    calls = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        TrackingConfig(wandb_project="project", wandb_mode="offline"),
        stage="s4-detection",
        run_config={"seed": 3},
    )
    best = tmp_path / "best.pt"
    final = tmp_path / "final.pt"
    best.touch()
    final.touch()

    tracker.log({"loss": 1.0}, step=2)
    tracker.summarize({"best_epoch": 2})
    tracker.log_model_artifact((best, final), metadata={"metric": "loss"})
    tracker.finish()

    assert calls[0]["project"] == "project"
    assert calls[0]["job_type"] == "s4-detection"
    assert run.logged == [({"loss": 1.0}, 2)]
    assert run.summary["best_epoch"] == 2
    assert [name for _, name in run.artifacts[0][0].files] == ["best.pt", "final.pt"]
    assert run.artifacts[0][1] == ["best", "final"]
    assert run.finished == 0


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


def test_hub_publisher_uploads_periodic_and_canonical_best_final_paths(
    tmp_path: Path,
) -> None:
    api = _FakeHubApi()
    publisher = HubCheckpointPublisher(
        HubConfig(repo_id="owner/model", path_prefix="trace", checkpoint_every_epochs=2),
        run_id="run-123",
        api=api,
    )
    checkpoint = tmp_path / "epoch.pt"
    best = tmp_path / "best.pt"
    final = tmp_path / "final.pt"
    metadata = tmp_path / "validity.json"
    for path in (checkpoint, best, final, metadata):
        path.touch()

    publisher.upload_periodic(checkpoint, epoch=2, variant="lora")
    url = publisher.upload_final_bundle(
        best_path=best,
        final_path=final,
        metadata_paths=(metadata,),
    )

    assert api.created[0]["exist_ok"] is True
    assert api.uploaded[0]["path_in_repo"] == (
        "trace/runs/run-123/checkpoints/lora/epoch-0002.pt"
    )
    destinations = {
        operation.path_in_repo for operation in api.commits[0]["operations"]
    }
    assert "trace/best_detector.pt" in destinations
    assert "trace/final_detector.pt" in destinations
    assert "trace/runs/run-123/best_detector.pt" in destinations
    assert "trace/runs/run-123/final_detector.pt" in destinations
    assert "trace/validity.json" in destinations
    assert "trace/runs/run-123/validity.json" in destinations
    assert url.endswith("/final")
