"""W&B tracking and durable Hugging Face checkpoint publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .config import HubConfig, TrackingConfig


class WandbTracker:
    """Small testable wrapper around one W&B training run."""

    def __init__(
        self,
        config: TrackingConfig,
        *,
        stage: str,
        run_config: Mapping[str, Any],
    ) -> None:
        try:
            import wandb
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError(
                "Install training dependencies with `uv sync --group train` to use W&B."
            ) from error
        self._wandb = wandb
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            job_type=stage,
            config=dict(run_config),
            save_code=True,
        )
        if self.run is None:  # pragma: no cover - defensive against SDK contract changes
            raise RuntimeError("wandb.init() did not return a run.")

    @property
    def run_id(self) -> str:
        return str(self.run.id)

    def watch(self, model: Any) -> None:
        self.run.watch(model, log="gradients", log_freq=100, log_graph=False)

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        self.run.log(dict(values), step=step)

    def summarize(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.run.summary[key] = value

    def log_model_artifact(
        self,
        paths: Sequence[Path],
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        artifact = self._wandb.Artifact(
            f"trace-rx-m-{self.run_id}",
            type="model",
            metadata=dict(metadata),
        )
        for path in paths:
            artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=["best", "final"])

    def finish(self, *, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


class HubCheckpointPublisher:
    """Synchronous Hub uploader: a successful S4 return means uploads completed."""

    def __init__(
        self,
        config: HubConfig,
        *,
        run_id: str,
        api: Any | None = None,
    ) -> None:
        config.validate(require_repo=True)
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError(
                "Install training dependencies with `uv sync --group train` to publish weights."
            ) from error
        self.config = config
        self.repo_id = str(config.repo_id)
        self.run_id = run_id
        self.api = HfApi() if api is None else api
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=config.private,
            exist_ok=True,
        )

    @property
    def run_prefix(self) -> str:
        prefix = self.config.path_prefix.strip("/")
        return str(PurePosixPath(prefix) / "runs" / self.run_id)

    def upload_periodic(self, path: Path, *, epoch: int, variant: str) -> str:
        destination = str(
            PurePosixPath(self.run_prefix)
            / "checkpoints"
            / variant
            / f"epoch-{epoch:04d}.pt"
        )
        result = self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=destination,
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.config.revision,
            commit_message=f"Upload TRACE-RX-M {variant} epoch {epoch}",
        )
        return str(getattr(result, "commit_url", result))

    def upload_final_bundle(
        self,
        *,
        best_path: Path,
        final_path: Path,
        metadata_paths: Sequence[Path] = (),
    ) -> str:
        from huggingface_hub import CommitOperationAdd

        prefix = PurePosixPath(self.config.path_prefix.strip("/"))
        run_prefix = PurePosixPath(self.run_prefix)
        operations = [
            CommitOperationAdd(
                path_in_repo=str(prefix / "best_detector.pt"),
                path_or_fileobj=best_path,
            ),
            CommitOperationAdd(
                path_in_repo=str(prefix / "final_detector.pt"),
                path_or_fileobj=final_path,
            ),
            CommitOperationAdd(
                path_in_repo=str(run_prefix / "best_detector.pt"),
                path_or_fileobj=best_path,
            ),
            CommitOperationAdd(
                path_in_repo=str(run_prefix / "final_detector.pt"),
                path_or_fileobj=final_path,
            ),
        ]
        for path in metadata_paths:
            operations.extend((
                CommitOperationAdd(
                    path_in_repo=str(prefix / path.name),
                    path_or_fileobj=path,
                ),
                CommitOperationAdd(
                    path_in_repo=str(run_prefix / path.name),
                    path_or_fileobj=path,
                ),
            ))
        result = self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.config.revision,
            operations=operations,
            commit_message="Upload TRACE-RX-M best and final weights",
        )
        return str(getattr(result, "commit_url", result))


class LocalCheckpointPublisher:
    """No-op publisher for explicitly local training runs.

    The training loop already writes every periodic and final artifact before
    invoking the publisher. This adapter keeps that durable local behavior
    without requiring Hugging Face credentials or silently attempting uploads.
    """

    def __init__(self, output: Path, *, run_id: str) -> None:
        self.output = Path(output).resolve()
        self.run_id = str(run_id)

    def upload_periodic(self, path: Path, *, epoch: int, variant: str) -> str:
        del epoch, variant
        return Path(path).resolve().as_uri()

    def upload_final_bundle(
        self,
        *,
        best_path: Path,
        final_path: Path,
        metadata_paths: Sequence[Path] = (),
    ) -> str:
        del best_path, metadata_paths
        return Path(final_path).resolve().as_uri()
