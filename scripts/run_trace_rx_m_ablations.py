#!/usr/bin/env python3
"""Run the five local TRACE-RX-M S4 ablations sequentially and safely."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


GIB = 1024**3
RESERVE_GIB = 64.0
MAX_ABLATION_GIB = 42.0
MIN_ABLATION_GIB = 24.0


@dataclass(frozen=True)
class Ablation:
    slug: str
    package: str
    repo_id: str
    uses_memory: bool = True


ABLATIONS = (
    Ablation(
        "holdout-gemini",
        "trace_rx_m_ablation_holdout_gemini",
        "techjam-aigc/trace-rx-m-v2-ablation-holdout-gemini",
    ),
    Ablation(
        "holdout-flux",
        "trace_rx_m_ablation_holdout_flux",
        "techjam-aigc/trace-rx-m-v2-ablation-holdout-flux",
    ),
    Ablation(
        "frozen-encoder",
        "trace_rx_m_ablation_frozen_encoder",
        "techjam-aigc/trace-rx-m-v2-ablation-frozen-encoder",
    ),
    Ablation(
        "no-memory",
        "trace_rx_m_ablation_no_memory",
        "techjam-aigc/trace-rx-m-v2-ablation-no-memory",
        uses_memory=False,
    ),
    Ablation(
        "bce-only",
        "trace_rx_m_ablation_bce_only",
        "techjam-aigc/trace-rx-m-v2-ablation-bce-only",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/techjam2026_v2-normalized/training-manifest.csv"),
    )
    parser.add_argument(
        "--source-artifacts",
        type=Path,
        default=Path("artifacts/trace-rx-m-techjam2026-v2"),
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_gib() -> tuple[float, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = float(value.split()[0]) * 1024 / GIB
    return values["MemTotal"], values["MemAvailable"]


def wait_for_allowance(*, poll_seconds: int, no_wait: bool) -> tuple[float, float, float]:
    while True:
        total, available = memory_gib()
        allowance = min(MAX_ABLATION_GIB, available - RESERVE_GIB)
        if allowance >= MIN_ABLATION_GIB:
            return total, available, allowance
        message = (
            f"Waiting for unified memory: available={available:.1f} GiB, "
            f"reserve={RESERVE_GIB:.1f} GiB, allowance={allowance:.1f} GiB"
        )
        if no_wait:
            raise RuntimeError(message)
        print(message, flush=True)
        time.sleep(max(10, poll_seconds))


def config_path(repo_root: Path, ablation: Ablation) -> Path:
    return repo_root / "src" / "techjam_aigc" / ablation.package / "config.json"


def output_path(root: Path, ablation: Ablation) -> Path:
    return root / ablation.slug


def validate_suite(repo_root: Path, manifest: Path, source_memory: Path) -> str:
    import torch

    manifest_hash = file_sha256(manifest)
    artifact = torch.load(source_memory, map_location="cpu", weights_only=True)
    if artifact.get("stage") != "S3":
        raise ValueError("Shared artifact is not an S3 authentic memory.")
    if artifact.get("manifest_sha256") != manifest_hash:
        raise ValueError("Shared S3 memory was built from a different manifest.")
    repos: set[str] = set()
    for ablation in ABLATIONS:
        config = json.loads(config_path(repo_root, ablation).read_text())
        if artifact.get("backbone_model_id") != config["backbone"]["model_id"]:
            raise ValueError(f"Shared memory backbone mismatch for {ablation.slug}.")
        if artifact.get("backbone_revision") != config["backbone"]["revision"]:
            raise ValueError(f"Shared memory revision mismatch for {ablation.slug}.")
        if config["hub"]["repo_id"] != ablation.repo_id:
            raise ValueError(f"Repository mismatch for {ablation.slug}.")
        if ablation.repo_id in repos:
            raise ValueError("Ablation Hugging Face repositories must be unique.")
        repos.add(ablation.repo_id)
        if config["data"]["batch_size"] not in {10, 16}:
            raise ValueError("Ablation batch sizes must be uniformly 10 or 16.")
        if config["data"]["workers"] != 2:
            raise ValueError("Ablation worker count must remain two.")
    if len({json.loads(config_path(repo_root, item).read_text())["data"]["batch_size"] for item in ABLATIONS}) != 1:
        raise ValueError("All ablations must use the same batch size.")
    return file_sha256(source_memory)


def seed_artifacts(root: Path, source_artifacts: Path, memory_hash: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    common = root / "common"
    common.mkdir(exist_ok=True)
    for name in (
        "s0_protocol.json",
        "s1_memory.pt",
        "s1_capacity.pt",
        "s3_capacity.json",
    ):
        source = source_artifacts / name
        destination = common / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    source_memory = source_artifacts / "s3_memory.pt"
    common_memory = common / "s3_memory.pt"
    if not common_memory.exists():
        shutil.copy2(source_memory, common_memory)
    if file_sha256(common_memory) != memory_hash:
        raise ValueError("Copied common memory failed SHA-256 verification.")
    for ablation in ABLATIONS:
        output = output_path(root, ablation)
        output.mkdir(exist_ok=True)
        if not ablation.uses_memory:
            if (output / "s3_memory.pt").exists():
                raise ValueError("No-memory ablation must not contain s3_memory.pt.")
            continue
        destination = output / "s3_memory.pt"
        if not destination.exists():
            shutil.copy2(common_memory, destination)
        if file_sha256(destination) != memory_hash:
            raise ValueError(f"Memory hash mismatch for {ablation.slug}.")


def command_prefix(repo_root: Path) -> list[str]:
    python = repo_root / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError("Expected project interpreter at .venv/bin/python.")
    prefix = ["nice", "-n", "10"]
    if shutil.which("ionice"):
        prefix.extend(["ionice", "-c2", "-n7"])
    prefix.append(str(python))
    return prefix


def run_process(
    command: list[str],
    *,
    repo_root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.wait()


def environment_for(total: float, allowance: float) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "MALLOC_ARENA_MAX": "2",
        "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TRACE_RX_M_CUDA_MEMORY_FRACTION": f"{allowance / total:.8f}",
        "TRACE_RX_M_RESERVE_GIB": str(RESERVE_GIB),
    })
    return environment


def training_command(
    repo_root: Path,
    ablation: Ablation,
    *,
    manifest: Path,
    output: Path,
    preflight: bool = False,
    resume: bool = False,
) -> list[str]:
    command = command_prefix(repo_root) + [
        "-m", f"techjam_aigc.{ablation.package}.train",
        "--stage", "detection",
        "--config", str(config_path(repo_root, ablation)),
        "--manifest", str(manifest),
        "--output", str(output),
    ]
    if preflight:
        command.append("--preflight-only")
    if resume:
        command.append("--resume")
    return command


def run_preflights(
    repo_root: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> None:
    for slug in ("holdout-gemini", "frozen-encoder", "no-memory"):
        ablation = next(item for item in ABLATIONS if item.slug == slug)
        output = output_path(args.artifacts_root, ablation)
        code = run_process(
            training_command(
                repo_root,
                ablation,
                manifest=args.manifest,
                output=output,
                preflight=True,
            ),
            repo_root=repo_root,
            environment=environment,
            log_path=output / "preflight.log",
        )
        if code:
            raise RuntimeError(f"GPU preflight failed for {slug}; see {output / 'preflight.log'}.")


def export_and_evaluate(
    repo_root: Path,
    ablation: Ablation,
    *,
    manifest: Path,
    output: Path,
    environment: dict[str, str],
) -> None:
    scores = output / "scores.csv"
    if not scores.exists():
        command = command_prefix(repo_root) + [
            "-m", f"techjam_aigc.{ablation.package}.export_scores",
            "--config", str(config_path(repo_root, ablation)),
            "--manifest", str(manifest),
            "--artifacts", str(output),
            "--output", str(scores),
            "--splits", "val",
        ]
        code = run_process(
            command,
            repo_root=repo_root,
            environment=environment,
            log_path=output / "export.log",
        )
        if code:
            raise RuntimeError(f"Score export failed for {ablation.slug}.")
    metrics = output / "metrics.json"
    command = command_prefix(repo_root) + [
        str(repo_root / "scripts" / "evaluate_trace_rx_m_ablation_scores.py"),
        "--scores", str(scores),
        "--output", str(metrics),
    ]
    code = run_process(
        command,
        repo_root=repo_root,
        environment=environment,
        log_path=output / "evaluation.log",
    )
    if code:
        raise RuntimeError(f"Canonical evaluation failed for {ablation.slug}.")


def upload_post_training(ablation: Ablation, output: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for name in ("scores.csv", "metrics.json", "run-metadata.json"):
        path = output / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"ablation-results/{name}",
                repo_id=ablation.repo_id,
                repo_type="model",
                commit_message=f"Upload {ablation.slug} validation results",
            )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.manifest = args.manifest.resolve()
    args.source_artifacts = args.source_artifacts.resolve()
    args.artifacts_root = args.artifacts_root.resolve()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_root / ".suite.lock"
    lock = lock_path.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    source_memory = args.source_artifacts / "s3_memory.pt"
    memory_hash = validate_suite(repo_root, args.manifest, source_memory)
    seed_artifacts(args.artifacts_root, args.source_artifacts, memory_hash)
    total, available, allowance = wait_for_allowance(
        poll_seconds=args.poll_seconds,
        no_wait=args.no_wait,
    )
    environment = environment_for(total, allowance)
    print(
        f"Memory gate passed: total={total:.1f} GiB, available={available:.1f} GiB, "
        f"ablation cap={allowance:.1f} GiB, reserve={RESERVE_GIB:.1f} GiB",
        flush=True,
    )
    if not args.skip_preflight:
        run_preflights(repo_root, args, environment)
    if args.preflight_only:
        return

    for ablation in ABLATIONS:
        output = output_path(args.artifacts_root, ablation)
        started = time.time()
        resumed = False
        while not (output / "s4_detector.pt").exists():
            total, available, allowance = wait_for_allowance(
                poll_seconds=args.poll_seconds,
                no_wait=args.no_wait,
            )
            environment = environment_for(total, allowance)
            code = run_process(
                training_command(
                    repo_root,
                    ablation,
                    manifest=args.manifest,
                    output=output,
                    resume=resumed,
                ),
                repo_root=repo_root,
                environment=environment,
                log_path=output / "training.log",
            )
            if code == 75:
                resumed = True
                continue
            if code:
                raise RuntimeError(f"Training failed for {ablation.slug}; see training.log.")
        metadata = {
            "slug": ablation.slug,
            "repo_id": ablation.repo_id,
            "uses_memory": ablation.uses_memory,
            "source_memory_sha256": memory_hash if ablation.uses_memory else None,
            "runtime_seconds_this_invocation": time.time() - started,
            "memory_reserve_gib": RESERVE_GIB,
            "maximum_ablation_gib": MAX_ABLATION_GIB,
            "batch_size": json.loads(config_path(repo_root, ablation).read_text())["data"]["batch_size"],
        }
        (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        total, _, allowance = wait_for_allowance(
            poll_seconds=args.poll_seconds,
            no_wait=args.no_wait,
        )
        environment = environment_for(total, allowance)
        export_and_evaluate(
            repo_root,
            ablation,
            manifest=args.manifest,
            output=output,
            environment=environment,
        )
        upload_post_training(ablation, output)

    summary_command = command_prefix(repo_root) + [
        str(repo_root / "scripts" / "summarize_trace_rx_m_ablations.py"),
        "--artifacts-root", str(args.artifacts_root),
        "--upload",
    ]
    code = run_process(
        summary_command,
        repo_root=repo_root,
        environment=dict(os.environ),
        log_path=args.artifacts_root / "summary.log",
    )
    if code:
        raise RuntimeError("Ablation summary generation failed.")


if __name__ == "__main__":
    main()
