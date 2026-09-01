#!/usr/bin/env python3
"""Fit and train the isolated three-branch model on all techjam2026_v2 train rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from techjam_aigc.three_branch.config import ThreeBranchConfig
from techjam_aigc.three_branch.data import (
    ThreeBranchDataset,
    load_all_training_rows,
    parent_digest,
)
from techjam_aigc.three_branch.training import (
    build_model,
    file_sha256,
    fit_dual_memories,
    load_dual_memory,
    seed_everything,
    train_ten_epochs,
)
from techjam_aigc.trace_rx_m.backbone import DinoV3PatchEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "fit-memories", "train", "all"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/three-branch.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/three-branch-techjam2026-v2"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(dataset, config: ThreeBranchConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.data.seed)
    return DataLoader(
        dataset,
        batch_size=config.optimizer.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=config.optimizer.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.optimizer.workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _resolved(path: str, repo_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = ThreeBranchConfig.load(config_path)
    config.validate(require_backbone_access=args.stage != "preflight")
    seed_everything(config.data.seed)
    frame = load_all_training_rows(config.data, repo_root)
    expected_ids = set(frame["parent_id"].astype(str))
    if len(expected_ids) != len(frame):
        raise ValueError("Training parent identifiers must be unique.")
    manifest_path = _resolved(config.data.manifest, repo_root)
    labels_path = _resolved(config.data.labels, repo_root)
    provenance = {
        "manifest_sha256": file_sha256(manifest_path),
        "labels_sha256": file_sha256(labels_path),
        "config_sha256": file_sha256(config_path),
        "training_parent_digest": parent_digest(list(expected_ids)),
    }
    run_contract = {
        "model_name": config.model_name,
        "dataset": "techjam2026_v2",
        "rows": len(frame),
        "class_rows": {
            "authentic": int(frame["target"].eq(0).sum()),
            "aigc": int(frame["target"].eq(1).sum()),
        },
        "training_pools": frame["training_pool"].value_counts().sort_index().to_dict(),
        "generator_families": frame.loc[
            frame["target"].eq(1), "generator_family"
        ].value_counts().sort_index().to_dict(),
        "split": "train",
        "all_training_pools_used": True,
        "validation_used": False,
        "test_used": False,
        "generator_holdout_used": False,
        "epochs": config.optimizer.epochs,
        **provenance,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run-contract.json").write_text(
        json.dumps(run_contract, indent=2, sort_keys=True) + "\n"
    )

    data = ThreeBranchDataset(frame, config.preprocessing, include_native_crops=True)
    sample = data[0]
    expected_global = (3, config.preprocessing.global_image_size, config.preprocessing.global_image_size)
    expected_native = (
        config.preprocessing.native_crop_count,
        3,
        config.preprocessing.native_crop_size,
        config.preprocessing.native_crop_size,
    )
    if sample["global_pixels"].shape != expected_global:
        raise RuntimeError("Global preprocessing shape does not match the config.")
    if sample["native_crops"].shape != expected_native:
        raise RuntimeError("Native forensic crop shape does not match the config.")
    print(json.dumps({"stage": "preflight", **run_contract}), flush=True)
    if args.stage == "preflight":
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    memory_path = output / "dual-memory.pt"
    if args.stage in {"fit-memories", "all"}:
        memory_data = ThreeBranchDataset(
            frame,
            config.preprocessing,
            include_native_crops=False,
        )
        memory_loader = _loader(memory_data, config, shuffle=False)
        unadapted_config = config.backbone.__class__(
            **{**config.backbone.__dict__, "lora_rank": 0}
        )
        encoder = DinoV3PatchEncoder(unadapted_config)
        metadata = fit_dual_memories(
            encoder,
            memory_loader,
            config=config,
            device=device,
            expected_parent_ids=expected_ids,
            provenance=provenance,
            output_path=memory_path,
        )
        print(json.dumps({"stage": "memories-complete", **metadata}), flush=True)
        if args.stage == "fit-memories":
            return

    memory = load_dual_memory(memory_path, config=config, provenance=provenance)
    model = build_model(config, memory)
    train_loader = _loader(data, config, shuffle=True)
    resume = args.resume.resolve() if args.resume is not None else None
    history = train_ten_epochs(
        model,
        train_loader,
        config=config,
        device=device,
        expected_parent_ids=expected_ids,
        provenance=provenance,
        memory_path=memory_path,
        output_directory=output,
        resume_path=resume,
    )
    print(json.dumps({
        "stage": "training-complete",
        "epochs": len(history),
        "final": history[-1],
        "artifact": str(output / "three-branch-final.pt"),
    }), flush=True)


if __name__ == "__main__":
    main()
