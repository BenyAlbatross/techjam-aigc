"""Adapter around the upstream `techjam-aigc` inference path for `trace-rx-m-v2`.

We deliberately do **not** reimplement the detector. The model card points at
https://github.com/BenyAlbatross/techjam-aigc; that repository is vendored at
`vendor/techjam-aigc` pinned to commit a3c393a (branch `feat/traincode`, "Train TRACE-RX-M
on TechJam dataset"), which is the branch that produced the shipped checkpoint -- it is the
only branch whose `OptimizerConfig` carries the `mixed_precision` key present in the
checkpoint's embedded config. Its `trace_rx_m/model.py` and `memory.py` are byte-identical
to the other branches carrying them, so the forward pass is unambiguous.

Scoring convention, taken from `scripts/export_trace_rx_m_scores.py`: the exported score is
the raw `model(pixels).logit`. Reliability fusion is skipped because the published model
repository ships no `s5_reliability.json`, and that script applies fusion only when present.
The decision threshold is 0.0 on the logit, matching `evaluation/metrics_by_condition.csv`.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_VENDOR_ROOT = Path(__file__).resolve().parents[3] / "vendor"

# The two detector families live on different branches whose `techjam_aigc` packages differ,
# so each gets its own checkout and only one may be on sys.path per process. `run_eval`
# therefore scores one family per invocation.
FAMILIES = {
    "trace_rx_m": {
        "vendor": _VENDOR_ROOT / "techjam-aigc",
        "branch": "feat/traincode",
        "commit": "a3c393a17d3eea3ac79b438df8cb0bc619630e0b",
        "training_module": "techjam_aigc.trace_rx_m.training",
        "config_class": ("techjam_aigc.trace_rx_m.config", "TraceRXMConfig"),
    },
    "trace_rx_parallel": {
        "vendor": _VENDOR_ROOT / "techjam-aigc-parallel",
        "branch": "feat/trace-rx-parallel",
        "commit": "f58ec937b416e15953b3cb7b48608a7df976ba3d",
        "training_module": "techjam_aigc.trace_rx_parallel.training",
        "config_class": ("techjam_aigc.trace_rx_parallel.config", "TraceRXParallelConfig"),
    },
}
DEFAULT_FAMILY = "trace_rx_m"
VENDOR_COMMIT = FAMILIES[DEFAULT_FAMILY]["commit"]

MODEL_DIR = Path("/home/joel/.cache/huggingface/hub/models--techjam-aigc--trace-rx-m-v2/"
                 "snapshots/3d4270323bbb2de437326fa59e436a1540da1110")


def _ensure_vendor(family: str = DEFAULT_FAMILY) -> dict:
    spec = FAMILIES[family]
    src = spec["vendor"] / "src"
    if not (src / "techjam_aigc").is_dir():
        raise RuntimeError(
            f"vendored upstream for {family!r} not found at {src}. Re-create with:\n"
            f"  git clone --branch {spec['branch']} https://github.com/BenyAlbatross/techjam-aigc.git "
            f"{spec['vendor']} && git -C {spec['vendor']} checkout {spec['commit']}")
    for other, o in FAMILIES.items():
        if other != family and str(o["vendor"] / "src") in sys.path:
            raise RuntimeError(
                f"{other!r} is already on sys.path; score one family per process.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return spec


def load(device: torch.device, model_dir: Path = MODEL_DIR, family: str = DEFAULT_FAMILY):
    """Returns (model, checkpoint_metadata, image_size) via the upstream loader."""
    import importlib

    spec = _ensure_vendor(family)
    load_detector_checkpoint = getattr(
        importlib.import_module(spec["training_module"]), "load_detector_checkpoint")
    cfg_mod, cfg_name = spec["config_class"]
    config_class = getattr(importlib.import_module(cfg_mod), cfg_name)

    model, meta = load_detector_checkpoint(
        model_dir / "s4_detector.pt", model_dir / "s3_memory.pt", device=device)
    cfg = config_class.from_dict(meta["config"])
    # encoder_mode 'frozen' makes the loader rebuild the backbone with lora_rank=0, so the
    # encoder is stock facebook/dinov2-base at the pinned revision -- nothing unrecoverable
    # was dropped by the checkpoint storing only heads and memory.
    if meta["encoder_mode"] != "frozen" or model.encoder.lora_modules:
        raise RuntimeError("expected a frozen-encoder checkpoint with no LoRA adapters")
    return model, meta, int(cfg.backbone.image_size)


def preprocess(raw: list[bytes], image_size: int, device: torch.device,
               family: str = DEFAULT_FAMILY) -> torch.Tensor:
    """Upstream `canonical_preprocess`: square BILINEAR resize + ImageNet normalisation.

    Shared by both families -- `export_trace_rx_parallel_scores.py` imports the same
    `trace_rx_m.data.TraceRXMDataset`, so the preprocessing path is identical.
    """
    _ensure_vendor(family)
    from techjam_aigc.trace_rx_m.augment import canonical_preprocess

    arrs = [canonical_preprocess(Image.open(io.BytesIO(b)).convert("RGB"), image_size=image_size)
            for b in raw]
    return torch.from_numpy(np.stack(arrs)).to(device)
