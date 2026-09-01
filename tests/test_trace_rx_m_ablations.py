from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "trace_rx_m_ablation_holdout_gemini",
    "trace_rx_m_ablation_holdout_flux",
    "trace_rx_m_ablation_frozen_encoder",
    "trace_rx_m_ablation_no_memory",
    "trace_rx_m_ablation_bce_only",
)


def _config(package: str) -> dict:
    path = ROOT / "src" / "techjam_aigc" / package / "config.json"
    return json.loads(path.read_text())


def _launcher_module():
    path = ROOT / "scripts" / "run_trace_rx_m_ablations.py"
    spec = importlib.util.spec_from_file_location("trace_rx_m_ablation_launcher", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ablation_configs_are_unique_and_resource_matched() -> None:
    configs = [_config(package) for package in PACKAGES]
    assert len({item["hub"]["repo_id"] for item in configs}) == 5
    assert len({item["hub"]["path_prefix"] for item in configs}) == 5
    assert len({item["tracking"]["wandb_run_name"] for item in configs}) == 5
    assert {item["data"]["batch_size"] for item in configs} == {16}
    assert {item["data"]["workers"] for item in configs} == {2}
    assert configs[0]["data"]["held_out_generator_family"] == "gemini_flash_image"
    assert configs[1]["data"]["held_out_generator_family"] == "flux_1_schnell"
    assert configs[2]["backbone"]["lora_rank"] == 0
    assert configs[4]["loss"]["pauc_weight"] == 0
    assert configs[4]["loss"]["pair_weight"] == 0


def test_holdout_counts_match_frozen_manifest() -> None:
    frame = pd.read_csv(
        ROOT / "data" / "techjam2026_v2-normalized" / "training-manifest.csv",
        usecols=["split", "training_pool", "target", "generator_family"],
    )
    train = frame[frame["split"].eq("train") & frame["training_pool"].eq("detector")]
    val = frame[frame["split"].eq("val")]
    assert int((train["generator_family"] == "gemini_flash_image").sum()) == 4572
    assert int((val["generator_family"] == "gemini_flash_image").sum()) == 1888
    assert int((train["generator_family"] == "flux_1_schnell").sum()) == 5580
    assert int((val["generator_family"] == "flux_1_schnell").sum()) == 1420


def test_frozen_encoder_exposes_no_trainable_backbone_parameters() -> None:
    from techjam_aigc.trace_rx_m_ablation_frozen_encoder.memory import AuthenticMemory
    from techjam_aigc.trace_rx_m_ablation_frozen_encoder.model import TraceRXM
    from techjam_aigc.trace_rx_m_ablation_frozen_encoder.config import HeadConfig

    class Encoder(nn.Module):
        output_dim = 8

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, pixels):
            return torch.ones(len(pixels), 4, 8) * self.weight

    model = TraceRXM(Encoder(), AuthenticMemory(4, 8, 2), HeadConfig())
    model.configure_for_detection(frozen_encoder_fallback=True)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.head_parameters())


def test_direct_probe_has_no_memory_dependency() -> None:
    from techjam_aigc.trace_rx_m_ablation_no_memory.config import HeadConfig
    from techjam_aigc.trace_rx_m_ablation_no_memory.model import TraceRXM

    class Encoder(nn.Module):
        output_dim = 8

        def forward(self, pixels):
            return torch.ones(len(pixels), 4, 8)

    model = TraceRXM(Encoder(), HeadConfig(hidden_dim=4))
    assert not hasattr(model, "memory")
    assert model(torch.zeros(2, 3, 8, 8)).logit.shape == (2,)


def test_bce_only_total_is_exactly_bce() -> None:
    from techjam_aigc.trace_rx_m_ablation_bce_only.config import LossConfig
    from techjam_aigc.trace_rx_m_ablation_bce_only.losses import detection_objective

    logits = torch.tensor([-0.8, -0.2, 0.4, 1.2])
    labels = torch.tensor([0, 0, 1, 1])
    result = detection_objective(
        logits,
        labels,
        torch.ones(4),
        dda_logits=torch.empty(0),
        source_real_logits=torch.empty(0),
        config=LossConfig(pauc_weight=0.0, pair_weight=0.0),
    )
    assert torch.equal(result.total, result.bce)


@pytest.mark.parametrize("package", PACKAGES)
def test_heldout_gate_is_report_only(package: str) -> None:
    source = (ROOT / "src" / "techjam_aigc" / package / "train.py").read_text()
    assert '"gate_action": "report_only"' in source
    assert "if auc < config.data.held_out_min_roc_auc" not in source


def test_memory_pause_and_resume_command(monkeypatch, tmp_path: Path) -> None:
    from techjam_aigc.trace_rx_m_ablation_holdout_gemini import train

    monkeypatch.setenv("TRACE_RX_M_RESERVE_GIB", "64")
    monkeypatch.setattr(train, "_available_memory_gib", lambda: 63.0)
    with pytest.raises(SystemExit) as error:
        train._pause_if_reserve_breached(tmp_path, epoch=3)
    assert error.value.code == 75
    assert json.loads((tmp_path / "memory-pause.json").read_text())["epoch"] == 3

    launcher = _launcher_module()
    command = launcher.training_command(
        ROOT,
        launcher.ABLATIONS[0],
        manifest=ROOT / "data/techjam2026_v2-normalized/training-manifest.csv",
        output=tmp_path,
        resume=True,
    )
    assert "--resume" in command


def test_common_memory_is_copied_with_identical_hash(tmp_path: Path) -> None:
    launcher = _launcher_module()
    source = ROOT / "artifacts/trace-rx-m-techjam2026-v2/s3_memory.pt"
    digest = launcher.file_sha256(source)
    launcher.seed_artifacts(tmp_path, source.parent, digest)
    for ablation in launcher.ABLATIONS:
        memory = tmp_path / ablation.slug / "s3_memory.pt"
        if ablation.uses_memory:
            assert launcher.file_sha256(memory) == digest
        else:
            assert not memory.exists()
