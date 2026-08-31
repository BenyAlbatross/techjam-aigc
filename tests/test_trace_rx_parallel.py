from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from techjam_aigc.trace_rx_m.config import OptimizerConfig
from techjam_aigc.trace_rx_m.memory import AuthenticMemory
from techjam_aigc.trace_rx_m.training import build_detection_optimizer
from techjam_aigc.trace_rx_parallel.config import (
    ParallelHeadConfig,
    TraceRXParallelConfig,
)
from techjam_aigc.trace_rx_parallel.model import TraceRXParallel
from techjam_aigc.trace_rx_parallel.training import save_detector_checkpoint


class TinyEncoder(nn.Module):
    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 6)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(2, 4))
        self.lora_B = nn.Parameter(torch.zeros(6, 2))

    def forward(self, pixels):
        features = self.base(pixels) + (pixels @ self.lora_A.t()) @ self.lora_B.t()
        return features.unsqueeze(1).repeat(1, 5, 1)


def _model() -> TraceRXParallel:
    return TraceRXParallel(
        TinyEncoder(),
        AuthenticMemory(8, 6, 3),
        ParallelHeadConfig(evidence_dim=7, hidden_dim=9, dropout=0.0),
    )


def test_parallel_branches_and_equal_weight_fusion_shapes() -> None:
    model = _model().eval()
    output = model(torch.randn(4, 4))

    assert output.logit.shape == (4,)
    assert output.global_logit.shape == (4,)
    assert output.memory_logit.shape == (4,)
    assert output.residual.shape == (4, 5, 6)
    torch.testing.assert_close(
        output.logit,
        0.5 * (output.global_logit + output.memory_logit),
    )
    torch.testing.assert_close(output.fusion_weights, torch.tensor([0.5, 0.5]))


def test_global_branch_does_not_depend_on_authentic_memory() -> None:
    model = _model().eval()
    pixels = torch.randn(3, 4)
    before = model(pixels)
    with torch.no_grad():
        model.memory.prototypes.copy_(
            torch.nn.functional.normalize(torch.randn_like(model.memory.prototypes), dim=-1)
        )
    after = model(pixels)

    torch.testing.assert_close(before.global_logit, after.global_logit)
    assert not torch.allclose(before.memory_logit, after.memory_logit)


def test_detection_freeze_ledger_trains_both_heads_fusion_and_lora() -> None:
    model = _model()
    model.configure_for_detection()

    assert not model.memory.prototypes.requires_grad
    assert not model.encoder.base.weight.requires_grad
    assert model.encoder.lora_A.requires_grad
    assert all(parameter.requires_grad for parameter in model.head_parameters())

    optimizer = build_detection_optimizer(model, OptimizerConfig())
    assert {group["name"] for group in optimizer.param_groups} == {"lora", "heads"}

    model.configure_for_memory_fit()
    assert model.memory.prototypes.requires_grad
    assert not model.encoder.lora_A.requires_grad
    assert not any(parameter.requires_grad for parameter in model.head_parameters())


def test_parallel_config_round_trip_and_quantile_validation() -> None:
    values = TraceRXParallelConfig().to_dict()
    values["backbone"]["lora_targets"] = ["q_proj", "v_proj"]
    config = TraceRXParallelConfig.from_dict(values)
    assert config.head.global_tail_quantile == pytest.approx(0.95)
    assert config.backbone.lora_targets == ("q_proj", "v_proj")
    assert config.tracking.wandb_project == "trace-rx-parallel"
    assert config.hub.path_prefix == "trace-rx-parallel"

    values["head"]["global_tail_quantile"] = 1.0
    with pytest.raises(ValueError, match="global_tail_quantile"):
        TraceRXParallelConfig.from_dict(values)


def test_parallel_checkpoint_is_architecture_tagged(tmp_path) -> None:
    model = _model()
    model.configure_for_detection()
    path = tmp_path / "parallel.pt"
    save_detector_checkpoint(
        model,
        path,
        config=TraceRXParallelConfig().to_dict(),
        memory_artifact_sha256="memory",
        manifest_sha256="manifest",
        history=[],
        epoch=1,
    )
    artifact = torch.load(path, map_location="cpu", weights_only=True)

    assert artifact["architecture"] == "trace-rx-parallel"
    assert artifact["stage"] == "S4"
    assert "global_classifier.4.weight" in artifact["model_state"]
    assert "memory_classifier.4.weight" in artifact["model_state"]
    assert "fusion.weight" in artifact["model_state"]
