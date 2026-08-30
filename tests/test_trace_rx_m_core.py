from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from techjam_aigc.trace_rx_m.backbone import LoRALinear, inject_lora
from techjam_aigc.trace_rx_m.config import HeadConfig, LossConfig, MemoryConfig, OptimizerConfig
from techjam_aigc.trace_rx_m.losses import detection_objective, partial_auc_surrogate
from techjam_aigc.trace_rx_m.memory import AuthenticMemory, residual_patch_statistics
from techjam_aigc.trace_rx_m.model import TraceRXM
from techjam_aigc.trace_rx_m.training import (
    CoverageMetrics,
    build_detection_optimizer,
    cache_patch_features,
    cosine_warmup_scheduler,
    load_feature_cache,
    load_memory_artifact,
    paired_dda_logits,
    primary_objective_mask,
    save_detector_checkpoint,
    save_memory_artifact,
    select_capacity,
)


def test_chunked_sparse_retrieval_matches_dense_reference_and_stays_fp32() -> None:
    generator = torch.Generator().manual_seed(4)
    prototypes = torch.randn(7, 5, generator=generator)
    tokens = torch.randn(2, 3, 5, generator=generator, dtype=torch.bfloat16)
    memory = AuthenticMemory(7, 5, 3, score_chunk_size=2, prototypes=prototypes)

    output = memory(tokens)
    queries = F.normalize(tokens.float(), dim=-1)
    keys = F.normalize(prototypes.float(), dim=-1)
    scores, indices = torch.topk(queries @ keys.t(), 3, dim=-1)
    weights = torch.softmax(scores / (5**0.5), dim=-1)
    selected = keys[indices]
    expected = (weights.unsqueeze(-1) * selected).sum(-2)

    torch.testing.assert_close(output.reference, expected)
    torch.testing.assert_close(output.max_attention, weights[..., 0])
    assert output.reference.dtype == torch.float32
    assert output.entropy.dtype == torch.float32


def test_memory_loss_uses_unsquared_frobenius_diversity() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    memory = AuthenticMemory(3, 2, 2, prototypes=prototypes)
    tokens = torch.randn(2, 4, 2)
    loss = memory.phase1_loss(tokens, diversity_weight=0.1)
    normalized = F.normalize(memory.prototypes, dim=-1)
    expected = torch.linalg.matrix_norm(normalized @ normalized.t() - torch.eye(3), ord="fro")
    torch.testing.assert_close(loss.diversity, expected)


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


def test_model_shapes_directional_pooling_and_freeze_ledger() -> None:
    encoder = TinyEncoder()
    memory = AuthenticMemory(8, 6, 3)
    model = TraceRXM(encoder, memory, HeadConfig(evidence_dim=7, hidden_dim=9, dropout=0.0))
    model.configure_for_detection()
    output = model(torch.randn(4, 4))

    assert output.logit.shape == (4,)
    assert output.residual.shape == (4, 5, 6)
    assert residual_patch_statistics(output.residual).shape == (4, 18)
    assert not model.memory.prototypes.requires_grad
    assert not model.encoder.base.weight.requires_grad
    assert model.encoder.lora_A.requires_grad
    assert all(parameter.requires_grad for parameter in model.head_parameters())

    model.configure_for_memory_fit()
    assert model.memory.prototypes.requires_grad
    assert not model.encoder.lora_A.requires_grad


def test_frozen_memory_still_backpropagates_retrieval_signal_to_queries() -> None:
    memory = AuthenticMemory(8, 6, 3)
    memory.requires_grad_(False)
    queries = torch.randn(4, 5, 6, requires_grad=True)
    output = memory(queries)
    (output.reference.sum() + output.max_attention.sum() + output.entropy.sum()).backward()

    assert memory.prototypes.grad is None
    assert queries.grad is not None
    assert torch.count_nonzero(queries.grad) > 0


def test_lora_injection_updates_only_low_rank_parameters_at_initialization() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ModuleDict({"query": nn.Linear(4, 4)}))
    names = inject_lora(model, targets=("query",), rank=2, alpha=4, dropout=0)
    assert names == ("1.query",)
    layer = model[1]["query"]
    assert isinstance(layer, LoRALinear)
    values = torch.randn(3, 4)
    torch.testing.assert_close(layer(values), layer.base(values))
    assert not layer.base.weight.requires_grad




def test_lora_defaults_match_transformers_dinov3_attention_layout() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.DINOv3ViTConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_register_tokens=4,
        image_size=32,
        patch_size=16,
    )
    backbone = transformers.DINOv3ViTModel(config)
    names = inject_lora(
        backbone, targets=("q_proj", "v_proj"), rank=2, alpha=4, dropout=0
    )
    assert len(names) == 4
    assert all(name.endswith(("q_proj", "v_proj")) for name in names)
    output = backbone(pixel_values=torch.randn(2, 3, 32, 32), return_dict=True)
    assert output.last_hidden_state[:, 1 + config.num_register_tokens:].shape == (2, 4, 64)
def test_primary_objective_excludes_dda_and_pair_loss_orders_it_above_source() -> None:
    logits = torch.tensor([-1.0, 0.2, 0.5, 1.0])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    mask = torch.tensor([True, True, False, True])
    config = LossConfig(pauc_weight=0.3, pair_weight=0.4)
    loss = detection_objective(
        logits,
        labels,
        torch.ones(4),
        dda_logits=logits[2:3],
        source_real_logits=logits[3:4],
        config=config,
        primary_mask=mask,
    )
    assert loss.total == pytest.approx(loss.bce + 0.3 * loss.pauc + 0.4 * loss.pair)
    assert loss.pair > 0
    assert partial_auc_surrogate(logits, labels) > 0


def test_primary_objective_excludes_both_members_of_a_dda_pair() -> None:
    batch = {
        "sample_kind": ["authentic", "native_aigc", "dda", "authentic"],
        "parent_id": ["paired-real", "native", "derived", "unpaired-real"],
        "source_parent_id": ["", "", "paired-real", ""],
    }
    assert primary_objective_mask(batch, include_dda=False) == [False, True, False, True]
    assert primary_objective_mask(batch, include_dda=True) == [True, True, True, True]


def test_capacity_selection_uses_worst_authentic_tail_not_mean() -> None:
    candidates = [
        CoverageMetrics(64, 4, 0.01, 0.2, 0.1, 0.30, {"rare": 0.30}),
        CoverageMetrics(128, 4, 0.02, 0.2, 0.1, 0.10, {"rare": 0.10}),
        CoverageMetrics(256, 8, 0.03, 0.2, 0.1, 0.0995, {"rare": 0.0995}),
    ]
    selected = select_capacity(candidates, relative_tolerance=0.01)
    assert (selected.size, selected.topk) == (128, 4)


def test_s1_cache_equivalence_and_s3_backbone_provenance(tmp_path) -> None:
    encoder = TinyEncoder().requires_grad_(False)
    cache_path = tmp_path / "cache.pt"
    cache_patch_features(
        encoder,
        [{
            "pixel_values": torch.randn(2, 4),
            "parent_id": ["one", "two"],
            "source_dataset": ["camera-a", "camera-b"],
        }],
        cache_path,
        device=torch.device("cpu"),
        backbone_model_id="test/backbone",
        backbone_revision="immutable",
        manifest_sha256="manifest",
    )
    cache = load_feature_cache(cache_path)
    assert cache["direct_forward_gate_passed"] is True
    assert cache["direct_forward_max_abs_error"] <= 0.01

    memory = AuthenticMemory(8, 6, 3)
    memory_path = tmp_path / "memory.pt"
    coverage = CoverageMetrics(8, 3, 0.1, 0.2, 0.05, 0.1, {"camera-a": 0.1})
    save_memory_artifact(
        memory,
        memory_path,
        source_cache_sha256="cache",
        backbone_model_id="test/backbone",
        backbone_revision="immutable",
        manifest_sha256="manifest",
        history=[],
        coverage=coverage,
    )
    loaded = load_memory_artifact(
        memory_path,
        expected_backbone_model_id="test/backbone",
        expected_backbone_revision="immutable",
        expected_manifest_sha256="manifest",
    )
    assert loaded.dimension == 6
    with pytest.raises(ValueError, match="backbone_revision"):
        load_memory_artifact(memory_path, expected_backbone_revision="different")


def test_pair_resolution_optimizer_groups_and_scheduler() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    dda, source = paired_dda_logits(
        {
            "sample_kind": ["authentic", "native_aigc", "dda"],
            "parent_id": ["real", "native", "aligned"],
            "source_parent_id": ["", "", "real"],
        },
        logits,
    )
    torch.testing.assert_close(dda, torch.tensor([2.0]))
    torch.testing.assert_close(source, torch.tensor([0.0]))

    model = TraceRXM(TinyEncoder(), AuthenticMemory(8, 6, 3), HeadConfig())
    model.configure_for_detection()
    config = OptimizerConfig(adapter_lr=1e-5, head_lr=1e-4)
    optimizer = build_detection_optimizer(model, config)
    assert {group["name"] for group in optimizer.param_groups} == {"lora", "heads"}
    scheduler = cosine_warmup_scheduler(optimizer, total_steps=10, warmup_fraction=0.2)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_s4_checkpoint_can_include_resume_state_and_selection_metadata(tmp_path) -> None:
    model = TraceRXM(TinyEncoder(), AuthenticMemory(8, 6, 3), HeadConfig())
    model.configure_for_detection()
    optimizer = build_detection_optimizer(model, OptimizerConfig())
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps=4,
        warmup_fraction=0.25,
    )
    path = tmp_path / "epoch-0001.pt"

    save_detector_checkpoint(
        model,
        path,
        config={},
        memory_artifact_sha256="memory",
        manifest_sha256="manifest",
        history=[],
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        selection_metric={"name": "training_total_loss", "value": 0.5},
    )
    artifact = torch.load(path, map_location="cpu", weights_only=True)

    assert artifact["epoch"] == 1
    assert artifact["selection_metric"]["name"] == "training_total_loss"
    assert "optimizer_state" in artifact
    assert "scheduler_state" in artifact
