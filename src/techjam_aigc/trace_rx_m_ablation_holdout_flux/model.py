"""The TRACE-RX-M v2 reference-comparison detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import HeadConfig
from .memory import AuthenticMemory, residual_patch_statistics


class PatchEncoder(Protocol):
    output_dim: int

    def __call__(self, pixel_values: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class TraceRXMOutput:
    logit: Tensor
    patch_tokens: Tensor
    reference: Tensor
    residual: Tensor
    s_max: Tensor
    s_ent: Tensor
    memory_indices: Tensor


class TraceRXM(nn.Module):
    """LoRA encoder -> authentic memory -> directional residual evidence."""

    def __init__(self, encoder: nn.Module, memory: AuthenticMemory, config: HeadConfig) -> None:
        super().__init__()
        output_dim = int(getattr(encoder, "output_dim"))
        if output_dim != memory.dimension:
            raise ValueError("Encoder and authentic-memory dimensions differ.")
        self.encoder = encoder
        self.memory = memory
        self.config = config
        self.perplexity_mlp = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.evidence_dim),
        )
        self.residual_projection = nn.Linear(3 * output_dim, config.evidence_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * config.evidence_dim),
            nn.Linear(2 * config.evidence_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, pixel_values: Tensor) -> TraceRXMOutput:
        patch_tokens = F.normalize(self.encoder(pixel_values).float(), dim=-1)
        # Frozen means the dictionary cannot update; it does not mean retrieval
        # is detached. Unlike MIRROR's surrounding torch.no_grad(), keeping this
        # path differentiable lets LoRA learn through both retrieval confidence
        # and reconstruction, as required by TRACE-RX-M's adapted S4 objective.
        retrieval = self.memory(patch_tokens)
        residual = patch_tokens - retrieval.reference
        pooled = residual_patch_statistics(residual, self.config.residual_tail_quantile)
        residual_evidence = self.residual_projection(pooled)

        # The report names two scalar retrieval statistics but does not specify
        # patch aggregation. Population means are the minimal faithful choice;
        # H2's mean/std/tail requirement is retained in the residual branch.
        s_max = retrieval.max_attention.mean(dim=1)
        s_ent = retrieval.entropy.mean(dim=1)
        retrieval_evidence = self.perplexity_mlp(torch.stack((s_max, s_ent), dim=-1))
        logit = self.classifier(torch.cat((retrieval_evidence, residual_evidence), dim=-1)).squeeze(-1)
        return TraceRXMOutput(
            logit=logit,
            patch_tokens=patch_tokens,
            reference=retrieval.reference,
            residual=residual,
            s_max=s_max,
            s_ent=s_ent,
            memory_indices=retrieval.indices,
        )

    def configure_for_memory_fit(self) -> None:
        self.requires_grad_(False)
        self.memory.prototypes.requires_grad_(True)

    def configure_for_detection(self, *, frozen_encoder_fallback: bool = False) -> None:
        self.requires_grad_(False)
        self.memory.prototypes.requires_grad_(False)
        if not frozen_encoder_fallback:
            for name, parameter in self.encoder.named_parameters():
                if name.endswith(("lora_A", "lora_B")):
                    parameter.requires_grad_(True)
        for module in (self.perplexity_mlp, self.residual_projection, self.classifier):
            module.requires_grad_(True)

    def head_parameters(self):
        for module in (self.perplexity_mlp, self.residual_projection, self.classifier):
            yield from module.parameters()
