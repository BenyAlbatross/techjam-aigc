"""TRACE-RX with parallel global and authentic-memory detector branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from techjam_aigc.trace_rx_m.memory import (
    AuthenticMemory,
    residual_patch_statistics,
)

from .config import ParallelHeadConfig


class PatchEncoder(Protocol):
    output_dim: int

    def __call__(self, pixel_values: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class TraceRXParallelOutput:
    """Final and branch-level predictions plus memory diagnostics."""

    logit: Tensor
    global_logit: Tensor
    memory_logit: Tensor
    fusion_weights: Tensor
    patch_tokens: Tensor
    reference: Tensor
    residual: Tensor
    s_max: Tensor
    s_ent: Tensor
    memory_indices: Tensor


class TraceRXParallel(nn.Module):
    """Run a global classifier and authentic-memory classifier in parallel.

    Both branches consume the same normalized encoder tokens. The global branch
    classifies pooled image-wide tokens directly, while the memory branch sees
    only authentic-reference residuals and retrieval statistics. A learned
    two-input linear layer performs transparent late fusion of the branch logits.
    """

    def __init__(
        self,
        encoder: nn.Module,
        memory: AuthenticMemory,
        config: ParallelHeadConfig,
    ) -> None:
        super().__init__()
        output_dim = int(getattr(encoder, "output_dim"))
        if output_dim != memory.dimension:
            raise ValueError("Encoder and authentic-memory dimensions differ.")
        self.encoder = encoder
        self.memory = memory
        self.config = config

        self.global_projection = nn.Sequential(
            nn.Linear(3 * output_dim, config.evidence_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.global_classifier = self._branch_classifier(config)

        self.perplexity_mlp = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.evidence_dim),
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(3 * output_dim, config.evidence_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.memory_classifier = self._branch_classifier(config, input_multiplier=2)

        self.fusion = nn.Linear(2, 1)
        with torch.no_grad():
            self.fusion.weight.fill_(0.5)
            self.fusion.bias.zero_()

    @staticmethod
    def _branch_classifier(
        config: ParallelHeadConfig,
        *,
        input_multiplier: int = 1,
    ) -> nn.Sequential:
        input_dim = input_multiplier * config.evidence_dim
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, pixel_values: Tensor) -> TraceRXParallelOutput:
        patch_tokens = F.normalize(self.encoder(pixel_values).float(), dim=-1)

        global_pooled = residual_patch_statistics(
            patch_tokens,
            self.config.global_tail_quantile,
        )
        global_evidence = self.global_projection(global_pooled)
        global_logit = self.global_classifier(global_evidence).squeeze(-1)

        retrieval = self.memory(patch_tokens)
        residual = patch_tokens - retrieval.reference
        residual_pooled = residual_patch_statistics(
            residual,
            self.config.residual_tail_quantile,
        )
        residual_evidence = self.residual_projection(residual_pooled)
        s_max = retrieval.max_attention.mean(dim=1)
        s_ent = retrieval.entropy.mean(dim=1)
        retrieval_evidence = self.perplexity_mlp(
            torch.stack((s_max, s_ent), dim=-1)
        )
        memory_evidence = torch.cat((retrieval_evidence, residual_evidence), dim=-1)
        memory_logit = self.memory_classifier(memory_evidence).squeeze(-1)

        branch_logits = torch.stack((global_logit, memory_logit), dim=-1)
        logit = self.fusion(branch_logits).squeeze(-1)
        return TraceRXParallelOutput(
            logit=logit,
            global_logit=global_logit,
            memory_logit=memory_logit,
            fusion_weights=self.fusion.weight.squeeze(0),
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
        for module in self._head_modules():
            module.requires_grad_(True)

    def _head_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.global_projection,
            self.global_classifier,
            self.perplexity_mlp,
            self.residual_projection,
            self.memory_classifier,
            self.fusion,
        )

    def head_parameters(self):
        for module in self._head_modules():
            yield from module.parameters()
