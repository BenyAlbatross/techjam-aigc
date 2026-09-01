"""Frozen, class-relative prototype retrieval for the three-branch detector."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ClassRetrieval:
    reference: Tensor
    residual: Tensor
    distance: Tensor
    max_similarity: Tensor
    entropy: Tensor
    indices: Tensor


@dataclass(frozen=True)
class DualRetrieval:
    authentic: ClassRetrieval
    synthetic: ClassRetrieval


class DualPrototypeMemory(nn.Module):
    """Two frozen prototype dictionaries fitted in the unadapted encoder space."""

    def __init__(
        self,
        authentic_prototypes: Tensor,
        synthetic_prototypes: Tensor,
        *,
        topk: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if authentic_prototypes.ndim != 2 or synthetic_prototypes.ndim != 2:
            raise ValueError("Prototype tensors must have shape [K, D].")
        if authentic_prototypes.shape != synthetic_prototypes.shape:
            raise ValueError("Authentic and synthetic memories must have identical shapes.")
        if not 0 < topk <= authentic_prototypes.shape[0]:
            raise ValueError("topk must lie in [1, K].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.register_buffer(
            "authentic_prototypes",
            F.normalize(authentic_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer(
            "synthetic_prototypes",
            F.normalize(synthetic_prototypes.detach().float(), dim=-1),
        )
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.dimension = int(authentic_prototypes.shape[1])

    def _retrieve(self, tokens: Tensor, prototypes: Tensor) -> ClassRetrieval:
        scores = tokens.float() @ prototypes.float().t()
        best_scores, indices = torch.topk(scores, k=self.topk, dim=-1, sorted=True)
        weights = torch.softmax(best_scores / self.temperature, dim=-1)
        selected = prototypes[indices]
        reference = F.normalize(torch.sum(weights.unsqueeze(-1) * selected, dim=-2), dim=-1)
        residual = tokens.float() - reference
        distance = residual.square().sum(dim=-1)
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
        if self.topk > 1:
            entropy = entropy / math.log(self.topk)
        else:
            entropy = torch.zeros_like(entropy)
        return ClassRetrieval(
            reference=reference,
            residual=residual,
            distance=distance,
            max_similarity=best_scores[..., 0],
            entropy=entropy,
            indices=indices,
        )

    def forward(self, tokens: Tensor) -> DualRetrieval:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dimension:
            raise ValueError(f"Expected normalized tokens with shape [B, P, {self.dimension}].")
        normalized = F.normalize(tokens.float(), dim=-1)
        return DualRetrieval(
            authentic=self._retrieve(normalized, self.authentic_prototypes),
            synthetic=self._retrieve(normalized, self.synthetic_prototypes),
        )

    def artifact_state(self) -> dict[str, object]:
        return {
            "authentic_prototypes": self.authentic_prototypes.detach().cpu(),
            "synthetic_prototypes": self.synthetic_prototypes.detach().cpu(),
            "topk": self.topk,
            "temperature": self.temperature,
            "dimension": self.dimension,
        }
