"""Authentic-only prototype memory and sparse FP32 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalOutput:
    reference: Tensor
    max_similarity: Tensor
    max_attention: Tensor
    entropy: Tensor
    indices: Tensor
    weights: Tensor


@dataclass(frozen=True)
class MemoryLoss:
    total: Tensor
    reconstruction: Tensor
    diversity: Tensor


class AuthenticMemory(nn.Module):
    """K authentic prototypes with exact sparse top-k cross-attention.

    This keeps MIRROR's prototype retrieval/reconstruction structure while
    implementing TRACE-RX-M's stricter S2 ledger: M is the only learned memory
    parameter, rather than a multi-head q/k/v/out attention block. The source
    comparison is MIRROR ``models/mirror.py`` at commit 18c56efa.

    Queries, keys, and values are L2-normalized as required by S2. Similarity
    scoring, top-k selection, softmax, and entropy always run in FP32.
    ``s_max`` is maximum sparse-attention probability; ``s_ent`` is entropy.
    Maximum cosine similarity remains available as a diagnostic.
    """

    def __init__(
        self,
        size: int,
        dimension: int,
        topk: int,
        *,
        score_chunk_size: int = 512,
        prototypes: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0 < topk <= size:
            raise ValueError("topk must lie in [1, size].")
        self.size = int(size)
        self.dimension = int(dimension)
        self.topk = int(topk)
        self.score_chunk_size = int(score_chunk_size)
        initial = torch.randn(size, dimension) if prototypes is None else prototypes.detach().clone()
        if initial.shape != (size, dimension):
            raise ValueError(f"Expected prototypes {(size, dimension)}, got {tuple(initial.shape)}.")
        self.prototypes = nn.Parameter(F.normalize(initial.float(), dim=-1))

    @torch.no_grad()
    def initialize_kmeans(self, tokens: Tensor, *, seed: int = 0, batch_size: int = 4096) -> None:
        """Initialize every prototype from normalized authentic tokens."""

        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as error:  # pragma: no cover - declared dependency
            raise RuntimeError("scikit-learn is required for prototype initialization.") from error
        flat = F.normalize(tokens.detach().float().flatten(0, -2), dim=-1).cpu().numpy()
        if flat.shape[0] < self.size:
            raise ValueError("K-means initialization needs at least K authentic patch tokens.")
        kmeans = MiniBatchKMeans(
            n_clusters=self.size,
            random_state=seed,
            batch_size=max(batch_size, self.size),
            n_init=3,
        ).fit(flat)
        centers = torch.from_numpy(np.asarray(kmeans.cluster_centers_)).to(self.prototypes)
        self.prototypes.copy_(F.normalize(centers, dim=-1))

    def forward(self, tokens: Tensor) -> RetrievalOutput:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dimension:
            raise ValueError(f"Expected [B, P, {self.dimension}] tokens.")
        batch, patches, _ = tokens.shape
        queries = F.normalize(tokens.float(), dim=-1).reshape(-1, self.dimension)
        keys = F.normalize(self.prototypes.float(), dim=-1)
        best_scores: Tensor | None = None
        best_indices: Tensor | None = None
        for start in range(0, self.size, self.score_chunk_size):
            stop = min(start + self.score_chunk_size, self.size)
            scores = queries @ keys[start:stop].t()
            indices = torch.arange(start, stop, device=scores.device).expand(scores.shape[0], -1)
            if best_scores is not None and best_indices is not None:
                scores = torch.cat((best_scores, scores), dim=-1)
                indices = torch.cat((best_indices, indices), dim=-1)
            keep = min(self.topk, scores.shape[-1])
            best_scores, positions = torch.topk(scores, keep, dim=-1, sorted=True)
            best_indices = torch.gather(indices, 1, positions)
        if best_scores is None or best_indices is None:  # pragma: no cover - size validated
            raise RuntimeError("Empty memory.")
        # TRACE-RX-M explicitly specifies sqrt(D), whereas MIRROR's multi-head
        # module uses sqrt(head_dim). There are no learned attention heads here.
        attention_logits = best_scores / math.sqrt(self.dimension)
        weights = torch.softmax(attention_logits, dim=-1)
        selected = keys[best_indices]
        reference = torch.sum(weights.unsqueeze(-1) * selected, dim=-2)
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)
        return RetrievalOutput(
            reference=reference.reshape(batch, patches, self.dimension),
            max_similarity=best_scores[:, 0].reshape(batch, patches),
            entropy=entropy.reshape(batch, patches),
            max_attention=weights[:, 0].reshape(batch, patches),
            indices=best_indices.reshape(batch, patches, self.topk),
            weights=weights.reshape(batch, patches, self.topk),
        )

    def phase1_loss(self, authentic_tokens: Tensor, diversity_weight: float) -> MemoryLoss:
        normalized = F.normalize(authentic_tokens.float(), dim=-1)
        retrieved = self(normalized)
        reconstruction = (normalized - retrieved.reference).square().sum(dim=-1).mean()
        normalized_memory = F.normalize(self.prototypes.float(), dim=-1)
        gram = normalized_memory @ normalized_memory.t()
        identity = torch.eye(self.size, dtype=gram.dtype, device=gram.device)
        diversity = torch.linalg.matrix_norm(gram - identity, ord="fro")
        return MemoryLoss(
            total=reconstruction + diversity_weight * diversity,
            reconstruction=reconstruction,
            diversity=diversity,
        )

    @torch.no_grad()
    def usage_histogram(self, tokens: Tensor) -> Tensor:
        indices = self(tokens).indices.flatten()
        return torch.bincount(indices, minlength=self.size)


def residual_patch_statistics(residual: Tensor, quantile: float = 0.95) -> Tensor:
    """Retain signed mean, standard deviation, and signed upper quantile (3D).

    This follows H2's stronger aggregation requirement. The following linear
    projection maps the 3D concatenation back to the evidence dimension.
    """

    if residual.ndim != 3:
        raise ValueError("Residual must have shape [batch, patches, dimension].")
    mean = residual.mean(dim=1)
    std = residual.std(dim=1, unbiased=False)
    upper = torch.quantile(residual.float(), quantile, dim=1).to(residual.dtype)
    return torch.cat((mean, std, upper), dim=-1)
