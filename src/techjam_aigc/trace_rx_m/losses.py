"""TRACE-RX-M v2 S4 objective terms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import LossConfig


@dataclass(frozen=True)
class DetectionLoss:
    total: Tensor
    bce: Tensor
    pauc: Tensor
    pair: Tensor


def balanced_binary_cross_entropy(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    """Weighted BCE; weights encode master, class, and within-class group balance."""

    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError("logits, labels, and weights must have equal shapes.")
    per_sample = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return torch.sum(per_sample * weights) / weights.sum().clamp_min(1e-12)


def partial_auc_surrogate(
    logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = 0.05,
    margin: float = 0.2,
) -> Tensor:
    """Pair every positive with the highest-scoring alpha tail of authentic negatives."""

    positives = logits[labels == 1]
    negatives = logits[labels == 0]
    if not positives.numel() or not negatives.numel():
        return logits.sum() * 0
    tail_size = max(1, math.ceil(alpha * negatives.numel()))
    hardest = torch.topk(negatives, k=tail_size).values
    return F.softplus(margin - positives[:, None] + hardest[None, :]).mean()


def dda_pair_ranking_loss(
    dda_logits: Tensor,
    source_real_logits: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    if dda_logits.shape != source_real_logits.shape:
        raise ValueError("Every DDA logit needs its paired real-source logit.")
    if not dda_logits.numel():
        return dda_logits.sum() * 0
    return F.softplus(margin - dda_logits + source_real_logits).mean()


def detection_objective(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    *,
    dda_logits: Tensor,
    source_real_logits: Tensor,
    config: LossConfig,
    primary_mask: Tensor | None = None,
) -> DetectionLoss:
    if primary_mask is None:
        primary_mask = torch.ones_like(labels, dtype=torch.bool)
    if primary_mask.shape != labels.shape or not primary_mask.any():
        raise ValueError("primary_mask must select at least one primary sample.")
    bce = balanced_binary_cross_entropy(logits[primary_mask], labels[primary_mask], weights[primary_mask])
    pauc = partial_auc_surrogate(
        logits[primary_mask],
        labels[primary_mask],
        alpha=config.pauc_alpha,
        margin=config.ranking_margin,
    )
    pair = dda_pair_ranking_loss(
        dda_logits,
        source_real_logits,
        margin=config.ranking_margin,
    )
    total = bce + config.pauc_weight * pauc + config.pair_weight * pair
    return DetectionLoss(total=total, bce=bce, pauc=pauc, pair=pair)
