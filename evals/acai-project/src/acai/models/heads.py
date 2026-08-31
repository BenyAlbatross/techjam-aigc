"""Classifier heads and the four DINOv3 x forensic-feature fusion designs (plan §3.2).

    H0  concat        [dino ; z(feats)] -> MLP.          Control arm.
    H1  FiLM          feats -> (gamma, beta) modulating the pooled embedding.
    H2  dense tokens  per-patch feature maps -> d_model -> added to patch tokens
                      inside the last N blocks.          Primary arm.
    H3  auxiliary     predict the feature values from DINOv3 tokens as an aux loss.

Why H2 is the one worth betting on: H0 and H1 can only ever see three *global* scalars, so
they can express "this image has heavy tails" but not "this region does". H2 keeps the
statistics spatially aligned with the tokens, which lets attention localise the evidence.
That is a difference in kind, not degree, and it is the hypothesis the ablation tests.

H0/H1/H3 read `feats` [B, 3] from acai.features.feature_vector.
H2 reads `feat_maps` [B, 3, g, g] from acai.features.feature_maps, on the same patch grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from acai.features.lowlevel import FEATURES

N_FEATS = len(FEATURES)


class FeatureNorm(nn.Module):
    """Standardise the handcrafted features using statistics frozen from the training split.

    Kurtosis is unbounded and routinely lands in the tens, so feeding it raw next to a
    LayerNorm'd 768-d embedding would let it dominate or vanish depending on the batch.
    Statistics are *frozen*, not batch-estimated: a BatchNorm here would leak information
    between samples in a batch and quietly inflate eval numbers.
    """

    def __init__(self, n: int = N_FEATS):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sigma", torch.ones(n))
        self.fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "FeatureNorm":
        self.mu.copy_(x.mean(0))
        # Median-absolute-deviation-like floor: kurtosis has a long right tail, and a
        # near-zero sigma on a degenerate split would blow the normalised values up.
        self.sigma.copy_(x.std(0).clamp_min(1e-3))
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.fitted:
            raise RuntimeError(
                "FeatureNorm used before fit(): call Detector.fit_feature_norm(train_feats) "
                "first. Training on raw kurtosis (values ~10 next to a LayerNorm'd embedding) "
                "silently changes what every fusion arm learns, and the ablation would then "
                "be comparing normalisation regimes rather than fusion designs.")
        return (x - self.mu) / self.sigma


def mlp(d_in: int, hidden: int, d_out: int = 1, p: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
        nn.Dropout(p), nn.Linear(hidden, d_out),
    )


# --------------------------------------------------------------------------- fusions

class NoFusion(nn.Module):
    """Pure DINOv3 baseline (plan §3.1). Linear or MLP probe on the pooled embedding."""

    def __init__(self, d: int, hidden: int = 0, p: float = 0.1):
        super().__init__()
        self.net = nn.Linear(d, 1) if hidden == 0 else mlp(d, hidden, 1, p)

    def forward(self, emb, feats=None, **_):
        return self.net(emb).squeeze(-1), {}


class ConcatFusion(nn.Module):
    """H0. The control. Three standardised scalars appended to a 768-d embedding.

    Expected to underperform H1/H2: at 768:3 the features contribute ~0.4% of the input
    width, so an unregularised MLP has little pressure to use them. That is the point of
    running it -- without this arm, a win from H1/H2 could not be attributed to the
    *fusion design* rather than to merely having the features at all.
    """

    def __init__(self, d: int, hidden: int = 256, p: float = 0.1):
        super().__init__()
        self.norm = FeatureNorm()
        self.net = mlp(d + N_FEATS, hidden, 1, p)

    def forward(self, emb, feats=None, **_):
        return self.net(torch.cat([emb, self.norm(feats)], -1)).squeeze(-1), {}


class FiLMFusion(nn.Module):
    """H1. Features generate (gamma, beta) that modulate the pooled embedding.

    Multiplicative rather than additive so the forensic signal can *reweight* semantic
    dimensions -- "given heavy tails, attend to these features differently" -- which
    concatenation cannot express at any width.
    """

    def __init__(self, d: int, hidden: int = 256, p: float = 0.1):
        super().__init__()
        self.norm = FeatureNorm()
        self.film = nn.Sequential(nn.Linear(N_FEATS, 64), nn.GELU(), nn.Linear(64, 2 * d))
        # Zero-init the last layer: gamma starts at 1 and beta at 0, so training begins
        # from exactly the pure-DINOv3 model and any gain is attributable to the fusion.
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.net = mlp(d, hidden, 1, p)

    def forward(self, emb, feats=None, **_):
        g, b = self.film(self.norm(feats)).chunk(2, -1)
        return self.net(emb * (1 + g) + b).squeeze(-1), {}


class DenseTokenFusion(nn.Module):
    """H2 (primary). Per-patch forensic maps projected to d_model and injected as a
    side-channel on the patch tokens inside the last `n_blocks` transformer blocks.

    Injection is done with forward pre-hooks rather than by reimplementing the backbone's
    forward pass: timm's DINOv3 is the RoPE variant, whose blocks take a `rope` kwarg and
    have a separate `rope_mixed` path. Duplicating that logic would silently break on a
    timm update; a hook rides on whatever the real forward does.

    The projection is zero-initialised, so the model starts identical to pure DINOv3 and
    the ablation measures only what the forensic channel adds.
    """

    def __init__(self, backbone, hidden: int = 256, n_blocks: int = 2, p: float = 0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.width
        self.map_norm = nn.GroupNorm(1, N_FEATS)      # per-image, per-channel; no batch leak
        self.proj = nn.Sequential(nn.Conv2d(N_FEATS, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.net = mlp(backbone.out_dim, hidden, 1, p)

        self._side: torch.Tensor | None = None
        self._handles = []
        blocks = backbone.model.blocks
        if n_blocks > len(blocks):
            raise ValueError(f"n_blocks={n_blocks} exceeds depth {len(blocks)}")
        for blk in blocks[-n_blocks:]:
            self._handles.append(blk.register_forward_pre_hook(self._inject, with_kwargs=True))

    def _inject(self, module, args, kwargs):
        if self._side is None:
            return None
        x = args[0]
        n = self.backbone.n_prefix
        if x.shape[1] - n != self._side.shape[1]:
            raise RuntimeError(
                f"forensic map has {self._side.shape[1]} positions but the backbone has "
                f"{x.shape[1] - n} patch tokens -- feature_maps(patch=16) and img_size disagree")
        x = torch.cat([x[:, :n], x[:, n:] + self._side], dim=1)
        return (x, *args[1:]), kwargs

    def forward(self, emb=None, feats=None, feat_maps=None, images=None, **_):
        """Runs the backbone itself, since the side-channel must be live during its forward."""
        if feat_maps is None or images is None:
            raise ValueError("H2 needs both `images` and `feat_maps`")
        side = self.proj(self.map_norm(feat_maps))            # [B, D, g, g]
        self._side = side.flatten(2).transpose(1, 2)          # [B, g*g, D]
        try:
            pooled = self.backbone(images)
        finally:
            self._side = None                                  # never leak across batches
        return self.net(pooled).squeeze(-1), {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class AuxRegressionFusion(nn.Module):
    """H3. Predict the three feature values from the embedding as an auxiliary task.

    Costs nothing at inference: the aux head is dropped. Tests whether forensic-awareness
    helps as a *regulariser* rather than as an input, which is a genuinely different
    hypothesis from H0-H2 and is the cheap one to be wrong about.
    """

    def __init__(self, d: int, hidden: int = 256, p: float = 0.1, aux_weight: float = 0.1):
        super().__init__()
        self.norm = FeatureNorm()
        self.net = mlp(d, hidden, 1, p)
        self.aux = nn.Linear(d, N_FEATS)
        self.aux_weight = aux_weight

    def forward(self, emb, feats=None, **_):
        extra = {}
        if feats is not None:
            extra["aux_pred"] = self.aux(emb)
            extra["aux_target"] = self.norm(feats)
            extra["aux_weight"] = self.aux_weight
        return self.net(emb).squeeze(-1), extra


HEADS = {"none": NoFusion, "h0": ConcatFusion, "h1": FiLMFusion,
         "h2": DenseTokenFusion, "h3": AuxRegressionFusion}


class Detector(nn.Module):
    """Backbone + head, with one forward signature shared by every arm.

    A single signature matters for the ablation: each arm must be swappable without
    touching the training loop, or differences between arms start including differences
    in how they were trained.
    """

    def __init__(self, backbone, fusion: str = "none", hidden: int = 256,
                 p: float = 0.1, **kw):
        super().__init__()
        if fusion not in HEADS:
            raise ValueError(f"fusion must be one of {list(HEADS)}")
        self.backbone, self.fusion = backbone, fusion
        self.head = (DenseTokenFusion(backbone, hidden, p=p, **kw) if fusion == "h2"
                     else HEADS[fusion](backbone.out_dim, hidden, p=p, **kw))

    def forward(self, images, feats=None, feat_maps=None):
        if self.fusion == "h2":
            return self.head(images=images, feat_maps=feat_maps, feats=feats)
        return self.head(self.backbone(images), feats=feats)

    def fit_feature_norm(self, feats: torch.Tensor) -> "Detector":
        """Freeze feature standardisation statistics from the **training split only**.

        Must be called before training any arm that consumes `feats` (H0/H1/H3). Fitting on
        train+eval would leak eval distribution into the model.
        """
        n = 0
        for m in self.modules():
            if isinstance(m, FeatureNorm):
                m.fit(feats)
                n += 1
        if n == 0 and self.fusion in ("h0", "h1", "h3"):
            raise RuntimeError(f"fusion {self.fusion} has no FeatureNorm to fit")
        return self

    def loss(self, logits, y, extra: dict) -> torch.Tensor:
        l = nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        if "aux_pred" in extra:
            l = l + extra["aux_weight"] * nn.functional.mse_loss(
                extra["aux_pred"], extra["aux_target"])
        return l
