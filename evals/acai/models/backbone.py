"""DINOv3 backbone wrapper.

Loaded from the `timm/*` repos rather than `facebook/*`: the latter are gated and 403 for
this account, while timm carries the same official LVD-1689M weights. timm is also the
better fit for what we need -- `dynamic_img_size` for the resolution sweep, and direct
access to blocks for LoRA and for the H2 token injection.

Input resolution is treated as a first-class experimental variable, not a default. DINOv3
at 224px discards most of the high-frequency content that the forensic features key on, and
high-frequency content is the entire premise of this project, so `img_size` is swept
(224/336/512) in plan §3.1 rather than left at the pretrained default.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn

VARIANTS = {
    "s": "vit_small_patch16_dinov3.lvd1689m",     # 384-d
    "b": "vit_base_patch16_dinov3.lvd1689m",      # 768-d
    "l": "vit_large_patch16_dinov3.lvd1689m",     # 1024-d
}

# ImageNet statistics, matching the pretrained config. Kept explicit rather than pulled
# from timm's resolve_data_config so the transform pipeline and the model agree visibly.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DinoV3(nn.Module):
    """DINOv3 feature extractor returning pooled embeddings and/or patch tokens.

    `pool`:
        cls      the CLS token
        mean     mean over patch tokens
        cls_mean concat of both (2x width) -- usually the strongest linear probe
    """

    def __init__(self, variant: str = "b", img_size: int = 224, pool: str = "cls_mean",
                 pretrained: bool = True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        if img_size % 16:
            raise ValueError(f"img_size must be a multiple of the patch size 16, got {img_size}")

        self.variant, self.img_size, self.pool = variant, img_size, pool
        self.model = timm.create_model(
            VARIANTS[variant], pretrained=pretrained, num_classes=0,
            img_size=img_size, dynamic_img_size=True,
        )
        self.width = self.model.embed_dim
        self.grid = img_size // 16
        # DINOv3 carries 4 register tokens alongside CLS (5 prefix tokens total). They must
        # be stripped before patch tokens are reshaped to a grid, or the grid misaligns and
        # every dense feature in H2 is silently wrong. Read from timm rather than hardcoded,
        # and asserted below, because this count differs across DINO generations.
        self.n_prefix = int(self.model.num_prefix_tokens)
        if self.n_prefix < 1:
            raise RuntimeError("expected at least a CLS token; timm layout changed?")

    @property
    def out_dim(self) -> int:
        return self.width * 2 if self.pool == "cls_mean" else self.width

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (cls [B, D], patch tokens [B, N, D]) with prefix/register tokens removed."""
        t = self.model.forward_features(x)
        return t[:, 0], t[:, self.n_prefix:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, patches = self.forward_tokens(x)
        if self.pool == "cls":
            return cls
        if self.pool == "mean":
            return patches.mean(1)
        if self.pool == "cls_mean":
            return torch.cat([cls, patches.mean(1)], dim=-1)
        raise ValueError(f"unknown pool {self.pool!r}")

    def patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens as a spatial grid [B, D, H/16, W/16], for H2 alignment."""
        _, p = self.forward_tokens(x)
        b, n, d = p.shape
        g = int(math.isqrt(n))
        if g * g != n:
            raise ValueError(f"{n} patch tokens is not a square grid; non-square input?")
        return p.transpose(1, 2).reshape(b, d, g, g)

    # ----------------------------------------------------------------- finetuning

    def freeze(self) -> "DinoV3":
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze_last(self, n_blocks: int = 1) -> "DinoV3":
        self.freeze()
        for blk in self.model.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.model.norm.parameters():
            p.requires_grad_(True)
        return self

    def add_lora(self, rank: int = 8, alpha: int = 16, targets=("qkv", "proj")) -> "DinoV3":
        """Attach LoRA adapters to attention projections; base weights stay frozen.

        Chosen over full finetuning for the hybrid arms because the composition study
        (plan §4) runs many training jobs, and full finetunes of a ViT-L across a mixture
        simplex would dominate the compute budget without changing the ranking.
        """
        self.freeze()
        n = 0
        for blk in self.model.blocks:
            for name in targets:
                mod = getattr(blk.attn, name, None)
                if isinstance(mod, nn.Linear):
                    setattr(blk.attn, name, LoRALinear(mod, rank, alpha))
                    n += 1
        if n == 0:
            raise RuntimeError(f"no LoRA targets matched {targets}; timm layout changed?")
        return self

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen nn.Linear: y = Wx + (alpha/r) * B(A(x))."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)      # starts as an exact no-op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.b(self.a(x)) * self.scale
