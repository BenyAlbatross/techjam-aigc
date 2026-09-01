"""Global, dual-memory, and native-forensic branches with transparent fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import HeadConfig
from .memory import DualPrototypeMemory


def score_distribution_pool(scores: Tensor, top_fraction: float) -> Tensor:
    """Pool patch/crop scores without discarding their tail behavior."""

    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("Scores must have shape [batch, locations].")
    count = max(1, math.ceil(scores.shape[1] * top_fraction))
    top = torch.topk(scores, k=count, dim=1).values
    return torch.stack(
        (
            scores.mean(dim=1),
            scores.std(dim=1, unbiased=False),
            top.mean(dim=1),
            scores.max(dim=1).values,
        ),
        dim=-1,
    )


class HighPassResidual(nn.Module):
    """Fixed separable five-tap Gaussian residual, evaluated in native pixels."""

    def __init__(self) -> None:
        super().__init__()
        vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(vector, vector)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("High-pass input must have shape [N, 3, H, W].")
        padded = F.pad(rgb, (2, 2, 2, 2), mode="reflect")
        low = F.conv2d(padded, self.kernel.to(dtype=rgb.dtype), groups=3)
        return rgb - low


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    groups = min(8, out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class NativeForensicEncoder(nn.Module):
    """Lightweight CNN over native RGB and a fixed high-pass residual."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.high_pass = HighPassResidual()
        self.network = nn.Sequential(
            _conv_block(6, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 96, stride=2),
            _conv_block(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.GELU(),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        residual = self.high_pass(rgb)
        return self.network(torch.cat((rgb, residual), dim=1))


@dataclass(frozen=True)
class ThreeBranchOutput:
    logit: Tensor
    global_logit: Tensor
    memory_logit: Tensor
    forensic_logit: Tensor
    patch_logits: Tensor
    crop_logits: Tensor
    authentic_distance: Tensor
    synthetic_distance: Tensor


class ThreeBranchDetector(nn.Module):
    """Fuse complementary semantic, relative-reference, and forensic evidence."""

    def __init__(
        self,
        encoder: nn.Module,
        memory: DualPrototypeMemory,
        config: HeadConfig,
    ) -> None:
        super().__init__()
        dimension = int(getattr(encoder, "output_dim"))
        if dimension != memory.dimension:
            raise ValueError("Encoder and dual-memory dimensions differ.")
        self.encoder = encoder
        self.memory = memory
        self.config = config
        self.global_head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        patch_input_dim = 2 * dimension + 7
        self.memory_patch_head = nn.Sequential(
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, config.patch_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.patch_hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.memory_pool_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.forensic_encoder = NativeForensicEncoder(config.forensic_embedding_dim)
        self.forensic_crop_head = nn.Sequential(
            nn.LayerNorm(config.forensic_embedding_dim),
            nn.Linear(config.forensic_embedding_dim, 1),
        )
        self.forensic_pool_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
        # Begin as an equal expert average. The residual fusion learns only if
        # branch disagreement is predictively useful.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, global_pixels: Tensor, native_crops: Tensor) -> ThreeBranchOutput:
        if native_crops.ndim != 5 or native_crops.shape[2] != 3:
            raise ValueError("native_crops must have shape [B, N, 3, H, W].")
        tokens = F.normalize(self.encoder(global_pixels).float(), dim=-1)
        global_logit = self.global_head(tokens.mean(dim=1)).squeeze(-1)

        retrieved = self.memory(tokens)
        scalars = torch.stack(
            (
                retrieved.authentic.max_similarity,
                retrieved.synthetic.max_similarity,
                retrieved.authentic.entropy,
                retrieved.synthetic.entropy,
                retrieved.authentic.distance,
                retrieved.synthetic.distance,
                retrieved.authentic.distance - retrieved.synthetic.distance,
            ),
            dim=-1,
        )
        patch_features = torch.cat(
            (retrieved.authentic.residual, retrieved.synthetic.residual, scalars),
            dim=-1,
        )
        patch_logits = self.memory_patch_head(patch_features).squeeze(-1)
        memory_logit = self.memory_pool_head(
            score_distribution_pool(patch_logits, self.config.top_fraction)
        ).squeeze(-1)

        batch, crops, channels, height, width = native_crops.shape
        crop_embeddings = self.forensic_encoder(
            native_crops.reshape(batch * crops, channels, height, width)
        )
        crop_logits = self.forensic_crop_head(crop_embeddings).reshape(batch, crops)
        forensic_logit = self.forensic_pool_head(
            score_distribution_pool(crop_logits, self.config.top_fraction)
        ).squeeze(-1)

        branches = torch.stack((global_logit, memory_logit, forensic_logit), dim=-1)
        disagreements = torch.stack(
            (
                (global_logit - memory_logit).abs(),
                (global_logit - forensic_logit).abs(),
                (memory_logit - forensic_logit).abs(),
            ),
            dim=-1,
        )
        logit = branches.mean(dim=-1) + self.fusion(
            torch.cat((branches, disagreements), dim=-1)
        ).squeeze(-1)
        return ThreeBranchOutput(
            logit=logit,
            global_logit=global_logit,
            memory_logit=memory_logit,
            forensic_logit=forensic_logit,
            patch_logits=patch_logits,
            crop_logits=crop_logits,
            authentic_distance=retrieved.authentic.distance,
            synthetic_distance=retrieved.synthetic.distance,
        )

    def configure_for_training(self) -> None:
        self.requires_grad_(False)
        for name, parameter in self.encoder.named_parameters():
            if name.endswith(("lora_A", "lora_B")):
                parameter.requires_grad_(True)
        for module in (
            self.global_head,
            self.memory_patch_head,
            self.memory_pool_head,
            self.forensic_encoder,
            self.forensic_crop_head,
            self.forensic_pool_head,
            self.fusion,
        ):
            module.requires_grad_(True)

    def adapter_parameters(self):
        return (
            parameter
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and name.endswith(("lora_A", "lora_B"))
        )

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and not name.startswith("encoder."):
                yield parameter

    def trainable_state_dict(self) -> dict[str, Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable
        }
