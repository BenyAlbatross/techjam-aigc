"""No-memory direct DINOv3 probe used by the ablation suite."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from .config import HeadConfig


@dataclass(frozen=True)
class TraceRXMOutput:
    logit: Tensor
    patch_tokens: Tensor


class TraceRXM(nn.Module):
    """LoRA encoder followed by a mean-pooled patch-token MLP classifier."""

    def __init__(self, encoder: nn.Module, config: HeadConfig) -> None:
        super().__init__()
        output_dim = int(getattr(encoder, "output_dim"))
        self.encoder = encoder
        self.config = config
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, pixel_values: Tensor) -> TraceRXMOutput:
        patch_tokens = F.normalize(self.encoder(pixel_values).float(), dim=-1)
        logit = self.classifier(patch_tokens.mean(dim=1)).squeeze(-1)
        return TraceRXMOutput(logit=logit, patch_tokens=patch_tokens)

    def configure_for_detection(self, *, frozen_encoder_fallback: bool = False) -> None:
        self.requires_grad_(False)
        if not frozen_encoder_fallback:
            for name, parameter in self.encoder.named_parameters():
                if name.endswith(("lora_A", "lora_B")):
                    parameter.requires_grad_(True)
        self.classifier.requires_grad_(True)

    def head_parameters(self):
        yield from self.classifier.parameters()
