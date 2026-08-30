"""DINOv3-L patch-token encoder with a small, explicit LoRA implementation."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from .config import BackboneConfig


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        update = (self.dropout(inputs) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(inputs) + update * self.scaling


def inject_lora(
    model: nn.Module,
    *,
    targets: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    """Replace matching linear leaves and return their fully qualified names."""

    target_tuple = tuple(targets)
    replacements: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not name.endswith(target_tuple):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((name, parent, child_name, module))
    for _, parent, child_name, module in replacements:
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    if rank and not replacements:
        raise ValueError(f"No linear modules matched LoRA targets {target_tuple!r}.")
    return tuple(name for name, _, _, _ in replacements)


class DinoV3PatchEncoder(nn.Module):
    """Exact DINOv3-L patch-token extraction, excluding CLS/register tokens."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        config.validate(require_access=True)
        try:
            from transformers import AutoModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the training dependency group to load DINOv3.") from error

        self.config = config
        self.backbone = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
        )
        self.backbone.requires_grad_(False)
        self.output_dim = int(self.backbone.config.hidden_size)
        self.patch_size = int(self.backbone.config.patch_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        if config.gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        self.lora_modules = inject_lora(
            self.backbone,
            targets=config.lora_targets,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        ) if config.lora_rank else ()
        if sum(parameter.numel() for parameter in self.backbone.parameters()) >= 2_000_000_000:
            raise ValueError("The selected backbone violates the challenge's <2B parameter rule.")

    def forward(self, pixel_values: Tensor) -> Tensor:
        output = self.backbone(pixel_values=pixel_values, return_dict=True)
        start = 1 + self.num_register_tokens
        tokens = output.last_hidden_state[:, start:, :]
        expected = (pixel_values.shape[-2] // self.patch_size) * (
            pixel_values.shape[-1] // self.patch_size
        )
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"Patch-token count {tokens.shape[1]} does not match input grid {expected}; "
                "special/register token exclusion or preprocessing is inconsistent."
            )
        return tokens

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return (
            parameter
            for name, parameter in self.backbone.named_parameters()
            if name.endswith(("lora_A", "lora_B"))
        )

    def trainable_adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.backbone.state_dict().items()
            if name.endswith(("lora_A", "lora_B"))
        }
