"""Configuration contract for the isolated three-branch detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from techjam_aigc.trace_rx_m.config import BackboneConfig


@dataclass(frozen=True)
class PreprocessingConfig:
    global_image_size: int = 224
    max_global_short_side: int = 512
    native_crop_size: int = 192
    native_crop_count: int = 5
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.global_image_size < 1 or self.max_global_short_side < self.global_image_size:
            raise ValueError("Global image dimensions are inconsistent.")
        if self.native_crop_size < 32:
            raise ValueError("native_crop_size must be at least 32 pixels.")
        if self.native_crop_count not in {1, 5}:
            raise ValueError("native_crop_count must be 1 (center) or 5 (corners and center).")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("Image normalization requires three-channel mean/std values.")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("Image standard deviations must be positive.")


@dataclass(frozen=True)
class MemoryConfig:
    prototypes_per_class: int = 256
    topk: int = 8
    retrieval_temperature: float = 0.07
    tokens_per_image: int = 16
    kmeans_batch_size: int = 8192
    group_balanced_fit: bool = True

    def validate(self) -> None:
        if self.prototypes_per_class < 2:
            raise ValueError("Each class memory needs at least two prototypes.")
        if not 0 < self.topk <= self.prototypes_per_class:
            raise ValueError("memory.topk must lie in [1, prototypes_per_class].")
        if self.retrieval_temperature <= 0:
            raise ValueError("memory.retrieval_temperature must be positive.")
        if self.tokens_per_image < 1:
            raise ValueError("memory.tokens_per_image must be positive.")
        if self.kmeans_batch_size < self.prototypes_per_class:
            raise ValueError("kmeans_batch_size must cover every prototype at initialization.")


@dataclass(frozen=True)
class HeadConfig:
    hidden_dim: int = 256
    patch_hidden_dim: int = 256
    forensic_embedding_dim: int = 128
    dropout: float = 0.10
    top_fraction: float = 0.20

    def validate(self) -> None:
        if min(self.hidden_dim, self.patch_hidden_dim, self.forensic_embedding_dim) < 1:
            raise ValueError("Head dimensions must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("head.dropout must lie in [0, 1).")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("head.top_fraction must lie in (0, 1].")


@dataclass(frozen=True)
class LossConfig:
    pauc_weight: float = 0.25
    pauc_alpha: float = 0.05
    ranking_margin: float = 0.20
    auxiliary_branch_weight: float = 0.10

    def validate(self) -> None:
        if self.pauc_weight < 0 or self.auxiliary_branch_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if not 0 < self.pauc_alpha <= 1:
            raise ValueError("loss.pauc_alpha must lie in (0, 1].")


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 10
    batch_size: int = 32
    workers: int = 8
    adapter_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"

    def validate(self) -> None:
        if self.epochs != 10:
            raise ValueError("The three-branch training contract requires exactly 10 epochs.")
        if self.batch_size < 2 or self.workers < 0:
            raise ValueError("Batch size must be at least two and workers cannot be negative.")
        if min(self.adapter_lr, self.head_lr, self.gradient_clip_norm) <= 0:
            raise ValueError("Learning rates and gradient clipping must be positive.")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0, 1).")
        if self.mixed_precision not in {"bf16", "fp32"}:
            raise ValueError("mixed_precision must be bf16 or fp32.")


@dataclass(frozen=True)
class DataConfig:
    manifest: str = "data/techjam2026_v2-normalized/training-manifest.csv"
    labels: str = "data/techjam2026_v2/labels.csv"
    source_root: str = "data/techjam2026_v2"
    split: str = "train"
    use_all_training_pools: bool = True
    group_balanced_loss: bool = True
    seed: int = 20260831

    def validate(self) -> None:
        if self.split != "train":
            raise ValueError("The training config may consume only the train split.")
        if not self.use_all_training_pools:
            raise ValueError("The requested run requires all train-only pools.")
        if self.seed < 0:
            raise ValueError("data.seed cannot be negative.")


@dataclass(frozen=True)
class ThreeBranchConfig:
    schema_version: int = 1
    model_name: str = "three-branch"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self, *, require_backbone_access: bool = False) -> None:
        if self.schema_version != 1 or self.model_name != "three-branch":
            raise ValueError("Expected schema_version=1 and model_name='three-branch'.")
        self.backbone.validate(require_access=require_backbone_access)
        self.preprocessing.validate()
        self.memory.validate()
        self.head.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.data.validate()
        if self.backbone.image_size != self.preprocessing.global_image_size:
            raise ValueError("Backbone and global preprocessing image sizes must match.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ThreeBranchConfig":
        raw = dict(values)
        for key, value_type in (
            ("backbone", BackboneConfig),
            ("preprocessing", PreprocessingConfig),
            ("memory", MemoryConfig),
            ("head", HeadConfig),
            ("loss", LossConfig),
            ("optimizer", OptimizerConfig),
            ("data", DataConfig),
        ):
            if key in raw:
                raw[key] = value_type(**raw[key])
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> "ThreeBranchConfig":
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Three-branch config must be a JSON object.")
        return cls.from_dict(values)

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
