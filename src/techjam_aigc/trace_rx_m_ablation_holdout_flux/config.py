"""Validated configuration for the TRACE-RX-M v2 implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


HUGGINGFACE_ORGANIZATION = "techjam-aigc"
DEFAULT_HUGGINGFACE_REPO_ID = f"{HUGGINGFACE_ORGANIZATION}/trace-rx-m"


@dataclass(frozen=True)
class BackboneConfig:
    model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    revision: str | None = None
    license_accepted: bool = False
    image_size: int = 224
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")
    gradient_checkpointing: bool = True

    def validate(self, *, require_access: bool = True) -> None:
        if require_access and not self.license_accepted:
            raise ValueError(
                "The backbone licence must be reviewed and license_accepted=true "
                "before feature extraction or training."
            )
        if require_access and not self.revision:
            raise ValueError("Pin backbone.revision to an immutable commit for reproducibility.")
        if self.image_size <= 0:
            raise ValueError("backbone.image_size must be positive.")
        if self.lora_rank < 0:
            raise ValueError("lora_rank cannot be negative.")
        if self.lora_rank and not self.lora_targets:
            raise ValueError("At least one LoRA target is required when lora_rank > 0.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Frozen pixel contract shared by v2 preparation and inference."""

    version: str = "center-crop-v1"
    image_size: int = 224
    max_short_side: int = 512
    resize_interpolation: str = "bicubic"
    crop_mode: str = "center"
    undersized_policy: str = "symmetric_zero_pad"
    convert_rgb: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.version != "center-crop-v1":
            raise ValueError("preprocessing.version must be center-crop-v1 for v2.")
        if self.image_size < 1:
            raise ValueError("preprocessing.image_size must be positive.")
        if self.max_short_side < self.image_size:
            raise ValueError(
                "preprocessing.max_short_side must be at least preprocessing.image_size."
            )
        if self.resize_interpolation != "bicubic":
            raise ValueError("v2 preprocessing requires bicubic resizing.")
        if self.crop_mode != "center":
            raise ValueError("v2 preprocessing requires a center crop.")
        if self.undersized_policy != "symmetric_zero_pad":
            raise ValueError("v2 preprocessing requires symmetric zero-padding.")
        if not self.convert_rgb:
            raise ValueError("v2 preprocessing requires RGB conversion.")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("preprocessing image_mean/image_std must have three channels.")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("preprocessing.image_std values must be positive.")


@dataclass(frozen=True)
class MemoryConfig:
    candidate_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    candidate_topk: tuple[int, ...] = (4, 8, 16, 32)
    selected_size: int | None = None
    selected_topk: int | None = None
    score_chunk_size: int = 512
    diversity_weight: float = 0.01
    tail_quantile: float = 0.95
    tail_error_threshold: float | None = None
    capacity_relative_tolerance: float = 0.01
    max_prototype_usage_share: float = 0.25

    def validate(self, *, require_selection: bool = False) -> None:
        if not self.candidate_sizes or any(value <= 0 for value in self.candidate_sizes):
            raise ValueError("memory.candidate_sizes must contain positive values.")
        if not self.candidate_topk or any(value <= 0 for value in self.candidate_topk):
            raise ValueError("memory.candidate_topk must contain positive values.")
        if require_selection and (self.selected_size is None or self.selected_topk is None):
            raise ValueError("S3 must record selected_size and selected_topk before S4.")
        if self.selected_size is not None and self.selected_topk is not None:
            if self.selected_topk > self.selected_size:
                raise ValueError("selected_topk cannot exceed selected_size.")
        if not 0 < self.max_prototype_usage_share <= 1:
            raise ValueError("max_prototype_usage_share must lie in (0, 1].")


@dataclass(frozen=True)
class HeadConfig:
    evidence_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1
    residual_tail_quantile: float = 0.95


@dataclass(frozen=True)
class LossConfig:
    pauc_weight: float = 0.25
    pair_weight: float = 0.25
    pauc_alpha: float = 0.05
    ranking_margin: float = 0.2
    dda_in_primary_objective: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    adapter_lr: float = 2e-5
    head_lr: float = 2e-4
    memory_lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    memory_epochs: int = 20
    detection_epochs: int = 10
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"


@dataclass(frozen=True)
class TrackingConfig:
    """Remote experiment tracking without embedding credentials in config."""

    wandb_project: str = "trace-rx-m-v2"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_name: str | None = None

    def validate(self) -> None:
        if not self.wandb_project.strip():
            raise ValueError("tracking.wandb_project cannot be empty.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "tracking.wandb_mode must be online, offline, or disabled."
            )


@dataclass(frozen=True)
class HubConfig:
    """Hugging Face model-repository publication settings."""

    repo_id: str | None = DEFAULT_HUGGINGFACE_REPO_ID
    private: bool = True
    revision: str = "main"
    path_prefix: str = "trace-rx-m-v2"
    checkpoint_every_epochs: int = 1

    def validate(self, *, require_repo: bool = False) -> None:
        if require_repo and not self.repo_id:
            raise ValueError(
                "hub.repo_id is required for S4 so periodic, best, and final "
                "weights cannot remain local-only."
            )
        if self.repo_id is not None and self.repo_id.count("/") != 1:
            raise ValueError("hub.repo_id must use the 'owner/name' form.")
        if self.repo_id is not None:
            owner, _ = self.repo_id.split("/", maxsplit=1)
            if owner != HUGGINGFACE_ORGANIZATION:
                raise ValueError(
                    "hub.repo_id must belong to the "
                    f"'{HUGGINGFACE_ORGANIZATION}' Hugging Face organization."
                )
        if not self.revision.strip():
            raise ValueError("hub.revision cannot be empty.")
        if not self.path_prefix.strip("/"):
            raise ValueError("hub.path_prefix cannot be empty.")
        if self.checkpoint_every_epochs < 1:
            raise ValueError("hub.checkpoint_every_epochs must be positive.")


@dataclass(frozen=True)
class ProtocolConfig:
    nuisance_max_auc: float = 0.55
    nuisance_folds: int = 5


@dataclass(frozen=True)
class ReliabilityConfig:
    quality_bins: tuple[int, int, int, int] = (4, 4, 4, 4)
    prior_strength: float = 20.0
    variance_floor: float = 1e-6
    hf_cutoff: float = 0.25
    heldout_min_spearman: float = 0.50
    heldout_min_samples_per_class: int = 3
    max_clean_noise_cell_overlap: float = 0.80
    availability_min_normalized_pauc_gain: float = 0.01


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    workers: int = 4
    seed: int = 20260831
    dda_positive_share: float = 0.20
    clean_probability: float = 0.20
    held_out_transform_family: str = "gaussian_noise"
    held_out_generator_family: str | None = None
    held_out_min_roc_auc: float = 0.60

    def validate(self) -> None:
        if self.batch_size < 10 or self.batch_size % 2:
            raise ValueError("data.batch_size must be even and at least ten.")
        if not 0.10 <= self.dda_positive_share <= 0.20:
            raise ValueError("The proposal constrains DDA to 10--20% of positives.")
        if not 0 <= self.clean_probability < 1:
            raise ValueError("clean_probability must lie in [0, 1).")
        if not 0.5 <= self.held_out_min_roc_auc <= 1:
            raise ValueError("held_out_min_roc_auc must lie in [0.5, 1].")


@dataclass(frozen=True)
class TraceRXMConfig:
    schema_version: int = 1
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(
        self,
        *,
        require_backbone_access: bool = False,
        require_memory: bool = False,
        require_remote_artifacts: bool = False,
    ) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported TRACE-RX-M config schema {self.schema_version}.")
        self.backbone.validate(require_access=require_backbone_access)
        self.preprocessing.validate()
        if self.preprocessing.image_size != self.backbone.image_size:
            raise ValueError(
                "preprocessing.image_size must equal backbone.image_size so training and "
                "inference use the same spatial input."
            )
        self.memory.validate(require_selection=require_memory)
        self.tracking.validate()
        self.hub.validate(require_repo=require_remote_artifacts)
        self.data.validate()
        if not 0.5 <= self.protocol.nuisance_max_auc <= 1:
            raise ValueError("protocol.nuisance_max_auc must lie in [0.5, 1].")
        if self.protocol.nuisance_folds < 2:
            raise ValueError("protocol.nuisance_folds must be at least two.")
        if len(self.reliability.quality_bins) != 4 or any(
            value < 1 for value in self.reliability.quality_bins
        ):
            raise ValueError("reliability.quality_bins must contain four positive integers.")
        if self.reliability.prior_strength <= 0 or self.reliability.variance_floor <= 0:
            raise ValueError("Reliability shrinkage and variance floor must be positive.")
        if not -1 <= self.reliability.heldout_min_spearman <= 1:
            raise ValueError("reliability.heldout_min_spearman must lie in [-1, 1].")
        if self.reliability.heldout_min_samples_per_class < 2:
            raise ValueError("Reliability held-out cells need at least two samples per class.")
        if not 0 <= self.reliability.max_clean_noise_cell_overlap <= 1:
            raise ValueError("Clean/noise cell overlap must lie in [0, 1].")
        if self.reliability.availability_min_normalized_pauc_gain < 0:
            raise ValueError("Availability pAUC gain threshold cannot be negative.")
        if not 0 < self.head.residual_tail_quantile < 1:
            raise ValueError("head.residual_tail_quantile must lie in (0, 1).")
        if not 0 < self.loss.pauc_alpha <= 1:
            raise ValueError("loss.pauc_alpha must lie in (0, 1].")
        if self.optimizer.mixed_precision not in {"bf16", "fp32"}:
            raise ValueError("optimizer.mixed_precision must be bf16 or fp32.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TraceRXMConfig":
        values = dict(values)
        nested: dict[str, Any] = {}
        for name, config_type in (
            ("backbone", BackboneConfig),
            ("preprocessing", PreprocessingConfig),
            ("memory", MemoryConfig),
            ("head", HeadConfig),
            ("loss", LossConfig),
            ("optimizer", OptimizerConfig),
            ("tracking", TrackingConfig),
            ("hub", HubConfig),
            ("protocol", ProtocolConfig),
            ("reliability", ReliabilityConfig),
            ("data", DataConfig),
        ):
            section = dict(values.pop(name, {}))
            for key, value in tuple(section.items()):
                if isinstance(value, list):
                    section[key] = tuple(value)
            nested[name] = config_type(**section)
        config = cls(**values, **nested)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> "TraceRXMConfig":
        return cls.from_dict(json.loads(path.read_text()))
