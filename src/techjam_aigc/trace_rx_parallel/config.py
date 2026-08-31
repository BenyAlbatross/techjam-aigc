"""Validated configuration for the TRACE-RX-Parallel detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from techjam_aigc.trace_rx_m.config import (
    BackboneConfig,
    DataConfig,
    HubConfig,
    LossConfig,
    MemoryConfig,
    OptimizerConfig,
    ProtocolConfig,
    ReliabilityConfig,
    TrackingConfig,
)


@dataclass(frozen=True)
class ParallelHeadConfig:
    """Widths and pooling choices for both branches and late fusion."""

    evidence_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1
    global_tail_quantile: float = 0.95
    residual_tail_quantile: float = 0.95

    def validate(self) -> None:
        if self.evidence_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("head evidence_dim and hidden_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("head.dropout must lie in [0, 1).")
        if not 0 < self.global_tail_quantile < 1:
            raise ValueError("head.global_tail_quantile must lie in (0, 1).")
        if not 0 < self.residual_tail_quantile < 1:
            raise ValueError("head.residual_tail_quantile must lie in (0, 1).")


@dataclass(frozen=True)
class TraceRXParallelConfig:
    """Top-level configuration with a parallel-head-specific section."""

    schema_version: int = 1
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    head: ParallelHeadConfig = field(default_factory=ParallelHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tracking: TrackingConfig = field(
        default_factory=lambda: TrackingConfig(wandb_project="trace-rx-parallel")
    )
    hub: HubConfig = field(
        default_factory=lambda: HubConfig(path_prefix="trace-rx-parallel")
    )
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
            raise ValueError(
                f"Unsupported TRACE-RX-Parallel config schema {self.schema_version}."
            )
        self.backbone.validate(require_access=require_backbone_access)
        self.memory.validate(require_selection=require_memory)
        self.head.validate()
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
            raise ValueError(
                "reliability.quality_bins must contain four positive integers."
            )
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
        if not 0 < self.loss.pauc_alpha <= 1:
            raise ValueError("loss.pauc_alpha must lie in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TraceRXParallelConfig":
        values = dict(values)
        nested: dict[str, Any] = {}
        for name, config_type in (
            ("backbone", BackboneConfig),
            ("memory", MemoryConfig),
            ("head", ParallelHeadConfig),
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
    def load(cls, path: Path) -> "TraceRXParallelConfig":
        return cls.from_dict(json.loads(path.read_text()))
