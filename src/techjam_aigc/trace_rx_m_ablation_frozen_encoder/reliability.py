"""Measured-survival reliability tables for TRACE-RX-M.

No function in this module is trained through the detector loss.  Quality-cell
statistics are estimated after detection training, with empirical-Bayes style
pseudo-count shrinkage to the corresponding global population statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from .quality import Cell, QuantileQualityBinner


_EPS = 1e-12


def normalized_partial_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> float:
    """Integrate ROC TPR over ``[0, max_fpr]`` and divide by its width."""

    targets = np.asarray(labels).reshape(-1)
    values = _validated_values(scores, "scores")
    if targets.size != values.size or set(np.unique(targets)) != {0, 1}:
        raise ValueError("Normalized pAUC requires aligned scores and both binary classes.")
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must lie in (0, 1].")
    false_positive, true_positive, _ = roc_curve(targets, values)
    boundary_tpr = float(np.interp(max_fpr, false_positive, true_positive))
    keep = false_positive < max_fpr
    clipped_fpr = np.concatenate((false_positive[keep], [max_fpr]))
    clipped_tpr = np.concatenate((true_positive[keep], [boundary_tpr]))
    return float(np.trapezoid(clipped_tpr, clipped_fpr) / max_fpr)


@dataclass(frozen=True)
class CellStatistics:
    """A shrunk population estimate for one quality cell."""

    count: int
    mean: float
    variance: float

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))


@dataclass(frozen=True)
class AvailabilityAudit:
    cell_count: int
    spearman: float
    measured_d_prime: dict[Cell, float]
    predicted_d_prime: dict[Cell, float]
    passed: bool


@dataclass(frozen=True)
class OccupancyAudit:
    clean_count: int
    noise_count: int
    distribution_overlap: float
    cell_counts: dict[Cell, int]
    passed: bool


def _validated_values(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _cell_tuples(cells: np.ndarray | Iterable[Cell], expected: int) -> list[Cell]:
    array = np.asarray(list(cells) if not isinstance(cells, np.ndarray) else cells, dtype=np.int64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != expected:
        raise ValueError("cells must contain one cell index per value")
    return [tuple(int(item) for item in row) for row in array]


def _shrunk_statistics(
    local: np.ndarray,
    *,
    global_mean: float,
    global_variance: float,
    prior_strength: float,
    variance_floor: float,
) -> CellStatistics:
    count = int(local.size)
    if count == 0:
        return CellStatistics(
            count=0,
            mean=float(global_mean),
            variance=max(float(global_variance), variance_floor),
        )
    local_mean = float(np.mean(local))
    local_second = float(np.mean(local**2))
    total = count + prior_strength
    mean = (count * local_mean + prior_strength * global_mean) / total
    second = (
        count * local_second
        + prior_strength * (global_variance + global_mean**2)
    ) / total
    variance = max(second - mean**2, variance_floor)
    return CellStatistics(count=count, mean=float(mean), variance=float(variance))


def fit_shrunk_cell_statistics(
    values: np.ndarray,
    cells: np.ndarray | Iterable[Cell],
    all_cells: Iterable[Cell],
    *,
    prior_strength: float = 20.0,
    variance_floor: float = 1e-6,
) -> tuple[dict[Cell, CellStatistics], CellStatistics]:
    """Fit cell means/variances shrunk to the global population distribution.

    ``prior_strength`` is the effective number of global pseudo-observations.
    Shrinking the second moment (rather than variance alone) correctly includes
    between-mean uncertainty.  Empty cells become the global estimate.
    """

    observations = _validated_values(values, "values")
    cell_rows = _cell_tuples(cells, observations.size)
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be finite and positive")
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    global_stats = CellStatistics(
        count=int(observations.size),
        mean=float(np.mean(observations)),
        variance=max(float(np.var(observations)), variance_floor),
    )
    grouped: dict[Cell, list[float]] = {}
    for cell, value in zip(cell_rows, observations, strict=True):
        grouped.setdefault(cell, []).append(float(value))
    result = {}
    for cell in all_cells:
        key = tuple(int(item) for item in cell)
        local = np.asarray(grouped.get(key, ()), dtype=np.float64)
        result[key] = _shrunk_statistics(
            local,
            global_mean=global_stats.mean,
            global_variance=global_stats.variance,
            prior_strength=prior_strength,
            variance_floor=variance_floor,
        )
    return result, global_stats


def measured_d_prime(authentic: CellStatistics, aigc: CellStatistics) -> float:
    """Compute signed d-prime, positive when AIGC logits are larger."""

    pooled = np.sqrt((authentic.variance + aigc.variance) / 2.0)
    if pooled <= _EPS:
        return 0.0
    return float((aigc.mean - authentic.mean) / pooled)


@dataclass
class ReliabilityTable:
    """Fitted authentic normalization and measured availability lookup.

    Reversible assumption: equation (5.4) in the proposal uses ``z_tilde`` but
    never defines it.  Here it is the detector logit z-scored by the dedicated
    authentic-null mean and standard deviation in the endpoint's quality cell:
    ``(z - mu_real[cell]) / sigma_real[cell]``.  This is the minimal
    quality-cell authentic normalization implied by section 4's warning about
    "mis-binned normalisation".  It is kept in this standalone method so a
    clarified definition can replace it without changing availability fitting.

    Validation d-prime statistics are independently shrunk within each class.
    Availability is exactly ``clip(d_prime[cell] / d_prime_clean, 0, 1)``.
    """

    binner: QuantileQualityBinner
    authentic_null: Mapping[Cell, CellStatistics]
    validation_authentic: Mapping[Cell, CellStatistics]
    validation_aigc: Mapping[Cell, CellStatistics]
    d_prime_by_cell: Mapping[Cell, float]
    availability_by_cell: Mapping[Cell, float]
    clean_d_prime: float
    authentic_global: CellStatistics
    variance_floor: float = 1e-6

    @classmethod
    def fit(
        cls,
        *,
        authentic_null_logits: np.ndarray,
        authentic_null_quality: np.ndarray,
        validation_logits: np.ndarray,
        validation_labels: np.ndarray,
        validation_quality: np.ndarray,
        clean_mask: np.ndarray,
        n_bins: int | Iterable[int] = 4,
        prior_strength: float = 20.0,
        variance_floor: float = 1e-6,
    ) -> ReliabilityTable:
        """Fit S5 using disjoint train/authentic-null and val populations.

        ``clean_mask`` identifies clean validation endpoints and is used only
        for the denominator d-prime.  It must contain both classes; making it
        explicit avoids silently guessing which quality cell represents clean.
        """

        null_logits = _validated_values(authentic_null_logits, "authentic_null_logits")
        val_logits = _validated_values(validation_logits, "validation_logits")
        labels = np.asarray(validation_labels).reshape(-1)
        clean = np.asarray(clean_mask, dtype=bool).reshape(-1)
        if labels.size != val_logits.size or clean.size != val_logits.size:
            raise ValueError("validation logits, labels, and clean_mask must have equal length")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("validation_labels must be binary (0 authentic, 1 AIGC)")
        if not np.any(clean & (labels == 0)) or not np.any(clean & (labels == 1)):
            raise ValueError("clean_mask must select at least one sample from each class")

        binner = QuantileQualityBinner(n_bins).fit(authentic_null_quality)
        null_cells = binner.transform(authentic_null_quality)
        val_cells = binner.transform(validation_quality)
        if null_cells.shape[0] != null_logits.size or val_cells.shape[0] != val_logits.size:
            raise ValueError("each logit must have one quality descriptor")
        all_cells = list(binner.iter_cells())

        null_stats, null_global = fit_shrunk_cell_statistics(
            null_logits,
            null_cells,
            all_cells,
            prior_strength=prior_strength,
            variance_floor=variance_floor,
        )
        authentic_mask = labels == 0
        aigc_mask = labels == 1
        authentic_stats, _ = fit_shrunk_cell_statistics(
            val_logits[authentic_mask],
            val_cells[authentic_mask],
            all_cells,
            prior_strength=prior_strength,
            variance_floor=variance_floor,
        )
        aigc_stats, _ = fit_shrunk_cell_statistics(
            val_logits[aigc_mask],
            val_cells[aigc_mask],
            all_cells,
            prior_strength=prior_strength,
            variance_floor=variance_floor,
        )
        d_primes = {
            cell: measured_d_prime(authentic_stats[cell], aigc_stats[cell])
            for cell in all_cells
        }

        clean_real = val_logits[clean & authentic_mask]
        clean_aigc = val_logits[clean & aigc_mask]
        clean_real_stats = CellStatistics(
            count=int(clean_real.size),
            mean=float(np.mean(clean_real)),
            variance=max(float(np.var(clean_real)), variance_floor),
        )
        clean_aigc_stats = CellStatistics(
            count=int(clean_aigc.size),
            mean=float(np.mean(clean_aigc)),
            variance=max(float(np.var(clean_aigc)), variance_floor),
        )
        clean_d_prime = measured_d_prime(clean_real_stats, clean_aigc_stats)
        if clean_d_prime <= _EPS:
            raise ValueError("clean d-prime must be positive to define measured survival")
        availability = {
            cell: float(np.clip(value / clean_d_prime, 0.0, 1.0))
            for cell, value in d_primes.items()
        }
        return cls(
            binner=binner,
            authentic_null=null_stats,
            validation_authentic=authentic_stats,
            validation_aigc=aigc_stats,
            d_prime_by_cell=d_primes,
            availability_by_cell=availability,
            clean_d_prime=clean_d_prime,
            authentic_global=null_global,
            variance_floor=variance_floor,
        )

    def cells(self, quality: np.ndarray) -> list[Cell]:
        return self.binner.transform_cells(quality)

    def normalize_authentic(self, logits: np.ndarray, quality: np.ndarray) -> np.ndarray:
        """Return the reversible-assumption definition of ``z_tilde``."""

        values = np.asarray(logits, dtype=np.float64)
        shape = values.shape
        flat = values.reshape(-1)
        cell_rows = self.cells(quality)
        if len(cell_rows) != flat.size:
            raise ValueError("each logit must have one quality descriptor")
        normalized = np.empty_like(flat)
        for index, (value, cell) in enumerate(zip(flat, cell_rows, strict=True)):
            stats = self.authentic_null.get(cell, self.authentic_global)
            normalized[index] = (value - stats.mean) / max(stats.std, np.sqrt(self.variance_floor))
        return normalized.reshape(shape)

    def availability(self, quality: np.ndarray) -> np.ndarray:
        """Look up clipped measured-survival availability for each endpoint."""

        return np.asarray(
            [self.availability_by_cell.get(cell, 0.0) for cell in self.cells(quality)],
            dtype=np.float64,
        )

    def fuse(self, logits: np.ndarray, quality: np.ndarray) -> np.ndarray:
        """Compute ``z_fused = availability * z_tilde`` from equation (5.4)."""

        normalized = self.normalize_authentic(logits, quality).reshape(-1)
        availability = self.availability(quality)
        if normalized.size != availability.size:
            raise ValueError("each logit must have one quality descriptor")
        return (availability * normalized).reshape(np.asarray(logits).shape)


    def to_dict(self) -> dict[str, object]:
        def cells(values: Mapping[Cell, object]) -> dict[str, object]:
            result = {}
            for cell, value in values.items():
                key = ",".join(map(str, cell))
                result[key] = value.__dict__ if isinstance(value, CellStatistics) else float(value)
            return result
        return {
            "binner": self.binner.to_dict(),
            "authentic_null": cells(self.authentic_null),
            "validation_authentic": cells(self.validation_authentic),
            "validation_aigc": cells(self.validation_aigc),
            "d_prime_by_cell": cells(self.d_prime_by_cell),
            "availability_by_cell": cells(self.availability_by_cell),
            "clean_d_prime": self.clean_d_prime,
            "authentic_global": self.authentic_global.__dict__,
            "variance_floor": self.variance_floor,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ReliabilityTable:
        def key(text: str) -> Cell:
            return tuple(map(int, text.split(",")))
        def statistics(name: str) -> dict[Cell, CellStatistics]:
            return {
                key(cell): CellStatistics(**stats)
                for cell, stats in values[name].items()
            }
        return cls(
            binner=QuantileQualityBinner.from_dict(values["binner"]),
            authentic_null=statistics("authentic_null"),
            validation_authentic=statistics("validation_authentic"),
            validation_aigc=statistics("validation_aigc"),
            d_prime_by_cell={
                key(cell): float(value)
                for cell, value in values["d_prime_by_cell"].items()
            },
            availability_by_cell={
                key(cell): float(value)
                for cell, value in values["availability_by_cell"].items()
            },
            clean_d_prime=float(values["clean_d_prime"]),
            authentic_global=CellStatistics(**values["authentic_global"]),
            variance_floor=float(values["variance_floor"]),
        )


def audit_heldout_availability(
    table: ReliabilityTable,
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    min_samples_per_class: int = 3,
    min_spearman: float = 0.5,
) -> AvailabilityAudit:
    """Test predicted cell survival against held-out-family d-prime."""

    scores = _validated_values(logits, "logits")
    targets = np.asarray(labels).reshape(-1)
    cells = table.cells(quality)
    if targets.size != scores.size or len(cells) != scores.size:
        raise ValueError("Held-out logits, labels, and quality must align.")
    measured: dict[Cell, float] = {}
    predicted: dict[Cell, float] = {}
    for cell in sorted(set(cells)):
        mask = np.asarray([value == cell for value in cells])
        real = scores[mask & (targets == 0)]
        aigc = scores[mask & (targets == 1)]
        if min(real.size, aigc.size) < min_samples_per_class:
            continue
        real_stats = CellStatistics(real.size, float(real.mean()), max(float(real.var()), table.variance_floor))
        aigc_stats = CellStatistics(aigc.size, float(aigc.mean()), max(float(aigc.var()), table.variance_floor))
        measured[cell] = measured_d_prime(real_stats, aigc_stats)
        predicted[cell] = table.availability_by_cell.get(cell, 0.0) * table.clean_d_prime
    if len(measured) < 2:
        correlation = float("nan")
        passed = False
    else:
        correlation = float(spearmanr(
            [predicted[cell] for cell in measured],
            [measured[cell] for cell in measured],
        ).statistic)
        passed = bool(np.isfinite(correlation) and correlation >= min_spearman)
    return AvailabilityAudit(len(measured), correlation, measured, predicted, passed)


def audit_quality_cell_occupancy(
    table: ReliabilityTable,
    *,
    quality: np.ndarray,
    conditions: Iterable[str],
    transform_families: Iterable[str],
    max_clean_noise_overlap: float = 0.8,
) -> OccupancyAudit:
    """Confirm noise endpoints do not collapse into clean quality cells."""

    cells = table.cells(quality)
    condition_values = np.asarray(list(map(str, conditions)))
    family_values = np.asarray(list(map(str, transform_families)))
    if len(cells) != len(condition_values) or len(cells) != len(family_values):
        raise ValueError("Occupancy inputs must contain one row per endpoint.")
    clean = condition_values == "clean"
    noise = family_values == "gaussian_noise"
    cell_set = sorted(set(cells))
    counts = {cell: cells.count(cell) for cell in cell_set}
    if not clean.any() or not noise.any():
        return OccupancyAudit(int(clean.sum()), int(noise.sum()), 1.0, counts, False)
    clean_distribution = np.asarray([
        sum(cell == value for cell, selected in zip(cells, clean, strict=True) if selected)
        for value in cell_set
    ], dtype=np.float64) / clean.sum()
    noise_distribution = np.asarray([
        sum(cell == value for cell, selected in zip(cells, noise, strict=True) if selected)
        for value in cell_set
    ], dtype=np.float64) / noise.sum()
    overlap = float(np.minimum(clean_distribution, noise_distribution).sum())
    return OccupancyAudit(
        int(clean.sum()), int(noise.sum()), overlap, counts,
        overlap <= max_clean_noise_overlap,
    )


@dataclass
class PassiveQualityStacker:
    """L2 logistic fallback over the detector logit and passive quality."""

    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, logits: np.ndarray, quality: np.ndarray, labels: np.ndarray) -> PassiveQualityStacker:
        scores = _validated_values(logits, "logits")
        descriptors = QuantileQualityBinner._validate(quality)
        targets = np.asarray(labels).reshape(-1)
        if len(scores) != len(descriptors) or len(scores) != len(targets):
            raise ValueError("Passive stacker inputs must align.")
        features = np.column_stack((scores, descriptors))
        scaler = StandardScaler().fit(features)
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(
            scaler.transform(features), targets
        )
        self.feature_mean = scaler.mean_
        self.feature_scale = scaler.scale_
        self.coefficients = model.coef_[0]
        self.intercept = float(model.intercept_[0])
        return self

    def fused_logits(self, logits: np.ndarray, quality: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None or self.coefficients is None:
            raise RuntimeError("Passive quality stacker has not been fitted.")
        scores = np.asarray(logits, dtype=np.float64)
        shape = scores.shape
        features = np.column_stack((scores.reshape(-1), QuantileQualityBinner._validate(quality)))
        return (((features - self.feature_mean) / self.feature_scale) @ self.coefficients + self.intercept).reshape(shape)

    def predict_proba(self, logits: np.ndarray, quality: np.ndarray) -> np.ndarray:
        return expit(self.fused_logits(logits, quality))

    def to_dict(self) -> dict[str, object]:
        if self.feature_mean is None or self.feature_scale is None or self.coefficients is None:
            raise RuntimeError("Passive quality stacker has not been fitted.")
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> PassiveQualityStacker:
        return cls(
            feature_mean=np.asarray(values["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(values["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(values["coefficients"], dtype=np.float64),
            intercept=float(values["intercept"]),
        )
