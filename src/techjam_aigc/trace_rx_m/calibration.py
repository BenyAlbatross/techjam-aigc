"""Final-stage positive temperature calibration for TRACE-RX-M."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def binary_logit_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean binary negative log likelihood after positive temperature scaling."""

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.size == 0 or scores.size != targets.size:
        raise ValueError("logits and labels must be non-empty and have equal length")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("logits and labels must be finite")
    if not np.all(np.isin(targets, [0.0, 1.0])):
        raise ValueError("labels must be binary")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled = scores / temperature
    return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))


@dataclass
class PositiveTemperatureScaler:
    """Fit one strictly positive temperature on the independent S6 split.

    Optimization is performed over ``log(T)`` in bounded scalar space, so no
    iteration can evaluate a zero or negative temperature.  The detector and
    reliability table must already be frozen before calling :meth:`fit`.
    """

    temperature: float = 1.0
    bounds: tuple[float, float] = (1e-3, 1e3)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> PositiveTemperatureScaler:
        lower, upper = self.bounds
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("bounds must be finite, positive, and increasing")
        # Validate eagerly so optimizer failures are never mistaken for bad data.
        binary_logit_nll(logits, labels, 1.0)
        result = minimize_scalar(
            lambda log_temperature: binary_logit_nll(
                logits, labels, float(np.exp(log_temperature))
            ),
            bounds=(float(np.log(lower)), float(np.log(upper))),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError(f"temperature optimization failed: {result.message}")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Divide logits by the fitted positive temperature."""

        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise RuntimeError("temperature is not finite and positive")
        values = np.asarray(logits, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("logits must be finite")
        return values / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated AIGC probabilities in ``[0, 1]``."""

        return expit(self.transform(logits))



    def to_dict(self) -> dict[str, object]:
        return {"temperature": self.temperature, "bounds": list(self.bounds)}

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PositiveTemperatureScaler:
        return cls(
            temperature=float(values["temperature"]),
            bounds=tuple(map(float, values.get("bounds", (1e-3, 1e3)))),
        )
