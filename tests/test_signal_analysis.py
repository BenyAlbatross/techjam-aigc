from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "run_signal_analysis",
    Path(__file__).parents[1] / "scripts/run_signal_analysis.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
dct_analysis = _MODULE.dct_analysis
feature_effects = _MODULE.feature_effects
spectrum_analysis = _MODULE.spectrum_analysis


def test_spectrum_harmonic_excess_responds_to_one_eighth_periodicity() -> None:
    x = np.arange(256)
    periodic = np.tile(np.sin(2 * np.pi * x / 8), (256, 1))
    smooth = np.tile(np.linspace(0, 1, 256), (256, 1))

    periodic_features, periodic_profile = spectrum_analysis(periodic)
    smooth_features, _ = spectrum_analysis(smooth)

    assert len(periodic_profile) == 32
    assert (
        periodic_features["fft_harmonic_excess_1_8"]
        > smooth_features["fft_harmonic_excess_1_8"]
    )


def test_dct_profile_partitions_all_ac_energy() -> None:
    y, x = np.mgrid[:256, :256]
    image = ((3 * x + 5 * y) % 256) / 255.0
    features, profile = dct_analysis(image)

    assert np.isfinite(list(features.values())).all()
    assert len(profile) == 14
    assert np.isclose(sum(row["dct_energy_ratio"] for row in profile), 1.0)


def test_feature_effects_preserves_aigc_high_direction() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["paired"] * 8,
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "fft_high_energy": [0.1, 0.2, 0.15, 0.18, 0.7, 0.8, 0.75, 0.9],
        }
    )
    registry = pd.DataFrame(
        {"name": ["fft_high_energy"], "family": ["fft_magnitude"]}
    )
    result = feature_effects(frame, registry)
    row = result[result["scope"] == "within:paired"].iloc[0]

    assert row["auc_aigc_high"] == 1.0
    assert row["rank_biserial"] == 1.0
