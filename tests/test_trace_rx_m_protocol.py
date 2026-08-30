from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from techjam_aigc.trace_rx_m.protocol import run_nuisance_probes


def _frame(*, shortcut: bool) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(8)
    for index in range(80):
        target = index % 2
        if shortcut:
            width = 128 if target else 1024
            image_format = "PNG" if target else "JPEG"
            byte_count = 10_000 if target else 900_000
        else:
            width = 256 + int(rng.integers(0, 4)) * 16
            image_format = "PNG" if index % 3 else "JPEG"
            byte_count = 50_000 + int(rng.integers(0, 5000))
        rows.append({
            "target": target,
            "lineage_id": f"lineage-{index}",
            "width": width,
            "height": width,
            "bytes": byte_count,
            "format": image_format,
        })
    return pd.DataFrame(rows)


def test_grouped_nuisance_probes_detect_class_shortcuts() -> None:
    values = run_nuisance_probes(_frame(shortcut=True), folds=4, seed=3)
    assert set(values) == {"dimension", "metadata", "codec"}
    assert min(values.values()) > 0.95


def test_grouped_nuisance_probes_stay_near_chance_without_shortcuts() -> None:
    values = run_nuisance_probes(_frame(shortcut=False), folds=4, seed=3)
    assert max(values.values()) < 0.65


def test_nuisance_probe_requires_reproducibility_metadata() -> None:
    with pytest.raises(ValueError, match="require columns"):
        run_nuisance_probes(pd.DataFrame({"target": [0, 1], "lineage_id": ["a", "b"]}))
