"""Gates for run logging and the shortcut audit."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from acai.audit import (NUISANCES, SHORTCUT_FEATURES, audit, nuisance_vector,
                        shortcut_vector)
from acai.runlog import Run


def preds(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.2, n), 0, 1)
    return y, s, pd.DataFrame({"id": np.arange(n), "group": np.arange(n) // 4,
                               "label": y, "score": s, "split": "val",
                               "source": "x", "transform": "clean", "severity": 0})


# ------------------------------------------------------------------------ runlog

def test_run_writes_config_env_and_status(tmp_path):
    with Run("t", {"fusion": "h2", "seed": 1}, root=tmp_path) as r:
        r.log(step=1, loss=0.7)
    d = r.dir
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["fusion"] == "h2" and "git" in cfg and "dirty" in cfg["git"]
    assert "python" in json.loads((d / "env.json").read_text())
    assert json.loads((d / "status.json").read_text())["status"] == "ok"
    assert json.loads((d / "train_log.jsonl").read_text().splitlines()[0])["loss"] == 0.7


def test_failed_run_is_still_recorded(tmp_path):
    """A crash that leaves no trace is the run you repeat by accident."""
    with pytest.raises(ValueError):
        with Run("boom", {}, root=tmp_path) as r:
            raise ValueError("x")
    assert json.loads((r.dir / "status.json").read_text())["status"] == "failed:ValueError"


def test_predictions_require_group_column(tmp_path):
    """Missing `group` would silently downgrade every grouped bootstrap to row-level."""
    _, _, df = preds()
    with Run("t", {}, root=tmp_path) as r:
        with pytest.raises(ValueError, match="group"):
            r.predictions(df.drop(columns=["group"]))
        r.predictions(df)
    assert (r.dir / "predictions.parquet").exists()


def test_predictions_roundtrip(tmp_path):
    """Every report table is recomputed from this file, so it must survive intact."""
    _, _, df = preds()
    with Run("t", {}, root=tmp_path) as r:
        p = r.predictions(df)
    pd.testing.assert_frame_equal(pd.read_parquet(p), df)


def test_index_csv_accumulates_one_row_per_run(tmp_path):
    for i in range(3):
        with Run(f"r{i}", {"fusion": "none"}, root=tmp_path) as r:
            r.metrics({"overall": {"auroc": 0.5 + i / 10}})
    rows = pd.read_csv(tmp_path / "index.csv")
    assert len(rows) == 3 and list(rows["auroc"]) == [0.5, 0.6, 0.7]


def test_metrics_json_accepts_numpy_and_dataclasses(tmp_path):
    from acai.metrics import compute
    y, s, _ = preds()
    with Run("t", {}, root=tmp_path) as r:
        r.metrics({"overall": compute(y, s), "arr": np.arange(3), "f": np.float64(1.5)})
    d = json.loads((r.dir / "metrics.json").read_text())
    assert d["arr"] == [0, 1, 2] and d["f"] == 1.5 and "auroc" in d["overall"]


# ------------------------------------------------------------------------- audit

def test_probe_vectors_have_expected_keys():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    assert set(nuisance_vector(img)) == set(NUISANCES)
    assert set(shortcut_vector(img)) == set(SHORTCUT_FEATURES)


def test_recompressed_bytes_is_size_normalised():
    """Un-normalised, this probe would mostly just re-measure resolution."""
    rng = np.random.default_rng(0)
    small = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    big = np.repeat(np.repeat(small, 4, 0), 4, 1)
    a, b = nuisance_vector(small)["recompressed_bytes"], nuisance_vector(big)["recompressed_bytes"]
    assert b < a * 1.5


def test_audit_flags_a_score_that_tracks_a_nuisance():
    y, s, _ = preds()
    r = audit(y, s, {"pixels": s * 1000, "mean_luminance": np.random.default_rng(1).random(len(y))})
    assert [f["nuisance"] for f in r["flags"]] == ["pixels"]


def test_audit_is_quiet_when_no_shortcut_present():
    y, s, _ = preds()
    rng = np.random.default_rng(2)
    r = audit(y, s, {n: rng.random(len(y)) for n in ("pixels", "mean_luminance")})
    assert r["flags"] == []


def test_audit_reports_the_cheat_baseline():
    """The dataset-card lesson: a 0.75 AUROC means little if luminance alone gets 0.73."""
    y, s, _ = preds()
    r = audit(y, s, {"mean_luminance": y + np.random.default_rng(3).normal(0, .1, len(y))})
    assert r["cheat_baseline"] > 0.95        # this nuisance alone nearly solves it


def test_audit_flags_native_resolution_collapse():
    """Reproduces the prior study's fft_mid_energy diagnosis: clean 0.650, native 0.495."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    native = rng.random(n) < 0.5
    s = np.where(native, rng.random(n), np.clip(0.5 + 0.4 * y + rng.normal(0, .1, n), 0, 1))
    r = audit(y, s, {"pixels": rng.random(n)}, native_mask=native)
    assert any(f["nuisance"] == "resolution" for f in r["flags"])
    assert r["native_resolution_auroc"] < r["overall_auroc"] - 0.10


def test_shortcut_features_are_not_importable_from_the_feature_module():
    """Structural guarantee that audit-only probes cannot become model inputs."""
    from acai.features import lowlevel
    assert not [f for f in SHORTCUT_FEATURES if hasattr(lowlevel, f)]
