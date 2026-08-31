"""Correctness gates for the metrics layer.

DeLong and the grouped bootstrap are the two pieces that decide whether a hybrid "beats"
pure DINOv3, so both are validated against independent references rather than trusted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from acai.metrics import (bootstrap_ci, by_group, compute, delong_test, worst_family)


def data(n=2000, sep=0.25, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, np.clip(0.5 + sep * y + rng.normal(0, 0.2, n), 0, 1)


def test_auroc_matches_sklearn():
    y, s = data()
    assert np.isclose(compute(y, s).auroc, roc_auc_score(y, s))


def test_auprc_lift_is_one_for_random_scores():
    """Guards the base-rate correction: a random scorer must show no lift at any prevalence."""
    rng = np.random.default_rng(0)
    for prev in (0.1, 0.5, 0.9):
        y = (rng.random(20000) < prev).astype(int)
        m = compute(y, rng.random(20000))
        assert abs(m.auprc_lift - 1.0) < 0.12, f"prevalence {prev}: lift {m.auprc_lift}"


def test_auprc_moves_with_prevalence_but_auroc_does_not():
    """The exact reason AUPRC must be reported with its base rate (plan §5)."""
    rng = np.random.default_rng(0)
    aurocs, auprcs = [], []
    for prev in (0.1, 0.5):
        n = 20000
        y = (rng.random(n) < prev).astype(int)
        s = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.2, n), 0, 1)
        m = compute(y, s)
        aurocs.append(m.auroc); auprcs.append(m.auprc)
    assert abs(aurocs[0] - aurocs[1]) < 0.03      # AUROC ~invariant to prevalence
    assert auprcs[1] - auprcs[0] > 0.20           # AUPRC is not


def test_perfect_and_random_separation():
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    assert compute(y, y.astype(float)).auroc == 1.0
    assert compute(y, y.astype(float)).eer == 0.0
    rng = np.random.default_rng(0)
    assert abs(compute(y, rng.random(1000)).auroc - 0.5) < 0.05


def test_tpr_at_fpr_clamps_when_unreachable():
    """A curve that never reaches FPR=1% must report 0.0, not a looser threshold's TPR."""
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    rng = np.random.default_rng(0)
    m = compute(y, rng.random(100))
    assert 0.0 <= m.tpr_at_fpr01 <= m.tpr_at_fpr05


def test_single_class_raises():
    with pytest.raises(ValueError):
        compute(np.ones(10, int), np.random.random(10))


def test_weights_simulate_a_different_eval_mixture():
    """§4.4: re-weighting fixed predictions must actually change the reported prevalence."""
    y, s = data()
    w = np.where(y == 1, 0.2, 1.0)
    assert compute(y, s, weights=w).prevalence < compute(y, s).prevalence - 0.2


# ------------------------------------------------------------------------- DeLong

def test_delong_aurocs_match_sklearn():
    y, a = data(seed=1)
    _, b = data(sep=0.15, seed=2)
    d = delong_test(y, a, b)
    assert np.isclose(d["auroc_a"], roc_auc_score(y, a), atol=1e-6)
    assert np.isclose(d["auroc_b"], roc_auc_score(y, b), atol=1e-6)


def test_delong_identical_models_give_p_one():
    y, s = data()
    d = delong_test(y, s, s)
    assert d["diff"] == 0.0 and d["p"] == 1.0


def test_delong_detects_a_real_difference():
    y, a = data(sep=0.30, seed=3)
    b = np.clip(a + np.random.default_rng(4).normal(0, 0.35, len(a)), 0, 1)
    d = delong_test(y, a, b)
    assert d["diff"] > 0 and d["p"] < 1e-4


def test_delong_se_agrees_with_bootstrap_se():
    """Independent check on the variance estimate, which is the part worth doubting."""
    y, a = data(sep=0.30, seed=5)
    b = np.clip(a + np.random.default_rng(6).normal(0, 0.30, len(a)), 0, 1)
    d = delong_test(y, a, b)

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(400):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    assert 0.5 < d["se"] / np.std(diffs) < 2.0, f"delong se {d['se']} vs boot {np.std(diffs)}"


def test_delong_is_paired_not_independent():
    """Correlated scores must yield a *smaller* SE than the unpaired approximation.

    This is the whole point of pairing: if the SE matched the independent-samples formula,
    we would be discarding the shared-image structure and losing power.
    """
    y, a = data(sep=0.30, seed=7)
    b = np.clip(a + np.random.default_rng(8).normal(0, 0.05, len(a)), 0, 1)   # near-identical
    _, c = data(sep=0.30, seed=9)                                            # independent
    assert delong_test(y, a, b)["se"] < delong_test(y, a, c)["se"]


# --------------------------------------------------------------------- bootstrap CI

def test_grouped_bootstrap_is_wider_than_row_level():
    """The sqrt(n_variants) understatement this guard exists to prevent.

    15 near-duplicate rows per source image, mimicking the 15 transform conditions.
    """
    rng = np.random.default_rng(0)
    n_src, k = 200, 15
    y_src = rng.integers(0, 2, n_src)
    s_src = 0.5 + 0.25 * y_src + rng.normal(0, 0.2, n_src)
    y = np.repeat(y_src, k)
    s = np.repeat(s_src, k) + rng.normal(0, 0.01, n_src * k)      # tiny within-group noise
    g = np.repeat(np.arange(n_src), k)

    _, glo, ghi = bootstrap_ci(y, s, groups=g, n_boot=400)
    _, rlo, rhi = bootstrap_ci(y, s, groups=None, n_boot=400)
    assert (ghi - glo) > 2.0 * (rhi - rlo)


def test_bootstrap_ci_brackets_point_estimate():
    y, s = data()
    p, lo, hi = bootstrap_ci(y, s, n_boot=300)
    assert lo < p < hi


def test_bootstrap_is_seed_reproducible():
    y, s = data()
    assert bootstrap_ci(y, s, n_boot=200, seed=3) == bootstrap_ci(y, s, n_boot=200, seed=3)


# --------------------------------------------------------------------- breakdowns

def test_by_group_skips_underpowered_and_single_class():
    y, s = data(n=600)
    key = np.array(["big"] * 560 + ["tiny"] * 40)
    out = by_group(y, s, key, min_n=50)
    assert out["tiny"]["skipped"] == "underpowered"
    assert "auroc" in out["big"]

    y2 = np.r_[np.ones(300, int), data(n=300)[0]]
    key2 = np.array(["allpos"] * 300 + ["mixed"] * 300)
    out2 = by_group(y2, np.random.default_rng(0).random(600), key2, min_n=10)
    assert out2["allpos"]["skipped"] == "single class"


def test_worst_family_picks_the_weakest_stratum():
    y, s = data(n=1200)
    key = np.array(["a", "b", "c"] * 400)
    s = s.copy()
    s[key == "c"] = np.random.default_rng(0).random((key == "c").sum())   # destroy stratum c
    assert worst_family(by_group(y, s, key, min_n=10))[0] == "c"
