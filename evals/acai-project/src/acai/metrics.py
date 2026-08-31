"""Evaluation metrics for AIGC detection.

Design rules, each of which exists because the obvious alternative is misleading here:

* **AUROC is primary, AUPRC always carries its base rate.** The composition study (plan §4)
  deliberately varies class prevalence, and AUPRC is not comparable across different
  prevalences -- its random baseline *is* the positive rate. Every AUPRC is therefore
  reported next to `prevalence` and as a lift over that baseline.
* **TPR at low FPR is reported explicitly.** A deployed detector runs at a low false-positive
  rate; AUROC integrates over the whole curve and hides that corner entirely. Two models can
  tie on AUROC while one is twice as good at FPR=1%.
* **Bootstrap resamples groups, not rows.** With ~15 transformed variants per source image,
  row-level resampling treats 15 correlated rows as 15 independent samples and shrinks the
  CI by roughly sqrt(15). Passing `groups=source_image_id` is not optional for any table
  that reports a CI.
* **Model comparisons use DeLong**, which accounts for the two AUROCs being computed on the
  *same* images. Comparing overlapping CIs instead is a well-known way to miss real
  differences.

Positive class is label 1 == AIGC throughout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ------------------------------------------------------------------ point estimates

@dataclass
class Metrics:
    n: int
    n_pos: int
    prevalence: float
    auroc: float
    auprc: float
    auprc_lift: float          # auprc / prevalence; 1.0 == no better than chance
    eer: float
    balanced_acc_50: float     # at the fixed 0.5 threshold
    balanced_acc_eer: float    # at the EER-optimal threshold
    tpr_at_fpr01: float
    tpr_at_fpr05: float
    brier: float
    ece: float

    def as_dict(self) -> dict:
        return asdict(self)


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """TPR at the largest operating point with FPR <= target.

    Interpolated, and clamped to 0.0 when no such point exists -- a model whose ROC curve
    never reaches FPR=1% genuinely cannot be operated there, and reporting the TPR of the
    nearest looser threshold would overstate it.
    """
    ok = fpr <= target
    if not ok.any():
        return 0.0
    return float(np.interp(target, fpr, tpr))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None) -> Metrics:
    """All point metrics. `score` is higher == more likely AIGC.

    `weights` supports the eval-composition sensitivity analysis (plan §4.4): re-weighting
    fixed predictions to simulate a different eval mixture, with no retraining.

    Calibration metrics (Brier, ECE) assume `score` is a probability. When it is an
    uncalibrated margin they are reported anyway but are only meaningful *relatively*,
    between models on the same scale.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    if len(np.unique(y)) < 2:
        raise ValueError("both classes must be present to compute AUROC")

    prev = float(np.average(y, weights=w))
    fpr, tpr, thr = roc_curve(y, score, sample_weight=w)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2)

    def bal_acc(t: float) -> float:
        pred = score >= t
        pos, neg = y == 1, y == 0
        wp = None if w is None else w
        sens = np.average(pred[pos], weights=None if wp is None else wp[pos])
        spec = np.average(~pred[neg], weights=None if wp is None else wp[neg])
        return float((sens + spec) / 2)

    ap = float(average_precision_score(y, score, sample_weight=w))
    return Metrics(
        n=len(y), n_pos=int(y.sum()), prevalence=prev,
        auroc=float(roc_auc_score(y, score, sample_weight=w)),
        auprc=ap, auprc_lift=float(ap / prev) if prev > 0 else float("nan"),
        eer=eer,
        balanced_acc_50=bal_acc(0.5), balanced_acc_eer=bal_acc(float(thr[i])),
        tpr_at_fpr01=_tpr_at_fpr(fpr, tpr, 0.01),
        tpr_at_fpr05=_tpr_at_fpr(fpr, tpr, 0.05),
        brier=float(np.average((score - y) ** 2, weights=w)),
        ece=_ece(y, np.clip(score, 0, 1)),
    )


# --------------------------------------------------------------------- uncertainty

def bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray | None = None,
                 metric: str = "auroc", n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) percentile CI, resampling **groups** with replacement.

    `groups` should be `source_image_id`: all transformed variants of one source image
    move together, because they are not independent observations. Omitting it silently
    understates the interval.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    point = getattr(compute(y, score), metric)

    if groups is None:
        groups = np.arange(len(y))
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue                      # degenerate resample; skip rather than error
        vals.append(getattr(compute(y[idx], score[idx]), metric))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ------------------------------------------------------------------- DeLong's test

def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def delong_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Paired DeLong test for AUROC(a) - AUROC(b) on the same samples.

    Fast algorithm of Sun & Xu (2014). Returns the two AUROCs, their difference, the
    standard error of the difference, z and a two-sided p-value.

    This is the test that decides whether a hybrid beats pure DINOv3 (plan §3.2). An
    unpaired comparison would ignore that both models scored the *same* images, throwing
    away most of the statistical power.
    """
    y = np.asarray(y).astype(int)
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("both classes must be present")

    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    px, py = preds[:, pos], preds[:, ~pos]
    k = 2

    tx = np.array([_midrank(px[r]) for r in range(k)])
    ty = np.array([_midrank(py[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([px[r], py[r]])) for r in range(k)])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)

    diff = float(aucs[0] - aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    return dict(auroc_a=float(aucs[0]), auroc_b=float(aucs[1]), diff=diff,
                se=se, z=float(z), p=float(2 * stats.norm.sf(abs(z))))


# --------------------------------------------------------------------- breakdowns

def by_group(y, score, key, groups=None, min_n: int = 30) -> dict[str, dict]:
    """Metrics split by `key` (transform, severity, generator, resolution bucket...).

    Strata with fewer than `min_n` rows, or only one class present, report `None` with a
    reason rather than a number. An AUROC from 8 images is noise, and printing it in a
    table invites it to be read as a finding.
    """
    y, score, key = np.asarray(y), np.asarray(score), np.asarray(key)
    out: dict[str, dict] = {}
    for k in sorted(set(key.tolist())):
        m = key == k
        if m.sum() < min_n:
            out[str(k)] = {"n": int(m.sum()), "skipped": "underpowered"}
        elif len(np.unique(y[m])) < 2:
            out[str(k)] = {"n": int(m.sum()), "skipped": "single class"}
        else:
            d = compute(y[m], score[m]).as_dict()
            if groups is not None:
                _, lo, hi = bootstrap_ci(y[m], score[m], np.asarray(groups)[m], n_boot=500)
                d["auroc_ci"] = [lo, hi]
            out[str(k)] = d
    return out


def worst_family(breakdown: dict[str, dict], metric: str = "auroc") -> tuple[str, float]:
    """The weakest non-skipped stratum -- the number a robustness claim actually rests on.

    Mean-over-transforms hides a single catastrophic family; if the hidden evaluation
    happens to weight that family, the mean was never the relevant statistic.
    """
    ok = {k: v for k, v in breakdown.items() if metric in v}
    if not ok:
        return "", float("nan")
    k = min(ok, key=lambda k: ok[k][metric])
    return k, float(ok[k][metric])
