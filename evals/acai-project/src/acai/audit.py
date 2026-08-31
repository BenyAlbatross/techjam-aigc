"""Shortcut audit: is the model detecting AIGC, or detecting the dataset's pipeline?

This module exists because the prior work on this corpus found shortcuts that beat real
detectors. From the dataset card for `techjam-aigc/wildfake-eval-subset`:

    def cheat(img):
        return 0 if img.size == (200, 200) else 1     # AUROC 1.000, learns nothing

and, with size removed, mean luminance still reaches AUROC 0.734 on `laion_matched`.

So every model is audited, not just suspicious ones. A model that beats the baseline while
its score correlates 0.8 with image resolution has learned the acquisition pipeline.

The four features that failed the prior study's shortcut audit live here, and *only* here:

    wavelet_level2_ratio   clean 0.660, falls to 0.469 under resize, r=0.773 with size
    fft_mid_energy         clean 0.650, native-resolution AUROC 0.495, r=0.833
    fft_spectral_entropy   clean 0.644, native-resolution AUROC 0.424
    fft_high_energy        clean 0.605, native-resolution AUROC 0.449, r=0.843

They are kept as probes precisely because they are good shortcut *detectors*: a model that
correlates strongly with them is likely leaning on resolution or provenance. Keeping them
out of acai.features makes it structurally impossible to feed them to a model by accident.
"""
from __future__ import annotations

import io

import numpy as np
import pywt
from PIL import Image
from scipy.stats import spearmanr

from acai.features.lowlevel import to_luma
from acai.metrics import compute

NUISANCES = ("pixels", "aspect", "mean_luminance", "saturation", "recompressed_bytes",
             "laplacian_var")

SHORTCUT_FEATURES = ("wavelet_level2_ratio", "fft_mid_energy", "fft_spectral_entropy",
                     "fft_high_energy")


# ------------------------------------------------------------------ nuisance probes

def nuisance_vector(img: np.ndarray) -> dict[str, float]:
    """Trivial image properties that must not explain a detector's score."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    h, w = a.shape[:2]
    luma = to_luma(a)
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    buf = io.BytesIO()
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=90)

    lap = (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
           - 4 * luma[1:-1, 1:-1])
    return {"pixels": float(h * w), "aspect": float(w / h),
            "mean_luminance": float(luma.mean()), "saturation": float(sat.mean()),
            # Normalised by pixel count: raw byte size is mostly just resolution.
            "recompressed_bytes": float(buf.tell() / (h * w)),
            "laplacian_var": float(lap.var())}


# --------------------------------------------------- shortcut features (probes only)

def shortcut_vector(img: np.ndarray) -> dict[str, float]:
    """The four features that failed the prior shortcut audit. Probes, never model inputs."""
    luma = to_luma(img)
    c = pywt.wavedec2(luma, "haar", level=2)
    l2 = np.concatenate([b.ravel() for b in c[1]])       # level-2 detail
    l1 = np.concatenate([b.ravel() for b in c[2]])       # level-1 detail
    e1 = float((l1 ** 2).mean())

    f = np.abs(np.fft.fftshift(np.fft.fft2(luma - luma.mean()))) ** 2
    h, w = f.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    tot = float(f.sum()) + 1e-12
    p = (f / tot).ravel()
    p = p[p > 0]
    return {"wavelet_level2_ratio": float((l2 ** 2).mean() / (e1 + 1e-12)),
            "fft_mid_energy": float(f[(r > 0.25) & (r <= 0.6)].sum() / tot),
            "fft_high_energy": float(f[r > 0.6].sum() / tot),
            "fft_spectral_entropy": float(-(p * np.log(p)).sum() / np.log(len(p)))}


# --------------------------------------------------------------------- the audit

def audit(y, score, nuisances: dict[str, np.ndarray], native_mask=None,
          shortcuts: dict[str, np.ndarray] | None = None,
          warn_at: float = 0.5) -> dict:
    """Full shortcut report for one model's predictions.

    * `nuisance_auroc` -- how well each nuisance *alone* separates the classes. This is a
      property of the dataset, not the model, and it bounds how much credit any AUROC
      deserves: if luminance alone gets 0.73, a model at 0.75 has shown almost nothing.
    * `score_correlation` -- Spearman between the model score and each nuisance. High
      correlation means the model may be riding the shortcut.
    * `native_resolution_auroc` -- AUROC restricted to un-resampled images, where
      resolution-derived shortcuts are unavailable.

    `flags` lists nuisances exceeding `warn_at` in absolute correlation. Reported, never
    silently corrected -- the point is to make the caveat impossible to omit.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    out: dict = {"n": int(len(y)), "overall_auroc": compute(y, score).auroc,
                 "nuisance_auroc": {}, "score_correlation": {}, "flags": []}

    for k, v in nuisances.items():
        v = np.asarray(v, float)
        a = compute(y, v).auroc
        out["nuisance_auroc"][k] = float(max(a, 1 - a))       # direction-agnostic
        rho = float(spearmanr(score, v).statistic)
        out["score_correlation"][k] = rho
        if abs(rho) >= warn_at:
            out["flags"].append({"nuisance": k, "spearman": rho,
                                 "note": "model score tracks a nuisance variable"})

    if shortcuts:
        out["shortcut_feature_correlation"] = {
            k: float(spearmanr(score, np.asarray(v, float)).statistic)
            for k, v in shortcuts.items()}

    if native_mask is not None:
        m = np.asarray(native_mask, bool)
        if m.sum() >= 30 and len(np.unique(y[m])) == 2:
            nat = compute(y[m], score[m]).auroc
            out["native_resolution_auroc"] = nat
            out["native_resolution_n"] = int(m.sum())
            # The prior study's own diagnosis of fft_mid_energy: clean 0.650 but native 0.495.
            if out["overall_auroc"] - nat > 0.10:
                out["flags"].append({
                    "nuisance": "resolution",
                    "note": f"AUROC drops {out['overall_auroc'] - nat:.3f} on native-resolution "
                            "images; the gain may be resampling provenance, not detection"})

    out["cheat_baseline"] = max(out["nuisance_auroc"].values(), default=float("nan"))
    return out
