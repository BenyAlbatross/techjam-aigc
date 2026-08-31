"""The three low-level forensic features carried forward from the feature study.

Prior results these must reproduce (step 4 gate in docs/PLAN.md):

    feature                    clean AUROC (95% CI)     worst official transform
    wavelet_hf_kurtosis        0.654 (0.575-0.726)      0.612 @ 0.25x resize
    residual_kurtosis          0.674 (0.596-0.736)      0.618 @ 0.25x resize
    phase_neighbor_coherence   0.624 (0.536-0.696)      0.597 @ blur sigma=1

Median values reported there: wavelet_hf_kurtosis 15.45 (AIGC) vs 11.96 (authentic);
residual_kurtosis 7.49 vs 4.99. Those medians are the calibration target -- they pin
down the two conventions the prose leaves open (excess vs Pearson kurtosis, and which
subband set is pooled), so `selftest_medians()` below is how we tell whether our
implementation is the same one those AUROCs came from.

The four shortcut-failing features from that study (wavelet_level2_ratio, fft_mid_energy,
fft_spectral_entropy, fft_high_energy) are deliberately NOT implemented here. They are
audit probes only and live in acai/audit.py -- keeping them out of this module makes it
structurally impossible to feed them to a model by accident.

Every feature is computed on the *post-transform* image, because that is the image the
deployed detector actually sees.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, uniform_filter

FEATURES = ("wavelet_hf_kurtosis", "residual_kurtosis", "phase_neighbor_coherence")

# Kurtosis convention: Fisher (excess, normal == 0.0) is scipy's default and is what the
# reported medians (15.45 / 11.96 / 7.49 / 4.99) are consistent with -- Pearson would put
# these ~3 higher. selftest_medians() re-checks this against real data rather than trusting
# the inference.
EXCESS = True

# Luma weights: Rec.601, matching the colour op in acai/transforms.py so the two modules
# do not disagree about what "grey" means.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_luma(a: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] or uint8 HxWx3 -> float32 luma [0,1]."""
    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 2:
        return a.astype(np.float32)
    return (a[..., :3] * _LUMA).sum(-1).astype(np.float32)


def _kurtosis(x: np.ndarray, axis=None) -> np.ndarray | float:
    """Kurtosis with a guarded denominator.

    Flat regions (a blown-out sky, a solid background) have ~zero variance, where the
    ratio is 0/0. Those return 0.0 -- "no tail information" -- rather than nan, which
    would poison a downstream mean over tiles.
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(axis=axis, keepdims=True)
    d = x - m
    var = (d ** 2).mean(axis=axis)
    m4 = (d ** 4).mean(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = m4 / np.maximum(var, 1e-12) ** 2
    raw = raw - 3.0 if EXCESS else raw
    # Degenerate (flat) case is set *after* the excess shift, so it lands on the neutral
    # value of whichever convention is active -- 0.0 for excess, 3.0 for Pearson.
    return np.where(var > 1e-12, raw, 0.0 if EXCESS else 3.0)


# ------------------------------------------------------------------ global features

def wavelet_hf_kurtosis(luma: np.ndarray) -> float:
    """Tail-heaviness of level-1 Haar detail coefficients, pooled over LH/HL/HH.

    Pooled rather than averaged per-subband: the three subbands share a scale under Haar,
    so pooling gives one estimator with 3x the samples instead of three noisier ones.
    """
    _, (lh, hl, hh) = pywt.dwt2(luma, "haar")
    return float(_kurtosis(np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()])))


def residual_kurtosis(luma: np.ndarray, sigma: float = 1.0) -> float:
    """Tail-heaviness of the residual left after Gaussian smoothing.

    sigma=1.0 is the natural reading of "the residual after Gaussian smoothing" and
    coincides with an official blur level, which conveniently makes the
    blur-sigma=1 condition a near-worst case for this feature -- exactly the
    interaction the robustness study needs to measure.
    """
    return float(_kurtosis((luma - gaussian_filter(luma, sigma)).ravel()))


def phase_neighbor_coherence(luma: np.ndarray) -> float:
    """Local organisation between neighbouring Fourier phases.

    Circular coherence |<exp(i d_phi)>| over horizontally- and vertically-adjacent
    frequency bins. Circular, not linear, because phase lives on a circle: a plain mean
    of phase differences would treat -pi and +pi as maximally distant when they are
    identical.

    Sign convention is frozen to the discovery direction: AIGC images had *lower*
    coherence, so this feature is negatively oriented and the scorer must not silently
    flip it. The prior study also found this feature only weakly correlated with the two
    kurtoses (rho ~ -0.21..-0.25), which is the entire reason it is kept despite being
    the weakest of the three.
    """
    f = np.fft.fft2(luma - luma.mean())
    ph = np.angle(np.fft.fftshift(f))
    dh = np.exp(1j * (ph[:, 1:] - ph[:, :-1]))
    dv = np.exp(1j * (ph[1:, :] - ph[:-1, :]))
    return float(abs(np.concatenate([dh.ravel(), dv.ravel()]).mean()))


def feature_vector(img: np.ndarray) -> np.ndarray:
    """The three global features, in FEATURES order."""
    luma = to_luma(img)
    return np.array([wavelet_hf_kurtosis(luma),
                     residual_kurtosis(luma),
                     phase_neighbor_coherence(luma)], dtype=np.float32)


# ------------------------------------------------- dense (per-patch) features for H2

def feature_maps(img: np.ndarray, patch: int = 16, window: int = 32) -> np.ndarray:
    """Per-patch feature maps aligned to DINOv3's patch grid -> (3, H/patch, W/patch).

    `window` > `patch` on purpose. A 16x16 patch holds 256 samples, and a fourth-moment
    estimator on 256 samples is badly noisy -- tile-level kurtosis would be mostly
    estimator variance. Using a 32x32 window at stride 16 gives each patch 4x the samples
    while still producing exactly one vector per DINOv3 token. Overlap is a feature here,
    not a bug: neighbouring tokens genuinely share local statistics.

    Returned in FEATURES order so the channel axis matches `feature_vector`.
    """
    luma = to_luma(img)
    H, W = luma.shape
    gh, gw = H // patch, W // patch
    pad = (window - patch) // 2
    p = np.pad(luma, pad, mode="reflect")

    # -- residual kurtosis, and wavelet HF kurtosis, via windowed moments -------------
    res = p - gaussian_filter(p, 1.0)
    # Haar level-1 details, kept at full resolution as the three directional differences
    # so they can be windowed on the same grid as everything else.
    d_h = np.zeros_like(p); d_h[:, :-1] = p[:, 1:] - p[:, :-1]
    d_v = np.zeros_like(p); d_v[:-1, :] = p[1:, :] - p[:-1, :]
    d_d = np.zeros_like(p); d_d[:-1, :-1] = p[1:, 1:] - p[:-1, :-1]

    def windowed_kurt(x: np.ndarray) -> np.ndarray:
        """Excess kurtosis in a `window` box at every pixel, then sampled at patch centres."""
        m1 = uniform_filter(x, window, mode="reflect")
        m2 = uniform_filter(x * x, window, mode="reflect")
        c = x - m1
        var = np.maximum(m2 - m1 * m1, 0.0)
        m4 = uniform_filter(c ** 4, window, mode="reflect")
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = m4 / np.maximum(var, 1e-12) ** 2
        raw = raw - 3.0 if EXCESS else raw
        k = np.where(var > 1e-12, raw, 0.0 if EXCESS else 3.0)
        # patch centres in padded coordinates
        r = pad + patch // 2 + patch * np.arange(gh)
        c_ = pad + patch // 2 + patch * np.arange(gw)
        return k[np.ix_(r, c_)]

    wav = windowed_kurt(np.abs(d_h) + np.abs(d_v) + np.abs(d_d))
    resid = windowed_kurt(res)

    # -- phase coherence, computed per tile (FFT does not localise via box filters) ---
    coh = np.empty((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            t = p[i * patch:i * patch + window, j * patch:j * patch + window]
            coh[i, j] = phase_neighbor_coherence(t)

    return np.stack([wav, resid, coh]).astype(np.float32)


# ------------------------------------------------------------------------ self-test

def selftest_medians(images: list[np.ndarray]) -> dict[str, float]:
    """Median of each global feature over a sample -- the step-4 calibration gate.

    Compare against the prior study's medians (AIGC / authentic):
        wavelet_hf_kurtosis  15.45 / 11.96
        residual_kurtosis     7.49 /  4.99
    A large offset means our convention differs from theirs, and every hybrid result
    downstream would be built on a different feature than the one that was validated.
    """
    v = np.stack([feature_vector(im) for im in images])
    return {f: float(np.median(v[:, i])) for i, f in enumerate(FEATURES)}
