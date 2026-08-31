"""Correctness gates for the three low-level forensic features.

Statistical properties are checked against synthetic signals with known answers, since
the real calibration (against the prior study's medians and AUROCs) needs the actual
dataset and lives in docs/RESULTS.md.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from acai.features.lowlevel import (FEATURES, _kurtosis, feature_maps, feature_vector,
                                    phase_neighbor_coherence, residual_kurtosis,
                                    to_luma, wavelet_hf_kurtosis)


def noise_img(seed=0, size=256):
    rng = np.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=np.float32)


def test_kurtosis_convention_is_excess():
    """Gaussian must give ~0, not ~3 -- the medians we calibrate against assume excess."""
    x = np.random.default_rng(0).normal(size=200_000)
    assert abs(_kurtosis(x)) < 0.1


def test_kurtosis_flat_region_is_zero_not_nan():
    """A solid-colour region is 0/0; it must not poison a mean over tiles."""
    assert _kurtosis(np.full(1000, 0.5)) == 0.0
    assert np.isfinite(feature_vector(np.full((64, 64, 3), 0.5, np.float32))).all()


def test_kurtosis_detects_heavy_tails():
    rng = np.random.default_rng(0)
    assert _kurtosis(rng.standard_t(3, 100_000)) > _kurtosis(rng.normal(size=100_000))


def test_feature_vector_shape_and_order():
    v = feature_vector(noise_img())
    assert v.shape == (3,) and v.dtype == np.float32
    luma = to_luma(noise_img())
    assert np.isclose(v[0], wavelet_hf_kurtosis(luma), rtol=1e-4)
    assert np.isclose(v[1], residual_kurtosis(luma), rtol=1e-4)
    assert np.isclose(v[2], phase_neighbor_coherence(luma), rtol=1e-4)


def test_uint8_and_float_inputs_agree():
    """Callers pass both; they must not silently produce different features."""
    f = noise_img()
    u = (f * 255).astype(np.uint8)
    assert np.allclose(feature_vector(u), feature_vector(u.astype(np.float32) / 255), rtol=1e-3)


def test_blur_reduces_wavelet_kurtosis_on_structured_input():
    """Smoothing removes the sparse high-frequency detail the feature keys on."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    im = gaussian_filter(rng.random((256, 256)), 3).astype(np.float32)   # structured
    im[::8, :] = 1.0                                                     # sparse edges
    assert wavelet_hf_kurtosis(gaussian_filter(im, 2)) < wavelet_hf_kurtosis(im)


def test_phase_coherence_ordered_beats_noise():
    """A smooth image has organised phase; white noise has none."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    smooth = gaussian_filter(rng.random((256, 256)), 8).astype(np.float32)
    assert phase_neighbor_coherence(smooth) > phase_neighbor_coherence(
        rng.random((256, 256)).astype(np.float32))


def test_phase_coherence_is_circular():
    """Bounded in [0,1]: a linear mean of phase differences would escape this range."""
    for s in range(5):
        assert 0.0 <= phase_neighbor_coherence(noise_img(s)[..., 0]) <= 1.0


def test_feature_maps_align_to_patch_grid():
    m = feature_maps(noise_img(size=256), patch=16)
    assert m.shape == (3, 16, 16) and np.isfinite(m).all()
    m = feature_maps(noise_img(size=512), patch=16)
    assert m.shape == (3, 32, 32)


def test_feature_maps_localise():
    """A patch of heavy-tailed content must light up its own tokens, not the whole map.

    This is the property H2 depends on: if the maps did not localise, dense tokens would
    carry no more information than the global scalar and H2 would collapse into H0.
    """
    rng = np.random.default_rng(0)
    im = rng.normal(0.5, 0.05, (256, 256)).astype(np.float32)
    im[:64, :64] += rng.standard_t(2, (64, 64)).astype(np.float32) * 0.05   # heavy tail
    m = feature_maps(np.clip(im, 0, 1), patch=16)[1]                        # residual kurt
    assert m[:4, :4].mean() > m[8:, 8:].mean()


def test_feature_maps_window_larger_than_patch():
    """The estimator-variance mitigation must actually be in effect."""
    from acai.features import lowlevel
    import inspect
    assert inspect.signature(lowlevel.feature_maps).parameters["window"].default > \
           inspect.signature(lowlevel.feature_maps).parameters["patch"].default


def test_shortcut_features_absent_from_this_module():
    """wavelet_level2_ratio / fft_* failed the shortcut audit and must never be model inputs."""
    from acai.features import lowlevel
    banned = ("wavelet_level2_ratio", "fft_mid_energy", "fft_spectral_entropy", "fft_high_energy")
    assert not [b for b in banned if hasattr(lowlevel, b)]
    assert set(FEATURES) == {"wavelet_hf_kurtosis", "residual_kurtosis", "phase_neighbor_coherence"}
