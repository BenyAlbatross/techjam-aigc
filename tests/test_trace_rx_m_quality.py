from __future__ import annotations

import numpy as np

from techjam_aigc.trace_rx_m.quality import (
    QuantileQualityBinner,
    extract_quality_descriptor,
    jpeg_blockiness,
    structural_high_frequency_energy,
    wavelet_mad_noise,
)


def test_quality_descriptor_has_proposal_order_and_finite_values() -> None:
    image = np.tile(np.linspace(0.0, 1.0, 32), (24, 1))
    descriptor = extract_quality_descriptor(image)

    assert descriptor.as_array().shape == (4,)
    assert np.isclose(descriptor.log_min_dimension, np.log(24))
    assert np.isfinite(descriptor.as_array()).all()
    assert descriptor.noise_sigma >= 0.0
    assert descriptor.structural_hf_energy >= 0.0


def test_wavelet_mad_tracks_white_noise_scale() -> None:
    rng = np.random.default_rng(7)
    sigma = 0.04
    image = np.clip(0.5 + rng.normal(0.0, sigma, size=(512, 512)), 0.0, 1.0)

    estimate = wavelet_mad_noise(image)

    assert np.isclose(estimate, sigma, rtol=0.08)


def test_blockiness_detects_eight_pixel_boundaries() -> None:
    block_levels = np.repeat(np.arange(8) % 2, 8).astype(np.float64)
    blocked = np.tile(block_levels, (64, 1))
    smooth = np.tile(np.linspace(0.0, 1.0, 64), (64, 1))

    assert jpeg_blockiness(blocked) > 0.9
    assert jpeg_blockiness(blocked) > jpeg_blockiness(smooth)


def test_structural_energy_subtracts_supplied_noise_floor_and_clips() -> None:
    rng = np.random.default_rng(9)
    image = rng.normal(0.5, 0.03, size=(128, 128))

    energy = structural_high_frequency_energy(image, noise_sigma=1.0)

    assert energy == 0.0


def test_quantile_binner_is_deterministic_and_collapses_duplicate_edges() -> None:
    descriptors = np.array(
        [
            [1.0, 0.0, 4.0, 2.0],
            [2.0, 0.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 2.0],
            [4.0, 0.0, 1.0, 2.0],
        ]
    )
    left = QuantileQualityBinner(2).fit(descriptors)
    right = QuantileQualityBinner(2).fit(descriptors[::-1])

    assert left.bin_counts == (2, 1, 2, 1)
    np.testing.assert_array_equal(left.transform(descriptors), right.transform(descriptors))
    assert list(left.iter_cells()) == [(0, 0, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0), (1, 0, 1, 0)]

