from __future__ import annotations

import numpy as np
from PIL import Image

from techjam_aigc.feature_lab.features import extract_features
from techjam_aigc.feature_lab.registry import FEATURE_REGISTRY


PLANNED_FAMILIES = {
    "bit_plane",
    "patch_distribution",
    "multiscale_residual",
    "steganalysis",
    "camera_proxy",
    "fft_magnitude",
    "fft_phase",
    "codec_resampling",
    "chroma",
}


def test_extension_registry_is_documented_and_unique() -> None:
    names = [spec.name for spec in FEATURE_REGISTRY]
    assert len(names) == len(set(names))
    assert PLANNED_FAMILIES <= {spec.family for spec in FEATURE_REGISTRY}

    extension_specs = [spec for spec in FEATURE_REGISTRY if spec.family in PLANNED_FAMILIES]
    assert extension_specs
    for spec in extension_specs:
        assert spec.measurement
        assert spec.hypothesis
        assert spec.expected_failure
        assert spec.role in {"candidate", "nuisance"}

    camera_names = [spec.name.lower() for spec in FEATURE_REGISTRY if spec.family == "camera_proxy"]
    assert camera_names
    assert all("proxy" in name for name in camera_names)
    assert all("prnu" not in name for name in camera_names)


def test_all_features_are_deterministic_and_finite_for_tiny_constant_images() -> None:
    images = [
        Image.new("L", (1, 1), 0),
        Image.new("RGB", (2, 3), (127, 127, 127)),
        Image.new("RGBA", (7, 1), (255, 0, 128, 64)),
    ]
    for image in images:
        metadata = {"width": image.width, "height": image.height, "bytes": 0, "format": "PNG"}
        first = extract_features(image, metadata)
        second = extract_features(image, metadata)
        assert first == second
        assert np.isfinite(list(first.values())).all()


def test_composed_bitplane_gradient_patch_measurement_responds_to_lsb_pattern() -> None:
    checkerboard = np.indices((32, 32)).sum(axis=0) % 2
    patterned = np.full((32, 32, 3), 128, dtype=np.uint8)
    patterned[..., 0] += checkerboard.astype(np.uint8)

    flat_features = extract_features(Image.fromarray(np.full_like(patterned, 128), "RGB"))
    patterned_features = extract_features(Image.fromarray(patterned, "RGB"))

    assert patterned_features["bitplane_gradient_patch_max"] > flat_features["bitplane_gradient_patch_max"]
    assert patterned_features["bitplane_low_entropy"] > flat_features["bitplane_low_entropy"]
    assert 0.0 <= patterned_features["stego_residual_cooc_diagonal"] <= 1.0


def test_representative_extension_values_have_interpretable_ranges() -> None:
    y, x = np.mgrid[0:48, 0:64]
    rgb = np.stack(
        [
            (5 * x + 3 * y) % 256,
            (2 * x + 7 * y) % 256,
            (11 * x + y) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    values = extract_features(Image.fromarray(rgb, "RGB"))

    unit_interval_features = {
        "bitplane_low_occupancy",
        "bitplane_low_entropy",
        "bitplane_directional_transition",
        "bitplane_cross_channel_agreement",
        "multiscale_residual_tail_mean",
        "stego_residual_cooc_entropy",
        "stego_residual_cooc_diagonal",
        "stego_residual_directional_gap",
        "camera_signal_noise_fit_proxy",
        "fft_multiring_angular_entropy",
        "fft_cross_channel_coherence",
        "fft_phase_magnitude_coupling",
        "ycbcr_chroma_tail_fraction",
    }
    for name in unit_interval_features:
        assert 0.0 <= values[name] <= 1.0, (name, values[name])
    assert -1.0 <= values["multiscale_residual_crossscale_corr"] <= 1.0
