"""Preregistered feature hypotheses for the robustness laboratory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    measurement: str
    hypothesis: str
    expected_failure: str
    role: str = "candidate"
    cost: str = "low"


def _f(
    name: str,
    family: str,
    measurement: str,
    hypothesis: str,
    expected_failure: str,
    role: str = "candidate",
    cost: str = "low",
) -> FeatureSpec:
    return FeatureSpec(name, family, measurement, hypothesis, expected_failure, role, cost)


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    _f("meta_log_pixels", "nuisance", "log native pixel count", "Dataset pipelines may couple size to label.", "Any common resize.", "nuisance"),
    _f("meta_aspect_ratio", "nuisance", "native width / height", "Dataset sources may couple framing to label.", "Cropping or source matching.", "nuisance"),
    _f("meta_bytes_per_pixel", "nuisance", "encoded bytes / native pixels", "Codec and quality may be label-correlated.", "Matched re-encoding.", "nuisance"),
    _f("meta_is_jpeg", "nuisance", "JPEG indicator", "File format may be a direct dataset shortcut.", "Format alignment.", "nuisance"),
    _f("meta_is_png", "nuisance", "PNG indicator", "File format may be a direct dataset shortcut.", "Format alignment.", "nuisance"),
    _f("luma_mean", "spatial", "mean luminance", "Generator tone distributions can differ from camera/web images.", "Color jitter and semantic mismatch."),
    _f("luma_std", "spatial", "luminance standard deviation", "Generated images may have different global contrast.", "Contrast jitter and content."),
    _f("luma_entropy", "spatial", "64-bin luminance entropy", "Synthetic tone usage may be unusually regular.", "Compression, noise, and content."),
    _f("luma_dynamic_range", "spatial", "95th minus 5th percentile", "Generated images may compress or exaggerate tone range.", "Color jitter and source mismatch."),
    _f("local_contrast_mean", "spatial", "mean local luminance deviation", "Local contrast can expose oversmoothing or enhancement.", "Blur, resize, and scene content."),
    _f("clipped_fraction", "color", "fraction of RGB values near 0 or 1", "Generation or post-processing may clip channels differently.", "Exposure changes and source pipelines."),
    _f("saturation_mean", "color", "mean HSV-like saturation", "Synthetic color distributions may be over-saturated.", "Saturation jitter and semantics."),
    _f("saturation_std", "color", "saturation standard deviation", "Synthetic color variation may differ locally.", "Saturation jitter and semantics."),
    _f("colorfulness", "color", "Hasler-Suesstrunk colorfulness", "Generators may favor vivid chroma.", "Color editing and content."),
    _f("chroma_energy", "color", "mean squared opponent chroma", "Decoder/color-pipeline statistics can affect chroma.", "Color jitter and codec subsampling."),
    _f("channel_correlation", "color", "mean RGB pair correlation", "Cross-channel dependence may differ without a camera pipeline.", "Flat images, jitter, and source processing."),
    _f("lbp_entropy", "texture", "entropy of 8-neighbor local binary patterns", "Local generator textures can have different microstructure.", "Blur, resize, and generator family."),
    _f("lbp_uniform_fraction", "texture", "fraction of low-transition LBP codes", "Synthetic textures may be too locally regular.", "Noise and low resolution."),
    _f("glcm_contrast", "texture", "16-level neighbor co-occurrence contrast", "Neighbor transitions encode texture formation statistics.", "Resize, blur, and semantic content."),
    _f("glcm_homogeneity", "texture", "co-occurrence inverse-distance homogeneity", "Generated surfaces may be unusually homogeneous.", "Noise and detailed scenes."),
    _f("residual_std", "residual", "standard deviation after Gaussian denoising", "Decoder traces can remain in high-pass residuals.", "Blur, JPEG, resizing, and noise."),
    _f("residual_mad", "residual", "median absolute Gaussian residual", "Robust high-frequency strength may differ by image origin.", "Blur, JPEG, and resolution."),
    _f("residual_kurtosis", "residual", "Gaussian-residual kurtosis", "Natural and synthetic residual tails may differ.", "Noise injection and compression."),
    _f("residual_neighbor_corr", "residual", "residual autocorrelation", "Generators may leave spatially dependent residuals.", "Resize kernels and JPEG blocks."),
    _f("noise_laplacian_mad", "noise", "robust Laplacian magnitude", "Real acquisition noise and synthetic decoder noise may differ.", "Any blur/noise and weak camera provenance."),
    _f("noise_flat_region_std", "noise", "residual deviation in low-gradient pixels", "Camera noise persists in flat areas while synthetic noise may not.", "Web processing, denoising, and non-camera reals."),
    _f("noise_intensity_slope", "noise", "slope of residual scale versus luminance", "Shot noise is signal-dependent; synthetic noise may be less so.", "Unknown camera pipeline and JPEG."),
    _f("fft_low_energy", "fft_magnitude", "normalized Fourier energy below 0.10 cycles/pixel", "Generators can shift energy between coarse and fine scales.", "Content and canonicalization."),
    _f("fft_mid_energy", "fft_magnitude", "normalized Fourier energy from 0.10 to 0.25", "Decoder/resampling statistics may affect mid bands.", "Resize and blur."),
    _f("fft_high_energy", "fft_magnitude", "normalized Fourier energy above 0.25", "Upsampling and decoder artifacts can enrich high frequencies.", "JPEG, blur, resize, and noise."),
    _f("fft_radial_slope", "fft_magnitude", "log radial-power slope", "Natural-image spectra often follow scale regularities generators may miss.", "Resolution, windowing, and content."),
    _f("fft_spectral_entropy", "fft_magnitude", "normalized power-spectrum entropy", "Periodic generator traces concentrate spectral energy.", "Noise and JPEG."),
    _f("fft_anisotropy", "fft_magnitude", "horizontal/vertical spectral imbalance", "Architecture or resampling may produce oriented artifacts.", "Scene edges and crops."),
    _f("fft_peak_ratio", "fft_magnitude", "strongest non-DC power relative to mean", "Periodic fingerprints create spectral peaks.", "Repeating real textures and resize aliasing."),
    _f("phase_resultant", "fft_phase", "circular resultant of Fourier phase", "Phase organization may differ despite similar magnitude.", "Translations and crop geometry."),
    _f("phase_entropy", "fft_phase", "18-bin Fourier phase entropy", "Generated geometry may have different phase statistics.", "Crop, resizing, and low texture."),
    _f("phase_neighbor_coherence", "fft_phase", "cosine coherence of adjacent phase", "Locally structured phase may expose synthesis regularity.", "Crop, blur, and resolution."),
    _f("dct_high_ac_ratio", "dct_jpeg", "8x8 DCT energy in u+v>=8", "Decoder or codec history changes block-frequency energy.", "Matched JPEG and blur."),
    _f("dct_ac_kurtosis", "dct_jpeg", "kurtosis of non-DC block DCT coefficients", "Quantization and synthesis affect coefficient tails.", "Re-encoding and content."),
    _f("dct_near_zero_fraction", "dct_jpeg", "fraction of small non-DC coefficients", "JPEG quantization creates many near-zero coefficients.", "Direct codec shortcut; matched re-encoding."),
    _f("jpeg_blockiness", "dct_jpeg", "8-pixel boundary discontinuity ratio", "JPEG/resampling grids can be label-correlated.", "Format alignment and non-JPEG sources."),
    _f("wavelet_hf_energy", "wavelet", "one-level Haar high-band energy fraction", "Synthetic decoders may distribute fine energy differently.", "Blur, resize, JPEG, and noise."),
    _f("wavelet_hf_kurtosis", "wavelet", "Haar high-band kurtosis", "Natural edges and decoder residuals have different tails.", "Noise and low resolution."),
    _f("wavelet_orientation_imbalance", "wavelet", "imbalance among LH/HL/HH energies", "Upsampling artifacts may be directionally structured.", "Oriented scene texture."),
    _f("wavelet_level2_ratio", "wavelet", "second-level high energy / first-level high energy", "Cross-scale consistency may differ by formation process.", "Resize and very small images."),
    _f("gradient_energy", "gradient", "mean Sobel magnitude", "Synthetic fine detail may have atypical edge strength.", "Blur, resize, noise, and content."),
    _f("gradient_orientation_entropy", "gradient", "weighted Sobel orientation entropy", "Generator artifacts may favor orientations.", "Scene geometry and crop."),
    _f("laplacian_std", "gradient", "standard deviation of discrete Laplacian", "Sharp transitions and microtexture differ by pipeline.", "Blur, noise, and compression."),
    _f("edge_density", "gradient", "fraction above adaptive gradient threshold", "Generated detail may be too smooth or over-sharpened.", "Blur and semantic content."),
    _f("bitplane_low_occupancy", "bit_plane", "mean occupancy of RGB bit planes 0--3", "Low-order sample codes may retain generator or camera-pipeline regularities.", "JPEG, noise, re-encoding, screenshots, and reduced bit depth."),
    _f("bitplane_low_entropy", "bit_plane", "mean Bernoulli entropy of RGB bit planes 0--3", "Synthetic low bits may be less random or differently balanced than acquired-image low bits.", "JPEG, noise, re-encoding, screenshots, and reduced bit depth."),
    _f("bitplane_directional_transition", "bit_plane", "horizontal/vertical transition imbalance over RGB bit planes 0--3", "Decoder upsampling can leave oriented structure in low bit planes.", "JPEG, noise, resampling, and strongly oriented content."),
    _f("bitplane_cross_channel_agreement", "bit_plane", "mean pairwise agreement of RGB low bit planes", "A camera color pipeline and a generator decoder can couple channel low bits differently.", "Grayscale content, color conversion, and chroma subsampling."),
    _f("bitplane_gradient_patch_max", "bit_plane", "maximum patch mean of directional transitions after low-bit slicing", "Localized generator artifacts may be visible in low-bit directional gradients even when global summaries dilute them.", "Crop selection, JPEG, noise, tiny images, and reduced bit depth."),
    _f("patch_residual_q90", "patch_distribution", "90th percentile of patch Gaussian-residual MAD", "Synthetic artifacts may be localized rather than image-wide.", "Tiny images, blur, crop selection, and semantic texture mismatch."),
    _f("patch_residual_heterogeneity", "patch_distribution", "IQR of patch Gaussian-residual MAD divided by its median", "Spatial variation in fine residuals can distinguish acquisition noise from synthesized texture.", "Flat scenes, injected noise, and small images."),
    _f("patch_gradient_q90", "patch_distribution", "90th percentile of patch Sobel energy", "Over-detailed synthetic regions can produce an upper tail not visible in global edge strength.", "Sharpening, blur, crop selection, and textured real scenes."),
    _f("patch_spectral_high_q90", "patch_distribution", "90th percentile of patch high-frequency spectral-energy fraction", "Local decoder or upsampling artifacts may concentrate high-frequency energy in selected regions.", "Small patches, JPEG, noise, and repetitive real texture."),
    _f("multiscale_residual_std_slope", "multiscale_residual", "slope of log residual standard deviation over Gaussian sigma 0.5, 1, 2, 4", "The scale response of generator residuals may differ from camera and scene detail.", "Blur, resize, injected noise, and sharpening."),
    _f("multiscale_residual_tail_mean", "multiscale_residual", "mean robust three-MAD tail fraction over Gaussian sigma 0.5, 1, 2, 4", "Natural edges and synthesized residuals may have different tail persistence across scale.", "Compression, noise, and sparse line art."),
    _f("multiscale_residual_kurtosis_spread", "multiscale_residual", "range of residual kurtosis over Gaussian sigma 0.5, 1, 2, 4", "Formation pipelines may differ in how residual tail shape changes with scale.", "Noise, low resolution, and clipping."),
    _f("multiscale_residual_crossscale_corr", "multiscale_residual", "mean correlation between adjacent-scale Gaussian residuals", "Decoder artifacts may remain unusually coherent across spatial scales.", "Blur, resizing, noise, and constant images."),
    _f("stego_residual_cooc_entropy", "steganalysis", "mean entropy of quantized co-occurrences from a compact directional high-pass bank", "Generator residual dependencies can differ from natural-image residual dependencies.", "JPEG alignment, noise, resolution, and strong scene edges."),
    _f("stego_residual_cooc_diagonal", "steganalysis", "mean equal-state mass of quantized directional residual co-occurrences", "Repeated residual states can reveal local synthesis or codec regularity.", "JPEG, denoising, noise, and flat images."),
    _f("stego_residual_directional_gap", "steganalysis", "horizontal/vertical gap in quantized high-pass co-occurrence concentration", "Anisotropic decoding or resizing can make residual dependence direction-specific.", "Oriented content, cropping, and codec grids."),
    _f("camera_cfa_periodicity_proxy", "camera_proxy", "normalized 2x2 phase contrast of high-pass opponent color", "Authentic camera pipelines can retain weak CFA/demosaicing periodicity absent from some generators.", "Non-camera authentic images, resizing, screenshots, denoising, and JPEG."),
    _f("camera_color_residual_coupling_proxy", "camera_proxy", "mean correlation between high-pass RGB channel residuals", "Demosaicing and camera processing can couple channel residuals differently from a generator decoder.", "Grayscale images, heavy processing, and synthetic camera-noise simulation."),
    _f("camera_signal_noise_fit_proxy", "camera_proxy", "fit quality of residual variance as a linear function of luminance", "Shot-noise-like signal dependence can support camera provenance.", "Non-camera authentic images, tone mapping, denoising, and JPEG."),
    _f("fft_one_over_f_residual_rmse", "fft_magnitude", "normalized RMSE around a fitted log radial 1/f power law", "Generators may deviate from natural-image scale regularity at selected radii.", "Scene content, resolution, windowing, and periodic real textures."),
    _f("fft_multiring_angular_entropy", "fft_magnitude", "mean normalized angular power entropy over low, mid, and high radial bands", "Architecture artifacts can create directionally concentrated energy at particular scales.", "Strong scene geometry, crops, and low texture."),
    _f("fft_cross_channel_coherence", "fft_magnitude", "mean magnitude-squared spectral coherence between RGB channel pairs", "Camera color formation and generator decoding may impose different cross-channel frequency coupling.", "Grayscale content, color conversion, and chroma subsampling."),
    _f("fft_phase_magnitude_coupling", "fft_phase", "absolute correlation between log magnitude and local phase-neighbor coherence", "Synthesis may couple spectral strength to phase organization differently from natural imaging.", "Translations, crops, blur, and low spectral energy."),
    _f("codec_grid_phase_contrast", "codec_resampling", "contrast of first-difference energy across the eight pixel-grid phases", "JPEG history or block-aligned processing can create an eight-pixel phase preference.", "Non-JPEG edges, crops that shift grid phase, and matched re-encoding."),
    _f("resampling_second_difference_periodicity", "codec_resampling", "lag-two autocorrelation of horizontal and vertical second differences", "Interpolation can leave periodic dependencies in discrete second derivatives.", "Natural periodic texture, noise, and unknown resize kernels."),
    _f("ycbcr_chroma_entropy", "chroma", "mean marginal entropy of Cb and Cr", "Synthetic color distributions may occupy chroma code space differently from real pipelines.", "Color jitter, grayscale content, and codec subsampling."),
    _f("ycbcr_chroma_joint_entropy", "chroma", "joint entropy of quantized Cb/Cr pairs", "Generator palettes and decoders may produce atypical joint chroma distributions.", "Semantic color mismatch, quantization, and color editing."),
    _f("ycbcr_chroma_residual_corr", "chroma", "correlation between high-pass Cb and Cr residuals", "Cross-chroma residual coupling may differ between camera and generator pipelines.", "Grayscale content, JPEG subsampling, resizing, and denoising."),
    _f("ycbcr_chroma_tail_fraction", "chroma", "fraction of chroma residual magnitudes above a robust three-MAD threshold", "Synthetic color microstructure may have atypical residual tails.", "Color noise, saturation edits, and compression."),
    _f("self_jpeg70_mse", "self_consistency", "MSE after a JPEG-Q70 probe", "Existing quantization/synthetic statistics change re-save sensitivity.", "Codec-specific and computationally repeated.", cost="medium"),
    _f("self_blur1_mse", "self_consistency", "MSE after a sigma-1 blur probe", "Fine-detail stability can differ by origin.", "Resolution and sharpness.", cost="medium"),
    _f("self_resize05_mse", "self_consistency", "MSE after a 0.5x rescale probe", "Generator textures may be less scale-consistent.", "Resize kernel and small inputs.", cost="medium"),
    _f("self_jpeg70_gradient_drop", "self_consistency", "relative Sobel-energy loss after JPEG-Q70", "Synthetic and natural edges respond differently to re-encoding.", "Codec history and flat images.", cost="medium"),
)


EXPANDED_V2_ONLY_NAMES: tuple[str, ...] = (
    "bitplane_low_occupancy",
    "bitplane_low_entropy",
    "bitplane_directional_transition",
    "bitplane_cross_channel_agreement",
    "bitplane_gradient_patch_max",
    "patch_residual_q90",
    "patch_residual_heterogeneity",
    "patch_gradient_q90",
    "patch_spectral_high_q90",
    "multiscale_residual_std_slope",
    "multiscale_residual_tail_mean",
    "multiscale_residual_kurtosis_spread",
    "multiscale_residual_crossscale_corr",
    "stego_residual_cooc_entropy",
    "stego_residual_cooc_diagonal",
    "stego_residual_directional_gap",
    "camera_cfa_periodicity_proxy",
    "camera_color_residual_coupling_proxy",
    "camera_signal_noise_fit_proxy",
    "fft_one_over_f_residual_rmse",
    "fft_multiring_angular_entropy",
    "fft_cross_channel_coherence",
    "fft_phase_magnitude_coupling",
    "codec_grid_phase_contrast",
    "resampling_second_difference_periodicity",
    "ycbcr_chroma_entropy",
    "ycbcr_chroma_joint_entropy",
    "ycbcr_chroma_residual_corr",
    "ycbcr_chroma_tail_fraction",
)

_EXPANDED_V2_ONLY_NAME_SET = frozenset(EXPANDED_V2_ONLY_NAMES)
FROZEN_V1_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = tuple(
    spec for spec in FEATURE_REGISTRY if spec.name not in _EXPANDED_V2_ONLY_NAME_SET
)
EXPANDED_V2_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = FEATURE_REGISTRY
FEATURE_PROFILES: dict[str, tuple[FeatureSpec, ...]] = {
    "frozen_v1": FROZEN_V1_FEATURE_REGISTRY,
    "expanded_v2": EXPANDED_V2_FEATURE_REGISTRY,
}
DEFAULT_FEATURE_PROFILE = "frozen_v1"

if len(FROZEN_V1_FEATURE_REGISTRY) != 53 or len(EXPANDED_V2_FEATURE_REGISTRY) != 82:
    raise RuntimeError("Feature-profile sizes changed; define a new versioned schema instead of mutating v1/v2.")


def get_feature_registry(profile: str) -> tuple[FeatureSpec, ...]:
    try:
        return FEATURE_PROFILES[profile]
    except KeyError as error:
        raise KeyError(f"Unknown feature profile {profile!r}; choose from {sorted(FEATURE_PROFILES)}") from error


def feature_schema_sha256(profile: str) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in get_feature_registry(profile)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def registry_frame(profile: str = "expanded_v2") -> pd.DataFrame:
    return pd.DataFrame(asdict(spec) for spec in get_feature_registry(profile))


def feature_names(*, role: str | None = None, profile: str = "expanded_v2") -> list[str]:
    return [
        spec.name
        for spec in get_feature_registry(profile)
        if role is None or spec.role == role
    ]
