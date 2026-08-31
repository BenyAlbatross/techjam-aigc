"""Classical, interpretable image features used by the laboratory."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn

from .registry import feature_names
from .transforms import apply_transform


_EPS = 1e-12


def _entropy(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> float:
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    probabilities = counts[counts > 0].astype(np.float64)
    if probabilities.size == 0:
        return 0.0
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _kurtosis(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    centered = flat - flat.mean()
    variance = np.mean(centered**2)
    if variance <= _EPS:
        return 0.0
    return float(np.mean(centered**4) / (variance**2) - 3.0)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.ravel(), right.ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _luma(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return rgb, rgb @ np.array([0.2126, 0.7152, 0.0722])


def _sobel(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = ndimage.sobel(luma, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(luma, axis=0, mode="reflect") / 8.0
    return gx, gy, np.hypot(gx, gy)


def _texture_features(luma: np.ndarray) -> dict[str, float]:
    if min(luma.shape) < 3:
        resized = np.asarray(Image.fromarray(np.uint8(luma * 255)).resize((8, 8)), dtype=np.float64) / 255.0
        luma = resized
    center = luma[1:-1, 1:-1]
    neighbors = (
        luma[:-2, :-2], luma[:-2, 1:-1], luma[:-2, 2:],
        luma[1:-1, 2:], luma[2:, 2:], luma[2:, 1:-1],
        luma[2:, :-2], luma[1:-1, :-2],
    )
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    transitions = np.zeros_like(codes, dtype=np.uint8)
    for bit in range(8):
        transitions += (((codes >> bit) & 1) != ((codes >> ((bit + 1) % 8)) & 1)).astype(np.uint8)

    quantized = np.minimum((luma * 16).astype(np.int16), 15)
    left = np.concatenate([quantized[:, :-1].ravel(), quantized[:-1, :].ravel()])
    right = np.concatenate([quantized[:, 1:].ravel(), quantized[1:, :].ravel()])
    matrix = np.zeros((16, 16), dtype=np.float64)
    np.add.at(matrix, (left, right), 1)
    np.add.at(matrix, (right, left), 1)
    matrix /= max(matrix.sum(), 1.0)
    i, j = np.indices(matrix.shape)
    return {
        "lbp_entropy": _entropy(codes, 256, (0, 256)),
        "lbp_uniform_fraction": float(np.mean(transitions <= 2)),
        "glcm_contrast": float(np.sum(matrix * (i - j) ** 2)),
        "glcm_homogeneity": float(np.sum(matrix / (1.0 + np.abs(i - j)))),
    }


def _fft_features(luma: np.ndarray) -> dict[str, float]:
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((luma - luma.mean()) * window))
    power = np.abs(spectrum) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    grid_x, grid_y = np.meshgrid(fx, fy)
    radius = np.hypot(grid_x, grid_y)
    non_dc = radius > max(1.0 / max(height, width), _EPS)
    total = float(power[non_dc].sum()) + _EPS

    radial_edges = np.linspace(0.01, min(0.70, radius.max()), 24)
    radial_centers, radial_power = [], []
    for low, high in zip(radial_edges[:-1], radial_edges[1:]):
        mask = (radius >= low) & (radius < high)
        if np.any(mask):
            radial_centers.append((low + high) / 2.0)
            radial_power.append(float(power[mask].mean()) + _EPS)
    slope = 0.0
    if len(radial_centers) >= 3:
        slope = float(np.polyfit(np.log(radial_centers), np.log(radial_power), 1)[0])

    probabilities = power[non_dc] / total
    spectral_entropy = float(-(probabilities * np.log2(probabilities + _EPS)).sum() / np.log2(len(probabilities)))
    horizontal = (np.abs(grid_y) < 0.035) & (np.abs(grid_x) > 0.08)
    vertical = (np.abs(grid_x) < 0.035) & (np.abs(grid_y) > 0.08)
    horizontal_energy, vertical_energy = float(power[horizontal].sum()), float(power[vertical].sum())

    phase = np.angle(spectrum)[non_dc]
    phase_grid = np.angle(spectrum)
    phase_coherence = 0.5 * (
        np.mean(np.cos(np.diff(phase_grid, axis=0))) + np.mean(np.cos(np.diff(phase_grid, axis=1)))
    )
    return {
        "fft_low_energy": float(power[(radius > 0) & (radius < 0.10)].sum() / total),
        "fft_mid_energy": float(power[(radius >= 0.10) & (radius < 0.25)].sum() / total),
        "fft_high_energy": float(power[radius >= 0.25].sum() / total),
        "fft_radial_slope": slope,
        "fft_spectral_entropy": spectral_entropy,
        "fft_anisotropy": abs(horizontal_energy - vertical_energy) / (horizontal_energy + vertical_energy + _EPS),
        "fft_peak_ratio": float(np.max(power[non_dc]) / (np.mean(power[non_dc]) + _EPS)),
        "phase_resultant": float(abs(np.mean(np.exp(1j * phase)))),
        "phase_entropy": _entropy(phase, 18, (-np.pi, np.pi)),
        "phase_neighbor_coherence": float(phase_coherence),
    }


def _dct_features(luma: np.ndarray) -> dict[str, float]:
    if min(luma.shape) < 8:
        luma = np.asarray(Image.fromarray(np.uint8(luma * 255)).resize((8, 8)), dtype=np.float64) / 255.0
    height, width = (luma.shape[0] // 8) * 8, (luma.shape[1] // 8) * 8
    cropped = luma[:height, :width] * 255.0
    blocks = cropped.reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3)
    coefficients = dctn(blocks, axes=(-2, -1), norm="ortho")
    ac_mask = np.ones((8, 8), dtype=bool)
    ac_mask[0, 0] = False
    u, v = np.indices((8, 8))
    high_mask = (u + v >= 8) & ac_mask
    ac = coefficients[..., ac_mask]
    total_ac = float(np.sum(ac**2)) + _EPS

    vertical_boundaries = np.arange(8, width, 8)
    horizontal_boundaries = np.arange(8, height, 8)
    boundary_values = []
    if len(vertical_boundaries):
        boundary_values.append(np.abs(cropped[:, vertical_boundaries] - cropped[:, vertical_boundaries - 1]).ravel())
    if len(horizontal_boundaries):
        boundary_values.append(np.abs(cropped[horizontal_boundaries, :] - cropped[horizontal_boundaries - 1, :]).ravel())
    boundary_mean = float(np.mean(np.concatenate(boundary_values))) if boundary_values else 0.0
    ordinary_mean = 0.5 * (float(np.mean(np.abs(np.diff(cropped, axis=0)))) + float(np.mean(np.abs(np.diff(cropped, axis=1)))))
    return {
        "dct_high_ac_ratio": float(np.sum(coefficients[..., high_mask] ** 2) / total_ac),
        "dct_ac_kurtosis": _kurtosis(ac),
        "dct_near_zero_fraction": float(np.mean(np.abs(ac) < 0.5)),
        "jpeg_blockiness": boundary_mean / (ordinary_mean + _EPS),
    }


def _haar(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = (luma.shape[0] // 2) * 2, (luma.shape[1] // 2) * 2
    if height < 2 or width < 2:
        luma = np.asarray(Image.fromarray(np.uint8(luma * 255)).resize((8, 8)), dtype=np.float64) / 255.0
        height, width = 8, 8
    a, b = luma[:height:2, :width:2], luma[:height:2, 1:width:2]
    c, d = luma[1:height:2, :width:2], luma[1:height:2, 1:width:2]
    return (a + b + c + d) / 2.0, (a - b + c - d) / 2.0, (a + b - c - d) / 2.0, (a - b - c + d) / 2.0


def _wavelet_features(luma: np.ndarray) -> dict[str, float]:
    ll, lh, hl, hh = _haar(luma)
    energies = np.array([np.mean(lh**2), np.mean(hl**2), np.mean(hh**2)])
    total = float(np.mean(ll**2) + energies.sum()) + _EPS
    level2 = _haar(ll)
    level2_high = sum(float(np.mean(band**2)) for band in level2[1:])
    first_high = float(energies.sum()) + _EPS
    return {
        "wavelet_hf_energy": float(energies.sum() / total),
        "wavelet_hf_kurtosis": _kurtosis(np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()])),
        "wavelet_orientation_imbalance": float((energies.max() - energies.min()) / (energies.sum() + _EPS)),
        "wavelet_level2_ratio": level2_high / first_high,
    }


def _gradient_features(luma: np.ndarray) -> dict[str, float]:
    gx, gy, magnitude = _sobel(luma)
    angles = np.arctan2(gy, gx)
    bins = np.linspace(-np.pi, np.pi, 19)
    weighted, _ = np.histogram(angles, bins=bins, weights=magnitude)
    weighted = weighted / (weighted.sum() + _EPS)
    orientation_entropy = float(-(weighted[weighted > 0] * np.log2(weighted[weighted > 0])).sum())
    laplacian = ndimage.laplace(luma, mode="reflect")
    return {
        "gradient_energy": float(np.mean(magnitude)),
        "gradient_orientation_entropy": orientation_entropy,
        "laplacian_std": float(np.std(laplacian)),
        "edge_density": float(np.mean(magnitude > (np.median(magnitude) + np.std(magnitude)))),
    }


def _robust_tail_fraction(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.median(values)
    mad = float(np.median(np.abs(centered)))
    if mad <= _EPS:
        return 0.0
    return float(np.mean(np.abs(centered) > 3.0 * 1.4826 * mad))


def _array_patches(array: np.ndarray, grid: int = 4) -> list[np.ndarray]:
    row_groups = np.array_split(np.arange(array.shape[0]), min(grid, array.shape[0]))
    col_groups = np.array_split(np.arange(array.shape[1]), min(grid, array.shape[1]))
    return [array[np.ix_(rows, cols)] for rows in row_groups for cols in col_groups if rows.size and cols.size]


def _bitplane_features(rgb: np.ndarray) -> dict[str, float]:
    samples = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    bit_indices = np.arange(4, dtype=np.uint8)
    planes = ((samples[..., None] >> bit_indices) & 1).astype(np.float64)
    occupancy = planes.mean(axis=(0, 1))
    entropy = -(occupancy * np.log2(occupancy + _EPS) + (1.0 - occupancy) * np.log2(1.0 - occupancy + _EPS))

    horizontal = np.mean(planes[:, 1:, ...] != planes[:, :-1, ...], axis=(0, 1)) if planes.shape[1] > 1 else np.zeros((3, 4))
    vertical = np.mean(planes[1:, ...] != planes[:-1, ...], axis=(0, 1)) if planes.shape[0] > 1 else np.zeros((3, 4))
    agreements = [
        np.mean(planes[..., 0, :] == planes[..., 1, :]),
        np.mean(planes[..., 0, :] == planes[..., 2, :]),
        np.mean(planes[..., 1, :] == planes[..., 2, :]),
    ]

    transition_map = np.zeros(planes.shape[:2], dtype=np.float64)
    if planes.shape[1] > 1:
        transition_map[:, 1:] += np.mean(planes[:, 1:, ...] != planes[:, :-1, ...], axis=(2, 3))
    if planes.shape[0] > 1:
        transition_map[1:, :] += np.mean(planes[1:, ...] != planes[:-1, ...], axis=(2, 3))
    patch_max = max(float(np.mean(patch)) for patch in _array_patches(transition_map))
    return {
        "bitplane_low_occupancy": float(np.mean(occupancy)),
        "bitplane_low_entropy": float(np.mean(entropy)),
        "bitplane_directional_transition": float(np.mean(np.abs(horizontal - vertical) / (horizontal + vertical + _EPS))),
        "bitplane_cross_channel_agreement": float(np.mean(agreements)),
        "bitplane_gradient_patch_max": patch_max,
    }


def _patch_distribution_features(luma: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    residual_values, gradient_values, spectral_values = [], [], []
    for patch, residual_patch in zip(_array_patches(luma), _array_patches(residual)):
        residual_values.append(float(np.median(np.abs(residual_patch - np.median(residual_patch)))))
        gradient_values.append(float(np.mean(_sobel(patch)[2])))

        centered = patch - np.mean(patch)
        window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
        power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
        fy = np.fft.fftshift(np.fft.fftfreq(patch.shape[0]))
        fx = np.fft.fftshift(np.fft.fftfreq(patch.shape[1]))
        radius = np.hypot(*np.meshgrid(fx, fy))
        spectral_values.append(float(power[radius >= 0.25].sum() / (power[radius > 0].sum() + _EPS)))

    residual_array = np.asarray(residual_values)
    return {
        "patch_residual_q90": float(np.quantile(residual_array, 0.90)),
        "patch_residual_heterogeneity": float(
            (np.quantile(residual_array, 0.75) - np.quantile(residual_array, 0.25))
            / (np.median(residual_array) + _EPS)
        ),
        "patch_gradient_q90": float(np.quantile(gradient_values, 0.90)),
        "patch_spectral_high_q90": float(np.quantile(spectral_values, 0.90)),
    }


def _multiscale_residual_features(luma: np.ndarray) -> dict[str, float]:
    sigmas = np.array([0.5, 1.0, 2.0, 4.0])
    residuals = [luma - ndimage.gaussian_filter(luma, sigma=sigma, mode="reflect") for sigma in sigmas]
    standard_deviations = np.array([np.std(item) for item in residuals])
    kurtoses = np.array([_kurtosis(item) for item in residuals])
    adjacent_correlations = [_correlation(left, right) for left, right in zip(residuals[:-1], residuals[1:])]
    return {
        "multiscale_residual_std_slope": float(np.polyfit(np.log(sigmas), np.log(standard_deviations + _EPS), 1)[0]),
        "multiscale_residual_tail_mean": float(np.mean([_robust_tail_fraction(item) for item in residuals])),
        "multiscale_residual_kurtosis_spread": float(np.ptp(kurtoses)),
        "multiscale_residual_crossscale_corr": float(np.mean(adjacent_correlations)),
    }


def _cooccurrence_summary(residual: np.ndarray, axis: int) -> tuple[float, float]:
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    quantized = np.clip(np.rint(residual / (scale + _EPS)), -2, 2).astype(np.int8) + 2
    if axis == 1:
        left, right = quantized[:, :-1].ravel(), quantized[:, 1:].ravel()
    else:
        left, right = quantized[:-1, :].ravel(), quantized[1:, :].ravel()
    matrix = np.zeros((5, 5), dtype=np.float64)
    if left.size:
        np.add.at(matrix, (left, right), 1.0)
    matrix /= matrix.sum() + _EPS
    populated = matrix[matrix > 0]
    entropy = float(-(populated * np.log2(populated)).sum() / np.log2(25.0)) if populated.size else 0.0
    return entropy, float(np.trace(matrix))


def _steganalysis_features(luma: np.ndarray) -> dict[str, float]:
    horizontal_kernel = np.array([[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]])
    vertical_kernel = horizontal_kernel.T
    horizontal = ndimage.convolve(luma, horizontal_kernel, mode="reflect")
    vertical = ndimage.convolve(luma, vertical_kernel, mode="reflect")
    horizontal_entropy, horizontal_diagonal = _cooccurrence_summary(horizontal, axis=1)
    vertical_entropy, vertical_diagonal = _cooccurrence_summary(vertical, axis=0)
    return {
        "stego_residual_cooc_entropy": 0.5 * (horizontal_entropy + vertical_entropy),
        "stego_residual_cooc_diagonal": 0.5 * (horizontal_diagonal + vertical_diagonal),
        "stego_residual_directional_gap": abs(horizontal_diagonal - vertical_diagonal),
    }


def _camera_proxy_features(rgb: np.ndarray, luma: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    channel_residuals = np.stack(
        [rgb[..., channel] - ndimage.gaussian_filter(rgb[..., channel], sigma=1.0, mode="reflect") for channel in range(3)],
        axis=2,
    )
    opponent_highpass = np.abs(channel_residuals[..., 0] - channel_residuals[..., 1]) + np.abs(
        channel_residuals[..., 2] - channel_residuals[..., 1]
    )
    phase_means = np.array(
        [np.mean(opponent_highpass[row_phase::2, col_phase::2]) for row_phase in range(2) for col_phase in range(2)]
    )
    coupling = np.mean(
        [
            _correlation(channel_residuals[..., 0], channel_residuals[..., 1]),
            _correlation(channel_residuals[..., 0], channel_residuals[..., 2]),
            _correlation(channel_residuals[..., 1], channel_residuals[..., 2]),
        ]
    )

    centers, variances = [], []
    for low, high in zip(np.linspace(0.0, 1.0, 9)[:-1], np.linspace(0.0, 1.0, 9)[1:]):
        mask = (luma >= low) & (luma < high if high < 1.0 else luma <= high)
        if np.count_nonzero(mask) >= 8:
            centers.append((low + high) / 2.0)
            variances.append(float(np.var(residual[mask])))
    fit_quality = 0.0
    if len(centers) >= 3 and np.var(variances) > _EPS:
        prediction = np.polyval(np.polyfit(centers, variances, 1), centers)
        variance_array = np.asarray(variances)
        fit_quality = max(0.0, 1.0 - float(np.sum((variance_array - prediction) ** 2) / (np.sum((variance_array - np.mean(variance_array)) ** 2) + _EPS)))
    return {
        "camera_cfa_periodicity_proxy": float(np.ptp(phase_means) / (np.mean(phase_means) + _EPS)),
        "camera_color_residual_coupling_proxy": float(coupling),
        "camera_signal_noise_fit_proxy": fit_quality,
    }


def _extended_spectrum_features(rgb: np.ndarray, luma: np.ndarray) -> dict[str, float]:
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((luma - luma.mean()) * window))
    power = np.abs(spectrum) ** 2
    fy, fx = np.fft.fftshift(np.fft.fftfreq(height)), np.fft.fftshift(np.fft.fftfreq(width))
    grid_x, grid_y = np.meshgrid(fx, fy)
    radius, angle = np.hypot(grid_x, grid_y), np.arctan2(grid_y, grid_x)

    radial_centers, radial_power = [], []
    for low, high in zip(np.linspace(0.02, min(0.70, radius.max()), 20)[:-1], np.linspace(0.02, min(0.70, radius.max()), 20)[1:]):
        mask = (radius >= low) & (radius < high)
        if np.any(mask):
            radial_centers.append((low + high) / 2.0)
            radial_power.append(float(np.mean(power[mask])) + _EPS)
    log_power = np.log(radial_power)
    prediction = np.polyval(np.polyfit(np.log(radial_centers), log_power, 1), np.log(radial_centers))
    one_over_f_rmse = float(np.sqrt(np.mean((log_power - prediction) ** 2)) / (np.std(log_power) + _EPS))

    angular_entropies = []
    angle_edges = np.linspace(-np.pi, np.pi, 19)
    for low, high in ((0.03, 0.12), (0.12, 0.25), (0.25, 0.50)):
        mask = (radius >= low) & (radius < high)
        weighted, _ = np.histogram(angle[mask], bins=angle_edges, weights=power[mask])
        probabilities = weighted / (weighted.sum() + _EPS)
        populated = probabilities[probabilities > 0]
        angular_entropies.append(float(-(populated * np.log2(populated)).sum() / np.log2(18.0)))

    channel_spectra = [
        np.fft.fftshift(np.fft.fft2((rgb[..., channel] - rgb[..., channel].mean()) * window)) for channel in range(3)
    ]
    spectral_coherence = []
    non_dc = radius > 1.0 / max(height, width)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        numerator = np.sum(np.abs(channel_spectra[left][non_dc] * np.conj(channel_spectra[right][non_dc])))
        denominator = np.sqrt(
            np.sum(np.abs(channel_spectra[left][non_dc]) ** 2) * np.sum(np.abs(channel_spectra[right][non_dc]) ** 2)
        )
        spectral_coherence.append(float(numerator / (denominator + _EPS)))

    phase = np.angle(spectrum)
    local_phase_coherence = 0.5 * (np.cos(phase - np.roll(phase, 1, axis=0)) + np.cos(phase - np.roll(phase, 1, axis=1)))
    phase_magnitude_coupling = abs(_correlation(np.log1p(power[non_dc]), local_phase_coherence[non_dc]))
    return {
        "fft_one_over_f_residual_rmse": one_over_f_rmse,
        "fft_multiring_angular_entropy": float(np.mean(angular_entropies)),
        "fft_cross_channel_coherence": float(np.mean(spectral_coherence)),
        "fft_phase_magnitude_coupling": float(phase_magnitude_coupling),
    }


def _codec_resampling_features(luma: np.ndarray) -> dict[str, float]:
    horizontal_difference = np.abs(np.diff(luma, axis=1))
    vertical_difference = np.abs(np.diff(luma, axis=0))
    phase_energies = [
        np.mean(horizontal_difference[:, phase::8]) for phase in range(8) if horizontal_difference[:, phase::8].size
    ] + [np.mean(vertical_difference[phase::8, :]) for phase in range(8) if vertical_difference[phase::8, :].size]
    second_horizontal = np.diff(luma, n=2, axis=1)
    second_vertical = np.diff(luma, n=2, axis=0)
    periodicities = [
        _correlation(second_horizontal[:, :-2], second_horizontal[:, 2:]),
        _correlation(second_vertical[:-2, :], second_vertical[2:, :]),
    ]
    return {
        "codec_grid_phase_contrast": float(np.ptp(phase_energies) / (np.mean(phase_energies) + _EPS)),
        "resampling_second_difference_periodicity": float(np.mean(periodicities)),
    }


def _chroma_features(rgb: np.ndarray, luma: np.ndarray) -> dict[str, float]:
    cb = (rgb[..., 2] - luma) / 1.8556
    cr = (rgb[..., 0] - luma) / 1.5748
    cb_residual = cb - ndimage.gaussian_filter(cb, sigma=1.0, mode="reflect")
    cr_residual = cr - ndimage.gaussian_filter(cr, sigma=1.0, mode="reflect")
    joint, _, _ = np.histogram2d(cb.ravel(), cr.ravel(), bins=16, range=((-0.5, 0.5), (-0.5, 0.5)))
    probabilities = joint[joint > 0] / max(joint.sum(), 1.0)
    magnitude = np.hypot(cb_residual, cr_residual)
    return {
        "ycbcr_chroma_entropy": 0.5 * (_entropy(cb, 32, (-0.5, 0.5)) + _entropy(cr, 32, (-0.5, 0.5))),
        "ycbcr_chroma_joint_entropy": float(-(probabilities * np.log2(probabilities)).sum()),
        "ycbcr_chroma_residual_corr": _correlation(cb_residual, cr_residual),
        "ycbcr_chroma_tail_fraction": _robust_tail_fraction(magnitude),
    }



def _self_consistency(image: Image.Image, luma: np.ndarray, gradient_energy: float) -> dict[str, float]:
    probes = {
        "jpeg": apply_transform(image, "jpeg_q70", parent_id="self-probe"),
        "blur": apply_transform(image, "blur_sigma1", parent_id="self-probe"),
        "resize": apply_transform(image, "resize_0.5", parent_id="self-probe"),
    }
    probe_luma = {name: _luma(probe)[1] for name, probe in probes.items()}
    jpeg_gradient = _gradient_features(probe_luma["jpeg"])["gradient_energy"]
    return {
        "self_jpeg70_mse": float(np.mean((luma - probe_luma["jpeg"]) ** 2)),
        "self_blur1_mse": float(np.mean((luma - probe_luma["blur"]) ** 2)),
        "self_resize05_mse": float(np.mean((luma - probe_luma["resize"]) ** 2)),
        "self_jpeg70_gradient_drop": float((gradient_energy - jpeg_gradient) / (gradient_energy + _EPS)),
    }


def extract_features(
    image: Image.Image,
    metadata: Mapping[str, object] | None = None,
    *,
    profile: str = "expanded_v2",
) -> dict[str, float]:

    metadata = metadata or {}
    computation_image = image.convert("RGB")
    if min(computation_image.size) < 16:
        computation_image = computation_image.resize(
            (max(computation_image.width, 16), max(computation_image.height, 16)),
            Image.Resampling.BILINEAR,
        )
    rgb, luma = _luma(computation_image)
    native_width = float(metadata.get("width", image.width))
    native_height = float(metadata.get("height", image.height))
    native_pixels = max(native_width * native_height, 1.0)
    encoded_bytes = float(metadata.get("bytes", 0.0))
    file_format = str(metadata.get("format", "")).upper()

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > _EPS)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.hypot(np.std(rg), np.std(yb)) + 0.3 * np.hypot(np.mean(rg), np.mean(yb))
    channel_correlations = [
        _correlation(rgb[..., 0], rgb[..., 1]),
        _correlation(rgb[..., 0], rgb[..., 2]),
        _correlation(rgb[..., 1], rgb[..., 2]),
    ]

    smooth = ndimage.gaussian_filter(luma, sigma=1.0, mode="reflect")
    residual = luma - smooth
    gx, gy, magnitude = _sobel(luma)
    residual_correlations = [_correlation(residual[:, :-1], residual[:, 1:]), _correlation(residual[:-1, :], residual[1:, :])]
    flat_mask = magnitude <= np.quantile(magnitude, 0.25)
    bins = np.quantile(luma, [0.0, 0.25, 0.5, 0.75, 1.0])
    centers, scales = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (luma >= low) & (luma <= high)
        if np.count_nonzero(mask) >= 8:
            centers.append((low + high) / 2.0)
            scales.append(float(np.std(residual[mask])))
    intensity_slope = float(np.polyfit(centers, scales, 1)[0]) if len(set(centers)) >= 2 else 0.0

    gradient = _gradient_features(luma)
    features = {
        "meta_log_pixels": float(np.log(native_pixels)),
        "meta_aspect_ratio": native_width / max(native_height, 1.0),
        "meta_bytes_per_pixel": encoded_bytes / native_pixels,
        "meta_is_jpeg": float(file_format in {"JPEG", "JPG"}),
        "meta_is_png": float(file_format == "PNG"),
        "luma_mean": float(np.mean(luma)),
        "luma_std": float(np.std(luma)),
        "luma_entropy": _entropy(luma, 64, (0.0, 1.0)),
        "luma_dynamic_range": float(np.quantile(luma, 0.95) - np.quantile(luma, 0.05)),
        "local_contrast_mean": float(np.mean(np.abs(luma - ndimage.uniform_filter(luma, size=9, mode="reflect")))),
        "clipped_fraction": float(np.mean((rgb <= 1.0 / 255.0) | (rgb >= 254.0 / 255.0))),
        "saturation_mean": float(np.mean(saturation)),
        "saturation_std": float(np.std(saturation)),
        "colorfulness": float(colorfulness),
        "chroma_energy": float(np.mean(rg**2 + yb**2)),
        "channel_correlation": float(np.mean(channel_correlations)),
        "residual_std": float(np.std(residual)),
        "residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
        "residual_kurtosis": _kurtosis(residual),
        "residual_neighbor_corr": float(np.mean(residual_correlations)),
        "noise_laplacian_mad": float(np.median(np.abs(ndimage.laplace(luma, mode="reflect")))),
        "noise_flat_region_std": float(np.std(residual[flat_mask])) if np.any(flat_mask) else 0.0,
        "noise_intensity_slope": intensity_slope,
        **_texture_features(luma),
        **_fft_features(luma),
        **_dct_features(luma),
        **_wavelet_features(luma),
        **gradient,
        **_self_consistency(computation_image, luma, gradient["gradient_energy"]),
    }
    expected = set(feature_names(profile=profile))
    if profile == "expanded_v2":
        features.update(
            _bitplane_features(rgb)
            | _patch_distribution_features(luma, residual)
            | _multiscale_residual_features(luma)
            | _steganalysis_features(luma)
            | _camera_proxy_features(rgb, luma, residual)
            | _extended_spectrum_features(rgb, luma)
            | _codec_resampling_features(luma)
            | _chroma_features(rgb, luma)
        )

    if set(features) != expected:
        raise RuntimeError(f"Feature registry mismatch: missing={expected - set(features)}, extra={set(features) - expected}")
    return {
        name: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        for name, value in features.items()
    }
