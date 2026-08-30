"""Endpoint quality descriptors used by TRACE-RX-M reliability fitting.

The proposal fixes the four descriptor *families* but does not prescribe the
FFT cut-off or the exact JPEG blockiness estimator.  This implementation uses
a radial cut-off of 0.25 cycles/pixel and a normalized excess of discontinuity
at 8-pixel boundaries.  Both choices are deterministic and intentionally
isolated here so that they can be replaced after the required occupancy audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator

import numpy as np
from PIL import Image


_EPS = np.finfo(np.float64).eps
_GAUSSIAN_MAD = 0.6744897501960817
QUALITY_DIMENSION = 4


def _as_luma(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or numeric array to finite luma in ``[0, 1]``."""

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    else:
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError("image must have shape (H, W), (H, W, 1), or (H, W, C)")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("image dimensions must be non-empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("image array must be numeric")
        original_dtype = array.dtype
        array = array.astype(np.float64)
        if np.issubdtype(original_dtype, np.integer):
            array /= float(np.iinfo(original_dtype).max)
        elif array.size and (np.nanmin(array) < 0.0 or np.nanmax(array) > 1.0):
            # Float images in conventional byte range are accepted, but
            # arbitrary per-image normalization is deliberately avoided.
            if np.nanmin(array) >= 0.0 and np.nanmax(array) <= 255.0:
                array /= 255.0
            else:
                raise ValueError("floating-point image values must be in [0, 1] or [0, 255]")

    if not np.isfinite(array).all():
        raise ValueError("image contains non-finite values")
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        return array
    if array.shape[2] == 1:
        return array[..., 0]
    if array.shape[2] < 3:
        raise ValueError("multi-channel image must have either one or at least three channels")
    return array[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


def _haar_diagonal(luma: np.ndarray) -> np.ndarray:
    """Return first-level orthonormal Haar diagonal-detail coefficients."""

    height = 2 * (luma.shape[0] // 2)
    width = 2 * (luma.shape[1] // 2)
    if height < 2 or width < 2:
        return np.empty((0, 0), dtype=np.float64)
    a = luma[:height:2, :width:2]
    b = luma[:height:2, 1:width:2]
    c = luma[1:height:2, :width:2]
    d = luma[1:height:2, 1:width:2]
    return (a - b - c + d) / 2.0


def wavelet_mad_noise(image: Image.Image | np.ndarray) -> float:
    """Estimate additive Gaussian noise sigma with wavelet MAD.

    The median absolute deviation is taken from first-level diagonal Haar
    detail coefficients and divided by ``Phi^-1(0.75)``.  Centering by the
    coefficient median makes the estimate robust to a non-zero detail bias.
    """

    diagonal = _haar_diagonal(_as_luma(image))
    if diagonal.size == 0:
        return 0.0
    median = float(np.median(diagonal))
    mad = float(np.median(np.abs(diagonal - median)))
    return mad / _GAUSSIAN_MAD


def jpeg_blockiness(image: Image.Image | np.ndarray, block_size: int = 8) -> float:
    """Measure normalized discontinuity excess at JPEG block boundaries.

    The score is ``(boundary - interior) / (boundary + interior)`` and lies in
    ``[-1, 1]`` (zero for a constant image).  Comparing boundary differences
    to non-boundary neighbours avoids confusing ordinary image contrast with
    blockiness; a high score indicates an 8x8 lattice.
    """

    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    luma = _as_luma(image)
    differences: list[np.ndarray] = []
    boundary_masks: list[np.ndarray] = []
    if luma.shape[1] > 1:
        horizontal = np.abs(np.diff(luma, axis=1))
        positions = np.arange(1, luma.shape[1])
        differences.append(horizontal.ravel())
        boundary_masks.append(np.broadcast_to(positions % block_size == 0, horizontal.shape).ravel())
    if luma.shape[0] > 1:
        vertical = np.abs(np.diff(luma, axis=0))
        positions = np.arange(1, luma.shape[0])[:, None]
        differences.append(vertical.ravel())
        boundary_masks.append(np.broadcast_to(positions % block_size == 0, vertical.shape).ravel())
    if not differences:
        return 0.0
    values = np.concatenate(differences)
    is_boundary = np.concatenate(boundary_masks)
    if not np.any(is_boundary) or np.all(is_boundary):
        return 0.0
    boundary = float(np.mean(values[is_boundary]))
    interior = float(np.mean(values[~is_boundary]))
    denominator = boundary + interior
    if denominator <= _EPS:
        return 0.0
    return float(np.clip((boundary - interior) / denominator, -1.0, 1.0))


def high_frequency_energy(
    image: Image.Image | np.ndarray,
    *,
    cutoff: float = 0.25,
) -> tuple[float, float]:
    """Return raw HF energy and the fraction of Fourier bins in the HF band.

    FFT normalization and division by the total number of bins make raw energy
    a per-pixel mean-square contribution.  Consequently white noise with
    variance sigma squared contributes, in expectation,
    ``sigma**2 * high_frequency_bin_fraction`` exactly as assumed in the PDF.
    """

    if not 0.0 < cutoff <= np.sqrt(0.5):
        raise ValueError("cutoff must be in (0, sqrt(0.5)] cycles/pixel")
    luma = _as_luma(image)
    centered = luma - float(np.mean(luma))
    spectrum = np.fft.fft2(centered, norm="ortho")
    fy = np.fft.fftfreq(luma.shape[0])
    fx = np.fft.fftfreq(luma.shape[1])
    radius = np.hypot(fy[:, None], fx[None, :])
    high = radius >= cutoff
    bin_fraction = float(np.mean(high))
    energy = float(np.sum(np.abs(spectrum[high]) ** 2) / spectrum.size)
    return energy, bin_fraction


def structural_high_frequency_energy(
    image: Image.Image | np.ndarray,
    *,
    noise_sigma: float | None = None,
    cutoff: float = 0.25,
) -> float:
    """Compute the proposal's noise-floor-subtracted structural HF energy."""

    sigma = wavelet_mad_noise(image) if noise_sigma is None else float(noise_sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_sigma must be finite and non-negative")
    raw_energy, bin_fraction = high_frequency_energy(image, cutoff=cutoff)
    return max(0.0, raw_energy - sigma**2 * bin_fraction)


@dataclass(frozen=True)
class QualityDescriptor:
    """The ordered TRACE-RX-M quality vector ``q(x)``."""

    log_min_dimension: float
    noise_sigma: float
    blockiness: float
    structural_hf_energy: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.log_min_dimension,
                self.noise_sigma,
                self.blockiness,
                self.structural_hf_energy,
            ],
            dtype=np.float64,
        )


def extract_quality_descriptor(
    image: Image.Image | np.ndarray,
    *,
    hf_cutoff: float = 0.25,
) -> QualityDescriptor:
    """Extract ``[log min(H,W), sigma_hat, q_block, E_HF_struct]``."""

    luma = _as_luma(image)
    sigma = wavelet_mad_noise(luma)
    return QualityDescriptor(
        log_min_dimension=float(np.log(min(luma.shape))),
        noise_sigma=sigma,
        blockiness=jpeg_blockiness(luma),
        structural_hf_energy=structural_high_frequency_energy(
            luma, noise_sigma=sigma, cutoff=hf_cutoff
        ),
    )


def quality_vector(image: Image.Image | np.ndarray, *, hf_cutoff: float = 0.25) -> np.ndarray:
    """Convenience array form of :func:`extract_quality_descriptor`."""

    return extract_quality_descriptor(image, hf_cutoff=hf_cutoff).as_array()


Cell = tuple[int, ...]


class QuantileQualityBinner:
    """Deterministic independent-quantile binning for endpoint quality cells.

    Duplicate quantile edges are removed.  Therefore a constant descriptor has
    one bin instead of multiple arbitrary empty bins.  Values equal to an edge
    deterministically enter the bin to its right via ``searchsorted(...,
    side='right')``.  Fit this object on the dedicated authentic-null split and
    only call :meth:`transform` on development/inference descriptors.
    """

    def __init__(self, n_bins: int | Iterable[int] = 4) -> None:
        bins = np.asarray(
            [n_bins] * QUALITY_DIMENSION if np.isscalar(n_bins) else list(n_bins),
            dtype=np.int64,
        )
        if bins.shape != (QUALITY_DIMENSION,) or np.any(bins < 1):
            raise ValueError("n_bins must be a positive integer or four positive integers")
        self.requested_bins = tuple(int(value) for value in bins)
        self.edges_: tuple[np.ndarray, ...] | None = None

    @staticmethod
    def _validate(descriptors: np.ndarray) -> np.ndarray:
        values = np.asarray(descriptors, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != QUALITY_DIMENSION:
            raise ValueError(f"descriptors must have shape (N, {QUALITY_DIMENSION})")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("descriptors must be non-empty and finite")
        return values

    def fit(self, descriptors: np.ndarray) -> QuantileQualityBinner:
        values = self._validate(descriptors)
        edges: list[np.ndarray] = []
        for column, bins in enumerate(self.requested_bins):
            if bins == 1:
                edges.append(np.empty(0, dtype=np.float64))
                continue
            quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
            cuts = np.quantile(values[:, column], quantiles, method="linear")
            cuts = np.unique(np.asarray(cuts, dtype=np.float64))
            minimum = float(np.min(values[:, column]))
            maximum = float(np.max(values[:, column]))
            # A cut at either endpoint creates an empty cell. Constant
            # descriptors therefore collapse to one meaningful bin.
            edges.append(cuts[(cuts > minimum) & (cuts < maximum)])
        self.edges_ = tuple(edges)
        return self

    @property
    def bin_counts(self) -> tuple[int, ...]:
        if self.edges_ is None:
            raise RuntimeError("binner has not been fitted")
        return tuple(len(edges) + 1 for edges in self.edges_)

    def iter_cells(self) -> Iterator[Cell]:
        """Yield the complete Cartesian cell set in lexicographic order."""

        return product(*(range(count) for count in self.bin_counts))

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        values = self._validate(descriptors)
        if self.edges_ is None:
            raise RuntimeError("binner has not been fitted")
        return np.column_stack(
            [np.searchsorted(edges, values[:, column], side="right") for column, edges in enumerate(self.edges_)]
        ).astype(np.int64, copy=False)

    def transform_cells(self, descriptors: np.ndarray) -> list[Cell]:
        return [tuple(int(value) for value in row) for row in self.transform(descriptors)]

    def fit_transform(self, descriptors: np.ndarray) -> np.ndarray:
        return self.fit(descriptors).transform(descriptors)


    def to_dict(self) -> dict[str, object]:
        if self.edges_ is None:
            raise RuntimeError("binner has not been fitted")
        return {
            "requested_bins": list(self.requested_bins),
            "edges": [edge.tolist() for edge in self.edges_],
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> QuantileQualityBinner:
        binner = cls(values["requested_bins"])
        edges = values.get("edges")
        if not isinstance(edges, list) or len(edges) != QUALITY_DIMENSION:
            raise ValueError("Serialized quality binner needs four edge arrays.")
        binner.edges_ = tuple(np.asarray(edge, dtype=np.float64) for edge in edges)
        return binner
