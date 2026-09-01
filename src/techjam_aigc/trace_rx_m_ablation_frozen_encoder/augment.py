"""Deterministic symmetric degradation and canonical image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PIL import Image
from torchvision.transforms.functional import center_crop

from techjam_aigc.feature_lab.transforms import (
    TRANSFORM_SPECS,
    apply_transform,
)

from .config import PreprocessingConfig


OFFICIAL_DEGRADATION_FAMILIES = {
    "jpeg",
    "gaussian_blur",
    "resize",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_PREPROCESS_SHORT_SIDE = 512


@dataclass(frozen=True)
class SymmetricTransformSampler:
    """Sample the same condition distribution for every label.

    Selection depends only on seed, parent identity, and epoch.  A complete
    transform family can be excluded for the proposal's held-out-family gate.
    """

    held_out_family: str
    base_seed: int = 20260830
    clean_probability: float = 0.20

    def __post_init__(self) -> None:
        if self.held_out_family not in OFFICIAL_DEGRADATION_FAMILIES:
            raise ValueError(
                "held_out_family must be one official degradation family: "
                f"{sorted(OFFICIAL_DEGRADATION_FAMILIES)}"
            )
        if not 0.0 <= self.clean_probability < 1.0:
            raise ValueError("clean_probability must lie in [0, 1).")
        conditions = [
            spec.name
            for spec in TRANSFORM_SPECS
            if spec.official
            and spec.family != self.held_out_family
            and spec.family != "clean"
        ]
        if not conditions:
            raise ValueError("At least one training condition is required.")
        object.__setattr__(self, "conditions", tuple(conditions))

    def sample_condition(self, *, parent_id: str, epoch: int) -> str:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        payload = f"{self.base_seed}:{epoch}:{parent_id}:trace-rx-m-transform".encode()
        digest = sha256(payload).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        if draw < self.clean_probability:
            return "clean"
        condition_draw = int.from_bytes(digest[8:16], "big")
        return self.conditions[condition_draw % len(self.conditions)]

    def apply(
        self,
        image: Image.Image,
        *,
        parent_id: str,
        epoch: int,
    ) -> tuple[str, Image.Image]:
        condition = self.sample_condition(parent_id=parent_id, epoch=epoch)
        transformed = apply_transform(
            image,
            condition,
            parent_id=parent_id,
            base_seed=self.base_seed + epoch,
        )
        return condition, transformed


def canonical_center_crop(
    image: Image.Image,
    *,
    image_size: int = 224,
    max_short_side: int = MAX_PREPROCESS_SHORT_SIDE,
) -> Image.Image:
    """Prepare a centered square view without upscaling small images.

    Images whose short side exceeds ``max_short_side`` are first resized to
    that limit with aspect ratio preserved and bicubic interpolation.
    ``torchvision`` center-crop supplies symmetric zero-padding whenever an
    input dimension is smaller than the requested crop.
    """

    if image_size < 1:
        raise ValueError("image_size must be positive.")
    if max_short_side < image_size:
        raise ValueError("max_short_side must be at least image_size.")
    prepared = image.convert("RGB")
    width, height = prepared.size
    short_side = min(width, height)
    if short_side > max_short_side:
        if width <= height:
            resized_size = (
                max_short_side,
                max(1, round(height * max_short_side / width)),
            )
        else:
            resized_size = (
                max(1, round(width * max_short_side / height)),
                max_short_side,
            )
        prepared = prepared.resize(resized_size, Image.Resampling.BICUBIC)
    return center_crop(prepared, [image_size, image_size])


def canonical_preprocess(
    image: Image.Image,
    *,
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    preprocessing: PreprocessingConfig | None = None,
) -> np.ndarray:
    """DINOv3 center crop and ImageNet CHW normalization.

    The canonical view is 224px by default. Images larger than 512px on their
    short side are aspect-preservingly downscaled before cropping; undersized
    dimensions are zero-padded by ``torchvision.center_crop``.
    """

    if preprocessing is not None:
        preprocessing.validate()
        image_size = preprocessing.image_size
        mean = preprocessing.image_mean
        std = preprocessing.image_std
        max_short_side = preprocessing.max_short_side
    else:
        max_short_side = MAX_PREPROCESS_SHORT_SIDE
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean/std must contain three channels and std must be positive.")
    cropped = canonical_center_crop(
        image,
        image_size=image_size,
        max_short_side=max_short_side,
    )
    values = np.asarray(cropped, dtype=np.float32) / 255.0
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1), dtype=np.float32)
