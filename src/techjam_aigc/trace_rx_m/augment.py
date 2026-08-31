"""Deterministic symmetric degradation and canonical image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PIL import Image

from techjam_aigc.feature_lab.transforms import (
    TRANSFORM_SPECS,
    apply_transform,
)


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


def canonical_preprocess(
    image: Image.Image,
    *,
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """DINOv3 square bilinear resize and ImageNet CHW normalization.

    This mirrors the checkpoint processor metadata without importing
    torchvision. Backbone-specific values remain explicit arguments.
    """

    if image_size < 1:
        raise ValueError("image_size must be positive.")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean/std must contain three channels and std must be positive.")
    rgb = image.convert("RGB")
    resized = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1), dtype=np.float32)
