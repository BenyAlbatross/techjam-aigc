"""All-train data contract and native-pixel crops for the three-branch model."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from techjam_aigc.trace_rx_m.data import load_training_manifest

from .config import DataConfig, PreprocessingConfig


def _resolve(path: str | Path, repo_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def parent_digest(parent_ids: list[str] | tuple[str, ...]) -> str:
    digest = sha256()
    for parent_id in sorted(map(str, parent_ids)):
        digest.update(parent_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_all_training_rows(config: DataConfig, repo_root: Path) -> pd.DataFrame:
    """Join source paths while proving that no train row was held out."""

    root = Path(repo_root).resolve()
    manifest_path = _resolve(config.manifest, root)
    labels_path = _resolve(config.labels, root)
    source_root = _resolve(config.source_root, root)
    complete = load_training_manifest(manifest_path)
    train = complete[complete["split"].eq("train")].copy()
    if len(train) != int(complete["split"].eq("train").sum()):  # pragma: no cover
        raise RuntimeError("Internal train-row selection mismatch.")
    if not config.use_all_training_pools:
        raise ValueError("All train-only pools are required for this run.")
    selected_pools = set(train["training_pool"].astype(str))
    expected_pools = set(complete.loc[complete["split"].eq("train"), "training_pool"].astype(str))
    if selected_pools != expected_pools:
        raise RuntimeError("A train-only pool was omitted.")

    labels = pd.read_csv(
        labels_path,
        usecols=["asset_id", "image_path", "label", "split"],
        low_memory=False,
    )
    if labels["asset_id"].duplicated().any():
        raise ValueError("techjam2026_v2 labels contain duplicate asset_id values.")
    source = labels.rename(columns={
        "asset_id": "parent_id",
        "image_path": "source_image_path",
        "label": "source_label",
        "split": "source_split",
    })
    train = train.merge(source, on="parent_id", how="left", validate="one_to_one")
    if train[["source_image_path", "source_label", "source_split"]].isna().any().any():
        raise ValueError("Every training manifest row must resolve to a source label row.")
    if not train["source_split"].eq("train").all():
        raise ValueError("A selected training row resolves outside the source train split.")
    expected_target = train["source_label"].map({"real": 0, "ai_full": 1})
    if expected_target.isna().any() or not expected_target.astype(int).eq(train["target"]).all():
        raise ValueError("Source labels and normalized manifest targets disagree.")

    train["source_path"] = train["source_image_path"].map(
        lambda value: str((source_root / str(value)).resolve())
    )
    missing = [path for path in train["source_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training source images are missing; first: {missing[0]}"
        )
    train["balance_group"] = np.where(
        train["target"].eq(0),
        train["authentic_subtype"].astype(str),
        train["generator_family"].astype(str),
    )
    train["group_weight"] = _group_balanced_weights(train)
    train["sample_weight"] = train["group_weight"] if config.group_balanced_loss else 1.0
    return train.reset_index(drop=True)


def _group_balanced_weights(frame: pd.DataFrame) -> pd.Series:
    """Equal total weight per class and per within-class provenance group."""

    keys = frame["target"].astype(str) + ":" + frame["balance_group"].astype(str)
    counts = keys.map(keys.value_counts()).astype(float)
    groups_per_class = keys.groupby(frame["target"]).transform("nunique").astype(float)
    raw = 1.0 / (counts * groups_per_class)
    return raw / raw.mean()


def _global_view(image: Image.Image, config: PreprocessingConfig) -> np.ndarray:
    prepared = image.convert("RGB")
    width, height = prepared.size
    short = min(width, height)
    if short > config.max_global_short_side:
        scale = config.max_global_short_side / short
        prepared = prepared.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
    array = np.asarray(prepared, dtype=np.uint8)
    size = config.global_image_size
    pad_h = max(0, size - array.shape[0])
    pad_w = max(0, size - array.shape[1])
    if pad_h or pad_w:
        before_h, before_w = pad_h // 2, pad_w // 2
        array = np.pad(
            array,
            (
                (before_h, pad_h - before_h),
                (before_w, pad_w - before_w),
                (0, 0),
            ),
            mode="constant",
        )
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    values = array[top : top + size, left : left + size].astype(np.float32) / 255.0
    values = (values - np.asarray(config.image_mean, dtype=np.float32)) / np.asarray(
        config.image_std, dtype=np.float32
    )
    return np.ascontiguousarray(values.transpose(2, 0, 1))


def _native_crops(image: Image.Image, config: PreprocessingConfig) -> np.ndarray:
    """Take source-pixel crops without resizing; edge-pad undersized inputs."""

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    size = config.native_crop_size
    pad_h = max(0, size - array.shape[0])
    pad_w = max(0, size - array.shape[1])
    if pad_h or pad_w:
        before_h, before_w = pad_h // 2, pad_w // 2
        array = np.pad(
            array,
            (
                (before_h, pad_h - before_h),
                (before_w, pad_w - before_w),
                (0, 0),
            ),
            mode="edge",
        )
    max_y = array.shape[0] - size
    max_x = array.shape[1] - size
    if config.native_crop_count == 1:
        positions = ((max_y // 2, max_x // 2),)
    else:
        positions = (
            (0, 0),
            (0, max_x),
            (max_y, 0),
            (max_y, max_x),
            (max_y // 2, max_x // 2),
        )
    crops = [
        np.ascontiguousarray(
            array[top : top + size, left : left + size].transpose(2, 0, 1),
            dtype=np.float32,
        )
        / 255.0
        for top, left in positions
    ]
    return np.stack(crops)


class ThreeBranchDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessing: PreprocessingConfig,
        *,
        include_native_crops: bool = True,
    ) -> None:
        if frame.empty or set(frame["split"]) != {"train"}:
            raise ValueError("ThreeBranchDataset requires non-empty train rows only.")
        self.frame = frame.reset_index(drop=True)
        self.preprocessing = preprocessing
        self.include_native_crops = include_native_crops

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(str(row["source_path"])) as opened:
            image = opened.convert("RGB")
            global_pixels = _global_view(image, self.preprocessing)
            native = (
                _native_crops(image, self.preprocessing)
                if self.include_native_crops
                else np.empty((0, 3, 0, 0), dtype=np.float32)
            )
        return {
            "global_pixels": global_pixels,
            "native_crops": native,
            "target": np.int64(row["target"]),
            "sample_weight": np.float32(row["sample_weight"]),
            "group_weight": np.float32(row["group_weight"]),
            "parent_id": str(row["parent_id"]),
            "generator_family": str(row["generator_family"]),
            "authentic_subtype": str(row["authentic_subtype"]),
            "balance_group": str(row["balance_group"]),
            "training_pool": str(row["training_pool"]),
        }
