"""Torch datasets over the canonical images and the frozen evaluation shards.

Two backends, matching the two storage policies in acai.build_transformed:

* `CanonicalDataset` reads canonical PNGs and applies a transform chain on the fly. Used for
  training, where the composition study needs to resample the mixture freely without
  rebuilding 105,000 files per mixture. Deterministic per (asset_id, chain), so a run is still
  exactly reproducible from its manifest.
* `FrozenEvalDataset` reads the frozen parquet shards. Used for every evaluation, so the sealed
  scorecard always scores the same bytes.

Both return the same tuple, so the training loop never branches on which one it has.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from acai.features.lowlevel import feature_maps, feature_vector
from acai.models.backbone import MEAN, STD
from acai.transforms import Chain, apply_chain


def _normalise(img: np.ndarray, size: int) -> torch.Tensor:
    """uint8 HWC -> normalised CHW float tensor at `size`."""
    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BICUBIC))
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class _Base(Dataset):
    """Shared feature computation and tensor packing.

    Features are computed on the **post-transform, pre-resize** image: that is the forensic
    content the deployed detector would actually receive, and computing them after the model's
    own downscale would measure the resize instead.
    """

    def __init__(self, img_size: int = 224, want_feats: bool = False,
                 want_maps: bool = False, patch: int = 16):
        self.img_size, self.want_feats, self.want_maps, self.patch = (
            img_size, want_feats, want_maps, patch)

    def _pack(self, img: np.ndarray, row: dict) -> dict:
        out = {"image": _normalise(img, self.img_size),
               "label": torch.tensor(int(row["label"])), "id": row["id"]}
        if self.want_feats:
            out["feats"] = torch.from_numpy(feature_vector(img))
        if self.want_maps:
            # Maps must be on the model's patch grid, so they are computed on the *resized*
            # image -- the grid is defined by img_size // patch, not by the canonical size.
            small = np.asarray(Image.fromarray(img).resize(
                (self.img_size, self.img_size), Image.BICUBIC))
            out["feat_maps"] = torch.from_numpy(feature_maps(small, self.patch))
        return out


class CanonicalDataset(_Base):
    """Canonical PNGs + on-the-fly deterministic transforms.

    `rows` must carry `asset_id`, `canonical_path`, `y`, and a `chain` column of Chain objects.
    """

    def __init__(self, rows: pd.DataFrame, **kw):
        super().__init__(**kw)
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        with Image.open(r.canonical_path) as im:
            im = im.convert("RGB")
        chain = r.chain if isinstance(r.chain, Chain) else Chain()
        img = np.asarray(apply_chain(im, chain, r.asset_id))
        return self._pack(img, {"label": r.y, "id": f"{r.asset_id}|{chain.name}"})


class FrozenEvalDataset(_Base):
    """The frozen evaluation shards. Bytes are never regenerated."""

    def __init__(self, shard_dir: str | Path, split: str | None = None, **kw):
        super().__init__(**kw)
        d = Path(shard_dir)
        shards = sorted(d.glob("eval-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no eval shards in {d}")
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        if split:
            df = df[df.split == split].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        with Image.open(io.BytesIO(r.image)) as im:
            img = np.asarray(im.convert("RGB"))
        return self._pack(img, {"label": r.label, "id": r.id})

    def meta(self) -> pd.DataFrame:
        """Provenance columns for the scorecard, aligned to __getitem__ order."""
        cols = ["id", "group", "label", "split", "source", "source_dataset", "generator",
                "authentic_subtype", "transform", "severity", "orig_width", "orig_height"]
        return self.df[[c for c in cols if c in self.df.columns]].copy()
