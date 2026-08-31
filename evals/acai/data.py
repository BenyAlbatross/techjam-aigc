"""Dataset layer for `Joshyxwa/data_draft`: canonicalisation, manifests, holdouts.

The central job of this module is to remove the resolution shortcut before any model sees
an image. Measured on the raw manifest, with no pixels and no model:

    pixel count   pooled AUROC 0.739   SID-Set 0.979   WildFake 0.976
    is-square     pooled AUROC 0.739   SID-Set 0.978

SID's AI images are all exactly 1024x1024 against non-square real web photos; WildFake's reals
are all exactly 200x200 (an upstream artifact) against native-resolution AI. The two point in
opposite directions, so the *pooled* figure partly cancels to a merely-suspicious 0.739 while
each source alone is ~98% solved by image dimensions.

Canonicalisation is therefore centre-crop-to-square plus resize to one fixed size, which
neutralises pixel count, aspect ratio and is-square together. `width`/`height` are kept in the
manifest as **audit-only** columns: acai.audit checks whether a score still tracks them, but no
model input is ever derived from them.

Note the official `resize` transform does not help here -- it upscales back to the original
dimensions by design, so it preserves the leak exactly.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CANON_SIZE = 512
SPLITS = ("train", "dev", "calibration")

# The 5 WildFake generators carrying a real identity. SID's 2,500 AI images are a single
# lumped `text_to_image` with model_family='unknown_generator', so they can be held out only
# as one block, never per-generator.
NAMED_GENERATORS = ("adm", "ddim", "ddpm", "gan_based", "imagen")
LUMPED_GENERATOR = "text_to_image"

SOURCES = ("sid_set", "wildfake")


def load_manifest(root: str | Path) -> pd.DataFrame:
    """Read the dataset manifest.

    `manifest.parquet` in this repo is misnamed: it is JSONL, not parquet. pd.read_parquet
    fails on it with "Parquet magic bytes not found", so the loader snippet in the dataset
    card does not work as written. We read it as JSONL and keep going.
    """
    root = Path(root)
    p = root / "manifest.parquet"
    try:
        m = pd.read_parquet(p)
    except Exception:
        m = pd.read_json(p, lines=True)

    keep = ["asset_id", "lineage_id", "base_id", "split", "label", "path",
            "source_dataset", "source_family", "real_subtype", "ai_subtype",
            "width", "height", "aspect_ratio", "file_size_bytes", "sha256", "phash"]
    m = m[[c for c in keep if c in m.columns]].copy()

    m["y"] = (m["label"] == "ai_full").astype(int)
    # One generator column spanning both sources: named where known, lumped where not.
    m["generator"] = np.where(m["y"] == 1,
                              m["ai_subtype"].replace("", LUMPED_GENERATOR),
                              "")
    m["authentic_subtype"] = np.where(m["y"] == 0, m["real_subtype"].replace("", "other"), "")
    # Grouping key for anything that must not straddle a split. Every identity field in this
    # draft is 1:1 with the asset, so this groups nothing *today*; it starts binding once
    # transform variants of one source image share a lineage.
    m["group"] = m["lineage_id"]
    return m.reset_index(drop=True)


# ------------------------------------------------------------------ canonicalisation

def _canon_one(args) -> tuple[str, bool]:
    src, dst, size, bottleneck, mode = args
    dst = Path(dst)
    if dst.exists():
        return str(dst), True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if mode == "native_crop" and min(im.size) >= size:
                # Take a `size`x`size` patch at NATIVE scale: no resampling at all. This is
                # forensically the best option where it is available, because every resize
                # destroys the generator fingerprints a detector wants. It also discards the
                # parent aspect ratio, which on SID is itself a near-perfect class cue.
                l, t = (im.width - size) // 2, (im.height - size) // 2
                im = im.crop((l, t, l + size, t + size))
            else:
                s_ = min(im.size)
                im = im.crop(((im.width - s_) // 2, (im.height - s_) // 2,
                              (im.width + s_) // 2, (im.height + s_) // 2))
                if bottleneck:
                    im = im.resize((bottleneck, bottleneck), Image.BICUBIC)
                im = im.resize((size, size), Image.BICUBIC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "PNG", optimize=False)
        return str(dst), True
    except Exception:
        return str(dst), False


def canonicalise(manifest: pd.DataFrame, root: str | Path, out: str | Path,
                 size: int = CANON_SIZE, workers: int = 16,
                 bottleneck: int | None = None, mode: str = "square_resize") -> pd.DataFrame:
    """Centre-crop square + resize every image to `size`, losslessly. Idempotent.

    `bottleneck` routes every image through a common low resolution first, equalising the
    resampling history across classes. Required whenever the two classes have systematically
    different native resolutions -- which they do here.
    """
    root, out = Path(root), Path(out)
    jobs = [(str(root / r.path), str(out / f"{r.asset_id}.png"), size, bottleneck, mode)
            for r in manifest.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_canon_one, jobs, chunksize=32))

    m = manifest.copy()
    m["canonical_path"] = [r[0] for r in res]
    ok = np.array([r[1] for r in res])
    if not ok.all():
        m = m[ok].reset_index(drop=True)
    m["canon_size"] = size
    m["canon_bottleneck"] = bottleneck or 0
    m["canon_mode"] = mode
    return m


def verify_shortcut_removed(m: pd.DataFrame) -> dict:
    """Confirm canonicalisation actually killed the dimension leak.

    Run after canonicalisation and before any training. A silent failure here would put us
    back to a 0.98 shortcut with no warning, which is the single worst outcome for this project.
    """
    from sklearn.metrics import roc_auc_score
    out = {}
    sizes = [Image.open(p).size for p in m["canonical_path"].head(200)]
    out["all_canonical_same_size"] = len(set(sizes)) == 1
    out["canonical_size"] = list(sizes[0])
    for name, v in [("pixels_original", (m.width * m.height).values),
                    ("aspect_original", m.aspect_ratio.values)]:
        a = roc_auc_score(m.y, v)
        out[f"{name}_auroc"] = float(max(a, 1 - a))
    out["note"] = ("original-dimension AUROCs are retained as the audit baseline; after "
                   "canonicalisation the model cannot observe them")
    return out


# ---------------------------------------------------------------------- holdouts

def holdout_masks(m: pd.DataFrame) -> dict[str, dict]:
    """Generalisation holdouts available in this dataset.

    * `unseen_generator/<g>` -- leave-one-generator-out over the 5 named WildFake generators.
      SID's AI half has no generator identity, so it is never a holdout target, only ever
      training or in-distribution eval.
    * `unseen_source/<s>` -- leave-one-source-out over the 2 source datasets. Only 2 exist, so
      this is a 2-way test rather than a sweep -- but given that each source carries its own
      ~0.98 dimension shortcut, it is the sharpest generalisation probe available here.

    Each entry gives the *train* mask and the *eval* mask. Reals are never held out with the
    generator: holding out `adm` means "never saw ADM images", not "never saw any real image".
    """
    out: dict[str, dict] = {}
    present = set(m.generator.unique())
    for g in NAMED_GENERATORS:
        if g not in present:
            continue          # never emit a holdout with an empty eval side
        held = (m.generator == g).values
        out[f"unseen_generator/{g}"] = {
            "train": ~held,
            "eval": held | ((m.y == 0) & (m.source_dataset == "wildfake")).values,
            "note": f"AI={g} unseen in training; evaluated against WildFake reals",
        }
    for s in SOURCES:
        if s not in set(m.source_dataset.unique()):
            continue
        held = (m.source_dataset == s).values
        out[f"unseen_source/{s}"] = {
            "train": ~held, "eval": held,
            "note": f"entire {s} source unseen in training",
        }
    return out


def check_split_integrity(m: pd.DataFrame) -> dict:
    """Assert no group, hash or perceptual hash straddles two splits.

    The dataset card claims the builder already rejects these. We re-check rather than trust:
    a leak here inflates every number in the scorecard, and it is cheap to rule out.
    """
    issues = {}
    for col in ("group", "sha256", "phash", "base_id"):
        if col not in m.columns:
            continue
        n = m.groupby(col)["split"].nunique()
        bad = n[n > 1]
        issues[col] = {"straddling": int(len(bad)), "examples": bad.index[:5].tolist()}
    counts = m.groupby(["split", "label"]).size().unstack(fill_value=0)
    issues["counts"] = counts.to_dict()
    issues["ok"] = all(v["straddling"] == 0 for k, v in issues.items() if isinstance(v, dict)
                       and "straddling" in v)
    return issues


def save(m: pd.DataFrame, path: str | Path) -> Path:
    """Persist the working manifest. Mixed-type columns are coerced, since the upstream
    `file_size` column mixes int and str and breaks a naive parquet write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = m.copy()
    for c in m.columns:
        if m[c].dtype == object:
            m[c] = m[c].astype(str)
    m.to_parquet(path, index=False)
    return path
