"""Build the canonical and frozen-evaluation datasets.

Two stages, with different storage policies chosen for different reasons:

**Stage 1 - canonicalise (all 10,000 images).** Centre-crop square + resize to 512, saved as
PNG. This is the step that removes the resolution shortcut (see acai.data). PNG rather than
JPEG because the official grid contains a `jpeg` condition: pre-compressing the canonical copy
would put a second, invisible quantisation into *every* image regardless of its assigned
transform, contaminating the one condition we most need to measure cleanly.

**Stage 2 - freeze the evaluation conditions (dev + calibration x 15 conditions).** Written
once, as bytes, and never regenerated. This is deliberate and it is what makes the sealed
evaluator meaningful: a constitution that evaluates on freshly-regenerated inputs is only as
fixed as the generator. Frozen bytes plus a sealed evaluator means two runs of the same model
produce the same scorecard by construction, not by convention.

Training transforms are **not** frozen. They are generated on the fly from the canonical PNGs,
deterministically per (asset_id, chain), which keeps the composition study (plan §4) free to
sample any mixture without rebuilding 105,000 files per mixture. Determinism means a training
run is still exactly reproducible from its manifest.
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from acai import data as D
from acai.transforms import Chain, apply_chain, conditions

EVAL_SPLITS = ("dev", "calibration")
SHARD_ROWS = 2000


def _render(args) -> tuple[str, bytes | None]:
    """Apply one chain to one canonical image and return PNG bytes."""
    asset_id, path, fam, param = args
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
        chain = Chain() if fam == "clean" else Chain(((fam, param),))
        out = apply_chain(im, chain, asset_id)
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=False)
        return asset_id, buf.getvalue()
    except Exception:
        return asset_id, None


def freeze_eval(m: pd.DataFrame, out: str | Path, workers: int = 16,
                splits=EVAL_SPLITS) -> pd.DataFrame:
    """Materialise every official condition for the evaluation splits, as parquet shards."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sub = m[m.split.isin(splits)].reset_index(drop=True)
    conds = conditions()
    print(f"freezing {len(sub)} images x {len(conds)} conditions = "
          f"{len(sub) * len(conds)} rows", flush=True)

    jobs, meta = [], []
    for r in sub.itertuples():
        for fam, param in conds:
            jobs.append((r.asset_id, r.canonical_path, fam, param))
            meta.append(dict(
                id=f"{r.asset_id}|{fam}:{param:g}", asset_id=r.asset_id,
                group=r.group, label=int(r.y), split=r.split,
                source=r.source_dataset, source_dataset=r.source_dataset,
                generator=r.generator, authentic_subtype=r.authentic_subtype,
                transform=fam, severity=float(param),
                orig_width=int(r.width), orig_height=int(r.height)))

    shard, rows, blobs, n_bad = 0, [], [], 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (aid, blob) in enumerate(ex.map(_render, jobs, chunksize=32)):
            if blob is None:
                n_bad += 1
                continue
            rows.append(meta[i])
            blobs.append(blob)
            if len(rows) >= SHARD_ROWS:
                shard = _write_shard(out, shard, rows, blobs)
                rows, blobs = [], []
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    if rows:
        shard = _write_shard(out, shard, rows, blobs)

    man = pd.DataFrame(meta)
    if n_bad:
        print(f"  WARNING: {n_bad} renders failed")
    D.save(man, out / "manifest.parquet")
    return man


def _write_shard(out: Path, shard: int, rows: list, blobs: list) -> int:
    df = pd.DataFrame(rows)
    df["image"] = blobs
    df.to_parquet(out / f"eval-{shard:05d}.parquet", index=False)
    print(f"  wrote eval-{shard:05d}.parquet ({len(df)} rows)", flush=True)
    return shard + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/data_draft")
    ap.add_argument("--out", default="data/build")
    ap.add_argument("--size", type=int, default=D.CANON_SIZE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=0,
                    help="common low resolution every image passes through first; equalises "
                         "resampling history across classes (see acai.data)")
    ap.add_argument("--stage", choices=["canon", "freeze", "all"], default="all")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m = D.load_manifest(root)
    print(f"manifest: {len(m)} rows, {m.label.value_counts().to_dict()}")

    integrity = D.check_split_integrity(m)
    print(f"split integrity ok: {integrity['ok']}")
    (out / "split_integrity.json").write_text(json.dumps(integrity, indent=2, default=str))

    canon_manifest = out / "canonical_manifest.parquet"
    if a.stage in ("canon", "all"):
        print(f"canonicalising to {a.size}x{a.size} ...", flush=True)
        m = D.canonicalise(m, root, out / "canonical", a.size, a.workers,
                           a.bottleneck or None)
        D.save(m, canon_manifest)
        chk = D.verify_shortcut_removed(m)
        print("shortcut check:", json.dumps(chk, indent=2))
        (out / "shortcut_check.json").write_text(json.dumps(chk, indent=2))
    else:
        m = pd.read_parquet(canon_manifest)
        m["y"] = m["y"].astype(int)

    if a.stage in ("freeze", "all"):
        freeze_eval(m, out / "frozen_eval", a.workers)
    print("done")


if __name__ == "__main__":
    main()
