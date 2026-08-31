"""Score `trace-rx-m-v2` over every config of `techjam-aigc/wildfake-eval-subset`.

Writes one `predictions.parquet` per config carrying the upstream logit alongside the label
and the per-row transform columns, so metrics (`report.py`) are recomputable without re-running
the backbone. Positive class is label 1 == AIGC; the logit is positive for AIGC.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import model as tracerx

CONFIGS = {
    "default": "data", "default_transformed": "default_transformed",
    "normalized": "normalized", "normalized_transformed": "normalized_transformed",
    "laion_matched": "laion_matched", "laion_matched_transformed": "laion_matched_transformed",
    "cross_generator": "cross_generator", "cross_generator_transformed": "cross_generator_transformed",
    "diverse": "diverse", "diverse_transformed": "diverse_transformed",
}
PASSTHROUGH = ["label", "source", "id", "orig_path",
               "primary_transform", "transform_chain", "n_transforms"]


def run_config(model, image_size, dev, root: Path, name: str, subdir: str,
               out_dir: Path, batch: int, family: str = "trace_rx_m") -> dict:
    files = sorted((root / subdir).glob("*.parquet"))
    if not files:
        return {"config": name, "status": "missing"}

    cols, logits = {c: [] for c in PASSTHROUGH}, []
    n, t0 = 0, time.time()
    for f in files:
        pf = pq.ParquetFile(f)
        have = set(pf.schema_arrow.names)
        keep = [c for c in PASSTHROUGH if c in have]
        for rb in pf.iter_batches(batch_size=batch, columns=["image"] + keep):
            d = rb.to_pydict()
            px = tracerx.preprocess([r["bytes"] for r in d["image"]], image_size, dev, family)
            with torch.inference_mode():
                logits.append(model(px).logit.float().cpu().numpy())
            for c in PASSTHROUGH:
                cols[c].extend(d[c] if c in d else [None] * len(d["image"]))
            n += len(d["image"])
            if n % (batch * 40) == 0:
                print(f"  {name}: {n} imgs, {n/(time.time()-t0):.0f} img/s", flush=True)

    tbl = {c: pa.array(v) for c, v in cols.items() if any(x is not None for x in v)}
    tbl["logit"] = pa.array(np.concatenate(logits).astype(np.float64))
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(tbl), out_dir / f"{name}.parquet")
    dt = time.time() - t0
    print(f"[done] {name}: {n} imgs in {dt:.0f}s ({n/dt:.0f} img/s)", flush=True)
    return {"config": name, "status": "ok", "n": n, "seconds": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/eval_subset_full")
    ap.add_argument("--out", default="runs/tracerx_v2_wildfake/predictions")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--family", default=tracerx.DEFAULT_FAMILY, choices=list(tracerx.FAMILIES),
                    help="which upstream detector family the checkpoint belongs to")
    ap.add_argument("--model-dir", default=str(tracerx.MODEL_DIR),
                    help="directory holding s4_detector.pt and s3_memory.pt")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, image_size = tracerx.load(dev, Path(a.model_dir), family=a.family)
    out = Path(a.out)
    manifest = [run_config(model, image_size, dev, Path(a.data_root), c, CONFIGS[c], out,
                           a.batch, a.family) for c in a.configs]
    (out.parent / "inference_manifest.json").write_text(json.dumps({
        "model_dir": a.model_dir, "family": a.family,
        "upstream_commit": tracerx.FAMILIES[a.family]["commit"],
        "encoder_mode": meta["encoder_mode"], "image_size": image_size,
        "configs": manifest}, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
