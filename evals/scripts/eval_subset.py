"""TRACE-RX-M v2 on every config of techjam-aigc/wildfake-eval-subset.

Scored through the repo's own `canonical_preprocess` (square bilinear resize to 224, ImageNet
normalisation) on the original stored bytes -- the pipeline the checkpoint was trained for.

Reported alongside every AUROC: the prevalence (AUPRC is not comparable across configs with
different base rates, and these range from 36% to 50% positive) and, where image dimensions are
recoverable, the size-only cheat baseline the dataset card warns about.
"""
import glob, io, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "feat_traincode" / "src"))
sys.path.insert(0, "/home/joel/Desktop/acai/src")
from techjam_aigc.trace_rx_m.training import load_detector_checkpoint
from techjam_aigc.trace_rx_m.augment import canonical_preprocess
from acai.metrics import compute, by_group

ROOT = Path("/home/joel/Desktop/acai/data/eval_subset")
CKPT = Path("/home/joel/.cache/huggingface/hub/models--techjam-aigc--trace-rx-m-v2/"
            "snapshots/3d4270323bbb2de437326fa59e436a1540da1110")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = ["data", "default_transformed", "normalized", "normalized_transformed",
           "laion_matched", "laion_matched_transformed", "cross_generator",
           "cross_generator_transformed", "diverse", "diverse_transformed"]
DISPLAY = {"data": "default"}


class Shards(Dataset):
    def __init__(self, df):
        self.b = [r["bytes"] if isinstance(r, dict) else r for r in df["image"]]
        self.y = df["label"].to_numpy()
    def __len__(self): return len(self.b)
    def __getitem__(self, i):
        with Image.open(io.BytesIO(self.b[i])) as im:
            x = canonical_preprocess(im, image_size=224)
        return torch.from_numpy(x), int(self.y[i])


@torch.no_grad()
def score(model, df, bs=64):
    dl = DataLoader(Shards(df), batch_size=bs, num_workers=10, pin_memory=True)
    S, Y = [], []
    for x, y in dl:
        S.append(model(x.to(DEV, non_blocking=True)).logit.float().cpu()); Y.append(y)
    return torch.cat(S).numpy().ravel(), torch.cat(Y).numpy()


def dims(df, n=4000):
    """Recover (w,h) from the stored bytes, for the size-cheat baseline."""
    out = []
    for r in df["image"].head(n):
        b = r["bytes"] if isinstance(r, dict) else r
        try:
            with Image.open(io.BytesIO(b)) as im: out.append(im.size)
        except Exception: out.append((0, 0))
    return np.array(out)


def main():
    model, meta = load_detector_checkpoint(CKPT / "s4_detector.pt", CKPT / "s3_memory.pt",
                                           device=DEV)
    print(f"TRACE-RX-M v2 | {meta['encoder_mode']} encoder | epoch {meta['epoch']}\n")
    results = {}
    print(f"{'config':30s} {'n':>6} {'prev':>5} {'AUROC':>7} {'AUPRC':>7} "
          f"{'TPR@1%':>7} {'EER':>6} {'size-cheat':>10}")
    for cfg in CONFIGS:
        shards = sorted((ROOT / cfg).glob("*.parquet"))
        if not shards:
            print(f"  {DISPLAY.get(cfg,cfg):28s} (missing)"); continue
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        s, y = score(model, df)
        m = compute(y, s)

        d = dims(df)
        cheat = ""
        if (d[:, 0] > 0).all() and len(np.unique(d[:, 0] * d[:, 1])) > 1:
            a = roc_auc_score(y[:len(d)], (d[:, 0] * d[:, 1]).astype(float))
            cheat = f"{max(a, 1-a):.4f}"

        name = DISPLAY.get(cfg, cfg)
        print(f"  {name:28s} {len(y):6d} {y.mean():5.3f} {m.auroc:7.4f} {m.auprc:7.4f} "
              f"{m.tpr_at_fpr01:7.4f} {m.eer:6.4f} {cheat:>10s}")
        r = m.as_dict(); r["size_cheat_auroc"] = cheat or None; r["n"] = int(len(y))
        # per-source and per-transform breakdowns where the columns exist
        if "source" in df.columns and df["source"].nunique() > 1:
            r["by_source"] = {k: v.get("auroc") for k, v in
                              by_group(y, s, df["source"].astype(str).values, min_n=50).items()}
        for col in ("transform", "transform_chain", "condition", "transform_family"):
            if col in df.columns:
                r[f"by_{col}"] = {k: v.get("auroc") for k, v in
                                  by_group(y, s, df[col].astype(str).values, min_n=50).items()}
                break
        results[name] = r
        pd.DataFrame({"label": y, "score": s}).to_parquet(
            f"/home/joel/Desktop/acai/runs/trace_evalsubset_{name}.parquet", index=False)

    Path("/home/joel/Desktop/acai/runs/trace_rx_m_evalsubset.json").write_text(
        json.dumps(results, indent=2, default=str))

    print("\n=== per-source breakdowns ===")
    for name, r in results.items():
        if "by_source" in r:
            print(f"  {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in r["by_source"].items()
                                            if v is not None))
    print("\n=== per-transform breakdowns ===")
    for name, r in results.items():
        k = next((k for k in r if k.startswith("by_") and k != "by_source"), None)
        if k:
            print(f"  {name} [{k}]: " + "  ".join(f"{a}={b:.3f}" for a, b in r[k].items()
                                                  if b is not None))
    print("\nwrote runs/trace_rx_m_evalsubset.json")


if __name__ == "__main__":
    main()
