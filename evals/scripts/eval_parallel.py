"""TRACE-RX-Parallel (albagon/trace-rx-parallel-techjam2026) on data_draft + wildfake-eval-subset.

A genuinely different architecture from trace-rx-m: two branches (a global patch-statistics
branch and the authentic-memory branch) fused by a learned 2-weight gate. Its own
`s4_validity.json` reports all three branch heads on the held-out generator, so we log all
three: `logit` (fused), `global_logit`, `memory_logit`.

Weights: best_detector.pt == final_detector.pt (byte-identical, epoch 8, frozen encoder).
Preprocessing is the repo's own canonical_preprocess, as for the other model.
"""
import io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

BR = Path(__file__).parent / "feat_trace-rx-parallel" / "src"
sys.path.insert(0, str(BR)); sys.path.insert(0, "/home/joel/Desktop/acai/src")
from techjam_aigc.trace_rx_parallel.training import load_detector_checkpoint
from techjam_aigc.trace_rx_m.augment import canonical_preprocess
from acai import data as D
from acai.metrics import compute, by_group

R, B = "albagon/trace-rx-parallel-techjam2026", "trace-rx-parallel-techjam2026/"
DRAFT = Path("/home/joel/Desktop/acai/data/data_draft")
SUBSET = Path("/home/joel/Desktop/acai/data/eval_subset")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/home/joel/Desktop/acai/runs")

CONFIGS = ["data", "default_transformed", "normalized", "normalized_transformed",
           "laion_matched", "laion_matched_transformed", "cross_generator",
           "cross_generator_transformed", "diverse", "diverse_transformed"]


class Files(Dataset):
    def __init__(self, rows): self.r = rows.reset_index(drop=True)
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        row = self.r.iloc[i]
        with Image.open(DRAFT / row.path) as im:
            return torch.from_numpy(canonical_preprocess(im, image_size=224)), int(row.y)


class Blobs(Dataset):
    def __init__(self, df):
        self.b = [r["bytes"] if isinstance(r, dict) else r for r in df["image"]]
        self.y = df["label"].to_numpy()
    def __len__(self): return len(self.b)
    def __getitem__(self, i):
        with Image.open(io.BytesIO(self.b[i])) as im:
            return torch.from_numpy(canonical_preprocess(im, image_size=224)), int(self.y[i])


@torch.no_grad()
def score(model, ds, bs=64):
    dl = DataLoader(ds, batch_size=bs, num_workers=10, pin_memory=True)
    F, G, M, Y = [], [], [], []
    for x, y in dl:
        o = model(x.to(DEV, non_blocking=True))
        F.append(o.logit.float().cpu()); G.append(o.global_logit.float().cpu())
        M.append(o.memory_logit.float().cpu()); Y.append(y)
    cat = lambda t: torch.cat(t).numpy().ravel()
    return cat(F), cat(G), cat(M), torch.cat(Y).numpy()


def row(tag, f, g, m, y, extra=""):
    a = compute(y, f)
    gg, mm = roc_auc_score(y, g), roc_auc_score(y, m)
    print(f"  {tag:30s} {len(y):6d} {y.mean():5.3f} {a.auroc:7.4f} {a.auprc:7.4f} "
          f"{a.tpr_at_fpr01:7.4f} {a.eer:6.4f} {gg:7.4f} {mm:7.4f} {extra}")
    d = a.as_dict(); d.update(global_auroc=float(gg), memory_auroc=float(mm), n=int(len(y)))
    return d


def main():
    ck = hf_hub_download(R, B + "best_detector.pt")
    mem = hf_hub_download(R, B + "s3_memory.pt")
    model, meta = load_detector_checkpoint(Path(ck), Path(mem), device=DEV)
    print(f"TRACE-RX-Parallel | {meta['architecture']} | {meta['encoder_mode']} | "
          f"epoch {meta['epoch']}\n")
    print(f"  {'population':30s} {'n':>6} {'prev':>5} {'FUSED':>7} {'AUPRC':>7} "
          f"{'TPR@1%':>7} {'EER':>6} {'global':>7} {'memory':>7}")

    res = {"data_draft": {}, "eval_subset": {}}

    print("\n--- data_draft ---")
    m = D.load_manifest(DRAFT)
    for name, sel in [("WildFake", m.source_dataset == "wildfake"),
                      ("SID-Set", m.source_dataset == "sid_set"),
                      ("Both pooled", m.index == m.index)]:
        sub = m[sel]
        f, g, mm_, y = score(model, Files(sub))
        res["data_draft"][name] = row(name, f, g, mm_, y)
        if name == "WildFake":
            gen = sub.generator.values; real = y == 0
            for gname in ["adm", "ddim", "ddpm", "gan_based", "imagen"]:
                k = (gen == gname) | real
                if len(np.unique(y[k])) == 2:
                    res["data_draft"][f"wf/{gname}"] = row(f"  gen:{gname}", f[k], g[k], mm_[k], y[k])

    print("\n--- wildfake-eval-subset ---")
    for cfg in CONFIGS:
        sh = sorted((SUBSET / cfg).glob("*.parquet"))
        if not sh:
            print(f"  {cfg:30s} (missing)"); continue
        df = pd.concat([pd.read_parquet(s) for s in sh], ignore_index=True)
        f, g, mm_, y = score(model, Blobs(df))
        nm = "default" if cfg == "data" else cfg
        d = row(nm, f, g, mm_, y)
        if "source" in df.columns and df["source"].nunique() > 1:
            d["by_source"] = {k: v.get("auroc") for k, v in
                              by_group(y, f, df["source"].astype(str).values, min_n=50).items()}
        res["eval_subset"][nm] = d
        pd.DataFrame({"label": y, "fused": f, "global": g, "memory": mm_}).to_parquet(
            OUT / f"parallel_{nm}.parquet", index=False)

    (OUT / "trace_rx_parallel_results.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n=== per-source (eval subset) ===")
    for k, v in res["eval_subset"].items():
        if "by_source" in v:
            print(f"  {k}: " + "  ".join(f"{a}={b:.3f}" for a, b in v["by_source"].items() if b))
    print("\nwrote runs/trace_rx_parallel_results.json")


if __name__ == "__main__":
    main()
