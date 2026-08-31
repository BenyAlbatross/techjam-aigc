"""Both TRACE-RX models on Joshyxwa/techjam2026 `calibration`, under all 15 official transforms.

Transform coverage is FULL FACTORIAL: every one of the 15 official conditions is applied to
every one of the 5,585 images. That is uniform by construction -- each condition gets identical
coverage on identical images -- and gives n=5,585 per condition rather than the ~372 a
round-robin assignment would leave. It also pairs the conditions, so a per-condition difference
cannot be an artefact of which images landed in which bucket.

Transforms are applied at the image's NATIVE resolution, before each model's own 224 preprocessing
-- the order a deployed detector actually sees. Note trace-rx-m's card records that their own
robustness numbers instead transformed already-224 BMPs, so per-condition figures here are not
expected to match theirs exactly.

Prevalence is 0.268 (4,090 real / 1,495 AI), so AUPRC is reported with its base rate and lift.
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
WHICH = sys.argv[1] if len(sys.argv) > 1 else "m"
BRANCH = "feat_traincode" if WHICH == "m" else "feat_trace-rx-parallel"
sys.path.insert(0, str(ROOT / BRANCH / "src"))
sys.path.insert(0, "/home/joel/Desktop/acai/src")

from techjam_aigc.trace_rx_m.augment import canonical_preprocess
from acai.transforms import SETTINGS, FN, label, apply as apply_chain
from acai.metrics import compute

DATA = Path("/home/joel/Desktop/acai/data/techjam2026")
OUT = Path("/home/joel/Desktop/acai/runs")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clean + the 14 official settings, from the team's own transforms module so these results
# stay consistent with how the `_transformed` eval configs were generated.
CONDS = [("clean", 0.0)] + [(f, float(p)) for f, p in SETTINGS]


class CalibDS(Dataset):
    """One (image, condition) pair per item. Deterministic per asset_id."""

    def __init__(self, rows):
        self.items = [(r.image_path, r.asset_id, f, p)
                      for r in rows.itertuples() for f, p in CONDS]
        self.y = {r.asset_id: int(r.label == "ai_full") for r in rows.itertuples()}

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, aid, fam, param = self.items[i]
        with Image.open(DATA / path) as im:
            im = im.convert("RGB")
            if fam != "clean":
                im = apply_chain(im, [(fam, param)], abs(hash(aid)) % (2**31))
            x = canonical_preprocess(im, image_size=224)
        return torch.from_numpy(x), self.y[aid], i


def meta_frame(ds, rows):
    info = rows.set_index("asset_id")
    recs = []
    for path, aid, fam, param in ds.items:
        r = info.loc[aid]
        recs.append({"asset_id": aid, "label": int(r.label == "ai_full"),
                     "transform": fam, "severity": float(param),
                     "generator": r.ai_subtype if isinstance(r.ai_subtype, str) else "",
                     "source_dataset": r.source_dataset,
                     "source_family": r.source_family})
    return pd.DataFrame(recs)


@torch.no_grad()
def run(model, ds, parallel: bool, bs=96):
    dl = DataLoader(ds, batch_size=bs, num_workers=12, pin_memory=True)
    n = len(ds)
    F = np.zeros(n, np.float32)
    G = np.zeros(n, np.float32) if parallel else None
    M = np.zeros(n, np.float32) if parallel else None
    for x, _, idx in dl:
        o = model(x.to(DEV, non_blocking=True))
        i = idx.numpy()
        F[i] = o.logit.float().cpu().numpy().ravel()
        if parallel:
            G[i] = o.global_logit.float().cpu().numpy().ravel()
            M[i] = o.memory_logit.float().cpu().numpy().ravel()
    return F, G, M


def report(name, meta, s, res):
    y = meta.label.values
    ov = compute(y, s)
    print(f"\n### {name}")
    print(f"  overall (all 15 conditions pooled)  n={len(y)}  prevalence={y.mean():.3f}")
    print(f"    AUROC {ov.auroc:.4f}  AUPRC {ov.auprc:.4f} (lift {ov.auprc_lift:.2f}x)  "
          f"TPR@1% {ov.tpr_at_fpr01:.4f}  EER {ov.eer:.4f}")
    r = {"overall": ov.as_dict()}

    print(f"  {'condition':16s} {'n':>6} {'AUROC':>7} {'AUPRC':>7} {'TPR@1%':>7}")
    per = {}
    clean_auc = None
    for f, p in CONDS:
        k = (meta["transform"] == f) & (meta.severity == float(p))
        m = compute(y[k.values], s[k.values])
        tag = "clean" if f == "clean" else label(f, p)
        if f == "clean": clean_auc = m.auroc
        per[tag] = m.as_dict()
        print(f"    {tag:14s} {int(k.sum()):6d} {m.auroc:7.4f} {m.auprc:7.4f} {m.tpr_at_fpr01:7.4f}")
    r["per_condition"] = per
    worst = min(per, key=lambda k: per[k]["auroc"])
    r["clean_auroc"] = clean_auc
    r["worst_condition"] = {"condition": worst, "auroc": per[worst]["auroc"]}
    r["clean_to_worst_drop"] = clean_auc - per[worst]["auroc"]
    print(f"  clean {clean_auc:.4f} -> worst {worst} {per[worst]['auroc']:.4f} "
          f"(drop {r['clean_to_worst_drop']:+.4f})")

    print(f"  per generator (vs all reals, clean rows only):")
    cl = (meta["transform"] == "clean").values
    gen = meta.generator.values
    real = (y == 0)
    pg = {}
    for g in sorted(set(gen) - {""}):
        k = ((gen == g) | real) & cl
        if len(np.unique(y[k])) == 2:
            a = compute(y[k], s[k])
            pg[g] = a.as_dict()
            flag = "  <- HELD OUT in training" if g == "gemini_flash_image" else ""
            print(f"    {g:22s} n={int(k.sum()):5d}  AUROC {a.auroc:.4f}{flag}")
    r["per_generator_clean"] = pg
    res[name] = r


def main():
    labels = pd.read_csv(DATA / "labels.csv")
    rows = labels[labels.split == "calibration"].reset_index(drop=True)
    ds = CalibDS(rows)
    meta = meta_frame(ds, rows)
    print(f"calibration: {len(rows)} images x {len(CONDS)} conditions = {len(ds)} evaluations")
    print(f"prevalence {meta.label.mean():.3f} ({int((meta.label==0).sum()/len(CONDS))} real / "
          f"{int(meta.label.sum()/len(CONDS))} ai)")

    res = {}
    if WHICH == "m":
        from techjam_aigc.trace_rx_m.training import load_detector_checkpoint as load_m
        P = Path("/home/joel/.cache/huggingface/hub/models--techjam-aigc--trace-rx-m-v2/"
                 "snapshots/3d4270323bbb2de437326fa59e436a1540da1110")
        model, _ = load_m(P / "s4_detector.pt", P / "s3_memory.pt", device=DEV)
        f, _, _ = run(model, ds, parallel=False)
        report("TRACE-RX-M v2", meta, f, res)
        meta.assign(score=f).to_parquet(OUT / "calib_trace_rx_m.parquet", index=False)
    else:
        from techjam_aigc.trace_rx_parallel.training import load_detector_checkpoint as load_p
        R, B = "albagon/trace-rx-parallel-techjam2026", "trace-rx-parallel-techjam2026/"
        ck = Path(hf_hub_download(R, B + "best_detector.pt"))
        mem = Path(hf_hub_download(R, B + "s3_memory.pt"))
        model, _ = load_p(ck, mem, device=DEV)
        f2, g2, m2 = run(model, ds, parallel=True)
        report("TRACE-RX-Parallel (fused)", meta, f2, res)
        for nm, sc in (("TRACE-RX-Parallel (global branch)", g2),
                       ("TRACE-RX-Parallel (memory branch)", m2)):
            y = meta.label.values
            a = compute(y, sc)
            res[nm] = {"overall": a.as_dict()}
            print(f"\n### {nm}\n    AUROC {a.auroc:.4f}  AUPRC {a.auprc:.4f}  "
                  f"TPR@1% {a.tpr_at_fpr01:.4f}")
        meta.assign(fused=f2, glob=g2, mem=m2).to_parquet(
            OUT / "calib_trace_parallel.parquet", index=False)

    (OUT / f"calibration_results_{WHICH}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote runs/calibration_results_{WHICH}.json")


if __name__ == "__main__":
    main()
