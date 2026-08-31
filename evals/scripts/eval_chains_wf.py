"""Both models on data_draft WILDFAKE under chains of 1..6 sequential transforms.

Unlike the calibration split, neither model trained on any part of WildFake, so every
generator here is unseen -- this is out-of-distribution throughout, not just in one slice.

Uniform on both axes, by construction rather than by sampling luck:

* **Chain length.** Every image gets every length 1..6 (full factorial over k), so each
  length has exactly n=5,585 and lengths are perfectly balanced.
* **Transform type.** For each (image, length) the 6 families are put in a random
  permutation seeded by (asset, k) and the first k are taken. By symmetry every family is
  equally likely at every position, so each family appears in a k-chain for k/6 of images
  and family coverage is uniform across families. The setting within a family (e.g. which
  JPEG quality) is drawn uniformly from that family's options.

There are exactly 6 families, so k=6 uses all of them and only the parameters vary.

* **Order.** Applied in the sampled order, so all k! orderings of the chosen families are
  equally likely. This deliberately differs from the repo's fixed physical order
  (crop -> resize -> jitter -> blur -> noise -> jpeg) used for the `_transformed` configs:
  ordering is a variable here, not a constant. The ordered chain is recorded per row so
  order effects can be measured afterwards.
"""
import hashlib, json, sys
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
from acai.transforms import SETTINGS, FN, ORDER, label
from acai.metrics import compute

DATA = Path("/home/joel/Desktop/acai/data/data_draft")
OUT = Path("/home/joel/Desktop/acai/runs")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KS = [1, 2, 3, 4, 5, 6]
FAMILIES = list(ORDER)                                   # 6 families
BY_FAM = {f: [s for s in SETTINGS if s[0] == f] for f in FAMILIES}


def chain_for(asset_id: str, k: int):
    """k distinct families, uniform; one uniform setting each; fixed physical order."""
    # Full-string hash, not a byte slice: taking the last 8 bytes and reducing mod 2**31
    # keeps only the low-order characters, which for fixed-width ids can be constant --
    # that collapsed every asset onto one seed and skewed family coverage badly.
    seed = int.from_bytes(hashlib.blake2b(asset_id.encode(), digest_size=8).digest(), "big")
    rng = np.random.default_rng([seed % (2**63), k])
    fams = list(rng.permutation(FAMILIES))[:k]
    # Applied in the SAMPLED order, NOT sorted into the repo's fixed physical order.
    # A uniform permutation of all 6 families truncated to k is a uniformly random ordered
    # k-sequence, so all k! orderings of the chosen families are equally likely.
    return [BY_FAM[f][int(rng.integers(len(BY_FAM[f])))] for f in fams]


class ChainDS(Dataset):
    def __init__(self, rows):
        self.items = [(r.image_path, r.asset_id, k) for r in rows.itertuples() for k in KS]
        self.y = {r.asset_id: int(r.y) for r in rows.itertuples()}

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, aid, k = self.items[i]
        chain = chain_for(aid, k)
        # Python's hash() is salted per process; use the same stable digest here so a
        # rerun reproduces byte-identical transformed images.
        pseed = int.from_bytes(hashlib.blake2b(aid.encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng([pseed % (2**63), k, 7])
        with Image.open(DATA / path) as im:
            im = im.convert("RGB")
            for fam, param in chain:
                im = FN[fam](im, param, rng)
            x = canonical_preprocess(im, image_size=224)
        return torch.from_numpy(x), self.y[aid], i


@torch.no_grad()
def run(model, ds, parallel, bs=96):
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


def main():
    from acai import data as D
    m = D.load_manifest(DATA)
    rows = m[m.source_dataset == "wildfake"].reset_index(drop=True)
    rows = rows.rename(columns={"path": "image_path"})
    rows["ai_subtype"] = rows["generator"]
    ds = ChainDS(rows)
    info = rows.set_index("asset_id")
    meta = pd.DataFrame([{
        "asset_id": aid, "k": k, "label": ds.y[aid],
        "generator": (info.loc[aid].ai_subtype
                      if isinstance(info.loc[aid].ai_subtype, str) else ""),
        "chain": "+".join(label(f, p) for f, p in chain_for(aid, k)),
    } for _, aid, k in ds.items])
    print(f"{len(rows)} images x {len(KS)} chain lengths = {len(ds)} evaluations")

    fam_counts = {f: int(meta.chain.str.contains(f).sum()) for f in FAMILIES}
    print("family coverage across all chains (should be near-equal):", fam_counts)
    print("rows per chain length:", meta.k.value_counts().sort_index().to_dict())

    if WHICH == "m":
        from techjam_aigc.trace_rx_m.training import load_detector_checkpoint as load
        P = Path("/home/joel/.cache/huggingface/hub/models--techjam-aigc--trace-rx-m-v2/"
                 "snapshots/3d4270323bbb2de437326fa59e436a1540da1110")
        model, _ = load(P / "s4_detector.pt", P / "s3_memory.pt", device=DEV)
        name = "TRACE-RX-M v2"
    else:
        from techjam_aigc.trace_rx_parallel.training import load_detector_checkpoint as load
        R, B = "albagon/trace-rx-parallel-techjam2026", "trace-rx-parallel-techjam2026/"
        model, _ = load(Path(hf_hub_download(R, B + "best_detector.pt")),
                        Path(hf_hub_download(R, B + "s3_memory.pt")), device=DEV)
        name = "TRACE-RX-Parallel"

    f, g, m = run(model, ds, parallel=(WHICH == "p"))
    y = meta.label.values
    res = {"model": name, "n": int(len(y)), "prevalence": float(y.mean())}

    ov = compute(y, f)
    print(f"\n### {name} — pooled over all chain lengths")
    print(f"  AUROC {ov.auroc:.4f}  AUPRC {ov.auprc:.4f} (lift {ov.auprc_lift:.2f}x)  "
          f"TPR@1% {ov.tpr_at_fpr01:.4f}  EER {ov.eer:.4f}")
    res["overall"] = ov.as_dict()

    print(f"\n  {'chain length':13s} {'n':>6} {'AUROC':>7} {'AUPRC':>7} {'TPR@1%':>7} {'EER':>6}")
    per_k = {}
    for k in KS:
        sel = (meta.k == k).values
        a = compute(y[sel], f[sel])
        per_k[k] = a.as_dict()
        print(f"    {k} transform{'s' if k>1 else ' '}  {int(sel.sum()):6d} {a.auroc:7.4f} "
              f"{a.auprc:7.4f} {a.tpr_at_fpr01:7.4f} {a.eer:6.4f}")
    res["per_chain_length"] = per_k
    print(f"  k=1 -> k=6 drop: {per_k[1]['auroc'] - per_k[6]['auroc']:+.4f}")

    print(f"\n  per generator x chain length (AUROC):")
    print(f"    {'generator':22s} " + "".join(f"{'k='+str(k):>8s}" for k in KS))
    pg = {}
    real = y == 0
    for gname in ["adm", "ddim", "ddpm", "gan_based", "imagen"]:
        vals = []
        for k in KS:
            sel = ((meta.generator == gname) | real).values & (meta.k == k).values
            vals.append(compute(y[sel], f[sel]).auroc if len(np.unique(y[sel])) == 2 else np.nan)
        pg[gname] = vals
        flag = ""   # every WildFake generator is unseen for both models
        print(f"    {gname:22s} " + "".join(f"{v:8.4f}" for v in vals) + flag)
    res["per_generator_by_k"] = pg

    if WHICH == "p":
        for nm, sc in (("global", g), ("memory", m)):
            a = compute(y, sc)
            res[f"branch_{nm}"] = a.as_dict()
            print(f"\n  branch {nm}: AUROC {a.auroc:.4f}  TPR@1% {a.tpr_at_fpr01:.4f}")

    meta.assign(score=f).to_parquet(OUT / f"chains_wf_{WHICH}.parquet", index=False)
    (OUT / f"chain_results_wf_{WHICH}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote runs/chain_results_wf_{WHICH}.json")


if __name__ == "__main__":
    main()
