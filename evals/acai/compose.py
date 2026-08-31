"""Training-data composition study (plan §4).

The key move is the **embedding bank**. A naive sweep would re-run the backbone for every
candidate mixture, because each mixture assigns different transforms to the training images.
Instead we extract embeddings once for *all* (image, condition) pairs in the training split --
7,000 images x 15 conditions = 105,000 vectors, about 320 MB at ViT-B width -- after which any
mixture is a **row selection** from that bank. A 100-mixture sweep becomes a few minutes of
logistic regression rather than days of forward passes.

Two constraints make the comparison mean something:

* **Constant training size.** Every mixture selects exactly one condition per training image,
  so all mixtures train on exactly 7,000 rows. Otherwise a "better" mixture might just be more
  data, which is a different experiment.
* **Fixed evaluation.** Every mixture is scored on the same frozen eval shards. Optimising a
  mixture against a moving evaluation measures nothing.

Reported both ways -- mean AUROC over transforms and **worst-family** AUROC. These do not share
an optimum, and a robustness claim rests on the second.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acai.dataset import CanonicalDataset
from acai.metrics import by_group, compute, worst_family
from acai.models.backbone import DinoV3
from acai.models.heads import Detector
from acai.runlog import Run
from acai.train import DEV, cached_extract, fit_probe, score_probe
from acai.transforms import Chain, FAMILY_ORDER, conditions

CELLS = ("real_clean", "real_transformed", "ai_clean", "ai_transformed")


def build_bank(build: Path, variant: str, img_size: int, pool: str,
               batch: int = 128, workers: int = 12) -> dict:
    """Embeddings for every (training image, official condition) pair. Cached on disk."""
    path = build / "cache" / f"bank_{variant}_{img_size}_{pool}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)

    m = pd.read_parquet(build / "canonical_manifest.parquet")
    m["y"] = m["y"].astype(int)
    tr = m[m.split == "train"].reset_index(drop=True)

    rows = []
    for fam, param in conditions():
        c = tr.copy()
        c["chain"] = [Chain() if fam == "clean" else Chain(((fam, param),))] * len(c)
        c["condition_family"] = fam
        c["condition_param"] = float(param)
        rows.append(c)
    allrows = pd.concat(rows, ignore_index=True)
    print(f"embedding bank: {len(allrows)} rows "
          f"({len(tr)} images x {len(conditions())} conditions)", flush=True)

    ds = CanonicalDataset(allrows, img_size=img_size)
    bb = DinoV3(variant, img_size, pool).freeze()
    out = cached_extract(bb, ds, path, batch=batch, workers=workers)
    out["meta"] = allrows[["asset_id", "y", "condition_family", "condition_param"]]
    torch.save(out, path)
    return out


def select(bank: dict, mixture: dict, seed: int = 0) -> np.ndarray:
    """Row indices realising `mixture`: exactly one condition per training image.

    `mixture` gives the four cell proportions and, optionally, `families` -- a sub-mixture over
    the six transform families used when an image is assigned to a transformed cell.
    """
    meta = bank["meta"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fams = mixture.get("families")
    by_asset = meta.groupby("asset_id").indices

    picks = []
    for asset, idx in by_asset.items():
        sub = meta.iloc[idx]
        y = int(sub.y.iloc[0])
        cell = "ai" if y == 1 else "real"
        pt = mixture.get(f"{cell}_transformed", 0.0)
        pc = mixture.get(f"{cell}_clean", 0.0)
        p = pt / max(pt + pc, 1e-9)
        if rng.random() >= p:
            cand = idx[(sub.condition_family == "clean").values]
        else:
            if fams:
                names = list(fams)
                w = np.array([fams[k] for k in names], float)
                fam = names[int(rng.choice(len(names), p=w / w.sum()))]
                cand = idx[(sub.condition_family == fam).values]
            else:
                cand = idx[(sub.condition_family != "clean").values]
        if len(cand) == 0:
            cand = idx
        picks.append(int(rng.choice(cand)))
    return np.array(picks)


def evaluate_mixture(bank: dict, ev: dict, ev_meta: pd.DataFrame, mixture: dict,
                     seed: int = 0, epochs: int = 30, hidden: int = 0) -> dict:
    """Fit a probe on one mixture and score it on the fixed evaluation set."""
    idx = select(bank, mixture, seed)
    train = {"emb": bank["emb"][idx], "label": bank["label"][idx]}
    bbw = train["emb"].shape[1]
    head = Detector.__new__(Detector)          # head only; no backbone needed on the cache path
    from acai.models.heads import NoFusion
    head = NoFusion(bbw, hidden=hidden)
    fit_probe(train, head, epochs=epochs, seed=seed)
    scores = score_probe(head, ev)

    y = ev_meta.label.values
    overall = compute(y, scores)
    fam = by_group(y, scores, ev_meta["transform"].astype(str).values, min_n=30)
    wname, wval = worst_family(fam)
    clean = ev_meta["transform"].values == "clean"
    return {
        "mixture": mixture, "n_train": int(len(idx)),
        "auroc": overall.auroc, "auprc": overall.auprc,
        "tpr_at_fpr01": overall.tpr_at_fpr01,
        "clean_auroc": float(compute(y[clean], scores[clean]).auroc),
        "transformed_auroc": float(compute(y[~clean], scores[~clean]).auroc),
        "worst_family": wname, "worst_family_auroc": wval,
        "mean_family_auroc": float(np.mean([v["auroc"] for v in fam.values() if "auroc" in v])),
        "per_family": {k: v.get("auroc") for k, v in fam.items()},
    }


def dirichlet_mixtures(n: int, seed: int = 0, with_families: bool = False) -> list[dict]:
    """Random points on the cell simplex, plus fixed references.

    References are included so the search is interpretable: all-clean is the naive baseline and
    uniform is the obvious default. A random mixture that cannot beat those is not a finding.
    """
    rng = np.random.default_rng(seed)
    out = [
        {"real_clean": 1.0, "real_transformed": 0.0, "ai_clean": 1.0, "ai_transformed": 0.0},
        {"real_clean": 0.0, "real_transformed": 1.0, "ai_clean": 0.0, "ai_transformed": 1.0},
        {"real_clean": .5, "real_transformed": .5, "ai_clean": .5, "ai_transformed": .5},
    ]
    for _ in range(n):
        r = rng.dirichlet(np.ones(2))
        a = rng.dirichlet(np.ones(2))
        mx = {"real_clean": float(r[0]), "real_transformed": float(r[1]),
              "ai_clean": float(a[0]), "ai_transformed": float(a[1])}
        if with_families:
            w = rng.dirichlet(np.ones(len(FAMILY_ORDER)))
            mx["families"] = {f: float(x) for f, x in zip(FAMILY_ORDER, w)}
        out.append(mx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="data/build")
    ap.add_argument("--name", default="compose")
    ap.add_argument("--variant", default="b")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pool", default="cls_mean")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["cells", "families"], default="cells")
    ap.add_argument("--eval-split", default="dev")
    a = ap.parse_args()

    build = Path(a.build)
    with Run(a.name, vars(a)) as run:
        bank = build_bank(build, a.variant, a.img_size, a.pool)

        from acai.dataset import FrozenEvalDataset
        eds = FrozenEvalDataset(build / "frozen_eval", split=a.eval_split, img_size=a.img_size)
        ck = build / "cache" / f"eval_{a.variant}_{a.img_size}_{a.pool}_{a.eval_split}.pt"
        bb = DinoV3(a.variant, a.img_size, a.pool).freeze()
        ev = cached_extract(bb, eds, ck, batch=128, workers=12)
        ev_meta = eds.meta()

        mixes = dirichlet_mixtures(a.n, a.seed, with_families=(a.stage == "families"))
        results = []
        for i, mx in enumerate(mixes):
            r = evaluate_mixture(bank, ev, ev_meta, mx, seed=a.seed, epochs=a.epochs)
            results.append(r)
            run.log(mixture_index=i, **{k: v for k, v in r.items()
                                        if k not in ("mixture", "per_family")})
            print(f"[{i+1}/{len(mixes)}] auroc {r['auroc']:.4f}  "
                  f"worst {r['worst_family']} {r['worst_family_auroc']:.4f}", flush=True)

        df = pd.DataFrame(results)
        df.to_parquet(run.dir / "mixtures.parquet", index=False)
        best_mean = df.loc[df.auroc.idxmax()]
        best_worst = df.loc[df.worst_family_auroc.idxmax()]
        summary = {
            "n_mixtures": len(df),
            "best_by_overall_auroc": json.loads(best_mean.to_json()),
            "best_by_worst_family": json.loads(best_worst.to_json()),
            "note": ("the two optima usually differ; the worst-family optimum is what a "
                     "robustness claim rests on"),
        }
        run.metrics({"overall": {"auroc": float(best_mean.auroc)}, "composition": summary})
        print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
