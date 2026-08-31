"""Overnight experiment suite. Three tasks, run in order, each independently resumable.

    1  leave-one-generator-out on WildFake       (the one clean generalisation axis here)
    2  external evaluation on wildfake-eval-subset
    3  fusion ablation (H0-H3) + composition study, judged on 1 and 2

Design decisions carried in from the day's findings, all recorded in docs/RESULTS.md:

* **WildFake only for training.** SID is 0.9984-separable from a 32x32 thumbnail: its two
  classes differ in subject matter (web photos vs text-to-image art), so any model trained on
  it learns content, not generation. WildFake sits at 0.71 on the same test, so most of its
  signal is something else. SID is still *scored* (as a transfer target) but never trained on.
* **224 preprocessing.** The only intervention all day that moved a transfer number
  (WildFake->SID clean 0.812 -> 0.881).
* **Margin over the trivial cheat is reported everywhere.** On this corpus a two-integer
  manifest lookup scores 0.973-0.979, so a raw AUROC of 0.98 can mean a contribution of 0.008.
* **Judged on held-out generators, never in-distribution.** In-distribution AUROC on this
  corpus ranks how well a variant fits source artefacts.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from acai import data as D
from acai.dataset import CanonicalDataset, FrozenEvalDataset
from acai.metrics import by_group, compute, delong_test, worst_family
from acai.models.backbone import DinoV3
from acai.models.heads import Detector, FeatureNorm
from acai.runlog import Run
from acai.train import DEV, cached_extract, fit_probe, score_probe
from acai.transforms import Chain, FAMILY_ORDER, conditions

BUILD = Path("data/build_224")
OUT = Path("runs/overnight")
GENERATORS = ("adm", "ddim", "ddpm", "gan_based", "imagen")


def log(msg: str) -> None:
    print(f"[overnight] {msg}", flush=True)


def cheat_auroc(meta: pd.DataFrame) -> float:
    """AUROC of the best manifest-only cue on this eval population."""
    from sklearn.metrics import roc_auc_score
    best = 0.5
    for v in [(meta.orig_width * meta.orig_height).values.astype(float),
              (meta.orig_width == meta.orig_height).astype(float).values,
              (meta.orig_width / meta.orig_height).values.astype(float)]:
        if len(np.unique(v)) > 1:
            a = roc_auc_score(meta.label, v)
            best = max(best, a, 1 - a)
    return float(best)


# --------------------------------------------------------------- shared embedding bank

def wildfake_bank(variant="b", img_size=224, pool="cls_mean", batch=128, workers=12) -> dict:
    """Embeddings for every (WildFake training image, official condition) pair.

    Extracted once; every experiment below is a row-selection from this. Without it the
    fusion ablation and composition study would each need their own forward passes.
    """
    path = BUILD / "cache" / f"wfbank_{variant}_{img_size}_{pool}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    m = pd.read_parquet(BUILD / "canonical_manifest.parquet")
    m["y"] = m.y.astype(int)
    tr = m[(m.split == "train") & (m.source_dataset == "wildfake")].reset_index(drop=True)
    rows = []
    for fam, param in conditions():
        c = tr.copy()
        c["chain"] = [Chain() if fam == "clean" else Chain(((fam, param),))] * len(c)
        c["condition_family"], c["condition_param"] = fam, float(param)
        rows.append(c)
    allrows = pd.concat(rows, ignore_index=True)
    log(f"building WildFake bank: {len(allrows)} rows ({len(tr)} imgs x {len(conditions())})")
    ds = CanonicalDataset(allrows, img_size=img_size, want_feats=True)
    bb = DinoV3(variant, img_size, pool).freeze()
    out = cached_extract(bb, ds, path, batch=batch, workers=workers, want_feats=True)
    out["meta"] = allrows[["asset_id", "y", "generator", "authentic_subtype",
                           "condition_family", "condition_param"]].reset_index(drop=True)
    torch.save(out, path)
    del bb; gc.collect(); torch.cuda.empty_cache()
    return out


def frozen_eval_cache(variant="b", img_size=224, pool="cls_mean", source=None,
                      batch=128, workers=12):
    tag = source or "all"
    path = BUILD / "cache" / f"ovn_eval_{variant}_{img_size}_{pool}_{tag}.pt"
    ds = FrozenEvalDataset(BUILD / "frozen_eval", img_size=img_size, want_feats=True)
    if source:
        ds.df = ds.df[ds.df.source_dataset == source].reset_index(drop=True)
    meta = ds.meta()
    if path.exists():
        return torch.load(path, weights_only=False), meta
    bb = DinoV3(variant, img_size, pool).freeze()
    c = cached_extract(bb, ds, path, batch=batch, workers=workers, want_feats=True)
    del bb; gc.collect(); torch.cuda.empty_cache()
    return c, meta


def fit_and_score(train: dict, evc: dict, fusion="none", hidden=0, epochs=40, seed=0):
    """Fit one head on cached embeddings, return eval scores."""
    d = train["emb"].shape[1]
    head = Detector.__new__(Detector)
    from acai.models.heads import HEADS
    head = HEADS[fusion](d, hidden or 256) if fusion != "none" else HEADS["none"](d, hidden)
    if fusion in ("h0", "h1", "h3"):
        for mod in head.modules():
            if isinstance(mod, FeatureNorm):
                mod.fit(train["feats"])
    fit_probe(train, head, epochs=epochs, seed=seed)
    return score_probe(head, evc)


def summarise(y, s, meta, label="") -> dict:
    m = compute(y, s)
    fam = by_group(y, s, meta["transform"].astype(str).values, min_n=30)
    wn, wv = worst_family(fam)
    clean = meta["transform"].values == "clean"
    d = m.as_dict()
    d.update(clean_auroc=float(compute(y[clean], s[clean]).auroc) if clean.sum() > 30 else None,
             worst_family=wn, worst_family_auroc=wv,
             per_family={k: v.get("auroc") for k, v in fam.items()},
             cheat_auroc=cheat_auroc(meta))
    if d["clean_auroc"] is not None:
        d["margin_over_cheat"] = d["clean_auroc"] - d["cheat_auroc"]
    d["label"] = label
    return d


# ============================================================== TASK 1: leave-one-generator-out

def task1_logo(bank: dict, epochs=40, seeds=(0, 1, 2)) -> dict:
    """Hold out one WildFake generator at a time; train on the rest; score on the held-out one.

    Reals are never held out -- "unseen ADM" means the model never saw ADM images, not that it
    never saw a real image. The held-out generator is scored against the same WildFake reals
    used everywhere else, so folds are mutually comparable.

    Three seeds per fold: with 350 training images per generator, single-seed differences
    between folds would be mostly initialisation noise.
    """
    log("TASK 1: leave-one-generator-out on WildFake")
    meta_tr = bank["meta"]
    evc, ev_meta = frozen_eval_cache(source="wildfake")
    y_ev = ev_meta.label.values

    results = {}
    for gen in GENERATORS:
        keep = (meta_tr.generator != gen).values          # drop that generator from training
        held = (ev_meta.generator == gen).values
        evalmask = held | (ev_meta.label == 0).values      # held-out gen vs all WildFake reals
        per_seed = []
        for sd in seeds:
            tr = {k: (v[keep] if hasattr(v, "shape") else v)
                  for k, v in bank.items() if k in ("emb", "label", "feats")}
            s = fit_and_score(tr, evc, epochs=epochs, seed=sd)
            per_seed.append(summarise(y_ev[evalmask], s[evalmask],
                                      ev_meta[evalmask].reset_index(drop=True), f"{gen}/seed{sd}"))
        aur = [p["auroc"] for p in per_seed]
        results[gen] = {
            "per_seed": per_seed,
            "auroc_mean": float(np.mean(aur)), "auroc_std": float(np.std(aur)),
            "clean_auroc_mean": float(np.mean([p["clean_auroc"] for p in per_seed])),
            "margin_mean": float(np.mean([p["margin_over_cheat"] for p in per_seed])),
            "worst_family": per_seed[0]["worst_family"],
            "n_eval": int(evalmask.sum()),
        }
        log(f"  {gen:10s} AUROC {results[gen]['auroc_mean']:.4f} "
            f"+/- {results[gen]['auroc_std']:.4f}  clean {results[gen]['clean_auroc_mean']:.4f}"
            f"  margin {results[gen]['margin_mean']:+.4f}")

    aur = [v["auroc_mean"] for v in results.values()]
    summary = {"per_generator": results,
               "mean_auroc": float(np.mean(aur)), "std_auroc": float(np.std(aur)),
               "worst_generator": min(results, key=lambda k: results[k]["auroc_mean"]),
               "worst_auroc": float(np.min(aur)),
               "note": ("mean over 5 held-out generators, 3 seeds each; this is the "
                        "'unseen generator' row of the scorecard")}
    log(f"  MEAN {summary['mean_auroc']:.4f}  WORST {summary['worst_generator']} "
        f"{summary['worst_auroc']:.4f}")
    return summary


# ================================================== TASK 2: external eval (wildfake-eval-subset)

EVAL_ROOT = Path("data/eval_subset")
EVAL_CONFIGS = {
    # `default` is deliberately excluded from headline reporting: its dataset card states
    # every real is exactly 200x200 and no fake is, so image size alone scores AUROC 1.000.
    # It is scored anyway, purely to demonstrate that shortcut is still live.
    "laion_matched": "the config the card recommends: both classes natively >=1024px, one "
                     "identical downscale, balanced 50/50",
    "cross_generator": "1,500 LAION reals vs DALL-E 3 / Midjourney v5 / SDXL / GigaGAN",
    "normalized": "size shortcut removed by 200x200 downscale; high-frequency detail destroyed",
    "default": "SPEC-COMPLIANCE ONLY -- size alone gives AUROC 1.000, never report alone",
}


def load_external(config: str) -> pd.DataFrame | None:
    d = EVAL_ROOT / config if config != "default" else EVAL_ROOT / "data"
    shards = sorted(d.glob("*.parquet"))
    if not shards:
        log(f"  {config}: no shards found, skipping")
        return None
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    return df


def check_leakage(ext: pd.DataFrame, config: str) -> dict:
    """Our training data and this eval set both derive from WildFake. Verify no overlap.

    data_draft states COCO and DALL-E families are excluded and uses adm/ddim/ddpm/imagen/
    gan_based against afhq/celebahq/church/ffhq/imagenet, while the eval subset uses COCO and
    LAION reals against DALLE3/Midjourney/SDXL/GigaGAN -- disjoint by construction. We check
    content hashes anyway, because a documented disjointness is not a verified one.
    """
    import hashlib
    m = pd.read_parquet(BUILD / "canonical_manifest.parquet")
    train_sha = set(m[m.split == "train"].sha256.astype(str))
    hits = 0
    sample = ext.sample(min(3000, len(ext)), random_state=0)
    for b in sample["image"]:
        raw = b["bytes"] if isinstance(b, dict) else b
        if hashlib.sha256(raw).hexdigest() in train_sha:
            hits += 1
    return {"config": config, "checked": int(len(sample)), "sha256_collisions": hits,
            "clean": hits == 0}


def _render_external(args):
    """Apply one official condition to one external image, at 224, matching our pipeline."""
    raw, aid, fam, param = args
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                          (im.width + s) // 2, (im.height + s) // 2)).resize((224, 224),
                                                                             Image.BICUBIC)
        from acai.transforms import apply_chain
        chain = Chain() if fam == "clean" else Chain(((fam, param),))
        return np.asarray(apply_chain(im, chain, aid), dtype=np.uint8)
    except Exception:
        return None


class ExternalDataset(torch.utils.data.Dataset):
    """External eval images + the 15 official conditions, preprocessed exactly like training."""

    def __init__(self, df: pd.DataFrame, img_size=224, want_feats=True, max_images=None):
        from acai.features.lowlevel import feature_vector
        self.fv = feature_vector
        if max_images and len(df) > max_images:
            # Index-based balanced subsample. NOT groupby.apply: pandas 2.x excludes the
            # grouping column from each group frame, which silently drops `label`.
            rng = np.random.default_rng(0)
            take = []
            for lab in sorted(df["label"].unique()):
                idx = df.index[df["label"] == lab].to_numpy()
                k = min(len(idx), max_images // max(df["label"].nunique(), 1))
                take.append(rng.choice(idx, k, replace=False))
            df = df.loc[np.concatenate(take)].reset_index(drop=True)
        self.rows = []
        for i, r in enumerate(df.itertuples()):
            raw = r.image["bytes"] if isinstance(r.image, dict) else r.image
            for fam, param in conditions():
                self.rows.append((raw, f"ext{i}", fam, float(param), int(r.label),
                                  getattr(r, "source", "")))
        self.img_size, self.want_feats = img_size, want_feats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw, aid, fam, param, label, src = self.rows[i]
        img = _render_external((raw, aid, fam, param))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        from acai.dataset import _normalise
        out = {"image": _normalise(img, self.img_size), "label": torch.tensor(label),
               "id": f"{aid}|{fam}:{param:g}"}
        if self.want_feats:
            out["feats"] = torch.from_numpy(self.fv(img))
        return out

    def meta(self) -> pd.DataFrame:
        return pd.DataFrame([{"id": f"{a}|{f}:{p:g}", "group": a, "label": l, "split": "external",
                              "source": s or "external", "source_dataset": "external",
                              "generator": "", "authentic_subtype": "",
                              "transform": f, "severity": p,
                              "orig_width": 224, "orig_height": 224}
                             for _, a, f, p, l, s in self.rows])


def task2_external(bank: dict, max_images=2400, epochs=40) -> dict:
    """Train on WildFake (all generators), score on the external benchmark configs."""
    log("TASK 2: external evaluation on techjam-aigc/wildfake-eval-subset")
    results = {}
    tr = {k: v for k, v in bank.items() if k in ("emb", "label", "feats")}

    for cfg, note in EVAL_CONFIGS.items():
        ext = load_external(cfg)
        if ext is None:
            results[cfg] = {"skipped": "shards not present"}
            continue
        leak = check_leakage(ext, cfg)
        log(f"  {cfg}: {len(ext)} rows, leakage check {leak}")
        ds = ExternalDataset(ext, max_images=max_images)
        meta = ds.meta()
        cache = BUILD / "cache" / f"ext_{cfg}_{max_images}.pt"
        bb = DinoV3("b", 224, "cls_mean").freeze()
        evc = cached_extract(bb, ds, cache, batch=128, workers=12, want_feats=True)
        del bb; gc.collect(); torch.cuda.empty_cache()

        s = fit_and_score(tr, evc, epochs=epochs)
        y = meta.label.values
        r = summarise(y, s, meta, cfg)
        r.update(note=note, leakage=leak, n_source_images=int(len(ds.rows) // len(conditions())))
        if cfg == "default":
            r["WARNING"] = ("size alone scores AUROC 1.000 on this config per its dataset card; "
                            "reported for spec compliance only")
        results[cfg] = r
        log(f"    AUROC {r['auroc']:.4f}  clean {r['clean_auroc']:.4f}  "
            f"worst {r['worst_family']} {r['worst_family_auroc']:.4f}")
    return results


# ============================ TASK 1b: leave-one-generator-out scored on EXTERNAL data

def task1b_logo_external(bank: dict, config="laion_matched", epochs=40) -> dict:
    """LOGO, but scored on the external benchmark instead of WildFake's own eval split.

    Why this exists: inside WildFake the internal LOGO evaluation is size-confounded. Every
    real is 200x200 and each generator has one fixed size (adm/ddim/ddpm 256x256,
    imagen 512x512), so a manifest-only size lookup scores **1.000 on four of the five folds**.
    A model can therefore look strong there while contributing nothing.

    Scoring the same held-out-generator models against LAION/COCO reals, whose sizes vary,
    removes that particular cue entirely. The held-out generator is not present in the external
    set, so this measures how much each generator contributes to *general* transfer rather than
    per-generator recall -- a different and, here, more trustworthy question.
    """
    log(f"TASK 1b: LOGO models scored on external `{config}`")
    ext = load_external(config)
    if ext is None:
        return {"skipped": f"{config} shards not present"}
    ds = ExternalDataset(ext, max_images=2400)
    meta = ds.meta()
    cache = BUILD / "cache" / f"ext_{config}_2400.pt"
    bb = DinoV3("b", 224, "cls_mean").freeze()
    evc = cached_extract(bb, ds, cache, batch=128, workers=12, want_feats=True)
    del bb; gc.collect(); torch.cuda.empty_cache()
    y = meta.label.values

    meta_tr = bank["meta"]
    out = {}
    full = {k: v for k, v in bank.items() if k in ("emb", "label", "feats")}
    out["all_generators"] = summarise(y, fit_and_score(full, evc, epochs=epochs), meta, "all")
    for gen in GENERATORS:
        keep = (meta_tr.generator != gen).values
        tr = {k: v[keep] for k, v in bank.items() if k in ("emb", "label", "feats")}
        r = summarise(y, fit_and_score(tr, evc, epochs=epochs), meta, f"minus_{gen}")
        out[f"minus_{gen}"] = r
        log(f"  without {gen:10s} external AUROC {r['auroc']:.4f} "
            f"(delta {r['auroc'] - out['all_generators']['auroc']:+.4f})")
    out["note"] = ("external eval has no 200x200-vs-generator size cue, so these numbers are "
                   "not inflated by the confound that makes internal LOGO look strong")
    return out


# ================================== TASK 3a: fusion ablation (H0-H3), judged on held-out generators

def logo_score(bank: dict, evc, ev_meta, fusion="none", hidden=0, epochs=40,
               seeds=(0, 1), select=None) -> dict:
    """Mean AUROC over the 5 leave-one-generator-out folds, for one head configuration.

    This is the objective for every comparison in task 3. In-distribution AUROC is not used:
    on this corpus it ranks how well a variant fits source artefacts.
    """
    meta_tr = bank["meta"]
    y_ev = ev_meta.label.values
    folds, all_scores = {}, {}
    for gen in GENERATORS:
        keep = (meta_tr.generator != gen).values
        if select is not None:
            keep = keep & select
        evalmask = ((ev_meta.generator == gen) | (ev_meta.label == 0)).values
        aur, sc = [], None
        for sd in seeds:
            tr = {k: v[keep] for k, v in bank.items() if k in ("emb", "label", "feats")}
            s = fit_and_score(tr, evc, fusion=fusion, hidden=hidden, epochs=epochs, seed=sd)
            aur.append(compute(y_ev[evalmask], s[evalmask]).auroc)
            sc = s if sc is None else sc + s
        folds[gen] = float(np.mean(aur))
        all_scores[gen] = (sc / len(seeds), evalmask)
    return {"per_generator": folds, "mean": float(np.mean(list(folds.values()))),
            "worst": float(np.min(list(folds.values()))),
            "worst_generator": min(folds, key=folds.get), "_scores": all_scores}


def task3a_fusion(bank: dict, epochs=40) -> dict:
    """Pure DINOv3 vs H0/H1/H3, DeLong-tested on the pooled held-out-generator predictions.

    H2 (dense forensic tokens) is excluded here: it injects per-patch maps inside the backbone,
    so it cannot run on cached embeddings and needs a full finetune. It is reported as deferred
    rather than silently dropped.
    """
    log("TASK 3a: fusion ablation (pure DINOv3 vs H0/H1/H3), judged on held-out generators")
    evc, ev_meta = frozen_eval_cache(source="wildfake")
    y_ev = ev_meta.label.values

    arms = {}
    for fusion in ("none", "h0", "h1", "h3"):
        r = logo_score(bank, evc, ev_meta, fusion=fusion, hidden=(0 if fusion == "none" else 256),
                       epochs=epochs)
        arms[fusion] = r
        log(f"  {fusion:5s} mean {r['mean']:.4f}  worst {r['worst_generator']} {r['worst']:.4f}")

    # Paired DeLong against the pure baseline, on the concatenated held-out-generator rows.
    base = arms["none"]["_scores"]
    tests = {}
    for fusion in ("h0", "h1", "h3"):
        ya, sa, sb = [], [], []
        for gen in GENERATORS:
            s_arm, mask = arms[fusion]["_scores"][gen]
            s_base, _ = base[gen]
            ya.append(y_ev[mask]); sa.append(s_arm[mask]); sb.append(s_base[mask])
        tests[fusion] = delong_test(np.concatenate(ya), np.concatenate(sa), np.concatenate(sb))
        log(f"  DeLong {fusion} vs none: diff {tests[fusion]['diff']:+.4f} "
            f"p={tests[fusion]['p']:.3g}")

    for a in arms.values():
        a.pop("_scores", None)
    return {"arms": arms, "delong_vs_pure": tests,
            "h2_status": ("deferred: dense forensic-token injection cannot run on cached "
                          "embeddings; needs a full finetune pass"),
            "objective": "mean AUROC over 5 leave-one-generator-out folds, 2 seeds each"}


# ================================================ TASK 3b: training-data composition study

def task3b_composition(bank: dict, n_mixtures=40, epochs=30, seed=0) -> dict:
    """Which training composition maximises held-out-generator AUROC?

    Two axes, both asked of the *training* set with the evaluation held fixed:
      * clean vs transformed, separately for real and AI rows
      * the proportion of each transform family within the transformed part

    Training size is held constant at one row per training image for every mixture, so a
    result is never just 'more data'.
    """
    log(f"TASK 3b: composition study ({n_mixtures} mixtures + references)")
    evc, ev_meta = frozen_eval_cache(source="wildfake")
    meta_tr = bank["meta"].reset_index(drop=True)
    by_asset = meta_tr.groupby("asset_id").indices
    rng = np.random.default_rng(seed)

    def select_rows(mx, sd):
        r = np.random.default_rng(sd)
        fams = mx.get("families")
        picks = []
        for asset, idx in by_asset.items():
            sub = meta_tr.iloc[idx]
            cell = "ai" if int(sub.y.iloc[0]) == 1 else "real"
            pt, pc = mx.get(f"{cell}_transformed", 0.0), mx.get(f"{cell}_clean", 0.0)
            if r.random() >= pt / max(pt + pc, 1e-9):
                cand = idx[(sub.condition_family == "clean").values]
            elif fams:
                names = list(fams); w = np.array([fams[k] for k in names], float)
                fam = names[int(r.choice(len(names), p=w / w.sum()))]
                cand = idx[(sub.condition_family == fam).values]
            else:
                cand = idx[(sub.condition_family != "clean").values]
            picks.append(int(r.choice(cand if len(cand) else idx)))
        sel = np.zeros(len(meta_tr), bool); sel[np.array(picks)] = True
        return sel

    mixes = [
        ("all_clean", {"real_clean": 1, "real_transformed": 0, "ai_clean": 1, "ai_transformed": 0}),
        ("all_transformed", {"real_clean": 0, "real_transformed": 1, "ai_clean": 0, "ai_transformed": 1}),
        ("uniform_50_50", {"real_clean": .5, "real_transformed": .5, "ai_clean": .5, "ai_transformed": .5}),
    ]
    for i in range(n_mixtures):
        r, a = rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2))
        mx = {"real_clean": float(r[0]), "real_transformed": float(r[1]),
              "ai_clean": float(a[0]), "ai_transformed": float(a[1])}
        if i >= n_mixtures // 2:                       # half the budget also samples families
            w = rng.dirichlet(np.ones(len(FAMILY_ORDER)))
            mx["families"] = {f: float(x) for f, x in zip(FAMILY_ORDER, w)}
        mixes.append((f"dirichlet_{i}", mx))

    results = []
    for name, mx in mixes:
        sel = select_rows(mx, seed)
        r = logo_score(bank, evc, ev_meta, epochs=epochs, seeds=(0,), select=sel)
        r.pop("_scores", None)
        results.append({"name": name, "mixture": mx, "n_train": int(sel.sum()), **r})
        log(f"  {name:16s} mean {r['mean']:.4f}  worst {r['worst']:.4f}")

    df = pd.DataFrame(results)
    best_mean = df.loc[df["mean"].idxmax()]
    best_worst = df.loc[df["worst"].idxmax()]
    return {"n_mixtures": len(df),
            "references": {r["name"]: {"mean": r["mean"], "worst": r["worst"]}
                           for r in results[:3]},
            "best_by_mean": json.loads(best_mean.to_json()),
            "best_by_worst_generator": json.loads(best_worst.to_json()),
            "all": results,
            "note": ("training size held constant at one row per image across every mixture, "
                     "so no result is merely more data")}


# ================================================================================== driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1,2,1b,3a,3b")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-mixtures", type=int, default=40)
    ap.add_argument("--max-ext-images", type=int, default=2400)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    want = set(a.tasks.split(","))
    results, errors = {}, {}

    with Run("overnight", vars(a), root=OUT) as run:
        bank = wildfake_bank()
        log(f"bank ready: {bank['emb'].shape}")

        stages = [
            ("1", "task1_logo", lambda: task1_logo(bank, epochs=a.epochs)),
            ("2", "task2_external", lambda: task2_external(bank, a.max_ext_images, a.epochs)),
            ("1b", "task1b_logo_external", lambda: task1b_logo_external(bank, epochs=a.epochs)),
            ("3a", "task3a_fusion", lambda: task3a_fusion(bank, epochs=a.epochs)),
            ("3b", "task3b_composition",
             lambda: task3b_composition(bank, a.n_mixtures, epochs=max(20, a.epochs // 2))),
        ]
        for key, name, fn in stages:
            if key not in want:
                continue
            # Each stage is isolated: an overnight run must not lose three finished tasks
            # because the fourth raised.
            try:
                results[name] = fn()
            except Exception as e:
                errors[name] = {"error": repr(e), "traceback": traceback.format_exc()}
                log(f"!! {name} FAILED: {e!r}")
            (OUT / "results.json").write_text(
                json.dumps({"results": results, "errors": errors}, indent=2, default=str))
            run.log(stage=name, done=True, failed=name in errors)

        run.metrics({"overall": {"auroc": results.get("task1_logo", {}).get("mean_auroc", 0.0)},
                     "tasks_completed": sorted(results), "tasks_failed": sorted(errors)})
        (OUT / "results.json").write_text(
            json.dumps({"results": results, "errors": errors}, indent=2, default=str))
    log(f"ALL DONE. completed={sorted(results)} failed={sorted(errors)}")


if __name__ == "__main__":
    main()
