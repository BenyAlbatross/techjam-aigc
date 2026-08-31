"""Train one detector and emit predictions the sealed scorecard can read.

Two paths, chosen by `--mode`:

* **probe** (default) -- freeze the backbone, cache its embeddings once, fit a head on the
  cache. The cache is the point: the composition study (plan §4) evaluates 60-100 training
  mixtures, and every mixture is a different *subset of the same embeddings*. Extracting once
  turns a 100-run sweep from days into minutes, and the ranking it produces is what the
  expensive finetunes then confirm.
* **finetune** -- LoRA or last-block updates through the backbone. Used for the top mixtures and
  for the H2 arm, whose forensic side-channel only means something if the blocks above it adapt.

Whatever the path, the output contract is identical: one `predictions.parquet` per run carrying
every column `acai.scorecard` requires, so the ruler never has to know how the model was made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from acai import data as D
from acai.dataset import CanonicalDataset, FrozenEvalDataset
from acai.models.backbone import DinoV3
from acai.models.heads import Detector
from acai.runlog import Run
from acai.transforms import Chain, conditions, random_chain

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ embedding cache

def cache_key(variant: str, img_size: int, pool: str, tag: str) -> str:
    return hashlib.sha256(f"{variant}|{img_size}|{pool}|{tag}".encode()).hexdigest()[:16]


@torch.no_grad()
def extract(backbone: DinoV3, ds, batch: int = 64, workers: int = 8,
            want_feats: bool = False) -> dict:
    """Run the frozen backbone once over a dataset; return embeddings (+ handcrafted feats)."""
    backbone.eval().to(DEV)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    embs, feats, labels, ids = [], [], [], []
    for b in dl:
        x = b["image"].to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            embs.append(backbone(x).float().cpu())
        labels.append(b["label"])
        ids += list(b["id"])
        if want_feats and "feats" in b:
            feats.append(b["feats"])
    out = {"emb": torch.cat(embs), "label": torch.cat(labels), "id": ids}
    if feats:
        out["feats"] = torch.cat(feats)
    return out


def cached_extract(backbone, ds, path: Path, **kw) -> dict:
    if path.exists():
        return torch.load(path, weights_only=False)
    out = extract(backbone, ds, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


# ---------------------------------------------------------------------- train paths

def fit_probe(train: dict, head: nn.Module, epochs: int = 40, lr: float = 1e-3,
              wd: float = 1e-4, batch: int = 512, run: Run | None = None,
              seed: int = 0) -> nn.Module:
    """Fit a head on cached embeddings."""
    seed_all(seed)
    head.to(DEV).train()
    X, y = train["emb"].to(DEV), train["label"].float().to(DEV)
    F = train.get("feats")
    F = F.to(DEV) if F is not None else None
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            logits, extra = head(X[j], feats=None if F is None else F[j])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[j])
            if "aux_pred" in extra:
                loss = loss + extra["aux_weight"] * nn.functional.mse_loss(
                    extra["aux_pred"], extra["aux_target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if run and (ep % 5 == 0 or ep == epochs - 1):
            run.log(epoch=ep, loss=tot / n, lr=sched.get_last_lr()[0])
    return head.eval()


@torch.no_grad()
def score_probe(head: nn.Module, cache: dict, batch: int = 1024) -> np.ndarray:
    head.eval().to(DEV)
    X = cache["emb"].to(DEV)
    F = cache.get("feats")
    F = F.to(DEV) if F is not None else None
    out = []
    for i in range(0, len(X), batch):
        logits, _ = head(X[i:i + batch], feats=None if F is None else F[i:i + batch])
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def finetune(model: Detector, train_ds, epochs: int, lr: float, wd: float, batch: int,
             workers: int, run: Run | None = None, seed: int = 0) -> Detector:
    seed_all(seed)
    model.to(DEV).train()
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * len(dl))
    step = 0
    for ep in range(epochs):
        for b in dl:
            x = b["image"].to(DEV, non_blocking=True)
            y = b["label"].to(DEV)
            f = b.get("feats"); f = f.to(DEV) if f is not None else None
            fm = b.get("feat_maps"); fm = fm.to(DEV) if fm is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                logits, extra = model(x, f, fm)
                loss = model.loss(logits.float(), y, extra)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if run and step % 50 == 0:
                run.log(step=step, epoch=ep, loss=float(loss),
                        lr=sched.get_last_lr()[0], grad_norm=float(gn))
    return model.eval()


@torch.no_grad()
def score_model(model: Detector, ds, batch: int, workers: int) -> np.ndarray:
    model.eval().to(DEV)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    for b in dl:
        x = b["image"].to(DEV, non_blocking=True)
        f = b.get("feats"); f = f.to(DEV) if f is not None else None
        fm = b.get("feat_maps"); fm = fm.to(DEV) if fm is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits, _ = model(x, f, fm)
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


# -------------------------------------------------------------------- training rows

def training_rows(m: pd.DataFrame, mixture: dict | None = None, seed: int = 0,
                  tier: str = "T1") -> pd.DataFrame:
    """Assign each training image a transform chain.

    `mixture` maps a cell name to a proportion; cells are the four in plan §4
    (`real_clean`, `real_transformed`, `ai_clean`, `ai_transformed`), optionally with a
    `families` sub-mixture over the six transform families. Absent, every image gets `clean`.

    Total row count is held constant across mixtures by construction -- one row per training
    image -- so a composition comparison never becomes a data-quantity comparison.
    """
    rng = np.random.default_rng(seed)
    tr = m[m.split == "train"].copy()
    if not mixture:
        tr["chain"] = [Chain()] * len(tr)
        return tr

    fams = mixture.get("families")
    conds = conditions(include_clean=False)
    chains = []
    for r in tr.itertuples():
        cell = ("ai" if r.y == 1 else "real")
        p_trans = mixture.get(f"{cell}_transformed", 0.0)
        p_trans = p_trans / max(p_trans + mixture.get(f"{cell}_clean", 0.0), 1e-9)
        if rng.random() >= p_trans:
            chains.append(Chain())
        elif tier == "T3":
            chains.append(random_chain(r.asset_id, rng))
        else:
            if fams:
                names = list(fams); w = np.array([fams[k] for k in names], float)
                fam = names[int(rng.choice(len(names), p=w / w.sum()))]
                opts = [c for c in conds if c[0] == fam]
            else:
                opts = conds
            chains.append(Chain((opts[int(rng.integers(len(opts)))],)))
    tr["chain"] = chains
    return tr


# --------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="data/build")
    ap.add_argument("--name", default="run")
    ap.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    ap.add_argument("--variant", default="b", choices=["s", "b", "l"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pool", default="cls_mean")
    ap.add_argument("--fusion", default="none", choices=["none", "h0", "h1", "h2", "h3"])
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixture", default=None, help="JSON cell mixture for the composition study")
    ap.add_argument("--tier", default="T1", choices=["T1", "T3"])
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--train-source", default=None, choices=[None, "sid_set", "wildfake"],
                    help="restrict TRAINING to one source dataset")
    ap.add_argument("--eval-source", default=None, choices=[None, "sid_set", "wildfake"],
                    help="restrict EVALUATION to one source dataset")
    ap.add_argument("--train-exclude-generator", default=None,
                    help="hold one named generator out of training (unseen-generator probe)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="subsample training rows AFTER extraction; used to control for the "
                         "data-quantity difference a source holdout introduces")
    a = ap.parse_args()

    build = Path(a.build)
    cfg = vars(a) | {"device": DEV}
    needs_feats = a.fusion in ("h0", "h1", "h3")
    needs_maps = a.fusion == "h2"

    with Run(a.name, cfg) as run:
        m = pd.read_parquet(build / "canonical_manifest.parquet")
        m["y"] = m["y"].astype(int)
        mixture = json.loads(a.mixture) if a.mixture else None
        tr = training_rows(m, mixture, a.seed, a.tier)
        # Generalisation holdouts. Applied to TRAINING only; the evaluation set stays frozen
        # and identical, so a transfer number is directly comparable to the in-distribution
        # one rather than being measured on a different eval population.
        if a.train_source:
            tr = tr[tr.source_dataset == a.train_source].reset_index(drop=True)
        if a.train_exclude_generator:
            tr = tr[tr.generator != a.train_exclude_generator].reset_index(drop=True)

        train_ds = CanonicalDataset(tr, img_size=a.img_size, want_feats=needs_feats,
                                    want_maps=needs_maps)
        eval_ds = FrozenEvalDataset(build / "frozen_eval", img_size=a.img_size,
                                    want_feats=needs_feats, want_maps=needs_maps)
        if a.eval_source:
            eval_ds.df = eval_ds.df[
                eval_ds.df.source_dataset == a.eval_source].reset_index(drop=True)
        run.write("manifest.json", {
            "n_train": len(train_ds), "n_eval": len(eval_ds), "mixture": mixture,
            "tier": a.tier, "train_source": a.train_source, "eval_source": a.eval_source,
            "train_exclude_generator": a.train_exclude_generator,
            "train_sources": tr.source_dataset.value_counts().to_dict(),
            "train_chains": tr.chain.map(lambda c: c.name).value_counts().to_dict()})

        backbone = DinoV3(a.variant, a.img_size, a.pool)

        if a.mode == "probe":
            backbone.freeze()
            # The holdout MUST be in the cache key. Without it a transfer run would silently
            # reuse the full-training cache and the holdout would have no effect at all --
            # a failure that produces a plausible number rather than an error.
            ck = cache_key(a.variant, a.img_size, a.pool,
                           f"{a.train_source}|{a.train_exclude_generator}")
            tr_cache = cached_extract(backbone, train_ds,
                                      build / "cache" / f"train_{ck}_{a.seed}_{a.tier}.pt",
                                      batch=a.batch, workers=a.workers, want_feats=needs_feats)
            evck = cache_key(a.variant, a.img_size, a.pool, f"eval|{a.eval_source}")
            ev_cache = cached_extract(backbone, eval_ds,
                                      build / "cache" / f"eval_{evck}.pt",
                                      batch=a.batch, workers=a.workers, want_feats=needs_feats)
            if a.train_frac < 1.0:
                # Subsample the cache, not the dataset, so no re-extraction is needed. This
                # exists to answer the obvious objection to a source-holdout result: that the
                # drop is just half the training data rather than the distribution shift.
                k = int(len(tr_cache["emb"]) * a.train_frac)
                sel = np.random.default_rng(a.seed).permutation(len(tr_cache["emb"]))[:k]
                tr_cache = {kk: (v[sel] if hasattr(v, "__getitem__")
                                 and not isinstance(v, list) else v)
                            for kk, v in tr_cache.items() if kk != "id"}
            head = Detector(backbone, a.fusion, hidden=a.hidden or 256).head
            if needs_feats and "feats" in tr_cache:
                from acai.models.heads import FeatureNorm
                for mod in head.modules():
                    if isinstance(mod, FeatureNorm):
                        mod.fit(tr_cache["feats"])
            fit_probe(tr_cache, head, a.epochs, a.lr, a.wd, run=run, seed=a.seed)
            scores = score_probe(head, ev_cache)
        else:
            if a.lora:
                backbone.add_lora(rank=a.lora)
            else:
                backbone.unfreeze_last(2)
            model = Detector(backbone, a.fusion, hidden=a.hidden or 256)
            if needs_feats:
                from acai.features.lowlevel import feature_vector
                sample = torch.stack([train_ds[i]["feats"] for i in
                                      np.random.default_rng(0).integers(0, len(train_ds), 512)])
                model.fit_feature_norm(sample)
            finetune(model, train_ds, a.epochs, a.lr, a.wd, a.batch, a.workers, run, a.seed)
            scores = score_model(model, eval_ds, a.batch, a.workers)

        meta = eval_ds.meta()
        meta["score"] = scores
        run.predictions(meta)

        from acai.scorecard import build as build_card, to_markdown
        card = build_card(run.dir / "predictions.parquet")
        run.metrics({"overall": card["1_overall"], "robustness": card["3_robustness"]})
        (run.dir / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
        (run.dir / "scorecard.md").write_text(to_markdown(card))
        print(to_markdown(card))


if __name__ == "__main__":
    main()
