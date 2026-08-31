"""Evaluate the frozen TRACE-RX-M v2 detector on both sources of data_draft.

Framing note: TRACE-RX-M ships pre-trained on a different corpus, so "train on WildFake, test on
SID" does not apply to it -- there is no training step. The equivalent question is whether an
independently-trained detector generalises to BOTH of our sources, which is what a genuinely
transferable detector would do. Clean images only, as requested.

Preprocessing is the repo's own `canonical_preprocess`: square BILINEAR resize to 224 with
ImageNet normalisation, applied to the ORIGINAL files -- not our centre-cropped canonical copies,
which would be a different pipeline from the one the checkpoint expects.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent / "feat_traincode" / "src"))
sys.path.insert(0, "/home/joel/Desktop/acai/src")
from techjam_aigc.trace_rx_m.training import load_detector_checkpoint
from techjam_aigc.trace_rx_m.augment import canonical_preprocess
from acai import data as D
from acai.metrics import compute

ROOT = Path("/home/joel/Desktop/acai/data/data_draft")
CKPT = Path("/home/joel/.cache/huggingface/hub/models--techjam-aigc--trace-rx-m-v2/"
            "snapshots/3d4270323bbb2de437326fa59e436a1540da1110")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Raw(Dataset):
    def __init__(self, rows): self.rows = rows.reset_index(drop=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        with Image.open(ROOT / r.path) as im:
            x = canonical_preprocess(im, image_size=224)
        return torch.from_numpy(x), int(r.y)


@torch.no_grad()
def score(model, rows, bs=64):
    dl = DataLoader(Raw(rows), batch_size=bs, num_workers=10, pin_memory=True)
    S, Y = [], []
    for x, y in dl:
        out = model(x.to(DEV, non_blocking=True))
        S.append(out.logit.float().cpu()); Y.append(y)
    return torch.cat(S).numpy().ravel(), torch.cat(Y).numpy()


def main():
    model, meta = load_detector_checkpoint(CKPT / "s4_detector.pt", CKPT / "s3_memory.pt",
                                           device=DEV)
    print(f"TRACE-RX-M v2 | encoder={meta['encoder_mode']} | epoch {meta['epoch']}\n")
    m = D.load_manifest(ROOT)

    print(f"{'population':34s} {'n':>5} {'AUROC':>7} {'AUPRC':>7} {'TPR@1%':>7} {'EER':>6}")
    rows_all = []
    for name, sel in [("WildFake (all splits)", m.source_dataset == "wildfake"),
                      ("SID-Set (all splits)", m.source_dataset == "sid_set"),
                      ("Both pooled", m.index == m.index)]:
        sub = m[sel]
        s, y = score(model, sub)
        mm = compute(y, s)
        print(f"  {name:32s} {len(y):5d} {mm.auroc:7.4f} {mm.auprc:7.4f} "
              f"{mm.tpr_at_fpr01:7.4f} {mm.eer:6.4f}")
        rows_all.append((name, s, y, sub))

    # Per WildFake generator, and SID's lumped bucket
    print("\n  per generator (AI rows vs that source's reals):")
    for src, gens in [("wildfake", ["adm", "ddim", "ddpm", "gan_based", "imagen"]),
                      ("sid_set", ["text_to_image"])]:
        sub = m[m.source_dataset == src]
        s, y = score(model, sub)
        gen = sub.generator.values
        real = y == 0
        for g in gens:
            k = (gen == g) | real
            if len(np.unique(y[k])) == 2:
                print(f"    {src:9s} {g:14s} n={int(k.sum()):5d}  AUROC {roc_auc_score(y[k], s[k]):.4f}")

    out = pd.DataFrame({"path": m.path, "label": m.y, "source": m.source_dataset,
                        "generator": m.generator,
                        "score": score(model, m)[0]})
    out.to_parquet("/home/joel/Desktop/acai/runs/trace_rx_m_scores.parquet", index=False)
    print("\n  wrote runs/trace_rx_m_scores.parquet")


if __name__ == "__main__":
    main()
