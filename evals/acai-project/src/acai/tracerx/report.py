"""Score the predictions written by `run_eval` and emit the report tables.

Metric choices follow `acai.metrics`: AUROC is primary, AUPRC always carries its prevalence,
and low-FPR operating points are reported explicitly. Two conventions specific to this model:

* The threshold is **0.0 on the logit**, matching the model's own
  `evaluation/metrics_by_condition.csv`. We pass `sigmoid(logit)` as the score so
  `balanced_acc_50` lands on exactly that operating point.
* Brier and ECE are computed but must be read as relative only -- the model card states the
  checkpoint is not probability-calibrated, so `sigmoid(logit)` is a margin, not a probability.

Per-generator AUROC pools each generator's fakes against *all* authentic rows of the same
config, because a generator-vs-generator split has no negatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..metrics import bootstrap_ci, by_group, compute, worst_family

CLEAN = ["default", "normalized", "laion_matched", "cross_generator", "diverse"]
ORDER = [c for pair in zip(CLEAN, [f"{c}_transformed" for c in CLEAN]) for c in pair]


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def load(pred_dir: Path, name: str):
    f = pred_dir / f"{name}.parquet"
    if not f.exists():
        return None
    t = pq.read_table(f)
    d = {c: t[c].to_numpy(zero_copy_only=False) for c in t.column_names}
    d["score"] = sigmoid(d["logit"])
    return d


def per_generator(d: dict) -> dict:
    """Each generator's fakes vs all authentic rows in the same config."""
    if "source" not in d:
        return {}
    y, s, src = d["label"], d["score"], d["source"]
    real = y == 0
    out = {}
    for g in sorted(set(src[y == 1].tolist())):
        m = real | ((y == 1) & (src == g))
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        r = compute(y[m], s[m])
        out[g] = {"n_fake": int(((y == 1) & (src == g)).sum()), "n_real": int(real.sum()),
                  "auroc": r.auroc, "balanced_acc_50": r.balanced_acc_50,
                  "tpr_at_fpr05": r.tpr_at_fpr05}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/tracerx_v2_wildfake/predictions")
    ap.add_argument("--out", default="runs/tracerx_v2_wildfake")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    pred, out = Path(a.pred), Path(a.out)

    report: dict = {"model": "techjam-aigc/trace-rx-m-v2",
                    "threshold": "logit >= 0 (== sigmoid >= 0.5), per the model's own metrics_by_condition.csv",
                    "configs": {}}
    for name in ORDER:
        d = load(pred, name)
        if d is None:
            continue
        y, s = d["label"], d["score"]
        m = compute(y, s)
        groups = d["id"] if "id" in d else np.arange(len(y))
        _, lo, hi = bootstrap_ci(y, s, groups, n_boot=a.n_boot)
        # The shipped threshold is a fixed point, not a fitted one, so it can travel badly
        # across a distribution shift. Recording where the logits actually sit separates
        # "cannot discriminate" from "discriminates, but the operating point moved".
        lg = d["logit"]
        entry = m.as_dict() | {
            "auroc_ci95": [lo, hi],
            "operating_point": {
                "frac_logit_positive": float(np.mean(lg > 0)),
                "frac_logit_positive_real": float(np.mean(lg[y == 0] > 0)),
                "frac_logit_positive_aigc": float(np.mean(lg[y == 1] > 0)),
                "median_logit": float(np.median(lg)),
                "balanced_acc_at_eer_threshold": m.balanced_acc_eer,
            },
        }

        if "primary_transform" in d:
            single = d["n_transforms"] == 1
            bd = by_group(y[single], s[single], d["primary_transform"][single], min_n=30)
            entry["by_transform_single"] = bd
            wk, wv = worst_family(bd)
            entry["worst_transform"] = {"setting": wk, "auroc": wv}
            two = d["n_transforms"] == 2
            if two.sum() >= 30 and len(np.unique(y[two])) > 1:
                entry["composed_two"] = {"n": int(two.sum()), "auroc": compute(y[two], s[two]).auroc}
                entry["single_only"] = {"n": int(single.sum()), "auroc": compute(y[single], s[single]).auroc}
        gens = per_generator(d)
        if gens:
            entry["by_generator"] = gens
        report["configs"][name] = entry
        print(f"{name:30s} n={m.n:6d} AUROC={m.auroc:.4f} [{lo:.4f},{hi:.4f}] "
              f"balacc={m.balanced_acc_50:.4f} prev={m.prevalence:.3f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    # ---------------------------------------------------------------- markdown
    L = ["# TRACE-RX-M v2 on wildfake-eval-subset", "",
         "Score = `sigmoid(logit)`; positive class is AIGC. Threshold 0.0 on the logit.",
         "AUROC CIs are 2000-resample percentile bootstrap.",
         "",
         "`bal.acc @0` is the shipped operating point; `bal.acc @EER` is the best this model\u2019s",
         "ranking could do if the threshold were re-fitted on this data. The gap between them,",
         "and `frac logit>0`, are the calibration story -- not the discrimination story.", "",
         "| config | n | prev | AUROC | 95% CI | bal.acc @0 | bal.acc @EER | EER | TPR@FPR1% | TPR@FPR5% | frac logit>0 |",
         "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, e in report["configs"].items():
        op = e["operating_point"]
        L.append(f"| `{name}` | {e['n']} | {e['prevalence']:.3f} | **{e['auroc']:.4f}** | "
                 f"{e['auroc_ci95'][0]:.4f}–{e['auroc_ci95'][1]:.4f} | {e['balanced_acc_50']:.4f} | "
                 f"{e['balanced_acc_eer']:.4f} | {e['eer']:.4f} | {e['tpr_at_fpr01']:.4f} | "
                 f"{e['tpr_at_fpr05']:.4f} | {op['frac_logit_positive']:.3f} |")
    for name, e in report["configs"].items():
        if "by_generator" in e:
            L += ["", f"## Per-generator — `{name}`", "",
                  "| generator | n fake | AUROC | bal.acc | TPR@FPR5% |", "|---|---:|---:|---:|---:|"]
            for g, v in sorted(e["by_generator"].items(), key=lambda kv: -kv[1]["auroc"]):
                L.append(f"| {g} | {v['n_fake']} | {v['auroc']:.4f} | "
                         f"{v['balanced_acc_50']:.4f} | {v['tpr_at_fpr05']:.4f} |")
    for name, e in report["configs"].items():
        if "by_transform_single" in e:
            L += ["", f"## Robustness — `{name}` (single-transform rows)", "",
                  "| setting | n | AUROC |", "|---|---:|---:|"]
            for k, v in sorted(e["by_transform_single"].items(),
                               key=lambda kv: kv[1].get("auroc", 9)):
                L.append(f"| {k} | {v['n']} | " +
                         (f"{v['auroc']:.4f} |" if "auroc" in v else f"_{v['skipped']}_ |"))
            if "composed_two" in e:
                L.append(f"\nOne transform: AUROC {e['single_only']['auroc']:.4f} "
                         f"(n={e['single_only']['n']}) · two composed: "
                         f"{e['composed_two']['auroc']:.4f} (n={e['composed_two']['n']}).")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out/'report.json'} and {out/'REPORT.md'}")


if __name__ == "__main__":
    main()
