"""THE SEALED EVALUATOR. One evaluation implementation, identical for every model.

    python -m acai.scorecard --predictions runs/<id>/predictions.parquet --out runs/<id>

Why sealing matters: if a proposing agent can edit the evaluator, it can move the goalposts --
usually not maliciously, but by "fixing" a metric that made its model look bad. So this module
hashes itself and its dependencies on every run and stamps the digest into the scorecard. A
scorecard whose `sealed` field is false is not a result; it is a claim about a modified ruler.

Sections produced (fixed; the scorecard refuses to emit a partial one):

    1  roc_auc + auprc                      overall, with base rate
    2  per_transform                        clean and every official condition
    3  robustness                           worst transform, clean-to-transform drop
    4  generalisation                       unseen generator, unseen source
    6  calibration                          Brier, ECE, reliability curve, fitted calibrator
    10 predictions + error_cards            representative FP/FN with provenance

Deliberately OUT OF SCOPE for this build, by decision on 2026-08-30:
    5  worst authentic-subtype FPR
    7  latency and memory
    8  expert error correlation      (not computable: dataset carries no human annotations)
    9  Master-Lineage bootstrap CIs  (degenerate today: every lineage_id is 1:1 with an asset)
`acai.metrics.bootstrap_ci` stays available and correct for when 9 becomes meaningful --
i.e. once transform variants share a lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acai.metrics import by_group, compute, worst_family

# Files whose content defines the ruler. Changing any of them changes every number.
SEALED_FILES = ("scorecard.py", "metrics.py")
LOCK = Path(__file__).parent.parent.parent / "evaluator.lock"

REQUIRED_COLUMNS = {"id", "group", "label", "score", "split", "source", "transform", "severity"}
OPTIONAL_COLUMNS = {"generator", "source_dataset", "authentic_subtype", "holdout"}


# ------------------------------------------------------------------------- sealing

def digest() -> dict[str, str]:
    d = {}
    here = Path(__file__).parent
    for f in SEALED_FILES:
        d[f] = hashlib.sha256((here / f).read_bytes()).hexdigest()
    d["combined"] = hashlib.sha256(
        "".join(d[f] for f in SEALED_FILES).encode()).hexdigest()
    return d


def seal_status() -> dict:
    """Compare the live digest against evaluator.lock.

    Reports rather than blocks. A hard refusal would just invite someone to delete the lock;
    an unmissable `sealed: false` in the output travels with the result.
    """
    cur = digest()
    if not LOCK.exists():
        return {"sealed": False, "reason": "evaluator.lock missing -- run --seal to create it",
                "digest": cur}
    locked = json.loads(LOCK.read_text())
    same = locked.get("combined") == cur["combined"]
    return {"sealed": bool(same), "digest": cur, "locked": locked,
            "reason": "" if same else
                      "EVALUATOR MODIFIED since seal: this scorecard used a different ruler",
            "changed": [f for f in SEALED_FILES if locked.get(f) != cur.get(f)]}


def write_lock() -> Path:
    LOCK.write_text(json.dumps(digest(), indent=2, sort_keys=True))
    return LOCK


# -------------------------------------------------------------------------- input

def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    if df["label"].nunique() < 2:
        raise ValueError("predictions contain a single class")
    if df["score"].isna().any():
        raise ValueError("predictions contain NaN scores")
    # `transform` (and `filter`, `pop`, ...) are DataFrame *methods*, so attribute access
    # silently returns the bound method instead of the column. Everything here uses
    # df["col"] indexing; this check makes the hazard visible if a new column is added.
    shadowed = {c for c in df.columns if callable(getattr(pd.DataFrame, c, None))}
    if shadowed:
        df.attrs["method_shadowed_columns"] = sorted(shadowed)
    return df


# ---------------------------------------------------------------------- sections

def s1_overall(df: pd.DataFrame) -> dict:
    m = compute(df.label, df.score).as_dict()
    m["note"] = ("AUPRC is reported with its base rate: its random baseline IS the positive "
                 "rate, so it is not comparable across evaluations with different prevalence")
    return m


def s2_per_transform(df: pd.DataFrame) -> dict:
    """Every official condition, and each transform family aggregated over its severities."""
    out = {"by_condition": by_group(df.label, df.score, df["transform"].astype(str)
                                    + ":" + df["severity"].astype(str), min_n=30)}
    fam = df["transform"].astype(str)
    out["by_family"] = by_group(df.label, df.score, fam, min_n=30)
    return out


def s3_robustness(df: pd.DataFrame) -> dict:
    """Worst transform, and the clean-to-transform drop.

    The drop is the number a robustness claim rests on. A model can hold a high mean over
    transforms while collapsing on one family, and if the hidden evaluation happens to weight
    that family, the mean was never the relevant statistic.
    """
    clean = df[df["transform"] == "clean"]
    trans = df[df["transform"] != "clean"]
    out: dict = {}
    if len(clean) and clean.label.nunique() == 2:
        out["clean_auroc"] = compute(clean.label, clean.score).auroc
    if len(trans) and trans.label.nunique() == 2:
        out["transformed_auroc_mean"] = compute(trans.label, trans.score).auroc
        fam = by_group(trans.label, trans.score, trans["transform"].astype(str), min_n=30)
        name, val = worst_family(fam)
        out["worst_family"] = {"family": name, "auroc": val}
        cond = by_group(trans.label, trans.score,
                        trans["transform"].astype(str) + ":" + trans["severity"].astype(str), min_n=30)
        cname, cval = worst_family(cond)
        out["worst_condition"] = {"condition": cname, "auroc": cval}
    if "clean_auroc" in out and "worst_family" in out:
        out["clean_to_worst_drop"] = out["clean_auroc"] - out["worst_family"]["auroc"]
        out["clean_to_mean_drop"] = out["clean_auroc"] - out["transformed_auroc_mean"]
    return out


def s4_generalisation(df: pd.DataFrame) -> dict:
    """Unseen-generator and unseen-source results.

    Caveats recorded inline so a reader cannot take these at more than face value:
    SID's 2,500 AI images carry no generator identity (a single lumped `text_to_image`), so
    only 5 named WildFake generators can be held out; and there are only 2 source datasets,
    making unseen-source a 2-way test rather than a sweep.
    """
    out: dict = {"caveats": [
        "SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named "
        "WildFake generators support leave-one-generator-out",
        "only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep",
    ]}
    if "generator" in df.columns:
        g = df[df.generator.astype(str) != ""]
        if len(g):
            # Each generator's AI images scored against all reals in the same rows.
            out["by_generator"] = {}
            # Each generator's AI images are scored against the *same* real set, so the
            # numbers are comparable across generators. They are not, however, standalone
            # AUROCs: the shared reals dominate the row count and compress the spread, so
            # read this table for the ranking and the gaps, not the absolute levels.
            reals = df[df.label == 0]
            out["by_generator_note"] = (
                "each generator's AI rows vs the common real set; n includes those shared "
                "reals, so compare generators against each other, not against section 1")
            for gen in sorted(set(g.generator) - {""}):
                ai = df[df.generator == gen]
                sub = pd.concat([ai, reals])
                if sub.label.nunique() == 2 and len(sub) >= 30:
                    d = compute(sub.label, sub.score).as_dict()
                    d["n_ai"] = int(len(ai))
                    d["n_real_shared"] = int(len(reals))
                    out["by_generator"][gen] = d
            if out["by_generator"]:
                k = min(out["by_generator"], key=lambda k: out["by_generator"][k]["auroc"])
                out["worst_generator"] = {"generator": k,
                                          "auroc": out["by_generator"][k]["auroc"]}
    if "source_dataset" in df.columns:
        out["by_source"] = by_group(df.label, df.score, df.source_dataset.astype(str), min_n=30)
    if "holdout" in df.columns:
        h = df[df.holdout.astype(str) != ""]
        if len(h):
            out["holdout_evaluations"] = by_group(h.label, h.score, h.holdout.astype(str),
                                                  min_n=30)
    return out


def s6_calibration(df: pd.DataFrame, calibration_split: str = "calibration") -> dict:
    """Brier, ECE, a reliability curve, and a calibrator fitted on the dedicated split.

    The dataset ships a 1,500-image `calibration` split for exactly this. Fitting on it and
    reporting the improvement keeps calibration an honest, separate step rather than something
    tuned on the evaluation data.
    """
    m = compute(df.label, df.score)
    out = {"brier": m.brier, "ece": m.ece}

    p = np.clip(df.score.values.astype(float), 0, 1)
    y = df.label.values.astype(int)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    out["reliability"] = [
        {"bin": [float(edges[b]), float(edges[b + 1])], "n": int((idx == b).sum()),
         "mean_score": float(p[idx == b].mean()) if (idx == b).any() else None,
         "empirical_rate": float(y[idx == b].mean()) if (idx == b).any() else None}
        for b in range(10)]

    cal = df[df.split == calibration_split]
    ev = df[df.split != calibration_split]
    if len(cal) >= 100 and cal.label.nunique() == 2 and len(ev) and ev.label.nunique() == 2:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        out["fitted_on"] = {"split": calibration_split, "n": int(len(cal))}

        plat = LogisticRegression().fit(cal.score.values.reshape(-1, 1), cal.label.values)
        iso = IsotonicRegression(out_of_bounds="clip").fit(cal.score.values, cal.label.values)
        for name, pred in (("platt", plat.predict_proba(ev.score.values.reshape(-1, 1))[:, 1]),
                           ("isotonic", iso.predict(ev.score.values))):
            c = compute(ev.label, pred)
            out[name] = {"brier": c.brier, "ece": c.ece, "auroc": c.auroc}
        base = compute(ev.label, ev.score)
        out["uncalibrated_on_eval"] = {"brier": base.brier, "ece": base.ece,
                                       "auroc": base.auroc}
        out["note"] = ("AUROC is rank-based and so is unchanged by monotone calibration; any "
                       "difference in the isotonic AUROC comes from its ties")
    else:
        out["fitted_on"] = None
        out["note"] = "no usable calibration split in these predictions; reporting raw only"
    return out


def s10_error_cards(df: pd.DataFrame, k: int = 12) -> dict:
    """Representative false positives and false negatives, at the EER threshold.

    Chosen as the most *confident* mistakes rather than the marginal ones: a borderline error
    tells you the threshold is close, while a confident error tells you what the model has
    actually misunderstood.
    """
    m = compute(df.label, df.score)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(df.label, df.score)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    t = float(thr[i])

    cols = [c for c in ("id", "group", "score", "source", "source_dataset", "generator",
                        "transform", "severity", "split") if c in df.columns]
    fp = df[(df.label == 0) & (df.score >= t)].nlargest(k, "score")[cols]
    fn = df[(df.label == 1) & (df.score < t)].nsmallest(k, "score")[cols]
    return {"threshold": t, "threshold_rule": "EER",
            "n_false_positive": int(((df.label == 0) & (df.score >= t)).sum()),
            "n_false_negative": int(((df.label == 1) & (df.score < t)).sum()),
            "eer": m.eer,
            "false_positives": fp.to_dict("records"),
            "false_negatives": fn.to_dict("records")}


# -------------------------------------------------------------------------- driver

EXCLUDED = {
    "5_worst_authentic_subtype_fpr": "excluded from this build by decision 2026-08-30",
    "7_latency_memory": "excluded from this build by decision 2026-08-30",
    "8_expert_error_correlation":
        "NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote "
        "fields; label_confidence is the constant 0.8; labels are path-derived)",
    "9_master_lineage_bootstrap":
        "excluded from this build; also degenerate today -- every lineage_id in data_draft is "
        "1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform "
        "variants share a lineage",
}


def build(predictions: str | Path, split: str | None = None) -> dict:
    """Produce the full scorecard. Every section runs; none is silently skipped."""
    df = load_predictions(predictions)
    if split:
        df = df[df.split == split]
    seal = seal_status()
    card = {
        "sealed": seal["sealed"],
        "seal": seal,
        "input": {"path": str(predictions), "n": int(len(df)),
                  "splits": sorted(df.split.astype(str).unique().tolist()),
                  "positive_class": "label == 1 (ai_full)"},
        "1_overall": s1_overall(df),
        "2_per_transform": s2_per_transform(df),
        "3_robustness": s3_robustness(df),
        "4_generalisation": s4_generalisation(df),
        "6_calibration": s6_calibration(df),
        "10_error_cards": s10_error_cards(df),
        "excluded_sections": EXCLUDED,
    }
    if not seal["sealed"]:
        card["WARNING"] = seal["reason"]
    return card


def to_markdown(card: dict) -> str:
    """Human-readable scorecard. Never the source of truth -- that is the JSON."""
    L = []
    ok = "SEALED" if card["sealed"] else "*** UNSEALED -- EVALUATOR MODIFIED ***"
    L.append(f"# Scorecard  [{ok}]\n")
    if not card["sealed"]:
        L.append(f"> **{card.get('WARNING','')}**\n")
    L.append(f"Evaluator digest: `{card['seal']['digest']['combined'][:16]}`  ·  "
             f"n = {card['input']['n']}  ·  positive = {card['input']['positive_class']}\n")

    o = card["1_overall"]
    L.append("## 1. Overall\n")
    L.append(f"| metric | value |\n|---|---|")
    for k in ("auroc", "auprc", "auprc_lift", "prevalence", "eer", "balanced_acc_50",
              "tpr_at_fpr01", "tpr_at_fpr05", "brier", "ece"):
        L.append(f"| {k} | {o[k]:.4f} |")
    L.append("")

    L.append("## 2. Per transform\n")
    L.append("| family | n | AUROC | AUPRC | TPR@FPR1% |\n|---|---|---|---|---|")
    for k, v in card["2_per_transform"]["by_family"].items():
        L.append(f"| {k} | {v['n']} | {v['auroc']:.4f} | {v['auprc']:.4f} | "
                 f"{v['tpr_at_fpr01']:.4f} |" if "auroc" in v
                 else f"| {k} | {v['n']} | — | — | {v.get('skipped','')} |")
    L.append("")

    r = card["3_robustness"]
    L.append("## 3. Robustness\n")
    for k in ("clean_auroc", "transformed_auroc_mean", "clean_to_mean_drop",
              "clean_to_worst_drop"):
        if k in r:
            L.append(f"- **{k}**: {r[k]:.4f}")
    if "worst_family" in r:
        L.append(f"- **worst family**: `{r['worst_family']['family']}` "
                 f"AUROC {r['worst_family']['auroc']:.4f}")
    if "worst_condition" in r:
        L.append(f"- **worst condition**: `{r['worst_condition']['condition']}` "
                 f"AUROC {r['worst_condition']['auroc']:.4f}")
    L.append("")

    g = card["4_generalisation"]
    L.append("## 4. Generalisation\n")
    if "by_generator" in g:
        L.append("| generator | n AI | AUROC (vs shared reals) |\n|---|---|---|")
        for k, v in g["by_generator"].items():
            L.append(f"| {k} | {v.get('n_ai', v['n'])} | {v['auroc']:.4f} |")
        if "worst_generator" in g:
            L.append(f"\n- **worst generator**: `{g['worst_generator']['generator']}` "
                     f"AUROC {g['worst_generator']['auroc']:.4f}")
    if "by_source" in g:
        L.append("\n| source | n | AUROC |\n|---|---|---|")
        for k, v in g["by_source"].items():
            L.append(f"| {k} | {v['n']} | {v['auroc']:.4f} |" if "auroc" in v
                     else f"| {k} | {v['n']} | {v.get('skipped','')} |")
    for c in g.get("caveats", []):
        L.append(f"\n> {c}")
    L.append("")

    c = card["6_calibration"]
    L.append("## 6. Calibration\n")
    L.append(f"- raw: Brier {c['brier']:.4f}, ECE {c['ece']:.4f}")
    for k in ("platt", "isotonic"):
        if k in c:
            L.append(f"- {k}: Brier {c[k]['brier']:.4f}, ECE {c[k]['ece']:.4f}")
    L.append("")

    e = card["10_error_cards"]
    L.append("## 10. Error cards\n")
    L.append(f"At the EER threshold {e['threshold']:.4f}: "
             f"{e['n_false_positive']} FP, {e['n_false_negative']} FN.\n")

    L.append("## Excluded sections\n")
    for k, v in card["excluded_sections"].items():
        L.append(f"- **{k}** — {v}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sealed AIGC-detection evaluator")
    ap.add_argument("--predictions")
    ap.add_argument("--out", default=None, help="directory for scorecard.json / scorecard.md")
    ap.add_argument("--split", default=None)
    ap.add_argument("--seal", action="store_true", help="(re)write evaluator.lock")
    ap.add_argument("--check", action="store_true", help="print seal status and exit")
    a = ap.parse_args()

    if a.seal:
        print(f"wrote {write_lock()}")
        print(json.dumps(digest(), indent=2))
        return
    if a.check:
        print(json.dumps(seal_status(), indent=2))
        return
    if not a.predictions:
        ap.error("--predictions is required")

    card = build(a.predictions, a.split)
    out = Path(a.out or Path(a.predictions).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, sort_keys=True, default=str))
    (out / "scorecard.md").write_text(to_markdown(card))
    print(to_markdown(card))
    if not card["sealed"]:
        raise SystemExit(f"UNSEALED: {card.get('WARNING')}")


if __name__ == "__main__":
    main()
