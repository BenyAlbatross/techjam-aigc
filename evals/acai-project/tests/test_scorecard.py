"""Gates for the sealed evaluator.

The seal tests matter most: the whole point of item 2 is that a proposing agent cannot quietly
change the ruler, so "modification is detected" has to be verified rather than assumed.
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from acai import scorecard as SC
from acai.transforms import conditions


def preds(n=400, seed=0):
    rng = np.random.default_rng(seed)
    conds = conditions()
    gens = ["adm", "ddim", "ddpm", "gan_based", "imagen", "text_to_image"]
    rows = []
    for i in range(n):
        y = i % 2
        ds = "sid_set" if i % 3 else "wildfake"
        for f, p in conds:
            deg = {"clean": 0, "jpeg": .05, "blur": .25, "resize": .15,
                   "noise": .10, "color": .02, "crop": .03}[f]
            rows.append(dict(
                id=f"a{i}|{f}:{p:g}", group=f"lin{i}", label=y,
                score=float(np.clip(0.5 + (0.30 - deg) * (y * 2 - 1) + rng.normal(0, .18), 0, 1)),
                split=["train", "dev", "calibration"][i % 3], source=ds,
                transform=f, severity=float(p),
                generator=gens[(i // 2) % len(gens)] if y else "", source_dataset=ds, holdout=""))
    return pd.DataFrame(rows)


@pytest.fixture
def pfile(tmp_path):
    p = tmp_path / "pred.parquet"
    preds().to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------- seal

def test_digest_is_stable_and_covers_metrics():
    d = SC.digest()
    assert set(d) == {"scorecard.py", "metrics.py", "combined"}
    assert d == SC.digest()


def test_seal_detects_a_modified_evaluator(tmp_path, monkeypatch):
    """The core guarantee: edit the ruler, and every scorecard says so."""
    lock = tmp_path / "evaluator.lock"
    monkeypatch.setattr(SC, "LOCK", lock)
    SC.write_lock()
    assert SC.seal_status()["sealed"] is True

    tampered = json.loads(lock.read_text())
    tampered["combined"] = "0" * 64          # as if metrics.py had been edited
    lock.write_text(json.dumps(tampered))
    st = SC.seal_status()
    assert st["sealed"] is False and "MODIFIED" in st["reason"]


def test_missing_lock_is_unsealed(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "LOCK", tmp_path / "nope.lock")
    assert SC.seal_status()["sealed"] is False


def test_unsealed_card_carries_a_loud_warning(pfile, tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "LOCK", tmp_path / "nope.lock")
    card = SC.build(pfile)
    assert card["sealed"] is False and "WARNING" in card
    assert "UNSEALED" in SC.to_markdown(card)


# --------------------------------------------------------------------------- input

def test_required_columns_enforced(tmp_path):
    df = preds(50).drop(columns=["group"])
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="group"):
        SC.load_predictions(p)


def test_single_class_and_nan_rejected(tmp_path):
    df = preds(50)
    p = tmp_path / "x.parquet"
    df.assign(label=1).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="single class"):
        SC.load_predictions(p)
    d2 = df.copy(); d2.loc[0, "score"] = np.nan
    d2.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="NaN"):
        SC.load_predictions(p)


def test_transform_column_is_not_shadowed_by_the_dataframe_method(pfile):
    """`df.transform` is DataFrame.transform, not the column. Regression guard."""
    df = SC.load_predictions(pfile)
    assert "transform" in df.attrs.get("method_shadowed_columns", [])
    assert set(df["transform"]) >= {"clean", "jpeg", "blur"}


# ------------------------------------------------------------------------ sections

def test_all_sections_present_and_none_silently_skipped(pfile):
    card = SC.build(pfile)
    for s in ("1_overall", "2_per_transform", "3_robustness", "4_generalisation",
              "6_calibration", "10_error_cards", "excluded_sections"):
        assert s in card, f"missing section {s}"


def test_excluded_sections_are_documented_with_reasons(pfile):
    ex = SC.build(pfile)["excluded_sections"]
    assert set(ex) == {"5_worst_authentic_subtype_fpr", "7_latency_memory",
                       "8_expert_error_correlation", "9_master_lineage_bootstrap"}
    assert "NOT COMPUTABLE" in ex["8_expert_error_correlation"]
    assert "1:1" in ex["9_master_lineage_bootstrap"]


def test_per_transform_covers_every_official_condition(pfile):
    card = SC.build(pfile)
    fams = set(card["2_per_transform"]["by_family"])
    assert fams == {"clean", "jpeg", "blur", "resize", "noise", "color", "crop"}
    assert len(card["2_per_transform"]["by_condition"]) == len(conditions())


def test_robustness_identifies_the_planted_worst_family(pfile):
    """blur was constructed to be the worst; the scorecard must find it."""
    r = SC.build(pfile)["3_robustness"]
    assert r["worst_family"]["family"] == "blur"
    assert r["clean_auroc"] > r["worst_family"]["auroc"]
    assert r["clean_to_worst_drop"] == pytest.approx(
        r["clean_auroc"] - r["worst_family"]["auroc"])


def test_generalisation_reports_generators_sources_and_caveats(pfile):
    g = SC.build(pfile)["4_generalisation"]
    assert set(g["by_generator"]) >= {"adm", "ddim", "ddpm", "gan_based", "imagen"}
    assert set(g["by_source"]) == {"sid_set", "wildfake"}
    assert any("no generator identity" in c for c in g["caveats"])
    assert "n_ai" in next(iter(g["by_generator"].values()))


def test_calibration_fits_only_on_the_calibration_split(pfile):
    c = SC.build(pfile)["6_calibration"]
    assert c["fitted_on"]["split"] == "calibration"
    assert "platt" in c and "isotonic" in c
    # Monotone calibration cannot change a rank-based metric.
    assert c["platt"]["auroc"] == pytest.approx(c["uncalibrated_on_eval"]["auroc"], abs=1e-9)
    assert len(c["reliability"]) == 10


def test_calibration_degrades_gracefully_without_the_split(tmp_path):
    df = preds(200)
    df = df[df.split != "calibration"]
    p = tmp_path / "nocal.parquet"
    df.to_parquet(p, index=False)
    c = SC.build(p)["6_calibration"]
    assert c["fitted_on"] is None and "brier" in c


def test_error_cards_are_the_confident_mistakes(pfile):
    e = SC.build(pfile)["10_error_cards"]
    assert e["threshold_rule"] == "EER"
    fp = [r["score"] for r in e["false_positives"]]
    fn = [r["score"] for r in e["false_negatives"]]
    assert fp == sorted(fp, reverse=True) and all(s >= e["threshold"] for s in fp)
    assert fn == sorted(fn) and all(s < e["threshold"] for s in fn)
    assert "transform" in e["false_positives"][0]      # provenance travels with the card


def test_markdown_renders_without_error(pfile):
    md = SC.to_markdown(SC.build(pfile))
    assert "# Scorecard" in md and "## 3. Robustness" in md and "Excluded sections" in md


def test_split_filter(pfile):
    card = SC.build(pfile, split="dev")
    assert card["input"]["splits"] == ["dev"]
