"""Gates for the data layer — canonicalisation, holdouts, split integrity.

The bottleneck tests matter most: canonicalising to a common *size* removes the dimension cue
but not the resampling *history*, and on this corpus the residual was strong enough to tie the
first baseline with a dimensions-only probe.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from acai import data as D


@pytest.fixture
def fake_root(tmp_path):
    """Mimics data_draft's confound: reals small and square, AI large and square."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(24):
        real = i % 2 == 0
        w = h = 200 if real else 1024
        if not real and i % 4 == 1:
            w, h = 1024, 768                       # some non-square AI
        p = f"images/x/{'real' if real else 'ai_full'}/a{i}.png"
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(tmp_path / p)
        rows.append(dict(
            asset_id=f"a{i}", lineage_id=f"lin{i}", base_id=f"b{i}",
            split=["train", "dev", "calibration"][i % 3],
            label="real" if real else "ai_full", path=p,
            source_dataset="wildfake" if (i // 2) % 2 else "sid_set",
            source_family="", real_subtype="afhq" if real else "",
            ai_subtype="" if real else ["adm", "ddim", ""][i % 3],
            width=w, height=h, aspect_ratio=w / h, file_size_bytes=1000,
            sha256=f"s{i}", phash=f"p{i}"))
    pd.DataFrame(rows).to_json(tmp_path / "manifest.parquet", orient="records", lines=True)
    return tmp_path


def test_load_manifest_reads_jsonl_despite_parquet_extension(fake_root):
    """Upstream ships JSONL named .parquet; the documented loader snippet fails on it."""
    m = D.load_manifest(fake_root)
    assert len(m) == 24 and set(m.y) == {0, 1}


def test_derived_columns(fake_root):
    m = D.load_manifest(fake_root)
    assert (m.loc[m.y == 0, "generator"] == "").all()          # reals carry no generator
    assert (m.loc[m.y == 1, "generator"] != "").all()          # AI always does
    assert D.LUMPED_GENERATOR in set(m.generator)              # SID's unnamed AI is lumped
    assert (m.loc[m.y == 0, "authentic_subtype"] != "").all()
    assert (m["group"] == m["lineage_id"]).all()


def test_canonicalise_produces_one_size(fake_root, tmp_path):
    m = D.load_manifest(fake_root)
    c = D.canonicalise(m, fake_root, tmp_path / "canon", size=64, workers=2)
    assert len(c) == len(m)
    assert {Image.open(p).size for p in c.canonical_path} == {(64, 64)}


def test_canonicalise_is_idempotent(fake_root, tmp_path):
    m = D.load_manifest(fake_root)
    a = D.canonicalise(m, fake_root, tmp_path / "c", size=64, workers=2)
    first = Image.open(a.canonical_path.iloc[0]).tobytes()
    b = D.canonicalise(m, fake_root, tmp_path / "c", size=64, workers=2)
    assert Image.open(b.canonical_path.iloc[0]).tobytes() == first


def test_bottleneck_equalises_the_resampling_history(fake_root, tmp_path):
    """Without a bottleneck, a natively-small image stays visibly softer after upscaling.

    That softness is what tied the first baseline to a dimensions-only probe: on WildFake
    every real is natively 200x200 and every one of them gets upscaled.
    """
    from scipy.ndimage import laplace
    m = D.load_manifest(fake_root)

    def sharpness(df):
        out = {}
        for lab in (0, 1):
            v = [laplace(np.asarray(Image.open(p).convert("L"), float)).var()
                 for p in df[df.y == lab].canonical_path]
            out[lab] = float(np.mean(v))
        return out

    plain = sharpness(D.canonicalise(m, fake_root, tmp_path / "p", 256, 2, bottleneck=None))
    matched = sharpness(D.canonicalise(m, fake_root, tmp_path / "m", 256, 2, bottleneck=64))

    gap_plain = abs(plain[0] - plain[1]) / max(plain[0], plain[1])
    gap_match = abs(matched[0] - matched[1]) / max(matched[0], matched[1])
    assert gap_match < gap_plain, (
        f"bottleneck did not narrow the class sharpness gap: {gap_plain:.3f} -> {gap_match:.3f}")


def test_bottleneck_recorded_in_manifest(fake_root, tmp_path):
    """A run must be able to say which config produced it."""
    m = D.load_manifest(fake_root)
    c = D.canonicalise(m, fake_root, tmp_path / "c", 64, 2, bottleneck=32)
    assert (c.canon_bottleneck == 32).all()


def test_holdout_masks_are_disjoint_and_named(fake_root):
    m = D.load_manifest(fake_root)
    h = D.holdout_masks(m)
    assert any(k.startswith("unseen_generator/") for k in h)
    assert set(f"unseen_source/{s}" for s in D.SOURCES) <= set(h)
    for k, v in h.items():
        if k.startswith("unseen_source/"):
            assert not (v["train"] & v["eval"]).any(), f"{k} train/eval overlap"
        assert v["train"].sum() > 0 and v["eval"].sum() > 0


def test_unseen_generator_keeps_reals_in_training(fake_root):
    """Holding out ADM must mean 'never saw ADM', not 'never saw a real image'."""
    m = D.load_manifest(fake_root)
    h = D.holdout_masks(m)
    for gen in D.NAMED_GENERATORS:
        k = f"unseen_generator/{gen}"
        if k in h:
            trained = m[h[k]["train"]]
            assert (trained.generator == gen).sum() == 0
            assert (trained.y == 0).sum() > 0


def test_check_split_integrity_flags_a_planted_leak(fake_root):
    m = D.load_manifest(fake_root)
    assert D.check_split_integrity(m)["ok"] is True
    leak = m.copy()
    leak.loc[1, "lineage_id"] = leak.loc[0, "lineage_id"]
    leak.loc[1, "group"] = leak.loc[0, "group"]
    leak.loc[1, "split"] = "dev" if leak.loc[0, "split"] != "dev" else "train"
    leak.loc[0, "split"] = "train"
    r = D.check_split_integrity(leak)
    assert r["group"]["straddling"] >= 1 and r["ok"] is False


def test_save_coerces_mixed_type_columns(fake_root, tmp_path):
    """Upstream `file_size` mixes int and str, which breaks a naive parquet write."""
    m = D.load_manifest(fake_root)
    m["mixed"] = [1 if i % 2 else "x" for i in range(len(m))]
    p = D.save(m, tmp_path / "out.parquet")
    assert len(pd.read_parquet(p)) == len(m)
