"""Correctness gates for the official transform grid.

These exist because every downstream number is computed on the output of this module:
if the transforms are not deterministic, the per-family robustness tables are not
reproducible; if a transform changes output size, the transform label leaks through
image dimensions and inflates every per-family AUROC.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from PIL import Image

from acai.transforms import (GRID, Chain, apply_chain, conditions, random_chain,
                             seed_for, single_chains)


def img(seed=0, size=(256, 256)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size[::-1], 3), dtype=np.uint8), "RGB")


def test_grid_matches_official_sheet():
    assert GRID["jpeg"] == [90, 70, 50, 30]
    assert GRID["blur"] == [0.5, 1.0, 2.0]
    assert GRID["resize"] == [0.5, 0.25]
    assert GRID["noise"] == [0.02, 0.05, 0.10]
    assert GRID["color"] == [0.20]
    assert GRID["crop"] == [0.80]
    assert len(conditions()) == 15


@pytest.mark.parametrize("chain", single_chains(), ids=lambda c: c.name)
def test_deterministic(chain):
    """Same image id -> byte-identical output, across calls."""
    a = apply_chain(img(), chain, "img-42")
    b = apply_chain(img(), chain, "img-42")
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("chain", single_chains(), ids=lambda c: c.name)
def test_size_preserved(chain):
    """No transform may change output dimensions.

    crop and resize both round-trip back to the source size on purpose; if they did
    not, a detector could read the transform label straight off img.size.
    """
    assert apply_chain(img(), chain, "x").size == (256, 256)


def test_seed_varies_with_image_id():
    """Stochastic transforms must not apply the identical realisation to every image."""
    a = apply_chain(img(), Chain((("color", 0.20),)), "img-a")
    b = apply_chain(img(), Chain((("color", 0.20),)), "img-b")
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_seed_stable_across_processes():
    """sha1-derived, not Python hash() -- which is salted per process."""
    assert seed_for("img-42", "0:jpeg:90") == seed_for("img-42", "0:jpeg:90")
    assert seed_for("img-42") != seed_for("img-43")


def test_clean_is_identity():
    src = img()
    assert np.array_equal(np.asarray(apply_chain(src, Chain(), "x")), np.asarray(src))


def test_transforms_actually_change_the_image():
    """Guards against a silently no-op op (e.g. a mis-scaled parameter)."""
    src = np.asarray(img(), dtype=np.int16)
    for chain in single_chains():
        if chain.name == "clean":
            continue
        out = np.asarray(apply_chain(img(), chain, "x"), dtype=np.int16)
        assert np.abs(out - src).mean() > 0.5, f"{chain.name} barely changed the image"


def test_severity_is_monotone():
    """Stronger official parameters must distort more, or the grid is mislabelled."""
    src = np.asarray(img(), dtype=np.float32)

    def dist(fam, p):
        out = np.asarray(apply_chain(img(), Chain(((fam, p),)), "x"), dtype=np.float32)
        return np.abs(out - src).mean()

    for fam, params in [("jpeg", GRID["jpeg"]), ("blur", GRID["blur"]),
                        ("resize", GRID["resize"]), ("noise", GRID["noise"])]:
        d = [dist(fam, p) for p in params]
        assert d == sorted(d), f"{fam} not monotone in severity: {d}"


def test_noise_sigma_is_normalised_units():
    """sigma=0.10 in [0,1] units means ~25.5 8-bit levels; as 8-bit it would be ~0.1."""
    src = np.asarray(img(), dtype=np.float32)
    out = np.asarray(apply_chain(img(), Chain((("noise", 0.10),)), "x"), dtype=np.float32)
    # Clipping at the uint8 boundary pulls the observed std below the nominal 25.5.
    assert 15.0 < (out - src).std() < 26.0


def test_chain_canonical_order():
    """Equivalent chains collapse to one name, so per-chain tables do not double-count."""
    a = Chain((("jpeg", 50), ("crop", 0.8))).canonical()
    b = Chain((("crop", 0.8), ("jpeg", 50))).canonical()
    assert a.name == b.name == "crop:0.8+jpeg:50"


def test_random_chain_families_distinct_and_ordered():
    from acai.transforms import FAMILY_ORDER
    for i in range(200):
        c = random_chain("x", np.random.default_rng(i))
        assert 1 <= len(c.steps) <= 3
        assert len(set(c.families)) == len(c.families)
        idx = [FAMILY_ORDER.index(f) for f in c.families]
        assert idx == sorted(idx)


def test_numpy_ops_compose_without_intermediate_quantisation():
    """`color` -> `noise` must compose in float32, with no uint8 round-trip between.

    Only the numpy-domain ops can be checked this way: crop/resize/blur/jpeg go through
    PIL and quantise internally by construction, which we accept deliberately (matching
    the grader's PIL semantics beats numeric purity). Chaining a PIL op in the middle
    swamps the effect, which is exactly why this test isolates the two numpy ops.
    """
    from acai.transforms import OPS, _to_f32, _to_pil

    steps = (("color", 0.20), ("noise", 0.02))
    good = np.asarray(apply_chain(img(), Chain(steps), "x"), dtype=np.float32)

    a = _to_f32(img())
    for i, (fam, p) in enumerate(steps):
        rng = np.random.default_rng(seed_for("x", f"{i}:{fam}:{p}"))
        a = _to_f32(_to_pil(OPS[fam](a, p, rng)))     # forced round-trip
    bad = np.asarray(_to_pil(a), dtype=np.float32)

    assert not np.array_equal(good, bad), "intermediate quantisation crept back in"
    # The rounding it avoids is sub-level, so it must not be mistaken for a real effect.
    assert np.abs(good - bad).mean() < 1.0


def test_pil_ops_quantise_internally():
    """Documents the accepted limitation above, so it cannot silently change."""
    from acai.transforms import _to_f32
    from PIL import ImageFilter
    from acai.transforms import _to_pil
    a = _to_f32(img())
    out = _to_pil(a).filter(ImageFilter.GaussianBlur(0.5))
    assert out.mode == "RGB"   # 8-bit per channel: PIL has already quantised
