"""Correctness gates for the fusion arms.

The zero-init tests are the important ones: every fusion arm must *start* numerically
identical to the pure-DINOv3 baseline. If it does not, a measured gain could come from a
different initialisation rather than from the fusion design, and the whole ablation in
plan §3.2 would be uninterpretable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from acai.models.backbone import DinoV3
from acai.models.heads import Detector, FeatureNorm, HEADS

B, S, G = 4, 224, 14


@pytest.fixture(scope="module")
def bb():
    return DinoV3("s", img_size=S, pretrained=False).eval()


def batch():
    torch.manual_seed(0)
    return (torch.randn(B, 3, S, S),
            torch.randn(B, 3) * 5 + 10,          # raw kurtosis scale
            torch.randn(B, 3, G, G))


@pytest.mark.parametrize("fusion", list(HEADS))
def test_every_arm_runs_and_shares_one_signature(bb, fusion):
    d = Detector(bb, fusion).eval()
    x, f, fm = batch()
    logits, extra = d(x, f, fm)
    assert logits.shape == (B,) and torch.isfinite(logits).all()
    assert torch.isfinite(d.loss(logits, torch.randint(0, 2, (B,)), extra)).all()


@pytest.mark.parametrize("fusion", ["h1", "h2"])
def test_zero_init_starts_identical_to_baseline(bb, fusion):
    """FiLM gamma=1/beta=0 and a zero-init projection must be exact no-ops at step 0."""
    torch.manual_seed(0)
    base = Detector(bb, "none").eval()
    torch.manual_seed(0)
    arm = Detector(bb, fusion).eval()
    arm.head.net.load_state_dict(base.head.net.state_dict())

    x, f, fm = batch()
    with torch.no_grad():
        a, _ = base(x, f, fm)
        b, _ = arm(x, f, fm)
    assert torch.allclose(a, b, atol=1e-5), f"{fusion} is not a no-op at init"


def test_h2_rejects_grid_mismatch(bb):
    """A silent misalignment here would corrupt every dense feature."""
    d = Detector(bb, "h2").eval()
    x, f, _ = batch()
    with pytest.raises(RuntimeError, match="patch tokens"):
        d(x, f, torch.randn(B, 3, 8, 8))          # wrong grid


def test_h2_does_not_leak_side_channel_between_batches(bb):
    """The hook state must be cleared, or batch N+1 silently gets batch N's forensics."""
    d = Detector(bb, "h2").eval()
    x, f, fm = batch()
    with torch.no_grad():
        d(x, f, fm)
    assert d.head._side is None
    # The bare backbone must now behave as if no hook existed.
    with torch.no_grad():
        assert torch.isfinite(bb(x)).all()


def test_h2_side_channel_actually_changes_output_once_trained(bb):
    """After the zero-init projection is perturbed, feature maps must matter."""
    d = Detector(bb, "h2").eval()
    torch.nn.init.normal_(d.head.proj[-1].weight, std=0.02)
    x, f, fm = batch()
    with torch.no_grad():
        a, _ = d(x, f, fm)
        b, _ = d(x, f, torch.zeros_like(fm))
    assert not torch.allclose(a, b)


def test_feature_norm_refuses_to_train_unfitted(bb):
    """Guards the H3 hazard: raw kurtosis (~10) flowing into an unfitted norm."""
    d = Detector(bb, "h3").train()
    x, f, fm = batch()
    with pytest.raises(RuntimeError, match="fit"):
        d(x, f, fm)
    d.fit_feature_norm(torch.randn(64, 3) * 5 + 10)
    assert torch.isfinite(d(x, f, fm)[0]).all()


def test_fit_feature_norm_freezes_stats(bb):
    """Frozen, not batch-estimated: a BatchNorm here would leak between samples."""
    d = Detector(bb, "h0")
    train = torch.randn(256, 3) * 5 + 10
    d.fit_feature_norm(train)
    n = [m for m in d.modules() if isinstance(m, FeatureNorm)][0]
    mu = n.mu.clone()
    d.train()
    d(*batch()[:1], batch()[1], batch()[2]) if False else None
    with torch.no_grad():
        d(batch()[0], torch.randn(B, 3) * 50, batch()[2])   # wildly different batch
    assert torch.equal(n.mu, mu), "statistics moved at inference time"


def test_feature_norm_sigma_floor():
    """A degenerate (constant) feature must not blow up the normalised value."""
    n = FeatureNorm().fit(torch.zeros(100, 3))
    assert torch.isfinite(n(torch.ones(4, 3))).all()


def test_h3_aux_loss_contributes(bb):
    d = Detector(bb, "h3").eval()
    d.fit_feature_norm(torch.randn(256, 3) * 5 + 10)
    x, f, fm = batch()
    logits, extra = d(x, f, fm)
    y = torch.randint(0, 2, (B,))
    with_aux = d.loss(logits, y, extra)
    without = d.loss(logits, y, {})
    assert with_aux > without and "aux_pred" in extra
