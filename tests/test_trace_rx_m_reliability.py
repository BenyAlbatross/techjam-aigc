from __future__ import annotations

import numpy as np

from techjam_aigc.trace_rx_m.calibration import (
    PositiveTemperatureScaler,
    binary_logit_nll,
)
from techjam_aigc.trace_rx_m.reliability import (
    audit_heldout_availability,
    audit_quality_cell_occupancy,
    fit_shrunk_cell_statistics,
    PassiveQualityStacker,
    ReliabilityTable,
    normalized_partial_auc,
)


def test_sparse_cell_statistics_shrink_toward_global_distribution() -> None:
    values = np.array([0.0, 0.0, 10.0])
    cells = np.array([[0], [0], [1]])
    statistics, global_statistics = fit_shrunk_cell_statistics(
        values,
        cells,
        all_cells=[(0,), (1,), (2,)],
        prior_strength=10.0,
    )

    assert statistics[(0,)].count == 2
    assert statistics[(1,)].count == 1
    assert statistics[(2,)].count == 0
    assert statistics[(0,)].mean > 0.0
    assert statistics[(1,)].mean < 10.0
    assert statistics[(2,)].mean == global_statistics.mean
    assert statistics[(2,)].variance == global_statistics.variance


def _reliability_fixture() -> ReliabilityTable:
    # One descriptor dimension varies; duplicate quantiles collapse the rest.
    null_quality = np.array([[value, 0.0, 0.0, 0.0] for value in range(8)])
    null_logits = np.array([-1.2, -0.8, -1.0, -1.0, 1.8, 2.2, 2.0, 2.0])

    dev_quality = np.array(
        [[value, 0.0, 0.0, 0.0] for value in [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]]
    )
    labels = np.array([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    # Cell 0 is clean and strongly separated; cell 1 has weaker survival.
    logits = np.array([-1.2, -0.8, -1.1, -0.9, 1.2, 0.8, 1.1, 0.9,
                       -0.2, 0.2, -0.1, 0.1, 0.4, 0.0, 0.3, 0.1])
    clean = np.array([True] * 8 + [False] * 8)
    return ReliabilityTable.fit(
        authentic_null_logits=null_logits,
        authentic_null_quality=null_quality,
        development_logits=logits,
        development_labels=labels,
        development_quality=dev_quality,
        clean_mask=clean,
        n_bins=(2, 1, 1, 1),
        prior_strength=0.1,
    )


def test_reliability_is_measured_d_prime_clipped_relative_to_clean() -> None:
    table = _reliability_fixture()
    quality = np.array([[1.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]])
    availability = table.availability(quality)

    assert 0.8 <= availability[0] <= 1.0
    assert 0.0 < availability[1] < availability[0]
    assert all(0.0 <= value <= 1.0 for value in table.availability_by_cell.values())


def test_authentic_cell_normalization_and_fusion() -> None:
    table = _reliability_fixture()
    quality = np.array([[1.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]])
    logits = np.array([-1.0, 2.0])

    normalized = table.normalize_authentic(logits, quality)
    fused = table.fuse(logits, quality)

    assert abs(normalized[0]) < 0.11
    np.testing.assert_allclose(fused, normalized * table.availability(quality))

    restored = ReliabilityTable.from_dict(table.to_dict())
    np.testing.assert_allclose(restored.fuse(logits, quality), fused)


def test_positive_temperature_fit_improves_overconfident_nll() -> None:
    logits = np.array([-8.0, -4.0, 4.0, 8.0])
    labels = np.array([0, 1, 0, 1])
    scaler = PositiveTemperatureScaler().fit(logits, labels)

    assert scaler.temperature > 1.0
    assert binary_logit_nll(logits, labels, scaler.temperature) < binary_logit_nll(logits, labels, 1.0)
    probabilities = scaler.predict_proba(logits)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    restored = PositiveTemperatureScaler.from_dict(scaler.to_dict())
    np.testing.assert_allclose(restored.predict_proba(logits), probabilities)


def test_normalized_partial_auc_respects_low_fpr_ranking() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    perfect = normalized_partial_auc(labels, np.array([-4, -3, -2, -1, 1, 2]))
    reversed_score = normalized_partial_auc(labels, np.array([4, 3, 2, 1, -1, -2]))
    assert perfect == 1.0
    assert reversed_score == 0.0


def test_heldout_availability_audit_and_passive_fallback_roundtrip() -> None:
    table = _reliability_fixture()
    quality = np.array(
        [[1.0, 0.0, 0.0, 0.0]] * 8 + [[7.0, 0.0, 0.0, 0.0]] * 8
    )
    labels = np.array(([0] * 4 + [1] * 4) * 2)
    logits = np.array([-1.1, -0.9, -1.0, -1.0, 1.1, 0.9, 1.0, 1.0,
                       -0.1, 0.1, -0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
    audit = audit_heldout_availability(
        table,
        logits=logits,
        labels=labels,
        quality=quality,
        min_samples_per_class=3,
        min_spearman=0.5,
    )
    assert audit.passed
    assert audit.cell_count == 2

    occupancy = audit_quality_cell_occupancy(
        table,
        quality=quality,
        conditions=["clean"] * 8 + ["noise_sigma0.10"] * 8,
        transform_families=["clean"] * 8 + ["gaussian_noise"] * 8,
        max_clean_noise_overlap=0.8,
    )
    assert occupancy.passed
    assert occupancy.distribution_overlap == 0.0

    stacker = PassiveQualityStacker().fit(logits, quality, labels)
    fused = stacker.fused_logits(logits, quality)
    restored = PassiveQualityStacker.from_dict(stacker.to_dict())
    np.testing.assert_allclose(restored.fused_logits(logits, quality), fused)
