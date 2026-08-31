"""Grouped evaluation, uncertainty, shortcut audits, and decision ledger."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .registry import FEATURE_REGISTRY, feature_names, registry_frame
from .transforms import transform_frame


_METRIC_COLUMNS = [
    "oriented_auprc", "positive_prevalence", "auprc_gain",
    "normalized_auprc", "ci_low", "ci_high", "n_real", "n_aigc",
    "parents", "low_power", "direction_reversal",
]
_DRIFT_COLUMNS = [
    "view", "condition", "feature", "target", "transform_family",
    "official_transform", "direction", "paired_parents", "signed_drift_mean",
    "signed_drift_median", "signed_drift_std", "absolute_drift_mean",
    "absolute_drift_median", "within_parent_transform_variance_mean",
    "within_parent_transform_variance_median",
]
_AUPRC_DROP_COLUMNS = [
    "view", "feature", "direction", "condition", "transform_family", "severity",
    "clean_auprc", "condition_auprc", "auprc_drop", "parents", "low_power",
    "direction_reversal",
]
_SEVERITY_AREA_COLUMNS = [
    "view", "feature", "direction", "transform_family", "severity_conditions",
    "max_severity", "clean_auprc", "normalized_severity_auprc_area",
    "normalized_severity_auprc_drop_area", "worst_condition_auprc",
    "worst_auprc_drop",
]
_INTERACTION_COLUMNS = [
    "view", "feature", "direction", "pair_condition", "first_operation",
    "second_operation", "first_single_condition", "second_single_condition",
    "clean_auprc", "first_single_auprc", "second_single_auprc", "pair_auprc",
    "first_single_auprc_drop", "second_single_auprc_drop", "pair_auprc_drop",
    "interaction_excess_auprc_drop", "parents", "low_power",
]
_ORDER_COLUMNS = [
    "view", "feature", "direction", "operation_a", "operation_b",
    "condition_a_then_b", "condition_b_then_a", "auprc_a_then_b",
    "auprc_b_then_a", "auprc_a_then_b_minus_b_then_a", "absolute_order_sensitivity",
    "interaction_excess_a_then_b", "interaction_excess_b_then_a",
]
_CHRONOLOGICAL_COLUMNS = [
    "window_column", "chronological_window", "phase", "view", "condition",
    "feature", "direction", *_METRIC_COLUMNS,
]
_ABLATION_COLUMNS = [
    "view", "baseline", "condition", "auprc", "positive_prevalence",
    "auprc_gain", "normalized_auprc", "balanced_accuracy", "parents",
]
_PREDICTION_COLUMNS = [
    "view", "condition", "parent_id", "local_path", "dataset", "generation_model",
    "source_dataset", "target", "pred",
]
_LEAVEOUT_COLUMNS = [
    "view", "heldout_dataset", "auprc", "positive_prevalence", "auprc_gain",
    "normalized_auprc", "balanced_accuracy", "train_parents", "test_parents",
]
_DECISION_COLUMNS = [
    "feature", "decision", "decision_confidence", "decision_basis",
    "clean_confirmation_auprc", "clean_confirmation_normalized_auprc",
    "native_clean_auprc", "native_clean_normalized_auprc",
    "preprocessing_normalized_auprc_gap", "worst_official_transform_auprc",
    "worst_official_transform_normalized_auprc", "worst_official_condition",
    "worst_powered_dataset_normalized_auprc",
    "best_powered_generator_normalized_auprc",
    "worst_powered_generator_normalized_auprc", "powered_generator_direction_reversals",
    "underpowered_generator_direction_reversals", "max_abs_nuisance_spearman",
    "strongest_nuisance", "provisional_due_to_sampling", "powered_generator_groups",
    "underpowered_generator_groups", "rule_note",
]
_FAILURE_COLUMNS = [*_PREDICTION_COLUMNS, "error_type", "rank"]


def _empty(columns: list[str]) -> pd.DataFrame:
    """Return an empty, schema-stable result table."""

    return pd.DataFrame(columns=columns)


def _available_feature_names(features: pd.DataFrame, *, role: str | None = None) -> list[str]:
    """Return registered feature names that are present in this extraction schema."""

    return [name for name in feature_names(role=role) if name in features.columns]


def _seed(text: str, base_seed: int) -> int:
    digest = sha256(f"{base_seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores) & pd.notna(labels)
    labels = labels[finite]
    scores = scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _normalized_auprc(auprc: float, positive_prevalence: float) -> float:
    """Map the prevalence baseline to 0 and perfect AUPRC to 1."""

    if not np.isfinite(auprc) or not 0 <= positive_prevalence < 1:
        return float("nan")
    return float((auprc - positive_prevalence) / (1.0 - positive_prevalence))


def _metric_row(
    frame: pd.DataFrame,
    feature: str,
    direction: int,
    *,
    repetitions: int,
    seed: int,
    min_group_images: int,
) -> dict[str, float | int | bool]:
    usable = frame[["target", feature, "parent_id"]].dropna()
    labels = usable["target"].to_numpy(dtype=int)
    scores = usable[feature].to_numpy(dtype=float) * direction
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    oriented_auprc = _auprc(labels, scores)
    positive_prevalence = float(np.mean(labels == 1)) if len(labels) else float("nan")
    rng = np.random.default_rng(seed)
    bootstraps: list[float] = []
    if len(positives) and len(negatives):
        for _ in range(repetitions):
            sampled_positive = rng.choice(positives, size=len(positives), replace=True)
            sampled_negative = rng.choice(negatives, size=len(negatives), replace=True)
            bootstrap_scores = np.concatenate([sampled_negative, sampled_positive])
            bootstrap_labels = np.concatenate([
                np.zeros(len(sampled_negative), dtype=int),
                np.ones(len(sampled_positive), dtype=int),
            ])
            bootstraps.append(_auprc(bootstrap_labels, bootstrap_scores))
    ci_low, ci_high = (
        np.quantile(bootstraps, [0.025, 0.975]) if bootstraps else (float("nan"), float("nan"))
    )
    return {
        "oriented_auprc": oriented_auprc,
        "positive_prevalence": positive_prevalence,
        "auprc_gain": oriented_auprc - positive_prevalence,
        "normalized_auprc": _normalized_auprc(oriented_auprc, positive_prevalence),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_real": int(np.sum(labels == 0)),
        "n_aigc": int(np.sum(labels == 1)),
        "parents": int(usable["parent_id"].nunique()),
        "low_power": bool(min(np.sum(labels == 0), np.sum(labels == 1)) < min_group_images),
        "direction_reversal": bool(
            np.isfinite(oriented_auprc) and oriented_auprc < positive_prevalence
        ),
    }


def discovery_directions(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    discovery = features[(features["phase"] == "discovery") & (features["condition"] == "clean")]
    for view, view_frame in discovery.groupby("view"):
        for feature in _available_feature_names(features):
            labels = view_frame["target"].to_numpy(dtype=int)
            values = view_frame[feature].to_numpy(dtype=float)
            forward_auprc = _auprc(labels, values)
            reverse_auprc = _auprc(labels, -values)
            direction = 1 if forward_auprc >= reverse_auprc else -1
            selected_auprc = max(forward_auprc, reverse_auprc)
            positive_prevalence = float(np.mean(labels == 1))
            rows.append({
                "view": view,
                "feature": feature,
                "forward_discovery_auprc": forward_auprc,
                "reverse_discovery_auprc": reverse_auprc,
                "selected_discovery_auprc": selected_auprc,
                "positive_prevalence": positive_prevalence,
                "normalized_discovery_auprc": _normalized_auprc(
                    selected_auprc, positive_prevalence
                ),
                "direction": direction,
            })
    return pd.DataFrame(
        rows,
        columns=[
            "view", "feature", "forward_discovery_auprc",
            "reverse_discovery_auprc", "selected_discovery_auprc",
            "positive_prevalence", "normalized_discovery_auprc", "direction",
        ],
    )


def _evaluate_groups(
    features: pd.DataFrame,
    directions: pd.DataFrame,
    group_columns: list[str],
    *,
    repetitions: int,
    base_seed: int,
    min_group_images: int,
) -> pd.DataFrame:
    rows = []
    direction_lookup = directions.set_index(["view", "feature"])["direction"].to_dict()
    grouped = features.groupby(group_columns, dropna=False, sort=True)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, keys))
        view = str(group_values["view"])
        for feature in _available_feature_names(features):
            if (view, feature) not in direction_lookup:
                continue
            direction = int(direction_lookup[(view, feature)])
            result = _metric_row(
                group,
                feature,
                direction,
                repetitions=repetitions,
                seed=_seed("|".join(map(str, keys)) + feature, base_seed),
                min_group_images=min_group_images,
            )
            rows.append({**group_values, "feature": feature, "direction": direction, **result})
    return pd.DataFrame(rows, columns=[*group_columns, "feature", "direction", *_METRIC_COLUMNS])


def univariate_tables(
    features: pd.DataFrame,
    *,
    repetitions: int,
    base_seed: int,
    min_group_images: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directions = discovery_directions(features)
    confirmation = features[features["phase"] == "confirmation"]
    transform_metrics = _evaluate_groups(
        confirmation,
        directions,
        ["view", "condition", "transform_family", "official_transform"],
        repetitions=repetitions,
        base_seed=base_seed,
        min_group_images=min_group_images,
    )
    clean = confirmation[confirmation["condition"] == "clean"]
    dataset_metrics = _evaluate_groups(
        clean,
        directions,
        ["view", "dataset"],
        repetitions=repetitions,
        base_seed=base_seed + 1,
        min_group_images=min_group_images,
    )

    generator_frames = []
    for (view, dataset), subset in clean.groupby(["view", "dataset"]):
        authentic = subset[subset["target"] == 0]
        for generator, generated in subset[subset["target"] == 1].groupby("generation_model"):
            comparison = pd.concat([authentic, generated], ignore_index=True)
            comparison = comparison.assign(comparison_generator=str(generator))
            generator_frames.append(comparison)
    generator_input = (
        pd.concat(generator_frames, ignore_index=True)
        if generator_frames
        else confirmation.assign(comparison_generator=pd.Series(dtype="object")).iloc[0:0]
    )
    generator_metrics = _evaluate_groups(
        generator_input,
        directions,
        ["view", "dataset", "comparison_generator"],
        repetitions=repetitions,
        base_seed=base_seed + 2,
        min_group_images=min_group_images,
    )
    return directions, transform_metrics, dataset_metrics, generator_metrics


def shortcut_correlations(features: pd.DataFrame) -> pd.DataFrame:
    clean = features[(features["phase"] == "discovery") & (features["condition"] == "clean")]
    candidates = _available_feature_names(features, role="candidate")
    nuisances = _available_feature_names(features, role="nuisance")
    rows = []
    for view, subset in clean.groupby("view"):
        for feature in candidates:
            correlations = {}
            for nuisance in nuisances:
                if subset[feature].nunique(dropna=True) < 2 or subset[nuisance].nunique(dropna=True) < 2:
                    coefficient = 0.0
                else:
                    coefficient = spearmanr(
                        subset[feature], subset[nuisance], nan_policy="omit"
                    ).statistic
                correlations[nuisance] = float(coefficient) if np.isfinite(coefficient) else 0.0
            if not correlations:
                continue
            strongest = max(correlations, key=lambda name: abs(correlations[name]))
            rows.append({
                "view": view,
                "feature": feature,
                "strongest_nuisance": strongest,
                "nuisance_spearman": correlations[strongest],
                "max_abs_nuisance_spearman": abs(correlations[strongest]),
            })
    return pd.DataFrame(rows, columns=[
        "view", "feature", "strongest_nuisance", "nuisance_spearman",
        "max_abs_nuisance_spearman",
    ])


def _fit_model(train: pd.DataFrame, columns: list[str], seed: int):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(train[columns], train["target"])
    return model


def family_ablation_tables(features: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = registry_frame()
    registry = registry[registry["name"].isin(features.columns)]
    candidates = registry[registry["role"] == "candidate"]
    families = {
        family: group["name"].tolist()
        for family, group in candidates.groupby("family", sort=True)
    }
    families["engineered_all"] = candidates["name"].tolist()
    families["nuisance_only"] = registry[registry["role"] == "nuisance"]["name"].tolist()

    result_rows, prediction_rows, leaveout_rows = [], [], []
    for view, subset in features.groupby("view"):
        train = subset[(subset["phase"] == "discovery") & (subset["condition"] == "clean")]
        test = subset[subset["phase"] == "confirmation"]
        for family, columns in families.items():
            if not columns or train.empty or train["target"].nunique() < 2:
                continue
            model = _fit_model(train, columns, seed)
            for condition, condition_frame in test.groupby("condition"):
                probabilities = model.predict_proba(condition_frame[columns])[:, 1]
                labels = condition_frame["target"].to_numpy(dtype=int)
                auprc = _auprc(labels, probabilities)
                positive_prevalence = float(np.mean(labels == 1))
                result_rows.append({
                    "view": view,
                    "baseline": family,
                    "condition": condition,
                    "auprc": auprc,
                    "positive_prevalence": positive_prevalence,
                    "auprc_gain": auprc - positive_prevalence,
                    "normalized_auprc": _normalized_auprc(auprc, positive_prevalence),
                    "balanced_accuracy": float(balanced_accuracy_score(condition_frame["target"], probabilities >= 0.5)),
                    "parents": int(condition_frame["parent_id"].nunique()),
                })
                if family == "engineered_all":
                    for row, probability in zip(condition_frame.to_dict("records"), probabilities):
                        prediction_rows.append({
                            "view": view,
                            "condition": condition,
                            "parent_id": row["parent_id"],
                            "local_path": row["local_path"],
                            "dataset": row["dataset"],
                            "generation_model": row["generation_model"],
                            "source_dataset": row["source_dataset"],
                            "target": row["target"],
                            "pred": float(probability),
                        })

        all_columns = families["engineered_all"]
        if not all_columns:
            continue
        for heldout_dataset in sorted(subset["dataset"].unique()):
            leave_train = train[train["dataset"] != heldout_dataset]
            leave_test = subset[
                (subset["phase"] == "confirmation")
                & (subset["condition"] == "clean")
                & (subset["dataset"] == heldout_dataset)
            ]
            if leave_train["target"].nunique() < 2 or leave_test["target"].nunique() < 2:
                continue
            model = _fit_model(leave_train, all_columns, seed)
            probabilities = model.predict_proba(leave_test[all_columns])[:, 1]
            labels = leave_test["target"].to_numpy(dtype=int)
            auprc = _auprc(labels, probabilities)
            positive_prevalence = float(np.mean(labels == 1))
            leaveout_rows.append({
                "view": view,
                "heldout_dataset": heldout_dataset,
                "auprc": auprc,
                "positive_prevalence": positive_prevalence,
                "auprc_gain": auprc - positive_prevalence,
                "normalized_auprc": _normalized_auprc(auprc, positive_prevalence),
                "balanced_accuracy": float(balanced_accuracy_score(leave_test["target"], probabilities >= 0.5)),
                "train_parents": int(leave_train["parent_id"].nunique()),
                "test_parents": int(leave_test["parent_id"].nunique()),
            })
    return (
        pd.DataFrame(result_rows, columns=_ABLATION_COLUMNS),
        pd.DataFrame(prediction_rows, columns=_PREDICTION_COLUMNS),
        pd.DataFrame(leaveout_rows, columns=_LEAVEOUT_COLUMNS),
    )



def parent_paired_feature_drift(
    features: pd.DataFrame,
    directions: pd.DataFrame | None = None,
    *,
    phase: str = "confirmation",
) -> pd.DataFrame:
    """Summarize oriented condition-minus-clean drift using the same parents.

    ``within_parent_transform_variance`` is computed across all available
    conditions for a parent/view/feature, then summarized within each reported
    condition and class. The orientation is learned from discovery-clean only.
    """

    required = {"phase", "condition", "view", "parent_id", "target"}
    if not required <= set(features.columns):
        return _empty(_DRIFT_COLUMNS)
    evaluation = features[features["phase"] == phase].copy()
    if evaluation.empty or "clean" not in set(evaluation["condition"]):
        return _empty(_DRIFT_COLUMNS)
    directions = discovery_directions(features) if directions is None else directions
    direction_lookup = directions.set_index(["view", "feature"])["direction"].to_dict()
    feature_columns = _available_feature_names(evaluation)
    rows: list[dict[str, object]] = []

    for feature in feature_columns:
        clean = (
            evaluation[evaluation["condition"] == "clean"]
            .groupby(["parent_id", "view", "target"], as_index=False, dropna=False)[feature]
            .mean()
            .rename(columns={feature: "clean_value"})
        )
        transformed = evaluation[evaluation["condition"] != "clean"].copy()
        if transformed.empty:
            continue
        transformed = transformed.merge(
            clean,
            on=["parent_id", "view", "target"],
            how="inner",
            validate="many_to_one",
        )
        parent_variance = (
            evaluation.groupby(["parent_id", "view", "target"], dropna=False)[feature]
            .var(ddof=0)
            .rename("within_parent_transform_variance")
            .reset_index()
        )
        transformed = transformed.merge(
            parent_variance,
            on=["parent_id", "view", "target"],
            how="left",
            validate="many_to_one",
        )
        transformed["direction"] = transformed["view"].map(
            lambda view: direction_lookup.get((str(view), feature), np.nan)
        )
        transformed = transformed.dropna(
            subset=[feature, "clean_value", "direction", "within_parent_transform_variance"]
        )
        transformed["signed_drift"] = transformed["direction"] * (
            transformed[feature] - transformed["clean_value"]
        )
        transformed["absolute_drift"] = transformed["signed_drift"].abs()
        if "transform_family" not in transformed:
            transformed["transform_family"] = transformed["condition"].map(
                transform_frame("all").set_index("name")["family"]
            )
        if "official_transform" not in transformed:
            transformed["official_transform"] = transformed["condition"].map(
                transform_frame("all").set_index("name")["official"]
            )
        group_columns = [
            "view", "condition", "target", "transform_family", "official_transform", "direction"
        ]
        for group_key, group in transformed.groupby(group_columns, dropna=False, sort=True):
            values = dict(zip(group_columns, group_key))
            rows.append({
                "view": values["view"],
                "condition": values["condition"],
                "feature": feature,
                "target": int(values["target"]),
                "transform_family": values["transform_family"],
                "official_transform": bool(values["official_transform"]),
                "direction": int(values["direction"]),
                "paired_parents": int(group["parent_id"].nunique()),
                "signed_drift_mean": float(group["signed_drift"].mean()),
                "signed_drift_median": float(group["signed_drift"].median()),
                "signed_drift_std": float(group["signed_drift"].std(ddof=0)),
                "absolute_drift_mean": float(group["absolute_drift"].mean()),
                "absolute_drift_median": float(group["absolute_drift"].median()),
                "within_parent_transform_variance_mean": float(
                    group["within_parent_transform_variance"].mean()
                ),
                "within_parent_transform_variance_median": float(
                    group["within_parent_transform_variance"].median()
                ),
            })
    return pd.DataFrame(rows, columns=_DRIFT_COLUMNS)


def auprc_degradation_tables(
    transform_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clean-to-condition AUPRC drops and severity-integrated AUPRC."""

    required = {"view", "feature", "condition", "oriented_auprc", "direction"}
    if transform_metrics.empty or not required <= set(transform_metrics.columns):
        return _empty(_AUPRC_DROP_COLUMNS), _empty(_SEVERITY_AREA_COLUMNS)
    metadata = transform_frame("all").set_index("name")
    official_singles = metadata[
        metadata["official"] & (metadata["step_count"] == 1)
    ]
    clean_lookup = (
        transform_metrics[transform_metrics["condition"] == "clean"]
        .set_index(["view", "feature"])["oriented_auprc"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for metric in transform_metrics.to_dict("records"):
        condition = str(metric["condition"])
        if condition not in official_singles.index:
            continue
        key = (metric["view"], metric["feature"])
        if key not in clean_lookup:
            continue
        clean_auprc = float(clean_lookup[key])
        condition_auprc = float(metric["oriented_auprc"])
        rows.append({
            "view": metric["view"],
            "feature": metric["feature"],
            "direction": int(metric["direction"]),
            "condition": condition,
            "transform_family": official_singles.loc[condition, "family"],
            "severity": float(official_singles.loc[condition, "severity"]),
            "clean_auprc": clean_auprc,
            "condition_auprc": condition_auprc,
            "auprc_drop": clean_auprc - condition_auprc,
            "parents": int(metric.get("parents", 0)),
            "low_power": bool(metric.get("low_power", False)),
            "direction_reversal": bool(metric.get("direction_reversal", False)),
        })
    drops = pd.DataFrame(rows, columns=_AUPRC_DROP_COLUMNS)

    area_rows: list[dict[str, object]] = []
    for (view, feature, family), group in drops.groupby(
        ["view", "feature", "transform_family"], sort=True
    ):
        # Signed color jitter and one-point crop do not define a monotone
        # nonnegative severity curve. Require at least two positive severities.
        if len(group) < 2 or (group["severity"] <= 0).any():
            continue
        group = group.sort_values("severity")
        clean_auprc = float(group["clean_auprc"].iloc[0])
        x = np.concatenate([[0.0], group["severity"].to_numpy(dtype=float)])
        y = np.concatenate([[clean_auprc], group["condition_auprc"].to_numpy(dtype=float)])
        if len(np.unique(x)) != len(x) or x[-1] <= 0:
            continue
        normalized_area = float(np.trapezoid(y, x) / x[-1])
        area_rows.append({
            "view": view,
            "feature": feature,
            "direction": int(group["direction"].iloc[0]),
            "transform_family": family,
            "severity_conditions": int(len(group)),
            "max_severity": float(x[-1]),
            "clean_auprc": clean_auprc,
            "normalized_severity_auprc_area": normalized_area,
            "normalized_severity_auprc_drop_area": clean_auprc - normalized_area,
            "worst_condition_auprc": float(group["condition_auprc"].min()),
            "worst_auprc_drop": float(group["auprc_drop"].max()),
        })
    return drops, pd.DataFrame(area_rows, columns=_SEVERITY_AREA_COLUMNS)


def _canonical_recipe_key(recipe: list[dict[str, object]]) -> str:
    """Normalize JSON's numerically equivalent ``1`` and ``1.0`` parameters."""

    def normalize(value: object) -> object:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return json.dumps(normalize(recipe), sort_keys=True, separators=(",", ":"))


def _single_condition_by_recipe(metadata: pd.DataFrame) -> dict[str, str]:
    singles = metadata[metadata["step_count"] == 1]
    return {
        _canonical_recipe_key(json.loads(row.ordered_recipe_json)): row.name
        for row in singles.itertuples()
    }


def directed_pair_interactions(transform_metrics: pd.DataFrame) -> pd.DataFrame:
    """Measure pair degradation beyond the sum of its constituent single effects."""

    required = {"view", "feature", "condition", "oriented_auprc", "direction"}
    if transform_metrics.empty or not required <= set(transform_metrics.columns):
        return _empty(_INTERACTION_COLUMNS)
    metadata = transform_frame("all")
    pairs = metadata[metadata["design"] == "directed_medium_pair"]
    available_conditions = set(transform_metrics["condition"])
    pairs = pairs[pairs["name"].isin(available_conditions)]
    if pairs.empty:
        return _empty(_INTERACTION_COLUMNS)
    single_by_recipe = _single_condition_by_recipe(metadata)
    metric_lookup = transform_metrics.set_index(["view", "feature", "condition"])
    rows: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        recipe = json.loads(pair["ordered_recipe_json"])
        if len(recipe) != 2:
            continue
        single_conditions = [
            single_by_recipe.get(_canonical_recipe_key([step]))
            for step in recipe
        ]
        if any(condition is None for condition in single_conditions):
            continue
        first_condition, second_condition = single_conditions
        for view, feature in transform_metrics[["view", "feature"]].drop_duplicates().itertuples(index=False):
            keys = [
                (view, feature, "clean"),
                (view, feature, first_condition),
                (view, feature, second_condition),
                (view, feature, pair["name"]),
            ]
            if any(key not in metric_lookup.index for key in keys):
                continue
            clean, first, second, combined = [metric_lookup.loc[key] for key in keys]
            clean_auprc = float(clean["oriented_auprc"])
            first_auprc = float(first["oriented_auprc"])
            second_auprc = float(second["oriented_auprc"])
            pair_auprc = float(combined["oriented_auprc"])
            first_drop = clean_auprc - first_auprc
            second_drop = clean_auprc - second_auprc
            pair_drop = clean_auprc - pair_auprc
            rows.append({
                "view": view,
                "feature": feature,
                "direction": int(combined["direction"]),
                "pair_condition": pair["name"],
                "first_operation": recipe[0]["operation"],
                "second_operation": recipe[1]["operation"],
                "first_single_condition": first_condition,
                "second_single_condition": second_condition,
                "clean_auprc": clean_auprc,
                "first_single_auprc": first_auprc,
                "second_single_auprc": second_auprc,
                "pair_auprc": pair_auprc,
                "first_single_auprc_drop": first_drop,
                "second_single_auprc_drop": second_drop,
                "pair_auprc_drop": pair_drop,
                "interaction_excess_auprc_drop": pair_drop - first_drop - second_drop,
                "parents": int(combined.get("parents", 0)),
                "low_power": bool(combined.get("low_power", False)),
            })
    return pd.DataFrame(rows, columns=_INTERACTION_COLUMNS)


def directed_pair_order_sensitivity(interactions: pd.DataFrame) -> pd.DataFrame:
    """Compare A->B and B->A for exact reversed registered recipes."""

    if interactions.empty or "pair_condition" not in interactions:
        return _empty(_ORDER_COLUMNS)
    metadata = transform_frame("all")
    pairs = metadata[metadata["design"] == "directed_medium_pair"].copy()
    recipe_lookup = {
        _canonical_recipe_key(json.loads(row.ordered_recipe_json)): row.name
        for row in pairs.itertuples()
    }
    reverse_lookup: dict[str, str] = {}
    for row in pairs.itertuples():
        recipe = json.loads(row.ordered_recipe_json)
        reverse_key = _canonical_recipe_key(list(reversed(recipe)))
        if reverse_key in recipe_lookup:
            reverse_lookup[row.name] = recipe_lookup[reverse_key]
    indexed = interactions.set_index(["view", "feature", "pair_condition"])
    rows: list[dict[str, object]] = []
    for (view, feature), group in interactions.groupby(["view", "feature"], sort=True):
        present = set(group["pair_condition"])
        for condition_ab in sorted(present):
            condition_ba = reverse_lookup.get(condition_ab)
            if condition_ba not in present or condition_ab >= condition_ba:
                continue
            ab = indexed.loc[(view, feature, condition_ab)]
            ba = indexed.loc[(view, feature, condition_ba)]
            delta = float(ab["pair_auprc"] - ba["pair_auprc"])
            rows.append({
                "view": view,
                "feature": feature,
                "direction": int(ab["direction"]),
                "operation_a": ab["first_operation"],
                "operation_b": ab["second_operation"],
                "condition_a_then_b": condition_ab,
                "condition_b_then_a": condition_ba,
                "auprc_a_then_b": float(ab["pair_auprc"]),
                "auprc_b_then_a": float(ba["pair_auprc"]),
                "auprc_a_then_b_minus_b_then_a": delta,
                "absolute_order_sensitivity": abs(delta),
                "interaction_excess_a_then_b": float(ab["interaction_excess_auprc_drop"]),
                "interaction_excess_b_then_a": float(ba["interaction_excess_auprc_drop"]),
            })
    return pd.DataFrame(rows, columns=_ORDER_COLUMNS)


def chronological_confirmation_metrics(
    features: pd.DataFrame,
    directions: pd.DataFrame | None = None,
    *,
    repetitions: int,
    base_seed: int,
    min_group_images: int,
) -> pd.DataFrame:
    """Evaluate frozen directions by chronological window without selecting on them.

    A window field is mandatory so merely adding a ``final_confirmation`` phase
    cannot accidentally masquerade as a chronological evaluation.
    """

    window_candidates = (
        "chronological_window", "generator_window", "release_window", "time_window", "window"
    )
    window_column = next((name for name in window_candidates if name in features.columns), None)
    required = {"phase", "view", "condition", "target", "parent_id"}
    if window_column is None or not required <= set(features.columns):
        return _empty(_CHRONOLOGICAL_COLUMNS)
    eligible = features[
        features["phase"].isin(["confirmation", "final_confirmation"])
        & features[window_column].notna()
    ].copy()
    if eligible.empty:
        return _empty(_CHRONOLOGICAL_COLUMNS)
    directions = discovery_directions(features) if directions is None else directions
    metrics = _evaluate_groups(
        eligible,
        directions,
        ["phase", "view", "condition", window_column],
        repetitions=repetitions,
        base_seed=base_seed,
        min_group_images=min_group_images,
    )
    if metrics.empty:
        return _empty(_CHRONOLOGICAL_COLUMNS)
    metrics = metrics.rename(columns={window_column: "chronological_window"})
    metrics.insert(0, "window_column", window_column)
    return metrics[_CHRONOLOGICAL_COLUMNS]

def decision_ledger(
    transform_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    evaluated = set(transform_metrics.get("feature", pd.Series(dtype=str)))
    candidates = [name for name in feature_names(role="candidate") if name in evaluated]
    official_names = set(transform_frame().query("official and name != 'clean'")["name"])
    rows = []
    for feature in candidates:
        clean_views = transform_metrics[(transform_metrics["feature"] == feature) & (transform_metrics["condition"] == "clean")]
        canonical = clean_views[clean_views["view"] == "canonical_128"]
        native = clean_views[clean_views["view"] == "native_capped"]
        if canonical.empty or native.empty:
            continue
        clean_auprc = float(canonical["oriented_auprc"].iloc[0])
        clean_normalized = float(canonical["normalized_auprc"].iloc[0])
        native_auprc = float(native["oriented_auprc"].iloc[0])
        native_normalized = float(native["normalized_auprc"].iloc[0])
        transformed = transform_metrics[
            (transform_metrics["feature"] == feature)
            & (transform_metrics["view"] == "canonical_128")
            & (transform_metrics["condition"].isin(official_names))
        ]
        datasets = dataset_metrics[(dataset_metrics["feature"] == feature) & (dataset_metrics["view"] == "canonical_128")]
        generators = generator_metrics[(generator_metrics["feature"] == feature) & (generator_metrics["view"] == "canonical_128")]
        powered_datasets = datasets[~datasets["low_power"]]
        powered_generators = generators[~generators["low_power"]]
        underpowered_generators = generators[generators["low_power"]]
        correlation_rows = correlations[
            (correlations["feature"] == feature)
            & (correlations["view"] == "canonical_128")
        ]
        if (
            transformed.empty
            or powered_datasets.empty
            or powered_generators.empty
            or correlation_rows.empty
        ):
            continue
        corr = correlation_rows.iloc[0]

        worst_transform_row = transformed.loc[transformed["normalized_auprc"].idxmin()]
        worst_transform = float(worst_transform_row["oriented_auprc"])
        worst_transform_normalized = float(worst_transform_row["normalized_auprc"])
        worst_condition = str(worst_transform_row["condition"])
        worst_dataset_normalized = float(powered_datasets["normalized_auprc"].min())
        best_generator_normalized = float(powered_generators["normalized_auprc"].max())
        worst_generator_normalized = float(powered_generators["normalized_auprc"].min())
        powered_reversals = int(powered_generators["direction_reversal"].sum())
        underpowered_reversals = int(underpowered_generators["direction_reversal"].sum())
        preprocessing_gap = clean_normalized - native_normalized
        nuisance_corr = float(corr["max_abs_nuisance_spearman"])

        if clean_normalized >= 0.20 and (abs(preprocessing_gap) > 0.20 or nuisance_corr >= 0.75):
            decision = "shortcut"
            decision_basis = "Strong separation is coupled to preprocessing or a preregistered nuisance."
        elif clean_normalized < 0.10:
            decision = "discard"
            decision_basis = "Aggregate confirmation AUPRC is too close to its prevalence baseline."
        elif best_generator_normalized >= 0.40 and (
            worst_generator_normalized < 0.10 or powered_reversals > 0
        ):
            decision = "specialist"
            decision_basis = "Powered generator groups disagree materially."
        elif clean_normalized >= 0.20 and worst_transform_normalized < 0.10:
            decision = "fragile"
            decision_basis = "Clean separation does not survive the worst official transform."
        elif (
            clean_normalized >= 0.20
            and worst_transform_normalized >= 0.10
            and worst_dataset_normalized >= 0.10
            and worst_generator_normalized >= 0.16
            and powered_reversals == 0
        ):
            decision = "keep"
            decision_basis = "The feature clears clean, powered-group, and official-transform screens."
        else:
            decision = "discard"
            decision_basis = "Evidence is insufficient for the other preregistered categories."

        provisional = bool(len(underpowered_generators) or len(powered_generators) < 3)
        decision_confidence = "low" if underpowered_reversals else "medium"
        rows.append({
            "feature": feature,
            "decision": decision,
            "decision_confidence": decision_confidence,
            "decision_basis": decision_basis,
            "clean_confirmation_auprc": clean_auprc,
            "clean_confirmation_normalized_auprc": clean_normalized,
            "native_clean_auprc": native_auprc,
            "native_clean_normalized_auprc": native_normalized,
            "preprocessing_normalized_auprc_gap": preprocessing_gap,
            "worst_official_transform_auprc": worst_transform,
            "worst_official_transform_normalized_auprc": worst_transform_normalized,
            "worst_official_condition": worst_condition,
            "worst_powered_dataset_normalized_auprc": worst_dataset_normalized,
            "best_powered_generator_normalized_auprc": best_generator_normalized,
            "worst_powered_generator_normalized_auprc": worst_generator_normalized,
            "powered_generator_direction_reversals": powered_reversals,
            "underpowered_generator_direction_reversals": underpowered_reversals,
            "max_abs_nuisance_spearman": nuisance_corr,
            "strongest_nuisance": corr["strongest_nuisance"],
            "provisional_due_to_sampling": provisional,
            "powered_generator_groups": int(len(powered_generators)),
            "underpowered_generator_groups": int(len(underpowered_generators)),
            "rule_note": (
                "Exploratory normalized-AUPRC thresholds, where 0 is the prevalence "
                "baseline; not a competition acceptance criterion."
            ),
        })
    result = pd.DataFrame(rows, columns=_DECISION_COLUMNS)
    if result.empty:
        return result
    return result.sort_values(
        [
            "decision", "worst_official_transform_normalized_auprc",
            "clean_confirmation_normalized_auprc",
        ],
        ascending=[True, False, False],
    )


def representative_failures(predictions: pd.DataFrame, ablations: pd.DataFrame, *, per_type: int = 5) -> pd.DataFrame:
    if predictions.empty or ablations.empty:
        return _empty(_FAILURE_COLUMNS)
    canonical_scores = ablations[
        (ablations["view"] == "canonical_128") & (ablations["baseline"] == "engineered_all")
    ]
    official = set(transform_frame().query("official")["name"])
    ranked = canonical_scores[
        canonical_scores["condition"].isin(official)
    ].sort_values("normalized_auprc")
    selected_conditions = ["clean", *[name for name in ranked["condition"] if name != "clean"][:3]]
    rows = []
    subset = predictions[
        (predictions["view"] == "canonical_128")
        & (predictions["condition"].isin(selected_conditions))
    ]
    for condition, condition_frame in subset.groupby("condition"):
        false_positive = condition_frame[condition_frame["target"] == 0].nlargest(per_type, "pred")
        false_negative = condition_frame[condition_frame["target"] == 1].nsmallest(per_type, "pred")
        for error_type, errors in (("false_positive", false_positive), ("false_negative", false_negative)):
            for rank, row in enumerate(errors.to_dict("records"), start=1):
                rows.append({**row, "error_type": error_type, "rank": rank})
    return pd.DataFrame(rows, columns=_FAILURE_COLUMNS)


def write_evaluation_cache(
    output_dir: Path,
    features: pd.DataFrame,
    *,
    repetitions: int = 200,
    seed: int = 20260830,
    min_group_images: int = 20,
) -> dict[str, pd.DataFrame]:
    directions, transforms, datasets, generators = univariate_tables(
        features,
        repetitions=repetitions,
        base_seed=seed,
        min_group_images=min_group_images,
    )
    correlations = shortcut_correlations(features)
    ablations, predictions, leaveout = family_ablation_tables(features, seed=seed)
    ledger = decision_ledger(transforms, datasets, generators, correlations)
    failures = representative_failures(predictions, ablations)
    drift = parent_paired_feature_drift(features, directions)
    auprc_drops, severity_areas = auprc_degradation_tables(transforms)
    interactions = directed_pair_interactions(transforms)
    order_sensitivity = directed_pair_order_sensitivity(interactions)
    chronological = chronological_confirmation_metrics(
        features,
        directions,
        repetitions=repetitions,
        base_seed=seed + 3,
        min_group_images=min_group_images,
    )
    tables = {
        "directions": directions,
        "transform_metrics": transforms,
        "dataset_metrics": datasets,
        "generator_metrics": generators,
        "shortcut_correlations": correlations,
        "ablations": ablations,
        "predictions": predictions,
        "leave_one_dataset_out": leaveout,
        "decision_ledger": ledger,
        "representative_failures": failures,
        "parent_paired_feature_drift": drift,
        "clean_to_condition_auprc_drop": auprc_drops,
        "severity_area": severity_areas,
        "directed_pair_interactions": interactions,
        "directed_pair_order_sensitivity": order_sensitivity,
        "chronological_confirmation_metrics": chronological,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv.gz", index=False, compression="gzip")
    evaluation_metadata = {
        "schema_version": 2,
        "primary_metric": "auprc",
        "estimator": "sklearn.metrics.average_precision_score",
        "positive_class": "AIGC",
        "positive_target": 1,
        "chance_baseline": "positive prevalence per evaluated group",
        "normalized_auprc": "(AUPRC - prevalence) / (1 - prevalence)",
        "direction_selection": "maximum forward/reverse AUPRC on clean discovery only",
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(evaluation_metadata, indent=2) + "\n"
    )
    return tables
