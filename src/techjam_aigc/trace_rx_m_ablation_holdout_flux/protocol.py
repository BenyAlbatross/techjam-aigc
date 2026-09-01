"""S0 metadata, codec, and dimension shortcut probes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUISANCE_COLUMNS = {"width", "height", "bytes", "format"}


def run_nuisance_probes(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260831,
) -> dict[str, float]:
    """Return lineage-grouped out-of-fold nuisance-only ROC-AUC values."""

    missing = NUISANCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"S0 nuisance probes require columns: {sorted(missing)}")
    if not {"target", "lineage_id"} <= set(frame.columns):
        raise ValueError("S0 nuisance probes require target and lineage_id.")
    if frame["target"].nunique() != 2:
        raise ValueError("S0 nuisance probes require both binary classes.")

    data = frame.copy()
    width = pd.to_numeric(data["width"], errors="coerce")
    height = pd.to_numeric(data["height"], errors="coerce")
    byte_count = pd.to_numeric(data["bytes"], errors="coerce")
    data["log_width"] = np.log1p(width)
    data["log_height"] = np.log1p(height)
    data["log_pixels"] = np.log1p(width * height)
    data["log_bytes"] = np.log1p(byte_count)
    data["aspect_ratio"] = width / height.replace(0, np.nan)
    data["bytes_per_pixel"] = byte_count / (width * height).replace(0, np.nan)

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    groups = data["lineage_id"].astype(str)
    labels = data["target"].astype(int)
    specifications = {
        "dimension": (["log_width", "log_height", "log_pixels", "aspect_ratio"], []),
        "metadata": (["log_bytes", "bytes_per_pixel"], []),
        "codec": (["log_bytes", "bytes_per_pixel"], ["format"]),
    }
    results: dict[str, float] = {}
    for name, (numeric, categorical) in specifications.items():
        transformer = ColumnTransformer(
            [
                ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
                ("categorical", make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ), categorical),
            ],
            remainder="drop",
        )
        estimator = make_pipeline(
            transformer,
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        probabilities = cross_val_predict(
            estimator,
            data,
            labels,
            groups=groups,
            cv=splitter,
            method="predict_proba",
        )[:, 1]
        auc = float(roc_auc_score(labels, probabilities))
        # Shortcut direction is irrelevant: perfect inverse separation is also
        # a perfect nuisance cue.
        results[name] = max(auc, 1.0 - auc)
    return results
