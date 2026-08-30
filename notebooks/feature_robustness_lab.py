import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup:
    from hashlib import sha256
    from io import BytesIO
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image
    from scipy import ndimage

    from techjam_aigc.feature_lab.data import load_binary_index, resolve_repo_root
    from techjam_aigc.feature_lab.transforms import analysis_view, apply_transform


@app.cell(hide_code=True)
def intro():
    mo.md("""
    # AIGC feature robustness laboratory

    This is a hypothesis tournament, not a pooled-label filter leaderboard. It tests whether an interpretable measurement separates authentic from purely generated images, preserves its discovery-learned direction, transfers across datasets and generators, survives every challenge transformation, and remains useful after resolution, codec, source, and preprocessing shortcuts are exposed.

    A filter output is only a probe. Every registered feature has a scalar measurement, causal hypothesis, and expected failure.
    """)
    return


@app.cell
def paths():
    repo_root = resolve_repo_root()
    cache_dir = repo_root / "data/derived/feature_lab"
    _required = [
        "features.csv.gz", "feature_registry.csv", "transform_registry.csv",
        "coverage.csv", "run_metadata.json", "directions.csv.gz",
        "transform_metrics.csv.gz", "dataset_metrics.csv.gz",
        "generator_metrics.csv.gz", "shortcut_correlations.csv.gz",
        "ablations.csv.gz", "leave_one_dataset_out.csv.gz",
        "decision_ledger.csv.gz", "representative_failures.csv.gz",
    ]
    _missing = [_name for _name in _required if not (cache_dir / _name).exists()]
    if _missing:
        raise FileNotFoundError(
            "Incomplete feature-lab cache: " + ", ".join(_missing)
            + ". Run uv run python scripts/run_feature_lab.py --workers 4 --bootstrap 200"
        )
    return cache_dir, repo_root


@app.cell
def load_results(cache_dir, repo_root):
    run_metadata = json.loads((cache_dir / "run_metadata.json").read_text())
    evaluation_metadata = json.loads((cache_dir / "evaluation_metadata.json").read_text())
    feature_registry = pd.read_csv(cache_dir / "feature_registry.csv")
    transform_registry = pd.read_csv(cache_dir / "transform_registry.csv")
    coverage = pd.read_csv(cache_dir / "coverage.csv")
    feature_table = pd.read_csv(cache_dir / "features.csv.gz")
    directions = pd.read_csv(cache_dir / "directions.csv.gz")
    transform_metrics = pd.read_csv(cache_dir / "transform_metrics.csv.gz")
    dataset_metrics = pd.read_csv(cache_dir / "dataset_metrics.csv.gz")
    generator_metrics = pd.read_csv(cache_dir / "generator_metrics.csv.gz")
    shortcut_correlations = pd.read_csv(cache_dir / "shortcut_correlations.csv.gz")
    ablations = pd.read_csv(cache_dir / "ablations.csv.gz")
    leave_one_dataset_out = pd.read_csv(cache_dir / "leave_one_dataset_out.csv.gz")
    decision_ledger = pd.read_csv(cache_dir / "decision_ledger.csv.gz")
    representative_failures = pd.read_csv(cache_dir / "representative_failures.csv.gz")

    def _optional_result(name):
        _path = cache_dir / f"{name}.csv.gz"
        return pd.read_csv(_path) if _path.exists() else pd.DataFrame()

    parent_paired_feature_drift = _optional_result("parent_paired_feature_drift")
    clean_to_condition_auprc_drop = _optional_result("clean_to_condition_auprc_drop")
    severity_area = _optional_result("severity_area")
    directed_pair_interactions = _optional_result("directed_pair_interactions")
    directed_pair_order_sensitivity = _optional_result("directed_pair_order_sensitivity")
    chronological_confirmation_metrics = _optional_result("chronological_confirmation_metrics")
    _run_index_path = Path(run_metadata["input_index"])
    if not _run_index_path.is_absolute():
        _run_index_path = repo_root / _run_index_path
    _run_index_digest = sha256(_run_index_path.read_bytes()).hexdigest()
    if _run_index_digest != run_metadata["input_index_sha256"]:
        raise RuntimeError(
            "The recorded input index has changed since feature extraction; refusing to expose mismatched images."
        )
    _loaded_binary_index = load_binary_index(_run_index_path)
    _cached_parent_ids = set(feature_table["parent_id"].astype(str))
    binary_index = _loaded_binary_index[
        _loaded_binary_index["parent_id"].astype(str).isin(_cached_parent_ids)
    ].reset_index(drop=True)
    if set(binary_index["parent_id"].astype(str)) != _cached_parent_ids:
        raise RuntimeError("The feature cache contains parent IDs missing from its recorded input index.")
    return (
        ablations,
        binary_index,
        chronological_confirmation_metrics,
        coverage,
        dataset_metrics,
        decision_ledger,
        directed_pair_interactions,
        directed_pair_order_sensitivity,
        directions,
        feature_registry,
        feature_table,
        generator_metrics,
        leave_one_dataset_out,
        parent_paired_feature_drift,
        representative_failures,
        run_metadata,
        severity_area,
        shortcut_correlations,
        transform_metrics,
        transform_registry,
    )


@app.cell(hide_code=True)
def contract(run_metadata):
    _semantic = run_metadata["semantic_control"]
    mo.vstack([
        mo.md("## Experiment contract"),
        mo.hstack([
            mo.stat(value=f"{run_metadata['binary_parent_images']:,}", label="Eligible parents", caption="SID tampered excluded"),
            mo.stat(value=f"{run_metadata['feature_rows']:,}", label="Experiment rows", caption="parent × condition × view"),
            mo.stat(value=str(run_metadata["features"]), label="Registered scalars", caption="candidate + nuisance"),
        ], widths="equal"),
        mo.callout(
            "Primary metric: AUPRC via average precision, with AIGC as target 1. Raw AUPRC is shown with its positive-prevalence baseline; cross-group grids use (AUPRC - prevalence) / (1 - prevalence), so zero means no gain over prevalence.",
            kind="info",
        ),
        mo.callout(
            "Feature orientation and model fitting use only discovery splits. Confirmation uses CIFAKE test, SID validation, and WildFake reconstructed test. The organizer demonstration-only split is rejected by the loader.",
            kind="success",
        ),
        mo.callout(
            "Implementation assumptions: Lanczos resize, 80% center crop followed by bicubic restoration, deterministic joint ±20% color factors, and four separately labeled non-official compositions.",
            kind="info",
        ),
        mo.callout(
            f"Semantic control: {_semantic['status']}. {_semantic['reason']} Optional control: {_semantic['recommended_optional_control']}",
            kind="warn",
        ),
    ])
    return


@app.cell
def implementation_audit(feature_registry, repo_root, run_metadata):
    from techjam_aigc.feature_lab.expansion import acquisition_audit, load_expansion_config
    from techjam_aigc.feature_lab.registry import registry_frame as source_registry_frame
    from techjam_aigc.feature_lab.transforms import TRANSFORM_PROFILES, get_transform_specs, transform_frame as source_transform_frame

    expansion_config = load_expansion_config(repo_root / "configs/data-expansion.json")
    data_source_audit = acquisition_audit(expansion_config)
    _source_features = source_registry_frame("expanded_v2").copy()
    source_transform_registry = source_transform_frame("all")
    _source_features["cache_status"] = np.where(
        _source_features["name"].isin(set(feature_registry["name"])),
        "present in current cache",
        "implemented; pending expanded-data rerun",
    )
    feature_implementation_status = (
        _source_features.groupby(["family", "cache_status"], as_index=False)
        .size()
        .rename(columns={"size": "features"})
    )
    transform_profile_status = pd.DataFrame([
        {
            "profile": _profile,
            "conditions": len(get_transform_specs(_profile)),
            "default": _profile == "core",
        }
        for _profile in TRANSFORM_PROFILES
    ])
    _cached_profile = run_metadata.get("feature_schema", {}).get("profile", "legacy cache before schema versioning")
    _blocked_sources = data_source_audit[
        data_source_audit["selected"] & ~data_source_audit["acquisition_allowed"]
    ]["source_id"].tolist()
    mo.vstack([
        mo.md("## Six-step implementation status"),
        mo.hstack([
            mo.stat(value=str(len(feature_registry)), label="Cached feature schema", caption=_cached_profile),
            mo.stat(value=str(len(_source_features)), label="Implemented expanded schema", caption="expanded_v2"),
            mo.stat(value="20", label="Default transform conditions", caption="core remains bounded"),
        ], widths="equal"),
        mo.callout(
            "External acquisition is blocked pending complete source revisions, file lists, hashes, and license allowlisting: "
            + (", ".join(_blocked_sources) if _blocked_sources else "no selected sources are blocked"),
            kind="warn" if _blocked_sources else "success",
        ),
        mo.hstack([
            mo.vstack([mo.md("### Feature implementation"), mo.ui.table(feature_implementation_status, selection=None, pagination=True, page_size=12)]),
            mo.vstack([mo.md("### Opt-in transform profiles"), mo.ui.table(transform_profile_status, selection=None, pagination=False)]),
            mo.vstack([mo.md("### Data/license gate"), mo.ui.table(data_source_audit, selection=None, pagination=True, page_size=8)]),
        ], widths="equal", align="start"),
        mo.callout(
            "Frozen-v1 contains exactly the original 53 features. Expanded-v2 adds 29 bit-plane, patch, multi-scale residual, residual-co-occurrence, camera-proxy, spectral, codec/resampling, and chroma measurements. Generic DINOv2/CLIP and reconstruction controls remain explicit licensed dependency decisions.",
            kind="info",
        ),
    ])
    return (source_transform_registry,)


@app.cell
def registries(feature_registry, transform_registry):
    _families = feature_registry.groupby(["family", "role"], as_index=False).size()
    mo.vstack([
        mo.md("## Preregistered hypotheses and transformations"),
        mo.md("Metadata is a negative-control family. High performance there is confounding evidence, not a detector contribution."),
        mo.hstack([
            mo.ui.table(_families, selection=None, pagination=False),
            mo.ui.table(transform_registry, selection=None, pagination=True, page_size=10),
        ], widths=[1, 3], align="start"),
        mo.accordion({"Full feature-hypothesis registry": mo.ui.table(feature_registry, selection=None, pagination=True, page_size=12)}),
    ])
    return


@app.cell
def coverage_audit(coverage, run_metadata):
    _totals = coverage.groupby(["dataset", "phase", "binary_label"], as_index=False)["images"].sum()
    _chart = alt.Chart(_totals).mark_bar().encode(
        x=alt.X("images:Q", title="Parents"),
        y=alt.Y("dataset:N", title=None),
        color=alt.Color("binary_label:N"),
        column=alt.Column("phase:N", title=None),
        tooltip=["dataset", "phase", "binary_label", "images"],
    ).properties(width=250, height=125)
    mo.vstack([
        mo.md("## Coverage and power audit"),
        _chart,
        mo.callout(
            f"{int(coverage['low_power'].sum())} strata have fewer than 20 images. WildFake has only 4–12 generated confirmation examples per displayed model and 12 authentic examples total. These cells remain visible but cannot establish generator-specific conclusions.",
            kind="warn",
        ),
        mo.ui.table(coverage, selection=None, pagination=True, page_size=15),
        mo.md("Input-index SHA-256: " + run_metadata["input_index_sha256"]),
    ])
    return


@app.cell
def image_controls(binary_index, transform_registry):
    feature_map_names = [
        "Luminance",
        "Gaussian low-pass",
        "Gaussian residual",
        "Laplacian magnitude",
        "Sobel magnitude",
        "Local standard deviation",
        "FFT log magnitude",
        "FFT phase",
        "Haar high-frequency energy",
        "Chroma magnitude",
    ]
    _options = {}
    for _row in binary_index.to_dict("records"):
        _caption = f"{_row['dataset']} · {_row['binary_label']} · {_row['generation_model']} · {Path(_row['local_path']).name}"
        _options[_caption] = _row["parent_id"]
    image_parent_control = mo.ui.dropdown(_options, value=next(iter(_options)), label="Parent", searchable=True)
    image_transform_control = mo.ui.dropdown(transform_registry["name"].tolist(), value="jpeg_q50", label="Transform")
    feature_map_control = mo.ui.dropdown(
        feature_map_names,
        value="FFT log magnitude",
        label="Representation",
    )
    mo.vstack([
        mo.md("## Original, transformed, and feature-map inspection"),
        mo.hstack([image_parent_control, image_transform_control, feature_map_control], widths=[3, 1, 1], align="end"),
    ])
    return (
        feature_map_control,
        feature_map_names,
        image_parent_control,
        image_transform_control,
    )


@app.cell
def map_helpers():
    def feature_map_array(image, representation):
        _rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        _luma = _rgb @ np.array([0.2126, 0.7152, 0.0722])
        if representation == "Luminance":
            return _luma
        if representation == "Gaussian low-pass":
            return ndimage.gaussian_filter(_luma, sigma=1, mode="reflect")
        if representation == "Gaussian residual":
            return np.abs(_luma - ndimage.gaussian_filter(_luma, sigma=1, mode="reflect"))
        if representation == "Laplacian magnitude":
            return np.abs(ndimage.laplace(_luma, mode="reflect"))
        if representation == "Sobel magnitude":
            return np.hypot(
                ndimage.sobel(_luma, axis=1, mode="reflect"),
                ndimage.sobel(_luma, axis=0, mode="reflect"),
            )
        if representation == "Local standard deviation":
            _mean = ndimage.gaussian_filter(_luma, sigma=2, mode="reflect")
            _mean_sq = ndimage.gaussian_filter(_luma**2, sigma=2, mode="reflect")
            return np.sqrt(np.maximum(_mean_sq - _mean**2, 0))
        if representation == "FFT log magnitude":
            return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(_luma - _luma.mean()))))
        if representation == "FFT phase":
            return (np.angle(np.fft.fftshift(np.fft.fft2(_luma - _luma.mean()))) + np.pi) / (2 * np.pi)
        if representation == "Chroma magnitude":
            _rg = _rgb[..., 0] - _rgb[..., 1]
            _bg = _rgb[..., 2] - _rgb[..., 1]
            return np.hypot(_rg, _bg)
        _h, _w = (_luma.shape[0] // 2) * 2, (_luma.shape[1] // 2) * 2
        _a, _b = _luma[:_h:2, :_w:2], _luma[:_h:2, 1:_w:2]
        _c, _d = _luma[1:_h:2, :_w:2], _luma[1:_h:2, 1:_w:2]
        return np.sqrt(
            ((_a - _b + _c - _d) / 2) ** 2
            + ((_a + _b - _c - _d) / 2) ** 2
            + ((_a - _b - _c + _d) / 2) ** 2
        )


    def normalized_map_bytes(mapped, low=None, high=None):
        _finite = np.asarray(mapped)[np.isfinite(mapped)]
        if low is None or high is None:
            low, high = np.quantile(_finite, [0.01, 0.99]) if _finite.size else (0.0, 1.0)
        _normalized = np.clip((mapped - low) / max(high - low, 1e-12), 0, 1)
        _buffer = BytesIO()
        Image.fromarray(np.uint8(_normalized * 255), mode="L").save(_buffer, format="PNG")
        return _buffer.getvalue()


    def map_image_bytes(image, representation):
        return normalized_map_bytes(feature_map_array(image, representation))


    def paired_map_image_bytes(real_image, fake_image, representation):
        _real_map = feature_map_array(real_image, representation)
        _fake_map = feature_map_array(fake_image, representation)
        _finite = np.concatenate([
            _real_map[np.isfinite(_real_map)].ravel(),
            _fake_map[np.isfinite(_fake_map)].ravel(),
        ])
        _low, _high = np.quantile(_finite, [0.01, 0.99]) if _finite.size else (0.0, 1.0)
        return (
            normalized_map_bytes(_real_map, _low, _high),
            normalized_map_bytes(_fake_map, _low, _high),
            (_low, _high),
        )

    return map_image_bytes, paired_map_image_bytes


@app.cell
def image_lab(
    binary_index,
    feature_map_control,
    image_parent_control,
    image_transform_control,
    map_image_bytes,
    repo_root,
):
    _row = binary_index[binary_index["parent_id"] == image_parent_control.value].iloc[0]
    with Image.open(repo_root / _row["local_path"]) as _opened:
        _original = _opened.convert("RGB")
    _transformed = apply_transform(_original, image_transform_control.value, parent_id=_row["parent_id"])
    _prepared = analysis_view(_transformed, "canonical_128")
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("### Published pixels"), mo.image(_original, width=300)]),
            mo.vstack([mo.md("### Transformed pixels"), mo.image(_transformed, width=300)]),
            mo.vstack([mo.md("### " + feature_map_control.value), mo.image(map_image_bytes(_prepared, feature_map_control.value), width=300)]),
        ], widths="equal"),
        mo.md(f"**{_row['dataset']} · {_row['binary_label']}** — {_row['generation_model']} — {_row['width']}×{_row['height']} {_row['format']} — {image_transform_control.value}"),
        mo.callout("Feature maps are normalized per image only for display; their brightness is not the evaluated scalar.", kind="info"),
    ])
    return


@app.cell
def pair_controls(binary_index, feature_map_names, transform_registry):
    _real_options = {}
    _fake_options = {}
    for _pair_row in binary_index.to_dict("records"):
        _pair_caption = f"{_pair_row['dataset']} · {_pair_row['generation_model']} · {Path(_pair_row['local_path']).name}"
        if _pair_row["binary_label"] == "authentic":
            _real_options[_pair_caption] = _pair_row["parent_id"]
        elif _pair_row["binary_label"] == "AIGC":
            _fake_options[_pair_caption] = _pair_row["parent_id"]
    comparison_real_control = mo.ui.dropdown(
        _real_options,
        value=next(iter(_real_options)),
        label="Authentic image",
        searchable=True,
    )
    comparison_fake_control = mo.ui.dropdown(
        _fake_options,
        value=next(iter(_fake_options)),
        label="AIGC image",
        searchable=True,
    )
    comparison_transform_control = mo.ui.dropdown(
        transform_registry["name"].tolist(),
        value="clean",
        label="Robustness transform",
    )
    comparison_filter_control = mo.ui.dropdown(
        feature_map_names,
        value="Gaussian residual",
        label="Filter / representation",
    )
    mo.vstack([
        mo.md("## Real versus AIGC: transform and filter microscope"),
        mo.md(
            "Apply exactly the same robustness transform and visualization to one authentic and one generated image. "
            "Change either sample to check whether an apparent difference persists beyond a convenient example."
        ),
        mo.hstack(
            [comparison_real_control, comparison_fake_control],
            widths="equal",
            align="end",
        ),
        mo.hstack(
            [comparison_transform_control, comparison_filter_control],
            justify="start",
            align="end",
        ),
    ])
    return (
        comparison_fake_control,
        comparison_filter_control,
        comparison_real_control,
        comparison_transform_control,
    )


@app.cell
def pair_view(
    binary_index,
    comparison_fake_control,
    comparison_filter_control,
    comparison_real_control,
    comparison_transform_control,
    paired_map_image_bytes,
    repo_root,
):
    _real_row = binary_index[binary_index["parent_id"] == comparison_real_control.value].iloc[0]
    _fake_row = binary_index[binary_index["parent_id"] == comparison_fake_control.value].iloc[0]
    with Image.open(repo_root / _real_row["local_path"]) as _real_opened:
        _real_original = _real_opened.convert("RGB")
    with Image.open(repo_root / _fake_row["local_path"]) as _fake_opened:
        _fake_original = _fake_opened.convert("RGB")
    _real_transformed = apply_transform(
        _real_original,
        comparison_transform_control.value,
        parent_id=_real_row["parent_id"],
    )
    _fake_transformed = apply_transform(
        _fake_original,
        comparison_transform_control.value,
        parent_id=_fake_row["parent_id"],
    )
    _real_prepared = analysis_view(_real_transformed, "canonical_128")
    _fake_prepared = analysis_view(_fake_transformed, "canonical_128")
    _real_map, _fake_map, _pair_scale = paired_map_image_bytes(
        _real_prepared,
        _fake_prepared,
        comparison_filter_control.value,
    )
    _pair_widths = [0.72, 1, 1, 1]
    mo.vstack([
        mo.hstack(
            [
                mo.md("**Class**"),
                mo.md("**Published pixels**"),
                mo.md(f"**After `{comparison_transform_control.value}`**"),
                mo.md(f"**{comparison_filter_control.value}**"),
            ],
            widths=_pair_widths,
            align="center",
        ),
        mo.hstack(
            [
                mo.md(
                    f"### Authentic  \n{_real_row['dataset']}  \n{_real_row['width']}×{_real_row['height']} {_real_row['format']}"
                ),
                mo.image(_real_original, width=230),
                mo.image(_real_transformed, width=230),
                mo.image(_real_map, width=230),
            ],
            widths=_pair_widths,
            align="center",
        ),
        mo.hstack(
            [
                mo.md(
                    f"### AIGC  \n{_fake_row['dataset']}  \n{_fake_row['generation_model']}  \n{_fake_row['width']}×{_fake_row['height']} {_fake_row['format']}"
                ),
                mo.image(_fake_original, width=230),
                mo.image(_fake_transformed, width=230),
                mo.image(_fake_map, width=230),
            ],
            widths=_pair_widths,
            align="center",
        ),
        mo.callout(
            f"Both feature maps use one shared 1st–99th percentile display scale "
            f"({_pair_scale[0]:.4g} to {_pair_scale[1]:.4g}). Display brightness is still a visualization, not the evaluated scalar feature.",
            kind="info",
        ),
    ])
    return


@app.cell
def analysis_controls(feature_registry):
    _features = feature_registry[feature_registry["role"] == "candidate"]["name"].tolist()
    selected_feature_control = mo.ui.dropdown(_features, value="residual_kurtosis", label="Feature", searchable=True)
    selected_view_control = mo.ui.dropdown(["canonical_128", "native_capped"], value="canonical_128", label="View")
    mo.hstack([selected_feature_control, selected_view_control], justify="start")
    return selected_feature_control, selected_view_control


@app.cell
def severity_control(source_transform_registry):
    _severity_families = source_transform_registry[
        source_transform_registry["official"] & source_transform_registry["family"].ne("clean")
    ]["family"].drop_duplicates().tolist()
    severity_family_control = mo.ui.dropdown(
        _severity_families,
        value="gaussian_blur",
        label="Severity family",
    )
    mo.hstack([severity_family_control], justify="start")
    return (severity_family_control,)


@app.cell
def feature_profile(
    dataset_metrics,
    directions,
    feature_registry,
    feature_table,
    generator_metrics,
    selected_feature_control,
    selected_view_control,
    transform_metrics,
):
    _feature, _view = selected_feature_control.value, selected_view_control.value
    _spec = feature_registry[feature_registry["name"] == _feature].iloc[0]
    _distribution = feature_table[
        (feature_table["phase"] == "confirmation") & (feature_table["condition"] == "clean") & (feature_table["view"] == _view)
    ][["dataset", "binary_label", _feature]].rename(columns={_feature: "value"})
    _box = alt.Chart(_distribution).mark_boxplot(extent="min-max").encode(
        x=alt.X("binary_label:N", title=None), y=alt.Y("value:Q", title=_feature),
        color=alt.Color("binary_label:N", legend=None), column=alt.Column("dataset:N", title=None),
        tooltip=["dataset", "binary_label", "value"],
    ).properties(width=190, height=220)
    _tm = transform_metrics[
        (transform_metrics["feature"] == _feature) & (transform_metrics["view"] == _view) & (transform_metrics["official_transform"])
    ]
    _interval = alt.Chart(_tm).mark_rule().encode(
        x=alt.X("ci_low:Q", scale=alt.Scale(domain=[0, 1]), title="Oriented AUPRC and 95% bootstrap interval"),
        x2="ci_high:Q", y=alt.Y("condition:N", sort="-x", title=None),
        color=alt.Color("direction_reversal:N"),
        tooltip=["condition", "oriented_auprc", "positive_prevalence", "normalized_auprc", "ci_low", "ci_high", "n_real", "n_aigc"],
    ).properties(width=520, height=350)
    _points = alt.Chart(_tm).mark_point(filled=True, size=70).encode(
        x="oriented_auprc:Q", y=alt.Y("condition:N", sort="-x", title=None), color="direction_reversal:N"
    )
    _baseline = alt.Chart(_tm).mark_rule(strokeDash=[5, 4], color="#555").encode(x="positive_prevalence:Q")
    _direction = directions[(directions["feature"] == _feature) & (directions["view"] == _view)].iloc[0]
    _dm = dataset_metrics[(dataset_metrics["feature"] == _feature) & (dataset_metrics["view"] == _view)]
    _gm = generator_metrics[(generator_metrics["feature"] == _feature) & (generator_metrics["view"] == _view)]
    mo.vstack([
        mo.md("## Feature profile: " + _feature),
        mo.md(f"**Measurement:** {_spec['measurement']}  \n**Hypothesis:** {_spec['hypothesis']}  \n**Expected failure:** {_spec['expected_failure']}"),
        mo.callout(f"Discovery orientation is frozen: selected AUPRC {_direction['selected_discovery_auprc']:.3f} against prevalence {_direction['positive_prevalence']:.3f}, direction {_direction['direction']:+.0f}. Confirmation below its own prevalence is a reversal.", kind="info"),
        _box, _interval + _points + _baseline,
        mo.hstack([
            mo.vstack([mo.md("### By dataset"), mo.ui.table(_dm, selection=None, pagination=False)]),
            mo.vstack([mo.md("### By generator"), mo.ui.table(_gm, selection=None, pagination=True, page_size=10)]),
        ], widths="equal"),
    ])
    return


@app.cell
def severity_trajectories(
    comparison_fake_control,
    comparison_real_control,
    feature_table,
    parent_paired_feature_drift,
    selected_feature_control,
    selected_view_control,
    severity_family_control,
    source_transform_registry,
):
    _trajectory_feature = selected_feature_control.value
    _trajectory_view = selected_view_control.value
    _trajectory_parents = {
        comparison_real_control.value: "Authentic",
        comparison_fake_control.value: "AIGC",
    }
    _trajectory = feature_table[
        feature_table["parent_id"].isin(_trajectory_parents)
        & feature_table["view"].eq(_trajectory_view)
        & (
            feature_table["transform_family"].eq(severity_family_control.value)
            | feature_table["condition"].eq("clean")
        )
    ][["parent_id", "condition", _trajectory_feature]].rename(
        columns={_trajectory_feature: "feature_value"}
    )
    _trajectory = _trajectory.merge(
        source_transform_registry[["name", "severity"]],
        left_on="condition",
        right_on="name",
        how="left",
    )
    _trajectory["severity"] = _trajectory["severity"].fillna(0.0)
    _trajectory["class"] = _trajectory["parent_id"].map(_trajectory_parents)
    _trajectory_chart = alt.Chart(_trajectory).mark_line(point=True).encode(
        x=alt.X("severity:Q", title="Registered severity (clean = 0)"),
        y=alt.Y("feature_value:Q", title=_trajectory_feature),
        color=alt.Color("class:N"),
        detail="parent_id:N",
        tooltip=["class", "parent_id", "condition", "severity", "feature_value"],
    ).properties(width=720, height=300)
    _drift_selected = parent_paired_feature_drift[
        parent_paired_feature_drift["feature"].eq(_trajectory_feature)
        & parent_paired_feature_drift["view"].eq(_trajectory_view)
        & parent_paired_feature_drift["transform_family"].eq(severity_family_control.value)
    ] if not parent_paired_feature_drift.empty else parent_paired_feature_drift
    mo.vstack([
        mo.md("## Parent-wise severity trajectories"),
        mo.md("The selected authentic and AIGC parents use the same registered severities. The table summarizes paired condition-minus-clean drift across all confirmation parents; the two lines are examples, not population evidence."),
        _trajectory_chart,
        mo.ui.table(_drift_selected, selection=None, pagination=True, page_size=10),
    ])
    return


@app.cell
def generator_map(feature_registry, generator_metrics, selected_view_control):
    _candidates = set(feature_registry.query("role == 'candidate'")["name"])
    _data = generator_metrics[(generator_metrics["view"] == selected_view_control.value) & (generator_metrics["feature"].isin(_candidates))].copy()
    _data["generator_axis"] = _data["dataset"] + " · " + _data["comparison_generator"]
    _heatmap = alt.Chart(_data).mark_rect().encode(
        x=alt.X("generator_axis:N", title="Generator comparison"), y=alt.Y("feature:N", title=None),
        color=alt.Color("normalized_auprc:Q", title="Normalized AUPRC", scale=alt.Scale(domain=[-0.3, 0, 0.4, 1], range=["#9b3a46", "#f3eee2", "#72a88d", "#174b58"])),
        opacity=alt.Opacity("low_power:N", scale=alt.Scale(domain=[False, True], range=[1, 0.35])),
        tooltip=["dataset", "comparison_generator", "feature", "oriented_auprc", "positive_prevalence", "normalized_auprc", "ci_low", "ci_high", "n_real", "n_aigc", "low_power", "direction_reversal"],
    ).properties(width=820, height=900)
    _reversals = alt.Chart(_data[_data["direction_reversal"]]).mark_text(text="↺", color="#7c1f2a").encode(x="generator_axis:N", y="feature:N")
    mo.vstack([
        mo.md("## Feature × generator failure map"),
        mo.callout("Color is prevalence-normalized AUPRC: 0 is the group-specific prevalence baseline. Faded cells are underpowered; ↺ means the frozen discovery direction fell below that baseline.", kind="warn"),
        _heatmap + _reversals,
    ])
    return


@app.cell
def transform_map(feature_registry, selected_view_control, transform_metrics):
    _candidates = set(feature_registry.query("role == 'candidate'")["name"])
    _data = transform_metrics[
        (transform_metrics["view"] == selected_view_control.value) & (transform_metrics["feature"].isin(_candidates)) & (transform_metrics["official_transform"])
    ]
    _heatmap = alt.Chart(_data).mark_rect().encode(
        x=alt.X("condition:N", title="Official condition"), y=alt.Y("feature:N", title=None),
        color=alt.Color("normalized_auprc:Q", title="Normalized AUPRC", scale=alt.Scale(domain=[-0.3, 0, 0.4, 1], range=["#9b3a46", "#f3eee2", "#72a88d", "#174b58"])),
        tooltip=["feature", "condition", "oriented_auprc", "positive_prevalence", "normalized_auprc", "ci_low", "ci_high", "direction_reversal"],
    ).properties(width=900, height=900)
    mo.vstack([mo.md("## Feature × transformation robustness map"), mo.md("Color uses prevalence-normalized AUPRC; zero is the AIGC prevalence baseline."), _heatmap])
    return


@app.cell
def composition_analysis(
    directed_pair_interactions,
    directed_pair_order_sensitivity,
    selected_feature_control,
    selected_view_control,
    severity_area,
    source_transform_registry,
    transform_metrics,
):
    _composition_ready = not directed_pair_order_sensitivity.empty
    if _composition_ready:
        _order_top = directed_pair_order_sensitivity.sort_values(
            "absolute_order_sensitivity", ascending=False
        ).head(20)
        _interaction_top = directed_pair_interactions.reindex(
            directed_pair_interactions["interaction_excess_auprc_drop"].abs().sort_values(ascending=False).index
        ).head(20)
        _severity_top = severity_area.sort_values("worst_auprc_drop", ascending=False).head(20)
        _order_selected = directed_pair_order_sensitivity[
            directed_pair_order_sensitivity["feature"].eq(selected_feature_control.value)
            & directed_pair_order_sensitivity["view"].eq(selected_view_control.value)
        ]
        _order_heatmap = alt.Chart(_order_selected).mark_rect().encode(
            x=alt.X("operation_a:N", title="First operation"),
            y=alt.Y("operation_b:N", title="Second operation"),
            color=alt.Color("absolute_order_sensitivity:Q", title="|AUPRC order gap|"),
            tooltip=["feature", "operation_a", "operation_b", "auprc_a_then_b", "auprc_b_then_a", "absolute_order_sensitivity"],
        ).properties(width=520, height=360)
        _composition_view = mo.vstack([
            mo.callout(
                "Pair effects use discovery-frozen feature directions and operation-level shared noise draws. Interaction excess compares the composed AUPRC drop with the two constituent single-transform drops; order sensitivity compares exact reversed recipes.",
                kind="info",
            ),
            mo.md(f"### Order heatmap: `{selected_feature_control.value}` / `{selected_view_control.value}`"),
            _order_heatmap,
            mo.hstack([
                mo.vstack([mo.md("### Largest order effects"), mo.ui.table(_order_top, selection=None, pagination=True, page_size=10)]),
                mo.vstack([mo.md("### Largest interaction excess"), mo.ui.table(_interaction_top, selection=None, pagination=True, page_size=10)]),
            ], widths="equal", align="start"),
            mo.vstack([mo.md("### Severity-curve summaries"), mo.ui.table(_severity_top, selection=None, pagination=True, page_size=10)]),
        ])
    else:
        _composition_view = mo.callout(
            "The implementation is ready, but the current cache predates the directed-pair run. Generate it with --feature-profile expanded_v2 --transform-profile directed_pairs. Core runs intentionally return schema-stable empty interaction tables.",
            kind="warn",
        )
    _profile_metrics = transform_metrics.merge(
        source_transform_registry[["name", "design"]],
        left_on="condition",
        right_on="name",
        how="left",
    )
    _profile_summary = (
        _profile_metrics[
            _profile_metrics["design"].isin(["preregistered_realistic_chain", "halton_covering_bank"])
        ]
        .groupby(["design", "view", "feature"], as_index=False)
        .agg(worst_normalized_auprc=("normalized_auprc", "min"), conditions=("condition", "nunique"))
        .sort_values("worst_normalized_auprc")
        .head(30)
    )
    _profile_view = (
        mo.ui.table(_profile_summary, selection=None, pagination=True, page_size=12)
        if not _profile_summary.empty
        else mo.callout(
            "No realistic-chain or covering-bank results are in this cache. Those profiles are opt-in so the bounded core cache cannot be mistaken for composition evidence.",
            kind="warn",
        )
    )
    mo.vstack([
        mo.md("## Sequential-transform interactions and order sensitivity"),
        mo.md("The full 12,959-condition Cartesian grid is deliberately not evaluated. The staged design uses official singles, 30 directed medium pairs, 12 realistic chains, and a 32-recipe covering bank."),
        _composition_view,
        mo.md("### Realistic and covering-bank summary"),
        _profile_view,
    ])
    return


@app.cell
def shortcut_audit(ablations, leave_one_dataset_out, shortcut_correlations):
    _clean = ablations[(ablations["view"] == "canonical_128") & (ablations["condition"] == "clean")].sort_values("normalized_auprc")
    _bars = alt.Chart(_clean).mark_bar().encode(
        x=alt.X("normalized_auprc:Q", scale=alt.Scale(domain=[-0.1, 0.75]), title="Clean prevalence-normalized AUPRC"),
        y=alt.Y("baseline:N", sort="-x", title=None),
        color=alt.condition(alt.datum.baseline == "nuisance_only", alt.value("#9b3a46"), alt.value("#2f6f75")),
        tooltip=["baseline", "auprc", "positive_prevalence", "normalized_auprc", "balanced_accuracy"],
    ).properties(width=520, height=330)
    _corr = shortcut_correlations.sort_values("max_abs_nuisance_spearman", ascending=False).head(15)
    mo.vstack([
        mo.md("## Shortcut audit and family ablations"),
        mo.callout("Nuisance-only performance is a warning. Leave-one-dataset-out performance tests whether pooled separation survives removal of dataset identity.", kind="warn"),
        _bars,
        mo.hstack([
            mo.vstack([mo.md("### Leave one dataset out"), mo.ui.table(leave_one_dataset_out, selection=None, pagination=False)]),
            mo.vstack([mo.md("### Strongest nuisance correlations"), mo.ui.table(_corr, selection=None, pagination=False)]),
        ], widths="equal"),
    ])
    return


@app.cell
def model_robustness(ablations):
    _data = ablations[
        (ablations["view"] == "canonical_128")
        & (ablations["baseline"].isin(["engineered_all", "nuisance_only", "residual", "wavelet", "fft_phase", "fft_magnitude"]))
    ]
    _chart = alt.Chart(_data).mark_line(point=True).encode(
        x=alt.X("condition:N", title=None, sort=None),
        y=alt.Y("normalized_auprc:Q", scale=alt.Scale(domain=[-0.3, 0.9]), title="Prevalence-normalized AUPRC"),
        color=alt.Color("baseline:N"), tooltip=["baseline", "condition", "auprc", "positive_prevalence", "normalized_auprc", "balanced_accuracy"],
    ).properties(width=1000, height=340)
    mo.vstack([
        mo.md("## Multivariate family robustness"),
        mo.md("Each model is fit once on clean discovery parents and evaluated without refitting on every confirmation condition. Zero is the condition-specific AIGC prevalence baseline."),
        _chart,
    ])
    return


@app.cell
def robustness_frontier_panel(ablations, selected_view_control):
    _frontier_source = ablations[ablations["view"].eq(selected_view_control.value)]
    _frontier_clean = _frontier_source[_frontier_source["condition"].eq("clean")][
        ["baseline", "normalized_auprc"]
    ].rename(columns={"normalized_auprc": "clean_normalized_auprc"})
    _frontier_worst = _frontier_source.groupby("baseline", as_index=False).agg(
        worst_condition_normalized_auprc=("normalized_auprc", "min")
    )
    robustness_frontier = _frontier_clean.merge(_frontier_worst, on="baseline", how="inner")
    robustness_frontier["pareto"] = [
        not (
            (robustness_frontier["clean_normalized_auprc"].ge(_row.clean_normalized_auprc))
            & (robustness_frontier["worst_condition_normalized_auprc"].ge(_row.worst_condition_normalized_auprc))
            & (
                robustness_frontier["clean_normalized_auprc"].gt(_row.clean_normalized_auprc)
                | robustness_frontier["worst_condition_normalized_auprc"].gt(_row.worst_condition_normalized_auprc)
            )
        ).any()
        for _row in robustness_frontier.itertuples()
    ]
    _frontier_chart = alt.Chart(robustness_frontier).mark_point(filled=True, size=110).encode(
        x=alt.X("clean_normalized_auprc:Q", scale=alt.Scale(domain=[0, 1]), title="Clean normalized AUPRC"),
        y=alt.Y("worst_condition_normalized_auprc:Q", scale=alt.Scale(domain=[0, 1]), title="Worst-condition normalized AUPRC"),
        color=alt.Color("pareto:N"),
        tooltip=["baseline", "clean_normalized_auprc", "worst_condition_normalized_auprc", "pareto"],
    ).properties(width=620, height=360)
    mo.vstack([
        mo.md("## Clean versus worst-case Pareto frontier"),
        mo.md("A point is marked Pareto-efficient only when no evaluated branch is at least as good on both clean and worst-condition prevalence-normalized AUPRC and strictly better on one."),
        _frontier_chart,
        mo.ui.table(robustness_frontier.sort_values(["pareto", "worst_condition_normalized_auprc"], ascending=False), selection=None, pagination=False),
    ])
    return


@app.cell
def ledger_view(decision_ledger):
    _scatter = alt.Chart(decision_ledger).mark_circle(size=100).encode(
        x=alt.X("clean_confirmation_normalized_auprc:Q", scale=alt.Scale(domain=[0.3, 0.75])),
        y=alt.Y("worst_official_transform_normalized_auprc:Q", scale=alt.Scale(domain=[0.3, 0.7])),
        color=alt.Color("decision:N"), shape=alt.Shape("decision_confidence:N"),
        tooltip=["feature", "decision", "decision_confidence", "decision_basis", "clean_confirmation_auprc", "clean_confirmation_normalized_auprc", "worst_official_transform_auprc", "worst_official_transform_normalized_auprc", "worst_powered_dataset_normalized_auprc", "worst_powered_generator_normalized_auprc", "max_abs_nuisance_spearman"],
    ).properties(width=650, height=420)
    _counts = decision_ledger["decision"].value_counts().rename_axis("decision").reset_index(name="features")
    mo.vstack([
        mo.md("## Final decision ledger"),
        mo.callout("Categories use fixed exploratory normalized-AUPRC rules, not competition thresholds. Decisions are provisional wherever modern generator strata remain underpowered.", kind="warn"),
        mo.hstack([_scatter, mo.ui.table(_counts, selection=None, pagination=False)], widths=[3, 1]),
        mo.ui.table(decision_ledger, selection=None, pagination=True, page_size=12),
    ])
    return


@app.function
def failure_card(row, root):
    with Image.open(root / row["local_path"]) as _opened:
        _image = apply_transform(_opened.convert("RGB"), row["condition"], parent_id=row["parent_id"])
    return mo.vstack([
        mo.image(_image, width=210),
        mo.md(f"**{row['error_type'].replace('_', ' ').title()} · score {row['pred']:.3f}**  \n{row['dataset']} · {row['generation_model']}  \n{row['condition']}"),
    ])


@app.cell
def failure_control(representative_failures):
    _options = representative_failures["condition"].drop_duplicates().tolist()
    failure_condition_control = mo.ui.dropdown(_options, value=_options[0], label="Failure condition")
    failure_condition_control
    return (failure_condition_control,)


@app.cell
def failure_gallery(
    failure_condition_control,
    repo_root,
    representative_failures,
):
    _rows = representative_failures[representative_failures["condition"] == failure_condition_control.value].sort_values(["error_type", "rank"])
    mo.vstack([
        mo.md("## Representative high-confidence failures"),
        mo.md("False positives are authentic confirmation images with the highest AIGC scores; false negatives are AIGC images with the lowest scores."),
        mo.hstack([failure_card(_row, repo_root) for _row in _rows.to_dict("records")], wrap=True, justify="start"),
    ])
    return


@app.cell
def diversity_and_deferrals(repo_root):
    _diversity_path = repo_root / "data/derived/data_expansion/diversity_cohorts.csv"
    if _diversity_path.exists():
        _diversity_cohorts = pd.read_csv(_diversity_path)
        _diversity_summary = _diversity_cohorts.groupby("diversity_cohort", as_index=False).agg(
            images=("parent_id", "nunique"),
            generators=("generator_id", "nunique"),
        )
        _diversity_view = mo.ui.table(_diversity_summary, selection=None, pagination=False)
    else:
        _diversity_view = mo.callout(
            "No diversity cohort is displayed. Community Forensics remains unselected and must not be downloaded wholesale; this panel activates only after an allowlisted indexed sample creates equal-image-count few-generator and many-generator cohorts.",
            kind="warn",
        )
    mo.vstack([
        mo.md("## Generator diversity versus image count"),
        _diversity_view,
        mo.callout(
            "Still explicitly deferred pending licensed data or dependencies: continuous interior color training samples, adaptive discovery-only worst-case search, prompt-matched/real-real/fake-fake and matched-reencoding controls, leave-one-generator model evaluation, sparse/out-of-fold branch fusion, a licensed DINOv2/CLIP control, and reconstruction evidence.",
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def conclusions(ablations, decision_ledger, leave_one_dataset_out):
    _clean = ablations[(ablations["view"] == "canonical_128") & (ablations["condition"] == "clean")].set_index("baseline")
    _kept = decision_ledger[decision_ledger["decision"] == "keep"]["feature"].tolist()
    _worst = leave_one_dataset_out[leave_one_dataset_out["view"] == "canonical_128"].sort_values("normalized_auprc").iloc[0]
    mo.vstack([
        mo.md("## What this pilot supports"),
        mo.callout(
            f"Nuisance-only clean AUPRC is {_clean.loc['nuisance_only', 'auprc']:.3f} ({_clean.loc['nuisance_only', 'normalized_auprc']:.3f} normalized), versus {_clean.loc['engineered_all', 'auprc']:.3f} ({_clean.loc['engineered_all', 'normalized_auprc']:.3f}) for all candidate features. Aggregate AIGC prevalence is {_clean.loc['engineered_all', 'positive_prevalence']:.3f}. Holding out {_worst['heldout_dataset']} yields {_worst['auprc']:.3f} AUPRC ({_worst['normalized_auprc']:.3f} normalized). Pooled separation is therefore not trustworthy by itself.",
            kind="warn",
        ),
        mo.callout(
            "Powered clean/transform screens retain " + (", ".join(_kept) if _kept else "no features") + ". Low-confidence warnings remain where underpowered WildFake groups reverse.",
            kind="info",
        ),
        mo.md("""
        - This run validates the laboratory and produces falsifiable candidates; it does not establish modern-generator generalization.
        - CIFAKE is 32×32 and single-generator. SID hides row-level generator identity. The local WildFake slice is underpowered.
        - The next data action is a larger generator-balanced confirmation slice rerun against this frozen registry.
        - A generic pretrained semantic control was not downloaded implicitly; adding a licensed frozen DINOv2 or CLIP control is a separate dependency decision.
        """),
    ])
    return


@app.cell
def sealed_confirmation(
    chronological_confirmation_metrics,
    feature_table,
    run_metadata,
):
    _sealed_generators = sorted(
        feature_table[
            feature_table["phase"].eq("final_confirmation")
            & feature_table["target"].eq(1)
        ]["generation_model"].dropna().astype(str).unique().tolist()
    )
    _sealed_run_state = run_metadata.get("final_confirmation", {})
    _sealed_first_evaluated_at = _sealed_run_state.get("evaluated_at")
    _sealed_metrics = chronological_confirmation_metrics[
        chronological_confirmation_metrics.get("phase", pd.Series(dtype=str)).eq("final_confirmation")
    ]
    if _sealed_metrics.empty:
        _confirmation_result = mo.callout(
            "No sealed final-confirmation result is displayed. This is the correct state until licensed generator IDs are assigned and sealed before selection; ordinary chronological confirmation is kept in the exploratory tables and an empty table cannot masquerade as final evidence.",
            kind="warn",
        )
    else:
        _confirmation_result = mo.ui.table(
            _sealed_metrics,
            selection=None,
            pagination=True,
            page_size=12,
        )
    mo.vstack([
        mo.md("## Sealed chronological confirmation"),
        mo.hstack([
            mo.stat(value="sealed", label="Policy", caption="feature/transform selection forbidden"),
            mo.stat(value=str(len(_sealed_generators)), label="Assigned generator IDs", caption="must be set before acquisition"),
            mo.stat(value=str(_sealed_first_evaluated_at), label="First evaluated", caption="None means this cache never touched the sealed window"),
        ], widths="equal"),
        _confirmation_result,
    ])
    return


@app.cell(hide_code=True)
def reproduce(cache_dir):
    mo.vstack([
        mo.md("## Reproduce"),
        mo.md("""
            uv sync
            uv run pytest -q

        Frozen preregistered baseline on the current local index:

            uv run python scripts/run_feature_lab.py --feature-profile frozen_v1 --transform-profile core --workers 4 --bootstrap 200

        Expanded features and directed-pair stress profile:

            uv run python scripts/run_feature_lab.py --feature-profile expanded_v2 --transform-profile directed_pairs --output data/derived/feature_lab_pairs --workers 4 --bootstrap 200

        License-gated data planning (metadata only; never downloads):

            uv run python scripts/plan_data_expansion.py path/to/expansion-manifest.csv

        Run an audited `selection.csv` with --index only after both license layers pass. Final rows stay withheld unless the frozen one-time run explicitly adds --evaluate-final-confirmation; success writes a repeat-blocking receipt beside the index.

        Metrics-only refresh:

            uv run python scripts/run_feature_lab.py --reuse-features --bootstrap 200

        Open the notebook:

            uv run marimo edit notebooks/feature_robustness_lab.py --no-token
        """),
        mo.md("Detailed contract: docs/aigc-exploration-implementation-plan.md  \nLocal cache: " + str(cache_dir)),
    ])
    return


if __name__ == "__main__":
    app.run()
