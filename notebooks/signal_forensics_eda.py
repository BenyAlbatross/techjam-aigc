import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    from io import BytesIO
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageOps
    from scipy import ndimage
    from scipy.fft import dctn

    return BytesIO, Image, ImageOps, Path, alt, dctn, json, mo, ndimage, np, pd


@app.cell(hide_code=True)
def intro(mo):
    mo.md("""
    # Frequency and camera-signal forensics

    A separate follow-up to the visual dataset EDA. This notebook tests **766 images** to answer four questions about Fourier upsampling traces, PRNU/DSNU, FFT/DCT signatures, and transfer to DDA-COCO, EvalGEN, and Community Forensics Eval.

    ## Bottom line

    1. **Periodic Fourier traces exist in some sources, but are not universal diffusion evidence.** Community Forensics has a strong 1/8-cycle harmonic difference in this slice; tightly paired DDA-COCO largely removes it.
    2. **This corpus cannot directly measure PRNU or DSNU.** It supports honest single-image camera-pipeline proxies only.
    3. **FFT and DCT effects are dataset- and codec-dependent.** Strong SID Set or WildFake signals collapse on paired DDA-COCO.
    4. **New generators differ in residual statistics, but EvalGEN is AIGC-only.** Its authentic comparison is cross-source and descriptive.
    """)
    return


@app.cell(hide_code=True)
def load_results(Path, json, pd):
    repo_root = Path.cwd()
    result_dir = repo_root / "data/derived/signal_analysis"
    external_index_path = repo_root / "data/metadata/signal_analysis_index.csv"
    source_metadata_path = repo_root / "data/metadata/signal_analysis_sources.json"
    _required_paths = [
        result_dir / "image_features.csv.gz",
        result_dir / "radial_profiles.csv.gz",
        result_dir / "dct_profiles.csv.gz",
        result_dir / "feature_effects.csv",
        result_dir / "dda_paired_effects.csv",
        result_dir / "feature_registry.csv",
        result_dir / "run_metadata.json",
        external_index_path,
        source_metadata_path,
    ]
    _missing_paths = [_path for _path in _required_paths if not _path.exists()]
    if _missing_paths:
        raise FileNotFoundError(
            "Missing signal-analysis artifacts. Run scripts/prepare_signal_analysis_samples.py "
            "and scripts/run_signal_analysis.py. Missing: "
            + ", ".join(str(_path) for _path in _missing_paths)
        )
    features = pd.read_csv(result_dir / "image_features.csv.gz")
    radial_profiles = pd.read_csv(result_dir / "radial_profiles.csv.gz")
    dct_profiles = pd.read_csv(result_dir / "dct_profiles.csv.gz")
    effects = pd.read_csv(result_dir / "feature_effects.csv")
    dda_paired = pd.read_csv(result_dir / "dda_paired_effects.csv")
    feature_registry = pd.read_csv(result_dir / "feature_registry.csv")
    external_index = pd.read_csv(external_index_path)
    run_metadata = json.loads((result_dir / "run_metadata.json").read_text())
    sources_metadata = json.loads(source_metadata_path.read_text())
    return (
        dct_profiles,
        dda_paired,
        effects,
        external_index,
        feature_registry,
        features,
        radial_profiles,
        repo_root,
        run_metadata,
        sources_metadata,
    )


@app.cell(hide_code=True)
def inventory(external_index, features, mo):
    _inventory = (
        features.groupby(["dataset", "label", "generator_family", "generation_model"], as_index=False)
        .size().rename(columns={"size": "images"})
    )
    _external_inventory = (
        external_index.groupby(
            ["dataset", "label", "generator_family", "generation_model", "license_name"],
            as_index=False,
        ).size().rename(columns={"size": "images"})
    )
    mo.vstack([
        mo.md("## Evidence inventory"),
        mo.hstack([
            mo.stat(value=f"{len(features):,}", label="Analyzed images", caption="existing EDA + external"),
            mo.stat(value=f"{len(external_index):,}", label="Newly acquired", caption="ignored under data/"),
            mo.stat(value=str(features["generation_model"].nunique()), label="Model/source labels"),
            mo.stat(value="256×256", label="Common view", caption="center-fit, Lanczos"),
        ], widths="equal"),
        mo.callout(
            "DDA-COCO: 120 exact COCO/reconstruction pairs across six groups. "
            "EvalGEN: 20 each from Flux, GoT, Infinity, NOVA, and OmniGen. "
            "Community Forensics Eval: 25 authentic + 25 AIGC, initially capped at two images per model.",
            kind="info",
        ),
        mo.accordion({
            "Full inventory": mo.ui.table(_inventory, selection=None, pagination=True, page_size=14),
            "Pinned sources and licenses": mo.ui.table(_external_inventory, selection=None, pagination=True, page_size=14),
        }),
    ])
    return


@app.cell(hide_code=True)
def methods(mo):
    mo.vstack([
        mo.md("""
    ## Measurement contract

    All images are center-fitted to 256×256. A Hann window reduces FFT boundary leakage. Measurements include radial power, entropy, anisotropy, peak prominence, phase, 1/8- and 1/4-cycle harmonic excess, residual autocorrelation, 8×8 DCT bands, AC tails/near-zero mass, blockiness, and camera-pipeline proxies.

    Tables preserve **AUC(AIGC-high)**, signed median differences, and rank-biserial direction. Direction-free separation AUC is descriptive ranking only.

    **Validity hierarchy:** exact DDA pairs > within-dataset comparisons > pooled/cross-source summaries. Codec, resizing, content, and camera processing can imitate every signal here.
    """),
        mo.callout(
            "A high scalar AUC is a hypothesis lead, not a deployable detector. The smallest balanced cells are 25+25.",
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def controls(mo, radial_profiles):
    dataset_control = mo.ui.dropdown(
        sorted(radial_profiles["dataset"].unique().tolist()),
        value="DDA-COCO",
        label="Dataset",
    )
    family_control = mo.ui.dropdown(
        ["FFT magnitude", "FFT phase", "DCT / JPEG", "Noise / camera proxy", "Residual / periodicity"],
        value="FFT magnitude",
        label="Feature family",
    )
    mo.hstack([dataset_control, family_control], justify="start", gap=2)
    return dataset_control, family_control


@app.cell(hide_code=True)
def spectra(alt, dataset_control, mo, radial_profiles):
    _radial = (
        radial_profiles[radial_profiles["dataset"] == dataset_control.value]
        .groupby(["label", "cycles_per_pixel"], as_index=False)
        .agg(log_power=("log_power", "median"))
    )
    _radial_chart = (
        alt.Chart(_radial).mark_line()
        .encode(
            x=alt.X("cycles_per_pixel:Q", title="Radial frequency (cycles / pixel)"),
            y=alt.Y("log_power:Q", title="Median log power"),
            color=alt.Color("label:N"),
            tooltip=["label", alt.Tooltip("cycles_per_pixel:Q", format=".3f"), alt.Tooltip("log_power:Q", format=".3f")],
        ).properties(width=760, height=320)
    )
    mo.vstack([
        mo.md(f"## Fourier spectrum shape — {dataset_control.value}"),
        _radial_chart,
        mo.md("Parallel curves suggest scale/energy differences; localized bumps are more consistent with periodic structure. DDA's matched curves are intentionally close."),
    ])
    return


@app.cell(hide_code=True)
def upsampling_answer(effects, mo):
    periodic_names = [
        "fft_harmonic_excess_1_8", "fft_harmonic_excess_1_4",
        "fft_local_peak_prominence_q999", "fft_peak_ratio",
        "residual_periodicity_lag2", "residual_periodicity_lag4",
        "residual_periodicity_lag8", "resampling_second_difference_periodicity",
    ]
    _periodic = effects[
        effects["feature"].isin(periodic_names)
        & effects["scope"].str.startswith("within:")
    ].copy()
    _periodic["dataset"] = _periodic["scope"].str.replace("within:", "", regex=False)
    _periodic_display = _periodic[[
        "dataset", "feature", "n_authentic", "n_aigc",
        "median_delta_aigc_minus_authentic", "auc_aigc_high",
        "separation_auc", "rank_biserial",
    ]].sort_values(["dataset", "separation_auc"], ascending=[True, False])
    mo.vstack([
        mo.md("## 1 · Does diffusion upsampling leave Fourier patterns cameras do not?"),
        mo.callout(
            "Sometimes, not reliably. Community Forensics shows 1/8-cycle harmonic separation AUC 0.805 (25+25), "
            "and WildFake has strong resampling/FFT effects. On 120 exact DDA pairs, the best tested scalar is only 0.603 "
            "and FFT radial slope is 0.549. Alignment and matched content largely erase the shortcut.",
            kind="warn",
        ),
        mo.ui.table(_periodic_display, selection=None, pagination=True, page_size=18),
        mo.md("**Conclusion:** periodic spectra are useful as one weak branch, not a rule that real cameras never produce. Real textures, demosaicing, JPEG grids, and resizing also create peaks."),
    ])
    return (periodic_names,)


@app.cell(hide_code=True)
def dda_pairs(dda_paired, effects, mo, np, periodic_names):
    _dda_top = (
        effects[effects["scope"] == "within:DDA-COCO"]
        .sort_values("separation_auc", ascending=False).head(18)
    )[["family", "feature", "n_authentic", "n_aigc", "median_delta_aigc_minus_authentic", "auc_aigc_high", "separation_auc"]]
    _dda_periodic = dda_paired[dda_paired["feature"].isin(periodic_names)].copy()
    _dda_periodic["direction_consistency"] = np.maximum(
        _dda_periodic["positive_pair_fraction"], 1 - _dda_periodic["positive_pair_fraction"]
    )
    _dda_periodic = _dda_periodic.sort_values(
        ["direction_consistency", "pairs"], ascending=[False, False]
    ).head(18)
    mo.vstack([
        mo.md("### Hard negative: exact DDA-COCO pairs"),
        mo.hstack([
            mo.vstack([mo.md("**Top DDA scalars**"), mo.ui.table(_dda_top, selection=None, pagination=True, page_size=10)]),
            mo.vstack([mo.md("**Paired periodic deltas**"), mo.ui.table(_dda_periodic, selection=None, pagination=True, page_size=10)]),
        ], widths="equal", align="start"),
        mo.callout("DDA's near-chance results directly contradict a universal upsampling-frequency rule.", kind="success"),
    ])
    return


@app.cell(hide_code=True)
def prnu_answer(effects, mo):
    _camera_names = [
        "camera_cfa_periodicity_proxy", "camera_color_residual_coupling_proxy",
        "camera_signal_noise_fit_proxy", "noise_patch_scale_cv",
        "noise_residual_mean_abs", "noise_flat_region_std", "noise_intensity_slope",
    ]
    _camera = effects[
        effects["feature"].isin(_camera_names) & effects["scope"].str.startswith("within:")
    ].copy()
    _camera["dataset"] = _camera["scope"].str.replace("within:", "", regex=False)
    _camera_display = _camera[[
        "dataset", "feature", "n_authentic", "n_aigc",
        "median_delta_aigc_minus_authentic", "auc_aigc_high", "separation_auc",
    ]].sort_values(["dataset", "separation_auc"], ascending=[True, False])
    mo.vstack([
        mo.md("## 2 · Can PRNU or DSNU distinguish authentic photos?"),
        mo.callout(
            "**Not from these data.** PRNU is a sensor-specific multiplicative pattern estimated by correlating denoising residuals "
            "across repeated images from the same known camera, preferably flat fields or RAW. DSNU is measured from dark frames. "
            "These datasets provide neither repeated camera IDs nor calibration frames/RAW.",
            kind="danger",
        ),
        mo.md("The tested values are only single-image proxies: 2×2 CFA phase contrast, high-pass RGB coupling, signal-dependent residual variance, and patch noise stationarity."),
        mo.ui.table(_camera_display, selection=None, pagination=True, page_size=20),
        mo.md("**Observed:** DDA's best camera proxy is about 0.603 AUC. Community color-residual coupling reaches 0.755. EvalGEN patch-noise CV reaches 0.869 only cross-source and is confounded. None certifies camera authenticity."),
    ])
    return


@app.cell(hide_code=True)
def family_answer(alt, effects, family_control, mo):
    _family_lookup = {
        "FFT magnitude": ["fft_magnitude"],
        "FFT phase": ["fft_phase"],
        "DCT / JPEG": ["dct_jpeg"],
        "Noise / camera proxy": ["noise", "camera_proxy"],
        "Residual / periodicity": ["residual", "multiscale_residual", "codec_resampling", "steganalysis", "patch_distribution"],
    }
    _family_rows = effects[
        effects["family"].isin(_family_lookup[family_control.value])
        & effects["scope"].str.startswith("within:")
    ].copy()
    _family_rows["dataset"] = _family_rows["scope"].str.replace("within:", "", regex=False)
    _family_rows = _family_rows.sort_values(["dataset", "separation_auc"], ascending=[True, False])
    _family_chart = (
        alt.Chart(_family_rows.groupby(["dataset", "feature"], as_index=False).head(6))
        .mark_bar()
        .encode(
            x=alt.X("separation_auc:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="Direction-free separation AUC"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("dataset:N"),
            row=alt.Row("dataset:N", title=None),
            tooltip=["dataset", "feature", "auc_aigc_high", "separation_auc", "median_delta_aigc_minus_authentic"],
        ).properties(width=610, height=120)
    )
    mo.vstack([
        mo.md(f"## 3 · {family_control.value} features in AIGC"),
        _family_chart,
        mo.callout("Feature directions can reverse across datasets. Inspect AUC(AIGC-high) and signed median delta; separation AUC hides reversals.", kind="info"),
        mo.ui.table(_family_rows[[
            "dataset", "family", "feature", "median_authentic", "median_aigc",
            "median_delta_aigc_minus_authentic", "auc_aigc_high", "separation_auc",
        ]], selection=None, pagination=True, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def dct_answer(alt, dataset_control, dct_profiles, effects, mo):
    _dct_summary = (
        dct_profiles[dct_profiles["dataset"] == dataset_control.value]
        .groupby(["label", "dct_band"], as_index=False)
        .agg(dct_energy_ratio=("dct_energy_ratio", "median"))
    )
    _dct_chart = (
        alt.Chart(_dct_summary).mark_line(point=True)
        .encode(
            x=alt.X("dct_band:O", title="8×8 DCT diagonal band (u+v)"),
            y=alt.Y("dct_energy_ratio:Q", title="Median fraction of AC energy"),
            color=alt.Color("label:N"),
            tooltip=["label", "dct_band", alt.Tooltip("dct_energy_ratio:Q", format=".5f")],
        ).properties(width=760, height=300)
    )
    _dct_effects = effects[
        (effects["family"] == "dct_jpeg") & effects["scope"].str.startswith("within:")
    ].copy()
    _dct_effects["dataset"] = _dct_effects["scope"].str.replace("within:", "", regex=False)
    mo.vstack([
        mo.md(f"### DCT profile — {dataset_control.value}"),
        _dct_chart,
        mo.callout(
            "DCT often describes codec history more than generation. SID near-zero/quantization signals reach ~0.83 AUC, "
            "Community is ~0.69, and paired DDA high-AC ratio is only ~0.54. Match JPEG history before using these features.",
            kind="warn",
        ),
        mo.ui.table(_dct_effects[[
            "dataset", "feature", "median_delta_aigc_minus_authentic", "auc_aigc_high", "separation_auc",
        ]].sort_values(["dataset", "separation_auc"], ascending=[True, False]), selection=None, pagination=True, page_size=18),
    ])
    return


@app.cell(hide_code=True)
def newer_answer(effects, features, mo):
    _newer = features[features["dataset"].isin(["DDA-COCO", "EvalGEN", "Community Forensics Eval"])].copy()
    _newer_metrics = [
        "fft_harmonic_excess_1_8", "fft_radial_slope", "fft_high_energy",
        "dct_high_ac_ratio", "dct_near_zero_fraction", "noise_patch_scale_cv",
        "camera_color_residual_coupling_proxy", "residual_kurtosis",
    ]
    _newer_summary = (
        _newer.groupby(["dataset", "label", "generation_model"], as_index=False)[_newer_metrics].median()
    )
    _eval_effects = effects[
        effects["scope"] == "cross-source:EvalGEN-vs-all-authentic"
    ].sort_values("separation_auc", ascending=False).head(15)
    mo.vstack([
        mo.md("## 4 · Does the analysis hold across newer generators?"),
        mo.callout(
            "Only partially. Pinned DDA includes FLUX.1 and SD 3.5 Large reconstruction groups and collapses nearly every scalar. "
            "EvalGEN Flux, GoT, Infinity, NOVA, and OmniGen show high patch-noise heterogeneity and residual tails versus mixed authentic "
            "references, but EvalGEN has no authentic class; that comparison is descriptive.",
            kind="warn",
        ),
        mo.md("### External model median signatures"),
        mo.ui.table(_newer_summary, selection=None, pagination=True, page_size=20),
        mo.md("### EvalGEN versus all authentic references — cross-source only"),
        mo.ui.table(_eval_effects[[
            "family", "feature", "n_authentic", "n_aigc",
            "median_delta_aigc_minus_authentic", "auc_aigc_high", "separation_auc",
        ]], selection=None, pagination=True, page_size=15),
        mo.md("**Decision:** retain a multibranch residual/frequency hypothesis; do not hard-code any FFT peak, DCT threshold, or absence of camera noise as universal."),
    ])
    return


@app.cell(hide_code=True)
def image_controls(Path, features, mo):
    _image_options = {}
    for _image_row in features.to_dict("records"):
        _caption = (
            f"{_image_row['dataset']} · {_image_row['label']} · "
            f"{_image_row['generation_model']} · {Path(_image_row['local_path']).name}"
        )
        _image_options[_caption] = _image_row["row_id"]
    image_control = mo.ui.dropdown(_image_options, value=next(iter(_image_options)), label="Image", searchable=True)
    representation_control = mo.ui.dropdown(
        ["FFT log magnitude", "Gaussian residual", "Mean 8×8 DCT magnitude"],
        value="FFT log magnitude",
        label="Representation",
    )
    mo.vstack([mo.md("## Signal microscope"), mo.hstack([image_control, representation_control], widths=[3, 1], align="end")])
    return image_control, representation_control


@app.cell(hide_code=True)
def helpers(BytesIO, Image, ImageOps, dctn, ndimage, np):
    def normalized_png(array):
        values = np.asarray(array, dtype=np.float64)
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [0.01, 0.99]) if finite.size else (0.0, 1.0)
        scaled = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
        buffer = BytesIO()
        Image.fromarray(np.uint8(255 * scaled), mode="L").save(buffer, format="PNG")
        return buffer.getvalue()


    def signal_representation(image, representation):
        fitted = ImageOps.fit(image.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS)
        rgb = np.asarray(fitted, dtype=np.float64) / 255.0
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
        if representation == "FFT log magnitude":
            window = np.outer(np.hanning(256), np.hanning(256))
            return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((luma - luma.mean()) * window))))
        if representation == "Gaussian residual":
            return np.abs(luma - ndimage.gaussian_filter(luma, sigma=1, mode="reflect"))
        blocks = luma.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3)
        coefficients = np.abs(dctn(blocks * 255.0 - 128.0, axes=(-2, -1), norm="ortho")).mean(axis=(0, 1))
        return np.asarray(Image.fromarray(coefficients).resize((256, 256), Image.Resampling.NEAREST))

    return normalized_png, signal_representation


@app.cell(hide_code=True)
def inspector(
    Image,
    features,
    image_control,
    mo,
    normalized_png,
    pd,
    repo_root,
    representation_control,
    signal_representation,
):
    _selected = features[features["row_id"] == image_control.value].iloc[0]
    with Image.open(repo_root / _selected["local_path"]) as _opened:
        _selected_image = _opened.convert("RGB")
    _representation = signal_representation(_selected_image, representation_control.value)
    _selected_metrics = pd.DataFrame([
        {"feature": _name, "value": _selected[_name]}
        for _name in [
            "fft_harmonic_excess_1_8", "fft_harmonic_excess_1_4",
            "fft_radial_slope", "fft_high_energy", "dct_high_ac_ratio",
            "dct_near_zero_fraction", "camera_cfa_periodicity_proxy",
            "camera_color_residual_coupling_proxy", "noise_patch_scale_cv",
        ]
    ])
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("### Published pixels"), mo.image(_selected_image, width=330)]),
            mo.vstack([mo.md("### " + representation_control.value), mo.image(normalized_png(_representation), width=330)]),
            mo.vstack([mo.md("### Scalar measurements"), mo.ui.table(_selected_metrics, selection=None, pagination=False)]),
        ], widths="equal", align="start"),
        mo.md(f"**{_selected['dataset']} · {_selected['label']} · {_selected['generation_model']}** — {_selected['local_path']}"),
        mo.callout("Maps are normalized per image for display. Compare scalars and population charts, not map brightness.", kind="info"),
    ])
    return


@app.cell(hide_code=True)
def conclusions(mo):
    mo.vstack([
        mo.md("""
    ## Decision record

    - **Fourier branch:** retain radial, phase, harmonic, and residual-periodicity features as diverse weak evidence. Require transform stress tests and generator holdouts.
    - **Camera branch:** retain CFA/color coupling and signal-noise proxies only as supporting evidence. Do not call them PRNU/DSNU.
    - **DCT branch:** treat near-zero AC mass, high-band energy, quantization fraction, and blockiness primarily as codec nuisance controls.
    - **New-generator finding:** residual heterogeneity/tails are the most consistent EvalGEN descriptive differences; DDA shows aligned generation can suppress scalar frequency evidence.
    - **Model implication:** combine these low-level measurements with semantic/patch evidence and train explicitly against codec, resize, source, and camera-pipeline shortcuts.
    """),
        mo.callout(
            "The contribution is not 'FFT detects AI.' It is a falsifiable, source-aware signal panel whose paired DDA hard negative exposes when frequency and camera proxies stop working.",
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def reproduce(feature_registry, mo, pd, run_metadata, sources_metadata):
    _source_rows = []
    for _source_name, _source in sources_metadata["sources"].items():
        _source_rows.append({
            "dataset": _source_name,
            "repository": _source["repository"],
            "revision": _source["revision"],
            "license": _source["license_name"],
            "dataset_url": _source["dataset_url"],
        })
    _source_table = pd.DataFrame(_source_rows)
    mo.vstack([
        mo.md("""
    ## Reproduce and audit

    From the repository root:

        uv run python scripts/prepare_signal_analysis_samples.py
        uv run python scripts/run_signal_analysis.py
        uv run marimo edit notebooks/signal_forensics_eda.py

    The acquisition uses byte ranges for large ZIPs, writes only ignored data, verifies image SHA-256 hashes, pins revisions, and downloads exact COCO originals for DDA pairing. Community Forensics is CC-BY-NC-SA-4.0 and limited to non-commercial research/education with attribution/share-alike duties.

    The organizer demonstration-only COCO/DALL-E Advanced split is not used. Official COCO originals appear only as DDA pair counterparts in this non-training analysis.
    """),
        mo.ui.table(_source_table, selection=None, pagination=False),
        mo.accordion({
            "Run cautions": mo.json(run_metadata["cautions"]),
            "Acquisition notes": mo.json(sources_metadata["notes"]),
            "Feature registry": mo.ui.table(feature_registry, selection=None, pagination=True, page_size=16),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
