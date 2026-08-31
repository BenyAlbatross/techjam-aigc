# /// script
# dependencies = [
#     "altair>=5.5",
#     "marimo",
#     "numpy==2.5.2",
#     "pandas>=2.3",
#     "pillow>=11.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import hashlib
    import io
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@app.cell(hide_code=True)
def _():
    mo.md("""
    # See the data before modeling it

    Visual EDA for **CIFAKE**, **SID Set**, and **WildFake**: dataset scale, class/source makeup, dimensions, file formats, image galleries, and qualitative robustness-transform inspection.

    > **Boundary for this phase:** use your eyes to identify dataset shortcuts, source differences, labeling caveats, and which documented redistribution transforms obscure visible cues. Frequency analysis, model features, training, and quantitative robustness claims remain out of scope here.
    """)
    return


@app.cell(hide_code=True)
def paths():
    repo_root = Path.cwd()
    index_path = repo_root / "data/samples/index.csv"
    summary_path = repo_root / "data/metadata/eda_summary.json"

    if not index_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "EDA data is missing. Run: uv run scripts/prepare_eda_samples.py"
        )
    return index_path, repo_root, summary_path


@app.cell(hide_code=True)
def load_data(index_path, summary_path):
    sample_index = pd.read_csv(index_path).assign(
        megapixels=lambda frame: frame["width"] * frame["height"] / 1_000_000,
        aspect_ratio=lambda frame: frame["width"] / frame["height"],
        kilobytes=lambda frame: frame["bytes"] / 1024,
    )
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    sample_index
    return dataset_summary, sample_index


@app.cell(hide_code=True)
def overview(dataset_summary, sample_index):
    _published = dataset_summary["published_datasets"]
    _local_counts = sample_index.groupby("dataset").size().to_dict()
    published_overview = pd.DataFrame([
        {
            "dataset": "CIFAKE",
            "full dataset": f"{_published['CIFAKE']['images']:,} images",
            "splits": "train 100,000 · test 20,000",
            "generation model(s)": "Stable Diffusion 1.4",
            "source(s)": "CIFAR-10",
            "published resolution": "32×32",
            "local images": _local_counts.get("CIFAKE", 0),
        },
        {
            "dataset": "SID Set",
            "full dataset": f"{_published['SID Set']['available_rows']:,} available rows",
            "splits": "train 210,000 · validation 30,000",
            "generation model(s)": "multiple; model is not exposed per viewer row",
            "source(s)": "OpenImages V7 / SID Set",
            "published resolution": "varied",
            "local images": _local_counts.get("SID Set", 0),
        },
        {
            "dataset": "WildFake",
            "full dataset": f"{_published['WildFake']['images']:,} ({_published['WildFake']['fake_images']:,} fake · {_published['WildFake']['real_images']:,} real)",
            "splits": "paper: 80/20 inside every generator/real source",
            "generation model(s)": ", ".join(_published["WildFake"]["generation_models"]),
            "source(s)": ", ".join(_published["WildFake"]["sources"]),
            "published resolution": "varied by generator/source",
            "local images": _local_counts.get("WildFake", 0),
        },
    ])

    local_overview = (
        sample_index.groupby(["dataset", "split", "label"], as_index=False)
        .agg(
            local_images=("local_path", "count"),
            width_min=("width", "min"),
            width_median=("width", "median"),
            width_max=("width", "max"),
            height_min=("height", "min"),
            height_median=("height", "median"),
            height_max=("height", "max"),
        )
    )

    _wildfake = sample_index[sample_index["dataset"] == "WildFake"].copy()
    _wildfake["resolution"] = _wildfake["width"].astype(str) + "×" + _wildfake["height"].astype(str)
    _wildfake_keys = ["label", "generator_family", "generation_model", "source_dataset"]
    _wildfake_base = (
        _wildfake.groupby(_wildfake_keys, as_index=False)
        .agg(
            local_images=("local_path", "count"),
            observed_resolutions=("resolution", lambda values: ", ".join(sorted(set(values))[:8])),
        )
    )
    _wildfake_splits = (
        _wildfake.groupby(_wildfake_keys + ["split"]).size()
        .unstack(fill_value=0).reset_index()
    )
    wildfake_local_coverage = _wildfake_base.merge(_wildfake_splits, on=_wildfake_keys)

    mo.vstack([
        mo.md("## Dataset inventory — start here"),
        mo.md(f"The browser is showing **{len(sample_index):,} exact published image files**. Full-dataset facts and local EDA counts are intentionally separate."),
        mo.ui.table(published_overview, selection=None, pagination=False),
        mo.md("### Local coverage of native splits and labels"),
        mo.ui.table(local_overview, selection=None, pagination=False),
        mo.md("### WildFake local generator/source coverage"),
        mo.callout(
            "WildFake publishes labels by generator and authentic source, but no fixed train/test membership. This notebook uses a stable hash within every sampled stratum to reconstruct the paper's 80/20 policy, then keeps four images from each side for balanced visual inspection. It is stratum-representative, not prevalence-weighted.",
            kind="info",
        ),
        mo.ui.table(wildfake_local_coverage, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def controls(sample_index):
    _all_datasets = ["All"] + sorted(sample_index["dataset"].unique().tolist())
    _all_labels = ["All"] + sorted(sample_index["label"].unique().tolist())
    _all_splits = ["All"] + sorted(sample_index["split"].unique().tolist())
    _all_families = ["All"] + sorted(sample_index["generator_family"].unique().tolist())
    _all_models = ["All"] + sorted(sample_index["generation_model"].unique().tolist())
    _all_sources = ["All"] + sorted(sample_index["source_dataset"].unique().tolist())

    dataset_control = mo.ui.dropdown(_all_datasets, value="All", label="Dataset")
    label_control = mo.ui.dropdown(_all_labels, value="All", label="Label (charts/all-label view)")
    split_control = mo.ui.dropdown(_all_splits, value="All", label="Split")
    family_control = mo.ui.dropdown(_all_families, value="All", label="Generator family")
    model_control = mo.ui.dropdown(_all_models, value="All", label="Generation model")
    source_control = mo.ui.dropdown(_all_sources, value="All", label="Source dataset")
    gallery_mode_control = mo.ui.dropdown(
        ["Paired binary target (recommended)", "All labels"],
        value="Paired binary target (recommended)",
        label="Gallery mode",
    )
    show_tampered_control = mo.ui.checkbox(
        value=False,
        label="Show SID tampered reference separately",
    )
    gallery_size_control = mo.ui.slider(
        2, 20, step=2, value=8, show_value=True,
        label="Pairs (paired) / cards (all labels)",
    )
    seed_control = mo.ui.slider(0, 100, step=1, value=29, show_value=True, label="Gallery seed")

    control_panel = mo.vstack([
        mo.md("## Choose what to inspect"),
        mo.hstack(
            [dataset_control, label_control, split_control, family_control, model_control, source_control],
            wrap=True,
            justify="start",
        ),
        mo.hstack(
            [gallery_mode_control, show_tampered_control, gallery_size_control, seed_control],
            wrap=True,
            justify="start",
        ),
    ])
    control_panel
    return (
        dataset_control,
        family_control,
        gallery_mode_control,
        gallery_size_control,
        label_control,
        model_control,
        seed_control,
        show_tampered_control,
        source_control,
        split_control,
    )


@app.cell(hide_code=True)
def filtered_data(
    dataset_control,
    family_control,
    label_control,
    model_control,
    sample_index,
    source_control,
    split_control,
):
    _filtered = sample_index
    if dataset_control.value != "All":
        _filtered = _filtered[_filtered["dataset"] == dataset_control.value]
    if label_control.value != "All":
        _filtered = _filtered[_filtered["label"] == label_control.value]
    if split_control.value != "All":
        _filtered = _filtered[_filtered["split"] == split_control.value]
    if family_control.value != "All":
        _filtered = _filtered[_filtered["generator_family"] == family_control.value]
    if model_control.value != "All":
        _filtered = _filtered[_filtered["generation_model"] == model_control.value]
    if source_control.value != "All":
        _filtered = _filtered[_filtered["source_dataset"] == source_control.value]
    filtered = _filtered.reset_index(drop=True)

    mo.md(f"**{len(filtered):,} local images match the current controls.**")
    return (filtered,)


@app.cell(hide_code=True)
def breakdown(filtered):
    breakdown_data = (
        filtered.groupby(
            ["dataset", "label", "generator_family", "generation_model", "source_dataset"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "images"})
    )
    breakdown_chart = (
        alt.Chart(breakdown_data)
        .mark_bar()
        .encode(
            x=alt.X("images:Q", title="Images in local slice"),
            y=alt.Y("generation_model:N", sort="-x", title="Generation model"),
            color=alt.Color("label:N", title="Published label"),
            row=alt.Row("dataset:N", title=None),
            tooltip=["dataset", "label", "generator_family", "generation_model", "source_dataset", "images"],
        )
        .properties(width=520, height=120)
    )

    mo.vstack([
        mo.md("## Class, generator, and source makeup"),
        mo.md("Look for label/source coupling: if a source or model appears under only one label, a detector may learn dataset identity rather than AI-generation evidence."),
        breakdown_chart if len(breakdown_data) else mo.callout("No rows match the controls.", kind="warn"),
    ])
    return


@app.cell(hide_code=True)
def dimensions(filtered):
    dimension_scatter = (
        alt.Chart(filtered)
        .mark_circle(opacity=0.55, size=70)
        .encode(
            x=alt.X("width:Q", title="Width (px)"),
            y=alt.Y("height:Q", title="Height (px)"),
            color=alt.Color("dataset:N"),
            shape=alt.Shape("label:N"),
            tooltip=["dataset", "label", "source", "width", "height", "format", "kilobytes"],
        )
        .properties(width=500, height=330)
        .interactive()
    )
    aspect_histogram = (
        alt.Chart(filtered)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("aspect_ratio:Q", bin=alt.Bin(maxbins=30), title="Aspect ratio (width / height)"),
            y=alt.Y("count():Q", title="Local images"),
            color=alt.Color("dataset:N"),
            tooltip=["dataset", "count()"],
        )
        .properties(width=500, height=240)
    )
    format_table = (
        filtered.groupby(["dataset", "format", "mode"], as_index=False)
        .agg(images=("local_path", "count"), median_kb=("kilobytes", "median"))
        .sort_values(["dataset", "images"], ascending=[True, False])
    )

    mo.vstack([
        mo.md("## Dimensions, aspect ratios, and encodings"),
        mo.md("These are prime places to spot shortcuts before training: CIFAKE's tiny fixed resolution is visually unlike the larger datasets."),
        dimension_scatter,
        aspect_histogram,
        mo.ui.table(format_table, selection=None, pagination=False),
    ])
    return


@app.function(hide_code=True)
def make_image_card(row, root):
    caption = (
        f"**{row['dataset']} · {row['label']} · {row['split']}**<br>"
        f"Model: {row['generation_model']} · family: {row['generator_family']}<br>"
        f"Source: {row['source_dataset']}<br>"
        f"{int(row['width'])}×{int(row['height'])} · {row['format']} · {row['kilobytes']:.1f} KB"
    )
    image_bytes = (root / row["local_path"]).read_bytes()
    return mo.vstack([
        mo.image(image_bytes, width=220),
        mo.md(caption),
    ])


@app.cell(hide_code=True)
def gallery(
    dataset_control,
    family_control,
    filtered,
    gallery_mode_control,
    gallery_size_control,
    model_control,
    repo_root,
    sample_index,
    seed_control,
    show_tampered_control,
    source_control,
    split_control,
):
    _pair_mode = gallery_mode_control.value == "Paired binary target (recommended)"
    _gallery_limit = int(gallery_size_control.value)

    if _pair_mode:
        _pair_base = sample_index
        if dataset_control.value != "All":
            _pair_base = _pair_base[_pair_base["dataset"] == dataset_control.value]
        if split_control.value != "All":
            _pair_base = _pair_base[_pair_base["split"] == split_control.value]

        _real_pool = _pair_base[_pair_base["label"] == "real"]
        _fake_pool = _pair_base[_pair_base["label"].isin(["fake", "full_synthetic"])]
        if family_control.value != "All":
            _fake_pool = _fake_pool[_fake_pool["generator_family"] == family_control.value]
        if model_control.value != "All":
            _fake_pool = _fake_pool[_fake_pool["generation_model"] == model_control.value]
        if source_control.value != "All":
            _fake_pool = _fake_pool[_fake_pool["source_dataset"] == source_control.value]

        _real_strata = set(zip(_real_pool["dataset"], _real_pool["split"]))
        _fake_groups = {
            _key: _group.sample(frac=1, random_state=int(seed_control.value) + _position)
            for _position, (_key, _group) in enumerate(
                _fake_pool.groupby(["dataset", "split"], sort=True)
            )
            if _key in _real_strata
        }
        _group_positions = {_key: 0 for _key in _fake_groups}
        _pair_records = []
        while len(_pair_records) < _gallery_limit:
            _added = False
            for _key, _group in _fake_groups.items():
                _position = _group_positions[_key]
                if _position >= len(_group) or len(_pair_records) >= _gallery_limit:
                    continue
                _fake_row = _group.iloc[_position]
                _group_positions[_key] += 1
                _real_candidates = _real_pool[
                    (_real_pool["dataset"] == _key[0]) & (_real_pool["split"] == _key[1])
                ]
                _same_source = _real_candidates[
                    _real_candidates["source_dataset"] == _fake_row["source_dataset"]
                ]
                _source_matched = len(_same_source) > 0
                if _source_matched:
                    _real_candidates = _same_source
                _real_row = _real_candidates.iloc[
                    (int(seed_control.value) + len(_pair_records)) % len(_real_candidates)
                ]
                _pair_records.append((_real_row, _fake_row, _source_matched))
                _added = True
            if not _added:
                break

        gallery_cards = []
        for _pair_number, (_real_row, _fake_row, _source_matched) in enumerate(_pair_records, start=1):
            _match_text = "source-matched" if _source_matched else "same dataset/split; source differs"
            gallery_cards.append(
                mo.vstack([
                    mo.md(
                        f"**Pair {_pair_number} · {_fake_row['dataset']} · {_fake_row['split']}** — {_match_text}"
                    ),
                    mo.hstack(
                        [
                            mo.vstack([
                                mo.md("### Authentic (binary negative)"),
                                make_image_card(_real_row, repo_root),
                            ]),
                            mo.vstack([
                                mo.md("### Pure-generated (binary positive)"),
                                make_image_card(_fake_row, repo_root),
                            ]),
                        ],
                        wrap=False,
                        justify="start",
                    ),
                ])
            )

        gallery_rows = pd.concat(
            [pd.DataFrame([_real_row, _fake_row]) for _real_row, _fake_row, _ in _pair_records],
            ignore_index=True,
        ) if _pair_records else sample_index.head(0)
        _main_gallery = (
            mo.vstack(gallery_cards)
            if gallery_cards
            else mo.callout(
                "No authentic/pure-generated pairs match the dataset, split, and generated-side filters.",
                kind="warn",
            )
        )
        _mode_note = mo.callout(
            "Pair mode maps only `real` → authentic and `fake`/`full_synthetic` → pure-generated. The label control is intentionally ignored here. Generator-family, model, and source filters apply to the generated side; authentic candidates come from the same dataset and split, with source matching preferred when available.",
            kind="info",
        )
    else:
        _gallery_count = min(_gallery_limit, len(filtered))
        gallery_rows = (
            filtered.sample(n=_gallery_count, random_state=int(seed_control.value))
            if _gallery_count else filtered.head(0)
        )
        gallery_cards = [make_image_card(_row, repo_root) for _, _row in gallery_rows.iterrows()]
        _main_gallery = (
            mo.hstack(gallery_cards, wrap=True, justify="start")
            if gallery_cards
            else mo.callout("No images match the current controls.", kind="warn")
        )
        _mode_note = mo.callout(
            "All-label mode respects every filter and may mix labels. Switch back to paired mode for the challenge's binary comparison.",
            kind="info",
        )

    if show_tampered_control.value:
        _tampered_pool = sample_index[sample_index["label"] == "tampered"]
        if dataset_control.value != "All":
            _tampered_pool = _tampered_pool[_tampered_pool["dataset"] == dataset_control.value]
        if split_control.value != "All":
            _tampered_pool = _tampered_pool[_tampered_pool["split"] == split_control.value]
        _tampered_count = min(6, len(_tampered_pool))
        _tampered_rows = (
            _tampered_pool.sample(n=_tampered_count, random_state=int(seed_control.value))
            if _tampered_count else _tampered_pool.head(0)
        )
        _tampered_cards = [make_image_card(_row, repo_root) for _, _row in _tampered_rows.iterrows()]
        _tampered_view = mo.vstack([
            mo.md("### SID tampered reference — outside the hackathon binary target"),
            mo.callout(
                "These are shown for dataset understanding only. They are neither authentic negatives nor purely generated positives for this track.",
                kind="warn",
            ),
            mo.hstack(_tampered_cards, wrap=True, justify="start")
            if _tampered_cards else mo.md("No tampered rows match the dataset/split controls."),
        ])
    else:
        _tampered_view = mo.md("")

    gallery_view = mo.vstack([
        mo.md("## Compare authentic and pure-generated images"),
        mo.md("Read each row left-to-right. Pairing is visual juxtaposition, not a claim that the scenes are semantically identical."),
        _mode_note,
        _main_gallery,
        _tampered_view,
    ])

    gallery_view
    return


@app.cell(hide_code=True)
def transform_helpers():
    TRANSFORM_FAMILIES = {
        "JPEG compression": [
            ("JPEG quality 90", ("jpeg", 90)),
            ("JPEG quality 70", ("jpeg", 70)),
            ("JPEG quality 50", ("jpeg", 50)),
            ("JPEG quality 30", ("jpeg", 30)),
        ],
        "Gaussian blur": [
            ("Gaussian blur · σ 0.5", ("blur", 0.5)),
            ("Gaussian blur · σ 1.0", ("blur", 1.0)),
            ("Gaussian blur · σ 2.0", ("blur", 2.0)),
        ],
        "Resize then upscale": [
            ("Resize 0.5× → upscale", ("resize", 0.5)),
            ("Resize 0.25× → upscale", ("resize", 0.25)),
        ],
        "Gaussian noise": [
            ("Gaussian noise · σ 0.02", ("noise", 0.02)),
            ("Gaussian noise · σ 0.05", ("noise", 0.05)),
            ("Gaussian noise · σ 0.10", ("noise", 0.10)),
        ],
        "Color jitter": [
            ("Brightness −20%", ("brightness", 0.8)),
            ("Brightness +20%", ("brightness", 1.2)),
            ("Contrast −20%", ("contrast", 0.8)),
            ("Contrast +20%", ("contrast", 1.2)),
            ("Saturation −20%", ("saturation", 0.8)),
            ("Saturation +20%", ("saturation", 1.2)),
        ],
        "Center crop": [
            ("Center crop · retain 80%", ("crop", 0.8)),
        ],
    }

    TRANSFORM_OPTIONS = {"None": None}
    for _family_variants in TRANSFORM_FAMILIES.values():
        for _variant_label, _variant_spec in _family_variants:
            TRANSFORM_OPTIONS[_variant_label] = _variant_spec

    CHAIN_PRESETS = {
        "Manual": None,
        "Light redistribution": [("resize", 0.5), ("jpeg", 70)],
        "Heavy redistribution": [("resize", 0.25), ("blur", 1.0), ("jpeg", 30)],
        "Noisy repost": [("noise", 0.02), ("resize", 0.5), ("jpeg", 50)],
        "Filtered repost": [("brightness", 1.2), ("contrast", 1.2), ("saturation", 1.2), ("resize", 0.5), ("jpeg", 70)],
        "Crop and repost": [("crop", 0.8), ("resize", 0.5), ("jpeg", 50)],
    }


    def load_eda_image(path):
        with Image.open(path) as _source:
            return ImageOps.exif_transpose(_source).convert("RGB")


    def transform_spec_label(spec):
        for _label, _candidate in TRANSFORM_OPTIONS.items():
            if _candidate == spec:
                return _label
        return f"{spec[0]} · {spec[1]}"


    def stable_transform_seed(path_key, spec, occurrence, base_seed):
        _payload = (
            f"{path_key}\0{spec[0]}\0{spec[1]}\0{occurrence}\0{int(base_seed)}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(_payload).digest()[:8], "big")


    def apply_robustness_transform(image, spec, path_key, base_seed, occurrence=0):
        _operation, _value = spec
        _image = image.convert("RGB")

        if _operation == "jpeg":
            _buffer = io.BytesIO()
            _image.save(
                _buffer,
                format="JPEG",
                quality=int(_value),
                subsampling=2,
                optimize=False,
                progressive=False,
            )
            _buffer.seek(0)
            with Image.open(_buffer) as _decoded:
                return _decoded.convert("RGB").copy()

        if _operation == "blur":
            return _image.filter(ImageFilter.GaussianBlur(radius=float(_value)))

        if _operation == "resize":
            _width, _height = _image.size
            _down_size = (
                max(1, int(round(_width * float(_value)))),
                max(1, int(round(_height * float(_value)))),
            )
            return _image.resize(_down_size, Image.Resampling.LANCZOS).resize(
                (_width, _height), Image.Resampling.LANCZOS
            )

        if _operation == "noise":
            _pixels = np.asarray(_image, dtype=np.float32) / 255.0
            _rng = np.random.default_rng(
                stable_transform_seed(path_key, spec, occurrence, base_seed)
            )
            _noise = _rng.normal(0.0, float(_value), size=_pixels.shape)
            _noisy = np.clip(_pixels + _noise, 0.0, 1.0)
            return Image.fromarray(np.rint(_noisy * 255.0).astype(np.uint8), mode="RGB")

        if _operation == "brightness":
            return ImageEnhance.Brightness(_image).enhance(float(_value))
        if _operation == "contrast":
            return ImageEnhance.Contrast(_image).enhance(float(_value))
        if _operation == "saturation":
            return ImageEnhance.Color(_image).enhance(float(_value))

        if _operation == "crop":
            _width, _height = _image.size
            _crop_width = max(1, int(round(_width * float(_value))))
            _crop_height = max(1, int(round(_height * float(_value))))
            _left = (_width - _crop_width) // 2
            _top = (_height - _crop_height) // 2
            return _image.crop((_left, _top, _left + _crop_width, _top + _crop_height))

        raise ValueError(f"Unknown transform operation: {_operation}")


    def apply_transform_chain(image, specs, path_key, base_seed):
        _current = image
        _stages = [("Original", _current)]
        _occurrences = {}
        for _spec in specs:
            _key = repr(_spec)
            _occurrence = _occurrences.get(_key, 0)
            _occurrences[_key] = _occurrence + 1
            _current = apply_robustness_transform(
                _current, _spec, path_key, base_seed, _occurrence
            )
            _stages.append((transform_spec_label(_spec), _current))
        return _stages


    def image_to_png_bytes(image):
        _buffer = io.BytesIO()
        image.save(_buffer, format="PNG")
        return _buffer.getvalue()


    def make_transform_card(image, stage_label, display_mode):
        _width, _height = image.size
        _display_image = image
        _display_note = "whole image"
        _image_style = {"max-width": "100%", "height": "auto"}
        _render_width = 360

        if display_mode == "Center detail":
            _side = max(1, min(128, _width, _height))
            _left = (_width - _side) // 2
            _top = (_height - _side) // 2
            _display_image = image.crop((_left, _top, _left + _side, _top + _side))
            _display_note = f"center {_side}×{_side}px detail; display-only zoom"
            _render_width = 384
        elif display_mode == "Pixel zoom":
            _side = max(1, min(64, _width, _height))
            _left = (_width - _side) // 2
            _top = (_height - _side) // 2
            _display_image = image.crop((_left, _top, _left + _side, _top + _side))
            _display_note = f"center {_side}×{_side}px nearest-neighbor display zoom"
            _image_style["image-rendering"] = "pixelated"
            _render_width = 384

        return mo.vstack([
            mo.image(
                image_to_png_bytes(_display_image),
                width=_render_width,
                alt=f"{stage_label}; {_width} by {_height} pixels",
                style=_image_style,
            ),
            mo.md(f"**{stage_label}**<br>{_width}×{_height}px · {_display_note}"),
        ]).style({"min-width": "380px"})


    def make_transform_strip(stages, display_mode):
        return mo.hstack(
            [
                make_transform_card(_stage_image, _stage_label, display_mode)
                for _stage_label, _stage_image in stages
            ],
            justify="start",
            align="start",
            wrap=False,
            gap=1.0,
        ).style({"overflow-x": "auto", "padding-bottom": "0.75rem"})


    def transform_item_title(item, position, hide_labels):
        if hide_labels:
            return f"Image {'AB'[position]} · labels hidden"
        if item["row"]["label"] == "real":
            return "Authentic · binary negative"
        return "Pure-generated · binary positive"


    return (
        CHAIN_PRESETS,
        TRANSFORM_FAMILIES,
        TRANSFORM_OPTIONS,
        apply_robustness_transform,
        apply_transform_chain,
        load_eda_image,
        make_transform_strip,
        transform_item_title,
        transform_spec_label,
    )


@app.cell(hide_code=True)
def transform_controls(sample_index, seed_control):
    _pair_choices = {}
    for _pair_dataset in ["SID Set", "WildFake", "CIFAKE"]:
        _dataset_rows = sample_index[sample_index["dataset"] == _pair_dataset]
        _real_pool = _dataset_rows[_dataset_rows["label"] == "real"].copy()
        _fake_pool = _dataset_rows[
            _dataset_rows["label"].isin(["fake", "full_synthetic"])
        ].copy()
        if _real_pool.empty or _fake_pool.empty:
            continue

        _fake_pool["_pixel_count"] = _fake_pool["width"] * _fake_pool["height"]
        _fake_pool = _fake_pool.sort_values(
            ["_pixel_count", "generation_model", "local_path"],
            ascending=[False, True, True],
        )
        _model_first = _fake_pool.drop_duplicates("generation_model", keep="first")
        _remaining = _fake_pool.drop(index=_model_first.index)
        _ranked_fakes = pd.concat([_model_first, _remaining]).head(8)

        _real_pool["_pixel_count"] = _real_pool["width"] * _real_pool["height"]
        for _pair_number, (_fake_index, _fake_row) in enumerate(
            _ranked_fakes.iterrows(), start=1
        ):
            _source_split_matches = _real_pool[
                (_real_pool["source_dataset"] == _fake_row["source_dataset"])
                & (_real_pool["split"] == _fake_row["split"])
            ]
            _source_matches = _real_pool[
                _real_pool["source_dataset"] == _fake_row["source_dataset"]
            ]
            _split_matches = _real_pool[_real_pool["split"] == _fake_row["split"]]
            if not _source_split_matches.empty:
                _real_candidates = _source_split_matches
                _match_note = "source/split matched"
            elif not _source_matches.empty:
                _real_candidates = _source_matches
                _match_note = "source matched"
            elif not _split_matches.empty:
                _real_candidates = _split_matches
                _match_note = "split matched"
            else:
                _real_candidates = _real_pool
                _match_note = "dataset matched"

            _real_candidates = _real_candidates.sort_values(
                ["_pixel_count", "local_path"], ascending=[False, True]
            )
            _real_index = _real_candidates.index[
                (_pair_number - 1) % len(_real_candidates)
            ]
            _caption = (
                f"{_pair_dataset} · {_fake_row['generation_model']} · "
                f"{int(_fake_row['width'])}×{int(_fake_row['height'])} · "
                f"pair {_pair_number} ({_match_note})"
            )
            _pair_choices[_caption] = [int(_real_index), int(_fake_index)]

    transform_pair_count = len(_pair_choices)
    _pair_default = next(iter(_pair_choices)) if _pair_choices else None
    transform_pair_control = mo.ui.dropdown(
        _pair_choices if _pair_choices else {"No binary pairs available": []},
        value=_pair_default if _pair_default is not None else "No binary pairs available",
        label="Dataset/model pair · larger generated images first",
        searchable=True,
        full_width=True,
        disabled=not _pair_choices,
    )
    transform_seed_control = mo.ui.slider(
        0, 100, step=1, value=int(seed_control.value), show_value=True,
        label="Transform seed",
    )
    transform_display_control = mo.ui.dropdown(
        ["Fitted whole image", "Center detail", "Pixel zoom"],
        value="Fitted whole image",
        label="Display mode",
    )
    transform_hide_labels_control = mo.ui.checkbox(
        value=False,
        label="Hide class labels and metadata",
    )

    transform_global_controls = mo.vstack([
        transform_pair_control,
        mo.hstack(
            [
                transform_seed_control,
                transform_display_control,
                transform_hide_labels_control,
            ],
            wrap=True,
            justify="start",
        ),
    ])

    return (
        transform_display_control,
        transform_global_controls,
        transform_hide_labels_control,
        transform_pair_control,
        transform_pair_count,
        transform_seed_control,
    )


@app.cell(hide_code=True)
def selected_transform_pair(
    load_eda_image,
    repo_root,
    sample_index,
    transform_hide_labels_control,
    transform_item_title,
    transform_pair_control,
    transform_pair_count,
):
    selected_transform_items = []
    if transform_pair_count and transform_pair_control.value:
        _selected_indices = [int(_index) for _index in transform_pair_control.value]
        _selected_rows = sample_index.loc[_selected_indices]
        for _position, (_, _selected_row) in enumerate(_selected_rows.iterrows()):
            selected_transform_items.append({
                "row": _selected_row,
                "image": load_eda_image(repo_root / _selected_row["local_path"]),
                "path_key": str(_selected_row["local_path"]),
                "position": _position,
            })

    if not selected_transform_items:
        transform_pair_summary = mo.callout(
            "No authentic/pure-generated pair is available in the local EDA sample.",
            kind="warn",
        )
    elif transform_hide_labels_control.value:
        transform_pair_summary = mo.callout(
            "Class labels and source metadata are hidden to reduce expectancy bias. Reveal them after making a visual judgment.",
            kind="info",
        )
    else:
        _summary_lines = []
        for _item in selected_transform_items:
            _row = _item["row"]
            _summary_lines.append(
                f"- **{transform_item_title(_item, _item['position'], False)}:** "
                f"{_row['dataset']} · {_row['split']} · model {_row['generation_model']} · "
                f"family {_row['generator_family']} · source {_row['source_dataset']} · "
                f"{int(_row['width'])}×{int(_row['height'])}"
            )
        transform_pair_summary = mo.md("\n".join(_summary_lines))

    return selected_transform_items, transform_pair_summary


@app.cell(hide_code=True)
def atlas_controls(TRANSFORM_FAMILIES):
    atlas_family_control = mo.ui.dropdown(
        list(TRANSFORM_FAMILIES),
        value="JPEG compression",
        label="Transform family",
    )
    atlas_controls_panel = mo.hstack(
        [atlas_family_control], wrap=True, justify="start"
    )
    return atlas_controls_panel, atlas_family_control


@app.cell(hide_code=True)
def transform_atlas(
    TRANSFORM_FAMILIES,
    apply_robustness_transform,
    atlas_controls_panel,
    atlas_family_control,
    make_transform_strip,
    selected_transform_items,
    transform_display_control,
    transform_hide_labels_control,
    transform_item_title,
    transform_seed_control,
):
    if not selected_transform_items:
        atlas_section = mo.callout(
            "No paired images are available for the transform atlas.", kind="warn"
        )
    else:
        _atlas_rows = []
        _atlas_variants = TRANSFORM_FAMILIES[atlas_family_control.value]
        for _item in selected_transform_items:
            _atlas_stages = [("Original", _item["image"])]
            for _variant_label, _variant_spec in _atlas_variants:
                _atlas_stages.append((
                    _variant_label,
                    apply_robustness_transform(
                        _item["image"],
                        _variant_spec,
                        _item["path_key"],
                        transform_seed_control.value,
                    ),
                ))
            _atlas_rows.append(mo.vstack([
                mo.md(
                    f"### {transform_item_title(_item, _item['position'], transform_hide_labels_control.value)}"
                ),
                make_transform_strip(_atlas_stages, transform_display_control.value),
            ]))

        atlas_section = mo.vstack([
            atlas_controls_panel,
            mo.md(
                "Each row repeats the unchanged original, then applies one documented "
                "severity directly to that original. Scroll horizontally to compare pixels."
            ),
            *_atlas_rows,
        ]).style({
            "max-height": "540px",
            "overflow-y": "auto",
            "padding-right": "0.5rem",
        })
    return (atlas_section,)


@app.cell(hide_code=True)
def chain_controls(CHAIN_PRESETS, TRANSFORM_OPTIONS):
    chain_preset_control = mo.ui.dropdown(
        list(CHAIN_PRESETS),
        value="Manual",
        label="Exploratory preset",
    )
    chain_step_1_control = mo.ui.dropdown(
        list(TRANSFORM_OPTIONS),
        value="Resize 0.5× → upscale",
        label="Step 1",
    )
    chain_step_2_control = mo.ui.dropdown(
        list(TRANSFORM_OPTIONS),
        value="JPEG quality 70",
        label="Step 2",
    )
    chain_step_3_control = mo.ui.dropdown(
        list(TRANSFORM_OPTIONS),
        value="None",
        label="Step 3",
    )
    chain_step_4_control = mo.ui.dropdown(
        list(TRANSFORM_OPTIONS),
        value="None",
        label="Step 4",
    )
    chain_step_controls = [
        chain_step_1_control,
        chain_step_2_control,
        chain_step_3_control,
        chain_step_4_control,
    ]
    chain_controls_panel = mo.vstack([
        mo.hstack([chain_preset_control], wrap=True, justify="start"),
        mo.hstack(chain_step_controls, wrap=True, justify="start"),
    ])
    return chain_controls_panel, chain_preset_control, chain_step_controls


@app.cell(hide_code=True)
def transform_chain(
    CHAIN_PRESETS,
    TRANSFORM_OPTIONS,
    apply_transform_chain,
    chain_controls_panel,
    chain_preset_control,
    chain_step_controls,
    make_transform_strip,
    selected_transform_items,
    transform_display_control,
    transform_hide_labels_control,
    transform_item_title,
    transform_seed_control,
    transform_spec_label,
):
    if chain_preset_control.value == "Manual":
        _chain_specs = [
            TRANSFORM_OPTIONS[_control.value]
            for _control in chain_step_controls
            if TRANSFORM_OPTIONS[_control.value] is not None
        ]
        _chain_source_note = "Manual chain"
    else:
        _chain_specs = CHAIN_PRESETS[chain_preset_control.value]
        _chain_source_note = f"Exploratory preset: {chain_preset_control.value}"

    _chain_description = (
        " → ".join(transform_spec_label(_spec) for _spec in _chain_specs)
        if _chain_specs else "No transforms selected"
    )

    if not selected_transform_items:
        chain_section = mo.callout(
            "No paired images are available for the chain explorer.", kind="warn"
        )
    else:
        _chain_rows = []
        for _item in selected_transform_items:
            _chain_stages = apply_transform_chain(
                _item["image"],
                _chain_specs,
                _item["path_key"],
                transform_seed_control.value,
            )
            _chain_rows.append(mo.vstack([
                mo.md(
                    f"### {transform_item_title(_item, _item['position'], transform_hide_labels_control.value)}"
                ),
                make_transform_strip(_chain_stages, transform_display_control.value),
            ]))

        chain_section = mo.vstack([
            chain_controls_panel,
            mo.callout(
                f"**{_chain_source_note}:** {_chain_description}. Presets are qualitative "
                "EDA hypotheses, not organizer-defined evaluation pipelines.",
                kind="info",
            ),
            *_chain_rows,
        ]).style({
            "max-height": "540px",
            "overflow-y": "auto",
            "padding-right": "0.5rem",
        })
    return (chain_section,)


@app.cell(hide_code=True)
def order_controls(TRANSFORM_OPTIONS):
    _order_option_labels = [
        _label for _label, _spec in TRANSFORM_OPTIONS.items() if _spec is not None
    ]
    order_a_control = mo.ui.dropdown(
        _order_option_labels,
        value="Resize 0.5× → upscale",
        label="Operation A",
    )
    order_b_control = mo.ui.dropdown(
        _order_option_labels,
        value="JPEG quality 70",
        label="Operation B",
    )
    order_controls_panel = mo.hstack(
        [order_a_control, order_b_control],
        wrap=True,
        justify="start",
    )
    return order_a_control, order_b_control, order_controls_panel


@app.cell(hide_code=True)
def transform_order(
    TRANSFORM_OPTIONS,
    apply_robustness_transform,
    make_transform_strip,
    order_a_control,
    order_b_control,
    order_controls_panel,
    selected_transform_items,
    transform_display_control,
    transform_hide_labels_control,
    transform_item_title,
    transform_seed_control,
):
    _order_a_spec = TRANSFORM_OPTIONS[order_a_control.value]
    _order_b_spec = TRANSFORM_OPTIONS[order_b_control.value]
    _order_second_occurrence = 1 if _order_a_spec == _order_b_spec else 0

    if not selected_transform_items:
        order_section = mo.callout(
            "No paired images are available for the order comparison.", kind="warn"
        )
    else:
        _order_rows = []
        for _item in selected_transform_items:
            _original = _item["image"]
            _path_key = _item["path_key"]
            _order_a_image = apply_robustness_transform(
                _original,
                _order_a_spec,
                _path_key,
                transform_seed_control.value,
                0,
            )
            _order_b_image = apply_robustness_transform(
                _original,
                _order_b_spec,
                _path_key,
                transform_seed_control.value,
                0,
            )
            _order_ab_image = apply_robustness_transform(
                _order_a_image,
                _order_b_spec,
                _path_key,
                transform_seed_control.value,
                _order_second_occurrence,
            )
            _order_ba_image = apply_robustness_transform(
                _order_b_image,
                _order_a_spec,
                _path_key,
                transform_seed_control.value,
                _order_second_occurrence,
            )
            _order_stages = [
                ("Original", _original),
                (f"A only · {order_a_control.value}", _order_a_image),
                ("A → B", _order_ab_image),
                (f"B only · {order_b_control.value}", _order_b_image),
                ("B → A", _order_ba_image),
            ]
            _order_rows.append(mo.vstack([
                mo.md(
                    f"### {transform_item_title(_item, _item['position'], transform_hide_labels_control.value)}"
                ),
                make_transform_strip(_order_stages, transform_display_control.value),
            ]))

        order_section = mo.vstack([
            order_controls_panel,
            mo.md(
                f"Compare **A = {order_a_control.value}** and **B = {order_b_control.value}**. "
                "For stochastic operations, the same image uses the same operation-specific "
                "random draw in both orders; the two classes use independent draws."
            ),
            *_order_rows,
        ]).style({
            "max-height": "540px",
            "overflow-y": "auto",
            "padding-right": "0.5rem",
        })
    return (order_section,)


@app.cell(hide_code=True)
def transform_lab(
    atlas_section,
    chain_section,
    order_section,
    transform_global_controls,
    transform_pair_summary,
):
    transform_lab_tabs = mo.ui.tabs(
        {
            "Single-transform atlas": atlas_section,
            "Sequential chain": chain_section,
            "Order comparison": order_section,
        },
        value="Single-transform atlas",
        lazy=True,
    )

    transform_lab_view = mo.vstack([
        mo.md("## Robustness transform lab"),
        mo.md(
            "Choose an authentic/generated pair independently of the main gallery filters, "
            "then judge—qualitatively—whether documented redistribution transforms obscure "
            "visible cues. The picker includes up to eight pairs each from SID Set, WildFake, "
            "and CIFAKE. It prioritizes larger generated images and model diversity as a "
            "navigation heuristic, not as a human-rated realism score. Display zoom never "
            "changes pipeline pixels."
        ),
        transform_global_controls,
        transform_pair_summary,
        mo.callout(
            "**Explicit EDA semantics.** Sources are EXIF-transposed and converted to RGB. "
            "JPEG uses fixed 4:2:0 subsampling, no optimization/progressive encoding, and is "
            "decoded before the next step. Pillow Gaussian radius is treated as pixel σ. "
            "Resize uses LANCZOS downsampling and upsampling back to the pre-step size. "
            "Noise σ is applied in clipped [0,1] RGB with a stable per-image seed. Color "
            "jitter varies brightness, contrast, or saturation independently at 0.8/1.2. "
            "Center crop retains 80% of each spatial dimension and is not resized back. "
            "The brief leaves transform composition and several implementation details open, "
            "so these are reproducible notebook choices rather than organizer-confirmed rules.",
            kind="info",
        ),
        transform_lab_tabs,
        mo.callout(
            "This is qualitative EDA, not evidence of detector robustness. Inspect several "
            "pairs, seeds, datasets, resolutions, and generator families before forming a hypothesis.",
            kind="warn",
        ),
    ])

    transform_lab_view

    return


@app.cell(hide_code=True)
def records(filtered):
    record_table = filtered[
        [
            "dataset", "split", "label", "generator_family", "generation_model",
            "source_dataset", "stratum", "split_method", "width", "height",
            "aspect_ratio", "format", "mode", "kilobytes", "local_path", "source_member",
        ]
    ].copy()

    mo.vstack([
        mo.md("## Inspect the records"),
        mo.ui.table(record_table, selection="multi", pagination=True, page_size=15),
    ])
    return


@app.cell(hide_code=True)
def boundary():
    mo.vstack([
        mo.md("## Label and sampling caveats"),
        mo.callout(
            "SID Set includes a **tampered** class. The hackathon target explicitly focuses on purely generated versus authentic images, so tampered rows are useful for understanding the source dataset but should not silently become a third training target.",
            kind="warn",
        ),
        mo.callout(
            "The WildFake gallery spans sampled authentic, GAN, diffusion, and other-model strata, but it is still a small visual slice. The opening inventory lists the full published model/source universe so omitted strata remain visible.",
            kind="info",
        ),
        mo.md("""
    ### Deliberately deferred

    The robustness lab performs only the six documented pixel-transform families for **qualitative inspection**. This notebook still computes no FFT, DCT, high-pass or spectral features, trains no detector, and makes no quantitative robustness claim.
    """),
    ])
    return


if __name__ == "__main__":
    app.run()
