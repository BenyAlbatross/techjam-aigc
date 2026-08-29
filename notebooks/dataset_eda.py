# /// script
# dependencies = [
#     "altair>=5.5",
#     "marimo",
#     "pandas>=2.3",
#     "pillow>=11.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md("""
    # See the data before modeling it

    Visual EDA for **CIFAKE**, **SID Set**, and **WildFake**. This notebook stays intentionally close to the published pixels: dataset scale, class/source makeup, dimensions, file formats, and image galleries.

    > **Boundary for this phase:** no FFT, DCT, high-pass filters, spectral plots, augmentations, or model features yet. First use your eyes and identify dataset shortcuts, source differences, and labeling caveats.
    """)
    return


@app.cell
def paths():
    repo_root = Path.cwd()
    index_path = repo_root / "data/samples/index.csv"
    summary_path = repo_root / "data/metadata/eda_summary.json"

    if not index_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "EDA data is missing. Run: uv run scripts/prepare_eda_samples.py"
        )
    return index_path, repo_root, summary_path


@app.cell
def load_data(index_path, summary_path):
    sample_index = pd.read_csv(index_path).assign(
        megapixels=lambda frame: frame["width"] * frame["height"] / 1_000_000,
        aspect_ratio=lambda frame: frame["width"] / frame["height"],
        kilobytes=lambda frame: frame["bytes"] / 1024,
    )
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    sample_index
    return dataset_summary, sample_index


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.function
def make_image_card(row, root):
    caption = (
        f"**{row['dataset']} · {row['label']} · {row['split']}**<br>"
        f"Model: {row['generation_model']} · family: {row['generator_family']}<br>"
        f"Source: {row['source_dataset']}<br>"
        f"{int(row['width'])}×{int(row['height'])} · {row['format']} · {row['kilobytes']:.1f} KB"
    )
    image_bytes = (root / row["local_path"]).read_bytes()
    return mo.vstack([
        mo.image(image_bytes, width=190),
        mo.md(caption),
    ])


@app.cell
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


@app.cell
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

    Only after reviewing these views should we decide which frequency representations or filters are worth testing. This notebook currently computes **none**.
    """),
    ])

    return


if __name__ == "__main__":
    app.run()
