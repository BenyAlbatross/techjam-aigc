# TechJam AIGC detector

This repository explores robust detection of purely AI-generated images. Read the authoritative [problem statement](docs/problem-statement.md) before implementation work.

## Visual dataset EDA

The first analysis phase is deliberately visual and metadata-focused. It does not apply frequency filters or train a model.

Prepare reproducible local slices of the three suggested datasets:

```bash
uv run scripts/prepare_eda_samples.py
```

Open the Marimo notebook using its pinned inline dependencies and a discoverable localhost session:

```bash
uvx marimo@latest edit notebooks/dataset_eda.py --no-token --sandbox
```


## Feature robustness laboratory

Run the deterministic classical-feature, shortcut, and transformation audit:

    uv sync
    uv run pytest -q
    uv run python scripts/run_feature_lab.py --feature-profile frozen_v1 --transform-profile core --workers 4 --bootstrap 200

Plan a larger slice without downloading anything, then run its selected index
only after both license layers are allowlisted:

    uv run python scripts/plan_data_expansion.py path/to/expansion-manifest.csv
    uv run python scripts/run_feature_lab.py --index data/derived/data_expansion/selection.csv --feature-profile expanded_v2 --transform-profile directed_pairs

Open the interactive result notebook:

    uv run marimo edit notebooks/feature_robustness_lab.py --no-token

The laboratory covers spatial, color, texture, residual, noise, FFT
magnitude/phase, DCT/JPEG, wavelet, gradient, and transform-self-consistency
features. See [the experiment contract](docs/feature-robustness-lab.md), the
[six-step plan](docs/aigc-exploration-implementation-plan.md), and its
[plan-to-code reconciliation](docs/aigc-exploration-reconciliation.md).

The original visual EDA notebook provides interactive dataset filters, summaries, galleries, and record-level inspection.

See [data/README.md](data/README.md) for the local data layout and sampling scope.

## TRACE-RX-M v2 training

The staged PyTorch implementation lives under `src/techjam_aigc/trace_rx_m`.
See the [training guide](docs/trace-rx-m-training.md) for the gated DINOv3
licence check, manifest contract, configuration assumptions, and S0--S6
commands.
