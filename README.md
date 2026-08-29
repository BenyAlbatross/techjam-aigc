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

The notebook provides interactive filters, published-scale and local-slice summaries, class/source breakdowns, dimension and aspect-ratio views, format statistics, image galleries, and record-level inspection.

See [data/README.md](data/README.md) for the local data layout and sampling scope.

