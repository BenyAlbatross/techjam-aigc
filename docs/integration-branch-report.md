# Integration branch report

Branch: `integration/gallery-model-workbench`

## Branch audit

- `public-baseline-robustness` is the latest compliance, manifest, inference,
  transformation benchmark, and report implementation.
- `ben` is the base of the separate model-development history.
- `feat/traincode` advances `ben` with TRACE-RX-M training code.
- `feat/trace-rx-parallel` advances `feat/traincode` with the latest parallel
  global and authentic-memory architecture.
- The model-development history and public-baseline history have no Git merge
  base. This branch therefore imports the latest model tree once instead of
  merging three unrelated histories and their large research binaries.

Imported from `feat/trace-rx-parallel` at `140fc91`:

- `src/techjam_aigc/feature_lab/`
- `src/techjam_aigc/trace_rx_m/`
- `src/techjam_aigc/trace_rx_parallel/`
- model and data-expansion configs
- associated tests, planning script, and authoritative problem statement

## Model schema

TRACE-RX Parallel uses a DINOv3 ViT-L/16 encoder with low-rank adapters. Shared
patch tokens feed two parallel detector branches:

1. a global branch over pooled patch statistics;
2. an authentic-memory branch over nearest-reference residuals and retrieval
   statistics.

A transparent learned linear fusion combines the two branch logits. Forward
output includes the final logit, both branch logits, fusion weights, patch
tokens, retrieved authentic references, residuals, retrieval confidence and
entropy, and memory indices. The UI exposes final and branch probabilities when
available, but treats internal tensors and memory indices as diagnostics rather
than calibrated evidence.

Training manifests require `parent_id`, `lineage_id`, `role`, `phase`, `label`,
`sample_kind`, `generator_family`, `generation_model`, `source_dataset`, and
`local_path`. Optional master, prompt, and duplicate groups cannot cross data
roles. DDA derivatives explicitly reference an authentic `source_parent_id`.

## Dataset merge

`scripts/merge_dataset_registry.py` creates a virtual grouped registry from the
public benchmark allowlist and TRACE-RX data-expansion plan. It merges duplicate
source identities and groups entries by approval state and intended role. It
does not copy or concatenate image data and cannot upgrade a pending or blocked
license decision.
