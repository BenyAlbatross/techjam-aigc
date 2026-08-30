# AIGC feature robustness laboratory

## Purpose

The laboratory turns classical image-processing operations into falsifiable
AIGC-detection hypotheses. It is an exploratory feature audit, not the final
detector and not a claim that any scalar is universally causal.

Open the notebook:

    uv run marimo edit notebooks/feature_robustness_lab.py --no-token

The notebook reads deterministic local artifacts produced by reusable code
under `src/techjam_aigc/feature_lab/`.

## Reproduce

From the repository root:

    uv sync
    uv run pytest -q
    uv run python scripts/run_feature_lab.py --feature-profile frozen_v1 --transform-profile core --workers 4 --bootstrap 200

The bounded expanded run is opt-in:

    uv run python scripts/run_feature_lab.py --feature-profile expanded_v2 --transform-profile directed_pairs --output data/derived/feature_lab_pairs --workers 4 --bootstrap 200

A metrics-only refresh can reuse the expensive parent/transform/view cache:

    uv run python scripts/run_feature_lab.py --reuse-features --bootstrap 200

Artifacts are written under `data/derived/feature_lab/` and intentionally
ignored by Git because they derive from ignored local datasets. The run records
the input-index SHA-256 and configuration in `run_metadata.json`.

## Frozen experiment design

### Binary scope and splits

Only `real`, `fake`, and `full_synthetic` rows are admitted. SID
`tampered` rows are excluded. The loader rejects rows matching the organizer
demonstration-only COCO `val2017` / DALL-E Advanced split.

Feature directions and all linear probes are fit on discovery rows only:

- CIFAKE train;
- SID Set train; and
- WildFake reconstructed train side.

Confirmation uses CIFAKE test, SID Set validation, and WildFake's reconstructed
test side. Every transformed copy retains the source image's `parent_id`, so
variants cannot cross phases.

### Analysis views

Every transformed image is measured twice:

- `native_capped`: preserve native resolution unless the long side exceeds
  256 pixels; never upscale;
- `canonical_128`: Lanczos-resize to 128×128.

Disagreement between these views is evidence of preprocessing sensitivity.

### Transform assumptions

The implementation covers every parameter in the challenge brief:

- JPEG Q90, Q70, Q50, Q30;
- Gaussian blur sigma 0.5, 1.0, 2.0;
- Lanczos 0.5× and 0.25× down/up resizing;
- seeded RGB Gaussian noise sigma 0.02, 0.05, 0.10 on `[0, 1]`;
- deterministic brightness, contrast, and saturation factors of 0.8 and 1.2;
- centered 80% crop followed by bicubic restoration to the original size.

The brief does not define interpolation, crop post-processing, jitter sampling,
or composition order. The unchanged `core` default therefore stays bounded at
20 conditions. Opt-in profiles add 12 axial/mixed color cases, 30 directed
medium pairs, 12 realistic chains, or a deterministic 32-recipe covering bank;
the notebook never presents these as organizer rules.

### Metrics and uncertainty

A feature's direction is chosen once from pooled clean discovery data. It is
never flipped separately for a confirmation group. Reported univariate
statistics use AUPRC (implemented as average precision) with AIGC as the
positive class. Every table carries the positive prevalence because that is the
uninformative AUPRC baseline. Cross-dataset and cross-generator comparisons use
`(AUPRC - prevalence) / (1 - prevalence)`, which maps the prevalence baseline
to zero and perfect ranking to one. Directions are selected by maximum forward
or reverse discovery AUPRC, then frozen. Reports include raw and normalized
AUPRC, stratified bootstrap intervals, class counts, low-power flags, and
direction-reversal warnings relative to prevalence.

The notebook includes:

- feature-by-generator and feature-by-transformation maps;
- within-dataset metrics;
- nuisance correlations;
- clean-discovery linear family ablations;
- leave-one-dataset-out tests;
- representative high-confidence errors; and
- a fixed-rule decision ledger.

Decision labels are exploratory:

- `keep`: clears clean, powered dataset/generator, and official-transform
  screens;
- `specialist`: powered generator groups materially disagree;
- `fragile`: clean separation collapses under an official transform;
- `shortcut`: strong separation is coupled to preprocessing or registered
  nuisance variables;
- `discard`: insufficient evidence.

Underpowered generator groups remain visible and can lower decision confidence,
but they do not drive the primary classification rule.

## Feature coverage

The `frozen_v1` preregistered registry contains 53 scalars spanning spatial intensity,
color, texture, Gaussian residuals, noise behavior, Fourier magnitude and
phase, block DCT/JPEG structure, Haar wavelets, gradients, and transform
self-consistency. Five metadata measures are explicit nuisance controls.

The opt-in `expanded_v2` profile adds 29 candidates across lower bit planes,
patch distributions, multi-scale residuals, compact residual co-occurrences,
camera-pipeline proxies, richer spectra, codec/resampling history, and chroma.
Every registered scalar records its family, measurement, hypothesis, expected
failure, role, and cost. The feature profile and schema hash are stored in the
cache metadata.

The project does not automatically download a generic pretrained backbone.
Consequently the semantic-control slot is reported as `not_run` in this local
pilot. The recommended optional control is a frozen, public DINOv2 or CLIP
embedding with a discovery-only linear probe. Adding it requires an explicit
dependency and model-weight decision.

## Pilot result boundary

The current local run contains 376 eligible parent images and 15,040
parent/condition/view rows. It is sufficient to validate the laboratory but
not to establish modern-generator generalization:

- CIFAKE is 32×32 and contains one Stable Diffusion version;
- SID does not expose per-row generator identity; and
- WildFake confirmation has only 4–12 generated images per displayed model
  and 12 authentic images total.

The notebook therefore presents WildFake intervals and reversal warnings
without treating those small cells as definitive. The next data step is a
larger generator-balanced confirmation slice evaluated against the frozen
registry, not additional tuning on the current confirmation results.

## License-gated data expansion

`scripts/plan_data_expansion.py` plans deterministic SID/WildFake,
AI-GenBench, and Community Forensics slices from metadata only. It performs no
download and blocks acquisition until selected sources have complete revisions,
file lists, hashes, and allowlisted licenses:

    uv run python scripts/plan_data_expansion.py path/to/expansion-manifest.csv

The resulting selected index can be passed explicitly to the laboratory:

    uv run python scripts/run_feature_lab.py --index data/derived/data_expansion/selection.csv --feature-profile frozen_v1

The selection validator preserves parent IDs and phases, rejects edited or
demonstration-only rows, prevents discovery/final generator overlap, and can
construct equal-image-count few-generator versus many-generator cohorts.
An index carrying `source_id` is refused unless its sibling `audit.json` is
fully allowlisted for both dataset-source and underlying-image terms.
The audit is bound to the configured source revision and exact SHA-256 of
`selection.csv`, so swapping rows after review is also refused.

Final-confirmation rows are withheld by default. After every feature,
transformation, threshold, fusion rule, and hyperparameter is frozen, the first
and intended one-time evaluation is explicit:

    uv run python scripts/run_feature_lab.py --index data/derived/data_expansion/selection.csv --evaluate-final-confirmation

A successful run writes a timestamped receipt beside the index containing its
hash, output location, feature schema, and transform conditions. Repeating the
evaluation against that index is refused.

## Additional evaluation artifacts

Expanded runs emit parent-paired feature drift, clean-to-condition AUPRC loss,
severity-curve area, directed-pair interaction excess, exact reversed-order
sensitivity, and guarded chronological-confirmation tables. Empty
chronological tables remain visibly empty until a licensed final window is
assigned and sealed; they are not treated as evidence.
