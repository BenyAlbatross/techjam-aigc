# Detector evaluation metrics

## Always-report set

Every frozen TRACE-RX-M detector evaluation must report all four metrics below
on the same rows:

| Metric | Purpose |
| --- | --- |
| ROC-AUC | Threshold-free ranking quality across authentic and AIGC images |
| Average precision | Precision-recall ranking quality, reported with class prevalence |
| Accuracy | Fraction correct at the declared decision threshold |
| Balanced accuracy (BAcc) | Mean of authentic recall and AIGC recall at that threshold |

BAcc is mandatory because evaluation slices can have different real/AIGC class
ratios. It prevents the majority class from dominating accuracy and makes
asymmetric real-versus-AIGC behavior visible. It does not replace ROC-AUC or
average precision because it depends on a decision threshold.

Every report must record the threshold and its score scale. For uncalibrated
logits, the default is logit `0.0`, equivalent to probability `0.5` after a
sigmoid. A different threshold must be selected using only an allowed
development or calibration split, frozen before locked evaluation, and reported
without retuning on locked results.

Run the canonical evaluator with:

```bash
uv run python scripts/evaluate_trace_rx_m_scores.py \
  --scores artifacts/trace-rx-m/scores.csv \
  --output artifacts/trace-rx-m/metrics.json \
  --score-column logit --threshold 0.0
```

Apply the always-report set to the overall split and to every sufficiently
powered generator, authentic-source, and transformation slice. Also report row
counts and positive/negative counts. For headline robustness claims, add clean
to transformed changes and the worst slice; do not substitute a pooled average
for subgroup results.

## Current shipping checkpoint

The existing clean-only run predates the canonical evaluator. At its
uncalibrated logit threshold of `0.0`, balanced accuracy is `0.88515` on the
6,091-row development split and `0.63900` on the 960-row locked split. These
values supplement, rather than replace, the ROC-AUC, average precision, and
accuracy already recorded in `techjam2026-training-run.md`.

## Transformed detector evaluation

The score-only evaluator above does not create transformed images. Use the
model evaluator for challenge robustness testing:

```bash
uv run --group train python scripts/evaluate_trace_rx_m.py \
  --config configs/trace-rx-m-techjam2026-local.json \
  --checkpoint artifacts/trace-rx-m-techjam2026/s4_detector.pt \
  --memory artifacts/trace-rx-m-techjam2026/s3_memory.pt \
  --dataset-spec configs/evaluation/techjam2026-development.json \
  --transform-profile core \
  --output artifacts/evaluation/trace-rx-m-techjam2026-development
```

`core` evaluates clean images, every transformation and severity explicitly
listed in the challenge, and four predeclared compositions. Pass
`--official-only` to omit the compositions. Additional profiles such as
`realistic`, `directed_pairs`, and `covering32` may be repeated with
`--transform-profile`; profiles are unioned and duplicate conditions are run
only once. `--condition` provides an exact allowlist for focused smoke tests.

The evaluator transforms decoded source pixels lazily and then applies the
model's canonical resize and normalization. It never writes transformed copies
or changes the source dataset. Noise is deterministic by dataset, parent,
operation, and seed. Every transformed endpoint retains the source
`parent_id`, enabling paired clean-to-condition drift analysis.

The output directory contains endpoint predictions, the exact transform
registry, metrics by dataset/condition/family/generator/authentic source,
clean-to-condition drops, paired score and correctness drift, a compact
worst-condition summary, and hashes for the checkpoint, memory, model config,
and dataset specifications.

### Dataset plug-in contract

Evaluation datasets are added through JSON specifications. A CSV-backed
dataset maps its source columns to the canonical schema:

```json
{
  "schema_version": 1,
  "dataset_id": "example-test",
  "adapter": "csv",
  "manifest": "data/example/labels.csv",
  "root": "data/example",
  "columns": {
    "image_path": "path",
    "target": "label",
    "parent_id": "image_id",
    "generator_family": "generator",
    "source_dataset": "real_source"
  },
  "filters": {
    "split": ["test"]
  },
  "label_map": {
    "real": 0,
    "generated": 1
  }
}
```

Only `image_path` and `target` mappings are required. Missing parent IDs are
derived deterministically from the dataset ID and path. Generator and authentic
source fields are optional, but should be supplied when available so subgroup
metrics remain meaningful. Paths in specifications are resolved relative to
`--repo-root`.

For a directory organized by class, no manifest is needed:

```json
{
  "schema_version": 1,
  "dataset_id": "example-folders",
  "adapter": "class_folders",
  "root": "data/example-folders",
  "classes": {
    "real": 0,
    "aigc": 1
  },
  "class_metadata": {
    "aigc": {
      "generator_family": "example-generator"
    }
  }
}
```

Repeat `--dataset-spec` to evaluate multiple datasets in one run. Dataset IDs
must be unique.

## Train/evaluation transform policy

Split immutable source images and their lineages before applying any
transformation. All endpoints derived from one source image remain in that
source's split.

- Training samples transformations stochastically and symmetrically across
  authentic and AIGC labels.
- Development and locked evaluation apply a deterministic exhaustive matrix so
  every source image is measured under the same conditions.
- The final training policy should include all official single transformations.
  Reserve composed journeys, operation-order changes, or additional severities
  as evaluation-only distribution shifts.
- Do not tune from locked transformed results. Freeze the model, threshold,
  transform profile, and reporting policy before opening that split.

The included TechJam development specification points at the existing 224-pixel
neutral BMP dataset. New datasets should point at the best licensed source-pixel
files available so transformations occur before canonical model preprocessing.
