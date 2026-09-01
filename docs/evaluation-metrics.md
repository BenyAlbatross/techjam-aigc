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
sigmoid. A different threshold must be selected using `val`, frozen before
`test`, and reported without retuning on test results.

The v2 benchmark tables use AIGC as positive class `1` and report the complete
schema below for every two-class dataset or slice:

- rows, positives, negatives, and positive prevalence;
- TP, TN, FP, and FN;
- accuracy, balanced accuracy, ROC-AUC, average precision (AUPRC), and
  normalized partial ROC-AUC through 5% FPR;
- precision, recall/sensitivity/TPR, specificity/TNR, F1, FPR, FNR, and
  Matthews correlation coefficient;
- predicted-positive rate, decision threshold, score scale, and pAUC limit;
- per-class logit and sigmoid-probability mean, population standard deviation,
  median, 5th percentile, and 95th percentile.

EvalGEN is positive-only. Its meaningful fields are rows/positive count, TP,
FN, recall/TPR, FNR, predicted-positive rate, threshold, and positive score
distributions. Its TN, FP, accuracy, balanced accuracy, ROC-AUC, AUPRC,
partial AUC, precision, specificity, F1, and MCC fields are written as
unavailable values; the evaluator never manufactures an authentic class.

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

## Transformed detector evaluation

The score-only evaluator above does not create transformed images. Use the
model evaluator's v2 benchmark mode for the frozen test suite:

```bash
uv run --group train python scripts/evaluate_trace_rx_m.py \
  --config configs/trace-rx-m-v2.json \
  --checkpoint artifacts/trace-rx-m-techjam2026-v2/s4_detector.pt \
  --memory artifacts/trace-rx-m-techjam2026-v2/s3_memory.pt \
  --as-is-dataset-spec configs/evaluation/techjam2026-v2-test.json \
  --uniform-chain-dataset-spec configs/evaluation/wildfake-reconstructed-test-20k.json \
  --uniform-chain-dataset-spec configs/evaluation/evalgen-positive-only.json \
  --output artifacts/evaluation/trace-rx-m-v2-test-suite
```

This mode evaluates every TechJam asset as supplied and assigns exactly one
deterministic 1--6-step transform chain to each external source. See
`trace-rx-m-v2-benchmark-evaluation.md` for the frozen assignment, slicing, and
artifact contracts. The evaluator rejects configs and checkpoints without
explicit v2 preprocessing metadata, including historical DINOv2 artifacts.

### Legacy exhaustive mode

The retained `--dataset-spec`/`--transform-profile` interface evaluates clean
images, every transformation and severity explicitly
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
- Val and test apply a deterministic exhaustive matrix so
  every source image is measured under the same conditions.
- The final training policy should include all official single transformations.
  Reserve composed journeys, operation-order changes, or additional severities
  as evaluation-only distribution shifts.
- Do not tune from test results. Freeze the model, threshold, transform profile,
  and reporting policy before evaluating that split.

The included TechJam val specification points at the existing 224-pixel
neutral BMP dataset. New datasets should point at the best licensed source-pixel
files available so transformations occur before canonical model preprocessing.
