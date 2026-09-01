# TRACE-RX-M v2 benchmark evaluation

## Frozen full inventories and reduced evaluation protocol

The adapters validate all 77,458 available source endpoints. The time-bounded
evaluation run deterministically samples 2,000 endpoints from each dataset for
an exact 6,000-prediction suite:

| Dataset | Policy | Full rows | Evaluated rows |
| --- | --- | ---: | ---: |
| `techjam2026-v2-test` | supplied assets as-is | 2,160 | 2,000 |
| `wildfake-reconstructed-test-20k` | one sequential chain | 20,000 | 2,000 |
| `evalgen-positive-only` | one sequential chain | 55,298 | 2,000 |

Sampling is class-proportional and source-stratified, deterministic from the
frozen seed and parent ID, and recorded in run metadata. WildFake therefore
retains 1,000 authentic and 1,000 AIGC images. EvalGEN is balanced across its
five generator models. The full inventories are always validated before the
subsets are selected.

WildFake is the local, stratified reconstruction of the authors' published
split policy, not their unpublished exact test membership. EvalGEN's published
inventory is 55,300, while both the local ZIP central directory and extracted
tree contain 55,298 image files: Flux, GoT, and Infinity have 11,060 each;
NOVA and OmniGen have 11,059 each.

TechJam receives no new transform. Its existing clean and supplied-transformed
assets are evaluated once and joined by `lineage_id` for score drift and
prediction-flip statistics.

Each WildFake and EvalGEN source receives exactly one chain of length 1--6; a
clean copy is not added. Assignment is frozen by seed and parent ID and is
independent of model scores. Within dataset, class, and generator/authentic
source stratum, length counts differ by at most one. At every active step
position, JPEG, Gaussian blur, resize, Gaussian noise, color jitter, and center
crop are similarly counterbalanced. Severity variants are counterbalanced
within family and step. Families may repeat within a chain. Operations run in
recorded order on decoded RGB source pixels, and deterministic noise uses the
parent/operation/occurrence seed policy.

## Pixel contract

All v2 training, evaluation, and inference use the config-embedded policy:

1. Decode and convert to RGB.
2. Apply the assigned external chain, if any.
3. If the short side exceeds 512 pixels, reduce it to 512 while preserving
   aspect ratio with bicubic interpolation; never upscale a smaller image.
4. Center-crop to 224×224, with symmetric zero-padding for undersized axes.
5. Convert to contiguous CHW float32 and apply ImageNet mean/std normalization.

The preparation sidecar must exactly match the training config. The evaluator
preflights the checkpoint's complete config before constructing the backbone;
missing or different preprocessing metadata and historical DINOv2 artifacts
are rejected.

## Validate inventories before inference

This command validates all three real inventories, constructs and audits every
external assignment, and decodes representative endpoints through the 224
pipeline without loading a model:

```bash
uv run python scripts/evaluate_trace_rx_m.py \
  --config configs/trace-rx-m-v2.json \
  --as-is-dataset-spec configs/evaluation/techjam2026-v2-test.json \
  --uniform-chain-dataset-spec configs/evaluation/wildfake-reconstructed-test-20k.json \
  --uniform-chain-dataset-spec configs/evaluation/evalgen-positive-only.json \
  --max-images-per-dataset 2000 \
  --validate-only \
  --output artifacts/evaluation/trace-rx-m-v2-6k-validation
```

The checked local EvalGEN archive was also extracted for visual inspection at
`data/evalgen/GenEval-JPEG/`. Evaluation uses the extracted tree because Python
detects overlapping entries in the ZIP central directory.

## Run inference

After freezing the v2 checkpoint and any validation-selected threshold:

```bash
uv run --group train python scripts/evaluate_trace_rx_m.py \
  --config configs/trace-rx-m-v2.json \
  --checkpoint artifacts/trace-rx-m-techjam2026-v2/s4_detector.pt \
  --memory artifacts/trace-rx-m-techjam2026-v2/s3_memory.pt \
  --as-is-dataset-spec configs/evaluation/techjam2026-v2-test.json \
  --uniform-chain-dataset-spec configs/evaluation/wildfake-reconstructed-test-20k.json \
  --uniform-chain-dataset-spec configs/evaluation/evalgen-positive-only.json \
  --max-images-per-dataset 2000 \
  --threshold 0.0 \
  --seed 20260831 \
  --output artifacts/evaluation/trace-rx-m-v2-6k-test-suite
```

Do not choose the threshold from these test results. Logit `0.0` is the frozen
default; an alternative must be selected from validation and supplied
explicitly.

## Statistics and artifacts

Two-class tables report counts/prevalence, TP/TN/FP/FN, accuracy, balanced
accuracy, ROC-AUC, average precision, normalized pAUC@5% FPR, precision,
recall/sensitivity/TPR, specificity/TNR, F1, FPR, FNR, MCC,
predicted-positive rate, threshold, and score scale. `score_distributions.csv`
adds per-class logit/probability mean, population standard deviation, median,
p05, and p95. Positive-only EvalGEN tables expose only meaningful positive
statistics and mark all two-class statistics unavailable.

Metrics are written overall and by endpoint class, chain length, TechJam
supplied transform family/variant, generator family/model, and authentic
source. TechJam also produces lineage-paired score drift and flip rates.
False-positive and false-negative detail tables retain the endpoint recipe.
`evaluation_report.md` provides the human-readable inventory, overall metric,
chain-length, TechJam transform-family, and paired-drift summary.

`predictions.csv` and challenge-compatible `predictions.json` contain at least
`image_path` and sigmoid AIGC confidence `pred`. Assignment CSVs persist every
recipe and SHA-256 plus expanded step rows. Audit tables count lengths, step
positions, families, severities, and repeated-family frequency. Run metadata
records config/data/checkpoint hashes, preprocessing policy, seed, threshold,
inventories, elapsed time, and throughput. No transformed images are written.
