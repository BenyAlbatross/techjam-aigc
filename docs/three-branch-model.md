# Three-branch model

## Status

The three-branch detector is an updated model implemented independently under
`src/techjam_aigc/three_branch`. It does not replace or mutate the existing
TRACE-RX-M implementation or its checkpoint format.

The configured final run uses every one of the 67,418 rows in the
`techjam2026_v2` **train** split. This includes rows assigned to the historical
`detector`, `memory`, `capacity`, and `authentic_null` train-only pools. No
generator family is excluded. The supplied `val` and `test` partitions remain
untouched because using them for gradient updates would violate their dataset
roles; neither is used for checkpoint selection. The deliverable is the final
checkpoint after exactly ten epochs.

## Architecture

The detector exposes three independently supervised image logits:

1. **Global branch.** A pinned DINOv3 ViT-H/16+ encoder receives one 224-pixel
   canonical view. Rank-8 LoRA adapters update query, key, and value
   projections; base weights remain frozen.
2. **Class-relative memory branch.** Separate 256-prototype authentic and AIGC
   dictionaries are fitted with streaming, group-weighted mini-batch k-means
   over tokens from every training image. For each patch, the head receives
   signed residuals to both dictionaries, both raw distances, their relative
   distance, maximum similarities, and normalized retrieval entropies. Patch
   logits are pooled using mean, dispersion, top-20% mean, and maximum.
3. **Native forensic branch.** Five source-pixel crops (four corners and
   center) are taken without resizing. A fixed Gaussian high-pass residual is
   concatenated with RGB and processed by a lightweight CNN. Crop logits use
   the same mean/dispersion/tail/max pooling.

Fusion begins as the equal average of the three branch logits. A small residual
fusion head sees the logits and their pairwise disagreements. Its last layer is
zero-initialized so it cannot suppress an untrained branch at initialization.
Every branch also receives an auxiliary BCE term.

The dual memories are frozen during detection training. Memory retrieval uses
cosine similarity with an explicit temperature rather than the historical
`sqrt(D)` scaling, which was nearly uniform for normalized vectors.

## Training contract

The authoritative configuration is `configs/three-branch.json`. The pipeline:

- validates and joins the normalized manifest to the original source images;
- fails if any train row, train-only pool, source image, or label is missing;
- fits both memories from every train image exactly once;
- shuffles but emits every train row exactly once in each epoch;
- asserts parent-ID coverage and uniqueness after memory fitting and every
  training epoch;
- uses group-balanced loss weights without resampling or omitting examples;
- writes atomic, resumable checkpoints after every epoch;
- never loads validation/test rows and has no holdout-family option; and
- publishes `three-branch-final.pt` only after epoch 10.

Run the data-only contract check:

```bash
uv run python scripts/train_three_branch.py \
  --stage preflight \
  --config configs/three-branch.json \
  --output artifacts/three-branch-techjam2026-v2
```

Fit memories and train all ten epochs:

```bash
uv run --group train python scripts/train_three_branch.py \
  --stage all \
  --config configs/three-branch.json \
  --output artifacts/three-branch-techjam2026-v2 \
  --device cuda
```

If a run stops after a completed epoch, resume without repeating completed
epochs:

```bash
uv run --group train python scripts/train_three_branch.py \
  --stage train \
  --config configs/three-branch.json \
  --output artifacts/three-branch-techjam2026-v2 \
  --resume artifacts/three-branch-techjam2026-v2/latest.pt \
  --device cuda
```

## Artifacts

- `run-contract.json`: exact row counts, pool counts, hashes, and the explicit
  no-holdout/no-validation contract.
- `dual-memory.pt`: both prototype banks plus full-train coverage provenance.
- `checkpoints/epoch-NNNN.pt`: resumable optimizer/scheduler and trainable
  model state.
- `training-history.json`: epoch-level loss and fused/branch training metrics.
- `three-branch-final.pt`: lightweight final epoch-10 trainable state. Frozen
  public backbone weights are referenced by pinned model ID and revision.

Training metrics are diagnostics only. External test results must be produced
after training without modifying the final checkpoint.

Directory inference uses the challenge-required `image_path`/`pred` JSON
records:

```bash
uv run --group train python scripts/infer_three_branch.py \
  --image-dir path/to/images \
  --output predictions.json \
  --device cuda
```
