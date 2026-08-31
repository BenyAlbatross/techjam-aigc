# TRACE-RX-Parallel training

TRACE-RX-Parallel replaces the sequential TRACE-RX-M detector head with two
parallel branches and learned late fusion:

```text
                         +-> global token statistics -> global classifier ---+
image -> DINOv2 patches -+                                                   +-> fused AIGC logit
                         +-> authentic memory -> residual classifier --------+
```

The branches share one DINOv2/LoRA patch encoder to keep training and inference
within hackathon-scale compute. The global branch sees normalized patch tokens
directly and captures image-wide evidence. The memory branch independently
retrieves frozen authentic prototypes and classifies the directional residual
plus retrieval confidence/entropy. A trainable two-input linear layer fuses the
two branch logits. It is initialized to an equal average (`0.5`, `0.5`) so both
branches contribute from the first update; its learned weights are exported for
inspection.

This is an implementation choice, not a requirement from the challenge brief.
The target remains binary pure-AIGC versus authentic classification. The model
does not use watermarks, the demonstration-only split, or edited/composited
images for training.

## Setup and configuration

Install the training dependencies:

```bash
uv sync --group train
```

`configs/trace-rx-parallel.json` pins the public Apache-2.0
`facebook/dinov2-base` revision and targets its `query` and `value` projections
with LoRA. It also configures online W&B tracking and the private model repo
`Joshyxwa/trace-rx-parallel-techjam2026`. Change `hub.repo_id` if the active
Hugging Face account does not own that namespace. Credentials stay in the
Hugging Face and W&B credential stores; do not place tokens in the config.

Authenticate before S4:

```bash
wandb login
hf auth login
hf auth whoami
```

## TechJam 2026 dataset and split policy

The preparation command downloads the immutable dataset revision, excludes
locked pixels by default, and normalizes train/dev/calibration images to RGB
224x224 uncompressed BMP. Uniform decoding prevents dimensions, codec, and byte
count from identifying the class. Passing the acknowledgement flag means the
operator reviewed the dataset card and its row-level licence metadata; the
generated audit reports rows whose recorded licence still contains `pending`.

```bash
uv run --group train python scripts/prepare_techjam2026_parallel.py \
  --acknowledge-dataset-terms
```

The command writes
`data/processed/techjam2026-parallel/training-manifest.csv` and its adjacent
summary. The fixed source splits are used as follows:

| Source split | Use |
| --- | --- |
| `train` | Supervised fitting; real-only content groups also supply memory pool, capacity validation, and authentic-null roles. |
| `dev` | Non-Gemini groups select the best epoch; a group-disjoint Gemini-vs-real subset is consulted once as the generator gate. |
| `calibration` | Reliability and probability calibration only. |
| `own_locked` | Never downloaded by default and never used for fitting, selection, fallback decisions, or calibration. |

`gemini_flash_image` is predeclared as the held-out generator because the
dataset's locked positives are Gemini. Content groups never cross internal
roles or the two dev purposes. The supplied dataset has no DDA rows, so the
balanced sampler automatically uses equal authentic/native-AIGC batches and
does not require artificial DDA examples.

The data roles, leakage checks, authentic-memory capacity selection, balanced
sampling, DDA pair ranking, one-time held-out-family gate, nuisance probes,
reliability fitting, and calibration protocol are unchanged from the
[TRACE-RX-M training contract](trace-rx-m-training.md). This variant deliberately
reuses those audited components; only the S4 detector architecture and its
checkpoint loader differ.

## Commands

Run all stages against one artifact directory:

```bash
uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage protocol --config configs/trace-rx-parallel.json \
  --manifest data/processed/techjam2026-parallel/training-manifest.csv \
  --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage cache --config configs/trace-rx-parallel.json \
  --manifest data/processed/techjam2026-parallel/training-manifest.csv \
  --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage capacity --config configs/trace-rx-parallel.json \
  --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage detection --config configs/trace-rx-parallel.json \
  --manifest data/processed/techjam2026-parallel/training-manifest.csv \
  --output artifacts/trace-rx-parallel
```

Export endpoint scores for S5/S6:

```bash
uv run --group train python scripts/export_trace_rx_parallel_scores.py \
  --config configs/trace-rx-parallel.json \
  --manifest data/processed/techjam2026-parallel/training-manifest.csv \
  --artifacts artifacts/trace-rx-parallel \
  --output artifacts/trace-rx-parallel/scores.csv
```

The export retains the pipeline's `logit` and adds `global_logit`,
`memory_logit`, `fusion_global_weight`, and `fusion_memory_weight`. After the
first reliability run, repeat the export to add `fused_logit`, then run the
calibration stage exactly as documented in the base training contract.

S4 checkpoints carry `architecture: trace-rx-parallel`; the loader rejects
sequential TRACE-RX-M checkpoints instead of silently accepting incompatible
head weights. The authentic memory stays frozen during S4, both branch heads
and fusion are trained at the configured head learning rate, and LoRA uses its
separate lower learning rate. If the held-out-family validity gate fails, the
same predeclared frozen-encoder fallback retrains all three head components.

Every epoch is synchronously uploaded to the configured Hugging Face model
repo, and every improvement immediately updates the canonical and run-scoped
best checkpoint. At successful S4 completion, one atomic commit publishes both
canonical files:

```text
trace-rx-parallel-techjam2026/best_detector.pt
trace-rx-parallel-techjam2026/final_detector.pt
```

The same files are retained below the W&B run ID, along with the selected
memory, config, split audit, manifest, and validity report. Missing or empty
files fail the upload rather than producing a nominally successful run.

W&B records training losses, authentic-subtype losses, gradient conflicts,
learning rates, global/memory/fused dev AUC, one-time held-out AUC, fusion
weights and bias, the exact manifest as a dataset artifact, and best/final
weights as a model artifact. `best_detector.pt` maximizes fused ROC-AUC on the
non-held-out dev groups; `final_detector.pt` is the last epoch.
