# TRACE-RX-Parallel training

TRACE-RX-Parallel replaces the sequential TRACE-RX-M detector head with two
parallel branches and learned late fusion:

```text
                         +-> global token statistics -> global classifier ---+
image -> DINOv3 patches -+                                                   +-> fused AIGC logit
                         +-> authentic memory -> residual classifier --------+
```

The branches share one DINOv3/LoRA patch encoder to keep training and inference
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

Start from `configs/trace-rx-parallel.json`. Before feature extraction, review
the gated DINOv3 licence, accept model access on Hugging Face, set
`backbone.license_accepted` to `true`, and pin `backbone.revision` to an
immutable commit. Set `data.held_out_generator_family` before detection
training and configure `hub.repo_id` as `owner/name`. Credentials stay in the
Hugging Face and W&B credential stores; do not place tokens in the config.

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
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage cache --config configs/trace-rx-parallel.json \
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage capacity --config configs/trace-rx-parallel.json \
  --output artifacts/trace-rx-parallel

uv run --group train python scripts/train_trace_rx_parallel.py \
  --stage detection --config configs/trace-rx-parallel.json \
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-parallel
```

Export endpoint scores for S5/S6:

```bash
uv run --group train python scripts/export_trace_rx_parallel_scores.py \
  --config configs/trace-rx-parallel.json \
  --manifest path/to/training-manifest.csv \
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
