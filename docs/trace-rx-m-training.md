# TRACE-RX-M v2 PyTorch training

This package implements the six-stage method in
`docs/method_proposal/TRACE-RX-M_v2_architecture_and_method_final.pdf`. The
proposal is a design report rather than a complete executable recipe, so every
unspecified numerical choice is visible in `configs/trace-rx-m-v2.json`.

## Safety and prerequisites

Install the optional training stack:

```bash
uv sync --group train
```

The operating backbone is the gated
`facebook/dinov3-vitl16-pretrain-lvd1689m`. Before S1, review Meta's DINOv3
licence for competition eligibility, accept access on Hugging Face, set an
immutable model commit in `backbone.revision`, and set
`backbone.license_accepted` to `true`. The command fails closed otherwise.
Tokens remain in the environment/Hugging Face credential store and must never
be committed.

The two trainable stages use [Weights & Biases](https://docs.wandb.ai/models/ref/python/functions/init).
S2/S3 logs every memory-candidate
loss and the selected coverage result; S4 logs epoch losses, authentic-subtype
losses, gradient-conflict diagnostics, learning rates, the one-time held-out
family result, and best/final model artifacts. Authenticate before training:

```bash
wandb login
hf auth login
hf auth whoami
```

Set `tracking.wandb_project` (and optionally `tracking.wandb_entity`) in the
config. `online` is the release default; `offline` and `disabled` are explicit
local-development modes. Set `hub.repo_id` to an `owner/name` model repository
before S4. S4 fails closed if it is missing. Hugging Face authentication comes
from the credential store or `HF_TOKEN`; no token belongs in the JSON config.
The configured repository is created if needed, with `hub.private` controlling
initial visibility.
Uploads use the official
[`huggingface_hub` file/commit APIs](https://huggingface.co/docs/huggingface_hub/guides/upload).

The manifest is validated before role filtering. It must contain:

`parent_id,lineage_id,role,phase,label,sample_kind,generator_family,generation_model,source_dataset,local_path`

Roles are `memory_pool`, `capacity_validation`, `supervised`,
`authentic_null`, `development`, `calibration`, and `locked_evaluation`.
Memory, capacity, and authentic-null rows must be authentic. DDA rows require
`source_parent_id`, share a lineage with that authentic source, and remain in
the supervised role. Optional `master_id`, `prompt_group_id`, and
`duplicate_group_id` are checked for cross-role leakage. Expansion manifests
with `source_id` must retain a valid acquisition `audit.json`; the organizer's
demonstration-only split and edited/composited labels are rejected.

Set `data.held_out_generator_family` before S4. S0 warns when fewer than eight
generator families are available, as required by the proposal.
S0 also requires `width,height,bytes,format` and runs lineage-grouped,
out-of-fold nuisance-only probes. The default gate rejects any metadata,
dimension, or codec ROC-AUC above 0.55. S1 verifies that this passed artifact
belongs to the same manifest before loading the backbone.

## Ordered stages

Run each command against the same output directory:

```bash
uv run --group train python scripts/train_trace_rx_m.py \
  --stage protocol --config configs/trace-rx-m-v2.json \
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-m

uv run --group train python scripts/train_trace_rx_m.py \
  --stage cache --config configs/trace-rx-m-v2.json \
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-m

uv run --group train python scripts/train_trace_rx_m.py \
  --stage capacity --config configs/trace-rx-m-v2.json \
  --output artifacts/trace-rx-m

uv run --group train python scripts/train_trace_rx_m.py \
  --stage detection --config configs/trace-rx-m-v2.json \
  --manifest path/to/training-manifest.csv --output artifacts/trace-rx-m
```

S1 uses a zero-rank, fully frozen encoder and caches BF16 patch tokens after
removing CLS/register tokens. It repeats a held-out forward and refuses the
cache if BF16 round-trip error exceeds the recorded gate tolerance. S2
initializes each authentic memory candidate
with k-means, normalizes tokens/prototypes, and performs chunked global top-k
retrieval with FP32 scores and entropy. Candidates with a dead prototype or a
prototype exceeding the configured maximum retrieval share are rejected, and
the complete usage histogram is reported. S3 selects the smallest candidate near
the best worst-subtype authentic-tail coverage. If no absolute tail threshold
is configured, the first candidate defines one common quantile threshold which
is reused for all candidates. If worst-subtype tail coverage is still improving
beyond the configured relative tolerance at the largest tested capacity, the
report records the proposal's undersizing warning and expected worst-subtype
false-positive direction.

S4 permanently freezes the selected memory. AdamW uses separate learning rates
for LoRA and heads, BF16 autocast on CUDA, short warm-up, cosine decay, and
gradient clipping. Global CPU/CUDA initialization is seeded from `data.seed`.
The S3 artifact must match the configured backbone model, immutable revision,
and manifest hash, so a same-dimensional but incompatible feature space cannot
be used silently. Batches are class-balanced; native positives rotate across
generator families. DDA defaults to 20% of positive slots and, following the
stricter prose interpretation, contributes only through paired ranking unless
`dda_in_primary_objective` is enabled; its co-batched real source is likewise
reserved for that pair term so forced pairing cannot overweight one authentic
master in BCE. If the one-time held-out generator gate
fails, adapters are discarded and heads are retrained with the encoder frozen;
LoRA is never retuned using that result.
The first batch of every epoch records the proposal's pairwise gradient-cosine
diagnostics for BCE, pAUC, and DDA ranking in the checkpoint history.

Every `hub.checkpoint_every_epochs` S4 epochs, a resumable checkpoint (model,
optimizer, scheduler, epoch, history, and provenance hashes) is uploaded
synchronously to:

`<path_prefix>/runs/<wandb-run-id>/checkpoints/<variant>/epoch-NNNN.pt`

Synchronous publication is intentional: a successful S4 command means the
remote checkpoint exists, rather than merely being queued when the process
exits. At the end, both run-scoped and canonical `best_detector.pt` and
`final_detector.pt` files are uploaded, together with `s3_memory.pt`, the config,
and validity report required to reconstruct and audit them. W&B receives the
same best/final model bundle. The final checkpoint means the last epoch of the
shipping variant.
The best checkpoint minimizes S4 training total loss. It is deliberately **not**
selected on the held-out generator family, because the proposal permits that
family to be consulted once as a validity gate, not repeatedly for model
selection. If the gate activates the frozen fallback, the published canonical
best and final files are from that fallback; the earlier LoRA checkpoints remain
under the run history for auditability.

For S5/S6, produce a CSV of frozen-model endpoint scores. S5 requires
`role,logit,target,condition,transform_family,detector_sha256,lineage_id,width,height,bytes,format`
plus:

`log_min_dimension,noise_sigma,blockiness,structural_hf_energy`

The exporter computes these columns from original endpoint pixels and records
the checkpoint hash (the manifest must carry `condition` and
`transform_family`):

```bash
uv run --group train python scripts/export_trace_rx_m_scores.py \
  --config configs/trace-rx-m-v2.json \
  --manifest path/to/training-manifest.csv \
  --artifacts artifacts/trace-rx-m --output artifacts/trace-rx-m/scores.csv
```

After S5, run the same command again; it detects `s5_reliability.json`, applies
the admitted availability table or passive fallback, and adds `fused_logit`
plus the reliability hash for S6.

Every `detector_sha256` value must match `s4_detector.pt`. S5 excludes the
configured held-out transform family while fitting the table, then measures
cell d-prime on that family. If predicted and measured survival fail the
configured Spearman gate, S5 writes the proposal's L2 passive-quality logistic
stacker fallback instead of an availability table. The same fallback is used
if noise endpoints collapse into clean quality cells, or if availability does
not beat the passive stacker by the predeclared normalized-pAUC gain (default
one point; increase it to twice the measured finalist seed standard deviation
when that is larger). For S6, add `fused_logit` and `reliability_sha256`
matching `s5_reliability.json`; both S5 and S6 also verify the originating
detector hash.
Before either S5 path is admitted, the S0 nuisance models are re-run against a
median split of the adapted model's ranking; an above-gate codec, size, or
metadata cue stops the pipeline.

Then run:

```bash
uv run --group train python scripts/train_trace_rx_m.py --stage reliability \
  --config configs/trace-rx-m-v2.json --scores path/to/scores.csv \
  --output artifacts/trace-rx-m

uv run --group train python scripts/train_trace_rx_m.py --stage calibration \
  --config configs/trace-rx-m-v2.json --scores path/to/scores-with-fused-logit.csv \
  --output artifacts/trace-rx-m
```

S5 fits quantile cells on authentic-null descriptors, shrinks sparse cell
statistics toward global populations, computes measured d-prime survival, and
clips availability relative to clean d-prime. S6 fits one positive temperature
on calibration-only fused logits.

## Explicit interpretations of underspecified text

- Residual pooling concatenates signed per-dimension mean, standard deviation,
  and 95th percentile (`3D`) before projection. This preserves page 4 even
  though page 3 calls the result a `D`-vector.
- `s_max` is maximum sparse-attention probability per patch; `s_ent` is sparse
  attention entropy. Both are mean-aggregated over patches.
- The undefined `z_tilde` is authentic-cell standardization:
  `(z - mu_real[cell]) / sigma_real[cell]`. It is isolated in
  `ReliabilityTable.normalize_authentic` for easy replacement.
- Wavelet-MAD uses first-level Haar diagonal detail; JPEG blockiness is the
  normalized excess discontinuity at 8-pixel boundaries; structural HF energy
  uses a 0.25 cycles/pixel radial cutoff and subtracts the noise floor. These
  are measurable assumptions, not claims that the PDF prescribed them.
- Head widths, LoRA settings, loss weights/margin, learning rates, epochs,
  memory grids, clean probability, and quality bins are configurable defaults.

## How the memory relates to MIRROR

The comparison target is
[MIRROR `models/mirror.py` at commit `18c56efa`](https://github.com/handsome-rich/MIRROR/blob/18c56efa303d96f038a6aac4e11ba1a512a2cbde/models/mirror.py).
The common core is the same: retrieve the top-k authentic prototypes, softmax
their scores, reconstruct an authentic reference, compute maximum attention and
entropy, and classify what the reference failed to explain.

TRACE-RX-M differs where its own proposal is explicit:

- **Only the prototypes are learned in S2.** MIRROR's code adds learned
  multi-head `q_proj`, `k_proj`, `v_proj`, and `out_proj` layers. The TRACE-RX-M
  trainable-parameter ledger says S2 trains `M` only, so this implementation uses
  normalized tokens/prototypes directly and has no trainable memory projections.
- **The memory is frozen but retrieval is differentiable in S4.** MIRROR wraps
  retrieval in `torch.no_grad()`. Here prototype gradients are disabled, while
  query-dependent retrieval remains in the graph. This lets LoRA receive the
  reconstruction and retrieval-statistic signal without allowing the authentic
  dictionary to chase the adapted encoder.
- **The residual keeps direction and tail behavior.** MIRROR mean-pools the
  residual and also concatenates the CLS feature. TRACE-RX-M excludes CLS and
  concatenates signed per-channel mean, standard deviation, and upper quantile,
  because the proposal requires reference-only evidence and H2 aggregation.
- **Retrieval is chunked and gathered exactly.** MIRROR materializes dense
  attention and masks it for clarity. TRACE-RX-M scans prototypes in chunks and
  gathers only the winning values, keeping FP32 scoring/entropy without a
  `[batch, heads, patches, memory]` allocation.
- **Scaling follows the proposal's single-head formula.** MIRROR scales each
  learned head by `sqrt(head_dim)`; TRACE-RX-M has no learned heads and uses the
  report's `sqrt(D)` scaling.

In simple terms: it keeps MIRROR's idea of asking “can authentic memory explain
this patch?”, but removes parts that conflict with TRACE-RX-M's frozen-memory
ledger and adds the residual/reliability details that are this project's stated
contribution.

The implementation never trains on `locked_evaluation`; calibration,
reliability, capacity, and supervised roles remain disjoint by lineage.
