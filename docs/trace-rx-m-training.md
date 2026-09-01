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
`facebook/dinov3-vith16plus-pretrain-lvd1689m`. Before S1, review Meta's DINOv3
licence for competition eligibility, accept access on Hugging Face, set an
immutable model commit in `backbone.revision`, and set
`backbone.license_accepted` to `true`. The command fails closed otherwise.
Tokens remain in the environment/Hugging Face credential store and must never
be committed. Detection training adapts the attention query, key, and value
projections with rank-8, alpha-16 LoRA; the remaining backbone weights stay
frozen.

For historical reproduction of the earlier local-compute run,
`configs/archive/trace-rx-m-dinov2-historical.json` pins the public Apache-2.0
`facebook/dinov2-base` checkpoint. The encoder supports both DINOv2 and
DINOv3 patch-token layouts and records the exact model ID and immutable
revision in every artifact. This archived configuration is not used by the
current DINOv3 training path, and the two backbones are not equivalent.

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
local-development modes. S4 always publishes trained weights to the
`techjam-aigc` Hugging Face organization and fails before loading the model if
`hub.repo_id` is missing or names a different owner. The default shared model
repository is `techjam-aigc/trace-rx-m`; `hub.path_prefix` keeps configurations
and runs separate within it. Hugging Face authentication comes
from the credential store or `HF_TOKEN`; no token belongs in the JSON config.
The configured repository is created if needed, with `hub.private` controlling
initial visibility.
Uploads use the official
[`huggingface_hub` file/commit APIs](https://huggingface.co/docs/huggingface_hub/guides/upload).

## TechJam 2026 dataset preparation

`data/techjam2026_v2` is the current training dataset. Its 94,490 rows already
define leakage-grouped `train`, `val`, and `test` splits. Preparation retains
every row, including every dual-data-alignment (DDA) row, and uses its supplied
binary label without special-case filtering. The fixed mapping is:

| Source split | TRACE-RX-M use | Rows |
| --- | --- | ---: |
| `train` | model fitting plus train-only internal pools | 67,418 |
| `val` | model validation only | 24,912 |
| `test` | final testing only | 2,160 |

Only real-only content groups inside `train` are reassigned to the required
authentic memory (256), capacity (256), and authentic-null (512)
pools. This leaves 66,394 rows in the detector training pool. No source group
crosses a dataset split or train-only pool. Run the preparation script with its
v2 defaults:

```bash
uv run python scripts/prepare_techjam2026_training.py
```

This writes normalized images and
`data/techjam2026_v2-normalized/training-manifest.csv`. Use `--manifest-only`
to validate the complete adapter and write the manifest without decoding the
image payloads.

The source has a class-correlated codec/geometry shortcut: all AIGC files are
PNG while many authentic files are JPEG, and their resolution distributions
differ. Preparation and native-resolution loading both limit images above a
512px short side with an aspect-preserving bicubic resize, then take a 224px
center crop; dimensions below 224px are zero-padded by
`torchvision.center_crop`. Preparation stores the result as fixed-size RGB BMP
inputs. The same deterministic spatial path is used by training and inference.
S0 audits the metadata of the inputs actually used by training rather than the
original download wrappers. The labels SHA-256, exact split and pool counts,
generator counts, and transform counts are saved beside the generated manifest. The v2
`NIL` transform sentinel becomes an explicit `clean` endpoint; transformed rows
retain their supplied transform family and variant. Rerunning preparation writes
versioned center-crop outputs, so files produced by the former square-resize
policy are not silently reused. `image_origin`, `pair_role`, and
`data_source_segment` remain in the manifest for slicing and audit, but are not
model inputs.

The local DGX Spark profile uses batches of 512 with CUDA BF16 autocast. Run a
single-step preflight when other GPU services are resident; if memory is
constrained, lower only `data.batch_size` without changing the split or
precision policy.

The manifest is validated before split or training-pool filtering. It must
contain:

`parent_id,lineage_id,split,training_pool,label,sample_kind,generator_family,generation_model,source_dataset,local_path`

`split` accepts only `train`, `val`, or `test`. `training_pool` is separate
model-stage metadata: train rows use `detector`, `memory`, `capacity`, or
`authentic_null`, while every val/test row uses `none`. Memory, capacity, and
authentic-null rows must be authentic. For v2, every
`ai_full` row—including DDA-origin rows—is mapped uniformly to `aigc` with
`sample_kind=native_aigc`. The generic manifest contract still supports an
optional paired `sample_kind=dda`, but the v2 adapter does not infer or impose
that mode. Optional `master_id`, `prompt_group_id`, and `duplicate_group_id` are
checked across dataset/train-pool partitions for leakage. Expansion manifests
with `source_id` must retain a valid acquisition `audit.json`; the organizer's
demonstration-only split and edited/composited labels are rejected.

### Current v2 settings with no effective holdout behavior

Two proposal-era fields remain in `configs/trace-rx-m-v2.json`, but they do not
provide the behavior their names suggest for the current v2 manifest:

- `data.dda_positive_share` has no effect. The v2 adapter maps every AIGC row,
  including DDA-origin rows, to `sample_kind=native_aigc`. The batch sampler
  therefore finds no `sample_kind=dda` rows and allocates zero DDA-specific
  positive slots. DDA images are still trained as ordinary AIGC positives.
- `data.held_out_transform_family` does not hold that family out of detector
  training. It only prevents the online sampler from adding another transform
  from that family and partitions `val` rows during S5. The packaged v2 `train`
  split already contains stored derivatives from every transform family; for
  the current `gaussian_noise` value, 5,618 Gaussian-noise rows remain in
  `train`. Consequently this field has no effect on whether the detector sees
  Gaussian-noise examples during training.

These fields are not challenge requirements. Treat them as inactive legacy
configuration for v2 unless the adapter and training protocol are deliberately
changed to support paired DDA sampling or a genuine transform-family holdout.

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
for LoRA and heads, per-batch BF16 autocast on CUDA (selected explicitly with
`optimizer.mixed_precision`), short warm-up, cosine decay, and
gradient clipping. Global CPU/CUDA initialization is seeded from `data.seed`.
The retrieval scores, sparse softmax, and authentic prototype memory remain
FP32 for numerical stability; BF16 does not require FP16-style loss scaling.
The S3 artifact must match the configured backbone model, immutable revision,
and manifest hash, so a same-dimensional but incompatible feature space cannot
be used silently. Batches are class-balanced; all v2 AIGC positives rotate
across generator families. The optional paired-DDA sampler and loss remain
available for other manifests that explicitly opt into `sample_kind=dda`; they
are inactive for the v2 manifest. If the one-time held-out generator gate
fails, adapters are discarded and heads are retrained with the encoder frozen;
LoRA is never retuned using that result.
The first batch of every epoch records the proposal's gradient-cosine
diagnostics for active loss terms in the checkpoint history.

Every `hub.checkpoint_every_epochs` S4 epochs, a resumable checkpoint (model,
optimizer, scheduler, epoch, history, and provenance hashes) is uploaded
synchronously to:

`<path_prefix>/runs/<wandb-run-id>/checkpoints/<variant>/epoch-NNNN.pt`

Synchronous publication is intentional: a successful S4 command means the
remote checkpoint exists, rather than merely being queued when the process
exits. Each attempted variant also retains its terminal `best_detector.pt` and
`final_detector.pt` beneath its run-scoped checkpoint directory. This includes
the LoRA weights even when the held-out-family gate selects the frozen fallback.
At the end, both run-scoped and canonical selected weights are uploaded under
the compatibility names `best_detector.pt` and `final_detector.pt` and the exact
local names `s4_best_detector.pt`, `s4_final_detector.pt`, and `s4_detector.pt`.
The bundle also contains `s3_memory.pt`, the config, and the validity report
required to reconstruct and audit the weights. W&B receives the selected
best/final model bundle. The final checkpoint means the last epoch of the
shipping variant. Frozen S1 feature caches are not model weights and are not
published; the public pretrained backbone remains referenced by its pinned Hub
ID and revision rather than being redistributed.
The best checkpoint minimizes S4 training total loss. It is deliberately **not**
selected on the held-out generator family, because the proposal permits that
family to be consulted once as a validity gate, not repeatedly for model
selection. If the gate activates the frozen fallback, the published canonical
best and final files are from that fallback; the earlier LoRA checkpoints remain
under the run history for auditability.

For S5, produce a CSV of frozen-model endpoint scores. S5 requires
`split,training_pool,logit,target,condition,transform_family,detector_sha256,lineage_id,width,height,bytes,format`
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

By default the exporter scores every `val` row plus the train-only
`authentic_null` pool needed by S5. Pass `--splits test` only after model and
reporting choices are frozen; test rows are never used for fitting.

Evaluate every frozen score table with the canonical always-report metric set
(ROC-AUC, average precision, accuracy, and balanced accuracy). The threshold is
explicit because accuracy and balanced accuracy are threshold-dependent:

```bash
uv run python scripts/evaluate_trace_rx_m_scores.py \
  --scores artifacts/trace-rx-m/scores.csv \
  --output artifacts/trace-rx-m/metrics.json \
  --score-column logit --threshold 0.0
```

See `docs/evaluation-metrics.md` for reporting and threshold-selection policy.

For exhaustive clean and transformed model evaluation across one or more
plug-in datasets, use `scripts/evaluate_trace_rx_m.py`. Dataset specifications,
transform policy, output tables, and the distinction between stochastic
training augmentation and deterministic evaluation are documented in
`docs/evaluation-metrics.md`.

After S5, run the same export command again; it detects
`s5_reliability.json`, applies the admitted availability table or passive
fallback, and adds `fused_logit` plus the reliability hash.

Every `detector_sha256` value must match `s4_detector.pt`. S5 excludes the
configured held-out transform family while fitting the table, then measures
cell d-prime on that family. If predicted and measured survival fail the
configured Spearman gate, S5 writes the proposal's L2 passive-quality logistic
stacker fallback instead of an availability table. The same fallback is used
if noise endpoints collapse into clean quality cells, or if availability does
not beat the passive stacker by the predeclared normalized-pAUC gain (default
one point; increase it to twice the measured finalist seed standard deviation
when that is larger). Fused score exports add `reliability_sha256` matching
`s5_reliability.json`; S5 verifies the originating detector hash.
Before either S5 path is admitted, the S0 nuisance models are re-run against a
median split of the adapted model's ranking; an above-gate codec, size, or
metadata cue stops the pipeline.

Then run:

```bash
uv run --group train python scripts/train_trace_rx_m.py --stage reliability \
  --config configs/trace-rx-m-v2.json --scores path/to/scores.csv \
  --output artifacts/trace-rx-m
```

S5 fits quantile cells on authentic-null descriptors, shrinks sparse cell
statistics toward global populations, computes measured d-prime survival, and
clips availability relative to clean d-prime. S5 is the final fitting stage.

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

The implementation fits model parameters only from `train`, uses `val` for
validation and reliability fitting, and leaves `test` untouched until final
evaluation. All train-only pools remain disjoint by lineage.
