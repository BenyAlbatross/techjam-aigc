# AIGC Detection under Transformations — Experiment Plan

Status: **B1 and B3 resolved; blocked only on B2 (dataset location)** (see §0). Written 2026-08-30.

---

## 0. Blockers / open questions

**B1 — RESOLVED (2026-08-30).** The official "Robustness transformations" spec has landed; the
grid in §2.1 is now the real one, not a placeholder. One genuine ambiguity remains *in the spec
itself*: it states "more than one transformation may potentially be applied to an image; the supplied
materials do not specify the exact composition policy." We therefore do not guess a policy — we
build both single and composed conditions and treat composition as an experimental axis (§2.3).

**B2 — Dataset location.** Pending from you. Needed: path/repo, and whether reals and AIGC are
already split, and what generator/source metadata exists per image (needed for grouped splits and
per-generator breakdowns).

**B3 — RESOLVED (2026-08-30).** Confirmed by the user: the target is the **training** data
composition, not the test set. §4 is the search over training mixtures. The eval-composition
sensitivity analysis (§4.4) is kept anyway — it is nearly free (a weighted recomputation over fixed
predictions, no retraining) and it quantifies how much the headline number depends on how the
eval set happened to be composed, which we need in order to report the headline number honestly.

---

## 1. Environment (verified)

| | |
|---|---|
| Machine | DGX Spark — NVIDIA GB10 (Blackwell, sm_121), aarch64, 121 GB unified memory, 3.5 TB free |
| Python | 3.11.16 in `.venv` (uv-managed) |
| Torch | **not installed** — needs the `cu130` aarch64 wheels |
| HF auth | `joelleoqiyi`, member of `techjam-aigc` |

**DINOv3 access — resolved.** `facebook/dinov3-*` is gated (`403 GatedRepoError` confirmed on this
account). The `timm/*` repos carry the same official weights and are **not** gated — verified
readable:

- `timm/vit_small_patch16_dinov3.lvd1689m` (384-d)
- `timm/vit_base_patch16_dinov3.lvd1689m` (768-d) ← default
- `timm/vit_large_patch16_dinov3.lvd1689m` (1024-d) ← ceiling run

So we go through `timm`, not `transformers`. This is also the better path: `timm` gives clean
variable-input-resolution support and `forward_intermediates()` for mid-block feature taps, both of
which the hybrid model in §3 needs. Still worth requesting `facebook/dinov3-*` access in parallel.

---

## 2. Transform pipeline

### 2.1 Official spec (confirmed)

Source: the "Robustness transformations" sheet. Evaluation considers a **subset** of these.

| family | official parameters | conditions | real-world analogue |
|---|---|---|---|
| `jpeg` | quality 90, 70, 50, 30 | 4 | social-media re-encoding, messaging |
| `blur` | Gaussian kernel sigma 0.5, 1.0, 2.0 | 3 | out-of-focus images |
| `resize` | scale to 0.5x or 0.25x, **then upscale back** | 2 | thumbnail generation |
| `noise` | Gaussian sigma 0.02, 0.05, 0.10 | 3 | low-light sensor noise |
| `color` | brightness, contrast, saturation +/-20% | 1 | filter apps, auto-enhancement |
| `crop` | centre crop, retain 80% | 1 | profile-picture cropping, reframing |
| `clean` | identity | 1 | control cell |

**15 single-transform conditions total.** Two readings had to be pinned down, both recorded here so
downstream numbers are interpretable:

- **Noise sigma is in normalised [0,1] units**, not 8-bit levels — 0.02/0.05/0.10 correspond to
  5.1/12.75/25.5 on an 8-bit scale. Read as 8-bit these would be imperceptible; read as [0,1] they
  span "barely visible" to "clearly grainy", which matches the stated analogue. Noise is applied in
  float32 before the single uint8 quantisation.
- **Colour jitter +/-20% is a *range*, not a fixed offset.** Brightness, contrast and saturation
  factors are each drawn from U(0.8, 1.2) under the per-image derived seed, so the condition is
  reproducible per image but not a constant shift. Op order is fixed (brightness -> contrast ->
  saturation) and recorded, since these do not commute.

Two operating points from the prior feature work — `0.25x` resize and `blur sigma=1` — appear in this
grid, which independently confirms the spec matches the earlier experiments.

**Note the resize family upscales back.** The forensic content is the resample round-trip damage, not
a size change, so `resize` does **not** alter model input dimensions and is not confounded with the
resolution shortcut audited in §5.

### 2.2 Implementation rules

- **Deterministic.** Transform id + severity + a per-image seed derived from `sha1(image_id)`, so
  the transformed dataset is byte-reproducible and re-derivable from the manifest alone.
- **Materialised, not on-the-fly.** Write the transformed set to parquet once. On-the-fly random
  augmentation would make the per-transform metric breakdown non-reproducible across runs.
- **Manifest per row:** `image_id, source_image_id, label, source/generator, transform, severity,
  seed, original (w,h)`. `source_image_id` is what the grouped split keys on.
- **Order matters and gets recorded.** Resize-then-JPEG and JPEG-then-resize are different forensic
  problems, so every applied chain is recorded verbatim in the manifest.
- **Bit-depth / colour discipline.** Noise and colour ops in float32, single quantisation to uint8
  at the end, so we don't accidentally measure our own rounding.

---

### 2.3 Composition policy — an experimental axis, not a guess

The spec explicitly declines to define how transforms compose, so we cannot match it by assumption.
We build three tiers and let the results say which matters:

- **T1 — single.** Exactly one transform per image, all 15 conditions. This is the *only* tier that
  supports clean per-family attribution, so all per-transform robustness tables come from it.
- **T2 — pairs.** Two transforms from different families, in a fixed canonical order that mirrors a
  plausible real pipeline (geometric -> photometric -> noise -> compression: crop/resize -> color ->
  blur/noise -> jpeg). JPEG last matches how images actually reach a platform.
- **T3 — random chains.** 1-3 transforms, families and severities sampled under the per-image seed.
  This is the closest stand-in for an unknown evaluation policy.

**Robustness is reported against the worst tier, not the average.** If the hidden evaluation composes
transforms and we only tuned on T1, a T1-only number would be optimistic — the guard against that is
reporting T3 alongside, always.

## 3. Models

### 3.1 Experiment 1 — pure DINOv3 baseline

Frozen backbone, no finetuning. This is the honest "what does DINOv3 already know" number.

- Features: CLS token, mean-pooled patch tokens, and `[CLS ; meanpool]` concat — pick by validation.
- Head: L2-regularised logistic regression (primary; convex, no seed variance) and a 2-layer MLP.
- **Input resolution is a first-class variable, not a default.** DINOv3 at 224 px throws away almost
  every high-frequency forensic cue — and high-frequency cues are the entire premise of the three
  handcrafted features. Sweep **224 / 336 / 512** (patch-16 handles all three) and report the curve.
  I expect this sweep to matter more than the head choice.
- Backbone sweep: ViT-S -> ViT-B -> ViT-L, to see whether capacity or resolution is the binding
  constraint.
- Also run finetuned variants as a second baseline tier: last-block-only, then LoRA over attention
  projections. Kept separate from the frozen number so the two are never conflated.

### 3.2 Experiment 2 — hybrid: DINOv3 backbone + low-level forensic features

The three carried-forward features, computed **on the post-transform image** (that is the image the
deployed model sees):

| feature | clean AUROC (95% CI) | worst official transform | note from prior work |
|---|---|---|---|
| `wavelet_hf_kurtosis` | 0.654 (0.575-0.726) | 0.612 @ 0.25x resize | most consistent |
| `residual_kurtosis` | 0.674 (0.596-0.736) | 0.618 @ 0.25x resize | strongest, but generator-sensitive; **direction reversal on WildFake** |
| `phase_neighbor_coherence` | 0.624 (0.536-0.696) | 0.597 @ blur sigma=1 | weakest, but complementary (rho ~ -0.21..-0.25 vs the kurtoses) |

The two kurtosis features correlate at 0.847, so they largely encode one signal. Realistic ceiling
from fusing these three alone is low — the value here is whether they add *anything* on top of
DINOv3, which is a question that has to be answered with a significance test, not a bare number.
The shortcut-failing features (`wavelet_level2_ratio`, `fft_mid_energy`, `fft_spectral_entropy`,
`fft_high_energy`) are **excluded from all model inputs** and retained only as audit probes (§5).

Four fusion designs, cheapest first, each an ablation arm:

- **H0 — concat baseline.** `[dino_emb ; z(feats)]` -> MLP. Control. Expect the 3 standardised
  scalars to be drowned by 768 dims; that is the point of having the control.
- **H1 — FiLM gating.** The feature vector produces `(gamma, beta)` that modulate the pooled DINOv3
  embedding. Lets the forensic signal *reweight* semantic features instead of merely appending.
- **H2 — dense forensic tokens (primary).** Compute the three statistics **per 16x16 tile**, aligned
  to DINOv3's patch grid -> a 3-channel forensic map at exactly patch resolution -> linear projection
  to `d_model` -> injected into the last N blocks as an additive side-channel on the patch tokens.
  This is the design worth betting on: it makes the handcrafted statistics *spatially aligned* with
  the tokens, so attention can localise "this region has AIGC-like tails", which a single global
  scalar structurally cannot express.
- **H3 — auxiliary regression.** Predict the three feature values from DINOv3 tokens as an auxiliary
  loss. No inference-time cost; tests whether forensic-awareness helps as a *regulariser*.

H2 is trained with LoRA on the backbone; H0/H1 can run frozen for cheap ablation.

**The comparison that decides it:** paired DeLong test of each hybrid against pure DINOv3 at matched
training budget and matched seeds, plus per-transform breakdown. A hybrid that wins on clean data and
loses under `0.25x resize` is not a win for this project.

---

## 4. Experiment 3 — data composition study

### 4.1 Cell mixture

Four cells: `real-clean`, `real-transformed`, `ai-clean`, `ai-transformed`.
Search over the mixture simplex with **total training image count held constant** — otherwise
composition is confounded with data quantity, which is the classic way this experiment goes wrong.

- Phase A: ~60-100 Dirichlet-sampled mixtures, evaluated with the *cheap* model (frozen DINOv3 +
  logistic head). Fast enough to actually cover the simplex.
- Phase B: top-5 mixtures + the uniform and clean-only references re-run with the full finetuned
  model, 3 seeds each.

### 4.2 Within-transformed proportions

Second-stage simplex over the 6 transform families (+ severity distribution). Same protocol.
Reported both ways: unweighted mean AUROC, and **worst-family AUROC** — the latter is what a
robustness claim actually rests on, and the two do not have the same optimum.

### 4.3 Guardrails

- **Fixed eval set**, chosen once, never touched by the search. Optimising composition against a
  moving eval target measures nothing.
- **Grouped splits** on `source_image_id`: all transformed variants of one source image live in one
  split. Otherwise a transformed copy in train and its clean twin in eval is straightforward leakage.
- **Held-out generator split** as a separate generalisation axis, since prior work showed
  `residual_kurtosis` reversing direction across generators.
- Selection is on a *validation* split; the test split is read once, at the end.

### 4.4 Eval-composition sensitivity

Re-weight the *fixed* per-image predictions under many hypothetical eval mixtures — no retraining
needed, it is a weighted metric recomputation — and report how much the headline AUROC/AUPRC moves.
This directly answers "how much does the reported number depend on how we composed the test set",
and it is nearly free.

---

## 5. Metrics and reporting

**Primary:** AUROC. **Secondary:** AUPRC (prevalence stated alongside — AUPRC is not comparable
across mixtures with different base rates, which is exactly what §4 varies, so every AUPRC is
reported with its positive-class rate and against a random-baseline reference line).

Also reported for every run:

- Balanced accuracy at 0.5 and at the EER threshold; EER.
- **TPR @ FPR = 1% and 5%** — a deployed detector operates at low FPR, and AUROC hides that region.
- Calibration: Brier score, ECE, reliability curve.
- **Uncertainty: stratified bootstrap 95% CIs, resampled by `source_image_id`, not by row.** With
  ~7 transformed variants per source image, row-level bootstrap would understate the CI by roughly
  sqrt(7). Paired comparisons use DeLong.

**Breakdowns:** clean vs transformed; per transform family x severity; per generator/source;
per original-resolution bucket.

**Shortcut audit — carried over from prior work, run on every model:**
correlation of the model score with image size, mean luminance, saturation, recompressed byte size;
AUROC restricted to the native-resolution subset; and the `img.size == (200,200)` cheat check. A
model beating the baseline while correlating 0.8 with resolution has learned the pipeline, not the
task. Prior findings this guards against: `default` config is AUROC 1.000 from image size alone, and
`laion_matched` still leaks 0.734 from mean luminance.

---

## 6. Logging

One directory per run, `runs/<utc-timestamp>__<name>/`:

```
config.json        fully-resolved config + git SHA + dirty flag + all seeds
env.json           package versions, CUDA/driver, GPU, hostname
manifest.json      dataset build hash, split hashes, row counts per cell
train_log.jsonl    per-step loss/lr/grad-norm; per-epoch val metrics
predictions.parquet  per-image: id, source_image_id, label, source, transform,
                     severity, score, split   <- every table in the report is
                     recomputed from this, so no metric is ever orphaned from
                     the predictions that produced it
metrics.json       all of §5, including breakdowns and CIs
report.md          generated, human-readable
```

Plus an appended row in `runs/index.csv` (one line per run) so runs are greppable, and
`docs/RESULTS.md` as the rolling narrative log of what we learned and what we rejected.
Final cross-run comparison gets published as an Artifact.

---

## 7. Repo layout

```
src/acai/
  transforms.py            official transform families, deterministic
  build_transformed.py     materialise the transformed dataset -> parquet + manifest
  features/lowlevel.py     wavelet_hf_kurtosis, residual_kurtosis, phase_neighbor_coherence
                           (global + per-tile variants for H2)
  models/backbone.py       timm DINOv3 wrapper, variable resolution, LoRA
  models/heads.py          linear / MLP / H0-H3 fusion heads
  data.py                  grouped splits, mixture sampler, loaders
  train.py                 one run = one config
  evaluate.py              §5 metrics, CIs, DeLong, breakdowns
  audit.py                 shortcut probes
  compose.py               §4 mixture search driver
  report.py                predictions.parquet -> report.md
docs/PLAN.md  docs/RESULTS.md
runs/
```

---

## 8. Sequencing

| # | step | depends on | status |
|---|---|---|---|
| 0 | install torch (cu130 aarch64), timm, sklearn, scipy, PyWavelets | — | **done** — verified on GB10, sm_121 |
| 1 | `transforms.py` + tests (determinism, size preservation, severity monotonicity) | B1 | **done** — 41 tests |
| 2 | ingest dataset, build manifest, grouped splits | **B2** | blocked |
| 3 | materialise transformed dataset | 1, 2 | blocked on B2 |
| 4 | `features/lowlevel.py` + reproduce prior per-feature AUROCs | 3 | **module done**, gate blocked on B2 |
| 5 | **Exp 1** — pure DINOv3, resolution x backbone sweep | 3 | backbone + heads done |
| 6 | **Exp 2** — H0-H3 hybrids + DeLong vs Exp 1 | 4, 5 | arms implemented |
| 7 | **Exp 3** — composition search, phases A and B | 5 | `compose.py` pending |
| 8 | shortcut audit, final report, Artifact | 5, 6, 7 | `audit.py` + `runlog.py` done |

Supporting modules complete and tested (97 tests green): `metrics.py` (AUROC/AUPRC/EER/
TPR@FPR/Brier/ECE, grouped bootstrap, DeLong), `audit.py`, `runlog.py`, `models/backbone.py`,
`models/heads.py`.

Step 0 runs now. Steps 1-3 unblock the moment B1 and B2 land; step 4 onward is then mechanical.

**Reproducing the prior feature AUROCs in step 4 is a gate, not a formality.** If our
`residual_kurtosis` does not land near 0.674 clean / 0.618 at 0.25x resize, our implementation
differs from the one those numbers came from, and every hybrid result downstream would be built on
sand.
