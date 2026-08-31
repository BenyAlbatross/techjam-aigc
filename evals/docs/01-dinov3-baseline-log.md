# Results log

Rolling record of what we ran, what we learned, and what we rejected. Newest first.
Every number here is reproducible from a `runs/<id>/predictions.parquet`, except the
pre-dataset diagnostics below, which name their inputs inline.

---

## 2026-08-31 — TRACE-RX-M v2 scored on all ten `wildfake-eval-subset` configs

**What ran.** `techjam-aigc/trace-rx-m-v2` (detector SHA `f811c464…`, frozen encoder) over all
106,444 images in the ten config paths. Inference uses the authors' own code, vendored at
`vendor/techjam-aigc` pinned to commit `a3c393a` — branch `feat/traincode`, the only branch whose
`OptimizerConfig` carries the `mixed_precision` key present in the checkpoint's embedded config,
and therefore the branch that produced the shipped weights. Predictions in
`runs/tracerx_v2_wildfake/predictions/`, metrics in `report.json` / `REPORT.md`.

| config | AUROC (95% CI) | bal.acc @0 | bal.acc @EER |
|---|---|---:|---:|
| `default` | 0.5813 (0.5682–0.5942) | 0.5272 | 0.5518 |
| `normalized` | 0.5623 (0.5490–0.5759) | 0.5226 | 0.5381 |
| **`laion_matched`** | **0.7379 (0.7251–0.7506)** | 0.5545 | 0.6724 |
| `cross_generator` | 0.6994 (0.6833–0.7150) | 0.5468 | 0.6414 |
| `diverse` | 0.6446 (0.6349–0.6535) | 0.5579 | 0.6060 |

Model card reports 0.9035 clean ROC-AUC on the authors' own development set. The drop to
0.74/0.64 here is a distribution shift (their reals are wikimedia/mirflickr/open-images, their
fakes FLUX/SDXL/GPT-Image-2), not a reproduction failure — the card already flags that
cross-generator generalisation is unsolved (Gemini Flash Image 0.328).

**Three findings.**

1. **`laion_matched` 0.7379 sits on top of this repo's own trivial-leak baseline of 0.734
   (mean luminance).** That is the config we tell people to compare on, and a full DINOv2
   prototype-memory detector currently matches what a one-line brightness statistic gets.
   This is not proof the model *is* using luminance — but the burden is now on showing it
   isn't, and the cheapest check is a luminance-decorrelated split.

2. **The shipped operating point does not transfer; the ranking partly does.** At the model's
   own threshold (logit >= 0) it calls ~90% of *everything* authentic — median logit -5.2 on
   `laion_matched`, and only 4.1% of real and 15.0% of AIGC rows clear zero. Balanced accuracy
   is therefore ~0.55 across the board while AUROC is 0.74. Re-fitting the threshold on this
   data recovers 0.6724. Any accuracy comparison against this checkpoint at its shipped
   threshold measures calibration drift, not detection.

3. **Robustness is genuinely flat — the one place it beats expectations.** On
   `laion_matched_transformed`, AUROC 0.7343 vs 0.7379 clean; the worst single setting is
   `blur_2.0` at 0.7031, and composing two transforms does not compound (0.7346 for two vs
   0.7343 for one). This reproduces the shape of the authors' claim (clean 0.9035 → macro
   0.9011) on data they never trained on.

**Note on `default`.** The 1.000-AUC image-size shortcut this repo warns about is unavailable
to this model: `canonical_preprocess` resizes every input to 224x224 before the backbone, so
resolution is destroyed. Its 0.5813 there is the honest content-only number, and is the
cleanest demonstration we have that the shortcut is a property of the *benchmark*, not of any
detector that must resize.

**Not run — TRACE-RX-Parallel.** No trained weights exist for it. The `feat/trace-rx-parallel`
branch carries `trace_rx_parallel/{config,model,training}.py` and a config, but no branch of the
GitHub repo contains a single `.pt`/`.safetensors` file, there are no releases or tags, and the
HF org publishes only `trace-rx-m-v2`. Its loader expects the same `s4_detector.pt` +
`s3_memory.pt` layout, so `--family trace_rx_parallel` will work as soon as those artifacts are
published.

---

## 2026-08-30 — Feature implementation calibrated against WildFake `cross_generator`

**What ran.** `feature_vector()` over 1,000 images (500 real / 500 AIGC, seed 0) from the
cached `techjam-aigc/wildfake-eval-subset` `cross_generator` config. 6.7 ms/image single-threaded.
This is a *diagnostic on eval data*, not training — the repo's no-train rule is intact.

| feature | median AIGC | median real | AUROC | prior study (their data) |
|---|---|---|---|---|
| `wavelet_hf_kurtosis` | 21.70 | 23.13 | 0.535 | 0.654 clean; medians 15.45 / 11.96 |
| `residual_kurtosis` | 9.57 | 9.25 | 0.506 | 0.674 clean; medians 7.49 / 4.99 |
| `phase_neighbor_coherence` | 0.17 | 0.16 | 0.533 | 0.624 clean |

Inter-feature Spearman: `wav~resid +0.908`, `wav~phase +0.416`, `resid~phase +0.470`.

**Read this carefully — the weak AUROCs are the expected result, not a red flag.**
`cross_generator` is WildFake, and the prior study *itself* reported these features collapsing on
WildFake specifically: `wavelet_hf_kurtosis` 0.597 pooled ("underpowered") and `residual_kurtosis`
**0.462 with a direction reversal**. Our 0.535 / 0.506 on a different WildFake slice is consistent
with that, not in contradiction of it. The prior study's headline AUROCs (0.654 / 0.674 / 0.624)
came from CIFAKE and SID, which we do not have here.

Two things this diagnostic does establish, and one it flags:

1. **The two kurtosis features are implemented consistently with the study.** Their inter-correlation
   reproduces at +0.908 against the reported 0.847 — close enough, on different data, to say we are
   computing the same pair of quantities.
2. **The medians are offset upward** (21.70 vs 15.45). Expected: `cross_generator` is downscaled to
   256x256, and resampling concentrates the high-frequency distribution. Not evidence of a convention
   mismatch on its own, but it means the median check cannot be run on this config.
3. **OPEN — the phase feature's correlation sign is flipped.** Prior study: `phase~kurtosis`
   rho = -0.21..-0.25. Ours: **+0.416 / +0.470**. Same magnitude class, opposite sign. Either our
   `phase_neighbor_coherence` differs from theirs in construction, or the relationship genuinely
   inverts on WildFake. This matters because the feature's entire justification is being
   *decorrelated* from the kurtoses — if it is positively correlated here, it adds much less in
   fusion than the study implies. **Resolve at step 4** against the real dataset before H1-H3 are
   trusted.

**Consequence for the plan.** The step-4 calibration gate cannot be closed on WildFake. It needs the
incoming dataset (B2), and ideally a CIFAKE or SID slice to reproduce the 0.654 / 0.674 / 0.624
headline numbers directly. Until then, no hybrid result is interpretable, because we cannot say
whether a null result means "these features do not help DINOv3" or "we implemented different features".

---

## 2026-08-30 — Environment and access

- **Torch 2.13.0+cu130 / torchvision 0.28.0** verified working on the GB10 (compute capability 12.1,
  aarch64). First install attempt exited 0 while having *failed* on a `pypi.nvidia.com` timeout —
  worth knowing, as `uv`'s exit code cannot be trusted alone here.
- **DINOv3 is gated on `facebook/*` and 403s for this account.** Routed via the ungated `timm/*`
  repos, which carry the same official weights. Also the better path: `timm` gives variable-input-
  resolution support and `forward_intermediates()`, both required by the H2 design.
- Transform module: 15 official single-transform conditions implemented, 41 tests green, including
  determinism, size-preservation, and severity monotonicity.

### Two bugs caught by tests, worth recording

- **`_kurtosis` returned -3.0 for flat regions**, not the intended neutral 0.0: the degenerate value
  was assigned *before* the excess-kurtosis shift. Would have biased `feature_maps` downward on every
  image containing a blown-out sky or solid background — a silent, content-dependent bias that no
  AUROC would have flagged as wrong.
- **A docstring overclaimed "float32 throughout, quantise once".** False: PIL-routed ops
  (`crop`, `resize`, `blur`, `jpeg`) quantise to uint8 internally. Accepted deliberately rather than
  worked around — PIL/torchvision semantics are what the official transforms are defined against, so
  matching the grader beats numeric purity. The docstring now says so, and a test pins it.

---

## 2026-08-30 — Infrastructure complete and tested (97 tests green)

Everything not requiring the dataset (B2) is built. What exists now:

| module | what it provides |
|---|---|
| `transforms.py` | 15 official conditions; T1/T2/T3 composition tiers; deterministic per-image seeds |
| `features/lowlevel.py` | the 3 carried-forward features, global + per-patch (for H2) |
| `metrics.py` | AUROC, AUPRC+base rate, EER, TPR@FPR 1%/5%, Brier, ECE, grouped bootstrap, DeLong |
| `models/backbone.py` | DINOv3 via timm, variable resolution, LoRA, patch-grid access |
| `models/heads.py` | pure baseline + H0/H1/H2/H3 fusion arms, one shared signature |
| `audit.py` | nuisance probes, the 4 shortcut features (probe-only), native-resolution check |
| `runlog.py` | per-run config/env/predictions/metrics + `runs/index.csv` |

**Verified, not assumed.** DeLong's AUROCs are checked against sklearn and its standard error
against a 400-sample bootstrap; the grouped bootstrap is checked to widen the CI by >2x versus
row-level resampling on 15-variant groups; H1 and H2 are checked to be exact no-ops at
initialisation, so any measured gain is attributable to the fusion rather than to a different init.

### Three bugs the tests caught, all silent-failure class

1. **`_kurtosis` returned -3.0 for flat regions** instead of the neutral 0.0 — the degenerate
   value was assigned before the excess-kurtosis shift. Would have biased `feature_maps` downward
   on any image with a sky or solid background. Content-dependent, and invisible in an AUROC.
2. **DINOv3 has 4 register tokens, not 0.** `num_reg_tokens` does not exist in this timm version;
   the correct attribute is `num_prefix_tokens` (= 5). The naive read misaligned the patch grid by
   4 positions, which would have corrupted every dense feature in H2 while still training happily.
   Caught by the square-grid assertion in `patch_grid`.
3. **`FeatureNorm` could be used unfitted**, passing raw kurtosis (~10) next to a LayerNorm'd
   embedding. Now raises during training rather than warning. Without this, the H0/H1/H3 ablation
   would have been comparing normalisation regimes as much as fusion designs.

### One design note

H2 injects via **forward pre-hooks**, not by reimplementing the backbone forward. timm's DINOv3 is
the RoPE variant: `_pos_embed` returns a tuple, blocks take a `rope=` kwarg, and there is a separate
`rope_mixed` code path. Hand-copying that would break silently on a timm update; a hook rides on
whatever the real forward does.

---

## 2026-08-30 — Dataset received (`Joshyxwa/data_draft`) — SEVERE resolution shortcut found

10,000 canonical RGB PNGs, 5,000 real / 5,000 `ai_full`, from two sources (SID-Set 5,000,
WildFake 5,000). Splits: train 7,000 / dev 1,500 / calibration 1,500, exactly class-balanced
within every split and source. 89-column manifest.

### The finding that changes the experimental design

**Each source is ~98% separable by image dimensions alone — no model, no pixels, manifest only.**

| probe (manifest fields only) | pooled AUROC | SID-Set only | WildFake only |
|---|---|---|---|
| pixel count (`w*h`) | 0.739 | **0.979** | **0.976** |
| is-square (`w == h`) | 0.739 | **0.978** | 0.500 |
| aspect ratio | 0.642 | | |
| file size (bytes) | 0.636 | | |
| `estimated_quality` | 0.668 | | |

Mechanism, and it is different in each source:

- **SID-Set:** all 2,500 AI images are exactly 1024x1024; SID reals are overwhelmingly
  non-square web photos (1024x768, 1024x683, ...). "Is it square?" answers it.
- **WildFake:** all 2,500 reals are exactly 200x200 — the known upstream WildFake
  downscaling artifact, previously flagged on `techjam-aigc/wildfake-eval-subset` — while the
  WildFake AI images keep native resolution. Only 104 AI images are 200x200 and only 101
  reals are 1024x1024.

**The pooled number actively conceals this.** The two shortcuts point in opposite directions
(SID's AI is the large/square class, WildFake's real is the small/square class), so they partly
cancel and the pooled figure lands at a merely-suspicious 0.739. Reporting only pooled AUROC
would hide a 0.98 leak. This is the single strongest argument for the per-source breakdown
being mandatory rather than nice-to-have.

**Consequence, and it is not optional.** Any model trained on these images as-is will learn
resolution, and would score near-perfectly while having learned nothing transferable. Note that
the official `resize` transform does **not** help: it upscales back to the original dimensions
by design, so it preserves the leak exactly.

Mitigation, to be applied before any training: centre-crop to square and resize every image to a
single fixed size. That neutralises pixel count, aspect ratio and is-square in one step. It costs
the native-resolution audit, so `width`/`height` are retained in the manifest as audit-only
columns — the model never sees them, but `audit.py` still checks whether its score tracks them.

### Structural facts that shape the scorecard

- **No identity field groups more than one asset.** `lineage_id`, `base_id`, `parent_asset_id`,
  `content_group_id`, `source_item_id` all have max group size 1 across all 10,000 rows. A
  lineage-grouped bootstrap is therefore *identical* to a row-level bootstrap today. It only
  becomes meaningful once transforms are materialised, since all 15 variants of an image will
  share its `lineage_id`. Wire it now; it starts binding at step 3.
- **No expert or human annotations of any kind.** No annotator, rater, vote, consensus or
  agreement fields; `label_confidence` is the constant 0.8 on every row; `label_evidence` is
  path/archive-derived (`wildfake_archive_role`, `path_pattern:synthetic`, `path_pattern:real`).
  Labels are source-provided, never human-adjudicated.
- **Generator identity is asymmetric.** WildFake gives 5 named generators (`adm`, `ddim`, `ddpm`,
  `imagen`, `gan_based`; 350 train / 75 dev / 75 cal each). SID's 2,500 AI images are a single
  lumped `text_to_image` with `model_family = unknown_generator`. Leave-one-generator-out works
  for 5 generators; the SID half can only be held out as one block.
- **Authentic subtypes are equally asymmetric.** 5 named WildFake subtypes (`afhq`, `celebahq`,
  `church`, `ffhq`, `imagenet`) plus `other` = all 2,500 SID reals undifferentiated.
- **Only two source datasets**, so "unseen source" is a 2-way leave-one-out, not a sweep.

### Packaging defect worth reporting upstream

`manifest.parquet` is **not parquet** — it is JSONL with a `.parquet` extension (`pd.read_parquet`
fails with "Parquet magic bytes not found"; `pd.read_json(..., lines=True)` works). The column
`file_size` is also mixed-type (int and str), so a naive parquet conversion fails too. Neither is
fatal, but both will break the documented loader snippet in the dataset card.

---

## 2026-08-30 — Step-4 calibration gate: CLOSED (on WildFake), and SID found to be unusable for it

Ran `feature_vector` over 1,000 images per source (500/class, seed 0), at native resolution and
at canonical 512, against the prior study's per-source numbers.

### WildFake — the confound-free source. Gate closes here.

| feature | ours (canonical 512) | prior study (WildFake) |
|---|---|---|
| `wavelet_hf_kurtosis` | 0.625 | 0.597 |
| `residual_kurtosis` | 0.543 | 0.462 (they reported a reversal) |
| `phase_neighbor_coherence` | 0.569 | 0.589 |

All three land within noise of the published WildFake figures (n=1,000 gives roughly +/-0.035).
`residual_kurtosis` sits just above chance where theirs sat just below; both are chance-level, so
the "reversal" is not a stable direction in either implementation. **We are computing the same
features they were.**

### SID-Set — numbers look far better, and are not trustworthy

| feature | ours (canonical 512) | prior study (SID) | rho vs `was_square` |
|---|---|---|---|
| `wavelet_hf_kurtosis` | 0.766 | 0.731 | +0.426 |
| `residual_kurtosis` | 0.783 | 0.797 | +0.453 |
| `phase_neighbor_coherence` | 0.804 | 0.644 | +0.488 |

**On SID, `is_square` alone predicts the label at AUROC 0.977**, and all three features correlate
0.43-0.49 with it. Their apparent strength here is substantially collinear with class geometry,
not evidence of forensic signal. The obvious control -- restricting to originally-square images --
is unavailable: SID has 500 square AI images against 23 square reals, and that imbalance *is* the
confound.

A subtlety worth stating, because canonicalisation does not fully remove it: centre-cropping is
itself class-dependent on SID. A 1024x1024 AI image loses nothing; a 1024x768 real loses a third
of its frame. The output dimensions are identical so no model can read the size, but the *content
statistics* still differ systematically. `audit.py` checks a model's score against original aspect
ratio for exactly this reason. On WildFake the issue does not arise -- every image is square.

### This resolves the earlier open item on the phase feature

The 2026-08-30 diagnostic flagged `phase_neighbor_coherence` correlating +0.42/+0.47 with the
kurtoses against the study's -0.21..-0.25. On the **confound-free** source those correlations fall
to **+0.152 / +0.184** -- near zero, which is materially consistent with the study's claim that the
feature is complementary rather than redundant. The large positive correlation was a property of
SID's geometry confound, not of the feature. The sign still differs from the published figure, so
the feature is treated as weakly-decorrelated rather than anti-correlated, and the fusion ablation
(H0-H3 + DeLong) is left to settle whether it actually contributes.

### Native vs canonical, which also validates the canonicalisation

On SID at *native* resolution the features score 0.812 / 0.818 / 0.774; canonicalising drops them
to 0.766 / 0.783 / 0.804. The drop is the resolution shortcut being removed. It is a useful
confirmation that canonicalisation bites.

### Absolute feature scales differ from the published medians, harmlessly

Ours on SID canonical: 61.6 (AI) / 31.6 (real) for `wavelet_hf_kurtosis`; the study reported
15.45 / 11.96. The study pooled CIFAKE, whose 32x32 images have far lower kurtosis, which drags
the pooled median down. AUROC is rank-based and unaffected, and the AUROCs match, so the gate is
closed on ranking behaviour rather than on absolute scale.

---

## 2026-08-30 — Pipeline validated on 3,454 locally-available images

- Shortcut reproduced on the partial set exactly as on the full manifest: pooled pixels 0.739,
  **SID 0.977, WildFake 0.977**.
- Canonicalisation to 512x512 produces a single distinct output size, so pixel count, aspect ratio
  and is-square all become constant: AUROC 0.500 by construction. 7 ms/image, ~1.2 min for 10,000.
- **Split integrity re-verified independently of the dataset card's claim**: zero straddling
  across `group`/`lineage_id`, `sha256`, `phash`, `base_id`.
- All 7 holdouts viable: 5 unseen-generator (9,500 train / 3,000 eval each) and 2 unseen-source
  (5,000 / 5,000).

---

## 2026-08-30 — First DINOv3 baseline, and why its 0.96 is not a detection result

**Run** `20260830T101530Z__dinov3_b224_probe_clean` — DINOv3 ViT-B/16, frozen, `cls_mean` pooling,
224px, linear probe, clean-only training (no transforms), scored on all 45,000 frozen eval rows.
Scorecard sealed, digest `2effdf053d812fe2`.

| metric | value |
|---|---|
| AUROC | 0.9608 |
| AUPRC (prevalence 0.500) | 0.9696 |
| TPR @ FPR 1% | 0.7594 |
| EER | 0.0963 |
| worst family | `noise` 0.9048 |
| worst condition | `noise:0.10` 0.8554 |
| clean → worst drop | 0.0841 |

Robustness looks strong for a clean-trained probe: `blur` 0.9867, `resize` 0.9836, `jpeg` 0.9626,
with only `noise` falling materially. That would be an encouraging result if the underlying number
meant what it appears to.

### The audit says it does not

| source | model AUROC (clean) | best trivial probe **alone** | score ↔ probe Spearman |
|---|---|---|---|
| WildFake | 0.9792 | `orig_pixels` **0.9787** | **+0.808** |
| SID-Set | 0.9983 | `is_square` **0.973** | **+0.751** |

The model is statistically tied with a probe that reads nothing but the *original* image
dimensions — dimensions it cannot see, because every input is 512x512.

Canonicalisation removed the dimension **cue** but not the resampling **history**, and the
mechanism differs per source:

- **WildFake:** every real is natively 200x200 and is **upscaled 2.56x** to reach 512, while the
  AI images are downscaled from 1024. "Was upscaled" is still approximately the label, now baked
  into the pixels where no manifest-level audit can see it. The control that would settle it is
  unavailable: of 1,500 clean WildFake rows only 154 were not upscaled, and **all 154 are AI**.
- **SID-Set:** centre-cropping is class-dependent. A 1024x1024 AI image loses nothing; a 1024x768
  real loses a third of its frame. Output dimensions match; content statistics do not.

**Correction to the earlier entry.** The 2026-08-30 pipeline-validation note presented
canonicalisation as having removed the shortcut. It removed the manifest-visible part, which is
all that entry actually verified. The residual pixel-level confound above is what the model
audit, not the manifest audit, exposed.

### Mitigation: a second, resolution-matched build

`data/build_matched` routes every image through a common **200x200 bottleneck** before the upscale
to 512, so both classes share one information ceiling and one final upsample. The cost is explicit:
it discards the high-frequency detail above 200x200 — precisely the band the three forensic
features operate in. This is the same trade the upstream `normalized` / `laion_matched` configs
faced, and there is no configuration of this corpus that avoids it: WildFake's reals are *all*
natively 200x200, so no resolution-matched subset exists within that source.

Both configs will therefore be reported side by side — 512 as high-fidelity-but-confounded,
matched as low-fidelity-but-trustworthy. A single headline number would misrepresent either way.

**The strongest available probe is neither**: `--train-source sid_set --eval-source wildfake`.
The two confounds differ in kind, so a model riding SID's cropping artefact should not survive
WildFake's upscaling artefact. Implemented in `train.py`; the holdout is part of the embedding
cache key, because a transfer run that silently reused the full-training cache would produce a
plausible number rather than an error.

---

## 2026-08-30 — Cross-source transfer: the baseline does NOT generalise. This is the headline result.

Four runs on `data/build_matched`, DINOv3 ViT-B/16 frozen + linear probe, 224px, identical
hyperparameters, all scored on the same frozen eval shards.

| run | n train | AUROC | TPR@FPR1% | clean | worst family |
|---|---|---|---|---|---|
| in-distribution (both sources) | 7,000 | **0.9658** | 0.769 | 0.988 | noise 0.917 |
| **control**: half data, both sources | 3,500 | **0.9601** | — | — | — |
| train SID → eval WildFake | 3,500 | **0.4133** | 0.028 | 0.419 | crop 0.399 |
| train WildFake → eval SID | 3,500 | **0.6733** | 0.080 | 0.812 | noise 0.486 |

**Train SID → eval WildFake lands at 0.413 — significantly *below* chance.** The model does not
merely fail to transfer; its learned direction is actively inverted on the other source. That is
the signature of a source-specific artefact, not of weak-but-real signal.

**The data-quantity objection is ruled out by the control.** Halving the training set while keeping
both sources costs 0.006 (0.9658 → 0.9601). Halving it by removing a source costs 0.55 (→ 0.413).
The collapse is distribution shift, not sample size. And no amount of missing data produces
*systematic inversion* — that requires having learned a rule that is wrong on the new source.

### Correcting the record twice, because I got this wrong in both directions

1. **First claim — "the 0.96 is a shortcut, because score correlates ρ=0.8 with `orig_pixels`."**
   The inference was invalid. `orig_pixels` alone scores AUROC 0.978 here, so it is nearly
   collinear with the label; *any* accurate model correlates ~0.8 with it. Correlation with a
   near-perfect nuisance is entailed by being accurate and is not evidence of using it.
2. **Second claim — "the bottleneck intervention shows the model is not relying on the artefact."**
   Also too strong. The bottleneck tested exactly one mechanism (resampling history) and correctly
   ruled that one out — WildFake AUROC moved 0.9792 → 0.9777. It could not speak to source-specific
   structure in general, which is what the transfer test now shows is present.

The correlation was uninformative, the bottleneck ruled out one mechanism, and only the
intervention that actually breaks the confound — training on a disjoint source — settles it.
Worth remembering as the general shape: **audit by intervention, not by correlation.**

### What this means for the project

- **The reportable number is cross-source transfer, not in-distribution AUROC.** An
  in-distribution 0.97 on this corpus isclose to unfalsifiable: both sources carry their own
  class-collinear structure (SID: AI is square and uncropped; WildFake: reals are all natively
  200x200), and a probe fits whichever is present.
- **The two directions are asymmetric and that is informative.** WildFake → SID reaches 0.812
  clean, SID → WildFake only 0.419. Training on WildFake, whose reals span five visual subtypes
  (afhq/celebahq/church/ffhq/imagenet), transfers meaningfully; training on SID, whose reals are
  one undifferentiated `other` bucket, does not transfer at all.
- **The transform-robustness picture is unchanged in shape but not in level.** Under transfer,
  WildFake → SID degrades hardest on `noise` (0.486 — chance) and `jpeg` (0.624), while clean
  holds at 0.812. So the robustness ranking (`noise` worst) survives, but the margins do not.
- **Every subsequent comparison — the H0–H3 hybrids, the composition study — must be judged on
  transfer.** Ranking fusion arms by in-distribution AUROC on this corpus would rank how well each
  arm fits source artefacts.

---

# 2026-08-31 — Overnight suite: leave-one-generator-out, external benchmark, fusion ablation, composition study

Five tasks, all complete. Training is WildFake-only (SID is 0.9984-separable from a 32x32
thumbnail, so it teaches content rather than generation); preprocessing is 224 crop-square-resize;
everything is judged on held-out generators or external data, never in-distribution.

## Task 1 — leave-one-generator-out on WildFake (5 folds x 3 seeds)

| held-out generator | AUROC | +/- | clean | size-cheat | margin |
|---|---|---|---|---|---|
| adm | 0.8813 | 0.0009 | 0.9192 | 1.000 | **-0.081** |
| ddim | 0.9807 | 0.0002 | 0.9916 | 1.000 | -0.008 |
| ddpm | 0.8796 | 0.0005 | 0.9021 | 1.000 | **-0.098** |
| gan_based | 0.8710 | 0.0005 | 0.8887 | 0.882 | -0.005 |
| imagen | 0.8801 | 0.0008 | 0.9584 | 1.000 | -0.042 |
| **mean** | **0.8986** | | | | |

**Every margin is negative** — the model is beaten by a manifest-only size lookup on every fold.
Inside WildFake each generator has exactly one native size (adm/ddim/ddpm 256x256, imagen 512x512)
against reals that are uniformly 200x200, so size alone scores 1.000 on four of five folds. The
exception is `gan_based`, the only generator with mixed sizes (256 and 224), whose cheat is 0.882
— and it is also the model's worst fold. The one place the cue is weak is the one place the model
is weak.

**So the internal LOGO number cannot be reported as an unseen-generator result.** It measures the
right thing on the wrong population. Task 1b exists because of this.

## Task 2 — external benchmark (`techjam-aigc/wildfake-eval-subset`)

Trained on WildFake (all 5 generators), scored on all four configs through the identical 224 +
15-condition pipeline. **Leakage verified: 0 sha256 collisions against our training set in 3,000
sampled rows per config**, on top of the documented family disjointness.

| config | AUROC | AUPRC | clean | TPR@FPR1% | EER | worst family |
|---|---|---|---|---|---|---|
| **`laion_matched`** (the card's recommended config) | **0.7055** | 0.6840 | 0.7355 | 0.043 | 0.352 | noise 0.652 |
| `cross_generator` | 0.6876 | 0.6619 | 0.7198 | 0.035 | 0.361 | noise 0.655 |
| `normalized` | 0.8220 | 0.8344 | 0.8715 | 0.227 | 0.256 | noise 0.757 |
| `default` | 0.8509 | 0.8659 | 0.9241 | 0.299 | 0.232 | noise 0.757 |

**This is the honest headline: ~0.71 on `laion_matched`.** Not 0.96.

The ordering is itself diagnostic and matches the card's warning exactly. `default` (0.851) >
`normalized` (0.822) > `laion_matched` (0.706): performance *rises* as the config's known shortcuts
are added back. `default` is the config whose card states image size alone gives AUROC 1.000 — and
our model scores highest there. `laion_matched`, the config built to remove that leak, is where it
scores lowest. A detector reading genuine signal would not be ordered this way.

TPR@FPR=1% is the number to sit with: **0.043 on `laion_matched`**. At a 1% false-positive rate the
model catches 4% of AI images. AUROC 0.71 hides that completely.

## Task 1b — LOGO models scored on external data (the confound-free version)

Same held-out-generator models, scored on `laion_matched` where the 200-vs-256 size cue does not
exist.

| training set | external AUROC | delta vs all |
|---|---|---|
| all 5 generators | 0.7094 | — |
| without `adm` | 0.7147 | **+0.005** |
| without `ddim` | 0.6987 | -0.011 |
| without `ddpm` | 0.6926 | -0.017 |
| without `imagen` | 0.6815 | -0.028 |
| without `gan_based` | 0.6686 | **-0.041** |

Two findings the internal LOGO could not have produced:

* **`gan_based` is the single most valuable generator in training** (-0.041 when removed) —
  by a clear margin over the diffusion models. GAN artefacts appear to teach something the
  diffusion generators do not, and it generalises to a benchmark containing GigaGAN.
* **`adm` contributes nothing** (+0.005, i.e. removing it slightly helps). Training-set diversity
  here is not uniform in value; four of the five generators are near-interchangeable.

## Task 3a — fusion ablation (H0/H1/H3 vs pure DINOv3)

Objective: mean AUROC over the 5 held-out-generator folds, 2 seeds, paired DeLong on ~67k rows.

| arm | LOGO mean | worst fold | vs pure | p |
|---|---|---|---|---|
| pure DINOv3 | 0.8990 | gan_based 0.8712 | — | — |
| H0 concat | 0.9023 | ddpm **0.8532** | +0.0032 | 1.1e-08 |
| H1 FiLM | 0.9015 | ddpm 0.8506 | +0.0024 | 2.9e-04 |
| H3 auxiliary | 0.9010 | ddpm 0.8556 | +0.0022 | 1.2e-04 |

**All three are statistically significant and practically worthless.** The p-values are small
because n is ~67,000 paired rows, not because the effect is large: +0.003 AUROC. And every arm has
a *worse* worst-generator fold than the baseline (0.851-0.856 vs 0.871), so the fusion trades a
negligible mean gain for a real robustness loss.

This is exactly the case the plan's DeLong requirement was meant to catch, and it caught it in the
opposite direction from the one anticipated — the danger was never a false negative, it was
reading a significant p-value as a meaningful gain.

**H2 (dense per-patch forensic tokens) was not run.** It injects maps inside the backbone, so it
cannot use cached embeddings and needs a full finetune. Deferred, not dropped — and it remains the
only arm whose hypothesis (spatially-localised forensic evidence) is untested.

## Task 3b — training composition study (43 mixtures)

| mixture | LOGO mean | worst fold |
|---|---|---|
| all clean | 0.9142 | 0.8701 |
| all transformed | 0.9103 | 0.8639 |
| uniform 50/50 | 0.9123 | 0.8677 |
| best of 40 Dirichlet | 0.9154 | 0.8688 |
| worst of 40 Dirichlet | 0.9050 | 0.8623 |

**Composition does not matter here.** The entire 43-mixture range spans 0.905-0.915 mean and
0.862-0.872 worst — a spread of 0.010, comparable to seed noise and with no interpretable structure
across the simplex. Training size was held constant at one row per image throughout, so this is not
a data-quantity artefact.

That is a clean negative answer to the original question: on this corpus, the proportion of
clean-vs-transformed and the mix of transform families in training make no measurable difference to
held-out-generator performance.

## What the suite establishes overall

1. **The real number is ~0.71 AUROC / 0.043 TPR@FPR1% on `laion_matched`**, not the 0.96 the
   in-distribution split reports.
2. **Model quality is inversely ordered with benchmark cleanliness** (0.851 on the config with a
   known perfect size shortcut, 0.706 on the config built to remove it). That ordering is the
   strongest single piece of evidence that the frozen-DINOv3 probe is riding dataset artefacts.
3. **`noise` is the worst transform family everywhere** — internal folds, all four external
   configs. It is the only robustness finding that has held across every experiment run today.
4. **Neither of the two planned interventions helps.** Feature fusion gives +0.003 with a worse
   worst-case; composition gives nothing at all.
5. **Generator diversity is not uniform**: `gan_based` is worth -0.041 external AUROC when removed,
   `adm` is worth nothing. If more training data is acquired, GAN-family coverage is the priority.

The frozen-probe approach has been characterised about as thoroughly as this corpus permits. The
untested directions that remain are full finetuning (which could learn a shared cross-source
direction that frozen features demonstrably lack), H2, and more diverse training sources.

---

## 2026-08-31 — CORRECTION: the step-4 calibration gate does NOT close

The 2026-08-30 entry "Step-4 calibration gate: CLOSED (on WildFake)" is **wrong** and is retracted.

Two errors compounded:

1. **It ran on a partial download.** Only 3,454 of 10,000 images were on disk at the time, so the
   WildFake sample was drawn from ~1,726 rather than 5,000 images.
2. **"Within noise" was asserted, not computed.** No confidence interval was calculated; the gaps
   were eyeballed.

Re-run on the full corpus, n=1,000 (500/class), 2,000 bootstrap resamples:

| feature | ours | 95% CI | prior study (WildFake) | prior inside CI? |
|---|---|---|---|---|
| `wavelet_hf_kurtosis` | 0.621 | [0.584, 0.657] | 0.597 | **yes** |
| `residual_kurtosis` | 0.532 | [0.503, 0.568] | 0.462 | **no** |
| `phase_neighbor_coherence` | 0.521 | [0.501, 0.556] | 0.589 | **no** |

CI half-width is ~0.028, so the gaps for the latter two are real, not sampling noise.
`phase_neighbor_coherence` moved from 0.569 on the partial sample to 0.521 on the full one --
i.e. from "weak but present" to chance.

**Only `wavelet_hf_kurtosis` reproduces.** It is also the feature the prior study called "the
safest single low-level candidate", so the one that reproduces is the one that was most robust
there -- consistent, but only one of three.

### Consequence for the fusion ablation

The task-3a result (H0/H1/H3 give +0.003 over pure DINOv3) is **weaker evidence than reported**.
It establishes that *our* three features add nothing. It does not cleanly establish that the prior
study's features add nothing, because two of the three are demonstrably different quantities.

This is exactly the ambiguity the gate was designed to prevent, and it was reported closed when it
was not. The fusion conclusion should be stated as conditional until `residual_kurtosis` and
`phase_neighbor_coherence` are reconciled against the original implementation -- which requires
that implementation, not more experiments on ours.

### Also noted

`hub/README.md` now documents the competition's official **composition policy**, which was open
question B1: 70% of rows get exactly one transform (round-robin over 14 settings), 30% get two
from different families, applied in the fixed physical order
`crop -> resize -> jitter -> blur -> noise -> jpeg`. That ordering matches `FAMILY_ORDER` in
`acai/transforms.py` exactly, and the 70/30 split is directly implementable as a T2 tier. The
eval subset also now ships `_transformed` twins of all four configs, which would replace our
locally-generated external transform conditions with the official ones.
