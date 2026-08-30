# AIGC exploration and implementation plan

## Status and authority

This is the implementation contract for the next feature-robustness phase.
[The challenge brief](problem-statement.md) remains authoritative. The target is
binary classification of purely generated versus authentic images. Use only
public or properly licensed data; never train on test labels or the
demonstration-only COCO val2017 / DALL-E Advanced split; keep the final model
below two billion parameters; do not reuse a pretrained AIGC detector as the
solution; do not rely on watermarks; and keep the full pipeline reproducible.

The brief leaves transform composition ambiguous. The compositions below are
documented stress-test choices, not claims about hidden evaluation.

Implementation status, evidence, and honest deferrals are recorded in the
[plan-to-code reconciliation](aigc-exploration-reconciliation.md). The checklist
below is the original review contract; use the reconciliation for its evaluated
status rather than interpreting unchecked source text as a current result.

## Goal and current evidence

Determine which signals transfer across datasets and generator families, which
survive sequential redistribution, which combinations are complementary, and
whether generator diversity matters more than image count.

The pilot has 376 eligible parents and only 188 confirmation parents. WildFake
confirmation contains 4--12 generated images per displayed model and 12
authentic images. AIGC prevalence in aggregate confirmation is 0.585. The
canonical nuisance-only model reaches 0.862 clean AUPRC (0.667 normalized above
prevalence), versus 0.819 (0.564 normalized) for all engineered candidates.
Dataset, resolution, format, and codec shortcuts can therefore dominate.
Residual kurtosis (0.764 AUPRC), Haar high-frequency kurtosis (0.757), and
Fourier phase-neighbor coherence (0.728) are provisionally retained but
low-confidence.

## Six-step execution plan

### 1. Incorporate larger SID Set and WildFake subsets

Use the larger subsets already being acquired by the team before adding another
external corpus.

- Admit authentic and fully synthetic images only; exclude SID tampered rows.
- Preserve official or reproducibly reconstructed splits and immutable parent
  IDs.
- Target at least 200 images per generator/source confirmation stratum, with
  500 preferred.
- Balance authentic and generated counts within each evaluation comparison.
- Record dataset, generator family and model, source dataset, resolution,
  format, encoded bytes, split method, source member, and provenance.
- Retain low-power warnings instead of hiding small groups through pooling.
- Prefer deterministic indexed slices to complete multi-terabyte archives.

### 2. Add a targeted, license-audited AI-GenBench subset

AI-GenBench is the priority external source because its chronological protocol
spans 36 generators and ends with SDXL, DALL-E 3, and FLUX.

- Pilot with 100 images per generator plus an equal authentic pool.
- Increase to 200--250 per generator for the serious run.
- Preserve generator identities, authentic sources, and chronological windows.
- Match or normalize resolution, format, compression, and semantic content.
- Hold out whole generators rather than random images.
- Store upstream source, revision, exact file list, hashes, source license,
  model/image license, decision, reviewer, and review date.
- Keep acquisition disabled until every selected source is allowlisted.

The hosted fake portion aggregates sources with different terms; public access
is not sufficient evidence of permission. See the
[AI-GenBench paper](https://arxiv.org/abs/2504.20865) and
[dataset card](https://huggingface.co/datasets/lrzpellegrini/AI-GenBench-fake_part/blob/main/README.md).

### 3. Run the frozen feature registry

Run the existing preregistered features before changing formulas or selecting
new candidates.

- Freeze formulas, discovery-learned directions, thresholds, and model
  hyperparameters.
- Results inspected for selection become discovery results, never
  retrospectively confirmation results.
- Preserve native-capped and canonical-128 views.
- Report nuisance-only controls beside candidate models.
- Evaluate leave-one-dataset, leave-one-generator, and chronological future
  windows.
- Keep every transformed derivative in its parent's phase.
- Save a schema/configuration hash so the frozen run is identifiable.

### 4. Add missing feature families

Every feature requires a measurement, hypothesis, expected failure, role, and
cost.

#### High-priority inexpensive families

1. **Lower bit planes**
   - RGB bit-0 through bit-3 occupancy and entropy.
   - Horizontal/vertical transitions and run structure.
   - Cross-channel agreement.
   - Directional-gradient tails and maximum-response patch summaries.
   - Expected failure: JPEG, noise, re-encoding, screenshots, and reduced bit
     depth.

2. **Patch distributions**
   - Compute residual, gradient, spectral, and bit-plane measures per patch.
   - Aggregate median, IQR, upper quantiles, maximum, and spatial
     heterogeneity instead of only global means.
   - Expected failure: tiny images, crop selection, flat scenes, and semantic
     texture mismatch.

3. **Multi-scale residuals**
   - Gaussian residuals at sigma 0.5, 1, 2, and 4.
   - Scale-wise standard deviation, MAD, kurtosis, tail fraction, neighbor
     correlation, and cross-scale correlation.
   - Expected failure: blur, resizing, injected noise, and sharpening.

4. **Compact steganalysis residual co-occurrences**
   - A small documented directional high-pass bank.
   - Quantized horizontal and vertical residual co-occurrences.
   - Avoid an opaque full SRM implementation in the prototype.
   - Expected failure: codec alignment, noise, and resolution.

5. **Camera-pipeline proxies**
   - CFA-like and color-difference periodicity, signal-dependent noise-fit
     residuals, and cross-channel residual coupling.
   - Do not call a single-image proxy PRNU; device PRNU normally requires
     multiple images.
   - Treat these as specialist/authenticity-supporting evidence because genuine
     web images may be scans, renders, or heavily processed.

#### Medium-priority families

6. **Richer spectrum**
   - Radial residual relative to a fitted 1/f spectrum.
   - Multi-radius angular entropy and anisotropy.
   - Cross-channel spectral coherence and compact phase-magnitude coupling.

7. **JPEG and resampling history**
   - Quantization periodicity, 8-pixel grid phase, double-compression proxies,
     chroma-subsampling structure, and resampling aliasing.
   - Treat these as codec-sensitive and audit them against format nuisances.

8. **Richer color distributions**
   - YCbCr/Lab marginal and joint statistics, chroma residuals, and
     cross-channel co-occurrences.

#### Explicit learned and expensive controls

9. **Generic semantic control**
   - Frozen public DINOv2 or CLIP embedding plus a discovery-only regularized
     linear probe.
   - This is a generic backbone control, not a pretrained AIGC detector.
   - Require an explicit model identifier, revision, preprocessing, weight
     hash, and license; never silently download weights.

10. **Optional reconstruction evidence**
    - Compare autoencoder/diffusion spatial, latent, and frequency-separated
      reconstruction error.
    - Defer until its compute and diffusion specialization justify the cost.

Primary precedents motivate experiments rather than direct replication:
[LOTA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.html),
[FIRE](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_FIRE_Robust_Detection_of_Diffusion-Generated_Images_via_Frequency-Guided_Reconstruction_Error_CVPR_2025_paper.html),
[SPAI](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html),
and [CO-SPY](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html).

### 5. Test Community Forensics as optional training diversity

Community Forensics is about 1.08 TB and contains roughly 2.7 million images
from 4,803 generators. Do not download it wholesale.

- Select 200--500 generator identities across architecture, family, release
  period, and audited license.
- Sample about 10--20 images from each selected generator.
- Use it as a training-diversity condition, not the primary per-generator
  evaluation set.
- Hold out complete generators and families.
- Compare equal-image-count few-generator and many-generator training sets.
- Evaluate both on the same adequately powered AI-GenBench confirmation groups.

This tests whether generator diversity helps more than raw image count. Dataset
terms do not replace underlying model/image license checks. See the
[paper](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_Detectors.html)
and [dataset card](https://huggingface.co/datasets/OwensLab/CommunityForensics).

### 6. Preserve untouched final confirmation

Seal the newest AI-GenBench chronological window, or an equivalent licensed
modern-generator group, before experimentation.

- Store the assignment in the manifest.
- Reject generator overlap with discovery.
- Exclude it from feature, transform, threshold, fusion, and hyperparameter
  selection.
- Record when it is first evaluated.
- Do not repeatedly inspect it while changing the method.
- Distinguish the frozen result from any later diagnostic run.
- The organizer demonstration-only split remains forbidden.

## Filter composition

Analysis filters and robustness transformations are different objects.

### Do not exhaustively sequence linear filters

Sequential linear shift-invariant convolutions collapse to one combined kernel,
apart from boundary and quantization effects. Gaussian sigma 1 followed by
sigma 2 approximates sigma sqrt(5); Gaussian followed by Sobel is a
derivative-of-Gaussian; convolution order normally commutes with fixed padding.
An exhaustive permutation grid creates redundant hypotheses and multiple-test
noise.

### Hypothesis-driven feature pipelines

Implement and ablate:

- lower bit plane -> directional gradient -> maximum-response patch;
- Gaussian denoise -> signed residual -> quantized co-occurrence;
- YCbCr -> chroma residual -> cross-channel spectrum;
- multi-scale wavelet bands -> patch-tail aggregation;
- JPEG probe -> difference image -> gradient/DCT summaries; and
- semantic embedding plus low-level residual evidence through late score
  fusion.

Compare every composed pipeline with its constituents. Do not enumerate all
2^N feature subsets. Compare univariate features, family-only regularized
models, sparse standardized early fusion, out-of-fold late fusion, and
semantic/low-level/optional reconstruction branches. Select only on grouped
discovery folds and report coefficient stability, ablations, cost, and
worst-condition behavior.

## Sequential robustness transformations

Current official singles cover JPEG quality 90/70/50/30, Gaussian blur sigma
0.5/1/2, down-up resize 0.5/0.25, Gaussian noise sigma 0.02/0.05/0.10,
brightness/contrast/saturation within +/-20%, and centered 80% crop.

Current color jitter tests only the diagonal corners (0.8, 0.8, 0.8) and
(1.2, 1.2, 1.2). It misses six mixed two-level corners, single-axis changes,
and continuous interior values.

The relevant grid sizes are:

- 4 JPEG * 3 blur * 2 resize * 8 color corners = **192**.
- Adding 3 noise levels and optional crop gives **1,152**.
- Allowing no-op per family and 0.8/1.0/1.2 per color axis gives
  5 * 4 * 3 * 4 * 27 * 2 - 1 = **12,959** fixed-order pipelines.

The last design produces about 9.7 million rows for 376 parents and two views,
before extra self-consistency probes or order permutations. It is not the
default.

### Staged transformation design

1. Retain clean plus every official single severity.
2. Add six single-axis color changes, all eight two-level color corners, and
   deterministic continuous interior samples for training.
3. Screen all 15 pairs among six transform families at medium severity and both
   orders where meaningful: about 30 directed conditions.
4. Expand severity only for material interactions, direction reversals, or
   large parent-wise drift.
5. Preregister 12--20 realistic chains: crop -> resize -> JPEG; color -> resize
   -> JPEG; blur -> noise -> JPEG; resize -> sharpen -> JPEG; and JPEG -> resize
   -> JPEG.
6. Add a deterministic 32--64 recipe covering-array, Latin-hypercube, or
   Sobol-style bank.
7. Search for worst cases only on discovery data, then freeze recipes before
   confirmation.

Use a realistic canonical order by default: geometric edit, photometric edit,
noise/blur, then final encoding. Explicitly compare order for noise/JPEG,
blur/noise, resize/noise, and repeated JPEG. Stable pipeline IDs must encode
ordered operations, parameters, interpolation/padding, and seed. Apply the same
recipe distribution to both classes.

## Pairing and counterfactuals

- Pair every transformed observation with its clean parent.
- Match authentic and generated groups by semantic domain, source, resolution,
  codec, and format where possible.
- Prefer prompt- or conditioning-matched real/generated pairs when licensed.
- Compare the same prompt across generators.
- Include real-real source and fake-fake generator controls.
- Re-encode both classes identically to expose codec shortcuts.
- Never choose pairings or transforms using confirmation labels.

## Metrics and notebook presentation

Report AUPRC (average precision, AIGC positive), positive prevalence,
prevalence-normalized AUPRC, and balanced accuracy together with frozen
direction, direction reversals relative to prevalence, bootstrap intervals,
counts/power flags, clean-to-corruption drop,
worst-condition lower confidence bound, area under severity curves, parent-wise
drift, within-parent transform variance, A->B versus B->A order sensitivity,
interaction excess beyond single effects, leave-one-dataset/generator and
chronological results, nuisance correlations, matched-reencoding controls,
calibration when required, and extraction/inference cost.

The marimo notebook should include:

- data, license, and power audit;
- real-versus-AIGC transform/filter microscope with shared map scaling;
- feature-gap registry and implementation status;
- feature x generator x condition views;
- parent-wise severity trajectories;
- directed-pair order-sensitivity heatmap;
- realistic/covering-bank summaries;
- branch ablations and clean/worst-case Pareto frontier;
- generator-diversity versus image-count experiment; and
- a visibly sealed final-confirmation section.

Visual maps must disclose normalization. Conclusions must use evaluated scalars
or scores, not display brightness.

## Reproducibility safeguards

- Store source URLs, revisions, file lists, hashes, licenses, audit decisions,
  reviewer, and review date.
- Default acquisition to dry-run until every selected source is allowlisted.
- Keep credentials out of the repository.
- Generate transforms at run time or in ignored deterministic caches.
- Record code/configuration/feature/transform hashes and random seeds per run.
- Prevent parent, prompt, near-duplicate, and generator leakage where metadata
  permits.
- Treat source metadata as a nuisance control, not a contribution.

## Deliverables

1. License-gated dataset expansion manifest.
2. Deterministic selection and split-audit tool.
3. Frozen-registry baseline mode.
4. Registered inexpensive new feature families with tests.
5. Opt-in single, interaction, realistic, and covering transform profiles.
6. Drift, interaction, chronological, and diversity evaluation tables.
7. Updated marimo notebook and documentation.
8. Final plan-to-code reconciliation with deferred dependencies identified.

## Acceptance checklist

### Data and licenses

- [ ] SID/WildFake expansion preserves provenance and excludes tampered rows.
- [ ] AI-GenBench counts, windows, sources, revisions, hashes, and license
      decisions are machine-readable.
- [ ] Acquisition is blocked while selected licenses are pending.
- [ ] Community Forensics is sampled by generator, not downloaded wholesale.
- [ ] Few-generator and many-generator conditions use equal image counts.
- [ ] The demonstration-only organizer split is rejected.
- [ ] Final-confirmation generators cannot overlap discovery.

### Features and fusion

- [ ] The original feature schema can run frozen.
- [ ] Bit-plane, patch, multi-scale residual, compact residual co-occurrence,
      camera-proxy, richer spectrum, JPEG/resampling, and richer color
      hypotheses are implemented or explicitly deferred.
- [ ] Every feature records measurement, hypothesis, failure, family, and role.
- [ ] Generic semantic weights require explicit licensed revision metadata.
- [ ] Reconstruction evidence is optional and costed.
- [ ] Fusion is compared with constituents under grouped discovery selection.

### Transformations

- [ ] Every official single severity remains covered.
- [ ] Color jitter has axial and mixed-direction cases.
- [ ] Medium directed pairs and order metadata are reproducible.
- [ ] Realistic and covering profiles are opt-in.
- [ ] The default never enumerates the 12,959-condition grid.
- [ ] Both classes receive the same transform distribution.
- [ ] Pipeline IDs capture order, parameters, interpolation, padding, and seed.

### Evaluation and presentation

- [ ] Parent grouping is preserved.
- [ ] Power, intervals, reversals, nuisance controls, drift, interaction, and
      order sensitivity are reported.
- [ ] Dataset, generator, chronological, and sealed-confirmation results remain
      distinct.
- [ ] The notebook compares real/AIGC under the same transform and scale.
- [ ] Exploration and sealed confirmation are visibly distinct.
- [ ] Tests, static notebook checks, and a small end-to-end run pass.

See the [final reconciliation](aigc-exploration-reconciliation.md) for the
PASS/deferred decision and verification evidence for every item.
