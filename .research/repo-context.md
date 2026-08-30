# Repository context for the detector literature review

_Repository snapshot inspected 30 August 2026. `.git` and `.venv` were excluded. This note distinguishes executable code and measured pilot results from research recommendations and the unimplemented TRACE-RX proposal._

## 1. Authority, scope, and hard constraints

`docs/problem-statement.md` is authoritative. The target is image-level binary classification of **purely generated AIGC** (`Y=1`) versus **authentic** (`Y=0`). AI edits and partial composites are out of focus. The public entry point must accept an image directory and write JSON records containing `image_path` and a continuous AIGC confidence in `pred`.

Hard challenge guardrails that constrain any paper-derived method:

- every final model must have fewer than 2 billion parameters;
- pretrained backbones must be public; a team must not merely use or reproduce an existing pretrained AIGC detector or existing detector approach;
- do not train on test labels or the organizer's demonstration-only COCO `val2017` / DALL-E Advanced split;
- use public or properly licensed data and release reproducible training, augmentation, evaluation, hyperparameters, weights if the team wins, and source code;
- do not depend on SynthID or any watermark;
- hidden evaluation should include old Stable Diffusion, newer DiT/flow-like families, and real-world redistribution.

The official metric, score calibration/range, JSON root shape, transform composition policy, exact hidden generators/data/formats, and SynthID handling are still open. A literature claim that assumes one of these is not direct evidence for this repository.

## 2. What is implemented now

### 2.1 Executable detector work is a feature laboratory, not a final detector

The implemented package is `src/techjam_aigc/feature_lab/`. It extracts deterministic classical scalars, applies transformations, trains simple discovery-only logistic probes for analysis, and writes audit tables. It is explicitly documented as hypothesis testing rather than a deployable detector.

The default `frozen_v1` registry has 53 scalars:

- 5 nuisance controls: log pixel count, aspect ratio, encoded bytes/pixel, JPEG flag, PNG flag;
- 48 candidates over spatial intensity, color, LBP/GLCM texture, Gaussian residuals, noise proxies, FFT magnitude, FFT phase, block-DCT/JPEG, Haar wavelets, gradients, and JPEG/blur/resize self-consistency.

`expanded_v2` is implemented in code with 82 scalars. Its 29 added candidates cover RGB bit planes 0--3, patch upper tails/heterogeneity, multi-scale Gaussian residuals, compact directional residual co-occurrences, camera-pipeline proxies, richer 1/f/angular/cross-channel/phase spectra, JPEG-grid/resampling history, and YCbCr chroma. These features have measurement, hypothesis, expected-failure, role, and cost metadata. **The checked-in/current derived run is still `frozen_v1`; there is no measured `expanded_v2` result in the current cache.**

Every image is evaluated in two analysis views:

- `native_capped`: preserve native size, cap the long side at 256, never upscale;
- `canonical_128`: Lanczos resize to 128 x 128.

The transform engine implements the official singles and deterministic optional compositions. The current `core` cache has 20 conditions: clean; JPEG Q90/70/50/30; blur sigma 0.5/1/2; Lanczos down/up 0.5/0.25; seeded RGB Gaussian noise sigma 0.02/0.05/0.10 on `[0,1]`; two diagonal color-jitter points `(0.8,0.8,0.8)` and `(1.2,1.2,1.2)`; centered 80% crop restored by bicubic resize; plus four repository-defined compositions. Opt-in profiles implement axial/mixed color cases, 30 medium-severity directed pairs, 12 realistic chains, and a deterministic 32-recipe covering bank. Operation order, parameters, interpolation/padding, seed policy, and hashes are recorded.

Data/split guards admit only real/authentic and fully synthetic labels, exclude SID tampered rows, reject demo-only rows, keep derivatives with their parent and phase, and withhold `final_confirmation` by default. An audited expansion index requires both dataset-source and underlying-image licenses, revision, and exact selected-index hash. A first final-confirmation evaluation writes a receipt; a repeat is refused.

Evaluation code chooses each scalar's orientation once on pooled clean discovery data. It reports average precision/AUPRC with AIGC positive, prevalence, normalized AUPRC `(AP-prevalence)/(1-prevalence)`, class-stratified bootstrap intervals for univariate tables, low-power and direction-reversal flags, nuisance correlations, family ablations, leave-one-dataset-out probes, paired drift, severity loss/area, directed-pair interaction and order effects, errors, and guarded chronological tables. Family probes are median-imputed, standardized, class-balanced `LogisticRegression` models fit only on clean discovery. They are analytical baselines, not TRACE-RX training.

### 2.2 Data and reproducibility tooling

The local feature pilot uses deterministic slices of CIFAKE, SID Set, and WildFake. WildFake's published 80/20 policy is reconstructed by stable hash because official split membership is absent. The local gallery is balanced for inspection, not prevalence-representative. A planner exists for larger SID/WildFake slices, targeted AI-GenBench chronological evaluation, and equal-image-count Community Forensics diversity cohorts, but acquisition is dry-run and blocked because revisions, file hashes, and both license layers are pending. No AI-GenBench or Community Forensics data has been downloaded by this workflow.

The repository has tests for binary/demo/tampered/final guards, license-bound selection, feature profiles, transform determinism/order/seeds, metrics, and cache metadata. At inspection time `uv run pytest -q` passed: **48 passed**.

There is **no trained neural detector, checkpoint, training loop for TRACE-RX, calibrated final score, parameter/latency measurement, or required directory-to-JSON inference implementation**. The installed CLI still prints `Hello from techjam-aigc!`.

## 3. Current measured state and its boundary

The cached pilot contains 376 eligible parent images: 188 discovery and 188 confirmation parents. With 20 conditions and two views it has 15,040 rows. Aggregate confirmation prevalence is 0.585. The cache predates the current schema-2/profile metadata: it is a `frozen_v1`/legacy-core result, not evidence for `expanded_v2`, directed pairs, realistic chains, or the covering bank. Current code implements those paths, but their result tables are empty or absent here. CIFAKE is only 32 x 32 and one Stable Diffusion version; SID lacks row-level generator identity; WildFake confirmation has only 4--12 fakes per displayed generator and 12 authentic images. The WildFake fake slice also includes StarGAN, MAE, and related categories whose pure-generation provenance needs a row/source audit before they are used for this challenge; image translation or reconstruction must not silently become positive training evidence. The pilot validates code, not modern-generator generalization.

Important current results:

- On `canonical_128` clean confirmation, the nuisance-only probe reaches **0.862 AP / 0.667 normalized AP**, while all engineered candidates reach **0.819 / 0.564**. This is direct evidence that source, size, codec, and format can dominate.
- On `native_capped`, all engineered candidates reach **0.898 AP / 0.755 normalized AP**, versus nuisance-only **0.862 / 0.667**. This apparent gain is not secure: the views behave very differently and the dataset cells are small/confounded.
- Family AP on clean `canonical_128`: residual 0.766, wavelet 0.748, FFT phase 0.720; FFT magnitude is 0.576, below the 0.585 prevalence baseline. Native results are often much higher, which makes preprocessing sensitivity a first-class issue.
- Only three scalar decisions are provisionally `keep`, all low confidence: residual kurtosis (clean AP 0.764; worst official 0.694 under noise 0.10), Haar high-frequency kurtosis (0.757; 0.695 under noise 0.10), and phase-neighbor coherence (0.728; 0.678 under blur sigma 2). Three frequency/wavelet features are labeled shortcut; three are specialists; 39 are discarded. Every decision is provisional due to sampling.
- Leave-one-dataset-out exposes failure: canonical CIFAKE AP is 0.506 (normalized 0.012); native WildFake AP is 0.784 against prevalence 0.786 (normalized -0.006). These results argue against interpreting pooled clean AP as transfer.
- The semantic VFM control is `not_run`; chronological-confirmation output is empty; prompt/content-matched and matched-reencoding controls are absent; there is no adequately powered leave-one-generator training experiment.

Do not cite the pilot as evidence that residual/phase features solve the challenge. Its strongest conclusion is that the present data can reward nuisance shortcuts and that native/canonical preprocessing changes conclusions.

## 4. Proposed TRACE-RX architecture (not implemented)

The PDFs under `docs/method_proposal/` propose **TRACE-RX: Transformation-Reliability-Aware AI Image Detection**. Its thesis is: do not force every forensic expert to vote after its signal has been erased; estimate evidence availability and uncertainty, account for correlated errors, and optionally report insufficient evidence.

### 4.1 Experts

1. **One global representation backbone**, selected in a controlled three-way frozen trial:
   - PE-Core-L/14, about 320M vision parameters, with language-aligned real/AIGC provenance prototypes;
   - DINOv3-L/16, about 300M, dense self-supervised structure, but gated/custom license risk;
   - DINOv2-L/14, about 300M, Apache 2.0 reproducible control.
   A shared head/schedule selects one primary using low-FPR and worst-group evidence. The runner-up is fused only once if error sets are complementary.
2. **Native-resolution forensic expert**: select 192--256 pixel patches from high texture, low texture, edges, faces/text, and uniform coverage. Tokenize RGB, Gaussian high-pass residual, block DCT, Fourier phase, and short-offset neighbour relations at multiple scales in a 40--60M transformer with modality embeddings. Aggregate patch logits by mean, variance, upper quantile, and attention; predict heteroscedastic log variance. Cross-patch logit variance is both locality evidence and uncertainty.
3. **Authentic-manifold evidence**: positive similarity/distance to camera, screenshot, scan, artwork, and conventional-CGI subtypes. Missing camera/provenance evidence must never alone imply AIGC.
4. **Controlled intervention responses**: mild deterministic JPEG Q95, blur sigma 0.5, resize 0.95/restore, saturation 0.95, mild denoise, and cross-patch consistency are candidates. For expert feature `h` and logit `z`, use absolute logit change, cosine feature change, and relative L2 feature change as fusion features. They are reliability probes, not independent AIGC votes. Greedy selection uses low-FPR/worst-group gain minus latency and instability.

### 4.2 Reliability fusion and output

For expert logit `z_k`, availability `a_k=sigmoid(g_k(q(x),r(x)))`, predicted variance, and low-rank residual covariance `Sigma`, the proposal gives

`w = Sigma^-1 a / (a^T Sigma^-1 a + epsilon)`, `z_fused = w^T z`, `p_AI = sigmoid(z_fused/T)`.

The system would abstain when entropy/disagreement is high or total availability is low, while still needing a continuous `pred` for competition scoring. Temperature and real/AI/insufficient-evidence thresholds are fitted only on calibration. The included critique recommends logistic stacking as the default and covariance weighting only as an ablation because expert logits are not necessarily unbiased, common-scale estimates and covariance can create negative/unstable weights.

### 4.3 Proposed data, training, selection, and evaluation

The proposal assumes two DGX Sparks and 24 hours. It posits 80,000 independent masters, balanced 40k authentic/40k synthetic, with one fixed redistribution endpoint per master: 160,000 records. Lineage, prompt group, and duplicate component stay together.

- expert train: 72,000 masters / 144,000 records;
- fusion train: 4,000 / 8,000;
- development: 2,000 / 4,000;
- calibration: 1,000 / 2,000;
- locked test opened once: 1,000 / exactly 2,000 (500 real and 500 fake masters).

Architecture ladder: B0--B2 backbone trial; A0 chosen global; A1 + forensic; A2 + authentic prototypes; A3 + selected interventions/reliability; one optional A4 adds the runner-up global backbone. Proposed selection score is 0.35 TPR@1% FPR + 0.25 worst-group AUC + 0.20 unseen-generator AUC + 0.10(1-worst-real FPR) + 0.10(1-ECE). A branch is dropped for less than 0.5 point gain or harm to low-FPR/unseen/worst-real results.

Objective hypotheses are class/source/subtype/generator-balanced BCE, worst-20%-group CVaR, low-FPR partial-AUC ranking against hard authentic examples, and a staged BCE -> CVaR -> pAUC sequence. The proposal monitors gradient conflict. It caches expensive expert outputs, fits fusion separately, then calibrates after all method selection.

Evaluation emphasizes ROC-AUC/AP but prioritizes realised FPR and TPR at a calibration-frozen nominal 1% FPR, Brier/ECE, risk-coverage, and worst predeclared group. Official single transforms, a fractional interaction design, both operation orders, repeated JPEG/resize/blur saturation, unseen generator/source/journey, paired master bootstrap, and error cards are proposed.

### 4.4 Other documented proposals, not implementation

The broad SOTA report recommends a 50/50 controlled corpus, prompt/content/dimension/codec/file-writer matching, generator-balanced GAN/U-Net-diffusion/DiT/autoregressive coverage, 0--3 label-independent transforms, a frozen DINOv2/CLIP/PE-class encoder with LoRA plus an original local/cross-layer head, and held-out generator/family/real-source/transform/time evaluation. Its example DDA-scale corpus is 236k, which is not reconciled with TRACE-RX's 80k-master plan.

The watermark/resolution idea brief rejects `watermark = AI`. It permits a visible-mark detector only as an explanation/counterfactual-tested bounded auxiliary cue. It recommends a quality/uncertainty cascade: cheap full-frame plus 1--2 native patches, exit only when calibrated and stable to a mild probe, and route uncertain cases to more native tiles/stronger VFM. Raw dimensions affect patch budget, not the label or routing decision. This cascade is not yet integrated into TRACE-RX code.

## 5. Exact design decisions the literature review should challenge

| Repository decision or claim | What a paper must establish or the review must challenge |
|---|---|
| Select among frozen PE-Core-L/14, DINOv3-L/16, and DINOv2-L/14, then at most one runner-up fusion | Evidence at the **L/300M scale**, not 1.9B/7B; frozen vs LoRA/adapter; preprocessing parity; public weight/license/release fit; low-FPR and transformed generalization rather than mean AUC. |
| PE prompt/prototype provenance evidence and authentic subtype prompts | Prompt sensitivity, prompt-selection leakage, coverage of camera/screenshot/scan/art/CGI, calibration across content, and whether text alignment encodes semantic/style shortcuts. |
| A new 40--60M multimodal forensic transformer trained from scratch in roughly four hours | Sample efficiency and actual Spark throughput; whether RGB+residual+DCT suffices; whether phase and neighbour features add robust, generator-held-out value; compare a pretrained no-early-downsampling fallback. |
| Native 192--256 patch selection by texture/edge/face/text/uniform rules | Crop and semantic-selection bias, arbitrary-resolution evidence, patch count/coverage, low-resolution behavior, aggregation, and native vs canonical trade-off. |
| Fourier phase as a modality | Translation/crop/resampling sensitivity, correct invariances, and incremental evidence after residual/DCT/global branches. Current pilot phase result is small/confounded evidence only. |
| Authentic manifolds as asymmetric positive-real evidence | Real subtype taxonomy, manifold estimation in high dimension, open-set genuine content, false positives on art/CGI/screenshots, and whether distance absence is prevented from acting as fake evidence. |
| Test-time controlled interventions estimate signal survival | Position against RIGID/test-time-consistency prior art; demonstrate multiple operators and learned availability add value beyond a standalone sensitivity score; prove response is not merely a codec/quality detector. |
| Availability `a_k` inferred from quality and response | Define a training target. The internal critique suggests known transform lineage/branch survival supervision. Test leakage, causal identifiability, unseen transforms, calibration, and collapse modes. |
| GLS-like covariance fusion | Its common-scale/unbiased-logit assumptions, covariance sample complexity, negative weights, low-rank rank choice, and comparison with calibrated logistic stacking, constrained mixtures, and simple averaging. |
| Heteroscedastic expert uncertainty and three-way real/AI/insufficient output | Whether uncertainty is calibrated under generator/source/transform shift; whether abstention improves risk-coverage while always emitting the required `pred`; no silent change of the binary task. |
| Five interventions plus expert reruns | Explicit mean/p95/worst latency and memory budget; marginal value per rerun; stop at one or two probes if the accuracy-latency Pareto frontier does not improve. |
| 80k masters, one clean plus one fixed journey | Actual licensed source composition, duplicate/prompt lineage, generator and authentic-source balance, transform distribution, canonicalization, storage/decoder throughput; fixed endpoint vs on-the-fly training augmentation. |
| 2k-record development and 1% FPR-driven selection | Only about 1,000 real dev masters means roughly ten observations define 1% FPR. Quantify uncertainty and selection overfit; reduce decisions or increase authentic calibration/dev size. The 0.5-point removal rule is likely below seed variance. |
| BCE/CVaR/pAUC/staged objectives after cached expert outputs | Clarify which weights actually update. Cached representations cannot test claims about robust representation learning. Require equal-budget, multi-seed Pareto evidence and conflict diagnostics. |
| Official singles plus selected pairs/order/repeats | Include held-out compositions and real platform/screenshot/recapture journeys; apply identical distributions to labels; evaluate paired survival and not only clean-to-transform averages. |
| Metadata canonicalization/matching | The proposal does not yet state exact re-encoding. Literature should prioritize causal data alignment and counterfactual codec/size/writer swaps over architecture gains. Run the metadata-only gate before model selection. |
| Watermark/provenance appears as an optional independent channel | Competition says not to rely on it and the inference schema has one passive score. Remove from the challenge model or prove it is strictly separate and absent/non-evidence never changes `pred`. |
| 24-hour two-Spark schedule | The current schedule includes three VFM caches, a new tower, architecture/objective/intervention sweeps, transform studies, calibration, and locked evaluation. Papers must include comparable throughput; otherwise define an hour-12 minimum system and kill gates. |
| Originality is reliability-aware fusion of multiple evidence families | Explicitly differentiate from RIGID, test-time augmentation/consistency, mixture-of-experts gating, uncertainty weighting, authentic-manifold work, and hybrid semantic/frequency detectors. Ablate the claimed new mechanism. |

## 6. Repo-specific paper scoring rubric (100 points)

Give each category a 0--5 evidence rating, then multiply by `weight/5`. Use 0 = absent/incompatible, 1 = anecdotal or same-generator only, 3 = relevant controlled evidence with material gaps, 5 = direct, well-powered evidence matching this repository's target. Score the paper's **usable evidence**, not its headline number.

| Category | Weight | A 5 requires | Common reasons to score <=2 |
|---|---:|---|---|
| Cross-generator/source/temporal generalization | 25 | Generator- and preferably family-held-out tests; modern DiT/flow plus older diffusion/GAN; authentic-source holdout; group-balanced metrics and intervals; no random derivative leakage. | Only within-dataset split, one generator, pooled mean, unknown real source, or tuning on test generators. |
| Post-processing robustness | 20 | Clean plus severity sweeps close to the official set; label-independent transforms; composed/unseen/order/repeated or real-platform journeys; paired-lineage analysis and worst-condition reporting. | JPEG-only, augmentation without transformed evaluation, best-case average, no crop/noise/resize, or transformed copies crossing splits. |
| Shortcut and confound control | 18 | Metadata-only baseline; content/prompt/source/size/codec/file-writer matching; identical re-encoding/augmentation across labels; duplicate controls; counterfactual swaps and nuisance ablations. | PNG-fake/JPEG-real, unmatched semantics/resolution, source equals class, no metadata audit, or performance collapses under re-encoding. |
| Compute and hackathon feasibility | 12 | Public sub-2B components; credible training/inference memory, time, FLOPs and latency near two DGX Sparks/24h; cache/fallback plan; small ablatable implementation. | Multi-billion VLM, diffusion reconstruction per image, unreported cost, large ensemble/many test-time reruns, custom tower needing long pretraining. |
| Originality and disqualification safety | 10 | Clear delta from prior art; uses generic public backbones rather than an off-the-shelf AIGC detector; original mechanism is necessary by ablation; licenses permit public release; no watermark dependence. | Directly adopts an existing detector/recipe, novelty is only fine-tuning, unclear license, detector checkpoint reuse, or watermark/provenance shortcut. |
| Interpretability and failure visibility | 7 | Per-branch evidence/reliability, transform survival, real-subtype false positives, calibrated uncertainty/risk-coverage, and representative lineage-aware errors; explanations are validated, not just heatmaps. | One opaque score, saliency only, no false-positive analysis, or causal claims from display brightness. |
| Reproducibility and protocol completeness | 8 | Public code/weights/data provenance; exact splits/hashes/model revision/preprocessing/hyperparameters/seeds; train-selection-calibration-test separation; confidence intervals and runnable inference. | Missing splits or provenance, unreleased code, single seed without variance, test-set tuning, ambiguous metric/preprocessing. |

**Formula:** `total = 5*G + 4*R + 3.6*S + 2.4*F + 2*O + 1.4*I + 1.6*P`, where each letter is its 0--5 rating. A paper can be highly relevant without being safe to copy; record both its total and a one-line adaptation warning.

**Hard red flags independent of score:** test-label/demo-split training; proprietary or unauditable data; a final >=2B model; direct reuse/replication of a pretrained AIGC detector as the solution; watermark dependence; or evaluation leakage. Such a paper may supply a baseline or negative lesson but cannot define the submission.

## 7. Decisions still missing

1. Exact licensed sources and counts for the claimed 40k real/40k fake masters, including authentic subtypes, generator families/versions, prompt/content matching, file writer, near-duplicate components, and licenses.
2. The final sealed modern-generator/chronological window and a powered real-source holdout. AI-GenBench selection is not yet licensed or assigned.
3. The exact canonicalization/matched-reencoding policy and on-the-fly journey distribution. Clean source files currently preserve large class-correlated nuisance differences.
4. Final backbone, revision, preprocessing, license, freeze/LoRA policy, checkpoint hash, parameter count, and whether two backbones fit the rule and latency budget.
5. The minimum native forensic branch: modalities, patch selector/count, tokenization, aggregation, pretrained fallback, optimizer/schedule, and deadline kill gate.
6. Authentic-manifold subtype definitions, prototype construction, distance model, and rule enforcing that missing authentic evidence cannot become positive fake evidence.
7. Availability supervision/target, quality features, calibration, and how to prevent availability from learning codec/source shortcuts.
8. Default fusion. The proposal and its critique disagree on GLS covariance versus logistic stacking. Logit scale alignment, covariance constraints/rank, and negative-weight handling are undefined.
9. Maximum mean/p95 latency, memory, intervention count, patch budget, and the utility coefficients `alpha/lambda/mu`.
10. Which training parameters O0--O3 update, group hierarchy for CVaR, pAUC implementation, seeds, and selection uncertainty.
11. How abstention maps to the mandatory continuous binary `pred`, and whether abstention is only a demo/reporting layer.
12. Final metric/threshold/calibration objective and JSON CLI/container/error-handling choices while organizer requirements remain ambiguous.
13. Screenshot, WebP, re-save, recapture, and held-out journey recipes; adversarial robustness is currently neither included nor explicitly excluded in TRACE-RX.
14. Whether the quality/uncertainty cascade and visible-watermark explanation tag are in TRACE-RX, separate stretch work, or out of scope. Provenance/watermark should not influence challenge `pred`.
15. A precise novelty statement against RIGID/test-time sensitivity, uncertainty-gated mixtures, authentic-manifold detectors, native-patch methods, and semantic/forensic hybrids.

## 8. Highest-value experiments, in order

1. **Data/shortcut gate before neural training.** On the intended manifest, fit metadata-only and tiny pixel-statistic baselines. Perform label-balanced decode/re-encode, size/codec/file-writer swaps, and source/content matching. Do not proceed if nuisance prediction remains strong or architecture gain disappears.
2. **Freeze an adequately powered modern confirmation protocol.** Use whole generator/family, authentic source, prompt/duplicate lineage, and newest chronological holdouts. Aim for >=200 images per generator/source comparison (500 preferred). Seal a final window before feature/backbone selection.
3. **Small equal-protocol global trial.** PE-L vs DINOv3-L vs DINOv2-L with the same preprocessing/head/data and official/composed evaluation; then frozen winner vs one LoRA/adapter run. Report TPR at calibration-frozen 1% FPR, worst group, AP/AUC, calibration, latency, and at least finalist seeds.
4. **Minimum viable TRACE-RX ablation.** Chosen global backbone + a finishable pretrained/native RGB-residual-DCT branch. Compare global, forensic, naive average/concatenation, calibrated logistic stacking, and stacking with one or two probe responses. This tests complementarity before a custom tower or covariance model.
5. **Availability falsification.** Supervise branch availability from known paired signal survival/transform lineage on discovery only. Compare no gate, quality-only gate, response-only gate, learned gate, and oracle gate on **unseen transforms and generators**. Counterfactually swap codec/quality to test whether the gate is a nuisance detector.
6. **Transformation survival matrix.** For each branch, evaluate every official severity, pre-registered realistic chains, held-out compositions, both orders for resize/JPEG and noise/JPEG, and repeated JPEG/resize/blur. Use paired masters and report survival, interaction, reversal, worst lower confidence bound, and per-rerun latency.
7. **Native-resolution and routing ablation.** Canonical full-frame vs native random/texture-diverse patches vs global+patch fusion; then always-cheap, always-strong, dimension rule, uncertainty rule, and quality-aware rule. Stratify by effective quality, not pixels alone, and compare mean/p95 latency Pareto fronts.
8. **Authentic false-positive challenge.** Build powered, source-held-out camera, screenshot, scan, art, conventional CGI, meme/text, and heavily processed real groups. Test prototype/manifold evidence and hard-real pAUC training. Reject a component that materially increases worst-real FPR.
9. **Generator diversity at fixed image count.** Use the implemented few-vs-many cohort planner after license audit. Hold total fake images and real distribution fixed; compare 1/8/64/~200 generators on one sealed modern target. This determines whether the expensive 80k-master mix is allocated well.
10. **Locked operational evaluation.** After choices freeze, fit temperature/thresholds only on calibration; open the final set once; report realised FPR/TPR with exact/Wilson intervals, AP/AUC, Brier/ECE, risk-coverage, worst groups, branch/error cards, parameters, throughput, and mean/p95 latency. Always emit a continuous score even for demo abstentions.

If time is short, experiments 1--5 have the highest information value. They can kill the central method or expose disqualification/shortcut risk before the 24-hour budget is spent on the custom transformer, covariance layer, or a large transform grid.

## 9. Repository documents most relevant to reviewers

- `docs/problem-statement.md`: authoritative requirements and ambiguities.
- `docs/feature-robustness-lab.md`, `docs/aigc-exploration-implementation-plan.md`, `docs/aigc-exploration-reconciliation.md`: executable feature/evaluation contract and honest deferrals.
- `docs/method_proposal/trace_rx_training_proposal.pdf`: unimplemented TRACE-RX design.
- `docs/method_proposal/TRACE-RX_Critique.pdf`: internal feasibility/statistical/prior-art critique; especially availability supervision, logistic stacking, development power, data provenance, LoRA, and minimum viable scope.
- `docs/research/aigc-detection-sota.html`: broad paper synthesis and alternative practical training recommendation.
- `docs/research/idea-validation-watermarks-resolution.html`: bounded watermark cue and quality/uncertainty routing decisions.
- `data/derived/feature_lab/run_metadata.json`, `evaluation_metadata.json`, and the derived CSVs: exact current pilot state.
