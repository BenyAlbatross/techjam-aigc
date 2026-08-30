# Skeptical architecture review: TRACE-RX against seven recent papers

**Scope.** This review compares the local full papers listed in the request with the TRACE-RX training proposal and its existing critique. Page references are **PDF page numbers in the local files**, not the papers' printed page numbers. The papers use different data, backbones, metrics, and thresholds. Their headline numbers must not be treated as a common leaderboard.

## Executive verdict

TRACE-RX has the right operational questions but the wrong experiment order.

The proposal's strongest ideas are not the five-modality forensic transformer or the covariance equation. They are the lineage-safe splits, explicit low-FPR objective, asymmetric treatment of missing authentic evidence, per-branch transform-survival analysis, calibration split, and locked test (TRACE-RX proposal, pp. 2–3, 7, 11–15). Keep those.

The seven papers change the build order in four important ways:

1. **Prove that the data are not the detector before adding architecture.** DDA shows that codec, size, content, and frequency mismatches can dominate a detector, and its gains come from changing training pairs rather than adding an inference branch (DDA, pp. 2–7, Figs. 1, 4–8; Table 2). TRACE-RX mentions this risk but schedules the metadata-only gate too late and does not specify sources or canonicalization (proposal, pp. 7, 14–15; critique, pp. 2–3).
2. **Establish a frozen modern-VFM baseline before any LoRA or specialist module.** A frozen DINOv3/PE/MetaCLIP2 probe is a much stronger baseline than TRACE-RX's narrative implies. In one controlled SD-v1.4 protocol, DINOv3-Linear reached 0.940 mean in-the-wild accuracy and 0.972 on AIGIHolmes (Simplicity Prevails, pp. 4–5, Tables 2–3). The same paper found LoRA sharply worse for MetaCLIP2, PE, and DINOv3 on Chameleon (p. 8, Table 10). This directly reverses the existing critique's advice to insert LoRA before A1 (TRACE-RX Critique, pp. 2, 4). It does **not** prove that all LoRA is harmful; GAPL, PPM-CLIP, SDID, DDA, and MIRROR obtain gains with LoRA on other encoders, data scales, and objectives. It makes LoRA a late, guarded ablation, not an assumed improvement.
3. **Test cheap representation and real-reference heads before training a new forensic transformer.** MIRROR reports gains from a real-only prototype memory and reconstruction residual (pp. 7–10, Fig. 4, Table 4). DoU reports gains from frozen multi-stage features with cue bottlenecks and decorrelation (pp. 4–8, Fig. 4, Table 1). Both are cheaper and closer to the selected global encoder than TRACE-RX's from-scratch 40–60M five-modality tower (proposal, p. 6).
4. **Earn reliability-aware fusion.** No reviewed paper validates TRACE-RX's per-image availability estimator or GLS-like logit fusion. DoU actually regularizes aggregation toward equal cue weights (p. 5, Eq. 10), while TRACE-RX proposes unequal per-example reliability weights (proposal, p. 9). These are not mutually exclusive because DoU regularizes internal stage cues during training and TRACE-RX gates experts at inference. But they require an explicit ablation: uniform/simple fusion versus learned static weights versus reliability gating. Reliability gating should be attempted only after experts show complementary, transformation-specific errors.

**Recommended MVP:** one frozen modern VFM, a linear head, a cheap real-reference distance feature, and a pretrained RGB/residual forensic branch, combined by regularized logistic stacking. Use symmetric on-the-fly transformations and source/format-matched data. Do not initially use Fourier phase, NPR, learned covariance, or inference interventions. Add them only when a branch-survival test shows that they improve TPR at the target FPR on source- and generator-held-out data.

## What each paper actually supports

| Paper | Evidence that is relevant to TRACE-RX | What it does **not** establish |
|---|---|---|
| **Dual Data Alignment (DDA)** | Real/fake data commonly differ in JPEG/PNG, size, and content (pp. 2–4, Figs. 1, 4–5). Matching a real image to a VAE reconstruction, applying estimated matching JPEG, and pixel mixup is a data intervention (pp. 5–7, Fig. 6). DDA reports 90.7 mean balanced accuracy across 11 benchmarks and improved JPEG/resize/blur robustness (pp. 7–10, Table 2, Fig. 9). | It does not isolate “causal generator artifacts.” The pipeline changes content, codec, frequency, and pixel mixture together. The synthetic training image is a VAE reconstruction mixed with authentic pixels, which is not the competition's target class of a purely generated image. Results use balanced accuracy, not low-FPR performance or calibration. |
| **MIRROR** | A frozen DINO feature space plus a real-only orthogonal prototype memory, sparse top-k reconstruction, perplexity, and residual can act as positive-real reference evidence (pp. 7–8, Eqs. 1–6; Fig. 4). Real-only memory beats mixed/generated memory in its ablation (p. 10, Table 4). | It does not prove that it recovered “the real manifold,” human cognition, or physical laws. Phase 2 is still supervised on SD-v1.4 fakes and uses a LoRA-tuned DINOv3 (p. 9). The method can learn COCO/source support, not authenticity. Its comparisons mix retrained and official baselines (p. 9, Table 1). |
| **Simplicity Prevails** | Frozen modern VFM features are a mandatory baseline. DINOv3-Linear, PE-Linear, and MetaCLIP2-Linear perform strongly on standard, recent-generator, and in-the-wild sets (pp. 3–5, Tables 1–4). LoRA degrades out-of-domain results in its setup (p. 8, Table 10). It also documents failures under recapture and pure VAE reconstruction (pp. 7–8, Tables 7–8). | It does not prove that synthetic-content exposure caused the capability. The web-versus-satellite DINOv3 comparison changes domain, corpus, and likely scale, and the authors admit that controlled pretraining is out of scope (p. 6) before using causal language on p. 7. Accuracy is not TRACE-RX's low-FPR target. Pretraining contamination and platform/source recognition remain plausible. |
| **GAPL** | Large generator diversity can make a fixed representation harder to separate, and a compact prototype adapter plus LoRA works well when training at 4.7K-generator scale (pp. 1–7, Figs. 1–5; Tables 1, 4). It motivates generator-balanced sampling and explicit tests of whether the backbone can use a diverse corpus. | Its variance derivation does not imply real/fake overlap or detection error (pp. 3–4; appendix p. 15, Eqs. 11–16). The toy study reduces images per generator as generator count rises and reuses a stable real pool (appendix p. 13), confounding diversity with per-domain coverage. Only 29 of 55 test subsets are completely unseen (appendix pp. 13–14). Compactness is therefore not a universal design law. |
| **PPM-CLIP** | Multiple image-conditioned prompt hypotheses and their ensemble can be tested as a semantic-head alternative; visual prompt diversity and MC dispersion may be useful *features* (pp. 3–5, Fig. 2; Eqs. 3–12). | This remains a discriminative classifier with an input-conditioned boundary; it is not generative modeling of image provenance. MC prompt variation is not demonstrated to be calibrated epistemic uncertainty. Figure 1 selects prompts using ground-truth class for its visualization (p. 1 caption). Its headline results use accuracy on ProGAN/SD-v1.4/DRCT protocols and do not test the official transformation matrix or low FPR (pp. 6–8, Tables 1–5). |
| **Diversity over Uniformity (DoU)** | A frozen CLIP can expose multiple stages to small trainable cue bottlenecks. Cue removal consistency and cross-stage decorrelation improved cross-generator accuracy in one SD-v1.4 protocol (pp. 4–8, Fig. 4, Table 1). Effective rank and expert-use concentration are useful diagnostics (pp. 3, 7, Figs. 2–3, 6). | High effective rank is correlated with performance, not shown to cause it. The information-bottleneck discussion is stronger than the implemented loss: the practical term is KL-to-prior plus removal consistency, with classification elsewhere (pp. 4–5, Eqs. 3–6). Uniform cue weights are not tested against signal erasure across the full official transform set. |
| **SDID** | RGB and NPR encoders can benefit from cross-modal contrastive learning and mutual distillation; RGB+NPR exceeds either alone in the GenImage ablation (pp. 6–8, Tables 4–7). It is evidence that relational training can make modalities complementary. | It does not show that NPR survives JPEG, resize, crop, or recapture. Evaluation is accuracy at a fixed 0.5 threshold on GenImage/DRCT/Co-Spy protocols (pp. 5–7, Tables 1–3). Compact intra-class distillation can propagate NPR shortcuts into RGB and increase expert correlation. A DINOv2-L plus ResNet-101 pair is also much heavier than the stated gain alone justifies. |

## Core contradictions and how to resolve them

### 1. Frozen features versus LoRA

There is real conflict in the literature:

- DDA fine-tunes DINOv2 with rank-8 LoRA after constructing aligned data (DDA, p. 7).
- MIRROR uses frozen DINOv3 for its real-prior phase but a LoRA-fine-tuned DINOv3 in the supervised phase (MIRROR, pp. 7, 9).
- GAPL attributes its scaling gains partly to LoRA and reports a large LoRA ablation gain (GAPL, pp. 5, 7, Fig. 4; Table 4).
- PPM-CLIP fine-tunes later CLIP blocks with rank-4 LoRA (PPM-CLIP, p. 6).
- SDID uses LoRA on DINOv2-L (SDID, p. 5).
- DoU freezes CLIP and trains only multi-stage cue modules/prompts (DoU, p. 4, Fig. 4).
- Simplicity Prevails reports that rank-4/8 LoRA reduces Chameleon accuracy from 0.930 to 0.817/0.880 for MetaCLIP2, from 0.959 to 0.719/0.635 for PE, and from 0.914 to 0.803/0.718 for DINOv3 (p. 8, Table 10).

These results are not logically inconsistent. They vary the encoder generation, data diversity, training objective, adaptation location, and metric. The safe architectural decision is:

- Freeze the three TRACE-RX candidates for the first comparison, exactly as proposed (proposal, pp. 4, 10).
- Select using source-held-out and generator-held-out low-FPR metrics, not average accuracy.
- Run **one** LoRA experiment only on the winner, after data matching and the frozen head are stable. Use rank 4 first, early stopping on worst-group/TPR@1%FPR, and three finalist seeds if time allows.
- Keep LoRA only if it improves the full transform matrix and unseen groups without worsening worst-real-subtype FPR. A same-source gain is insufficient.

This replaces the existing critique's recommendation to put LoRA before A1 (critique, pp. 2, 4). That recommendation was plausible from older tuned-detector literature, but Table 10 of Simplicity Prevails is more direct evidence for the modern backbones TRACE-RX plans to use.

### 2. Compact prototypes versus diverse representations

GAPL says thousands of generator modes should be mapped into a few canonical prototypes and a bounded-variance space (pp. 1–5; appendix p. 15). DoU says collapsing evidence into a few dominant directions is the failure mode and explicitly preserves a high-rank, decorrelated representation (pp. 1–5, 7, Figs. 2–4, 6). MIRROR adds a third view: compact **real-only** prototypes define a reference, while deviations carry the detection evidence (pp. 7–10).

The apparent contradiction comes from using “compactness” at different levels:

- GAPL compacts generator variation for a final binary decision.
- DoU preserves diversity among internal cues.
- MIRROR compacts the reference support but keeps the residual.

**Recommendation:** do not compress all TRACE-RX evidence into one prototype space. Use compact prototypes only within a branch:

1. A source/subtype-conditioned real-reference branch may use prototypes and residual distance.
2. A semantic branch may use several prompt prototypes.
3. Preserve separate global, reference, and forensic logits through fusion.
4. Monitor effective rank, pairwise error correlation, and per-condition expert-use entropy. Do not optimize rank as an end in itself.
5. If adding a diversity penalty, apply it to representations or branch dropout, not to force unreliable branches to vote.

This preserves GAPL/MIRROR's useful within-branch structure without violating DoU's warning about a single collapsed decision path.

### 3. Data intervention versus architecture

DDA is the clearest warning that architecture comparisons are meaningless before acquisition mismatches are controlled. TRACE-RX says “every label-correlated acquisition choice is a confound” (proposal, p. 3) but then allocates the first five hours primarily to three backbone caches and a new forensic tower (p. 14). That order lets the fastest shortcut win the backbone trial.

**Evidence-backed change:** before B0–B2, compare at least these data conditions with the same frozen head:

- raw decoded inputs;
- symmetric canonicalization of both classes to a matched container/codec policy;
- symmetric on-the-fly official transformations;
- content/source/generator-balanced sampling;
- an optional paired/aligned training condition.

Do not copy DDA literally as the first training set. DDA's VAE reconstruction and pixel mixup are useful research augmentations, but a VAE reconstruction of an authentic image is outside the challenge's “purely generated versus authentic” target, and pixel mixup creates a hybrid training example (DDA, pp. 5–6, Eq. 3). First take the robust lesson—match non-causal acquisition variables—not the exact label construction.

### 4. Uniform weighting versus reliability weighting

DoU's Eq. 10 penalizes aggregation weights away from uniform (p. 5). TRACE-RX uses availability, uncertainty, and inverse covariance to make weights deliberately non-uniform per image (proposal, p. 9). Uniform weighting protects against training collapse. Reliability weighting protects against a cue erased at test time. Both goals matter.

Use the following hierarchy:

1. **Control A:** standardized-logit mean or fixed equal weights.
2. **Control B:** regularized logistic stacking on out-of-fold expert scores.
3. **Control C:** stacking plus branch dropout/cue-removal consistency.
4. **Experiment D:** low-capacity per-example reliability gate.

Only run D after A–C. Regularize D toward the static stacking weights on clean examples where all branches retain class separation. Allow deviations only where out-of-fold transformation data show branch-specific loss. Report expert-use entropy and maximum branch share to detect gate collapse.

Do not use TRACE-RX's GLS formula as the default. Raw expert logits are not unbiased measurements on a common scale, the inverse covariance can create negative weights, and the proposal does not specify how per-example heteroscedastic variance is trained (proposal, p. 9; critique, p. 2). Calibrate/standardize each expert and default to logistic stacking. Treat low-rank covariance as a stretch ablation.

## Unsupported or overstated causal claims

### Claims in the papers

- **DDA:** Better performance after its combined pipeline does not show that “dual alignment” alone caused transfer. VAE choice, matched content, JPEG handling, mixup, and DINOv2 LoRA all change together (pp. 5–7, Fig. 6). Its high-frequency masking experiment shows sensitivity of SAFE, not that all high-frequency evidence is spurious (pp. 4–5, Fig. 5).
- **MIRROR:** A learned memory bank is not evidence that the system models the complete real manifold or reproduces human active inference. The reference-classifier ablation shows utility of the module, not the claimed cognitive mechanism (pp. 5–8, Figs. 1–4; p. 10, Table 4). The “superhuman crossover” compares a model with dataset-scale supervision against only 50 participants, and the hard subset is selected from generated samples using realism/response time (pp. 7–8, Eq. 7). It is not evidence of production replacement of human experts.
- **Simplicity Prevails:** The paper properly admits that a controlled pretraining study is out of scope (p. 6), but later says the satellite/web result “proves” that capability is entirely contingent on synthetic exposure (p. 7, Table 6). Architecture, natural-image domain, corpus size, and training distribution are confounded. Platform-name similarity in Table 5 may show semantic/source recognition, not provenance reasoning (p. 6).
- **GAPL:** Increasing mixture variance does not mathematically imply poorer class separability. A bounded prototype variance does not imply lower error. The toy data also lower samples per generator as generator count grows (pp. 3–4; appendix pp. 13, 15). “Benefit then conflict” is an empirical result for that construction, not a general law.
- **PPM-CLIP:** Sampling input-conditioned prompts and averaging softmax scores does not change the task from discriminative classification to generative provenance modeling. It yields a nonlinear ensemble boundary. The paper does not compare calibrated uncertainty, Brier score, or risk-coverage, so “embracing uncertainty” is not established (pp. 1–5, 8).
- **DoU:** Effective rank and PCA sensitivity are diagnostic correlations. Table 1 supports the combined training regularizers in one protocol, not the universal claim that higher-rank representations cause generalization (pp. 3, 7–8).
- **SDID:** Cross-generator accuracy does not establish that NPR captures source-invariant intrinsic clues. The paper omits transformation survival; other reviewed evidence shows local/frequency detectors can collapse after transmission (Simplicity Prevails, p. 7, Table 7; GAPL, p. 8, Fig. 6).

### Claims in TRACE-RX

- An intervention response is not an identifiable measure of “signal availability.” A small response can mean genuine invariance, a saturated codec, an already-erased cue, or a locally flat but wrong classifier. TRACE-RX acknowledges saturation (proposal, p. 8) but still names the latent quantity availability.
- Cross-patch logit variance is not specifically crop robustness. It also measures normal content heterogeneity and patch selection bias (proposal, pp. 6, 8).
- Heteroscedastic variance is not uncertainty unless trained with a proper probabilistic loss or validated against conditional error. The proposal only says the expert predicts log variance (p. 6).
- A low-rank covariance of residual errors does not automatically prevent double counting, especially when fitted on only 8,000 fusion records and when experts are biased, uncalibrated logits (pp. 7, 9; critique, p. 2).
- “Authentic manifold distance” is positive-real evidence only inside the coverage of its training sources. Distance from COCO-like prototypes can mean artwork, screenshot, scan, CGI, a new camera pipeline, or an unseen country/domain rather than AI (proposal, pp. 3, 15; MIRROR, p. 9).
- Abstention can reduce reported risk at coverage, but the required inference still needs a continuous score for every image. It does not repair ranking on the hidden binary metric (proposal, pp. 9, 13; critique, p. 3).

Use narrower language in the implementation: **estimated branch usefulness**, **reference-support distance**, **intervention sensitivity**, and **empirical disagreement**, not signal survival, manifold truth, or uncertainty unless directly validated.

## Protocol non-comparability

| Work | Training protocol | Main reporting | Main mismatch with TRACE-RX |
|---|---|---|---|
| DDA | DINOv2 + rank-8 LoRA; MSCOCO paired with VAE reconstructions, estimated JPEG alignment, pixel mixup (p. 7) | Balanced accuracy; official checkpoints; some test synthetic images JPEG-96; own DDA-COCO/EvalGEN (pp. 7–10, Tables 1–10) | Different training labels/data intervention; no TPR@1%FPR, calibration, or official six-transform matrix. |
| MIRROR | Real-only phase on stated “200k” MSCOCO images; supervised LoRA DINOv3 phase on GenImage SD-v1.4/JPEG-96 (p. 9) | Balanced accuracy; JPEG robustness at Q90 and resize at 0.9; mixture of retrained and official baselines (p. 9, Tables 1–2) | Source-heavy one-class phase, mild transform summaries, no low-FPR or calibration; baseline training is not uniform. The “200k MSCOCO” count is ambiguous because it exceeds the usual number of unique COCO images. |
| Simplicity Prevails | Frozen heads, GenImage SD-v1.4, two epochs, native model resolution, no augmentation (pp. 3–4) | Accuracy on standard/in-wild/recent sets; separate recapture and reconstruction tests (pp. 3–8) | Strong baseline evidence, but no TRACE-RX source mixture, low-FPR threshold, or calibration. Pretraining exposure may overlap web sources. |
| GAPL | CLIP-L + LoRA; 550K Community-Forensics images spanning about 4.7K generators; prototype seed set (pp. 5–6) | Accuracy/AP over six benchmarks; only 29/55 generator subsets completely unseen (appendix pp. 13–14) | Roughly 3.4x TRACE-RX's record count, partial generator overlap, different source distribution; robustness uses separate SD-v1.4 checkpoints and JPEG/blur only (p. 8). |
| PPM-CLIP | CLIP-L rank-4 LoRA, one epoch; benchmark-specific ProGAN, SD-v1.4, or DRCT protocols (pp. 6–7) | Accuracy and throughput; prompt sampling ablations (pp. 6–8, Tables 1–7) | No common training set across headline tables, no official transform sweep, low-FPR, or calibration evidence. DR/inpainting is partly outside challenge scope. |
| DoU | Frozen CLIP-L multi-stage features; same 320K SD-v1.4 protocol for compared models (pp. 4, 7–8) | AP/accuracy and JPEG/blur curves (pp. 6–8, Fig. 8, Table 1) | Better internal comparison, but one training generator and only two distortions; no source-held-out low-FPR or calibration. |
| SDID | DINOv2-L LoRA + ImageNet ResNet-101 NPR branch; GenImage/SD-v1.4 or DRCT/SD-v1.4 protocol (pp. 5–7) | Accuracy at threshold 0.5 (pp. 5–7, Tables 1–3) | No official transform survival, class/source-held-out evaluation, calibration, latency table, or low-FPR result. |
| TRACE-RX | Planned 80K masters/160K records, mixed real subtypes and generators, lineage-safe partitions (proposal, p. 7) | TPR@1%FPR, worst-group AUC/FPR, ECE, Brier, risk-coverage, master bootstrap (pp. 10, 12–13) | Its protocol is more operational, but its small dev/calibration real counts make 1% FPR selection noisy (critique, p. 2). Literature accuracy deltas cannot set its 0.5-point removal rule. |

Consequently, do not choose an architecture because it has the highest paper accuracy. Reimplement only a small number of principles under one TRACE-RX manifest, decoder, split, transformation implementation, and metric suite.

## Shortcut, contamination, and leakage risks

### Data and preprocessing

1. **Codec/format/size leakage.** DDA documents the problem directly (pp. 2–4, Figs. 1, 4–5). Canonicalize symmetrically or deliberately randomize codec independent of label. Do not compress only fake images in the final training path.
2. **Transformation-implementation leakage.** If all train-time JPEG/resize endpoints and all inference interventions use one library and parameterization, the gate can recognize the library's signature rather than cue survival. Use at least two codec/resize implementations on development or reserve an implementation-held-out test.
3. **Fixed journey leakage.** One fixed journey per master lets the model memorize a transformation regime. Use on-the-fly symmetric augmentation for expert training and fixed endpoints only for evaluation. The existing critique makes the same point (p. 2).
4. **Content/source leakage.** COCO real versus generator-web fake is an easy semantic/source task. Match content strata, camera/non-camera real subtype, geography if available, and resolution. Leave entire authentic sources out.
5. **DDA target mismatch.** A VAE reconstruction of a real and an authentic/synthetic pixel mix are not purely generated challenge examples. Use them as an auxiliary invariance/debiasing view, not as the only positive definition.
6. **Modern camera pipelines.** Smartphone enhancement can create synthetic-like residuals, noted by DDA itself (p. 10). Hold out phone models/pipelines and include computational-photography hard negatives.

### Backbone and pretraining

7. **Web pretraining contamination.** Modern VFM strength may include seen benchmark images, near-duplicates, platform templates, or generator-name associations. Deduplicate local masters against known public benchmark hashes where possible. Compare a language-aligned encoder with DINOv3 under source-held-out data; agreement does not prove independent evidence.
8. **Semantic provenance shortcut.** PE/MetaCLIP prompt similarity to “AI generated” may recognize style, watermarks, captions rendered in images, or platform-specific visual tropes (Simplicity Prevails, p. 6, Table 5). Evaluate content-matched real/fake and remove visible platform marks.
9. **LoRA source overfit.** A LoRA gain on SD-v1.4 can overwrite broad pretrained structure. Require unseen-generator, unseen-source, and transformed improvement, as Table 10 of Simplicity Prevails warns (p. 8).

### Branches and fusion

10. **Real-manifold coverage.** A real-only memory trained on COCO can call scans, screenshots, digital art, medical images, or conventional CGI anomalous. Build subtype-conditioned prototypes and require leave-one-real-source-out false-positive results. Never interpret “far from real memory” as AI by itself.
11. **Prototype generator leakage.** GAPL builds prototypes from named ProGAN/SD-v1.4/Midjourney sources and evaluates 26/55 subsets with some training-domain overlap (pp. 4–6; appendix p. 13). Generator IDs may be used for balancing, but no generator label or source-specific head should become an inference requirement.
12. **High-frequency shortcut.** PPM-CLIP's PWCL assumes top-frequency patches are informative (pp. 3–4), while DDA shows frequency richness can be a codec artifact (pp. 4–6). Any DCT/residual branch must pass symmetric codec and source-held-out controls.
13. **NPR fragility and cross-modal contamination.** SDID's mutual distillation can transfer a brittle NPR decision into RGB. Before adopting it, test RGB-only, NPR-only, late fusion, and distillation under JPEG/resize/crop. Do not infer complementarity from clean accuracy.
14. **Patch-selector content leakage.** TRACE-RX's high/low texture, face, and text patch selection can correlate with class/source (proposal, p. 6). Include random-grid and uniform-spatial controls. Train the selector without label access and report selection frequencies by class/source.
15. **Availability-as-codec detector.** A learned gate can identify JPEG blocking and map it directly to a class-weight pattern. Train on label-symmetric transforms, hold out codec implementations, and test whether gate output predicts label after conditioning on expert scores.
16. **Intervention-induced cue.** Response to TRACE-RX's own JPEG/resize/blur can reflect starting container, resampler phase, dimensions, and saturation, not forensic information (proposal, p. 8). Interventions are stress features, not independent causal probes.
17. **Correlated experts.** RGB, residual, DCT, phase, and NPR are deterministic functions of the same pixels. SDID's two modalities become explicitly aligned; PPM/DoU encourage shared decision structure. Measure error correlation per transform and group. Branch count is not evidence count.
18. **Selection leakage and low-FPR noise.** About 1,000 authentic development masters produce only about ten errors at nominal 1% FPR. Repeated branch, loss, intervention, and transform selection will overfit development (critique, p. 2). Use fewer gates, bootstrap deltas, and a coarser removal threshold tied to seed variance.
19. **Organizer demo leakage.** The demonstration split is COCO-real/DALL-E-fake and has extreme source confounding. Keep it out of all training, selection, calibration, and threshold tuning, as the proposal says (p. 15).

## Revised architecture recommendation

### Minimum viable architecture

1. **Decode and audit layer**
   - Preserve the original path and dimensions for reporting only.
   - Decode to RGB through one documented path.
   - Apply label-symmetric canonicalization/augmentation.
   - Log codec, dimensions, estimated quality, and source for audits. Do not feed raw metadata to the class head.

2. **Global branch**
   - Trial frozen PE-L, DINOv3-L, and DINOv2-L with the same standardized linear head, as TRACE-RX already proposes (pp. 4, 10).
   - Keep the selected encoder frozen for the MVP.
   - Add a cheap multi-stage/token-statistics adapter only after the linear probe. A DoU-like cue-removal/decorrelation variant is one bounded experiment, not the baseline.

3. **Positive-real reference feature**
   - On the selected frozen space, fit subtype- and source-balanced real prototypes or a shrinkage Mahalanobis/kNN distance.
   - Use distance, reconstruction residual norm, and prototype entropy as **features**, not a one-class verdict.
   - Compare against the plain global head and require lower worst-real-subtype FPR. A 4,096-vector MIRROR memory is unnecessary for the first test; its exact capacity was tuned to its own data (MIRROR, p. 11, Fig. 5).

4. **Forensic branch**
   - Start from the critique's pretrained no-early-downsampling fallback, not a new 40–60M transformer (critique, p. 3).
   - Initial input: native RGB plus a high-pass residual. Add block-DCT only as a separate ablation.
   - Defer Fourier phase and NPR. Phase is shift/crop sensitive; SDID gives NPR cross-generator evidence but no transformation-survival evidence.
   - Use uniform spatial patches plus a random-grid control. Aggregate mean, variance, and a robust quantile. Cross-patch variance is an empirical dispersion feature, not “crop robustness.”

5. **Fusion**
   - Calibrate or standardize each out-of-fold expert logit.
   - Default to L2-regularized logistic stacking with expert logits, reference-support features, quality variables, and cross-patch dispersion.
   - Compare equal-weight averaging and branch dropout/cue-removal consistency.
   - Add a low-capacity reliability gate only if expert rankings cross under transformations. Train it on out-of-fold transformed losses. Call its target “relative branch usefulness,” not availability.
   - Do not use covariance inversion or predicted variance until proper-loss training and numerical stability are demonstrated.

6. **Output**
   - Always emit a continuous `pred` score.
   - Abstention is an extra reporting flag/region, not a third organizer-facing class unless the interface is clarified.

### Stretch components, in order

1. One guarded rank-4 LoRA run on the winning modern VFM.
2. One DoU-style multi-stage diversity adapter.
3. One DCT or NPR modality, chosen by official-transform survival.
4. At most two inference interventions, selected under a hard latency budget.
5. Learned reliability gate.
6. Low-rank covariance fusion.
7. Full custom forensic transformer.

This order is almost the reverse of the current risk profile. The least supported and most expensive claims come last.

## Literature-driven experiment order

### Gate 0 — protocol and confound audit, before GPU model selection

- Freeze manifests by master, derivative, prompt group, and duplicate component.
- List actual authentic sources, subtypes, camera/phone distributions, generator families, prompt sources, codecs, dimensions, and licenses.
- Run file/metadata-only and low-resolution/color-histogram baselines.
- Run duplicate and near-duplicate checks across all partitions.
- Define symmetric canonicalization and on-the-fly journey sampling.
- Add codec/resizer implementation holdouts.

**Exit:** metadata/source baselines are near chance after balancing; no cross-split lineage; every group has enough masters for the metrics used. This moves the critique's metadata gate from “before hour 5” to the first gate (critique, p. 2).

### Gate 1 — frozen modern-VFM baseline on data ablations

For PE-L, DINOv3-L, and DINOv2-L, train the same linear head under:

1. raw inputs;
2. canonicalized inputs;
3. symmetric on-the-fly official transformations;
4. balanced source/generator/content sampling;
5. optional paired/aligned auxiliary views.

Report TPR at calibration-selected 1% FPR, worst-real FPR, unseen-generator AUC, Brier/ECE, and transformation survival. Cache only after the data path is selected; otherwise cached features bake in the wrong preprocessing.

**Why first:** DDA makes data intervention the leading hypothesis; Simplicity Prevails makes the frozen VFM the leading architecture baseline.

### Gate 2 — cheap representation structure

On the selected frozen backbone, compare:

- linear pooled-token head;
- pooled plus token mean/variance;
- real-reference distance/entropy;
- one multi-stage cue adapter with and without diversity/cue-dropout regularization.

Use source-held-out folds. Monitor effective rank and error correlation, but select on operational metrics.

**Why before forensics:** MIRROR and DoU offer evidence that pretrained-space structure can add value without a new pixel tower.

### Gate 3 — forensic evidence survival, one modality at a time

Train the pretrained native-patch fallback on RGB/residual. Then independently add DCT, NPR, and phase only if time permits. For every candidate, measure:

- clean and official-transform `d'`/AUC;
- TPR@1%FPR and worst-real FPR;
- generator/source holdouts;
- error correlation with the global and reference branches;
- latency.

Drop a modality if it adds only clean accuracy, if it reverses under JPEG/resize/crop, or if its fused gain is within bootstrap/seed noise. This is where SDID's NPR hypothesis is tested rather than assumed.

### Gate 4 — simple fusion controls

Fit equal-weight, regularized logistic stacking, and stacking with branch dropout on the dedicated fusion partition. Use out-of-fold expert scores to avoid optimistic stacking. Do not tune covariance or interventions yet.

**Exit:** at least two branches have complementary held-out errors and a stable fused low-FPR gain.

### Gate 5 — LoRA and objective ablations

Run one LoRA adaptation of the winning VFM against its frozen counterpart. Then compare balanced BCE and, if statistically stable, CVaR/pAUC on the actual trainable parameters.

Clarify the current inconsistency: if experts are trained once and only cached heads/fusion are swept, CVaR/pAUC do not shape the expert representation (proposal, p. 11; critique, p. 3). State whether each objective applies to a head, adapter, LoRA, forensic branch, or fusion.

**Keep LoRA only** after unseen-source/generator/transformation and low-FPR success. This late gate reflects the direct modern-VFM warning in Simplicity Prevails without ignoring positive LoRA results under GAPL/SDID/PPM/DDA protocols.

### Gate 6 — reliability and intervention tests

First determine whether branch rankings actually cross by condition. If not, stop: static stacking is sufficient.

If they do:

1. Train a low-capacity gate from image-quality and out-of-fold intervention-response features.
2. Compare uniform/static stacking versus gated stacking.
3. Test only JPEG q95 and blur 0.5 initially.
4. Hold out a codec/resizer implementation and at least one transform composition.
5. Report gate collapse, label predictability, latency, and marginal low-FPR gain.

Do not supervise a per-image “signal survived” label as if it were observed. A defensible target is out-of-fold branch excess loss or correctness under a known transform, with group/source balancing. Even that estimates usefulness, not physical observability.

### Gate 7 — calibration, abstention, and locked test

Freeze architecture, weights, and selected interventions. Fit calibration and any insufficient-evidence threshold on calibration only. Open the locked test once. Always retain a continuous score for all images. Use master-clustered intervals and state that 500 real test masters cannot certify production 1% FPR (proposal, pp. 7, 13–14).

## Concrete ablation matrix

The smallest convincing matrix is:

| ID | Data | Global | Reference | Forensic | Fusion | Purpose |
|---|---|---|---|---|---|---|
| D0 | raw | frozen winner, linear | – | – | – | Expose raw shortcuts |
| D1 | matched/canonical | same | – | – | – | Measure data-only effect |
| D2 | D1 + symmetric online journeys | same | – | – | – | Robust training baseline |
| R1 | D2 | same | real-reference features | – | logistic | MIRROR principle |
| V1 | D2 | multi-stage adapter | – | – | – | DoU principle, no diversity loss |
| V2 | D2 | multi-stage adapter | – | – | – | Diversity/cue-dropout delta |
| F1 | D2 | frozen winner | – | RGB/residual | logistic | Minimal two-branch system |
| F2 | D2 | frozen winner | reference | RGB/residual | logistic | MVP TRACE-RX |
| M1 | D2 | frozen winner | reference | + DCT **or** NPR | logistic | One forensic modality test |
| L1 | D2 | rank-4 LoRA winner | reference | chosen | logistic | Guarded adaptation |
| G1 | D2 | chosen | chosen | chosen | branch dropout/static | Anti-collapse control |
| G2 | D2 | chosen | chosen | chosen | learned gate, no interventions | Reliability value |
| I1 | D2 | chosen | chosen | chosen | gate + two interventions | Intervention marginal value |
| C1 | D2 | chosen | chosen | chosen | covariance fusion | Stretch only |

Run the backbone/data rows before building F1. Run only finalists for multiple seeds. Replace TRACE-RX's fixed “drop below 0.5 point” rule with a paired master-bootstrap interval and a minimum effect larger than observed seed variance.

## Final decision rules

- **Keep frozen VFM as primary** unless LoRA wins on low-FPR, worst group, and official transforms.
- **Keep real-reference features** only if leave-one-authentic-source-out FPR improves. Otherwise they are source support, not provenance evidence.
- **Keep a forensic modality** only if it has positive survival and incremental fused value after canonicalization.
- **Keep diversity regularization** only if it improves held-out operational metrics; high effective rank alone is not success.
- **Keep reliability gating** only if expert rankings cross by condition and gating beats static stacking under implementation-held-out transforms.
- **Keep inference interventions** only if their paired gain survives latency penalty and does not increase worst-authentic FPR.
- **Do not keep covariance fusion** merely because it looks principled. It must beat regularized stacking across seeds and remain numerically stable.

## Bottom line

The literature does not justify a larger TRACE-RX. It justifies a more disciplined one.

- DDA says **fix the data first**.
- Simplicity Prevails says **respect the frozen modern representation** and distrust automatic LoRA gains.
- MIRROR says **test positive-real reference support**, but not as proof of authenticity.
- GAPL says **balance and structure generator diversity**, but its compactness claim is protocol-bound.
- DoU says **do not collapse all cues into one direction**.
- SDID says **cross-modal training can help**, but NPR must earn its place under transformations.
- PPM-CLIP says **multiple semantic hypotheses may be useful**, but prompt sampling is not calibrated provenance uncertainty.

The defensible technical contribution is therefore: **a small set of independently validated evidence branches, trained on label-symmetric matched data, fused only after their transformation-conditioned usefulness and correlation are measured.** That is narrower than the current diagram, but it is more original, more testable, and much more likely to finish within the stated compute window.
