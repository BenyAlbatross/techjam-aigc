# Deep reading notes: data alignment and generator-aware prototypes for AIGI detection

**Scope.** These notes are based on the complete current arXiv PDFs, including their supplementary material, not only the abstracts. Page references are PDF page numbers unless a numbered table/figure is enough to identify the location. I also inspected the authors' public code on 2026-08-30 where the PDF leaves an implementation detail unclear. I label that evidence **code check**, not a paper claim.

**Project lens.** The target repository needs a public, reproducible, under-2B-parameter binary classifier for *purely generated* versus authentic images. The main risks are new generator families and JPEG, blur, resize, noise, color change, and crop. Copying an existing detector is not allowed, so the useful output here is principles to adapt and experiments that support a distinct contribution.

---

# 1. Chen et al., “Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable”

## Bibliographic record and version read

- **Authors:** Ruoxin Chen, Junwei Xi, Zhiyuan Yan, Ke-Yue Zhang (rendered “Keyue Zhang” in the PDF), Shuang Wu, Jingyi Xie, Xu Chen, Lei Xu, Isabel Guan, Taiping Yao, and Shouhong Ding.
- **Affiliations:** Tencent YouTu Lab; East China University of Science and Technology; Peking University; Renmin University of China; Shenzhen University; Hong Kong University of Science and Technology (PDF p. 1).
- **Record:** arXiv:2505.14359, cs.CV. First submitted 2025-05-20. The PDF read is **v6, 2025-10-21**.
- **Publication status:** NeurIPS 2025 Spotlight; PDF footer says 39th Conference on Neural Information Processing Systems (p. 1). The arXiv comment says “13 Pages, 10 figures,” although the generated PDF has the paper, appendix, and checklist.
- **Code:** <https://github.com/roy-ch/Dual-Data-Alignment>.

## One-sentence thesis

**Paper claim:** reconstruction-paired training is only genuinely bias-reducing if the fake reconstruction is aligned to its real source not just in semantic/pixel content but also in compression/frequency statistics; a detector trained on such very hard pairs learns a tight, transferable boundary (Figs. 3, 6–8; pp. 3, 5–6).

## Motivation, argument, and assumptions

### What the paper argues

1. Real/fake datasets often leak format, resolution, and content shortcuts. A classifier can learn “JPEG or variable size means real” rather than a causal generation trace (Fig. 1, p. 2).
2. Caption-to-image, diffusion reconstruction, and VAE reconstruction progressively improve semantic/pixel alignment, but even VAE reconstruction normally outputs clean PNG-like high frequencies that its compressed JPEG source no longer has (Eq. 1–2, Fig. 4, pp. 4–5).
3. SAFE recognizes ordinary VAE reconstructions at 93%, yet its hit rate collapses after only the upper DCT region is masked. The authors take this as evidence that the real/reconstruction task contains an exploitable high-frequency shortcut (Fig. 5, p. 5).
4. Therefore create fake samples at the edge of the real manifold: preserve the VAE trace, match the source JPEG quality, and partially blend the source pixels back into the fake. Such hard, paired negatives should leave fewer non-causal alternatives and force a small boundary around authentic data (Figs. 3, 6–8).

### Assumptions required by that story

- A Stable-Diffusion VAE decoder leaves features shared by many diffusion models and even sufficiently transferable to GAN and autoregressive outputs (p. 6).
- The main spectral mismatch between web photographs and VAE reconstructions is JPEG, and a recoverable/estimable JPEG quality factor is a useful description of it (p. 6).
- Matched JPEG compression removes a nuisance cue without erasing the forensic trace of interest.
- A convex mix `real + reconstruction`, still labelled fake, retains enough generation trace to be a valid hard negative.
- “Closer to real” in MSE, Fourier error, or a t-SNE view predicts a better out-of-domain decision boundary (Figs. 7–8).
- MSCOCO photographs are a broad enough authentic source for a universal detector.

### Critique of the motivation

- The shortcut diagnosis is persuasive, but Fig. 5 is intervention-on-input evidence rather than causal feature attribution: deleting high frequencies also deletes legitimate model fingerprints. It shows sensitivity, not that every removed signal was spurious.
- JPEG quality is only one spectral nuisance. Demosaicing, denoising, sharpening, social-media resampling, phone HDR, and repeated compression can create or suppress similar statistics. The authors acknowledge smartphone computational photography as a limitation (p. 10).
- “VAE artifacts are universal” is much stronger than the evidence warrants. The strong EvalGEN scores are useful evidence, but the training recipe and backbone can learn more than a decoder fingerprint. GAN transfer in Table 6 is mixed.
- Pixel MSE proximity does not itself establish label validity. As the real coefficient approaches one, the fake label becomes label noise. The endpoint degradation in Fig. 10 is consistent with that concern.

## Exact method and data flow

### Offline/paired sample construction

For each MSCOCO real image `x` (pp. 4, 6–7; Eq. 2):

1. **Geometry:** center-crop height and width to the largest multiples of 8 so that the VAE accepts the image and source/reconstruction sizes agree.
2. **VAE reconstruction / pixel-semantic alignment:** encode and decode once with the Stable Diffusion 2.1 `AutoencoderKL`, without diffusion denoising or prompt conditioning:
   `z = Encoder(x); x_vae = Decoder(z)`.
   The PDF writes this deterministically, but **code check** samples `latent_dist.sample()` from the VAE posterior, so the actual reconstruction also has stochastic latent variation. The training set has about **118K real / 118K fake** images (Table 11); this corresponds to COCO train2017 scale.
3. **Frequency/format alignment:** estimate the JPEG quality of each real image and, with 50% probability during training, JPEG-compress its reconstructed counterpart at that same quality. The other half remains uncompressed so fake inputs span JPEG and PNG (p. 7). Despite the introductory phrase “high-frequency fusion,” the core PDF method is matched JPEG compression, not explicit DCT-band replacement (p. 6).
4. **Pixel alignment:** with probability `P_pixel`, replace the fake by
   `x_mix = r_pixel x_real + (1-r_pixel) x_syn`,
   where `r_pixel ~ Uniform(0, R_pixel)` (Eq. 3, p. 6). The selected sample remains in the synthetic class.
5. Apply training crop/normalization and form paired batches that contain a real image and its aligned fake counterpart (supplement p. 15).

The PDF says the model is trained solely on these MSCOCO pairs and that SD2.1 is the reconstruction VAE (p. 7; supplement p. 15). Figure 10 varies `P_pixel`, `R_pixel`, and VAE choice. It says values between 0.2 and 0.8 are stable and endpoints 0 or 1 hurt; SD2.1 is best (p. 10).

### Detector

- **Backbone:** DINOv2 ViT-L/14, about 304M base parameters, well below the challenge's 2B cap (p. 7; supplement Tables 12–13).
- **Adaptation:** LoRA rank 8 plus a binary head; the frozen DINO weights remain fixed (p. 7). The released implementation inserts LoRA into every transformer block's attention `qkv`, attention output projection, and MLP `fc1/fc2`, plus trains the final linear head. That is roughly 3.15M adapter parameters by architecture count, so training memory is much smaller than full fine-tuning.
- **Input:** 336×336. Random crop for training, center crop for validation/test, and padding if an image is too small (p. 7). Supplement Table 12 tests 224–504.
- **Batch/optimizer:** base batch 16, four-step accumulation for effective batch 64, learning rate `1e-4`; eight V100 GPUs. Critically, the supplement says balanced accuracy on **all datasets** is measured every 10,000 iterations and used for early stopping (supplement p. 15). Those are the reported test benchmarks, not a named development set. The PDF does not give optimizer, weight decay, epoch count, selected checkpoint, threshold calibration, or random seeds.
- **Loss:** the PDF does **not** specify an equation or weights for the detector objective. **Code check:** public training code uses `0.5 * BCEWithLogitsLoss + 0.5 * ContrastiveLoss(pos_margin=0, neg_margin=1)` on DINO features, AdamW, and cosine decay. This is an important unreported component, not something that can safely be inferred from the paper.
- **Inference:** center crop 336, DINOv2-LoRA, scalar head, sigmoid. One checkpoint is used everywhere; no target-dataset tuning or threshold adjustment (p. 7). Reported balanced accuracy implies a fixed binary threshold, but the PDF does not explicitly state its numerical value.

### Important public-code differences from the PDF

These matter to reproducibility and to attributing the gains:

- The released loader adds strong generic augmentations: JPEG quality 55–100, Gaussian and pepper noise, Gaussian/median/motion blur, sharpening, color jitter, grayscale, flips, and random crop. It also adds paired down- and up-resized real/fake copies (random factors 0.2–1 and 1–3.5) to every item.
- It offers an extra **frequency blend mix** between real/fake and the checked-in `train.sh` turns it on (`P=0.2, R=0.8`), although this fourth operation is not the three-step PDF method.
- The checked-in pixel mixer defines its random coefficient as the *fake* weight (`a*fake + (1-a)*real`), opposite to Eq. 3's definition of `r_pixel` as the real weight. With its `[0,0.8]` setting, the implemented real weight is `[0.2,1.0]`, not the PDF's `[0,0.8]` interpretation.
- The public code supplies the BCE-plus-contrastive objective omitted from the PDF. It concatenates original plus two resized views per class (six crops per pair), so “batch 16/effective 64” also understates the current code's image-forward count.

Thus a faithful reimplementation of the text and a reproduction of the released checkpoint are different experiments. The robustness result cannot be attributed only to matched JPEG and pixel mixing without separating these augmentations.

## Why each component might work

- **VAE pairing:** holds scene, object, framing, nominal resolution, and source almost constant. It destroys many easy semantic shortcuts and makes the remaining class difference closer to an encode/decode trace.
- **Quality-matched JPEG:** makes compression a within-class variable rather than a label proxy. Randomly leaving half the fakes uncompressed prevents “JPEG means real” from simply reversing.
- **Real-to-fake mix:** weakens any reconstruction artifact that is too obvious. It resembles hard-negative mining or vicinal risk minimization around the real manifold. Moderate mixing can teach a margin; too much makes labels inconsistent.
- **Paired batches:** expose both sides of the local boundary simultaneously and reduce batch-to-batch content imbalance.
- **DINOv2 + LoRA:** DINO features preserve local visual structure better than CLIP's semantic alignment, while LoRA gives enough capacity to surface subtle traces without destroying pretraining. Table 13 strongly supports the need for nonlinear adaptation rather than a frozen linear probe.
- **Contrastive loss and robust augmentations (code, not described method):** explicitly tighten class geometry and expose the exact transformations used in the project's target. They are plausible large contributors to robustness.

## Data, splits, evaluation protocol, and evidence

### Training and new test sets

- **Train:** MSCOCO train images paired with SD2.1-VAE reconstructions; 118K/118K (Table 11; pp. 7, 10).
- **DDA-COCO:** COCO validation real images and quality-aligned VAE reconstructions. Table 1 says 5K real/25K fake and five SD models, while Table 9 displays six fake columns (`XL`, `EMA`, `MSE`, `SD21`, `SD35`, `FLUX.1`). This is an internal count/model inconsistency (pp. 7, 9).
- **EvalGEN:** 553 GenEval prompts, five generators (FLUX, GoT, Infinity, NOVA, OmniGen), 20 samples per prompt/model: 55,300 complete fakes, JPEG Q96. Quantitative tables use only index 0, hence 553×5 = **2,765** fakes (pp. 7; supplement pp. 17–18). There are no real samples. Infinity/NOVA are autoregressive and the remainder include diffusion/hybrid families.

### Eleven benchmarks (Table 1, p. 7)

Seven curated: GenImage (48K/48K; 8 generators), DRCT-2M (5K/80K; 16 diffusion), Synthbuster (1K/9K; 9 diffusion), DDA-COCO, EvalGEN, AIGCDetectionBenchmark (76.25K/76.25K; 17 GAN/diffusion), ForenSynths (36.2K/36.2K; 11 GAN). Four in the wild: Chameleon (14.9K/11.2K), WildRF (500/500), SynthWildx (500/1.5K), BFree-Online (303/641). The latter include internet redistribution and unknown generators.

The primary metric is balanced accuracy. For GenImage, ForenSynths, and AIGCDetectionBenchmark, all synthetic test images are JPEG-compressed at Q96 to reduce format bias. Competitors use official checkpoints; B-Free is omitted from the main table because its code was unavailable (p. 7).

### Overall and cross-generator results

**Paper results, Table 2 (p. 7):** DDA averages **90.7 ± 5.3** across the 11 benchmark aggregates with minimum **81.4**, versus AlignedForensics 75.0 ± 11.1, DRCT 70.1 ± 14.6, C2P-CLIP 62.1 ± 15.6, FatFormer 59.6 ± 14.6, UnivFD 56.3 ± 16.5, AIDE 54.1 ± 12.8, SAFE 47.6 ± 16.0, NPR 46.1 ± 16.1. The `±` is dispersion across dataset/model subsets, not repeated-seed uncertainty.

Notable dataset aggregates:

- GenImage **91.7 ± 7.8**; DRCT 84.7 (Table 4, p. 8).
- DRCT-2M **98.1 ± 1.4**; AlignedForensics 95.5 (Table 3, p. 8).
- Synthbuster **90.1 ± 5.6**; DRCT 84.8 (Table 7, p. 9). DDA is 86.3 DALL-E 2, 90.0 DALL-E 3, 91.9 Firefly, 76.5 GLIDE, 93.5 Midjourney, and about 92.7–93.5 on SD variants.
- AIGCDetectionBenchmark **87.8 ± 12.6** (Table 5, p. 8), but only 52.1 on WFR and 72.5/72.7 on CycleGAN/StarGAN.
- ForenSynths **81.4 ± 13.9**, below FatFormer 90.0 and C2P-CLIP 92.0 (Table 6, p. 8). The paper fairly notes those methods trained on ProGAN contained in this benchmark; DDA also struggles on SeeingDark 58.6 and WFR 52.1.
- In the wild: Chameleon **82.4**, SynthWildx **90.9 ± 3.1**, WildRF **90.3 ± 3.5**, BFree-Online **95.1** (Tables 2 and 8, pp. 7, 9).
- DDA-COCO **92.2 ± 10.6** (Table 9, p. 9). It has 99.0 real accuracy; fake recalls range from 50.2 on FLUX.1 and 68.1 on SD3.5 to 95–99.7 on the other VAEs. This is good evidence of remaining decoder dependence, not universal success.
- EvalGEN fake detection average **97.2 ± 4.2**: FLUX 89.9, GoT 99.5, Infinity 97.8, NOVA 99.5, OmniGen 99.5 (Table 10, p. 9).

**Metric caveat:** EvalGEN has zero real images, so its Table 10 values are fake recall/accuracy, not balanced accuracy. Including that 97.2 as one equal term in the “balanced accuracy over 11 benchmarks” average makes the headline metric conceptually inconsistent and gives a one-class set equal weight to whole paired benchmarks.

### Perturbation evidence

Figure 9 (p. 10) tests GenImage-Q96 under:

- JPEG quality from 100 to 60;
- resize scale 0.5 to 2.0;
- Gaussian blur sigma 0 to 2.

**Paper claim:** at the hardest plotted points DDA beats the second best by **10.5 points at JPEG Q60, 4.1 at resize 2.0, and 5.7 at blur sigma 2.0**. The plotted DDA curve stays in the low 90s through Q60 and resize 2, then falls to the low/mid 80s at blur 2. This is strong relative evidence for those three operations.

**Coverage gap:** no reported Gaussian/shot noise, color jitter, crop severity, repeated JPEG, WebP, social-media pipeline, or compositions of transformations. The released training code explicitly augments several of these; Figure 9 therefore tests a mixture of the proposed alignment principle and direct augmentation exposure.

## Strongest ablations and what they do or do not establish

1. **Backbone/training strategy, Table 13 (supplement p. 17):** frozen linear probes are weak: CLIP variants 74.3–78.6 average and DINOv2-S/B/L only 65.0–66.8. LoRA improves CLIP-L to 83.2; **DINOv2-L LoRA reaches 91.9 ± 6.1** on the five selected aggregates. This is the strongest architecture evidence and says trainable low-level adaptation matters.
2. **Input size, Table 12 (supplement p. 17):** 336 is unusually best: 91.9 average versus 87.1–89.3 for every other tested size. Chameleon jumps from roughly 65.7–73.2 at other sizes to 82.4 at 336. This could be a real scale sweet spot, but the isolated peak also calls for repeated seeds and evaluation without center-crop confounds.
3. **Mix probabilities/strength and VAE, Fig. 10 (p. 10):** moderate `P_pixel` and `R_pixel` are stable; 0 and 1 hurt. SD2.1 is the best reconstruction VAE; SD3.5 is visibly much worse. This supports a moderate-hard-negative regime and reveals source-VAE dependence.
4. **High-frequency masking, Fig. 5 (p. 5):** provides a useful shortcut diagnostic, not a clean component ablation.

**Most important missing ablation:** the PDF never gives a factorial comparison of (a) raw unpaired data, (b) VAE pairs only, (c) matched JPEG only, (d) pixel mix only, (e) both, with identical backbone, objective, and augmentations. It also does not isolate the contrastive loss or the strong released-code augmentations. Therefore the paper demonstrates a powerful recipe but does not identify the independent causal gain of “dual alignment.”

## Limitations, confounds, and fairness

### Stated by the paper

Authentic smartphone photographs increasingly include AI denoising/sharpening/HDR and can show synthetic-like traces; heavy real-world post-processing remains a gap (p. 10). Regional scores vary, so local detection may help (Fig. 13, supplement pp. 18–19).

### Additional critique

- **Test-set leakage through model selection:** the appendix says the detector is evaluated on all reported datasets every 10,000 steps and early stopping is applied (supplement p. 15). With no separate development suite named, checkpoint choice is tuned to the benchmark suite. This materially weakens every zero-shot/generalization number and is the strongest validity concern.
- **Benchmark hyperparameter selection:** input size, VAE, `P_pixel`, and `R_pixel` are also compared on named reported benchmarks rather than a clearly isolated development split (Fig. 10; supplement Tables 12–13).
- **Comparator mismatch:** official checkpoints differ in training data, backbone, image size, augmentations, and objectives. Only the DDA backbone table is controlled, and it does not retrain peer methods on the same pairs.
- **Proposed-set advantage:** DDA-COCO is generated by the same alignment idea and from COCO validation, adjacent to the COCO training domain. It is a diagnostic set, not an independent neutral benchmark.
- **EvalGEN construction:** all prompts come from one prompt suite, all files are Q96, only the first of 20 samples is scored, and no real class exists. Results can overstate real-world calibration.
- **Modified external tests:** all fake images in GenImage, ForenSynths, and AIGCDetectionBenchmark are re-encoded at JPEG Q96 for every method. This is a defensible shortcut control but no longer the native benchmark distribution, and it can remove cues on which peer checkpoints legitimately or spuriously rely. Both native and aligned results are needed.
- **Uncertainty:** no multi-seed error bars or confidence intervals. The NeurIPS checklist explicitly says statistical significance is not reported. The table `±` values reflect heterogeneity among subsets.
- **Data leakage ambiguity:** COCO is common in several benchmark real sources. Image-level overlap/deduplication is not documented.
- **Unspecified quality estimator:** the paper does not say how JPEG quality is inferred or how non-JPEG/multiply-compressed sources are treated. The code consumes a precomputed JSON and defaults missing items to Q96.
- **Artifact survivability:** real/fake pixel blending can create an unnatural double-image/resampling trace of its own and makes fake labels increasingly questionable.
- **Scope:** AI-edited/composite and localized manipulations are not evaluated. That is consistent with this repository's stated pure-AIGC scope, but local heatmaps should not be presented as localization validation.
- **Internal reporting inconsistencies:** DDA-COCO model/count mismatch noted above; EvalGEN called balanced accuracy despite no real class; appendix reports EvalGEN 96.3 in Tables 12–13 while main Table 10 reports 97.2, without explaining checkpoint/evaluation variation.
- **Reproducibility mismatch:** public code contains important loss and augmentation modules absent from the PDF and a frequency mix beyond the named three-step algorithm.

## Implementation cost

- **Data:** download COCO train2017 and produce one 118K-image SD2.1 VAE reconstruction set. Table 11 reports 0.1792 ± 0.0704 s/image and **5.9 hours** for DDA construction, versus DRCT 64.6 h, AlignedForensics 8.73 h, and B-Free 258.79 h (p. 10). This timing omits download/storage and appears measured on unspecified hardware.
- **Training:** authors used 8×V100, effective batch 64. LoRA reduces trainable weights to about 3.15M, but the ViT-L forward/backward and 336 crops are still substantial. A hackathon could train on fewer modern GPUs with gradient accumulation, at longer wall time.
- **Inference:** one ~304M ViT-L pass at 336. No VAE is needed at inference. This fits the <2B limit comfortably but is heavier than a ViT-B/ConvNeXt-T prototype.
- **Engineering:** paired filenames, accurate crop geometry, JPEG quality metadata, deterministic augmentations, and strict split/deduplication are the main complexity. The public code is useful but must be reconciled with the paper before claiming a reproduction.

## Architecture-taste lessons

- Spend novelty budget on **constructing the right contrast**, not stacking forensic branches. Paired hard negatives can be more valuable than a complicated frequency module.
- Treat acquisition/redistribution state as a nuisance variable that must vary within both labels.
- A frozen semantic encoder can be too rigid for subtle forensics; small LoRA adaptation on a self-supervised visual backbone is a good middle ground.
- “Frequency-aware” need not mean a permanent FFT branch. It can mean eliminating frequency shortcuts in the data.
- Prefer minima/worst-family and class-conditional recalls to a single mean. DDA's reduced variance is at least as relevant as its average.
- Test whether a claimed invariant survives **composed** operations. Single-operation curves are not a platform pipeline.

---

# 2. Qin et al., “Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes”

## Bibliographic record and version read

- **Authors:** Ziheng Qin, Yuheng Ji, Renshuai Tao, Yuxuan Tian, Yuyang Liu, Yipu Wang, and Xiaolong Zheng.
- **Affiliations:** Institute of Automation, Chinese Academy of Sciences; Institute of Information Science, Beijing Jiaotong University; University of Chinese Academy of Sciences (PDF p. 1).
- **Record:** arXiv:2512.12982, cs.CV. First submitted 2025-12-15. The PDF read is **v2, 2026-04-12**.
- **Status:** the PDF/arXiv record has no venue line, while the public repository news says accepted to CVPR 2026. Treat that repository statement, not the PDF, as the venue evidence.
- **Code:** <https://github.com/UltraCapture/GAPL>.

## One-sentence thesis

**Paper claim:** naively adding generators first helps and then harms because fake-class variance grows and a frozen CLIP space cannot reorganize around new artifacts; learn a small real/fake PCA basis from three canonical generator types, then LoRA-tune CLIP to map 4.7K generator domains through attention onto that fixed prototype space (Figs. 1–4, pp. 1–5).

## Motivation, “benefit then conflict,” and assumptions

### Paper's diagnostic argument

- A detector trained on one source often generalizes, so the natural next step is to concatenate many generators. Yet performance eventually stagnates or falls (Fig. 1).
- In a toy series, every dataset contains 8,000 fake and 8,000 ImageNet real images, but fake sources increase from 1, 2, 4, 8 GenImage generators to about 4.7K Community-Forensics generators. The fake scatter trace rises while real scatter is stable (Eq. 4, Fig. 3a; main pp. 3–4; supplement p. 13).
- LDA Fisher ratio and validation accuracy fall as source count grows. An end-to-end model retains more separability than a classifier on frozen CLIP, especially at thousands of sources (Eqs. 5–6, Fig. 3b). The plotted 1/2/4/8/thousands conditions give frozen accuracy 99.2/99.3/93.0/88.2/67.5 versus end-to-end 99.9/99.1/98.5/95.4/93.2; frozen Fisher `J` 45.03/43.12/19.55/12.29/8.14 versus end-to-end 2200.28/859.56/128.30/22.56/18.55. Fake scatter trace rises 0.5168→0.5967→0.6105→0.6343 then 0.6322, while real trace is about 0.51 for the GenImage points and 0.5864 after the Community-Forensics source shift (Fig. 3, p. 4).
- Their GMM derivation decomposes fake covariance into within-generator fitting variance plus cross-generator variance, so generator diversity adds a variance term (Eqs. 2–3; supplement Eqs. 11–13).

### Assumptions

- Scatter trace in one chosen encoder space is an adequate proxy for task heterogeneity.
- The real distribution remains stable while the fake mixture expands. This depends heavily on holding the real source fixed; real cameras/web pipelines are themselves heterogeneous.
- Three sources—ProGAN, SD1.4, and Midjourney—span useful canonical GAN, latent-diffusion, and commercial-API concepts.
- Top PCA directions are general real/fake forensic concepts; low-variance directions are generator-specific or noise (p. 5).
- Mapping any sample to a convex attention combination of 64 fixed directions reduces nuisance variance without erasing new-source evidence.
- LoRA can reshape the encoder enough to enter that basis while preserving CLIP's semantic prior.

### Critique of the diagnosis/theory

- Generator count is confounded with generator identity, architecture, source dataset, and (for the 4.7K point) image-source shift. Equal image count is good, but this is not a randomized monotone scaling curve.
- The GMM statement that cross-generator variance grows with diversity is intuitive, not guaranteed: adding a generator near the current mean can reduce or barely change it.
- Real diversity is artificially held fixed in the toy sets. A detector for web media faces camera, scanner, editing, and redistribution diversity on the real side too.
- PCA eigenvectors are directions around a mean, not exemplar centroids. Calling them “generator-aware prototypes” is semantically loose. Separate real/fake PCA does not directly optimize class separation.
- The variance proof (supplement Eqs. 14–16) overclaims `Var(F) <= D^2/4` for an arbitrary multi-prototype vector distribution of diameter `D`. From their own `1/2 sum_i,j w_i w_j ||v_i-v_j||^2`, the immediate general bound is up to `D^2/2`; `D^2/4` holds for two endpoint masses/scalar Popoviciu-style conditions, not arbitrary multi-point geometry (an equilateral three-point support is a counterexample). Also attention value projections are trainable, so the effective geometry is not fixed during optimization. The useful claim is only that a finite fixed-size support prevents variance from scaling directly with generator count.

## Exact architecture and two-stage data flow

### Stage 1: make a forensic subspace and extract prototypes (Fig. 4; Eqs. 7–8, pp. 4–5)

1. Build a **prototype set** with `M=2,000` fake images from each of ProGAN, SD1.4, and Midjourney (6,000 fake total) plus the same number of paired real images (6,000 real; 12,000 total). Sources are ForenSynths and GenImage (p. 6).
2. Center/random crop to 224; pass through frozen OpenAI CLIP ViT-L/14 image encoder. Use the pooled/[CLS] 1024-D representation.
3. Train a two-layer MLP `1024 -> 128 -> 1` using binary cross-entropy. Normalize the 128-D projected feature before the scalar classifier (Eq. 7; supplement pp. 14–15).
4. Freeze the learned `1024 -> 128` forensic projection. Extract its 128-D intermediate features for real and fake subsets separately.
5. Center each subset, compute its covariance/PCA, retain the top `N/2=32` eigenvectors from fake and 32 from real, and concatenate them into fixed `P in R^(64x128)` (Eq. 8).

### Stage 2: scale to thousands of generators (Fig. 4; Eqs. 9–10, p. 5)

1. Train on the 550K-image **Community-Forensics-Small** set, which retains about 4.7K generator identities from the 5.4M original: 12 GANs, 3 pixel diffusion models, and roughly 4,000 Hugging Face latent-diffusion models (p. 6).
2. For input `x`, CLIP ViT-L/14 produces pooled feature `phi_lora(x)`. Apply the Stage-1 `1024 -> 128` projection and L2 normalize it.
3. Make the single 128-D image vector the **query** and the 64 PCA vectors the keys/values in multi-head cross-attention:
   `f_tilde = softmax((f Wq)(P Wk)^T / sqrt(128)) P Wv` (Eq. 10).
   The public code uses PyTorch `MultiheadAttention`, four heads.
4. A bias-free `128 -> 1` fully connected layer emits a logit. Train with BCE; inference applies sigmoid and threshold 0.5 (main p. 5; supplement pp. 13–15).
5. Adapt CLIP only through LoRA on every image-transformer `q_proj`, `k_proj`, and `v_proj`, with rank 16, alpha 32, dropout 0.1. The forensic projection, cross-attention, LoRA, and final head are trainable in Stage 2; the 64 PCA tensor is fixed. The rest of CLIP is frozen (supplement p. 14; public model code).

### Training and inference details

- AdamW, learning rate `1e-4`, weight decay 0.01 in both stages (supplement pp. 14–15).
- PDF: Stage 1 for 20 epochs. Stage 2 until validation accuracy reaches 99.9% or fails to improve for 3 epochs; 5% automatic validation split; two RTX 4090 GPUs (pp. 6, 15).
- 224×224 random crop with zero padding for too-small training images; 224 center crop for GAPL test. B-Free remains at its native 504 and Co-SPY at 384 to avoid handicapping them (supplement p. 15).
- **Code check:** the current example scripts say Stage 1 two epochs and Stage 2 ten epochs, not the PDF's 20/early-stop recipe. They use a 1% Stage-2 validation default, cosine schedule with 20% warmup, AMP, and a random-state augmentation pool that includes JPEG, resize, crop, flips, rotation, translation, shear, padding, and cutout. The PDF describes only random crop. Robustness attribution must account for this mismatch.
- Approximate model size is the CLIP ViT-L/14 vision tower, about 303M, plus small modules—comfortably under 2B. Stage-2 trainable count is roughly 2.36M q/k/v LoRA parameters + 0.13M projection + ~0.066M attention + head, around **2.56M**.

### Inference meaning of “prototype”

There is no generator-ID prediction and no nearest-prototype decision. The 64 directions are a fixed memory. One query attends to all of them, produces one 128-D weighted representation, then a binary head scores it. This is closer to a learned low-rank basis/memory bottleneck than explicit clustering of generators.

## Why the modules might work

- **Stage-1 supervised projection:** removes some semantic dimensions before PCA. PCA on raw CLIP would mostly recover content/style; PCA after BCE training is more likely to contain class-related variation.
- **Separate real/fake PCA:** guarantees representation capacity for both authentic and fake modes rather than letting the larger-variance class dominate a single PCA.
- **Prototype attention:** replaces an irregular classifier over heterogeneous features with a low-dimensional learned mixture of a fixed dictionary. It regularizes and gives the classifier stable coordinates.
- **LoRA:** the largest empirical contributor. It lets deep CLIP attention respond to forensic details while limiting catastrophic drift. Supplement Fig. 9–10 claims shallow semantic attention is preserved and deep attention becomes broader/more artifact-focused (supplement p. 16).
- **Three source types:** inject deliberately distinct priors without making the dictionary itself as heterogeneous as the full 4.7K-source set.
- **Two stages:** separates “choose a compact coordinate system” from “fit many domains into it,” which is easier to optimize and easier to audit than jointly learning free prototypes.

## Datasets, splits, and reported evidence

### Toy scaling sets (supplement Table 5, p. 13)

Each GenImage set is 8,000 fake + 8,000 ImageNet real, class/category balanced. Fake groups are: SD1.4; SD1.4+BigGAN; those plus VQDM+GLIDE; all eight GenImage generators. For each of 1,000 ImageNet categories, sample `n_s` per generator so `1000*n_s*n_g=8000`. The thousands-generator point samples about two images per Community-Forensics generator (~9,000 stated, despite the nominal 8,000 target) and 8,000 reals.

### Main training and six benchmarks

- Stage-1 prototype set as above; Stage-2 Community-Forensics-Small 550K/4.7K generators (p. 6). The Stage-1 data are sampled from ForenSynths and GenImage, which are later used as test benchmarks; ProGAN, SD1.4, and Midjourney are therefore not cleanly unseen domains even if the benchmark test images are disjoint.
- **ForenSynths:** 31K/31K, eight selected GAN subsets.
- **UFD diffusion extension:** 8K/8K, eight diffusion subsets.
- **GenImage:** 48K/48K, eight GAN/diffusion/API subsets.
- **Synthbuster:** 1K/9K, nine aligned diffusion/API subsets.
- **Chameleon:** 14.9K/11.2K internet/unknown, one aggregate.
- **Community-Forensics evaluation:** 25K/25K, 21 diffusion generators intended to be excluded from its training set (Table 2, p. 6; supplement pp. 13–14).

There are 55 test subsets. The authors state only **29 are completely unseen**; the remaining 26 overlap the broad training domain/generator set but differ in generation condition (supplement p. 13). Accuracy uses threshold 0.5 and AP is threshold-free.

### Overall comparison (Table 1, p. 6)

GAPL reports:

| Benchmark | Accuracy | AP |
|---|---:|---:|
| ForenSynths | 97.2 | 99.5 |
| UFD | 97.2 | 99.8 |
| GenImage | 96.7 | 99.6 |
| Synthbuster | 91.1 | 97.2 |
| Chameleon | 71.0 | 75.6 |
| Community-Forensics eval | 89.4 | 97.8 |
| **Mean of six benchmark aggregates** | **90.4** | **94.9** |

The best prior aggregate mean is Community Forensics at 86.9/93.4, so the claimed accuracy improvement is 3.5 points. GAPL is not column-best everywhere: scaling AIDE is higher on UFD and GenImage, B-Free on Synthbuster, and DRCT/B-Free/Community-Forensics are higher on Chameleon. GAPL wins by consistency and is best on ForenSynths and Community-Forensics eval. Its AP gain is much smaller than its accuracy gain: only 1.1 points over B-Free and 1.5 over Community Forensics. The phrase “mean accuracy across 55 subsets” is imprecise: 90.4 is the unweighted arithmetic mean of the six benchmark aggregates, not an equal average of all 55 rows.

On the expressly newer Community-Forensics evaluation set, GAPL's 21-subset mean is 89.4 Acc/97.8 AP. Examples include DFGAN 99.0, Kandinsky 93.0, Stable Cascade 93.3, LCM-LoRA-SDv1.5 93.9, DeciDiffusion 92.3, FLUX-dev 87.3, FLUX-schnell 88.0, Imagen3 87.0, and Hourglass 78.5 (supplement Tables 10–11). High AP with lower accuracy suggests threshold/calibration shift, not only ranking failure.

Detailed tables in supplement pp. 17–21 show, among other cases, ForenSynths 88.2 on Deepfake while near 97–100 elsewhere; UFD 93.4 on Guided versus ~97.7–97.9 elsewhere; GenImage 90.3 Midjourney and 95.0 ADM versus ~98 on most other subsets. These residual drops support reporting generator-family minima.

### Strong controlled architecture evidence (Table 3, p. 7)

The paper reimplements architectures on the same prototype+scaling data and evaluates four benchmarks (FS/UFD/GI/SB):

- GAPL **95.5** mean accuracy.
- Swin-T 89.7, ConvNeXt 86.2, ResNet 73.3, plain CLIP-ViT 73.2.
- Specialized AIGI models: AIDE 85.2, Effort 77.5, D3 65.9, UniFD 59.8, NPR 55.4, Co-SPY 50.4.

This is more diagnostic than the official-checkpoint table. It says the architecture/optimization matters beyond simply owning 4.7K-generator data. However, the PDF gives too little detail about how each peer was adapted and tuned to the shared data to fully reproduce the comparison.

### Robustness (Fig. 6, p. 8)

On GenImage, using checkpoints trained on the SD1.4 subset for a “comparable starting point,” the paper plots JPEG quality 100, 90, 80, 70, 60, 50 and Gaussian blur sigma 0, 1, 2, 3.

- GAPL falls about 96.7→85.6, or **11.09 absolute accuracy points**, at JPEG Q50 and about 96.7→71.6, or **25.12 points**, at blur sigma 3. The paper calls these percentages/degradation, but they are read most naturally as absolute-point changes.
- SAFE drops to chance under modest JPEG/blur; NPR/AIDE/Ojha also degrade more.

This is useful evidence, but it is not the main 4.7K-generator checkpoint, the exact construction of the SD1.4 GAPL checkpoint is underexplained, and the public augmentation pipeline already includes JPEG and related spatial operations. There are no resize, noise, color, crop, or composed-perturbation results.

## Strongest ablations

### Component factorial (Table 4, p. 7; mean over FS/UFD/GI/SB)

| PCA dictionary | prototype mapping | LoRA | mAcc | mAP |
|---|---|---|---:|---:|
| no | no | no | 60.05 | 66.07 |
| no | yes | no | 68.59 | 72.43 |
| no | no | yes | 88.52 | 97.91 |
| yes | yes | no | 71.88 | 82.18 |
| no | yes | yes | 90.35 | 95.40 |
| yes | yes | yes | **95.54** | **98.97** |

Interpretation: **LoRA is the dominant module** (+28.47 mAcc over the bare group). Mapping alone adds 8.54; PCA adds 3.29 without LoRA and 5.19 on top of mapping+LoRA. The AP non-monotonicity (LoRA only 97.91 vs mapping+LoRA 95.40) warns that the decision threshold/calibration and ranking gains differ.

### Dictionary size/source diversity (Fig. 5, p. 7)

- `N=16/32/64` gives mAcc 95.28/95.31/95.54 and mAP 98.98/98.99/98.97. Prototype count barely matters.
- Random vectors: 90.35/95.40. One, two, three, four generator types yield mAcc 93.67, 94.10, **95.54**, 95.29 and mAP 96.89, 97.18, **98.97**, 98.66.

This supports a small diverse dictionary and saturation around three types. It also weakens a literal semantic-prototype story: random vectors plus trainable attention/LoRA are already very strong, and changing 16→64 directions is negligible. The bottleneck/regularization and LoRA may matter more than named forensic concepts.

### Visualization (Fig. 7, p. 8; supplement Figs. 9–10)

The authors cluster images with high attention to selected directions: fake prototypes correlate with distortion lines and oversmooth surfaces; real prototypes with complex natural scenes and consistent portrait lighting. These are plausible qualitative semantics, not proof that the model uses causal generator traces. Some cues are content/style shortcuts of exactly the type the first paper warns about.

## Limitations, confounds, and fairness

### Stated limitation

Future, fundamentally new generator domains remain unknown; artifacts may vanish or change (p. 8). The supplement proposes visual reasoning, 3D/physical constraints, and embodied perception as future directions (p. 16).

### Additional critique

- **Seen/unseen mixture:** 26/55 test subsets are not completely unseen. The headline mean does not isolate the 29 unseen subsets, even though unseen-generator generalization is the main claim.
- **Training-data advantage:** official-checkpoint baselines range from 144K one-generator data to GAPL's 550K/4.7K sources and Community Forensics' 5M/4.7K. Table 3 partly fixes this, but its reimplementation/tuning protocol is sparse.
- **Benchmark averaging:** each of six benchmarks gets equal weight regardless of 1 versus 21 subsets and regardless of sample count. Chameleon's weak 71.0 is diluted by easy legacy suites.
- **Source shortcuts:** Community-Forensics aggregates many Hugging Face pipelines. File encoders, default resolution, uploader, dataset, and prompt conventions can correlate with generator identity and the fake label. No acquisition-matched training ablation is shown.
- **Prototype interpretability:** oversmooth texture, portrait light, and “complex natural scene” can encode style/content or camera priors, not forensic causality. No counterfactual intervention validates the prototype names.
- **Toy diagnostic:** the 4.7K point changes dataset and has about 9K fakes rather than the same exact 8K; one point cannot establish a smooth benefit-then-conflict law.
- **Theory:** GMM and fixed-real assumptions are simplifying; the `D^2/4` vector bound is not generally valid as written.
- **Robustness:** two operations only, one at a time; direct augmentation exposure and checkpoint ambiguity confound the result.
- **No uncertainty:** no repeated seeds, confidence intervals, calibration error, real false-positive rate at fixed fake recall, or subgroup/source bootstrap.
- **Release mismatch:** Stage-1 epochs, validation fraction/stopping, schedule, and augmentations differ between PDF and current scripts. The code uses ImageNet normalization while its CLIP normalization is commented out, has a hard-coded author cache path, and the prototype extraction script in the inspected commit refers to missing/renamed objects. It is not yet a clean clone-and-reproduce path.
- **Model fairness:** B-Free and Co-SPY are tested at their preferred larger resolutions, which is a considerate choice, but compute/FLOPs and latency are not normalized.
- **Data ethics/bias:** no analysis by photographic geography, skin tone, camera/phone model, art/photo content, or web source. For a platform detector, false positives on heavily processed photos and digital art matter.

## Implementation cost

- **Data is the main barrier:** Community-Forensics-Small has 550K images but approximately 4.7K generator identities. It is public, yet download/storage/decoding and license bookkeeping are much heavier than COCO pairing. Stage 1 adds only 12K samples.
- **Compute:** reported training uses 2×RTX 4090. No wall-clock time is given. Two stages and PCA add workflow complexity, but PCA is only on 12K×128 features and is trivial.
- **Model:** about 303M inference parameters; roughly 2.56M trainable in Stage 2. One 224×224 CLIP vision pass + tiny attention is practical and below 2B.
- **Engineering:** must preserve generator/source metadata, construct leakage-safe splits by generator family, train Stage 1, save the projection, extract two PCAs, then train Stage 2. Current code has hard-coded local CLIP cache paths and recipe discrepancies that need cleanup for a public release.
- **Inference:** fixed prototype tensor and one binary head; no generator bank lookup or per-generator calibration. Deployment cost is close to CLIP ViT-L/14.

## Architecture-taste lessons

- Once sources are highly diverse, **frozen features become the bottleneck**. A small trainable subspace can beat a forensic head with a more elaborate fixed input representation.
- A small fixed memory can regularize a heterogeneous fake superclass, but do not overinterpret PCA axes as causal concepts.
- Use a two-stage curriculum when jointly learning the dictionary and mapping would let them drift together.
- Measure family-wise minima and leave-family-out performance; “more generators” is not a sufficient generalization claim.
- A simple ConvNeXt/Swin baseline on the exact same data is mandatory. Table 3 shows generic strong backbones can embarrass specialized methods.
- Prototype count insensitivity is a design hint: try the smallest memory first and spend compute on better source balancing and nuisance control.

---

# Cross-paper synthesis for this repository

## What the papers jointly say

The papers attack opposite failure modes:

- **DDA:** reduce *irrelevant real/fake distance* within each pair so the classifier cannot take shortcuts.
- **GAPL:** compress *relevant but heterogeneous source variation* across many generators so the classifier can maintain a usable boundary.

They agree on a more important point: a frozen general-purpose feature plus a linear head is not enough. Both end at a large pretrained ViT with a few million trainable LoRA parameters. They differ in backbone taste: DDA's controlled ablation favors DINOv2 for low-level traces; GAPL chooses CLIP and builds a forensic projection to recover class-related structure.

The natural temptation is to bolt DDA and GAPL together. That would be an unoriginal replication and may collapse useful variability twice. The project should instead derive a distinct hypothesis and make it falsifiable.

## Adopt

1. **Acquisition-matched evaluation.** Normalize or factorially balance format, resolution, and transformation state across labels. Always report real recall and fake recall separately.
2. **Small trainable adaptation.** Start with DINOv2-B/L or a strong ConvNeXt and LoRA/adapters. This fits the cap and makes controlled ablation cheap.
3. **Paired or near-paired hard negatives.** Pair authentic images with generated images sharing content, crop, and output pipeline when licenses permit. Use them as one stratum, not the entire fake universe.
4. **Generator-family metadata.** Keep family IDs for training samplers and leave-family-out tests even though inference remains binary.
5. **Worst-group metrics.** In addition to AUROC/AP/balanced accuracy, report minimum family balanced accuracy, authentic FPR at fixed TPR, and perturbation worst case.
6. **Transformation stress matrix.** Evaluate clean and severity curves for JPEG, blur, resize, noise, color, and crop, plus realistic compositions.

## Adapt into an original contribution

### Recommended concept: nuisance-conditioned consistency with residual evidence routing

Build one under-2B image encoder with two small heads/paths:

- a **global semantic/structure token** from DINOv2 or ConvNeXt;
- a **local residual token set** computed from high-pass/noise residuals or shallow backbone patches;
- a small **nuisance-state encoder** that predicts/embeds JPEG quality, resize scale clues, blur/noise level, and crop/view state *without using the class label*;
- a learned gate that routes residual/global evidence conditioned on nuisance state;
- train binary BCE/focal loss plus **same-image transformation consistency** and a **nuisance-adversarial or cross-nuisance contrastive loss** so class features remain stable across transformations while the nuisance branch retains transformation information.

This uses DDA's lesson (“do not let frequency state equal class”) and GAPL's lesson (“compress diverse evidence into a stable low-dimensional decision space”) without copying their reconstruction recipe or PCA prototype attention. The original technical claim becomes: *explicitly modeling redistribution state lets the detector distinguish “artifact absent because transformed” from “artifact absent because authentic,” and route to more stable semantic/structural evidence.*

A cheaper alternative is a **transformation-conditioned mixture of 4–8 learned evidence tokens**, not generator-aware PCA prototypes. Tokens represent evidence regimes (clean residual, compressed residual, blurred/low-resolution, semantic/geometry) and are learned end-to-end with leave-generator-family-out meta-validation. This is distinct because the conditioning variable is observable nuisance state, not generator identity.

### Data curriculum

- Phase A: public authentic sets with strict licenses and multiple cameras/sources; public generated sets spanning GAN, VAE diffusion, DiT, autoregressive/API outputs.
- Phase B: for every image, sample two independently composed redistribution views. Apply the same transform distribution to both classes.
- Phase C: add a modest amount of content-aligned real/fake data, but use several reconstruction/generation families and cap its batch fraction so the detector cannot become a VAE detector.
- Batch sampler balances real source, fake generator family, and perturbation severity.

## Avoid

- Do not reproduce DDA's SD2.1-VAE + matched JPEG + pixel-mix recipe as the contribution.
- Do not reproduce GAPL's three-generator PCA dictionary + CLIP cross-attention recipe.
- Do not call PCA eigenvectors semantic prototypes without intervention tests.
- Do not train all authentic files as JPEG and all generated files as PNG, even temporarily.
- Do not select a checkpoint on the organizer demonstration split or use its labels for tuning.
- Do not claim unseen-generator generalization from a mixed seen/unseen average.
- Do not use frequency-only detection. Both papers' curves show how easily it breaks under redistribution.
- Do not treat strong augmentations as a footnote. They must be logged and ablated because they can explain most robustness gains.
- Do not average a one-class fake-only set into balanced accuracy.

# Concrete discriminating experiments

The following experiments can decide between the papers' mechanisms and the proposed direction with hackathon-scale compute.

## E1. Shortcut factorial: alignment versus augmentation

Use one backbone/head and one fixed training set. Train at least:

1. ordinary balanced real/fake;
2. + matched format/size only;
3. + same-label robust augmentation only;
4. + paired VAE hard negatives only;
5. + DDA-like matched JPEG/pixel mix only as a **research baseline**;
6. + proposed nuisance-conditioned consistency;
7. + both paired hard negatives and proposed consistency.

Test leave-generator-family-out and every perturbation. If (6) beats (5) under noise/color/crop and composed transformations while matching JPEG, it supports the new mechanism. Keep losses and training steps identical.

## E2. Does decoder pairing teach a universal trace?

Train paired-hard-negative models separately using SD1.5, SD2.1, SDXL, and a non-SD autoencoder; test a matrix over GAN, diffusion-VAE, pixel diffusion, DiT, and autoregressive/API outputs. Include a no-VAE baseline. A universal trace predicts high off-diagonal transfer. Decoder specialization predicts a strong diagonal and failures like DDA's FLUX.1/SD3.5 rows.

## E3. Nuisance counterfactual test

For the *same image*, create a grid of JPEG×resize×blur severities. Measure:

- variance of the final logit;
- cosine distance of class embeddings;
- nuisance prediction accuracy;
- false-positive/false-negative flips.

Desired behavior: nuisance embedding changes, class embedding/logit stays stable until information is genuinely destroyed. Compare plain augmentation, DDA baseline, prototype bottleneck baseline, and proposed conditioning.

## E4. Unseen-family protocol, not random generator split

Group generators by mechanisms: GAN; latent diffusion sharing SD VAE; independent-VAE diffusion; pixel diffusion; DiT; autoregressive/token; closed API. Hold out the entire family, its prompts, and its image sources. Deduplicate perceptually against training. Report macro family AP/balanced accuracy and worst family. This distinguishes fingerprint memorization from transferable evidence.

## E5. Real-source false-positive audit

Hold out authentic domains: smartphone JPEG/HEIC converted losslessly, DSLR RAW-derived JPEG, social-media downloads, scanned material, digital art, screenshots, HDR/night mode. Test both clean and reposted. Measure FPR with bootstrap CIs. This directly targets DDA's stated smartphone limitation and reveals whether the nuisance branch mistakes computational photography for generation.

## E6. Prototype/bottleneck reality check

Implement a small, non-contribution GAPL-style diagnostic on frozen features and compare:

- learned class prototypes;
- PCA directions;
- random orthogonal directions;
- no memory, same-parameter MLP;
- proposed transformation-conditioned evidence tokens.

Match parameters. Intervene by removing a purported concept (blur texture, alter color/content while preserving acquisition). If PCA and random are similar, its benefit is regularization rather than generator semantics. If nuisance-conditioned tokens win chiefly under transformations, that supports the repository's distinct design.

## E7. Loss ablation

On the chosen architecture compare BCE only, BCE+supervised contrastive, BCE+view consistency, BCE+nuisance adversary, and the full loss. Report clean/unseen/perturbed trade-offs. This is essential because DDA's public code silently adds a 50/50 contrastive objective.

## E8. Crop/locality stress

Evaluate center crop, random crop, 5-crop mean, patch-logit robust aggregation, and full resize. Sweep retained area 100/75/50/25%. Use pure generated and pure authentic images only. DDA's regional heatmaps suggest uneven evidence; a local residual path should degrade more gracefully than a single center token.

## E9. Composed “repost” pipelines

Sample fixed, reproducible chains such as resize down → mild sharpen → JPEG; crop → resize → JPEG; color/filter → noise/denoise → JPEG; and two successive JPEG encodes. Compare both average and worst-chain accuracy. Single-operation robustness does not predict these interactions.

## E10. Scale/cost frontier

Train the same method with ConvNeXt-T, DINOv2-B, DINOv2-L, and (if licensed/available) CLIP-L. Record parameters, trainable parameters, images/s, VRAM, latency, and macro/worst-group performance. Choose the smallest model within 1–2 points of the best worst-group score. Both papers show large backbones help, but neither establishes that ViT-L is cost-optimal.

# Suggested minimum decision gate

Advance the proposed architecture only if, on a generator-family-held-out validation suite that is separate from the organizer demo split, it satisfies all of these relative to a strong augmented DINO/ConvNeXt baseline:

- higher worst-family balanced accuracy;
- lower maximum logit variance across nuisance views;
- no material increase in authentic smartphone/social-media FPR;
- gain under at least four of JPEG, blur, resize, noise, color, crop and under composed repost chains;
- improvement survives matched training steps, parameter count, and three random seeds.

Otherwise keep the simple augmented backbone and treat data curation/evaluation rigor as the primary contribution rather than adding a decorative prototype or frequency branch.
