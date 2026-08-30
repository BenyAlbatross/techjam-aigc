# Paper notes — arXiv:2512.12982, *Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes*

## Reading scope and citation convention

I read the full **20-page v2 PDF**, including the eight-page supplementary material, rather than only the abstract. Page references below use the printed PDF sequence: **main p. 1–8** and **supp. p. 1–8** (which are PDF pages 13–20). “Paper claim” and “Critique” are kept separate. I also spot-checked the public code to resolve implementation cost and noted code/PDF mismatches separately.

## 1. Metadata and version

- **Title:** *Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes*.
- **Authors:** Ziheng Qin, Yuheng Ji (equal contribution), Renshuai Tao, Yuxuan Tian, Yuyang Liu, Yipu Wang, Xiaolong Zheng (corresponding author).
- **Affiliations:** Institute of Automation, Chinese Academy of Sciences; Institute of Information Science, Beijing Jiaotong University; School of Artificial Intelligence, University of Chinese Academy of Sciences; School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences (main p. 1).
- **Identifier / subject:** arXiv:2512.12982 [cs.CV]; DOI `10.48550/arXiv.2512.12982`.
- **Version history:** v1 submitted 15 Dec 2025 04:58:08 UTC; current PDF examined is **v2, revised 12 Apr 2026 02:37:04 UTC**. The v2 PDF header says `arXiv:2512.12982v2 [cs.CV] 12 Apr 2026`.
- **Venue status:** the PDF is formatted as CVPR 2026, and the linked repository says accepted to CVPR 2026. The arXiv record itself is still an arXiv preprint record.
- **Length:** 8 main pages, 4 reference pages, 8 supplementary pages.
- **Code:** <https://github.com/UltraCapture/GAPL>; repository states MIT. The arXiv manuscript uses arXiv's non-exclusive distribution license.
- **Funding:** multiple National Natural Science Foundation of China grants (main p. 8).

## 2. Core question, motivation, and assumptions

### Paper claims

The paper studies what happens when an AIGC detector changes from “train on one generator” to “train on many,” eventually thousands. Its main observation is a **“Benefit then Conflict”** pattern: adding a few generator domains helps, but adding many can flatten or reduce performance for existing detectors (Fig. 1, main p. 1).

It diagnoses two causes:

1. **Data-level heterogeneity.** A mixture of generators has an extra between-generator covariance term. Generated features therefore spread out and overlap real features more as generator diversity rises (Eqs. 2–4, main p. 3; Fig. 3a, main p. 4).
2. **Model-level bottleneck.** A frozen pretrained encoder cannot learn new forensic traces. It can only move a classifier boundary over its existing representation. End-to-end adaptation becomes more useful, not less useful, at large generator scale (Eqs. 5–6 and Fig. 3b, main pp. 3–4).

The proposed response is **Generator-Aware Prototype Learning (GAPL)**. Its stated philosophy is “turn thousand into a few”: create a small fixed set of real/fake forensic basis vectors, then LoRA-adapt CLIP so every image is represented through those prototypes (main pp. 4–5, Fig. 4).

### Quantitative diagnostic evidence

The controlled GenImage toy sets keep 8,000 fake and 8,000 ImageNet real images while using 1, 2, 4, or 8 generators. The nested generator sets are:

- 1: SD v1.4
- 2: + BigGAN
- 4: + VQDM and Glide
- 8: all GenImage generators

For every ImageNet category, the authors sample `n_s` images per generator such that `1000 × n_s × n_g = 8000`. A final “thousands” condition instead samples about two images from every Community-Forensics training generator, giving about 9,000 fake plus 8,000 real images (supp. p. 1, Table 5).

From Fig. 3 (main p. 4):

| Generators | Real feature trace | Fake feature trace | Frozen mAcc | End-to-end mAcc | Frozen Fisher J | End-to-end Fisher J |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5117 | 0.5168 | 99.2 | 99.9 | 45.03 | 2200.28 |
| 2 | 0.5191 | 0.5967 | 99.3 | 99.1 | 43.12 | 859.56 |
| 4 | 0.5132 | 0.6105 | 93.0 | 98.5 | 19.55 | 128.30 |
| 8 | 0.5162 | 0.6343 | 88.2 | 95.4 | 12.29 | 22.56 |
| Thousands | 0.5864 | 0.6322 | 67.5 | 93.2 | 8.14 | 18.55 |

This strongly supports the narrower claim that a frozen representation becomes inadequate as diversity grows. It is less clean evidence that generator count alone causes monotonic variance growth; see critique below.

### Assumptions behind the method

- Real and generated feature distributions are approximated by Gaussian mixtures whose modes roughly represent semantic categories (main p. 3; supp. p. 3).
- The added between-generator covariance is a main cause of poor separability.
- A supervised 128-D projection can remove much of CLIP's irrelevant semantic structure.
- In class-separated PCA, high-variance eigenvectors capture general forensic concepts; low-variance directions are generator-specific or noise (main p. 5).
- A fixed, small prototype geometry constrains output variance even while the encoder adapts (main p. 5; supp. p. 3).
- Three prototype generator types—GAN, latent diffusion, commercial API—span enough forensic concepts to guide thousands of training generators.

## 3. Exact architecture and data flow

### Stage I — supervised forensic projection, then post-hoc PCA

**Inputs.** The prototype set contains 2,000 fake images from each of ProGAN, Stable Diffusion v1.4, and Midjourney: 6,000 fake total, plus the same total number of real images. Sources are ForenSynths and GenImage (main pp. 5–6).

**Flow.** For image `x`:

1. A frozen **CLIP ViT-L/14 vision encoder** produces its pooled/[CLS] representation `phi(x) ∈ R^1024`.
2. A two-linear-layer head `1024 → 128 → 1` is trained with binary cross-entropy. The 128-D intermediate vector is the “forgery-related embedding.” The paper calls this an MLP, although it describes only the two affine dimensions and does not state an activation (main p. 5; supp. p. 2).
3. After training, embeddings are split into real and fake matrices. PCA is fit **separately per binary class**.
4. Keep the top 32 real eigenvectors and top 32 fake eigenvectors. Concatenate them into a fixed matrix `P ∈ R^(64×128)` (Eq. 8, main p. 5).

No generator-identity label, prototype classification loss, contrastive loss, or prototype update is used. “Generator-aware” comes from which generator images populate the binary fake set. The prototypes are PCA directions, not learned class centers or representative examples.

**Trainable in Stage I:** only the `1024→128→1` head; CLIP is frozen. **Loss:** binary cross-entropy (main p. 5).

### Stage II — LoRA-adapted encoder and prototype matching

**Training data.** Community-Forensics-Small, about 550,000 images while retaining about 4,700 generator identities. The full dataset is described as 5.4M images. Its generator pool includes 12 GANs, 3 pixel-space diffusion models, and roughly 4,000 Hugging Face latent-diffusion models (main p. 6).

**Flow.** For every image:

1. CLIP ViT-L/14 outputs the 1024-D pooled image representation.
2. The retained Stage-I projection maps it to 128-D and it is normalized.
3. Use the image vector as a single cross-attention **query**. Use the 64 fixed PCA vectors as **keys and values**:

   `f_tilde = softmax((f Wq)(P Wk)^T / sqrt(128)) · P Wv`.

   Thus the image selects a dynamic mixture of a fixed small dictionary (Eq. 10, Fig. 4, main p. 5).
4. A linear `128→1` classifier emits a logit. Sigmoid gives the fake probability; 0.5 is used for accuracy (main p. 5; supp. p. 2).

**Trainable in Stage II:** rank-16 LoRA on every CLIP vision self-attention `q_proj`, `k_proj`, and `v_proj`; the 1024→128 projection; cross-attention projections; and final classifier. The base CLIP weights and the 64 prototype vectors remain fixed. LoRA uses `r=16`, `alpha=32`, dropout 0.1 (supp. p. 2).

**Loss:** only binary cross-entropy on the final logit. There is no explicit covariance, prototype-separation, generator classification, or transformation-consistency loss (main p. 5).

### Preprocessing, optimization, and stopping

- Train: random crop to 224×224, zero-pad smaller images (supp. p. 2).
- Test: center crop to 224×224. Baselines retain their native resolution where needed: B-Free 504×504 and Co-SPY 384×384 (supp. pp. 2–3).
- AdamW in both stages, learning rate `1e-4`, weight decay `0.01`.
- Stage I: 20 epochs.
- Stage II: stop when validation accuracy reaches 99.9% or has not improved for 3 epochs.
- Validation: random 5% image split, not a generator-held-out split.
- Hardware: two GeForce RTX 4090 GPUs (main p. 6; supp. pp. 2–3).

### Inference

A single 224 crop passes through CLIP ViT-L/14, the 128-D projection, fixed 64-vector prototype attention, and a binary FC layer. The paper reports no test-time augmentation, ensemble, reconstruction model, frequency transform, or generator-ID prediction. Output is one fake logit/probability.

### Why each part is supposed to help

| Part | Paper rationale | My technical reading |
|---|---|---|
| Frozen CLIP in Stage I | Preserve broad pretrained semantics while finding a basic forensic subspace | Cheap way to initialize a supervised bottleneck without damaging CLIP |
| 128-D binary projection | Remove semantic dominance and expose forgery-related concepts | Most direct supervised dimensional bottleneck; likely does substantial work itself |
| Separate real/fake PCA | Keep common high-variance directions and discard generator-specific/noisy directions | Creates an inexpensive fixed dictionary, but PCA axes are not naturally “concept prototypes” |
| Small fixed prototype set | Bound feature spread independent of number of generators | Cross-attention output lies in a transformed, finite dictionary hull; useful regularization |
| q/k/v LoRA | Escape the frozen-feature ceiling while preserving pretrained knowledge | Parameter-efficient adaptation is the dominant ablated component |
| Cross-attention | Dynamically express each image as a mixture of forensic basis vectors | A learned gated bottleneck with one query; simpler dictionary/MLP alternatives need comparison |
| Two stages | Prevent heterogeneous Stage-II data from defining an unconstrained feature space | Separates dictionary construction from broad adaptation, but introduces extra pipeline complexity |

## 4. Datasets, splits, overlap, and metrics

### Evaluation suites (main p. 6, Table 2; supp. pp. 1–2)

| Benchmark | Real / fake | Subsets/models | Main character |
|---|---:|---:|---|
| ForenSynths | 31k / 31k | 8 | ProGAN, StyleGAN/2, CycleGAN, StarGAN, GauGAN, BigGAN, Deepfake |
| UFD diffusion portion | 8k / 8k | 8 | DALL-E, Glide variants, Guided, latent diffusion variants |
| GenImage | 48k / 48k | 8 | GAN, diffusion, and API models trained around ImageNet |
| SynthBuster | 1k / 9k | 9 | PNG-aligned real/fake; DALL-E, Firefly, SD, Glide, Midjourney |
| Chameleon | 14.9k / 11.2k | 1 | internet/unknown-source, human-indistinguishable images |
| Community-Forensics Eval | 25k / 25k | 21 | newer unseen generators including FLUX, Imagen3, DALL-E, LCM-LoRA, Stable Cascade |

There are **55 test subsets**, of which the authors call **29 completely unseen generator subsets**. The rest may share a generator name with training but differ in generation conditions (supp. p. 1). This is therefore not a pure 55-way unseen-generator test.

Important overlap: the prototype set is sampled from ForenSynths and GenImage, which are also test benchmarks. ProGAN, SD v1.4, and Midjourney are explicitly in Stage I. Broad Community-Forensics training also overlaps some older model families/names. The Community-Forensics Eval set is expressly designed for unseen-generator evaluation, and Fig. 8 labels examples such as DALL-E 2, Firefly, Imagen3, and Midjourney v6 as out of domain (supp. p. 2).

**Metrics:** average precision (threshold-free) and binary accuracy at threshold 0.5. Reported “Mean” is the unweighted mean of the six benchmark-level means, not an image-weighted score: e.g. GAPL's six accuracies average to 90.4 (supp. p. 2; Table 1, main p. 6).

## 5. Main numerical evidence

### Cross-benchmark result

Table 1 (main p. 6):

| Method | Train scale | ForenSynths Acc/AP | UFD | GenImage | SynthBuster | Chameleon | CommFor Eval | Mean Acc/AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B-Free | SD2.1 / 360k | 81.1/94.2 | 87.1/95.3 | 87.3/97.7 | **94.9/98.8** | **78.2/85.2** | 81.5/91.8 | 85.0/93.8 |
| D3 | 9 gens / 2M | 93.0/97.9 | 94.8/99.0 | 95.4/99.2 | 81.3/89.5 | 61.0/56.5 | 73.6/86.0 | 83.2/88.0 |
| Community Forensics | 4.7k gens / 5M | 92.3/98.2 | 94.0/97.0 | 84.0/93.4 | 87.0/94.7 | 77.5/83.5 | 86.8/93.9 | 86.9/93.4 |
| **GAPL** | 4.7k gens / 550k | **97.2/99.5** | 97.2/99.8 | 96.7/99.6 | 91.1/97.2 | 71.0/75.6 | **89.4/97.8** | **90.4/94.9** |

**Paper claim:** GAPL improves average accuracy by 3.5 points over the strongest prior mean and reaches 94.9 mAP across a broad set of GANs and diffusion models (main p. 6).

**Nuance:** it does not dominate every domain. It loses to B-Free on SynthBuster and Chameleon, and to the Community-Forensics baseline on Chameleon. The 71.0/75.6 Chameleon result is far from “universal.” Its mean AP advantage over B-Free is only 1.1 points and over Community Forensics 1.5 points.

### Strictly newer/unseen Community-Forensics Eval

GAPL's 21-subset mean is 89.4 Acc / 97.8 AP. Individual accuracies include DFGAN 99.0, Kandinsky 93.0, Stable Cascade 93.3, LCM-LoRA-SDv1.5 93.9, DeciDiffusion 92.3, FLUX-dev 87.3, FLUX-schnell 88.0, Imagen3 87.0, and Hourglass 78.5 (Tables 10–11, supp. p. 6). High AP with lower accuracy suggests much of the remaining problem is calibration/threshold shift rather than ranking alone.

### Matched-data architecture comparison

All models are reimplemented on the same prototype + scaling dataset, then tested only on ForenSynths, UFD, GenImage, and SynthBuster (Table 3, main p. 7):

- GAPL: **95.5** mean accuracy.
- Swin-T: 89.7; ConvNeXt: 86.2; AIDE: 85.2; Effort: 77.5.
- A plain CLIP ViT: 73.2.
- Several specialized detectors collapse: D3 65.9, UniFD 59.8, NPR 55.4, Co-SPY 50.4.

This is strong architecture evidence under common data, but it excludes the two difficult suites, Chameleon and Community-Forensics Eval, and the paper gives no per-method tuning details or repeated-run variance.

### Strongest ablations

Table 4 (main p. 7), mean over the same four easier benchmarks:

| PCA | Prototype matching | LoRA | mAcc | mAP |
|---|---|---|---:|---:|
| no | no | no | 60.05 | 66.07 |
| no | yes | no | 68.59 | 72.43 |
| no | no | yes | 88.52 | 97.91 |
| yes | yes | no | 71.88 | 82.18 |
| no | yes | yes | 90.35 | 95.40 |
| **yes** | **yes** | **yes** | **95.54** | **98.97** |

The most important fact is that **LoRA is the dominant component**: +28.47 mAcc over the bare baseline. Full GAPL adds +7.02 mAcc over LoRA alone and +5.19 over random-prototype matching + LoRA. The full model's AP is only +1.06 over LoRA alone. Prototype modules chiefly improve fixed-threshold accuracy/calibration, although they also add some ranking value versus random prototypes.

Prototype count is insensitive: 16/32/64 gives 95.28/95.31/95.54 mAcc and 98.98/98.99/98.97 mAP. Prototype-source types matter more: random 90.35/95.40; one type 93.67/96.89; two 94.10/97.18; three 95.54/98.97; four 95.29/98.66 (Fig. 5, main p. 7). Three types are best, but the fourth type is not identified in the paper and there are no error bars.

### Perturbation evidence

The paper tests only JPEG compression and Gaussian blur on GenImage (Fig. 6, main p. 8):

- JPEG qualities plotted: 100, 90, 80, 70, 60, 50. GAPL drops from about 96.7 to about 85.6 mAcc. The authors call this **11.09% degradation**; numerically it is about 11.09 absolute accuracy points.
- Blur sigmas plotted: 0, 1, 2, 3. GAPL is about 95.5 at sigma 1, 84 at sigma 2, and 71.6 at sigma 3: **25.12 absolute points** down at the extreme.
- From the plot, at JPEG Q90 the baselines are roughly UniFD 73, SAFE 50, NPR 56, and AIDE 80, versus GAPL about 92. At blur sigma 1, baselines are roughly 55–75 while GAPL remains around 95.

**Paper claim:** GAPL has the best curve and smallest degradation; frequency-focused methods are fragile. The prose says methods “such as AIDE and SAFE” reach chance under mild JPEG. The plot supports that for SAFE, but AIDE is still around 80 at Q90, so that sentence overstates AIDE's collapse.

**Direct relevance to this challenge:** it covers challenge JPEG Q90/Q70/Q50 and blur sigma 1/2, but not JPEG Q30 or blur sigma 0.5. It gives **no tests for resize, Gaussian noise, color jitter, center crop, WebP, screenshots, or composed transforms**. There are no confidence intervals and no transformed Chameleon/unseen-generator result.

### Interpretability evidence

Average prototype attention is sparse: some prototypes are used much more than others. High-attention image clusters are described as distorted objects/unrealistic lines (fake prototype #36), oversmooth surfaces (#50), complex natural scenes (real #4), and consistent portrait light (real #13) (Fig. 7, main p. 8). Attention maps suggest shallow CLIP semantics are retained while deeper LoRA layers attend to more patches (Figs. 9–10, supp. pp. 7–8). These are qualitative, hand-interpreted examples rather than causal evidence that a prototype represents the named concept.

## 6. Limitations, confounds, and fairness

### Limitations the paper acknowledges

The authors explicitly say behavior on truly new domains and future generator techniques is unknown. Future models may stop leaving detectable artifacts (main p. 8). The supplementary also argues that future systems may need semantic/logical, 3D, and physical reasoning rather than only 2D artifact cues (supp. p. 4).

### Additional critique

1. **The scale experiment confounds count with generator identity/order.** The 1/2/4/8 sets are one nested ordering, not multiple random subsets. The “thousands” point changes from GenImage/ImageNet to Community-Forensics and has about 9k fake vs 8k real. Multiple generator orders and seeds are needed before calling the effect causal.
2. **The variance story is not monotonic at the final point.** Fake trace rises 0.5168→0.6343 through eight generators, then slightly falls to 0.6322. Real trace jumps from about 0.516 to 0.5864 when the dataset source changes. Thus “real heterogeneity remains stable” holds only for the four ImageNet-controlled points.
3. **Feature trace setup is underspecified.** The paper does not clearly identify the exact encoder/layer/normalization used for Fig. 3a. Trace values depend strongly on normalization and dimension.
4. **The theoretical constant is questionable.** The supplement claims `Var(F) ≤ D²/4` for an arbitrary convex mixture of prototypes with diameter `D` (supp. p. 3). In more than one dimension, the stated derivation does not generally justify `1/4`; the immediate iid bound is `Var(F)=1/2 E||F-F'||² ≤ D²/2`. A finite dictionary still bounds variance, but the stronger constant needs extra assumptions.
5. **PCA rationale conflicts with the heterogeneity diagnosis.** The method says excessive cross-generator variance is harmful, then retains the *largest-variance* class directions as universal concepts. That may work empirically, but “large variance = general forensic information; low variance = generator-specific/noise” is asserted, not tested.
6. **PCA axes are not prototypes in the usual sense.** They are centered covariance eigenvectors with arbitrary sign, not class means, exemplars, or generator clusters. Separate real/fake PCA bases can span similar subspaces. The paper does not compare class centroids, K-means, learned codebooks, supervised PCA/LDA, or random orthogonal bases under matched capacity.
7. **“Generator-aware” has no generator supervision.** Stage I only uses binary labels. There is no generator ID objective. This makes the name stronger than the mechanism.
8. **Training/test overlap weakens “cross-generator” wording.** Stage-I sources are also benchmarks; only 29/55 subsets are completely unseen. Results should be split into seen-name, unseen-condition, and unseen-family aggregates.
9. **Random image validation leaks generator domains.** A 5% random split from thousands of generators makes early stopping easy (99.9% validation accuracy) and does not estimate unseen-generator generalization. Hold out entire generators and real-source datasets.
10. **Robustness comparison is confounded by training scale.** GAPL uses 4.7k generators/550k images, while robustness baselines use their SD1.4-subset checkpoints. Similar clean starting points do not equal matched training data or augmentation.
11. **Threshold behavior is underanalyzed.** SynthBuster DALL-E 3 has only 60.1 fake accuracy while AP is 85.3, with real accuracy 90.0 (Table 9, supp. p. 6). This is a practical calibration failure hidden by high macro AP. No calibration curve, ECE, TPR-at-fixed-FPR, or locked deployment threshold is reported.
12. **No repeated-run uncertainty.** “±” in detailed tables is spread across generator subsets, not variation over seeds. The ablation gains of tenths of a point are not statistically supported.
13. **Shortcut controls remain incomplete.** SynthBuster's PNG/content alignment is a useful check, but there is no leave-real-source-out study, deduplication report, codec/metadata scrub, or content-matched test across all training sources.
14. **Fairness is not evaluated.** There are no demographic, geography, camera, content, or art-style strata; no class-conditional false-positive analysis; and no discussion of harm to artists or unusual authentic imagery. Chameleon's 71% accuracy suggests hard authentic-looking content is not solved.
15. **Pure-vs-edited scope is unclear.** Benchmarks include “Deepfake” face imagery, while the paper frames binary generated-vs-real and does not discuss partial edits. For this project's pure-AIGC scope, those samples must be audited rather than blindly imported.

## 7. Implementation cost and reproducibility

### From the paper

- Two-stage training on two RTX 4090 GPUs.
- Stage I: roughly 12k images for 20 epochs.
- Stage II: 550k images and an early-stopping rule, but no reported epoch count, wall time, batch size, memory, FLOPs, latency, energy, or dataset storage.
- CLIP ViT-L/14 is far below the challenge's 2B limit.

### Parameter estimate

The CLIP ViT-L/14 **vision tower alone** is about **303.18M parameters**. Rank-16 q/k/v LoRA over 24 1024-wide vision blocks adds about 2.36M trainable parameters. The 1024→128 projection, 128-D attention, and classifier add about 0.20M. A faithful model is therefore about **305.7M total**, with roughly **2.56M trainable in Stage II (~0.84%)**. This is comfortably below 2B, but inference still pays for the full 24-layer ViT-L tower.

### Code spot-check — not a paper claim

Repository snapshot `ea32aeb2d619b548daffa6db83296ee1aee70605` (2026-03-24) implements four-head `nn.MultiheadAttention`, normalizes the 128-D query, and uses bias-free projection/classifier layers. Its public FP32 checkpoint is about **1.22 GB**, consistent with saving the whole vision tower.

There are important paper/code mismatches that raise reproduction risk:

- Paper says Stage I 20 epochs and validation fraction 5%; committed scripts use 2 Stage-I epochs and the general default validation fraction is 1%.
- Paper documents only resize/crop-style training, but the Stage-II code defaults to a broad random augmentation list including in-memory JPEG, resize/interpolation, multiple crops, flips, rotation, translation, shear, padding, and cutout. This is a major robustness confound not disclosed in the PDF.
- Code uses ImageNet normalization while loading a CLIP backbone; CLIP's own normalization is present but commented out.
- The prototype extraction script refers to missing/renamed items (`new_dataset`, `aproj`) in the inspected commit.
- The model loader defaults to an author's absolute Hugging Face cache path with the portable online load commented out.

The architectural idea is feasible at hackathon scale, but the published pipeline is not yet “clone and reproduce” quality without fixes.

## 8. Architecture-taste lessons for our detector

### Useful lessons

- **Do not freeze a semantic backbone by default at large generator diversity.** Fig. 3 and the LoRA ablation make a strong case that light q/k/v adaptation can preserve pretrained priors while learning forensic evidence.
- **Separate representation capacity from data scale.** Adding more generators is not automatically useful if the model can only bend a linear boundary in a fixed feature space.
- **Use a small, explicit bottleneck.** The successful part of GAPL may be less “PCA concept discovery” and more the regularizing effect of a 128-D gated dictionary.
- **Measure both ranking and operating-point stability.** AP can stay excellent while 0.5 accuracy fails badly. Our API needs a stable score and likely calibration on source-held-out validation.
- **Macro means are not enough.** Worst-generator, worst-transform, real-source false-positive rate, and seen/unseen-family partitions are more discriminating.
- **Data diversity and perturbation diversity should be evaluated orthogonally.** GAPL is strong on clean unseen models but leaves most challenge transformations untested.

### Bad tastes to avoid

- Naming PCA axes “generator-aware concepts” without generator labels or a concept-validation experiment.
- Treating a random image split from 4,700 generators as evidence of universal generalization.
- Using overlapping benchmark sources while reporting one cross-generator mean.
- Claiming a tight theoretical result when only a looser boundedness intuition is secure.
- Attributing robustness to architecture when training data and augmentations are unmatched.
- Shipping a 1.22-GB checkpoint without latency, memory, or quantized alternatives for a demo.

## 9. Adopt / adapt / avoid for this repository

### Adopt

- The **principle** of parameter-efficient q/k/v adaptation of a public vision backbone.
- Generator-family-held-out and real-source-held-out evaluation.
- A compact supervised forensic bottleneck and explicit clean-vs-perturbed score tracking.
- Diverse, properly licensed generator data, with deduplication and source metadata.

### Adapt, with an original contribution

We cannot replicate an existing detector. Adapt the broad lesson, not GAPL's exact PCA + cross-attention pipeline. A promising original direction is a **transformation-consistent multi-scale evidence bottleneck**:

- semantic backbone features plus our own low-level/local evidence branch;
- learned or analytically defined evidence slots tied to *nuisance stability* rather than generator identity;
- consistency loss between clean and JPEG/blur/resize/noise/color/crop views;
- generator- and real-source-balanced training;
- calibrated fusion and uncertainty/worst-view score.

This directly targets the challenge's transformations and differs technically from post-hoc classwise PCA prototypes.

### Avoid

- Exact reproduction of GAPL, which is both disallowed by the challenge and insufficiently tested for our perturbation suite.
- Importing its trained checkpoint or presenting PCA prototype attention as our original contribution.
- Depending on Stage-I overlap with evaluation generators.
- Center-crop-only inference without scale/crop robustness tests.
- A clean-only 0.5 threshold selected from random image validation.

## 10. Discriminating experiments to run

1. **Does the prototype machinery add more than LoRA?** Under identical data, augmentations, parameters, and seeds compare: frozen backbone + head; LoRA + head; LoRA + 128-D bottleneck; random orthogonal dictionary; separate real/fake PCA; centroids; K-means; learned slots; cross-attention vs cosine/MLP gating. Report three seeds. This is the most important falsification test suggested by Table 4.
2. **Causal generator scaling.** Hold total fake images fixed. For each generator count, sample many random generator subsets/orders. Separately run a setting where images grow with generator count. Hold the real source fixed. Plot mean and confidence interval for AP, balanced accuracy, Fisher ratio, and covariance trace.
3. **Strict unseen-family evaluation.** Hold out entire GAN, convolutional diffusion, latent diffusion, diffusion-transformer, and commercial-API families in turn. Also leave one authentic dataset/camera source out. Deduplicate near neighbors before splitting.
4. **Full challenge perturbation matrix.** JPEG Q90/70/50/30; blur sigma .5/1/2; resize .5/.25; Gaussian noise .02/.05/.10; color jitter ±20%; 80% center crop. Test clean, each single perturbation, and realistic compositions. Report clean delta, worst severity, worst composition, AUROC/AP, balanced accuracy, real FPR, fake TPR, and calibration.
5. **Locked-threshold test.** Select one threshold only on a generator- and real-source-held-out validation set. Freeze it. Measure every generator and perturbation. This will expose cases like SynthBuster DALL-E 3 where AP hides deployment failure.
6. **Shortcut audit.** Re-encode both classes identically to PNG/JPEG/WebP; remove metadata; content-match real/fake images; test codec-label prediction; stratify by resolution/aspect ratio; leave dataset source out; inspect nearest train neighbors.
7. **Mechanism test for evidence slots.** Remove or permute slots, measure attention entropy/effective slot count, and intervene on image regions claimed to activate a slot. Compare performance with an equally sized MLP. Qualitative clusters alone are not enough.
8. **Robustness-source ablation.** Cross architecture (plain vs bottleneck) with augmentation (none vs exact challenge transforms). If robustness follows augmentation rather than the prototype module, say so.
9. **Cost benchmark.** Record train wall time, peak VRAM, checkpoint size, batch-1 CPU/GPU latency, throughput, and FP16/INT8 accuracy. Compare ViT-L against a smaller backbone under the same head.
10. **Fairness/error strata.** Break authentic false positives down by people/skin tone where licensed labels permit, geography, art/CG, low light, screenshots, text-heavy content, camera type, and heavy editing. Show representative false positives and false negatives.

## Bottom line

**Paper claim:** diverse generator training needs an adaptive encoder and a structured low-variance space; GAPL combines Stage-I classwise PCA directions with Stage-II CLIP q/k/v LoRA and cross-attention, reaching 90.4 mean accuracy / 94.9 mean AP and strong JPEG/blur curves.

**My assessment:** the strongest reusable evidence is for **LoRA adaptation plus large, diverse data**, not yet for PCA “generator-aware prototypes” as the unique cause. GAPL is a capable ~306M-parameter reference and a valuable warning against frozen features, but generator overlap, random validation, limited perturbations, a questionable variance-bound constant, and code/PDF mismatches prevent direct adoption. For this project: **adopt the scaling diagnosis, adapt compact evidence bottlenecks around transformation consistency, and avoid replicating GAPL.**
