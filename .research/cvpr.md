# CVPR 2026 reading notes: PPM-CLIP and *Diversity over Uniformity*

**Project context.** These notes read the two papers as design evidence for a binary, pure-AIGC-versus-authentic detector that must generalize to unseen generators and survive JPEG, blur, resize, noise, color jitter, and crop. The challenge forbids merely copying an existing detector. Therefore the final section extracts principles and proposes experiments, not a reproduction recipe.

## Sources and citation convention

- Xinyuan Wang, Yingxin Lai, Zhiming Luo, and Zhihui Liu, **“PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection,”** CVPR 2026, pp. 21316–21325. [CVF page](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PPM-CLIP_Probabilistic_Prompt_Modeling_for_Generalizable_AI-Generated_Image_Detection_CVPR_2026_paper.html), [paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_PPM-CLIP_Probabilistic_Prompt_Modeling_for_Generalizable_AI-Generated_Image_Detection_CVPR_2026_paper.pdf), [two-page supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PPM-CLIP_Probabilistic_Prompt_CVPR_2026_supplemental.pdf), [code](https://github.com/bandaidssssss/PPM_CLIP).
- Qinghui He, Haifeng Zhang, Qiao Qin, Bo Liu, Xiuli Bi, and Bin Xiao, **“Diversity over Uniformity: Rethinking Representation in Generated Image Detection,”** CVPR 2026, pp. 40407–40417. [CVF page](https://openaccess.thecvf.com/content/CVPR2026/html/He_Diversity_over_Uniformity_Rethinking_Representation_in_Generated_Image_Detection_CVPR_2026_paper.html), [paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Diversity_over_Uniformity_Rethinking_Representation_in_Generated_Image_Detection_CVPR_2026_paper.pdf), [code](https://github.com/Yanmou-Hui/DoU). No supplemental file is linked by CVF.

“Paper p. N” means the Nth PDF page, followed where useful by the printed proceedings page. Approximate values are explicitly marked when the paper provides only a plot.

---

# 1. PPM-CLIP

## 1.1 Metadata and scope

- **Authors/affiliations:** Wang, Lai, and Luo are at Xiamen University; Luo is also affiliated with its Key Laboratory of Multimedia Trusted Perception and Efficient Computing; Liu is at Truesight Technology Co. Ltd. (paper p. 1 / p. 21316).
- **Task:** binary real/generated image detection with emphasis on cross-architecture and cross-generator transfer.
- **Backbone:** OpenAI CLIP ViT-L/14.
- **Named contributions:** Probabilistic Prompt Modeling (PPM), the Prompt Flow Module (PFM), a learnable prompt repository, and Patch-Wise Contrastive Learning (PWCL).
- **Release:** the abstract says code “will be released”; the linked repository is now public. A small implementation audit is separated from paper claims below.

## 1.2 Motivating argument — author claim

The authors divide detectors into low-level forensic methods and semantic/VLM methods, but argue both remain **static discriminative systems**. A fixed classifier, adapter, or learned prompt pair seeks one decision boundary and therefore memorizes the most convenient generator-specific artifacts. That boundary becomes obsolete when a new generator changes the artifact distribution (paper pp. 1–2, Fig. 1).

Their proposed analogy is an expert who entertains several possible explanations rather than matching one fixed template. PPM therefore draws an image-conditioned distribution of “real/fake” prompt hypotheses. One sampled prompt pair can be unreliable, but averaging many pairs should marginalize sample noise and retain consensus. PWCL separately makes the CLIP visual representation sensitive to subtle local high-frequency evidence that semantic pretraining may ignore (paper pp. 2–3, Fig. 2).

### Intuition in plain terms

1. **PPM:** do not ask one learned textual prototype whether the image is fake. Generate several slightly different prototype pairs, conditioned partly on this image, ask all of them, then average.
2. **Prompt repository:** stochastic offsets need stable full-length prompt scaffolds; otherwise samples may be diverse but incoherent.
3. **PWCL:** those prompts cannot reason about forensic traces if the visual token contains only object semantics. Use DCT-ranked patch groups to fine-tune late visual attention toward locally complex texture.

## 1.3 Exact data flow and architecture — paper claim

### A. Visual path and PWCL

For a resized image `I` of 224×224, CLIP ViT-L/14 divides the image into non-overlapping 14×14 patches in the released configuration. The paper writes a general patch size `M`. For patch `m`, it computes a 2-D DCT independently over RGB channels. `N_f` diagonal frequency bands partition positions by `i+j`. The supplementary definition is

`F^(k)_(i,j)=1` when `(2M/N_f)k <= i+j < (2M/N_f)(k+1)`, otherwise zero (supplement p. 1, Eq. 14).

The score is

`G_m = Σ_k 2^k Σ_c Σ_i Σ_j F^(k)_(i,j) log(|DCT_m(i,j,c)|+1)`

(paper p. 3, Eq. 1; supplement p. 1, Eq. 15). The exponential `2^k` factor makes higher bands contribute more.

Patches are sorted within each image. With selection ratio `alpha=0.5`, the highest-scoring half forms the “high-frequency” set. The single highest-scoring patch is the anchor; the rest of that half are positives; every lower-scoring patch is negative. At selected ViT layers, PWCL applies

`L_con = Σ_p ||e_a-e_p||² + Σ_n max(0, m-||e_a-e_n||²)`

with margin `m=1` (paper p. 4, Eq. 2). Thus the learned late visual representation makes high-frequency patches mutually similar and separates them from low-frequency patches. The final CLIP class token is `X_cls`.

The visual CLIP is not fully tuned. LoRA with rank 4 and scale 0.5 is applied to visual blocks 12–23; Fig. 2 depicts all earlier visual blocks as frozen and LoRA in the latter half (paper pp. 3, 6). The public implementation applies PWCL at those same 12 blocks.

### B. Prompt Flow Module

PPM generates four length-one adjustment vectors per Monte Carlo draw:

- **IGD (image-generic distribution):** conditioned on one learned global vector `X_g`; intended to represent prompt-level universal/syntactic structure.
- **ISD (image-specific distribution):** conditioned on `X_cls`; intended to retain the particular image’s visual evidence.
- **ICD (image-class distribution):** a shared-weight encoder is separately conditioned on learned vectors `X_r` and `X_f`; intended to place real and fake at opposing semantic poles.

(paper p. 4, §3.2.1).

Each input is mapped to the mean and diagonal variance of a simple Gaussian base distribution `q_0`. Samples use the reparameterization trick. Each sample then traverses `K` planar-flow transformations:

`Phi_(i+1) = Phi_i + u h(w^T Phi_i + b)`, with `h=tanh`,

and

`log q_K(Phi_K) = log q_0(Phi_0) - Σ_k log|1 + u_k^T phi(Phi_k)|`,

where `phi(Phi)=h'(w^T Phi+b)w` (paper p. 4, Eqs. 3–4). The main experiment uses `K=10`. The paper calls these transformations learnable and invertible.

### C. Prompt repository and fusion

There are `B=2` learned base prompt pairs. Each base `b` has three token segments:

- generic context `G_(b,1..L_g)`, shared by real and fake;
- specific content `S_(b,1..L_s)`, also shared;
- distinct class concepts `C^r_(b,1..L_c)` and `C^f_(b,1..L_c)`.

The `n`th sampled one-token offsets are broadcast across their corresponding sequences:

- real: `[G_b + phi^g_n][S_b + phi^s_n][C^r_b + phi^r_n]`
- fake: `[G_b + phi^g_n][S_b + phi^s_n][C^f_b + phi^f_n]`

(paper pp. 4–5, Eqs. 5–6). This creates `B×N` real/fake pairs per image. They pass through the frozen CLIP text encoder. The released defaults clarify lengths that the paper does not state: `L_g=3`, `L_s=7`, and `L_c=10`, for 20 learned prompt tokens, but these are code details rather than reported paper hyperparameters.

### D. Similarity, training, and inference

For each prompt pair `i`, cosine similarities to `X_cls` are converted to a two-class softmax at temperature `tau`:

`P_i(real) = exp(s_i^r/tau) / [exp(s_i^r/tau)+exp(s_i^f/tau)]` and `P_i(fake)=1-P_i(real)`

(paper p. 5, Eqs. 10–11).

- **Training:** only one Monte Carlo sample is generated for efficiency. The text says a single prompt-pair index is randomly selected for each image and ordinary cross-entropy is applied (paper p. 5, §3.3). The repository’s orthogonal loss is computed across its `B` real text embeddings and separately across its `B` fake embeddings:
  `L_ort = Σ_i Σ_(j!=i) [cos(t_i^r,t_j^r)^2 + cos(t_i^f,t_j^f)^2]` (Eq. 7).
- **Inference:** use `N=4` Monte Carlo draws in the stated main setup, hence `B×N=8` prompt pairs, and average probabilities, not logits: `Pbar(real)=(1/(BN))Σ_i P_i(real)` (Eq. 12; implementation details on paper p. 6).

### E. Variational losses and complete objective

A standard-normal prior is implied in the supplement/released code. For IGD, ISD, and ICD, the flow posterior is regularized by

`L_KL = E_q0[log q_K(Phi_K)-log p(Phi_K)]`

(paper p. 5, Eq. 8). Only ISD is decoded back toward `X_cls`, using summed MSE `L_rec=Σ_d(x_d-xhat_d)^2` (Eq. 9). The supplement factorizes `q(Phi|D)` into independent specific, generic, and class components; only the specific latent is image-conditioned, whereas generic and class latents are global across the dataset (supplement p. 1, Eqs. 16–21). It interprets MSE as a Gaussian negative log likelihood for `p(X_cls|Phi_s)` (supplement pp. 1–2, Eq. 22).

The full loss is

`L = L_cls + lambda_con L_con + lambda_ort L_ort + lambda_kl L_KL + lambda_rec L_rec`

(paper p. 5, Eq. 13), with reported optima `lambda_con=0.5`, `lambda_ort=1.0`, `lambda_kl=0.001`, and `lambda_rec=0.5` (paper p. 8, Table 6 and Fig. 4).

### F. What is trainable

**Reported/depicted:** prompt repository; PFM encoders, decoders, and flow parameters; learned conditioning vectors; LoRA parameters in visual layers 12–23. The CLIP text encoder and the rest of the CLIP visual encoder are frozen (Fig. 2; implementation paragraph, paper p. 6).

**Not fully disclosed in the prose:** the release also has a learned 768→768 mapping after text encoding and learned temperatures. The paper does not give parameter counts for any trainable subset.

## 1.4 Training data, protocols, and headline results — author claim

### Training setup

- Adam, learning rate `1e-4`, batch 48, **one epoch**.
- Resize 224×224; random crop in training, center crop at test.
- CLIP ViT-L/14; LoRA `r=4`, `alpha_lora=0.5` on visual layers 12–23.
- `alpha=0.5`, margin 1; repository `B=2`; 10 flow steps; 4 test-time samples.

(paper p. 6, §4.1). No training hardware, time, number of seeds, split sizes, or scheduler is reported.

The paper retrains under the designated source protocol for each benchmark. It does **not** train one detector once and then carry that same frozen detector across Ojha, GenImage, and DRCT (paper p. 6, §4.1).

### Ojha/UniversalFakeDetect-style cross-architecture protocol

Train on ProGAN, test on unseen GAN/diffusion sources. Table 1 reports accuracy over DALL-E, three GLIDE settings, ADM, and three LDM settings. PPM-CLIP obtains:

| DALL-E | GLIDE 100/10 | GLIDE 100/27 | GLIDE 50/27 | ADM | LDM 100 | LDM 200 | LDM 200-cfg | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 98.4±1.3 | 99.3±0.0 | 99.3±0.1 | 99.3±0.1 | 97.0±0.4 | 99.3±0.0 | 99.1±0.1 | 98.4±0.3 | **98.8±0.2** |

The nearest listed means are CoD 97.5, SAFE 95.7, and NPR 95.1 (paper p. 6, Table 1). The paper attributes the 11.9-point gap over UnivFD’s 86.9 mean to dynamic hypotheses rather than its static linear head.

### GenImage cross-generator protocol

Train on the Stable Diffusion v1.4 subset; test on eight generator subsets. The reported “cross-generator” macro-mean includes the seen training generator, SD1.4, so it is not a pure unseen-generator mean. PPM-CLIP reports:

| Midjourney | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 98.7±1.5 | 100.0±0.0 | 99.9±0.0 | 99.2±0.9 | 99.4±0.7 | 100.0±0.0 | 100.0±0.0 | 99.9±0.2 | **99.6±0.4** |

The closest listed detector is LOTA at 98.9 mean; CoD is 96.2 and C2P-CLIP 95.8 (paper p. 6, Table 2).

### DRCT protocol

Table 3 covers base latent diffusion, SD variants, SDXL/refiner, turbo, latent-consistency models, ControlNet, and diffusion-reconstruction/inpainting (“DR”) variants. PPM-CLIP is roughly 99.78–99.85 on the first 13 subsets, then **88.92 on SD1-DR, 88.68 on SD2-DR, and only 56.67 on SDXL-DR**, for **95.72 mean**. The DRCT detector has 91.35 mean and does better on SDXL-DR (67.61). Thus “95.72 SOTA” is true for the reported macro-average, but the hardest listed local/reconstruction shift remains close to chance for PPM-CLIP (paper p. 7, Table 3). The prose claims an SD1/2-DR mean of 87.80; the two table entries actually average to 88.80.

### Degradation robustness

The supplement tests only two degradations on GenImage:

- **Gaussian blur:** sigma 0, 1, 2, 3; metric AUC.
- **JPEG:** quality 100, 95, 90, 85, 80; metric AP.

From the plotted curves, not a numeric table, PPM-CLIP is approximately **99→96→91→89 AUC** as blur sigma rises 0→3 and approximately **99→96→95→93→87 AP** as JPEG quality falls 100→80. It stays above UnivFD, LOTA, and SAFE at the plotted settings (supplement p. 2, Fig. 7). These readings are approximate. The supplement’s text only commits to “approximately 90%” AUC at blur sigma 3 and does not state exact JPEG points.

No resize, Gaussian noise, color jitter, crop, WebP, or composed-degradation evaluation is reported.

## 1.5 Ablations and diagnostics — author claim

### Core modules (GenImage accuracy)

Table 4 replaces components with zero tensors:

| Prompt flow | repository | PWCL | mAcc |
|---|---|---|---:|
| no | no | yes | 91.2 |
| no | yes | yes | 92.1 |
| yes | no | yes | 88.9 |
| yes | yes | no | 80.0 |
| yes | yes | yes | **99.6** |

Removing flow costs 7.5 points; removing the repository costs 10.7; removing PWCL costs 19.6. Replacing DCT selection with random patch selection gives 95.0 (paper p. 7, §4.3 and Table 4).

### Prompt distribution roles

Table 7 starts from repository+PWCL at 92.1 and enables the distributions:

| IGD | ISD | ICD | mAcc |
|---|---|---|---:|
| no | no | no | 92.1 |
| no | no | yes | 92.6 |
| no | yes | no | 95.5 |
| yes | no | no | 91.3 |
| no | yes | yes | 98.0 |
| yes | no | yes | 93.1 |
| yes | yes | no | 96.3 |
| yes | yes | yes | **99.6** |

ISD supplies most standalone benefit; IGD alone slightly hurts; all three together are best (paper p. 8, Table 7). Fig. 5 uses t-SNE to argue that the components capture distinct information.

### Regularization sensitivity

- `lambda_KL`: 0.0001→92.3, 0.001→**99.6**, 0.01→98.6, 0.1→97.1 (paper p. 8, Table 6).
- Curves in Fig. 4 peak around `lambda_rec=0.5`, `lambda_con=0.5`, `lambda_ort=1.0`, and selection ratio 0.5. Exact non-peak values are not tabulated.
- The authors interpret the very small optimum KL weight as preserving distribution variance rather than collapsing toward the prior.

### Sampling/cost trade-off

| method / N | memory | GenImage accuracy | FPS |
|---|---:|---:|---:|
| UnivFD | 3116 MB | 88.8 | 87.44 |
| PPM, 2 | 3200 MB | 98.6 | 50.51 |
| PPM, 4 | 4764 MB | 99.6 | 35.61 |
| PPM, 6 | 5542 MB | 99.8 | 27.17 |
| PPM, 8 | 6496 MB | 99.9 | 22.19 |
| PPM, 10 | 7976 MB | 100.0 | 18.76 |

(paper p. 8, Table 5). The main `N=4` setup is about 2.45× slower (59.3% lower throughput) and uses 1.53× (52.9% more) reported memory than UnivFD. A forward pass encodes the image once but sends `2BN` class-specific soft-token sequences through the text transformer—16 sequences per image for `B=2,N=4`, or 40 at `N=10`. A code-shape estimate gives roughly 25.8M added/trainable parameters and roughly 453M total including the approximately 428M-parameter CLIP ViT-L/14; the paper itself reports no counts. It does not identify GPU, batch size, precision, or whether memory means peak allocated/reserved, so the speed/memory numbers are useful only as within-paper relative costs.

### Qualitative diagnostics

- Fig. 1 shows that an individual sampled hypothesis can separate poorly, while the ensemble separates the PCA projection.
- Supplement Fig. 6 gives 20 individual confidences (`B=2, N=10`) across GAN→GAN, GAN→SD, SD→GAN, and SD→SD examples. Individual GAN→SD scores can fall to 0.18 or 0.22, while averages are 0.66 and 0.90 for the two shown rows. In-domain averages are ~0.97–1.00. This supports variance reduction but is a tiny illustrative sample, not a calibrated uncertainty study.
- Grad-CAM in Fig. 3 shows the PWCL model focusing more locally than vanilla CLIP. No localization ground truth or quantitative pointing metric is provided.

## 1.6 Critical reading: limitations and confounds

The following are **our critique**, not author claims.

### Conceptual and mathematical issues

1. **It is still discriminative at the output.** PPM learns from binary cross-entropy and ends in cosine logits for two classes. Calling it a wholesale shift from “discrimination” to “generation” overstates the change. It is better described as a stochastic, input-conditioned prototype ensemble with variational regularization.
2. **Multiple samples do not guarantee multiple forensic perspectives.** All hypotheses share one visual token, two base templates, one text encoder, and the same learned flow. Orthogonality across `B=2` templates and latent variance can produce numerical spread without semantically distinct evidence. Fig. 1/Fig. 6 demonstrate ensemble variance reduction, not that samples discover novel artifact types.
3. **The flow’s invertibility is not established.** A planar flow generally needs a constraint/reparameterization on `u` relative to `w` to guarantee invertibility. Neither Eq. 3 nor the release shows that constraint. The code takes `log(abs(1+...))`, which makes a density correction computable but does not itself make the map one-to-one.
4. **The ELBO story is loose.** The supplement defines the “data” as images and labels, but reconstructs only a frozen/tuned feature `X_cls` from ISD; classification remains a separate discriminative CE. The likelihood model, full joint generative process, and prior are not fully specified in the main paper. This is a useful regularizer, but not a complete generative model of authenticity.
5. **The independence factorization is assumed, not validated.** Generic, specific, and class latent independence may be convenient, but image content and class evidence can be correlated. The paper shows t-SNE, not independence or mutual-information estimates.

### PWCL risks

6. **High DCT energy is not equivalent to a forgery artifact.** Text, foliage, hair, fabric, demosaicing, sharpening, JPEG ringing, and sensor noise can all rank highly. Treating all high-frequency patches within an image as positives can encode content or camera-pipeline shortcuts. Treating all low-frequency patches as negatives can discard smooth but useful diffusion cues.
7. **The most fragile branch dominates the ablation.** Removing PWCL loses 19.6 points, much more than removing flow. That makes the near-perfect result at least as much a frequency-guided LoRA result as evidence for probabilistic prompting. It weakens the paper’s central causal claim that prompt generation is the main source of generalization.
8. **Robustness evidence is narrow.** Frequency emphasis should be stress-tested against severe resizing, denoising, sensor noise, sharpening, screenshot pipelines, and class-matched codec histories. Only blur and JPEG are shown, and only as plots. There is no composed-transform test or clean/robust trade-off table.

### Evaluation issues

9. **Accuracy can hide threshold and class-balance behavior.** The main tables mostly report accuracy, without AUROC, AP, per-class accuracy, calibration, or fixed-threshold transfer. Near-100 results on standard paired benchmarks can arise from real/fake preprocessing or source shortcuts.
10. **Unseen generator is not necessarily unseen content or real domain.** GenImage subsets share benchmark construction and often real-image pools/categories. Training on SD1.4 and testing other GenImage generators is valuable, but it does not prove cross-dataset or cross-camera generalization.
11. **The hardest DR result contradicts broad “universal” language.** SDXL-DR is 56.67. The strong mean is dominated by thirteen almost-perfect subsets. The paper should report worst-group accuracy and the DR-only mean, not only the overall macro-average.
12. **Statistical reporting is incomplete.** “±” appears only for PPM-CLIP, but the paper does not say whether it is across seeds, runs, or subsets. There are no confidence intervals or paired significance tests.
13. **Hyperparameters appear tuned on the evaluation benchmark.** Tables/figures give detailed GenImage sensitivity and select the best settings, but the validation split and model-selection rule are not described. This matters when the final headline is also GenImage.

### Missing implementation/cost detail

14. Missing from the paper: base Gaussian network dimensions; `N_f`; prompt segment lengths; exact LoRA targets (Q/K/V/etc.); temperature; parameter counts; seed count; training time/hardware; inference batch/precision; and exact DRCT train split.
15. Image-dependent prompts require running CLIP’s text transformer for every image and every hypothesis. Unlike static class prompts, their text embeddings cannot be globally cached. Table 5 confirms the resulting memory/throughput penalty. This is a poor fit for a small hackathon inference pipeline unless `N` is tiny or text features are approximated.

### Public-release audit, separate from paper evidence

At repository commit `09d05b9fc4be6a2b079356bbf337a05e193db0be` (inspected 2026-08-30):

- defaults say 100 epochs and `sample_num=10`, whereas the paper says one epoch and `N=4` for the main setup;
- `main.py` evaluates the test loaders during training and passes mean **test accuracy** to `EarlyStopping`, which saves the checkpoint on that score once validation exceeds 90%;
- prompt lengths and `N_f=6` are revealed in code, not the paper;
- extra learned mappings/temperatures are present.

This does **not prove** the published tables used test-selected checkpoints, but the released training script would contaminate a test set if used as written. A clean reimplementation must select checkpoints only on a generator-disjoint validation set before treating the headline numbers as reproducible.

## 1.7 Architecture-taste lessons

- **Good taste:** express epistemic uncertainty as an ensemble and train the visual backbone to expose the evidence that the head needs.
- **Good taste:** use a small number of parameter-efficient visual updates rather than unconstrained full fine-tuning.
- **Good taste:** explicitly measure accuracy/latency as the sample budget changes.
- **Questionable taste:** build a complex normalizing-flow/text-decoder stack when most gain comes from the visual auxiliary loss and two prompt bases. A simpler low-rank stochastic prototype head should be tested first.
- **Questionable taste:** equate “diverse prompt vectors” with diverse causal cues without intervention-based validation.
- **Bad fit for our challenge:** depend heavily on high frequency without paired class-balanced degradation training and shortcut audits.

---

# 2. Diversity over Uniformity (DoU)

## 2.1 Metadata and scope

- **Authors/affiliations:** Qinghui He, Haifeng Zhang, Qiao Qin, Bo Liu, Xiuli Bi, Bin Xiao; Chongqing Key Laboratory of Image Cognition and Chongqing University of Posts and Telecommunications, with Jinan Inspur Data Technology listed for one affiliation (paper p. 1 / p. 40407).
- **Method name:** the prose sometimes calls the framework DoU; its central mechanism is Anti-Feature-Collapse Learning (AFCL), combined with Cue Information Bottlenecks (CIB) and Class-Specific Prompt Learning (CSP).
- **Backbone:** frozen CLIP ViT-L/14 visual and text encoders.
- **Supplement:** none is linked by CVF. This absence matters because several implementation details are missing from the 11-page paper.

## 2.2 Motivating argument — author claim

DoU rejects the claim that detectors lack possible cues. A pretrained representation already contains frequency, noise, smoothing, style, and semantic evidence. The problem is **representational homogenization during supervised training**: optimization selects a few easiest cues, compresses many signals into a narrow subspace, and forms a brittle shortcut boundary (paper pp. 1–4, Figs. 1–3).

The evidence offered is:

- Pretrained features occupy a broader manifold. CNNDet and VIB-Net misclassify unseen fakes near real samples and have effective ranks only 1.37 and 1.92, versus 99.08 for the shown pretrained features (paper p. 3, Fig. 2).
- Existing CNN and ViT detectors reach nearly all their accuracy with only a few principal components. Adding more dimensions contributes variance but little discrimination. With the proposed heterogeneity constraint, accuracy continues rising with subspace dimension (paper pp. 3–4, Fig. 3).

The design goal is therefore not “add more signals,” but keep several existing hierarchical cues useful, task-relevant, decorrelated, and individually nonessential.

### Intuition in plain terms

1. Read the frozen CLIP class token at many depths; early/middle/late layers see different details.
2. Compress each layer so it retains authenticity information and drops nuisance information.
3. Penalize dependence between layer cues so they do not all learn the same shortcut.
4. Randomly/virtually remove each cue and require a stable prediction, so no single cue becomes a point of failure.
5. Learn a balanced aggregate and compare it with learned “real” and “fake” text prototypes.

## 2.3 Exact architecture and optimization — paper claim

### A. Multi-stage frozen encoder

A frozen CLIP ViT-L/14 emits `N` class-token features `v_i in R^D` from multiple visual stages (paper p. 4, Eq. 1 and Fig. 4). Each stage has its own learned CIB transform, producing `vtilde_i` (Eq. 2). The paper does not state `N`, the exact tapped blocks, or the hidden/output sizes.

### B. Cue Information Bottleneck

The stated information-bottleneck objective is

`max Σ_i [I(vtilde_i;y) - beta I(vtilde_i;x)]`

(paper p. 4, Eq. 3): retain label-relevant information but suppress input-specific nuisance. In practice, each stage predicts a variational posterior `q(z_i|v_i)`, regularized to `N(0,I)`:

`L_IB = Σ_i KL[q(z_i|v_i) || N(0,I)]`

(paper p. 5, Eq. 4).

It also compares the full prediction with the prediction after removing cue `i`:

`L_cons = Σ_i KL[p(y|Vtilde) || p(y|Vtilde \ vtilde_i)]`

(Eq. 5). `L_CIB=L_IB+L_cons` (Eq. 6). The first term compresses each stage; the second is intended to stop exclusive reliance on one stage.

After stage aggregation, Fig. 4 and Eq. 11 describe another global CIB to remove residual noise.

### C. Anti-Feature-Collapse Learning

For a batch of `B` samples, DoU computes centered Gram matrices `K_i`, `K_j` for every pair of stage representations and uses empirical HSIC:

`HSIC(vtilde_i,vtilde_j)=Tr(K_i H K_j H)/(B-1)^2`

(paper p. 5, Eq. 7). The mean over all ordered stage pairs is `L_AFCL` (Eq. 8). Minimizing it should make stages independent/decorrelated and therefore complementary.

### D. Aggregation and weight regularization

The paper states a simplex-weighted sum

`v_agg=Σ_i alpha_i vtilde_i`, with nonnegative `alpha_i` summing to one (Eq. 9).

It discourages a peaked gate with

`L_reg=(Σ_i alpha_i² - 1/N)²`

(Eq. 10), whose minimum under the simplex constraint is the uniform vector. A global CIB then yields `vtilde_final` (Eq. 11).

### E. Class-specific prompt learning and inference

For each class `c in {real,fake}`, CSP learns `M` continuous context tokens followed by the literal class token `[C]`:

`T_c=[T_c^1,...,T_c^M,[C]]` (paper p. 5, Eq. 12).

The frozen CLIP text encoder maps each to a class prototype `e_c`. The final visual feature is projected to CLIP’s shared space, cosine similarities are divided by `tau`, and a two-class softmax produces probabilities (Eqs. 13–14). Cross-entropy is `L_CSP` (paper p. 6, Eq. 15). At inference there is one aggregated visual representation and one static real/fake prompt pair; unlike PPM, there is no prompt ensemble in the paper.

### F. Full loss and trainable parts

`L = L_CSP + lambda_1 L_CIB + lambda_2 L_AFCL + lambda_3 L_reg`

(paper p. 6, Eq. 16).

The CLIP image and text encoders are frozen. Trainable components are the per-stage CIBs, aggregation weights, final CIB, any visual projection, and class-specific prompt context tokens (Fig. 4). The paper gives neither trainable nor total parameter counts.

## 2.4 Datasets, protocol, and results — author claim

### Training

- GenImage Stable Diffusion v1.4 subset only, stated as 320k images at 100% in the few-shot section. SD1.4 is also retained in the reported test average, so “avg” mixes a seen generator with unseen generators; the exact membership of “cross” is not enumerated.
- Adam, learning rate `1e-4`, batch 512, early stopping.
- CLIP ViT-L/14; PyTorch; NVIDIA H100 PCIe GPU(s).
- “Same data augmentation as [44]” (VIB-Net), without restating it.

(paper pp. 6–7, §5.1; p. 8, §5.5). Epoch count, early-stop split/rule, image size, number of GPUs, time, precision, seeds, and loss weights are absent.

### Test sources

The prose names:

- **UniversalFakeDetect:** BigGAN, CycleGAN, GauGAN, ProGAN, StarGAN, StyleGAN, DeepFakes, Guided Diffusion, LDM, GLIDE, DALL-E.
- **GenImage:** ADM, GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong.
- **AIGI-Holmes:** Infinity, LlamaGen, PixArt-XL, SD3.5-L, Show-o, VAR. The paper notes that its real images are compressed/web-scraped and its generated images are high resolution.

(paper pp. 6–7). Fig. 5 says 21 sources and visually labels six GANs plus DALL-E/GLIDE/ADM/GLIDE/Midjourney/SD1.4/SD1.5/VQDM/Wukong and six AIGI-Holmes models. This does not cleanly match all prose-listed subsets; several UFD subsets are omitted from the radar.

Metrics are AP and accuracy, with F1, real accuracy, and fake accuracy for some comparisons (paper p. 7).

### Aggregate results

The paper reports **99.52% mean AP and 92.81% mean accuracy** over all evaluated test sets. It says this exceeds VIB-Net by 3.39 AP points and 5.68 accuracy points, and CLIPping by 12.20 AP points and 5.00 accuracy points (paper p. 7, §5.2 and Fig. 5).

Fig. 1 reports diffusion-model accuracy / GAN accuracy for representative detectors:

| detector | diffusion ACC | GAN ACC |
|---|---:|---:|
| DoU | **93.85** | **90.02** |
| CLIPping | 90.27 | 85.56 |
| VIB-Net | 88.93 | 83.46 |
| DRCT | 90.60 | 77.32 |
| NPR | 89.19 | 64.23 |
| CO-SPY | 76.99 | 66.98 |
| UnivFD | 72.51 | 56.70 |
| CNNDet | 74.06 | 50.13 |

(paper p. 1, Fig. 1). The abstract’s “5.02%” improvement is not tied clearly to one table; the body instead gives +5.68 overall ACC versus VIB-Net.

Fig. 5 is a radar plot, not a numeric result table. Exact per-generator AP/ACC/F1/real-ACC/fake-ACC values cannot be recovered reliably from the paper. That is a reporting limitation, not missing analysis here.

### Representation analysis

DoU’s final representation has effective rank **67.38**, versus CNNDet 1.37 and VIB-Net 1.92. For 90% explained variance, the text says DoU removes only 26 principal components relative to the pretrained structure, while existing detectors remove 479 (paper p. 7, Fig. 6). The authors interpret this as retaining more pretrained knowledge and a more heterogeneous manifold.

### Few-shot scaling

| SD1.4 training fraction | examples | ACC | AP |
|---:|---:|---:|---:|
| 0.1% | 320 | 80.98±9.30 | 90.81±6.67 |
| 1% | 3.2k | 81.90±9.10 | 92.79±5.07 |
| 10% | 32k | 85.56±9.05 | 95.85±3.79 |
| 50% | 160k | 89.55±7.46 | 96.94±2.48 |
| 100% | 320k | 92.81±6.75 | 99.52±0.29 |

(paper p. 8, Fig. 7). The paper does not define what the ± dispersion is over; it appears more likely to summarize variation across test sources than repeated training seeds, but that is not stated.

### JPEG and blur robustness

Fig. 8 evaluates JPEG quality 100, 95, 90, 85, 80, 75, 70 and Gaussian blur sigma 0, 0.5, 1, 1.5, 2. DoU is the best plotted method at every strength. Approximate visual readings are:

- JPEG: digitized ACC about **93.8→89.7→89.5→88.6→87.6→86.6→86.6** and AP about **99.6→97.9→97.5→96.9→94.6→94.1→93.8** from quality 100→70.
- Blur: digitized ACC about **93.9→93.8→92.6→91.5→91.1** and AP about **99.6→99.5→99.0→98.6→98.3** from sigma 0→2.

These are approximate values digitized to the nearest roughly 0.1 point because the paper supplies curves without a numeric table (paper p. 8, Fig. 8). The digitized clean→strongest endpoints make the gap clearer:

| method | JPEG ACC Q100→70 | JPEG AP Q100→70 | blur ACC sigma 0→2 | blur AP sigma 0→2 |
|---|---|---|---|---|
| UnivFD | 65.7→65.9 | 78.1→75.8 | 65.7→59.7 | 77.9→66.3 |
| NPR | 75.6→49.2 | 89.9→49.4 | 76.0→64.3 | 89.9→68.4 |
| DRCT | 84.1→61.8 | 93.4→67.8 | 84.3→65.5 | 93.4→74.9 |
| VIB-Net | 87.6→73.1 | 97.1→84.5 | 87.7→66.5 | 97.1→73.6 |
| CLIPping | 89.0→50.6 | 86.8→50.6 | 89.4→62.2 | 87.2→58.5 |
| DoU | **93.8→86.6** | **99.6→93.8** | **93.9→91.1** | **99.6→98.3** |

The benchmark aggregation, exact transform library, input ordering, and whether the selected training augmentation already included blur/JPEG are not specified. Resize, noise, color jitter, crop, and transform compositions are absent.

## 2.5 Ablations — author claim, then what they do not establish

Table 1 on paper p. 7:

| CIB | AFCL | weight reg | avg ACC | cross ACC | avg AP | cross AP |
|---|---|---|---:|---:|---:|---:|
| no | no | no | 87.81 | 85.56 | 87.32 | 84.69 |
| yes | no | yes | 89.72 | 85.99 | 99.38 | 98.73 |
| no | yes | yes | 91.60 | **91.15** | 99.40 | 98.73 |
| yes | yes | yes | **92.81** | 90.02 | **99.52** | **98.90** |

Author interpretation: CIB filters superfluous information; AFCL gives the larger cross-model accuracy improvement; all components yield the best overall ACC/AP and thus work synergistically (paper pp. 7–8).

Important qualification: adding CIB to AFCL **reduces cross-generator accuracy from 91.15 to 90.02**, even while it improves overall accuracy and cross AP slightly. The paper does not discuss this trade-off. The table also never disables weight regularization when either CIB or AFCL is active, so it does not isolate `L_reg`. There are no ablations for the KL versus cue-removal parts of CIB, the final CIB, CSP, number/choice of layers, HSIC kernel, aggregation method, or loss weights.

## 2.6 Critical reading: limitations and confounds

The following are **our critique**, not author claims.

### Conceptual tension

1. **Independence is not the same as complementary class evidence.** HSIC can be reduced by keeping unrelated nuisance differences between stages. Effective rank can likewise be high because of content, generator identity, or noise. Neither proves that extra dimensions causally support real/fake transfer.
2. **CIB and cue-removal consistency pull in opposite directions.** HSIC asks stage cues to be nonredundant; `L_cons` asks the prediction to remain unchanged after deleting any one cue, which encourages redundancy or makes each cue individually sufficient. The paper does not analyze this tension.
3. **Uniform aggregation can preserve weak cues, not only diversity.** `L_reg` explicitly prefers equal weights. That prevents gate collapse, but may force noisy or transformation-fragile layers into every decision. A reliability-aware gate under degradation could be better.
4. **A frozen backbone preserves biases too.** Keeping CLIP’s high-rank space avoids supervised collapse, but also retains semantic, dataset, and web-pretraining shortcuts. CIB is supposed to remove them, yet no content/domain intervention establishes that it does.

### Mathematical/implementation gaps

5. Eq. 4 implements only the compression term of the information bottleneck. Label relevance is supplied indirectly by the final CE and consistency loss; the paper does not derive the variational lower bound for `I(z;y)`.
6. The HSIC kernel is unspecified. Linear and RBF HSIC behave differently, and raw magnitude/normalization matters. Pairwise Gram matrices can cost `O(N²B²)` memory/time; the large batch of 512 makes the omitted implementation important.
7. The paper omits `N`, tapped layer indices, CIB architecture, latent size, posterior variance treatment at inference, context length `M`, temperatures, loss weights, and cue-removal renormalization. Without a supplement, the method is not self-contained.

### Evaluation/reporting gaps

8. **No numeric main table.** A radar plot is poor audit material. It hides exact per-source failures and makes macro-average reconstruction impossible.
9. **Source-count mismatch.** The test prose and Fig. 5 do not enumerate the same apparent set. “avg” and “cross” therefore lack a fully auditable denominator.
10. **GenImage protocol can reuse real content/domain.** Cross-generator testing within one benchmark may keep real-image distributions and semantic categories familiar. AIGI-Holmes helps with a new domain, but also has a stated codec/resolution asymmetry between scraped real and high-resolution fake images.
11. **High AP with lower accuracy needs threshold analysis.** Full DoU has 99.52 AP but 92.81 ACC; the CIB-only row jumps from 87.32 to 99.38 AP while ACC rises only 1.91. This suggests excellent ranking but substantial score/threshold or class-specific behavior. Calibration, per-class error, AUROC, ECE, and fixed-threshold cross-domain transfer are essential.
12. **Robustness is incomplete and possibly augmentation-aligned.** The paper references another paper’s augmentation without listing it, then evaluates JPEG and blur only. If those were training augmentations, Fig. 8 is robustness to seen transform families, not unseen perturbations as the heading implies.
13. No confidence intervals across independent training runs, no worst-source metric, no statistical test, and no error examples are reported.

### Compute and reproducibility

14. The frozen ViT-L/14 is comfortably below the challenge’s 2B limit, but is still a large backbone. Batch 512 on H100 is not hackathon-friendly. No FPS, memory, FLOPs, trainable parameters, epochs, or wall time are reported.
15. Static text prototypes can be cached at inference, which should make DoU materially cheaper than PPM’s image-conditioned prompt ensemble. The paper does not report or exploit this comparison.

### Public-release audit, separate from paper evidence

At repository commit `230856fabd137e00580d5457b3b1ce7b664bcf99` (inspected 2026-08-30):

- the README says training code, configs, and reproducibility details are still “coming next”; only evaluation and a checkpoint are presented;
- the model hooks all 24 ViT-L/14 `ln_2` outputs, so `N=24` in that release;
- each cue uses independent 1024→1024 linear mean and log-variance heads, samples `z=mu+sigma*epsilon`, and aggregation weights have shape `[1,24,1024]`—per-feature weights—not the scalar `alpha_i` in paper Eq. 9;
- the released model has no separate final/global CIB shown in paper Eq. 11/Fig. 4;
- the stochastic reparameterization is still active in `.eval()`, so repeated inference can change unless the random seed is controlled or scores are averaged;
- from the released dimensions, the 48 mean/log-variance linears alone contain about **50.38M trainable parameters**; with aggregation and 2×16 learned prompt tokens, the visible adapter total is about **50.43M**, excluding any component not present in the release;
- current default options (`lr=1e-3`, batch 256, 30 epochs) do not match the paper (`1e-4`, batch 512, early stopping).

These discrepancies prevent a clean paper-to-code reproduction and make the paper’s cost impossible to infer precisely.

## 2.7 Architecture-taste lessons

- **Good taste:** inspect representations before inventing a new input signal. A detector can fail because optimization discards useful pretrained structure.
- **Good taste:** expose intermediate layers and test whether added subspace dimensions improve actual held-out accuracy, not only variance.
- **Good taste:** use cue removal as an intervention. If one cue disappears under JPEG or blur, the detector should degrade gracefully.
- **Good fit for deployment:** frozen encoders and cacheable static text prototypes simplify inference.
- **Questionable taste:** maximize raw feature diversity without conditioning on label and transformation stability. Diversity can preserve nuisance.
- **Questionable taste:** claim synergy from an ablation where cross-generator accuracy is actually best without one of the modules.
- **Bad reproducibility taste:** reference augmentation externally, omit all loss weights and layer details, and report the main benchmark as a radar plot.

---

# 3. Representation philosophies compared

| Question | PPM-CLIP | Diversity over Uniformity |
|---|---|---|
| Where is diversity created? | In the **prompt/prototype hypothesis distribution** for each image. | In the **multi-layer visual evidence representation** retained across the frozen backbone. |
| Core failure theory | One static decision boundary memorizes generator artifacts. | Supervised training collapses many available cues into a few dominant directions. |
| Adaptation to each input | Strong: ISD conditions prompt offsets on that image’s `X_cls`. | Limited: one learned stage aggregation and static class prompt pair; visual features vary with the image but the decision machinery is global. |
| Stochasticity | Explicit normalizing-flow Monte Carlo; probability averaging. | Paper presents one aggregate; public code samples each variational CIB even at test. |
| Visual encoder | Late visual LoRA is trained, strongly guided by DCT-ranked PWCL. | Visual CLIP is frozen; hierarchical CLS tokens are filtered and combined. |
| Text encoder | Frozen but rerun for every image/hypothesis because prompts are image-dependent. | Frozen; static learned class prompts can be precomputed at inference. |
| Diversity control | Orthogonal repository prompts + latent distributions + ensemble. | Pairwise HSIC decorrelation + balanced aggregation + cue-removal consistency. |
| Principal invariance mechanism | Ensemble marginalization over prompt samples. | Robustness to deletion of any stage cue; retention of high-rank features. |
| Main fragility | High-frequency visual shortcut and correlated/expensive hypotheses. | High-rank nuisance preservation; internal tension between independence and redundancy. |
| Strongest causal ablation | Flow −7.5, repository −10.7, PWCL −19.6 points on GenImage. | AFCL gives +5.59 cross ACC over baseline; adding CIB then lowers cross ACC by 1.13 but raises overall metrics. |
| Cost profile | 35.61 FPS / 4.764GB at `N=4` in the paper’s unspecified setup. | Not reported; one visual pass plus many feature heads, with cacheable text features. |

### The key difference

PPM-CLIP says, **“keep several possible decisions.”** DoU says, **“keep several possible reasons.”** The first marginalizes a distribution of class prototypes after forcing the image representation toward high-frequency evidence. The second preserves a heterogeneous visual basis but makes one aggregated decision. They are complementary in principle, yet neither proves that its multiple vectors correspond to independent, transformation-stable forensic mechanisms.

A better synthesis for our project is not “stack both papers.” It is to create diversity over **measured evidence channels and their degradation behavior**, then estimate which channels remain trustworthy for a particular transformed image.

---

# 4. Adopt / adapt / avoid for this repository

## Adopt as principles

1. **Keep multiple evidence paths.** Use at least semantic/context, local residual/texture, and acquisition/frequency summaries. Do not let one branch dominate.
2. **Train with cue removal.** Randomly drop or corrupt a branch and require the remaining branches to preserve the binary prediction. This directly models real redistribution failures.
3. **Use paired clean/transformed views.** Optimize score consistency and calibration across JPEG, blur, resize, noise, color jitter, and crop while retaining class separation.
4. **Report diversity and utility together.** Track effective rank/CKA/HSIC, but also accuracy as principal components or branches are added, and performance after interventions.
5. **Use parameter-efficient adaptation.** A frozen or mostly frozen public backbone plus small adapters is feasible, reproducible, and well below 2B.
6. **Report an explicit cost frontier.** Params, trainable params, peak memory, latency/FPS, and accuracy at different branch/sample budgets.

## Adapt, do not copy

### Proposed original direction: degradation-conditioned complementary evidence

Build a small set of **named evidence experts** rather than stochastic prompt clones:

- semantic expert from a frozen ViT/CLIP/DINO feature;
- spatial residual expert from high-pass/noise residual patches;
- compact spectral/acquisition expert from radially/diagonally pooled DCT/FFT and codec statistics.

For a clean image and sampled transformed view, learn:

1. a **shared authenticity subspace** that must remain stable across transformations;
2. **expert residual subspaces** whose class-conditional residuals are decorrelated *after projecting out the shared class signal*;
3. a **degradation/reliability gate** predicted from transform-sensitive measurements, with entropy/floor constraints so it cannot collapse to one expert;
4. **counterfactual expert dropout**, requiring stable scores when the currently highest-weight expert is removed;
5. a consistency/calibration loss on clean/transformed score pairs.

This differs materially from PPM-CLIP: no image-conditioned prompt flow, no Monte Carlo text encoder, and diversity is tied to named evidence and measured degradation. It differs materially from DoU: diversity is across signal families and **conditional residuals**, not raw multi-layer features, and the gate is allowed to respond to JPEG/blur/resize reliability rather than being pushed uniformly.

A compact alternative, if only one backbone is feasible, is to use intermediate layers as experts but decorrelate their **class-conditional transformation-response vectors**, not their raw embeddings. That targets complementary robustness rather than cosmetic feature rank.

## Avoid

- Do not implement PPM’s prompt flow/PWCL or DoU’s CIB+HSIC recipe verbatim; that risks challenge disqualification as detector replication.
- Do not equate DCT energy with fake evidence. Balance codecs/resolutions between classes and verify cues on real camera images.
- Do not select checkpoints or hyperparameters on the demonstration WildFake split or any test generator.
- Do not report only mean accuracy. Include worst-generator, worst-transform, real accuracy/false-positive rate, AP/AUROC, and calibration.
- Do not use a stochastic test path without a fixed seed, deterministic expectation, or explicit multi-sample average.
- Do not build a ViT-L/14 text-ensemble bottleneck before a cheaper ViT-B/16 or DINOv2-base experiment establishes value.

---

# 5. Recommended experiment plan

## 5.1 Protocol first

1. **Generator-family holdout:** split by generator family, not random image. At minimum hold out GAN, convolutional diffusion, diffusion-transformer, and autoregressive families in turn.
2. **Dataset/source holdout:** keep real-image datasets/camera domains disjoint where licenses permit. Hash/deduplicate real images across all splits.
3. **Codec/resolution balancing:** re-encode both classes through the same randomized JPEG/PNG/WebP pipeline; match resolution histograms. Run a metadata-only/codec-only probe to expose shortcuts.
4. **Transformation matrix:** clean plus the exact challenge grid:
   - JPEG 90/70/50/30;
   - Gaussian blur sigma 0.5/1/2;
   - downscale 0.5×/0.25× then upscale;
   - Gaussian noise 0.02/0.05/0.10;
   - brightness/contrast/saturation ±20%;
   - center crop retaining 80%.
5. **Compositions:** at least resize→JPEG, crop→resize→JPEG, blur→JPEG, and noise→JPEG. Randomize order in a separate stress suite.
6. **No hidden selection:** select once on generator-disjoint validation; lock threshold and temperature; then evaluate held-out generators/transforms.

## 5.2 Baselines and ablations

Run these in increasing complexity:

1. frozen backbone + linear head;
2. semantic expert only;
3. residual expert only;
4. spectral/acquisition expert only;
5. simple concatenation;
6. reliability gate without diversity loss;
7. conditional-residual decorrelation without gate;
8. expert dropout without gate;
9. full proposed system;
10. full system minus each expert and under forced corruption of each expert.

Also compare raw HSIC decorrelation versus class-conditional residual decorrelation. If raw HSIC raises effective rank but not worst-group accuracy, reject it.

## 5.3 Tests inspired by the two papers, but diagnostic rather than reproductive

- **Hypothesis/cue removal:** delete each branch and the highest-weight branch; report score KL, accuracy loss, and worst-group loss.
- **Subspace utility:** plot held-out accuracy versus PCA dimensions and PCs@90%, alongside effective rank. Compute it separately by class and generator so generator identity cannot masquerade as authenticity diversity.
- **Cue similarity:** CKA/linear-HSIC between experts before and after projecting out the shared class direction.
- **Reliability validity:** correlate gate weights with controlled degradation strength and each expert’s measured accuracy loss.
- **Uncertainty:** variance across transformed views or lightweight stochastic heads; evaluate error detection, risk–coverage, Brier score, ECE, and NLL. Do not assume ensemble variance is calibrated.
- **Localization sanity:** if a patch branch is used, test whether selected patches remain stable after content-preserving transforms and whether selection is class-balanced; do not rely on Grad-CAM anecdotes.

## 5.4 Metrics and reporting

For every clean/transformed and seen/unseen cell, report:

- AUROC and AP;
- accuracy at one validation-locked threshold;
- real accuracy, fake accuracy, F1, FPR at selected TPRs;
- ECE/Brier score;
- macro mean, worst generator, worst transform, and worst generator×transform;
- mean±standard deviation over at least three seeds for trainable heads;
- total/trainable parameters, peak VRAM, images/s, and per-image latency at batch 1 and a practical batch.

## 5.5 Go/no-go decisions

- **Keep multi-evidence diversity** only if it improves worst held-out generator×transform, not just average AP or effective rank.
- **Keep a reliability gate** only if weights predict per-expert degradation and outperform fixed averaging under transform compositions.
- **Keep stochastic inference** only if `N=2–4` materially improves calibration/worst-group results enough to justify latency.
- **Drop a frequency expert** if codec-balanced real false positives rise or blur/resize erases its gain.
- **Prefer a smaller backbone** if the larger backbone’s gain is less than the improvement from data/protocol controls; originality should live in the evidence-and-robustness design, not model scale.

---

# Bottom line

PPM-CLIP offers a useful engineering idea—average several image-conditioned prototype hypotheses—but its strongest ablation is actually a high-frequency visual fine-tuning loss, its “generative” framing is stronger than its mathematics, and its text-side Monte Carlo is expensive. DoU offers the more useful architectural diagnosis for this project: supervised detection can discard transferable evidence. Yet raw high rank and HSIC independence are not sufficient, its ablations contain a cross-accuracy trade-off, and its paper/release omit key reproducibility details.

For this repository, adopt **graceful multi-cue failure, intervention-based validation, and explicit uncertainty/cost reporting**. Adapt diversity to the challenge by learning **degradation-conditioned complementary evidence under paired transformations**. Avoid reproducing either detector’s named modules.
