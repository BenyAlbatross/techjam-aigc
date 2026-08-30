# PPM-CLIP (CVPR 2026): deep paper and supplement notes

## 1. Source and metadata

- **Title:** *PPM-CLIP: Probabilistic Prompt Modeling for Generalizable AI-Generated Image Detection*.
- **Authors:** Xinyuan Wang\*, Yingxin Lai\*, Zhiming Luo (corresponding), Zhihui Liu. Affiliations: Xiamen University; Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China; Truesight Technology Co. Ltd. [paper p. 21316]
- **Venue:** IEEE/CVF CVPR 2026, June 2026, pp. **21316–21325**. Main paper is 10 proceedings pages including references; supplement is 2 pages.
- **Open-access landing page:** <https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PPM-CLIP_Probabilistic_Prompt_Modeling_for_Generalizable_AI-Generated_Image_Detection_CVPR_2026_paper.html>
- **Paper PDF:** <https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_PPM-CLIP_Probabilistic_Prompt_Modeling_for_Generalizable_AI-Generated_Image_Detection_CVPR_2026_paper.pdf>
- **Supplement:** <https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PPM-CLIP_Probabilistic_Prompt_CVPR_2026_supplemental.pdf>
- **Authors' code:** <https://github.com/bandaidssssss/PPM_CLIP>. I inspected commit `09d05b9fc4be6a2b079356bbf337a05e193db0be` (2026-05-20). Code-derived observations below are explicitly marked because the paper, rather than the code, is the primary source.

## 2. One-paragraph technical summary

PPM-CLIP is a binary real/fake classifier built on frozen CLIP ViT-L/14 plus LoRA in the upper half of the visual transformer. It adds (1) **PWCL**, an auxiliary patch contrastive loss that uses DCT energy to label high-frequency image patches as mutually positive and low-frequency patches as negative, and (2) **PPM**, a soft-prompt generator. PPM samples four length-one adjustment vectors: global/generic, image-conditioned/specific, real-class, and fake-class. It broadcasts them over three segments of each of `B` learned soft-prompt templates, encodes the resulting real/fake prompts with the frozen CLIP text transformer, and compares each text embedding with the image CLS embedding. Training uses one Monte Carlo draw and one randomly chosen prompt pair. Inference averages the two-class softmax probabilities from `B × N` sampled prompt pairs. [Fig. 2 and Secs. 3–3.3, pp. 21318–21320]

## 3. Motivation and claimed contribution

### What the authors claim

1. Existing pixel/frequency detectors learn generator-specific artifacts. Existing CLIP methods improve semantic transfer but still learn one static adapter, classifier, or prompt pair. The authors argue that any such single boundary is too rigid for a changing generator distribution. [Sec. 1, pp. 21316–21317; Sec. 2, p. 21317]
2. Their human-expert analogy is to test multiple possible authenticity cues rather than find one perfect cue. PPM generates an image-tailored distribution of prompt hypotheses; individual hypotheses can be unreliable, but probability averaging marginalizes their noise. [Sec. 1, p. 21317; Fig. 1, p. 21316]
3. A vanilla CLIP vision encoder attends to object semantics and can miss forensic texture. PWCL forces it to retain local high-frequency evidence before the image feature conditions PPM. [Sec. 1, p. 21317; Sec. 3.1, pp. 21318–21319]
4. The paper calls this a shift from “static discriminative” detection to “conditional generative modeling,” and claims SOTA cross-generator results. [Abstract, p. 21316; contributions, p. 21317]

### Immediate critical reading

- This is still a supervised discriminative classifier optimized by real/fake cross-entropy. The generated objects are latent **soft text embeddings**, not images or an explicit density model of real/fake image data. “Generative paradigm” is therefore a useful design metaphor, but stronger than what is established mathematically.
- Only the image-specific latent is conditioned on the input. The generic and class latents are global variables. The most accurate description is **conditional stochastic prompt ensembling with frequency-guided visual PEFT**.
- Figure 1 is suggestive, not causal evidence. It plots the class prompt chosen using ground truth (“real” prompt for a real image and “fake” prompt for a fake image), so class identity is partly built into what is visualized. It also compares against a different method (FatFormer), not controlled deterministic and probabilistic versions of the same model. [Fig. 1 caption, p. 21316]

## 4. Exact architecture and data flow

### 4.1 Visual branch and trainable backbone parts

1. Input `I ∈ R^{h×w×3}` is cropped to `224×224` and passed through **OpenAI CLIP ViT-L/14**. The output global feature is `X_cls`. [Sec. 3, p. 21318; implementation details, pp. 21320–21321]
2. The base CLIP weights are frozen. **LoRA** is inserted into visual layers **12–23**, with rank `r=4` and LoRA scale `α_lora=0.5`. The text encoder is frozen. [Fig. 2, p. 21318; Sec. 4.1, pp. 21320–21321]
3. PWCL supervises patch-token embeddings during training; it does not add a second visual inference tower. Its DCT ranking is a training-only way to choose contrastive groups. [Sec. 3.1, pp. 21318–21319]
4. **Code clarification:** the released implementation applies LoRA to Q, K, and V projections in visual blocks 12–23 and applies/averages the PWCL loss at every one of those 12 blocks (`networks/model_engine.py`, `loralib/utils.py`). This target-module and multi-layer detail is absent from the paper.

### 4.2 PWCL: patch-wise contrastive learning

For non-overlapping patches `p_m ∈ R^{M×M×3}`, take a per-channel 2-D DCT. Divide the DCT plane into `N_f` diagonal frequency bands:

\[
F^{(k)}_{ij}=\mathbf 1\left[\frac{2M}{N_f}k\leq i+j<\frac{2M}{N_f}(k+1)\right].
\]

Score patch `m` by an exponentially high-frequency-weighted log magnitude:

\[
G_m=\sum_{k=0}^{N_f-1}2^k\sum_{c=0}^{2}\sum_{i,j=0}^{M-1}
F^{(k)}_{ij}\log\left(|p_m^{dct}(i,j,c)|+1\right).
\]

[Eq. 1, p. 21318; expanded Eqs. 14–15, supplement p. 1]

Rank patches by `G_m`. The top fraction `α` is the high-frequency set. The highest-scoring patch is anchor `e_a`, the rest of the high set are positives `P`, and every patch in the low set is a negative `N`. The auxiliary loss is

\[
L_{con}=\sum_{e_p\in P}\|e_a-e_p\|_2^2+
\sum_{e_n\in N}\max(0,m-\|e_a-e_n\|_2^2).
\]

Thus positives are collapsed around the most high-frequency patch; negatives are required to be at least squared-distance margin `m` away. Default `α=0.5`, `m=1`. [Eq. 2 and Sec. 3.1, p. 21319; Sec. 4.1, p. 21321]

**Intuition.** This is not contrastive learning from semantic augmentations. DCT rank supplies pseudo-labels about local texture. LoRA then makes later CLIP patch tokens encode the distinction between high-frequency and smoother regions. The global CLS token used for classification is indirectly changed by attention to those patch tokens.

**Code clarification:** with ViT-L/14, `M=stride=14`, so a 224 image gives 256 patches; `α=.5` gives 128 high and 128 low patches. Code uses `N_f=6`. It computes DCT after CLIP normalization, not literally on an RGB-valued image as the prose says. It also normalizes each band by its number of coefficients, while printed Eq. 15 does not show that normalization (`networks/DCT_score.py`).

### 4.3 PPM/PFM: the probabilistic prompt generator

PPM separates a prompt into three semantic levels. There are effectively four sampled adjustments because the class level has separate real and fake draws:

- **IGD, image-generic:** input is a learned global vector `X_g`; sample `φ^g`. Intended to encode common prompt syntax/structure.
- **ISD, image-specific:** input is `X_cls`; sample `φ^s`. This is the only input-conditioned latent and carries per-image evidence.
- **ICD, image-class:** learned inputs `X_r` and `X_f` pass through a shared-weight encoder; sample `φ^r`, `φ^f`. Intended as opposite authenticity poles in a shared space.

[Sec. 3.2.1, p. 21319; factorization in Eq. 17, supplement p. 1]

For each input, a small encoder produces the mean and diagonal scale of a base Gaussian `q_0(Φ_0)=N(μ_ξ,Σ_ξ)`. Reparameterized samples pass through `K` planar-flow transforms:

\[
Φ_{i+1}=Φ_i+u_i\,\tanh(w_i^TΦ_i+b_i),\quad i=0,\ldots,K-1.
\]

The transformed density uses the change of variables:

\[
\log q_K(Φ_K)=\log q_0(Φ_0)-\sum_{k=1}^{K}
\log|1+u_k^T\phi(Φ_k)|,
\quad \phi(Φ)=\tanh'(w^TΦ+b)w.
\]

Default flow length is `K=10`. [Eqs. 3–4, p. 21319; Sec. 4.1, p. 21321]

**Intuition.** A diagonal Gaussian is cheap but unimodal. Ten planar residual transforms bend it into a richer soft-prompt distribution. Reparameterization lets classification gradients train its mean, variance, and flow parameters.

**Important qualification:** the paper calls every planar transform “invertible” but does not give the usual constraint/reparameterization on `u` needed to guarantee planar-flow invertibility. The released code takes `log(abs(1+ψ^T u))` but does not impose such a constraint (`networks/PFL.py`). The density interpretation is therefore not fully secured by the provided implementation.

### 4.4 Prompt repository and fusion

There are `B` learned pairs of soft-prompt templates. Template `b` is

\[
g_b^r=[G_{b,1:L_g}][S_{b,1:L_s}][C^r_{b,1:L_c}],\qquad
 g_b^f=[G_{b,1:L_g}][S_{b,1:L_s}][C^f_{b,1:L_c}].
\]

`G` is shared generic context, `S` is shared specific content, and `C^r,C^f` are separate class segments. [Eq. 5 and Sec. 3.2.2, p. 21319]

For MC sample `n`, broadcast one adjustment over every token in its matching segment:

\[
\begin{aligned}
g_{b,n}^r&=[G_b+φ_n^g][S_b+φ_n^s][C_b^r+φ_n^r],\\
g_{b,n}^f&=[G_b+φ_n^g][S_b+φ_n^s][C_b^f+φ_n^f].
\end{aligned}
\]

This turns `B` base pairs into `B×N` real/fake prompt pairs. [Eq. 6, p. 21320]

Defaults are `B=2`. **Code clarification:** the segment lengths are `L_g=3`, `L_s=7`, `L_c=10`, for 20 learned soft tokens plus SOS/EOS. The latent/text embedding dimension is 768. These lengths are not reported in the paper (`options.py`). These are entirely continuous prompts; no literal words such as “real photo” are used.

**Intuition.** The repository supplies stable, diverse scaffolds. The four length-one adjustments are cheap residuals shared across segment tokens. This factorization limits per-image freedom, which can regularize a small training domain, while the repository prevents every sample from being just a noisy shift of one template.

### 4.5 Text encoding, similarity, and prediction

The frozen CLIP text transformer maps each fused prompt to `f^r_{text,i}` or `f^f_{text,i}`. For pair `i`:

\[
s_i^r=\cos(X_{cls},f^r_{text,i}),\qquad s_i^f=\cos(X_{cls},f^f_{text,i}),
\]
\[
P_i^r=\frac{e^{s_i^r/τ}}{e^{s_i^r/τ}+e^{s_i^f/τ}},\qquad P_i^f=1-P_i^r.
\]

[Eq. 10–11, p. 21320]

At inference:

\[
\bar P^r=\frac1{BN}\sum_{i=1}^{BN}P_i^r,\qquad \bar P^f=1-\bar P^r.
\]

The predicted label is the larger averaged probability. Default paper setting is `N=4`, hence 8 prompt pairs and 16 text sequences per image. [Eq. 12, p. 21320; implementation details, p. 21321]

**Code clarification:** released code adds an unreported trainable 768→768 `class_mapping` after text encoding and uses a learned temperature initialized to `τ=.07`. It defaults to `N=10`, not the paper's stated `N=4` (`networks/model_engine.py`, `options.py`).

## 5. Training objective

### 5.1 Prompt diversity

With one MC draw, the same-class embeddings of different repository entries are pushed toward orthogonality:

\[
L_{ort}=\sum_{i=1}^B\sum_{j\ne i}
\left[\cos(t^r_{i,1},t^r_{j,1})^2+\cos(t^f_{i,1},t^f_{j,1})^2\right].
\]

[Eq. 7, p. 21320]

### 5.2 Variational terms

For each latent distribution, regularize the flowed posterior toward prior `p(Φ)`:

\[
L_{KL}=E_{q_0(Φ_0)}[\log q_K(Φ_K)-\log p(Φ_K)].
\]

Only ISD is reconstructed. A decoder maps `Φ_K^s` back to `X_rec`, with

\[
L_{rec}=\sum_{j=1}^d(x_j-\hat x_j)^2.
\]

[Eqs. 8–9, p. 21320]

The supplement frames data as `D={I,Y}`, factorizes

\[
q_γ(Φ|D)\approx q_{γ_s}(Φ_s|I)q_{γ_g}(Φ_g)q_{γ_c}(Φ_c),
\]

and derives the negative ELBO as KL minus expected log likelihood. With a Gaussian observation model, `-log p(X_cls|Φ_s)` is proportional to the MSE above. The joint KL is the sum over generic, specific, and class components. [Eqs. 16–22, supplement pp. 1–2]

**Critical qualification:** this derivation reconstructs `X_cls`, not label `Y`, despite defining `D={I,Y}`. The actual class likelihood is supplied separately by discriminative cross-entropy. Thus the claimed ELBO is a regularizer on prompt latents, not a complete generative likelihood for the detection task.

### 5.3 Full loss and sampling schedule

\[
L=L_{cls}+λ_{con}L_{con}+λ_{ort}L_{ort}+λ_{kl}L_{kl}+λ_{rec}L_{rec}.
\]

Default weights inferred from the stated optima are `λ_con=.5`, `λ_ort=1.0`, `λ_kl=.001`, `λ_rec=.5`. [Eq. 13, p. 21320; Fig. 4/Table 6, p. 21323]

- **Training:** one MC sample for efficiency. The paper says it randomly chooses one of the `B` prompt-pair predictions *for each input* and applies two-class cross-entropy. [Sec. 3.3, p. 21320]
- **Inference:** sample `N` times and average probabilities over all `B×N` pairs. [Eq. 12, p. 21320]
- **Code mismatch:** the implementation samples one repository index for the whole batch, not independently per input (`networks/model_engine.py`).

## 6. Data and evaluation protocol

The paper says it follows prior protocols and retrains on the designated source of each benchmark. It does **not** demonstrate one frozen detector trained once and evaluated across all three benchmarks. [Sec. 4.1, p. 21321]

1. **Ojha / UniversalFakeDetect:** train on ProGAN; test unseen GAN/diffusion sources: DALL-E, GLIDE `(100_10, 100_27, 50_27)`, ADM, and LDM `(100, 200, 200_cfg)`. Metric: accuracy and macro mean over eight subsets. [Dataset paragraph and Table 1, p. 21321]
2. **GenImage:** train on Stable Diffusion v1.4; evaluate Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN. Metric: accuracy and unweighted mean over eight subsets. Note that the source/training generator SD v1.4 is included in the reported “cross-generator” mean. [Dataset paragraph and Table 2, p. 21321]
3. **DRCT:** described as testing high-fidelity diffusion reconstruction and localized inpainting. Table 3 covers 16 generators/variants, grouped as SD, Turbo, LCM, ControlNet, and diffusion-reconstruction (DR) variants. The paper does not state the exact DRCT training split/source in enough detail to reproduce it. [Dataset paragraph, p. 21321; Table 3, p. 21322]
4. **Degradations:** on GenImage only, Gaussian blur `σ={0,1,2,3}` and JPEG quality `{100,95,90,85,80}`. Blur uses AUC; JPEG uses AP. SAFE and LOTA use their official models. [Sec. 8.2 and Fig. 7, supplement p. 2]

### Training recipe reported in the paper

- CLIP ViT-L/14; Adam; LR `1e-4`; batch 48; **1 epoch**.
- Paper says resize to 224, random crop in training, center crop in testing.
- LoRA `r=4`, `α_lora=.5`, visual blocks 12–23.
- PWCL `α=.5`, margin `m=1`; repository `B=2`; flow length `K=10`; inference `N=4`.

[Sec. 4.1, pp. 21320–21321]

**Released-code additions/mismatches:** it first ensures the short side is at least 256, then takes a 224 crop; adds random horizontal flip; uses gradient accumulation 2 (effective batch 96); fixes seed 1029; defaults to 100 epochs, not 1; and contains validation-driven LR scheduling and test-accuracy-based checkpoint selection. No DRCT config, degradation evaluation script, pretrained checkpoint, or usable README is present at the inspected commit. `test.py` refers to a nonexistent default checkpoint. These issues make the headline results hard to reproduce from the release as-is.

## 7. Main quantitative results

### 7.1 Ojha, accuracy (%)

PPM-CLIP's complete row is:

| DALL-E | GLIDE 100_10 | GLIDE 100_27 | GLIDE 50_27 | ADM | LDM 100 | LDM 200 | LDM 200_cfg | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 98.4±1.3 | 99.3±0.0 | 99.3±0.1 | 99.3±0.1 | 97.0±0.4 | 99.3±0.0 | 99.1±0.1 | 98.4±0.3 | **98.8±0.2** |

Baseline mean accuracies: CNNDet 52.8, FreDect 54.5, F3Net 79.1, LGrad 90.9, UnivFD 86.9, PatchCraft 84.0, FreqNet 89.5, NPR 95.1, FatFormer 93.6, SAFE 95.7, CoD 97.5. Thus the gain over the strongest listed mean, CoD, is **+1.3 points**. The prose emphasizes +11.9 over UnivFD, but UnivFD is not the strongest comparison. [Table 1 and discussion, p. 21321]

### 7.2 GenImage, accuracy (%)

| Midjourney | SD 1.4 | SD 1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 98.7±1.5 | 100.0±0.0 | 99.9±0.0 | 99.2±0.9 | 99.4±0.7 | 100.0±0.0 | 100.0±0.0 | 99.9±0.2 | **99.6±0.4** |

Baseline means: CNNDet 53.3, FreDect 41.2, LGrad 61.8, UnivFD 88.8, PatchCraft 86.3, FreqNet 86.8, NPR 88.6, FatFormer 88.9, AIDE 86.9, Effort 90.1, SAFE 95.6, C2P-CLIP 95.8, CoD 96.2, LOTA 98.9. Gain over LOTA is **+0.7 point**. FatFormer illustrates domain instability despite an 88.9 mean: 100.0 on SD1.4 but 75.9 ADM and 55.8 BigGAN. [Table 2 and discussion, pp. 21321–21322]

The tables show `±` for PPM-CLIP on Ojha/GenImage but do not state the number of runs, whether these are seeds or repeated MC evaluations, or how the mean/dispersion was computed.

### 7.3 DRCT, accuracy (%)

| LDM | SD1.4 | SD1.5 | SD2 | SDXL Refiner | SDXL-Refiner | SD-Turbo | SDXL-Turbo | LCM-SD1.5 | LCM-SDXL | SD1-Ctrl | SD2-Ctrl | SDXL-Ctrl | SD1-DR | SD2-DR | SDXL-DR | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99.83 | 99.82 | 99.80 | 99.84 | 99.81 | 99.80 | 99.85 | 99.83 | 99.80 | 99.82 | 99.78 | 99.83 | 99.84 | 88.92 | 88.68 | **56.67** | **95.72** |

Baseline means: CNNSpot 81.12, F3Net 77.13, CLIP/RN50 80.05, GramNet 76.62, De-fake 75.52, Conv-B 79.11, UnivFD 83.46, DIRE 72.55, DRCT 91.35. PPM-CLIP gains **+4.37** mean points over DRCT. [Table 3, p. 21322]

But the distribution matters: PPM-CLIP is weak on SDXL-DR (56.67) and loses to DRCT (67.61) there. Its SD1/SD2-DR mean is `(88.92+88.68)/2 = 88.80`, while the prose reports **87.80**, an arithmetic/typographical inconsistency. [Table 3 and DRCT discussion, p. 21322]

### 7.4 Robustness to degradations

Figure 7 has no value labels. The following are approximate readings from the plotted markers, not author-tabulated exact values:

| Blur σ (AUC %) | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| PPM-CLIP | ~100 | ~98 | ~91 | ~89–90 |
| UnivFD | ~98 | ~85 | ~78 | ~76 |
| LOTA | ~99 | ~61 | ~59 | ~59 |
| SAFE | ~99 | ~77 | ~70 | ~70 |

| JPEG quality (AP %) | 100 | 95 | 90 | 85 | 80 |
|---|---:|---:|---:|---:|---:|
| PPM-CLIP | ~100 | ~95 | ~94 | ~92 | ~88 |
| UnivFD | ~98 | ~90 | ~89 | ~88 | ~86 |
| LOTA | ~98 | ~50 | ~50 | ~50 | ~50 |
| SAFE | ~100 | ~60 | ~58 | ~61 | ~60 |

The authors' only explicit numeric robustness statement is approximately **90% AUC at blur σ=3**. PPM-CLIP is clearly best throughout the plot. However, only blur/JPEG are tested; there is no resize, noise, crop, color, screenshot, WebP, or composed-corruption experiment. AUC for one corruption and AP for the other also prevents a single operational robustness comparison. [Sec. 8.2 and Fig. 7, supplement p. 2]

### 7.5 Individual prompt reliability

Figure 6 uses `B=2,N=10` (20 hypotheses) on two examples for each train|test scenario. Ensemble confidence values are `.97, .98` (GAN|GAN), `.90, .66` (GAN|SD), `.99, .99` (SD|GAN), and `1.00,1.00` (SD|SD). Some single GAN→SD hypotheses fall to `.18, .19, .22`; averaging raises them to `.66` on one example and `.90` on the other. This supports variance reduction, but `.66` is not strong confidence and the supplement does not define how the eight examples were selected. [Fig. 6 and Sec. 8.1, supplement p. 2]

## 8. Ablations

### 8.1 Top-level modules

Replacement with zero tensor on GenImage: [Table 4, p. 21322]

| Prompt Flow | Repository | PWCL | mAcc |
|:--:|:--:|:--:|---:|
| – | – | ✓ | 91.2 |
| – | ✓ | ✓ | 92.1 |
| ✓ | – | ✓ | 88.9 |
| ✓ | ✓ | – | 80.0 |
| ✓ | ✓ | ✓ | **99.6** |

Additional text result: replace DCT frequency selection with random patch selection → **95.0**, versus 99.6 full. [Sec. 4.3, p. 21322]

**Reading:** PWCL is essential in the full system (−19.6 without it). Prompt flow adds +7.5 over repository+PWCL. Repository adds +10.7 over flow+PWCL. But flow alone actually hurts the PWCL-only row (88.9 vs 91.2); the gain is strongly interaction-dependent. Zero replacement is not a matched deterministic baseline.

### 8.2 IGD / ISD / ICD combinations

[Table 7, p. 21323]

| IGD | ISD | ICD | mAcc |
|:--:|:--:|:--:|---:|
| – | – | – | 92.1 |
| – | – | ✓ | 92.6 |
| – | ✓ | – | 95.5 |
| ✓ | – | – | 91.3 |
| – | ✓ | ✓ | 98.0 |
| ✓ | – | ✓ | 93.1 |
| ✓ | ✓ | – | 96.3 |
| ✓ | ✓ | ✓ | **99.6** |

ISD is the strongest singleton (+3.4); IGD alone slightly degrades performance (−0.8); ICD gives +0.5. Full performance needs the interactions. Figure 5's t-SNE shows increasing visual separation for ISD, IGD+ISD, and ICD+ISD, but t-SNE is qualitative and can manufacture apparent clusters; no quantitative representation metric is supplied. [Fig. 5 and discussion, p. 21323]

### 8.3 Sampling/cost sweep

[Table 5, p. 21323]

| Method / N | Memory MB | Acc % | FPS |
|---|---:|---:|---:|
| UnivFD | 3116 | 88.8 | 87.44 |
| PPM, 2 | 3200 | 98.6 | 50.51 |
| PPM, 4 | 4764 | 99.6 | 35.61 |
| PPM, 6 | 5542 | 99.8 | 27.17 |
| PPM, 8 | 6496 | 99.9 | 22.19 |
| PPM, 10 | 7976 | 100.0 | 18.76 |

At the paper default `N=4`, memory is +52.9% and throughput is −59.3% relative to UnivFD. At `N=10`, memory is +156.0% and throughput is −78.5%. The “negligible inference overhead” wording near Eq. 6 is contradicted by this measured text-ensemble cost. Hardware, precision, batch size, and timing method are not stated, so FPS is only an internal relative comparison.

### 8.4 Loss and selection hyperparameters

- `λ_KL={.0001,.001,.01,.1}` → mAcc `{92.3,99.6,98.6,97.1}`. A very small but nonzero `.001` is sharply optimal. [Table 6, p. 21323]
- Authors state `λ_rec=.5`, `λ_con=.5`, `λ_ort=1.0`, and patch ratio `α=.5` are optimal. Figure 4 only plots markers, so exact non-optimal values are not tabulated. [Fig. 4 and text, p. 21323]
- Interpretation offered by authors: too much KL collapses useful prompt variance; auxiliary losses should gently regularize rather than dominate; `α=.5` balances high- and low-frequency patches. [Sec. 4.3, p. 21323]

### Missing decisive ablations

The paper does **not** compare against: deterministic conditional prompt MLP with equal parameters; Gaussian sampling without flows (`K=0`); flow mean only; multiple deterministic repository prompts without MC; test-time augmentation ensemble at equal compute; averaging logits versus probabilities; frozen CLIP+PWCL with an ordinary classifier; different backbones; or equal-FPS/equal-memory baselines. These omissions prevent attributing the gain specifically to probabilistic flow modeling rather than conditional prompting, extra capacity, frequency supervision, or ensembling.

## 9. Compute and implementation cost

### Reported

- Training: 1 epoch, batch 48, 224 crops, Adam 1e-4. No GPU model, wall time, FLOPs, energy, number of training images, or trainable parameter count. [Sec. 4.1, pp. 21320–21321]
- Inference memory/FPS are in Table 5 above. Accuracy saturates quickly: 98.6 at `N=2`, 99.6 at 4, then only +0.4 through 10. `N=2` is the best practical point if 98.6 is sufficient.
- A forward pass encodes one image but **2BN** soft-text sequences. With paper defaults, that is 16 sequences/image; at `N=10`, 40.

### Code-based estimate

OpenAI CLIP ViT-L/14 is roughly 428M parameters. The released implementation adds about **25.8M trainable parameters** (approximate count from layer shapes, including some registered but apparently unused mappings): about 0.295M QKV-LoRA parameters, 0.046M repository embeddings, and roughly 25M PFM encoders/decoders/flow and mapping weights. Total model size is roughly **453M**, far below 2B but not a tiny detector. The PFM's amortized `u,w` heads are large because each predicts 10×768 values.

## 10. Limitations and confounds

### Limitations visible in the paper/supplement

1. **The central causal claim is under-ablated.** A static prompt ablation is not an equal-capacity deterministic conditional prompt baseline. Zeroing modules changes optimization and representation scale.
2. **Most of the full-system dependency is PWCL.** Removing PWCL costs 19.6 points, larger than removing PFM (7.5). This weakens the claim that the generative reframing alone explains SOTA.
3. **Cross-generator is not cross-everything.** Each benchmark has its own source training protocol. Cross-dataset/domain deployment, camera images, social-media images, and generators newer than the benchmark are not tested.
4. **Potential dataset/source shortcuts.** Near-perfect GenImage accuracy can reflect generator-correlated content, real-image sources, resolution, or encoding as well as universal synthesis cues. No source-matched controls, deduplication analysis, semantic balancing, or leave-real-source-out test is reported.
5. **Limited corruption suite.** Only standalone blur and JPEG are studied, with different metrics. No composed perturbations or classwise/FPR analysis.
6. **Accuracy hides operating risk.** There is no ROC at low FPR, calibration/ECE, confidence interval, threshold transfer, real/fake class accuracy, or prevalence analysis. These matter more than balanced accuracy for authentic-image false positives.
7. **Frequency prior can be brittle.** It assumes high DCT energy is forensic evidence. Authentic textures, text, foliage, demosaicing/sharpening, and sensor noise are natural high-frequency sources; heavy compression/blur destroys the cue. The paper gives no false-positive taxonomy.
8. **DRCT average hides a failure.** SDXL-DR is 56.67, close to chance and below DRCT's 67.61, despite a 95.72 macro mean.
9. **Unspecified statistical protocol.** `±` values appear only for the proposed method on two tables, with no run count or baseline variance. Hyperparameters appear tuned and ablated on GenImage, with no clear held-out selection protocol.
10. **No stated author limitations section.** The conclusion only restates benefits. [Sec. 5, p. 21323]

### Mathematical/implementation concerns

1. Planar-flow invertibility is asserted but not guaranteed in the equations/code as released.
2. The supplement's “ELBO on `D={I,Y}`” does not model `Y` in its likelihood; reconstruction is only of `X_cls`.
3. The prior `p(Φ)`, PFM hidden dimensions, prompt lengths, and temperature are absent from the paper. The code suggests standard normal, 512 hidden units, lengths 3/7/10, and learned temperature.
4. Code and paper disagree on one prompt index per image versus per batch, `N=4` versus default 10, 1 epoch versus default 100, and some preprocessing/PWCL details.
5. The released training loop can evaluate the test benchmarks during training and choose checkpoints by test mean accuracy when run for multiple epochs. That is test leakage/model selection, though a strict one-epoch reproduction would reduce its effect.
6. The repository currently lacks checkpoints and key scripts/configs needed for DRCT and Fig. 7. Reproducibility is therefore incomplete.
7. Inference is stochastic and the paper does not report variance across inference draws or specify deterministic seed handling. It uses the ensemble only for a mean score, not a calibrated epistemic-uncertainty signal.

## 11. Architecture-taste lessons for a robust hackathon detector

### Good ideas worth borrowing

1. **Make the forensic prior an auxiliary training signal, not a mandatory inference branch.** PWCL injects DCT knowledge while deployment still uses one visual encoder. This is cleaner than concatenating a costly fixed frequency tower.
2. **Use parameter-efficient late-layer adaptation.** LoRA in blocks 12–23 changes CLIP enough to encode texture without full fine-tuning. It is a sensible compute/overfit compromise.
3. **Separate stable basis from input-conditioned residual.** Learned repository templates plus a compact per-image adjustment are a strong pattern. It constrains adaptation and supplies diversity.
4. **Train cheap, ensemble only when budget permits.** One sample in training and 2–4 at inference gives most of the gain. Table 5 shows little value beyond 4.
5. **Make stochasticity semantically factorized.** Generic, image-specific, and class-specific factors are easier to inspect and ablate than one opaque large prompt generator.
6. **Average probabilities, not hard votes.** This preserves each hypothesis's confidence and produces a smooth score usable by the required JSON interface.

### What to simplify or fix before adoption

1. Start with an equal-capacity **deterministic image-conditioned prompt** and/or small MLP classifier. Add Gaussian/flow sampling only if cross-generator validation shows a real gain at fixed compute.
2. Use `N=2` or cache global/class text embeddings. Only ISD varies per image. Do not run 16–40 full text sequences if latency matters.
3. Pair PWCL with robust augmentation. DCT-only selection risks learning codec/sharpness. Include JPEG/WebP, resize, blur, noise, crop, and their compositions during training and evaluation.
4. Replace the single extreme-frequency anchor with several anchors or robust prototypes. One noisy/textured patch can dominate the current loss.
5. Guarantee flow invertibility or use a simpler diagonal Gaussian. A probabilistic story is not worth fragile density code if sampling diversity, rather than exact likelihood, creates the benefit.
6. Report and optimize classwise metrics, AUC/AP, TPR at fixed low FPR, and calibration—not only balanced-set accuracy.
7. Lock a validation set and never select on benchmark test subsets. Record multiple seeds and deterministic inference settings.
8. For this project's real-world robustness focus, the strongest transferable concept is **frequency-guided LoRA + compact conditional ensemble**, not the paper's broader “generative replaces discriminative” claim.

## 12. Bottom line: claims versus evidence

- **Well supported:** the full model is very strong on the three reported benchmark protocols; PPM+repository+PWCL interact strongly; frequency-ranked PWCL beats random selection; MC averaging reduces some individual prompt failures; PPM-CLIP is substantially more robust than the plotted baselines to blur and JPEG.
- **Partially supported:** stochastic prompt hypotheses improve over the authors' static/zeroed variant, but the experiment does not isolate probability, planar flows, dynamic conditioning, extra parameters, and ensembling.
- **Not established:** that static discriminative methods “inevitably” fail, that PPM models the true forgery distribution, that it is universally generalizable, or that the method is production-robust under broad real-world transformations and low false-positive constraints.
