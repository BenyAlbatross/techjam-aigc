# Deep reading notes: arXiv:2602.02222 and arXiv:2602.01738

These notes are based on the **full current PDFs**, including the appendices, rather than only the abstracts. Page numbers below are PDF page numbers. I use **Paper claim** for the authors' stated interpretation and **Critique** for my own assessment.

## Why these papers matter to this repository

The repository's target is binary, image-level classification of **purely generated** versus **authentic** images, with a final model below 2B parameters. The hidden test is expected to include old generators, new diffusion-transformer generators, and JPEG, blur, resize, noise, color jitter, and crop. Direct replication of an existing detector is disallowed. The two papers therefore serve best as:

1. strong baselines and controls;
2. evidence about which design choices generalize;
3. warnings about evaluation leakage and robustness claims; and
4. ingredients to transform into a distinct, challenge-specific contribution, not recipes to copy.

---

# 1. MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection

## 1.1 Bibliographic record and version read

- **Title:** *MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection*.
- **Authors:** Ruiqi Liu, Manni Cui, Ziheng Qin, Zhiyuan Yan, Ruoxin Chen, Yi Han, Zhiheng Li, Junkai Chen, ZhiJin Chen, Kaiqing Lin, Jialiang Shen, Lubin Weng, Jing Dong, Yan Wang, and Shu Wu.
- **Affiliations:** Institute of Automation, Chinese Academy of Sciences; School of Advanced Interdisciplinary Sciences, UCAS; Huazhong University of Science and Technology; Tencent YouTu Lab; Southwest University; Peking University; The University of Sydney; Shenzhen University; and Tsinghua University (PDF p.1). The first three authors are marked equal contributors. Yan Wang and Shu Wu are corresponding authors.
- **Identifier:** arXiv:2602.02222; arXiv DOI `10.48550/arXiv.2602.02222`.
- **Version read:** **v1**, submitted 2 February 2026 at 15:28:17 UTC. The PDF footer says `arXiv:2602.02222v1 [cs.CV] 2 Feb 2026` (p.1). The arXiv history lists only v1, 26,826 KB.
- **Subjects:** Computer Vision and Pattern Recognition (`cs.CV`, primary) and Cryptography and Security (`cs.CR`).
- **Publication status visible in this version:** arXiv preprint. No DOI or journal reference is listed on the arXiv record.
- **Paper length:** 15 PDF pages: 8 pages of main paper, references on pp.9–10, and appendices on pp.11–15.
- **Code link stated in paper:** <https://github.com/349793927/MIRROR> (abstract, p.1).

## 1.2 Motivating argument and assumptions

### Paper claim

The paper says conventional detectors learn the moving distribution of fake artifacts. That distribution changes whenever generators, decoders, or post-processing change. Scaling the backbone does not fix the unstable decision anchor. Humans instead compare an observation with an internal prediction of how a real scene should look. MIRROR tries to encode that same stable anchor: learn a representation of the real-image manifold, reconstruct an input using only “reality” prototypes, and classify the mismatch (Fig.1, pp.1–2; Sec.3, pp.3–4).

The argument has three linked premises:

1. **Manifold premise:** authentic images occupy a lower-dimensional manifold in the encoder feature space (Sec.4.1, p.4).
2. **Reference-comparison premise:** human perception constructs an expected real reference and reacts to prediction error. The paper cites predictive-coding and recognition work as motivation (Sec.3, pp.3–4).
3. **Stationarity premise:** real-world regularities are more stable across time than generator artifacts, so a real-only reference should generalize to unseen generators (pp.2–4).

The psychophysics study is used to support the analogy. In Fig.3, the explicit-reference condition is plotted as improving 65.8% to 88.4%, and CV experts score 86.5% versus lay users at 65.8%; the fake pool is shown as 76.9% easy and 23.1% hard. These numbers do not match Fig.2's group scores, and the paper does not explain the different task/aggregation. The authors infer that stronger real-world priors, rather than memorized generator artifacts, explain human performance (Fig.3 and Sec.3, p.4).

### Why the idea could work

- A real-only dictionary cannot deliberately memorize a particular fake generator in phase 1.
- Sparse reconstruction can expose feature components that are uncommon in the real training distribution.
- Patch tokens preserve local evidence that a single globally pooled embedding can wash out.
- Retrieval uncertainty and reconstruction residual are complementary. A sample can have no close prototype, or it can retrieve one confidently while retaining a large unexplained residual.
- Fixing the reference module during binary training is intended to stop the memory from absorbing fake artifacts.
- A modern DINO representation supplies semantic and structural invariances that can survive mild JPEG, resize, and blur better than raw high-frequency fingerprints.

### Critique of the assumptions

1. **“The real manifold” is not stationary or singular.** COCO photographs are a narrow subset of authentic content. Authentic illustrations, scanned documents, screenshots, low-light phone photos, CGI used in legitimate media, medical imagery, and unusual camera pipelines can all lie far from a COCO-derived reference. A one-class reference can turn authentic-domain novelty into a false positive.
2. **K prototypes are not a manifold proof.** The learned attention dictionary is a useful model, but it does not establish that the reconstructed tokens lie on a physical-world manifold. Learned key/value projections can create a discriminative codebook without a literal manifold interpretation.
3. **The “orthogonal prototypes” description is mathematically impossible at the reported dimensions.** The main experiment uses `K=4096`; DINOv3-L features are lower dimensional (the released implementation uses 1024). More than D mutually orthogonal nonzero vectors cannot exist in a D-dimensional space. The loss `||MM^T-I||_F` can encourage a diverse/tight frame but cannot produce 4,096 pairwise orthogonal 1,024-D prototypes. This should be described as a diversity regularizer, not exact orthogonality (Eq.3, p.4; implementation details, p.6).
4. **The human analogy is suggestive, not causal.** CV experts may have seen generated images and learned artifact cues. Reference images may make a paired comparison easier for reasons unrelated to an internal real manifold. The paper does not isolate these alternatives.
5. **Modern-backbone exposure is a major alternate explanation.** The second paper reviewed below shows that DINOv3 and recent VLM features already linearly separate real/fake data. MIRROR does not provide the most important matched control: the same DINOv3 checkpoint, same input pipeline, and a frozen linear head versus MIRROR.

## 1.3 Exact architecture and data flow

The architecture is presented in Fig.4 and Eqs.1–6 (pp.4–5).

### Phase 1: encode reality priors

1. Input real image `I` is cropped to 224×224.
2. A **frozen DINO encoder** `E` produces patch features:
   `F = E(I) ∈ R^(N×D)` (Eq.1, p.4).
3. A trainable memory `M ∈ R^(K×D)` contains K prototype vectors.
4. Cross-attention maps each patch onto memory prototypes. `Q` comes from `F`; `K` and `V` come from `M`. The scaled dot products are restricted to the top-k entries and softmax-normalized:
   `A = Softmax(Top-k(QK^T / sqrt(D)))`, `F_hat = A V` (Eq.2, p.4).
5. `F_hat` is the “Ideal Reference”: a sparse linear combination of real-derived prototypes for each input patch.
6. Only the memory/projection path is meant to learn. The DINO encoder stays frozen.
7. The phase-1 loss on **real images only** is:
   `L_phase1 = ||F - F_hat||_2^2 + λ ||M M^T - I||_F` (Eq.3, p.4).
   The first term fits authentic patch features; the second discourages redundant prototypes.

### Phase 2: compare input and reference

1. The phase-1 memory **and projection module are frozen** (Sec.4.2, p.4).
2. The input may now be real or fake. DINO produces patch features `F`.
3. Frozen sparse memory attention returns `A` and `F_hat`.
4. **Reconstruct-perplexity branch:** compute maximum attention score `s_max` and attention entropy `s_ent`; send the two values through `MLP_per` to form evidence `V_per` (Eq.4, p.5).
5. **Comparison-residual branch:** compute `ΔF = F-F_hat`, then a linear projection gives `V_res` (Eq.4, p.5).
6. Concatenate the evidence and predict fake probability:
   `y_pred = MLP_C(Concat[V_per, V_res])` (Eq.5, p.5).
7. Optimize binary cross-entropy (Eq.6, p.5).

### Backbone and trainable parts in the reported experiment

- Main setup: DINOv3-Large, 224×224 input, patch tokens (Sec.6.1, p.6).
- Phase 1: frozen DINOv3-L; train memory and its attention/projection parameters using 200,000 MSCOCO real images.
- Memory size `K=4096`; final top-k is 128 (Sec.6.1, p.6; Sec.6.3 and Fig.5, p.8).
- Phase 2: the paper says it uses a **LoRA-finetuned DINOv3** feature extractor while memory and projection remain frozen. The detector heads and LoRA parameters are therefore trainable; the base backbone weights are not directly updated.
- Binary data: the SD v1.4 portion of GenImage. The paper says “finetune the model on SDv1.4” but does not give real/fake counts or sampling ratio (p.6).
- Optimizer: AdamW; initial learning rate `1e-4`; cosine schedule; 5 epochs (p.6).
- Fake SD v1.4 images are JPEG-compressed at quality 96 to align their format with real images (footnote, p.6).
- All images are cropped to 224×224 for both training and inference (p.6).
- Inference returns a binary probability/logit from the classifier. The main metric uses the resulting hard decision; score calibration is not discussed.

### Important specification gaps and an external reproducibility warning

The PDF does **not** specify phase-1 optimizer, learning rate, epoch count, batch size, λ, head sizes, number of attention heads, LoRA rank/targets, augmentation beyond crop/JPEG alignment, validation selection, threshold selection, or seed. Training code was not available in the public repository snapshot inspected while writing these notes; its README marked training code and Human-AIGI release as “coming soon.” This makes exact reproduction impossible from the paper alone.

There is also a material paper/code discrepancy worth checking before borrowing any result. Eq.5 says the final classifier uses only `[V_per,V_res]`. The public inference model at commit `18c56efa...` additionally concatenated the raw DINO CLS token into the final classifier. That creates a direct conventional-classification route and weakens the claim that decisions strictly come from reference comparison. The same code instantiated rank-8 Q/K/V LoRA and an eight-head memory attention, details absent from the PDF. Treat this as an implementation snapshot, not as a paper claim, but require an ablation that removes CLS if testing the idea.

## 1.4 Human-AIGI benchmark

### Construction claimed by the paper

- More than 30,000 generated images from **27 generators** in T2I-COREBENCH (Sec.5.1, p.5).
- Generator families include diffusion, autoregressive, unified, and closed-source systems. Table 3 lists SD3/3.5, FLUX variants, PixArt, HiDream, Qwen-Image, Infinity, GoT-R1, BAGEL, Show-o2, Janus-Pro, BLIP3o, OmniGen2, Seedream 3.0, Gemini 2.0 Flash, Nano Banana, Imagen 4, HunyuanImage 3.0, and GPT-Image (p.7).
- 50 participants divided into lay users, CV experts, and AIGI-detection experts (Sec.5.1, p.5; Fig.2, p.2).
- Each trial records binary response, perceived-realism/confidence `S ∈ {1,2,3,4}`, and response time `RT`.
- The hard set contains generated images that are confidently perceived as real or produce unusually long hesitation:
  `D_hard={x∈X_gen | S(x)≥τ_real OR RT(x)>μ_rt+σ_rt}` (Eq.7, p.5).
- Figure 2 reports Human-AIGI scaling: NPR ResNet-18/50/101 = 64.5/65.1/64.2; UnivFD CLIP-S/B/L = 76.8/77.0/79.4; DDA DINOv2-S/B/L = 84.5/86.5/87.2; lay/CV/AIGI-detector experts = 76.9/88.3/94.5; MIRROR DINOv3-B/L/H = 86.7/89.6/93.6 (p.2).

### Critical validity gaps

The PDF omits participant demographics, recruitment criteria, compensation/ethics, display and viewing conditions, exact reference-image protocol, number of ratings per image, class balance, trial randomization, training trials, the value of `τ_real`, aggregation across observers, and confidence intervals. These are not minor omissions for a psychophysical benchmark.

More importantly, Eq.7 explicitly selects from `X_gen`, and Table 3 is organized only by fake generator. Its “accuracy” is therefore effectively fake recall on the selected samples, not balanced real/fake accuracy. Calling 89.5% on `D_hard` a balanced accuracy on p.8 is at least unclear. A detector can obtain high hard-set fake recall by lowering its threshold and simultaneously create unacceptable false positives on authentic images. “Superhuman crossover” needs a matched ROC operating point or matched false-positive rate, not fake-only accuracy.

The hard-set cell increments suggest that several per-generator subsets are small. Examples such as 93.8%, 14.3%, and 3.3% are consistent with only tens of samples. No uncertainty is reported. The benchmark was also not released at the time inspected, blocking independent verification.

## 1.5 Datasets, splits, protocols, and headline evidence

### Training data

- Phase 1: 200k **real-only MSCOCO** images (p.6).
- Phase 2: GenImage Stable Diffusion v1.4 real/fake training data, with fake images re-encoded at JPEG Q=96 for format alignment (p.6 and footnote).
- The paper does not state exact counts, class sampling, a validation split, or whether any benchmark images overlap DINOv3 pretraining.

### Six standard benchmarks: Table 1, p.6

The paper reports Balanced Accuracy, mild JPEG robustness at Q=90, and mild resize robustness at scale 0.9. The DINOv3 version obtains:

| Benchmark | B.Acc | JPEG Q90 | resize 0.9 |
|---|---:|---:|---:|
| AIGCDetect | 91.7 | 91.5 | 91.5 |
| GenImage | 94.2 | 96.7 | 94.9 |
| DRCT-2M | 93.0 | 90.4 | 93.4 |
| UnivFakeDetect | 88.2 | 88.3 | 87.5 |
| Synthbuster | 98.1 | 97.5 | 97.6 |
| EvalGEN | 99.0 | 98.5 | 98.6 |
| **Macro average** | **94.0** | **93.8** | **93.9** |

The strongest prior aggregate in the table is B-Free at 91.9/91.8/90.4. Thus MIRROR's reported gains are +2.1 B.Acc, +2.0 JPEG, and +3.5 resize. DINOv2 MIRROR reaches 92.4/92.1/92.6.

Appendix Tables 5–9 (p.14) expose useful per-generator detail:

- **AIGCDetect:** MIRROR averages 91.7 across 17 sources, but is weaker on ADM 75.6, CycleGAN 73.3, and WhichFaceIsReal 78.9 (while BigGAN is 97.2 and Wukong 99.8). This is broad transfer, not uniform transfer.
- **DRCT-2M:** MIRROR averages only 93.0 versus B-Free 99.2 and DDA 97.0. It drops on SDv1-DR 80.9, SDv2-DR 67.9, and SDXL-DR 59.4. This is important evidence that reality reconstruction does not automatically solve diffusion/VAE reconstruction traces.
- **UniversalFakeDetect:** 88.2, narrowly above B-Free 87.8.
- **Synthbuster:** 98.1 across DALL-E 2/3, Firefly, GLIDE, Midjourney, SD1.x/2/XL.
- **EvalGEN:** 99.0 across Flux, GoT, Infinity, NOVA, and OmniGen, versus DDA 94.7 and B-Free 89.6.

### Seven in-the-wild benchmark groups: Table 2, p.6

The macro table expands each group into 16 sources:

- Chameleon;
- SynthWildx: DALL-E 3, Firefly, Midjourney;
- WildRF: Facebook, Reddit, Twitter;
- AIGIBench: SocialRF and CommunityAI;
- CO-SPY in the wild: Civitai, DALL-E 3, instavibe.ai, Lexica, Midjourney v6;
- RRDataset;
- BFree-Online.

MIRROR DINOv3 averages **91.2**, versus DDA 83.1 and B-Free 82.2; DINOv2 MIRROR averages 87.4. The gain is broad but not universal. DINOv3 is only 75.0 on instavibe.ai, 78.9 on RRDataset, and 83.0 on BFree-Online; DDA/B-Free remain stronger on some sources, such as Midjourney v6 and BFree-Online. The largest aggregate claim, +8.1, is relative to DDA's 83.1.

### Human-AIGI: Table 3, p.7

Macro results across 27 fake generators:

| Method | Original | Human-hard |
|---|---:|---:|
| B-Free | 82.9 | 80.8 |
| DDA | 88.1 | 86.9 |
| **MIRROR** | **89.6** | **89.5** |

MIRROR is near 100% on several PixArt, Infinity, GoT, Show-o, and BLIP generators, but its weakest hard-set fake recalls are FLUX.1-Krea-dev 62.5, Nano Banana 64.4, and Imagen 4 52.5. Those failures are especially relevant to a hidden test containing recent commercial generators.

### Corruption evidence: Appendix B and Fig.8, pp.11–12

- JPEG quality 100, 90, 80, 60, 40.
- Resize scales 0.5, 0.75, 1.0, 1.5, 2.0.
- Gaussian blur sigma 0, 0.5, 1.0, 1.5, 2.0.
- MIRROR's plotted balanced-accuracy curve is the most stable among the compared methods.

This is directionally relevant, but the appendix gives curves without a numeric table, dataset identity/aggregation detail, uncertainty, or compound corruptions. It does **not** test the challenge's JPEG Q=30, resize 0.25, Gaussian noise, ±20% color jitter, or 80% center crop.

## 1.6 Strongest ablations and what they actually establish

### Memory source: Table 4a, p.7

| Memory source | Standard | in the wild | Human-AIGI | average |
|---|---:|---:|---:|---:|
| Generated only | 90.5 | 84.4 | 84.9 | 86.6 |
| Mixed real+generated | 87.1 | 82.3 | 84.3 | 84.5 |
| **Real only** | **94.0** | **86.5** | **89.8** | **90.1** |

This supports a real-only memory under their configuration. It does not prove the reason is a real manifold unless sample count, semantic coverage, and optimization are controlled across sources.

### Reference-comparison components: Table 4b, p.7

| Configuration | Standard | in the wild | Human-AIGI | average |
|---|---:|---:|---:|---:|
| Direct classification baseline | 88.4 | 82.2 | 83.7 | 84.7 |
| + retrieval perplexity | 90.1 | 84.3 | 85.0 | 86.4 |
| + residual | 92.5 | 85.1 | 88.5 | 88.7 |
| **Full MIRROR** | **94.0** | **86.5** | **89.8** | **90.1** |

The residual is the larger single addition; combining residual and perplexity is best. However, the baseline architecture, parameter matching, CLS usage, and DINO tuning are not specified well enough to isolate the mechanism.

### Capacity and sparsity: Fig.5, pp.6 and 8

- K swept from 512 to 32,768; average performance peaks around **4096**.
- top-k swept from 16 to 1,024; performance peaks around **128**.
- The authors interpret too-small settings as under-reconstruction and too-large settings as anomaly leakage/redundancy.

The inverted-U result is useful engineering evidence. It is not evidence of prototype orthogonality, and the huge K sweep likely changes compute and regularization strength as well as coverage.

### Scaling: Fig.2, p.2

MIRROR gains 86.7→89.6→93.6 as DINOv3 grows B→L→H, while the chosen NPR/UnivFD/DDA families show smaller scaling. This is intriguing but not a controlled scaling study because architecture families, pretraining data, feature dimensions, and tuning recipes differ.

## 1.7 Limitations, confounds, and comparison fairness

1. **Backbone confound:** no same-DINOv3 frozen-linear baseline. Much of the result may be DINOv3, as arXiv:2602.01738 demonstrates.
2. **Phase mismatch:** phase 1 learns memory in frozen DINO coordinates, while phase 2 LoRA changes the encoder coordinates but freezes memory/projection. LoRA can deliberately reshape fake tokens to generate residuals, reintroducing an artifact classifier through the feature space.
3. **Potential direct CLS shortcut:** current public inference code concatenates raw CLS into the classifier, despite Eq.5. If used for reported checkpoints, the detector is not reference-only.
4. **Real-domain coverage:** COCO-only memory can confuse authentic-domain novelty with synthetic evidence. No authentic-domain leave-one-source-out false-positive table is shown.
5. **Format control is partial:** JPEG-aligning SD1.4 fakes to the real set is good, but all external datasets have different collection, resizing, and encoding histories. Dataset-source prediction remains possible.
6. **Unequal comparator protocols:** DRCT, Aligned, B-Free, and DDA use official weights; architecture methods are retrained on MIRROR's set. Appendix Table 10 usefully reports official versus retrained results and shows very large swings. Neither protocol alone is perfectly fair. A matched-backbone, matched-data comparison is still absent.
7. **Fake-only benchmarks:** Synthbuster and EvalGEN are paired with MSCOCO real performance to create “balanced” scores (Appendix Table 10, p.15), which can introduce a real-source/domain shortcut instead of using source-matched authentic controls.
8. **Fixed-threshold metric only:** Balanced accuracy can collapse because of dataset-dependent calibration even if ranking is good. AUC, AP, TPR at fixed FPR, threshold-selection protocol, and calibration error are omitted.
9. **No uncertainty:** no repeated seeds, confidence intervals, or significance tests. Small per-generator Human-hard cells are particularly fragile.
10. **Robustness is milder than this challenge:** Q90/resize0.9 headline metrics are very mild. The appendix stops at JPEG40 and resize0.5, omits four required corruption families, and does not test chains.
11. **Psychophysics does not establish operational superiority:** the hard subset is fake-only and lacks an equal-FPR comparison. The detector remains below AIGI-detection experts in Fig.2 (93.6 versus 94.5 for the largest models), though it exceeds lay and CV experts.
12. **Interpretability claim is unvalidated:** residual maps in Figs.7 and 9 (pp.8,12–13) look localized, but there are no pixel annotations or insertion/deletion tests showing that hot regions causally correspond to generation errors.
13. **No adversarial or future-real analysis:** a generated image close to common real prototypes may evade detection; rare authentic images may be rejected. No tail-risk study addresses either case.

## 1.8 Implementation cost

### Paper-reported cost

- DINOv3-L backbone plus a 4,096×D memory and attention projections.
- One DINO forward plus patch-to-memory attention at inference.
- 20.03 images/s on one Tesla V100; 1,000 images in 49.91 s (Fig.6, p.8).
- The authors say memory overhead is small relative to the backbone.

### Practical estimate for this repository

- DINOv3-L is comfortably below 2B parameters, but exact checkpoint, license, and packaged parameter count still need to be recorded. DINOv3-H variants may also be below 2B, but memory and latency are much higher.
- A 4,096×1,024 memory itself is about 4.2M parameters. Q/K/V/output projections add roughly another 4.2M at D=1024, plus the detector and LoRA.
- Compute is less benign than parameter count. With about 196 patches, eight heads, and 4,096 slots, attention materializes millions of similarities per image. Phase-1 backpropagation through this module is costly.
- Phase 1 needs feature passes over 200k authentic images. Feature caching can reduce backbone cost but consumes substantial disk.
- Missing training code and hyperparameters make implementation risk high for a hackathon. Exact replication would also violate the challenge's originality guardrail.

## 1.9 Architecture-taste lessons

### Adopt as taste

- Choose a stable reference distribution instead of only chasing known fake artifacts.
- Keep local patch evidence until late in the model.
- Treat uncertainty and residual magnitude as different signals.
- Align encoding/format across classes before training.
- Report per-generator failures; averages hide new-generator blind spots.

### Do not adopt uncritically

- Do not call an overcomplete dictionary “orthogonal.”
- Do not change the feature coordinate system after freezing a memory trained in the old coordinate system without measuring drift.
- Do not let an unexplained raw CLS path bypass the advertised module.
- Do not infer a universal real manifold from COCO.
- Do not use a fake-only hard subset to claim operating-point superiority over humans.

---

# 2. Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models

## 2.1 Bibliographic record and version read

- **Title:** *Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models*.
- **Authors:** Yue Zhou, Xinan He, Kaiqing Lin, Bing Fan, Feng Ding, and Bin Li.
- **Affiliations:** Shenzhen University; Nanchang University; University of North Texas. Bin Li is marked corresponding author (p.1).
- **Identifier:** arXiv:2602.01738; arXiv DOI `10.48550/arXiv.2602.01738`.
- **Version read:** **v2**, submitted 15 April 2026 at 07:37:32 UTC. Version history: v1 on 2 February 2026 at 07:20:02 UTC (314 KB); v2 on 15 April 2026 (656 KB). The PDF says `arXiv:2602.01738v2 [cs.CV] 15 Apr 2026` (p.1).
- **Subject:** Computer Vision and Pattern Recognition (`cs.CV`).
- **Venue text in PDF:** ACM MM ’26, October 2026, Amsterdam, The Netherlands. The headers still say “Anonymous Author(s)” despite the author list (pp.2–8), so this should be cited conservatively as an arXiv preprint/draft unless publication is independently confirmed.
- **No DOI or journal reference** appears on the arXiv record.
- **Length:** 9 PDF pages, with references continuing through p.9. The paper promises detailed model specifications in an appendix (p.3), but this PDF contains no appendix.

## 2.2 Motivating argument and assumptions

### Paper claim

Specialized detectors approach saturation on curated benchmarks but collapse in the wild. Inspired by the “Bitter Lesson,” the paper asks whether newer general-purpose vision foundation models (VFMs) already contain stronger forensic features than bespoke detector modules. Its answer is yes: freeze a modern VFM, attach one linear classifier, and train only that classifier on Stable Diffusion 1.4 (pp.1–3).

The authors attribute this “emergent” capability mainly to synthetic content in newer web-scale pretraining corpora:

- **Vision-language models:** image/text co-occurrence injects explicit concepts such as “AI generated,” “deepfake,” or a platform name.
- **Self-supervised models:** unlabeled web exposure creates an implicit feature separation between authentic and generated distributions.

Figure 1 counts Common Crawl URLs from Civitai and Liblib from 2022–2025 and shows rapid growth after 2023 (pp.1–2). Section 4 then uses text-image probes and a DINOv3 web-versus-satellite comparison as indirect evidence (pp.5–7).

### Why the simple module could work

- Large, diverse pretraining has already learned semantics, material/geometry regularities, image styles, and web-distribution cues.
- A frozen backbone preserves that broad knowledge and reduces overfitting to SD1.4.
- A linear boundary has low capacity, so it is less able to memorize the training generator.
- Newer web-pretrained models may have directly encountered many synthetic families or adjacent styles.
- Global pooled features are a good match for this repository's pure-image, image-level task; localized edits are out of scope.

### Critique of the causal story

The performance evidence for modern frozen features is strong. The causal explanation—synthetic exposure—is plausible but not proven.

- Common Crawl URL counts do not show that a particular model downloaded, retained, or trained on those images.
- Newer and older models differ in data scale, curation, resolution, objectives, architectures, optimization, and compute, not only synthetic exposure.
- Benchmark or near-duplicate exposure is possible because the pretraining corpora are huge and incompletely disclosed.
- Explicit text similarity to “AI generated” can reflect a visual style shortcut, watermark, or source pattern; it does not prove the model learned physical authenticity.
- The DINO satellite counterfactual changes both dataset domain and scale, so it is not a clean synthetic-content ablation.

The paper itself appropriately says a fully controlled pretraining study is beyond scope (abstract and Sec.4, pp.1,6), but later prose uses causal terms such as “proves” and “entirely contingent” (p.7), which exceed the evidence.

## 2.3 Exact detector architecture, training, and inference

### Data flow

1. Resize and center-crop an image to the selected model's native resolution.
2. Run a modern VFM as a **completely frozen** feature extractor.
3. Take the model's **pooled output feature**.
4. Apply one trainable linear layer to predict real versus fake.
5. At inference, run the same preprocessing, one backbone pass, and the linear head. No text encoder or prompt is needed for the deployed classifier.

The evaluated family includes:

- MetaCLIP and MetaCLIP 2;
- SigLIP and SigLIP 2;
- Perception Encoder (PE-CLIP);
- DINOv2 and DINOv3 (Sec.3.1, pp.3–4).

### Trainable parts and optimization

- **Trainable:** only the linear classification head.
- **Frozen:** every backbone parameter.
- **Training data:** GenImage Stable Diffusion v1.4 training subset only.
- **Optimizer:** AdamW.
- **Learning rate:** `1e-3`.
- **Batch size:** 128.
- **Duration:** 2 epochs.
- **Augmentation:** none beyond resize and center crop.
- **Resolution:** each model's native resolution. PE is specified as ViT-L/14 at 336 px (p.4).

The PDF does **not** state the classification loss, weight decay, warmup/schedule, exact pooled layer for each model, checkpoint identifiers, model sizes for most backbones, class counts/balance, seed, threshold selection, or feature normalization. Binary cross-entropy/softmax is likely but should not be presented as specified. The missing appendix prevents exact replication.

### Eligibility problem for this repository

Section 4.2 identifies the web DINOv3 used for the matching headline/analysis numbers as **DINOv3 ViT-7B** on LVD-1689M (p.6), over the challenge's 2B limit. Those DINOv3-Linear results are therefore not an eligible implementation result for this repository. PE ViT-L/14 is far below 2B and is the clearest compliant candidate in the text. Exact MetaCLIP2/SigLIP2 checkpoints also need a parameter and license audit.

## 2.4 Datasets and splits

### Training

All simple probes, and all competitors except DDA, are trained on GenImage SD v1.4 for the authors' fairness protocol. DDA uses official pretrained weights because its contribution depends on VAE-reconstruction alignment (Sec.3.1, p.3). Exact image counts are not given.

### Standard benchmark: GenImage

Eight source columns are evaluated: ADM, BigGAN, Midjourney, VQDM, GLIDE, SD v1.4, SD v1.5, and Wukong (Table 1, p.3). The paper calls evaluation “unseen generators,” but the macro average includes the in-domain SD v1.4 test subset and the closely related SD v1.5 subset. This does not erase the result, but it makes the label imprecise.

### In-the-wild sets

Chameleon, WildRF, SocialRF, and CommunityAI are described as social-media/forum collections with diverse, uncontrolled processing and unknown generators (Sec.3.1, p.3; Table 2, p.4). Per-class real and fake accuracies are shown.

### Recent/unseen-generator sets

- **AIGIHolmes:** ten newer autoregressive, unified, and diffusion-transformer systems: FLUX, Infinity, Janus, Janus-Pro 1B/7B, LlamaGen, PixArt-XL, SD3.5-L, Show-o, and VAR (Table 3, p.5).
- **AIGI-Now:** nine APIs/models, each with a `pix` subset and a strongly degraded `sem` subset: FLUX-dev, FLUX-Krea, FLUX-Kontext, FLUX-Pro, GPT-4o, Jimeng, Keling, Minimax, and Nano Banana (Table 4, p.5). The paper says `pix` format-aligns images to preserve pixel artifacts, while `sem` aggressively degrades them to force semantic detection. Exact degradation operations are not provided here.

### Mechanism-analysis sets

- Existing in-the-wild sets plus **Midjourney-CC**, 3,000 images collected from `reddit.com/r/midjourney` in late 2025, for zero-shot text-image similarity (Sec.4.1, p.6).
- A web-pretrained DINOv3 ViT-7B on LVD-1689M versus a satellite-pretrained version on Sat-493M (Sec.4.2, pp.6–7).

### Robustness/limitation sets

- JPEG and Gaussian blur on GenImage and Chameleon (Sec.5.1 and Fig.2, p.7).
- RRDataset: Original, Redigital screen/print recapture, and Transfer through social apps (Table 7, p.7).
- DDA-COCO: VAE reconstruction by SDXL, SD2, SD3.5-L.
- BR-Gen: local editing with Brush, Power, and SDXL (Table 8, p.8).

No dataset sizes, source-level split counts, duplicate controls, confidence intervals, or public split hashes are given in the PDF.

## 2.5 Main evidence

### GenImage cross-generator: Table 1, p.3

| Probe | Avg accuracy |
|---|---:|
| MetaCLIP | 76.6 |
| MetaCLIP 2 | 89.2 |
| SigLIP | 85.1 |
| SigLIP 2 | 94.5 |
| PE-CLIP | 93.8 |
| DINOv2 | 85.2 |
| **DINOv3** | **96.4** |
| best specialized, OMAT | 94.6 |
| DDA | 89.0 |

DINOv3 is strong on BigGAN 99.1, VQDM 99.2, SD1.4 99.8, SD1.5 99.6, and Wukong 99.4, but its hardest source is ADM 84.9. The leap DINOv2→DINOv3 is 11.2 points; MetaCLIP→MetaCLIP2 is 12.6. Excluding the in-domain SD1.4 column, DINOv3 still averages about 96.0, so the basic result is not caused by that column alone.

### In the wild: Table 2, p.4

| Probe/method | Macro avg over four sets |
|---|---:|
| DINOv2-Linear | 63.6 |
| MetaCLIP2-Linear | 84.2 |
| PE-CLIP-Linear | 89.9 |
| **DINOv3-Linear** | **94.0** |
| DDA | 85.0 |
| most other specialized methods | 52.3–63.6 |

DINOv3 balances both classes well: Chameleon 93.3 real/89.5 fake, WildRF 94.8/97.5, SocialRF 93.7/94.8, CommunityAI 94.9/94.6. Many specialized detectors retain high real accuracy but label almost every fake as real, which is a notable failure mode rather than a small average decline.

### AIGIHolmes: Table 3, p.5

- PE-CLIP: **97.8** average.
- SigLIP2: 97.3.
- DINOv3: 97.2.
- AIDE: 97.0.
- DDA: 96.3.
- MetaCLIP2: 94.2.

PE and DINOv3 remain strong on autoregressive and diffusion-transformer sources. SD3.5-L and VAR are relatively harder for DINOv3 (89.1 and 92.2), but still far above chance.

### AIGI-Now: Table 4, p.5

- **MetaCLIP2:** 90.7 average over 18 model×split cells.
- MetaCLIP: 89.2.
- PE: 89.1.
- DINOv3: 86.4.
- Best specialized result, DDA: 69.5.
- AIDE: 67.2.

The most striking evidence is that modern VFM probes retain useful accuracy on the degraded `sem` halves, while several artifact methods sit near 50%. It supports high-level representation value. It does not show exactly which degradation-invariant cue the probes use because the `sem` construction is not detailed.

## 2.6 Mechanism evidence and critique

### VLM text-image probing: Table 5, p.6

The authors build three text pools: authenticity/forgery concepts, neutral content concepts, and generator/source names. They compare cosine similarities with frozen image embeddings.

- MetaCLIP/MetaCLIP2/PE commonly retrieve “AI generated” on Chameleon, SocialRF, and CommunityAI.
- MetaCLIP2 and PE have stronger similarity than MetaCLIP.
- On Midjourney-CC, MetaCLIP2/PE retrieve `midjourney_images` first and “AI generated” second.
- CLIP, SigLIP, and SigLIP2 generally retrieve content or authenticity-neutral words instead.

**Paper claim:** newer VLMs explicitly internalized forgery/source concepts from web image-text exposure.

**Critique:** this is good evidence of an association in the embedding space, not “definitive” proof of a training-data pathway. The text-pool size, exact templates, preprocessing, and statistical uncertainty are not stated. `midjourney_images` retrieval can be driven by a consistent visual style or embedded marks. Model-specific prompt sensitivity is also uncontrolled.

### Web versus satellite DINOv3: Table 6, pp.6–7

| Pretraining | GenImage avg | Chameleon real | fake | avg |
|---|---:|---:|---:|---:|
| DINOv3-Web, LVD-1689M | 96.5 | 93.3 | 89.5 | 91.4 |
| DINOv3-Sat, Sat-493M | 70.6 | 94.8 | 12.1 | 53.5 |

The satellite model calls almost every fake real. This is strong evidence that architecture alone is insufficient and that pretraining distribution matters.

It is **not** a clean test of synthetic exposure: web and satellite sets differ in image count by more than 3×, subject domain, texture statistics, camera geometry, curation, and probably optimization. The satellite model may simply lack broad natural-image support. The result justifies screening checkpoints by pretraining distribution; it does not prove fake exposure is the sole cause.

## 2.7 Robustness evidence

### Common perturbations: Fig.2, p.7

The paper tests:

- JPEG Q = 95, 85, 75, 65;
- Gaussian blur sigma = 0.5, 1.0, 1.5, 2.0;
- on GenImage and Chameleon.

From the plotted curves:

- DINOv3 is roughly 96% clean on GenImage and remains around 91% at blur 2; on Chameleon it remains around the high 80s.
- MetaCLIP2 is especially stable on Chameleon, near the low/mid 90s across corruptions.
- PE begins very high but falls to **77.8% at blur sigma 2**, cited explicitly in text.
- Older MetaCLIP/SigLIP/DINOv2 are lower and/or more volatile.

This is relevant evidence that modern embeddings carry robust structural/semantic cues. It is incomplete for this challenge: no JPEG below 65, resize, noise, color jitter, crop, or composition. The curves have no numeric table, confidence interval, or calibration analysis.

### Recapture and transfer: Table 7, p.7

Per-class accuracies are reported. Balanced averages computed from the table are:

| Detector | Original | recapture | social transfer |
|---|---:|---:|---:|
| DDA | 91.9 | 63.0 | 73.3 |
| SigLIP2-linear | 90.1 | 72.2 | 59.7 |
| MetaCLIP2-linear | 86.2 | 75.5 | 82.2 |
| PE-linear | 93.7 | 73.1 | 83.7 |
| **DINOv3-linear** | **94.1** | **80.6** | **84.6** |

Modern VFMs clearly reduce the collapse but do not solve it. DINOv3 fake recall falls from 93.0 to 64.7 after recapture and 71.2 after transfer.

There is a narrative inconsistency: the text says MetaCLIP2 “leads with ~72% accuracy across both scenarios,” but DINOv3 has the best balanced average in both, and PE is also higher on transfer. The sentence appears to refer only to fake recall, not total/balanced accuracy.

### Reconstruction and local edit: Table 8, p.8

- On DDA-COCO VAE reconstruction, modern VFM fake-detection rates are only 0.4–17%; DDA is 68.2–99.7%.
- On BR-Gen local edits, modern probes hover around 45–62%; Effort reaches 76.7–80.1% and OMAT 68.1–71.5%.

The stated explanation is global pooling: unchanged real regions dominate local-edit features. These are genuine limitations, but pure VAE reconstruction and partial editing are outside this challenge's stated target. They matter mainly as evidence that “foundation models solve forensics” would be too broad a conclusion.

## 2.8 Strongest ablations

### Backbone generations

The old→new gaps on the same broad model families are among the paper's strongest evidence:

- MetaCLIP→MetaCLIP2: 76.6→89.2 GenImage, 65.4→84.2 in the wild.
- DINOv2→DINOv3: 85.2→96.4 GenImage, 63.6→94.0 in the wild.
- SigLIP→SigLIP2: 85.1→94.5 GenImage, 61.0→82.2 in the wild.

They show that “CLIP-like” or “DINO-like” is not a sufficient model specification. Checkpoint generation and data matter greatly.

### Specialized heads on modern backbones: Table 9, p.8

Replacing legacy backbones improves some expert methods, but the simple probe remains best:

- PE-linear: 93.8 GenImage / 95.9 Chameleon / 97.8 AIGIHolmes.
- AIDE+PE: 88.3 / 91.4 / 94.7.
- Effort+PE: 85.6 / 76.1 / 92.4.
- DDA+DINOv3: 57.2 / 75.1 / 71.3.

**Paper claim:** specialized inductive biases bottleneck strong generic features.

**Critique:** backbone swaps are rarely plug-compatible. Specialized modules depend on layer locations, token statistics, channel widths, normalization, and training schedules. DDA's collapse after replacement looks more like an unretuned integration than proof that its inductive bias is inherently harmful. A fair test needs each method retuned and parameter-matched.

### LoRA versus frozen probe: Table 10, p.8

| Backbone | frozen Gen/Cham/Holmes | LoRA r=4 | LoRA r=8 |
|---|---|---|---|
| MetaCLIP2 | .892/.930/.942 | .734/.817/.823 | .780/.880/.896 |
| PE | .938/.959/.978 | .810/.719/.879 | .761/.635/.891 |
| DINOv3 | .964/.914/.972 | .954/.803/.977 | .928/.718/.945 |

The result warns that narrow single-generator fine-tuning can destroy OOD performance. But the conclusion “LoRA actively dismantles generalizability” is too general. MetaCLIP2 and PE LoRA even lose large amounts on GenImage, the source-family benchmark, suggesting under-optimized hyperparameters or an implementation problem rather than only catastrophic forgetting. The PDF omits LoRA target layers, learning rate, epochs, regularization, whether the linear head is jointly trained, and model selection. Multi-generator training, lower learning rates, partial-layer LoRA, or feature-preservation penalties could change the result.

## 2.9 Limitations, confounds, and fairness

1. **Scale/eligibility:** the DINOv3-Web model tied to the headline analysis is ViT-7B, so the strongest DINOv3 result is ineligible here. Exact variants for several other VFM rows remain unstated. The missing appendix is a major reproducibility defect.
2. **Opaque pretraining overlap:** modern web models may have seen benchmark images, close derivatives, generator-community pages, or the same real sources. High performance can combine generalization with contamination.
3. **Pretraining cutoff assertions are uncertain:** the paper calls some closed APIs unseen at pretraining time, but exact crawl dates and model data are not disclosed.
4. **No exact loss/checkpoint/feature spec:** “pooled output” is insufficient for PE in particular, whose title/reference emphasizes that its best embeddings are not necessarily at the final output.
5. **Training-set average called unseen:** GenImage averages include SD1.4 and SD1.5.
6. **DDA fairness asymmetry:** DDA uses official weights/training data, while all other methods are retrained on SD1.4. This is understandable but does not isolate architecture.
7. **No AUC or calibration:** all main results are accuracy averaged over real/fake. Dataset-specific threshold shifts can make a detector look worse or better independent of separability.
8. **No repeated runs/error bars:** a two-epoch linear probe is cheap enough that multiple seeds should have been reported.
9. **In-the-wild source shortcuts:** social/community datasets can have platform compression, resolution, captions/watermarks baked into pixels, or unmatched real/fake topics. Strong VLMs may exploit those signals.
10. **AIGI-Now `sem` protocol underspecified:** without exact degradations, it is impossible to know whether the result reflects semantic anomaly, remaining low-level cues, or dataset source.
11. **Mechanism probes are indirect:** Common Crawl counts, text retrieval, and web-versus-satellite results converge on a plausible story but do not isolate synthetic exposure.
12. **LoRA and backbone-swap claims are over-broad:** both experiments lack tuning details and matched optimization.
13. **Robustness remains incomplete:** recapture fake recall near 65–71% is not deployment-ready; challenge-severity JPEG/resize and four required corruptions are missing.
14. **Global task only:** the authors correctly limit the conclusion; VAE reconstruction and local edit remain unsolved. For this repository that is acceptable because those cases are explicitly out of scope.

## 2.10 Implementation cost

- The head itself is one linear layer. Training can be done by caching features and fitting the head in minutes to hours.
- Inference cost is exactly one foundation-backbone forward plus a tiny matrix multiply.
- This is the lowest engineering-risk baseline in the two papers.
- Cost is dominated by which checkpoint is chosen. PE ViT-L/14 at 336 px is much more practical than a 7B DINOv3 and is under 2B. A compliant smaller DINOv3 must be tested rather than assuming the 7B result transfers.
- Native resolutions differ, so comparisons also differ in pixels/FLOPs. The paper does not report latency, memory, parameter counts, or throughput.
- As a final submission, a frozen linear probe alone is unlikely to meet the challenge's originality rule. It should be a mandatory baseline and possibly a frozen branch inside a new system.

## 2.11 Architecture-taste lessons

- Start with the strongest **current**, eligible foundation checkpoint. Architecture names without release/data version are misleading.
- Freeze first. Add trainable capacity only after it beats the frozen probe on leave-generator-out and corruption stress tests.
- Preserve broad pretrained geometry during adaptation; add an explicit feature-drift penalty if using LoRA.
- A simple global branch fits pure-AIGC detection well. Do not spend hackathon capacity on local-edit machinery that the task excludes.
- Semantic robustness is useful, but it can hide source/style shortcuts. Always pair it with format/content controls.
- Treat web exposure as a feature and a contamination risk at the same time.

---

# 3. Synthesis for `techjam-aigc`

## 3.1 What the papers jointly imply

The papers are less contradictory than their slogans suggest. The strongest shared explanation is:

1. a modern frozen foundation representation provides most of the cross-generator reach;
2. narrow tuning can destroy it;
3. real-centric residual/density information may add value, especially when it is kept separate from the fake distribution; and
4. neither semantic foundation features nor a real-reference module automatically guarantees severe corruption robustness.

The main unresolved causal question is **backbone versus head**. MIRROR attributes large gains to reference comparison but does not show a same-DINOv3 linear control. *Simplicity Prevails* attributes gains to the backbone, but its strongest DINOv3 may be 7B and its specialized-head swaps are not well controlled. This is exactly the comparison our repository should run first.

## 3.2 Adopt / adapt / avoid

### Adopt now

1. **Mandatory frozen-probe baseline.** Screen eligible PE, MetaCLIP2/SigLIP2, and smaller DINOv3 checkpoints with frozen features and the same balanced, source-controlled training set. Record exact checkpoint, license, parameter count, resolution, feature layer, and preprocessing.
2. **Format alignment.** Re-encode both classes through the same randomized codec/quality path, normalize resolution policy, strip metadata, and audit source predictability.
3. **Per-generator and per-corruption reporting.** Report real accuracy, fake accuracy, AUC, AP, worst-generator score, and worst-corruption score rather than only one macro average.
4. **Preserve foundation geometry.** Default to frozen backbone. If adaptation is used, compare it with frozen and measure feature drift on held-out authentic and fake data.
5. **Patch/mid-layer evidence as an auxiliary signal.** Global features are strong, but patch dispersion and mid-layer residuals can expose evidence that pooled output loses.

### Adapt into an original contribution

Do **not** implement MIRROR's exact memory/top-k/perplexity recipe. A distinct, challenge-aligned direction is a **transformation-response and authentic-density head** on a frozen, eligible VFM:

- Extract selected early/mid/late token summaries from the clean image and a small deterministic set of degraded views, for example JPEG70, resize0.5-upsample, and mild blur.
- Learn an authentic feature distribution with a simple, auditable estimator such as source-balanced shrinkage covariance, low-rank PCA residual, or mixture prototypes. This is not MIRROR's learned sparse memory.
- Compute (a) authentic-density residual, (b) cross-view feature displacement, and (c) displacement consistency across layers.
- Train a small gated head with binary loss plus a **worst-view consistency/ranking loss** so the AIGC score stays stable across transformations.
- Keep the frozen global VFM score as one branch. Gate or calibrate auxiliary low-level evidence down when it is unstable under compression.

This shifts the original contribution from “reconstruct a real reference” to **modeling how evidence survives redistribution**, directly matching the hidden evaluation. It is also explainable: show which layers/views changed the decision and whether the model relied on unstable evidence.

A second viable adaptation is a **feature-preserving LoRA** experiment: if LoRA is needed, penalize cosine/CKA drift from frozen features on a broad authentic replay buffer and use leave-one-generator-out meta-validation. The novelty is controlled adaptation that preserves web-scale generalization, motivated by Table 10 rather than copying it.

### Avoid

1. Exact MIRROR reproduction, both because it is disallowed and because training details are missing.
2. A linear probe as the final claimed innovation. Use it as a strong baseline or ensemble branch.
3. Any >2B checkpoint, especially the paper's DINOv3 ViT-7B.
4. COCO-only “real manifold” claims. Use multiple licensed authentic sources and hold entire sources out.
5. A trainable CLS bypass around an advertised forensic module.
6. Headline robustness based only on JPEG90 or resize0.9.
7. Evaluation on fake-only benchmarks paired with unrelated COCO reals.
8. Threshold-only accuracy without AUC and a held-out calibration protocol.
9. Watermark, metadata, file-format, resolution, or dataset-source shortcuts.
10. Spending major effort on local edit/VAE-only detection unless organizers expand scope.

## 3.3 Concrete discriminating experiments

### Experiment A: Is the backbone already the detector?

A 2×N matrix:

- Backbones: DINOv2-L, one eligible smaller DINOv3, PE-L/14, MetaCLIP2 or SigLIP2.
- Heads: frozen linear, two-layer MLP, authentic-density residual head, and the proposed transformation-response head.

Use identical images, crop policy, class balance, optimizer budget, and threshold calibration. Evaluate SD1.4 in-domain, leave-one-generator-family-out, all unseen families, and in-the-wild sets. If the proposed head does not beat the same-backbone linear probe on the **worst unseen generator**, reject it.

### Experiment B: Separate backbone gain from reference/density gain

Factorial control:

- old versus modern backbone;
- direct global head versus density/residual auxiliary branch;
- frozen versus feature-preserving LoRA.

Add random-projection and randomly initialized prototype controls. Match head parameter counts. Do not allow raw CLS into “residual-only” variants. This directly resolves the major missing MIRROR ablation.

### Experiment C: Test the real-manifold false-positive risk

Train the authentic density estimator on:

1. COCO only;
2. a diverse licensed mix of camera photos;
3. camera photos plus screenshots/documents/artwork if those are plausible authentic test inputs.

Hold each authentic source out in turn. Report per-domain FPR at a single threshold and TPR at 1%/5% FPR. If COCO memory yields high false positives on a held-out authentic domain, do not use it as the decision anchor.

### Experiment D: Frozen versus LoRA under generator diversity

For each eligible backbone compare:

- frozen linear;
- LoRA r=4/r=8 trained on SD1.4 only;
- LoRA trained on multiple generator families;
- LoRA with authentic-replay feature-distillation to the frozen backbone.

Use equal optimization search and report train-source accuracy as well as OOD. This tests whether the second paper's LoRA collapse is intrinsic or a tuning artifact.

### Experiment E: Full challenge robustness grid

For every clean test image create deterministic single corruptions at the exact challenge levels:

- JPEG 90/70/50/30;
- blur 0.5/1/2;
- resize 0.5/0.25 and upscale;
- Gaussian noise 0.02/0.05/0.10;
- brightness/contrast/saturation ±20%;
- center crop retaining 80%.

Also add realistic two-stage chains, such as crop→resize→JPEG50 and color→noise→JPEG70. Report clean AUC, each-severity AUC, average corruption AUC, **worst-severity AUC**, B.Acc at one clean-validation threshold, score drift, and ECE. A head that improves clean average but harms worst severity should not ship.

### Experiment F: Does transformation response add information?

Compare:

- clean single-view VFM;
- test-time score average across views;
- feature-displacement statistics only;
- clean + displacement gated fusion;
- paired-consistency training.

Measure gain separately for old GAN, UNet diffusion, DiT, autoregressive/unified, and each authentic source. Run a label permutation/control where transformation parameters are class-balanced to ensure the model cannot infer class from the augmentation pipeline.

### Experiment G: Shortcut and contamination audit

- Decode and re-encode every image through the same RGB/JPEG pipeline.
- Match size/aspect/quality distributions per class.
- Strip EXIF and filenames before the model sees data.
- Train a classifier to predict dataset/source from pixels and from model features; high source accuracy flags shortcut risk.
- Perceptual-hash and embedding-deduplicate across train/test.
- Use source-matched real images and, where possible, prompt/content-matched real/fake pairs.
- Probe whether a text-aligned backbone's score correlates with watermarks, typography, illustration style, or platform logos.

### Experiment H: Calibration and operating point

Fit temperature or isotonic calibration on a source-held-out validation set, never on the demonstration-only WildFake split. Freeze one threshold. Report ROC-AUC, PR-AUC, B.Acc, ECE/Brier, TPR at fixed FPR, and class accuracy for every generator/corruption. This avoids the fixed-threshold confound visible in both papers.

### Experiment I: Compute/feasibility gate

For every promising model record:

- total and trainable parameter counts;
- peak VRAM at train/inference;
- images/s at batch 1 and a practical batch;
- preprocessing cost, including multi-view inference;
- serialized weight size.

Set a hard gate before large experiments. A smaller PE/DINOv3 plus three views may beat a huge single-view model in robustness per unit latency, but this must be measured.

## 3.4 Recommended near-term decision

1. Reproduce only the **frozen linear probe baseline** on two or three verified sub-2B modern backbones.
2. Run the exact corruption grid and source-held-out authentic tests before adding modules.
3. Build one small original transformation-response/density head on the best frozen backbone.
4. Use the backbone×head factorial and no-CLS-bypass ablations to establish that the contribution is real.
5. Keep MIRROR as conceptual prior art in the write-up and clearly explain how the final method differs.

The strongest architectural story for the hackathon is not “we copied a real-manifold memory” or “we fine-tuned a foundation model.” It is: **we preserve a modern foundation model's broad generator knowledge, then explicitly learn which evidence remains stable through the transformations used in real redistribution, under strict shortcut and authentic-domain controls.**
