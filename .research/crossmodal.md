# Cross-modal Representation Learning for Diffusion-generated Image Detection

## Identity, full text, and title ambiguity

**Resolved paper.** Tao Gong, Dayong Wang, Qi Chu (corresponding author), Bin Liu, and Nenghai Yu, **“Cross-modal Representation Learning for Diffusion-generated Image Detection,”** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2026, pp. 36092–36102. The authors are affiliated with the University of Science and Technology of China and the Anhui Province Key Laboratory of Digital Security.

Reliable primary sources:

- [Official CVF Open Access record](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.html)
- [Official full-text PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.pdf)
- [Official CVPR virtual poster record](https://cvpr.thecvf.com/virtual/2026/poster/37096) (the conference site currently shows a June 7, 2026 poster slot)
- Local research copy: `.research/Gong_Crossmodal_CVPR2026.pdf`

The CVF record says its PDF is identical to the accepted version except for the CVF watermark. It supplies the canonical BibTeX above but does not display a DOI. The official page links only the paper PDF and BibTeX; it does not link code or supplementary material.

**Why the title can look ambiguous.** The prompt's title is not merely approximate: it exactly matches the official CVPR title, including the singular “Image Detection” and the hyphenated “Diffusion-generated.” It is easy to misidentify because it is a very new CVPR 2026 record and is not necessarily present in older bibliographic indexes or arXiv. A title search can instead surface similarly named but different works, notably *On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection* (NeurIPS 2024). That is a **video** paper and is not the requested work. “Cross-modal” here also does **not** mean image–text learning: the two modalities are RGB pixels and a deterministic local pixel-difference representation called NPR.

### Citation convention

The PDF has 11 pages. Below, “p. 4 / 36095” means PDF page 4 and printed proceedings page 36095. Tables and figures refer to the paper's numbering.

## Core argument

Most generated-image detectors feed RGB into an encoder originally optimized for semantic recognition, such as ResNet, CLIP, or DINOv2. The authors argue that such a feature space is not inherently organized around generator artifacts. Their response is not a new visual backbone. It is a representation-learning recipe that combines:

1. **RGB**, which contains semantic and ordinary visual evidence; and
2. **Neighboring Pixel Relationships (NPR)**, a handcrafted local residual view intended to expose source-invariant upsampling traces.

They train the two views jointly with:

- **Cross-Modal Contrastive Learning (CMCL):** enlarge the real/fake inter-class gap in both modalities.
- **Cross-Modal Mutual Distillation (CMMD):** make the within-class neighborhood geometry of the two modalities agree, which is intended to tighten each real or fake cluster.

A normal cross-entropy head on the concatenated features supplies the binary decision. The claim is that CMCL gives inter-class separability while CMMD gives intra-class compactness, and the two cooperate to produce a “forgery-aware” embedding rather than a generic semantic one (pp. 1–2 / 36092–36093; Figs. 1–2).

## Exact architecture and data flow

### Inputs and modalities

For an RGB image `I_RGB`, NPR is computed with no learned transform:

1. Divide each channel into non-overlapping `2 x 2` patches.
2. In each patch and channel, subtract the patch's top-left pixel value from all four pixels.
3. Feed the resulting three-channel difference image `I_NPR` to a second encoder.

Thus every patch contains a zero at its top-left position and three local differences per channel. The authors inherit this representation from Tan et al., *Rethinking the Up-sampling Operations...*; they explicitly do not claim NPR as their contribution (p. 4 / 36095, §3.2).

### Encoders and classifier

The main SDID model uses:

- **RGB encoder `E_RGB`:** pretrained **DINOv2 ViT-L/14**, fine-tuned with **LoRA**.
- **NPR encoder `E_NPR`:** **ResNet-101**, pretrained on ImageNet.
- **Decision head:** concatenate `F_RGB` and `F_NPR`, then predict real versus fake with cross-entropy.

This is specified on p. 5 / 36096, §§3.5 and 4.1. The overview is Fig. 3 on p. 4 / 36095. DINOv2, rather than CLIP, is the primary semantic encoder. CLIP's image encoder appears only in a backbone-transfer ablation (Table 7).

### Queue state

Training maintains **four class-and-modality queues**:

- `Q_real_RGB`, `Q_fake_RGB`
- `Q_real_NPR`, `Q_fake_NPR`

Each stores 2,048 features. Current features are enqueued and the oldest are dequeued. There is no described momentum/key encoder. The queues are training state, not model inputs at inference (pp. 3–5 / 36094–36096).

### CMCL: supervised, cross-modal inter-class contrast

For a real image, its RGB and NPR embeddings from the **same image** are a positive pair. Its RGB feature is contrasted against NPR features of fake images in `Q_fake_NPR`; symmetrically, its NPR feature is contrasted against fake RGB queue entries. The analogous two directions are applied to fake anchors against real queues. The four InfoNCE terms are therefore:

1. real RGB → matching real NPR, with fake NPR negatives;
2. real NPR → matching real RGB, with fake RGB negatives;
3. fake RGB → matching fake NPR, with real NPR negatives;
4. fake NPR → matching fake RGB, with real RGB negatives.

Equation 1 gives the per-direction InfoNCE loss; Equation 2 sums all four. The reported temperature is `tau = 0.07` (p. 4 / 36095). This is supervised class separation, not ordinary instance-only SimCLR: negatives are explicitly drawn from the opposite class.

### CMMD: bidirectional within-class relational distillation

CMCL does not use the relationships among different examples of the same class. CMMD addresses that gap:

1. For an embedding `z`, compute cosine similarity to same-class anchors from its modality queue.
2. Select the top `K = 128` nearest neighbors out of the 2,048 entries.
3. Softmax the `K` similarities into a local neighborhood probability distribution (Eq. 3).
4. Minimize KL divergence between the RGB and NPR distributions for corresponding queue entries (Eq. 4).
5. Do this in both modality directions and separately for real and fake, for four KL terms total (Eq. 5).

Figure 3 labels the use of the **same top-K indices** when comparing distributions across modalities. Each view continuously serves as teacher and student; there is no fixed teacher network (p. 5 / 36096, §3.4).

### Total objective

The total loss is (Eq. 6):

`L = L_cls + lambda_1 L_CMCL + lambda_2 L_CMMD`

with `lambda_1 = lambda_2 = 0.1`. `L_cls` is binary real/fake cross-entropy on the concatenated RGB and NPR features (p. 5 / 36096).

### Trainable parts: what is stated and what is missing

The paper clearly states LoRA fine-tuning for DINOv2-L/14 and ImageNet initialization for ResNet-101. It necessarily trains a decision head and some compatible cross-modal feature mapping. However, it does **not** say:

- whether all ResNet-101 layers are fine-tuned or any are frozen;
- LoRA rank, target modules, scaling, dropout, or whether non-LoRA DINO parameters are frozen;
- the projection/pooling heads or common embedding dimension;
- the classifier architecture;
- whether features are explicitly L2-normalized before CMCL;
- optimizer, learning rate, schedule, epochs, batch size, resolution, augmentation, or random seeds;
- whether the “target” side of each KL term is stop-gradient;
- how top-K correspondence is implemented when the two modality rankings differ.

The missing projection detail is material: a standard DINOv2-L feature is 1,024-dimensional while a standard ResNet-101 pooled feature is 2,048-dimensional, yet Eq. 1 takes their dot product. A shared-dimensional projection must exist unless one encoder is modified, but it is absent from the text and figures. CMMD has a similar ambiguity. Consequently, the paper gives the learning idea exactly enough to understand, but **not** enough to reproduce faithfully.

### Inference

For the main SDID model:

`image → RGB view + computed NPR view → DINOv2-L/14 + ResNet-101 → concatenate → classifier → real/fake score`

Queues, CMCL, and CMMD disappear after training; only the two encoders and classification path remain (pp. 3 and 5 / 36094, 36096). Table 6 also studies a valuable **training-only NPR** variant: use NPR for CMCL/CMMD during training, but classify with only the RGB encoder at test time. It reaches 96.4% GenImage average and adds no NPR inference cost relative to the vanilla RGB encoder (p. 8 / 36099).

## Conceptual intuition

The strongest intuition is geometric:

- RGB and NPR are two observations of the same underlying source event.
- A real RGB point should align with its own real NPR point and sit far from fake NPR points; the reverse holds for fake examples.
- Two views can still form diffuse, modality-specific real/fake clouds after contrastive training. Matching their **local same-class neighborhood distributions** transfers structure without demanding identical raw features.
- The classifier then sees two complementary features: one can preserve semantics and broad texture, while the other emphasizes local pixel relations associated with generation/resampling.

This is more thoughtful than naive late fusion. CMCL asks both encoders to agree on source class across views. CMMD asks them to agree on which same-class examples are locally similar. Figure 4's t-SNE plots qualitatively show progressively better class separation after CMCL and tighter real clusters after CMMD, although t-SNE is not quantitative evidence (p. 8 / 36099).

A useful alternative interpretation is multi-view regularization: NPR need not be intrinsically a separate “modality.” It is a fixed high-pass-like transform of RGB. The benefit may come from forcing the semantic representation to respect a handcrafted forensic inductive bias.

## Datasets, protocols, metrics, and main numbers

All reported metrics are **average accuracy at a fixed 0.5 threshold**. “Avg.” is the arithmetic macro-average across the displayed generator subsets. The paper reports no AUC, calibration, FPR, per-class recall, confidence intervals, or repeated-run variation (p. 5 / 36096, §4.1).

Useful primary benchmark context, beyond the short descriptions in SDID:

- The [official GenImage repository](https://github.com/GenImage-Dataset/GenImage) describes over one million real/generated images across ImageNet's 1,000 semantic classes and eight generators. It defines both cross-generator and degraded-image tasks. **SDID evaluates the cross-generator task but not GenImage's low-resolution/JPEG/blur degradation task.**
- The [official DRCT repository](https://github.com/beibuwandeluori/DRCT) describes DRCT-2M as two million images covering 16 Stable-Diffusion-family conditions, including text-to-image, ControlNet, and diffusion-reconstruction variants.
- The [Co-Spy-Bench dataset card](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench) documents 22 generation models and captions drawn from several public caption datasets. SDID's Table 3 trains on DRCT-2M/SD v1.4, not Co-Spy-Bench, so it is both cross-dataset and cross-generator evaluation.

### GenImage — Table 1, p. 6 / 36097

Protocol: train on **GenImage/Stable Diffusion v1.4**, then test across Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM, and BigGAN.

- **SDID average: 98.4%**
- Best listed competitor: CoD 96.2%; DLFE 96.0%; CoDE 93.5%
- Selected SDID results: Midjourney 96.9, ADM 95.9, GLIDE 98.7, BigGAN 98.8, SD v1.4/v1.5 both 99.9

This table also tests GAN-generated BigGAN despite the paper title saying diffusion-generated, so the empirical claim is broader than diffusion alone.

### DRCT-2M — Table 2, p. 6 / 36097

Protocol: train on **DRCT-2M/SD v1.4**, following DRCT, then test 16 diffusion/reconstruction subsets.

- **SDID average: 92.4%**
- DLFE: 90.9%; DRCT: 90.5%; next listed method UnivFD: 83.5%
- Hard diffusion-reconstruction subsets: SD v1-DR 86.7, SD v2-DR 75.9, SDXL-DR 71.9
- Modern/accelerated subsets: SDXL 93.7, SD-Turbo 93.2, SDXL-Turbo 96.8, LCM-SDXL 91.3

Nuance: SDID is not best on every subset. For example, DRCT scores 94.1 on SD v1-DR versus SDID's 86.7, and DLFE has a nearly flat 97.1 on several standard subsets. The claim is strongest on the mean and on SD v2/SDXL reconstruction variants.

### Co-Spy-Bench — Table 3, p. 7 / 36098

Protocol: train on **DRCT-2M/SD v1.4**, following Co-Spy, then test 22 generator subsets ranging from LDM and SD versions to SD3, PixArt variants, and FLUX.

- **SDID average: 96.1%**
- Co-Spy: 87.1%; UnivFD: 76.5%; DRCT: 76.0%
- SD3-medium 95.0; PixArt/PG-v2-256 90.5; FLUX.1-schnell 93.9; FLUX.1-dev 93.6
- Most other SDID subset results are 94.8–97.6

This is the most striking result: exactly +9.0 points over the strongest listed baseline average. It is also the result most in need of independent reproduction because implementation details and variance are absent.

## Ablations

### Component ablation — Table 4, p. 7 / 36098

GenImage average, training on SD v1.4:

| RGB | NPR | CMCL | CMMD | Accuracy |
|---:|---:|---:|---:|---:|
| yes | no | no | no | 86.7 |
| no | yes | no | no | 88.5 |
| yes | yes | no | no | 90.1 |
| yes | yes | yes | no | 95.4 |
| yes | yes | yes | yes | **98.4** |

Simple fusion gives +3.4 over RGB; CMCL then gives +5.3; CMMD gives another +3.0. This supports both the extra view and the objectives, but it does not isolate parameter count: the dual model is much larger than RGB-only.

### Alternate second views — Table 5, p. 7 / 36098

| Views | no CMCL/CMMD | +CMCL | +CMCL+CMMD |
|---|---:|---:|---:|
| RGB + augmented RGB | 86.9 | 93.0 | 93.2 |
| RGB + high-frequency image | 88.3 | 93.8 | 95.5 |
| RGB + NPR | 90.1 | 95.4 | **98.4** |

CMMD adds only 0.2 with two RGB views, 1.7 with high frequency, and 3.0 with NPR. This is evidence that genuinely different inductive biases matter more than merely duplicating the image encoder. It is not a proof that NPR is robust under redistribution.

### NPR as training-only auxiliary — Table 6, p. 8 / 36099

Only RGB features drive the test classifier:

- RGB baseline: 86.7
- + CMCL: 93.6
- + CMMD: **96.4**

This is a 9.7-point improvement without an added inference branch. It is arguably the most practical ablation for this repository.

### CLIP image encoder — Table 7, p. 8 / 36099

Replacing DINOv2 with the CLIP image encoder:

- CLIP RGB alone: 81.3
- NPR alone: 88.5
- fusion: 89.3
- + CMCL: 93.2
- + CMMD: **95.5**

The objectives transfer to CLIP, but DINOv2-L remains better in the reported setting (98.4 versus 95.5 final GenImage average). The paper does not identify the CLIP variant or its fine-tuning policy, so this is qualitative backbone-transfer evidence, not a controlled DINO-versus-CLIP benchmark.

### Missing ablations

There is no reported sweep for queue length, top-K, temperatures, loss weights, NPR patch size, LoRA configuration, embedding dimension, projection heads, encoder scale, training data size, augmentation, or corruption strength. There is no dual-branch comparison matched for parameters/compute, no multiple seeds, and no quantitative compactness/separation metric beyond accuracy and t-SNE.

## Relation to CLIP, prompts, and multimodal detection

- **Not image–text:** no text encoder, caption, class-name prompt, learned prompt, or language supervision is used.
- **Not CLIP-dependent:** main RGB backbone is DINOv2-L/14. CLIP vision is an ablation only.
- **CLIP-adjacent idea:** like CLIP, CMCL aligns paired views with contrastive learning. Unlike CLIP's web-scale image–text instance discrimination, SDID uses two image-derived views, binary labels, same-image positives, and opposite-class queue negatives.
- **Compared method families:** the related-work section contrasts UnivFD's CLIP features, De-Fake's CLIP-based multimodal fusion, FatFormer adapters, CLIPMoLE's LoRA mixture, AIDE's experts, Co-Spy's semantic plus VAE artifact features, and CoDE's RGB-only contrastive embedding (p. 3 / 36094).
- **Prompt learning:** prompt-based forgery methods are cited in related work, but SDID itself has no prompt component.

The paper's “cross-modal” label is defensible as multi-view learning but slightly grander than the mechanism: NPR is deterministically computed from RGB and contains no independent information source. It is closer to RGB-plus-forensic-residual fusion than conventional multimodal vision-language learning.

## Limitations, confounds, and robustness risks

### Stated versus unstated limitations

The paper has no dedicated limitations section. Its conclusion restates the method and SOTA claim (p. 8 / 36099). The following limitations follow from the presented evidence and should be treated as analysis, not author admissions.

1. **No real-world corruption study.** There are no JPEG, WebP, blur, resize, crop, noise, color jitter, screenshot, or transform-chain tests. NPR is based on exact 2×2 local relations, so resize, blur, denoising, sharpening, and re-encoding can erase native generator traces or create new resampling traces in authentic images. This is the largest mismatch with the TechJam target.
2. **Fixed-threshold accuracy only.** A 0.5 threshold says little about ranking quality or calibration after domain shift. The challenge needs a continuous `pred`; AUC, PR-AUC, Brier/ECE, FPR at useful TPRs, and corruption-specific threshold drift remain unknown.
3. **Benchmark provenance shortcuts.** Training follows benchmark-specific one-generator protocols. Without matched file formats, resizing, JPEG histories, and real/fake content sources, a local residual model can learn dataset pipelines rather than generator mechanisms. The paper does not document shortcut controls.
4. **Heavy main inference.** Two large encoders are required in the headline model. A standard DINOv2-L/14 is about 304M parameters and ResNet-101 about 45M, so the model is comfortably below 2B but roughly 350M parameters before heads. At 224×224 it is on the order of ~90 GFLOPs, depending on preprocessing. These are external architectural estimates; the paper reports no parameter, FLOP, latency, memory, or energy numbers.
5. **Incomplete reproducibility.** The absent projector, embedding size, optimizer, training schedule, LoRA details, preprocessing, and code link prevent a faithful build from the paper alone.
6. **Potential stale queues.** Features are FIFO queue entries from earlier model states. No momentum encoder or stale-feature analysis is described.
7. **CMMD mechanics are underspecified.** Stop-gradient and cross-modal top-K matching matter. If both distributions receive gradients, “teacher” and “student” can move together; if one is detached, direction and update ordering matter.
8. **No uncertainty.** Single-run point estimates and large gains are reported without standard deviations or statistical tests.
9. **Scope language drifts.** The introduction mentions image “creating and manipulating,” but experiments are image-level generated-versus-real benchmarks. Nothing demonstrates AI-edited or localized-composite detection. This is not fatal for TechJam, which also focuses on purely generated images, but claims should stay narrow.
10. **Possible content dependence.** RGB semantic features can help if generator and real classes have different subject distributions. No content-balanced or cross-real-dataset diagnostic is shown.
11. **NPR attribution is not causation.** The strong NPR result is consistent with upsampling fingerprints, but also with codec, resizing, color pipeline, or dataset acquisition differences. Cross-generator accuracy alone does not separate these explanations.

## Implementation cost and architecture taste

### Cost

- **Parameter rule:** safe under the repository's `<2B` cap, at roughly 350M backbone parameters plus small heads.
- **Training:** expensive for a hackathon. Each example runs a ViT-L and a ResNet-101; four 2,048-entry queues and top-128 searches add memory traffic. The queues themselves are modest, but activations for two encoders and large-scale GenImage/DRCT training dominate.
- **Inference:** the full model doubles preprocessing/data flow and adds the ResNet branch. The Table 6 RGB-only inference variant avoids this cost and is much more attractive.
- **Engineering:** NPR is simple and deterministic. CMCL is straightforward. CMMD needs careful aligned queues, masked class sampling, top-K search, KL detachment policy, and distributed-training synchronization.
- **Reproduction risk:** high because critical head and training specifications are absent.

### Architecture taste

**Good taste:** The paper puts the novelty in the embedding objective, not another huge backbone. It uses a cheap forensic transform to regularize a semantic encoder and tests the idea with alternate inputs and CLIP. The training-only auxiliary result is especially elegant.

**Questionable taste:** The headline model uses two oversized encoders when the core idea does not require them. “Cross-modal” overstates a deterministic RGB transform. The handcrafted 2×2 signal may be exactly the sort of brittle low-level shortcut that disappears after redistribution. The paper optimizes benchmark averages without the corruption evaluation needed to justify operational robustness.

## Repo-specific recommendation

### Adopt

1. **Adopt the principle, not the published detector:** jointly shape a semantic representation with an explicit low-level forensic view.
2. **Adopt training-time auxiliary NPR/high-pass supervision:** reproduce the Table 6 pattern with a smaller public backbone, then discard the auxiliary branch at inference if it preserves gains.
3. **Adopt class-aware cross-view contrast:** same-image RGB/residual positives and opposite-class negatives are a clear fit for binary pure-AIGC versus authentic detection.
4. **Adopt strict held-out-generator evaluation:** the paper's train-one-generator/test-many protocol matches the hidden-generator concern.

### Adapt

1. Replace DINOv2-L + ResNet-101 with a hackathon-scale pair, for example DINOv2-S/B or CLIP ViT-B plus a small CNN residual branch. Report parameters, FLOPs, latency, and peak memory.
2. Make the original contribution **transformation-aware**: apply label-symmetric corruption chains, align clean and transformed embeddings, and gate/down-weight NPR when degradation destroys local evidence.
3. Use an explicit, documented projection head and normalized embedding. Define stop-gradient, queue synchronization, top-K correspondence, and all LoRA/trainability choices.
4. Consider a robust residual family rather than NPR alone: 2×2 NPR, high-pass/DCT bands, and re-encoding residuals. Randomize which view is used so the semantic encoder cannot overfit one artifact.
5. Optimize and report ranking and calibration, not only thresholded accuracy. Preserve a continuous AIGC confidence for the required JSON interface.
6. Treat reconstructed-from-real diffusion outputs carefully in documentation. They are fully regenerated outputs in these benchmarks, not partial AI edits.

### Avoid

1. Do **not** copy SDID end to end. Challenge rules disallow merely replicating an existing detector, and the missing details make exact replication unsafe anyway.
2. Do not rely on NPR as the sole detector or claim it is source-invariant without corruption evidence.
3. Do not use unmatched transform/file pipelines by class. A perfect shortcut detector will fail hidden evaluation.
4. Do not present “multimodal” as image–text or claim prompt learning; this method uses no language.
5. Do not select a fixed 0.5 threshold on one clean benchmark and assume its score stays calibrated under JPEG/blur/resize.

## Concrete experiment plan

1. **Controlled core ablation:** semantic RGB baseline → add small NPR branch with late fusion → add CMCL → add CMMD. Match the paper's Tables 4–6 and include a parameter-matched RGB-only control.
2. **Training-only branch test:** compare full dual inference against auxiliary-NPR training with RGB-only inference. This is the priority feasibility experiment.
3. **Held-out generator matrix:** train on one or a subset of public generators; reserve entire generator families, especially DiT/FLUX-like families, for evaluation. Keep the organizer's WildFake COCO/DALL-E Advanced split demonstration-only.
4. **Required corruption grid:** clean plus JPEG 90/70/50/30; blur 0.5/1/2; resize 0.5×/0.25× then upscale; noise 0.02/0.05/0.10; ±20% color jitter; and 80% center crop. Add WebP and short transform chains as secondary tests.
5. **Metrics:** ROC-AUC, PR-AUC, balanced accuracy, per-class recall, FPR at chosen TPR, ECE/Brier, clean-to-corrupt drop, and worst-transform/worst-generator performance. Bootstrap confidence intervals over images.
6. **Shortcut audit:** force the same decoder, resize, JPEG, dimensions, and augmentation distribution for both classes; test cross-real-dataset splits; check content-category balance and nearest duplicates.
7. **Signal survival:** measure how NPR feature norms and separability change under each transform. Compare 2×2 NPR with high-frequency, DCT, and learned residual views.
8. **Objective sweeps:** queue lengths `{256, 1024, 2048}`, top-K `{16, 64, 128}`, loss weights, with/without CMMD stop-gradient, and within-batch versus queued negatives. Repeat key runs with at least three seeds.
9. **Deployment benchmark:** images/sec, p50/p95 latency, peak RAM/VRAM, model bytes, and accuracy for small/base/large semantic encoders and with/without the NPR inference branch.
10. **Original extension:** add clean↔transformed consistency plus degradation-conditioned modality gating. Test whether it improves the worst-case corruption metric without erasing clean cross-generator gains. This directly turns the paper's useful representation-learning idea into a contribution aligned with the challenge.

## Bottom line

This is a real, exact-title CVPR 2026 paper, not the similarly named diffusion-video work. Its central idea is strong: use RGB and NPR as complementary views, make them class-discriminative with CMCL, and align their within-class neighborhood geometry with CMMD. Its reported cross-generator averages are excellent—98.4 GenImage, 92.4 DRCT-2M, and 96.1 Co-Spy-Bench—but the work does not test the transformations that define this repository's problem, and it omits critical reproduction details. The best repo fit is a **smaller, transformation-aware, training-only forensic auxiliary branch**, not a direct two-large-encoder SDID clone.
