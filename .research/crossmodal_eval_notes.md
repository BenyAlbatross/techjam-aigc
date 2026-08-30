# Evaluation notes: *Cross-modal Representation Learning for Diffusion-generated Image Detection*

## Identity and sources

- **Paper:** Tao Gong, Dayong Wang, Qi Chu, Bin Liu, and Nenghai Yu, “Cross-modal Representation Learning for Diffusion-generated Image Detection,” *CVPR 2026*, pp. 36092–36102.
- **Primary source:** [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.html); [accepted-paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.pdf). Page references below are the printed CVPR pages.
- **Benchmark context sources:** [GenImage official repository](https://github.com/GenImage-Dataset/GenImage), [DRCT paper/repository](https://github.com/beibuwandeluori/DRCT), and [Co-Spy-Bench dataset card](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench).
- The official CVF landing page has a PDF but no code or supplement link. I found no author repository in a targeted GitHub search as of this note. Thus, claims below distinguish what the paper states from estimates or omissions.

## One-paragraph take

The proposed **Strong Diffusion-generated Image Detector (SDID)** is not a CLIP/prompt method. Its main model combines a LoRA-tuned pretrained **DINOv2 ViT-L/14** RGB encoder with an ImageNet-pretrained **ResNet-101** encoder operating on a deterministic Neighborhood Pixel Relationship (**NPR**) transform. During supervised training, Cross-Modal Contrastive Learning (**CMCL**) aligns the RGB and NPR views of the same labeled image and contrasts them against the opposite class; Cross-Modal Mutual Distillation (**CMMD**) aligns within-class neighborhood distributions between the two modalities. The main detector concatenates both features at inference. It reports 98.4% macro-average accuracy on GenImage, 92.4% on DRCT-2M, and 96.1% on Co-Spy-Bench. These are strong cross-generator results, especially Co-Spy-Bench, but the paper omits much of the training recipe, reports only fixed-threshold accuracy, provides no uncertainty or robustness-to-transformation study, and does not report parameters, FLOPs, latency, training time, or hardware. [P, pp. 36092–36099, Tables 1–7]

## What the method actually learns

### NPR input

- NPR is borrowed from prior work, not claimed as a contribution. Split RGB into non-overlapping `2 x 2` patches. For each channel, subtract the patch’s top-left pixel from every pixel in that patch. The resulting image-like tensor is fed to the NPR encoder. The authors motivate it as a local trace of generator upsampling rather than high-level content. [P, pp. 36092–36095, §3.2]
- NPR is therefore a **deterministic derived view of RGB**, not an independent sensor modality. Calling this “cross-modal” is reasonable in a representation-learning sense, but it does not add external information.

### CMCL: inter-class separation

- For a real image, its RGB feature and its own NPR feature are a positive pair. Its RGB feature and queued NPR features from fake images are negatives. A symmetric term reverses RGB/NPR, and the same two directions are used with a fake query and real-class negatives. This gives four InfoNCE terms. Temperature is `tau = 0.07`. [P, p. 36095, Eqs. 1–2]
- This is **label-aware supervised contrastive learning**, despite the paper’s discussion of conventional self-supervised contrastive learning. Labels decide which class-specific queue supplies negatives. Same-class samples are not used as negatives.
- There are four moving queues: RGB-real, NPR-real, RGB-fake, NPR-fake. Each has `N_Q = 2048`; the current batch is enqueued and the oldest batch dequeued. [P, pp. 36094–36096]

### CMMD: intra-class compactness / cross-view knowledge transfer

- For an embedding and same-modality queue, cosine similarity to the top-`K` nearest neighbors is softmaxed with temperature `tau` into a neighborhood probability distribution. `K = 128`. [P, pp. 36095–36096, Eq. 3]
- A KL loss matches the RGB and NPR neighborhood distributions within the **same class**, bidirectionally. Four terms cover RGB→NPR and NPR→RGB for real and fake. There is explicitly no real↔fake distillation. Unlike fixed teacher/student distillation, each modality is both teacher and student while it changes during training. [P, p. 36096, Eqs. 4–5]
- The paper/diagram implies corresponding queue samples or selected neighbor indices are needed to make cross-modal probability entries comparable, but the precise indexing procedure is not fully specified in prose.

### Final objective and inference

- `L = L_cls + 0.1 L_CMCL + 0.1 L_CMMD`, with cross-entropy classification on concatenated RGB and NPR features. Main-model inference recomputes NPR, runs both encoders, concatenates their features, and classifies. Queues and representation losses are training-only. [P, p. 36096, Eq. 6 and §4.1]
- A useful alternate ablation trains both views but predicts only from RGB. That version reaches 96.4% on GenImage and can discard the NPR branch at inference, which the authors correctly describe as no added **test-time** model/time cost relative to its vanilla RGB encoder. [P, pp. 36098–36099, Table 6]

## Datasets and exact protocols

The paper evaluates **binary real versus generated classification**. Every reported value is accuracy at a fixed decision threshold of `0.5`; there is no AUROC, AP, F1, calibration result, or confidence interval. “Avg.” is the arithmetic macro-average over generator/test subsets (this can be reproduced from the displayed per-subset values). [P, p. 36096, §4.1; pp. 36097–36098, Tables 1–3]

### GenImage

- Officially, GenImage has over one million paired generated/real images, covers ImageNet’s 1,000 semantic classes, and has eight generator subsets: Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM, and BigGAN. Its benchmark defines cross-generator and degraded-image tasks. [GenImage official repository](https://github.com/GenImage-Dataset/GenImage)
- **SDID protocol:** train only on `GenImage/SDv1.4`, following the GenImage cross-generator protocol; evaluate separately on all eight subsets. The paper does **not** evaluate GenImage’s low-resolution/JPEG/blur degraded-image protocol. [P, p. 36097, Table 1]

### DRCT-2M

- The source DRCT paper describes two million generated images from 16 Stable-Diffusion-family conditions (120k/type): 10 text-to-image types, three ControlNet types, and three diffusion-reconstruction types. Its real/reference content comes from MSCOCO; image sizes span 256–1024. It also describes a separate 136k “Wild” collection from eight platforms. [DRCT paper/repository](https://github.com/beibuwandeluori/DRCT)
- **SDID protocol:** train on `DRCT-2M/SDv1.4` “following DRCT,” then test on the 16 displayed subsets: LDM; SD v1.4/v1.5/v2; SDXL and Refiner; SD-Turbo and SDXL-Turbo; LCM-SDv1.5 and LCM-SDXL; SDv1/2/XL-ControlNet; and SDv1/2/XL diffusion reconstruction (`-DR`). [P, p. 36097, Table 2]
- The wording indicates the standard detector comparison protocol—known fake training images from SD v1.4, rather than training on all 16 generators. However, SDID does not spell out exact image counts, split files, augmentations, or whether any DRCT-specific reconstructed training samples are included. This is a material reproducibility ambiguity.

### Co-Spy-Bench

- The benchmark’s dataset card says captions come from MSCOCO2017, CC3M, Flickr, TextCaps, and SBU; it contains 22 generation models and varies diffusion steps/guidance scales. A typical model/source archive contains 5,000 synthetic images. [Co-Spy-Bench dataset card](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench)
- **SDID protocol:** still train on `DRCT-2M/SDv1.4`, following CO-SPY, and evaluate on Co-Spy-Bench’s 22 generator subsets. Thus Table 3 is a cross-dataset and cross-generator evaluation, not Co-Spy training. [P, p. 36098, Table 3]

## Main numerical results

### Table 1 — GenImage accuracy (%)

| Method | Midjourney | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SDID | **96.9** | **99.9** | **99.9** | **95.9** | **98.7** | 99.1 | **98.0** | **98.8** | **98.4** |

- Prior average leaders are CoD 96.2 and DLFE 96.0, so SDID gains **+2.2 points** over the best displayed average. CoDE is 93.5, DRCT 89.4, UnivFD 79.4. [P, p. 36097, Table 1]
- SDID is not best on every column: several methods tie 99.9 on known SD subsets; DLFE ties it at 95.9 ADM; and DIRE/LaRE reach 99.9 Wukong versus SDID’s 99.1. The defensible SOTA claim is the displayed **average**, not universal per-generator dominance.

### Table 2 — DRCT-2M accuracy (%)

| Test subset | SDID |
|---|---:|
| LDM | 97.6 |
| SDv1.4 / SDv1.5 / SDv2 | 96.9 / 96.6 / 97.2 |
| SDXL / SDXL-Refiner | 93.7 / 94.4 |
| SD-Turbo / SDXL-Turbo | 93.2 / 96.8 |
| LCM-SDv1.5 / LCM-SDXL | 96.3 / 91.3 |
| SDv1-Ctrl / SDv2-Ctrl / SDXL-Ctrl | 97.0 / 95.7 / 96.6 |
| SDv1-DR / SDv2-DR / SDXL-DR | 86.7 / 75.9 / 71.9 |
| **Macro average** | **92.4** |

- Average comparators: DLFE 90.9, DRCT 90.5, UnivFD 83.5, CNNSpot 81.1; hence **+1.5 points** over the best displayed average. [P, p. 36097, Table 2]
- Important nuance: SDID is weaker than several baselines on some ordinary subsets. For example, DLFE is 97.1 on SDXL vs SDID 93.7. It is also weaker than DRCT on SDv1-DR (86.7 vs 94.1). Its gain is strongest on the harder unseen reconstruction variants SDv2-DR (**75.9 vs 69.6 best prior, +6.3**) and SDXL-DR (**71.9 vs 57.4, +14.5**), although absolute accuracy there remains the paper’s lowest.

### Table 3 — Co-Spy-Bench accuracy (%)

- SDID’s 22 per-generator scores, in table order, are: `97.5, 97.6, 97.6, 96.2, 97.5, 97.5, 97.5, 97.1, 95.0, 96.1, 96.3, 95.5, 94.8, 96.4, 96.0, 90.5, 96.4, 97.3, 97.4, 97.5, 93.9, 93.6`.
- These correspond to LDM, SD-v1.4, SD-v1.5, SSD-1B, tiny-sd, SegMoE-SD, small-sd, SD-2-1, SD-3-medium, SDXL-turbo, SD-2, SDXL, PG-v2.5-1024, PG-v2-1024, PG-v2-512, PG-v2-256, PAXL-2-1024, PAXL-2-512, LCM-sdxl, LCM-sdv1-5, FLUX.1-schnell, and FLUX.1-dev. [P, p. 36098, Table 3]
- Macro average is **96.1**, versus CO-SPY 87.1, DRCT 76.0, UnivFD 76.5, and NPR 65.7: exactly **+9.0 points** over the best displayed baseline. The weakest SDID cases are PG-v2-256 (90.5), FLUX.1-dev (93.6), and FLUX.1-schnell (93.9). This is the paper’s most compelling generalization result.

## Ablations (all GenImage macro accuracy, trained on SD v1.4)

### Table 4 — branch and loss contribution

| RGB | NPR | CMCL | CMMD | Acc. |
|---:|---:|---:|---:|---:|
| yes | no | no | no | 86.7 |
| no | yes | no | no | 88.5 |
| yes | yes | no | no | 90.1 |
| yes | yes | yes | no | 95.4 |
| yes | yes | yes | yes | **98.4** |

Interpretation: simple fusion adds +3.4 over RGB; CMCL adds +5.3 over plain fusion; CMMD adds another +3.0. NPR alone is +1.8 over RGB, supporting the artifact-view premise. [P, p. 36098, Table 4]

### Table 5 — is NPR special?

| Pair of views | no representation loss | +CMCL | +CMCL+CMMD |
|---|---:|---:|---:|
| RGB + augmented RGB | 86.9 | 93.0 | 93.2 |
| RGB + high-frequency image | 88.3 | 93.8 | 95.5 |
| RGB + NPR | 90.1 | 95.4 | **98.4** |

- CMCL helps all view choices. CMMD barely helps two RGB views (+0.2), more with a high-frequency view (+1.7), and most with NPR (+3.0). This supports both “distinct complementary view” and NPR-specific claims. [P, p. 36098, Table 5]
- The augmentation distribution for the second RGB view and the exact high-frequency transform are not reported, limiting replication and the fairness of this ablation.

### Table 6 — NPR only as training-time auxiliary

| RGB prediction | CMCL | CMMD | Acc. |
|---|---:|---:|---:|
| yes | no | no | 86.7 |
| yes | yes | no | 93.6 |
| yes | yes | yes | **96.4** |

NPR participates in CMCL/CMMD training but is omitted from classification/inference. The total gain is +9.7 over vanilla RGB and only 2.0 below full dual-branch SDID. This is likely the best cost/accuracy deployment point in the paper. [P, pp. 36098–36099, Table 6]

### Figure 4 — embedding visualization

The t-SNE plots qualitatively show CMCL separating real/fake clusters and CMMD making real embeddings more compact. They are illustrative only: no cluster metric, seed, perplexity, or quantitative representation measure is provided. [P, p. 36099, Fig. 4]

### Table 7 — changing DINOv2 to a CLIP image encoder

| RGB | NPR | CMCL | CMMD | Acc. |
|---:|---:|---:|---:|---:|
| yes | no | no | no | 81.3 |
| no | yes | no | no | 88.5 |
| yes | yes | no | no | 89.3 |
| yes | yes | yes | no | 93.2 |
| yes | yes | yes | yes | **95.5** |

CMCL adds +3.9 to plain CLIP+NPR fusion and CMMD adds +2.3. This shows the losses transfer to a second RGB backbone, but full CLIP-based SDID remains 2.9 points below DINOv2-based SDID (95.5 vs 98.4). [P, p. 36099, Table 7]

## CLIP and prompt-learning relation

- **Main SDID does not use CLIP.** It uses DINOv2 ViT-L/14 for RGB and ResNet-101 for NPR. [P, p. 36096, §4.1]
- CLIP appears in the motivation as an example of a pretrained visual encoder optimized for high-level semantics rather than forensic traces, and in related work (UnivFD, FatFormer, De-fake, CLIPMoLE, Co-Spy). [P, pp. 36092–36094]
- The only direct CLIP experiment is Table 7’s unnamed **CLIP image encoder** replacement. The paper gives no CLIP architecture (ViT-B/L, patch size), pretraining source, frozen/fine-tuned policy, or LoRA details for it.
- **No CLIP text encoder, text-image similarity, class-name template, learned text prompt, prompt ensemble, or prompt tuning is used.** Searching the paper body finds no method use of “prompt”; the sole occurrence is in a cited paper title. Thus, this work should not be described as prompt learning or multimodal image-text learning.
- Its “cross-modal” relation is **RGB ↔ NPR**, not image ↔ language. A prompt-based detector could be complementary, but that combination is not tested.

## Implementation and deployment cost

### What is specified

- RGB backbone: pretrained DINOv2 ViT-L/14, tuned with LoRA.
- NPR backbone: ImageNet-pretrained ResNet-101.
- NPR patch size 2; queue length 2048; `K=128`; `tau=0.07`; `lambda_1=lambda_2=0.1`.
- Main inference runs two encoders; the RGB-only Table 6 variant runs one encoder. [P, pp. 36095–36099]

### Approximate scale

- Meta’s [DINOv2 model table](https://github.com/facebookresearch/dinov2) lists ViT-L/14 at about **300M parameters**. TorchVision lists [ResNet-101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html) at **44,549,160 parameters**. Therefore, the two main pretrained backbones alone are roughly **345M parameters**, before projections/classifier/LoRA parameters. This is an estimate because SDID reports no model-size accounting.
- Main inference requires one large ViT-L forward plus one ResNet-101 forward and the cheap NPR transform. This is materially more costly than a single CLIP/DINO detector. The paper reports no latency or FLOPs.
- If queues store a common 1024-D FP32 projection, four `2048 x 1024` queues consume about **32 MiB**. Raw DINO-1024 and ResNet-2048 queues would be about **48 MiB**. This is only an order-of-magnitude estimate; the paper omits the projection dimension and dtype. Queue compute and memory are training-only; encoder activations will dominate training memory.
- The Table 6 deployment variant retains the roughly 300M-parameter DINO backbone but drops ResNet/NPR inference while preserving 96.4% GenImage average. It still needs the NPR branch and queues during training.

### Critical missing implementation details

The paper does not report: image resolution/cropping; normalization; data augmentation; optimizer; learning rate/schedule; weight decay; epochs/steps; batch size; class sampling; LoRA rank/target layers/dropout; whether ResNet-101 is frozen or fully tuned; classifier/projection-head architecture; embedding dimension; CLIP variant/tuning; seed/repetitions; GPU type/count; mixed precision; training time; throughput; latency; FLOPs; checkpoint size; or code. [P, §4.1 contains only the short backbone/hyperparameter paragraph]

A particularly important architectural omission is the common space used by CMCL: standard DINOv2 ViT-L/14 outputs 1024-D features while standard ResNet-101 outputs 2048-D pooled features, so the cross-modal dot product in Eq. 1 requires a projection or modified head. No such projection or its dimension is described. Also, “only the encoders are reserved” at inference omits the logically necessary binary classification head. These gaps prevent faithful reimplementation from the paper alone.

## Limitations and validity cautions

The paper has **no explicit limitations section**. The following limits follow directly from its design/results/reporting:

1. **Only fixed-threshold accuracy.** A threshold of 0.5 tests both discrimination and calibration, yet no calibration procedure is described. No AUROC/AP and no class-prior analysis are given.
2. **No statistical evidence.** Every result is a single number. There are no seeds, standard deviations, confidence intervals, or significance tests. Improvements such as +1.5 on DRCT-2M cannot be checked for run variance.
3. **No real-world transformation robustness.** There are no JPEG, resizing, blur, screenshot, crop, or social-platform tests, even though GenImage defines a degraded-image task. NPR is a local pixel-difference representation, making this omission especially important.
4. **Narrow target.** Tests cover fully generated images. There is no partially AI-edited/composited detection, localization, attribution, or open-set rejection.
5. **Artifact assumption.** NPR is motivated by common upsampling traces. A generator or post-process that removes/changes those traces could weaken it. The low DRCT reconstruction accuracies (71.9 SDXL-DR, 75.9 SDv2-DR) show residual hard cases.
6. **Heavy main inference.** Roughly 345M backbone parameters and two image encoder passes are feasible under a 2B cap but costly for a hackathon or high-throughput service. No measured cost is provided.
7. **Incomplete reproducibility.** The omitted recipe and unexplained cross-modal projection are substantial. No code is linked on the official page.
8. **Benchmark comparison ambiguity.** The paper says methods follow standard training protocols but does not say which baseline numbers were rerun versus copied, nor document equal augmentations/checkpoints. Backbones and pretraining differ substantially, so the tables compare full systems, not isolated representation-learning algorithms.
9. **Ablation breadth is limited.** There is no sweep for queue length, top-K, temperatures, loss weights, LoRA rank, NPR patch size, projection size, or backbone capacity. There is also no parameter-matched dual-RGB control.
10. **Semantic/content leakage is not analyzed.** GenImage real images and generated images may differ in source/compression pipelines. The use of paired semantic content helps, but no source-balanced or content-controlled diagnostic is reported by SDID.

## Practical relevance to this repository/challenge

- The core idea is compatible with the challenge’s originality requirement: an ordinary public pretrained vision backbone is trained with an original forensic cross-view objective rather than importing a pretrained AIGC detector.
- The full estimated model is comfortably below 2B parameters, but main SDID is not cheap. The Table 6 auxiliary-NPR training scheme is the more attractive prototype: DINO-only inference at 96.4 GenImage vs dual-branch 98.4.
- For a robust detector, reproduce the **training objective**, but add the missing essentials: explicit shared projection heads; smaller public backbone(s); transform-heavy evaluation/training; AUROC and calibration; multiple seeds; held-out generator families; and measured parameters/FLOPs/latency.
- Do not frame SDID as CLIP prompt learning. If CLIP is selected, use its image encoder as a replaceable backbone and independently decide whether a text/prompt branch adds value.
