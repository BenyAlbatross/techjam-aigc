# Identity notes: *Cross-modal Representation Learning for Diffusion-generated Image Detection*

Research checked: **2026-08-30**.

## Bottom line

This is a real **CVPR 2026** paper, not an arXiv title and not the similarly named NeurIPS video-forgery paper. The authoritative freely accessible full text is the Computer Vision Foundation (CVF) Open Access PDF. The method itself is named **SDID** (*Strong Diffusion-generated Image Detector*). It trains two visual branches on (1) RGB and (2) Neighboring Pixel Relationships (NPR), using cross-modal contrastive learning (**CMCL**) plus bidirectional cross-modal mutual distillation (**CMMD**).

- Official landing page: <https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.html>
- Official full-text PDF: <https://openaccess.thecvf.com/content/CVPR2026/papers/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026_paper.pdf>
- Official CVPR program/poster page: <https://cvpr.thecvf.com/virtual/2026/poster/37096>
- Local PDF: `.research/Gong_Cross-modal_Representation_Learning_for_Diffusion-generated_Image_Detection_CVPR_2026.pdf`
- Local extracted text: `.research/Gong_Crossmodal_CVPR2026.txt`
- PDF integrity: 970,396 bytes; SHA-256 `25520fd8d413f2ffa7affa9ff548fc61824dc2354e55f17a419895175d0f1344`.

CVF says its Open Access version is identical to the accepted paper except for the CVF watermark; it also says the final proceedings version is on IEEE Xplore. Copyright remains with the authors/rightsholders. Thus “open access” here must not be read as an MIT/Apache code or model license.

## Bibliographic identity

**Exact official title:** “Cross-modal Representation Learning for Diffusion-generated Image Detection”

**Authors, in order:** Tao Gong; Dayong Wang; Qi Chu; Bin Liu; Nenghai Yu.

**Corresponding author:** Qi Chu (starred in the paper; the PDF footnote misspells “Corresponding” as “Correponding”).

**Affiliations for the authors:**

1. School of Cyber Science and Technology, University of Science and Technology of China (USTC)
2. Anhui Province Key Laboratory of Digital Security

**Venue:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

**Publication month:** June 2026 (official CVF BibTeX). The CVPR program lists the poster for Sunday, 7 June 2026, 10:45 AM–12:45 PM PDT, ExHall F 698.

**Proceedings pages:** 36092–36102. The PDF is 11 pages including references.

**DOI / arXiv / identifiers:** No DOI is present in the official CVF metadata or BibTeX. Exact-title searches on Crossref, DBLP, OpenAlex, and Semantic Scholar did not return this work on 2026-08-30. The arXiv API returned no exact-title or matching Tao Gong/NPR record. Do **not** invent a DOI or arXiv ID. Database ingestion may lag the conference publication, so recheck before final publication of a bibliography.

**Code and supplement:** The official CVF “Related Material” section links only the PDF and BibTeX. No official code or supplement is linked. GitHub repository search by exact title found no repository on 2026-08-30. This is important for reproducibility and for the challenge rule against directly replicating an existing detector.

### Recommended citation

```text
T. Gong, D. Wang, Q. Chu, B. Liu, and N. Yu, “Cross-modal Representation
Learning for Diffusion-generated Image Detection,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
2026, pp. 36092–36102.
```

Official CVF BibTeX:

```bibtex
@InProceedings{Gong_2026_CVPR,
    author    = {Gong, Tao and Wang, Dayong and Chu, Qi and Liu, Bin and Yu, Nenghai},
    title     = {Cross-modal Representation Learning for Diffusion-generated Image Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36092-36102}
}
```

### Citation status and citational context

Google Scholar showed **“Cited by 1”** on 2026-08-30. Citation counts are volatile and Scholar is not authoritative metadata. The one result was:

- Davide Cozzolino, Giovanni Poggi, and Luisa Verdoliva, “Understanding Why Foundation Models Work for Diffusion-Generated Image Detection,” arXiv:2608.12155v1, 12 Aug. 2026, <https://arxiv.org/abs/2608.12155>. It cites Gong et al. as ref. [21], in a group of dedicated AI-image detectors that use generative-process traces. Local inspection copy: `.research/Cozzolino_2026_Understanding_Why_Foundation_Models.pdf`.

Crossref had no record for the target paper, so no Crossref cited-by count was available.

## Title ambiguity and disambiguation

1. **Capitalization is semantic, not a new paper.** The official spelling is “Cross-**modal**” and “Diffusion-**generated**.” Search engines or title-case bibliographies may display “Cross-Modal” or “Diffusion-Generated.” Use the official CVF form in a formal citation.
2. **“Cross-modal” does not mean image–text.** Both modalities are visual: ordinary RGB and NPR, a deterministic local pixel-difference transform computed from the same RGB image (pp. 36093–36095, Figs. 1–3).
3. **Do not confuse it with a different video paper:** Xiufeng Song et al., “On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection,” NeurIPS 37 (2024), pp. 122054–122077, DOI `10.52202/079017-3878`. Crossref ranks that different paper highly for a loose title search. It has different authors, a video task, “multi-modal,” and no hyphen in “Diffusion Generated.”
4. **SDID is the method name, not an alternate paper title.** The expansion “Strong Diffusion-generated Image Detector” appears in contribution 3 on p. 36094.
5. **The scope is broader than the title suggests.** The paper repeatedly discusses fake/generated-image detection and reports BigGAN as well as diffusion-generator results in GenImage (Table 1, p. 36097). Its central claim is cross-generator real/fake detection, though the title foregrounds diffusion-generated images.

## Architecture and training mechanics

Page references below are printed proceedings pages; parenthetical PDF page numbers make local checking easier.

### 1. Two input views and two encoders

- An RGB image is used directly by the **RGB encoder** `E^RGB`.
- The **NPR input** is computed from that same RGB image and sent to a separate **NPR encoder** `E^NPR`.
- Main implementation: RGB encoder = pretrained **DINOv2 ViT-L/14**, fine-tuned using **LoRA**; NPR encoder = ImageNet-pretrained **ResNet-101** (p. 36096, §4.1).
- The real/fake paths drawn twice in Figure 3 share branch weights. There is one RGB encoder and one NPR encoder, not class-specific encoders (Fig. 3, p. 36095).
- The two final branch features are concatenated and classified with a cross-entropy head. At inference, the encoders, concatenation, and classifier remain; queues and representation-learning losses are training-only (pp. 36094, 36096).

**NPR computation:** Divide each RGB channel into non-overlapping `2 × 2` patches. Within every patch and channel, subtract the top-left pixel value from every pixel. This produces the NPR view. The authors explicitly credit NPR to Tan et al. [44] and do not claim it as their contribution (pp. 36093, 36095, §3.2).

### 2. Four feature queues

Training maintains modality- and class-specific queues:

- `Q_real^RGB`, `Q_real^NPR`
- `Q_fake^RGB`, `Q_fake^NPR`

The current batch is enqueued and the oldest batch is dequeued. Each queue length is **2,048** (pp. 36094–36096). Paired RGB/NPR entries preserve the correspondence used for “same top-K” neighbor comparisons in CMMD (Fig. 3).

### 3. Cross-Modal Contrastive Learning (CMCL)

CMCL increases **inter-class separability**. For an RGB feature from a real image:

- positive = its NPR feature from the *same image*;
- negatives = NPR features of fake images stored in `Q_fake^NPR`.

The method applies InfoNCE symmetrically in both modality directions and both classes, producing four terms:

1. real RGB anchor / matching real NPR positive / fake NPR queue negatives;
2. real NPR anchor / matching real RGB positive / fake RGB queue negatives;
3. fake RGB anchor / matching fake NPR positive / real NPR queue negatives;
4. fake NPR anchor / matching fake RGB positive / real RGB queue negatives.

The dot-product InfoNCE temperature is **0.07**. Equations (1)–(2), p. 36095, give the exact formula. Figures 2a (p. 36093) and 3 (p. 36095) show the pairing.

This is supervised by real/fake class membership even though InfoNCE is used: opposite-class queued examples are deliberately selected as negatives. It is not ordinary self-supervised augmentation-only contrastive learning.

### 4. Cross-Modal Mutual Distillation (CMMD)

CMMD increases **intra-class compactness** and transfers neighborhood structure across modalities (pp. 36093, 36096, §3.4).

For a feature `z` and same-class queue anchors:

1. compute cosine similarity to queued embeddings;
2. select the **top K = 128** nearest neighbors;
3. softmax the similarities with temperature `τ` into a neighborhood probability distribution (Eq. 3);
4. use the corresponding same top-K sample identities in the other modality;
5. minimize KL divergence between RGB and NPR neighborhood distributions.

Distillation is bidirectional. Each modality is both teacher and student, and its knowledge changes during training. It is applied within real images and within fake images, never from a real sample distribution to a fake one. Four KL terms cover RGB→NPR and NPR→RGB for real and fake classes (Eqs. 4–5, p. 36096). No extra model forward pass is needed for anchors because the contrastive queues already contain them.

### 5. Total objective and inference

Equation (6), p. 36096:

```text
L = L_cls + λ1 L_CMCL + λ2 L_CMMD
```

- `L_cls`: cross-entropy on the concatenated RGB and NPR features.
- `λ1 = 0.1`, `λ2 = 0.1`.
- Main SDID inference uses both encoders and concatenates both features.
- A separate Table 6 ablation trains with NPR/CMCL/CMMD but predicts from RGB alone. That ablation gets a large gain with no NPR branch at test time, but it is not the primary two-branch SDID configuration (p. 36099).

### Architecture figures

- **Figure 1, p. 36093 (PDF p. 2):** top-level two-branch detector, RGB → RGB encoder and computed NPR → NPR encoder, then CMCL/CMMD.
- **Figure 2, p. 36093 (PDF p. 2):** intuition for same-image positive/opposite-class negatives in CMCL and neighborhood-distribution matching in CMMD.
- **Figure 3, p. 36095 (PDF p. 4):** full training pipeline, weight sharing across real/fake class paths, four queues, concatenation/classification, and two KL directions.
- **Figure 4, p. 36099 (PDF p. 8):** t-SNE. Adding CMCL separates classes; adding CMMD makes the real class more compact.

## Experiments and exact table takeaways

All reported metrics are **average accuracy with a fixed 0.5 decision threshold**, not AUC (p. 36096).

### Main comparisons

- **Table 1, p. 36097:** train on GenImage/SDv1.4 and test across Midjourney, SDv1.4, SDv1.5, ADM, GLIDE, Wukong, VQDM, BigGAN. SDID average = **98.4%**; next listed result CoD = 96.2%. Notable SDID values: Midjourney 96.9, ADM 95.9, GLIDE 98.7, BigGAN 98.8.
- **Table 2, p. 36097:** train on DRCT-2M/SDv1.4 and test on 16 subsets. SDID average = **92.4%**; next DLFE = 90.9%. Hard reconstructed-image subsets remain weaker: SDv1-DR 86.7, SDv2-DR 75.9, SDXL-DR 71.9.
- **Table 3, p. 36098:** train on DRCT-2M/SDv1.4 following Co-Spy and test across 22 Co-Spy-Bench subsets. SDID average = **96.1%**; Co-Spy = 87.1%. Lowest SDID results are PG-v2-256 90.5, FLUX.1-sch 93.9, and FLUX.1-dev 93.6.

### Ablations

- **Table 4, p. 36098:** RGB only 86.7; NPR only 88.5; RGB+NPR 90.1; +CMCL 95.4; +CMMD **98.4** on GenImage. Thus simple fusion adds 1.6 points over NPR alone, CMCL adds 5.3 over fusion, and CMMD adds another 3.0.
- **Table 5, p. 36098:**
  - RGB+RGB augmentation view: 86.9 → 93.0 with CMCL → 93.2 with CMMD.
  - RGB+high-frequency view: 88.3 → 93.8 → 95.5.
  - RGB+NPR: 90.1 → 95.4 → **98.4**.
  The RGB+RGB CMCL case is conventional contrastive learning; CMMD barely helps when both inputs are the same modality.
- **Table 6, p. 36099:** use NPR only for representation learning but classify with RGB features: 86.7 → 93.6 with CMCL → **96.4** with CMMD. This supports an optional single-RGB-encoder deployment path, with a 9.7-point gain and no added test-time branch cost.
- **Table 7, p. 36099:** swapping DINOv2 for a CLIP image encoder still gives gains: RGB+NPR 89.3 → 93.2 with CMCL → **95.5** with CMMD. RGB alone is 81.3 and NPR alone 88.5 in this setting.

## Important prior citations and what is original

- **NPR is prior art, explicitly not a contribution:** Chuangchuang Tan et al., “Rethinking the Up-Sampling Operations in CNN-Based Generative Network for Generalizable Deepfake Detection,” CVPR 2024, pp. 28130–28139, target ref. [44].
- **Closest contrastive predecessor named by the authors:** CoDE, Lorenzo Baraldi et al., ECCV 2024, pp. 199–216, target ref. [2]. It uses RGB-only conventional contrastive learning; the target paper argues CMCL/CMMD and NPR add a cross-modal structure.
- **InfoNCE / queues:** SimCLR [6], MoCo [17], plus MoCo v2/v3 [7,8].
- **Relational distillation lineage:** relational KD [34,36,45], PKT [35], CompRess [1], SEED [13].
- **Other RGB+NPR work distinguished by the authors:** AIGI-Holmes [55] uses RGB and NPR with an MLLM for explanations, whereas SDID focuses on optimizing a forgery-aware embedding.
- **Backbones/adaptation:** ResNet [16], ImageNet [11], LoRA [18]. The paper names DINOv2 in implementation details but, unusually, does not include a DINOv2 reference in its bibliography.

The defensible original contribution is therefore **not NPR** and not “two visual branches” alone. It is the paired training scheme: class-aware cross-modal InfoNCE plus bidirectional, same-class neighborhood-distribution distillation over aligned RGB/NPR queues.

## Reproducibility gaps and applicability cautions

The main paper does **not** report the optimizer, learning rate, schedule, batch size, epochs, input resolution/crop policy, training augmentations, LoRA rank/target layers, feature or projection dimensions, classifier-head design, random seeds, hardware, training time, parameter count, inference latency, or confidence calibration. It delegates dataset protocols to GenImage/DRCT/Co-Spy precedents. No official code is linked. Exact replication from this paper alone is therefore not possible.

It also does not present a controlled JPEG/blur/resize/noise/crop robustness table. The experiments establish cross-generator and benchmark generalization, not the full real-world transformation robustness required by the TechJam brief. Accuracy at threshold 0.5 does not establish score calibration or AUC. For the project, the paper is best used as architectural inspiration and a cited comparison, with an independently designed contribution and explicit transformation evaluation, rather than as a directly copied detector.

## Source reliability notes

1. **CVF Open Access landing page/PDF** — primary full text and authoritative bibliographic source.
2. **CVPR virtual program** — primary conference schedule and author/title confirmation.
3. **Google Scholar** — used only for a dated, volatile citation-count observation and to identify the one citing paper; not used to create DOI metadata.
4. **Crossref / DBLP / OpenAlex / Semantic Scholar / arXiv API** — negative identity checks as of 2026-08-30. Absence can reflect indexing lag.
5. **Paper text itself** — all method equations, figures, tables, implementation values, results, and limitations above were checked against the downloaded 11-page CVF PDF.
