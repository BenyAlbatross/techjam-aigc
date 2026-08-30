# TRACE-RX-Alternative3 Bias Register

This is the shareable B1–B18, M1–M4, and S1–S3 checklist for the two-head model: a shared DINOv3 + LoRA encoder with a global classifier head, plus an authentic-memory comparison head over DINO patch tokens.

## Dataset and data-pipeline biases

### B1 — Benchmark/test leakage and licensing

Do not train on hidden-test images, test labels, demonstration-only validation images, or their near-duplicates. Use only public or properly licensed data. The organizer-selected WildFake subset—COCO val2017 authentic images and DALL-E Advanced fake images—is demonstration-only and must not be used for training.

### B2 — Dataset–label confounding

If authentic images come from one dataset and fake images from another, the model can learn dataset identity instead of provenance. Each major source ecosystem should contribute both labels where possible; otherwise, give it an explicitly matched acquisition and post-processing counterpart.

### B3 — Semantic-content bias

The two classes must not have systematically different subjects or scenes—for example, COCO everyday photographs as authentic and fantasy portraits from Civitai as fake. Exact content pairing is unnecessary, but coarse semantic clusters should contain both labels and should not have extreme class skew.

### B4 — Style bias

Photorealism, illustration, anime, 3D rendering, graphic design, low-light imagery, or amateur photography must not become label proxies. This matters especially for WildFake and community-platform data, whose synthetic side may contain conspicuous aesthetic styles.

### B5 — Authentic-source and acquisition bias

COCO, FFHQ, OpenImages, LAION, social-media images, and camera-native collections have different capture and curation pipelines. Balance authentic source families and evaluate on authentic sources withheld from training so that stock-photo polish, face framing, or camera noise does not stand in for authenticity.

### B6 — File-format, codec, and compression bias

JPEG, WebP, or PNG choice; JPEG quality; chroma subsampling; quantization tables; repeat saves; and encoder implementation can reveal the source pipeline. Match or randomize these variables independently of class, strip metadata, and apply re-encoding symmetrically.

### B7 — Native-resolution, aspect-ratio, and preprocessing bias

Generator APIs, benchmarks, and camera datasets often have distinctive native sizes and aspect ratios. A final 224 × 224 crop does not erase earlier resizing, upsampling, or interpolation traces. Balance native geometry and use the same resize/crop pipeline for both labels.

### B8 — Platform and post-processing bias

SynthWildX and the social/community subsets of AIGIBench contain website or social-network laundering. Platform recompression, screenshots, overlays, filters, and download tools can dominate the signal. Comparable platform histories must occur in both classes.

### B9 — Challenge-transformation distribution bias

JPEG compression, blur, downscale/upscale, noise, color jitter, and crop must be sampled independently of label. Clean, single-transform, and transform-chain proportions must be identical across authentic and fake examples.

### B10 — Generator-family imbalance

Do not let Stable Diffusion variants or another prolific family dominate the fake class. Balance GANs, convolutional diffusion models, diffusion transformers, autoregressive or image-token models, and licensed commercial APIs. Hold out whole generator families for evaluation.

### B11 — Generator-version and configuration bias

Checkpoints, versions, samplers, guidance scales, step counts, VAEs, resolutions, seeds, safety filters, and upscalers can leave distinct signatures. Record and diversify these variables, and group related versions during splitting instead of treating them as independent generators.

### B12 — Prompt-template bias

Repeated templates, category words, negative prompts, prompt enhancers, and synthetic caption styles change image content and composition. Use diverse human and machine-written prompts, vary length and specificity, and keep prompt families from crossing held-out generator splits.

### B13 — API-output-pipeline bias

An API may impose characteristic resolution, codec, safety filtering, enhancement, invisible preprocessing, or visible marks. Preserve provenance records, normalize or symmetrically randomize delivery handling, and hold out entire APIs—not merely individual images—for evaluation.

### B14 — Curation and aesthetic-selection bias

Human filtering, aesthetic scorers, NSFW filters, obvious-artifact removal, and popularity thresholds change task difficulty. AIGIBench uses prompt categories, aesthetic filtering, and manual removal of obvious fakes. Apply comparable quality selection to both labels and retain high-fidelity hard fakes.

### B15 — Label and task-scope noise

Social and community sources may contain incorrect tags, undisclosed AI generation, authentic images on AI sites, AI edits, composites, or mixed screenshots. Because the task is purely generated versus authentic, exclude or isolate ambiguous and partially edited cases and audit every source.

### B16 — Duplicate and lineage leakage

Exact duplicates, resized or recompressed copies, crops, prompt-seed siblings, bursts, and near-identical outputs must stay in one split. Deduplicate with cryptographic hashes plus perceptual or embedding similarity, then split by source lineage, generator family, API, and prompt or seed group.

### B17 — Watermark, overlay, and metadata bias

Logos, community-site marks, borders, captions, filenames, EXIF, software tags, ICC profiles, and SynthID-like signals must not become shortcuts. Strip metadata, audit overlays, and exclude or reproduce visible marks symmetrically. Do not rely on SynthID.

### B18 — Temporal drift

Generators, APIs, platform codecs, and authentic-camera pipelines change over time. Preserve timestamps where possible and include a forward-in-time evaluation split with later generator versions and changed delivery pipelines.

## Authentic-memory-head biases

### M1 — Authentic-memory coverage and source domination

A real-only memory estimates the manifold of authentic patch features. If COCO, FFHQ, or another source dominates it, rare legitimate images may sit far from its prototypes and become false positives. Balance the memory across authentic source, semantic, style, and quality strata.

### M2 — Memory quality-state mismatch

A clean-only authentic memory may interpret JPEG artifacts, blur, resizing, noise, or crop as fake evidence. Populate it with authentic transformation endpoints and condition or normalize residual statistics by observable quality state.

### M3 — Memory contamination

The authentic memory must contain only verified authentic images. Synthetic, AI-edited, mislabeled, or leaked evaluation images corrupt the reference distribution. MIRROR's real-only-memory result supports preserving this interpretation rather than mixing fake samples into the bank.

### M4 — Memory capacity and sparsity

A memory that is too small under-covers legitimate diversity; one that is too large or flexible can find close prototypes for synthetic patches and absorb the anomaly. Use a bounded prototype budget, balanced routes, minimum support per prototype, and regularized similarity statistics.

## Shared-encoder biases

### S1 — Phase-1/phase-2 representation drift

The memory is built in frozen base-DINO space, while phase 2 adapts DINO with LoRA. If the patch representation moves too far, distance to the frozen memory loses its original meaning. Freeze base weights and memory components, constrain LoRA capacity, and anchor adapted real-image features to stopped-gradient base features:

$$
\mathcal{L}_{\mathrm{anchor}}
= 1 - \cos\!\left(E_{\mathrm{LoRA}}(x_{\mathrm{real}}),
\operatorname{sg}\!\left[E_{\mathrm{base}}(x_{\mathrm{real}})\right]\right).
$$

### S2 — Shared-encoder correlation

The global and memory heads are not independent because both consume the same adapted DINO features. A source, content, or codec shortcut learned by LoRA can move both logits together and make fusion falsely confident. Treat the heads as correlated measurements and use simple regularized fusion.

### S3 — DINO pretraining overlap

DINOv3 pretraining may overlap semantically—or possibly at image level—with downstream authentic datasets. Document the backbone and disclosed data, deduplicate where possible, and emphasize held-out-source, held-out-generator, transformed, and temporal evaluation instead of random-split claims.

## What the planned 20% DDA allocation addresses

The planned 20% Dual Data Alignment allocation primarily helps B3 content, B5 acquisition appearance, B6 format/frequency, and B7 size/resolution bias. It does not by itself solve B2 dataset identity, B8 platform history, B10–B14 generator/prompt/API/curation bias, B15 label noise, B16 lineage leakage, B18 temporal drift, M1–M4 memory bias, or S1–S3 shared-encoder bias.

The most dangerous combination is community or social fake imagery from WildFake, SynthWildX, or AIGIBench paired with clean benchmark authentic imagery. Source, format, quality, content, and post-processing then all correlate with the label.

> Minimum defensible rule: each major source ecosystem must contribute both labels where possible, or receive an explicitly matched acquisition and post-processing counterpart.

## Literature grounding

- [WildFake: A Large-scale Challenging Dataset for AI-Generated Image Detection (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/download/32363/34518): generator, architecture, weight, time, version, and source diversity.
- [Raising the Bar of AI-Generated Image Detection with CLIP (CVPR Workshops 2024)](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Cozzolino_Raising_the_Bar_of_AI-generated_Image_Detection_with_CLIP_CVPRW_2024_paper.pdf): SynthWildX and in-the-wild post-processing or laundering.
- [AIGIBench: A Comprehensive Evaluation of Image-Level AI-Generated Content Detection (2025)](https://arxiv.org/html/2505.12335): controlled, advanced-generator, social, and community sources; transformations and preprocessing.
- [A Bias-Free Training Paradigm for More General AI-Generated Image Detection (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html): content, format, and resolution bias.
- [Dual Data Alignment for Robust AI-Generated Image Detection (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/991c9f1799cfeebb4217baaacc462d86-Abstract-Conference.html): content, frequency/format, and size alignment.
- [A Sanity Check for AI-Generated Image Detection (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0303773962ea1b5394c3a83cc7dd066-Abstract-Conference.html): conspicuously flawed generations versus curated high-fidelity fakes.
- [MIRROR: A Real-Image Memory for AI-Generated Image Detection (2026)](https://arxiv.org/html/2602.02222): real-only visual memory, memory composition, and phase-2 LoRA adaptation.

## Evaluation constraint

If AIGIBench or another source reserves advanced-generator, social, community, or test subsets for evaluation, do not train on those subsets while claiming held-out generalization. Use only designated training partitions and public or properly licensed data. The same prohibition applies to the TechJam demonstration-only WildFake subset.

