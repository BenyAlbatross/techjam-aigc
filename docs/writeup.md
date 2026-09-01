# TRACE LENS

An AI-image detector designed to remain useful after compression, blur, noise, resizing, and cropping, supported by an evidence browser that shows how and where predictions fail.

## Submission snapshot

- **Track:** Track #5: Robust Detection of AI-Generated Images Under Real-World Transformations
- **Primary product:** An AI-image detector designed for resilience to common real-world transformations
- **Auxiliary product:** An evidence browser for explaining model behaviour, robustness, and failure cases
- **GitHub Repository:** https://github.com/BenyAlbatross/techjam-aigc/
- **Public demo video:** TBD. 

## Project story

### Inspiration

AI-generated images rarely stay pristine. They are recompressed, blurred, resized, cropped, recoloured, and shared through platforms that alter their pixels. A detector may perform well on clean images yet fail after an ordinary transformation. Many published AIGC detection pipelines are evaluated on pristine images and outputs from older diffusion models [1, 2]. We built TRACE LENS to expose these failure modes instead of hiding them behind a single headline score.

### What it does

TRACE LENS is an AI-image detection system designed around transformation resilience. The detector is the primary product. Its supporting evidence browser turns saved benchmark outputs into an inspectable analysis layer, allowing users to see what the model predicted and how common transformations affected its errors.

The browser provides:

- A gallery of authentic and fully synthetic images
- Fixed-threshold outputs from ten pretrained comparison detectors [3]
- Filters for model, condition, ground truth, and TP, TN, FP, or FN outcome
- Per-image probabilities, thresholds, metadata, provenance, and transformation chains
- Confusion matrices, error tables, mismatch slices, and model-condition heatmaps
- Direct paths from aggregate analytics to the exact samples behind each result

The hosted experience reads saved benchmark results instead of running GPU inference or accepting uploads. This keeps the application lightweight and makes every displayed result traceable to a committed evaluation artifact.

## How we built it

We spent the first two days auditing and generating the dataset because the training distribution defines what the detector is allowed to learn. A powerful semantic backbone can separate two datasets for the wrong reason. For example, authentic images may contain everyday photography while synthetic images contain mostly portraits or fantasy art. This can produce an impressive test score without producing a detector that generalises.

Our data strategy was shaped by two ideas from AIGC detection research.

1. **Make real and synthetic images semantically comparable.** We used text-to-image generation from open-source models, including FLUX.1 Schnell [4] and SDXL 1.0 [5], and API models, including GPT Image 2 and Gemini Flash Image. We generated content across the same broad domains represented by our authentic sources. This reduced correlations between subject matter and label, pushing the detector to look beyond what an image depicts and towards evidence of how it was formed.

2. **Include synthetic images that are genuinely difficult.** Inspired by the Dual Data Alignment (DDA) paper [6], we added synthetic reconstructions closely aligned with their authentic source images. Many were difficult for us to distinguish by eye, and the model struggled with them as well. In an earlier run, DDA examples repeatedly appeared among the false negatives: synthetic images classified as authentic.

We also wanted "authentic" to cover more than the usual benchmark mix of celebrity faces and COCO scenes. Our public-data mixture includes everyday photography [7], animation frames from Blender Open Productions [8], product images from Amazon Berkeley Objects [9] and Open Food Facts [10], artwork from The Met Open Access [11], and images from Wikimedia Commons [12] and Dollar Street [13]. This breadth is deliberate. Real online imagery includes products, illustrations, art, food, homes, and imperfect photographs, not only curated portraits.

To evaluate transformation robustness, we uniformly sampled chains of one to six sequential transformations. This reflects how images are handled in practice. An image may be converted into a thumbnail, blurred, and then cropped again.

The final training split contained 67,418 images:

| Class | Images |
|---|---:|
| Authentic | 36,824 |
| AI-generated | 30,594 |

## Model architecture

Our architectural hypothesis is that authentic patch embeddings occupy recurring regions of DINOv3 [14] feature space.

First, DINOv3 converts each image into a grid of semantic patch embeddings. We fine-tune the model with rank-8 LoRA adapters [15] on its query, key, and value projections.

Before detector training, we collect patch embeddings from authentic training images and compress them into 2,048 learned prototypes. These prototypes approximate different regions of the authentic-image feature manifold.

For every new patch, the model retrieves its 32 nearest authentic prototypes and combines them into an expected authentic reference. This resembles sparse dictionary lookup and prototype attention [16]. It is also related to local kernel regression because closer neighbours receive greater weight.

The model then extracts two complementary forms of evidence:

- **Residual evidence:** The signed mean, standard deviation, and 95th-percentile residual across patches
- **Retrieval evidence:** Whether a patch matches one authentic prototype confidently or distributes its attention across several prototypes

A classifier head combines this evidence to produce the final confidence score and classification.

## Evaluation

We evaluated the model on a held-out split from our curated data and on external datasets, including WildFake [17] and EvalGen. This allowed us to measure in-distribution performance, external generalisation, and resilience to transformation chains.

### What our first detector taught us

Our earlier DINOv2-B detector [18], which used direct 224 x 224 bilinear resizing, exceeded 90% on parts of an in-distribution test set. That result was encouraging, but it did not demonstrate universal detection. Performance fell to roughly 0.4 on WildFake GAN images and 0.6 on DDIM and DDPM images.

For the current run, we moved to a centre-crop-based 224 x 224 preprocessing pipeline to better match established practice.

## Results

AI-generated images are treated as the positive class. Recall measures how many AI images were detected, while the false-negative rate measures how many were incorrectly classified as real.

### Overall benchmark performance

| Evaluation dataset | Images | Accuracy | Balanced accuracy | AUROC | AUPRC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TechJam v2 test | 2,000 | 68.55% | 73.68% | 79.49% | 89.21% | 54.61% | 7.25% | 45.39% |
| WildFake reconstructed test | 2,000 | 60.25% | 60.25% | 69.57% | 73.37% | 22.20% | 1.70% | 77.80% |
| EvalGen positive-only | 2,000 | N/A | N/A | N/A | N/A | 64.30% | N/A | 35.70% |

The detector achieved 60.25% accuracy on the external WildFake benchmark. Its false-positive rate was only 1.70%, indicating conservative predictions, but this came at the cost of missing many AI-generated images. Performance was stronger on the TechJam test set, with an AUROC of 79.49% and AUPRC of 89.21%.

### Robustness under transformation chains

| Number of transformations | WildFake AUROC | EvalGen recall |
|---:|---:|---:|
| 1 | 69.86% | 77.01% |
| 2 | 72.49% | 67.76% |
| 3 | 66.52% | 64.18% |
| 4 | 72.05% | 59.70% |
| 5 | 71.33% | 59.70% |
| 6 | 64.58% | 57.27% |
| Change from 1 to 6 | -5.28 points | -19.74 points |

After six transformations, the detector retained an AUROC of 64.58% on WildFake, a decrease of 5.28 percentage points from the single-transformation condition. Recall on the positive-only EvalGen set decreased more noticeably, from 77.01% to 57.27%. The detector therefore retains useful signal under long transformation chains, although repeated processing still removes evidence needed to identify some AI-generated images.

### TechJam validation performance

| Validation subset | Images | Accuracy | AUROC | AUPRC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean TechJam validation sample | 2,000 | 88.15% | 94.63% | 92.97% | 76.34% | 5.11% | 23.66% |

The model performed strongly on the clean TechJam validation sample, reaching 94.63% AUROC while keeping the false-positive rate close to 5%.

### Performance by generator family

| Generator family | Images | Accuracy | AUROC | AUPRC | Recall | FNR |
|---|---:|---:|---:|---:|---:|---:|
| FLUX.1 Schnell | 1,387 | 95.24% | 99.76% | 98.33% | 99.12% | 0.88% |
| SDXL 1.0 | 1,436 | 95.47% | 99.87% | 98.85% | 100.00% | 0.00% |
| GPT Image 2 | 1,571 | 92.23% | 94.24% | 88.01% | 80.87% | 19.13% |
| Gemini Flash Image | 1,425 | 87.44% | 85.94% | 35.89% | 25.00% | 75.00% |

Performance varied substantially across generator families. The detector identified nearly all FLUX and SDXL images, while Gemini Flash Image was considerably harder to detect. This variation shows why aggregate metrics alone do not fully explain model behaviour.

### DDA error analysis

| DDA generator | Evaluated images | Correctly detected | Missed | Recall | False-negative rate |
|---|---:|---:|---:|---:|---:|
| GPT Image 2 | 100 | 58 | 42 | 58.00% | 42.00% |
| Gemini Flash Image | 733 | 172 | 561 | 23.47% | 76.53% |

On the evaluated DDA subsets, GPT Image 2 recall was 34.53 percentage points higher than Gemini recall. Gemini DDA images were therefore more difficult for this detector to distinguish from their real references.

One possible explanation is that Gemini preserves more structure or pixel-level content from the reference image, leaving weaker generator-specific artefacts. This remains a hypothesis. Paired-reference similarity and forensic analysis would be needed to test it.

## Challenges

- Finding representative, properly licensed public images was difficult. We narrowed the task to detecting specific generator and forensic fingerprints, which allowed us to prioritise diversity within the real-image class instead of attempting to cover every kind of content shared online.

- Comparing ten detectors fairly required a common evaluation layer. Each detector used different data loaders, label conventions, score directions, and thresholds, but its original inference behaviour still had to be preserved.

- Every image needed clear licensing and provenance records. Restricted, private, or insufficiently reviewed sources had to remain outside the public dataset and release pipeline.

- Metric rankings alone were not enough. We needed to separate small differences in point estimates from conclusions supported by confidence intervals and statistical testing.

- Hundreds of thousands of prediction records had to remain auditable. The analysis interface needed to connect aggregate metrics and error slices back to the exact image, source, generator, transformation, and model prediction.

- The hosted demo needed to remain responsive without GPU inference or large model downloads.

## Accomplishments

We built a reproducible fixed-threshold benchmark, a deterministic transformation suite, grouped-bootstrap confidence intervals, model and dataset registries, and a condition-linked analysis interface. Our evaluation also identified a notable difference between DDA generators: detector recall was 58.00% on the evaluated GPT Image 2 DDA subset and 23.47% on the Gemini Flash Image DDA subset. This gave us a specific failure mode to investigate and reinforced the value of generator-level error analysis instead of relying only on aggregate scores. The repository also contains a directory-to-JSON inference contract and TRACE-RX Parallel research code for a future global-plus-authentic-reference architecture. That research architecture is not presented as a deployed or benchmarked detector.

## Responsible use and limitations

- TRACE LENS does not authenticate an image, identify people, match faces, or infer identity from EXIF data. It also does not cover every generator or distribution.
- Results apply only to the pinned data, models, thresholds, and transformations.
- Model training-data provenance and release status remain explicit review items.
- Detector output should support human investigation. It must not be the sole basis for enforcement, attribution, or other high-impact decisions.

## What's next
- Expand transformation-chain evaluation across more datasets and realistic user journeys
- Measure latency, VRAM use, throughput, and accessibility in the intended deployment environment

## Built with

Python, PyTorch, Transformers, Pillow, NumPy, pytest, Pixi, Next.js, React, TypeScript, Node.js, Vercel, CUDA, GitHub, and Hugging Face.

Computer vision, data visualisation, machine learning, and artificial intelligence.

## References

[1] Z. Li, J. Yan, Z. He, K. Zeng, et al. Is Artificial Intelligence Generated Image Detection a Solved Problem? arXiv preprint, 2025.
[2] C. Li, X. Wang, M. Li, B. Miao, et al. Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios (RRDataset). arXiv preprint, 2025.
[3] U. Ojha, Y. Li, and Y. J. Lee. Towards Universal Fake Image Detectors that Generalize Across Generative Models. CVPR 2023. https://arxiv.org/abs/2302.10174
[4] Black Forest Labs. FLUX.1-schnell. Hugging Face model card, 2024. https://huggingface.co/black-forest-labs/FLUX.1-schnell
[5] D. Podell, Z. English, K. Lacey, A. Blattmann, et al. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. 2023. https://arxiv.org/abs/2307.01952
[6] R. Chen, J. Xi, Z. Yan, K. Zhang, S. Wu, et al. Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable. 2025. https://arxiv.org/abs/2505.14359
[7] Z. Huang, J. Hu, X. Li, et al. SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model (SID-Set). CVPR 2025. https://arxiv.org/abs/2412.04292
[8] Blender Studio. Open movie projects. https://studio.blender.org/films
[9] J. Collins, S. Goel, K. Deng, et al. ABO: Dataset and Benchmarks for Real-World 3D Object Understanding. CVPR 2022. https://arxiv.org/abs/2110.06199
[10] Open Food Facts. The open food products database. https://world.openfoodfacts.org
[11] The Metropolitan Museum of Art. Open Access collection. https://www.metmuseum.org/about-the-met/policies-and-documents/open-access
[12] Wikimedia Commons. https://commons.wikimedia.org
[13] W. Gaviria Rojas, S. Diamos, K. Kini, D. Kanter, V. Janapa Reddi, and C. Coleman. The Dollar Street Dataset: Images Representing the Geographic and Socioeconomic Diversity of the World. NeurIPS 2022 Datasets and Benchmarks Track.
[14] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, et al. DINOv3. 2025. https://arxiv.org/abs/2508.10104
[15] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. https://arxiv.org/abs/2106.09685
[16] Z. Qin, Y. Ji, R. Tao, et al. Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes. 2026. https://arxiv.org/abs/2512.12982
[17] Y. Hong, J. Feng, H. Chen, J. Lan, H. Zhu, W. Wang, and J. Zhang. WildFake: A Large-Scale and Hierarchical Dataset for AI-Generated Images Detection. AAAI 2025. https://arxiv.org/abs/2402.11843
[18] M. Oquab, T. Darcet, T. Moutakanni, et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024. https://arxiv.org/abs/2304.07193
