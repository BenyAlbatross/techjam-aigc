# Robust Detection of AI-Generated Images Under Real-World Transformations

## Document status and source hierarchy

This document is the repository's source of truth for Track 5 of TikTok TechJam 2026. It consolidates:

1. the supplied written problem statement;
2. rules and clarifications presented during the technical workshop webinar on 28 August 2026, 5:00–5:45 PM; and
3. participant notes from the webinar Q&A.

When statements conflict or remain ambiguous, the written brief and explicit competition rules take precedence. Organizers reserve final authority. Items that need organizer confirmation are listed under [Open questions and ambiguities](#open-questions-and-ambiguities).

Webinar recording filename: `#5 Robust Detection of AI-Generated Images Under Real-World Transformations.mp4`

## Challenge summary

Build a hackathon-scale prototype that performs **binary, image-level classification** between:

- **AIGC:** a purely AI-generated image; and
- **non-AIGC:** an authentic image.

The detector must return a confidence score for each input image and remain effective after realistic post-processing and redistribution. It should generalize across both older generators, such as Stable Diffusion, and newer state-of-the-art generator families, such as diffusion transformers.

AI-edited or composited images are not the focus of this track. For example, a real poster with a small AI-generated icon pasted into it is outside the stated evaluation focus.

## Background

Generative AI tools make it possible to create realistic synthetic images at scale. This creates risks for online platforms, including misinformation, impersonation, fraud, and reduced trust in digital content.

Detection becomes harder after images are compressed, cropped, reposted, filtered, or otherwise lightly transformed. The challenge therefore emphasizes robustness under real-world redistribution, not only accuracy on clean laboratory data.

## Problem statement

Develop a prototype that distinguishes AI-generated images from authentic images while maintaining strong performance after realistic transformations. A complete solution should include:

- a clear technical approach;
- an evaluation strategy covering clean and transformed images;
- robust handling of common post-processing operations;
- error analysis; and
- a thoughtful discussion of trade-offs, including robustness, generalization, and false positives.

## Robustness transformations

Evaluation will consider a subset of the following transformations. More than one transformation may potentially be applied to an image; the supplied materials do not specify the exact composition policy.

| Transform | Parameters | Real-world analogue |
| --- | --- | --- |
| JPEG compression | Quality `90`, `70`, `50`, `30` | Social-media re-encoding and messaging |
| Gaussian blur | Kernel sigma `0.5`, `1.0`, `2.0` | Out-of-focus images |
| Resize | Scale to `0.5x` or `0.25x`, then upscale | Thumbnail generation |
| Gaussian noise | Sigma `0.02`, `0.05`, `0.10` | Low-light sensor noise |
| Color jitter | Brightness, contrast, and saturation `+/-20%` | Filter apps and auto-enhancement |
| Center crop | Retain `80%` | Profile-picture cropping and reframing |

The workshop also described WebP compression, re-saving, filtering, noise, and re-screenshoting as realistic scenarios, but did not provide formal parameter ranges for them.

## Scope

### In scope

- Image-level AIGC detection
- Robustness to common image transformations
- Feature engineering
- Model and architecture design
- Evaluation design
- Error analysis
- Explainability ideas

### Out of scope

- Full production deployment
- Platform-wide moderation systems
- Video, audio, and other non-image modalities
- AI-edited or partially composited images as a distinct detection task

### Prototype assumptions

- Work at hackathon scale with limited compute and no access to internal production systems.
- Optimize for a convincing proof of concept rather than a production-grade service.
- Reasonable deployment assumptions are allowed when stated clearly.

## Competition rules and constraints

Violations may result in disqualification, and organizers reserve final authority on all matters.

### Models and source code

- The final solution must use models with **fewer than 2 billion parameters**.
- Pretrained backbones must be publicly available. Examples from the workshop include ResNet, ViT, CLIP, and DINOv2.
- Custom architectures must be released under an MIT or Apache license.
- A pretrained backbone may be fine-tuned, but a solution that merely uses or directly replicates an existing pretrained AIGC detector or existing approach is disallowed and may be disqualified.
- The workshop indicated that a simple backbone fine-tune is also unlikely to score best on technical criteria. Participants are expected to demonstrate their own technical contribution.
- Winning teams must open-source their training pipeline, hyperparameters, evaluation code, and model weights.

### Data

- Use only public or properly licensed datasets.
- Do not use proprietary or production data.
- Do not train on test labels.
- Teams may create transformed samples from allowed datasets, but must include the generation or augmentation scripts for reproducibility.
- Dataset selection is **not limited** to the three suggested datasets in this brief.

### Evaluation generalization

The hidden evaluation is expected to cover:

- older generators, such as Stable Diffusion;
- state-of-the-art generators, including diffusion-transformer-based models; and
- real-world image transformations.

The supplied materials do not enumerate every hidden generator or transformation.

### Watermarks

The webinar answer regarding SynthID was: "For the test set, we don't consider the SynthID watermark." This wording is ambiguous. Do not build a solution that depends on SynthID or assume without confirmation whether it is absent, stripped, or simply excluded as an evaluation signal.

## Available resources and data

Teams may use public or properly licensed AIGC-detection and image-forensics datasets, self-created transformed samples, and public documentation for relevant machine-learning and computer-vision libraries.

Suggested datasets:

- [SID Set on Hugging Face](https://huggingface.co/datasets/saberzl/SID_Set)
- [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [WildFake on ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — use the site's translation control if needed

These are suggestions, not an exhaustive allowlist.

### Demonstration-only validation dataset

Organizers selected the following subset of WildFake so teams can demonstrate performance and track iterative improvements:

| Class | Source | Number of images |
| --- | --- | ---: |
| Non-AIGC | COCO `val2017` | 4,998 |
| AIGC | DALL-E Advanced | 8,843 |

This validation dataset:

- is for demonstration and reference benchmarking only;
- does **not** contribute to the final score; and
- must **not** be used for training.

## Required inference interface

The public repository must contain a runnable script that:

1. accepts an image directory as input;
2. runs inference for each image;
3. outputs a confidence score indicating the likelihood that each image is AIGC-generated; and
4. writes a JSON file with `image_path` and `pred` for every image.

The supplied brief does not define the script name, command-line arguments, JSON container shape, score range, or exact calibration requirement. Document those choices clearly in the project README unless organizers publish a stricter interface.

## Expected deliverables

### 1. Written project description on Devpost

Include:

- how the solution addresses the problem;
- development tools used, such as VS Code, Colab, or Jupyter;
- models or APIs used;
- libraries and frameworks used, such as Hugging Face Transformers, PyTorch, scikit-learn, or pandas; and
- datasets and assets used.

### 2. Public code repository

Submit a public GitHub or equivalent code repository containing:

- well-structured, commented code for all solution components;
- the required directory-to-JSON inference script; and
- a README containing:
  - a project overview;
  - setup and installation instructions;
  - steps to reproduce results;
  - limitations and improvements that would be made with more time; and
  - team-member contributions, where applicable.

The competition-rules slide additionally requires reproducible training and augmentation code, evaluation code, hyperparameters, and—if the team wins—model weights.

### 3. Demo video

Submit a **2–4 minute** end-to-end demonstration that:

- shows the solution working, such as inference results, a dashboard, or model predictions;
- is uploaded to YouTube with public visibility;
- is linked in the Devpost description; and
- contains no third-party trademarks or copyrighted content used without permission.

### 4. Robustness evaluation summary

Include a compact table or visual comparison of performance on clean images and transformed images.

### 5. Error analysis note

Show representative false positives and false negatives, and discuss relevant trade-offs.

## Judging criteria

| Criterion | Definition | Weight |
| --- | --- | ---: |
| Technical Execution | Strong engineering fundamentals, well-structured code, thoughtful architecture, effective model or API use, a reliable demo, and technical complexity reflecting deliberate decisions | 35% |
| Innovation & Problem Insight | Originality of the idea and approach, sharp framing of the challenge, understanding of why it matters, and a direct response to the problem | 20% |
| Impact & Relevance | Potential to deliver value to real users or stakeholders, with meaningful reach, tangible benefit, and relevance beyond the prompt | 20% |
| Feasibility & Practicality | A realistic, sustainable approach with proportionate resource usage and an architecture grounded in real-world conditions | 15% |
| Presentation & Communication | Clear communication; at the final event, a coherent problem-to-solution story and the ability to answer questions with depth | 10% |

## Non-binding workshop guidance

This section records technical suggestions from the webinar. It is **not** a list of mandatory architecture requirements.

### Why the problem is difficult

- **Generalization:** New diffusion, GAN, and future generator families may leave fingerprints that differ from the training set.
- **Robustness:** Compression, blur, cropping, rescaling, color shifts, filtering, and re-saving can destroy the signals used by a detector.
- A detector that performs well only on clean, generator-native images may fail on real-world feeds.

### Potential detection signals

- **Frequency artifacts:** Periodic patterns left by GAN or diffusion upsampling in the Fourier spectrum
- **Noise and sensor fingerprints:** Real-camera sensor noise such as PRNU, which synthetic images may lack or imitate poorly
- **Texture and fine detail:** Skin, hair, foliage, text, and reflections
- **Semantic or physical inconsistencies:** Lighting, hands, text, shadows, and other scene-level errors

The workshop encouraged hybrid systems that combine high-level semantic features, such as CLIP-like representations, with low-level frequency or patch features. The rationale is that each branch may capture evidence missed by the other. This is a suggested direction, not a required architecture.

### Baseline described in the workshop

The illustrative baseline was:

`input image -> resize/normalize -> CNN or ViT backbone -> classifier head -> real-vs-AI score`

A frequency branch using FFT or DCT features was suggested as an optional upgrade. The workshop advised returning a confidence score rather than only a hard label.

### Training for redistribution

The workshop emphasized training-time transformations that simulate real feeds:

- JPEG compression
- blur
- resize and crop
- color jitter
- noise
- re-screenshoting

It also highlighted the risk of shortcuts, such as learning dataset-specific JPEG characteristics instead of generator artifacts. Suggested mitigations included aligning pixel and frequency characteristics between classes and checking whether augmentations introduce class-correlated cues.

### Trade-offs to report

- **Robustness vs. clean accuracy:** Heavy augmentation may lower clean-image performance.
- **Generalization vs. specialization:** Tuning for one generator may hurt performance on unseen generators.
- **Complexity vs. feasibility:** A multi-branch ensemble may improve results but increase training, inference, and demo risk.

The workshop explicitly described this as an open problem with no single best solution and encouraged approaches beyond the examples presented.

## Webinar Q&A clarifications

| Question | Recorded clarification | Implementation consequence |
| --- | --- | --- |
| Are teams limited to the three listed datasets? | No. | Other public or properly licensed datasets may be used. |
| What generator types will be tested? | Both older models such as Stable Diffusion and newer state-of-the-art models such as diffusion transformers. | Evaluate cross-generator generalization and avoid specializing to one generator family. |
| May teams fine-tune a pretrained backbone? | Yes, but a plain fine-tune is unlikely to be strongest on technical criteria. Merely using an existing pretrained AIGC detector may lead to disqualification. | Add an original, explainable technical contribution and document it. |
| Are AI-edited or composited images considered AIGC? | This track focuses on purely generated images, not AI-edited images. | Treat the task as binary classification between pure AIGC and authentic images. |
| Will SynthID be stripped? | "For the test set, we don't consider the SynthID watermark." | Do not rely on SynthID; the exact test-set handling remains ambiguous. |
| Will calibrated probabilities or a fixed threshold be evaluated in addition to AUC? | An informal note records the answer as "discontinuous probability score (or something like that)." | No reliable requirement can be derived from this answer; preserve score ranking and avoid assuming calibration is irrelevant. Seek organizer confirmation. |

## Open questions and ambiguities

Confirm these with organizers before treating them as implementation requirements:

1. **Official metric:** A participant question refers to AUC, but the supplied written brief does not state the final technical metric or how it relates to the weighted judging rubric.
2. **Score semantics:** The required range, calibration expectations, and any operating threshold for `pred` are unspecified.
3. **JSON schema:** Required fields are known, but the root structure, filename, ordering, and error handling are unspecified.
4. **Transform composition:** Evaluation uses a subset of listed transforms, but it is unclear whether multiple transforms can be chained and in what order.
5. **SynthID:** It is unclear whether the watermark is removed, absent, ignored during scoring, or merely not guaranteed.
6. **Hidden evaluation coverage:** Exact generators, datasets, class balance, image formats, and transformations are not enumerated.

Until clarified, implement deterministic, documented behavior; emit a continuous AIGC score; support common image formats robustly; and do not depend on watermarks or dataset-specific shortcuts.

