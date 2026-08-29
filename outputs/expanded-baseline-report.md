# TikTok TechJam 2026 Track 5: zero-shot baseline study

Date: 29 August 2026

## Bottom line

The strongest no-fine-tuning baseline tested is **Ateeqq SigLIP** (92.9M parameters). On the larger 80-image confirmation sample it achieved:

- clean balanced accuracy **0.925**, ROC-AUC **0.994**;
- worst required-condition balanced accuracy **0.825** at 0.25× resize;
- ROC-AUC between **0.988 and 0.998** across all 15 conditions;
- **29/30 synthetic recall** on a small external slice: 10 DALL·E 3, 10 Midjourney, and 10 SDXL images.

Use it as the immediate technical baseline, not yet as an automatically approved submission dependency. Its weights are Apache-2.0, but its model card describes 60,000 AI and 60,000 human training images without naming their sources. Clear that provenance with the team or organizers first.

The practical project direction is now:

1. SigLIP as the zero-shot reference implementation;
2. explicit uncertainty/calibration for blurred and heavily resized inputs;
3. error handling aimed at compressed, watermarked, and staged real web images;
4. generator/source-disjoint validation before any fine-tuning decision.

Current ensembles are not worth their complexity. Voting fixes some SigLIP false positives under blur/resize but causes severe AI false negatives under noise because the other detectors fail together.

## Requirements applied

The test design follows the supplied [Track 5 brief](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc) and [TechJam rules](https://tiktoktechjam2026.devpost.com/rules):

- binary whole-image prediction;
- fewer than 2 billion parameters;
- no fine-tuning in this baseline phase;
- class-symmetric transformations;
- fixed, published/default thresholds—not thresholds fitted to the test images;
- clean and robustness reporting;
- eventual directory-to-JSON output with `image_path` and `pred`;
- only assets with an identifiable license or explicit authorization.

The complete confirmation grid was clean; JPEG quality 90/70/50/30; Gaussian blur σ 0.5/1/2; downscale-and-restore 0.5×/0.25×; Gaussian noise σ 0.02/0.05/0.10; deterministic ±20% color jitter; and 80% center crop.

## What was run

Ten learned checkpoints or ensemble members, five elementary heuristics, transform aggregation, and six decision-level ensemble rules were tested. Every learned checkpoint was under the track's parameter ceiling.

The initial gate used 12 real and 12 fully synthetic images from the `validation` split of [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set), with label 2 excluded. It tested clean plus the severe endpoint of each transformation family. This gate is intentionally cheap and only ranks what deserves more compute.

### Learned-model gate

| Model | Params | Clean BA | Clean AUC | Worst endpoint BA | Worst condition |
|---|---:|---:|---:|---:|---|
| Ateeqq SigLIP | 92.9M | **1.000** | **1.000** | **0.875** | resize 0.25× |
| Frontier Community Forensics | 21.8M | **1.000** | **1.000** | 0.667 | noise σ=0.10 |
| Community Forensics | 21.8M | 0.958 | **1.000** | 0.500 | noise σ=0.10 |
| Steganograph ViT | 85.8M | 0.958 | 0.979 | 0.583 | noise σ=0.10 |
| Divine EfficientNet-B0 | 4.0M | 0.833 | 0.972 | 0.625 | noise σ=0.10 |
| wkaandemir CLIP | 85.8M | 0.750 | 0.938 | 0.667 | JPEG Q30 |
| Divine ConvNeXt-Tiny | 27.8M | 0.750 | 0.910 | 0.542 | noise σ=0.10 |
| Divine ResNet-50 | 23.5M | 0.583 | 0.576 | 0.458 | color jitter |
| CapCheck ViT | 85.8M | 0.542 | 0.826 | 0.500 | color jitter |
| UnivFD | 428M | 0.500 | 0.431 | 0.500 | multiple |

BA is balanced accuracy. The gate is only 24 images; apparent perfect values are not final estimates.

### Elementary heuristics

| Heuristic | Clean AUC | Interpretation |
|---|---:|---|
| Square aspect ratio | 0.917 | Strong SID_Set shortcut, not forensic evidence |
| Exactly 1024×1024 | 0.917 | Same dataset/generator shortcut |
| AI keyword in metadata | 0.500 | Chance |
| JPEG blockiness | 0.375 | Worse than chance on this sample |
| High-frequency energy | 0.417 | Worse than chance on this sample |

The dimension heuristics are useful as a warning: a detector can look strong while learning capture/export conventions. They should be reported as dataset diagnostics, not used as primary evidence.

### Ensemble findings

- SigLIP OR Frontier exactly preserved SigLIP's endpoint pattern: clean/JPEG/noise/color/crop BA 1.0, blur 0.917, resize 0.875.
- SigLIP AND Frontier was perfect except noise, where BA fell to 0.667 with 8/12 AI misses.
- Majority vote of SigLIP, Frontier, and Steganograph fixed blur/resize but fell to BA 0.708 under noise with 7/12 AI misses.
- The three Divine members voted to only 0.833 clean BA and 0.583 noise BA.

No tested vote dominates SigLIP alone. Probability averaging was rejected because the checkpoints' probability scales and published thresholds are not comparable without a held-out calibration set.

## Larger confirmation: SigLIP

The survivor was rerun on the first 40 real and first 40 fully synthetic SID_Set validation examples. Label 2 remained excluded. The threshold stayed at the model's default 0.5.

| Condition | BA | AUC | FP | FN |
|---|---:|---:|---:|---:|
| Clean | 0.925 | 0.994 | 6 | 0 |
| JPEG Q90 | 0.900 | 0.994 | 8 | 0 |
| JPEG Q70 | 0.900 | 0.995 | 8 | 0 |
| JPEG Q50 | 0.925 | 0.997 | 6 | 0 |
| JPEG Q30 | 0.912 | 0.994 | 7 | 0 |
| Blur σ=0.5 | 0.925 | 0.994 | 6 | 0 |
| Blur σ=1 | 0.900 | 0.995 | 8 | 0 |
| Blur σ=2 | 0.838 | 0.995 | 13 | 0 |
| Resize 0.5× | 0.887 | 0.995 | 9 | 0 |
| Resize 0.25× | **0.825** | 0.995 | 14 | 0 |
| Noise σ=0.02 | 0.938 | 0.995 | 5 | 0 |
| Noise σ=0.05 | **0.975** | 0.997 | 2 | 0 |
| Noise σ=0.10 | **0.975** | 0.988 | 1 | 1 |
| Color jitter ±20% | 0.938 | 0.990 | 5 | 0 |
| Center crop 80% | 0.938 | 0.998 | 5 | 0 |

The run produced 1,200 predictions in 608 seconds on CPU. Across all conditions there were 104 errors involving only 17 unique images. One real image was falsely positive under all 15 conditions; the next two persisted under 14 and 13 conditions. Error concentration is therefore high.

Two of the most persistent real false positives were visually inspected. One is an older event photograph with a large website watermark and visible web compression; the other is a heavily staged product photograph. This is suggestive, not a statistically complete taxonomy.

The high AUC with poorer fixed-threshold BA under blur/resize means separation largely survives while the score distribution shifts. A future abstention/calibration layer is more justified than adding another uncalibrated detector.

## External generator slice

Thirty clean synthetic samples were retrieved from OpenAI's MIT-licensed [DALL·E 3 evaluation repository](https://github.com/openai/dalle3-eval-samples) at revision `d7c88c07b492ad7b9fd3003126d00719a2edabb1`.

| Generator | Detected / 10 | Mean p(AI) | Minimum p(AI) |
|---|---:|---:|---:|
| DALL·E 3 | 10 | 0.9999 | 0.9990 |
| Midjourney | 10 | 0.9999 | 0.9995 |
| SDXL | 9 | 0.8999 | 0.0013 |

The SDXL miss was an abstract ornamental typography image and was classified as real with extreme confidence. This slice is positive-only, tiny, and not proven disjoint from SigLIP's undocumented training set. Treat it as a source smoke test, not a generalization estimate.

## What works and what does not

Works:

- SigLIP's representation is the most consistent across the required transformation families.
- JPEG robustness is strong: all 40 AI images remained detected from Q90 through Q30.
- Noise robustness is unusually strong, including σ=0.10.
- DALL·E 3 and Midjourney recall was perfect in the external 10-image slices.
- The model is small enough for the track and practical CPU inference.

Does not work:

- Blur and severe resize cause many real false positives.
- Community Forensics, Steganograph, and most CNN baselines lose AI recall under heavy noise.
- CapCheck is badly calibrated on this sample.
- UnivFD is directionally uninformative here.
- Simple frequency, metadata, and JPEG-block heuristics do not generalize.
- Ensemble voting creates correlated failures rather than robust complementarity.
- The strong square/1024 heuristic exposes dataset construction bias.

## License and provenance register

| Asset | Stated license | Current use decision |
|---|---|---|
| SID_Set | CC BY 4.0 | Used; retain attribution and revision |
| OpenAI DALL·E 3 eval samples | MIT repository license | Used for positive-only smoke test |
| Ateeqq SigLIP | Apache-2.0 | Technical winner; training-image sources undocumented, clear before submission |
| Community Forensics | MIT | Usable baseline; poor JPEG/noise threshold robustness |
| Frontier Community Forensics | MIT | Promising; independently fine-tuned, source licenses are source-specific, confirm organizer suitability |
| Steganograph | MIT | Usable baseline |
| CapCheck | Apache-2.0 | Usable but weak |
| wkaandemir CLIP | MIT | Usable but weak on this sample |
| Divine trio | MIT | Usable but weak/mixed |
| xRayon ConvNeXtV2-Base | MIT | Not run: 1.05 GB training checkpoint; defer to GPU |
| Sentry | Restricted model/data terms | Excluded without written permission |
| Hussein detector | CC BY-NC 4.0 | Excluded from prize/submission candidate set |
| UMM detector | CC BY-ND | Excluded from modifiable submission candidate set |
| PatchCraft/NPR checkpoints | Unclear | Excluded until checkpoint licensing is explicit |

Model sources: [Ateeqq SigLIP](https://huggingface.co/Ateeqq/ai-vs-human-image-detector), [Community Forensics](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), [Frontier Community Forensics](https://huggingface.co/Thermostatic/community-forensics-frontier-detector-2026-08), [Steganograph](https://huggingface.co/delpot/steganograph-ia-detector), [CapCheck](https://huggingface.co/capcheck/ai-image-detection), [wkaandemir CLIP](https://huggingface.co/wkaandemir/ai-image-detector), [Divine ensemble](https://huggingface.co/divine2k/ai-image-detectors), and [xRayon ConvNeXt](https://huggingface.co/xRayon/convnext-ai-images-detector).

## Next experiment

Before fine-tuning, build one legal, lineage-controlled clean benchmark with separate real-source families and generator families. At minimum:

- COCO/Open Images/owned-camera real images, with per-image license records;
- DALL·E 3, Midjourney, SDXL, FLUX, GPT Image, and at least one held-out current generator;
- SHA-256 and perceptual-duplicate grouping before splitting;
- a calibration split separate from the final generator/source-disjoint test;
- SigLIP scores plus an abstain region for shifted/ambiguous inputs;
- the same 15-condition grid only after the clean source-disjoint test is frozen.

Do not fine-tune yet. The next decisive question is whether SigLIP's ranking survives genuine source and generator holdouts, not whether its SID_Set score can be improved.

## Reproducibility limits

- SID samples were selected deterministically as the first examples by class, not randomly or by source strata.
- SID lineage/source-family fields were not available in this spike, so hidden correlations are possible.
- The 24-image gate has wide statistical uncertainty.
- The 80-image confirmation remains a single dataset.
- The external generator slice has no real comparator and no proven training-set disjointness.
- Threshold sweeps in raw evaluator output are diagnostic only and were not used for reported fixed-threshold decisions.
