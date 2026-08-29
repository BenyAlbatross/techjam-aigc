# Spark A916 baseline evaluation

## Outcome

The frozen, no-fine-tuning Ateeqq SigLIP baseline was executed on `spark-a916` using its NVIDIA GB10 GPU. The complete evaluation took 37.99 seconds for 1,230 predictions: 80 SID images across 15 conditions plus 30 external generator images.

GPU evidence recorded by the run:

- PyTorch 2.13.0+cu130 and TorchVision 0.28.0+cu130;
- CUDA 13.0 available;
- NVIDIA GB10, compute capability 12.1;
- requested device `cuda:0`;
- a separate tensor probe executed on `cuda:0` before evaluation.

## SID robustness results

| Condition | Balanced accuracy | ROC-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|
| Clean | 0.9250 | 0.994375 | 6 | 0 |
| JPEG Q90 | 0.9000 | 0.993750 | 8 | 0 |
| JPEG Q70 | 0.9000 | 0.995000 | 8 | 0 |
| JPEG Q50 | 0.9250 | 0.996875 | 6 | 0 |
| JPEG Q30 | 0.9125 | 0.994375 | 7 | 0 |
| Blur sigma 0.5 | 0.9250 | 0.994375 | 6 | 0 |
| Blur sigma 1 | 0.9000 | 0.995000 | 8 | 0 |
| Blur sigma 2 | 0.8375 | 0.995000 | 13 | 0 |
| Resize 0.5x | 0.8875 | 0.995000 | 9 | 0 |
| Resize 0.25x | 0.8250 | 0.995000 | 14 | 0 |
| Noise sigma 0.02 | 0.9375 | 0.995000 | 5 | 0 |
| Noise sigma 0.05 | 0.9750 | 0.996875 | 2 | 0 |
| Noise sigma 0.10 | 0.9750 | 0.988125 | 1 | 1 |
| Color jitter 20% | 0.9375 | 0.989375 | 5 | 0 |
| Center crop 80% | 0.9375 | 0.997500 | 5 | 0 |

The main observed weakness remains false-positive growth after severe downscaling or blur. The quarter-size condition produced 14 real-image false positives and no synthetic-image misses.

## External positive-only smoke test

| Generator | Images | AI recall |
|---|---:|---:|
| DALL-E 3 | 10 | 1.0 |
| Midjourney | 10 | 1.0 |
| SDXL | 10 | 0.9 |

This slice has no matched real-image comparator and is too small to estimate generalization.

## Cross-platform replication

- All 1,200 SID binary decisions matched the local CPU run.
- All 30 external binary decisions matched the local CPU run.
- Fourteen of fifteen SID condition aggregates matched exactly.
- Color-jitter balanced accuracy and confusion counts matched, while ROC-AUC changed from 0.990000 locally to 0.989375 on the GB10.
- Maximum absolute probability difference was 0.00095775 on SID and 1.98e-9 on the external slice.

## Scope and provenance caution

No model weights were fine-tuned and no thresholds were fitted. SID_Set is recorded as CC BY 4.0; the external images come from OpenAI's MIT-licensed `dalle3-eval-samples` repository. The Ateeqq checkpoint is Apache-2.0, but its model card does not identify the claimed 120,000 training images. That unresolved training-data provenance remains a blocker for treating this checkpoint as a submission dependency, even though it is a useful technical baseline.
