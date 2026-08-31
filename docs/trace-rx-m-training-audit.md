# TRACE-RX-M training audit and visual error analysis

## Verdict

The recorded run is internally consistent: the authentic prototype memory was
fitted only on real training images, the LoRA adapters were present and updated,
and the predeclared Gemini holdout was excluded from gradient updates. The main
failure is cross-distribution generalization, not a swapped label or an unused
LoRA optimizer group.

The shipping checkpoint is nevertheless **not the LoRA detector**. The trained
LoRA variant scored 0.33056 ROC-AUC on the one-time held-out Gemini development
slice, below the configured 0.60 validity gate. The pipeline consequently
discarded that model for deployment and retrained the heads with a completely
frozen encoder. `s4_detector.pt` is this frozen fallback; its held-out Gemini
ROC-AUC is 0.32793. In other words, the fallback was slightly worse than LoRA
on the gate slice. It shipped because the protocol prescribed "use frozen when
LoRA fails," not because the frozen variant won a comparison.

## Evidence checked

| Question | Evidence | Assessment |
| --- | --- | --- |
| Was the authentic memory trained on real data? | `s1_memory_pool.pt` contains 256 unique images of shape `[256, 256, 768]`; every corresponding manifest row has target 0 and role `memory_pool`. The cache, memory, and detector carry the same manifest SHA-256. | Yes. |
| Was the prototype actually optimized? | The selected 64-prototype memory records three epochs. Reconstruction loss fell from 0.84042 to 0.82027 and the diversity term fell from 7.84088 to 2.61444. | Yes. |
| Was the memory kept frozen during detector training? | The S4 implementation rejects a trainable memory, excludes it from optimizer groups, and the S4 artifacts reference the unchanged S3 memory hash. | Yes. |
| Was Gemini excluded from fitting? | The S4 code removes `gemini_flash_image` supervised rows before constructing the sampler and retains only real plus Gemini development rows for the one-time gate. | Yes. |
| Did LoRA train? | The LoRA final artifact contains 48 LoRA tensors (24 query/value pairs). All 24 B matrices are nonzero; their norms range from 0.02944 to 0.08567. LoRA total loss fell from 0.66742 to 0.09798. | Yes. |
| Is LoRA used by the shipping model? | `s4_detector.pt` declares `encoder_mode: frozen`, contains 13 head/memory tensors and no LoRA tensors. Its SHA-256 matches the frozen final checkpoint. | No. This is the intended fallback path. |

There is one global detector in either variant. LoRA, when enabled, adapts the
DINO encoder inside that detector; it is not an auxiliary classifier. The LoRA
artifact remains available at `checkpoints/lora/final_detector.pt`, but the
standard score-export and inference loading path reads `s4_detector.pt` and
therefore reconstructs DINO with rank zero.

## Why “real-only” is not yet a strong authentic prototype

The prototype construction is valid but its coverage is weak. The 256-image
memory pool is selected by a seeded global shuffle, not stratified by authentic
source. It contains 148 SID-Set images and 63 MIRFlickr images, while Blender
Open Productions is absent and Dollar Street and Wikimedia Commons contribute
one image each. This makes the learned authentic manifold heavily dependent on
two sources even though all inputs are genuinely real.

The capacity criterion also saturates. It derives one tail threshold from the
first 64-prototype/top-4 candidate. Several larger-top-k candidates then report
zero patches beyond that common threshold. The selector chooses the smallest
zero-tail candidate (64/top-16) even though its mean reconstruction error
(0.82705) is worse than 64/top-8 (0.78965). This is deterministic and consistent
with the implemented rule, but zero tail coverage should not be interpreted as
proof that the authentic manifold is complete.

## Observed failure pattern

| Evaluation slice | TN | FP | FN | TP | Accuracy at 0.5 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development, all families | 3,885 | 218 | 351 | 1,637 | 90.66% | 0.90351 |
| Development, Gemini positives only versus all real | — | — | 205 of 209 Gemini images | 4 of 209 Gemini images | 1.91% Gemini recall | 0.32793 |
| Locked, Gemini versus SID-Set | 358 | 37 | 355 | 210 | 59.17% | 0.66224 |

The model is close to perfect on FLUX and SDXL but assigns a mean AIGC
probability of only 0.022 to development Gemini. The locked Gemini distribution
is different enough to improve mean probability to 0.392 and AUC to 0.662, but
most positives still fall below 0.5. Generator name therefore does not define a
stable distribution, and the overall development score hides a severe family
failure.

Other limitations matter when interpreting the result:

- only four generator families are present, and only three contribute positive
  training gradients after the Gemini holdout;
- no DDA examples exist in this manifest, so the DDA pair loss is exactly zero
  for every epoch;
- the authentic memory is learned from clean normalized endpoints and remains
  fixed while only the detector stage sees redistribution augmentations; and
- S5 reliability and S6 calibration were not fitted, so 0.5 is an uncalibrated
  diagnostic threshold rather than an established operating point.

## Reproduce the visual report

The HTML generator embeds thumbnails so the output is self-contained and keeps
licensed/local images under the ignored `artifacts/` directory:

```bash
.venv/bin/python scripts/generate_error_analysis_report.py \
  --scores artifacts/trace-rx-m-techjam2026/locked-scores.csv \
  --output artifacts/trace-rx-m-techjam2026/locked-error-analysis.html \
  --title "TRACE-RX-M unseen locked evaluation — visual error analysis"
```

The report shows confusion counts, score histograms, per-source and
per-generator summaries, then separate galleries for incorrect and correct
predictions. Filters support false positives, false negatives, truth label,
generator family, source dataset, asset ID, model, and path. Images are the
normalized model-facing inputs referenced by the score CSV.
