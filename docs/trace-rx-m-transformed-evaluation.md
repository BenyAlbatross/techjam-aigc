# TRACE-RX-M official-transform evaluation

## Scope and provenance

The frozen shipping checkpoint was evaluated on 31 August 2026 against every
transformation and severity explicitly listed in the TechJam problem statement.
The run used all 6,091 development images (4,103 authentic and 1,988 AIGC) and
created one clean plus 15 deterministic transformed endpoints per parent, for
97,456 scored endpoints. No organizer demonstration-only or locked-evaluation
rows were used.

Checkpoint SHA-256:
`f811c4641a644e1eaed30891f9c075932a1de8680dce812adaf03c9a7daaf25e`.
The decision threshold was uncalibrated logit `0.0`; normalized pAUC uses a 5%
maximum false-positive rate. Full predictions, subgroup tables, transform
recipes, and artifact hashes are in
`artifacts/evaluation/trace-rx-m-techjam2026-development-official/`.

The available TechJam manifest points to neutralized 224-pixel BMP inputs.
Consequently, this run applies redistribution transforms to those decoded
pixels before model normalization. Future external evaluations should point to
licensed source-pixel files when available.

## Results

| Condition | ROC-AUC | Average precision | Normalized pAUC@5% | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.9035 | 0.9063 | 0.7794 | 0.8852 |
| JPEG Q90 | 0.9019 | 0.9051 | 0.7747 | 0.8858 |
| JPEG Q70 | 0.9017 | 0.9047 | 0.7724 | 0.8843 |
| JPEG Q50 | 0.9008 | 0.9041 | 0.7686 | 0.8849 |
| JPEG Q30 | 0.8996 | 0.9035 | 0.7668 | 0.8851 |
| Blur sigma 0.5 | 0.9032 | 0.9060 | 0.7779 | 0.8841 |
| Blur sigma 1.0 | 0.9031 | 0.9054 | 0.7745 | 0.8857 |
| Blur sigma 2.0 | 0.9036 | 0.9058 | 0.7711 | 0.8823 |
| Resize 0.5x | 0.9022 | 0.9057 | 0.7777 | 0.8848 |
| Resize 0.25x | 0.9003 | 0.9033 | 0.7655 | 0.8832 |
| Noise sigma 0.02 | 0.9011 | 0.9048 | 0.7738 | 0.8876 |
| Noise sigma 0.05 | 0.8978 | 0.9026 | 0.7646 | 0.8821 |
| Noise sigma 0.10 | 0.8936 | 0.8973 | 0.7461 | 0.8734 |
| Color jitter -20% | 0.9026 | 0.9057 | 0.7777 | 0.8850 |
| Color jitter +20% | 0.9039 | 0.9064 | 0.7763 | 0.8845 |
| Center crop 80% | 0.9005 | 0.9040 | 0.7739 | 0.8849 |

Across the 15 transformed conditions, macro ROC-AUC was 0.9011, macro average
precision 0.9043, macro normalized pAUC@5% 0.7708, and macro balanced accuracy
0.8839. Gaussian noise sigma 0.10 was worst by ROC-AUC. Relative to clean, it
reduced ROC-AUC by 0.0100, average precision by 0.0090, normalized pAUC by
0.0333, and balanced accuracy by 0.0118. It also caused the largest prediction
flip rate, 4.10%.

## Interpretation

The detector's pooled ranking is stable under the specified transformations;
no official condition reduces ROC-AUC by more than one point. This does not
resolve its generator-generalization failure. Clean Gemini Flash Image ROC-AUC
is 0.3279 and ranges only from 0.2828 to 0.3306 across conditions. At the logit
zero threshold, only 1.91% of clean Gemini images are detected, falling to
1.44% under noise sigma 0.10. By contrast, clean true-positive rates are 81.69%
for GPT Image 2, 99.16% for SDXL 1.0, and 100% for FLUX.1 Schnell.

The main model priority therefore remains cross-generator evidence, not further
tuning for the listed single transformations. Gaussian noise is the clearest
robustness weakness and deserves explicit final-training coverage because the
current run held that family out of training. Pooled transformation metrics
must continue to be reported alongside per-generator results so the easy
families cannot hide the Gemini reversal.
