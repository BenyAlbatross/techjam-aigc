# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.4133 |
| auprc | 0.4682 |
| auprc_lift | 0.9364 |
| prevalence | 0.5000 |
| eer | 0.5844 |
| balanced_acc_50 | 0.5056 |
| tpr_at_fpr01 | 0.0276 |
| tpr_at_fpr05 | 0.0569 |
| brier | 0.4831 |
| ece | 0.4794 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.4082 | 0.4734 | 0.0356 |
| clean | 1500 | 0.4188 | 0.4887 | 0.0427 |
| color | 1500 | 0.4293 | 0.4990 | 0.0493 |
| crop | 1500 | 0.3992 | 0.4600 | 0.0267 |
| jpeg | 6000 | 0.4095 | 0.4631 | 0.0187 |
| noise | 4500 | 0.4210 | 0.4517 | 0.0142 |
| resize | 3000 | 0.4145 | 0.4705 | 0.0347 |

## 3. Robustness

- **clean_auroc**: 0.4188
- **transformed_auroc_mean**: 0.4129
- **clean_to_mean_drop**: 0.0058
- **clean_to_worst_drop**: 0.0196
- **worst family**: `crop` AUROC 0.3992
- **worst condition**: `noise:0.02` AUROC 0.3880

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.3623 |
| ddim | 2250 | 0.3964 |
| ddpm | 2250 | 0.3482 |
| gan_based | 2250 | 0.3409 |
| imagen | 2250 | 0.6188 |

- **worst generator**: `gan_based` AUROC 0.3409

| source | n | AUROC |
|---|---|---|
| wildfake | 22500 | 0.4133 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.4831, ECE 0.4794
- platt: Brier 0.2498, ECE 0.0078
- isotonic: Brier 0.2487, ECE 0.0036

## 10. Error cards

At the EER threshold 0.0001: 6574 FP, 6575 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
