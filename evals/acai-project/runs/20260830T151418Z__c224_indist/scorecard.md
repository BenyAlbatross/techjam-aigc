# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 45000  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9312 |
| auprc | 0.9465 |
| auprc_lift | 1.8931 |
| prevalence | 0.5000 |
| eer | 0.1369 |
| balanced_acc_50 | 0.8686 |
| tpr_at_fpr01 | 0.6462 |
| tpr_at_fpr05 | 0.7813 |
| brier | 0.1159 |
| ece | 0.1046 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 9000 | 0.9756 | 0.9795 | 0.7916 |
| clean | 3000 | 0.9889 | 0.9910 | 0.8980 |
| color | 3000 | 0.9886 | 0.9907 | 0.8907 |
| crop | 3000 | 0.9833 | 0.9864 | 0.8493 |
| jpeg | 12000 | 0.9355 | 0.9476 | 0.6268 |
| noise | 9000 | 0.8347 | 0.8661 | 0.3956 |
| resize | 6000 | 0.9506 | 0.9573 | 0.6433 |

## 3. Robustness

- **clean_auroc**: 0.9889
- **transformed_auroc_mean**: 0.9265
- **clean_to_mean_drop**: 0.0624
- **clean_to_worst_drop**: 0.1541
- **worst family**: `noise` AUROC 0.8347
- **worst condition**: `noise:0.1` AUROC 0.7750

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9614 |
| ddim | 2250 | 0.9450 |
| ddpm | 2250 | 0.9652 |
| gan_based | 2250 | 0.9373 |
| imagen | 2250 | 0.9706 |
| text_to_image | 11250 | 0.9065 |

- **worst generator**: `text_to_image` AUROC 0.9065

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9096 |
| wildfake | 22500 | 0.9569 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.1159, ECE 0.1046
- platt: Brier 0.1075, ECE 0.0173
- isotonic: Brier 0.0999, ECE 0.0127

## 10. Error cards

At the EER threshold 0.0374: 3079 FP, 3081 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
