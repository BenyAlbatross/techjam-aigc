# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9960 |
| auprc | 0.9964 |
| auprc_lift | 1.9927 |
| prevalence | 0.5000 |
| eer | 0.0322 |
| balanced_acc_50 | 0.9608 |
| tpr_at_fpr01 | 0.9404 |
| tpr_at_fpr05 | 0.9791 |
| brier | 0.0325 |
| ece | 0.0330 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.9996 | 0.9996 | 0.9920 |
| clean | 1500 | 0.9998 | 0.9998 | 0.9973 |
| color | 1500 | 0.9998 | 0.9998 | 0.9973 |
| crop | 1500 | 0.9993 | 0.9993 | 0.9853 |
| jpeg | 6000 | 0.9965 | 0.9967 | 0.9310 |
| noise | 4500 | 0.9868 | 0.9878 | 0.8231 |
| resize | 3000 | 0.9994 | 0.9994 | 0.9853 |

## 3. Robustness

- **clean_auroc**: 0.9998
- **transformed_auroc_mean**: 0.9957
- **clean_to_mean_drop**: 0.0041
- **clean_to_worst_drop**: 0.0130
- **worst family**: `noise` AUROC 0.9868
- **worst condition**: `noise:0.1` AUROC 0.9785

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| text_to_image | 11250 | 0.9960 |

- **worst generator**: `text_to_image` AUROC 0.9960

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9960 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0325, ECE 0.0330
- platt: Brier 0.0302, ECE 0.0198
- isotonic: Brier 0.0230, ECE 0.0093

## 10. Error cards

At the EER threshold 0.0357: 362 FP, 363 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
