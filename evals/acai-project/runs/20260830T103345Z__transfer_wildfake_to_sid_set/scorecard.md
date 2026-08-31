# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.6733 |
| auprc | 0.6977 |
| auprc_lift | 1.3955 |
| prevalence | 0.5000 |
| eer | 0.3745 |
| balanced_acc_50 | 0.6107 |
| tpr_at_fpr01 | 0.0803 |
| tpr_at_fpr05 | 0.2365 |
| brier | 0.3458 |
| ece | 0.3251 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.7801 | 0.7778 | 0.0989 |
| clean | 1500 | 0.8118 | 0.8089 | 0.1238 |
| color | 1500 | 0.8079 | 0.8037 | 0.1257 |
| crop | 1500 | 0.7957 | 0.8032 | 0.1553 |
| jpeg | 6000 | 0.6238 | 0.6384 | 0.0463 |
| noise | 4500 | 0.4862 | 0.5037 | 0.0129 |
| resize | 3000 | 0.7606 | 0.7563 | 0.0889 |

## 3. Robustness

- **clean_auroc**: 0.8118
- **transformed_auroc_mean**: 0.6645
- **clean_to_mean_drop**: 0.1473
- **clean_to_worst_drop**: 0.3256
- **worst family**: `noise` AUROC 0.4862
- **worst condition**: `noise:0.1` AUROC 0.3732

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| text_to_image | 11250 | 0.6733 |

- **worst generator**: `text_to_image` AUROC 0.6733

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.6733 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.3458, ECE 0.3251
- platt: Brier 0.2354, ECE 0.0521
- isotonic: Brier 0.2247, ECE 0.0168

## 10. Error cards

At the EER threshold 0.8884: 4213 FP, 4213 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
