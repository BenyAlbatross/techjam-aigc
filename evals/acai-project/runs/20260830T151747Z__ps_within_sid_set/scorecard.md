# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9742 |
| auprc | 0.9794 |
| auprc_lift | 1.9587 |
| prevalence | 0.5000 |
| eer | 0.0775 |
| balanced_acc_50 | 0.9019 |
| tpr_at_fpr01 | 0.8115 |
| tpr_at_fpr05 | 0.8996 |
| brier | 0.0866 |
| ece | 0.0897 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.9976 | 0.9978 | 0.9591 |
| clean | 1500 | 0.9991 | 0.9991 | 0.9853 |
| color | 1500 | 0.9988 | 0.9989 | 0.9760 |
| crop | 1500 | 0.9963 | 0.9967 | 0.9573 |
| jpeg | 6000 | 0.9894 | 0.9903 | 0.8557 |
| noise | 4500 | 0.9278 | 0.9371 | 0.5133 |
| resize | 3000 | 0.9916 | 0.9923 | 0.8787 |

## 3. Robustness

- **clean_auroc**: 0.9991
- **transformed_auroc_mean**: 0.9726
- **clean_to_mean_drop**: 0.0264
- **clean_to_worst_drop**: 0.0713
- **worst family**: `noise` AUROC 0.9278
- **worst condition**: `noise:0.1` AUROC 0.8821

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| text_to_image | 11250 | 0.9742 |

- **worst generator**: `text_to_image` AUROC 0.9742

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9742 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0866, ECE 0.0897
- platt: Brier 0.0710, ECE 0.0241
- isotonic: Brier 0.0582, ECE 0.0118

## 10. Error cards

At the EER threshold 0.0026: 870 FP, 873 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
