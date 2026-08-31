# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.6449 |
| auprc | 0.6842 |
| auprc_lift | 1.3683 |
| prevalence | 0.5000 |
| eer | 0.3975 |
| balanced_acc_50 | 0.6022 |
| tpr_at_fpr01 | 0.0996 |
| tpr_at_fpr05 | 0.2385 |
| brier | 0.3530 |
| ece | 0.3338 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.7611 | 0.7692 | 0.1147 |
| clean | 1500 | 0.8813 | 0.8786 | 0.2603 |
| color | 1500 | 0.8745 | 0.8721 | 0.2579 |
| crop | 1500 | 0.8626 | 0.8702 | 0.2567 |
| jpeg | 6000 | 0.6198 | 0.6314 | 0.0400 |
| noise | 4500 | 0.4094 | 0.4324 | 0.0009 |
| resize | 3000 | 0.6663 | 0.6699 | 0.0451 |

## 3. Robustness

- **clean_auroc**: 0.8813
- **transformed_auroc_mean**: 0.6314
- **clean_to_mean_drop**: 0.2498
- **clean_to_worst_drop**: 0.4719
- **worst family**: `noise` AUROC 0.4094
- **worst condition**: `noise:0.1` AUROC 0.3655

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| text_to_image | 11250 | 0.6449 |

- **worst generator**: `text_to_image` AUROC 0.6449

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.6449 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.3530, ECE 0.3338
- platt: Brier 0.2377, ECE 0.0453
- isotonic: Brier 0.2267, ECE 0.0188

## 10. Error cards

At the EER threshold 0.5215: 4472 FP, 4472 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
