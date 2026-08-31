# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 45000  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9658 |
| auprc | 0.9730 |
| auprc_lift | 1.9461 |
| prevalence | 0.5000 |
| eer | 0.0901 |
| balanced_acc_50 | 0.9119 |
| tpr_at_fpr01 | 0.7693 |
| tpr_at_fpr05 | 0.8796 |
| brier | 0.0764 |
| ece | 0.0687 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 9000 | 0.9863 | 0.9889 | 0.8764 |
| clean | 3000 | 0.9877 | 0.9901 | 0.8940 |
| color | 3000 | 0.9875 | 0.9899 | 0.8933 |
| crop | 3000 | 0.9820 | 0.9854 | 0.8427 |
| jpeg | 12000 | 0.9673 | 0.9730 | 0.7400 |
| noise | 9000 | 0.9168 | 0.9332 | 0.5958 |
| resize | 6000 | 0.9837 | 0.9867 | 0.8383 |

## 3. Robustness

- **clean_auroc**: 0.9877
- **transformed_auroc_mean**: 0.9643
- **clean_to_mean_drop**: 0.0234
- **clean_to_worst_drop**: 0.0709
- **worst family**: `noise` AUROC 0.9168
- **worst condition**: `noise:0.1` AUROC 0.8768

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9710 |
| ddim | 2250 | 0.9584 |
| ddpm | 2250 | 0.9736 |
| gan_based | 2250 | 0.9522 |
| imagen | 2250 | 0.9915 |
| text_to_image | 11250 | 0.9623 |

- **worst generator**: `gan_based` AUROC 0.9522

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9637 |
| wildfake | 22500 | 0.9688 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0764, ECE 0.0687
- platt: Brier 0.0744, ECE 0.0229
- isotonic: Brier 0.0675, ECE 0.0094

## 10. Error cards

At the EER threshold 0.0460: 2027 FP, 2027 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
