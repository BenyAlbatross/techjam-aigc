# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 45000  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9608 |
| auprc | 0.9696 |
| auprc_lift | 1.9392 |
| prevalence | 0.5000 |
| eer | 0.0963 |
| balanced_acc_50 | 0.9063 |
| tpr_at_fpr01 | 0.7594 |
| tpr_at_fpr05 | 0.8663 |
| brier | 0.0818 |
| ece | 0.0732 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 9000 | 0.9867 | 0.9893 | 0.8798 |
| clean | 3000 | 0.9889 | 0.9910 | 0.8933 |
| color | 3000 | 0.9886 | 0.9907 | 0.8920 |
| crop | 3000 | 0.9852 | 0.9881 | 0.8820 |
| jpeg | 12000 | 0.9626 | 0.9695 | 0.7253 |
| noise | 9000 | 0.9048 | 0.9248 | 0.5562 |
| resize | 6000 | 0.9836 | 0.9865 | 0.8260 |

## 3. Robustness

- **clean_auroc**: 0.9889
- **transformed_auroc_mean**: 0.9588
- **clean_to_mean_drop**: 0.0300
- **clean_to_worst_drop**: 0.0841
- **worst family**: `noise` AUROC 0.9048
- **worst condition**: `noise:0.1` AUROC 0.8554

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9726 |
| ddim | 2250 | 0.9589 |
| ddpm | 2250 | 0.9758 |
| gan_based | 2250 | 0.9555 |
| imagen | 2250 | 0.9878 |
| text_to_image | 11250 | 0.9515 |

- **worst generator**: `text_to_image` AUROC 0.9515

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9556 |
| wildfake | 22500 | 0.9689 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0818, ECE 0.0732
- platt: Brier 0.0787, ECE 0.0244
- isotonic: Brier 0.0726, ECE 0.0105

## 10. Error cards

At the EER threshold 0.0384: 2168 FP, 2167 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
