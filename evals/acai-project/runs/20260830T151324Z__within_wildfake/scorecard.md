# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9796 |
| auprc | 0.9839 |
| auprc_lift | 1.9679 |
| prevalence | 0.5000 |
| eer | 0.0637 |
| balanced_acc_50 | 0.9393 |
| tpr_at_fpr01 | 0.8401 |
| tpr_at_fpr05 | 0.9290 |
| brier | 0.0502 |
| ece | 0.0368 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.9860 | 0.9893 | 0.8933 |
| clean | 1500 | 0.9864 | 0.9897 | 0.8960 |
| color | 1500 | 0.9862 | 0.9895 | 0.8880 |
| crop | 1500 | 0.9797 | 0.9845 | 0.8747 |
| jpeg | 6000 | 0.9780 | 0.9825 | 0.8470 |
| noise | 4500 | 0.9684 | 0.9747 | 0.7831 |
| resize | 3000 | 0.9835 | 0.9872 | 0.8893 |

## 3. Robustness

- **clean_auroc**: 0.9864
- **transformed_auroc_mean**: 0.9790
- **clean_to_mean_drop**: 0.0073
- **clean_to_worst_drop**: 0.0180
- **worst family**: `noise` AUROC 0.9684
- **worst condition**: `noise:0.1` AUROC 0.9540

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9855 |
| ddim | 2250 | 0.9868 |
| ddpm | 2250 | 0.9795 |
| gan_based | 2250 | 0.9526 |
| imagen | 2250 | 0.9936 |

- **worst generator**: `gan_based` AUROC 0.9526

| source | n | AUROC |
|---|---|---|
| wildfake | 22500 | 0.9796 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0502, ECE 0.0368
- platt: Brier 0.0517, ECE 0.0214
- isotonic: Brier 0.0488, ECE 0.0141

## 10. Error cards

At the EER threshold 0.3358: 717 FP, 716 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
