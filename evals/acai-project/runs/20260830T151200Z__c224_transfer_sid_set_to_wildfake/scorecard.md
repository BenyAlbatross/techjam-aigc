# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.4851 |
| auprc | 0.4983 |
| auprc_lift | 0.9966 |
| prevalence | 0.5000 |
| eer | 0.5304 |
| balanced_acc_50 | 0.5084 |
| tpr_at_fpr01 | 0.0260 |
| tpr_at_fpr05 | 0.0603 |
| brier | 0.4841 |
| ece | 0.4805 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.4920 | 0.5099 | 0.0302 |
| clean | 1500 | 0.5077 | 0.5473 | 0.0533 |
| color | 1500 | 0.5114 | 0.5524 | 0.0573 |
| crop | 1500 | 0.4686 | 0.5010 | 0.0333 |
| jpeg | 6000 | 0.4764 | 0.4793 | 0.0153 |
| noise | 4500 | 0.4645 | 0.4560 | 0.0098 |
| resize | 3000 | 0.5075 | 0.5170 | 0.0320 |

## 3. Robustness

- **clean_auroc**: 0.5077
- **transformed_auroc_mean**: 0.4834
- **clean_to_mean_drop**: 0.0243
- **clean_to_worst_drop**: 0.0431
- **worst family**: `noise` AUROC 0.4645
- **worst condition**: `noise:0.02` AUROC 0.4535

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.4689 |
| ddim | 2250 | 0.4687 |
| ddpm | 2250 | 0.4406 |
| gan_based | 2250 | 0.4239 |
| imagen | 2250 | 0.6237 |

- **worst generator**: `gan_based` AUROC 0.4239

| source | n | AUROC |
|---|---|---|
| wildfake | 22500 | 0.4851 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.4841, ECE 0.4805
- platt: Brier 0.2490, ECE 0.0040
- isotonic: Brier 0.2462, ECE 0.0091

## 10. Error cards

At the EER threshold 0.0001: 5966 FP, 5967 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
