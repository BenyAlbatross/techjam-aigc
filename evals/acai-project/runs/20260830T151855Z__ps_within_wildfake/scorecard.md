# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 22500  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9651 |
| auprc | 0.9716 |
| auprc_lift | 1.9432 |
| prevalence | 0.5000 |
| eer | 0.0943 |
| balanced_acc_50 | 0.9115 |
| tpr_at_fpr01 | 0.7607 |
| tpr_at_fpr05 | 0.8699 |
| brier | 0.0746 |
| ece | 0.0593 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 4500 | 0.9830 | 0.9860 | 0.8693 |
| clean | 1500 | 0.9877 | 0.9907 | 0.9067 |
| color | 1500 | 0.9874 | 0.9904 | 0.8933 |
| crop | 1500 | 0.9804 | 0.9848 | 0.8573 |
| jpeg | 6000 | 0.9593 | 0.9667 | 0.7240 |
| noise | 4500 | 0.9470 | 0.9573 | 0.6676 |
| resize | 3000 | 0.9624 | 0.9678 | 0.6800 |

## 3. Robustness

- **clean_auroc**: 0.9877
- **transformed_auroc_mean**: 0.9632
- **clean_to_mean_drop**: 0.0246
- **clean_to_worst_drop**: 0.0407
- **worst family**: `noise` AUROC 0.9470
- **worst condition**: `noise:0.1` AUROC 0.9210

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9711 |
| ddim | 2250 | 0.9789 |
| ddpm | 2250 | 0.9717 |
| gan_based | 2250 | 0.9358 |
| imagen | 2250 | 0.9680 |

- **worst generator**: `gan_based` AUROC 0.9358

| source | n | AUROC |
|---|---|---|
| wildfake | 22500 | 0.9651 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0746, ECE 0.0593
- platt: Brier 0.0781, ECE 0.0294
- isotonic: Brier 0.0722, ECE 0.0179

## 10. Error cards

At the EER threshold 0.2427: 1060 FP, 1062 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
