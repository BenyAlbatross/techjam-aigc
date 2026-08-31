# Scorecard  [SEALED]

Evaluator digest: `2effdf053d812fe2`  ·  n = 45000  ·  positive = label == 1 (ai_full)

## 1. Overall

| metric | value |
|---|---|
| auroc | 0.9601 |
| auprc | 0.9679 |
| auprc_lift | 1.9359 |
| prevalence | 0.5000 |
| eer | 0.1006 |
| balanced_acc_50 | 0.9034 |
| tpr_at_fpr01 | 0.7114 |
| tpr_at_fpr05 | 0.8571 |
| brier | 0.0831 |
| ece | 0.0720 |

## 2. Per transform

| family | n | AUROC | AUPRC | TPR@FPR1% |
|---|---|---|---|---|
| blur | 9000 | 0.9799 | 0.9840 | 0.8227 |
| clean | 3000 | 0.9819 | 0.9858 | 0.8507 |
| color | 3000 | 0.9818 | 0.9857 | 0.8453 |
| crop | 3000 | 0.9754 | 0.9803 | 0.7933 |
| jpeg | 12000 | 0.9598 | 0.9664 | 0.6710 |
| noise | 9000 | 0.9142 | 0.9286 | 0.5144 |
| resize | 6000 | 0.9767 | 0.9812 | 0.7930 |

## 3. Robustness

- **clean_auroc**: 0.9819
- **transformed_auroc_mean**: 0.9586
- **clean_to_mean_drop**: 0.0233
- **clean_to_worst_drop**: 0.0676
- **worst family**: `noise` AUROC 0.9142
- **worst condition**: `noise:0.1` AUROC 0.8819

## 4. Generalisation

| generator | n AI | AUROC (vs shared reals) |
|---|---|---|
| adm | 2250 | 0.9547 |
| ddim | 2250 | 0.9524 |
| ddpm | 2250 | 0.9627 |
| gan_based | 2250 | 0.9337 |
| imagen | 2250 | 0.9891 |
| text_to_image | 11250 | 0.9617 |

- **worst generator**: `gan_based` AUROC 0.9337

| source | n | AUROC |
|---|---|---|
| sid_set | 22500 | 0.9595 |
| wildfake | 22500 | 0.9613 |

> SID AI images have no generator identity (lumped 'text_to_image'); only the 5 named WildFake generators support leave-one-generator-out

> only 2 source datasets exist, so unseen-source is a 2-way test, not a sweep

## 6. Calibration

- raw: Brier 0.0831, ECE 0.0720
- platt: Brier 0.0815, ECE 0.0187
- isotonic: Brier 0.0747, ECE 0.0118

## 10. Error cards

At the EER threshold 0.0894: 2264 FP, 2265 FN.

## Excluded sections

- **5_worst_authentic_subtype_fpr** — excluded from this build by decision 2026-08-30
- **7_latency_memory** — excluded from this build by decision 2026-08-30
- **8_expert_error_correlation** — NOT COMPUTABLE: the dataset carries no human annotations (no annotator/rater/vote fields; label_confidence is the constant 0.8; labels are path-derived)
- **9_master_lineage_bootstrap** — excluded from this build; also degenerate today -- every lineage_id in data_draft is 1:1 with an asset, so a lineage bootstrap equals a row bootstrap until transform variants share a lineage
