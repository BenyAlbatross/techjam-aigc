# Findings

All numbers recomputable from `acai-project/runs/*.parquet`. Positive class = AI throughout.
Read [DEFINITIONS.md](DEFINITIONS.md) first — several dataset names are near-homonyms.

---

## 1. Both models on `data_draft` (clean, 10,000 images)

| population | n | TRACE-RX-M | Parallel fused | Parallel global | Parallel memory |
|---|---|---|---|---|---|
| SID-Set | 5,000 | **0.8549** | 0.8310 | 0.8313 | 0.7002 |
| WildFake | 5,000 | **0.5674** | 0.4633 | 0.4613 | 0.5796 |
| pooled | 10,000 | 0.6966 | 0.6333 | 0.6328 | 0.6272 |

Per generator (AI rows vs that source's reals):

| generator | TRACE-RX-M | Parallel fused | Parallel memory |
|---|---|---|---|
| `imagen` | 0.8464 | 0.6850 | 0.8180 |
| `ddim` | 0.6827 | 0.5194 | 0.7292 |
| `ddpm` | 0.5816 | 0.3905 | 0.5397 |
| `adm` | **0.3932** | **0.3552** | 0.3859 |
| `gan_based` | **0.3333** | **0.3662** | 0.4254 |
| `text_to_image` (SID) | 0.8549 | 0.8310 | 0.7002 |

**Both models are below chance on `adm` and `gan_based`** — independently, on a corpus neither
trained on. Not weakness; systematic inversion.

## 2. Both models on the eval subset (all 10 configs)

| config | n | prev | TRACE-RX-M | Parallel fused | Parallel memory | size-cheat |
|---|---|---|---|---|---|---|
| `default` | 13,841 | 0.639 | 0.5813 | **0.4260** | 0.5686 | **1.000** |
| `default_transformed` | 13,841 | 0.639 | 0.6025 | 0.4367 | 0.5625 | **1.000** |
| `normalized` | 13,841 | 0.639 | 0.5623 | 0.4183 | 0.5444 | — |
| `normalized_transformed` | 13,841 | 0.639 | 0.5327 | **0.3984** | 0.5238 | 0.502 |
| **`laion_matched`** | 7,652 | 0.500 | **0.7379** | 0.6909 | 0.6641 | — |
| `laion_matched_transformed` | 7,652 | 0.500 | 0.7343 | 0.6892 | 0.6569 | 0.503 |
| `cross_generator` | 5,494 | 0.727 | 0.6994 | 0.6889 | 0.6356 | — |
| `cross_generator_transformed` | 5,494 | 0.727 | 0.6926 | 0.6839 | 0.6317 | 0.508 |
| `diverse` | 14,394 | 0.500 | 0.6446 | 0.5223 | **0.6471** | — |
| `diverse_transformed` | 14,394 | 0.500 | 0.6213 | 0.5174 | 0.6450 | 0.504 |

**TRACE-RX-M beats Parallel on every config.** The parallel architecture is a regression here.

**Read `cross_generator`'s AUPRC carefully.** 0.847 looks strong; prevalence is 0.727, so the random
baseline *is* 0.727 and the lift is 1.16x. AUPRC is not comparable across these configs.

**Neither model uses the size shortcut, and that is to their credit.** The card's one-line cheat
scores 1.000 on `default`; the models get 0.581 and 0.426. My own DINOv3 probe scored **0.851** there
— the highest of anything tested, because it *was* riding the shortcut. Their `nuisance_max_auc: 0.55`
protocol worked.

## 3. Both models on `techjam2026` calibration, 15 official conditions

5,585 images x 15 conditions = **83,775 evaluations per model**. Prevalence 0.268.

| | TRACE-RX-M | Parallel fused |
|---|---|---|
| AUROC | 0.8737 | **0.8781** |
| AUPRC (lift) | 0.8573 (3.20x) | **0.8744 (3.27x)** |
| TPR@FPR1% | 0.6929 | **0.7607** |
| EER | 0.1910 | **0.1770** |
| clean -> worst drop | 0.0049 | 0.0036 |

Worst condition for both is `crop_0.8` — the only transform that removes content rather than
degrading it. Every other setting lands within 0.005 of clean; several score *above* it.

Per generator (clean rows):

| generator | TRACE-RX-M | Parallel | trained on? |
|---|---|---|---|
| `flux_1_schnell` | 0.9979 | **0.9993** | yes |
| `sdxl_1_0` | 0.9963 | **0.9982** | yes |
| `gpt_image_2` | 0.8966 | **0.9368** | yes |
| **`gemini_flash_image`** | **0.3357** | **0.2401** | **no — held out** |

My 0.3357 independently reproduces their own recorded 0.3279 (they measured on dev, I on
calibration), which is a useful check that this harness matches theirs.

## 4. Chains of 1–6 sequential transforms, uniform on length / family / order

### On calibration (5,585 x 6 = 33,510 per model)

| chain length | TRACE-RX-M | Parallel |
|---|---|---|
| 1 | 0.8732 | 0.8792 |
| 2 | 0.8732 | 0.8779 |
| 3 | 0.8726 | 0.8769 |
| 4 | 0.8714 | 0.8767 |
| 5 | 0.8700 | 0.8760 |
| 6 | 0.8684 | 0.8753 |
| **k=1 -> k=6** | **−0.0048** | **−0.0039** |

Per generator at k=1 -> k=6: the three trained generators are flat (<= 0.009 change); Gemini
degrades 6x faster under TRACE-RX-M (0.3363 -> 0.3068) and is flat-but-inverted under Parallel
(~0.247 throughout).

### On `data_draft` WildFake (5,000 x 6 = 30,000 per model) — everything unseen

| | TRACE-RX-M | Parallel fused |
|---|---|---|
| pooled AUROC | **0.5721** | 0.4730 |
| AUPRC (lift over 0.500) | 0.6163 (**1.23x**) | 0.5119 (**1.02x**) |
| TPR@FPR1% | 0.0646 | 0.0379 |
| k=1 -> k=6 | −0.0008 | **+0.0044** |

**Parallel's AUPRC lift of 1.02x is indistinguishable from random.**

Per generator, k=1 -> k=6:

| generator | TRACE-RX-M | Parallel |
|---|---|---|
| `imagen` | 0.8459 -> 0.8382 | 0.6758 -> 0.6479 |
| `ddim` | 0.6816 -> 0.6358 | 0.5105 -> 0.4547 |
| `ddpm` | 0.5785 -> 0.5652 | 0.3858 -> 0.3448 |
| `adm` | 0.3837 -> **0.3927** | 0.3742 -> **0.4701** |
| `gan_based` | 0.3662 -> **0.4199** | 0.4013 -> **0.4521** |

**`adm` and `gan_based` get BETTER with more transforms** (+0.096, +0.054). Degradation is removing
a cue that was actively misleading the models — evidence the failure is a false positive signal being
read, not a true signal being missed.

### The parallel model's fusion gate is broken off-distribution

| Parallel branch | WildFake chains | eval-subset `default` | eval-subset `diverse` |
|---|---|---|---|
| fused (shipped) | 0.4730 | 0.4260 | 0.5223 |
| global | 0.4709 | 0.4232 | 0.5202 |
| **memory** | **0.5960** | **0.5686** | **0.6471** |

Fused ~= global to three decimals on essentially every dataset — the learned gate is ignoring the
memory branch. On in-distribution data that is fine (global is better there). Off-distribution it is
a **0.12 AUROC self-inflicted loss**, and the memory branch alone beats both models' shipped outputs.

## 5. Why the calibration headline overstates things

The calibration split is 88% trained-generator AI images. So its 0.874 headline is mostly measuring
recognition of three memorised programs. Weight it toward unseen generators and it collapses. Always
report the per-generator split beside it.

## 6. The overall conclusion

| what breaks the models | cost |
|---|---|
| six stacked random transforms | <= 0.005 AUROC |
| one unseen generator | **0.66 AUROC** |

About **130x** difference. Transform robustness is solved on this data; generator generalisation is
not, and no amount of augmentation will fix it — some transforms make it *better*, which is the
opposite of an augmentation problem.
