# TRACE-RX-M v2 on wildfake-eval-subset

Score = `sigmoid(logit)`; positive class is AIGC. Threshold 0.0 on the logit.
AUROC CIs are 2000-resample percentile bootstrap.

`bal.acc @0` is the shipped operating point; `bal.acc @EER` is the best this model’s
ranking could do if the threshold were re-fitted on this data. The gap between them,
and `frac logit>0`, are the calibration story -- not the discrimination story.

| config | n | prev | AUROC | 95% CI | bal.acc @0 | bal.acc @EER | EER | TPR@FPR1% | TPR@FPR5% | frac logit>0 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `default` | 13841 | 0.639 | **0.5813** | 0.5682–0.5942 | 0.5272 | 0.5518 | 0.4482 | 0.0355 | 0.0952 | 0.124 |
| `default_transformed` | 13841 | 0.639 | **0.6025** | 0.5900–0.6151 | 0.5291 | 0.5696 | 0.4304 | 0.0336 | 0.1026 | 0.117 |
| `normalized` | 13841 | 0.639 | **0.5623** | 0.5490–0.5759 | 0.5226 | 0.5381 | 0.4619 | 0.0326 | 0.0931 | 0.119 |
| `normalized_transformed` | 13841 | 0.639 | **0.5327** | 0.5196–0.5457 | 0.5168 | 0.5168 | 0.4832 | 0.0280 | 0.0821 | 0.099 |
| `laion_matched` | 7652 | 0.500 | **0.7379** | 0.7251–0.7506 | 0.5545 | 0.6724 | 0.3276 | 0.0619 | 0.1793 | 0.095 |
| `laion_matched_transformed` | 7652 | 0.500 | **0.7343** | 0.7214–0.7471 | 0.5473 | 0.6681 | 0.3319 | 0.0604 | 0.1660 | 0.087 |
| `cross_generator` | 5494 | 0.727 | **0.6994** | 0.6833–0.7150 | 0.5468 | 0.6414 | 0.3586 | 0.0611 | 0.1627 | 0.103 |
| `cross_generator_transformed` | 5494 | 0.727 | **0.6926** | 0.6764–0.7086 | 0.5447 | 0.6393 | 0.3607 | 0.0531 | 0.1710 | 0.094 |
| `diverse` | 14394 | 0.500 | **0.6446** | 0.6349–0.6535 | 0.5579 | 0.6060 | 0.3940 | 0.0872 | 0.2191 | 0.082 |
| `diverse_transformed` | 14394 | 0.500 | **0.6213** | 0.6115–0.6306 | 0.5505 | 0.5883 | 0.4117 | 0.0794 | 0.2034 | 0.072 |

## Per-generator — `default`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 8843 | 0.5813 | 0.5272 | 0.0952 |

## Per-generator — `default_transformed`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 8843 | 0.6025 | 0.5291 | 0.1026 |

## Per-generator — `normalized`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 8843 | 0.5623 | 0.5226 | 0.0931 |

## Per-generator — `normalized_transformed`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 8843 | 0.5327 | 0.5168 | 0.0821 |

## Per-generator — `laion_matched`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 3826 | 0.7379 | 0.5545 | 0.1793 |

## Per-generator — `laion_matched_transformed`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| dalle3_advanced | 3826 | 0.7343 | 0.5473 | 0.1660 |

## Per-generator — `cross_generator`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| sdxl | 1000 | 0.7609 | 0.5713 | 0.2280 |
| dalle3 | 1000 | 0.7244 | 0.5488 | 0.1690 |
| midjourney_v5 | 999 | 0.6654 | 0.5429 | 0.1391 |
| gigagan | 995 | 0.6467 | 0.5240 | 0.1146 |

## Per-generator — `cross_generator_transformed`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| sdxl | 1000 | 0.7556 | 0.5658 | 0.2320 |
| dalle3 | 1000 | 0.7081 | 0.5428 | 0.1700 |
| gigagan | 995 | 0.6669 | 0.5326 | 0.1437 |
| midjourney_v5 | 999 | 0.6396 | 0.5374 | 0.1381 |

## Per-generator — `diverse`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| sdxl | 1800 | 0.7320 | 0.5908 | 0.3133 |
| dalle3 | 1800 | 0.6715 | 0.5560 | 0.2172 |
| midjourney_v5 | 1800 | 0.5957 | 0.5549 | 0.1917 |
| gigagan | 1794 | 0.5788 | 0.5298 | 0.1538 |

## Per-generator — `diverse_transformed`

| generator | n fake | AUROC | bal.acc | TPR@FPR5% |
|---|---:|---:|---:|---:|
| sdxl | 1800 | 0.7121 | 0.5827 | 0.2911 |
| dalle3 | 1800 | 0.6329 | 0.5452 | 0.1906 |
| gigagan | 1794 | 0.5932 | 0.5326 | 0.1689 |
| midjourney_v5 | 1800 | 0.5467 | 0.5416 | 0.1628 |

## Robustness — `default_transformed` (single-transform rows)

| setting | n | AUROC |
|---|---:|---:|
| blur_2.0 | 705 | 0.5266 |
| resize_0.5 | 682 | 0.5711 |
| blur_1.0 | 692 | 0.5733 |
| jpeg_q50 | 662 | 0.5755 |
| jpeg_q70 | 694 | 0.5768 |
| noise_0.02 | 690 | 0.5871 |
| noise_0.05 | 711 | 0.6035 |
| blur_0.5 | 709 | 0.6145 |
| jpeg_q30 | 699 | 0.6203 |
| jitter_0.2 | 692 | 0.6277 |
| jpeg_q90 | 680 | 0.6318 |
| resize_0.25 | 693 | 0.6358 |
| crop_0.8 | 689 | 0.6399 |
| noise_0.1 | 714 | 0.6587 |

One transform: AUROC 0.6025 (n=9712) · two composed: 0.6023 (n=4129).

## Robustness — `normalized_transformed` (single-transform rows)

| setting | n | AUROC |
|---|---:|---:|
| blur_2.0 | 705 | 0.4603 |
| resize_0.5 | 682 | 0.5146 |
| noise_0.05 | 711 | 0.5197 |
| blur_1.0 | 692 | 0.5199 |
| resize_0.25 | 693 | 0.5236 |
| noise_0.1 | 714 | 0.5297 |
| jpeg_q50 | 662 | 0.5342 |
| noise_0.02 | 690 | 0.5367 |
| jpeg_q70 | 694 | 0.5378 |
| jpeg_q30 | 699 | 0.5661 |
| blur_0.5 | 709 | 0.5823 |
| jpeg_q90 | 680 | 0.5963 |
| crop_0.8 | 689 | 0.6005 |
| jitter_0.2 | 692 | 0.6061 |

One transform: AUROC 0.5435 (n=9712) · two composed: 0.5067 (n=4129).

## Robustness — `laion_matched_transformed` (single-transform rows)

| setting | n | AUROC |
|---|---:|---:|
| blur_2.0 | 384 | 0.7031 |
| noise_0.1 | 396 | 0.7098 |
| jitter_0.2 | 384 | 0.7239 |
| crop_0.8 | 357 | 0.7289 |
| jpeg_q90 | 369 | 0.7307 |
| resize_0.25 | 388 | 0.7326 |
| blur_0.5 | 387 | 0.7349 |
| jpeg_q30 | 371 | 0.7431 |
| resize_0.5 | 376 | 0.7434 |
| blur_1.0 | 392 | 0.7450 |
| jpeg_q70 | 384 | 0.7499 |
| noise_0.02 | 376 | 0.7503 |
| noise_0.05 | 406 | 0.7507 |
| jpeg_q50 | 366 | 0.7512 |

One transform: AUROC 0.7343 (n=5336) · two composed: 0.7346 (n=2316).

## Robustness — `cross_generator_transformed` (single-transform rows)

| setting | n | AUROC |
|---|---:|---:|
| noise_0.02 | 268 | 0.6372 |
| crop_0.8 | 253 | 0.6401 |
| jpeg_q50 | 265 | 0.6656 |
| jpeg_q30 | 265 | 0.6682 |
| blur_0.5 | 278 | 0.6700 |
| jitter_0.2 | 283 | 0.6818 |
| noise_0.05 | 287 | 0.6873 |
| blur_2.0 | 282 | 0.6907 |
| noise_0.1 | 285 | 0.6949 |
| resize_0.5 | 266 | 0.6969 |
| blur_1.0 | 287 | 0.7034 |
| resize_0.25 | 273 | 0.7248 |
| jpeg_q70 | 276 | 0.7275 |
| jpeg_q90 | 265 | 0.7806 |

One transform: AUROC 0.6916 (n=3833) · two composed: 0.6962 (n=1661).

## Robustness — `diverse_transformed` (single-transform rows)

| setting | n | AUROC |
|---|---:|---:|
| resize_0.5 | 709 | 0.5743 |
| blur_1.0 | 715 | 0.5927 |
| resize_0.25 | 717 | 0.5987 |
| noise_0.1 | 745 | 0.6014 |
| jpeg_q90 | 708 | 0.6104 |
| noise_0.05 | 739 | 0.6130 |
| blur_2.0 | 735 | 0.6181 |
| jpeg_q50 | 692 | 0.6355 |
| jpeg_q30 | 725 | 0.6360 |
| noise_0.02 | 719 | 0.6438 |
| jpeg_q70 | 728 | 0.6544 |
| blur_0.5 | 738 | 0.6569 |
| crop_0.8 | 719 | 0.6636 |
| jitter_0.2 | 718 | 0.6710 |

One transform: AUROC 0.6254 (n=10107) · two composed: 0.6108 (n=4287).
