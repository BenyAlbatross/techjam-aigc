# TechJam 2026 local training run

## Dataset and split policy

The run pins public dataset `Joshyxwa/techjam2026` at commit
`fd6ff453e8214359423c8ab8e44150b2660ce36c`. Its supplied grouped split is
authoritative:

| Source split | TRACE-RX-M use | Rows |
| --- | --- | ---: |
| `train` | supervised fitting plus authentic-only internal pools | 32,035 |
| `dev` | held-out family validity check and error analysis | 6,091 |
| `calibration` | score calibration only | 5,585 |
| `own_locked` | final locked evaluation only | 960 |

Only real, real-only content groups from `train` are subdivided: 256 rows for
authentic memory, 256 for memory-capacity validation, and 512 for the
authentic-null reliability pool. This leaves 31,011 supervised rows (18,153
real and 12,858 AIGC). The development, calibration, and locked assignments are
not modified. Gemini Flash Image is the predeclared held-out generator family;
its 1,180 supervised rows are excluded from gradient updates and its 209 dev
rows are consulted once after fitting.

All AIGC source files are PNG while the authentic class mixes 9,710 JPEGs into
the training partition, and source dimensions also differ by class. The
preparation script decodes both classes identically, bilinearly resizes to
224x224 RGB, and writes fixed-size uncompressed BMP inputs. The S0 grouped
nuisance probes are therefore run on the actual model inputs and report 0.50
ROC-AUC for dimension, encoded-size metadata, and codec.

## Local training profile

- Host: NVIDIA DGX Spark, one GB10 GPU
- Backbone: Apache-2.0 `facebook/dinov2-base` at immutable revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- Trainable adaptation: rank-4 LoRA on attention query/value projections plus
  TRACE-RX-M evidence and classification heads
- Precision: CUDA BF16 autocast for S4; FP32 prototype retrieval/scoring
- Batch size: 512, class-balanced with native positives rotated across the
  three non-held-out generator families
- Schedule: 5 detector epochs, AdamW, short warm-up, cosine decay
- Tracking/publication: W&B offline; checkpoints retained locally
- Configuration: `configs/trace-rx-m-techjam2026-local.json`

The mixed-precision preflight observed BF16 attention-projection activations
and nonzero gradients in all 24 query/value LoRA pairs (48 LoRA tensors).

## Stage results

S0 passed. S1 cached two tensors of shape `[256, 256, 768]` in BF16. S3 selected
an authentic memory of 64 prototypes with top-k 16; the tested grid was
64/128/256 prototypes with top-k 4/8/16.

The five-epoch LoRA run reached training total loss 0.09798, but failed the
predeclared one-time unseen-Gemini gate (ROC-AUC 0.33056 versus the required
0.60). The prescribed fallback discarded the adapters and trained only the
heads for another five epochs. Its final loss was 0.10533 and held-out Gemini
ROC-AUC was 0.32793, so it also failed the gate. The frozen fallback is still
the shipping variant because that choice was declared before observing the
held-out result.

| Development slice | Positives | ROC-AUC |
| --- | ---: | ---: |
| All dev | 1,988 | 0.90351 |
| FLUX.1 Schnell vs all real | 412 | 0.99930 |
| Gemini Flash Image vs all real | 209 | 0.32793 |
| GPT Image 2 vs all real | 770 | 0.93544 |
| SDXL 1.0 vs all real | 597 | 0.99772 |

Full-dev average precision is 0.90632 and accuracy at an uncalibrated 0.5
threshold is 0.90658. These aggregates should not obscure the severe Gemini
reversal. At the final epoch, SID-Set and MIRFlickr remained the hardest real
subtypes (mean BCE 0.17734 and 0.16582 respectively).

After model selection was closed, `own_locked` was evaluated exactly once. It
contains 565 Gemini positives and 395 SID-Set real images. ROC-AUC is 0.66224,
average precision is 0.78601, and uncalibrated 0.5-threshold accuracy is
0.59167. The large gap between dev-Gemini and locked-Gemini confirms that
generator name alone does not define a stable distribution.

The shipping checkpoint is
`artifacts/trace-rx-m-techjam2026/s4_detector.pt`, SHA-256
`f811c4641a644e1eaed30891f9c075932a1de8680dce812adaf03c9a7daaf25e`.
Local score tables and slice metrics are retained beside it.

S5 reliability and S6 calibration are not claimed for this run. The canonical
dataset rows contain only clean endpoints, while S5 requires predeclared clean
and transformed endpoint groups including a held-out transform family. Fitting
a reliability table from fabricated endpoint labels after seeing locked results
would violate the protocol. The supplied `calibration` split remains untouched
for a future run that generates those endpoints reproducibly before training.
