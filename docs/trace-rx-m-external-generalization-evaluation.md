# TRACE-RX-M external generalization evaluation

## Outcome

The frozen TRACE-RX-M checkpoint does not generalize reliably to the sampled
WildFake and AIGIBench test distributions. Clean ROC-AUC is 0.5258 on
WildFake and 0.4822 on AIGIBench. At the existing logit-zero decision
threshold, it detects only 8.08% and 10.44% of fake images respectively while
correctly rejecting 97.60% and 96.56% of authentic images. The principal
failure is therefore unseen-generator generalization, not transformation
robustness: applying one to three transforms changes rankings and binary
predictions only modestly, but performance is already near chance on clean
external images.

This is a diagnostic evaluation, not a training or threshold-selection set.
Its labels must not be used to fit the final detector or calibrate its
threshold.

## Frozen protocol

The evaluation was run on 1 September 2026 with:

- checkpoint `s4_detector.pt`, SHA-256
  `f811c4641a644e1eaed30891f9c075932a1de8680dce812adaf03c9a7daaf25e`;
- memory `s3_memory.pt`, SHA-256
  `71f0bf9edeedc5af21de1b22e3705560a1b0e8adef0c1312266cfff3eb59a7a2`;
- uncalibrated logit threshold `0.0` and normalized pAUC at 5% maximum FPR;
- selection seed `20260831` and WildFake split-reconstruction seed `20260829`;
- 15,000 source images and 60,000 scored endpoints.

Every source image contributes exactly four paired endpoints: clean, one
deterministically assigned one-step transform, one two-step chain, and one
three-step chain. Assignment hashes the parent identity, seed, and chain
length, so it is reproducible and independent of the label. The one-step bank
contains all 15 official JPEG, blur, resize, noise, color-jitter, and crop
conditions. The two-step bank contains the 30 directed pairs of medium-strength
representatives from six transform families. The three-step bank contains ten
preregistered realistic compositions. Transforms are applied sequentially to
decoded source pixels before the model's 224-pixel preprocessing.

This assigned-chain design evaluates every parent at every requested chain
length without multiplying the run into every possible recipe. All 15
one-step, 30 two-step, and 10 three-step recipes are represented in the final
predictions.

## External data

| Dataset | Sources | Authentic | AIGC | Sampling |
| --- | ---: | ---: | ---: | --- |
| AIGIBench test | 5,000 | 2,500 | 2,500 | 100 authentic and 100 fake from each of 25 official test archives |
| WildFake reconstructed test | 10,000 | 5,000 | 5,000 | Near-equal authentic-source strata and exactly 200 fakes from each of 25 generator strata |

[AIGIBench](https://huggingface.co/datasets/HorizonTEL/AIGIBench) was pinned to revision
`e44ec40efe5117a5ccdaa6ff0e89ed934d03d310` and is licensed
CC-BY-NC-SA-4.0. [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
was read from its ModelScope `master` revision and is licensed Apache-2.0.
Because the full sources are very large, the preparation script reads remote
ZIP indexes and downloads only the selected members. The result is
6,072,240,296 source bytes (about 5.66 GiB), rather than downloading the
complete repositories.

WildFake documents a random 80/20 split within each stratum but does not
publish an exact membership file. The sample therefore reconstructs that
policy deterministically and is named `wildfake-reconstructed-test`; it must
not be represented as the authors' exact hidden test split. Every downloaded
file was decoded successfully, exact source-byte duplicates were removed, and
reserve candidates filled the original quotas. The final manifest has 15,000
unique parent IDs, paths, and content hashes. The CSV file SHA-256 is
`e671af6fd5191f1c6a7c02768f0fefc654c3d0d429c4197551ae7ad18214e36c`;
the metadata's ordered parent/source-content digest is
`fef626f0732f3f6c520722844fe3ae61f99793f0b0c8aff21edcc344873941a5`.

AIGIBench includes image manipulation and identity-conditioned generation
subsets as well as purely generated images. Its pooled result is useful as an
external stress test, but it does not exactly match the challenge's narrower
pure-generation target.

## Results by sequential-transform count

`TPR` and `TNR` use the frozen logit-zero threshold. Each row remains exactly
class balanced.

| Dataset | Steps | ROC-AUC | AP | n-pAUC@5% | Balanced accuracy | TPR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIGIBench | 0 (clean) | 0.4822 | 0.5484 | 0.0832 | 0.5350 | 0.1044 | 0.9656 |
| AIGIBench | 1 | 0.4897 | 0.5508 | 0.0835 | 0.5340 | 0.1004 | 0.9676 |
| AIGIBench | 2 | 0.4972 | 0.5540 | 0.0808 | 0.5342 | 0.0992 | 0.9692 |
| AIGIBench | 3 | 0.4945 | 0.5511 | 0.0792 | 0.5328 | 0.0976 | 0.9680 |
| WildFake | 0 (clean) | 0.5258 | 0.5704 | 0.0766 | 0.5284 | 0.0808 | 0.9760 |
| WildFake | 1 | 0.5351 | 0.5726 | 0.0748 | 0.5272 | 0.0766 | 0.9778 |
| WildFake | 2 | 0.5348 | 0.5731 | 0.0745 | 0.5238 | 0.0674 | 0.9802 |
| WildFake | 3 | 0.5384 | 0.5740 | 0.0728 | 0.5245 | 0.0726 | 0.9764 |

The clean-to-three-step n-pAUC change is -0.0039 on AIGIBench and -0.0038 on
WildFake. ROC-AUC instead increases slightly, which confirms that transformed
images are not the source of the dominant failure. The thresholded fake recall
does decline with longer chains, most clearly on WildFake's two-step endpoints.

## Paired transformation stability

| Dataset | Steps | Logit Pearson vs clean | Mean absolute logit drift | Prediction flip rate |
| --- | ---: | ---: | ---: | ---: |
| AIGIBench | 1 | 0.9885 | 0.3137 | 0.50% |
| AIGIBench | 2 | 0.9769 | 0.5145 | 1.16% |
| AIGIBench | 3 | 0.9817 | 0.4434 | 0.94% |
| WildFake | 1 | 0.9751 | 0.4607 | 0.98% |
| WildFake | 2 | 0.9595 | 0.6746 | 1.68% |
| WildFake | 3 | 0.9677 | 0.6041 | 1.43% |

All mean logit drifts are negative, so redistribution generally pushes scores
farther toward the authentic decision. Even so, the high correlations and low
flip rates show that the external rankings are mostly preserved.

## Generator-level diagnosis

The pooled averages hide severe generator reversals. On clean AIGIBench, 13 of
25 generator strata have ROC-AUC below 0.5. ProGAN is weakest at 0.1427,
followed by StyleGAN3 at 0.2008; both have zero fake recall at the frozen
threshold. CommunityAI (0.8662), SDXL (0.7827), IP-Adapter (0.7726), FLUX1-dev
(0.7697), and Imagen3 (0.7173) rank highest.

On clean WildFake, 10 of 25 generator strata are below 0.5. DF-GAN (0.1892),
BigGAN (0.1928), and VQDM (0.2392) are weakest. Imagen (0.8297), Stable
Diffusion with ControlNet (0.7736), original Stable Diffusion Advanced
(0.7442), personalized Stable Diffusion fine-tunes (0.7071), and Stable
Diffusion with LoRA (0.7044) are strongest. Even the stronger external families
usually have low thresholded recall, showing both ranking failures and a large
score-distribution shift toward negative logits.

Threshold recalibration could improve balanced accuracy, but it cannot repair
the below-chance generator rankings. The next model iteration should prioritize
generator-diverse training and generator-family holdouts, especially for GAN
families, while preserving these external labels as evaluation-only data.

## Reproduction and artifacts

Prepare the exact stratified source sample:

```bash
uv run --script scripts/prepare_external_evaluation.py \
  --output data/evaluation/wildfake-aigibench-stratified \
  --aigibench-count 5000 \
  --wildfake-count 10000 \
  --selection-seed 20260831 \
  --wildfake-split-seed 20260829 \
  --workers 8
```

Run the four-endpoint evaluation:

```bash
uv run python scripts/evaluate_trace_rx_m.py \
  --config configs/archive/trace-rx-m-dinov2-historical.json \
  --checkpoint artifacts/trace-rx-m-techjam2026/s4_detector.pt \
  --memory artifacts/trace-rx-m-techjam2026/s3_memory.pt \
  --dataset-spec configs/evaluation/aigibench-stratified-test.json \
  --dataset-spec configs/evaluation/wildfake-reconstructed-test.json \
  --assigned-chain-length 1 \
  --assigned-chain-length 2 \
  --assigned-chain-length 3 \
  --batch-size 128 \
  --workers 8 \
  --device cpu \
  --output artifacts/evaluation/trace-rx-m-wildfake-aigibench-stratified-sequential-1-3
```

The run used CPU because a separate, non-task GPU process occupied the local
device. It completed all 60,000 endpoints in 1 hour 23 minutes. Summaries can
be regenerated without inference:

```bash
uv run python scripts/summarize_external_evaluation.py \
  --predictions artifacts/evaluation/trace-rx-m-wildfake-aigibench-stratified-sequential-1-3/predictions.csv \
  --output artifacts/evaluation/trace-rx-m-wildfake-aigibench-stratified-sequential-1-3 \
  --threshold 0 \
  --max-fpr 0.05
```

The artifact directory contains all 60,000 predictions, run metadata,
condition and chain-length metrics, generator/source subgroup tables, paired
drift tables, and transform recipes. Source
selection metadata lives beside the ignored image manifest under
`data/evaluation/wildfake-aigibench-stratified/`.
