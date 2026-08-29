# TechJam 2026 AIGC detection research

Research and zero-fine-tuning baselines for robust detection of fully AI-generated still images under distribution shift.

## Current result

Ateeqq SigLIP was the strongest tested open-source baseline. On the frozen 80-image SID_Set confirmation slice:

- clean balanced accuracy: **0.925**;
- clean ROC-AUC: **0.994375**;
- worst balanced accuracy: **0.825** after 0.25x downscaling;
- external positive-only recall: DALL-E 3 **10/10**, Midjourney **10/10**, SDXL **9/10**.

The same frozen evaluator ran on `spark-a916` using an NVIDIA GB10 and CUDA 13.0. It completed 1,230 predictions in 37.99 seconds. All binary decisions matched the local CPU reference; color-jitter ROC-AUC differed by 0.000625 while its balanced accuracy and confusion counts matched.

No checkpoint was fine-tuned and no threshold was fitted on the evaluation samples.

## Repository map

- `research/`: master execution plan and recovered architecture/research review;
- `work/baseline-spike/`: experiment harness, report packager, verification test, and per-model raw results;
- `outputs/`: published reports, combined result JSON, and runnable copies of experiment entry points;
- `outputs/spark-a916-evaluation/`: frozen cluster evaluator source, dependency pins, model configuration, and SID sample manifest;
- `outputs/spark-a916-results/`: raw GB10 results and local-versus-cluster comparison.

## Quick verification

The report/result consistency test uses only Python's standard library:

```bash
python work/baseline-spike/verify_outputs.py
```

With the baseline dependencies installed:

```bash
python outputs/spark-a916-evaluation/run.py self-test
python -m compileall -q outputs work/baseline-spike
```

## Reproducing experiments

Install baseline dependencies from `requirements-baseline.txt`. DGX Spark uses the separate CUDA 13 pins in `outputs/spark-a916-evaluation/requirements-cuda.txt`.

Large model weights and copied image datasets are intentionally excluded from Git. Restore these inputs before a full rerun:

- model: `Ateeqq/ai-vs-human-image-detector`, commit `60e82406916921b823616bee33397baab38af3f0`, Apache-2.0;
- SID_Set: `saberzl/SID_Set`, revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`, CC BY 4.0;
- external generator samples: `openai/dalle3-eval-samples`, revision `d7c88c07b492ad7b9fd3003126d00719a2edabb1`, MIT repository license.

The SID manifest records exact row selection and image hashes. Result JSON remains committed for auditability.

## Important limitation

The Ateeqq checkpoint is Apache-2.0, but its model card does not identify the claimed 120,000 training images. Resolve that training-data provenance before using the checkpoint as a final submission dependency. Current external generator samples are tiny positive-only smoke tests, not generalization estimates.
