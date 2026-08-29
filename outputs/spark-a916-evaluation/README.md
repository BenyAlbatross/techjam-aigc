# Spark A916 evaluation bundle

Frozen, no-fine-tuning replication bundle for the Ateeqq SigLIP baseline.

Contents:

- local Apache-2.0 model snapshot;
- 40 real and 40 fully synthetic SID_Set validation images, CC BY 4.0;
- 10 each DALL·E 3, Midjourney, and SDXL images from OpenAI's MIT-licensed evaluation repository;
- exact 15-condition robustness evaluator;
- CUDA verification probe.

## Git snapshot versus frozen bundle

The original frozen bundle included `model/model.safetensors` and all 110 evaluation images. Normal Git excludes those large copied binaries. This repository retains evaluator source, pinned dependencies, model configuration, exact SID selection/hashes, raw results, and reports.

Restore the following before running the full evaluator:

- `model/model.safetensors` from `Ateeqq/ai-vs-human-image-detector` commit `60e82406916921b823616bee33397baab38af3f0`;
- SID images listed in `data/sid/manifest.json` from `saberzl/SID_Set` revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`;
- 10 DALL-E 3, 10 Midjourney, and 10 SDXL files from `openai/dalle3-eval-samples` revision `d7c88c07b492ad7b9fd3003126d00719a2edabb1`.

## Environment

On DGX Spark, create a private environment inside this directory and install the pinned ARM64 CUDA 13 builds from PyTorch's official wheel index:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_cuda.py
.venv/bin/python run.py self-test
.venv/bin/python run.py run --device cuda:0
```

Outputs are written to `results/sid.json`, `results/external.json`, and `results/environment.json`.

The runner uses local model and image files only. It does not download data, fit a threshold, or fine-tune weights.

## Expected replication values

Local CPU reference:

- clean balanced accuracy 0.925, ROC-AUC 0.994375;
- worst balanced accuracy 0.825 at resize 0.25x;
- external synthetic recall: DALL-E 3 1.0, Midjourney 1.0, SDXL 0.9.

Small probability differences may occur across devices. Binary decisions and aggregate metrics should match unless a score lies extremely close to 0.5.

## Observed Spark A916 result

The frozen bundle completed on `spark-a916` in 37.99 seconds using an NVIDIA GB10 on `cuda:0`. All 1,230 binary decisions matched the local CPU reference. Fourteen of fifteen condition aggregates matched exactly; color-jitter ROC-AUC was 0.989375 on the GB10 versus 0.990000 locally, with identical balanced accuracy and confusion counts.
