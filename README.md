# TechJam AIGC Detector

## Project overview

This repository implements a robust binary detector for **purely AI-generated versus authentic images** for TikTok TechJam 2026. The detector must generalize across generator families (GAN → diffusion → flow → autoregressive → unified multimodal) and survive common real-world transformations (JPEG recompression, blur, resize, noise, color jitter, crop).

The final model is **TRACE-RX-M v2** (`src/techjam_aigc/trace_rx_m`), an authentic-reference memory detector over a DINOv3 ViT-B/16 backbone: each semantic patch is compared against 2,048 learned authentic prototypes, and residual plus retrieval evidence feeds a compact classifier head. The repository also contains the **three-branch detector** (`src/techjam_aigc/three_branch`) as a research extension. The full architecture, training methodology, results, and references are in the [written project description](docs/writeup.md); the method proposal, bias register, evaluation contracts, and research reports live under [`docs/`](docs/README.md), and the authoritative challenge brief is [`docs/problem-statement.md`](docs/problem-statement.md).

Challenge guardrails respected throughout: public/licensed data only, final model under 2B parameters, no reuse of existing pretrained AIGC detectors, no watermark reliance, and a directory-inference entry point emitting `image_path`/`pred` JSON records.

## Setup and installation

Requirements: Python ≥ 3.12, [`uv`](https://docs.astral.sh/uv/), and a CUDA GPU for training (BF16 autocast; inference also runs on CPU, slowly).

```bash
# Core (analysis, evaluation tooling)
uv sync

# Training stack (torch, torchvision, transformers, huggingface-hub)
uv sync --group train
```

Hugging Face authentication is required to download the gated backbone; Weights & Biases is used by the TRACE-RX-M training stages (which also publish weights to the Hugging Face `techjam-aigc` organization):

```bash
hf auth login
wandb login   # only needed for the TRACE-RX-M stages
```

The operating backbone is the **gated** `facebook/dinov3-vitb16-pretrain-lvd1689m` (the three-branch research configs use `facebook/dinov3-vith16plus-pretrain-lvd1689m`). Before any training stage: review Meta's DINOv3 licence, accept access on Hugging Face, pin an immutable commit in `backbone.revision`, and set `backbone.license_accepted: true` in the config — the pipeline fails closed otherwise. A public Apache-2.0 DINOv2 fallback config is archived at `configs/archive/trace-rx-m-dinov2-historical.json`. Tokens stay in the environment or credential store; never in configs or commits.

Verify the environment:

```bash
uv run pytest -q
```

## Reproducing our results

### 1. Prepare the TechJam 2026 dataset

`data/techjam2026_v2` (94,490 rows) ships with leakage-grouped `train` (67,418) / `val` (24,912) / `test` (2,160) splits. Preparation normalizes images through a deterministic 512px-bounded resize + 224px center crop (neutralizing the class-correlated PNG/JPEG codec and resolution shortcut) and writes the training manifest:

```bash
uv run python scripts/prepare_techjam2026_training.py
```

Output: `data/techjam2026_v2-normalized/training-manifest.csv` plus audit metadata (labels SHA-256, split/pool/generator/transform counts).

### 2. Train the final TRACE-RX-M v2 model

The authoritative configuration is `configs/trace-rx-m-v2-vitb16-5ep.json` (DINOv3 ViT-B/16, 20 memory epochs, 5 detection epochs). Training runs as strictly ordered stages (full details and per-stage artifacts: [`docs/trace-rx-m-training.md`](docs/trace-rx-m-training.md)):

```bash
for stage in protocol cache capacity detection reliability; do
  uv run --group train python scripts/train_trace_rx_m.py \
    --stage "$stage" \
    --config configs/trace-rx-m-v2-vitb16-5ep.json
done
```

The published checkpoint is on the Hugging Face `techjam-aigc` organization; the frozen backbone is referenced by pinned model ID and revision, and the decision threshold is selected on the validation split.

The three-branch research extension trains via `scripts/train_three_branch.py` with `configs/three-branch.json` (see [`docs/three-branch-model.md`](docs/three-branch-model.md)); it is not the evaluated final checkpoint.

### 3. Run inference (challenge entry point)

```bash
uv run --group train python scripts/infer_three_branch.py \
  --image-dir path/to/images \
  --output predictions.json \
  --device cuda
```

This emits one JSON record per image with `image_path` and the AIGC confidence score in `pred`, as required by the challenge.

### 4. Evaluate

Canonical metric definitions are in [`docs/evaluation-metrics.md`](docs/evaluation-metrics.md). The frozen benchmark protocol ([details](docs/trace-rx-m-v2-benchmark-evaluation.md)) validates the full inventories, then deterministically samples 2,000 endpoints per dataset from a frozen seed:

| Dataset | Full rows | Evaluated rows |
| --- | ---: | ---: |
| TechJam 2026 v2 test | 2,160 | 2,000 |
| WildFake reconstructed test | 20,000 | 2,000 |
| EvalGEN (positive-only: Flux, GoT, Infinity, NOVA, OmniGen) | 55,298 | 2,000 |

WildFake and EvalGEN endpoints each receive one deterministic transform chain (length 1–6) assigned by seed and parent ID, independent of labels and scores. The TRACE-RX-M score-export and evaluation pipeline:

```bash
uv run --group train python scripts/export_trace_rx_m_scores.py --help
uv run python scripts/evaluate_trace_rx_m_scores.py --help
uv run python scripts/summarize_external_evaluation.py --help
```

Written evaluation reports: [official-transform evaluation](docs/trace-rx-m-transformed-evaluation.md), [WildFake/AIGIBench external generalization](docs/trace-rx-m-external-generalization-evaluation.md), [v2 benchmark evaluation](docs/trace-rx-m-v2-benchmark-evaluation.md), and the [training audit with visual error analysis](docs/trace-rx-m-training-audit.md).

### 5. Exploratory analysis (optional)

The pre-modeling phases are reproducible as well:

```bash
# Visual dataset EDA
uv run scripts/prepare_eda_samples.py
uvx marimo@latest edit notebooks/dataset_eda.py --no-token --sandbox

# Deterministic classical-feature, shortcut, and transformation audit
uv run python scripts/run_feature_lab.py --feature-profile frozen_v1 --transform-profile core --workers 4 --bootstrap 200
uv run marimo edit notebooks/feature_robustness_lab.py --no-token
```

See the [feature robustness lab contract](docs/feature-robustness-lab.md) and [signal-forensics notebook](docs/signal-forensics-eda.md).

## Limitations and future work

- **Model size was kept small due to the limited hackathon time.** The trainable footprint is narrow — LoRA adapters and lightweight heads over a frozen ViT-B/16 backbone — and the final run is a single five-epoch detection training.
- **Evaluations can be enriched with more recent generators.** WildFake was created in 2024, so its inventory ends at earlier GAN/diffusion families. The [generator coverage plan](docs/generator-coverage-benchmark-plan.md) documents the intended additions (EvalGEN's autoregressive and unified-multimodal models first).

## Team contributions

| Member | Contributions |
| --- | --- |
| **Joshua** | Dataset curation, model architecture design |
| **Xu An** | Demo video and GUI design |
| **Hibiki** | Model architecture design, evaluations |
| **Benjamin** | Model training, model architecture design |
| **Joel** | Model training, evaluations |
