# Submission handoff

Status date: 2026-09-01. This document is a handoff and evidence index, not a
claim that the project has been submitted, publicly deployed, or approved for
competition use.

## What is in the repository

TRACE LENS is a local evidence browser for a controlled AIGC-detection
benchmark. Given the required local artifacts, it presents a 72-image gallery,
source/rights metadata, recorded and selected transformation chains, per-model
scores, confusion outcomes, and condition analytics. Its upload panel can run
one supported cached detector locally. The implementation is in `web/`.

The project also contains TRACE-RX Parallel model-development code. It uses a
DINOv3 ViT-L/16 patch encoder, a global pooled-token branch, an
authentic-reference memory/residual branch, and learned late fusion. This is
research code only in the current checkout: it is not a model key in
`models.toml`, is not loaded by `scripts/infer.py`, and is not an enabled web
upload model. Do not present it as the deployed or benchmarked detector without
a trained, released, and separately evaluated checkpoint.

The runnable directory inference path is `scripts/infer.py`. It recursively
reads local images, applies EXIF transpose and RGB conversion, and writes a JSON
array whose objects contain exactly `image_path` and `pred`, where `pred` is a
finite AI probability in `[0, 1]`. Corrupt files are excluded from that JSON and
can be recorded in the optional diagnostics file.

## Setup and local run

The locked Pixi workspace declares only Linux AArch64 CPU and CUDA 13.0
platforms. Install the CPU environment and run its checks:

```bash
pixi install --platform linux-aarch64-cpu
pixi run --platform linux-aarch64-cpu test
```

Install and verify CUDA only on a compatible NVIDIA host:

```bash
pixi install --platform linux-aarch64-cuda
pixi run --platform linux-aarch64-cuda cuda-check
```

For the Next.js browser, install the lockfile-resolved packages and start the
development server from `web/`:

```bash
cd web
npm ci
npm run typecheck
npm run dev
```

This starts a local server. A manual Cloud Run workflow is configured in
`.github/workflows/deploy-cloud-run.yml`, but no successful public deployment
URL is evidenced in this checkout. Without `work/manifests/sid_set_1000x2_canonical.json`, the UI
intentionally renders its empty state. A populated gallery additionally needs
the corresponding ignored `work/data/` images, `work/predictions/` shards, and
`work/app-gallery/` derivatives. After the manifest and images are available:

```bash
pixi run --platform linux-aarch64-cpu python scripts/build_gallery_derivatives.py
```

The upload API accepts PNG, JPEG, or WebP files up to 10 MB. It invokes the
CUDA Pixi inference task offline with a cached checkpoint and enables only
`ateeqq_siglip`, `community_forensics`, and `univfd`. It returns HTTP 503 if
that local model environment is unavailable. Uploaded files are written beneath
ignored `work/ad-hoc/`; an explicit delete request removes the upload session.

For command-line inference, fetch an approved immutable snapshot, then run one
registered model:

```bash
pixi run fetch -- --model ateeqq_siglip --cache work/hf-cache
HF_HUB_OFFLINE=1 pixi run --platform linux-aarch64-cuda infer -- \
  --model ateeqq_siglip --input path/to/images \
  --output work/submission/predictions.json \
  --diagnostics work/submission/diagnostics.json \
  --device cuda:0 --cache work/hf-cache
```

The fetch script verifies registered primary and auxiliary SHA-256 hashes. The
registry's model revisions, parameter counts, thresholds, label directions, and
loader contracts are authoritative in `models.toml`.

## Data, evaluation, and analytics

The controlled public gate uses only `SID_Set` validation rows from the pinned
revision in `datasets.toml`: 1,000 valid real and 1,000 valid generated rows,
with label 2 excluded. The manifest builder records the selection and content
hashes. It is deliberately separate from blocked or review-only datasets.

The benchmark applies fifteen deterministic, class-symmetric conditions: clean;
JPEG q90/q70/q50/q30; Gaussian blur at 0.5/1/2; resize-and-restore at 0.5x and
0.25x; Gaussian noise at 0.02/0.05/0.10; 20% color jitter; and an 80% center
crop. `scripts/report.py` computes confusion counts, error, FPR, FNR, balanced
accuracy, AUROC, grouped-bootstrap 95% confidence intervals, score changes,
decision flips, throughput, invalid counts, contamination, and worst cohorts.
The browser independently recomputes per-model/per-condition confusion and
mismatch analytics from local prediction JSONL files.

The committed full report covers ten registry models over the 2,000-image gate.
It gives the displayed top three as `wkaandemir_clip`, `ateeqq_siglip`, and
`frontier_community_forensics`, while reporting the winner as **unresolved**.
That qualification matters: its ranking rule requires the relevant 95% interval
comparisons to support the result, and it does not report that support for a
winner. The report, not the smaller 80-image cluster confirmation, is the
source for these full-gate findings.

## Guardrails and limits

- No evaluation image may tune, calibrate, ensemble, or alter a published
  threshold. The current work is a fixed-threshold baseline, not a trained final
  submission model.
- `scripts/compliance.py` gates benchmark/submission scope using the data and
  model registries. Current release configuration keeps the problem brief,
  repository public visibility, and free judging access unverified; every model
  remains `submission_status = "review"`.
- The Ateeqq checkpoint's stated training-image provenance is unresolved. The
  repository calls it a technical baseline rather than a cleared final
  submission dependency.
- Do not claim image authentication, universal generator coverage, performance
  beyond the pinned data/conditions, or generalization from the positive-only
  smoke tests. No identity recognition, face matching, EXIF-based person
  identification, or lookup of depicted people is part of the project.
- Keep images, model weights, caches, manifests, prediction shards, secrets,
  and ad-hoc uploads out of Git. `work/` runtime assets are intentionally
  ignored.

## Submission status and owner actions

| Deliverable | Repository evidence | Current status / next action |
| --- | --- | --- |
| Runnable directory-to-JSON inference | `scripts/infer.py`; output schema enforced | Implemented for registered cached models; select a cleared final model before claiming it as the competition submission. |
| Robustness report and error evidence | `outputs/public-baseline-robustness-report.md` | Committed technical evidence; use its scope limits and unresolved-winner language. |
| Local TRACE LENS demo | `web/` application and local artifact loaders | Implemented locally; populate approved local artifacts and record a demo. No hosted access is evidenced. |
| Public repository | `compliance.toml` has `repository_public = false` | Unverified/blocked; set public only after a human audit confirms no restricted artifacts or secrets. |
| README and project documentation | README, this handoff, data/model registries | In progress; project-level license, team attribution, and clean-checkout proof are not present as completed evidence. |
| Devpost description and final entry | No Devpost export or submission receipt in the repository | Not evidenced; complete manually and capture confirmation. |
| Public demo video | No video URL or recording asset in the repository | Not evidenced; record and validate logged-out playback. |
| Track-specific package | `docs/problem-statement.md`; `problem_brief_verified = false` | Brief is documented but human verification remains required before final packaging. |

Before any release, run the documented compliance check, confirm repository and
video access while logged out, verify the official track requirements, and make
the Devpost narrative match the report rather than the research roadmap.

## Submission drafts

- [`docs/devpost-draft.md`](devpost-draft.md) is the evidence-backed narrative
  template. It retains placeholders for the final project/track/team/repository/video
  details and preserves the report's unresolved-winner qualification.
- [`docs/demo-script.md`](demo-script.md) is a 2–4 minute local-demo narration
  with screen actions, claim limits, and privacy checks.

## Evidence map

- Product/UI behavior: `web/lib/data.ts`, `web/lib/analytics.ts`,
  `web/app/api/inference/route.ts`, and `web/components/gallery-workbench.tsx`.
- Inference schema and safeguards: `scripts/infer.py` and
  `scripts/model_adapters.py`.
- Benchmark protocol and metrics: `scripts/benchmark.py`, `scripts/report.py`,
  and `outputs/public-baseline-robustness-report.md`.
- Dataset/model provenance and release controls: `datasets.toml`, `models.toml`,
  and `compliance.toml`.
- TRACE-RX research architecture: `src/techjam_aigc/trace_rx_parallel/model.py`
  and `configs/trace-rx-parallel.json`.
