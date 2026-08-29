# TODO

Last updated: 29 August 2026

Current rule: **do not fine-tune yet**. First determine whether the SigLIP ranking survives legal, source-disjoint and generator-disjoint evaluation.

## Completed

- [x] Research Track 5 requirements, current detector literature, architecture options, and legal constraints.
- [x] Run a zero-fine-tuning gate across 10 learned models plus basic heuristics.
- [x] Confirm Ateeqq SigLIP on 80 SID images across 15 clean/degraded conditions.
- [x] Run positive-only DALL-E 3, Midjourney, and SDXL smoke tests.
- [x] Reproduce all 1,230 decisions on `spark-a916` using an NVIDIA GB10.
- [x] Commit research, code, tests, manifests, and raw results to branch `xuan`.

## P0: Build the decisive clean holdout benchmark

- [ ] Freeze a rights-approved data panel in `research/data-panel.md`.
  - Real groups: camera photographs, processed photographs, CGI/renders, digital art, and screenshots.
  - AI groups: DALL-E 3, Midjourney, SDXL, FLUX, GPT Image, plus at least one fully held-out current generator.
  - Record source URL, licence or permission basis, collection date, generator/API version, and intended competition use.
- [ ] Add `data/manifest.schema.json` covering image ID, base lineage, SHA-256, perceptual hash, class, subtype, source family, generator family, rights, dimensions, encoding, and split.
- [ ] Add `scripts/validate_manifest.py` and `tests/test_manifest.py`.
  - Reject missing rights, duplicate IDs, invalid labels, absent files, and hash mismatches.
  - Test must pass on a small committed fixture without downloading datasets.
- [ ] Add `scripts/build_splits.py` and `tests/test_splits.py`.
  - Group all derivatives and perceptual near-duplicates into one split.
  - Hold out complete generator and real-source families.
  - Produce separate development, calibration, and locked-test manifests.
- [ ] Run a duplicate/leakage audit before scoring.
  - Acceptance: zero SHA-256 duplicates and zero grouped lineages crossing splits.
- [ ] Score the current SigLIP baseline on the clean frozen benchmark.
  - Report AUROC, balanced accuracy, TPR at fixed low FPR, worst-generator performance, worst-real-source FPR, and bootstrap confidence intervals.
  - Treat the published threshold as fixed; use only the calibration split for any new threshold or abstention region.
- [ ] Commit the frozen clean benchmark definition before applying robustness transformations.

## P1: Robustness and error analysis

- [ ] Generalise `outputs/spark-a916-evaluation/run.py` to consume a manifest instead of bundled fixed paths; add a fixture-based self-test first.
- [ ] Re-run the 15-condition grid class-symmetrically on the frozen holdout benchmark.
- [ ] Add selected realistic ordered chains: resize then JPEG, JPEG then resize, blur then JPEG, crop then resize, and screenshot-style resampling then JPEG.
- [ ] Measure class-specific score shift for every transform, not only AUROC.
- [ ] Report worst-condition metrics and an explicit instability/abstention rate.
- [ ] Build error cohorts for severe downscaling, blur, CGI, digital art, and screenshots.
  - Prioritise persistent real-image false positives already observed under 0.25x resize and sigma-2 blur.

## P1: Representation gate

- [ ] Compare frozen DINOv3-L features with logistic and Gaussian/LDA heads.
- [ ] Compare DINOv3-L global-token and spatial PatchHead-style aggregation.
- [ ] Test PE-L linear probing before freezing the backbone choice.
- [ ] Rank candidates by minimum performance across held-out generators, degradations, and real-source shifts; use hard-negative real FPR as a guardrail.
- [ ] Fine-tune or add LoRA only if frozen representations plateau after the clean holdout gate.

## P2: Submission readiness

- [ ] Resolve the undocumented training-image provenance of the Ateeqq checkpoint or replace it with a submission-safe dependency.
- [ ] Verify exact official Track 5 parameter, packaging, external-data, and inference-output requirements against the current brief.
- [ ] Record parameter count, peak VRAM, latency, throughput, dependency licences, and model/data revisions.
- [ ] Freeze model, threshold, transforms, and code before any provided external evaluation.
- [ ] Add final robustness tables, limitations, three-minute demo, and required prediction JSON validation.

## Deferred unless evidence justifies them

- Multi-backbone inference ensembles.
- Frequency/residual expert branches.
- Exhaustive transform-pair matrices.
- Active generator-boundary sample search.
- ONNX/TensorRT optimisation before model freeze.
