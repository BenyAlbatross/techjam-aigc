# Devpost draft

This is an evidence-backed draft for manual submission. Replace every bracketed
placeholder only after human verification. Do not describe the project as
publicly deployed, submission-approved, or a universal image-authentication
system.

## Project identity

- **Project name:** TRACE LENS *(confirm final submission name)*
- **Tagline:** Local evidence browser for stress-testing AI-generated-image
  detectors under common image transformations. *(confirm final wording)*
- **Selected track:** [Confirm exact official TechJam track name]
- **Team members and contributions:** [Add verified names and contributions]
- **Public repository URL:** [Add verified public URL]
- **Public demo video URL:** [Add verified public YouTube URL]

## Problem

AI-generated images can be redistributed after JPEG recompression, blur,
resizing, noise, color changes, and cropping. Those routine transformations can
change a detector's behavior, so a clean-image score alone does not explain how
a model might behave in a real feed. TRACE LENS addresses the image-level binary
AIGC-versus-authentic problem by making fixed-threshold detector evidence,
transformation conditions, and errors inspectable together.

## Solution

TRACE LENS is a local benchmark evidence browser plus a directory-to-JSON
inference runner. The browser presents approved local benchmark images with
source/rights metadata, a transformation chain, detector probabilities and
threshold decisions, and filters for true/false positives and negatives. Its
analytics view recomputes per-model and per-condition confusion metrics from
local prediction shards. An optional local upload panel can invoke a cached
CUDA detector for an unknown image; that upload is excluded from benchmark,
training, and calibration.

The command-line inference script recursively accepts an image directory and
writes a JSON array with exactly `image_path` and `pred`, where `pred` is a
finite AI probability in `[0, 1]`. Invalid images can be reported separately in
an optional diagnostics file.

## User workflow

1. Prepare the approved local SID_Set manifest, images, predictions, and gallery
   derivatives.
2. Open TRACE LENS locally and choose a model and transformation condition.
3. Inspect the gallery for prediction mismatches, then open an image to review
   its recorded metadata, provenance chain, probability, threshold, and outcome.
4. Use the analytics view to compare FPR, FNR, balanced accuracy, mismatch rate,
   and transformation-linked changes across available prediction shards.
5. Optionally test one new PNG, JPEG, or WebP image locally, treating the result
   as an unbenchmarked score rather than proof of authenticity or manipulation.

## Technical approach

### Runnable baseline and browser

The current runnable inference path uses immutable registry entries in
`models.toml`. Each entry records a pinned source revision, SHA-256, parameter
count, loader, label direction, and fixed threshold. The browser reads the
canonical manifest and prediction JSONL files locally; it has no evidenced public deployment. A manual Cloud Run workflow is committed,
but it requires user-owned Google Cloud identity and approved runtime artifacts before use.

The benchmark uses 1,000 valid real and 1,000 valid generated `SID_Set`
validation rows from the pinned registry revision, excluding label 2. It runs
fifteen deterministic, class-symmetric conditions: clean; JPEG q90/q70/q50/q30;
Gaussian blur at 0.5/1/2; 0.5x and 0.25x resize-and-restore; Gaussian noise at
0.02/0.05/0.10; 20% color jitter; and 80% center crop.

### Research architecture

The repository also contains TRACE-RX Parallel model-development code. It
combines a DINOv3 ViT-L/16 patch encoder with a global pooled-token branch and
an authentic-reference memory/residual branch, then applies learned late fusion.
It is not currently a registered inference model, browser upload model, or
released checkpoint, so it is presented here as a research direction rather
than a deployed result.

## Measured results

The committed full public-gate report covers ten registry models on the
2,000-image SID_Set gate across fifteen conditions. Its deterministic display
ordering lists `wkaandemir_clip`, `ateeqq_siglip`, and
`frontier_community_forensics` as the top three. It explicitly marks the
**winner unresolved** because the required 95% confidence-interval comparisons
are not all conclusive. Do not replace that qualification with a winner claim.

A smaller 80-image frozen confirmation run for Ateeqq SigLIP reported clean
balanced accuracy of 0.925 and clean ROC-AUC of 0.994375. Its worst balanced
accuracy was 0.825 after 0.25x downscaling. The same evaluator made matching
binary decisions on the recorded local CPU and NVIDIA GB10 runs. This small
confirmation is a cross-platform check, not the model-selection result.

The report also highlights transform-specific failure modes. For example, the
full-gate Ateeqq SigLIP result records its worst real-source FPR at 0.525 after
0.25x resize-and-restore. This is why the product exposes error outcomes and
condition context instead of presenting a single score as decisive.

## Innovation and differentiation

- Makes transformation context, provenance metadata, probabilities, fixed
  thresholds, and error outcomes inspectable in one local workflow.
- Treats robustness as a condition-by-condition evaluation problem instead of
  reporting only aggregate clean accuracy.
- Keeps rights status, model revision, checkpoint hash, and dataset approval
  state in explicit registries.
- Separates a runnable fixed-threshold technical baseline from an experimental
  hybrid global-plus-authentic-memory research architecture.

## Impact and feasibility

The prototype can help researchers, reviewers, or safety teams inspect where a
detector is stable, where it fails, and which transformations are associated
with those failures. It uses local manifests, cached checkpoints, and
file-backed prediction shards, making its evidence traceable without requiring
a production backend. The current web experience is intentionally local; it
requires approved runtime artifacts and, for ad hoc inference, a compatible
Linux AArch64 CUDA environment with cached model files.

## Challenges and accomplishments

**Challenges:** Detector performance changes across realistic transformations;
public data/model licenses and training provenance require review; and point
rankings can be misleading when confidence intervals overlap.

**Accomplishments:** The repository implements a pinned-model benchmark,
deterministic transformation suite, grouped-bootstrap reporting, JSON inference
contract, local evidence browser, and a cross-platform confirmation run. It also
keeps release blockers visible instead of silently treating technical evidence
as submission approval.

## Limitations and responsible use

This is a fixed-threshold technical baseline and robustness analysis. It does
not authenticate an image, identify people, match faces, use EXIF-based person
identification, perform external lookup of depicted people, cover every model or
generator, or establish performance beyond the pinned data and conditions.
Positive-only smoke tests are not generalization estimates. The Ateeqq
checkpoint's stated training-image provenance remains unresolved, and all
registry models remain under submission review. Results should be used as
investigative evidence, with human review, rather than as a sole automated
decision.

## Future work

- Select and clear a final submission model after rights, provenance, and track
  requirements are human-verified.
- Build a frozen source- and generator-disjoint holdout with complete rights
  evidence, then evaluate clean and chained transformations.
- Evaluate the TRACE-RX Parallel research model only after a trained checkpoint,
  reproducible training artifacts, and separate held-out results are available.
- Measure deployment-relevant latency, VRAM, throughput, and accessibility on
  the eventual target environment.

## Technology stack and disclosures

- Python 3.12/Pixi, PyTorch, Transformers, Pillow, NumPy, and pytest for
  inference, benchmarking, reporting, and checks.
- Next.js 15, React 19, TypeScript, and local Node.js file access for the
  evidence browser.
- Pinned third-party model and dataset inventories are documented in
  `models.toml` and `datasets.toml`; their use remains subject to the listed
  status and release guardrails.

## Submission checks before publishing

- [ ] Replace all project, track, team, repository, and video placeholders.
- [ ] Verify the official track requirements and public-access conditions.
- [ ] Confirm the description matches the final report and selected model.
- [ ] Confirm no restricted images, weights, caches, private paths, credentials,
  Tailnet details, or unverified claims appear in the repository, screenshots,
  video, or Devpost entry.
