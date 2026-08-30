# AIGC exploration plan reconciliation

This audit checks the implementation against
[`aigc-exploration-implementation-plan.md`](aigc-exploration-implementation-plan.md),
the challenge brief, and the decisions made while planning the exploration. It
distinguishes executable work from work that is deliberately blocked by absent
licensed data or an unapproved heavyweight dependency.

## Six-step status

| Step | Status | Implemented evidence | Remaining boundary |
|---|---|---|---|
| 1. Larger SID/WildFake subsets | Ready, acquisition blocked | Runnable manifest schema, deterministic strata, binary/tampered/demo guards, provenance and power tables | Populate exact revisions, hashes, and both license layers before using more bytes |
| 2. Targeted AI-GenBench | Ready, acquisition blocked | Per-generator pilot/serious limits, chronological metadata propagation, whole-generator final-window guard | Audit every constituent source and assign generator IDs before acquisition |
| 3. Frozen registry | Implemented | `frozen_v1` fixes the original 53 scalars; profile and schema hashes are cached | Run against expanded licensed data without changing the schema |
| 4. Missing feature families | Implemented for inexpensive v2; controls deferred | `expanded_v2` contains 82 scalars and all eight planned inexpensive families with hypothesis/failure/role/cost metadata | Licensed DINOv2/CLIP, reconstruction evidence, additional exact composed scalars, sparse/OOF fusion |
| 5. Community Forensics diversity | Sampler implemented, experiment blocked | Deterministic disjoint few/many-generator cohorts with equal image counts | Keep source unselected until audited; then train and evaluate both cohorts on one frozen target |
| 6. Untouched final confirmation | Operationally sealed | Default extraction withholds final rows; explicit one-time flag, timestamp, index/hash/schema/recipe receipt, and repeat guard | Assign a licensed final generator window only after the method is frozen |

## Implemented safeguards and analyses

- Planner output contains every field required by the feature loader.
- A planned index carrying `source_id` cannot enter extraction without a sibling
  `audit.json` whose source and underlying-image licenses are fully allowlisted
  and whose configured revision and selected-index SHA-256 match exactly.
- Parent IDs and discovery, confirmation, and final-confirmation phases survive
  extraction. Chronological-window, prompt, source, and revision metadata are
  propagated when present.
- Final-confirmation rows are excluded unless
  `--evaluate-final-confirmation` is given. A successful first evaluation writes
  a persistent receipt beside the index and a repeated run is refused.
- Core remains 20 bounded conditions. Opt-in profiles add axial/mixed color
  cases, all 30 directed medium pairs, 12 realistic chains, and a deterministic
  32-recipe covering bank. The 12,959-condition Cartesian grid is never the
  default.
- Ordered recipes record parameters, interpolation, padding, seed policy, and a
  SHA-256. Stochastic steps use operation-level seeds so exact reversed recipes
  share the same noise draw.
- Evaluation includes frozen directions, intervals, power/reversal warnings,
  dataset/generator breakdowns, nuisance correlations, family ablations,
  leave-one-dataset-out tests, parent drift, severity area, clean-to-condition
  loss, pair interaction excess, exact order sensitivity, and guarded
  chronological metrics.
- Run metadata records input/schema hashes, conditions, seeds, final-window
  state, and measured extraction throughput.
- The marimo notebook binds its microscope to the run's recorded input index and
  verifies its SHA-256, limits selectors to parents actually extracted into the
  cache, and therefore cannot expose withheld final rows. It presents
  shared-scale real/AIGC maps, parent severity trajectories, feature-specific order
  heatmaps, realistic/covering summaries, a clean/worst-case Pareto frontier,
  diversity status, and a separately filtered sealed-final section whose
  generator IDs and evaluation timestamp come from the evaluated cache.

## Explicit deferrals

These are not silently counted as completed results:

- deterministic continuous-interior color samples for training and adaptive
  discovery-only worst-case severity expansion;
- prompt-matched real/fake, real-real, fake-fake, near-duplicate, and matched
  re-encoding experiments where the current index lacks the necessary keys;
- leave-one-generator model training, which needs adequately powered repeated
  generators across phases;
- sparse early fusion, out-of-fold late fusion, coefficient stability, and the
  exact chroma-spectrum, wavelet-patch, JPEG-difference/DCT, and
  Gaussian-residual/co-occurrence composed branches;
- a generic DINOv2/CLIP control until identifier, revision, preprocessing,
  weight hash, and license are approved; and
- reconstruction evidence until its model choice, license, compute, and
  diffusion-family specialization justify the cost.

No AI-GenBench or Community Forensics data was downloaded. That is intentional:
the checked-in policy remains dry-run and all selected external sources remain
blocked pending complete review.

## Validation gate

Completion requires the full test suite, marimo static check, live-kernel cell
execution, diff check, and the audited planner-to-extraction-to-evaluation smoke
test to pass. The final handoff records the exact results of those checks.
