# Public Baseline Robustness Benchmark Design

Date: 29 August 2026  
Branch: `xuan`  
Status: approved design, pending implementation plan

## 1. Objective

Build a reproducible, zero-fine-tuning benchmark that answers two questions:

1. Which existing image AIGC detectors are strongest on legally usable public real and fully AI-generated images?
2. Which detectors remain reliable after realistic redistribution transformations, and which transformations cause false positives or false negatives?

The benchmark will register all ten existing learned baselines and evaluate every baseline authorized for the selected data at its published threshold. Any excluded baseline will have a machine-readable compliance reason. Eligible models will be ranked by worst-condition robustness rather than clean accuracy alone, and the three strongest will run on larger public panels. The benchmark will not train, fine-tune, calibrate, or ensemble models.

The result is a baseline and error-analysis system, not the final hackathon detector. It must not claim universal AI detection, authenticity proof, or generator-unseen performance where training overlap cannot be excluded.

## 2. Scope

### Included

- A deterministic 2,000-image balanced gate drawn from an approved controlled public panel.
- Fifteen total conditions: clean; JPEG quality 90, 70, 50, and 30; Gaussian blur sigma 0.5, 1, and 2; downscale-and-restore 0.5x and 0.25x; Gaussian noise sigma 0.02, 0.05, and 0.10; deterministic 20% color jitter; and 80% center crop.
- A labeled shard from the NTIRE 2026 Robust AI-Generated Image Detection training release as aggregate in-the-wild validation, subject to the compliance gate.
- Full controlled-panel and one-shard NTIRE evaluation for the three strongest eligible models.
- Per-model and per-condition errors, false-positive rate, false-negative rate, balanced accuracy, AUROC, confidence intervals, clean-to-transformed score change, decision-flip rate, throughput, and worst cohort.
- Training-overlap and license/provenance flags.
- CPU and NVIDIA GB10 execution through Pixi environments.

### Excluded

- Fine-tuning, threshold fitting, probability calibration, and ensembling.
- The gated social-media robustness dataset. It remains KIV.
- The team's private dataset, which is still under development.
- The competition-provided COCO/DALL-E evaluation data before complete method freeze.
- Restricted, noncommercial, no-derivatives, unclear-license, or unauthorized assets unless the organizers provide written approval applicable to prize competition use.
- Containers, virtual machines, weight storage in Git, a model-serving platform, and mirroring every baseline checkpoint.

## 3. Binding compliance policy

Compliance is a prerequisite, not a reporting afterthought. A model or dataset with an unknown or incompatible authorization status may be researched at the metadata level but may not be downloaded, evaluated, packaged, demonstrated, or submitted. Unknown detector training composition is recorded separately: it permits a technical baseline when the checkpoint itself is authorized, but blocks use as a final submission dependency.

### 3.1 Track 5 constraints

The repository's supplied-brief records define these task constraints:

- Binary whole-image prediction for fully AI-generated versus non-generative images.
- Fewer than 2 billion total learned inference-time parameters, including every head and auxiliary branch.
- Directory input and prediction JSON objects containing exactly `image_path` and `pred` for required submission output.
- `pred` must be a finite AI probability in `[0, 1]`.
- Robustness reporting for the specified JPEG, blur, resize, noise, color, and crop transformations.
- No use of the provided external evaluation set for training, feature discovery, model selection, threshold selection, or calibration.
- Freeze model, checkpoint, threshold, preprocessing, and code before the provided external evaluation run.

The official problem-statement URL resolved from the current Devpost rules returned HTTP 404 from the development environment on 29 August 2026. The known constraints above are enforced immediately. Final release remains blocked until a team member revalidates the current brief through an authorized browser or obtains written organizer clarification and records the verification date and source.

### 3.2 Official TechJam rules

The implementation and release checklist must enforce the [current official rules](https://tiktoktechjam2026.devpost.com/rules), checked on 29 August 2026:

- Work submitted must be the team's original work and must be newly created or significantly updated during the submission period.
- Every third-party SDK, API, model, dataset, library, and asset must be authorized and used under its applicable license.
- Submission must disclose development tools, APIs, assets, and libraries used.
- Submission must link a public code repository with a README.
- A working project must be available free of charge and without restriction for judging through the end of the judging period.
- Submission materials and testing instructions must be in English.
- Submission content must not violate intellectual-property, contract, privacy, publicity, or other third-party rights.
- No attempt may be made to re-identify any person or personal information.
- All data used or processed must be deleted at competition completion.
- No substantive submission changes are allowed after the submission deadline except changes explicitly permitted by the Sponsor or Devpost.

Official rules prevail over this design, repository notes, and secondary summaries. Ambiguous terms require written clarification from the organizers; the system must not guess a permissive interpretation.

### 3.3 Compliance register and gates

`models.toml` and dataset records will contain:

- canonical source URL;
- exact immutable revision;
- file name and downloaded SHA-256;
- model or dataset license identifier and source;
- training-data provenance summary where known;
- parameter count and counting method for models;
- intended use: internal benchmark, public report, demo, or submission;
- status: `approved`, `review`, or `blocked`;
- approval basis and verification date.

Only `approved` entries may pass `fetch` or `benchmark`. Public availability alone does not constitute authorization. Community Forensics PublicEval and the NTIRE shard are preferred candidates, but both remain blocked until their exact release terms and underlying asset permissions are recorded as compatible with hackathon use.

Models trained on Community Forensics data will be marked contaminated for Community Forensics evaluation. Those scores may be reported, but they cannot contribute to winner selection on that panel. A model with undocumented or unresolved training-data provenance may remain a technical baseline but cannot become a submission dependency.

Release checks must also verify public repository visibility, English README and testing instructions, complete third-party disclosures, exact output schema, final model parameter count, and a recorded brief revalidation. Any failed check blocks release.

No identity recognition, face matching, EXIF-based person identification, or external lookup of depicted people is permitted. Reports use opaque sample IDs and omit secrets, usernames, and private local paths.

The data inventory will enumerate every downloaded raw image, transformed cache, and image-bearing temporary artifact. A deliberate `cleanup` task will list targets and require operator confirmation before deleting them at competition completion. The team will record a deletion attestation; cleanup will not silently delete broad directories.

## 4. Environment and model hosting

Pixi manages environments; it does not host weights.

- Commit `pixi.toml` and `pixi.lock`.
- Define CPU and CUDA rich-platform environments for Linux without containers or virtual machines.
- Ignore `.pixi/`, image datasets, checkpoint caches, prediction work files, and generated transformations.
- Use a shared local Hugging Face cache under `work/hf-cache` and download each approved upstream checkpoint once.
- Pin every upstream model by immutable revision and downloaded file hash.
- Run one model at a time with bounded batches to fit the GB10.
- Do not mirror baseline checkpoints merely for convenience.
- After license and provenance clearance, mirror only the selected final model into a private `techjam-aigc/<model>` Hugging Face repository if reliable submission hosting requires it.
- Produce an offline Pixi export only if the final submission platform cannot restore the locked environment.

Primary Pixi tasks:

- `pixi run fetch`: validate compliance and fetch explicitly selected approved assets.
- `pixi run test`: run network-free CPU tests.
- `pixi run benchmark`: execute or resume predictions.
- `pixi run report`: validate coverage and generate tables.
- `pixi run compliance`: print pass/fail status and release blockers.
- `pixi run cleanup`: preview competition-data cleanup; deletion requires explicit confirmation.

## 5. Components

### 5.1 `models.toml`

One entry per existing learned baseline. Each entry defines display name, Hugging Face or upstream source, immutable revision, expected file hash, loader name, native preprocessing, AI label mapping, published threshold, parameter count, license, provenance status, and contamination tags.

The ten-model panel is:

1. Ateeqq SigLIP;
2. Frontier Community Forensics;
3. Community Forensics;
4. Steganograph ViT;
5. Divine EfficientNet-B0;
6. wkaandemir CLIP;
7. Divine ConvNeXt-Tiny;
8. Divine ResNet-50;
9. CapCheck ViT;
10. UnivFD.

Existing adapters and model preprocessing in `work/baseline-spike/expanded.py` will be reused. No generalized plugin framework will be introduced.

### 5.2 Public data manifests

The controlled panel will use Community Forensics PublicEval only after compliance approval. Selection will be deterministic, class-balanced, and source-stratified. Exact row IDs, source families, labels, revisions, rights records, and content hashes will be stored in a manifest; images remain untracked.

The in-the-wild panel will use one labeled shard from `deepfakesMSU/NTIRE-RobustAIGenDetection-train` only after compliance approval. It measures aggregate real-world performance; it will not be used to claim exact transformation attribution when transformation metadata is absent.

Manifest validation rejects missing rights, ambiguous primary labels, duplicate IDs, missing files, hash mismatch, and cross-panel duplicate or lineage leakage where lineage is available. Partially AI-edited, recaptured, and unknown-provenance images remain outside the primary binary analysis unless separately labeled as secondary slices.

### 5.3 `scripts/benchmark.py`

The benchmark command will minimally consolidate the current loader, transformation, inference, and metric code. It accepts model names, panel manifest, conditions, batch size, device, output directory, and deterministic seed.

Transformations are generated in memory from the decoded base image. They are applied identically across classes and are not stored unless a debugging command explicitly requests one sample. Label-dependent filenames, transform parameters, or preprocessing are forbidden.

Every prediction row records:

- model name, revision, and weight hash;
- dataset name, revision, sample ID, and content hash;
- base or lineage ID when available;
- class and source/generator cohort;
- condition and deterministic parameters;
- raw score, AI probability, fixed threshold, and binary decision;
- device, effective batch size, runtime, code revision, and seed.

### 5.4 Reporting

The report command consumes prediction rows only; it never reruns inference or changes thresholds. It produces machine-readable JSON/CSV plus a concise Markdown report.

For every model and condition it reports error rate, false-positive rate, false-negative rate, balanced accuracy, AUROC, confusion counts, confidence intervals, score shift from clean, clean-decision flip rate, throughput, and worst source or generator cohort. It also lists persistent false positives and false negatives by opaque ID for later team review.

Controlled-panel uncertainty uses paired bootstrap resampling by base image so transformed copies do not act as independent observations. NTIRE uses image-level stratified bootstrap unless lineage metadata supports grouped resampling. The confidence level and bootstrap seed are fixed in configuration.

## 6. Evaluation protocol

### Stage 1: deterministic gate

- Select 1,000 real and 1,000 fully AI-generated controlled-panel images before model execution.
- Run every eligible model on all fifteen total conditions.
- Run every eligible model on 1,000 real and 1,000 fully AI-generated NTIRE images as distributed, without adding the controlled transformation grid. Select the first 1,000 valid manifest rows per class after sorting by stable sample ID.
- Use only published/default thresholds; never inspect test labels to tune them.
- Keep model-native preprocessing except for the explicitly defined external transformations.

### Stage 2: model ranking

Rank only models eligible for the evaluated panel. The primary key is worst-condition balanced accuracy. Tie-breakers are lower worst real-source false-positive rate, lower worst AI-source false-negative rate, and then higher aggregate AUROC. Clean accuracy alone cannot select a winner.

Contaminated model-panel pairs are reported but excluded from all ranking keys. Rankings must retain bootstrap uncertainty and may declare models statistically unresolved instead of forcing an unsupported strict order.

### Stage 3: full confirmation

- Run the top three eligible models on the full approved controlled panel.
- Run the same three on every valid image in one full approved NTIRE labeled shard at its pinned revision. Record the actual manifest row count; do not assume a fixed shard size.
- Preserve the Stage 1 thresholds, revisions, preprocessing, ranking rule, and report definitions.
- Make no model or threshold changes based on Stage 3 errors under the same claimed benchmark version.

## 7. Reliability and failure handling

Preflight fails before a long run if any selected entry lacks compliance approval, immutable revision, expected hash, label mapping, threshold, parameter count, or license record. It also verifies the Pixi lock, required files, image balance, output writability, selected device, and CUDA availability when requested.

Prediction shards use an identity composed of model revision, model hash, dataset revision, image hash, condition, and code/config revision. Writes are atomic. Resume skips complete identities, rejects conflicting duplicates, and never overwrites completed results silently.

Corrupt or unsupported images are logged with opaque sample ID and excluded from metrics. The run fails if missing or invalid images exceed 0.1% of the selected panel. Exact attempted, valid, excluded, and per-class counts always appear in reports.

CUDA out-of-memory handling halves batch size until one, records the effective size, and retries the failed batch. Another exception fails with model, condition, and sample IDs. It does not silently skip inference failures.

## 8. Testing

`tests/test_benchmark.py` will use a tiny committed, redistributable image fixture and a deterministic dummy model. It requires no network or GPU.

Tests cover:

- registry rejection for missing revision, hash, license, threshold, parameter count, label mapping, unknown loader, or non-approved status;
- fewer-than-2-billion parameter enforcement;
- deterministic transformation output, geometry, severity, and class-symmetric parameter generation;
- exact metrics on known labels and scores;
- grouped bootstrap behavior on repeated lineages;
- atomic resume, idempotence, and duplicate-conflict detection;
- valid-input coverage and corrupt-input accounting;
- exact submission JSON schema, finite `[0, 1]` probabilities, and no leaked local paths;
- a network-free end-to-end CPU benchmark and report.

Existing output-consistency and Spark self-tests remain in use where relevant. New code will not duplicate established loaders or transformation implementations.

## 9. Acceptance gates

Implementation is accepted only when:

1. `pixi run test` passes in the CPU environment.
2. CUDA preflight and one-fixture inference pass on the NVIDIA GB10.
3. Compliance rejects every unresolved or incompatible model and dataset.
4. Every one of the ten registered models has either exact expected model-sample-condition coverage or an explicit compliance exclusion. Completed predictions have zero conflicting duplicates and recorded revisions and hashes.
5. The report includes all required metrics, confidence intervals, contamination flags, exclusions, invalid counts, and known limitations.
6. The top three are selected by the preregistered robustness rule without test-time fitting.
7. No images, weights, credentials, Hugging Face tokens, private paths, caches, or large predictions are committed.
8. Project instructions state Caveman Ultra communication mode, minimal implementation expectations, the no-fine-tuning rule, compliance gates, and benchmark priorities.
9. Final release checks verify the current problem brief, task output contract, public repository, English README/testing instructions, third-party disclosures, working free test access, parameter ceiling, and freeze record.
10. The data inventory and safe end-of-competition cleanup procedure are documented and testable without deleting data during ordinary verification.

## 10. Deliberate simplifications

- One benchmark command and one registry replace a new framework.
- Existing model adapters and transformations are reused.
- Transformations are computed in memory instead of materializing a multiplied dataset.
- One model runs at a time instead of maintaining services or schedulers.
- JSON/CSV/Markdown outputs replace a database and dashboard.
- Bootstrap confidence intervals replace a larger statistical modeling stack.
- Social-media and degradation-tree benchmarks remain deferred until access, licensing, and current results justify their cost.

These constraints keep the benchmark auditable and finishable within the hackathon while preserving the legal, methodological, and task requirements that cannot be simplified away.
