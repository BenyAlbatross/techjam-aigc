# TechJam 2026 AIGC detection research

Research and zero-fine-tuning baselines for robust detection of fully AI-generated still images under distribution shift.

## Submission handoff

The concrete, evidence-backed submission inventory is maintained in
[`docs/submission-handoff.md`](docs/submission-handoff.md). It distinguishes
what is runnable in this checkout from research code that is not wired into the
inference or web paths, records the local-only access requirements, and lists
the remaining human-release decisions. The working status checklist is
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md).

The repository is currently a **local prototype**, not a published service.
The web application can be started without benchmark artifacts, but it shows an
empty-state message until the ignored canonical manifest is present. A populated
gallery additionally needs local benchmark images, prediction shards, and the
generated gallery derivatives described below. Ad hoc uploads require a Linux
AArch64 CUDA Pixi environment plus cached, hash-verified model files; otherwise
the route returns a local-model-unavailable response.

For a clean local setup, install Pixi and a Node.js runtime, then install the
locked web dependencies and run the CPU checks:

```bash
pixi install --platform linux-aarch64-cpu
pixi run --platform linux-aarch64-cpu test

cd web
npm ci
npm run typecheck
npm run dev
```

The CUDA inference and full benchmark paths additionally require the
`linux-aarch64-cuda` Pixi platform and an NVIDIA CUDA-capable host; see the
handoff for the exact artifact and access requirements. No credentials or
runtime artifacts belong in Git.

The integrated local evidence browser lives in `web/`. It presents canonical images as a gallery, traces known transformations and lineage, compares detector outputs, and supports isolated ad hoc testing.

Build its ignored, benchmark-exact transformation cache with:

```bash
pixi run --platform linux-aarch64-cpu python scripts/build_gallery_derivatives.py
```

Then run `cd web && npm install && npm run dev`.

See [`TODO.md`](TODO.md) for prioritised next steps and acceptance gates.

To continue the interrupted full public-gate run on another machine, follow
[`docs/handoffs/2026-08-30-machine-transfer.md`](docs/handoffs/2026-08-30-machine-transfer.md),
including its artifact checks and copy-paste continuation prompt.

## Reproducible public gate

The allowed public gate uses only the pinned SID_Set validation split. It selects
the first 1,000 unique valid real rows and first 1,000 unique valid generated
rows in the revision-pinned stream, records content hashes, and excludes label 2 as
required by [`datasets.toml`](datasets.toml). No evaluation image is used to
fine-tune, calibrate, ensemble, or change a published decision threshold.

The committed Pixi platforms retain their rich platform names. On the current
Linux AArch64 NVIDIA GB10 host, install and verify them with:

```bash
pixi install --platform linux-aarch64-cpu
pixi run --platform linux-aarch64-cpu test

pixi install --platform linux-aarch64-cuda
pixi run --platform linux-aarch64-cuda cuda-check
```

Model snapshots live under the ignored `work/hf-cache/` directory. Fetch only
immutable registry revisions and verify every primary and auxiliary checkpoint
against its registered SHA-256:

```bash
pixi run fetch -- --model all --cache work/hf-cache
```

Build the controlled manifest without admitting any other dataset:

```bash
pixi run --platform linux-aarch64-cpu python scripts/data_manifest.py build-sid \
  --per-class 1000 \
  --output work/manifests/sid_set_1000x2.json
pixi run --platform linux-aarch64-cpu python scripts/data_manifest.py validate \
  work/manifests/sid_set_1000x2.json
```

Build the non-ranking canonical shortcut-control panel from the native manifest.
This applies the same metadata-free RGB conversion, 512×512 Lanczos fit, and
PNG encoding to both classes:

```bash
pixi run python scripts/data_manifest.py canonicalize work/manifests/sid_set_1000x2.json \
  --output work/manifests/sid_set_1000x2_canonical.json \
  --image-dir work/data/sid_set_canonical/images --size 512
```

Run CUDA inference offline and one model at a time. Each completed model must
produce 30,000 rows in fifteen atomic, resumable shards. An out-of-memory retry
may lower the effective batch size, but it may not reduce coverage.

```bash
HF_HUB_OFFLINE=1 pixi run --platform linux-aarch64-cuda benchmark -- \
  --models ateeqq_siglip \
  --dataset sid_set \
  --manifest work/manifests/sid_set_1000x2.json \
  --conditions all \
  --device cuda:0 \
  --batch-size 32 \
  --output work/predictions
```

Repeat that command sequentially for `community_forensics`,
`frontier_community_forensics`, `wkaandemir_clip`, `divine_resnet50`,
`divine_efficientnet`, `divine_convnext`, `steganograph`, `capcheck`, and
`univfd`. Do not change a threshold, revision, expected hash, loader, or
preprocessing rule in response to a result or failure.

The fifteen preregistered conditions are clean; JPEG quality 90, 70, 50, and
30; Gaussian blur sigma 0.5, 1, and 2; 0.5x and 0.25x downscale-and-restore;
Gaussian noise sigma 0.02, 0.05, and 0.10; deterministic 20% color jitter; and
80% center crop. Transformations are deterministic and class-symmetric.

The report includes error rate, FPR, FNR, balanced accuracy, AUROC, confusion
counts, 95% grouped-bootstrap confidence intervals, clean-to-transformed score
change, decision-flip rate, throughput, contamination, invalid counts, and
worst cohorts. Controlled intervals resample `base_id` 2,000 times. Ranking
admits only approved, uncontaminated model/dataset pairs and orders them by:

1. higher worst-condition balanced accuracy;
2. lower worst-real-source FPR;
3. lower worst-generated-source FNR;
4. higher aggregate AUROC;
5. model key, only as a deterministic final tie-break.

The point ordering is not sufficient to declare a winner or top three. A
winner is reported only when its relevant 95% interval strictly supports every
comparison, and a top-three set only when every selected/excluded boundary
comparison is supported. Otherwise the report marks the winner, boundary, or
both unresolved.

Generate and validate the report with:

```bash
pixi run report -- \
  --predictions work/predictions \
  --manifest work/manifests/sid_set_1000x2.json \
  --json work/reports/summary.json \
  --csv work/reports/metrics.csv \
  --markdown outputs/public-baseline-robustness-report.md
pixi run --platform linux-aarch64-cpu python scripts/report.py validate \
  --predictions work/predictions \
  --manifest work/manifests/sid_set_1000x2.json \
  --models all
```

## Pinned third-party inventory

Checkpoint hashes, parameter counts, fixed thresholds, label directions, and
loader names are authoritative in [`models.toml`](models.toml). Dataset status,
selection, and exclusions are authoritative in
[`datasets.toml`](datasets.toml).

| Model key | Repository and immutable revision | License |
| --- | --- | --- |
| `ateeqq_siglip` | [`Ateeqq/ai-vs-human-image-detector@60e82406916921b823616bee33397baab38af3f0`](https://huggingface.co/Ateeqq/ai-vs-human-image-detector/tree/60e82406916921b823616bee33397baab38af3f0) | Apache-2.0 |
| `community_forensics` | [`buildborderless/CommunityForensics-DeepfakeDet-ViT@ac6ee457bea904a373065754107451793b56db00`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/tree/ac6ee457bea904a373065754107451793b56db00) | MIT |
| `frontier_community_forensics` | [`Thermostatic/community-forensics-frontier-detector-2026-08@16db135220b318d811b207db576d90368980b595`](https://huggingface.co/Thermostatic/community-forensics-frontier-detector-2026-08/tree/16db135220b318d811b207db576d90368980b595) | MIT |
| `wkaandemir_clip` | [`wkaandemir/ai-image-detector@fefa013737a0c3477961d36ee8dbbdc751352366`](https://huggingface.co/wkaandemir/ai-image-detector/tree/fefa013737a0c3477961d36ee8dbbdc751352366) | MIT |
| `divine_resnet50` | [`divine2k/ai-image-detectors@5dd08026ea41f07ad7c37b79ffaed08282667655`](https://huggingface.co/divine2k/ai-image-detectors/tree/5dd08026ea41f07ad7c37b79ffaed08282667655) | MIT |
| `divine_efficientnet` | same pinned `divine2k/ai-image-detectors` snapshot | MIT |
| `divine_convnext` | same pinned `divine2k/ai-image-detectors` snapshot | MIT |
| `steganograph` | [`delpot/steganograph-ia-detector@b395557b96de82e5aaf97206054af416d657655a`](https://huggingface.co/delpot/steganograph-ia-detector/tree/b395557b96de82e5aaf97206054af416d657655a) | MIT |
| `capcheck` | [`capcheck/ai-image-detection@a6661e07d38f1a097bba07ca9415538819278f09`](https://huggingface.co/capcheck/ai-image-detection/tree/a6661e07d38f1a097bba07ca9415538819278f09) | Apache-2.0 |
| `univfd` | [`siddharthksah/deepsafe-weights@71706520a9e12494b3ebc24e6091f5b22b9efcaf`](https://huggingface.co/siddharthksah/deepsafe-weights/tree/71706520a9e12494b3ebc24e6091f5b22b9efcaf) | MIT |

| Dataset | Revision and status | License evidence |
| --- | --- | --- |
| SID_Set | [`saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd`](https://huggingface.co/datasets/saberzl/SID_Set/tree/dc03ead57929879319ce30a82bfcfb8d317b10bd), approved only for the controlled gate | [CC BY 4.0 card](https://huggingface.co/datasets/saberzl/SID_Set/blob/dc03ead57929879319ce30a82bfcfb8d317b10bd/README.md) |
| Community Forensics Eval | [`OwensLab/CommunityForensics-Eval@7d4a74a88d2cac93b513c0853bf92c260eaceea0`](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval/tree/7d4a74a88d2cac93b513c0853bf92c260eaceea0), blocked | CC BY-NC-SA 4.0 is not cleared for a prize competition |
| NTIRE 2026 train | [`deepfakesMSU/NTIRE-RobustAIGenDetection-train@700b6d08a3268b1e7a191306dec7321dd953b12f`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train/tree/700b6d08a3268b1e7a191306dec7321dd953b12f), review-blocked | no license or usage grant is declared |
| Social-media robustness panel | unavailable revision, blocked | no license is declared; the user deferred this panel |
| Data draft | [`Joshyxwa/data_draft@e800837`](https://huggingface.co/datasets/Joshyxwa/data_draft/tree/e800837135e10a2dca7cbfa49ecf0b5b68830537), blocked | mixed `other`; 5,000 WildFake files require a licence audit and public visibility conflicts with its private-only card |
| ELSA 1M Track 1 | [`elsaEU/ELSA1M_track1@199bf76`](https://huggingface.co/datasets/elsaEU/ELSA1M_track1/tree/199bf769e2ddb673d68442a9756212ddd204426a), approved for AI-only stress testing, never ranking | CC BY 4.0 card; no packaged real class |
| CIFAKE repack | [`yanbax/CIFAKE_autotrain_compatible@f67ae8d`](https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible/tree/f67ae8dedee6bb83e7523f6ce3b715a12147a200), review-blocked | MIT is asserted by a third-party repack; original image-rights chain needs confirmation |
| AIGC Detection Benchmark repack | [`TheKernel01/AIGC-Detection-Benchmark@c91d902`](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark/tree/c91d9024a5a77ef06e2ec681b53f9caf08675663), review-blocked | Apache-2.0 is asserted without per-source image-rights evidence |
| Synthbuster Plus | [`marco-willi/synthbuster-plus@dbfb72f`](https://huggingface.co/datasets/marco-willi/synthbuster-plus/tree/dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5), review-blocked | no license declared |

Community Forensics Eval, NTIRE, the social panel, competition COCO/DALL-E
assets, Data Draft, and any WildFake addendum are outside this gate. Without written
permission there is no download, transformation, benchmark, demo, or
submission use of those datasets.

## Directory inference

Run a selected cached checkpoint offline against a local directory. The output
is a JSON array whose objects contain exactly `image_path` and `pred`; paths are
relative and `pred` is finite in `[0, 1]`. Diagnostics must use a separate file.

```bash
HF_HUB_OFFLINE=1 pixi run --platform linux-aarch64-cuda infer -- \
  --model ateeqq_siglip \
  --input path/to/images \
  --output work/submission/predictions.json \
  --diagnostics work/submission/diagnostics.json \
  --device cuda:0 \
  --cache work/hf-cache
```

## Cleanup and release blockers

Preview cleanup at any time; it writes `work/cleanup-inventory.json` and deletes
nothing:

```bash
pixi run --platform linux-aarch64-cpu cleanup
```

Only at the end of the competition, after reviewing that inventory, an
operator may explicitly delete the confined safe targets and write
`outputs/data-deletion-attestation.json`:

```bash
pixi run --platform linux-aarch64-cpu python scripts/cleanup.py delete --confirm
```

Do not run confirmed cleanup during ordinary development or verification.

The technical benchmark gate is not a submission approval. Check the release
scope with:

```bash
pixi run --platform linux-aarch64-cpu python scripts/compliance.py check \
  --scope submission --models all --dataset sid_set
```

Release remains blocked while any model has `submission_status = "review"`,
the current problem brief has not been human-verified, the repository is not
public, or free judging access has not been human-verified. These blockers must
not be bypassed by editing the registry or release flags without evidence.

## Claim and privacy limits

This is a fixed-threshold technical baseline and robustness error analysis. It
does not authenticate an image, cover every detector or generator, establish
performance outside the pinned data and conditions, or resolve undocumented
training-data overlap. Positive-only legacy smoke tests are not generalization
estimates and are not part of the controlled gate.

No identity recognition, face matching, EXIF-based person identification, or
external lookup of depicted people is permitted. Reports use opaque sample IDs
and omit secrets, usernames, tokens, and private local paths.

## Published result scopes

The small frozen confirmation run documented below is a separate 80-image
cross-platform check, not the basis for selecting a submission model. Its
results are:

- clean balanced accuracy: **0.925**;
- clean ROC-AUC: **0.994375**;
- worst balanced accuracy: **0.825** after 0.25x downscaling;
- external positive-only recall: DALL-E 3 **10/10**, Midjourney **10/10**, SDXL **9/10**.

The same frozen evaluator ran on `spark-a916` using an NVIDIA GB10 and CUDA 13.0. It completed 1,230 predictions in 37.99 seconds. All binary decisions matched the local CPU reference; color-jitter ROC-AUC differed by 0.000625 while its balanced accuracy and confusion counts matched.

No checkpoint was fine-tuned and no threshold was fitted on the evaluation
samples. The full 2,000-image-per-condition public-gate report is
[`outputs/public-baseline-robustness-report.md`](outputs/public-baseline-robustness-report.md).
Its fixed ordering places `wkaandemir_clip`, `ateeqq_siglip`, and
`frontier_community_forensics` in the displayed top three, but it explicitly
reports the winner as unresolved because the relevant confidence-interval
comparisons are not all conclusive. It is technical evidence only: every model
is still marked `submission_status = "review"` in `models.toml`.

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
