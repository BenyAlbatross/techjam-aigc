# Corrected Dataset Flow And Candidate Audit

Audit date: 2026-08-30
Branch: `public-baseline-robustness`

## Raw results first

### Canonical shortcut-control panel

| Check | Result |
| --- | ---: |
| Source records | 2,000 |
| Real | 1,000 |
| AI | 1,000 |
| Output format | 2,000 PNG |
| Output geometry | 2,000 at 512×512 |
| Unique SHA-256 | 2,000 |
| Manifest validation errors | 0 |
| Profile | `rgb_png_c1_square_512_lanczos_v1` |
| Ranking eligible | No |

The native SID panel had 999 JPEG + 1 MPO real files versus 1,000 PNG AI files,
and every AI image was 1024×1024 while the real class had 206 dimension pairs.
The canonical panel removes file-container, metadata, color-mode, and geometry
separation by applying one class-symmetric decode/resize/encode path. It does
not remove inherited compression traces, source semantics, or the fact that one
source family maps to each label. It is therefore a diagnostic control, not a
new leaderboard.

### `Joshyxwa/data_draft` audit

| Check | Hub/API evidence | Assessment |
| --- | --- | --- |
| Pinned revision | `e800837135e10a2dca7cbfa49ecf0b5b68830537` | Reproducible |
| Repository access | Public, ungated | Conflicts with private-only card |
| Declared license | `other` | No aggregate reuse grant |
| Stated package | 10,000 PNG images | 5,000 SID + 5,000 WildFake |
| Stated split counts | train 7,000; dev 1,500; calibration 1,500 | Internally documented |
| HF generated rows | train 7,000; validation 1,500; total 8,500 | Calibration 1,500 absent from generated view |
| Restricted component | WildFake 5,000 | `licence_audit_required` |
| Verification file | 10,000 decode; balanced; zero reported cross-split collisions | Useful but self-reported aggregate only |

Decision: **blocked** for download, transformation, benchmarking, training,
demo, and redistribution. The card itself says to keep the repository private
and not use WildFake until its upstream licence review is complete. Making the
Hub repository public does not override those restrictions. The repository
should be made private immediately if its public visibility was accidental.

The 8,500-versus-10,000 discrepancy is a packaging defect rather than evidence
that rows are missing from the Git repository: Hugging Face maps `train` and
`dev` (`validation`) but does not expose the top-level `calibration` directory
as a generated split. Any future cleared revision should provide an explicit
dataset configuration mapping all intended splits.

### Candidate dataset decisions

| Dataset | Revision | Decision | Permitted role | Main limitation |
| --- | --- | --- | --- | --- |
| ELSA 1M Track 1 | `199bf769…` | Approved | AI-only external stress | No real class; no BA/AUROC or ranking alone |
| CIFAKE repack | `f67ae8d…` | Review-blocked | Possible low-resolution diagnostic | Third-party rights assertion; 32×32 and old SD 1.4 |
| AIGC Detection Benchmark repack | `c91d902…` | Review-blocked | Possible generator-breadth holdout | Per-source image rights/provenance absent |
| Synthbuster Plus | `dbfb72f…` | Review-blocked | Possible modern-generator holdout | No license declared |
| Community Forensics Eval | `7d4a74a…` | Blocked | None for competition | Non-commercial ShareAlike license |
| Data Draft | `e800837…` | Blocked | Local smoke test only after rights resolution | Mixed rights and public/private mismatch |

No candidate was silently upgraded into a ranking dataset. ELSA is the only new
approved source because its immutable card declares CC BY 4.0, but it contains
only generated images. It can measure generator-side false-negative rate and
transformation survival. It cannot be paired casually with arbitrary real
images because that would recreate the source/format confound.

## Corrected flow

1. Keep the completed native SID results as historical, shortcut-exposed data.
2. Use `canonicalize` to build an equal-format/equal-geometry SID control.
3. Rerun frozen models and published thresholds on that panel; compare paired
   native-versus-canonical decisions by `base_id`.
4. Use ELSA only as a pinned, sampled AI stress cohort and report FNR by model,
   generator, and transform. Do not compute or imply aggregate BA/AUROC.
5. Admit a ranking holdout only when both classes have compatible provenance,
   source/content coverage, processing, and competition-use rights.
6. Keep all review/blocked entries behind the compliance gate.

Command implemented:

```bash
pixi run python scripts/data_manifest.py canonicalize \
  work/manifests/sid_set_1000x2.json \
  --output work/manifests/sid_set_1000x2_canonical.json \
  --image-dir work/data/sid_set_canonical/images --size 512
```

## Remaining risks and automated checks

- High: class still equals SID source family. Require at least two independent
  real and two independent AI source families before ranking.
- High: generator mixture is undisclosed. Do not claim generator-unseen
  generalization from SID.
- High: `data_draft` is publicly visible against its own restriction. Change
  Hub visibility or remove restricted files.
- Medium: native SID selection is the first qualifying rows in a pinned stream,
  not a globally ID-sorted sample. Fix the README claim or replace selection
  with an auditable index-based sampler before refreshing the native panel.
- Medium: canonical square fitting changes composition. Preserve the native
  panel and report paired deltas; never present canonical results alone.
- Automated gates now cover path confinement, exact hashes, label validity,
  output-root confinement, canonical format/geometry, and duplicate canonical
  bytes on committed fixtures.

## Evidence links

- [Data Draft pinned card](https://huggingface.co/datasets/Joshyxwa/data_draft/blob/e800837135e10a2dca7cbfa49ecf0b5b68830537/README.md)
- [ELSA 1M Track 1 pinned card](https://huggingface.co/datasets/elsaEU/ELSA1M_track1/blob/199bf769e2ddb673d68442a9756212ddd204426a/README.md)
- [CIFAKE repack pinned card](https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible/blob/f67ae8dedee6bb83e7523f6ce3b715a12147a200/README.md)
- [AIGC Detection Benchmark pinned card](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark/blob/c91d9024a5a77ef06e2ec681b53f9caf08675663/README.md)
- [Synthbuster Plus pinned card](https://huggingface.co/datasets/marco-willi/synthbuster-plus/blob/dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5/README.md)
