# Machine transfer handoff: public baseline robustness

Date: 2026-08-30 (Asia/Singapore)

## Authoritative Git state

- Repository: `https://github.com/BenyAlbatross/techjam-aigc.git`
- Branch: `public-baseline-robustness`
- Base branch: `xuan`
- Last recovery code commit before this handoff: `6826116`
- Governing plan: `docs/superpowers/plans/2026-08-30-public-baseline-robustness.md`
- Governing design: `docs/superpowers/specs/2026-08-29-public-baseline-robustness-design.md`

Tasks 1 through 7 are committed and reviewed. Task 8 reached complete inference
coverage, but the user stopped the final report process after approximately 34
minutes to move to another machine. The report process exited cleanly and wrote
no partial public report.

Three recovery commits after Task 7 are important:

- `57b5c4d fix: recover public baseline gate` fixes aggregate model fetching,
  exact registered parameter counts, safe UniVFD TorchScript loading, and the
  report validation CLI. It adds regression tests.
- `2f335a9 fix: preserve univfd torchscript preprocessing` retains the pinned
  model's TorchScript image metadata and tests the preprocessing contract.
- `6826116 fix: complete univfd torchscript compatibility` completes the safe
  OpenCLIP compatibility path and adds focused regression coverage.

Frozen model revisions, hashes, thresholds, label directions, preprocessing,
and dataset authorization were not weakened.

## Current local compute state

These ignored artifacts exist on the old machine and are not in Git:

| Directory | Approximate size | Contents |
| --- | ---: | --- |
| `work/data/` | 1.4 GB | Pinned SID_Set images used by the manifest |
| `work/hf-cache/` | 2.6 GB | Hash-verified model snapshots |
| `work/manifests/` | 972 KB | SID_Set 1,000-real/1,000-generated manifest |
| `work/predictions/` | 320 MB | Complete atomic prediction shards |

Prediction state at stop time:

- 10 models;
- 15 conditions per model;
- 150 JSONL shards;
- 300,000 total prediction rows;
- zero temporary shard files.

Do not commit or push these ignored artifacts. Transfer them privately if the
new machine should resume without downloading data and rerunning GPU inference.

## New-machine setup

The target machine is expected to match the current Linux AArch64 NVIDIA setup.

```bash
git clone https://github.com/BenyAlbatross/techjam-aigc.git
cd techjam-aigc
git fetch origin
git switch public-baseline-robustness

curl -fsSL https://pixi.sh/install.sh | bash
# Start a new shell if `pixi` is not yet on PATH.

pixi install --platform linux-aarch64-cpu
pixi install --platform linux-aarch64-cuda
pixi run --platform linux-aarch64-cpu test
pixi run --platform linux-aarch64-cuda cuda-check
```

The `.pixi/` environment is reproducible from `pixi.toml` and `pixi.lock`; do
not transfer it between machines.

If Hugging Face access requires authentication:

```bash
hf auth login
```

## Transfer ignored runtime artifacts

Use a private encrypted channel such as SSH/rsync. Replace placeholders with
the old host and checkout path:

```bash
mkdir -p work/data work/hf-cache work/manifests work/predictions
rsync -a --info=progress2 <old-host>:<old-checkout>/work/data/ work/data/
rsync -a --info=progress2 <old-host>:<old-checkout>/work/hf-cache/ work/hf-cache/
rsync -a --info=progress2 <old-host>:<old-checkout>/work/manifests/ work/manifests/
rsync -a --info=progress2 <old-host>:<old-checkout>/work/predictions/ work/predictions/
```

Verify transferred state before reporting:

```bash
pixi run --platform linux-aarch64-cpu python scripts/data_manifest.py validate \
  work/manifests/sid_set_1000x2.json
find work/predictions -type f -name '*.jsonl' | wc -l
find work/predictions -type f -name '*.tmp' | wc -l
find work/predictions -type f -name '*.jsonl' -print0 | xargs -0 wc -l | tail -n 1
```

Expected values are `150`, `0`, and `300000 total`.

If runtime artifacts cannot be transferred, use README's pinned fetch,
manifest, and sequential benchmark commands. That path recreates the same
state but reruns all GPU inference.

## Exact resume point

First rerun the full test suite on the new machine. The pre-recovery Task 8 run
recorded `121 passed`; recovery commits added focused tests, but the interrupted
worker did not finish writing its final evidence report before shutdown.

Then generate the report. This is CPU-heavy and had not completed after 34
minutes on the old machine:

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

After report generation:

1. Review uncertainty-supported winner and top-three resolution. Do not force a
   point-order winner when intervals remain unresolved.
2. Run the prohibited-claim search from Task 8 against README and the report.
3. Run submission compliance. Remaining human and model-submission blockers
   must stay visible.
4. Run `git diff --check` and the full network-free verification sequence.
5. Commit only README, `compliance.toml`, the generated Markdown report, and
   any reviewed code/test changes. Never commit local reports, images, model
   weights, caches, manifests, or prediction shards.
6. Run Task 8 review, whole-branch review, and branch-finishing workflow before
   merge.

## Pending team Hugging Face dataset request

The user requested adding the team's dataset after the current SID run:

- Repository: `techjam-aigc/wildfake-eval-subset`
- Observed revision: `9e1fade16b3f83c0d0dd2dba2d617e165e55b055`
- Configs: `default`, `normalized`, `laion_matched`, `cross_generator`

This request was not implemented or run before shutdown. The card says upstream
WildFake terms are research/non-commercial, the final test corpus overlaps, and
training on any config leaks. It also documents a perfect image-size shortcut
in `default`. Proposed handling was to register it as blocked from competition
benchmarking, use `laion_matched` as the primary future diagnostic,
`cross_generator` as a generalization probe, and keep `default`/`normalized`
non-ranking. Obtain explicit design approval and written competition-use
clearance before download or execution. Never train on it.

## Release blockers that remain intentional

- Every model still has `submission_status = "review"`.
- `problem_brief_verified = false`.
- `repository_public = false` until the remote visibility is human-confirmed.
- `free_judging_access_verified = false`.
- Community Forensics Eval remains blocked.
- NTIRE remains review-blocked.
- Confirmed cleanup has not run and must not run during development.
