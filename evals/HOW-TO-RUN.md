# How to run

## Environment

```bash
# torch for the GB10 / aarch64 box this was developed on
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers timm scikit-learn scipy PyWavelets pandas pyarrow pillow huggingface-hub
```

## Getting the model code

The two models live on **different branches**, and their modules collide on `sys.path`
(see DEFINITIONS.md). Check both out separately:

```bash
curl -sL https://codeload.github.com/BenyAlbatross/techjam-aigc/tar.gz/refs/heads/feat/traincode \
  | tar xz && mv techjam-aigc-feat-traincode trace_m
curl -sL https://codeload.github.com/BenyAlbatross/techjam-aigc/tar.gz/refs/heads/feat/trace-rx-parallel \
  | tar xz && mv techjam-aigc-feat-trace-rx-parallel trace_p
```

Each script picks its branch from `sys.argv[1]` (`m` or `p`) and puts only that one on the path.

## Getting the data

```bash
hf download Joshyxwa/data_draft            --repo-type dataset --local-dir data/data_draft
hf download Joshyxwa/techjam2026           --repo-type dataset --local-dir data/techjam2026 \
    --include "images/calibration/**" "labels.csv"     # note: ** not *, or nested paths are missed
hf download techjam-aigc/wildfake-eval-subset --repo-type dataset --local-dir data/eval_subset
```

Model weights download automatically via `huggingface_hub` on first run.

## The six scripts

| script | what it does | cost |
|---|---|---|
| `eval_trace.py` | TRACE-RX-M on `data_draft`, clean | 10k images |
| `eval_parallel.py` | Parallel on `data_draft` + all 10 eval-subset configs | ~105k |
| `eval_subset.py` | TRACE-RX-M on all 10 eval-subset configs | ~95k |
| `eval_calib.py m` / `p` | both on calibration x 15 official conditions | 83,775 each |
| `eval_chains.py m` / `p` | both on calibration x chains of 1..6 | 33,510 each |
| `eval_chains_wf.py m` / `p` | both on `data_draft` WildFake x chains of 1..6 | 30,000 each |

Run one model per process:

```bash
python eval_chains.py m > chains_m.log 2>&1
python eval_chains.py p > chains_p.log 2>&1
```

Roughly 4–8 minutes each on a GB10.

## The supporting library (`acai/`)

| module | what it is |
|---|---|
| `metrics.py` | AUROC, AUPRC + base rate, EER, TPR@FPR, Brier/ECE, grouped bootstrap, DeLong |
| `transforms.py` | the team's official transform menu and composition policy |
| `data.py` | `data_draft` manifest loading (handles the JSONL-named-`.parquet` defect), canonicalisation, holdouts |
| `scorecard.py` | sealed evaluator — hashes itself and `metrics.py`, stamps the digest into every scorecard |
| `audit.py` | nuisance/shortcut probes, including the size-cheat baseline |
| `runlog.py` | per-run config/env/predictions/metrics logging |

`metrics.py` is the piece most worth reusing. Its DeLong implementation is validated against sklearn
for the AUROCs and against a 400-sample bootstrap for the standard error.

**Do not** import `dataset.py`, `train.py`, `compose.py`, `overnight.py` or `build_transformed.py`
without migrating them first — they predate the `transforms.py` rewrite.

## Two traps that cost me time

**Seeding.** My first chain sampler used
`int.from_bytes(asset_id.encode()[-8:], "little") % (2**31)`. Taking the last 8 bytes and reducing
mod 2^31 keeps only low-order characters, which for fixed-width IDs is constant padding — it
collapsed 3,000 assets onto **one** seed and skewed family coverage to 19/19/19/19/14/9.5%. Use a
full-string `blake2b` digest. Always print the coverage histogram before trusting a sampler.

**`df.transform` is a DataFrame method**, not your column. Attribute access silently returns the
bound method. Use `df["transform"]`.
