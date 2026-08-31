# Joel's evaluation work — handover

Everything here was produced 30 Aug – 1 Sep 2026. It is an **independent evaluation** of the two
TRACE-RX detectors plus a from-scratch DINOv3 baseline, run across three datasets.

If you are picking this up cold, read in this order:

1. **[DEFINITIONS.md](DEFINITIONS.md)** — what every dataset, split and term means. Read this first;
   the names are genuinely confusing and several are near-homonyms.
2. **[FINDINGS.md](FINDINGS.md)** — every result, with the numbers.
3. **[OPEN-QUESTIONS.md](OPEN-QUESTIONS.md)** — what is unresolved and what I would do next.
4. **[HOW-TO-RUN.md](HOW-TO-RUN.md)** — reproduce or extend any of it.

---

## The one-paragraph summary

Both TRACE-RX models are **essentially immune to the official robustness transforms** — six stacked
transforms in random order cost under 0.005 AUROC — and both **fail badly on generators they did not
train on**, by 0.66 AUROC. Effort spent on transform robustness is aimed at the wrong axis by roughly
two orders of magnitude. Separately, TRACE-RX-Parallel's learned fusion gate **destroys 0.12 AUROC**
on out-of-distribution data: its memory branch alone (0.596) beats its shipped fused output (0.473)
and beats TRACE-RX-M (0.572).

## The three headline numbers

| what | number | where |
|---|---|---|
| Six stacked random transforms cost | **≤ 0.005 AUROC** | both models, both datasets |
| One unseen generator costs | **0.66 AUROC** | 0.997 trained vs 0.336 held-out |
| Parallel's fusion gate costs | **0.12 AUROC** off-distribution | memory 0.596 vs fused 0.473 |

## What is in here

```
evals/
  README.md              this file
  DEFINITIONS.md         every dataset / split / term explained
  FINDINGS.md            all results
  OPEN-QUESTIONS.md      unresolved threads + recommended next steps
  HOW-TO-RUN.md          reproduction instructions
  scripts/               the six evaluation scripts actually used
  acai-project/          the complete standalone project this work was done in:
                           src/acai/   supporting library (metrics, transforms, sealed scorecard)
                           runs/       every logged run + per-image predictions as parquet
                           docs/       the earlier DINOv3-baseline work and original plan
                           tests/      124 tests (see caveat below)
  docs/                  copies of the two key project docs, for reading without digging
```

Every table in FINDINGS.md is recomputable from `acai-project/runs/*.parquet`. If a number here
disagrees with a table, trust the parquet.

## Caveats to carry forward

- **I could not push this myself** — no GitHub credentials in my environment. This branch was
  assembled locally and handed over for pushing.
- **`acai/transforms.py` was replaced mid-project** by the team's own official-policy version.
  The older `acai` modules (`dataset.py`, `train.py`, `compose.py`, `overnight.py`,
  `build_transformed.py`) still import the pre-rewrite API and **will not run** until migrated.
  The six scripts in `scripts/` all use the current API and do run.
- **Nothing here was trained.** Every model run is inference under `torch.no_grad()` against
  published checkpoints.
