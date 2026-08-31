# Definitions

The names in this project are confusing and several are near-homonyms. This page is the glossary.
Read it before FINDINGS.md.

---

## Datasets (there are three, and they are easy to mix up)

### 1. `Joshyxwa/data_draft` — "the draft corpus"

10,000 images, 5,000 real / 5,000 AI. **Two unrelated archives glued together, 5,000 each.**

| | SID-Set half | WildFake half |
|---|---|---|
| real | 2,500 web photos, one undifferentiated bucket | 2,500 across 5 domains: ffhq, celebahq, afhq, church, imagenet |
| AI | 2,500, all lumped as `text_to_image` | 2,500 across 5 generators: adm, ddim, ddpm, imagen, gan_based |
| real sizes | 1024x768, 1024x683 (varied, non-square) | **200x200, every single one** |
| AI sizes | **1024x1024, every single one** | 256x256, 512x512, 224x224 |

Splits: `train` 7,000 / `dev` 1,500 / `calibration` 1,500.

**"WildFake from data_draft"** — used repeatedly below — means the 5,000-image WildFake half of this
corpus. Not the separate eval-subset dataset (see #3). This is the phrase to watch for.

**Known defect:** `manifest.parquet` is actually JSONL with a `.parquet` extension.
`pd.read_parquet` fails on it; use `pd.read_json(..., lines=True)`. The loader snippet in that
dataset's card is broken as written. `acai/data.py:load_manifest` handles both.

### 2. `Joshyxwa/techjam2026` — "the training corpus"

44,671 images. **This is what both TRACE-RX models were trained on** (their training manifests name
it explicitly, revision `fd6ff453`).

Splits: `train` 32,035 / `dev` 6,091 / **`calibration` 5,585** / `own_locked` 960.

**The `calibration` split** is the 5,585-image partition the dataset reserves for threshold and
probability calibration only — *not* for fitting weights. Confirmed by the team that it was never
used in training, so it is a legitimate held-out set, though still **in-distribution** (same corpus,
same generators).

Its composition, which matters for reading any number from it:

- 4,090 real / 1,495 AI → **prevalence 0.268**, not 0.5. AUPRC's random baseline is therefore 0.268.
- AI comes from 4 programs: `gpt_image_2` 589, `sdxl_1_0` 419, `flux_1_schnell` 298,
  **`gemini_flash_image` 189**.

### 3. `techjam-aigc/wildfake-eval-subset` — "the eval subset"

A separate benchmark repackaged from upstream WildFake + COCO/LAION. **Ten configs**, five base and
five `_transformed` twins:

| config | rows | contents |
|---|---|---|
| `default` (dir is `data/`) | 13,841 | 4,998 COCO + 8,843 DALL-E 3 |
| `normalized` | 13,841 | same images at 200x200 |
| `laion_matched` | 7,652 | LAION-5B vs DALL-E 3, both natively >=1024px |
| `cross_generator` | 5,494 | LAION vs DALL-E 3 / Midjourney v5 / SDXL / GigaGAN |
| `diverse` | 14,394 | present in the YAML but **not described in the card's prose table** |

Its own card warns: on `default`, `img.size == (200,200)` alone scores **AUROC 1.000**. Verified —
see FINDINGS.

**Do not confuse** "WildFake from data_draft" (#1) with "the wildfake-eval-subset" (#3). Different
images, different generators, different purposes. I verified they do not overlap: 0 SHA-256
collisions against our training set across 3,000 sampled rows per config.

---

## Models (there are two, plus my baseline)

### `techjam-aigc/trace-rx-m-v2` — "TRACE-RX-M"

DINOv2-base backbone (pinned revision `f9e44c81`) + a frozen 64x768 authentic-prototype memory +
a head that classifies directional reconstruction residuals and retrieval statistics.

- Shipped checkpoint is `s4_detector.pt`, **epoch 5, frozen encoder** (`encoder_mode: "frozen"`).
- Only 3.4 MB — it holds heads and memory, not the encoder, which its code downloads separately.
- Code lives on GitHub branch **`feat/traincode`** (module `techjam_aigc.trace_rx_m`).

### `albagon/trace-rx-parallel-techjam2026` — "TRACE-RX-Parallel"

**A genuinely different architecture**, not a variant. Two branches — a *global* patch-statistics
branch and the *memory* branch — combined by a learned 2-weight fusion gate. Exposes three outputs:
`logit` (fused), `global_logit`, `memory_logit`.

- Shipped as `best_detector.pt`, **epoch 8, frozen encoder**. `best_detector.pt` and
  `final_detector.pt` are **byte-identical**.
- Code lives on GitHub branch **`feat/trace-rx-parallel`** (module `techjam_aigc.trace_rx_parallel`).

> **Trap:** the `trace_rx_m` module exists on *both* branches and is nearly identical, which makes
> it look like there is only one model. The parallel model is a **separate module** that exists only
> on `feat/trace-rx-parallel`. Also, that branch **removed the `optimizer.mixed_precision` config
> field** that trace-rx-m's shipped `config.json` still contains — so if both branches are on
> `sys.path` at once, whichever shadows the other breaks the other's checkpoint load. **Load each
> model in its own process.** Every script here does.

> The repo's default branch is `xuan` and it does **not** contain the model code. Neither does `ben`.

### My DINOv3 baseline

Frozen DINOv3 ViT-B/16 (via `timm`, because `facebook/dinov3-*` is gated and 403s) + a linear or MLP
probe. Built before the TRACE-RX models were available. Superseded, but its shortcut-audit method is
still useful — see `docs/01-dinov3-baseline-log.md`.

---

## Terms

**Held-out generator.** The TRACE-RX training config declares
`held_out_generator_family: "gemini_flash_image"` with `held_out_min_roc_auc: 0.6`. One generator is
excluded from training entirely and the model must score >= 0.60 on it to count as valid. Chosen
because it is the smallest family (2,143 training images), so holding it out costs least.

**Both models failed that gate and shipped anyway** — 0.328 and 0.358. To their credit the failure is
recorded in the shipped `s4_validity.json` and stated on the model card.

**Why the gemini number shows up in calibration results.** It is not a separate experiment. 189 of
the calibration split's 1,495 AI images happen to be Gemini, so grouping the calibration results by
the `ai_subtype` column surfaces it. All four generator groups are equally novel *as images*; only
Gemini is novel *as a program*. That isolates the variable cleanly.

**Official transforms.** Six families, 14 settings: `jpeg` q90/70/50/30, `blur` sigma 0.5/1.0/2.0,
`resize` 0.5x/0.25x (then upscaled back), `noise` sigma 0.02/0.05/0.10 (normalised [0,1] units, not
8-bit levels), `jitter` +/-20% brightness/contrast/saturation, `crop` retain 80%. "15 conditions"
means these 14 plus clean.

**Official composition policy** (from the eval-subset card): 70% of rows get exactly one transform
assigned round-robin; 30% get two from different families; applied in the fixed physical order
`crop -> resize -> jitter -> blur -> noise -> jpeg`.

**"Uniform along 3 axes"** — my chain experiments, which deliberately depart from that policy:
chain *length* uniform over 1..6 (every image gets every length), *family* uniform (~16.7% each,
verified), and *order* uniform (all k! orderings equally likely). Verified empirically before running.

**Size-cheat baseline.** AUROC of a predictor that reads only the image's original width/height and
never opens the pixels. Reported next to every score because on several configs it beats the models.

**Margin over cheat.** Model AUROC minus that baseline. On data_draft the models beat a
two-integer manifest lookup by only 0.008–0.027, which is the honest way to read a 0.98.
