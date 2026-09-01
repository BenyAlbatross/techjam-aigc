# Generator Coverage Benchmark Plan

## Objective

Our current benchmark suite already provides strong coverage of earlier image-generation paradigms through **WildFake**, while **SID-Set** adds substantial coverage of modern FLUX-based generation.

The next stage of evaluation should therefore prioritize **generator-family and generation-paradigm coverage**, especially for architectures that are poorly represented in the current suite.

The primary goal is not simply to increase the number of benchmark datasets. Instead, we want to minimize redundancy and cover distinct generations of image-generation technology.

Alongside coverage, **total dataset size is treated as a quantity to minimize**, not merely as a cost factor to note. The guiding formulation is:

> **Maximize generator/paradigm coverage subject to a minimal total image budget** — acquire the smallest number of images that still yields statistically stable per-generator metrics.

Statistical power sets the floor for this budget: for proportion-style metrics (accuracy, TPR/FPR), roughly **2,000–5,000 images per generator** already gives 95% confidence intervals of about ±1–2 percentage points. Acquiring more than that per generator buys almost no additional statistical precision — only storage and evaluation cost.

---

## 1. Operational Taxonomy of Generator Generations

The following generation taxonomy is used only as an operational framework for benchmark selection. It is not intended as a canonical historical classification.

| Generation | Approx. Era | Architecture Family | Representative Types | Current Coverage | Need for Additional Coverage |
|---|---|---|---|---|---|
| **Gen 1** | ≤2021 | **GAN** | ProGAN, StyleGAN, BigGAN, CycleGAN, GauGAN | **Strong — WildFake** | Low |
| **Gen 2** | 2020–2022 | **VAE / VQ / tokenized latent models** | VQ-VAE, VQGAN, related discrete-latent generators | **Strong — WildFake** | Low |
| **Gen 3** | 2021–2023 | **Classic diffusion / latent diffusion** | DDPM, DDIM, ADM, GLIDE, early Stable Diffusion | **Strong — WildFake** | Low |
| **Gen 4** | 2023–2025 | **Transformer / DiT-based modern diffusion** | SD3-class models, PixArt-class models, recent T2I systems | Partial | **High** |
| **Gen 5** | 2024– | **Flow-matching / rectified-flow / modern Transformer T2I** | FLUX-class models | **Strong for FLUX — SID-Set** | Medium |
| **Gen 6** | 2024– | **Autoregressive image generation** | autoregressive token / continuous image generators | Very limited | **Highest** |
| **Gen 7** | 2024– | **Unified multimodal generation** | unified image-language understanding/generation models | Very limited | **Highest** |
| **Gen 8** | 2025– | **Frontier proprietary / hybrid systems** | recent closed-source or architecture-undisclosed generators | Partial | **Very high** |

### Current Coverage Summary

Our existing benchmarks can therefore be summarized as:

- **WildFake:** broad coverage of **Gen 1–3**
- **SID-Set:** deep coverage of **Gen 5**, specifically FLUX
- **Major remaining gaps:** **Gen 4, Gen 6, Gen 7, and Gen 8**

The most important gaps are currently **autoregressive image generation** and **unified multimodal generation**.

---

## 2. Benchmark Dataset Comparison

| Priority | Dataset | Main Generations | Main Generator / Architecture Coverage | Approx. Scale | Overlap with WildFake | Marginal Coverage Added | Recommended Role |
|---:|---|---|---|---:|---|---|---|
| **Current** | **WildFake** | **Gen 1–3** | GAN, VQ/VAE-style generators, classic diffusion, latent diffusion, early T2I | Large-scale | — | **Broad classical generator coverage** | Core baseline |
| **Current** | **SID-Set** | **Gen 5** | FLUX / modern flow-era T2I | ~100k fully synthetic images | Low | **Deep modern-flow coverage** | Modern-generator anchor |
| **1** | **EvalGEN** | **Gen 5–7** | FLUX, autoregressive generators, unified/multimodal generators | ~55k synthetic images | **Low** | **Gen 6 autoregressive + Gen 7 unified generation** | **Highest-priority addition** |
| **2** | **AIGIBench** | **Gen 1–5 + specialized generation** | newer GANs, SD3-class models, FLUX, DALL·E-class systems, Imagen, Midjourney, personalized generation, community/social images | ~521k full benchmark | Medium | **Gen 4–5 breadth, recent commercial systems, personalized generation, unknown/community sources** | Capped subset (see Section 4) |
| **3** | **Human-AIGI** | **Gen 4–8** | modern diffusion, flow models, autoregressive generation, unified multimodal generation, recent proprietary systems | 30k+ generated images | Low | **Broad frontier-generation coverage** | Frontier stress test once available |
| **4** | **Chameleon** | Primarily Gen 3–5 | high-fidelity Web-collected AI imagery | ~26k | Medium | Primarily adds **artifact-poor / high-fidelity difficulty**, rather than a new architecture family | Hard-generation stress test |
| **5** | **SynthWildX** | Gen 3–5 | DALL·E-class, Firefly, Midjourney | ~2k | Medium | Modern commercial generators in a shared social-media domain | Small but useful sanity test |
| **6** | **WildRF** | Mixed / partially unknown | real-world social-media AI imagery | ~5.3k | Medium | Unknown-generator and deployment-domain coverage | Real-world generalization |
| **7** | **RRDataset** | Mixed modern generators | multiple generators followed by transmission / re-digitization pipelines | 20k masters + transformed views | Medium | Primarily adds **post-generation pipeline variation** | Robustness / redistribution evaluation |
| **8** | **BFree-Online** | Mixed / unknown | viral and redistributed AI imagery | ~1.4k | Medium | Real-world reposting and recompression | Small deployment stress test |
| **9** | **CO-SPY-Bench / in-the-wild** | Gen 3–5 | many modern generators + Web-collected synthetic images | ~50k wild synthetic images | Medium | Additional modern Web-generator breadth | Useful, but source-confound audit required |
| **10** | **Synthbuster** | Gen 3–4 | DALL·E, Firefly, Midjourney, Stable Diffusion family | ~10k | **High** | Mainly commercial-generator supplementation | Lower priority after AIGIBench |
| **11** | **GenImage** | Gen 1–3 | BigGAN, ADM, GLIDE, Stable Diffusion, Midjourney, VQ-family models | Million-scale | **Very high** | Primarily standard-benchmark compatibility | Excluded (size vs. marginal coverage) |
| **12** | **DRCT-2M** | Gen 3 | multiple diffusion variants | ~2M-scale | **Very high** | Depth within the diffusion family | Excluded (size vs. marginal coverage) |
| **13** | **UnivFakeDetect** | Gen 1–3 | GAN-to-diffusion cross-family evaluation | Tens of thousands | High | Historical cross-family control | Low marginal coverage |
| **14** | **AIGCDetect** | Gen 1–3 | legacy GAN and early diffusion generators | ~150k-scale evaluation | **Very high** | Legacy benchmark compatibility | Low priority; bias audit required |
| **15** | **COCO val2017 vs. DALL·E Advanced** | Likely Gen 4-class | DALL·E-family generator; exact version not yet verified | **13,841** total: 4,998 real + 8,843 synthetic | Medium | Target-like DALL·E-family distribution | Separate target/custom evaluation; source bias must be audited |

### Dataset Size as a Selection and Evaluation Factor

The datasets fall into four size tiers with distinct practical implications:

| Size Tier | Approx. Range | Datasets | Practical Implications |
|---|---|---|---|
| **XL** | ≥1M images | WildFake, GenImage, DRCT-2M | High storage and evaluation cost. Justified only when marginal coverage is high (WildFake). For high-overlap XL sets (GenImage, DRCT-2M), size is an additional argument **against** acquisition, not just neutral. |
| **L** | 100k–1M | AIGIBench (~521k), SID-Set (~100k synthetic), AIGCDetect (~150k) | Meaningful but manageable cost. Full evaluation feasible; per-generator subsets remain large enough for stable metrics. |
| **M** | 10k–100k | EvalGEN (~55k), CO-SPY wild (~50k), Human-AIGI (30k+), Chameleon (~26k), RRDataset (20k masters), COCO-vs-DALL·E (~13.8k), Synthbuster (~10k), UnivFakeDetect | Low cost. Full evaluation trivially feasible; per-generator counts are usually sufficient for point estimates but confidence intervals should be reported. |
| **S** | <10k | WildRF (~5.3k), SynthWildX (~2k), BFree-Online (~1.4k) | Negligible cost, but **limited statistical power**. Treat as qualitative stress tests; always report confidence intervals and avoid strong claims from these alone. |

Two consequences follow:

1. **Coverage efficiency (marginal coverage per image).** EvalGEN is the most efficient acquisition in the entire list: ~55k images buying two otherwise-uncovered generation paradigms (Gen 6 AR, Gen 7 unified). Conversely, GenImage and DRCT-2M are the least efficient: million-scale cost for coverage that is almost entirely redundant with WildFake. Under a size-minimization objective, these XL redundant sets are **excluded outright**.

2. **Size-aware aggregation.** Because per-dataset sizes span three orders of magnitude, any per-image (micro-averaged) overall score would be dominated by the largest sets. All headline metrics must be **macro-averaged**: first per generator, then per generation family, then across families. With per-generator caps in place (Section 4), sample counts become roughly uniform anyway, which makes macro- and micro-averages nearly coincide — a side benefit of the capped design.

---

## 3. Recommended Core Benchmark Suite

The recommended generator-coverage suite is:

| Coverage Layer | Dataset | Purpose |
|---|---|---|
| **Gen 1–3: classical breadth** | **WildFake** | Broad coverage of GAN, VQ/VAE-style, and classic diffusion generators |
| **Gen 5: modern-flow depth** | **SID-Set** | Large-scale FLUX evaluation |
| **Gen 4–5: modern breadth** | **AIGIBench** | Modern diffusion/DiT, commercial generators, personalized generation, and community sources |
| **Gen 5–7: emerging architectures** | **EvalGEN** | Adds **autoregressive and unified multimodal generation** |
| **Gen 4–8: frontier coverage** | **Human-AIGI** | Cross-generation stress test including recent proprietary systems |

This gives an approximately continuous coverage path:

**GAN → VQ/VAE → classic diffusion → modern Transformer/DiT → flow-based generation → autoregressive generation → unified multimodal generation → frontier proprietary systems**

---

## 4. Minimal-Footprint Acquisition Strategy

The default policy is **capped, stratified acquisition**, not full downloads:

1. **Per-generator cap.** Acquire at most **N = 5,000 fake images per generator** (stratified by sub-category where the dataset provides one, e.g., prompt source or resolution). For generators with fewer images, take all of them.
2. **Matched reals.** Acquire the dataset's own real images at a comparable count per dataset (reals must come from the same dataset to preserve the intended real-vs-fake pairing; do not substitute a shared external real pool, which would introduce source bias).
3. **Deterministic manifests.** Every subset is defined by a committed manifest file (file list + SHA hashes), sampled once with a fixed random seed. All detectors are evaluated on exactly the same manifest, and the manifest — not the download — is the reproducibility unit.
4. **Prune redundancy at acquisition time.** Generators already covered by an existing dataset are *skipped* rather than downloaded again, with one deliberate exception below.

### Retained cross-dataset controls

A small, controlled overlap is kept **on purpose** so that dataset/domain effects remain measurable, but at capped size (≤5k per control), for example:

- FLUX on **SID-Set** vs. FLUX on **EvalGEN**
- one or two classic-diffusion families on **WildFake** vs. the same families in **AIGIBench**

These controls are what allow us to distinguish **generator-family generalization** from **dataset/domain generalization** and **source/preprocessing effects** — they are the only redundancy worth paying for. All other overlapping generators are dropped at download time.

### Approximate acquired footprint under this policy

| Dataset | Full size | Acquired under cap (fake side, approx.) |
|---|---:|---:|
| WildFake | million-scale (already held) | keep on disk; evaluate via capped manifest only |
| SID-Set | ~100k synthetic | ~5–10k (single FLUX family; slight extra depth allowed) |
| EvalGEN | ~55k | ~30–55k (many generators, most under the cap already) |
| AIGIBench | ~521k | **~100k or less** (largest saving; cap per generator, skip WildFake-redundant families except designated controls) |
| Human-AIGI | 30k+ | full (already near cap per generator) |

Net effect: the added footprint for the entire expansion stays on the order of **~150–200k images** instead of ~700k+, with no meaningful loss of statistical precision.

---

## 5. Reporting Strategy

Results should be reported:

1. **Per dataset**
2. **Per generator**
3. **Per generation family**
4. **Per architecture family**

A useful aggregate reporting structure would be:

- **Gen 1–3:** classical generators
- **Gen 4:** modern Transformer / DiT diffusion
- **Gen 5:** flow-era generation
- **Gen 6:** autoregressive generation
- **Gen 7:** unified multimodal generation
- **Gen 8:** frontier / proprietary generation

This prevents a benchmark with many closely related diffusion generators from dominating the overall score and makes the generalization claim easier to interpret.

Consistent with the size analysis in Section 2, all aggregate scores should be **macro-averaged** (per generator → per generation family → overall) so that dataset size never acts as an implicit weight. For S-tier datasets (<10k images), report confidence intervals and treat results as stress-test evidence rather than headline metrics.

---

## 6. Current Priority

Given the existing **WildFake + SID-Set** coverage, the immediate acquisition priority is:

**1. EvalGEN — near-full under the per-generator cap** (~30–55k acquired; highest marginal coverage per image in the entire list)
**2. AIGIBench — capped subset** (~100k or less acquired out of ~521k; per-generator cap plus redundancy pruning, keeping designated cross-dataset controls)
**3. Human-AIGI — full once publicly available** (30k+; already near the cap per generator)

Dataset size reinforces this ordering: the two remaining XL-tier candidates (GenImage, DRCT-2M) combine the **highest acquisition cost with the lowest marginal coverage**, and under the size-minimization objective they are **excluded**, not merely deprioritized.

The remaining benchmarks should primarily be considered for **domain robustness, redistribution robustness, high-fidelity difficulty, or compatibility with prior literature**, rather than for closing major gaps in generator architecture coverage.
