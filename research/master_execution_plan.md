# Master execution plan: robust detection of AI-generated images

Version 1.0 — 27 August 2026
Scope: TikTok TechJam 2026, Track 5
Primary constraint: detector must contain fewer than 2 billion parameters

## 0. Executive objective

Build and evaluate a reproducible detector for **fully AI-generated still images** from a declared panel of current generator families versus **non-generative images**, including edited, denoised, filtered, recompressed, reposted, screenshot, human-created digital-art, and conventional rendered images.

The project will not claim universal detection. It will:

1. create content-aligned real/AI reference groups;
2. identify scalar and multivariate signatures associated with generation;
3. separate signatures shared across generator families from model-specific fingerprints;
4. measure signature survival under the six required transformations;
5. measure non-additive and order-dependent effects of compound transformations;
6. train one compact supervised detector using only justified improvements;
7. convert error analysis into mathematically prioritised data collection or methodology changes;
8. freeze the complete method before using the provided COCO val2017/DALL·E Advanced evaluation data;
9. return calibrated `AI-generated`, `likely non-generative`, or `uncertain` decisions while still emitting the required numeric `pred` score.

The research contribution is the **measurement and reasoning framework**, not only the final benchmark number.

---

## 1. Frozen problem definition

### 1.1 Primary labels

`Y = 0 — non-generative/real`

Includes:

- native camera photographs;
- photographs edited using conventional tools;
- denoised, sharpened, tone-mapped, HDR, filtered, cropped, or colour-graded photographs;
- repeatedly resized or recompressed images;
- screenshots;
- human-created illustrations and digital art;
- procedural graphics;
- conventional CGI and 3D renders;
- images that look artificial but have no generative-AI origin.

`Y = 1 — fully AI-generated`

Includes images whose complete visual content was synthesised by an image-generation model from text, noise, or multimodal conditioning. Caption-only text-to-image is the primary generation regime.

### 1.2 Edge-case labels

Keep these out of the primary binary training/evaluation claim unless sufficient data exists:

- `partial_ai`: inpainting, object replacement, generative fill, or local AI edits;
- `ai_assisted`: ambiguous workflows combining human and generative operations;
- `recaptured_ai`: AI image displayed/printed and photographed;
- `unknown_provenance`: source cannot be established reliably;
- `invalid`: corrupt, unsupported, or unusably small input.

Evaluate `partial_ai` and `recaptured_ai` as secondary slices. Do not silently merge them into the primary label.

### 1.3 Claimed operating scope

The headline claim must name:

- generator families used during training;
- generator families held out for testing;
- real-image source families;
- transformations and severity ranges;
- whether results are native-file or canonical-pixel results;
- minimum validated input resolution;
- whether partial edits and recapture are excluded;
- the threshold operating point and uncertainty coverage.

Approved claim template:

> The system detects fully generated images from the stated generator mixture against a broad non-generative mixture, under the specified digital post-processing conditions, and its transfer is measured on held-out generator and data-source families.

Prohibited claims:

- “proves an image is authentic”;
- “detects all AI images”;
- “absence of a watermark means real”;
- “universal” without a genuinely independent multi-family test;
- “generator-unseen” if that provider/family influenced model or signal selection.

---

## 2. Research questions and preregistered hypotheses

### RQ1 — Existence

Which measurable real-versus-AI contrasts exist after content is approximately aligned?

`H1:` At least one probe family has a non-zero paired effect on held-out base images after false-discovery control.

### RQ2 — Commonality

Which contrasts are shared across generators rather than model-specific fingerprints?

`H2:` A low-dimensional subspace estimated from training generator families captures a positive fraction of the paired contrast of a held-out generator.

### RQ3 — Transformation survival

How do JPEG, blur, resize, noise, colour jitter, and crop alter each signature?

`H3:` Low-level residual and magnitude-spectrum signatures contract faster under blur/resize/JPEG than semantic signatures, while colour signatures contract most under colour jitter.

### RQ4 — Compound behaviour

Are transformation effects additive and order-independent?

`H4:` At least one ordered transform pair has a statistically non-zero interaction or order effect.

### RQ5 — Training benefit

Does content-aligned paired supervision improve robustness or reduce content bias?

`H5:` BCE plus paired ranking improves held-out-generator or worst-group performance over BCE alone without materially worsening clean performance.

### RQ6 — Operational reliability

Can instability and input quality identify unsafe predictions?

`H6:` A fixed uncertainty policy lowers accepted-sample error relative to forced binary classification at useful coverage.

All hypotheses are tested on internal data. The provided evaluation set is for frozen external confirmation only.

---

## 3. Required deliverables

### 3.1 Code and model

- training command;
- evaluation command;
- transformation-generation command;
- signal-analysis command;
- error-analysis command;
- inference command accepting an image directory;
- inference JSON containing exactly `image_path` and `pred` for required submission output;
- optional extended JSON with status, uncertainty, and evidence for the demo;
- model checkpoint below 2B parameters;
- locked dependency file and configuration files;
- automated checks for leakage, transformations, metrics, and inference schema.

### 3.2 Research artifacts

- dataset inventory and licence manifest;
- split manifest by `base_id` and source/generator family;
- prompt and generation manifest;
- native and canonical evaluation manifests;
- signal-survival tables;
- shared/model-specific signature analysis;
- single and compound transform results;
- ablation table;
- calibration and risk-coverage plots;
- representative error cards;
- limitations and scope statement;
- final external evaluation report.

### 3.3 Hackathon artifacts

- public repository;
- README with setup, reproduction, data, model, limitations, and contributions;
- Devpost narrative;
- compact clean-versus-transformed table;
- error-analysis note;
- architecture/method diagram;
- three-minute demo script and video.

---

## 4. System architecture

```text
Licensed real sources
        │
        ├── provenance + licence audit
        ├── base-image deduplication
        ├── real subtype annotation
        └── content caption / structured prompt
                         │
                         ▼
               Current generator panel
                         │
             clean content-aligned groups
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     Native-file track          Canonical-pixel track
            │                         │
            └────────────┬────────────┘
                         ▼
          single transforms + ordered pairs + chains
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
       Signal probes             Learned detector
            │                         │
            ├── paired effects        ├── calibrated score
            ├── survival/alignment    ├── multi-view stability
            ├── shared SVD            └── uncertainty decision
            └── interactions
                         │
                         ▼
               grouped error analysis
                         │
                         ▼
           risk-prioritised data/method iteration
                         │
                         ▼
                freeze → locked tests
```

---

## 5. Repository and artifact layout

Use the smallest structure that cleanly separates immutable data, code, and outputs:

```text
project/
  README.md
  pyproject.toml
  uv.lock
  .gitignore
  configs/
    data.yaml
    generators.yaml
    transforms.yaml
    model.yaml
    experiments/
  data/
    manifests/
      assets.parquet
      prompts.parquet
      lineage.parquet
      splits.parquet
      licences.csv
      external_hashes.txt
    raw/                 # not committed
    canonical/           # not committed
    cache/               # reproducible, not committed
  src/
    audit_data.py
    generate.py
    transforms.py
    probes.py
    train.py
    calibrate.py
    evaluate.py
    error_analysis.py
    infer.py
    schemas.py
  tests/
    test_splits.py
    test_transforms.py
    test_metrics.py
    test_infer.py
  artifacts/
    checkpoints/
    metrics/
    figures/
    error_cards/
    reports/
  outputs/
    predictions.json
```

Raw images, API credentials, generated images, and model checkpoints stay out of Git unless hosting rules and licences explicitly allow them.

---

## 6. Tooling and reproducibility defaults

### 6.1 Runtime

- Python 3.11 or 3.12;
- PyTorch with the installed CUDA-compatible build;
- torchvision for deterministic image transforms;
- Pillow for decoding/encoding;
- NumPy, SciPy, pandas or Polars, scikit-learn;
- statsmodels only if needed for mixed-effects fitting;
- timm/Transformers/OpenCLIP for pretrained encoders;
- matplotlib/seaborn for static figures;
- pydantic or dataclasses for manifest validation;
- `uv` for dependency locking.

Avoid adding a service dependency for experiment tracking. Store each run as:

```text
artifacts/metrics/<run_id>/
  config.json
  environment.json
  metrics.json
  predictions.parquet
  checkpoint.sha256
```

### 6.2 Determinism

Record:

- global seed;
- split seed;
- generation replicate or provider seed;
- transform seed;
- model initialisation seed;
- package versions;
- GPU and precision;
- API model snapshot/version and request date;
- prompt text and all exposed generation parameters;
- SHA-256 of every input/output file;
- configuration hash;
- checkpoint hash.

Full GPU determinism is not required if it causes major performance loss, but repeat the selected training configuration with three seeds before final claims.

### 6.3 Secrets

- load API credentials from environment variables;
- never store keys in manifests, notebooks, logs, screenshots, or Git;
- redact provider request headers and account identifiers;
- stop generation immediately when the authorised budget is reached.

---

## 7. Data manifest contract

Every asset must have one manifest row with these fields:

### Identity and lineage

- `asset_id`: unique immutable ID;
- `base_id`: original reference group ID;
- `lineage_id`: generation/transformation lineage;
- `parent_asset_id`;
- `sha256`;
- `phash`;
- `split`;
- `created_at`.

### Label and subtype

- `label`: `real`, `ai_full`, `partial_ai`, `recaptured_ai`, `unknown`, `invalid`;
- `real_subtype`: camera, edited_photo, recompressed, screenshot, illustration, procedural, cgi_render, other;
- `ai_subtype`: text_to_image, image_conditioned, reconstruction, other;
- `label_evidence`;
- `label_confidence`.

### Source and rights

- `source_dataset`;
- `source_uri`;
- `source_creator` where required;
- `source_upload_date` where available;
- `licence`;
- `licence_uri`;
- `attribution_required`;
- `allowed_for_training`;
- `allowed_for_public_demo`.

### Image properties

- width, height, aspect ratio;
- decoded colour mode and ICC profile;
- file format and file size;
- EXIF present;
- alpha present;
- orientation;
- estimated blur/quality/compression indicators.

### Generator properties

- provider;
- model family;
- exact model/version/snapshot;
- access mode: official API, local weights, approved interactive workflow;
- prompt ID and exact prompt;
- negative prompt if supported;
- provider seed or replicate ID;
- image-conditioning flag and strength;
- guidance, steps, sampler where exposed;
- requested/returned resolution;
- provider moderation/filter outcome;
- C2PA/watermark status when known.

### Transformation properties

- `processing_track`: native or canonical;
- transform chain in execution order;
- parameters for every step;
- random seed;
- encoder/library version;
- output format and JPEG quality where applicable.

Manifest validation failure blocks the asset from training or evaluation.

---

## 8. Real-image collection plan

### 8.1 Real distribution strata

Target a deliberately broad real mixture:

| Stratum | Target share | Purpose |
|---|---:|---|
| Native camera photographs | 30% | Establish real capture distribution |
| Conventionally edited/denoised photographs | 20% | Prevent “processed means AI” shortcut |
| Recompressed/reposted images | 15% | Model normal web degradation |
| Screenshots/UI/documents/graphics | 10% | High-risk false positives |
| Human illustration/digital art | 10% | Non-photographic hard negatives |
| Conventional CGI/3D renders | 10% | Synthetic-looking but non-generative hard negatives |
| Procedural/scientific/other graphics | 5% | Broaden non-generative support |

Adjust these weights only using the intended product context, not test performance.

### 8.2 Source policy

Use only:

- public or properly licensed datasets;
- public-domain, CC0, or appropriately attributed CC material;
- self-created/captured material with documented provenance;
- conventional renders with documented non-generative workflow.

Prefer sources uploaded before 2021 for web-derived hard negatives where provenance is otherwise uncertain. This reduces contamination by modern text-to-image systems without excluding conventional editing.

Candidate source roles:

- SID-Set/OpenImages real images for licensed training material;
- established camera datasets for native-capture images;
- pre-2021 Wikimedia Commons material with machine-readable licences;
- Blender Open Movie or other explicitly licensed conventional renders;
- approved CC0/CC-BY illustration and procedural-art sources;
- self-created screenshots and conventional edits.

Do not use COCO in training, development, calibration, or internal testing. This preserves the provided COCO val2017 real subset as a genuinely source-external evaluation.

### 8.3 Conventional processing expansion

Derive additional real-labelled variants from licensed real images using conventional operations:

- denoise;
- sharpen;
- tone curve/gamma;
- white balance;
- saturation/contrast filters;
- crop/rotate;
- screenshot capture;
- standard resize and recompression;
- repeated web-style recompression.

These remain real-labelled because their provenance is non-generative. Record their lineage so derived versions never cross splits.

### 8.4 Contamination audit

- inspect source date and provenance;
- scan metadata for declared generative tools;
- search exact/perceptual duplicates against known AI assets where available;
- manually review a stratified sample;
- mark ambiguous samples `unknown_provenance` and exclude them from primary training;
- never relabel an image solely because the detector thinks it is AI.

---

## 9. Content-aligned generation plan

### 9.1 Current primary generator panel

Preferred coverage:

1. OpenAI GPT-Image-2;
2. Midjourney V8.2;
3. Google Imagen 4;
4. Adobe Firefly Image Model 5;
5. ByteDance Seedream 5.0 Pro;
6. Black Forest Labs FLUX.2;
7. FLUX.2 Klein 4B open weights;
8. Stable Diffusion 3.5 Medium/Large;
9. Qwen-Image if compute permits.

No unofficial API, scraping, or terms-violating automation is permitted. If a model has no approved scalable access, use an approved interactive sample set or move it to evaluation-only coverage.

### 9.2 Minimum executable six-family panel

Default if access/budget is limited:

- training/development families: Imagen 4, Seedream 5.0 Pro, FLUX.2 Klein, Stable Diffusion 3.5;
- completely held-out internal families: GPT-Image-2 and Midjourney V8.2;
- completely external family: provided DALL·E Advanced data.

If a listed provider cannot be accessed lawfully/reproducibly, replace it with Firefly Image Model 5 or FLUX.2 API. Record the substitution before model training.

### 9.3 Prompt construction

For each real reference:

1. obtain existing licensed caption if available;
2. generate a structured content description using one fixed captioning workflow;
3. include subject, count, approximate layout, setting, lighting, viewpoint, style/medium, and major colours;
4. remove proper names, private data, and unnecessary trademarks;
5. avoid provider-specific prompt engineering;
6. use the same base prompt across providers;
7. store exact prompts and revisions.

Default prompt format:

```text
Create a [medium/style] image showing [subjects and counts] in [setting].
Composition: [layout and viewpoint].
Lighting: [lighting].
Important visible details: [details].
Colour and atmosphere: [description].
Do not add borders, captions, logos, signatures, or watermarks.
```

Provider-added watermarks are not removed; they are recorded and analysed separately.

### 9.4 Match-quality policy

Measure, but do not optimise away, content mismatch using:

- pretrained embedding cosine similarity;
- optional segmentation/layout similarity;
- perceptual similarity where spatial correspondence exists;
- aspect-ratio/resolution difference;
- manual review of a small stratified sample.

Assign `match_quality = high`, `medium`, or `low` using thresholds fixed on development data.

- Use high/medium pairs for the main paired signal analysis.
- Retain low-match generations in ordinary detector training/evaluation as realistic text-to-image outputs.
- Report sensitivity to match-quality strata.

### 9.5 Seed/replicate design

- use at least three seeds/replicates per model-reference pair in the signal-discovery anchor panel;
- use one or two replicates in the scale-training panel;
- record provider seed when supported;
- when seed control is unavailable, store a monotonically numbered replicate ID;
- never split replicates of one `base_id` across partitions.

### 9.6 Cost-efficient balanced design

Do not generate every reference with every model at scale.

Use:

**Anchor panel**

- 300–500 development references balanced across content and real subtype;
- every training/development generator;
- three replicates each;
- used for cross-model signature comparison and variance decomposition.

**Scale panel**

- 3,000–10,000 training references depending budget;
- assign each reference to two generator families using a balanced incomplete block design;
- one replicate per assignment by default;
- ensure every generator and pair of generators has comparable content coverage.

**Held-out panel**

- 400–1,000 own locked-test references from unseen real sources;
- outputs from the two held-out generator families;
- two replicates where possible;
- generated/stored without entering training or development manifests.

This design yields broad training coverage while preserving a smaller fully crossed panel for mathematical comparison.

---

## 10. Split protocol

### 10.1 Partitions

| Partition | May train weights? | May choose features/method? | May set threshold? | May inspect errors? |
|---|---:|---:|---:|---:|
| Training | Yes | Yes | No | Yes |
| Development/error-analysis | No | Yes | No | Yes |
| Calibration | No | No | Yes | Aggregate only |
| Own locked test | No | No | No | Only after final metrics |
| Provided external evaluation | No | No | No | Only after final metrics; no subsequent model claim |

### 10.2 Grouping rules

- split real `base_id` before captioning, generation, editing, or transformation;
- group exact and near-duplicate families into one split;
- all AI outputs derived from a real reference inherit its split;
- all conventional edits and transforms inherit the original split;
- hold out whole real-source families for own test;
- hold out whole generator families for own test;
- do not use COCO anywhere before external evaluation;
- do not use DALL·E outputs anywhere before external evaluation.

### 10.3 Default proportions

Within training/development sources:

- 70% training base groups;
- 15% development/error-analysis base groups;
- 15% calibration base groups.

Own locked test is collected from separate source families and is not carved randomly from this pool.

### 10.4 Leakage checks

Before every run:

- no shared SHA-256 across splits;
- no perceptual-hash distance below configured threshold across splits;
- no shared lineage or `base_id` across splits;
- no train/dev source URI in own test;
- no held-out generator family in training or development;
- no COCO or DALL·E family in train/dev/calibration/own test;
- no provided external hashes in any earlier partition;
- no label encoded in filename passed to the model.

Failure of any leakage check stops training.

---

## 11. Canonical image processing

### 11.1 Decoder policy

- apply EXIF orientation;
- decode to a defined RGB colour space;
- composite alpha against a fixed neutral background and record it;
- reject corrupt files with reason;
- never use filenames or metadata as learned inputs;
- retain metadata separately for provenance analysis.

### 11.2 Native track

Preserve provider/source pixels and dimensions after decoding. Use model preprocessing only at inference. This track measures realistic delivery-pipeline evidence.

### 11.3 Canonical track

- decode all images identically;
- convert colour consistently;
- save through one controlled lossless path before experimental transformations;
- use content-preserving size handling defined in config;
- strip file metadata from the pixel-only copy while retaining it in the manifest.

Native-versus-canonical performance difference estimates the contribution of service/export/file shortcuts.

---

## 12. Transformation specification

Implement transformations once in a shared module used by training and evaluation.

### 12.1 Singles from the brief

**JPEG compression**

- qualities: 90, 70, 50, 30;
- encode/decode RGB using a fixed library/version;
- record chroma-subsampling default or set explicitly;
- return decoded pixels to the model.

**Gaussian blur**

- sigma: 0.5, 1.0, 2.0;
- kernel size: next odd integer at least `6σ + 1`;
- fixed border behaviour.

**Resize**

- downscale factors: 0.5 and 0.25;
- bicubic interpolation with antialiasing enabled;
- upscale back to original decoded dimensions using the same declared interpolation;
- retain aspect ratio exactly.

**Gaussian noise**

- sigma: 0.02, 0.05, 0.10 in `[0,1]` pixel scale;
- independent channel noise unless the brief specifies otherwise;
- clip to `[0,1]` after addition;
- deterministic per asset/recipe seed.

**Colour jitter**

- brightness, contrast, and saturation factors sampled from `[0.8, 1.2]`;
- training: random factors;
- evaluation: fixed published factor triplets and seeds;
- fixed operation order or explicitly varied order in the compound study.

**Centre crop**

- default interpretation: retain 80% of each spatial dimension, then resize to original dimensions;
- keep this parameter configurable because “crop 80%” can mean 80% area;
- ask the workshop organisers for the intended definition; update only the config and document the interpretation.

### 12.2 Label symmetry

- use identical parameter distributions for real and AI classes;
- do not make compression quality, output format, or transform count class-dependent;
- derive deterministic seeds from `asset_id + recipe_id`, not the class label;
- in paired analysis, match transform parameters across the real/AI pair while using independently generated stochastic noise fields unless an explicit same-noise ablation is run.

### 12.3 Ordered pairs

At middle severity use:

- JPEG 70;
- blur σ=1.0;
- resize 0.5;
- noise σ=0.05;
- one fixed moderate colour-jitter triplet;
- crop 80%.

Evaluate all `C(6,2)=15` unordered pairs in both orders: 30 conditions.

### 12.4 Realistic chains

Freeze at least these recipes:

1. `crop80 → resize0.5 → JPEG70`;
2. `colour_jitter → resize0.25 → JPEG30`;
3. `blur1.0 → resize0.5 → JPEG50`;
4. `noise0.05 → resize0.5 → JPEG50`;
5. `JPEG90 → JPEG70 → JPEG50`;
6. `colour_jitter → crop80 → resize0.5 → JPEG70`.

### 12.5 Random-chain holdout

- sample length 2–4;
- no repeated transform except repeated JPEG recipe;
- sample severities from the stated ranges;
- store recipe before inference;
- reserve a fixed seed family that training never uses.

### 12.6 Storage policy

- clean originals are immutable;
- training transforms run on the fly;
- evaluation recipes are generated deterministically and cached only when it saves time;
- transformed images never become new independent observations.

---

## 13. Signal probe bank

Implement transparent diagnostics before adding architecture complexity.

### 13.1 File/provenance diagnostics

- format, dimensions, colour profile, EXIF/software tags;
- C2PA validation where supported;
- known watermark detector where authorised;
- never feed these into the passive pixel classifier;
- absence never implies real.

### 13.2 Spatial/residual probes

- high-pass residual energy;
- neighbouring-pixel correlations horizontally/vertically/diagonally;
- local variance and kurtosis;
- residual autocorrelation peak strength and location;
- patch-wise distribution summaries.

### 13.3 Frequency probes

- 2D Fourier log-magnitude summary;
- radial energy bands;
- angular energy bands;
- spectral slope;
- phase-coherence summaries;
- DCT coefficient histograms/block periodicity;
- residual-spectrum versions of the above.

### 13.4 Colour probes

- channel means/variances/skewness;
- RGB/Lab cross-channel covariance;
- saturation and clipping rates;
- chroma/luma frequency summaries.

### 13.5 Learned/reconstruction probes

- pretrained encoder embeddings;
- frozen linear-probe margin;
- diffusion/autoencoder reconstruction error on a resource-limited subset;
- optional second-order reconstruction difference inspired by DID;
- multi-crop score variance.

### 13.6 Probe acceptance

A candidate signal is retained only if:

- discovery effect is non-trivial;
- validation effect has the same direction;
- false-discovery-rate-adjusted evidence is reported for large probe families;
- it generalises to at least one held-out content/source slice;
- its transformation behaviour is measured;
- it is not explained entirely by native-file metadata or format.

---

## 14. Mathematical analysis specification

### 14.1 Paired contrast

For probe vector `φ`:

`δ_{i,m,s,t} = φ(AI_{i,m,s,t}) − φ(Real_{i,t})`.

Estimate mean `μ_{m,t}` and covariance `Σ_{m,t}` using base-image-grouped resampling.

### 14.2 Scalar effect

`d' = mean(δ) / sd(δ)`.

Report median or mean, effect size, and clustered-bootstrap 95% interval.

### 14.3 Multivariate separability

`Q_{m,t} = μᵀ(Σ + λI)⁻¹μ`.

Choose shrinkage/regularisation on development data. Supplement with MMD and grouped permutation where Gaussian assumptions are poor.

### 14.4 Retention and direction

Using clean metric `W₀=(Σ₀+λI)⁻¹`:

`ρ_t = ||μ_t||_{W₀}/||μ₀||_{W₀}`

`κ_t = <μ_t,μ₀>_{W₀}/(||μ_t||_{W₀}||μ₀||_{W₀})`

Report both. Do not collapse amplitude and direction into one unexplained number.

### 14.5 Difference-in-differences

`DID_t = E[(φ(T(AI))−φ(T(Real))) − (φ(AI)−φ(Real))]`.

### 14.6 Compound interaction

`Γ_{A,B}=μ_{A→B}−μ_A−μ_B+μ₀`.

### 14.7 Order effect

`Ω_{A,B}=μ_{A→B}−μ_{B→A}`.

### 14.8 Shared signature subspace

- form contrast matrix from training generators;
- whiten using development real/paired covariance;
- compute SVD;
- choose component count using development explained variance and held-out development families only;
- project completely held-out generators without refitting;
- report shared fraction and residual model-specific fraction;
- repeat after transformations to measure shared-subspace survival.

### 14.9 Hierarchical variance

Fit probe or detector margin with:

`z = β₀ + β_model + β_transform + β_model×transform + b_base + u_base×model + ε_seed`.

Report variance attributed to:

- content/base image;
- generator;
- seed;
- transform;
- model×transform interaction;
- residual.

Use these estimates to decide whether the next budget should buy more references, models, seeds, or transform coverage.

---

## 15. Model-selection plan

### 15.1 Parameter constraint

Count all inference-time learned parameters, including heads and auxiliary branches. Save the count in the model card. Reject any configuration at or above 2B.

### 15.2 Candidate backbones

Pilot only:

- DINOv2 ViT-B/14 class;
- OpenCLIP ViT-B/16 class.

Both are far below the limit. Train a frozen linear head on the same pilot data and select one using this ordered criterion:

1. macro AUROC across development generator families;
2. worst real-subtype FPR;
3. worst single-transform balanced accuracy;
4. calibration;
5. inference latency/memory.

Do not carry both into the full pipeline unless results are effectively tied and an ensemble is shown to earn its cost.

### 15.3 Input policy ablation

Compare on pilot data:

- resize-to-input;
- native-resolution random/centre crop;
- two-crop inference.

Prefer cropping when it preserves forensic traces and fits the image. Select the largest input resolution that maintains a stable effective batch size on available hardware. Default fallback is 224px crops; upgrade to 448/504 only when measured benefit justifies compute.

### 15.4 Training stages

**B0 — frozen encoder + linear head**

- establishes minimal baseline;
- no augmentation beyond required crop/normalisation.

**B1 — partial fine-tuning**

- unfreeze final blocks or use a parameter-efficient adapter;
- BCE only;
- class-balanced sampling.

**B2 — paired ranking**

`L = L_BCE + λ_pair log(1+exp(−[s(AI)−s(Real)]))`.

Use `λ_pair` from a small development sweep such as `{0.1, 0.3, 1.0}`.

**B3 — transformation-aware training**

- random clean or transformed view;
- label-symmetric singles and random chain lengths;
- preserve a meaningful clean-image probability, default 0.35;
- optional consistency penalty only within validated severities.

**B4 — group balancing/DRO**

- groups: real subtype, generator family, and transform family;
- compare balanced group sampling with group DRO;
- keep only if worst-group development performance improves.

### 15.5 Training defaults

Starting values, adjusted only on development data:

- optimiser: AdamW;
- head learning rate: `1e-3` for frozen probe;
- fine-tuned backbone learning rate: `1e-5`;
- weight decay: `0.05`;
- effective batch size: 32 or largest stable equivalent;
- mixed precision;
- 5% warmup;
- maximum 10 epochs with early stopping;
- selection metric: mean of generator-macro AUROC and transform worst-group balanced accuracy, with real FPR used as a guardrail;
- three final training seeds for the selected configuration.

### 15.6 Optional forensic branch gate

Do not build a spectral/residual branch initially. Add one only if:

- probe analysis identifies a shared signal surviving important transforms;
- it is not already captured by the selected encoder;
- a small branch improves worst-group balanced accuracy by at least 2 percentage points or has a positive paired-bootstrap interval;
- clean performance falls by no more than 1 point;
- latency remains demo-feasible.

Otherwise retain the single backbone.

---

## 16. Calibration and inference harness

### 16.1 Calibration

- fit temperature scaling using only own calibration data;
- freeze temperature before own locked test;
- report Brier score, ECE/reliability curve, and class-wise calibration;
- never recalibrate on the provided dataset.

### 16.2 Fixed multi-view inference

Default views:

1. canonical model input from full image;
2. centre crop;
3. two deterministic overlapping crops;
4. one mild downscale/re-encode diagnostic view.

Compute:

- median logit or probability;
- median absolute deviation;
- score range;
- number of decision flips.

Keep the view set fixed after development.

### 16.3 Three-way decision

On calibration data select:

- `τ_high`: AI threshold satisfying the chosen real FPR target;
- `τ_low`: likely-real threshold satisfying the chosen accepted-real error target;
- `u_max`: maximum permitted view disagreement;
- minimum validated resolution/quality envelope.

Decision:

```text
if score >= τ_high and instability <= u_max and input is in-domain:
    AI-generated
elif score <= τ_low and instability <= u_max and input is in-domain:
    likely non-generative
else:
    uncertain
```

Required submission `pred` remains the calibrated AI probability. The demo may additionally expose the three-way status.

### 16.4 Provenance/watermark harness

- verify known C2PA credentials separately;
- run known watermark detectors only where authorised;
- valid positive evidence may support AI attribution;
- missing/invalid/negative evidence never produces a real decision;
- passive classifier results must be reported independently from this channel.

---

## 17. Experiment matrix and decision gates

### 17.1 Mandatory baselines

1. simple frozen linear probe;
2. fine-tuned BCE detector;
3. paired-ranking detector;
4. best detector with transformation-aware training;
5. forced binary versus uncertainty policy.

### 17.2 Mandatory evaluation tracks

1. clean in-distribution;
2. own unseen base images;
3. own unseen real sources;
4. own unseen generator families;
5. every single transform/severity;
6. all 30 ordered middle-severity pairs;
7. fixed realistic chains;
8. held-out random chains;
9. native versus canonical;
10. provided external evaluation after freeze.

### 17.3 Keep/discard rule

Keep a method change only when, on development data:

- generator-macro or worst-group performance improves with positive paired evidence or at least a practically meaningful predefined gain;
- real-image FPR does not materially worsen;
- clean balanced accuracy falls no more than 1 point unless worst-chain gain clearly compensates;
- calibration and latency remain acceptable;
- improvement occurs across more than one training seed.

Discard speculative components that only improve overall accuracy or one seen generator.

### 17.4 Freeze rule

Freeze when:

- all mandatory ablations are complete;
- data/error iteration budget is exhausted or the last iteration fails the keep rule;
- selected model has three-seed results;
- calibration thresholds are locked;
- all tests pass;
- repository inference contract works;
- model/config/checkpoint hashes are recorded.

After freeze, no model or threshold change is permitted based on own locked-test or provided-set results.

---

## 18. Metrics and statistical reporting

### 18.1 Primary metrics

- AUROC;
- average precision;
- balanced accuracy;
- TPR at 1% and 5% real FPR where sample size permits;
- real FPR;
- AI FNR;
- Brier score;
- ECE/reliability;
- prediction flip rate;
- worst-group performance;
- clean-to-transform delta;
- risk-coverage curve for uncertainty;
- latency, memory, and parameter count.

### 18.2 Aggregation

Report:

- micro aggregate;
- macro by generator family;
- macro by real subtype;
- macro by transform family;
- worst generator;
- worst real subtype;
- worst transformation/chain;
- native and canonical separately.

Declare mixture weights before evaluation. Do not allow the larger DALL·E class in the provided set to dominate the story through raw accuracy.

### 18.3 Confidence intervals

- bootstrap `base_id`, not transformed files;
- keep all descendants and seeds together during bootstrap;
- use paired bootstrap for method comparisons;
- use grouped permutation for probe two-sample tests;
- report 95% intervals;
- correct large exploratory probe families using Benjamini-Hochberg FDR;
- confirm selected signals once on locked/held-out generators without refitting.

### 18.4 Sample-size targets

For headline group proportions:

- approximately 384 independent base images gives about ±5 percentage-point worst-case 95% precision;
- approximately 1,068 gives about ±3 points.

Use at least 400 base images for each primary held-out generator family if feasible. Use hierarchical shrinkage for smaller secondary groups rather than overinterpreting raw rates.

---

## 19. Error-analysis system

### 19.1 Error cohorts

Generate cohorts on development/error-analysis data:

1. stable false positives;
2. stable false negatives;
3. transformation-induced flips;
4. high-confidence errors;
5. unstable multi-view cases;
6. held-out-generator failures;
7. compound super-additive failures;
8. order-sensitive failures.

### 19.2 Error record

Each record contains:

- base and transformed image references;
- label and provenance evidence;
- real subtype or generator/version;
- content cluster;
- clean and transformed scores;
- multi-view score distribution;
- transform recipe and order;
- quality/resolution/format;
- probe values and changes;
- hypothesised failure mechanism;
- counterfactual ablation result;
- proposed response;
- whether response was accepted.

### 19.3 Counterfactual tests

- grayscale;
- low-pass/high-pass views;
- patch shuffle;
- common JPEG applied to both classes;
- native versus canonical;
- equalised resolution and format;
- metadata stripped;
- crop location changes;
- content-match stratification;
- provider watermark/provenance separated;
- conventional-render and digital-art hard-negative comparison.

### 19.4 Group posterior

For `e_h` errors among `n_h` base images:

`p_h | data ~ Beta(e_h+0.5, n_h−e_h+0.5)`.

Report posterior mean and interval.

### 19.5 Data-acquisition priority

`A_h = π_h × q95(p_h) × novelty_h / sqrt(cost_h)`.

Use this to rank new-data collection. `π_h` is the declared deployment relevance, not the observed test frequency alone.

### 19.6 Iteration budget

Permit at most two major error-driven collection/retraining cycles during the hackathon:

1. baseline error analysis and first targeted collection/method change;
2. robust-model error analysis and final targeted change.

Every change is evaluated on fresh development holdout data. Own locked test and provided data remain untouched.

### 19.7 Method-response rules

| Finding | Response |
|---|---|
| Frequent real subtype drives FPs | Add licensed hard negatives from that subtype; group-balance training |
| Unseen generator drives FNs | Add generator diversity only if it is not a locked test family; otherwise narrow claim |
| High content variance | Improve captions/pairing or semantic stratification |
| High seed variance | Generate more replicates for affected families |
| Large model×transform interaction | Add balanced transform/model sampling or uncertainty routing |
| Signal attenuated but aligned | Improve sensitivity/training exposure |
| Signal direction reverses | Disable/downweight it under that condition or abstain |
| Native strong, canonical weak | Report delivery-pipeline dependence; prevent shortcut claims |
| Multi-view disagreement predicts error | Tighten uncertainty policy |
| No consistent gain | Make no change |

---

## 20. Locked internal and provided external evaluation

### 20.1 Before accessing either locked layer

- final checkpoint hash recorded;
- code/config hash recorded;
- thresholds/calibration frozen;
- transform recipes/seeds frozen;
- data leakage checks passed;
- evaluation script dry-run on synthetic fixtures;
- output schema validated;
- no manual model selection remains.

### 20.2 Own locked test

Purpose:

- unseen real sources;
- unseen base images;
- two unseen generator families;
- unseen random chain seeds.

Run once for official results. Post-hoc error inspection is allowed for the limitations section, but any subsequent system change creates a new project version and invalidates these as final untouched-test results.

### 20.3 Provided evaluation dataset

Composition from brief:

- 4,998 non-AIGC COCO val2017 images;
- 8,843 DALL·E Advanced images.

Rules:

- never train on it;
- never use it for feature discovery;
- never use it for checkpoint/model selection;
- never set thresholds or calibration from it;
- never use its errors to collect data before the final reported run;
- never use COCO or DALL·E in earlier partitions if claiming source/provider externality;
- run clean images and, after method freeze, deterministic transformed copies for external robustness analysis;
- preserve originals unchanged;
- report the exact frozen checkpoint/config hashes.

This evaluation demonstrates external source and generator-family transfer only because COCO and DALL·E were excluded earlier.

### 20.4 Final hidden evaluation

Treat competition scoring data as completely unknown. The inference script must not assume:

- balanced labels;
- fixed resolution;
- JPEG only;
- metadata availability;
- a known generator;
- RGB without alpha/orientation;
- a writable input directory.

---

## 21. Automated verification

### 21.1 Data tests

- manifest schema validation;
- file existence and hash validation;
- no corrupt decoded assets;
- no cross-split base/lineage/duplicate leakage;
- class-symmetric transform distributions;
- no COCO/DALL·E before external phase;
- licence field present for every training/demo asset.

### 21.2 Transformation tests

- deterministic repeat gives identical output;
- output dimensions are correct;
- JPEG quality levels produce expected file ordering/quality trend;
- blur kernel/sigma correct;
- resize returns original size;
- noise standard deviation within tolerance before clipping;
- crop geometry correct;
- chain order changes output for known noncommuting fixture;
- real/AI parameter sampler has identical distribution.

### 21.3 Mathematics tests

Use synthetic arrays with known effects to verify:

- paired contrast;
- Mahalanobis regularisation;
- retention/alignment;
- DID;
- compound interaction equals zero under additive fixture;
- order effect detects noncommuting fixture;
- grouped bootstrap samples base IDs;
- posterior calculations.

### 21.4 Model/inference tests

- parameter count below 2B;
- CPU inference on one small fixture if feasible;
- GPU inference where available;
- corrupt/unsupported images handled without aborting directory run;
- output includes every valid input exactly once;
- `pred` is finite and in `[0,1]`;
- required JSON schema exactly matches submission contract;
- secrets and local paths absent from logs/report.

---

## 22. Resource and access plan

### 22.1 Required authorisations

Before paid or external generation:

- explicit API spending cap;
- approved provider credentials;
- confirmation that automated use complies with provider terms;
- storage quota;
- GPU availability;
- permission to publish selected generated/licensed examples.

No paid API call is made without the authorised cap.

### 22.2 Default budget controller

Configure:

```yaml
generation_budget_usd: <authorised amount>
warning_fraction: 0.80
hard_stop_fraction: 1.00
per_provider_caps: {...}
```

At 80%, stop scale-panel expansion and preserve budget for held-out evaluation generation. At 100%, stop all paid generation.

### 22.3 Compute tiers

**No/limited GPU**

- frozen embeddings and linear head;
- transparent probes;
- 224px inputs;
- no reconstruction-heavy branch.

**Single 16–24GB GPU**

- ViT-B partial fine-tuning;
- mixed precision and gradient accumulation;
- 224px or measured feasible crop size;
- on-the-fly transforms.

**Single 48–80GB GPU or equivalent**

- larger crop resolution;
- larger batches;
- expanded seed/model analysis;
- reconstruction probes on larger subset.

The project must remain runnable in the middle tier.

### 22.4 Storage controls

- calculate expected storage before generation;
- prefer lossless clean master plus on-the-fly transforms;
- cache only locked evaluation derivatives;
- deduplicate exact files by content hash;
- preserve generation outputs even if low match quality, subject to rights;
- never delete immutable raw data during active experiments.

---

## 23. Execution sequence and dependencies

### Phase 0 — Charter and preflight

Tasks:

- freeze definitions and scope;
- confirm competition rules and allowed preparation;
- inventory compute, storage, credentials, budget, and licences;
- create repository and locked environment;
- implement manifests and test fixtures.

Exit criteria:

- scope approved;
- environment reproducible;
- no missing blocking credentials for minimum panel or fallback chosen;
- automated smoke test passes.

### Phase 1 — Data audit and split creation

Tasks:

- collect licensed real sources;
- assign real subtypes;
- verify provenance/licences;
- hash/deduplicate;
- create base/source group splits;
- reserve own locked-test sources;
- exclude COCO and DALL·E.

Exit criteria:

- target strata represented;
- no cross-split leakage;
- every asset has licence and lineage;
- own test is inaccessible to training configuration.

### Phase 2 — Captioning and generation

Tasks:

- create structured captions/prompts;
- run pilot across accessible generators;
- validate output manifests and costs;
- generate anchor and scale panels;
- generate/store held-out panel separately;
- quantify content-match strata.

Exit criteria:

- minimum generator coverage achieved;
- anchor panel has required replicates;
- budget reserve remains for held-out evaluation;
- prompts and exact model versions recorded.

### Phase 3 — Transformation engine

Tasks:

- implement all singles;
- implement ordered-pair and chain recipes;
- implement native/canonical tracks;
- run deterministic and symmetry tests;
- create evaluation manifests.

Exit criteria:

- all transform unit tests pass;
- recipe hashes frozen;
- no label-dependent parameters.

### Phase 4 — Data exploration and signal discovery

Tasks:

- plot source/format/resolution/content by label;
- run shortcut audits;
- compute signal probes;
- estimate paired effects and variance components;
- compute shared signature subspace on development generators;
- run single-transform survival analysis;
- run ordered-pair interactions on an anchor subset.

Exit criteria:

- candidate signal list fixed for validation;
- obvious shortcuts documented/removed or isolated;
- transform survival and interaction tables available;
- held-out generators untouched.

### Phase 5 — Baseline and model selection

Tasks:

- frozen linear probes for candidate backbones;
- select one backbone/input policy;
- fine-tune BCE baseline;
- calibrate only on calibration split when configuration is fixed;
- produce baseline error cohorts on development data.

Exit criteria:

- reproducible baseline checkpoint;
- required metrics and latency recorded;
- failure strata ranked mathematically.

### Phase 6 — Error-driven iteration 1

Tasks:

- inspect top risk-contribution development cohorts;
- run causal counterfactuals;
- select one data repair and one methodology change at most;
- likely choices: paired ranking and targeted hard negatives;
- retrain and apply keep/discard rule.

Exit criteria:

- accepted changes have evidence;
- rejected changes removed;
- no test-set feedback used.

### Phase 7 — Transformation-aware robustness

Tasks:

- add class-symmetric transformation training;
- compare singles versus random chains;
- test group balancing/DRO only if worst groups remain poor;
- evaluate signal and classifier interaction/order effects;
- implement multi-view uncertainty.

Exit criteria:

- clean, singles, pairs, chains, and risk-coverage results complete;
- final architecture remains minimal;
- external data untouched.

### Phase 8 — Error-driven iteration 2

Tasks:

- repeat posterior/risk contribution on robust development results;
- make one final targeted change only if keep criteria are met;
- run three seeds for final candidate;
- select final model and thresholds.

Exit criteria:

- method, calibration, transforms, and thresholds frozen;
- hashes recorded;
- all automated checks pass.

### Phase 9 — Locked evaluation

Tasks:

- run own locked test once;
- run provided external clean evaluation once;
- run frozen external transformation recipes;
- compute all metrics and intervals;
- inspect errors only after metrics are archived;
- make no model changes under the same claimed version.

Exit criteria:

- internal and external reports immutable;
- generalisation axes stated accurately;
- limitations written from observed failures.

### Phase 10 — Submission and demo

Tasks:

- final inference packaging;
- reproduction test in clean environment;
- README and Devpost write-up;
- figures/tables/error cards;
- demo UI only if it exposes evidence clearly;
- record three-minute video;
- public repository audit for secrets and rights.

Exit criteria:

- inference command produces required JSON;
- setup works from README;
- public assets are licensed;
- demo fits time;
- all deliverables submitted.

---

## 24. Three-day hackathon schedule

If all work must occur during the event, use this compressed sequence.

### Day 1 — trustworthy data and baseline

Hours 0–3:

- initialise environment, manifests, tests, and access;
- freeze scope and model panel.

Hours 3–10:

- collect/audit real data and create grouped splits;
- start captioning and generation jobs;
- implement transformation engine concurrently.

Hours 10–18:

- finish pilot anchor panel;
- run EDA/shortcut audit;
- train frozen probe and BCE baseline.

Hours 18–24:

- baseline evaluation;
- first error cohorts and priority ranking;
- decide first targeted change.

Day 1 exit:

- leakage-free baseline works end-to-end;
- required inference schema exists;
- at least four training/development generator families represented.

### Day 2 — signal mathematics and robustness

Hours 24–34:

- compute probes, paired contrasts, shared signature SVD, and single-transform survival;
- train paired-ranking candidate.

Hours 34–44:

- transformation-aware training;
- ordered pairs and realistic-chain evaluation;
- calculate DID, interaction, and order effects.

Hours 44–52:

- second error analysis;
- targeted hard-negative/data or group-balancing iteration;
- multi-view uncertainty and calibration.

Day 2 exit:

- final candidate chosen using development data;
- clean/transform/error-analysis story supported mathematically;
- no provided data accessed.

### Day 3 — freeze, external evaluation, and communication

Hours 52–56:

- run final seeds/validation;
- freeze code, checkpoint, thresholds, and recipe hashes;
- complete automated checks.

Hours 56–61:

- own locked evaluation;
- provided external clean and transformed evaluation;
- archive immutable metrics.

Hours 61–67:

- final tables, plots, error cards, limitations;
- package inference and reproduce from clean setup.

Hours 67–72:

- record demo;
- finish Devpost and README;
- secret/licence/public-repository audit;
- submit.

---

## 25. Risk register and fallback actions

| Risk | Detection | Default response |
|---|---|---|
| Paid/API model unavailable | Access smoke test fails | Substitute approved Firefly/FLUX endpoint; record before training |
| Midjourney cannot be automated lawfully | No official scalable workflow | Use approved manually generated evaluation subset or replace; never scrape |
| Generation budget exceeded | Budget controller reaches 80/100% | Stop scale expansion; preserve held-out generation budget |
| Real data AI contamination | Provenance/date/metadata ambiguity | Mark unknown and exclude primary set |
| Provided-set leakage | Hash/source/family check fails | Remove contaminated training lineage and retrain from clean split |
| Poor content alignment | Similarity strata heavily low | Improve captioning; use high/medium for paired math but retain low for ordinary detection |
| Detector learns file format | Native strong, canonical weak | Equalise encoding for passive model; report delivery evidence separately |
| Clean performance high, unseen generator near chance | Group results diverge | Increase train generator diversity or narrow scope; do not tune on held-out family |
| Transform augmentation harms real FPR | Class-wise dev metrics worsen | reduce severity/probability or discard augmentation |
| Compound transforms erase evidence | high uncertainty/error | abstain outside validated envelope; report irrecoverable channel limitation |
| Reconstruction probe too expensive | runtime/storage exceeds cap | run on anchor subset only or drop it |
| Spectral branch adds no robust value | ablation fails gate | delete branch |
| Small groups produce unstable error rates | wide posterior intervals | hierarchical/Beta shrinkage and targeted evaluation sampling |
| Training instability | seed variance high | lower LR, increase effective batch, freeze more backbone layers |
| GPU unavailable | preflight failure | frozen encoder + linear head + probes |
| External score disappoints | frozen test result low | report honestly; diagnose after archive; no same-version retuning |
| Demo assets have rights concerns | licence audit fails | replace with CC0/self-created examples |

---

## 26. Final report structure

1. Problem and precise provenance-label policy.
2. Why content-aligned pairing controls semantics.
3. Generator and real-source coverage.
4. Data lineage, licences, and split isolation.
5. Information-channel view of transformations.
6. Signal probe taxonomy.
7. Paired effect, survival, alignment, and common-subspace mathematics.
8. Compact detector and training objectives.
9. Clean/single/compound/native/canonical results.
10. Calibration, uncertainty, and selective risk.
11. Error cohorts and causal ablations.
12. How error analysis changed data/methodology.
13. Own locked and provided external evaluation.
14. Limitations, including irrecoverable information loss.
15. Reproduction and inference instructions.

---

## 27. Demo narrative

### 0:00–0:30 — problem

Show the same real/AI content before and after repost-like transformations. Explain that clean accuracy is insufficient.

### 0:30–1:10 — method

Show content-aligned groups, signal probes, and shared/model-specific signature decomposition.

### 1:10–1:50 — transformation science

Show one signal-survival curve and one compound order interaction. Explain retention `ρ`, alignment `κ`, and why order matters.

### 1:50–2:25 — detector

Run directory inference. Show score, multi-view stability, and `uncertain` handling while producing required JSON.

### 2:25–2:50 — error feedback

Show one false-positive cohort, counterfactual ablation, mathematical risk contribution, and the resulting targeted data/method change.

### 2:50–3:00 — result and limits

Show clean, transformed, worst-chain, unseen-generator, and untouched external results. State exact scope and one honest limitation.

---

## 28. Definition of done

The project is complete only when all are true:

- label policy explicitly treats conventional edits/renders/recompression as real;
- all assets have provenance/licence/lineage metadata;
- no base or near-duplicate lineage crosses splits;
- COCO and DALL·E are absent before external evaluation;
- at least four generator families are used for training/development and at least one complete family is held out;
- all six required transformations are implemented and verified;
- compound pairs and realistic chains are evaluated;
- signal retention, direction, separability, and interaction are reported;
- shared versus model-specific signature analysis is performed;
- baseline and every retained change have ablations;
- error analysis causes either an evidence-backed change or a documented decision not to change;
- model parameter count is below 2B;
- calibration and uncertainty use only own calibration data;
- own locked and provided external data are evaluated only after freeze;
- required inference JSON is correct and reproducible;
- README, robustness summary, error note, demo, and public repository are complete;
- claims match the measured scope.

## 29. Immediate first actions

Execute in this order:

1. create the repository skeleton and manifest schemas;
2. run compute/storage/provider-access smoke tests;
3. freeze the six-family minimum panel and spending cap;
4. collect a 100-reference pilot spanning every real subtype;
5. generate two replicates per accessible training generator;
6. implement and verify the six transforms;
7. run the native/canonical shortcut audit;
8. train frozen linear baseline;
9. compute pilot paired effects and variance decomposition;
10. use pilot estimates to finalise data volume, seed allocation, and the anchor/scale panel sizes;
11. proceed through the phases without accessing the provided evaluation data.
