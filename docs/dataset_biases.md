  ## Dataset biases

  ### B1. Benchmark/test-set leakage

  AIGIBench is primarily an evaluation benchmark: its advanced generators, SocialRF, and CommunityAI subsets are test sets.
  Training on them would invalidate AIGIBench reporting and leak future-generator coverage.

  Required control:

  - Use only explicitly designated training partitions.
  - Never use organizer demonstration data.
  - Once a public subset is used for training, do not report it as unseen evaluation.
  - Verify every dataset’s licence.

  AIGIBench defines separate ProGAN/SD1.4 training settings and 23 advanced evaluation subsets.

  ### B2. Dataset-label confounding

  This occurs when one dataset contributes mostly real images and another mostly fake images:

  [
  P(Y\mid D)\gg P(Y).
  ]

  WildFake, for example, draws generated images from generation pipelines, Civitai, Midjourney, GitHub and Hugging Face, while
  authentic images come from COCO, FFHQ, ImageNet, LSUN, CelebA-HQ, AFHQ, LAION and Wukong. A detector can distinguish those
  source ecosystems instead of authenticity. WildFake construction

  Required control:

  - Sample dataset first, then label.
  - Obtain both labels from each source ecosystem where possible.
  - Cap each dataset’s batch contribution.
  - Record dataset ID and test whether it predicts the label.

  ### B3. Semantic-content bias

  Portraits, fantasy art, illustrations, product images or landscapes may be disproportionately fake or real. This
  particularly threatens the global DINO head.

  B-Free identifies content bias as one of the main reasons detectors fail across datasets. B-Free, CVPR 2025

  Required control:

  - Create 16–32 clusters using frozen, detection-naive DINO features.
  - Keep each cluster’s label ratio below approximately 2:1.
  - Generate API images from captions associated with authentic images.
  - Include both photographic and non-photographic authentic content if both can appear in evaluation.

  ### B4. Style bias

  Generated datasets may contain cinematic, polished, centered or illustration-like images, while authentic datasets contain
  ordinary photographs. Style is related to content but remains separately exploitable.

  Required control:

  - Track photographic, illustration, rendered, screenshot and artwork clusters.
  - Use varied prompts and API styles.
  - Include high-quality authentic photography and ordinary generated outputs.
  - Do not make “beautiful image” synonymous with fake.

  ### B5. Authentic-source/acquisition bias

  Authentic images from FFHQ, COCO and Open Images have characteristic camera, cropping and curation pipelines. AIGIBench, for
  example, uses FFHQ, CelebA-HQ and Open Images as its authentic set, while generated subsets come from very different
  pipelines. AIGIBench data construction

  Required control:

  - Use multiple authentic sources: cameras, web images, scanned material and social-media images.
  - Balance authentic sources during training.
  - Balance them again when constructing memory.
  - Evaluate every authentic source’s false-positive rate separately.

  ### B6. File-format and codec bias

  Real images may be JPEG while API images are PNG or WebP. Historical JPEG artifacts can survive even after later conversion.

  DDA specifically demonstrates that pixel alignment alone leaves frequency differences caused by compression history. DDA,
  NeurIPS 2025

  Required control:

  - Decode everything through the same RGB pipeline.
  - Track original format and estimated JPEG quality.
  - Match codec-quality distributions across labels.
  - Apply final re-encoding independently of the label.
  - Avoid double-JPEG compression affecting only one class.

  ### B7. Native-resolution and aspect-ratio bias

  Cropping everything to (224\times224) does not remove evidence of previous upsampling, downsampling or aspect-ratio changes.
  B-Free explicitly identifies resolution bias. B-Free

  Required control:

  - Record native width, height and aspect ratio.
  - Balance their distributions across labels.
  - Use exactly the same resize and crop implementation.
  - Simulate degradation before final (224\times224) preprocessing.

  ### B8. Platform/post-processing bias

  SynthWildX contains social-network imagery; WildFake mixes clean generated outputs and community images; AIGIBench includes
  SocialRF from X, Facebook and Reddit. Social images may have repeated compression, resizing or overlays, unlike clean
  authentic datasets. SynthWildX was assembled specifically to study in-the-wild and laundered data. SynthWildX paper

  Required control:

  - Do not combine social-media fakes exclusively with clean authentic images.
  - Collect authentic and fake images from the same platforms.
  - Otherwise apply matched platform-like journeys to both labels.
  - Track clean, web and social provenance separately.

  ### B9. Challenge-transformation bias

  If generated images receive more JPEG or resize augmentation, the model learns the augmentation.

  Required control:

  [
  T\perp Y.
  ]

  - Use the same transformation-family and severity distributions for both labels.
  - Apply identical journeys to paired DDA real/fake examples.
  - Include clean, single-transformation and chained-transformation groups.
  - Do not assume cropping always helps: AIGIBench found cropping gains could mainly improve real accuracy while fake accuracy
    stagnated or fell. AIGIBench findings

  ### B10. Generator-family imbalance

  Millions of Stable Diffusion images can overwhelm a small number of DALL-E, Midjourney, GAN or autoregressive images.

  Required control:

  sample generator family
  → sample generator/version
  → sample image

  Do not sample uniformly over all files. WildFake’s hierarchy explicitly distinguishes generator type, architecture, weights,
  versions and time. WildFake

  ### B11. Generator-version and configuration bias

  API version, sampler, step count, guidance scale, output resolution, LoRA, personalization and safety settings all change
  the generated distribution.

  Required control:

  - Record provider, model version, date, resolution and all exposed parameters.
  - Randomize supported parameters.
  - Do not use one configuration per provider.
  - Reserve later versions and configurations from training.

  ### B12. Prompt-template bias

  If every API image is generated from one LLM template, the global head can learn the corresponding compositions and visual
  vocabulary.

  AIGIBench generates prompts with Gemini across four predefined categories, demonstrating how prompt construction directly
  determines dataset content. AIGIBench prompt construction

  Required control:

  - Use authentic captions, human prompts and several LLM prompt templates.
  - Reuse the same prompt across multiple APIs.
  - Vary prompt length, specificity, language, style and negative prompts.
  - Keep prompt source independent of generator provider.

  ### B13. API-output pipeline bias

  Each provider may return a characteristic resolution, PNG encoder, colour profile, watermark, safety treatment or sharpening
  pipeline.

  Required control:

  - Save the original API output and metadata for auditing.
  - Feed the model only canonical RGB pixels.
  - Apply a common randomized final encoding journey.
  - Do not let every provider correspond to one resolution or codec.
  - Test whether API provider can be predicted from trivial image statistics.

  ### B14. Curation and aesthetic-selection bias

  Selecting only attractive or realistic generated images changes the fake distribution. Selecting obvious failures creates
  the opposite problem.

  AIGIBench explicitly applies aesthetic filtering and manual removal of obviously fake images. Chameleon was deliberately
  curated for high-fidelity, human-deceptive images, unlike benchmarks built from simple prompts. AIGIBench, AIDE/Chameleon

  Required control:

  - Define deterministic acceptance rules before generation.
  - Retain random draws rather than hand-picking outputs.
  - Record every rejection.
  - Apply comparable quality filtering to authentic images.
  - Include both ordinary and difficult images in each class.

  ### B15. Label and task-scope noise

  Hashtag-based collections such as #aiart may be mislabeled. They may also contain AI-edited, composited or upscaled images,
  while the challenge targets purely generated images.

  Required control:

  - Require reliable provenance for training labels.
  - Quarantine uncertain web images.
  - Separate pure generation, AI editing, compositing and unknown.
  - Do not silently label partially edited images as purely generated.
  - Uncertain web data may be useful for stress testing, but not clean supervision.

  ### B16. Duplicate and lineage leakage

  The same image may appear in WildFake, SynthWildX, AIGIBench, Civitai and API-derived collections. Crops and recompressions
  can evade hash matching.

  Required control:

  - Assign a master-image ID.
  - Use exact hashes, perceptual hashes and DINO/CLIP nearest-neighbour checks.
  - Keep all derivatives in one split.
  - Deduplicate across datasets, not just within each dataset.
  - Split authentic parents and generated derivatives together.

  ### B17. Watermark, overlay and metadata bias

  Community images may contain visible signatures, platform overlays or AI-community watermarks. The challenge prohibits
  watermark dependence.

  Required control:

  - Never expose EXIF, filenames or directory structure to the model.
  - Remove systematic borders and platform UI.
  - Either remove visible watermarks or balance similar overlays across labels.
  - Verify performance on watermark-free images.

  ### B18. Temporal bias

  New APIs and model versions differ from older training generators. Platform recompression pipelines also change.

  Required control:

  - Record collection and generation dates.
  - Keep a chronological validation split.
  - Reserve later generator/API versions.
  - Never randomly mix every version across train and evaluation.

  ## Two-head architecture biases

  ### M1. Authentic-memory coverage bias

  The memory may contain many COCO photographs but few documents, screenshots, paintings or unusual authentic images. Rare
  authentic content then receives a large residual.

  Required control:

  - Allocate memory quotas by authentic source, semantic cluster and quality state.
  - Do not let the largest real dataset dominate.
  - Preserve rare-real coverage even if it is inconvenient for class balance.
  - Interpret memory distance as finite support, not “probability of being fake.”

  ### M2. Memory quality-state bias

  A clean authentic memory may poorly reconstruct JPEG-30 or heavily resized authentic images.

  Required control:

  - Build memory from clean and transformed authentic endpoints.
  - Normalize residuals against authentic images of comparable observed quality.
  - Keep quality variables out of the direct classifier input.

  ### M3. Memory contamination bias

  One mislabeled generated image can teach the memory that its pattern is authentic. MIRROR reports that mixed real/generated
  memory performs substantially worse than real-only memory. MIRROR

  Required control:

  - Use only high-confidence, licensed authentic images.
  - Exclude uncertain social-media “real” images from memory.
  - Version and audit every memory source.

  ### M4. Memory capacity and sparsity bias

  Too little capacity undercovers authentic diversity; too much capacity can reconstruct generated anomalies. MIRROR reports
  an inverted-U relationship for memory size and top-(k) sparsity. MIRROR

  Required control:

  - Fix a bounded memory budget.
  - Balance slots across authentic groups.
  - Prevent one content cluster from consuming most prototypes.
  - Keep sparse retrieval so generated anomalies are not explained by arbitrary prototype combinations.

  ### S1. Phase-1/Phase-2 representation drift

  Memory is built using base DINO features, while LoRA changes the feature space during supervised training.

  Required control:

  - Freeze base DINO parameters.
  - Keep LoRA low-rank and regularized.
  - Anchor adapted real features near their base features.
  - Recalculate authentic residual calibration after LoRA training.
  - Treat the memory and LoRA checkpoint as one versioned pair.

  ### S2. Shared-encoder correlation bias

  The global and memory heads share DINOv3. They are not independent experts. If LoRA learns a JPEG or dataset shortcut, both
  heads may agree for the same wrong reason.

  Required control:

  - Do not multiply probabilities or treat agreement as independent confirmation.
  - Fuse standardized evidence using regularized logistic regression.
  - Include memory coverage and retrieval concentration as evidence.
  - Report the system as two correlated views of one representation, not an ensemble of independent detectors.

  ### S3. Pretraining-overlap bias

  Public images may already have appeared in DINO’s large pretraining corpus, which can make familiar authentic images
  unusually easy to represent.

  Required control:

  - Deduplicate against every locally available training source.
  - Prefer recent API generations and post-pretraining authentic collections for locked evaluation.
  - Avoid claiming that strong performance on old public benchmarks proves open-world generalization.