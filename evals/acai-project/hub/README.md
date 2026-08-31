# WildFake Eval Subset

Reference benchmark for the AIGC-detection track, repackaged from
[WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake) as parquet so it loads in one
line. Four configs: the spec-faithful set, plus three that remove artifacts which make the
spec-faithful set trivially gameable.

> [!WARNING]
> **Demonstration purposes only. Do not train on any config here.**
> These exist so you can sanity-check a model and track iterative improvements. They do **not**
> contribute to the final score, and the final test set is drawn from the same corpus — training
> on this leaks.

## Start here

**Access.** This repo is private to the `techjam-aigc` org. If `load_dataset` 401s, you are either
not a member or not logged in — ask an org admin for an invite, then `hf auth login`.

**Quick start.**

```python
from datasets import load_dataset

ds = load_dataset("techjam-aigc/wildfake-eval-subset", "laion_matched", split="validation")
# configs: default | normalized | laion_matched | cross_generator
```

**Three rules.**

1. **Do not train on any of this.** Not the images, not a subset, not "just for augmentation".
   The final test set comes from the same corpus, so training here leaks and your real score
   will not survive it.
2. **Report which config your number came from.** "0.98 AUC" is meaningless without it — the same
   model can score 1.00 on `default` and 0.75 on `laion_matched`.
3. **Never report accuracy alone on `default`.** It is 36% real / 64% fake, so predicting "fake"
   for everything scores 64%. Use AUC or balanced accuracy.

**Sanity-check yourself before you believe a good number.** Run this first:

```python
def cheat(img):
    return 0 if img.size == (200, 200) else 1
```

On `default` that scores **AUC 1.000** with no model at all. If your detector is near 1.00 on
`default` but falls to ~0.75 on `laion_matched`, it learned image size, not detection.

**Robustness.** Each config has a `_transformed` twin with JPEG/blur/resize/noise/jitter/crop
applied, and per-row columns saying which. Evaluation will use degraded images, so check your
degradation curve before submitting — see [Robustness transforms](#robustness-transforms).

**Suggested reporting template.**

| config | AUC | balanced acc | notes |
|---|---|---|---|
| `default` | | | spec compliance only — expect ~1.0, it means little |
| `laion_matched` | | | **the number to actually compare on** |
| `cross_generator`, per source | | | does it hold past DALL·E 3? |

**Known issue worth escalating.** In `default`, every real image is 200x200 and no fake image is,
so the two classes are perfectly separable without looking at content. This is upstream WildFake
preprocessing. If the **final test set** shares it, the leaderboard will rank resolution detectors
rather than AIGC detectors — worth raising with the organizers before tuning against it.

## Which config to use

| config | rows | contents | resolution | use it for |
|---|---|---|---|---|
| `default` | 13,841 | 4,998 COCO val2017 + 8,843 DALL·E 3 | as upstream | matching the official spec exactly |
| `normalized` | 13,841 | same images | 200x200 | the same benchmark without the size giveaway |
| `laion_matched` | 7,652 | 3,826 LAION-5B + 3,826 DALL·E 3, both natively >=1024px | 512x512 | **the most meaningful number** |
| `cross_generator` | 5,494 | 1,500 LAION + DALL·E 3, Midjourney v5, SDXL, GigaGAN | 256x256 | does it generalize past DALL·E? |

Every config above has a `_transformed` twin holding the **same rows** with robustness
transformations applied — `default_transformed`, `normalized_transformed`,
`laion_matched_transformed`, `cross_generator_transformed`. Same images, same labels, same
counts; see [Robustness transforms](#robustness-transforms).

```python
from datasets import load_dataset

ds = load_dataset("techjam-aigc/wildfake-eval-subset", "laion_matched", split="validation")
# omit the config name to get `default`
```

**If you only run one, run `laion_matched`.** `default` is reported for spec compliance, but see
below for why its headline number means nothing on its own.

## Read this before trusting any score

The classes in `default` are **100% separable by image size**, no model required:

| | count | dimensions |
|---|---|---|
| COCO val2017 (real) | 4,998 | **every** image is exactly 200x200 |
| DALL·E 3 (fake) | 8,843 | **none** is 200x200; min side-max 346, max 3056 |

```python
def cheat(img):
    return 0 if img.size == (200, 200) else 1   # AUC 1.000, learns nothing
```

This is upstream WildFake preprocessing — its COCO copies are downscaled to 200x200 while the
DALL·E 3 images keep native resolution — not an artifact of this repackaging. The same applies to
everything in WildFake's `Typical` trees; the `Advanced` trees keep native resolution.

### How much shortcut survives in each config

Single trivial features, no learning. 1.000 = perfect shortcut, 0.500 = no signal:

| feature | `default` | `normalized` | `laion_matched` | `cross_generator` |
|---|---|---|---|---|
| **image size** | **1.000** | 0.500 | 0.500 | 0.500 |
| mean luminance | — | 0.529 | **0.734** | 0.701 |
| recompressed bytes | — | 0.602 | 0.696 | 0.565 |
| saturation | — | 0.582 | 0.615 | 0.609 |
| high-freq energy | — | 0.580 | 0.561 | 0.513 |
| Laplacian variance | — | 0.569 | 0.526 | 0.640 |

Two honest observations:

`laion_matched` has a **stronger** trivial leak than `normalized` (0.734 vs 0.602), which is
counterintuitive. LAION web photos differ from DALL·E generations in brightness and saturation
more than COCO photos do, and the aggressive 200x200 downscale partly washes that out. The
difference in kind still matters: 1.000 from pixel dimensions is a pure artifact with no
relationship to the task, whereas ~0.73 from brightness is a genuine stylistic difference between
web imagery and AI generations — closer to real signal, though a model leaning on it will not
survive a distribution shift.

**Benchmark your model against this table.** If your AUC on `default` is ~1.00 but drops to ~0.75
on `laion_matched`, you have learned the shortcut, not the task.

## Robustness transforms

The `_transformed` configs apply the competition's robustness menu, so you can measure
degradation before anyone runs your model on mangled images.

| transform | settings | real-world analogue |
|---|---|---|
| JPEG compression | q90, q70, q50, q30 | social-media re-encoding |
| Gaussian blur | sigma 0.5, 1.0, 2.0 | out-of-focus images |
| Resize | 0.5x or 0.25x, then upscale back | thumbnail generation |
| Gaussian noise | sigma 0.02, 0.05, 0.10 | low-light sensor noise |
| Color jitter | brightness/contrast/saturation +/-20% | filter apps, auto-enhancement |
| Center crop | retain 80% | profile-picture cropping |

14 settings in total.

### Composition policy

The competition materials say more than one transform may be applied but do not specify the
composition policy, so this is the one used here — recorded per row so you can slice around it:

- **70% of rows get exactly one transform**, assigned round-robin so every setting gets
  near-equal coverage (988-989 rows per setting in the 13,841-row configs).
- **30% get two**, drawn from different families. 92 distinct chains occur.
- When two apply, they run in a fixed physical order regardless of sampling order:
  **crop -> resize -> jitter -> blur -> noise -> jpeg**, mirroring how an image actually gets
  mangled in the wild (cropped, filtered, blurred, then re-encoded on upload).

Assignment is seeded and deterministic — the same row always gets the same chain.

### Extra columns

`_transformed` configs carry three columns the clean ones do not:

- `primary_transform` — the round-robin-assigned setting, e.g. `jpeg_q30`, `blur_1.0`, `crop_0.8`.
- `transform_chain` — everything applied, in application order, e.g. `blur_2.0|jpeg_q50`.
- `n_transforms` — 1 or 2.

### Measuring robustness

```python
ds = load_dataset("techjam-aigc/wildfake-eval-subset",
                  "laion_matched_transformed", split="validation")

clean = ds.filter(lambda x: x["n_transforms"] == 1)   # isolate single transforms
for setting in sorted(set(clean["primary_transform"])):
    sub = clean.filter(lambda x, s=setting: x["primary_transform"] == s)
    ...  # AUC per setting -> a degradation curve
```

Filter `n_transforms == 1` for clean per-setting curves; use the `n_transforms == 2` rows to
check whether composed degradations compound.

> [!NOTE]
> Every image in a `_transformed` config is stored as JPEG q95, so all rows share one final
> encode. That makes comparison **across settings within a transformed config** clean. Comparing
> a transformed config against its clean twin carries that extra q95 re-encode as a small
> confound — for `default` the clean config stores original bytes, and for the others q92.

## Config details

### `default` — spec-faithful
Exactly the official demo subset: `real_coco.csv` filtered to `/val2017/` (4,998) and all of
`dalle3.csv`, i.e. WildFake DALLE3 with `IsAdvanced=1` (8,843). Bytes are untouched — no resizing,
re-encoding, or filtering. Classes are unbalanced (36% real / 64% fake), so report AUC or balanced
accuracy rather than raw accuracy.

### `normalized` — size shortcut removed, cheaply
The same 13,841 images, each center-cropped to a square and resized to 200x200, re-encoded at
JPEG q92. Removes size as a cue but destroys high-frequency detail — the very signal a forensic
detector should use. Treat it as a smoke test.

### `laion_matched` — the fair comparison
Native-resolution pairing is not achievable here: LAION clusters at 800x800 and DALL·E 3 at
1024x1024, giving only **66** exact (w,h) matches across 14,000 sampled LAION images. So instead
both classes are restricted to natively >=1024px images and put through one identical downscale to
512x512. Both classes therefore start large and receive the same resampling, unlike `default`
where the reals were pre-destroyed and the fakes were not. LAION-5B is also the training
distribution for these models, making it the apt real counterpart. Balanced 50/50.

### `cross_generator` — generalization probe
Every fake in the other configs is DALL·E 3, so a strong score there says nothing about other
generators. This config holds 1,500 LAION reals against four generators, all through an identical
pipeline at 256x256:

| source | label | n |
|---|---|---|
| `laion5b` | 0 | 1,500 |
| `dalle3` | 1 | 1,000 |
| `midjourney_v5` | 1 | 999 |
| `sdxl` | 1 | 1,000 |
| `gigagan` | 1 | 995 |

DALL·E 3 is included as a same-pipeline reference point, so the drop from DALL·E to the others is
measurable within one config:

```python
ds = load_dataset("techjam-aigc/wildfake-eval-subset", "cross_generator", split="validation")
real = ds.filter(lambda x: x["label"] == 0)
for gen in ["dalle3", "midjourney_v5", "sdxl", "gigagan"]:
    fake = ds.filter(lambda x: x["source"] == gen)
    ...  # score real vs fake, compare across generators
```

256x256 rather than 512 is deliberate: text-to-image GANs output natively smaller than diffusion
models (GigaGAN is 512x512), so a 512 target would have left GigaGAN as the only un-resampled
source and turned the GAN probe into a resampling detector.

## Fields

Identical across all configs.

- `image` — the image, embedded in the parquet shards.
- `label` — `ClassLabel`, `0=real`, `1=fake`.
- `source` — origin, e.g. `coco_val2017`, `dalle3_advanced`, `laion5b`, `midjourney_v5`, `sdxl`, `gigagan`.
- `orig_path` — path within the upstream WildFake archive, for tracing a row back to source.
- `id` — `"{source}/{basename}"`.

`_transformed` configs add `primary_transform`, `transform_chain`, and `n_transforms`.

Rows in every config are shuffled with a fixed seed (0), so a truncated or streamed read still
sees every class. Streaming works if you don't want the ~3 GB `default` locally:

```python
ds = load_dataset("techjam-aigc/wildfake-eval-subset", "laion_matched",
                  split="validation", streaming=True)
```

## Provenance

Built by range-reading the upstream ModelScope archives over HTTP — parsing each Zip64 central
directory and fetching only the needed byte spans, rather than downloading ~28 GB of zips. Every
extracted member was CRC-verified against its central-directory entry.

Sources: `Images/Real/coco.zip`, `Images/Real/laion5b.zip`, `Images/Diffusion_based/DALLE.zip`,
`Images/Diffusion_based/Midjourney/Advanced/part_1.zip`,
`Images/Diffusion_based/SD/originalSD/Advanced/part_1.zip`, `Images/GAN_based.zip`.

Known upstream defect: 5 of 1,000 sampled GigaGAN PNGs are undecodable. They pass the zip CRC —
the bytes match the archive exactly — but fail to parse, so they are dropped (hence 995).

## License / attribution

Upstream WildFake terms apply (research / non-commercial). Underlying images retain their own
terms: COCO under the [COCO terms of use](https://cocodataset.org/#termsofuse), LAION-5B under its
own license, and generated images under their respective providers' terms. Redistributed here for
benchmark use within the org.

```bibtex
@article{hong2024wildfake,
  title={WildFake: A Large-scale Challenging Dataset for AI-Generated Images Detection},
  author={Hong, Yan and Zhang, Jianfu},
  journal={arXiv preprint arXiv:2402.11843},
  year={2024}
}
```
