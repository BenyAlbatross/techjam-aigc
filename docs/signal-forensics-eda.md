# Frequency and camera-signal forensics

[`notebooks/signal_forensics_eda.py`](../notebooks/signal_forensics_eda.py) is a
separate marimo follow-up to the visual dataset EDA. It measures FFT, 8×8 DCT,
residual-periodicity, and single-image camera-pipeline proxies on the existing
EDA slice plus deterministic subsets of DDA-COCO, EvalGEN, and Community
Forensics Eval.

## Reproduce

From the repository root:

```bash
uv run python scripts/prepare_signal_analysis_samples.py
uv run python scripts/run_signal_analysis.py
uv run marimo edit notebooks/signal_forensics_eda.py
```

The preparation command writes downloaded images and provenance manifests only
under the Git-ignored `data/` tree. It uses bounded byte ranges for the large
DDA-COCO and EvalGEN archives, pins every dataset revision, records licenses and
SHA-256 hashes, and can reuse already verified local files. The current
deterministic sample contains:

| Dataset | Authentic | AIGC | Design |
| --- | ---: | ---: | --- |
| DDA-COCO | 120 | 120 | exact COCO/reconstruction pairs, 20 per reconstruction group |
| EvalGEN | 0 | 100 | 20 each from Flux, GoT, Infinity, NOVA, and OmniGen |
| Community Forensics Eval | 25 | 25 | balanced exploratory slice |

Community Forensics Eval is licensed CC-BY-NC-SA-4.0, so its local subset is
for non-commercial research and education subject to attribution and
share-alike obligations. DDA-COCO and EvalGEN declare Apache-2.0. The source
manifest at `data/metadata/signal_analysis_sources.json` is authoritative for
the pinned revisions and acquisition notes.

## Findings and limits

- Periodic Fourier differences exist in some sources, but do not generalize as
  a universal diffusion signature. The Community Forensics slice has strong
  1/8-cycle harmonic separation, while the matched DDA-COCO comparison is near
  chance for the tested scalar frequency features.
- PRNU and DSNU are **not directly measurable** here. PRNU needs repeated images
  from the same identified sensor, ideally flat-field or RAW captures. DSNU
  needs controlled dark frames. The notebook labels CFA periodicity, residual
  color coupling, and signal/noise fits as proxies rather than PRNU or DSNU.
- FFT and DCT directions vary by source. Strong effects in SID Set and WildFake
  can collapse on DDA-COCO, and DCT measurements are especially vulnerable to
  JPEG history and resizing.
- EvalGEN contains no authentic class. Its comparison with pooled authentic
  images is deliberately marked cross-source and descriptive, not evidence of
  causal or deployment-ready separation.

The organizer demonstration-only validation split is not used. Official COCO
images are downloaded only as the authentic counterparts of DDA-COCO pairs for
this non-training analysis.

## Outputs

`scripts/run_signal_analysis.py` center-fits images to 256×256 and writes the
following reproducible, ignored artifacts to `data/derived/signal_analysis/`:

- image-level features;
- radial FFT and 8×8 DCT profiles;
- within-source and explicitly labelled cross-source effect tables;
- exact-pair DDA-COCO deltas;
- a feature registry and run metadata with cautions and input hashes.

These signals are suitable as weak branches or nuisance controls in a later
detector, not as standalone authenticity rules.
