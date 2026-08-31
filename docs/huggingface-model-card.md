---
library_name: pytorch
pipeline_tag: image-classification
tags:
  - image-forensics
  - ai-generated-image-detection
  - dinov2
  - techjam-2026
---

# TRACE-RX-M v2

TRACE-RX-M v2 is a binary image-level detector for purely AI-generated versus
authentic images. It compares normalized DINOv2 patch tokens against a frozen
authentic-only prototype memory and classifies directional reconstruction
residuals plus retrieval statistics.

This repository contains the frozen TechJam 2026 shipping detector. It is not a
standalone copy of the DINOv2 backbone: loading downloads the pinned public
`facebook/dinov2-base` revision recorded in `config.json`.

## Files

- `s4_detector.pt`: shipping detector heads and checkpoint metadata.
- `s3_memory.pt`: required frozen authentic prototype memory.
- `config.json`: pinned model and training configuration.
- `s4_validity.json`: held-out-generator validity decision.
- `s3_capacity.json`: authentic-memory capacity audit.
- `evaluation/summary.json`: clean and official-transform summary.
- `evaluation/metrics_by_condition.csv`: metrics for all official conditions.

## Loading

Clone the public implementation and install its training dependencies:

```bash
git clone https://github.com/BenyAlbatross/techjam-aigc.git
cd techjam-aigc
uv sync --group train
```

Download this model repository, then reconstruct the frozen detector:

```python
from pathlib import Path

import torch

from techjam_aigc.trace_rx_m.training import load_detector_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, metadata = load_detector_checkpoint(
    Path("s4_detector.pt"),
    Path("s3_memory.pt"),
    device=device,
)
```

Images must use the canonical preprocessing defined by the public repository.
The detector logit is positive for AIGC; `sigmoid(logit)` is the exported AIGC
confidence score. The current checkpoint is not probability-calibrated.

## Evaluation

The frozen checkpoint was evaluated on 6,091 development images under clean
pixels and all 15 official TechJam transformation settings, producing 97,456
paired endpoints.

| Evaluation | ROC-AUC | Average precision | Normalized pAUC@5% | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.9035 | 0.9063 | 0.7794 | 0.8852 |
| Macro across transformed conditions | 0.9011 | 0.9043 | 0.7708 | 0.8839 |
| Worst condition: Gaussian noise sigma 0.10 | 0.8936 | 0.8973 | 0.7461 | 0.8734 |

The evaluation did not use the organizer demonstration-only set or the locked
split for model selection.

## Important limitations

- Cross-generator generalization is not solved. Gemini Flash Image development
  ROC-AUC is 0.3279, despite strong performance on FLUX, SDXL, and GPT Image 2.
- The current TechJam robustness evaluation transforms neutralized 224-pixel
  BMP inputs rather than original source-resolution files.
- The model targets purely generated images. AI-edited and partially composited
  images are outside its trained scope.
- Scores are not calibrated probabilities and no production operating threshold
  is claimed.
- The detector must not be treated as sole evidence for moderation or provenance
  decisions.

## Reproducibility and provenance

- Detector SHA-256:
  `f811c4641a644e1eaed30891f9c075932a1de8680dce812adaf03c9a7daaf25e`
- Authentic memory SHA-256:
  `71f0bf9edeedc5af21de1b22e3705560a1b0e8adef0c1312266cfff3eb59a7a2`
- Backbone: `facebook/dinov2-base`
- Backbone revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`

Training, augmentation, evaluation, and inference source code is maintained at
https://github.com/BenyAlbatross/techjam-aigc.
