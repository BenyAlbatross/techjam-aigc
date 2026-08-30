# Project instructions

- Communicate with the user in Caveman Ultra unless they disable it.
- Use Superpowers skills before applicable work and Ponytail minimalism for code.
- Work on branch `xuan`; preserve groupmates' and users' changes.
- Current priority: zero-fine-tuning public baselines, transformation robustness, and error rates.
- Do not fine-tune, calibrate, ensemble, or change published thresholds during this phase.
- Enforce fewer than 2 billion learned inference-time parameters.
- Run `pixi run compliance` before downloads or benchmarks. Block unknown, non-commercial, incompatible, or unauthorized assets.
- Never use competition-provided evaluation data for model or threshold selection.
- Keep images, weights, tokens, caches, and prediction shards out of Git.
- Use deterministic, class-symmetric transformations and report FPR, FNR, balanced accuracy, AUROC, confidence intervals, and contamination.
- Required submission output objects contain exactly `image_path` and `pred`.
- Run the smallest relevant network-free test after each code change.
