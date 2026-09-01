# Direct DINOv3 probe / Gemini holdout

This control removes the authentic memory, retrieval statistics, and residual reconstruction. It mean-pools normalized DINOv3 patch tokens and applies an MLP classifier while retaining LoRA, data, augmentation, schedule, and loss settings from the full Gemini-holdout reference.
