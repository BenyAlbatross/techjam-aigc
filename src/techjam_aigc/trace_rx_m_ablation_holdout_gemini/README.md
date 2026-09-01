# Full TRACE-RX-M / Gemini holdout

Frozen snapshot of the current TRACE-RX-M training implementation. The only scientific delta from the suite reference is the complete exclusion of `gemini_flash_image` from S4 training. The held-out ROC-AUC gate is report-only; it never replaces this LoRA model with a fallback.
