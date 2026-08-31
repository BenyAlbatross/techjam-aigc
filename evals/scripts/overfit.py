"""Classic overfitting, or something else? Look at the score distributions."""
import numpy as np, pandas as pd
d = pd.read_parquet('/home/joel/Desktop/acai/runs/calib_trace_rx_m.parquet')
c = d[d["transform"] == "clean"]
real = c[c.label == 0].score.values

print("Model outputs a LOGIT: higher = 'more likely AI'. Real photos should score LOW.\n")
print(f"  {'group':24s} {'n':>5} {'median score':>13} {'% scoring above the real median':>34}")
rmed = np.median(real)
print(f"  {'REAL photos':24s} {len(real):5d} {rmed:13.3f} {'(reference)':>34}")
for g in ["flux_1_schnell", "sdxl_1_0", "gpt_image_2", "gemini_flash_image"]:
    s = c[c.generator == g].score.values
    above = (s > rmed).mean() * 100
    flag = "  <- HELD OUT" if g == "gemini_flash_image" else ""
    print(f"  {g:24s} {len(s):5d} {np.median(s):13.3f} {above:33.1f}%{flag}")

print("\nIf the model had simply MEMORISED training images, it would score ~0.5 (random)")
print("on anything new. Instead:")
g = c[c.generator == "gemini_flash_image"].score.values
print(f"  median Gemini AI score  {np.median(g):.3f}")
print(f"  median real-photo score {rmed:.3f}")
print(f"  -> Gemini AI images score LOWER than real photos." if np.median(g) < rmed else "")
print("     The model is not guessing. It is confidently calling them authentic.")
