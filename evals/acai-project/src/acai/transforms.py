"""Robustness transforms for the *_transformed configs.

Menu and parameters come from the competition's robustness table. The composition policy
is not specified there, so we pick one and record it per row:

  - 70% of images get exactly ONE transform, assigned round-robin over the 14 settings so
    every setting gets near-equal coverage (filter n_transforms == 1 for clean per-setting
    curves).
  - 30% get TWO, from different families, to exercise composition.

When two are applied they run in a fixed, physically sensible order rather than the order
sampled: geometry -> photometric -> degradation -> compression, mirroring how an image
actually gets mangled in the wild (cropped, filtered, blurred, then re-encoded on upload).
"""
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# (family, param) — 14 settings total
SETTINGS = [
    ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30),
    ("blur", 0.5), ("blur", 1.0), ("blur", 2.0),
    ("resize", 0.5), ("resize", 0.25),
    ("noise", 0.02), ("noise", 0.05), ("noise", 0.10),
    ("jitter", 0.2),
    ("crop", 0.8),
]

ORDER = ["crop", "resize", "jitter", "blur", "noise", "jpeg"]
P_SECOND = 0.30


def _jpeg(im, q, rng):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=int(q))
    b.seek(0)
    return Image.open(b).convert("RGB")


def _blur(im, sigma, rng):
    return im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _resize(im, scale, rng):
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def _noise(im, sigma, rng):
    a = np.asarray(im, dtype=np.float32)
    a = a + rng.normal(0.0, float(sigma) * 255.0, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _jitter(im, amt, rng):
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = enh(im).enhance(float(rng.uniform(1 - amt, 1 + amt)))
    return im


def _crop(im, keep, rng):
    w, h = im.size
    nw, nh = max(1, int(w * keep)), max(1, int(h * keep))
    return im.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))


FN = {"jpeg": _jpeg, "blur": _blur, "resize": _resize,
      "noise": _noise, "jitter": _jitter, "crop": _crop}


def label(fam, param):
    if fam == "jpeg":
        return f"jpeg_q{int(param)}"
    if fam == "jitter":
        return "jitter_0.2"
    if fam == "crop":
        return "crop_0.8"
    return f"{fam}_{param}"


def plan(i, seed=0):
    """Deterministic transform plan for row i: (chain, primary_label, n)."""
    rng = np.random.default_rng([seed, i])
    fam, param = SETTINGS[i % len(SETTINGS)]
    chain = [(fam, param)]
    if rng.random() < P_SECOND:
        others = [s for s in SETTINGS if s[0] != fam]
        chain.append(others[rng.integers(len(others))])
    chain.sort(key=lambda fp: ORDER.index(fp[0]))
    return chain, label(fam, param), len(chain)


def apply(im, chain, i, seed=0):
    rng = np.random.default_rng([seed + 1, i])
    im = im.convert("RGB")
    for fam, param in chain:
        im = FN[fam](im, param, rng)
    return im
