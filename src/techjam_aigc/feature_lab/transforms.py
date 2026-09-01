"""Deterministic challenge transformations and staged composition profiles."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class TransformStep:
    operation: str
    parameters: tuple[tuple[str, Any], ...] = ()

    def values(self) -> dict[str, Any]:
        return dict(self.parameters)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    family: str
    severity: float
    description: str
    official: bool = True
    steps: tuple[TransformStep, ...] = ()
    profile: str = "core"
    design: str = "single"


def _step(operation: str, **parameters: Any) -> TransformStep:
    return TransformStep(operation, tuple(parameters.items()))


def _jpeg(q: int) -> TransformStep:
    return _step("jpeg", quality=q, subsampling=2)


def _blur(sigma: float) -> TransformStep:
    return _step("gaussian_blur", sigma=sigma, padding="pillow_default")


def _resize(scale: float) -> TransformStep:
    return _step("resize", scale=scale, down_interpolation="lanczos", up_interpolation="lanczos")


def _noise(sigma: float) -> TransformStep:
    return _step(
        "gaussian_noise",
        sigma=sigma,
        seed="sha256(base_seed,parent_id,step_token,occurrence)",
    )


def _color(brightness: float, contrast: float, saturation: float) -> TransformStep:
    return _step("color_jitter", brightness=brightness, contrast=contrast, saturation=saturation, operation_order="brightness_contrast_saturation")


def _crop() -> TransformStep:
    return _step("center_crop", retain=0.8, up_interpolation="bicubic")


def _sharpen() -> TransformStep:
    return _step("unsharp_mask", radius=2.0, percent=150, threshold=3, padding="pillow_default")


TRANSFORM_SPECS: tuple[TransformSpec, ...] = (
    TransformSpec("clean", "clean", 0.0, "Published pixels; no robustness transform."),
    *(TransformSpec(f"jpeg_q{q}", "jpeg", float(100-q), f"JPEG round trip at quality {q}.", steps=(_jpeg(q),)) for q in (90, 70, 50, 30)),
    *(TransformSpec(f"blur_sigma{s:g}", "gaussian_blur", s, f"Gaussian blur with sigma {s:g}.", steps=(_blur(s),)) for s in (0.5, 1.0, 2.0)),
    *(TransformSpec(f"resize_{s:g}", "resize", 1-s, f"Lanczos downscale to {s:g}x then upscale.", steps=(_resize(s),)) for s in (0.5, 0.25)),
    *(TransformSpec(f"noise_sigma{s:.2f}", "gaussian_noise", s, f"Seeded RGB Gaussian noise with sigma {s:.2f}.", steps=(_noise(s),)) for s in (0.02, 0.05, 0.10)),
    TransformSpec("color_jitter_minus20", "color_jitter", -0.2, "Brightness, contrast, and saturation factors all 0.8.", steps=(_color(0.8, 0.8, 0.8),)),
    TransformSpec("color_jitter_plus20", "color_jitter", 0.2, "Brightness, contrast, and saturation factors all 1.2.", steps=(_color(1.2, 1.2, 1.2),)),
    TransformSpec("center_crop_80", "center_crop", 0.2, "Retain centered 80% of each axis, then bicubic-resize.", steps=(_crop(),)),
    TransformSpec("resize_0.5__jpeg_q70", "composition", 2.0, "0.5x resize followed by JPEG Q70.", False, (_resize(0.5), _jpeg(70)), design="legacy_composition"),
    TransformSpec("crop_80__resize_0.5__jpeg_q50", "composition", 3.0, "80% crop, 0.5x resize, then JPEG Q50.", False, (_crop(), _resize(0.5), _jpeg(50)), design="legacy_composition"),
    TransformSpec("blur_1__resize_0.5__jpeg_q70", "composition", 3.0, "Sigma-1 blur, 0.5x resize, then JPEG Q70.", False, (_blur(1), _resize(0.5), _jpeg(70)), design="legacy_composition"),
    TransformSpec("noise_0.02__jpeg_q70", "composition", 2.0, "Sigma-0.02 noise followed by JPEG Q70.", False, (_noise(0.02), _jpeg(70)), design="legacy_composition"),
)


def _num(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _token(step: TransformStep) -> str:
    p = step.values()
    if step.operation == "jpeg": return f"jpeg-q{p['quality']}-sub{p['subsampling']}"
    if step.operation == "gaussian_blur": return f"blur-s{_num(p['sigma'])}-pad-pillow"
    if step.operation == "resize": return f"resize-s{_num(p['scale'])}-down-lanczos-up-lanczos"
    if step.operation == "gaussian_noise": return f"noise-s{_num(p['sigma'])}-seed-parent"
    if step.operation == "color_jitter": return f"color-b{_num(p['brightness'])}-c{_num(p['contrast'])}-s{_num(p['saturation'])}-order-bcs"
    if step.operation == "center_crop": return "crop-r0p8-up-bicubic"
    if step.operation == "unsharp_mask": return "sharpen-r2-p150-t3-pad-pillow"
    raise KeyError(step.operation)


def _pipeline(prefix: str, steps: tuple[TransformStep, ...], profile: str, design: str, description: str) -> TransformSpec:
    return TransformSpec(f"{prefix}__{'__then__'.join(map(_token, steps))}", "composition", float(len(steps)), description, False, steps, profile, design)


def _color_factorial() -> tuple[TransformSpec, ...]:
    specs = []
    for axis in ("brightness", "contrast", "saturation"):
        for factor in (0.8, 1.2):
            values = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0}
            values[axis] = factor
            specs.append(_pipeline("color-factorial-axial", (_color(**values),), "color_factorial", "color_axial", f"Single-axis {axis} factor {factor:g}."))
    for b in (0.8, 1.2):
        for c in (0.8, 1.2):
            for s in (0.8, 1.2):
                if b == c == s:  # official diagonal cases already exist in core
                    continue
                specs.append(_pipeline("color-factorial-corner", (_color(b, c, s),), "color_factorial", "color_corner", f"Two-level color corner ({b:g}, {c:g}, {s:g})."))
    return tuple(specs)


_MEDIUM = (("jpeg", _jpeg(70)), ("gaussian_blur", _blur(1)), ("resize", _resize(0.5)), ("gaussian_noise", _noise(0.05)), ("color_jitter", _color(1.2, 1.2, 1.2)), ("center_crop", _crop()))


def _directed_pairs() -> tuple[TransformSpec, ...]:
    return tuple(_pipeline("directed-pair", (a, b), "directed_pairs", "directed_medium_pair", f"Medium ordered pair: {af} -> {bf}.") for af, a in _MEDIUM for bf, b in _MEDIUM if af != bf)


def _realistic() -> tuple[TransformSpec, ...]:
    recipes = (
        ("Crop, resize, JPEG.", (_crop(), _resize(0.5), _jpeg(50))),
        ("Color, resize, JPEG.", (_color(1.2, 0.8, 1.2), _resize(0.5), _jpeg(70))),
        ("Blur, noise, JPEG.", (_blur(1), _noise(0.02), _jpeg(70))),
        ("Noise, blur, JPEG.", (_noise(0.02), _blur(1), _jpeg(70))),
        ("Resize, sharpen, JPEG.", (_resize(0.5), _sharpen(), _jpeg(70))),
        ("Sharpen, resize, JPEG.", (_sharpen(), _resize(0.5), _jpeg(70))),
        ("Repeated JPEG around resize.", (_jpeg(90), _resize(0.5), _jpeg(50))),
        ("Repeated JPEG around noise.", (_jpeg(90), _noise(0.02), _jpeg(50))),
        ("Repeated JPEG around blur.", (_jpeg(90), _blur(1), _jpeg(50))),
        ("Crop, color, JPEG.", (_crop(), _color(0.8, 1.2, 0.8), _jpeg(70))),
        ("Crop, resize, color, JPEG.", (_crop(), _resize(0.5), _color(1.2, 0.8, 1.2), _jpeg(50))),
        ("Crop, resize, sharpen, JPEG.", (_crop(), _resize(0.5), _sharpen(), _jpeg(50))),
    )
    return tuple(_pipeline("realistic", steps, "realistic", "preregistered_realistic_chain", description) for description, steps in recipes)


def _radical_inverse(index: int, base: int) -> float:
    result, fraction = 0.0, 1/base
    while index:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


def _pick(levels: tuple[Any, ...], index: int, base: int) -> Any:
    return levels[min(len(levels)-1, math.floor(_radical_inverse(index, base)*len(levels)))]


def _covering32() -> tuple[TransformSpec, ...]:
    colors = (None, (0.8,0.8,0.8), (0.8,0.8,1.2), (0.8,1.2,0.8), (0.8,1.2,1.2), (1.2,0.8,0.8), (1.2,0.8,1.2), (1.2,1.2,0.8), (1.2,1.2,1.2))
    specs = []
    for i in range(1, 33):
        q, blur, resize, noise, color, crop = (_pick((None,90,70,50,30),i,2), _pick((None,0.5,1.0,2.0),i,3), _pick((None,0.5,0.25),i,5), _pick((None,0.02,0.05,0.10),i,7), _pick(colors,i,11), _pick((None,0.8),i,13))
        steps = []  # canonical: geometry, photometry, blur/noise, encoding
        if crop is not None: steps.append(_crop())
        if resize is not None: steps.append(_resize(resize))
        if color is not None: steps.append(_color(*color))
        if blur is not None: steps.append(_blur(blur))
        if noise is not None: steps.append(_noise(noise))
        if q is not None: steps.append(_jpeg(q))
        specs.append(_pipeline(f"cover32-{i:02d}", tuple(steps), "covering32", "halton_covering_bank", f"Fixed Halton-style covering recipe {i}/32."))
    return tuple(specs)


COLOR_FACTORIAL_SPECS, DIRECTED_PAIR_SPECS, REALISTIC_SPECS, COVERING32_SPECS = _color_factorial(), _directed_pairs(), _realistic(), _covering32()
TRANSFORM_PROFILES = ("core", "color_factorial", "directed_pairs", "realistic", "covering32", "all")
_ADDITIONS = {"color_factorial": COLOR_FACTORIAL_SPECS, "directed_pairs": DIRECTED_PAIR_SPECS, "realistic": REALISTIC_SPECS, "covering32": COVERING32_SPECS}


def get_transform_specs(profile: str = "core") -> tuple[TransformSpec, ...]:
    if profile == "core": return TRANSFORM_SPECS
    if profile == "all": additions = tuple(spec for name in TRANSFORM_PROFILES[1:-1] for spec in _ADDITIONS[name])
    elif profile in _ADDITIONS: additions = _ADDITIONS[profile]
    else: raise KeyError(f"Unknown transform profile {profile!r}; choose from {TRANSFORM_PROFILES}")
    specs = (*TRANSFORM_SPECS, *additions)
    if len({spec.name for spec in specs}) != len(specs): raise RuntimeError(f"Duplicate condition in {profile}")
    return specs


_LOOKUP = {spec.name: spec for spec in get_transform_specs("all")}


def get_transform_spec(condition: str) -> TransformSpec:
    try: return _LOOKUP[condition]
    except KeyError: raise KeyError(f"Unknown transform condition: {condition}") from None


def transform_frame(profile: str = "core", *, conditions: tuple[str, ...] | None = None) -> pd.DataFrame:
    specs = get_transform_specs(profile)
    if conditions is not None:
        missing = set(conditions) - {spec.name for spec in specs}
        if missing: raise KeyError(f"Conditions are not in profile {profile!r}: {sorted(missing)}")
        specs = tuple(spec for spec in specs if spec.name in conditions)
    rows = []
    for spec in specs:
        values = [step.values() for step in spec.steps]
        recipe = json.dumps([{"operation": step.operation, "parameters": step.values()} for step in spec.steps], sort_keys=True, separators=(",", ":"))
        rows.append({"name": spec.name, "family": spec.family, "severity": spec.severity, "description": spec.description, "official": spec.official, "declared_profile": spec.profile, "design": spec.design, "step_count": len(spec.steps), "ordered_operations": ">".join(step.operation for step in spec.steps) or "clean", "ordered_recipe_json": recipe, "interpolation_json": json.dumps([v for p in values for k,v in p.items() if "interpolation" in k]), "padding_json": json.dumps([p["padding"] for p in values if "padding" in p]), "seed_policy_json": json.dumps([p["seed"] for p in values if "seed" in p]), "recipe_sha256": sha256(recipe.encode()).hexdigest()})
    return pd.DataFrame(rows)


def deterministic_seed(parent_id: str, condition: str, base_seed: int) -> int:
    return int.from_bytes(sha256(f"{base_seed}:{parent_id}:{condition}".encode()).digest()[:8], "big") % 2**32


def deterministic_step_seed(
    parent_id: str,
    step: TransformStep,
    occurrence: int,
    base_seed: int,
) -> int:
    """Seed stochastic steps independently of pipeline order.

    Exact reverse-order recipes therefore reuse the same noise draw while
    repeated identical stochastic steps remain distinct by occurrence.
    """

    return deterministic_seed(parent_id, f"step:{_token(step)}:{occurrence}", base_seed)


def _jpeg_image(image: Image.Image, quality: int, subsampling: int) -> Image.Image:
    buffer = BytesIO(); image.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=subsampling); buffer.seek(0)
    with Image.open(buffer) as decoded: return decoded.convert("RGB").copy()


def _resize_image(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size; down = (max(1, round(width*scale)), max(1, round(height*scale)))
    return image.resize(down, Image.Resampling.LANCZOS).resize((width,height), Image.Resampling.LANCZOS)


def _noise_image(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)/255
    noisy = np.clip(array + np.random.default_rng(seed).normal(0, sigma, array.shape), 0, 1)
    return Image.fromarray(np.round(noisy*255).astype(np.uint8), "RGB")


def _crop_image(image: Image.Image, retain: float) -> Image.Image:
    width,height=image.size; cw,ch=max(1,round(width*retain)),max(1,round(height*retain)); left,top=(width-cw)//2,(height-ch)//2
    return image.crop((left,top,left+cw,top+ch)).resize((width,height), Image.Resampling.BICUBIC)


def _apply_step(image: Image.Image, step: TransformStep, seed: int) -> Image.Image:
    p = step.values()
    if step.operation == "jpeg": return _jpeg_image(image, int(p["quality"]), int(p["subsampling"]))
    if step.operation == "gaussian_blur": return image.filter(ImageFilter.GaussianBlur(float(p["sigma"])))
    if step.operation == "resize": return _resize_image(image, float(p["scale"]))
    if step.operation == "gaussian_noise": return _noise_image(image, float(p["sigma"]), seed)
    if step.operation == "color_jitter":
        result = ImageEnhance.Brightness(image.convert("RGB")).enhance(float(p["brightness"])); result = ImageEnhance.Contrast(result).enhance(float(p["contrast"])); return ImageEnhance.Color(result).enhance(float(p["saturation"]))
    if step.operation == "center_crop": return _crop_image(image, float(p["retain"]))
    if step.operation == "unsharp_mask": return image.filter(ImageFilter.UnsharpMask(radius=float(p["radius"]), percent=int(p["percent"]), threshold=int(p["threshold"])))
    raise KeyError(step.operation)


def apply_transform_steps(
    image: Image.Image,
    steps: tuple[TransformStep, ...],
    *,
    parent_id: str = "image",
    base_seed: int = 20260830,
) -> Image.Image:
    """Apply an explicit ordered recipe with deterministic stochastic steps."""

    result = image.convert("RGB").copy()
    occurrences: dict[str, int] = {}
    for step in steps:
        token = _token(step)
        occurrence = occurrences.get(token, 0)
        occurrences[token] = occurrence + 1
        seed = deterministic_step_seed(parent_id, step, occurrence, base_seed)
        result = _apply_step(result, step, seed)
    return result.convert("RGB")


def apply_transform(image: Image.Image, condition: str, *, parent_id: str = "image", base_seed: int = 20260830) -> Image.Image:
    return apply_transform_steps(
        image,
        get_transform_spec(condition).steps,
        parent_id=parent_id,
        base_seed=base_seed,
    )


def analysis_view(image: Image.Image, view: str) -> Image.Image:
    image = image.convert("RGB")
    if view == "canonical_128": return image.resize((128,128), Image.Resampling.LANCZOS)
    if view == "native_capped":
        width,height=image.size; scale=min(1.0,256/max(width,height))
        if scale == 1: return image.copy()
        return image.resize((max(1,round(width*scale)),max(1,round(height*scale))), Image.Resampling.LANCZOS)
    raise KeyError(f"Unknown analysis view: {view}")
