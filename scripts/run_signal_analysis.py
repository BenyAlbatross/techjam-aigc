#!/usr/bin/env python3
"""Extract FFT, DCT, and camera-noise proxy evidence for the signal notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.fft import dctn
from sklearn.metrics import roc_auc_score

from techjam_aigc.feature_lab.features import extract_features
from techjam_aigc.feature_lab.registry import registry_frame


EPS = 1e-12
SEED = 20260830
SIGNAL_FAMILIES = {
    "fft_magnitude",
    "fft_phase",
    "dct_jpeg",
    "noise",
    "residual",
    "camera_proxy",
    "codec_resampling",
    "multiscale_residual",
    "steganalysis",
    "patch_distribution",
}


def canonical_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.fit(
            opened.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def luma_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).ravel()
    right_flat = np.asarray(right, dtype=np.float64).ravel()
    if left_flat.size == 0 or np.std(left_flat) < EPS or np.std(right_flat) < EPS:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def spectrum_analysis(luma: np.ndarray, bins: int = 32) -> tuple[dict[str, float], list[dict[str, float]]]:
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((luma - luma.mean()) * window))
    power = np.abs(spectrum) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    grid_x, grid_y = np.meshgrid(fx, fy)
    radius = np.hypot(grid_x, grid_y)
    log_power = np.log1p(power)
    non_dc = radius > (2 / min(height, width))

    radial_rows: list[dict[str, float]] = []
    edges = np.linspace(0.0, np.sqrt(0.5), bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (radius >= low) & (radius < high)
        radial_rows.append(
            {
                "frequency_bin": index,
                "cycles_per_pixel": float((low + high) / 2),
                "log_power": float(np.mean(log_power[mask])) if np.any(mask) else np.nan,
            }
        )

    harmonic_excesses = []
    tolerance = 1.5 / min(height, width)
    for harmonic in (0.125, 0.25):
        line = (
            (np.abs(np.abs(grid_x) - harmonic) <= tolerance)
            | (np.abs(np.abs(grid_y) - harmonic) <= tolerance)
        ) & non_dc
        shoulder = (
            (
                (np.abs(np.abs(grid_x) - harmonic) > 3 * tolerance)
                & (np.abs(np.abs(grid_x) - harmonic) <= 7 * tolerance)
            )
            | (
                (np.abs(np.abs(grid_y) - harmonic) > 3 * tolerance)
                & (np.abs(np.abs(grid_y) - harmonic) <= 7 * tolerance)
            )
        ) & non_dc
        harmonic_excesses.append(float(np.mean(log_power[line]) - np.mean(log_power[shoulder])))

    local_background = ndimage.gaussian_filter(log_power, sigma=3, mode="nearest")
    prominence = log_power - local_background
    highpass = luma - ndimage.gaussian_filter(luma, sigma=1, mode="reflect")
    periodicities = {}
    for lag in (2, 4, 8):
        periodicities[f"residual_periodicity_lag{lag}"] = 0.5 * (
            correlation(highpass[:, :-lag], highpass[:, lag:])
            + correlation(highpass[:-lag, :], highpass[lag:, :])
        )
    features = {
        "fft_harmonic_excess_1_8": harmonic_excesses[0],
        "fft_harmonic_excess_1_4": harmonic_excesses[1],
        "fft_local_peak_prominence_q999": float(np.quantile(prominence[non_dc], 0.999)),
        **periodicities,
    }
    return features, radial_rows


def dct_analysis(luma: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    height, width = (luma.shape[0] // 8) * 8, (luma.shape[1] // 8) * 8
    blocks = luma[:height, :width].reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3)
    coefficients = dctn(blocks * 255.0 - 128.0, axes=(-2, -1), norm="ortho")
    yy, xx = np.mgrid[:8, :8]
    order = yy + xx
    total_ac = float(np.sum(coefficients[..., order > 0] ** 2)) + EPS
    rows = []
    for band in range(1, 15):
        mask = order == band
        energy = float(np.sum(coefficients[..., mask] ** 2) / total_ac)
        rows.append({"dct_band": band, "dct_energy_ratio": energy})
    flat_ac = coefficients[..., order > 0].ravel()
    features = {
        "dct_abs_median": float(np.median(np.abs(flat_ac))),
        "dct_quantized_fraction": float(np.mean(np.isclose(flat_ac, np.rint(flat_ac), atol=0.05))),
    }
    return features, rows


def noise_analysis(luma: np.ndarray) -> dict[str, float]:
    residual = luma - ndimage.gaussian_filter(luma, sigma=1, mode="reflect")
    patch_scales = []
    patch_size = 32
    for top in range(0, luma.shape[0] - patch_size + 1, patch_size):
        for left in range(0, luma.shape[1] - patch_size + 1, patch_size):
            patch = residual[top : top + patch_size, left : left + patch_size]
            patch_scales.append(1.4826 * np.median(np.abs(patch - np.median(patch))))
    scales = np.asarray(patch_scales)
    return {
        "noise_patch_scale_cv": float(np.std(scales) / (np.mean(scales) + EPS)),
        "noise_residual_mean_abs": float(abs(np.mean(residual))),
    }


def load_index(repo_root: Path) -> pd.DataFrame:
    base = pd.read_csv(repo_root / "data/samples/index.csv")
    base = base[base["label"].isin(["real", "fake", "full_synthetic"])].copy()
    base["label"] = base["label"].map(
        {"real": "authentic", "fake": "AIGC", "full_synthetic": "AIGC"}
    )
    base["paired_id"] = ""
    base["revision"] = "repository EDA sample"
    base["license_name"] = "See source dataset card"
    external = pd.read_csv(repo_root / "data/metadata/signal_analysis_index.csv")
    shared = sorted(set(base.columns) | set(external.columns))
    return pd.concat(
        [base.reindex(columns=shared), external.reindex(columns=shared)],
        ignore_index=True,
    )


def feature_effects(features: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    feature_names = registry[registry["family"].isin(SIGNAL_FAMILIES)]["name"].tolist()
    feature_names += [
        "fft_harmonic_excess_1_8",
        "fft_harmonic_excess_1_4",
        "fft_local_peak_prominence_q999",
        "residual_periodicity_lag2",
        "residual_periodicity_lag4",
        "residual_periodicity_lag8",
        "dct_abs_median",
        "dct_quantized_fraction",
        "noise_patch_scale_cv",
        "noise_residual_mean_abs",
    ]
    feature_names = [name for name in feature_names if name in features.columns]
    family_map = dict(zip(registry["name"], registry["family"])) | {
        "fft_harmonic_excess_1_8": "fft_magnitude",
        "fft_harmonic_excess_1_4": "fft_magnitude",
        "fft_local_peak_prominence_q999": "fft_magnitude",
        "residual_periodicity_lag2": "codec_resampling",
        "residual_periodicity_lag4": "codec_resampling",
        "residual_periodicity_lag8": "codec_resampling",
        "dct_abs_median": "dct_jpeg",
        "dct_quantized_fraction": "dct_jpeg",
        "noise_patch_scale_cv": "camera_proxy",
        "noise_residual_mean_abs": "camera_proxy",
    }
    scopes: list[tuple[str, pd.DataFrame, str]] = []
    for dataset, group in features.groupby("dataset", sort=True):
        if group["target"].nunique() == 2:
            scopes.append((f"within:{dataset}", group, "within-source"))
    scopes.append(("pooled:all", features, "pooled; confounded by source"))
    authentic_reference = features[features["target"] == 0]
    for dataset in ("EvalGEN",):
        generated = features[(features["dataset"] == dataset) & (features["target"] == 1)]
        scopes.append(
            (
                f"cross-source:{dataset}-vs-all-authentic",
                pd.concat([authentic_reference, generated], ignore_index=True),
                "cross-source descriptive only",
            )
        )

    rows = []
    for scope, frame, validity in scopes:
        if frame["target"].nunique() != 2:
            continue
        target = frame["target"].to_numpy()
        for name in feature_names:
            values = frame[name].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() < 4 or np.unique(target[finite]).size != 2:
                continue
            real = values[finite & (target == 0)]
            aigc = values[finite & (target == 1)]
            auc = float(roc_auc_score(target[finite], values[finite]))
            pooled_iqr = float(np.subtract(*np.quantile(values[finite], [0.75, 0.25])))
            rows.append(
                {
                    "scope": scope,
                    "validity": validity,
                    "family": family_map[name],
                    "feature": name,
                    "n_authentic": len(real),
                    "n_aigc": len(aigc),
                    "median_authentic": float(np.median(real)),
                    "median_aigc": float(np.median(aigc)),
                    "median_delta_aigc_minus_authentic": float(np.median(aigc) - np.median(real)),
                    "robust_effect": float((np.median(aigc) - np.median(real)) / (pooled_iqr + EPS)),
                    "auc_aigc_high": auc,
                    "separation_auc": max(auc, 1 - auc),
                    "rank_biserial": 2 * auc - 1,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "separation_auc"], ascending=[True, False], kind="stable"
    )


def paired_dda_effects(features: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    dda = features[features["dataset"] == "DDA-COCO"].copy()
    dda["pair_group"] = np.where(
        dda["target"] == 1,
        dda["generation_model"],
        dda["stratum"].str.replace("real paired to ", "", regex=False),
    )
    names = effects[effects["scope"] == "within:DDA-COCO"]["feature"].unique()
    rows = []
    for group, subset in dda.groupby("pair_group", sort=True):
        real = subset[subset["target"] == 0].set_index("paired_id")
        fake = subset[subset["target"] == 1].set_index("paired_id")
        common = real.index.intersection(fake.index)
        for name in names:
            differences = fake.loc[common, name].to_numpy() - real.loc[common, name].to_numpy()
            differences = differences[np.isfinite(differences)]
            if not len(differences):
                continue
            rows.append(
                {
                    "generation_model": group,
                    "feature": name,
                    "pairs": len(differences),
                    "median_paired_delta": float(np.median(differences)),
                    "positive_pair_fraction": float(np.mean(differences > 0)),
                }
            )
    return pd.DataFrame(rows)


def run(repo_root: Path, size: int) -> None:
    index = load_index(repo_root)
    output_dir = repo_root / "data/derived/signal_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows, radial_rows, dct_rows = [], [], []
    for position, row in enumerate(index.to_dict("records"), start=1):
        path = repo_root / row["local_path"]
        image = canonical_image(path, size)
        luma = luma_array(image)
        metadata = {"width": size, "height": size, "bytes": row["bytes"], "format": row["format"]}
        values = extract_features(image, metadata, profile="expanded_v2")
        spectrum_values, spectrum_profile = spectrum_analysis(luma)
        dct_values, dct_profile = dct_analysis(luma)
        values.update(spectrum_values)
        values.update(dct_values)
        values.update(noise_analysis(luma))
        row_id = hashlib.sha256(f"{row['dataset']}|{row['local_path']}".encode()).hexdigest()[:20]
        identity = {
            "row_id": row_id,
            "dataset": row["dataset"],
            "label": row["label"],
            "target": int(row["label"] == "AIGC"),
            "generation_model": row["generation_model"],
            "generator_family": row["generator_family"],
            "source_dataset": row["source_dataset"],
            "stratum": row["stratum"],
            "paired_id": row.get("paired_id", ""),
            "local_path": row["local_path"],
        }
        feature_rows.append(identity | values)
        radial_rows.extend(identity | profile for profile in spectrum_profile)
        dct_rows.extend(identity | profile for profile in dct_profile)
        if position % 50 == 0 or position == len(index):
            print(f"features: {position}/{len(index)}", flush=True)

    features = pd.DataFrame(feature_rows)
    radial = pd.DataFrame(radial_rows)
    dct = pd.DataFrame(dct_rows)
    registry = registry_frame("expanded_v2")
    effects = feature_effects(features, registry)
    paired = paired_dda_effects(features, effects)
    features.to_csv(output_dir / "image_features.csv.gz", index=False)
    radial.to_csv(output_dir / "radial_profiles.csv.gz", index=False)
    dct.to_csv(output_dir / "dct_profiles.csv.gz", index=False)
    effects.to_csv(output_dir / "feature_effects.csv", index=False)
    paired.to_csv(output_dir / "dda_paired_effects.csv", index=False)
    registry.to_csv(output_dir / "feature_registry.csv", index=False)
    run_metadata = {
        "schema_version": 1,
        "canonical_size": size,
        "images": len(features),
        "datasets": features.groupby(["dataset", "label"]).size().to_dict(),
        "input_external_index_sha256": hashlib.sha256(
            (repo_root / "data/metadata/signal_analysis_index.csv").read_bytes()
        ).hexdigest(),
        "cautions": [
            "PRNU and DSNU are not directly measurable from this single-image, processed-image corpus.",
            "Camera features are explicitly proxies for CFA/color coupling and signal-dependent residual behavior.",
            "Cross-source comparisons are descriptive and may contain codec, content, and resolution confounding.",
            "EvalGEN contains generated images only; its authentic comparison is necessarily cross-source.",
        ],
    }
    run_metadata["datasets"] = {"|".join(key): int(value) for key, value in run_metadata["datasets"].items()}
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(effects.groupby("scope").head(8).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    run(args.repo_root, args.size)


if __name__ == "__main__":
    main()
