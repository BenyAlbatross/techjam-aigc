from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results"
EXPANDED = ROOT / "expanded-results"
OUTPUT = ROOT.parents[1] / "outputs" / "expanded-baseline-results.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gate_row(name: str, path: Path) -> dict:
    run = read(path)
    conditions = run["conditions"]
    worst_name, worst = min(
        conditions.items(), key=lambda item: item[1]["balanced_accuracy"]
    )
    clean = conditions["clean"]
    parameters = run.get("model_info", {}).get("parameters")
    if parameters is None:
        parameters = run.get("declared", {}).get("parameters")
    return {
        "model": name,
        "parameters": parameters,
        "clean_balanced_accuracy": clean["balanced_accuracy"],
        "clean_roc_auc": clean["roc_auc"],
        "worst_endpoint_condition": worst_name,
        "worst_endpoint_balanced_accuracy": worst["balanced_accuracy"],
        "worst_endpoint_roc_auc": worst["roc_auc"],
    }


def main() -> None:
    gate_files = {
        "Ateeqq SigLIP": EXPANDED / "ateeqq_siglip.json",
        "Frontier Community Forensics": EXPANDED / "frontier_community_forensics.json",
        "Community Forensics": EXPANDED / "community_forensics.json",
        "Steganograph ViT": BASE / "steganograph.json",
        "Divine EfficientNet-B0": EXPANDED / "divine_efficientnet.json",
        "wkaandemir CLIP": EXPANDED / "wkaandemir_clip.json",
        "Divine ConvNeXt-Tiny": EXPANDED / "divine_convnext.json",
        "Divine ResNet-50": EXPANDED / "divine_resnet50.json",
        "CapCheck ViT": BASE / "capcheck.json",
        "UnivFD": BASE / "univfd.json",
    }
    heuristics = read(BASE / "heuristics.json")["metrics"]
    ensembles = read(EXPANDED / "all_ensembles.json")
    confirmation = read(EXPANDED / "confirmation_sid_40x2.json")
    external = read(EXPANDED / "external_generator_recall.json")
    payload = {
        "generated_on": "2026-08-29",
        "scope": "zero-shot baselines only; no fine-tuning",
        "competition_constraints": {
            "parameter_limit": "under 2 billion parameters",
            "required_output_fields": ["image_path", "pred"],
            "required_robustness_conditions": list(confirmation["conditions"]),
            "track_brief": "https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc",
            "rules": "https://tiktoktechjam2026.devpost.com/rules",
        },
        "gate": {
            "dataset": "saberzl/SID_Set validation",
            "selection": "12 real + 12 fully synthetic; first rows by class; label-2 excluded",
            "threshold_policy": "published/default threshold, frozen before evaluation",
            "conditions": [
                "clean",
                "jpeg_q30",
                "blur_sigma2",
                "resize_0.25",
                "noise_sigma0.10",
                "color_jitter_20",
                "center_crop_80",
            ],
            "models": [gate_row(name, path) for name, path in gate_files.items()],
            "heuristics": heuristics,
            "decision_ensembles": ensembles,
        },
        "confirmation": {
            "model": confirmation["model"],
            "dataset": confirmation["dataset"],
            "selection": "40 real + 40 fully synthetic; first rows by class; label-2 excluded",
            "conditions": confirmation["conditions"],
            "seconds_total_cpu": confirmation["seconds_total"],
            "persistent_error_summary": {
                "total_errors_across_1200_predictions": 104,
                "unique_images_ever_wrong": 17,
                "most_persistent_real_false_positive_conditions": 15,
            },
        },
        "external_generator_recall": {
            "source": "openai/dalle3-eval-samples",
            "revision": "d7c88c07b492ad7b9fd3003126d00719a2edabb1",
            "sample": "10 clean images per generator; synthetic recall only",
            "results": external["results"],
        },
        "legal_and_scope_register": {
            "datasets_used": [
                {
                    "id": "saberzl/SID_Set",
                    "license": "CC BY 4.0",
                    "use": "gate and confirmation",
                },
                {
                    "id": "openai/dalle3-eval-samples",
                    "license": "MIT repository license",
                    "use": "positive-only DALL-E 3, Midjourney, SDXL slice",
                },
            ],
            "conditional_models": [
                "Ateeqq SigLIP: Apache-2.0 weights, but the card does not identify the 120,000 training images; clear provenance before submission",
                "Frontier Community Forensics: MIT weights, but organizer suitability should be confirmed and source licenses remain source-specific",
            ],
            "not_run": [
                "xRayon ConvNeXtV2-Base: MIT, but checkpoint is a 1.05 GB training-state download; defer to GPU",
                "Sentry: model/data commercial use requires written permission",
                "Hussein detector: CC BY-NC 4.0",
                "UMM detector: CC BY-ND",
                "PatchCraft/NPR checkpoints: licensing not clear enough for submission",
            ],
        },
        "decision": {
            "technical_baseline": "Ateeqq SigLIP",
            "reason": "best fixed-threshold robustness, strong larger-sample AUC, and 29/30 external generator recall",
            "main_failure": "false positives on compressed/watermarked or staged real web images; worst under 0.25x resize and sigma-2 blur",
            "ensemble_decision": "do not use current vote ensembles; they reduce heavy-noise recall",
            "submission_caveat": "training-image provenance must be cleared before treating Ateeqq SigLIP as a final bundled dependency",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
