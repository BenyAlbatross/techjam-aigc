from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "sid_validation" / "manifest.json"
DELIVERABLES = ROOT.parents[1] / "outputs"
MODEL_ORDER = ("steganograph", "capcheck", "univfd")
DISPLAY = {
    "steganograph": "Steganograph modern-generator ViT",
    "capcheck": "CapCheck CIFAKE ViT",
    "univfd": "UnivFD CLIP ViT-L/14",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    heuristics = json.loads((RESULTS / "heuristics.json").read_text(encoding="utf-8"))
    models = {name: load(name) for name in MODEL_ORDER}

    public_manifest = {**manifest}
    public_manifest["samples"] = [
        {key: value for key, value in sample.items() if key != "path"}
        for sample in manifest["samples"]
    ]
    aggregate = {
        "scope": {
            "purpose": "Disposable zero-training feasibility spike; not a leaderboard estimate",
            "sample_size": 24,
            "class_balance": {"real": 12, "ai": 12},
            "threshold": 0.5,
            "fine_tuning": False,
            "hardware": "CPU-only Torch 2.13.0",
            "runner_commit": "d3f4976d36c59974a25f55a0a7850b9866d3223b",
        },
        "manifest": public_manifest,
        "heuristics": heuristics["metrics"],
        "models": models,
    }
    output_json = DELIVERABLES / "baseline-results.json"
    output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    rows = []
    for name in MODEL_ORDER:
        for condition, metric in models[name]["conditions"].items():
            confusion = metric["confusion"]
            rows.append(
                f"| {DISPLAY[name]} | {condition} | {metric['balanced_accuracy']:.3f} | "
                f"{metric['roc_auc']:.3f} | {confusion['false_positive']} | "
                f"{confusion['false_negative']} |"
            )

    heuristic_rows = []
    for name, metric in heuristics["metrics"].items():
        heuristic_rows.append(f"| {name} | {metric['roc_auc']:.3f} |")

    best_predictions = models["steganograph"]["predictions"]
    errors = [row for row in best_predictions if row["truth"] != row["pred"]]
    error_counts = Counter(row["sample_id"] for row in errors)
    repeated = ", ".join(f"`{sample_id}` ({count} conditions)" for sample_id, count in error_counts.most_common(3))

    report = f"""# TikTok TechJam Track 5: zero-training baseline spike

## Decision

Carry **`delpot/steganograph-ia-detector`** into the next, larger no-training benchmark. Keep CapCheck only as a calibration-mismatch control and drop UnivFD from the competition prototype unless an independent implementation check reverses this result.

This is a 24-image feasibility spike, not a model-selection benchmark. Its strongest finding is qualitative: the modern-generator ViT separates this SID-Set slice well, but its fixed threshold fails differently under different transformations. Noise produces false negatives; blur and resize produce false positives. A single clean accuracy number would conceal that.

## Requirements that shaped the test

The official Track 5 brief asks for an image AIGC detector robust to redistribution transforms, a model below 2B parameters, a robustness table, and error analysis. Its reference transformations include JPEG quality 30, Gaussian blur sigma 2, resize to 0.25x then upscale, Gaussian noise sigma 0.10, brightness/contrast/saturation jitter of 20%, and center crop to 80%. The final prototype must process an image directory and emit JSON objects with `image_path` and `pred`. [Official Track 5 brief](https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc)

The event runs for 72 hours and requires a public repository, written technical description, and public three-minute demo. [Official Devpost](https://tiktoktechjam2026.devpost.com/)

The rules permit third-party open-source code, models, and data only when the team is authorized to use them and complies with their licences. [Official rules](https://tiktoktechjam2026.devpost.com/rules)

## Test design

- **Data:** first 12 real and first 12 fully synthetic examples from the SID-Set validation stream; tampered label 2 excluded. SID-Set declares CC BY 4.0. Original bytes and SHA-256 hashes were preserved locally. No images are included in these deliverables. [SID-Set card](https://huggingface.co/datasets/saberzl/SID_Set)
- **Models:** modern-generator ViT (85.8M, MIT), CapCheck/CIFAKE ViT (85.8M, Apache-2.0), and UnivFD CLIP ViT-L/14 (~428M, MIT). All are below 2B. [Modern ViT](https://huggingface.co/delpot/steganograph-ia-detector), [CapCheck](https://huggingface.co/capcheck/ai-image-detection), [UnivFD](https://github.com/WisconsinAIVision/UniversalFakeDetect)
- **Protocol:** no fine-tuning, fitting, ensembling, or threshold selection. Each model used its native preprocessing and fixed 0.5 threshold. Transformations were deterministic and class-symmetric.
- **Metrics:** AUROC measures ranking; balanced accuracy and FP/FN use the untouched 0.5 threshold.
- **Runtime:** CPU-only Torch. The harness is pinned to runner commit `d3f4976d36c59974a25f55a0a7850b9866d3223b`; HF revisions are recorded in the JSON.

## Results

| Model | Condition | Balanced accuracy | AUROC | FP / 12 | FN / 12 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Interpretation

- **Modern-generator ViT:** clean balanced accuracy 0.958 and AUROC 0.979. Worst thresholded condition is noise sigma 0.10: balanced accuracy 0.583 with 9/12 synthetic images missed. Blur causes 5/12 real images to be accused; resize causes 6/12.
- **CapCheck:** clean AUROC 0.826 but balanced accuracy only 0.542 because it labels 11/12 real images as AI at threshold 0.5. It ranks reasonably but is badly miscalibrated out of domain.
- **UnivFD:** all conditions predict every image as real at threshold 0.5; clean AUROC is 0.431. On this slice, its larger backbone is a negative result.

Do not tune a new threshold on these 24 images. That would turn the test set into training data and exaggerate progress.

## Shortcut audit

| Heuristic | AUROC |
|---|---:|
{chr(10).join(heuristic_rows)}

The 0.917 result for square/exact-1024 geometry is a dataset shortcut: most synthetic examples in this slice are 1024×1024 while real photos vary in aspect ratio. Metadata keywords are chance. Native-dimension heuristics must therefore be reported separately from canonicalized detector evaluation and never presented as forensic evidence.

## Representative errors

The modern ViT repeatedly misclassified: {repeated}.

- Real sample `f2b02fe7b2ddcb9c` is a controlled dental photograph with strong blown highlights, shallow depth of field, PPE, and smooth background regions. It is the lone clean false positive and remains a false positive under several transforms.
- Synthetic samples `full_synthetic_007473` (sunlit horses) and `full_synthetic_000008` (aircraft against a smooth sky) become false negatives after JPEG/noise. Both resemble conventional photography and contain broad smooth gradients; the transformations likely erase or overwhelm the cues used by the classifier.

These are visual hypotheses, not causal proof. A larger source-balanced error set is needed before designing features around them.

## Legal dataset expansion

- **Green for a synthetic-source extension:** DiffusionDB declares CC0 1.0 and can add Stable Diffusion diversity, but it is old-generator-heavy and should be treated as one generator family, not proof of generalization. [DiffusionDB card](https://huggingface.co/datasets/poloclub/diffusiondb)
- **Conditional:** SID-Set declares CC BY 4.0, but public redistribution or demo display should retain dataset citation and the underlying image-level attribution/licence metadata. Do not publish this local sample until that manifest is complete.
- **Conditional:** Open Images lists photos as CC BY but explicitly tells users to verify each image's licence. Use only entries with retained creator/source/licence fields. [Open Images notice](https://github.com/openimages/dataset/blob/main/READMEV1.md)
- **Hold:** do not add CIFAKE, WildFake, or another large benchmark merely because it is public. Require an explicit dataset licence, permitted competition use, redistribution terms, and an attribution path first.

This is an engineering screen, not legal advice.

## Next experiment

1. Freeze a 200–400 image no-training evaluation set with at least two real-source families and three AI-generator families. Keep every source/licence/generator field.
2. Run the modern-generator ViT over the complete official severity grid, not only the severe endpoints used here.
3. Report clean-to-transform deltas, per-condition FP/FN, AUROC, balanced accuracy, and score-distribution plots. Keep the 0.5 threshold frozen.
4. Add the official reference validation set when available, without training on it.
5. Only after that decide whether a second representation or ensemble is justified.

## Limitations

The sample is tiny, the first balanced rows are not a random population sample, SID-Set generator identities are unavailable in this manifest, confidence intervals are wide, and one dataset cannot establish cross-generator or cross-source generalization. Results are useful for pruning options, not for claiming detector performance.
"""
    output_report = DELIVERABLES / "baseline-spike-report.md"
    output_report.write_text(report, encoding="utf-8")

    shutil.copy2(ROOT / "spike.py", DELIVERABLES / "baseline-spike.py")
    print(output_report)
    print(output_json)


if __name__ == "__main__":
    main()
