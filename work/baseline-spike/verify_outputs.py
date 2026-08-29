from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
RAW = ROOT / "work" / "baseline-spike" / "expanded-results"
BASE = ROOT / "work" / "baseline-spike" / "results"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


summary = read_json(OUTPUTS / "expanded-baseline-results.json")
report = (OUTPUTS / "expanded-baseline-report.md").read_text(encoding="utf-8")
report_without_emphasis = report.replace("**", "")
confirmation = read_json(RAW / "confirmation_sid_40x2.json")
external = read_json(RAW / "external_generator_recall.json")

assert summary["scope"] == "zero-shot baselines only; no fine-tuning"
assert len(summary["gate"]["models"]) == 10
assert len(confirmation["predictions"]) == 80 * 15
assert len(confirmation["conditions"]) == 15
assert len(external["predictions"]) == 30
assert all(not read_json(path).get("fine_tuning", True) for path in BASE.glob("*.json") if "fine_tuning" in read_json(path))
assert all(
    not read_json(path).get("fine_tuning", True)
    for path in RAW.glob("*.json")
    if "fine_tuning" in read_json(path)
)

labels = {
    "clean": "Clean",
    "jpeg_q90": "JPEG Q90",
    "jpeg_q70": "JPEG Q70",
    "jpeg_q50": "JPEG Q50",
    "jpeg_q30": "JPEG Q30",
    "blur_sigma0.5": "Blur σ=0.5",
    "blur_sigma1": "Blur σ=1",
    "blur_sigma2": "Blur σ=2",
    "resize_0.5": "Resize 0.5×",
    "resize_0.25": "Resize 0.25×",
    "noise_sigma0.02": "Noise σ=0.02",
    "noise_sigma0.05": "Noise σ=0.05",
    "noise_sigma0.10": "Noise σ=0.10",
    "color_jitter_20": "Color jitter ±20%",
    "center_crop_80": "Center crop 80%",
}
for condition, metric in confirmation["conditions"].items():
    confusion = metric["confusion"]
    expected = (
        f"| {labels[condition]} | {metric['balanced_accuracy']:.3f} | "
        f"{metric['roc_auc']:.3f} | {confusion['false_positive']} | "
        f"{confusion['false_negative']} |"
    )
    assert expected in report_without_emphasis, expected
    assert summary["confirmation"]["conditions"][condition] == metric

assert summary["external_generator_recall"]["results"] == external["results"]
assert "C:\\Users" not in report
assert "C:\\Users" not in (OUTPUTS / "expanded-baseline-results.json").read_text(encoding="utf-8")
assert all((OUTPUTS / name).is_file() for name in (
    "expanded-baseline-report.md",
    "expanded-baseline-results.json",
    "expanded-baseline.py",
    "confirmation-baseline.py",
))
print("output verification passed: 10 models, 15 conditions, 1,200 confirmation predictions, 30 external predictions")
