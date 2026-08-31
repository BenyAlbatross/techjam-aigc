from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_error_analysis_report.py"
SPEC = importlib.util.spec_from_file_location("generate_error_analysis_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepare_predictions_converts_logits_and_assigns_all_outcomes() -> None:
    frame = pd.DataFrame({"target": [0, 0, 1, 1], "logit": [-2.0, 2.0, -2.0, 2.0]})

    result, column, kind = MODULE.prepare_predictions(frame)

    assert column == "logit"
    assert kind == "logit"
    assert result["outcome"].tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    assert result["correct"].tolist() == [True, False, False, True]


def test_report_embeds_correct_and_wrong_image_galleries(tmp_path: Path) -> None:
    image_paths = []
    for index, color in enumerate(("navy", "orange", "purple", "green")):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (24 + index, 18 + index), color).save(path)
        image_paths.append(path.name)
    scores = tmp_path / "scores.csv"
    pd.DataFrame({
        "parent_id": [f"asset-{index}" for index in range(4)],
        "local_path": image_paths,
        "target": [0, 0, 1, 1],
        "logit": [-2.0, 2.0, -2.0, 2.0],
        "source_dataset": ["real-a", "real-b", "fake-a", "fake-b"],
        "generator_family": ["authentic", "authentic", "gen-a", "gen-b"],
    }).to_csv(scores, index=False)
    output = tmp_path / "report.html"

    metrics = MODULE.generate_report(scores, output, repo_root=tmp_path, title="Test report")

    assert metrics == {
        "rows": 4,
        "accuracy": 0.5,
        "correct": 2,
        "wrong": 2,
        "false_negative": 1,
        "false_positive": 1,
        "true_positive": 1,
        "true_negative": 1,
        "embedded_correct": 2,
        "embedded_wrong": 2,
        "score_column": "logit",
        "score_type": "logit",
    }
    html = output.read_text()
    assert "data:image/jpeg;base64," in html
    assert html.count('class="image-card ') == 4
    assert "False negative" in html
    assert "False positive" in html
    assert "Test report" in html
