from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "infer_trace_rx_m.py"


def _module():
    spec = importlib.util.spec_from_file_location("infer_trace_rx_m", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_images_emits_submission_contract(tmp_path: Path) -> None:
    module = _module()
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), "navy").save(image)

    class Detector:
        def __call__(self, pixels):
            assert tuple(pixels.shape) == (1, 3, 16, 16)
            return type("Output", (), {"logit": torch.tensor([0.0])})()

    preprocessing = type("Preprocessing", (), {
        "validate": lambda self: None,
        "image_size": 16,
        "image_mean": (0.485, 0.456, 0.406),
        "image_std": (0.229, 0.224, 0.225),
        "max_short_side": 32,
    })()

    rows = module.predict_images(Detector(), preprocessing, [image], torch.device("cpu"))

    assert rows == [{"image_path": str(image), "pred": 0.5}]
    assert set(rows[0]) == {"image_path", "pred"}


def test_write_predictions_preserves_exact_keys(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "predictions.json"
    module.write_predictions([{"image_path": "x.png", "pred": 0.25}], output)
    assert json.loads(output.read_text()) == [{"image_path": "x.png", "pred": 0.25}]
