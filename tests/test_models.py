from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import scripts.fetch_models as fetch_models
from scripts.fetch_models import fetch_model
from scripts.fetch_models import sha256_file
from scripts.model_adapters import ModelAdapter
from scripts.model_adapters import assert_parameter_limit


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "weight.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_parameter_limit_is_strict():
    assert_parameter_limit(1_999_999_999)
    with pytest.raises(RuntimeError, match="2B"):
        assert_parameter_limit(2_000_000_000)


class FakeModel(torch.nn.Module):
    def __init__(self, logits: list[list[float]], labels: dict[int, str] | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.logits = torch.tensor(logits)
        self.config = SimpleNamespace(id2label=labels or {})

    def forward(self, *args, **kwargs):
        return SimpleNamespace(logits=self.logits + self.weight * 0)


def entry(loader: str, **values) -> dict:
    return {
        "loader": loader,
        "threshold": 0.37,
        "parameters": 1,
        "revision": "abc123",
        "sha256": "f" * 64,
        **values,
    }


def processor(*, images, return_tensors):
    return {"pixel_values": torch.zeros(len(images), 1)}


def test_hf_label_tokens_score_ai_and_retain_registry_metadata():
    adapter = ModelAdapter(
        "ateeqq_siglip",
        entry("hf_multiclass"),
        FakeModel([[0.0, 2.0]], {0: "human", 1: "AI-generated"}),
        "cpu",
        processor=processor,
    )

    [(probability_ai, raw_score)] = adapter.score_batch([Image.new("RGB", (1, 1))])

    assert probability_ai == pytest.approx(0.880797)
    assert raw_score == 2.0
    assert (adapter.name, adapter.threshold, adapter.parameter_count) == (
        "ateeqq_siglip", 0.37, 1,
    )
    assert (adapter.revision, adapter.weight_sha256) == ("abc123", "f" * 64)


@pytest.mark.parametrize(
    ("loader", "extra", "expected"),
    [
        ("timm_ai_logit", {}, 0.731059),
        ("timm_real_logit", {"temperature": 0.595}, 0.157006),
    ],
)
def test_single_logit_direction(loader: str, extra: dict, expected: float):
    adapter = ModelAdapter(
        "detector",
        entry(loader, **extra),
        FakeModel([[1.0]]),
        "cpu",
        preprocess=lambda image: torch.zeros(1),
    )

    [(probability_ai, raw_score)] = adapter.score_batch([Image.new("RGB", (1, 1))])

    assert probability_ai == pytest.approx(expected, abs=1e-6)
    assert raw_score == 1.0


def test_adapter_rejects_registry_parameter_mismatch():
    with pytest.raises(RuntimeError, match="parameter count mismatch"):
        ModelAdapter(
            "detector",
            entry("timm_ai_logit", parameters=2),
            FakeModel([[0.0]]),
            "cpu",
            preprocess=lambda image: torch.zeros(1),
        )


def test_fetch_verifies_auxiliary_and_deletes_only_mismatch(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    (snapshot / "nested").mkdir(parents=True)
    primary = snapshot / "model.bin"
    auxiliary = snapshot / "nested" / "aux.bin"
    primary.write_bytes(b"primary")
    auxiliary.write_bytes(b"wrong")
    registry = tmp_path / "models.toml"
    registry.write_text(
        '[models.demo]\nstatus="approved"\nsubmission_status="review"\n'
        'repository="owner/repo"\nrevision="fixed"\nfile="model.bin"\n'
        f'sha256="{sha256_file(primary)}"\nlicense="MIT"\nthreshold=0.5\n'
        'parameters=1\nloader="aidetector_univfd"\n'
        '[models.demo.auxiliary]\nfile="nested/aux.bin"\nsha256="' + "0" * 64 + '"\n'
    )

    def local_snapshot_download(repo_id, revision, cache_dir, allow_patterns):
        snapshots = {("owner/repo", "fixed"): snapshot}
        return str(snapshots[(repo_id, revision)])

    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_models, "snapshot_download", local_snapshot_download)

    with pytest.raises(RuntimeError, match="auxiliary checkpoint hash mismatch"):
        fetch_model("demo", tmp_path / "cache")

    assert primary.exists()
    assert not auxiliary.exists()


@pytest.mark.parametrize("script", ["scripts/fetch_models.py", "scripts/model_adapters.py"])
def test_model_cli_scripts_import_from_direct_execution(script: str):
    result = subprocess.run(
        [sys.executable, script, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
