import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import scripts.fetch_models as fetch_models
import scripts.model_adapters as model_adapters
from scripts.fetch_models import fetch_model
from scripts.fetch_models import sha256_file
from scripts.model_adapters import ModelAdapter
from scripts.model_adapters import assert_parameter_limit
from scripts.model_adapters import load_model


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

    [(raw_score, probability_ai)] = adapter.score_batch([Image.new("RGB", (1, 1))])

    assert probability_ai == pytest.approx(0.880797)
    assert raw_score == 2.0
    assert (adapter.name, adapter.threshold, adapter.parameter_count) == (
        "ateeqq_siglip", 0.37, 1,
    )
    assert (adapter.revision, adapter.weight_sha256) == ("abc123", "f" * 64)


@pytest.mark.parametrize(
    ("loader", "extra", "expected"),
    [
        ("hf_ai_logit", {}, 0.731059),
        ("timm_ai_logit", {}, 0.731059),
        ("timm_real_logit", {"temperature": 0.595}, 0.157006),
        ("torchvision_real_logit", {}, 0.268941),
    ],
)
def test_single_logit_tuple_is_raw_then_ai_probability(
    loader: str, extra: dict, expected: float
):
    adapter = ModelAdapter(
        "detector",
        entry(loader, **extra),
        FakeModel([[1.0]]),
        "cpu",
        preprocess=lambda image: torch.zeros(1),
    )

    [(raw_score, probability_ai)] = adapter.score_batch([Image.new("RGB", (1, 1))])

    assert probability_ai == pytest.approx(expected, abs=1e-6)
    assert raw_score == 1.0


class FakeDetector:
    def __init__(self):
        self.model = FakeModel([[0.0]])

    def predict_images(self, images):
        return [
            SimpleNamespace(raw_score=-0.25, probability_ai=0.75)
            for image in images
        ]


@pytest.mark.parametrize("loader", ["aidetector_hf", "aidetector_univfd"])
def test_aidetector_tuple_is_raw_then_ai_probability(loader: str):
    adapter = ModelAdapter("detector", entry(loader), FakeDetector(), "cpu")

    assert adapter.score_batch([Image.new("RGB", (1, 1))]) == [(-0.25, 0.75)]


def test_adapter_rejects_registry_parameter_mismatch():
    with pytest.raises(RuntimeError, match="parameter count mismatch"):
        ModelAdapter(
            "detector",
            entry("timm_ai_logit", parameters=2),
            FakeModel([[0.0]]),
            "cpu",
            preprocess=lambda image: torch.zeros(1),
        )




def test_torchscript_auxiliary_loader_avoids_pickle_loading(tmp_path: Path, monkeypatch):
    archive = tmp_path / "auxiliary.pt"
    module = torch.jit.trace(torch.nn.Linear(1, 1), torch.zeros(1, 1))
    torch.jit.save(module, archive)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("pickle load"))

    loaded = model_adapters.load_torchscript(archive)

    assert torch.equal(loaded(torch.zeros(1, 1)), module(torch.zeros(1, 1)))


def test_univfd_loader_assigns_openai_metadata_preprocess(tmp_path: Path, monkeypatch):
    from types import ModuleType

    class FakeDetector:
        def _load_head(self, path):
            self.loaded_head = path

    class FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = SimpleNamespace(
                output_dim=1,
                image_size=224,
                image_mean=(0.1, 0.2, 0.3),
                image_std=(0.4, 0.5, 0.6),
            )

    detector_module = ModuleType("aidetector.model")
    detector_module.AIImageDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "aidetector.model", detector_module)
    import open_clip.model as open_clip_model
    monkeypatch.setattr(
        open_clip_model, "build_model_from_openai_state_dict", lambda state, cast_dtype: FakeClip()
    )
    monkeypatch.setattr(open_clip_model, "get_cast_dtype", lambda precision: None)
    monkeypatch.setattr(
        model_adapters, "load_torchscript", lambda path: SimpleNamespace(state_dict=dict)
    )
    entry_data = entry("aidetector_univfd", architecture="ViT-L-14")
    entry_data["repository"] = "owner/repo"
    entry_data["file"] = "head.pth"
    entry_data["auxiliary"] = {"file": "auxiliary.pt"}

    detector = model_adapters._load_univfd(entry_data, tmp_path, "cpu")

    assert detector.preprocess(Image.new("RGB", (8, 8))).shape == (3, 224, 224)
def test_univfd_preprocess_uses_openai_torchscript_image_metadata():
    visual = SimpleNamespace(
        image_size=224,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
    )

    tensor = model_adapters.univfd_preprocess(visual)(Image.new("RGB", (8, 8)))

    assert tensor.shape == (3, 224, 224)
def test_pinned_hf_detector_parameter_counts_match_verified_weights():
    with (model_adapters.ROOT / "models.toml").open("rb") as handle:
        models = tomllib.load(handle)["models"]

    assert models["steganograph"]["parameters"] == 85_800_194
    assert models["capcheck"]["parameters"] == 85_800_194
    assert models["univfd"]["parameters"] == 427_617_282
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

    def local_snapshot_download(
        repo_id, revision, cache_dir, allow_patterns, local_files_only=False
    ):
        snapshots = {("owner/repo", "fixed"): snapshot}
        return str(snapshots[(repo_id, revision)])

    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_models, "snapshot_download", local_snapshot_download)

    with pytest.raises(RuntimeError, match="auxiliary checkpoint hash mismatch"):
        fetch_model("demo", tmp_path / "cache")

    assert primary.exists()
    assert not auxiliary.exists()


def test_offline_fetch_requires_an_existing_snapshot(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weight = snapshot / "model.bin"
    weight.write_bytes(b"verified")
    (tmp_path / "models.toml").write_text(
        '[models.demo]\nstatus="approved"\nsubmission_status="review"\n'
        'repository="owner/repo"\nrevision="fixed"\nfile="model.bin"\n'
        f'sha256="{sha256_file(weight)}"\nlicense="MIT"\nthreshold=0.5\n'
        'parameters=1\nloader="hf_ai_logit"\n'
    )

    def cached_snapshot_download(
        repo_id, revision, cache_dir, allow_patterns, *, local_files_only
    ):
        if not local_files_only:
            raise RuntimeError("network access remained enabled")
        return str(snapshot)

    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_models, "snapshot_download", cached_snapshot_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert fetch_model("demo", tmp_path / "cache") == snapshot



def test_fetch_cli_all_expands_to_every_registered_model(tmp_path: Path, monkeypatch, capsys):
    registry = tmp_path / "models.toml"
    registry.write_text(
        '[models.first]\nstatus="approved"\nsubmission_status="review"\n'
        'repository="owner/first"\nrevision="fixed"\nfile="one.bin"\n'
        'sha256="' + "0" * 64 + '"\nlicense="MIT"\nthreshold=0.5\n'
        'parameters=1\nloader="hf_ai_logit"\n'
        '[models.second]\nstatus="approved"\nsubmission_status="review"\n'
        'repository="owner/second"\nrevision="fixed"\nfile="two.bin"\n'
        'sha256="' + "1" * 64 + '"\nlicense="MIT"\nthreshold=0.5\n'
        'parameters=1\nloader="hf_ai_logit"\n'
    )
    fetched = []
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
    monkeypatch.setattr(
        fetch_models, "fetch_model",
        lambda name, cache: fetched.append((name, cache)) or cache / name,
    )

    assert fetch_models.main(["--model", "all", "--cache", str(tmp_path / "cache")]) == 0
    assert fetched == [
        ("first", tmp_path / "cache"),
        ("second", tmp_path / "cache"),
    ]
    assert "first verified one.bin" in capsys.readouterr().out
def test_load_model_sets_offline_before_fetch(monkeypatch, tmp_path: Path):
    observed = []

    class FetchObserved(RuntimeError):
        pass

    def observe_fetch(name, cache):
        observed.append(os.environ.get("HF_HUB_OFFLINE"))
        raise FetchObserved

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_adapters, "fetch_model", observe_fetch)

    with pytest.raises(FetchObserved):
        load_model("ateeqq_siglip", "cpu", tmp_path)

    assert observed == ["1"]


@pytest.mark.parametrize("script", ["scripts/fetch_models.py", "scripts/model_adapters.py"])
def test_model_cli_scripts_import_from_direct_execution(script: str):
    result = subprocess.run(
        [sys.executable, script, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
