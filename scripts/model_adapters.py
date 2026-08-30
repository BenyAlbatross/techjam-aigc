from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

if __package__:
    from scripts.compliance import check_models, load_registry
    from scripts.fetch_models import fetch_model
else:
    from compliance import check_models, load_registry
    from fetch_models import fetch_model


ROOT = Path(__file__).resolve().parents[1]


def load_torchscript(path: Path):
    return torch.jit.load(path, map_location="cpu")

def assert_parameter_limit(count: int) -> None:
    if count >= 2_000_000_000:
        raise RuntimeError(f"Model exceeds 2B parameter limit: {count}")


def _count_parameters(model) -> int:
    if isinstance(model, torch.nn.Module):
        return sum(parameter.numel() for parameter in model.parameters())
    if hasattr(model, "model"):
        return sum(parameter.numel() for parameter in model.model.parameters())
    return sum(
        parameter.numel()
        for component in (model.clip_model, model.head)
        for parameter in component.parameters()
    )


class ModelAdapter:
    def __init__(
        self,
        name: str,
        entry: dict,
        model,
        device: str,
        *,
        processor=None,
        preprocess=None,
    ) -> None:
        self.name = name
        self.threshold = float(entry["threshold"])
        self.revision = entry["revision"]
        self.weight_sha256 = entry["sha256"]
        self.kind = entry["loader"]
        self.temperature = float(entry.get("temperature", 1.0))
        self.model = model
        self.device = torch.device(device)
        self.processor = processor
        self.preprocess = preprocess
        self.parameter_count = _count_parameters(model)
        assert_parameter_limit(self.parameter_count)
        expected = int(entry["parameters"])
        if self.parameter_count != expected:
            raise RuntimeError(
                f"{name}: parameter count mismatch: registry={expected}, actual={self.parameter_count}"
            )

    def to_ai_scores(self, logits: torch.Tensor) -> list[tuple[float, float]]:
        logits = logits.detach().float().cpu()
        if self.kind == "hf_multiclass":
            fake_tokens = ("ai", "fake", "synthetic", "generated")
            real_tokens = ("human", "hum", "real", "authentic", "natural")
            fake_indices = []
            for index, label in self.model.config.id2label.items():
                label = str(label).lower()
                if any(token in label for token in fake_tokens) and not any(
                    token in label for token in real_tokens
                ):
                    fake_indices.append(int(index))
            if not fake_indices:
                raise RuntimeError(f"Cannot infer AI label: {self.model.config.id2label}")
            probabilities = torch.softmax(logits, dim=-1)
            return [
                (
                    float(raw[fake_indices].mean().item()),
                    float(row[fake_indices].sum().item()),
                )
                for row, raw in zip(probabilities, logits, strict=True)
            ]

        raw_values = logits.reshape(-1)
        if self.kind in {"hf_ai_logit", "timm_ai_logit"}:
            probabilities = torch.sigmoid(raw_values)
        elif self.kind in {"timm_real_logit", "torchvision_real_logit"}:
            probabilities = 1.0 - torch.sigmoid(raw_values / self.temperature)
        else:
            raise RuntimeError(f"Unsupported score conversion: {self.kind}")
        return [
            (float(raw.item()), float(probability.item()))
            for probability, raw in zip(probabilities, raw_values, strict=True)
        ]

    def score_batch(self, images: list[Image.Image]) -> list[tuple[float, float]]:
        if not images:
            return []
        images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        if self.kind in {"aidetector_hf", "aidetector_univfd"}:
            return [
                (result.raw_score, result.probability_ai)
                for result in self.model.predict_images(images)
            ]
        if self.processor is not None:
            values = self.processor(images=images, return_tensors="pt")
            values = {key: value.to(self.device) for key, value in values.items()}
            with torch.inference_mode():
                output = self.model(**values)
        else:
            values = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            with torch.inference_mode():
                output = self.model(values)
        return self.to_ai_scores(getattr(output, "logits", output))

    def score_one(self, image: Image.Image) -> tuple[float, float]:
        return self.score_batch([image])[0]


def _hf_components(entry: dict, device: str, cache: Path):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    kwargs = {
        "revision": entry["revision"],
        "cache_dir": cache,
        "local_files_only": True,
    }
    processor = AutoImageProcessor.from_pretrained(entry["repository"], **kwargs)
    model = AutoModelForImageClassification.from_pretrained(
        entry["repository"], **kwargs
    ).to(device).eval()
    return model, processor


def _load_aidetector_hf(entry: dict, device: str, cache: Path):
    from aidetector.model import HuggingFaceImageDetector

    model, processor = _hf_components(entry, device, cache)
    detector = HuggingFaceImageDetector.__new__(HuggingFaceImageDetector)
    detector._torch = torch
    detector.device = torch.device(device)
    detector.threshold = float(entry["threshold"])
    detector.model_id = entry["repository"]
    detector.model = model
    detector.id2label = model.config.id2label
    detector.processor = processor
    return detector


def univfd_preprocess(visual):
    from open_clip.transform import PreprocessCfg, image_transform_v2

    return image_transform_v2(PreprocessCfg(
        size=visual.image_size, mean=visual.image_mean, std=visual.image_std
    ), is_train=False)

def _load_univfd(entry: dict, snapshot: Path, device: str):
    from aidetector.model import AIImageDetector
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    from open_clip.model import build_model_from_openai_state_dict, get_cast_dtype

    detector = AIImageDetector.__new__(AIImageDetector)
    detector._torch = torch
    detector.device = torch.device(device)
    detector.threshold = float(entry["threshold"])
    detector.model_name = entry["architecture"]
    detector.pretrained = str(snapshot / entry["auxiliary"]["file"])
    detector.repo_id = entry["repository"]
    archive = load_torchscript(Path(detector.pretrained))
    detector.clip_model = build_model_from_openai_state_dict(
        archive.state_dict(), cast_dtype=get_cast_dtype("fp32")
    ).to(detector.device).float().eval()
    detector.clip_model.visual.image_mean = OPENAI_DATASET_MEAN
    detector.clip_model.visual.image_std = OPENAI_DATASET_STD
    detector.preprocess = univfd_preprocess(detector.clip_model.visual)
    output_dim = getattr(detector.clip_model.visual, "output_dim", 768)
    detector.head = torch.nn.Linear(int(output_dim), 1).to(detector.device)
    detector._load_head(snapshot / entry["file"])
    detector.head.eval()
    return detector

def _load_timm(entry: dict, snapshot: Path, device: str):
    import timm
    from safetensors.torch import load_file
    from torchvision import transforms

    if entry["loader"] == "timm_real_logit":
        model = timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=False,
            num_classes=1,
            img_size=256,
        )
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]
            ),
        ])
    else:
        model = timm.create_model(
            "vit_small_patch16_384", pretrained=False, num_classes=1
        )
        preprocess = transforms.Compose([
            transforms.Resize(440),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
    model.load_state_dict(load_file(snapshot / entry["file"]), strict=True)
    return model.to(device).eval(), preprocess


def _load_torchvision(entry: dict, snapshot: Path, device: str):
    from torch import nn
    from torchvision import models, transforms

    builders = {
        "ResNet-50": (models.resnet50, "fc", 0),
        "EfficientNet-B0": (models.efficientnet_b0, "classifier", 1),
        "ConvNeXt": (models.convnext_tiny, "classifier", 2),
    }
    builder, head, index = builders[entry["architecture"]]
    model = builder(weights=None)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.classifier[index] = nn.Linear(model.classifier[index].in_features, 1)
    model.load_state_dict(
        torch.load(snapshot / entry["file"], map_location="cpu", weights_only=True),
        strict=True,
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    return model.to(device).eval(), preprocess


def load_model(name: str, device: str, cache: Path) -> ModelAdapter:
    entries = load_registry(ROOT / "models.toml", "models")
    entry = entries[name]
    errors = check_models({name: entry}, [name], "benchmark")
    if errors:
        raise RuntimeError("\n".join(errors))
    os.environ["HF_HUB_OFFLINE"] = "1"
    snapshot = fetch_model(name, cache)
    kind = entry["loader"]
    if kind in {"hf_multiclass", "hf_ai_logit"}:
        model, processor = _hf_components(entry, device, cache)
        return ModelAdapter(name, entry, model, device, processor=processor)
    if kind in {"timm_ai_logit", "timm_real_logit"}:
        model, preprocess = _load_timm(entry, snapshot, device)
        return ModelAdapter(name, entry, model, device, preprocess=preprocess)
    if kind == "torchvision_real_logit":
        model, preprocess = _load_torchvision(entry, snapshot, device)
        return ModelAdapter(name, entry, model, device, preprocess=preprocess)
    if kind == "aidetector_hf":
        return ModelAdapter(
            name, entry, _load_aidetector_hf(entry, device, cache), device
        )
    if kind == "aidetector_univfd":
        return ModelAdapter(name, entry, _load_univfd(entry, snapshot, device), device)
    raise RuntimeError(f"Unsupported loader: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", type=Path, default=Path("work/hf-cache"))
    args = parser.parse_args()
    adapter = load_model(args.model, args.device, args.cache)
    with Image.open(args.image) as image:
        raw_score, probability_ai = adapter.score_one(image)
    if not math.isfinite(probability_ai):
        raise RuntimeError("non-finite AI probability")
    print(
        f"{adapter.name} probability_ai={probability_ai:.8f} "
        f"raw_score={raw_score:.8f} parameters={adapter.parameter_count}"
    )


if __name__ == "__main__":
    main()
