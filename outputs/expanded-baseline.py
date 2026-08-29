from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import spike
from aidetector.evaluation import compute_metrics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "expanded-results"
BASE_RESULTS = ROOT / "results"
os.environ.setdefault("HF_HOME", str(ROOT.parent / "hf-cache"))

CANDIDATES = {
    "community_forensics": {
        "model_id": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        "kind": "single_ai_logit",
        "threshold": 0.5,
        "license": "MIT",
    },
    "frontier_community_forensics": {
        "model_id": "Thermostatic/community-forensics-frontier-detector-2026-08",
        "kind": "timm_single_ai_logit",
        "threshold": 1 / (1 + math.exp(-1.359375)),
        "license": "MIT; organizer suitability requires confirmation",
    },
    "ateeqq_siglip": {
        "model_id": "Ateeqq/ai-vs-human-image-detector",
        "kind": "multiclass",
        "threshold": 0.5,
        "license": "Apache-2.0",
    },
    "wkaandemir_clip": {
        "model_id": "wkaandemir/ai-image-detector",
        "kind": "timm_real_logit",
        "threshold": 0.08,
        "license": "MIT",
        "temperature": 0.595,
    },
    "divine_resnet50": {
        "model_id": "divine2k/ai-image-detectors",
        "filename": "resnet50_ai_real_final.pth",
        "architecture": "resnet50",
        "kind": "torchvision_real_logit",
        "threshold": 0.475,
        "license": "MIT",
        "binary_assumption": "published uncertain band collapsed at its midpoint",
    },
    "divine_efficientnet": {
        "model_id": "divine2k/ai-image-detectors",
        "filename": "efficientNet_BO_Final.pth",
        "architecture": "efficientnet_b0",
        "kind": "torchvision_real_logit",
        "threshold": 0.475,
        "license": "MIT",
        "binary_assumption": "published uncertain band collapsed at its midpoint",
    },
    "divine_convnext": {
        "model_id": "divine2k/ai-image-detectors",
        "filename": "convNext_final.pth",
        "architecture": "convnext_tiny",
        "kind": "torchvision_real_logit",
        "threshold": 0.475,
        "license": "MIT",
        "binary_assumption": "published uncertain band collapsed at its midpoint",
    },
}


def load_base(name: str) -> dict:
    return json.loads((BASE_RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def metric_payload(labels: list[int], scores: list[float], threshold: float, name: str) -> dict:
    return compute_metrics(labels, scores, threshold=threshold, dataset=name).as_dict()


def analyze_existing() -> dict:
    names = ("steganograph", "capcheck", "univfd")
    runs = {name: load_base(name) for name in names}
    by_model_condition = {
        name: {
            condition: {row["sample_id"]: row for row in run["predictions"] if row["condition"] == condition}
            for condition in spike.CONDITIONS
        }
        for name, run in runs.items()
    }
    sample_ids = [row["sample_id"] for row in runs["steganograph"]["predictions"] if row["condition"] == "clean"]
    labels = [
        1 if by_model_condition["steganograph"]["clean"][sample_id]["truth"] == "ai" else 0
        for sample_id in sample_ids
    ]

    ensembles = {}
    combinations = [
        ("steganograph", "capcheck"),
        ("steganograph", "univfd"),
        ("capcheck", "univfd"),
        names,
    ]
    for combination in combinations:
        key = "mean_" + "_".join(combination)
        ensembles[key] = {}
        for condition in spike.CONDITIONS:
            scores = [
                statistics.fmean(
                    by_model_condition[name][condition][sample_id]["probability_ai"]
                    for name in combination
                )
                for sample_id in sample_ids
            ]
            ensembles[key][condition] = metric_payload(labels, scores, 0.5, f"{key}:{condition}")

    vote_key = "majority_vote_all3"
    ensembles[vote_key] = {}
    for condition in spike.CONDITIONS:
        scores = [
            statistics.fmean(
                by_model_condition[name][condition][sample_id]["probability_ai"] >= 0.5
                for name in names
            )
            for sample_id in sample_ids
        ]
        ensembles[vote_key][condition] = metric_payload(labels, scores, 0.5, f"{vote_key}:{condition}")

    transform_aggregation = {}
    reducers = {
        "mean": statistics.fmean,
        "median": statistics.median,
        "min": min,
        "max": max,
    }
    for name in names:
        transform_aggregation[name] = {}
        for reducer_name, reducer in reducers.items():
            scores = [
                reducer(
                    by_model_condition[name][condition][sample_id]["probability_ai"]
                    for condition in spike.CONDITIONS
                )
                for sample_id in sample_ids
            ]
            transform_aggregation[name][reducer_name] = metric_payload(
                labels, scores, 0.5, f"transform_aggregation:{name}:{reducer_name}"
            )

    result = {"ensembles": ensembles, "transform_aggregation": transform_aggregation}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "existing_ensembles.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze_all() -> dict:
    groups = {
        "top2_or": ("ateeqq_siglip", "frontier_community_forensics"),
        "top2_and": ("ateeqq_siglip", "frontier_community_forensics"),
        "top3_vote_with_steganograph": (
            "ateeqq_siglip",
            "frontier_community_forensics",
            "steganograph",
        ),
        "top3_vote_with_community": (
            "ateeqq_siglip",
            "frontier_community_forensics",
            "community_forensics",
        ),
        "top5_vote": (
            "ateeqq_siglip",
            "frontier_community_forensics",
            "steganograph",
            "community_forensics",
            "divine_efficientnet",
        ),
        "divine3_vote": ("divine_resnet50", "divine_efficientnet", "divine_convnext"),
    }
    needed = set(itertools.chain.from_iterable(groups.values()))
    runs = {
        name: json.loads(
            ((BASE_RESULTS if name in {"steganograph", "capcheck", "univfd"} else RESULTS) / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        for name in needed
    }
    rows = {
        name: {(row["condition"], row["sample_id"]): row for row in run["predictions"]}
        for name, run in runs.items()
    }
    reference = runs["ateeqq_siglip"]["predictions"]
    results = {}
    for group_name, members in groups.items():
        results[group_name] = {}
        for condition in spike.CONDITIONS:
            condition_rows = [row for row in reference if row["condition"] == condition]
            labels, scores = [], []
            for row in condition_rows:
                votes = [
                    1.0 if rows[member][(condition, row["sample_id"])]["pred"] == "ai" else 0.0
                    for member in members
                ]
                if group_name == "top2_or":
                    score = max(votes)
                elif group_name == "top2_and":
                    score = min(votes)
                else:
                    score = statistics.mean(votes)
                labels.append(1 if row["truth"] == "ai" else 0)
                scores.append(score)
            results[group_name][condition] = metric_payload(
                labels, scores, 0.5, f"decision_ensemble:{group_name}:{condition}"
            )
    payload = {"method": "published-threshold decision votes; no fitted calibration", "groups": groups, "results": results}
    (RESULTS / "all_ensembles.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class CandidateDetector:
    def __init__(self, name: str):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.name = name
        self.config = CANDIDATES[name]
        self.kind = self.config["kind"]
        self.threshold = float(self.config["threshold"])
        self.processor = None
        if self.kind == "torchvision_real_logit":
            from huggingface_hub import hf_hub_download
            from torch import nn
            from torchvision import models

            builders = {
                "resnet50": (models.resnet50, "fc"),
                "efficientnet_b0": (models.efficientnet_b0, "classifier"),
                "convnext_tiny": (models.convnext_tiny, "classifier"),
            }
            builder, head = builders[self.config["architecture"]]
            self.model = builder(weights=None)
            if head == "fc":
                self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            else:
                index = 1 if self.config["architecture"] == "efficientnet_b0" else 2
                self.model.classifier[index] = nn.Linear(
                    self.model.classifier[index].in_features, 1
                )
            path = hf_hub_download(self.config["model_id"], self.config["filename"])
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif self.kind in {"timm_real_logit", "timm_single_ai_logit"}:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            import timm

            path = hf_hub_download(self.config["model_id"], "model.safetensors")
            if self.kind == "timm_real_logit":
                self.model = timm.create_model(
                    "vit_base_patch16_clip_224.openai",
                    pretrained=False,
                    num_classes=1,
                    img_size=256,
                )
                self.model.load_state_dict(load_file(path), strict=True)
                self.preprocess = transforms.Compose(
                    [
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]
                        ),
                    ]
                )
            else:
                self.model = timm.create_model(
                    "vit_small_patch16_384", pretrained=False, num_classes=1
                )
                self.model.load_state_dict(load_file(path), strict=True)
                self.preprocess = transforms.Compose(
                    [
                        transforms.Resize(440),
                        transforms.CenterCrop(384),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.config["model_id"])
            self.model = AutoModelForImageClassification.from_pretrained(self.config["model_id"])
        self.model.eval()
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        if self.parameters >= 2_000_000_000:
            raise RuntimeError(f"Model exceeds 2B parameter limit: {self.parameters}")

    def predict(self, image: Image.Image) -> tuple[float, float]:
        if self.kind in {"torchvision_real_logit", "timm_real_logit", "timm_single_ai_logit"}:
            values = self.preprocess(image).unsqueeze(0)
            with torch.inference_mode():
                raw = float(self.model(values).flatten()[0].item())
            if self.kind == "timm_single_ai_logit":
                return torch.sigmoid(torch.tensor(raw)).item(), raw
            p_real = torch.sigmoid(
                torch.tensor(raw / self.config.get("temperature", 1.0))
            ).item()
            return 1.0 - p_real, raw

        values = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**values).logits[0].detach().cpu()
        if self.kind == "single_ai_logit":
            raw = float(logits.flatten()[0].item())
            return torch.sigmoid(torch.tensor(raw)).item(), raw

        probabilities = torch.softmax(logits, dim=-1)
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
        probability = float(probabilities[fake_indices].sum().item())
        return probability, float(logits[fake_indices].mean().item())

    def info(self) -> dict:
        model_config = getattr(self.model, "config", None)
        return {
            **self.config,
            "parameters": self.parameters,
            "id2label": getattr(model_config, "id2label", None),
            "resolved_commit": getattr(model_config, "_commit_hash", None),
        }


def run_candidate(name: str) -> dict:
    samples = spike.prepare_sample()
    detector = CandidateDetector(name)
    metrics = {}
    predictions = []
    started = time.perf_counter()
    for condition in spike.CONDITIONS:
        labels, scores = [], []
        condition_started = time.perf_counter()
        for sample in samples:
            image = spike.transform(spike.load_image(sample), condition, sample["sample_id"])
            score, raw = detector.predict(image)
            labels.append(sample["label"])
            scores.append(score)
            predictions.append(
                {
                    "model": name,
                    "condition": condition,
                    "sample_id": sample["sample_id"],
                    "truth": sample["truth"],
                    "probability_ai": score,
                    "raw_score": raw,
                    "pred": "ai" if score >= detector.threshold else "real",
                }
            )
        metric = compute_metrics(
            labels,
            scores,
            threshold=detector.threshold,
            dataset=f"SID_Set validation gate:{condition}",
            seconds=time.perf_counter() - condition_started,
        )
        metrics[condition] = metric.as_dict()
        print(
            f"{name:30} {condition:18} BA={metric.balanced_accuracy:.3f} "
            f"AUC={metric.roc_auc:.3f} FP={metric.false_positive} FN={metric.false_negative}",
            flush=True,
        )
    result = {
        "model": name,
        "model_info": detector.info(),
        "sample_count": len(samples),
        "fine_tuning": False,
        "conditions": metrics,
        "predictions": predictions,
        "seconds_total": time.perf_counter() - started,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def self_test() -> None:
    result = analyze_existing()
    assert len(result["ensembles"]) == 5
    assert all(len(value) == len(spike.CONDITIONS) for value in result["ensembles"].values())
    assert set(CANDIDATES) == {
        "community_forensics",
        "frontier_community_forensics",
        "ateeqq_siglip",
        "wkaandemir_clip",
        "divine_resnet50",
        "divine_efficientnet",
        "divine_convnext",
    }
    assert all(float(config["threshold"]) < 1 for config in CANDIDATES.values())
    assert CANDIDATES["divine_resnet50"]["threshold"] == 1 - ((0.45 + 0.60) / 2)
    if all((RESULTS / f"{name}.json").exists() for name in CANDIDATES):
        assert len(analyze_all()["results"]) == 6
    print("expanded self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("self-test", "ensembles", "all-ensembles", "model"))
    parser.add_argument("--model", choices=tuple(CANDIDATES))
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
    elif args.action == "ensembles":
        analyze_existing()
    elif args.action == "all-ensembles":
        analyze_all()
    else:
        if not args.model:
            parser.error("--model is required")
        run_candidate(args.model)


if __name__ == "__main__":
    main()
