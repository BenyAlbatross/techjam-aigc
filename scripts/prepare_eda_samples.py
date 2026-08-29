#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow>=11.3.0",
#   "remotezip>=0.12.3",
#   "requests>=2.32.5",
#   "stream-unzip>=0.0.99",
# ]
# ///
"""Create exact-pixel visual slices of CIFAKE, SID Set, and WildFake.

The deterministic samples are for visual EDA only. No transforms or
frequency-domain processing are performed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests
from PIL import Image
from remotezip import RemoteZip
from stream_unzip import stream_unzip


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
SEED = 20260829
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SID_ROWS = "https://datasets-server.huggingface.co/rows"
MS_FILE = "https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"
WILDFAKE_SPLIT_METHOD = (
    "deterministic reconstruction of paper's per-stratum 80/20 policy"
)


def choose(values: Iterable[Any], count: int, key: str) -> list[Any]:
    ordered = sorted(values, key=str)
    return random.Random(f"{SEED}:{key}").sample(ordered, min(count, len(ordered)))


def metadata(payload: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or "unknown",
            "bytes": len(payload),
        }


def store(payload: bytes, destination: Path) -> dict[str, Any]:
    info = metadata(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return info


def record(
    *, dataset: str, split: str, label: str, source: str,
    generator_family: str, generation_model: str, source_dataset: str,
    split_method: str, stratum: str, member: str, destination: Path,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "label": label,
        "source": source,
        "generator_family": generator_family,
        "generation_model": generation_model,
        "source_dataset": source_dataset,
        "split_method": split_method,
        "stratum": stratum,
        "source_member": member,
        "local_path": destination.relative_to(ROOT).as_posix(),
        **store(payload, destination),
    }


def cifake_sample(per_group: int = 36) -> list[dict[str, Any]]:
    archive = DATA / "raw/cifake/cifake.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing complete CIFAKE archive: {archive}")
    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zipped:
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in zipped.namelist():
            path = Path(name)
            parts = {part.lower() for part in path.parts}
            if path.suffix.lower() not in SUFFIXES:
                continue
            split = "train" if "train" in parts else "test" if "test" in parts else "unknown"
            label = "real" if "real" in parts else "fake" if "fake" in parts else "unknown"
            groups[(split, label)].append(name)
        for (split, label), names in sorted(groups.items()):
            for name in choose(names, per_group, f"cifake:{split}:{label}"):
                destination = SAMPLES / "cifake" / split / label / Path(name).name
                output.append(record(
                    dataset="CIFAKE", split=split, label=label,
                    source="CIFAR-10" if label == "real" else "Stable Diffusion 1.4",
                    generator_family="authentic" if label == "real" else "diffusion",
                    generation_model="none" if label == "real" else "Stable Diffusion 1.4",
                    source_dataset="CIFAR-10", split_method="official CIFAKE split",
                    stratum=f"{split}/{label}", member=name, destination=destination,
                    payload=zipped.read(name),
                ))
    return output


def sid_rows(session: requests.Session, split: str, offset: int) -> list[dict[str, Any]]:
    params = {
        "dataset": "saberzl/SID_Set", "config": "default", "split": split,
        "offset": offset, "length": 100,
    }
    for attempt in range(6):
        response = session.get(SID_ROWS, params=params, timeout=120)
        if response.status_code < 500:
            response.raise_for_status()
            return response.json()["rows"]
        if attempt < 5:
            time.sleep(2**attempt)
    response.raise_for_status()
    raise AssertionError("unreachable")


def sid_sample(per_label: int = 30) -> list[dict[str, Any]]:
    labels = {0: "real", 1: "full_synthetic", 2: "tampered"}
    output: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "techjam-aigc-visual-eda/1.0"
    for split, count in {"train": 210_000, "validation": 30_000}.items():
        quotas = {key: per_label for key in labels}
        offsets = list(range(0, count, 100))
        random.Random(f"{SEED}:sid:{split}").shuffle(offsets)
        for offset in offsets:
            for wrapped in sid_rows(session, split, offset):
                row = wrapped["row"]
                label_id = int(row["label"])
                if quotas.get(label_id, 0) <= 0 or not row.get("image"):
                    continue
                response = session.get(row["image"]["src"], timeout=120)
                response.raise_for_status()
                label = labels[label_id]
                image_id = str(row["img_id"])
                destination = SAMPLES / "sid_set" / split / label / f"{image_id}.jpg"
                output.append(record(
                    dataset="SID Set", split=split, label=label,
                    source="SID Set published row",
                    generator_family="authentic" if label == "real" else "multiple / not row-annotated",
                    generation_model="none" if label == "real" else "multiple SID Set generators",
                    source_dataset="OpenImages V7 / SID Set",
                    split_method="official Hugging Face split", stratum=f"{split}/{label}",
                    member=f"{split}:{wrapped['row_idx']}:{image_id}",
                    destination=destination, payload=response.content,
                ))
                quotas[label_id] -= 1
                if not any(quotas.values()):
                    break
            if not any(quotas.values()):
                break
        if any(quotas.values()):
            raise RuntimeError(f"Could not fill SID Set quotas for {split}: {quotas}")
    return output


def modelscope_cdn_url(path: str) -> str:
    query = urlencode({"Revision": "master", "FilePath": path})
    response = requests.head(f"{MS_FILE}?{query}", allow_redirects=False, timeout=120)
    response.raise_for_status()
    if "Location" not in response.headers:
        raise RuntimeError(f"ModelScope returned no download redirect for {path}")
    return response.headers["Location"]


def reconstructed_wildfake_split(member: str) -> str:
    """Reproduce the paper's 80/20 policy, not its unpublished membership."""
    bucket = int.from_bytes(
        hashlib.sha256(f"{SEED}:{member}".encode()).digest()[:8], "big"
    ) % 5
    return "test (reconstructed 20%)" if bucket == 0 else "train (reconstructed 80%)"


def other_source(member: str) -> str:
    lowered = member.lower()
    if "ffhq" in lowered:
        return "FFHQ"
    if "coco" in lowered:
        return "COCO"
    if "ade20k" in lowered:
        return "ADE20K"
    return "not encoded in archive path"


def wildfake_random_access_sample(per_split: int = 4) -> list[dict[str, Any]]:
    archives = [
        ("real", "authentic", "none", "CelebA-HQ", "Images/Real/celebahq.zip"),
        ("real", "authentic", "none", "AFHQ", "Images/Real/afhq.zip"),
        ("real", "authentic", "none", "FFHQ", "Images/Real/ffhq.zip"),
        ("fake", "diffusion", "DDIM", "not encoded in archive path", "Images/Diffusion_based/DDIM.zip"),
        ("fake", "diffusion", "DDPM", "not encoded in archive path", "Images/Diffusion_based/DDPM.zip"),
        ("fake", "diffusion", "ADM", "not encoded in archive path", "Images/Diffusion_based/ADM.zip"),
    ]
    output: list[dict[str, Any]] = []
    for label, family, model, source_dataset, archive in archives:
        with RemoteZip(modelscope_cdn_url(archive)) as remote:
            names = [
                item.filename for item in remote.infolist()
                if not item.is_dir() and Path(item.filename).suffix.lower() in SUFFIXES
            ]
            groups: dict[str, list[str]] = defaultdict(list)
            for name in names:
                groups[reconstructed_wildfake_split(f"{archive}:{name}")].append(name)
            for split, members in sorted(groups.items()):
                for name in choose(members, per_split, f"wildfake:{archive}:{split}"):
                    token = hashlib.sha256(name.encode()).hexdigest()[:10]
                    source = source_dataset if label == "real" else model
                    destination = SAMPLES / "wildfake" / model.lower() / split.split()[0] / f"{token}-{Path(name).name}"
                    output.append(record(
                        dataset="WildFake", split=split, label=label, source=source,
                        generator_family=family, generation_model=model,
                        source_dataset=source_dataset, split_method=WILDFAKE_SPLIT_METHOD,
                        stratum=f"{label}/{model if label == 'fake' else source_dataset}",
                        member=f"{archive}:{name}", destination=destination,
                        payload=remote.read(name),
                    ))

    other_archive = "Images/Other_based.zip"
    with RemoteZip(modelscope_cdn_url(other_archive)) as remote:
        groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for item in remote.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in SUFFIXES:
                continue
            parts = Path(item.filename).parts
            if len(parts) < 3:
                continue
            model = parts[2]
            source_dataset = other_source(item.filename)
            split = reconstructed_wildfake_split(f"{other_archive}:{item.filename}")
            groups[(model, source_dataset, split)].append(item.filename)
        for (model, source_dataset, split), members in sorted(groups.items()):
            for name in choose(members, per_split, f"wildfake:{model}:{source_dataset}:{split}"):
                token = hashlib.sha256(name.encode()).hexdigest()[:10]
                destination = SAMPLES / "wildfake" / model.lower() / split.split()[0] / f"{token}-{Path(name).name}"
                output.append(record(
                    dataset="WildFake", split=split, label="fake", source=model,
                    generator_family="other", generation_model=model,
                    source_dataset=source_dataset, split_method=WILDFAKE_SPLIT_METHOD,
                    stratum=f"fake/{model}/{source_dataset}",
                    member=f"{other_archive}:{name}", destination=destination,
                    payload=remote.read(name),
                ))
    return output


def wildfake_streamed_gan_sample(per_split: int = 4) -> list[dict[str, Any]]:
    """Read only the start of the 47 GB GAN ZIP, stopping when quotas fill."""
    archive = "Images/GAN_based.zip"
    quotas = {
        "train (reconstructed 80%)": per_split,
        "test (reconstructed 20%)": per_split,
    }
    output: list[dict[str, Any]] = []
    with requests.get(modelscope_cdn_url(archive), stream=True, timeout=120) as response:
        response.raise_for_status()
        for raw_name, _, chunks in stream_unzip(response.iter_content(1024 * 1024)):
            name = raw_name.decode("utf-8", "replace")
            if Path(name).suffix.lower() not in SUFFIXES:
                for _ in chunks:
                    pass
                continue
            split = reconstructed_wildfake_split(f"{archive}:{name}")
            if quotas[split] <= 0:
                for _ in chunks:
                    pass
                if not any(quotas.values()):
                    break
                continue
            payload = b"".join(chunks)
            parts = Path(name).parts
            model = parts[2] if len(parts) > 2 else "GAN (archive prefix)"
            source_dataset = parts[3] if len(parts) > 3 else "not encoded in archive path"
            token = hashlib.sha256(name.encode()).hexdigest()[:10]
            destination = SAMPLES / "wildfake" / model.lower() / split.split()[0] / f"{token}-{Path(name).name}"
            output.append(record(
                dataset="WildFake", split=split, label="fake", source=model,
                generator_family="GAN", generation_model=model,
                source_dataset=f"archive category: {source_dataset}",
                split_method=WILDFAKE_SPLIT_METHOD,
                stratum=f"fake/{model}/{source_dataset}",
                member=f"{archive}:{name}", destination=destination, payload=payload,
            ))
            quotas[split] -= 1
            if not any(quotas.values()):
                break
    if any(quotas.values()):
        raise RuntimeError(f"Could not fill streamed WildFake GAN quotas: {quotas}")
    return output


def wildfake_sample(per_split: int = 4) -> list[dict[str, Any]]:
    return wildfake_random_access_sample(per_split) + wildfake_streamed_gan_sample(per_split)


def write_index(rows: list[dict[str, Any]]) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "split", "label", "source", "generator_family",
        "generation_model", "source_dataset", "split_method", "stratum",
        "source_member", "local_path", "width", "height", "mode", "format", "bytes",
    ]
    with (SAMPLES / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seed": SEED,
        "local_sample_images": len(rows),
        "samples_by_dataset": {
            name: sum(row["dataset"] == name for row in rows)
            for name in sorted({row["dataset"] for row in rows})
        },
        "published_datasets": {
            "CIFAKE": {
                "images": 120_000, "download_bytes": 109_625_224,
                "splits": {"train": 100_000, "test": 20_000},
                "generation_models": ["Stable Diffusion 1.4"],
                "sources": ["CIFAR-10"], "published_resolution": "32×32",
            },
            "SID Set": {
                "available_rows": 240_000, "download_bytes": 140_056_462_172,
                "splits": {"train": 210_000, "validation": 30_000},
                "generation_models": ["multiple; not exposed per viewer row"],
                "sources": ["OpenImages V7 / SID Set"], "published_resolution": "varied",
            },
            "WildFake": {
                "images": 3_570_724, "real_images": 1_013_446,
                "fake_images": 2_557_278, "repository_bytes": 1_287_902_462_734,
                "split_policy": "paper: random 80/20 within each generator and real source; exact membership not published",
                "generation_models": [
                    "ADM", "BigGAN", "DALL-E 2", "DALL-E 3", "DDIM", "DDPM",
                    "DF-GAN", "GALIP", "GigaGAN", "Imagen", "MAE", "MAGE",
                    "Midjourney v4", "Midjourney v5", "Original Stable Diffusion",
                    "Personalized Stable Diffusion", "SD with adapter", "SDXL",
                    "StarGAN", "StyleGAN", "VQDM", "VQGAN", "VQVAE",
                ],
                "sources": [
                    "AFHQ", "CelebA-HQ", "Church/LSUN", "COCO", "FFHQ",
                    "ImageNet", "LAION-5B", "Wukong",
                ],
                "published_resolution": "varied by generator/source",
            },
        },
    }
    metadata_dir = DATA / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    print("Preparing CIFAKE visual sample...", flush=True)
    rows = cifake_sample()
    print("Preparing SID Set visual sample...", flush=True)
    rows.extend(sid_sample())
    print("Preparing stratified WildFake visual sample...", flush=True)
    rows.extend(wildfake_sample())
    write_index(rows)
    print(f"Wrote {len(rows)} image records to data/samples/index.csv")


if __name__ == "__main__":
    main()
