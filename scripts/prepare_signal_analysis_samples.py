#!/usr/bin/env python3
"""Download deterministic, license-recorded subsets for signal analysis.

The two large ZIP releases are sampled with HTTP byte ranges; the complete
4.3 GB and 14.5 GB archives are never downloaded. Community Forensics is read
one row at a time through the official Hugging Face dataset-server API.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import pandas as pd
from PIL import Image


SEED = "techjam-aigc-signal-analysis-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SOURCES = {
    "DDA-COCO": {
        "repository": "mengyao121/DDA-COCO",
        "revision": "b38361e905f2533c2d847f43d0178c83a79a2602",
        "archive": "DDA-COCO.zip",
        "license_name": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_url": "https://huggingface.co/datasets/mengyao121/DDA-COCO",
    },
    "EvalGEN": {
        "repository": "Junwei-Xi/EvalGEN",
        "revision": "86c719a9bfc72b7e5d91bfbb04432774feb4e3b7",
        "archive": "EvalGEN.zip",
        "license_name": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_url": "https://huggingface.co/datasets/Junwei-Xi/EvalGEN",
    },
    "Community Forensics Eval": {
        "repository": "OwensLab/CommunityForensics-Eval",
        "revision": "7d4a74a88d2cac93b513c0853bf92c260eaceea0",
        "license_name": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "dataset_url": "https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval",
    },
}


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP file backed by bounded byte-range requests."""

    def __init__(self, url: str, block_size: int = 1 << 19, cache_blocks: int = 24):
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(
                response.headers.get("Content-Length")
                or response.headers["X-Linked-Size"]
            )
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.downloaded_bytes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _read_block(self, index: int) -> bytes:
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise OSError(
                    f"Range request was ignored for {self.url}: HTTP {response.status}"
                )
            value = response.read()
        self.downloaded_bytes += len(value)
        self.cache[index] = value
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return value

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        remaining = min(size, self.size - self.position)
        if remaining <= 0:
            return b""
        pieces: list[bytes] = []
        while remaining:
            block_index, within = divmod(self.position, self.block_size)
            block = self._read_block(block_index)
            take = min(remaining, len(block) - within)
            pieces.append(block[within : within + take])
            self.position += take
            remaining -= take
        return b"".join(pieces)


def stable_key(*parts: str) -> str:
    return hashlib.sha256("|".join((SEED, *parts)).encode()).hexdigest()


def archive_url(source: dict[str, str]) -> str:
    repository = urllib.parse.quote(source["repository"], safe="/")
    archive = urllib.parse.quote(source["archive"])
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{source['revision']}/{archive}"
    )


def image_metadata(payload: bytes) -> dict[str, object]:
    with Image.open(io.BytesIO(payload)) as image:
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or "unknown",
        }


def save_record(
    payload: bytes,
    *,
    destination: Path,
    dataset: str,
    label: str,
    generation_model: str,
    generator_family: str,
    source_dataset: str,
    source_member: str,
    repository: str,
    revision: str,
    license_name: str,
    license_url: str,
    paired_id: str,
    stratum: str,
    source_uri: str | None = None,
) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing != digest:
            raise RuntimeError(f"Refusing to overwrite changed local file: {destination}")
    else:
        destination.write_bytes(payload)
    metadata = image_metadata(payload)
    return {
        "dataset": dataset,
        "split": "external_confirmation",
        "label": label,
        "source": source_uri or f"hf://datasets/{repository}@{revision}/{source_member}",
        "generator_family": generator_family,
        "generation_model": generation_model,
        "source_dataset": source_dataset,
        "split_method": f"deterministic subset; seed={SEED}",
        "stratum": stratum,
        "source_member": source_member,
        "local_path": destination.as_posix(),
        **metadata,
        "bytes": len(payload),
        "sha256": digest,
        "repository": repository,
        "revision": revision,
        "license_name": license_name,
        "license_url": license_url,
        "paired_id": paired_id,
    }


def archive_groups(dataset: str) -> tuple[HTTPRangeReader, zipfile.ZipFile, dict[str, list[str]]]:
    source = SOURCES[dataset]
    reader = HTTPRangeReader(archive_url(source))
    archive = zipfile.ZipFile(reader)
    groups: dict[str, list[str]] = defaultdict(list)
    for info in archive.infolist():
        member = info.filename
        if info.is_dir() or Path(member).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = member.split("/")
        if len(parts) < 3:
            continue
        groups[parts[1]].append(member)
    return reader, archive, dict(groups)


def inspect_archives() -> None:
    for dataset in ("DDA-COCO", "EvalGEN"):
        reader, archive, groups = archive_groups(dataset)
        try:
            print(f"{dataset}: {reader.size:,} archive bytes")
            for group, members in sorted(groups.items()):
                print(f"  {group}: {len(members):,} images")
            print(f"  byte-range metadata read: {reader.downloaded_bytes:,} bytes")
        finally:
            archive.close()


def dda_label(group: str) -> str:
    normalized = group.lower().replace("_", "-")
    if normalized in {"real", "coco", "ms-coco", "mscoco", "val2017"}:
        return "authentic"
    return "AIGC"


def evalgen_family(group: str) -> str:
    normalized = group.lower()
    if normalized in {"infinity", "nova"}:
        return "autoregressive"
    if normalized == "flux":
        return "diffusion transformer"
    return "multimodal diffusion"


def sample_archive(dataset: str, output_root: Path, per_group: int) -> list[dict[str, object]]:
    source = SOURCES[dataset]
    reader, archive, groups = archive_groups(dataset)
    records: list[dict[str, object]] = []
    try:
        for group, members in sorted(groups.items()):
            chosen = sorted(members, key=lambda name: stable_key(dataset, group, name))[
                :per_group
            ]
            if len(chosen) < per_group:
                raise RuntimeError(f"{dataset}/{group} has only {len(chosen)} images")
            for member in chosen:
                suffix = Path(member).suffix.lower() or ".bin"
                filename = f"{stable_key(dataset, member)[:16]}{suffix}"
                destination = output_root / dataset.lower().replace(" ", "_") / group / filename
                payload = destination.read_bytes() if destination.exists() else archive.read(member)
                if dataset == "DDA-COCO":
                    label = dda_label(group)
                    family = "camera/web photo" if label == "authentic" else "VAE reconstruction"
                    paired_id = Path(member).stem
                    source_dataset = "MSCOCO val2017"
                else:
                    label = "AIGC"
                    family = evalgen_family(group)
                    paired_id = Path(member).stem.rsplit("_", 1)[0]
                    source_dataset = "GenEval prompts"
                records.append(
                    save_record(
                        payload,
                        destination=destination,
                        dataset=dataset,
                        label=label,
                        generation_model=group,
                        generator_family=family,
                        source_dataset=source_dataset,
                        source_member=member,
                        repository=source["repository"],
                        revision=source["revision"],
                        license_name=source["license_name"],
                        license_url=source["license_url"],
                        paired_id=paired_id,
                        stratum=group,
                    )
                )
                if dataset == "DDA-COCO":
                    coco_name = Path(member).name
                    # The official S3 endpoint currently presents a mismatched
                    # certificate over HTTPS; its documented HTTP endpoint works.
                    coco_url = f"http://images.cocodataset.org/val2017/{coco_name}"
                    real_destination = output_root / "dda-coco" / "real" / group / coco_name
                    if real_destination.exists():
                        real_payload = real_destination.read_bytes()
                    else:
                        with urllib.request.urlopen(coco_url, timeout=120) as response:
                            real_payload = response.read()
                    records.append(
                        save_record(
                            real_payload,
                            destination=real_destination,
                            dataset=dataset,
                            label="authentic",
                            generation_model="authentic",
                            generator_family="camera/web photo",
                            source_dataset="MSCOCO val2017",
                            source_member=f"val2017/{coco_name}",
                            repository="cocodataset/val2017",
                            revision="official-val2017",
                            license_name="COCO source-image terms (per image)",
                            license_url="https://cocodataset.org/#termsofuse",
                            paired_id=Path(member).stem,
                            stratum=f"real paired to {group}",
                            source_uri=coco_url,
                        )
                    )
        print(
            f"{dataset}: saved {len(records)} images; byte-range traffic "
            f"{reader.downloaded_bytes / (1 << 20):.1f} MiB"
        )
    finally:
        archive.close()
    return records


def fetch_community_row(offset: int, retries: int = 2) -> dict[str, object]:
    query = urllib.parse.urlencode(
        {
            "dataset": SOURCES["Community Forensics Eval"]["repository"],
            "config": "default",
            "split": "CompEval",
            "offset": offset,
            "length": 1,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                result = json.load(response)
            return result["rows"][0]["row"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError):
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def sample_community(
    output_root: Path,
    per_label: int,
    max_per_model: int,
) -> list[dict[str, object]]:
    source = SOURCES["Community Forensics Eval"]
    cached_paths = sorted(
        (output_root / "community_forensics_eval").glob("**/row_*")
    )
    cached_by_label = {
        label: [
            int(path.name.split("_")[1])
            for path in cached_paths
            if path.parent.name == label.lower()
        ]
        for label in ("authentic", "AIGC")
    }
    use_cached_offsets = all(
        len(cached_by_label[label]) >= per_label for label in cached_by_label
    )
    if use_cached_offsets:
        records: list[dict[str, object]] = []
        for label in ("authentic", "AIGC"):
            label_paths = [
                path for path in cached_paths if path.parent.name == label.lower()
            ][:per_label]
            for path in label_paths:
                offset = int(path.name.split("_")[1])
                payload = path.read_bytes()
                records.append(
                    save_record(
                        payload,
                        destination=path,
                        dataset="Community Forensics Eval",
                        label=label,
                        generation_model=(
                            "authentic"
                            if label == "authentic"
                            else "mixed; selected with <=2 images/model"
                        ),
                        generator_family=(
                            "real" if label == "authentic" else "mixed published architectures"
                        ),
                        source_dataset="CompEval paired authentic sources",
                        source_member=f"dataset-server/CompEval/{offset}",
                        repository=source["repository"],
                        revision=source["revision"],
                        license_name=source["license_name"],
                        license_url=source["license_url"],
                        paired_id=f"community-{offset}",
                        stratum=f"{label}:cached diverse subset",
                    )
                )
        print(
            "Community Forensics Eval: reused "
            f"{sum(record['label'] == 'authentic' for record in records)} real and "
            f"{sum(record['label'] == 'AIGC' for record in records)} AIGC cached rows; "
            "per-row model metadata deferred after HTTP 429"
        )
        return records
    else:
        randomizer = random.Random(int(stable_key("Community Forensics Eval"), 16))
        offsets = list(range(51_836))
        randomizer.shuffle(offsets)
    label_counts: Counter[str] = Counter()
    model_counts: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, object]] = []
    failures = 0
    for offset in offsets:
        if label_counts["authentic"] >= per_label and label_counts["AIGC"] >= per_label:
            break
        try:
            row = fetch_community_row(offset)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError):
            failures += 1
            if failures > 100:
                raise RuntimeError("Too many Community Forensics dataset-server failures")
            continue
        label = "AIGC" if int(row["label"]) == 1 else "authentic"
        model = str(row.get("model_name") or "unknown")
        model_key = (label, model)
        if label_counts[label] >= per_label or model_counts[model_key] >= max_per_model:
            continue
        if str(row.get("nsfw_flag", "False")).lower() == "true":
            continue
        encoded = row["image_data"]
        payload = base64.b64decode(encoded)
        suffix = "." + str(row.get("format") or "png").lower().replace("jpeg", "jpg")
        member = f"dataset-server/CompEval/{offset}/{row['image_name']}"
        filename = f"row_{offset:05d}_{stable_key(member)[:12]}{suffix}"
        records.append(
            save_record(
                payload,
                destination=output_root / "community_forensics_eval" / label.lower() / filename,
                dataset="Community Forensics Eval",
                label=label,
                generation_model=model if label == "AIGC" else "authentic",
                generator_family=str(row.get("architecture") or "unknown"),
                source_dataset=str(row.get("real_source") or "unknown"),
                source_member=member,
                repository=source["repository"],
                revision=source["revision"],
                license_name=source["license_name"],
                license_url=source["license_url"],
                paired_id=f"community-{offset}",
                stratum=f"{label}:{model}",
            )
        )
        label_counts[label] += 1
        model_counts[model_key] += 1
        print(
            f"Community Forensics Eval: {label_counts['authentic']}/{per_label} real, "
            f"{label_counts['AIGC']}/{per_label} AIGC",
            end="\r",
            flush=True,
        )
    print()
    if min(label_counts.values(), default=0) < per_label or len(label_counts) < 2:
        raise RuntimeError(f"Could not fill Community Forensics quotas: {label_counts}")
    return records


def write_outputs(records: list[dict[str, object]], repo_root: Path) -> None:
    manifest_path = repo_root / "data/metadata/signal_analysis_index.csv"
    metadata_path = repo_root / "data/metadata/signal_analysis_sources.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "label", "generation_model", "source_member"], kind="stable"
    )
    frame.to_csv(manifest_path, index=False)
    metadata = {
        "schema_version": 1,
        "selection_seed": SEED,
        "images": len(frame),
        "counts": frame.groupby(["dataset", "label"]).size().to_dict(),
        "sources": SOURCES,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "notes": [
            "DDA-COCO and EvalGEN were read with byte ranges; full ZIPs were not downloaded.",
            "Community Forensics rows came from the official comprehensive evaluation release.",
            "Community Forensics is licensed for non-commercial research/education; downstream use must preserve attribution and share-alike terms.",
            "The organizer demonstration-only COCO val2017/DALL-E Advanced split is not used.",
        ],
    }
    metadata["counts"] = {
        "|".join(key): int(value) for key, value in metadata["counts"].items()
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(frame.groupby(["dataset", "label", "generation_model"]).size())
    print(f"Wrote {manifest_path} and {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--per-archive-group", type=int, default=20)
    parser.add_argument("--community-per-label", type=int, default=25)
    parser.add_argument("--community-max-per-model", type=int, default=2)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        inspect_archives()
        return
    output_root = args.repo_root / "data/samples/signal_analysis"
    records = []
    records.extend(sample_archive("DDA-COCO", output_root, args.per_archive_group))
    records.extend(sample_archive("EvalGEN", output_root, args.per_archive_group))
    records.extend(
        sample_community(
            output_root,
            per_label=args.community_per_label,
            max_per_model=args.community_max_per_model,
        )
    )
    write_outputs(records, args.repo_root)


if __name__ == "__main__":
    main()
