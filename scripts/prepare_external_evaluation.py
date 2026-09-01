#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow>=11.3.0",
#   "remotezip>=0.12.3",
#   "requests>=2.32.5",
# ]
# ///
"""Prepare deterministic WildFake and AIGIBench evaluation samples.

The source repositories are too large to download wholesale.  This script
reads ZIP central directories over HTTP ranges, selects a fixed class- and
source-stratified sample, and range-downloads only selected members.  Source
pixels are stored without re-encoding.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import heapq
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
from urllib.parse import quote, urlencode

from PIL import Image
from remotezip import RemoteZip
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/evaluation/wildfake-aigibench-stratified"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
USER_AGENT = "techjam-aigc-external-evaluation/1.0"

HF_REPO = "HorizonTEL/AIGIBench"
HF_REVISION = "e44ec40efe5117a5ccdaa6ff0e89ed934d03d310"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_LICENSE = "CC-BY-NC-SA-4.0"

MODELSCOPE_REPO = "hy2628982280/WildFake"
MODELSCOPE_API = f"https://modelscope.cn/api/v1/datasets/{MODELSCOPE_REPO}"
MODELSCOPE_FILE_API = f"{MODELSCOPE_API}/repo"
MODELSCOPE_TREE_API = f"{MODELSCOPE_FILE_API}/tree"
MODELSCOPE_LICENSE = "Apache-2.0"
KNOWN_INVALID_MEMBERS = {
    (
        "Images/GAN_based.zip",
        "GAN_based/Advanced/GigaGAN/fake_images/05255.png",
    ),
}


@dataclass(frozen=True)
class Candidate:
    dataset_id: str
    archive: str
    member: str
    target: int
    stratum: str
    generator_family: str
    generation_model: str
    source_dataset: str
    authentic_subtype: str
    priority: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aigibench-count", type=int, default=5_000)
    parser.add_argument("--wildfake-count", type=int, default=10_000)
    parser.add_argument(
        "--wildfake-only",
        action="store_true",
        help="Prepare only the WildFake reconstructed-test sample.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260831)
    parser.add_argument(
        "--wildfake-split-seed",
        type=int,
        default=20260829,
        help="Seed for the published-but-unmaterialized per-stratum 80/20 split.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Resolve the deterministic selection without downloading image members.",
    )
    return parser.parse_args()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _json_get(url: str, *, params: dict[str, Any] | None = None) -> Any:
    with _session() as session:
        response = session.get(url, params=params, timeout=120)
        response.raise_for_status()
        return response.json()


def _stable_priority(seed: int, *values: str) -> int:
    material = ":".join((str(seed), *values))
    return int.from_bytes(sha256(material.encode()).digest()[:16], "big")


def _in_reconstructed_test(seed: int, archive: str, member: str) -> bool:
    material = f"{seed}:{archive}:{member}"
    return int.from_bytes(sha256(material.encode()).digest()[:8], "big") % 5 == 0


def _is_image(member: str) -> bool:
    return (
        not member.endswith("/")
        and "__MACOSX" not in PurePosixPath(member).parts
        and PurePosixPath(member).suffix.casefold() in IMAGE_SUFFIXES
    )


def _structurally_invalid_zip_member(info: Any) -> bool:
    """Reject implausible zero-filled image members before sampling.

    WildFake's GigaGAN directory contains PNG-named members hundreds of
    kilobytes long that compress to only a few hundred bytes and cannot be
    decoded.  Valid source images are still verified after download.
    """

    return bool(
        info.file_size > 10_000
        and info.compress_size > 0
        and info.compress_size / info.file_size < 0.01
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unknown"


def _balanced_quotas(total: int, available: dict[str, int]) -> dict[str, int]:
    """Allocate near-equal quotas and redistribute shortages deterministically."""

    names = sorted(name for name, count in available.items() if count > 0)
    if not names:
        raise ValueError("No non-empty strata were available for sampling.")
    if sum(available[name] for name in names) < total:
        raise ValueError(f"Requested {total} rows from only {sum(available.values())} candidates.")
    quotas = {name: 0 for name in names}
    remaining = total
    while remaining:
        progressed = False
        for name in names:
            if quotas[name] >= available[name]:
                continue
            quotas[name] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("Could not redistribute sample quotas.")
    return quotas


def _keep_smallest(
    heaps: dict[str, list[tuple[int, str, Candidate]]],
    candidate: Candidate,
    *,
    limit: int,
) -> None:
    heap = heaps.setdefault(candidate.stratum, [])
    item = (-candidate.priority, candidate.member, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif candidate.priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _modelscope_tree() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tree = _json_get(
        MODELSCOPE_TREE_API,
        params={"Revision": "master", "Recursive": "true"},
    )
    if tree.get("Code") != 200:
        raise RuntimeError(f"ModelScope tree request failed: {tree}")
    files = tree["Data"]["Files"]
    dataset = _json_get(MODELSCOPE_API)
    if dataset.get("Code") != 200:
        raise RuntimeError(f"ModelScope metadata request failed: {dataset}")
    return files, dataset["Data"]


def _modelscope_url(path: str) -> str:
    with _session() as session:
        response = session.head(
            MODELSCOPE_FILE_API,
            params={"Revision": "master", "FilePath": path},
            allow_redirects=False,
            timeout=120,
        )
        response.raise_for_status()
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"ModelScope returned no download redirect for {path!r}.")
    return location


def _huggingface_url(path: str) -> str:
    encoded = "/".join(quote(part) for part in PurePosixPath(path).parts)
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{encoded}?download=true"


def _aigibench_family(generator: str) -> str:
    if generator in {"ProGAN", "R3GAN", "StyleGAN-XL", "StyleGAN3"}:
        return "gan"
    if generator in {"BlendFace", "E4S", "FaceSwap", "InSwap", "SimSwap", "StyleSwim"}:
        return "face_manipulation"
    if generator in {"IP-Adapter", "Infinite-ID", "InstantID", "PhotoMaker"}:
        return "personalized_diffusion"
    if generator in {"CommunityAI", "SocialRF", "WFIR"}:
        return "open_platform"
    return "diffusion"


def _aigibench_inventory(
    total: int,
    selection_seed: int,
    workers: int,
    reserve_per_stratum: int = 5,
) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, int]]:
    repo = _json_get(HF_API)
    if repo.get("sha") != HF_REVISION:
        raise RuntimeError(
            f"AIGIBench revision changed from {HF_REVISION} to {repo.get('sha')}; "
            "review and pin the new revision before evaluating."
        )
    # The expanded tree exposes LFS sizes; the base repo response does not.
    tree = _json_get(
        f"{HF_API}/tree/{HF_REVISION}",
        params={"recursive": "true", "expand": "true"},
    )
    archives = sorted(
        ({"path": item["path"], "size": int(item["size"])} for item in tree
         if item["path"].startswith("test/") and item["path"].endswith(".zip")),
        key=lambda item: item["path"],
    )
    if total % (2 * len(archives)):
        raise ValueError(
            f"--aigibench-count must be divisible by {2 * len(archives)} "
            "to balance every archive and class exactly."
        )
    per_class = total // (2 * len(archives))

    def inspect(item: dict[str, Any]) -> tuple[list[Candidate], dict[str, Any]]:
        archive = item["path"]
        generator = Path(archive).stem
        groups: dict[int, list[Candidate]] = {0: [], 1: []}
        with RemoteZip(_huggingface_url(archive)) as remote:
            for info in remote.infolist():
                member = info.filename
                if not _is_image(member):
                    continue
                parts = PurePosixPath(member).parts
                if "0_real" in parts:
                    target = 0
                elif "1_fake" in parts:
                    target = 1
                else:
                    continue
                priority = _stable_priority(selection_seed, "aigibench", archive, member)
                groups[target].append(Candidate(
                    dataset_id="aigibench-stratified-test",
                    archive=archive,
                    member=member,
                    target=target,
                    stratum=f"{generator}/{target}",
                    generator_family=_aigibench_family(generator) if target else "authentic",
                    generation_model=generator if target else "none",
                    source_dataset=(
                        f"AIGIBench/{generator}/real-reference" if not target else "AIGIBench"
                    ),
                    authentic_subtype=(
                        f"AIGIBench/{generator}/real-reference" if not target else "not_applicable"
                    ),
                    priority=priority,
                ))
        selected: list[Candidate] = []
        for target in (0, 1):
            if len(groups[target]) < per_class:
                raise ValueError(f"{archive} has too few target={target} rows.")
            selected.extend(
                sorted(groups[target], key=lambda row: (row.priority, row.member))[
                    : per_class + reserve_per_stratum
                ]
            )
        inventory = {
            "archive": archive,
            "archive_bytes": item["size"],
            "real_images": len(groups[0]),
            "aigc_images": len(groups[1]),
            "selected_real": per_class,
            "selected_aigc": per_class,
        }
        print(
            f"AIGIBench inventory {archive}: {len(groups[0])} real, "
            f"{len(groups[1])} AIGC; selected {2 * per_class}",
            flush=True,
        )
        return selected, inventory

    selected: list[Candidate] = []
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inspect, item) for item in archives]
        for future in as_completed(futures):
            rows, details = future.result()
            selected.extend(rows)
            inventory.append(details)
    quotas = {
        f"{Path(item['path']).stem}/{target}": per_class
        for item in archives
        for target in (0, 1)
    }
    return (
        sorted(selected, key=lambda row: (row.archive, row.target, row.priority)),
        sorted(inventory, key=lambda row: row["archive"]),
        quotas,
    )


def _wildfake_candidate(
    archive: str,
    member: str,
    selection_seed: int,
) -> Candidate:
    path = PurePosixPath(member)
    parts = path.parts
    if archive.startswith("Images/Real/"):
        source = Path(archive).stem.upper() if Path(archive).stem == "ffhq" else Path(archive).stem
        return Candidate(
            "wildfake-reconstructed-test", archive, member, 0, f"real/{source}",
            "authentic", "none", source, source,
            _stable_priority(selection_seed, "wildfake", archive, member),
        )

    family = "diffusion"
    model = Path(archive).stem
    if archive.endswith("GAN_based.zip"):
        family = "gan"
        model = parts[2] if len(parts) > 2 else "GAN-unknown"
    elif archive.endswith("Other_based.zip"):
        family = "other"
        model = parts[2] if len(parts) > 2 else "other-unknown"
    elif archive.endswith("SDwithAdaptor.zip"):
        subtype = parts[1] if len(parts) > 1 else "unknown"
        model = f"SD-with-adapter/{subtype}"
    elif archive.endswith("personalizedSD.zip"):
        subtype = parts[1] if len(parts) > 1 else "unknown"
        model = f"personalized-SD/{subtype}"
    elif "/originalSD/" in archive:
        tier = PurePosixPath(archive).parts[-2]
        model = f"original-SD/{tier}"
    elif "/Midjourney/" in archive:
        tier = PurePosixPath(archive).parts[-2]
        model = f"Midjourney/{tier}"
    elif archive.endswith("DALLE.zip") and parts:
        model = parts[0]

    return Candidate(
        "wildfake-reconstructed-test", archive, member, 1, f"fake/{model}",
        family, model, "WildFake", "not_applicable",
        _stable_priority(selection_seed, "wildfake", archive, member),
    )


def _wildfake_inventory(
    total: int,
    selection_seed: int,
    split_seed: int,
    workers: int,
    reserve_per_stratum: int = 20,
) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    if total % 2:
        raise ValueError("--wildfake-count must be even for exact class balance.")
    files, dataset_metadata = _modelscope_tree()
    archives = sorted(
        ({"path": item["Path"], "size": int(item["Size"])} for item in files
         if item["Type"] == "blob"
         and item["Path"].startswith("Images/")
         and item["Path"].endswith(".zip")
         and int(item["Size"]) > 1_000),
        key=lambda item: item["path"],
    )
    per_class = total // 2
    # This is above the expected quota for 7 real and >20 generated strata.
    local_cap = max(2_000, per_class)

    def inspect(item: dict[str, Any]) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
        archive = item["path"]
        heaps: dict[str, list[tuple[int, str, Candidate]]] = {}
        total_images = 0
        test_images = 0
        structurally_invalid = 0
        with RemoteZip(_modelscope_url(archive)) as remote:
            for info in remote.infolist():
                member = info.filename
                if not _is_image(member):
                    continue
                total_images += 1
                if (
                    (archive, member) in KNOWN_INVALID_MEMBERS
                    or _structurally_invalid_zip_member(info)
                ):
                    structurally_invalid += 1
                    continue
                if not _in_reconstructed_test(split_seed, archive, member):
                    continue
                test_images += 1
                candidate = _wildfake_candidate(archive, member, selection_seed)
                _keep_smallest(heaps, candidate, limit=local_cap)
        candidates = {
            stratum: [item[2] for item in heap]
            for stratum, heap in heaps.items()
        }
        inventory = {
            "archive": archive,
            "archive_bytes": item["size"],
            "images": total_images,
            "reconstructed_test_images": test_images,
            "structurally_invalid_images_excluded": structurally_invalid,
            "strata": sorted(candidates),
        }
        print(
            f"WildFake inventory {archive}: {total_images} images, "
            f"{test_images} reconstructed test",
            flush=True,
        )
        return candidates, inventory

    global_heaps: dict[str, list[tuple[int, str, Candidate]]] = {}
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inspect, item) for item in archives]
        for future in as_completed(futures):
            candidates, details = future.result()
            inventory.append(details)
            for rows in candidates.values():
                for candidate in rows:
                    _keep_smallest(global_heaps, candidate, limit=local_cap)

    available = {
        stratum: len(heap)
        for stratum, heap in global_heaps.items()
    }
    real_available = {name: count for name, count in available.items() if name.startswith("real/")}
    fake_available = {name: count for name, count in available.items() if name.startswith("fake/")}
    quotas = {
        **_balanced_quotas(per_class, real_available),
        **_balanced_quotas(per_class, fake_available),
    }
    selected: list[Candidate] = []
    for stratum, quota in sorted(quotas.items()):
        rows = [item[2] for item in global_heaps[stratum]]
        selected.extend(
            sorted(rows, key=lambda row: (row.priority, row.archive, row.member))[
                : quota + reserve_per_stratum
            ]
        )
    return (
        sorted(selected, key=lambda row: (row.archive, row.target, row.priority)),
        sorted(inventory, key=lambda row: row["archive"]),
        dataset_metadata,
        quotas,
    )


def _image_metadata(payload: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        return {
            "width": int(image.width),
            "height": int(image.height),
            "mode": str(image.mode),
            "format": str(image.format or "unknown"),
        }


def _destination(output: Path, candidate: Candidate) -> Path:
    token = sha256(f"{candidate.archive}:{candidate.member}".encode()).hexdigest()[:20]
    suffix = PurePosixPath(candidate.member).suffix.casefold()
    label = "1_aigc" if candidate.target else "0_authentic"
    return output / "images" / candidate.dataset_id / label / _slug(candidate.stratum) / f"{token}{suffix}"


def _local_record(output: Path, candidate: Candidate, payload: bytes | None = None) -> dict[str, Any]:
    destination = _destination(output, candidate)
    if payload is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    stored = destination.read_bytes()
    try:
        details = _image_metadata(stored)
    except Exception as exc:
        raise ValueError(
            f"Invalid source image {candidate.archive}:{candidate.member} "
            f"stored at {destination}"
        ) from exc
    try:
        local_path = destination.relative_to(ROOT).as_posix()
    except ValueError:
        local_path = str(destination.resolve())
    parent_token = sha256(
        f"{candidate.dataset_id}:{candidate.archive}:{candidate.member}".encode()
    ).hexdigest()[:24]
    return {
        "dataset_id": candidate.dataset_id,
        "parent_id": f"{candidate.dataset_id}-{parent_token}",
        "lineage_id": f"{candidate.dataset_id}-{parent_token}",
        "image_path": local_path,
        "target": candidate.target,
        "generator_family": candidate.generator_family,
        "generation_model": candidate.generation_model,
        "source_dataset": candidate.source_dataset,
        "authentic_subtype": candidate.authentic_subtype,
        "stratum": candidate.stratum,
        "split": "test" if candidate.dataset_id.startswith("aigibench") else "reconstructed_test_20pct",
        "archive": candidate.archive,
        "archive_member": candidate.member,
        "selection_priority": f"{candidate.priority:032x}",
        "content_sha256": sha256(stored).hexdigest(),
        "bytes": len(stored),
        **details,
    }


def _download_archive(
    output: Path,
    archive: str,
    candidates: list[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    is_hf = candidates[0].dataset_id.startswith("aigibench")
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            url = _huggingface_url(archive) if is_hf else _modelscope_url(archive)
            records: list[dict[str, Any]] = []
            invalid: list[dict[str, str]] = []
            with RemoteZip(url) as remote:
                for index, candidate in enumerate(candidates, start=1):
                    destination = _destination(output, candidate)
                    try:
                        payload = None if destination.is_file() else remote.read(candidate.member)
                        records.append(_local_record(output, candidate, payload))
                    except ValueError as exc:
                        destination.unlink(missing_ok=True)
                        invalid.append({
                            "archive": candidate.archive,
                            "member": candidate.member,
                            "reason": str(exc),
                        })
                        print(f"Excluded invalid source member: {exc}", flush=True)
                    if index % 100 == 0 or index == len(candidates):
                        print(
                            f"Downloaded/verified {index}/{len(candidates)} from {archive}",
                            flush=True,
                        )
            return records, invalid
        except Exception as exc:  # retry expiring redirects and transient range failures
            last_error = exc
            print(f"Retry {attempt}/{attempts} for {archive}: {exc!r}", flush=True)
    assert last_error is not None
    raise last_error


def _download_selected(
    output: Path,
    candidates: Iterable[Candidate],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_archive: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_archive.setdefault(candidate.archive, []).append(candidate)
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_archive, output, archive, sorted(rows, key=lambda row: row.member))
            for archive, rows in sorted(by_archive.items())
        ]
        for future in as_completed(futures):
            archive_records, archive_invalid = future.result()
            records.extend(archive_records)
            invalid.extend(archive_invalid)
    return records, invalid


def _apply_final_quotas(
    records: list[dict[str, Any]],
    quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["dataset_id"]), str(row["stratum"])), []).append(row)
    selected: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for key, quota in sorted(quotas.items()):
        available = sorted(
            grouped.get(key, []),
            key=lambda row: (str(row["selection_priority"]), str(row["archive_member"])),
        )
        unique: list[dict[str, Any]] = []
        candidate_content = set(seen_content)
        for row in available:
            digest = str(row["content_sha256"])
            if digest in candidate_content:
                continue
            unique.append(row)
            candidate_content.add(digest)
        if len(unique) < quota:
            raise RuntimeError(f"Only {len(unique)} unique valid rows remain for {key}; need {quota}.")
        chosen = unique[:quota]
        selected.extend(chosen)
        seen_content.update(str(row["content_sha256"]) for row in chosen)
    return selected


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.wildfake_count < 2:
        raise ValueError("--wildfake-count must be at least two.")
    if not args.wildfake_only and args.aigibench_count < 2:
        raise ValueError("--aigibench-count must be at least two unless --wildfake-only is set.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    aigibench: list[Candidate] = []
    aigibench_inventory: list[dict[str, Any]] = []
    aigibench_quotas: dict[str, int] = {}
    if not args.wildfake_only:
        print("Resolving AIGIBench test inventory and selection...", flush=True)
        aigibench, aigibench_inventory, aigibench_quotas = _aigibench_inventory(
            args.aigibench_count, args.selection_seed, args.workers
        )
    print("Resolving WildFake reconstructed-test inventory and selection...", flush=True)
    wildfake, wildfake_inventory, wildfake_metadata, wildfake_quotas = _wildfake_inventory(
        args.wildfake_count,
        args.selection_seed,
        args.wildfake_split_seed,
        args.workers,
    )
    selected = [*aigibench, *wildfake]
    if args.inventory_only:
        print(
            f"Inventory selected {len(selected)} rows including deterministic reserves; "
            "image download skipped.",
            flush=True,
        )
        return

    print(
        f"Range-downloading {len(selected)} selected source images including reserves...",
        flush=True,
    )
    downloaded_records, invalid_members = _download_selected(output, selected, args.workers)
    final_quotas = {
        **{("aigibench-stratified-test", key): value for key, value in aigibench_quotas.items()},
        **{("wildfake-reconstructed-test", key): value for key, value in wildfake_quotas.items()},
    }
    records = _apply_final_quotas(downloaded_records, final_quotas)
    expected_records = args.wildfake_count + (0 if args.wildfake_only else args.aigibench_count)
    if len(records) != expected_records:
        raise RuntimeError(
            f"Final manifest has {len(records)} rows instead of "
            f"{expected_records}."
        )
    records.sort(key=lambda row: (row["dataset_id"], row["target"], row["stratum"], row["parent_id"]))
    _write_manifest(output / "manifest.csv", records)

    content_hash = sha256()
    for row in records:
        content_hash.update(f"{row['parent_id']}:{row['content_sha256']}\n".encode())
    metadata = {
        "schema_version": 1,
        "selection_seed": args.selection_seed,
        "wildfake_split_seed": args.wildfake_split_seed,
        "selection_policy": {
            "aigibench": "equal counts from 0_real and 1_fake in each of 25 official test archives",
            "wildfake": (
                "equal class totals; near-equal allocation across authentic-source and generated-model "
                "strata after deterministic reconstruction of the paper's unpublished 80/20 split"
            ),
            "priority": "smallest SHA-256(seed,dataset,archive,member) within each stratum",
            "deduplication": "retain the highest-priority occurrence of each exact source-byte SHA-256",
        },
        "datasets": {
            **({"aigibench-stratified-test": {
                "source": f"https://huggingface.co/datasets/{HF_REPO}",
                "revision": HF_REVISION,
                "license": HF_LICENSE,
                "selected_images": args.aigibench_count,
                "inventory": aigibench_inventory,
            }} if not args.wildfake_only else {}),
            "wildfake-reconstructed-test": {
                "source": f"https://modelscope.cn/datasets/{MODELSCOPE_REPO}/summary",
                "revision": "master",
                "last_updated_time": wildfake_metadata.get("LastUpdatedTime"),
                "license": MODELSCOPE_LICENSE,
                "selected_images": args.wildfake_count,
                "split_note": (
                    "WildFake publishes an 80/20 random per-stratum policy but no membership file; "
                    "this is a deterministic reconstruction, not the authors' exact hidden split."
                ),
                "source_integrity_filter": (
                    "exclude ZIP members larger than 10 KB with compressed/uncompressed ratio "
                    "below 0.01, then verify every downloaded image with Pillow"
                ),
                "inventory": wildfake_inventory,
            },
        },
        "manifest": str((output / "manifest.csv").resolve()),
        "invalid_members_encountered": sorted(
            invalid_members, key=lambda row: (row["archive"], row["member"])
        ),
        "content_manifest_sha256": content_hash.hexdigest(),
        "downloaded_bytes": sum(int(row["bytes"]) for row in records),
        "rows": len(records),
    }
    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {len(records)} images ({metadata['downloaded_bytes'] / 2**30:.2f} GiB) "
        f"at {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
