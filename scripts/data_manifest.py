"""Build and validate deterministic local image manifests."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from scripts.compliance import check_datasets, load_registry
else:
    from compliance import check_datasets, load_registry


ROOT = Path(__file__).resolve().parents[1]
SID_NAME = "sid_set"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(payload: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return output


def _dataset(name: str) -> dict:
    entries = load_registry(ROOT / "datasets.toml", "datasets")
    errors = check_datasets(entries, [name])
    if errors:
        raise ValueError("; ".join(errors))
    return entries[name]


def validate_manifest(payload: dict, root: Path) -> list[str]:
    errors, ids, hashes = [], set(), {}
    for row in payload.get("samples", []):
        sample_id = str(row.get("sample_id", ""))
        relative = Path(str(row.get("path", "")))
        if not sample_id or sample_id in ids:
            errors.append(f"{sample_id}: duplicate or empty sample_id")
        ids.add(sample_id)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{sample_id}: path must stay relative")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{sample_id}: absent file")
            continue
        digest = _digest(path)
        if digest != row.get("sha256"):
            errors.append(f"{sample_id}: hash mismatch")
        label = row.get("label")
        if label not in (0, 1):
            errors.append(f"{sample_id}: invalid label")
        if not row.get("license"):
            errors.append(f"{sample_id}: missing rights")
        if digest in hashes and hashes[digest] != label:
            errors.append(f"{sample_id}: same bytes have conflicting labels")
        hashes[digest] = label
    return errors


def _image_bytes(image: object) -> tuple[bytes, str]:
    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
        raise ValueError("SID row has no undecoded image bytes")
    return image["bytes"], Path(str(image.get("path", ""))).suffix or ".img"


def build_sid(output: Path, per_class: int) -> Path:
    entry = _dataset(SID_NAME)
    from datasets import Image as HFImage, load_dataset

    rows = load_dataset(
        entry["repository"], split=entry["split"], streaming=True,
        revision=entry["revision"],
    ).cast_column("image", HFImage(decode=False))
    images = ROOT / "work/data/sid_set/images"
    selected = {0: 0, 1: 0}
    samples = []
    hashes = {}
    excluded = set(entry["excluded_labels"])
    for source_row, row in enumerate(rows):
        label = row.get("label")
        if label in excluded or label not in selected or selected[label] >= per_class:
            continue
        image, suffix = _image_bytes(row.get("image"))
        digest = hashlib.sha256(image).hexdigest()
        if digest in hashes:
            if hashes[digest] != label:
                raise ValueError("SID_Set has same bytes with conflicting labels")
            continue
        hashes[digest] = label
        filename = f"{digest}{suffix.lower()}"
        local = images / filename
        images.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            temporary = local.with_suffix(local.suffix + ".tmp")
            temporary.write_bytes(image)
            os.replace(temporary, local)
        source_id = str(row.get("id", row.get("image_id", source_row)))
        samples.append({
            "sample_id": digest, "base_id": source_id, "label": label,
            "truth": "ai" if label else "real", "path": str(Path("data/sid_set/images") / filename),
            "sha256": digest, "source_row": source_row,
            "source_family": "OpenImages_V7" if label == 0 else "SID_Set_generated",
            "generator_family": "" if label == 0 else "SID_Set_undisclosed_mixture",
            "license": entry["license"],
        })
        selected[label] += 1
        if all(count == per_class for count in selected.values()):
            break
    if any(count != per_class for count in selected.values()):
        raise ValueError(f"SID_Set lacks {per_class} samples for each class")
    return _write_json({
        "dataset": SID_NAME, "repository": entry["repository"], "revision": entry["revision"],
        "split": entry["split"], "license": entry["license"], "per_class": per_class,
        "limitations": "Generated images use an undisclosed SID_Set generator mixture.",
        "samples": samples,
    }, output)


def _build_ntire(shard_dir: Path, output: Path) -> Path:
    samples = []
    with (shard_dir / "labels.csv").open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["image"])
    images = (shard_dir / "images").resolve()
    for row in rows:
        image = Path(row["image"])
        path = (images / image).resolve()
        if not path.is_relative_to(images):
            raise ValueError(f"{image}: path must stay under images")
        label = int(row["label"])
        samples.append({
            "sample_id": image.name, "base_id": image.name, "label": label,
            "truth": "ai" if label else "real", "path": str(Path("images") / image),
            "sha256": _digest(path), "source_family": "NTIRE_2026_train",
            "generator_family": "NTIRE_2026_train" if label else "",
            "license": "UNDECLARED",
        })
    return _write_json({"dataset": "ntire_2026_train", "samples": samples}, output)


def build_ntire(shard_dir: Path, output: Path) -> Path:
    _dataset("ntire_2026_train")
    return _build_ntire(shard_dir, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sid = commands.add_parser("build-sid")
    sid.add_argument("--output", type=Path, required=True)
    sid.add_argument("--per-class", type=int, default=1000)
    ntire = commands.add_parser("build-ntire")
    ntire.add_argument("shard_dir", type=Path)
    ntire.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "build-sid":
        build_sid(args.output, args.per_class)
    elif args.command == "build-ntire":
        build_ntire(args.shard_dir, args.output)
    else:
        errors = validate_manifest(json.loads(args.manifest.read_text()), args.manifest.parent.parent)
        if errors:
            print("\n".join(errors))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
