"""Preview or explicitly confirm confined competition-data cleanup."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SAFE_TARGETS = (
    Path("work/data"),
    Path("work/manifests"),
    Path("work/hf-cache"),
    Path("work/model-cache"),
    Path("work/predictions"),
    Path("work/reports"),
)


def _root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cleanup root does not exist: {root}") from error
    if not resolved.is_dir() or root.is_symlink():
        raise ValueError(f"cleanup root must be a real directory: {root}")
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"cleanup root is too broad: {root}")
    return resolved


def _validate_target(root: Path, target: Path) -> Path:
    allowed = tuple(root / relative for relative in SAFE_TARGETS)
    if not target.is_absolute() or target not in allowed:
        raise ValueError(f"target is not an exact safe target: {target}")
    if target.is_symlink():
        raise ValueError(f"target is a symlink escape: {target}")
    try:
        resolved = target.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"target does not exist: {target}") from error
    if resolved != target:
        raise ValueError(f"target contains a symlink or path escape: {target}")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"target escapes cleanup root: {target}") from error
    if not resolved.is_dir():
        raise ValueError(f"safe target is not a directory: {target}")
    return resolved


def cleanup_targets(root: Path) -> list[Path]:
    root = _root(root)
    targets = []
    for relative in SAFE_TARGETS:
        target = root / relative
        if target.is_symlink() or target.exists():
            targets.append(_validate_target(root, target))
    return targets


def _inventory(root: Path, targets: list[Path]) -> list[dict]:
    records = []
    for target in targets:
        byte_count = 0
        file_count = 0
        for directory, directory_names, file_names in os.walk(
            target, followlinks=False
        ):
            directory_path = Path(directory)
            names = file_names + [
                name for name in directory_names if (directory_path / name).is_symlink()
            ]
            for name in names:
                byte_count += (directory_path / name).lstat().st_size
                file_count += 1
        records.append(
            {
                "path": target.relative_to(root).as_posix(),
                "byte_count": byte_count,
                "file_count": file_count,
            }
        )
    return records


def _safe_parent(root: Path, relative: Path) -> Path:
    parent = root / relative
    if parent.is_symlink():
        raise ValueError(f"output directory is a symlink escape: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError(f"output directory escapes cleanup root: {parent}")
    return parent


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preview(root: Path) -> Path:
    root = _root(root)
    targets = cleanup_targets(root)
    inventory = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": _inventory(root, targets),
    }
    path = _safe_parent(root, Path("work")) / "cleanup-inventory.json"
    _write_json(path, inventory)
    print(f"Preview only: {len(targets)} cleanup targets; deleted 0")
    return path


def delete_targets(root: Path, targets: list[Path]) -> None:
    root = _root(root)
    validated = [_validate_target(root, Path(target)) for target in targets]
    if len(set(validated)) != len(validated):
        raise ValueError("duplicate cleanup target")
    records = _inventory(root, validated)
    output = _safe_parent(root, Path("outputs"))
    for target in validated:
        shutil.rmtree(target)
    _write_json(
        output / "data-deletion-attestation.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deleted_targets": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("--root", type=Path, default=ROOT)
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--root", type=Path, default=ROOT)
    delete_parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.command == "preview":
        preview(args.root)
        return
    if not args.confirm:
        parser.error("delete requires --confirm")
    targets = cleanup_targets(args.root)
    delete_targets(args.root, targets)


if __name__ == "__main__":
    main()
