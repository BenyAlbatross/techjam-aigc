"""Preview or explicitly confirm confined competition-data cleanup."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import stat


ROOT = Path(__file__).resolve().parents[1]
SAFE_TARGETS = (
    Path("work/data"),
    Path("work/manifests"),
    Path("work/hf-cache"),
    Path("work/model-cache"),
    Path("work/predictions"),
    Path("work/reports"),
)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


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


def _open_root(root: Path) -> int:
    expected = root.stat(follow_symlinks=False)
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    if not os.path.samestat(expected, os.fstat(descriptor)):
        os.close(descriptor)
        raise OSError("cleanup root changed while opening")
    return descriptor


def _open_relative_dir(root_descriptor: int, relative: Path, *, create=False) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for part in relative.parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _anchor_targets(root_descriptor: int, root: Path, targets: list[Path]):
    anchors = []
    try:
        for target in targets:
            relative = target.relative_to(root)
            parent_descriptor = _open_relative_dir(root_descriptor, relative.parent)
            try:
                target_descriptor = os.open(
                    relative.name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor
                )
            except BaseException:
                os.close(parent_descriptor)
                raise
            anchors.append(
                (relative, parent_descriptor, target_descriptor)
            )
        return anchors
    except BaseException:
        for _, parent_descriptor, target_descriptor in anchors:
            os.close(target_descriptor)
            os.close(parent_descriptor)
        raise


def _measure_directory(descriptor: int) -> tuple[int, int]:
    byte_count = 0
    file_count = 0
    with os.scandir(descriptor) as iterator:
        entries = list(iterator)
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                if not os.path.samestat(entry_stat, os.fstat(child)):
                    raise OSError(f"inventory directory changed: {entry.name}")
                child_bytes, child_files = _measure_directory(child)
            finally:
                os.close(child)
            byte_count += child_bytes
            file_count += child_files
        else:
            byte_count += entry_stat.st_size
            file_count += 1
    return byte_count, file_count


def _inventory(anchors) -> list[dict]:
    records = []
    for relative, _, target_descriptor in anchors:
        byte_count, file_count = _measure_directory(target_descriptor)
        records.append(
            {
                "path": relative.as_posix(),
                "byte_count": byte_count,
                "file_count": file_count,
            }
        )
    return records


def _write_json_at(parent_descriptor: int, name: str, value: dict) -> None:
    payload = json.dumps(value, indent=2) + "\n"
    temporary = None
    descriptor = None
    try:
        for _ in range(32):
            temporary = f".{name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("cannot allocate exclusive cleanup JSON temp file")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def preview(root: Path) -> Path:
    root = _root(root)
    targets = cleanup_targets(root)
    root_descriptor = _open_root(root)
    anchors = []
    try:
        anchors = _anchor_targets(root_descriptor, root, targets)
        inventory = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "targets": _inventory(anchors),
        }
        work_descriptor = _open_relative_dir(
            root_descriptor, Path("work"), create=True
        )
        try:
            _write_json_at(work_descriptor, "cleanup-inventory.json", inventory)
        finally:
            os.close(work_descriptor)
    finally:
        for _, parent_descriptor, target_descriptor in anchors:
            os.close(target_descriptor)
            os.close(parent_descriptor)
        os.close(root_descriptor)
    print(f"Preview only: {len(targets)} cleanup targets; deleted 0")
    return root / "work/cleanup-inventory.json"


def delete_targets(root: Path, targets: list[Path]) -> None:
    root = _root(root)
    validated = [_validate_target(root, Path(target)) for target in targets]
    if len(set(validated)) != len(validated):
        raise ValueError("duplicate cleanup target")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("platform lacks symlink-safe directory deletion")

    root_descriptor = _open_root(root)
    anchors = []
    output_descriptor = None
    try:
        anchors = _anchor_targets(root_descriptor, root, validated)
        records = _inventory(anchors)
        output_descriptor = _open_relative_dir(
            root_descriptor, Path("outputs"), create=True
        )
        for relative, parent_descriptor, target_descriptor in anchors:
            current = os.stat(
                relative.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
                current, os.fstat(target_descriptor)
            ):
                raise OSError(f"cleanup target changed before deletion: {relative}")
        for relative, parent_descriptor, _ in anchors:
            shutil.rmtree(relative.name, dir_fd=parent_descriptor)
        _write_json_at(
            output_descriptor,
            "data-deletion-attestation.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deleted_targets": records,
            },
        )
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        for _, parent_descriptor, target_descriptor in anchors:
            os.close(target_descriptor)
            os.close(parent_descriptor)
        os.close(root_descriptor)


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
