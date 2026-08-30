from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

if __package__:
    from scripts.compliance import check_models, load_registry
else:
    from compliance import check_models, load_registry


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assets(entry: dict) -> list[dict[str, str]]:
    assets = [{"file": entry["file"], "sha256": entry["sha256"]}]
    if auxiliary := entry.get("auxiliary"):
        assets.append(auxiliary)
    assets.extend(entry.get("auxiliary_files", []))
    return assets


def fetch_model(name: str, cache: Path) -> Path:
    entry = load_registry(ROOT / "models.toml", "models")[name]
    errors = check_models({name: entry}, [name], "benchmark")
    if errors:
        raise RuntimeError("\n".join(errors))
    assets = _assets(entry)
    snapshot = Path(snapshot_download(
        repo_id=entry["repository"],
        revision=entry["revision"],
        cache_dir=cache,
        allow_patterns=[
            *(asset["file"] for asset in assets),
            "config.json",
            "preprocessor_config.json",
        ],
    ))
    for index, asset in enumerate(assets):
        path = snapshot / asset["file"]
        if sha256_file(path) != asset["sha256"]:
            path.unlink()
            kind = "checkpoint" if index == 0 else "auxiliary checkpoint"
            raise RuntimeError(f"{name}: {kind} hash mismatch")
    return snapshot


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, default=Path("work/hf-cache"))
    args = parser.parse_args()
    snapshot = fetch_model(args.model, args.cache)
    entry = load_registry(ROOT / "models.toml", "models")[args.model]
    for asset in _assets(entry):
        print(f"{args.model} verified {asset['file']} sha256={asset['sha256']}")
    print(snapshot)


if __name__ == "__main__":
    main()
