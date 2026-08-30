"""Network-free registry checks for benchmark and release inputs."""

import argparse
from pathlib import Path
import tomllib


REQUIRED_MODEL = {
    "status", "submission_status", "repository", "revision", "file",
    "sha256", "license", "threshold", "parameters", "loader",
}
REQUIRED_DATASET = {"status", "repository", "revision", "split", "license"}
ROOT = Path(__file__).resolve().parents[1]


def load_registry(path: Path, section: str) -> dict[str, dict]:
    with path.open("rb") as handle:
        return tomllib.load(handle)[section]


def check_models(entries: dict, names: list[str], scope: str) -> list[str]:
    errors = []
    for name in names:
        entry = entries.get(name)
        if entry is None:
            errors.append(f"{name}: absent from registry")
            continue
        missing = REQUIRED_MODEL - entry.keys()
        if missing:
            errors.append(f"{name}: missing {sorted(missing)}")
        if entry.get("status") != "approved":
            errors.append(f"{name}: status is {entry.get('status')}")
        if scope == "submission" and entry.get("submission_status") != "approved":
            errors.append(f"{name}: submission status is {entry.get('submission_status')}")
        if int(entry.get("parameters", 2_000_000_000)) >= 2_000_000_000:
            errors.append(f"{name}: parameter count must be below 2,000,000,000")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            errors.append(f"{name}: SHA-256 must contain 64 hex characters")
    return errors


def check_datasets(entries: dict, names: list[str]) -> list[str]:
    errors = []
    for name in names:
        entry = entries.get(name)
        if entry is None:
            errors.append(f"{name}: absent from registry")
            continue
        missing = REQUIRED_DATASET - entry.keys()
        if missing:
            errors.append(f"{name}: missing {sorted(missing)}")
        if entry.get("status") != "approved":
            errors.append(f"{name}: status is {entry.get('status')}")
    return errors


def check_release(path: Path) -> list[str]:
    with path.open("rb") as handle:
        release = tomllib.load(handle).get("release", {})
    return [name for name, value in release.items() if isinstance(value, bool) and not value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--scope", choices=["benchmark", "submission"], default="benchmark")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--dataset", action="append", default=[])
    args = parser.parse_args()

    models = load_registry(ROOT / "models.toml", "models")
    datasets = load_registry(ROOT / "datasets.toml", "datasets")
    model_names = list(models) if args.models == ["all"] else args.models
    dataset_names = args.dataset or ["sid_set"]
    errors = check_models(models, model_names, args.scope)
    errors.extend(check_datasets(datasets, dataset_names))
    if args.scope == "submission":
        errors.extend(check_release(ROOT / "compliance.toml"))
    if errors:
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
