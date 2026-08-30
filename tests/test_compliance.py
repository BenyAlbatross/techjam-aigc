from pathlib import Path

from scripts.compliance import check_datasets, check_models, load_registry


def test_rejects_blocked_and_oversized_assets(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[models.bad]\nstatus="blocked"\nsubmission_status="blocked"\n'
        'revision="abc"\nsha256="deadbeef"\nlicense="unknown"\n'
        'threshold=0.5\nparameters=2000000000\nloader="hf_multiclass"\n'
    )
    entries = load_registry(path, "models")
    errors = check_models(entries, ["bad"], "benchmark")
    assert any("status" in error for error in errors)
    assert any("2,000,000,000" in error for error in errors)


def test_approved_asset_passes():
    entry = {"ok": {
        "status": "approved", "submission_status": "review",
        "repository": "owner/repo", "file": "model.bin",
        "revision": "abc", "sha256": "a" * 64, "license": "MIT",
        "threshold": 0.5, "parameters": 1, "loader": "hf_multiclass",
    }}
    assert check_models(entry, ["ok"], "benchmark") == []


def test_dataset_requires_approval():
    entry = {"ntire": {
        "status": "review", "repository": "owner/data",
        "revision": "abc", "split": "shard_0", "license": "UNDECLARED",
    }}
    assert check_datasets(entry, ["ntire"]) == ["ntire: status is review"]


def test_sid_set_requires_exact_label_2_exclusion():
    entry = {"sid_set": {
        "status": "approved", "repository": "owner/data",
        "revision": "abc", "split": "validation", "license": "CC-BY-4.0",
    }}
    assert check_datasets(entry, ["sid_set"]) == [
        "sid_set: excluded_labels must be [2]",
    ]
