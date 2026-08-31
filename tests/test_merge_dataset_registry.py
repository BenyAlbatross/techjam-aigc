import json
from pathlib import Path

from scripts.merge_dataset_registry import merge_registries


ROOT = Path(__file__).resolve().parents[1]


def test_registry_groups_without_upgrading_permissions() -> None:
    merged = merge_registries(
        ROOT / "datasets.toml", ROOT / "configs/data-expansion.json"
    )
    by_id = {item["id"]: item for item in merged["datasets"]}
    assert by_id["sid_set"]["status"] == "approved"
    assert "larger_local_discovery_and_confirmation" in by_id["sid_set"]["roles"]
    assert by_id["community_forensics_eval"]["status"] == "blocked"
    assert by_id["community_forensics_diversity"]["status"] == "review"
    assert by_id["wildfake_large"]["status"] == "review"
    assert set(merged["groups"]) == {
        "approved", "review", "blocked", "training_candidates", "evaluation_only"
    }
    json.dumps(merged)
