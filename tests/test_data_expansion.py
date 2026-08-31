from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from techjam_aigc.feature_lab.data import load_binary_index
from techjam_aigc.feature_lab.evaluation import write_evaluation_cache
from techjam_aigc.feature_lab.expansion import (
    acquisition_audit,
    assert_acquisition_allowed,
    audit_expansion_manifest,
    build_audit_summary,
    load_expansion_config,
    plan_diversity_cohorts,
    select_expansion_rows,
    write_expansion_artifacts,
)
from techjam_aigc.feature_lab.pipeline import ExperimentConfig, run_extraction


ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_is_dry_run_and_blocks_pending_sources() -> None:
    config = load_expansion_config(ROOT / "configs/data-expansion.json")
    assert config["acquisition"]["default_mode"] == "dry_run"
    assert {source["dataset"] for source in config["sources"]} >= {
        "SID Set", "WildFake", "AI-GenBench", "Community Forensics"
    }
    audit = acquisition_audit(config)
    assert not audit.loc[audit["selected"], "acquisition_allowed"].any()
    with pytest.raises(PermissionError, match="Acquisition blocked"):
        assert_acquisition_allowed(config)


def test_allowlisting_requires_license_reviewer_file_list_and_hashes() -> None:
    config = _config()
    source = config["sources"][0]
    source.update(
        {
            "revision": "commit-123",
            "file_list": "manifests/source-files.txt",
            "file_list_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "license_name": "Apache-2.0",
            "license_url": "https://example.test/license",
            "license_decision": "allowlisted",
            "underlying_image_license_name": "Apache-2.0",
            "underlying_image_license_url": "https://example.test/output-license",
            "underlying_image_license_decision": "allowlisted",
            "reviewer": "reviewer@example.test",
            "review_date": "2026-08-30",
        }
    )
    assert_acquisition_allowed(config)
    source["file_list_sha256"] = "not-a-hash"
    with pytest.raises(PermissionError):
        assert_acquisition_allowed(config)


@pytest.mark.parametrize("label", ["tampered", "AI-edited", "partial_composite", "inpainted"])
def test_rejects_out_of_scope_labels(label: str) -> None:
    with pytest.raises(ValueError, match="Tampered, edited, or composited"):
        audit_expansion_manifest(pd.DataFrame([_row("one", label=label)]), _config())


def test_rejects_organizer_demo_split() -> None:
    row = _row("demo", generation_model="DALL-E Advanced", source_dataset="COCO val2017")
    with pytest.raises(ValueError, match="demonstration-only"):
        audit_expansion_manifest(pd.DataFrame([row]), _config())


def test_final_confirmation_is_sealed_and_has_no_generator_overlap() -> None:
    config = _config(final_generators=["future-gen"])
    valid = pd.DataFrame([
        _row("old", generation_model="old-gen", phase="discovery"),
        _row("future", generation_model="future-gen", phase="final_confirmation"),
    ])
    audited = audit_expansion_manifest(valid, config)
    assert set(audited["phase"]) == {"discovery", "final_confirmation"}

    overlap = valid.copy()
    overlap.loc[0, "generation_model"] = "FUTURE-GEN"
    with pytest.raises(ValueError, match="overlap|appears in discovery"):
        audit_expansion_manifest(overlap, config)

    unassigned = valid.copy()
    unassigned.loc[1, "generation_model"] = "other-future"
    with pytest.raises(ValueError, match="Unassigned generators"):
        audit_expansion_manifest(unassigned, config)


def test_ordinary_confirmation_is_distinct_from_sealed_final_confirmation() -> None:
    manifest = pd.DataFrame([
        _row("discover", generation_model="seen-gen", phase="discovery"),
        _row("confirm", generation_model="evaluation-gen", phase="confirmation"),
    ])
    audited = audit_expansion_manifest(manifest, _config())
    assert set(audited["phase"]) == {"discovery", "confirmation"}


def test_selection_is_stable_stratified_and_preserves_provenance() -> None:
    config = _config()
    config["sources"][0]["target_images_per_stratum"] = 3
    rows = []
    for generator in ("g1", "g2"):
        rows.extend(_row(f"{generator}-{index}", generation_model=generator) for index in range(8))
    rows.extend(
        _row(f"real-{index}", label="authentic", generation_model="none", source_dataset="camera-a")
        for index in range(8)
    )
    manifest = pd.DataFrame(rows)
    first = select_expansion_rows(manifest, config)
    second = select_expansion_rows(manifest.sample(frac=1, random_state=8), config)
    assert first["parent_id"].tolist() == second["parent_id"].tolist()
    assert first.groupby("selection_stratum").size().eq(3).all()
    assert {"source_member", "local_path", "revision", "parent_id"} <= set(first)


def test_few_and_many_generator_cohorts_are_disjoint_and_equal_size() -> None:
    config = _config()
    rows = [
        _row(f"g{generator}-{index}", generation_model=f"g{generator}")
        for generator in range(7)
        for index in range(5)
    ]
    selected = audit_expansion_manifest(pd.DataFrame(rows), config)
    cohorts = plan_diversity_cohorts(
        selected,
        source_id="test_source",
        few_generators=2,
        many_generators=4,
        total_images=8,
        seed="fixed",
    )
    counts = cohorts.groupby("diversity_cohort").size()
    assert counts.to_dict() == {"few_generators": 8, "many_generators": 8}
    generator_sets = cohorts.groupby("diversity_cohort")["generator_id"].agg(set)
    assert generator_sets["few_generators"].isdisjoint(generator_sets["many_generators"])
    repeat = plan_diversity_cohorts(
        selected.sample(frac=1, random_state=4),
        source_id="test_source",
        few_generators=2,
        many_generators=4,
        total_images=8,
        seed="fixed",
    )
    assert cohorts["parent_id"].tolist() == repeat["parent_id"].tolist()


def test_writes_coverage_and_audit_artifacts_without_network(tmp_path: Path) -> None:
    config = _config()
    manifest = pd.DataFrame([_row("fake"), _row("real", label="real", generation_model="none")])
    selected = select_expansion_rows(manifest, config)
    write_expansion_artifacts(tmp_path, manifest, selected, config)
    assert {path.name for path in tmp_path.iterdir()} == {
        "selection.csv", "coverage.csv", "source_audit.csv", "audit.json"
    }
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["network_actions_performed"] is False
    assert audit["selected_parent_ids_unique"] is True
    assert audit["selection_sha256"] == sha256(
        (tmp_path / "selection.csv").read_bytes()
    ).hexdigest()
    assert build_audit_summary(manifest, selected, config)["final_confirmation"]["sealed"] is True


def test_feature_lab_loads_expansion_selection_and_preserves_parent_ids(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    frame = pd.DataFrame([
        _feature_row("stable-parent-a", phase="discovery", generation_model="old-gen"),
        _feature_row("stable-parent-b", phase="confirmation", generation_model="eval-gen"),
        _feature_row("stable-parent-c", phase="final_confirmation", generation_model="future-gen"),
    ])
    frame.to_csv(path, index=False)
    loaded = load_binary_index(path)
    assert loaded["parent_id"].tolist() == frame["parent_id"].tolist()
    assert set(loaded["phase"]) == {"discovery", "confirmation", "final_confirmation"}
    assert loaded["binary_label"].tolist() == ["AIGC", "AIGC", "AIGC"]


def test_feature_lab_rejects_unsafe_expansion_selection(tmp_path: Path) -> None:
    unsafe_cases = [
        [_feature_row("edited", label="AI-edited")],
        [_feature_row("mismatch", label="real", target=1)],
        [
            _feature_row(
                "demo",
                generation_model="DALL-E Advanced",
                source_dataset="COCO val2017",
            )
        ],
        [
            _feature_row("overlap-a", phase="discovery", generation_model="same-gen"),
            _feature_row("overlap-b", phase="final_confirmation", generation_model="SAME-GEN"),
        ],
    ]
    for index, rows in enumerate(unsafe_cases):
        path = tmp_path / f"unsafe-{index}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_binary_index(path)


def test_feature_lab_requires_discovery_and_an_evaluation_phase(tmp_path: Path) -> None:
    for phase in ("discovery", "confirmation"):
        path = tmp_path / f"only-{phase}.csv"
        pd.DataFrame([_feature_row(phase, phase=phase)]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="discovery and at least one evaluation"):
            load_binary_index(path)


def test_run_extraction_uses_and_hashes_requested_index(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in (("discovery.png", "navy"), ("confirmation.png", "gold")):
        Image.new("RGB", (16, 16), color).save(image_dir / name)
    index_path = tmp_path / "expansion.csv"
    rows = [
        _feature_row(
            "parent-discovery",
            phase="discovery",
            generation_model="old-gen",
            local_path="images/discovery.png",
        ),
        _feature_row(
            "parent-confirmation",
            phase="confirmation",
            generation_model="eval-gen",
            local_path="images/confirmation.png",
        ),
    ]
    pd.DataFrame(rows).to_csv(index_path, index=False)
    output = tmp_path / "output"
    _, binary, metadata = run_extraction(
        tmp_path,
        output,
        index_path=Path("expansion.csv"),
        config=ExperimentConfig(
            views=("canonical_128",),
            conditions=("clean",),
            feature_profile="frozen_v1",
            bootstrap_repetitions=1,
        ),
    )
    assert binary["parent_id"].tolist() == ["parent-discovery", "parent-confirmation"]
    assert metadata["input_index"] == str(index_path.resolve())
    assert metadata["input_index_sha256"] == sha256(index_path.read_bytes()).hexdigest()
    assert metadata["config"]["feature_profile"] == "frozen_v1"
    assert metadata["config"]["transform_profile"] == "core"
    assert metadata["extraction_cost"]["wall_seconds"] > 0
    assert metadata["extraction_cost"]["seconds_per_parent"] > 0
    assert metadata["extraction_cost"]["rows_per_second"] > 0


def test_planner_to_extraction_to_evaluation_is_license_gated_and_chronological(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "data"
    image_dir.mkdir()
    rows = []
    cases = (
        ("d-real", "real", "none", "discovery", "black"),
        ("d-fake", "full_synthetic", "old-gen", "discovery", "white"),
        ("c-real", "real", "none", "confirmation", "navy"),
        ("c-fake", "full_synthetic", "eval-gen", "confirmation", "gold"),
    )
    for parent_id, label, generator, phase, color in cases:
        Image.new("RGB", (16, 16), color).save(image_dir / f"{parent_id}.png")
        row = _row(parent_id, label=label, generation_model=generator, phase=phase)
        row["chronological_window"] = "pilot-2026"
        rows.append(row)

    config = _config()
    source = config["sources"][0]
    source.update({
        "revision": "test-revision",
        "file_list": "manifests/source-files.txt",
        "file_list_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "license_name": "Apache-2.0",
        "license_url": "https://example.test/license",
        "license_decision": "allowlisted",
        "underlying_image_license_name": "Apache-2.0",
        "underlying_image_license_url": "https://example.test/output-license",
        "underlying_image_license_decision": "allowlisted",
        "reviewer": "reviewer@example.test",
        "review_date": "2026-08-30",
    })
    manifest = pd.DataFrame(rows)
    selected = select_expansion_rows(manifest, config)
    plan_dir = tmp_path / "plan"
    write_expansion_artifacts(plan_dir, manifest, selected, config)

    features, binary, _ = run_extraction(
        tmp_path,
        tmp_path / "cache",
        index_path=plan_dir / "selection.csv",
        config=ExperimentConfig(
            views=("canonical_128",),
            conditions=("clean",),
            feature_profile="frozen_v1",
            bootstrap_repetitions=2,
            min_group_images=1,
        ),
    )
    tables = write_evaluation_cache(
        tmp_path / "cache",
        features,
        repetitions=2,
        seed=7,
        min_group_images=1,
    )
    assert len(binary) == 4
    assert set(features["chronological_window"]) == {"pilot-2026"}
    assert not tables["chronological_confirmation_metrics"].empty


def test_planned_index_is_blocked_without_positive_license_audit(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    rows = [
        {**_feature_row("discover"), "source_id": "test_source"},
        {
            **_feature_row("confirm", phase="confirmation", generation_model="eval-gen"),
            "source_id": "test_source",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "audit.json").write_text(
        json.dumps({"acquisition_allowed": False}), encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="license audit"):
        run_extraction(tmp_path, tmp_path / "cache", index_path=path)


def test_planned_index_hash_cannot_be_swapped_after_audit(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    rows = [
        {**_feature_row("discover"), "source_id": "test_source"},
        {
            **_feature_row("confirm", phase="confirmation", generation_model="eval-gen"),
            "source_id": "test_source",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "audit.json").write_text(
        json.dumps({"acquisition_allowed": True, "selection_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="SHA-256"):
        run_extraction(tmp_path, tmp_path / "cache", index_path=path)


def test_final_confirmation_requires_opt_in_and_records_evaluation(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in (("discover", "black"), ("confirm", "gray"), ("final", "white")):
        Image.new("RGB", (16, 16), color).save(image_dir / f"{name}.png")
    rows = [
        _feature_row("discover", local_path="images/discover.png"),
        _feature_row(
            "confirm",
            phase="confirmation",
            generation_model="eval-gen",
            local_path="images/confirm.png",
        ),
        {
            **_feature_row(
                "final",
                phase="final_confirmation",
                generation_model="future-gen",
                local_path="images/final.png",
            ),
            "chronological_window": "future-2026",
        },
    ]
    path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    base = ExperimentConfig(views=("canonical_128",), conditions=("clean",))

    withheld_features, withheld_index, withheld_metadata = run_extraction(
        tmp_path, tmp_path / "withheld", index_path=path, config=base
    )
    assert "final_confirmation" not in set(withheld_features["phase"])
    assert "final_confirmation" not in set(withheld_index["phase"])
    assert withheld_metadata["final_confirmation"] == {
        "available_parents": 1,
        "evaluated": False,
        "evaluated_at": None,
        "policy": "Explicit --evaluate-final-confirmation opt-in is required; otherwise sealed rows are withheld from extraction.",
    }
    assert withheld_metadata["eligible_binary_parent_images"] == 3
    assert withheld_metadata["binary_parent_images"] == 2
    assert withheld_metadata["withheld_final_confirmation_images"] == 1
    assert withheld_metadata["excluded_nonbinary_images"] == 0

    opened_features, _, opened_metadata = run_extraction(
        tmp_path,
        tmp_path / "opened",
        index_path=path,
        config=ExperimentConfig(
            views=("canonical_128",),
            conditions=("clean",),
            evaluate_final_confirmation=True,
        ),
    )
    final = opened_features[opened_features["phase"] == "final_confirmation"]
    assert final["chronological_window"].tolist() == ["future-2026"]
    assert opened_metadata["final_confirmation"]["evaluated"] is True
    assert opened_metadata["final_confirmation"]["evaluated_at"]
    receipt = path.with_suffix(path.suffix + ".final-confirmation-evaluated.json")
    assert receipt.is_file()
    with pytest.raises(PermissionError, match="first-evaluation receipt"):
        run_extraction(
            tmp_path,
            tmp_path / "repeat",
            index_path=path,
            config=ExperimentConfig(
                views=("canonical_128",),
                conditions=("clean",),
                evaluate_final_confirmation=True,
            ),
        )


def _config(*, final_generators: list[str] | None = None) -> dict:
    return {
        "schema_version": 1,
        "selection_seed": "test-seed",
        "acquisition": {
            "default_mode": "dry_run",
            "require_all_selected_sources_allowlisted": True,
        },
        "final_confirmation": {
            "phase": "final_confirmation",
            "sealed": True,
            "assignment_required_before_selection": True,
            "generator_ids": final_generators or [],
            "first_evaluated_at": None,
        },
        "sources": [
            {
                "source_id": "test_source",
                "dataset": "test",
                "purpose": "test",
                "selected": True,
                "source_url": "https://example.test/data",
                "revision": None,
                "file_list": None,
                "file_list_sha256": None,
                "content_sha256": None,
                "license_name": None,
                "license_url": None,
                "license_decision": "pending",
                "underlying_image_license_name": None,
                "underlying_image_license_url": None,
                "underlying_image_license_decision": "pending",
                "reviewer": None,
                "review_date": None,
            }
        ],
    }


def _row(
    parent_id: str,
    *,
    label: str = "full_synthetic",
    generation_model: str = "generator-a",
    source_dataset: str = "source-a",
    phase: str = "discovery",
) -> dict[str, object]:
    return {
        "source_id": "test_source",
        "parent_id": parent_id,
        "dataset": "test",
        "split": phase,
        "label": label,
        "generation_model": generation_model,
        "generator_family": "diffusion" if label not in {"real", "authentic"} else "authentic",
        "source_dataset": source_dataset,
        "source_member": f"archive/{parent_id}.png",
        "local_path": f"data/{parent_id}.png",
        "width": 16,
        "height": 16,
        "format": "PNG",
        "bytes": 80,
        "revision": "test-revision",
        "phase": phase,
    }



def _feature_row(
    parent_id: str,
    *,
    label: str = "full_synthetic",
    target: int = 1,
    phase: str = "discovery",
    generation_model: str = "generator-a",
    source_dataset: str = "source-a",
    local_path: str | None = None,
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "target": target,
        "phase": phase,
        "dataset": "expansion-test",
        "split": phase,
        "label": label,
        "generator_family": "diffusion" if target else "authentic",
        "generation_model": generation_model,
        "source_dataset": source_dataset,
        "source_member": f"archive/{parent_id}.png",
        "local_path": local_path or f"images/{parent_id}.png",
        "width": 16,
        "height": 16,
        "mode": "RGB",
        "format": "PNG",
        "bytes": 80,
    }
