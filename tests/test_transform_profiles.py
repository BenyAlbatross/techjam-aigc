from __future__ import annotations

import json

import numpy as np
from PIL import Image

from techjam_aigc.feature_lab.pipeline import ExperimentConfig
from techjam_aigc.feature_lab.transforms import (
    COLOR_FACTORIAL_SPECS,
    COVERING32_SPECS,
    DIRECTED_PAIR_SPECS,
    REALISTIC_SPECS,
    TRANSFORM_SPECS,
    apply_transform,
    apply_transform_steps,
    deterministic_step_seed,
    get_transform_specs,
    transform_frame,
)


def _image() -> Image.Image:
    y, x = np.mgrid[0:19, 0:23]
    pixels = np.stack((x * 9, y * 11, (x + y) * 5), axis=2).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def test_profile_counts_are_bounded_and_core_is_the_unchanged_default() -> None:
    assert len(TRANSFORM_SPECS) == 20
    assert len(COLOR_FACTORIAL_SPECS) == 12  # six axial + six new corners
    assert len(DIRECTED_PAIR_SPECS) == 30  # all directed pairs among six families
    assert len(REALISTIC_SPECS) == 12
    assert len(COVERING32_SPECS) == 32
    assert len(get_transform_specs("color_factorial")) == 32
    assert len(get_transform_specs("directed_pairs")) == 50
    assert len(get_transform_specs("realistic")) == 32
    assert len(get_transform_specs("covering32")) == 52
    assert len(get_transform_specs("all")) == 106
    assert ExperimentConfig().transform_profile == "core"
    assert ExperimentConfig().conditions is None
    assert get_transform_specs() == TRANSFORM_SPECS
    assert len(get_transform_specs()) < 100  # never expands to the 12,959 grid


def test_color_factorial_has_axial_cases_and_all_corners_without_duplicates() -> None:
    assert sum(spec.design == "color_axial" for spec in COLOR_FACTORIAL_SPECS) == 6
    assert sum(spec.design == "color_corner" for spec in COLOR_FACTORIAL_SPECS) == 6
    color_steps = [
        spec.steps[0].values()
        for spec in get_transform_specs("color_factorial")
        if len(spec.steps) == 1 and spec.steps[0].operation == "color_jitter"
    ]
    corners = {
        (p["brightness"], p["contrast"], p["saturation"])
        for p in color_steps
        if {p["brightness"], p["contrast"], p["saturation"]} <= {0.8, 1.2}
    }
    assert len(corners) == 8


def test_directed_pairs_cover_both_orders_and_record_order() -> None:
    operation_pairs = [tuple(step.operation for step in spec.steps) for spec in DIRECTED_PAIR_SPECS]
    assert len(operation_pairs) == len(set(operation_pairs)) == 30
    assert all((second, first) in operation_pairs for first, second in operation_pairs)
    frame = transform_frame("directed_pairs").query("design == 'directed_medium_pair'")
    assert len(frame) == 30
    assert frame["step_count"].eq(2).all()
    assert frame["ordered_operations"].str.count(">").eq(1).all()
    assert frame["name"].str.contains("__then__", regex=False).all()


def test_realistic_set_contains_preregistered_chains() -> None:
    orders = {tuple(step.operation for step in spec.steps) for spec in REALISTIC_SPECS}
    assert ("center_crop", "resize", "jpeg") in orders
    assert ("color_jitter", "resize", "jpeg") in orders
    assert ("gaussian_blur", "gaussian_noise", "jpeg") in orders
    assert ("resize", "unsharp_mask", "jpeg") in orders
    assert ("jpeg", "resize", "jpeg") in orders


def test_composed_conditions_are_deterministic_per_parent_and_preserve_geometry() -> None:
    image = _image()
    for spec in (*COLOR_FACTORIAL_SPECS, *DIRECTED_PAIR_SPECS, *REALISTIC_SPECS, *COVERING32_SPECS):
        first = apply_transform(image, spec.name, parent_id="stable-parent", base_seed=17)
        second = apply_transform(image, spec.name, parent_id="stable-parent", base_seed=17)
        assert first.mode == "RGB"
        assert first.size == image.size
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_registry_encodes_reproducibility_metadata() -> None:
    frame = transform_frame("all")
    assert frame["name"].is_unique
    assert frame["recipe_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    noise_rows = frame[frame["ordered_operations"].str.contains("gaussian_noise")]
    assert noise_rows["seed_policy_json"].str.contains("parent_id", regex=False).all()
    for recipe in frame["ordered_recipe_json"]:
        assert isinstance(json.loads(recipe), list)
    assert len(frame.query("design == 'halton_covering_bank'")) == 32


def test_reversed_noise_pair_uses_the_same_operation_level_draw() -> None:
    pairs = [
        spec
        for spec in DIRECTED_PAIR_SPECS
        if {step.operation for step in spec.steps} == {"jpeg", "gaussian_noise"}
    ]
    assert len(pairs) == 2
    noise_steps = [
        next(step for step in spec.steps if step.operation == "gaussian_noise")
        for spec in pairs
    ]
    seeds = [
        deterministic_step_seed("same-parent", step, 0, 17)
        for step in noise_steps
    ]
    assert seeds[0] == seeds[1]


def test_public_step_pipeline_preserves_sequential_order_and_noise_replay() -> None:
    spec = next(
        item
        for item in DIRECTED_PAIR_SPECS
        if tuple(step.operation for step in item.steps) == ("gaussian_noise", "jpeg")
    )
    image = _image()

    direct = apply_transform_steps(
        image,
        spec.steps,
        parent_id="stable-parent",
        base_seed=17,
    )
    registered = apply_transform(
        image,
        spec.name,
        parent_id="stable-parent",
        base_seed=17,
    )
    replay = apply_transform_steps(
        image,
        spec.steps,
        parent_id="stable-parent",
        base_seed=17,
    )

    np.testing.assert_array_equal(np.asarray(direct), np.asarray(registered))
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(replay))
