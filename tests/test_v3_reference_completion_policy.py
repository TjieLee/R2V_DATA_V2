from __future__ import annotations

import numpy as np
from PIL import Image

from r2v_data_v2.v3.reference_completion_policy import (
    build_conservative_completion_policy,
    restore_protected_source_pixels,
    validate_conservative_completion_candidate,
)


def _rgba(mask: np.ndarray) -> Image.Image:
    pixels = np.full((*mask.shape, 4), 255, dtype=np.uint8)
    rows, columns = np.indices(mask.shape)
    pixels[..., 0] = (40 + columns) % 220
    pixels[..., 1] = (70 + rows) % 220
    pixels[..., 2] = 130
    pixels[..., 3] = np.where(mask, 255, 0)
    pixels[..., :3][~mask] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _coherent_two_component_source() -> Image.Image:
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:50, 35:65] = True
    mask[20:30, 66:76] = True
    return _rgba(mask)


def _valid_directional_candidate_mask() -> tuple[object, Image.Image]:
    policy = build_conservative_completion_policy(
        _coherent_two_component_source(),
        whole_entity_recognizable=True,
    )
    protected = np.asarray(policy.protected_component_mask) == 255
    rows, columns = np.nonzero(protected)
    candidate = protected.copy()
    candidate[
        int(rows.min()) - 1,
        int(columns.min()) : int(columns.max()) + 1,
    ] = True
    return policy, Image.fromarray(candidate.astype(np.uint8) * 255, mode="L")


def test_reasonable_separated_source_components_are_all_protected() -> None:
    source = _coherent_two_component_source()
    source_alpha = np.asarray(source.getchannel("A")) == 255
    policy = build_conservative_completion_policy(
        source,
        whole_entity_recognizable=True,
    )
    cleaned_alpha = np.asarray(policy.cleaned_source_rgba.getchannel("A")) == 255
    protected = np.asarray(policy.protected_component_mask) == 255

    assert policy.eligible is True
    assert policy.recovery_directions == ("top",)
    assert policy.protected_component_ids == ("component_1", "component_2")
    assert np.array_equal(cleaned_alpha, source_alpha)
    assert int(protected.sum()) == int(source_alpha.sum())
    secondary_reasons = policy.source_component_stats[1]["protection_reasons"]
    assert "near_main_component" in secondary_reasons
    assert "minimum_area" not in secondary_reasons


def test_obviously_fragmented_source_skips_completion() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:5, 10:20] = True
    for index in range(6):
        row = 15 + (index // 3) * 15
        column = 10 + (index % 3) * 25
        mask[row : row + 2, column : column + 5] = True
    policy = build_conservative_completion_policy(
        _rgba(mask),
        whole_entity_recognizable=True,
    )

    assert policy.eligible is False
    assert policy.completion_skipped_reason == "completion_source_fragmented"
    assert policy.source_metrics["component_count"] == 7
    assert policy.source_metrics["largest_component_ratio"] < 0.60


def test_candidate_with_far_new_fragments_is_rejected() -> None:
    policy, valid_mask = _valid_directional_candidate_mask()
    candidate = np.asarray(valid_mask).copy()
    candidate[-8:-3, 3:8] = 255
    candidate[-8:-3, -8:-3] = 255

    result = validate_conservative_completion_candidate(
        policy,
        Image.fromarray(candidate, mode="L"),
    )

    assert result.passed is False
    assert "completion_new_disconnected_fragments" in result.reasons
    assert "completion_growth_outside_recovery_corridor" in result.reasons
    stats = result.report
    assert stats["disconnected_significant_component_count"] == 2
    assert stats["outside_recovery_corridor_pixels"] == 50


def test_directional_local_completion_passes_and_restores_source_pixels() -> None:
    policy, candidate_mask = _valid_directional_candidate_mask()
    result = validate_conservative_completion_candidate(policy, candidate_mask)

    assert result.passed is True
    assert result.reasons == ()
    assert result.report["matching_recovery_directions"] == ["top"]
    assert result.report["accepted_new_region_pixels"] > 0

    generated = Image.new("RGB", policy.completion_input_rgb.size, (1, 2, 3))
    restored = restore_protected_source_pixels(policy, generated)
    protected = np.asarray(policy.protected_component_mask) == 255
    restored_pixels = np.asarray(restored)
    baseline_pixels = np.asarray(policy.completion_input_rgb)
    assert np.array_equal(
        restored_pixels[protected],
        baseline_pixels[protected],
    )
    assert int((np.asarray(result.mask) == 255).sum()) > int(protected.sum())


def test_candidate_without_directional_growth_is_rejected() -> None:
    policy = build_conservative_completion_policy(
        _coherent_two_component_source(),
        whole_entity_recognizable=True,
    )
    result = validate_conservative_completion_candidate(
        policy,
        policy.protected_component_mask,
    )

    assert result.passed is False
    assert result.reasons == ("completion_no_directional_improvement",)


def test_candidate_with_excessive_corridor_growth_is_rejected() -> None:
    policy = build_conservative_completion_policy(
        _coherent_two_component_source(),
        whole_entity_recognizable=True,
    )
    protected = np.asarray(policy.protected_component_mask) == 255
    corridor = np.asarray(policy.recovery_corridor_mask) == 255
    candidate = protected | corridor
    result = validate_conservative_completion_candidate(
        policy,
        Image.fromarray(candidate.astype(np.uint8) * 255, mode="L"),
    )

    assert result.passed is False
    assert "completion_abnormal_shape_growth" in result.reasons


def test_unrecognizable_local_source_is_ineligible_without_geometry_loss() -> None:
    source = _coherent_two_component_source()
    policy = build_conservative_completion_policy(
        source,
        whole_entity_recognizable=False,
    )

    assert policy.eligible is False
    assert policy.completion_skipped_reason == (
        "completion_source_not_whole_recognizable"
    )
    assert policy.protected_component_ids == ("component_1", "component_2")
    assert np.array_equal(
        np.asarray(policy.cleaned_source_rgba),
        np.asarray(source),
    )
