from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.pair as pair_module
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.pair import EntityReferenceCandidate
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.reference_judge import (
    COMPACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    EntityReferenceJudgeFailure,
    QwenEntityReferenceJudge,
    build_entity_reference_request_payload,
    build_paired_candidate_evidence_card,
    subject_has_nontrivial_detached_component,
    validate_entity_reference_decision,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    EntityReferenceState,
    RawEntityReferenceDecision,
)


def _entity() -> AnnotationEntity:
    return AnnotationEntity(
        entity_id="e1",
        reference_type="subject",
        phrase="a woman in a yellow coat",
        grounding_prompt="the woman wearing a yellow coat",
    )


def _candidate(
    candidate_id: str = "candidate_1",
    *,
    significant_component_count: int = 1,
    largest_component_ratio: float = 1.0,
    second_largest_component_ratio: float = 0.0,
) -> EntityReferenceCandidate:
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True
    return EntityReferenceCandidate(
        candidate_id=candidate_id,
        entity_id="e1",
        frame_slot=5,
        source_frame_index=50,
        image_path="clips/clip-1/frames/05.jpg",
        mask=mask,
        bbox_xyxy=(2, 1, 5, 4),
        area_pixels=9,
        area_ratio=0.3,
        bbox_fill_ratio=1.0,
        border_contact_count=0,
        normalized_center_distance=0.05,
        significant_component_count=significant_component_count,
        largest_component_ratio=largest_component_ratio,
        second_largest_component_ratio=second_largest_component_ratio,
    )


def _candidate_with_mask(mask: np.ndarray) -> EntityReferenceCandidate:
    rows, columns = np.nonzero(mask)
    diagnostics = pair_module.mask_component_diagnostics(mask)
    return EntityReferenceCandidate(
        candidate_id="candidate_1",
        entity_id="e1",
        frame_slot=5,
        source_frame_index=50,
        image_path="clips/clip-1/frames/05.jpg",
        mask=mask,
        bbox_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        area_pixels=int(np.count_nonzero(mask)),
        area_ratio=float(np.mean(mask)),
        bbox_fill_ratio=1.0,
        border_contact_count=0,
        normalized_center_distance=0.05,
        significant_component_count=diagnostics.significant_component_count,
        largest_component_ratio=diagnostics.largest_component_ratio,
        second_largest_component_ratio=(
            diagnostics.second_largest_component_ratio
        ),
    )


def _mask_with_main(
    *,
    shape: tuple[int, int] = (180, 180),
    main_bbox: tuple[int, int, int, int] = (20, 20, 100, 100),
    secondary_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = main_bbox
    mask[y1:y2, x1:x2] = True
    if secondary_bbox is not None:
        x1, y1, x2, y2 = secondary_bbox
        mask[y1:y2, x1:x2] = True
    return mask


def _payload(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "selected_candidate_id": "candidate_1",
        "image_quality": "high",
        "completeness": "complete",
        "reference_scope": "full",
        "visible_region": "whole",
        "whole_entity_recognizable": True,
        "identity_features_visible": True,
        "viewpoint": "front",
        "independent_reference_value": True,
        "requires_substantial_invention": False,
        "primary_identity_region_visible": True,
        "major_structure_visible": True,
        "truncation_severity": "none",
        "discrete_foreground_instance": True,
        "mask_matches_target": True,
        "completion_needed_for_reference_use": False,
        "detached_target_fragments_present": False,
        "scope_reason": "The whole entity and identity are visible.",
    }
    result.update(updates)
    return result


def _source_images() -> dict[str, Image.Image]:
    pixels = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    return {"clips/clip-1/frames/05.jpg": Image.fromarray(pixels, mode="RGB")}


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"scope_reason": ""}, "scope_reason"),
        ({"selected_candidate_id": "candidate_0"}, "candidate_N"),
        ({"selected_candidate_id": None}, "requires selected"),
        (
            {
                "reference_scope": "reject",
                "selected_candidate_id": "candidate_1",
            },
            "must not select",
        ),
        ({"extra": "forbidden"}, "Extra inputs"),
        ({"whole_entity_recognizable": 1}, "valid boolean"),
    ],
)
def test_raw_decision_schema_is_strict(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        RawEntityReferenceDecision.model_validate(_payload(**updates))


@pytest.mark.parametrize(
    "field",
    [
        "viewpoint",
        "independent_reference_value",
        "requires_substantial_invention",
        "primary_identity_region_visible",
        "major_structure_visible",
        "truncation_severity",
        "discrete_foreground_instance",
        "mask_matches_target",
        "completion_needed_for_reference_use",
        "detached_target_fragments_present",
    ],
)
def test_new_reference_gate_fields_are_required(field: str) -> None:
    payload = _payload()
    payload.pop(field)

    with pytest.raises(ValidationError, match=field):
        RawEntityReferenceDecision.model_validate(payload)


def test_reject_decision_contract_is_valid() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            image_quality="acceptable",
            completeness="fragmented",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            scope_reason="Identity is occluded.",
        )
    )
    assert (
        validate_entity_reference_decision(
            decision,
            candidate_ids={"candidate_1"},
            reference_type="subject",
        )
        == []
    )


@pytest.mark.parametrize(
    ("completeness", "scope", "selected"),
    [
        ("complete", "full", "candidate_1"),
        ("repairable", "local", "candidate_1"),
        ("local_usable", "local", "candidate_1"),
        ("severely_incomplete", "reject", None),
        ("fragmented", "reject", None),
    ],
)
def test_completeness_routes_to_deterministic_reference_scope(
    completeness: str,
    scope: str,
    selected: str | None,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness=completeness,
            reference_scope=scope,
            selected_candidate_id=selected,
            visible_region=(
                "whole"
                if scope == "full"
                else "central"
                if scope == "local"
                else "custom"
            ),
            whole_entity_recognizable=scope == "full",
            identity_features_visible=scope != "reject",
            truncation_severity=("minor" if completeness == "repairable" else "none"),
            completion_needed_for_reference_use=(completeness == "repairable"),
        )
    )

    assert decision.completeness == completeness
    assert decision.reference_scope == scope


def test_poor_image_quality_fails_closed() -> None:
    with pytest.raises(ValidationError, match="poor image quality"):
        RawEntityReferenceDecision.model_validate(_payload(image_quality="poor"))


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"selected_candidate_id": "candidate_2"}, "unknown_candidate_id"),
        ({"visible_region": "upper_body"}, "full_requires_whole"),
        ({"whole_entity_recognizable": False}, "full_requires_recognizable"),
        ({"identity_features_visible": False}, "full_requires_identity"),
        (
            {
                "completeness": "repairable",
                "reference_scope": "local",
                "visible_region": "whole",
                "whole_entity_recognizable": False,
                "completion_needed_for_reference_use": True,
            },
            "local_must_be_non_whole",
        ),
        (
            {
                "completeness": "repairable",
                "reference_scope": "local",
                "visible_region": "central",
                "whole_entity_recognizable": False,
                "identity_features_visible": False,
                "truncation_severity": "minor",
                "completion_needed_for_reference_use": True,
            },
            "local_requires_identity",
        ),
    ],
)
def test_request_level_semantic_validation(
    updates: dict[str, object],
    code: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(_payload(**updates))
    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )
    assert code in {issue.code for issue in issues}


def test_recognizable_local_reference_is_valid() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
        )
    )
    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


@pytest.mark.parametrize("viewpoint", ["front", "three_quarter"])
def test_identity_bearing_subject_viewpoints_are_accepted(viewpoint: str) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(viewpoint=viewpoint)
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


def test_rear_subject_must_reject() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(viewpoint="rear")
    )

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )

    assert {issue.code for issue in issues} == {"rear_subject_not_rejected"}


@pytest.mark.parametrize("identity_visible", [True, False])
def test_side_subject_requires_explicit_identity(identity_visible: bool) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(identity_features_visible=identity_visible, viewpoint="side")
    )
    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )

    codes = {issue.code for issue in issues}
    if identity_visible:
        assert codes == set()
    else:
        assert "side_subject_requires_identity" in codes
        assert "full_requires_identity" in codes


@pytest.mark.parametrize("reference_type", ["object", "group"])
def test_non_subject_viewpoint_is_not_applicable(reference_type: str) -> None:
    accepted = RawEntityReferenceDecision.model_validate(
        _payload(viewpoint="not_applicable")
    )
    invalid = RawEntityReferenceDecision.model_validate(_payload(viewpoint="front"))

    assert validate_entity_reference_decision(
        accepted,
        candidate_ids={"candidate_1"},
        reference_type=reference_type,
    ) == []
    assert "non_subject_viewpoint_not_applicable" in {
        issue.code
        for issue in validate_entity_reference_decision(
            invalid,
            candidate_ids={"candidate_1"},
            reference_type=reference_type,
        )
    }


def test_no_independent_reference_value_must_reject() -> None:
    with pytest.raises(ValidationError, match="independent value"):
        RawEntityReferenceDecision.model_validate(
            _payload(independent_reference_value=False)
        )
    rejected = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            completeness="fragmented",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            independent_reference_value=False,
            scope_reason="The crop is only a wall fragment.",
        )
    )
    assert rejected.selected_candidate_id is None


def test_substantial_invention_cannot_route_to_repairable() -> None:
    with pytest.raises(ValidationError, match="substantial invention"):
        RawEntityReferenceDecision.model_validate(
            _payload(
                completeness="repairable",
                reference_scope="local",
                visible_region="central",
                whole_entity_recognizable=False,
                requires_substantial_invention=True,
                truncation_severity="minor",
            )
        )


def test_minor_crop_can_route_to_repairable() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
            requires_substantial_invention=False,
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            scope_reason="Only one small limb terminal is missing.",
        )
    )
    assert decision.completeness == "repairable"


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        (
            {"primary_identity_region_visible": False},
            "primary_identity_region_not_visible",
        ),
        ({"major_structure_visible": False}, "major_structure_not_visible"),
        (
            {"discrete_foreground_instance": False},
            "non_discrete_foreground_instance",
        ),
        ({"mask_matches_target": False}, "mask_target_mismatch"),
        (
            {
                "completeness": "repairable",
                "reference_scope": "local",
                "visible_region": "central",
                "whole_entity_recognizable": True,
                "truncation_severity": "major",
                "completion_needed_for_reference_use": True,
            },
            "major_truncation_not_rejected",
        ),
        ({"truncation_severity": "minor"}, "complete_truncation_mismatch"),
        ({"truncation_severity": "major"}, "complete_truncation_mismatch"),
    ],
)
def test_objective_evidence_fails_closed(
    updates: dict[str, object],
    expected_code: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(_payload(**updates))

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )

    assert expected_code in {issue.code for issue in issues}


def test_repairable_minor_and_complete_none_are_valid() -> None:
    repairable = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
        )
    )
    complete = RawEntityReferenceDecision.model_validate(_payload())

    for decision in (repairable, complete):
        assert validate_entity_reference_decision(
            decision,
            candidate_ids={"candidate_1"},
            reference_type="subject",
        ) == []


@pytest.mark.parametrize(
    (
        "significant_component_count",
        "largest_component_ratio",
        "second_largest_component_ratio",
        "expected",
    ),
    [
        (1, 1.0, 0.0, False),
        (2, 0.95, 0.03, False),
        (2, 0.88, 0.10, True),
    ],
)
def test_subject_detached_component_signal_uses_nontrivial_gray_zone(
    significant_component_count: int,
    largest_component_ratio: float,
    second_largest_component_ratio: float,
    expected: bool,
) -> None:
    candidate = _candidate(
        significant_component_count=significant_component_count,
        largest_component_ratio=largest_component_ratio,
        second_largest_component_ratio=second_largest_component_ratio,
    )

    assert subject_has_nontrivial_detached_component(candidate) is expected


@pytest.mark.parametrize(
    ("secondary_bbox", "expected"),
    [
        (None, False),
        ((130, 120, 142, 128), True),
        ((101, 50, 113, 56), False),
        ((130, 120, 135, 125), False),
        ((130, 120, 138, 128), False),
    ],
)
def test_small_spatial_subject_component_signal(
    secondary_bbox: tuple[int, int, int, int] | None,
    expected: bool,
) -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(secondary_bbox=secondary_bbox)
    )

    assert subject_has_nontrivial_detached_component(candidate) is expected


def test_far_component_below_half_percent_does_not_trigger() -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(
            shape=(260, 260),
            main_bbox=(20, 20, 200, 140),
            secondary_bbox=(225, 180, 237, 186),
        )
    )

    assert candidate.second_largest_component_ratio < 0.005
    assert subject_has_nontrivial_detached_component(candidate) is False


def test_two_percent_component_with_short_bbox_does_not_trigger() -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(
            shape=(140, 140),
            main_bbox=(20, 20, 76, 76),
            secondary_bbox=(100, 100, 108, 108),
        )
    )

    assert candidate.second_largest_component_ratio == pytest.approx(
        64 / (56 * 56 + 64)
    )
    assert subject_has_nontrivial_detached_component(candidate) is False


@pytest.mark.parametrize(
    "secondary_bbox",
    [
        (130, 120, 142, 128),
        (125, 115, 130, 145),
    ],
)
def test_one_to_three_percent_spatially_detached_limb_triggers(
    secondary_bbox: tuple[int, int, int, int],
) -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(secondary_bbox=secondary_bbox)
    )

    assert 0.01 <= candidate.second_largest_component_ratio <= 0.03
    assert subject_has_nontrivial_detached_component(candidate) is True


def test_small_spatial_signal_blocks_local_but_allows_repairable() -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(secondary_bbox=(130, 120, 142, 128))
    )
    local = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="local_usable",
            reference_scope="local",
            visible_region="upper_body",
        )
    )
    repairable = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="upper_body",
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            detached_target_fragments_present=True,
        )
    )
    candidate_by_id = {candidate.candidate_id: candidate}

    local_issues = validate_entity_reference_decision(
        local,
        candidate_ids=set(candidate_by_id),
        reference_type="subject",
        candidate_by_id=candidate_by_id,
    )

    assert "local_usable_has_detached_subject_fragments" in {
        issue.code for issue in local_issues
    }
    assert validate_entity_reference_decision(
        repairable,
        candidate_ids=set(candidate_by_id),
        reference_type="subject",
        candidate_by_id=candidate_by_id,
    ) == []


@pytest.mark.parametrize(
    ("completeness", "expected_code"),
    [
        ("complete", "complete_has_detached_target_fragments"),
        ("local_usable", "local_usable_has_detached_target_fragments"),
    ],
)
def test_non_repair_routes_reject_declared_detached_target_fragments(
    completeness: str,
    expected_code: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness=completeness,
            reference_scope="full" if completeness == "complete" else "local",
            visible_region="whole" if completeness == "complete" else "upper_body",
            detached_target_fragments_present=True,
        )
    )

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )

    assert expected_code in {issue.code for issue in issues}


def test_repairable_route_accepts_limited_detached_target_fragments() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="upper_body",
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            detached_target_fragments_present=True,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


def test_selected_subject_fragment_signal_blocks_local_usable() -> None:
    candidate = _candidate(
        significant_component_count=2,
        largest_component_ratio=0.88,
        second_largest_component_ratio=0.10,
    )
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="local_usable",
            reference_scope="local",
            visible_region="upper_body",
        )
    )

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={candidate.candidate_id},
        reference_type="subject",
        candidate_by_id={candidate.candidate_id: candidate},
    )

    assert "local_usable_has_detached_subject_fragments" in {
        issue.code for issue in issues
    }


def test_selected_subject_fragment_signal_blocks_complete() -> None:
    candidate = _candidate(
        significant_component_count=2,
        largest_component_ratio=0.88,
        second_largest_component_ratio=0.10,
    )
    decision = RawEntityReferenceDecision.model_validate(_payload())

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={candidate.candidate_id},
        reference_type="subject",
        candidate_by_id={candidate.candidate_id: candidate},
    )

    assert "complete_has_detached_subject_fragments" in {
        issue.code for issue in issues
    }


def test_selected_subject_fragment_signal_allows_repairable_route() -> None:
    candidate = _candidate(
        significant_component_count=2,
        largest_component_ratio=0.88,
        second_largest_component_ratio=0.10,
    )
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="upper_body",
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            detached_target_fragments_present=True,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={candidate.candidate_id},
        reference_type="subject",
        candidate_by_id={candidate.candidate_id: candidate},
    ) == []


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        (
            {"completion_needed_for_reference_use": True},
            "complete_must_not_need_completion",
        ),
        (
            {
                "completeness": "local_usable",
                "reference_scope": "local",
                "visible_region": "upper_body",
                "whole_entity_recognizable": True,
                "completion_needed_for_reference_use": True,
            },
            "local_usable_must_not_need_completion",
        ),
        (
            {
                "completeness": "repairable",
                "reference_scope": "local",
                "visible_region": "upper_body",
                "whole_entity_recognizable": True,
                "truncation_severity": "minor",
                "completion_needed_for_reference_use": False,
            },
            "repairable_requires_completion",
        ),
    ],
)
def test_completion_necessity_matches_routing(
    updates: dict[str, object],
    expected_code: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(_payload(**updates))

    issues = validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    )

    assert expected_code in {issue.code for issue in issues}


@pytest.mark.parametrize("truncation_severity", ["none", "minor"])
def test_local_usable_does_not_require_completion(
    truncation_severity: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="local_usable",
            reference_scope="local",
            visible_region="upper_body",
            whole_entity_recognizable=True,
            truncation_severity=truncation_severity,
            completion_needed_for_reference_use=False,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


@pytest.mark.parametrize(
    ("visible_region", "viewpoint", "scope_reason"),
    [
        ("head_shoulders", "front", "Clear head-and-shoulders portrait."),
        ("upper_body", "front", "Clear waist-up portrait."),
        (
            "upper_body",
            "front",
            "Face, head, and coherent upper body are clear; legs are out of frame.",
        ),
        ("upper_body", "three_quarter", "Clear three-quarter upper-body view."),
        ("side", "side", "Clear side upper-body view with visible identity."),
    ],
)
def test_natural_local_subject_framing_is_local_usable(
    visible_region: str,
    viewpoint: str,
    scope_reason: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="local_usable",
            reference_scope="local",
            visible_region=visible_region,
            whole_entity_recognizable=True,
            viewpoint=viewpoint,
            truncation_severity="minor",
            completion_needed_for_reference_use=False,
            scope_reason=scope_reason,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


@pytest.mark.parametrize(
    ("reference_type", "scope_reason"),
    [
        ("subject", "A clipped hand terminal needs a small safe repair."),
        ("object", "A clipped object edge needs a small safe repair."),
    ],
)
def test_small_real_defect_can_route_to_repairable(
    reference_type: str,
    scope_reason: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
            viewpoint=("front" if reference_type == "subject" else "not_applicable"),
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            scope_reason=scope_reason,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type=reference_type,
    ) == []


@pytest.mark.parametrize(
    "scope_reason",
    [
        "A clear upper body remains with detached trouser and leg pieces.",
        "A clear upper body remains with one small detached hand.",
    ],
)
def test_limited_detached_subject_parts_route_to_repairable(
    scope_reason: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            completeness="repairable",
            reference_scope="local",
            visible_region="upper_body",
            truncation_severity="minor",
            completion_needed_for_reference_use=True,
            detached_target_fragments_present=True,
            scope_reason=scope_reason,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


@pytest.mark.parametrize(
    ("viewpoint", "scope_reason"),
    [
        ("front", "The head is missing."),
        ("rear", "Only a rear view is visible."),
        ("front", "Only a torso without identity is visible."),
        ("front", "Most of the body structure is missing."),
    ],
)
def test_major_subject_loss_remains_rejected(
    viewpoint: str,
    scope_reason: str,
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            completeness="severely_incomplete",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            viewpoint=viewpoint,
            primary_identity_region_visible=False,
            major_structure_visible=False,
            truncation_severity="major",
            completion_needed_for_reference_use=False,
            scope_reason=scope_reason,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


@pytest.mark.parametrize(
    "updates",
    [
        {
            "discrete_foreground_instance": False,
            "scope_reason": "The crop is an environment fragment.",
        },
        {
            "mask_matches_target": False,
            "scope_reason": "The snorkel mask is dominated by unrelated equipment.",
        },
        {
            "primary_identity_region_visible": False,
            "major_structure_visible": False,
            "truncation_severity": "major",
            "scope_reason": "Only a torso and back remain visible.",
        },
    ],
)
def test_rejected_objective_evidence_is_representable(
    updates: dict[str, object],
) -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            completeness="severely_incomplete",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            **updates,
        )
    )

    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
        reference_type="subject",
    ) == []


def test_missing_head_and_most_body_routes_to_severe_rejection() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            completeness="severely_incomplete",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            requires_substantial_invention=True,
            scope_reason="The head and most of the body are missing.",
        )
    )
    assert decision.reference_scope == "reject"


def test_legacy_reference_state_without_new_judge_fields_loads() -> None:
    legacy = EntityReferenceState.model_validate(
        {
            "entity_id": "e1",
            "status": "ready",
            "reference_scope": "full",
            "visible_region": "whole",
            "whole_entity_recognizable": True,
            "identity_features_visible": True,
            "scope_reason": "legacy ready reference",
            "image_path": "clips/clip-1/selected/e1.png",
            "source_frame_index": 5,
        }
    )
    assert legacy.viewpoint is None
    assert legacy.independent_reference_value is None
    assert legacy.requires_substantial_invention is None
    assert legacy.primary_identity_region_visible is None
    assert legacy.major_structure_visible is None
    assert legacy.truncation_severity is None
    assert legacy.discrete_foreground_instance is None
    assert legacy.mask_matches_target is None
    assert legacy.completion_needed_for_reference_use is None
    assert legacy.detached_target_fragments_present is None


@pytest.mark.parametrize("status", ["ready", "rejected"])
def test_reference_state_persists_new_judge_fields(status: str) -> None:
    payload: dict[str, object] = {
        "entity_id": "e1",
        "status": status,
        "reference_scope": "full" if status == "ready" else "reject",
        "visible_region": "whole" if status == "ready" else "custom",
        "whole_entity_recognizable": status == "ready",
        "identity_features_visible": status == "ready",
        "scope_reason": "strict reference decision",
        "viewpoint": "front",
        "independent_reference_value": status == "ready",
        "requires_substantial_invention": status == "rejected",
        "primary_identity_region_visible": status == "ready",
        "major_structure_visible": status == "ready",
        "truncation_severity": "none" if status == "ready" else "major",
        "discrete_foreground_instance": status == "ready",
        "mask_matches_target": status == "ready",
        "completion_needed_for_reference_use": False,
        "detached_target_fragments_present": status == "rejected",
    }
    if status == "ready":
        payload.update(
            image_path="clips/clip-1/selected/e1.png",
            source_frame_index=5,
        )
    state = EntityReferenceState.model_validate(payload)
    restored = EntityReferenceState.model_validate(state.model_dump(mode="json"))
    assert restored.viewpoint == "front"
    assert restored.independent_reference_value is (status == "ready")
    assert restored.requires_substantial_invention is (status == "rejected")
    assert restored.primary_identity_region_visible is (status == "ready")
    assert restored.major_structure_visible is (status == "ready")
    assert restored.truncation_severity == (
        "none" if status == "ready" else "major"
    )
    assert restored.discrete_foreground_instance is (status == "ready")
    assert restored.mask_matches_target is (status == "ready")
    assert restored.completion_needed_for_reference_use is False
    assert restored.detached_target_fragments_present is (status == "rejected")


def test_prompt_distinguishes_repairable_from_stable_local_views() -> None:
    lowered = " ".join(SYSTEM_PROMPT.lower().split())
    for phrase in (
        "torso",
        "hips",
        "buttocks",
        "legs",
        "arms",
        "single connected component is not evidence",
        "main object structure",
        "natural local framing",
        "head-and-shoulders",
        "upper-body",
        "whole body is not visible",
        "completion_needed_for_reference_use",
        "detached target fragments",
        "nontrivial_detached_component_signal",
        "minor truncation does not automatically mean repairable",
    ):
        assert phrase in lowered


def test_request_payload_contains_only_required_evidence() -> None:
    payload = build_entity_reference_request_payload(
        _entity(),
        [_candidate()],
    )
    assert payload == {
        "entity_id": "e1",
        "reference_type": "subject",
        "phrase": "a woman in a yellow coat",
        "grounding_prompt": "the woman wearing a yellow coat",
        "candidates": [
            {
                "candidate_id": "candidate_1",
                "frame_slot": 5,
                "source_frame_index": 50,
                "area_ratio": 0.3,
                "bbox_fill_ratio": 1.0,
                "border_contact_count": 0,
                "sharpness_score": 0.0,
                "significant_component_count": 1,
                "largest_component_ratio": 1.0,
                "second_largest_component_ratio": 0.0,
                "nontrivial_detached_component_signal": False,
            }
        ],
    }


@pytest.mark.parametrize("non_subject_type", ["object", "group"])
def test_request_payload_applies_fragment_signal_only_to_subject(
    non_subject_type: str,
) -> None:
    candidate = _candidate_with_mask(
        _mask_with_main(secondary_bbox=(130, 120, 142, 128))
    )

    subject_payload = build_entity_reference_request_payload(_entity(), [candidate])
    non_subject_payload = build_entity_reference_request_payload(
        _entity().model_copy(update={"reference_type": non_subject_type}),
        [candidate],
    )

    assert subject_payload["candidates"][0][
        "nontrivial_detached_component_signal"
    ] is True
    assert non_subject_payload["candidates"][0][
        "nontrivial_detached_component_signal"
    ] is False


def test_subject_fragment_signal_enters_existing_structured_repair_lifecycle() -> None:
    candidate = _candidate(
        significant_component_count=2,
        largest_component_ratio=0.88,
        second_largest_component_ratio=0.10,
    )
    completions = _Completions(
        [
            _payload(
                completeness="local_usable",
                reference_scope="local",
                visible_region="upper_body",
            ),
            _payload(
                completeness="repairable",
                reference_scope="local",
                visible_region="upper_body",
                truncation_severity="minor",
                completion_needed_for_reference_use=True,
                detached_target_fragments_present=True,
            ),
        ]
    )
    judge = _judge(completions)

    attempt = judge.decide(
        entity=_entity(),
        candidates=[candidate],
        source_images=_source_images(),
    )

    assert attempt.repair_attempts == 1
    assert attempt.decision.completeness == "repairable"
    assert attempt.decision.detached_target_fragments_present is True
    repair_text = completions.calls[1]["messages"][1]["content"][0]["text"]
    assert "local_usable_has_detached_subject_fragments" in repair_text


class _Completions:
    def __init__(
        self,
        responses: list[dict[str, object] | str],
        *,
        strict_failure: bool = False,
    ) -> None:
        self.responses = iter(responses)
        self.strict_failure = strict_failure
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.strict_failure and len(self.calls) == 1:
            raise BadRequestError(
                "json_schema unsupported",
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST",
                        "http://127.0.0.1:8000/v1/chat/completions",
                    ),
                ),
                body={},
            )
        response = next(self.responses)
        raw = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )


def _judge(
    completions: _Completions,
    *,
    repair_retries: int = 1,
    evidence_mode: str = "separate",
    card_panel_max_side: int = 512,
    system_prompt: str = SYSTEM_PROMPT,
    prompt_mode: str = "baseline",
) -> QwenEntityReferenceJudge:
    return QwenEntityReferenceJudge(
        QwenServiceConfig(
            model="local-qwen-vl",
            temperature=0.0,
            max_tokens=500,
        ),
        repair_retries=repair_retries,
        evidence_mode=evidence_mode,
        card_panel_max_side=card_panel_max_side,
        system_prompt=system_prompt,
        prompt_mode=prompt_mode,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


def test_messages_use_ordered_in_memory_context_and_crop_data_urls() -> None:
    completions = _Completions([_payload()])
    judge = _judge(completions)

    judge.decide(
        entity=_entity(),
        candidates=[_candidate()],
        source_images=_source_images(),
    )

    call = completions.calls[0]
    messages = call["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    content = messages[1]["content"]
    labels = [item["text"] for item in content if item["type"] == "text"]
    assert labels[1:] == [
        "Candidate candidate_1 context",
        "Candidate candidate_1 isolated crop",
    ]
    images = [
        item["image_url"]["url"] for item in content if item["type"] == "image_url"
    ]
    assert len(images) == 2
    for data_url in images:
        assert data_url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(BytesIO(decoded)) as image:
            assert image.format == "PNG"
    assert call["response_format"]["type"] == "json_schema"
    assert "video_url" not in json.dumps(messages)


def test_compact_prompt_keeps_separate_images_payload_schema_and_validation() -> None:
    candidates = [_candidate(f"candidate_{index}") for index in range(1, 4)]
    baseline_completions = _Completions([_payload()])
    compact_completions = _Completions([_payload()])
    baseline_judge = _judge(baseline_completions)
    compact_judge = _judge(
        compact_completions,
        system_prompt=COMPACT_SYSTEM_PROMPT,
        prompt_mode="compact_v1",
    )

    baseline = baseline_judge.decide(
        entity=_entity(),
        candidates=candidates,
        source_images=_source_images(),
    )
    compact = compact_judge.decide(
        entity=_entity(),
        candidates=candidates,
        source_images=_source_images(),
    )

    baseline_call = baseline_completions.calls[0]
    compact_call = compact_completions.calls[0]
    baseline_content = baseline_call["messages"][1]["content"]
    compact_content = compact_call["messages"][1]["content"]
    assert baseline_call["messages"][0]["content"] == SYSTEM_PROMPT
    assert compact_call["messages"][0]["content"] == COMPACT_SYSTEM_PROMPT
    assert baseline_content[0]["text"] == compact_content[0]["text"]
    assert [
        item["text"] for item in baseline_content[1:] if item["type"] == "text"
    ] == [item["text"] for item in compact_content[1:] if item["type"] == "text"]
    assert sum(item["type"] == "image_url" for item in compact_content) == 6
    baseline_schema = baseline_call["response_format"]["json_schema"]["schema"]
    compact_schema = compact_call["response_format"]["json_schema"]["schema"]
    assert baseline_schema == compact_schema
    assert compact_schema == (
        RawEntityReferenceDecision.model_json_schema()
    )
    assert baseline.decision == compact.decision
    assert validate_entity_reference_decision(
        compact.decision,
        candidate_ids={candidate.candidate_id for candidate in candidates},
        reference_type="subject",
        candidate_by_id={candidate.candidate_id: candidate for candidate in candidates},
    ) == []


def test_compact_prompt_preserves_required_semantics_and_is_shorter() -> None:
    normalized = " ".join(COMPACT_SYSTEM_PROMPT.casefold().split())
    for fragment in (
        "visible evidence only",
        "reject a rear subject",
        "local_usable",
        "repairable",
        "severely_incomplete",
        "fragmented",
        "major truncation",
        "independent reference value",
        "substantial invention",
        "detached fragments",
        "nontrivial_detached_component_signal",
        "context image shows scene placement",
        "isolated crop shows the actual content",
    ):
        assert fragment in normalized
    assert len(COMPACT_SYSTEM_PROMPT) < len(SYSTEM_PROMPT) * 0.60


def test_paired_candidate_card_is_bounded_rgb_and_preserves_sources() -> None:
    context = Image.new("RGB", (800, 400), (20, 40, 60))
    isolated = Image.new("RGBA", (100, 200), (80, 100, 120, 170))
    context_before = (context.mode, context.size, context.tobytes())
    isolated_before = (isolated.mode, isolated.size, isolated.tobytes())

    card = build_paired_candidate_evidence_card(
        context,
        isolated,
        panel_max_side=384,
    )

    assert card.mode == "RGB"
    assert card.size == (768, 384)
    assert (context.mode, context.size, context.tobytes()) == context_before
    assert (isolated.mode, isolated.size, isolated.tobytes()) == isolated_before
    pixels = np.asarray(card)
    left_nonwhite = np.argwhere(np.any(pixels[:, :384] != 255, axis=2))
    right_nonwhite = np.argwhere(np.any(pixels[:, 384:] != 255, axis=2))
    assert tuple(np.ptp(left_nonwhite, axis=0) + 1) == (192, 384)
    assert tuple(np.ptp(right_nonwhite, axis=0) + 1) == (200, 100)


@pytest.mark.parametrize(
    ("evidence_mode", "expected_image_count"),
    [("separate", 6), ("paired_card", 3)],
)
def test_three_candidates_keep_order_with_selected_evidence_presentation(
    evidence_mode: str,
    expected_image_count: int,
) -> None:
    completions = _Completions([_payload()])
    judge = _judge(
        completions,
        evidence_mode=evidence_mode,
        card_panel_max_side=384,
    )
    candidates = [_candidate(f"candidate_{index}") for index in range(1, 4)]

    attempt = judge.decide(
        entity=_entity(),
        candidates=candidates,
        source_images=_source_images(),
    )

    assert attempt.decision.selected_candidate_id == "candidate_1"
    call = completions.calls[0]
    content = call["messages"][1]["content"]
    request_text = content[0]["text"]
    expected_payload = build_entity_reference_request_payload(
        _entity(),
        candidates,
    )
    assert request_text.endswith(json.dumps(expected_payload, ensure_ascii=False))
    labels = [item["text"] for item in content if item["type"] == "text"][1:]
    assert [label.split()[1] for label in labels] == (
        [
            candidate_id
            for candidate_id in ("candidate_1", "candidate_2", "candidate_3")
            for _ in range(2)
        ]
        if evidence_mode == "separate"
        else ["candidate_1", "candidate_2", "candidate_3"]
    )
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == expected_image_count
    if evidence_mode == "paired_card":
        assert all("left panel = scene context" in label for label in labels)
        for item in images:
            encoded = item["image_url"]["url"].split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as card:
                assert card.mode == "RGB"
                assert card.size == (768, 384)
    assert call["response_format"]["json_schema"]["schema"] == (
        RawEntityReferenceDecision.model_json_schema()
    )


def test_structured_repair_includes_original_schema_issues_and_response() -> None:
    completions = _Completions(
        [
            _payload(visible_region="upper_body"),
            _payload(),
        ]
    )
    judge = _judge(completions)

    attempt = judge.decide(
        entity=_entity(),
        candidates=[_candidate()],
        source_images=_source_images(),
    )

    assert attempt.repair_attempts == 1
    assert len(attempt.raw_responses) == 2
    repair_messages = completions.calls[1]["messages"]
    repair_text = repair_messages[1]["content"][0]["text"]
    assert "Original request" in repair_text
    assert "JSON Schema" in repair_text
    assert "full_requires_whole" in repair_text
    assert "upper_body" in repair_text


def test_candidate_judge_repair_profiles_payload_shape_without_mutation(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    entity = _entity()
    payload_before = build_entity_reference_request_payload(entity, [candidate])
    completions = _Completions(
        [
            _payload(visible_region="upper_body"),
            _payload(),
        ]
    )
    judge = _judge(completions)
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")

    with active_profiler(profiler):
        attempt = judge.decide(
            entity=entity,
            candidates=[candidate],
            source_images=_source_images(),
        )

    assert attempt.repair_attempts == 1
    assert build_entity_reference_request_payload(entity, [candidate]) == payload_before
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["component"] for event in events] == [
        "qwen_candidate_judge",
        "qwen_candidate_judge",
    ]
    assert [event["retry_index"] for event in events] == [0, 1]
    assert all(event["input_image_count"] == 2 for event in events)
    assert all(
        event["metadata"]
        | {
            "candidate_count": 1,
            "context_image_count": 1,
            "isolated_crop_count": 1,
        }
        == event["metadata"]
        for event in events
    )
    assert all(event["metadata"]["evidence_mode"] == "separate" for event in events)
    assert all(event["metadata"]["paired_card_count"] == 0 for event in events)
    assert all(event["metadata"]["prompt_mode"] == "baseline" for event in events)


def test_compact_prompt_profiles_mode_with_six_images(tmp_path: Path) -> None:
    completions = _Completions([_payload()])
    judge = _judge(
        completions,
        system_prompt=COMPACT_SYSTEM_PROMPT,
        prompt_mode="compact_v1",
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")

    with active_profiler(profiler):
        judge.decide(
            entity=_entity(),
            candidates=[
                _candidate(f"candidate_{index}") for index in range(1, 4)
            ],
            source_images=_source_images(),
        )

    event = json.loads(
        profiler.events_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert event["input_image_count"] == 6
    assert event["metadata"]["candidate_count"] == 3
    assert event["metadata"]["evidence_mode"] == "separate"
    assert event["metadata"]["prompt_mode"] == "compact_v1"


def test_paired_candidate_card_profiles_three_images_and_mode(tmp_path: Path) -> None:
    completions = _Completions([_payload()])
    judge = _judge(
        completions,
        evidence_mode="paired_card",
        card_panel_max_side=384,
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")

    with active_profiler(profiler):
        judge.decide(
            entity=_entity(),
            candidates=[
                _candidate(f"candidate_{index}") for index in range(1, 4)
            ],
            source_images=_source_images(),
        )

    event = json.loads(
        profiler.events_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert event["input_image_count"] == 3
    assert event["metadata"] | {
        "candidate_count": 3,
        "context_image_count": 3,
        "isolated_crop_count": 3,
        "paired_card_count": 3,
        "evidence_mode": "paired_card",
        "card_panel_max_side": 384,
    } == event["metadata"]


def test_repair_exhaustion_fails_closed_with_all_raw_responses() -> None:
    completions = _Completions(
        [
            _payload(selected_candidate_id="candidate_2"),
            "not json",
        ]
    )
    judge = _judge(completions)

    with pytest.raises(EntityReferenceJudgeFailure) as error:
        judge.decide(
            entity=_entity(),
            candidates=[_candidate()],
            source_images=_source_images(),
        )

    assert error.value.attempt_count == 2
    assert len(error.value.raw_responses) == 2
    assert error.value.issues[0].code == "invalid_json"


def test_bad_request_falls_back_to_json_object_without_network() -> None:
    completions = _Completions([_payload()], strict_failure=True)
    judge = _judge(completions, repair_retries=0)

    result = judge.decide(
        entity=_entity(),
        candidates=[_candidate()],
        source_images=_source_images(),
    )

    assert result.decision.reference_scope == "full"
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
    ]


@pytest.mark.parametrize(
    "candidate_ids",
    [
        [],
        [_candidate("candidate_2")],
        [_candidate(), _candidate("candidate_3")],
    ],
)
def test_candidate_ids_must_be_nonempty_contiguous_and_ordered(
    candidate_ids: list[EntityReferenceCandidate],
) -> None:
    judge = _judge(_Completions([_payload()]))
    with pytest.raises(ValueError, match="requires candidates|contiguous"):
        judge.decide(
            entity=_entity(),
            candidates=candidate_ids,
            source_images=_source_images(),
        )


def test_system_prompt_reserves_semantics_for_qwen_not_geometry() -> None:
    required = (
        "Candidate IDs and candidate order are immutable",
        "never infer",
        "do not downgrade only because of border contact",
        "Local means",
        "Reject when",
        "Do not output tokens or crop coordinates",
    )
    assert all(fragment in SYSTEM_PROMPT for fragment in required)
