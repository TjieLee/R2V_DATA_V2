from __future__ import annotations

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.pair import EntityReferenceCandidate
from r2v_data_v2.v3.reference_judge import (
    SYSTEM_PROMPT,
    EntityReferenceJudgeFailure,
    QwenEntityReferenceJudge,
    build_entity_reference_request_payload,
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


def _candidate(candidate_id: str = "candidate_1") -> EntityReferenceCandidate:
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
    )


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


def test_prompt_distinguishes_repairable_from_stable_local_views() -> None:
    lowered = SYSTEM_PROMPT.lower()
    for phrase in (
        "torso",
        "hips",
        "buttocks",
        "legs",
        "arms",
        "single connected component is not evidence",
        "main object structure",
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
            }
        ],
    }


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
) -> QwenEntityReferenceJudge:
    return QwenEntityReferenceJudge(
        QwenServiceConfig(
            model="local-qwen-vl",
            temperature=0.0,
            max_tokens=500,
        ),
        repair_retries=repair_retries,
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
