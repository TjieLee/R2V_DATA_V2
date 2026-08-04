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
        "reference_scope": "full",
        "visible_region": "whole",
        "whole_entity_recognizable": True,
        "identity_features_visible": True,
        "scope_reason": "The whole entity and identity are visible.",
    }
    result.update(updates)
    return result


def _source_images() -> dict[str, Image.Image]:
    pixels = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    return {
        "clips/clip-1/frames/05.jpg": Image.fromarray(pixels, mode="RGB")
    }


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


def test_reject_decision_contract_is_valid() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            selected_candidate_id=None,
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            scope_reason="Identity is occluded.",
        )
    )
    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
    ) == []


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"selected_candidate_id": "candidate_2"}, "unknown_candidate_id"),
        ({"visible_region": "upper_body"}, "full_requires_whole"),
        ({"whole_entity_recognizable": False}, "full_requires_recognizable"),
        ({"identity_features_visible": False}, "full_requires_identity"),
        (
            {
                "reference_scope": "local",
                "visible_region": "whole",
                "whole_entity_recognizable": False,
            },
            "local_must_be_non_whole",
        ),
        (
            {
                "reference_scope": "local",
                "visible_region": "central",
                "whole_entity_recognizable": False,
                "identity_features_visible": False,
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
    )
    assert code in {issue.code for issue in issues}


def test_recognizable_local_reference_is_valid() -> None:
    decision = RawEntityReferenceDecision.model_validate(
        _payload(
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
        )
    )
    assert validate_entity_reference_decision(
        decision,
        candidate_ids={"candidate_1"},
    ) == []


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
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
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
    images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
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
    assert [
        call["response_format"]["type"] for call in completions.calls
    ] == ["json_schema", "json_object"]


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
