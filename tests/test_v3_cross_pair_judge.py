from __future__ import annotations

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.cross_pair_judge import (
    SYSTEM_PROMPT,
    CrossPairJudgeFailure,
    QwenCrossPairJudge,
    build_cross_pair_request_payload,
)
from r2v_data_v2.v3.schemas import AnnotationEntity, RawCrossPairDecision


def _entity(entity_id: str, phrase: str) -> AnnotationEntity:
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type="subject",
        phrase=phrase,
        grounding_prompt=f"the visible {phrase}",
    )


def _payload(*, accept: bool = True, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "verdict": "accept" if accept else "reject",
        "same_physical_entity": accept,
        "identity_features_match": accept,
        "reference_is_usable": accept,
        "reason": "Visible identity evidence is consistent.",
    }
    result.update(updates)
    return result


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"reason": ""}, "reason"),
        ({"same_physical_entity": 1}, "valid boolean"),
        ({"extra": "forbidden"}, "Extra inputs"),
        (
            {"verdict": "accept", "same_physical_entity": False},
            "if and only if",
        ),
        (
            {
                "verdict": "reject",
                "same_physical_entity": True,
                "identity_features_match": True,
                "reference_is_usable": True,
            },
            "if and only if",
        ),
    ],
)
def test_cross_pair_decision_schema_is_strict(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        RawCrossPairDecision.model_validate(_payload(**updates))


def test_cross_pair_payload_keeps_target_and_donor_semantics_separate() -> None:
    payload = build_cross_pair_request_payload(
        target_clip_uid="target-clip",
        target_entity=_entity("e1", "target woman"),
        donor_clip_uid="donor-clip",
        donor_entity=_entity("e2", "donor woman"),
    )
    assert payload == {
        "target": {
            "clip_uid": "target-clip",
            "entity_id": "e1",
            "reference_type": "subject",
            "phrase": "target woman",
            "grounding_prompt": "the visible target woman",
        },
        "donor": {
            "clip_uid": "donor-clip",
            "entity_id": "e2",
            "phrase": "donor woman",
            "grounding_prompt": "the visible donor woman",
        },
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
) -> QwenCrossPairJudge:
    return QwenCrossPairJudge(
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


def _decide(judge: QwenCrossPairJudge):
    donor = Image.new("RGBA", (3, 2), (40, 50, 60, 255))
    donor.putpixel((0, 0), (1, 2, 3, 0))
    return judge.decide(
        target_clip_uid="target-clip",
        target_entity=_entity("e1", "target woman"),
        target_context_image=Image.new("RGB", (6, 4), (10, 20, 30)),
        target_entity_crop=Image.new("RGBA", (3, 2), (70, 80, 90, 255)),
        donor_clip_uid="donor-clip",
        donor_entity=_entity("e2", "donor woman"),
        donor_reference_image=donor,
    )


def test_messages_send_three_ordered_images_and_white_donor_background() -> None:
    completions = _Completions([_payload()])
    attempt = _decide(_judge(completions))

    assert attempt.decision.verdict == "accept"
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    messages = call["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    content = messages[1]["content"]
    labels = [item["text"] for item in content if item["type"] == "text"]
    assert labels[1:] == [
        "Target context frame",
        "Target entity crop",
        "Donor source-faithful reference",
    ]
    image_urls = [
        item["image_url"]["url"]
        for item in content
        if item["type"] == "image_url"
    ]
    assert len(image_urls) == 3
    donor_bytes = base64.b64decode(image_urls[-1].split(",", 1)[1])
    with Image.open(BytesIO(donor_bytes)) as donor:
        assert donor.mode == "RGB"
        assert donor.getpixel((0, 0)) == (255, 255, 255)


def test_structured_repair_reuses_schema_and_fails_closed() -> None:
    completions = _Completions(
        [
            _payload(same_physical_entity=False),
            _payload(),
        ]
    )
    attempt = _decide(_judge(completions))

    assert attempt.repair_attempts == 1
    repair_text = completions.calls[1]["messages"][1]["content"][0]["text"]
    assert "Original request" in repair_text
    assert "JSON Schema" in repair_text
    assert "if and only if" in repair_text

    failed = _judge(_Completions(["not json"]), repair_retries=0)
    with pytest.raises(CrossPairJudgeFailure) as error:
        _decide(failed)
    assert error.value.issues[0].code == "invalid_json"
    assert error.value.attempt_count == 1


def test_json_schema_bad_request_falls_back_only_to_json_object() -> None:
    completions = _Completions([_payload()], strict_failure=True)
    attempt = _decide(_judge(completions, repair_retries=0))

    assert attempt.decision.verdict == "accept"
    assert [
        call["response_format"]["type"] for call in completions.calls
    ] == ["json_schema", "json_object"]


def test_prompt_requires_visual_identity_and_uncertain_rejection() -> None:
    required = (
        "same physical entity",
        "Matching category does not prove matching identity",
        "Similar clothing, color",
        "Reject whenever identity is uncertain",
        "Never accept from text alone",
        "donor is not a faithful usable reference",
        "never infer an identity relationship that is not visible",
    )
    assert all(fragment in SYSTEM_PROMPT for fragment in required)
