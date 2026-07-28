from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from r2v_data_v2.config import QwenConfig
from r2v_data_v2.pairing import QwenCrossPairJudge
from r2v_data_v2.ranking import QwenCandidateJudge
from r2v_data_v2.schemas import (
    CandidateJudgeResult,
    CrossPairJudgeResult,
    QwenAnnotationResult,
)
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    parse_qwen_json_response,
)
from tests.test_caption_validation import _valid_payload


def _candidate_payload() -> dict[str, object]:
    return {
        "entity_id": "e1",
        "candidates": [
            {
                "frame_slot": 2,
                "completeness": 0.9,
                "recognizability": 0.9,
                "occlusion": 0.1,
                "mask_quality": 0.9,
                "visual_quality": 0.8,
                "identity_features_visible": True,
                "rejection_reasons": [],
            }
        ],
        "best_frame_slot": 2,
    }


def _cross_pair_payload() -> dict[str, object]:
    return {
        "same_exact_instance": "yes",
        "confidence": 0.95,
        "context_difference": "large",
        "near_duplicate": False,
        "conflicting_attributes": [],
        "reason": "matching details in a different context",
    }


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (QwenAnnotationResult, _valid_payload()),
        (CandidateJudgeResult, _candidate_payload()),
        (CrossPairJudgeResult, _cross_pair_payload()),
    ],
)
def test_all_qwen_schemas_share_strict_json_and_fence_parsing(
    model: type[BaseModel],
    payload: dict[str, object],
) -> None:
    raw = json.dumps(payload)
    assert parse_qwen_json_response(raw, model)
    assert parse_qwen_json_response(f"```json\n{raw}\n```", model)
    with pytest.raises(json.JSONDecodeError):
        parse_qwen_json_response(f"Result: {raw}", model)
    payload_with_extra = {**payload, "unexpected": True}
    with pytest.raises(ValidationError):
        parse_qwen_json_response(json.dumps(payload_with_extra), model)


class _FakeCandidateJudge(QwenCandidateJudge):
    def __init__(self, responses: list[str]) -> None:
        self.config = QwenConfig()
        self.responses: Iterator[str] = iter(responses)
        self.requests: list[str] = []

    def _request(self, *, prompt: str, encoded_image: str) -> str:
        assert encoded_image
        self.requests.append(prompt)
        return next(self.responses)


class _FakeCrossPairJudge(QwenCrossPairJudge):
    def __init__(self, responses: list[str]) -> None:
        self.config = QwenConfig()
        self.responses: Iterator[str] = iter(responses)
        self.requests: list[str] = []

    def _request(
        self,
        *,
        prompt: str,
        target: dict[str, object],
        candidate: dict[str, object],
    ) -> str:
        del target, candidate
        self.requests.append(prompt)
        return next(self.responses)


def test_candidate_judge_repairs_once_with_original_image(tmp_path: Path) -> None:
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"image")
    judge = _FakeCandidateJudge(["not json", json.dumps(_candidate_payload())])

    result = judge.review(
        entity_id="e1",
        entity_phrase="gray-haired man",
        contact_sheet=contact_sheet,
        frame_slots=[2],
    )

    assert result.entity_id == "e1"
    assert len(judge.requests) == 2
    assert "Original request:" in judge.requests[1]
    assert "Original invalid response:\nnot json" in judge.requests[1]


def test_candidate_judge_two_failures_preserve_raw_responses(
    tmp_path: Path,
) -> None:
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"image")
    judge = _FakeCandidateJudge(["not json", '{"still": "wrong"}'])

    with pytest.raises(StructuredOutputFailure) as caught:
        judge.review(
            entity_id="e1",
            entity_phrase="gray-haired man",
            contact_sheet=contact_sheet,
            frame_slots=[2],
        )
    assert caught.value.raw_responses == ["not json", '{"still": "wrong"}']


def test_cross_pair_judge_repairs_once() -> None:
    judge = _FakeCrossPairJudge(
        ["not json", f"```json\n{json.dumps(_cross_pair_payload())}\n```"]
    )
    result = judge.judge(
        target={"phrase": "silver watch"},
        candidate={"canonical_label": "watch"},
    )
    assert result.same_exact_instance == "yes"
    assert len(judge.requests) == 2


def test_cross_pair_judge_two_failures_preserve_raw_responses() -> None:
    judge = _FakeCrossPairJudge(["not json", '{"still": "wrong"}'])
    with pytest.raises(StructuredOutputFailure) as caught:
        judge.judge(
            target={"phrase": "silver watch"},
            candidate={"canonical_label": "watch"},
        )
    assert caught.value.raw_responses == ["not json", '{"still": "wrong"}']
