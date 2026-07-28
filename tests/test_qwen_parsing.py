from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.config import PipelineConfig, QwenConfig
from r2v_data_v2.qwen_client import (
    QwenAnnotationClient,
    QwenAnnotationFailure,
    annotate_manifest,
    parse_qwen_json_response,
)
from r2v_data_v2.schemas import QwenAnnotationResult
from tests.test_caption_validation import _valid_payload


def test_strict_json_object_parses() -> None:
    result = parse_qwen_json_response(
        json.dumps(_valid_payload()),
        QwenAnnotationResult,
    )
    assert result.entities[0].entity_id == "e1"


def test_complete_markdown_json_fence_parses() -> None:
    raw = f"```json\n{json.dumps(_valid_payload())}\n```"
    assert parse_qwen_json_response(raw, QwenAnnotationResult).caption


@pytest.mark.parametrize(
    "raw",
    [
        'Here is the JSON: {"caption": "invalid"}',
        "{'caption': 'single quotes are not JSON'}",
    ],
)
def test_explanations_and_python_literals_fail(raw: str) -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_qwen_json_response(raw, QwenAnnotationResult)


def test_extra_schema_field_fails() -> None:
    payload = _valid_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        parse_qwen_json_response(json.dumps(payload), QwenAnnotationResult)


def test_qwen_cannot_supply_tokens_or_final_prompt() -> None:
    payload = _valid_payload()
    payload["prompt_with_refs"] = payload["caption"]
    payload["entities"][0]["ref_token"] = "<ref_subject_99>"  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_qwen_json_response(json.dumps(payload), QwenAnnotationResult)


class _FakeAnnotationClient(QwenAnnotationClient):
    def __init__(self, responses: list[str]) -> None:
        self.config = QwenConfig(repair_retries=1)
        self._responses: Iterator[str] = iter(responses)

    def _request(self, messages: list[dict[str, object]]) -> str:
        assert messages
        return next(self._responses)


def test_one_repair_can_succeed() -> None:
    client = _FakeAnnotationClient(
        ["not json", json.dumps(_valid_payload())],
    )
    result, _ = client.annotate(frame_paths=[], caption_raw="draft", metadata={})
    assert result.entities[0].entity_id == "e1"


def test_two_failed_responses_are_preserved() -> None:
    client = _FakeAnnotationClient(["not json", '{"caption": "still invalid"}'])
    with pytest.raises(QwenAnnotationFailure) as caught:
        client.annotate(frame_paths=[], caption_raw="draft", metadata={})
    assert caught.value.raw_responses == [
        "not json",
        '{"caption": "still invalid"}',
    ]
    assert caught.value.issues


def test_two_failed_responses_write_complete_failure_log(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    source_manifest = output_root / "manifests" / "source.jsonl"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "video_path": "/read-only/video.mp4",
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "clip_order": [1, 0],
                "caption_raw": "draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frame_dir = output_root / "frames" / "clip-1"
    frame_dir.mkdir(parents=True)
    for slot in range(8):
        (frame_dir / f"frame_{slot:02d}.jpg").write_bytes(b"jpeg")

    stats = annotate_manifest(
        PipelineConfig(dataset_json=tmp_path / "source.jsonl", output_root=output_root),
        client=_FakeAnnotationClient(
            ["not json", '{"caption": "still invalid"}'],
        ),
    )

    failure = json.loads(
        (output_root / "logs" / "qwen_failed.jsonl").read_text(encoding="utf-8")
    )
    assert stats.qwen_failed == 1
    assert failure["attempt_count"] == 2
    assert failure["raw_responses"] == [
        "not json",
        '{"caption": "still invalid"}',
    ]
    assert failure["issues"]
    assert not (output_root / "annotations" / "clip-1.json").exists()
    assert (output_root / "manifests" / "annotations.jsonl").read_text(
        encoding="utf-8"
    ) == ""
