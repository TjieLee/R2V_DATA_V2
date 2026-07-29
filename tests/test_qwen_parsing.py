from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from prompts.qwen_annotation_prompt import SYSTEM_PROMPT
from r2v_data_v2.caption_validation import validate_annotation
from r2v_data_v2.config import PipelineConfig, QwenConfig, QwenVideoConfig
from r2v_data_v2.qwen_client import (
    QwenAnnotationClient,
    QwenAnnotationFailure,
    _sanitize_final_structure,
    _video_content,
    _video_processor_extra_body,
    annotate_manifest,
)
from r2v_data_v2.schemas import QwenAnnotationResult
from r2v_data_v2.structured_output import parse_qwen_json_response
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


def test_qwen_prompt_and_content_use_complete_video(tmp_path: Path) -> None:
    normalized_prompt = " ".join(SYSTEM_PROMPT.split())
    assert "complete video" in normalized_prompt
    assert "ten frames" not in normalized_prompt
    assert (
        "background.phrase must be copied as one exact, contiguous substring"
        in normalized_prompt
    )
    assert (
        "It must not combine separated parts of the caption."
        in normalized_prompt
    )
    video_path = tmp_path / "clip.mp4"

    content = _video_content(video_path)

    assert content == [
        {
            "type": "video_url",
            "video_url": {"url": video_path.resolve().as_uri()},
        }
    ]
    assert not any(item["type"] == "image_url" for item in content)


def test_qwen_video_processor_parameters_omit_null_pixel_budgets() -> None:
    default = _video_processor_extra_body(QwenConfig())
    assert default == {
        "mm_processor_kwargs": {
            "fps": 2.0,
            "do_sample_frames": True,
        }
    }
    configured = _video_processor_extra_body(
        QwenConfig(
            video=QwenVideoConfig(
                fps=3.5,
                do_sample_frames=False,
                max_pixels=640_000,
                total_pixels=2_560_000,
            )
        )
    )
    assert configured["mm_processor_kwargs"] == {
        "fps": 3.5,
        "do_sample_frames": False,
        "max_pixels": 640_000,
        "total_pixels": 2_560_000,
    }


class _FakeAnnotationClient(QwenAnnotationClient):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.config = QwenConfig(repair_retries=1)
        self._responses: Iterator[str | Exception] = iter(responses)
        self.requests: list[list[dict[str, object]]] = []

    def _request(self, messages: list[dict[str, object]]) -> str:
        assert messages
        self.requests.append(messages)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_one_repair_can_succeed() -> None:
    client = _FakeAnnotationClient(
        ["not json", json.dumps(_valid_payload())],
    )
    video_path = Path("/read-only/video.mp4")
    result, _ = client.annotate(
        video_path=video_path,
        caption_raw="draft",
        metadata={},
    )
    assert result.entities[0].entity_id == "e1"
    for messages in client.requests:
        user_content = messages[-1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0] == {
            "type": "video_url",
            "video_url": {"url": video_path.as_uri()},
        }


def test_two_failed_responses_are_preserved() -> None:
    client = _FakeAnnotationClient(["not json", '{"caption": "still invalid"}'])
    with pytest.raises(QwenAnnotationFailure) as caught:
        client.annotate(
            video_path=Path("/read-only/video.mp4"),
            caption_raw="draft",
            metadata={},
        )
    assert caught.value.raw_responses == [
        "not json",
        '{"caption": "still invalid"}',
    ]
    assert caught.value.issues


def test_repair_request_failure_preserves_first_raw_response() -> None:
    client = _FakeAnnotationClient(["not json", RuntimeError("service unavailable")])
    with pytest.raises(QwenAnnotationFailure) as caught:
        client.annotate(
            video_path=Path("/read-only/video.mp4"),
            caption_raw="draft",
            metadata={},
        )
    assert caught.value.attempt_count == 2
    assert caught.value.raw_responses == ["not json"]
    assert caught.value.issues[0].code == "qwen_request_failed"


def test_final_structure_sanitizer_aligns_deferred_background_phrase() -> None:
    caption = (
        "The camera reveals a deep blue ocean shimmering under bright sunlight, "
        "bordered by arid, rocky hills."
    )
    source_phrase = (
        "a vast expanse of deep blue ocean bordered by barren, rocky hills"
    )
    semantic = QwenAnnotationResult.model_validate(
        {
            "caption": caption,
            "entities": [],
            "relations": [],
            "background": {
                "phrase": source_phrase,
                "grounding_prompt": source_phrase,
                "reference_worthy": False,
            },
        }
    )

    sanitized, warnings = _sanitize_final_structure(semantic)

    assert sanitized.background is not None
    assert sanitized.background.phrase == "deep blue ocean"
    assert sanitized.background.reference_worthy is False
    assert any(
        warning.startswith("aligned_reference_phrase:background:")
        for warning in warnings
    )


def test_final_structure_sanitizer_demotes_drops_and_caps() -> None:
    caption = (
        "A red bicycle stands beside a blue car while a tall lamp lights a "
        "small bench in a broad city plaza during a calm evening tracking shot."
    )

    def entity(
        entity_id: str,
        phrase: str,
        salience: str,
    ) -> dict[str, object]:
        return {
            "entity_id": entity_id,
            "phrase": phrase,
            "grounding_prompt": phrase,
            "canonical_label": phrase,
            "category": "object",
            "reference_worthy": True,
            "salience": salience,
            "genericity": "descriptive",
            "name_evidence": "none",
            "separability": "independent",
            "selection_reason": "visible",
        }

    semantic = QwenAnnotationResult.model_validate(
        {
            "caption": caption,
            "entities": [
                entity("e1", "A red bicycle", "secondary"),
                entity("e2", "a blue car", "primary"),
                entity("e3", "a tall lamp", "incidental"),
                entity("e4", "a small bench", "primary"),
                entity("e5", "a green boat", "primary"),
            ],
            "relations": [
                {
                    "subject_id": "e2",
                    "predicate": "near",
                    "object_id": "missing",
                }
            ],
            "background": {
                "phrase": "a forest clearing",
                "grounding_prompt": "forest clearing",
                "reference_worthy": True,
            },
        }
    )

    sanitized, warnings = _sanitize_final_structure(semantic)

    retained = [
        item.entity_id for item in sanitized.entities if item.reference_worthy
    ]
    assert retained == ["e1", "e2", "e4"]
    assert sanitized.relations == []
    assert sanitized.background is not None
    assert sanitized.background.reference_worthy is False
    assert "demoted_unalignable_reference:e5" in warnings
    assert "dropped_invalid_relation:0" in warnings
    assert "demoted_reference_cap:e3" in warnings
    assert "demoted_unalignable_reference:background" in warnings
    assert validate_annotation(
        sanitized,
        caption_raw="",
        metadata={},
    ) == []


def test_two_failed_responses_write_complete_failure_log(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    source_manifest = output_root / "manifests" / "source.jsonl"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "video_path": str(video_path),
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "clip_order": [1, 0],
                "caption_raw": "draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
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


def test_qwen_annotation_does_not_require_sampled_frames(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    source_manifest = output_root / "manifests" / "source.jsonl"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "video_path": str(video_path),
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "clip_order": [1, 0],
                "caption_raw": "draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stats = annotate_manifest(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        client=_FakeAnnotationClient([json.dumps(_valid_payload())]),
    )

    assert stats.processed == 1
    assert stats.qwen_failed == 0
    assert not (output_root / "frames").exists()
    assert (output_root / "annotations" / "clip-1.json").is_file()


def test_missing_source_video_is_logged_before_qwen_request(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    source_manifest = output_root / "manifests" / "source.jsonl"
    source_manifest.parent.mkdir(parents=True)
    missing_video = tmp_path / "missing.mp4"
    source_manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "video_path": str(missing_video),
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "clip_order": [1, 0],
                "caption_raw": "draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = _FakeAnnotationClient([])

    stats = annotate_manifest(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        client=client,
    )

    failure = json.loads(
        (output_root / "logs" / "qwen_failed.jsonl").read_text(encoding="utf-8")
    )
    assert stats.qwen_failed == 1
    assert failure["issues"][0]["message"] == (
        f"source video does not exist: {missing_video}"
    )
    assert client.requests == []
