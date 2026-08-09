from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.scale_collapse_fallback_guard import (
    SYSTEM_PROMPT,
    QwenScaleCollapseFallbackJudge,
)


def _review(*, accepted: bool = True) -> dict[str, object]:
    return {
        "verdict": "accept" if accepted else "reject",
        "identity_or_primary_region_visible": accepted,
        "coherent_structure": accepted,
        "independent_reference_value": accepted,
        "not_severely_fragmented": accepted,
        "reason": "usable local reference" if accepted else "fragmented torso",
    }


class _Completions:
    def __init__(self, responses: list[str | dict[str, object]]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        response = next(self.responses)
        raw = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )


def _judge(completions: _Completions) -> QwenScaleCollapseFallbackJudge:
    return QwenScaleCollapseFallbackJudge(
        QwenServiceConfig(model="local-qwen"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


def test_scale_collapse_prompt_is_short() -> None:
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", SYSTEM_PROMPT)
    assert len(words) <= 120


def test_scale_collapse_guard_sends_exactly_one_source_image_and_minimal_text(
    tmp_path: Path,
) -> None:
    completions = _Completions([_review()])
    judge = _judge(completions)
    image = Image.new("RGBA", (7, 5), (20, 40, 60, 120))
    profiler = V3Profiler(tmp_path / "profile", git_commit="test")

    with active_profiler(profiler):
        attempt = judge.review(
            image=image,
            reference_type="subject",
            entity_phrase="a seated man",
        )

    assert attempt.review.verdict == "accept"
    call = completions.calls[0]
    content = call["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    assert content[0]["text"] == (
        "Reference type: subject\nEntity phrase: a seated man"
    )
    serialized = json.dumps(call["messages"], ensure_ascii=False)
    for forbidden in (
        "grounding_prompt",
        "candidate alternatives",
        "mask diagnostics",
        "boogu failed candidate",
    ):
        assert forbidden not in serialized.casefold()
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.mode == "RGBA"
        assert decoded.size == image.size
        assert decoded.tobytes() == image.tobytes()
    event = json.loads(
        profiler.events_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert event["component"] == "qwen_scale_collapse_fallback_guard"
    assert event["input_image_count"] == 1
    assert event["metadata"] | {
        "reference_type": "subject",
        "image_count": 1,
        "mode": "qwen_v1",
        "response_format": "json_schema",
    } == event["metadata"]


def test_malformed_scale_collapse_review_is_repaired_once() -> None:
    inconsistent = {**_review(), "coherent_structure": False}
    completions = _Completions([inconsistent, _review()])
    judge = _judge(completions)

    attempt = judge.review(
        image=Image.new("RGB", (6, 6), "white"),
        reference_type="object",
        entity_phrase="an ornate panel",
    )

    assert attempt.review.verdict == "accept"
    assert len(attempt.raw_responses) == 2
    assert len(completions.calls) == 2
    assert completions.calls[1]["messages"][-1]["role"] == "user"
    assert "Repair the JSON" in completions.calls[1]["messages"][-1]["content"]
    assert sum(
        item.get("type") == "image_url"
        for item in completions.calls[1]["messages"][1]["content"]
    ) == 1
