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
from r2v_data_v2.v3.removal_judge import (
    FULL_FRAME_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    QwenBackgroundRemovalJudge,
)


def _payload(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "verdict": "accept",
        "foreground_absent": True,
        "foreground_not_reconstructed": True,
        "no_new_salient_entity": True,
        "background_only_in_repaired_region": True,
        "background_continuity_ok": True,
        "no_visible_artifacts": True,
        "reason": "All visible checks pass.",
    }
    value.update(updates)
    return value


class _Completions:
    def __init__(
        self,
        response: dict[str, object] | str,
        *,
        strict_failure: bool = False,
    ) -> None:
        self.response = response
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
        raw = (
            self.response
            if isinstance(self.response, str)
            else json.dumps(self.response)
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )


def _judge(completions: _Completions) -> QwenBackgroundRemovalJudge:
    return QwenBackgroundRemovalJudge(
        QwenServiceConfig(
            model="local-qwen-vl",
            temperature=0.0,
            max_tokens=500,
        ),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )


def _review(
    judge: QwenBackgroundRemovalJudge,
    *,
    candidate_mode: str = "masked_local",
):
    source = Image.new("RGB", (4, 3), (10, 20, 30))
    candidate = Image.new("RGB", (4, 3), (40, 50, 60))
    source_mask = Image.new("L", (4, 3), 0)
    generation_mask = Image.new("L", (4, 3), 255)
    return judge.review(
        source_image=source,
        candidate_image=candidate,
        source_mask=source_mask,
        generation_mask=generation_mask,
        removal_phrases=["person", "bicycle"],
        background_phrase="empty park path",
        candidate_mode=candidate_mode,  # type: ignore[arg-type]
    )


def test_review_uses_four_in_memory_png_data_urls_and_semantics() -> None:
    completions = _Completions(_payload())
    review = _review(_judge(completions))

    assert review.verdict == "accept"
    request = completions.calls[0]
    content = request["messages"][1]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    assert len(image_items) == 4
    for item in image_items:
        url = item["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        with Image.open(BytesIO(decoded)) as opened:
            assert opened.format == "PNG"
            assert opened.size == (4, 3)
    assert "person; bicycle" in content[0]["text"]
    assert "empty park path" in content[0]["text"]


def test_request_uses_temperature_zero_and_strict_schema() -> None:
    completions = _Completions(_payload())

    _review(_judge(completions))

    request = completions.calls[0]
    assert request["temperature"] == 0.0
    assert request["model"] == "local-qwen-vl"
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert set(response_format["json_schema"]["schema"]["properties"]) == {
        "verdict",
        "foreground_absent",
        "foreground_not_reconstructed",
        "no_new_salient_entity",
        "background_only_in_repaired_region",
        "background_continuity_ok",
        "no_visible_artifacts",
        "reason",
    }


def test_response_format_bad_request_falls_back_to_json_object() -> None:
    completions = _Completions(_payload(), strict_failure=True)

    review = _review(_judge(completions))

    assert review.verdict == "accept"
    assert len(completions.calls) == 2
    assert completions.calls[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "\x60\x60\x60json\n{}\n\x60\x60\x60",
        json.dumps({**_payload(), "extra": "forbidden"}),
    ],
)
def test_malformed_or_extra_field_response_fails_strictly(raw: str) -> None:
    with pytest.raises(ValidationError):
        _review(_judge(_Completions(raw)))


@pytest.mark.parametrize(
    "updates",
    [
        {"verdict": "accept", "foreground_absent": False},
        {"verdict": "reject"},
        {"reason": "   "},
        {"foreground_absent": 1},
    ],
)
def test_semantically_contradictory_review_is_not_repaired(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _review(_judge(_Completions(_payload(**updates))))


def test_valid_rejection_is_returned_unchanged() -> None:
    payload = _payload(
        verdict="reject",
        no_visible_artifacts=False,
        reason="A visible seam remains.",
    )

    review = _review(_judge(_Completions(payload)))

    assert review.verdict == "reject"
    assert review.no_visible_artifacts is False
    assert review.reason == "A visible seam remains."


def test_empty_response_fails_explicitly() -> None:
    with pytest.raises(RuntimeError, match="empty background removal review"):
        _review(_judge(_Completions("")))


def test_system_prompt_defines_four_images_and_fail_closed_criteria() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()

    for index in range(1, 5):
        assert f"{index}." in normalized
    assert "judge only visible facts" in normalized
    assert "reject if any original foreground remains" in normalized
    assert "changes outside the generation mask" in normalized
    assert "ghosts, double exposure, seams" in normalized
    assert "accept if and only if every boolean is true" in normalized


def test_legacy_mode_keeps_exact_existing_prompt_and_local_label() -> None:
    completions = _Completions(_payload())

    _review(_judge(completions))

    messages = completions.calls[0]["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    labels = [
        item["text"]
        for item in messages[1]["content"]
        if item["type"] == "text"
    ]
    assert "Image 2: locally composited candidate" in labels


def test_full_frame_mode_uses_registration_tolerant_fail_closed_prompt() -> None:
    completions = _Completions(_payload())

    _review(_judge(completions), candidate_mode="full_frame")

    messages = completions.calls[0]["messages"]
    assert messages[0]["content"] == FULL_FRAME_SYSTEM_PROMPT
    normalized = " ".join(FULL_FRAME_SYSTEM_PROMPT.split()).casefold()
    for allowed in (
        "registration",
        "scale",
        "resampling",
        "translation",
        "photometric",
    ):
        assert allowed in normalized
    for rejection in (
        "original foreground remains",
        "is reconstructed",
        "new person",
        "disappearance or material alteration of unrelated salient entities",
        "material unrelated scene or layout changes",
        "severe background hallucination",
        "severe artifacts",
    ):
        assert rejection in normalized
    assert "reject changes outside the generation mask" not in normalized
    labels = [
        item["text"]
        for item in messages[1]["content"]
        if item["type"] == "text"
    ]
    assert "Image 2: resized full-frame candidate" in labels


def test_unknown_candidate_mode_fails_before_model_call() -> None:
    completions = _Completions(_payload())

    with pytest.raises(ValueError, match="unsupported removal candidate mode"):
        _review(_judge(completions), candidate_mode="other")

    assert completions.calls == []


def test_close_delegates_to_underlying_client() -> None:
    closed: list[bool] = []
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions(_payload())),
        close=lambda: closed.append(True),
    )
    judge = QwenBackgroundRemovalJudge(
        QwenServiceConfig(model="local"),
        client=client,
    )

    judge.close()

    assert closed == [True]
