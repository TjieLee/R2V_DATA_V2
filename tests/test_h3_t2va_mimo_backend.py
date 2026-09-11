import base64
import json

import pytest

from r2v_data_v2.h3.mimo25_backend import MIMO25_CANONICAL_ABSENT_SOUNDSCAPE
from r2v_data_v2.h3.t2va_mimo_backend import (
    T2VA_SYSTEM_PROMPT,
    T2VAMimoBackend,
    T2VAMimoConfig,
)
from r2v_data_v2.h3.t2va_shadow import T2VAMimoDraft
from tests import test_h3_t2va_shadow as shadow_tests
from tests.test_h3_t2va_shadow import Client, config, draft_for

job = shadow_tests.job


@pytest.mark.parametrize("transport", ["sglang", "xiaomi"])
def test_exactly_one_original_av_request(job, tmp_path, transport):
    cfg = config(tmp_path)
    cfg = T2VAMimoConfig(**{**cfg.__dict__, "transport": transport})
    client = Client([draft_for(job).model_dump_json()])
    backend = T2VAMimoBackend(cfg, client=client)
    raw = backend.annotate(job, "a" * 64)
    assert raw.model_call_count == 1 and raw.error is None and len(client.calls) == 1
    request = client.calls[0]
    assert request["temperature"] == 0 and request["stream"] is False
    assert request["extra_body"]["use_audio_in_video"] is True
    content = request["messages"][1]["content"]
    assert [c["type"] for c in content] == ["video_url", "text"]
    assert content[0]["fps"] == 4.0
    assert content[0]["media_resolution"] == "default"
    assert set(content[0]["video_url"]) == {"url"}
    url = content[0]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"original AV fixture"
    contract = json.loads(content[1]["text"])
    assert contract["speech_facts"] == [
        {
            "segment_id": s.segment_id,
            "source_speaker_cluster": s.source_speaker_cluster,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "language": s.language,
            "has_transcript": s.text is not None,
        }
        for s in job.speech_facts
    ]
    assert contract["dialogue_segment_ids"] == [s.segment_id for s in job.speech_facts]
    assert all(
        s.text not in json.dumps(request, ensure_ascii=False) for s in job.speech_facts
    )
    for banned in (
        "entity_id",
        "reference_subjects",
        "reference_images",
        "audio_url",
        "input_audio",
        "Picture",
        "Subject",
    ):
        assert banned not in json.dumps(request)
    if transport == "sglang":
        assert request["response_format"]["json_schema"]["strict"] is True
        assert (
            request["response_format"]["json_schema"]["schema"]
            == T2VAMimoDraft.model_json_schema()
        )
        assert request["reasoning_effort"] == "none"
        assert request["extra_body"]["chat_template_kwargs"] == {
            "thinking": False,
            "enable_thinking": False,
        }
    else:
        assert request["response_format"] == {"type": "json_object"}
        assert request["extra_body"]["thinking"] == {"type": "disabled"}
    assert backend.provenance().maximum_attempts == 1


def test_no_retry_on_transport_error(job, tmp_path):
    client = Client([RuntimeError("synthetic timeout")])
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert len(client.calls) == raw.model_call_count == 1
    assert "timeout" in raw.error


def test_prompt_adapts_existing_quality_and_authority_without_staged_concepts():
    for phrase in (
        "composition/framing",
        "foreground/background",
        "lighting/color",
        "gaze/expression",
        "early-to-middle-to-late",
        "Prefer observable description over plot interpretation",
        "Original target AV is factual authority",
        "exact and authoritative",
        "Never paraphrase, translate, correct, normalize, invent, drop or reorder",
        "stable S1/S2/",
        "visual presence alone never assigns a speaker",
        "Acoustic grouping is evidence, not visual identity",
        "offscreen",
        "BEFORE every corresponding <d>",
        "non-verbal human sound",
        "audience-only BGM",
        "N/A",
        MIMO25_CANONICAL_ABSENT_SOUNDSCAPE,
    ):
        assert phrase in T2VA_SYSTEM_PROMPT
    for phrase in (
        "Turn 1",
        "Turn 2",
        "Turn 3",
        "Turn 4",
        "entity_id",
        "<Picture",
        "<Subject",
        "retention",
        "reference_subjects",
        "speech stem",
        "music stem",
        "sfx stem",
    ):
        assert phrase not in T2VA_SYSTEM_PROMPT


def test_sdk_retries_disabled(tmp_path, monkeypatch):
    import openai

    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", construct)
    T2VAMimoBackend(config(tmp_path))
    assert captured["max_retries"] == 0


def test_typed_response_schema_and_one_call_exact_dialogue(job, tmp_path):
    from r2v_data_v2.h3.t2va_shadow import render_t2va_prompt, validate_t2va_draft

    schema = T2VAMimoDraft.model_json_schema()
    assert "integrated_multimodal_description" not in schema["properties"]
    assert (
        schema["properties"]["integrated_sequence"]["items"]["discriminator"][
            "propertyName"
        ]
        == "kind"
    )
    speech = schema["$defs"]["T2VASpeechPart"]
    assert set(speech["properties"]) == {"kind", "segment_id", "lead_in"}
    assert speech["additionalProperties"] is False
    assert "Reproducing them is NOT your task" in T2VA_SYSTEM_PROMPT
    assert (
        "Do NOT return ASR words, translations, transliterations" in T2VA_SYSTEM_PROMPT
    )
    client = Client([draft_for(job).model_dump_json()])
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    core = validate_t2va_draft(job, T2VAMimoDraft.model_validate_json(raw.response))
    assert "<d>[Chinese] 你好。</d>" in render_t2va_prompt(core)
    assert raw.model_call_count == len(client.calls) == 1


def test_request_separates_all_segments_from_dialogue_slots(job, tmp_path):
    from r2v_data_v2.h3.t2va_shadow import render_t2va_prompt, validate_t2va_draft

    job.speech_facts[0].text = job.speech_facts[0].language = None
    before = job.model_dump()
    backend = T2VAMimoBackend(config(tmp_path), client=Client([]))
    contract = json.loads(
        backend.build_request(job)["messages"][1]["content"][1]["text"]
    )
    assert [s["segment_id"] for s in contract["speech_facts"]] == [
        "segment_1",
        "segment_2",
    ]
    assert [s["has_transcript"] for s in contract["speech_facts"]] == [False, True]
    assert contract["dialogue_segment_ids"] == ["segment_2"]
    assert all("text" not in s for s in contract["speech_facts"])
    assert job.model_dump() == before
    draft = draft_for(job)
    core = validate_t2va_draft(job, draft)
    assert len(core.speaker_assignments) == 2
    assert "(S2) <d>[Chinese] 好的！</d>" in render_t2va_prompt(core)
    draft.integrated_sequence[-1].segment_id = "segment_1"
    with pytest.raises(ValueError, match="speech placement inventory"):
        validate_t2va_draft(job, draft)
    job.speech_facts[1].text = job.speech_facts[1].language = None
    empty = json.loads(backend.build_request(job)["messages"][1]["content"][1]["text"])
    assert empty["dialogue_segment_ids"] == []
    assert len(empty["speech_facts"]) == 2


def test_v3_prompt_inventory_and_positive_audio_contract():
    for rule in (
        "speaker_assignments = every supplied segment",
        "Speech slots = dialogue_segment_ids only",
        "Non-transcribed segments still receive speaker assignments, but NEVER speech slots",
        "neither S1 nor (S1) belongs there",
        "Listen across the ENTIRE embedded audio from beginning to end",
        "ambience / room tone / outdoor background",
        "physical movement/contact/impact",
        "mechanical/electronic sounds",
        "non-verbal human sounds",
        "excluding dialogue, singing and music",
        "Only when none of those soundscape layers are actually discernible",
        "Preserve clearly audible music",
        "Use N/A only when no such music is established",
    ):
        assert rule in T2VA_SYSTEM_PROMPT


@pytest.mark.parametrize("problem", ["length", "audio_zero", "video_zero"])
def test_incomplete_av_response_fails_once(job, tmp_path, problem):
    client = Client([draft_for(job).model_dump_json()])
    original = client.create

    def create(**kwargs):
        response = original(**kwargs)
        if problem == "length":
            response["choices"][0]["finish_reason"] = "length"
        else:
            response["usage"]["prompt_tokens_details"][
                problem.replace("_zero", "_tokens")
            ] = 0
        return response

    client.chat.completions.create = create
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert raw.error and raw.model_call_count == len(client.calls) == 1
