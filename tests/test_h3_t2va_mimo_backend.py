import base64
import json

import pytest

from r2v_data_v2.h3.mimo25_backend import (
    AUDIO_FINALIZE_SYSTEM_PROMPT,
    MimoAudioFinalizeDraft,
)
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
def test_semantic_and_frozen_audio_finalize_requests(job, tmp_path, transport):
    cfg = config(tmp_path)
    cfg = T2VAMimoConfig(**{**cfg.__dict__, "transport": transport})
    client = Client([draft_for(job).model_dump_json()])
    backend = T2VAMimoBackend(cfg, client=client)
    raw = backend.annotate(job, "a" * 64)
    assert raw.model_call_count == 2 and raw.error is None and len(client.calls) == 2
    assert raw.semantic_model_call_count == raw.audio_finalize.model_call_count == 1
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
    finalizer = client.calls[1]
    assert finalizer["messages"][0]["content"] == AUDIO_FINALIZE_SYSTEM_PROMPT
    media = finalizer["messages"][1]["content"]
    assert [p["type"] for p in media] == [
        "video_url",
        "text",
        "text",
        "audio_url",
        "text",
        "audio_url",
    ]
    assert media[0] == content[0]
    for index, kind in [(3, "music"), (5, "sfx")]:
        from pathlib import Path

        assert media[index]["audio_url"]["url"] == cfg.media_resolver.resolve(
            Path(getattr(job.audio_evidence, f"{kind}_path"))
        )
    if transport == "sglang":
        assert (
            finalizer["response_format"]["json_schema"]["schema"]
            == MimoAudioFinalizeDraft.model_json_schema()
        )
    else:
        assert "RESPONSE SCHEMA" in media[1]["text"]


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
    from r2v_data_v2.h3.t2va_shadow import render_t2va_prompt

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
    core = shadow_tests.materialize(
        job, T2VAMimoDraft.model_validate_json(raw.response)
    )
    assert "<d>[Chinese] 你好。</d>" in render_t2va_prompt(core)
    assert raw.model_call_count == len(client.calls) == 2


def test_request_separates_all_segments_from_dialogue_slots(job, tmp_path):
    from r2v_data_v2.h3.t2va_shadow import render_t2va_prompt

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
    assert [s["segment_id"] for s in contract["required_speech_slots"]] == ["segment_2"]
    assert all("text" not in s for s in contract["speech_facts"])
    assert job.model_dump() == before
    draft = draft_for(job)
    core = shadow_tests.materialize(job, draft)
    assert len(core.speaker_assignments) == 2
    assert "(S2) <d>[Chinese] 好的！</d>" in render_t2va_prompt(core)
    draft.integrated_sequence[-1].segment_id = "segment_1"
    with pytest.raises(ValueError, match="speech placement inventory"):
        shadow_tests.materialize(job, draft)
    job.speech_facts[1].text = job.speech_facts[1].language = None
    empty = json.loads(backend.build_request(job)["messages"][1]["content"][1]["text"])
    assert empty["dialogue_segment_ids"] == []
    assert empty["required_speech_slots"] == []
    assert len(empty["speech_facts"]) == 2


def test_v4_prompt_preserves_inventory_without_final_audio_fields():
    for rule in (
        "speaker_assignments = every supplied segment",
        "Speech slots = dialogue_segment_ids only",
        "Non-transcribed segments still receive speaker assignments, but NEVER speech slots",
        "neither S1 nor (S1) belongs there",
        "Describe each stable visual fact once",
        "Stop once the observed clip progression has been covered",
    ):
        assert rule in T2VA_SYSTEM_PROMPT
    for field in ("overall_soundscape", "non_diegetic_music"):
        assert field not in T2VA_SYSTEM_PROMPT
        assert field not in T2VAMimoDraft.model_json_schema()["properties"]


def test_required_slots_are_structural_only(job, tmp_path):
    backend = T2VAMimoBackend(config(tmp_path), client=Client([]))
    request = backend.build_request(job)
    contract = json.loads(request["messages"][1]["content"][1]["text"])
    assert contract["required_speech_slots"] == [
        {
            "kind": "speech",
            "segment_id": fact.segment_id,
            "lead_in": "<model writes source/presentation/delivery only>",
        }
        for fact in job.speech_facts
    ]
    assert len(contract["required_speech_slots"]) == 2
    assert backend.build_request(job) == request
    for fact in job.speech_facts:
        assert fact.text not in json.dumps(request, ensure_ascii=False)
    slots = json.dumps(contract["required_speech_slots"])
    for forbidden in ("S1", "S2", "onscreen_spoken", "offscreen_spoken", "voice_over"):
        assert forbidden not in slots
    for rule in (
        "dialogue_segment_ids is a hard structural inventory",
        "NEVER substitutes for a required speech part",
        "returning zero speech parts is invalid",
        "not insertion positions or answers",
    ):
        assert rule in T2VA_SYSTEM_PROMPT
    assert backend.provenance().prompt_version == "h3_t2va_joint_av_v5"
    assert backend.provenance().schema_version == "r2v.h3.t2va_mimo_backend.5"


def test_prose_substitution_fails_before_audio_finalize(job, tmp_path):
    draft = draft_for(job).model_dump()
    draft["integrated_sequence"] = [
        {"kind": "prose", "text": "[Shot 1] The man speaks in Chinese."}
    ]
    client = Client([json.dumps(draft)])
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert "speech placement inventory differs" in raw.error
    assert "possible_prose_substitution_for_required_speech_slots" in raw.error
    assert raw.model_call_count == 1
    assert raw.audio_finalize is None
    assert len(client.calls) == 1


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


@pytest.mark.parametrize("failure", ["not JSON", RuntimeError("finalizer unavailable")])
def test_audio_finalize_failure_has_two_attempts_no_fallback(job, tmp_path, failure):
    client = Client([draft_for(job).model_dump_json()], audio_response=failure)
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert raw.error and raw.model_call_count == len(client.calls) == 2
    assert raw.response and raw.audio_finalize is not None


def test_audio_finalize_supplies_final_audio_and_preserves_semantic_raw(job, tmp_path):
    from r2v_data_v2.h3 import t2va_shadow as t

    payload = draft_for(job).model_dump()
    payload["integrated_sequence"][1]["lead_in"] = "S1 speaks calmly"
    semantic_raw = json.dumps(payload)
    audio = MimoAudioFinalizeDraft(
        overall_soundscape="Footsteps tap on the floor.",
        non_diegetic_music="A gentle piano melody plays.",
    )
    client = Client([semantic_raw], audio_response=audio.model_dump_json())
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert raw.error is None and raw.model_call_count == 2
    assert raw.response == semantic_raw
    assert raw.deterministic_corrections == {
        "redundant_speech_lead_in_speaker_marker": 1
    }
    draft, _ = t.parse_t2va_semantic(job, raw.response)
    core = t.validate_t2va_draft(
        job,
        draft,
        MimoAudioFinalizeDraft.model_validate_json(raw.audio_finalize.response),
    )
    assert core.non_diegetic_music == audio.non_diegetic_music
    assert list(t.SECTIONS) == [
        line[:-1]
        for line in t.render_t2va_prompt(core).splitlines()
        if line.endswith(":")
    ]


def test_missing_or_changed_auxiliary_audio_fails_before_model(job, tmp_path):
    from pathlib import Path

    Path(job.audio_evidence.music_path).write_bytes(b"changed")
    client = Client([])
    raw = T2VAMimoBackend(config(tmp_path), client=client).annotate(job, "a" * 64)
    assert raw.error and raw.model_call_count == 0 and not client.calls
