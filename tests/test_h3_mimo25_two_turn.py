from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3 import mimo25_backend as mb
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import _annotation, _backend, _job_fixture


def _run(backend, job):
    return backend.reconcile(
        job, segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_evidence={
            "music_separator_candidate": "A cello melody.", "sfx_separator_candidate": "A clink.",
        },
    )


def test_intermediate_schema_field_ownership():
    visual = mb.MimoVisualDraft.model_json_schema()
    assert set(visual["properties"]) == {
        "segment_views", "subject_definitions", "visual_retention_analysis",
        "style_opening", "shot1_visual_description",
    }
    keys = set()

    def collect(value):
        if isinstance(value, dict):
            keys.update(value.get("properties", {}))
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(visual)
    assert not {"visual_observation", "visual_blocks"} & keys
    assert "MimoVisualObservation" not in visual["$defs"]
    assert "MimoVisualBlock" not in visual["$defs"]
    assert "maxLength" not in visual["properties"]["shot1_visual_description"]
    assert not keys & {
        "audio_observation", "av_grounding", "overall_soundscape", "non_diegetic_music",
        "speaker_group", "primary_speaker_group", "confidence", "evidence_codes",
        "speech_presentation", "binding_status", "delivery_style",
    }
    assembly = mb.MimoSpeechAVAssemblyDraft.model_json_schema()
    assert set(assembly["properties"]) == {
        "audio_observation", "av_grounding", "shot1_caption", "warnings",
    }
    finalize = mb.MimoAudioFinalizeDraft.model_json_schema()
    assert set(finalize["properties"]) == {
        "summary", "shot1_caption", "overall_soundscape", "non_diegetic_music",
    }
    assert finalize["additionalProperties"] is False
    assert set(finalize["required"]) == set(finalize["properties"])
    assert visual["additionalProperties"] is False and assembly["additionalProperties"] is False
    assert mb.MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.20"
    assert mb.MIMO25_ICL_VERSION == "h3_official_ref2va_detailed_shot1_v4"
    assert mb.MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION == "h3_mimo25_speaker_marker_polish_v4"


@pytest.mark.parametrize("transport", ["sglang", "xiaomi"])
def test_three_turn_prefix_is_literal_and_visual_facts_are_isolated(tmp_path, monkeypatch, transport):
    expected = _annotation()
    visual_raw, speech_raw, finalize_raw = split_annotation(expected.model_dump_json())
    visual = json.loads(visual_raw)
    visual["shot1_visual_description"] = "<Subject 1> turns, then lowers a hand. The camera stays static."
    expected.visual_observation.visual_blocks[0].text = visual["shot1_visual_description"]
    visual_raw = json.dumps(visual, indent=2)
    job = _job_fixture(tmp_path)
    backend, calls = _backend(
        tmp_path, [(visual_raw, 0), (speech_raw, 8), (finalize_raw, 8)], transport=transport,
        icl="official_ref2va_v1",
    )
    validate = mb.validate_annotation
    validations = []

    def checked(annotation, **kwargs):
        assert len(calls.requests) == 3
        validations.append(annotation)
        return validate(annotation, **kwargs)

    monkeypatch.setattr(mb, "validate_annotation", checked)
    result = _run(backend, job)
    assert result.annotation == expected
    assert result.annotation.visual_observation.visual_blocks[0].block_id == "v1"
    assert result.annotation.visual_observation.visual_blocks[0].text == visual["shot1_visual_description"]
    assert len(result.annotation.visual_observation.visual_blocks) == 1
    assert validations == [expected]
    assert result.raw_responses == (visual_raw, speech_raw, finalize_raw)
    assert result.visual_raw_response == visual_raw
    assert result.speech_av_raw_response == speech_raw
    assert result.audio_finalize_raw_response == finalize_raw
    assert result.visual_model_call_count == 1 and result.text_model_call_count == 0
    assert result.model_call_count == len(calls.requests) == 3
    assert result.http_retry_count == result.recheck_count == 0
    first, second, third = calls.requests
    assert second["messages"][:-2] == first["messages"]
    assert second["messages"][-2] == {"role": "assistant", "content": visual_raw}
    assert third["messages"][:-2] == second["messages"]
    assert third["messages"][-2] == {"role": "assistant", "content": speech_raw}
    assert first["messages"][0]["content"] == mb.VISUAL_SYSTEM_PROMPT
    assert first["messages"][1:-1] == mb._official_detailed_description_icl_messages()
    media = first["messages"][-1]["content"]
    assert sum(item["type"] == "video_url" for item in media) == 1
    assert sum(item["type"] == "image_url" for item in media) == len(job.reference_images)
    assert not any(item["type"] in {"audio_url", "input_audio"} for item in media)
    contract = json.loads(media[-1]["text"].split("VISUAL-ONLY INPUT:\n")[1].split("\nRESPONSE SCHEMA:")[0])
    assert contract["segment_windows"] == [{"segment_id": "segment_1", "start_time": job.segments[0].start_time, "end_time": job.segments[0].end_time}]
    for key in ("asr_text", "asr_language", "speaker_cluster_id", "binding_status",
                "evidence_codes", "delivery_style", "music_separator_candidate", "sfx_separator_candidate"):
        assert key not in json.dumps(contract)
    assert job.segments[0].asr_text not in json.dumps(contract)
    final_user = second["messages"][-1]["content"]
    assert isinstance(final_user, str)
    assert "video_url" not in final_user and "image_url" not in final_user
    assert job.segments[0].asr_text in final_user
    assert "music_separator_candidate" not in final_user and "sfx_separator_candidate" not in final_user
    finalize_user = third["messages"][-1]["content"]
    assert isinstance(finalize_user, str)
    assert "A cello melody." in finalize_user and "A clink." in finalize_user
    for key in ("video_url", "image_url", "asr_text", "segments", "reference_selection", "speaker_cluster_id"):
        assert key not in finalize_user
    assert job.segments[0].asr_text not in finalize_user
    assert third["extra_body"]["use_audio_in_video"] is True
    final_contract = json.loads(final_user.split("AUTHORITATIVE INPUT:\n", 1)[1].split(
        "\nAUXILIARY AUDIO CANDIDATES", 1,
    )[0])
    assert not {"reference_selection", "reference_image_mapping", "subject_definition_requirements"} & final_contract.keys()
    assert first["extra_body"]["use_audio_in_video"] is False
    assert second["extra_body"]["use_audio_in_video"] is True
    if transport == "sglang":
        for request, schema in zip(calls.requests, (mb.MimoVisualDraft, mb.MimoSpeechAVAssemblyDraft, mb.MimoAudioFinalizeDraft), strict=True):
            assert request["response_format"] == {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True,
            }}
    else:
        assert all(request["response_format"] == {"type": "json_object"} for request in calls.requests)
    assert [d.input_modality for d in result.diagnostics] == ["target_video_visual_only", "target_video_speech_assembly", "target_video_audio_finalize"]
    assert all(d.usage.cached_tokens is None for d in result.diagnostics)


@pytest.mark.parametrize("failure_stage", [1, 2, 3])
@pytest.mark.parametrize("failure", ["malformed", "truncated", "http"])
def test_three_turn_failure_stops_and_preserves_completed_evidence(tmp_path, failure_stage, failure):
    visual_raw, speech_raw, finalize_raw = split_annotation(_annotation().model_dump_json())
    responses = [(visual_raw, 0), (speech_raw, 8), (finalize_raw, 8)]
    if failure == "malformed":
        responses[failure_stage - 1] = ("not JSON", 0)
    elif failure == "truncated":
        responses[failure_stage - 1] = (responses[failure_stage - 1][0], 8, "length")
    backend, calls = _backend(tmp_path, responses)
    create = calls.create
    if failure == "http":
        def fail(**request):
            if len(calls.requests) == failure_stage - 1:
                calls.requests.append(request)
                raise OSError("synthetic stage failure")
            return create(**request)
        calls.create = fail
    with pytest.raises(mb.MimoBackendFailure) as error:
        _run(backend, _job_fixture(tmp_path))
    result = error.value
    assert result.model_call_count == len(calls.requests) == failure_stage
    assert result.visual_model_call_count == 1 and result.text_model_call_count == 0
    assert len(result.diagnostics) == failure_stage
    assert result.http_retry_count == result.recheck_count == 0
    if failure_stage >= 2:
        assert result.visual_raw_response == visual_raw
    assert result.raw_responses == tuple(raw for raw in (
        result.visual_raw_response, result.speech_av_raw_response, result.audio_finalize_raw_response,
    ) if raw is not None)


def test_visual_and_assembly_prompt_ownership():
    assert "from beginning to end" in mb.VISUAL_SYSTEM_PROMPT
    assert "early through middle to late" in mb.VISUAL_SYSTEM_PROMPT
    assert "Do not infer sounds or dialogue" in mb.VISUAL_SYSTEM_PROMPT
    assert "No hard word-count requirement" in mb.VISUAL_SYSTEM_PROMPT
    assert "Turn 1 visual draft owns observable visual facts" in mb.SYSTEM_PROMPT
    assert "minimum local grammatical edits" in mb.SYSTEM_PROMPT
    assert "speaker markers and exact dialogue" in mb.SYSTEM_PROMPT


def test_speech_prompt_defers_sound_and_finalizer_is_short():
    for rule in (
        "Defer ALL non-dialogue audio to Turn 3",
        "Do NOT add music, hum, rumble, room tone, SFX",
        "exact <d> ASR dialogue", "speaker markers",
    ):
        assert rule in mb.SYSTEM_PROMPT
    for removed in ("AUXILIARY AUDIO EVIDENCE", "FINAL H3 FIELD OWNERSHIP",
                    "music_separator_candidate", "sfx_separator_candidate",
                    "overall_soundscape", "non_diegetic_music", "Weak/masked"):
        assert removed not in mb.SYSTEM_PROMPT
    prompt = mb.AUDIO_FINALIZE_SYSTEM_PROMPT
    for rule in (
        "primary audio authority", "useful hints, not mandatory truth",
        "Continuous ambience / room tone / hum / rumble",
        "overall_soundscape", "Audience-only background score/music -> non_diegetic_music",
        "Localized diegetic or synchronized sound", "Diegetic/in-scene music",
        "Route each audible event once", "NON-DIALOGUE AUDIO wording only",
        "exact <d> dialogue text, Sx, speaker identity and Subject/Picture labels",
        "Do not turn closed lips into moving lips",
    ):
        assert rule in prompt
    for prohibited in ("unless", "must preserve", "threshold", "Weak/masked", "concrete factual contradiction"):
        assert prohibited not in prompt
    assert len(prompt.split()) < 250


def test_turn2_is_minimal_speech_integration_without_visual_priming():
    prompt = mb.SYSTEM_PROMPT
    for forbidden in (
        "lips", "mouth", "articulation", "visible_lip_motion", "no_visible_lip_motion",
        "speech_correlated_articulation", "lip motion",
        "PRIMARY H3 WRITING TASK", "SHOT1 CAPTION / DETAILED DESCRIPTION",
        "VISIBLE SUBJECT PRESERVATION", "Observe the ENTIRE target video",
        "shot scale", "framing", "lighting", "Continue through the end",
    ):
        assert forbidden.lower() not in prompt.lower()
    assert [line for line in prompt.splitlines() if line.isupper()] == [
        "ROLE / OUTPUT", "SPEECH-ONLY ASSEMBLY", "VISUAL OWNERSHIP", "AUTHORITY",
        "AUDIO + AV GROUNDING", "VISIBLE SPEAKER BINDING", "SHOT1 CAPTION", "SPEAKER MARKERS",
    ]
    for rule in (
        "Start from the Turn 1 shot1_visual_description",
        "Preserve its visual content and ordering",
        "Only make the minimum edits needed",
        "establish the correct speaker with (Sx)",
        "insert exact authoritative <d> dialogue at the appropriate point",
        "make the resulting prose grammatical",
        "Do not add new visual observations",
        "Do not add, reverse, or invent visual observations to justify speaker grounding",
        "Speaker grounding is a separate AV decision",
        "does not require an independent visual speech cue",
        "Use evidence_codes only for evidence actually supported by the current input",
        "Keep the list concise and do not repeat the same code",
        "supporting clues, NOT mandatory prerequisites",
        "bind that entity directly",
    ):
        assert rule in prompt
    assert len(prompt) < 8000  # Prompt cleanup only, not a generated-caption length gate.


def test_turn3_prompt_and_both_assembly_schemas_are_frozen():
    assert hashlib.sha256(mb.AUDIO_FINALIZE_SYSTEM_PROMPT.encode()).hexdigest() == (
        "7218c8074f36f8f211ffee8711c179ca72c0216ad3c7a02af90ca4c3f5c3d019"
    )
    for schema, digest in (
        (mb.MimoSpeechAVAssemblyDraft, "d0382d939597478c30b23a5b791641fbde93984ecf74ca427c23cbe95e456aa5"),
        (mb.MimoAudioFinalizeDraft, "82967c3b5918b051613dbe6c5cae774469cf34066e1a570e408e60a485c54239"),
    ):
        assert hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode()).hexdigest() == digest


def test_visual_v2_keeps_only_caption_long_and_definitions_concise():
    prompt = mb.VISUAL_SYSTEM_PROMPT
    for requirement in (
        "shot1_visual_description is the ONLY full visual description",
        "Do not generate visual_blocks",
        "exactly one definition for every required Subject, in required order",
        "ONE concise sentence", "stable visual identity / appearance only",
        "Do NOT describe actions, frame position, chronology, camera, or speaker state",
        "Do NOT mention <Picture N>",
        "Never repeat a fact or clause inside a definition",
        "Subject definitions are not mini-captions",
        'Do NOT use provenance/analysis boilerplate such as "consistent with the visual evidence"',
        '"consistent with the video frames"', '"primary subject of the shot"',
        '"only visible person"',
        "one concise retention statement per required item",
        "Do not repeat appearance descriptions",
        "foreground/midground/background composition", "appearance",
        "lighting and color", "camera motion or clearly static camera",
        "body/hand/arm motion", "gaze", "facial expression and visible expression changes",
        "early through middle to late",
    ):
        assert requirement in prompt
    visual, _, _ = split_annotation(_annotation().model_dump_json())
    payload = json.loads(visual)
    # No word cap, repetition detector, truncation, or deduplication.
    payload["shot1_visual_description"] = "<Subject 1> remains still. " * 600
    parsed = mb.MimoVisualDraft.model_validate(payload)
    block = mb.MimoVisualBlock(block_id="v1", text=parsed.shot1_visual_description)
    assert block.text == payload["shot1_visual_description"]


@pytest.mark.parametrize("syntax", ["<Picture 1>", "<Audio 1>", "(S1)", "<d>x</d>", "[[segment:x]]", "<Subject 0>"])
def test_reused_visual_caption_still_rejects_non_visual_syntax(syntax):
    with pytest.raises(ValidationError, match="non-visual or pipeline syntax"):
        mb.MimoVisualBlock(block_id="v1", text=f"<Subject 1> stands. {syntax}")


def test_visual_prompt_and_schema_remain_frozen():
    # Turn 1 is frozen at the working visual v2 contract.
    assert hashlib.sha256(mb.VISUAL_SYSTEM_PROMPT.encode()).hexdigest() == (
        "d838c696eab622b16f2aa2f2cc83408eab5bed4b0f225e413eb345d914f9ca1d"
    )
    visual_schema = json.dumps(mb.MimoVisualDraft.model_json_schema(), sort_keys=True)
    assert hashlib.sha256(visual_schema.encode()).hexdigest() == (
        "12c091fc52f25d61a205cfd13a8b0d5603cca7006b276b368500653ff9022045"
    )


@pytest.mark.parametrize("cached_tokens", [0, 73])
def test_cache_is_diagnostic_only_and_three_turn_provenance_is_fingerprinted(tmp_path, cached_tokens):
    visual, speech, finalized = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(visual, 0), (speech, 8), (finalized, 8)])
    create = calls.create

    def with_cache(**request):
        response = create(**request)
        response.usage["prompt_tokens_details"]["cached_tokens"] = cached_tokens
        return response

    calls.create = with_cache
    result = _run(backend, _job_fixture(tmp_path))
    assert result.model_call_count == 3
    assert [d.usage.cached_tokens for d in result.diagnostics] == [cached_tokens] * 3
    provenance = backend.provenance
    assert provenance.schema_version == "r2v.h3.mimo25_backend.51"
    assert provenance.prompt_version == "h3_mimo25_speech_assembly_v42"
    assert provenance.visual_prompt_version == "h3_mimo25_visual_only_v2"
    assert provenance.materializer_version == "h3_mimo25_materializer_v23"
    assert provenance.policy_version == "h3_mimo25_av_authority_contract_v17"
    values = provenance.model_dump(mode="json", exclude={"configuration_fingerprint"})
    assert provenance.configuration_fingerprint == mb._sha256_text(mb._compact_json(values))
    assert provenance.audio_finalize_prompt_version == "h3_mimo25_audio_finalize_v1"
    del values["audio_finalize_prompt_version"]
    assert provenance.configuration_fingerprint != mb._sha256_text(mb._compact_json(values))


def test_final_fields_come_from_their_owning_turns(tmp_path):
    source = _annotation()
    visual_raw, speech_raw, finalize_raw = split_annotation(source.model_dump_json())
    visual = json.loads(visual_raw)
    visual["shot1_visual_description"] = "<Subject 1> stands with closed lips and lowers a hand."
    speech = json.loads(speech_raw)
    speech["shot1_caption"] = source.h3_semantics.shot1_caption + " Her lips remain closed."
    finalized = json.loads(finalize_raw)
    finalized.update(
        summary="A standing person speaks as a door closes.",
        shot1_caption=speech["shot1_caption"] + " A door slams.",
        overall_soundscape="A low steady hum.",
        non_diegetic_music="A soft instrumental score.",
    )
    raws = tuple(json.dumps(item) for item in (visual, speech, finalized))
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws])
    result = _run(backend, _job_fixture(tmp_path))
    annotation = result.annotation
    assert annotation.visual_observation.visual_blocks[0].text == visual["shot1_visual_description"]
    assert annotation.visual_observation.segment_views == source.visual_observation.segment_views
    assert annotation.audio_observation == source.audio_observation
    assert annotation.av_grounding == source.av_grounding
    assert annotation.warnings == source.warnings
    for key in ("subject_definitions", "visual_retention_analysis", "style_opening"):
        assert annotation.h3_semantics.model_dump(mode="json")[key] == visual[key]
    for key, value in finalized.items():
        assert getattr(annotation.h3_semantics, key) == value
    assert annotation.h3_semantics.shot1_caption != speech["shot1_caption"]
    assert result.raw_responses == raws
    assert result.model_call_count == len(calls.requests) == 3
    assert not result.speaker_marker_polish.attempted


@pytest.mark.parametrize("schema,foreign_field", [
    (mb.MimoSpeechAVAssemblyDraft, "summary"),
    (mb.MimoSpeechAVAssemblyDraft, "overall_soundscape"),
    (mb.MimoSpeechAVAssemblyDraft, "non_diegetic_music"),
    (mb.MimoAudioFinalizeDraft, "audio_observation"),
    (mb.MimoAudioFinalizeDraft, "av_grounding"),
    (mb.MimoAudioFinalizeDraft, "visual_observation"),
])
def test_intermediate_schema_rejects_other_turn_fields(schema, foreign_field):
    _, speech, finalized = split_annotation(_annotation().model_dump_json())
    payload = json.loads(speech if schema is mb.MimoSpeechAVAssemblyDraft else finalized)
    payload[foreign_field] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        schema.model_validate(payload)
