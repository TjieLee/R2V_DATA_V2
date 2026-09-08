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
    assembly = mb.MimoFinalAVAssemblyDraft.model_json_schema()
    assert set(assembly["properties"]) == {
        "audio_observation", "av_grounding", "summary", "shot1_caption",
        "overall_soundscape", "non_diegetic_music", "warnings",
    }
    assert visual["additionalProperties"] is False and assembly["additionalProperties"] is False
    assert mb.MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.20"
    assert mb.MIMO25_ICL_VERSION == "h3_official_ref2va_detailed_shot1_v4"
    assert mb.MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION == "h3_mimo25_speaker_marker_polish_v4"


@pytest.mark.parametrize("transport", ["sglang", "xiaomi"])
def test_two_turn_prefix_is_literal_and_visual_facts_are_isolated(tmp_path, monkeypatch, transport):
    expected = _annotation()
    visual_raw, assembly_raw = split_annotation(expected.model_dump_json())
    visual = json.loads(visual_raw)
    visual["shot1_visual_description"] = "<Subject 1> turns, then lowers a hand. The camera stays static."
    expected.visual_observation.visual_blocks[0].text = visual["shot1_visual_description"]
    visual_raw = json.dumps(visual, indent=2)
    job = _job_fixture(tmp_path)
    backend, calls = _backend(
        tmp_path, [(visual_raw, 0), (assembly_raw, 8)], transport=transport,
        icl="official_ref2va_v1",
    )
    validate = mb.validate_annotation
    validations = []

    def checked(annotation, **kwargs):
        assert len(calls.requests) == 2
        validations.append(annotation)
        return validate(annotation, **kwargs)

    monkeypatch.setattr(mb, "validate_annotation", checked)
    result = _run(backend, job)
    assert result.annotation == expected
    assert result.annotation.visual_observation.visual_blocks[0].block_id == "v1"
    assert result.annotation.visual_observation.visual_blocks[0].text == visual["shot1_visual_description"]
    assert len(result.annotation.visual_observation.visual_blocks) == 1
    assert validations == [expected]
    assert result.raw_responses == (visual_raw, assembly_raw)
    assert result.visual_raw_response == visual_raw and result.final_av_raw_response == assembly_raw
    assert result.visual_model_call_count == 1 and result.text_model_call_count == 0
    assert result.model_call_count == len(calls.requests) == 2
    assert result.http_retry_count == result.recheck_count == 0
    first, second = calls.requests
    assert second["messages"][:-2] == first["messages"]
    assert second["messages"][-2] == {"role": "assistant", "content": visual_raw}
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
    assert "A cello melody." in final_user and "A clink." in final_user
    final_contract = json.loads(final_user.split("AUTHORITATIVE INPUT:\n", 1)[1].split(
        "\nAUXILIARY AUDIO CANDIDATES", 1,
    )[0])
    assert not {"reference_selection", "reference_image_mapping", "subject_definition_requirements"} & final_contract.keys()
    assert first["extra_body"]["use_audio_in_video"] is False
    assert second["extra_body"]["use_audio_in_video"] is True
    if transport == "sglang":
        for request, schema in zip(calls.requests, (mb.MimoVisualDraft, mb.MimoFinalAVAssemblyDraft), strict=True):
            assert request["response_format"] == {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True,
            }}
    else:
        assert all(request["response_format"] == {"type": "json_object"} for request in calls.requests)
    assert [d.input_modality for d in result.diagnostics] == ["target_video_visual_only", "target_video_av_assembly"]
    assert all(d.usage.cached_tokens is None for d in result.diagnostics)


@pytest.mark.parametrize("failure_stage", [1, 2])
@pytest.mark.parametrize("failure", ["malformed", "truncated", "http"])
def test_two_turn_failure_stops_and_preserves_completed_evidence(tmp_path, failure_stage, failure):
    visual_raw, assembly_raw = split_annotation(_annotation().model_dump_json())
    responses = [(visual_raw, 0), (assembly_raw, 8)]
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
    if failure_stage == 2:
        assert result.visual_raw_response == visual_raw
    assert result.raw_responses == tuple(raw for raw in (
        result.visual_raw_response, result.final_av_raw_response,
    ) if raw is not None)


def test_visual_and_assembly_prompt_ownership():
    assert "from beginning to end" in mb.VISUAL_SYSTEM_PROMPT
    assert "early through middle to late" in mb.VISUAL_SYSTEM_PROMPT
    assert "Do not infer sounds or dialogue" in mb.VISUAL_SYSTEM_PROMPT
    assert "No hard word-count requirement" in mb.VISUAL_SYSTEM_PROMPT
    assert "authoritative visual draft" in mb.SYSTEM_PROMPT
    assert "Do NOT replace the visual draft with a new shorter visual summary" in mb.SYSTEM_PROMPT
    assert "Minimal connective rephrasing" in mb.SYSTEM_PROMPT
    assert "retain all materially useful visual observations" in mb.SYSTEM_PROMPT


def test_final_av_has_one_canonical_audio_ownership_contract():
    prompt = mb.SYSTEM_PROMPT
    assert prompt.count("\nFINAL H3 FIELD OWNERSHIP\n") == 1
    ownership = prompt.split("FINAL H3 FIELD OWNERSHIP\n", 1)[1].split("VISUAL OWNERSHIP\n", 1)[0]
    for rule in (
        "Only localized diegetic or shot-synchronized audible events",
        "MUST NOT contain continuous ambience / room tone / hum / rumble",
        "Continuous ambience / room tone / hum / rumble / environmental noise -> overall_soundscape ONLY",
        "audience-only score/background music -> non_diegetic_music ONLY",
        "Diegetic/in-scene music -> shot1_caption ONLY",
        "non_diegetic_music must be N/A for that music",
        "Every audible fact has ONE narrative home",
        "Do not duplicate an audible event across those three fields",
        '"Preserve evidence" means preserve the factual event in its CORRECT owning field',
        "not copy it into multiple fields",
        "summary is exempt from this exclusivity",
    ):
        assert rule in ownership
    caption = prompt.split("SHOT1 CAPTION / DETAILED DESCRIPTION\n", 1)[1].split(
        "VISIBLE SUBJECT PRESERVATION\n", 1,
    )[0]
    assert "Follow FINAL H3 FIELD OWNERSHIP for all audible content." in caption
    markers = prompt.split("SPEAKER MARKERS\n", 1)[1]
    for section in (caption, markers):
        assert "overall_soundscape" not in section
        assert "non_diegetic_music" not in section
        assert "hum" not in section and "rumble" not in section
    assert len(prompt) < 17905  # v39 baseline size, not a generated-caption gate.


def test_visual_facts_are_not_reversed_to_justify_speaker_binding():
    prompt = mb.SYSTEM_PROMPT
    assert prompt.count("\nVISUAL OWNERSHIP\n") == 1
    visual = prompt.split("VISUAL OWNERSHIP\n", 1)[1].split("\nAUTHORITY\n", 1)[0]
    for rule in (
        "MUST NOT invent or reverse a Turn 1 observable visual fact merely to support speaker grounding",
        'If Turn 1 says "Her lips remain closed.", do NOT change it to "Her lips move subtly"',
        "Speaker binding and visual articulation are separate decisions",
        "Stage A articulation is an observable visual clue, not speaker authority",
        "speech_correlated_articulation=not_observed",
        "NOT evidence that the person is not speaking; it does not block visible binding",
        "Full AV may still bind a visible entity",
        "LR-ASD support is absent",
        "Do not manufacture visible_lip_motion, mouth movement, or another visual observation",
        "not_observed must not be converted into visible_lip_motion without actual positive visual observation",
        "concrete, unambiguous visual contradiction",
        "Grounding preference alone is not such a contradiction",
    ):
        assert rule in visual
    assert "supporting clues, NOT mandatory prerequisites" in prompt
    assert "bind that entity directly" in prompt
    assert "Reinspect conflicting Stage A articulation and Stage C lip-motion evidence" not in prompt


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
    visual, _ = split_annotation(_annotation().model_dump_json())
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


def test_visual_prompt_and_both_schemas_remain_frozen():
    # 4e631ab: only the final AV prompt is changing.
    assert hashlib.sha256(mb.VISUAL_SYSTEM_PROMPT.encode()).hexdigest() == (
        "d838c696eab622b16f2aa2f2cc83408eab5bed4b0f225e413eb345d914f9ca1d"
    )
    visual_schema = json.dumps(mb.MimoVisualDraft.model_json_schema(), sort_keys=True)
    assert hashlib.sha256(visual_schema.encode()).hexdigest() == (
        "12c091fc52f25d61a205cfd13a8b0d5603cca7006b276b368500653ff9022045"
    )
    schema = json.dumps(mb.MimoFinalAVAssemblyDraft.model_json_schema(), sort_keys=True)
    assert hashlib.sha256(schema.encode()).hexdigest() == (
        "f7d154d4d6add878a387d0a910d8fe2b7a710d4277b0118db1fd1c2f40a1e1d5"
    )


@pytest.mark.parametrize("cached_tokens", [0, 73])
def test_cache_is_diagnostic_only_and_two_turn_provenance_is_fingerprinted(tmp_path, cached_tokens):
    visual, assembly = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(visual, 0), (assembly, 8)])
    create = calls.create

    def with_cache(**request):
        response = create(**request)
        response.usage["prompt_tokens_details"]["cached_tokens"] = cached_tokens
        return response

    calls.create = with_cache
    result = _run(backend, _job_fixture(tmp_path))
    assert result.model_call_count == 2
    assert [d.usage.cached_tokens for d in result.diagnostics] == [cached_tokens] * 2
    provenance = backend.provenance
    assert provenance.schema_version == "r2v.h3.mimo25_backend.49"
    assert provenance.prompt_version == "h3_mimo25_two_turn_av_reconcile_v40"
    assert provenance.visual_prompt_version == "h3_mimo25_visual_only_v2"
    assert provenance.materializer_version == "h3_mimo25_materializer_v23"
    assert provenance.policy_version == "h3_mimo25_av_authority_contract_v17"
    values = provenance.model_dump(mode="json", exclude={"configuration_fingerprint"})
    assert provenance.configuration_fingerprint == mb._sha256_text(mb._compact_json(values))
    del values["visual_prompt_version"]
    assert provenance.configuration_fingerprint != mb._sha256_text(mb._compact_json(values))
