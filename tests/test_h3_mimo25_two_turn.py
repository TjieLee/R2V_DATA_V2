from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3 import mimo25_backend as mb
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import _annotation, _backend, _job_fixture


def _run(backend, job):
    return backend.reconcile(
        job, segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_paths={kind: Path(job.target_full_audio_path) for kind in ("speech", "music", "sfx")},
    )


def _visual_structure_fixture():
    visual_raw, _, _ = split_annotation(_annotation().model_dump_json())
    visual = json.loads(visual_raw)
    row = visual["segment_views"][0]
    first = row["entity_observations"][0]
    second = {**first, "entity_id": "e2"}
    row["visible_entity_ids"] = ["e1", "e2"]
    return visual, first, second


def test_duplicate_visual_segment_rows_merge_in_first_seen_order():
    visual, first, second = _visual_structure_fixture()
    row = visual["segment_views"][0]
    row["segment_id"] = "segment_0001"
    other = {"segment_id": "segment_0002", "visible_entity_ids": [], "entity_observations": []}
    visual["segment_views"] = [row, other, {
        "segment_id": "segment_0001", "visible_entity_ids": ["e2"],
        "entity_observations": [second, first],
    }]
    raw = json.dumps(visual)
    canonical, corrections = mb._canonicalize_visual_draft_payload(raw)
    draft = mb.MimoVisualDraft.model_validate_json(canonical)
    assert [view.segment_id for view in draft.segment_views] == ["segment_0001", "segment_0002"]
    assert draft.segment_views[0].visible_entity_ids == ["e1", "e2"]
    assert [obs.model_dump() for obs in draft.segment_views[0].entity_observations] == [first, second]
    assert draft.shot1_visual_description == visual["shot1_visual_description"]
    assert corrections == {"visual_segment_row_merge": 1}
    assert mb._canonicalize_visual_draft_payload(canonical) == (canonical, {})
    assert json.loads(raw) == visual


def test_partial_visual_observations_do_not_invent_missing_entity_metadata():
    visual, first, _ = _visual_structure_fixture()
    draft = mb.MimoVisualDraft.model_validate(visual)
    assert draft.segment_views[0].visible_entity_ids == ["e1", "e2"]
    assert [obs.model_dump() for obs in draft.segment_views[0].entity_observations] == [first]


def test_visual_observation_order_is_not_a_presence_requirement():
    visual, first, second = _visual_structure_fixture()
    visual["segment_views"][0]["entity_observations"] = [second, first]
    draft = mb.MimoVisualDraft.model_validate(visual)
    assert [obs.entity_id for obs in draft.segment_views[0].entity_observations] == ["e2", "e1"]


@pytest.mark.parametrize("invalid", ["duplicate_visible", "duplicate_observation", "outside_visible"])
def test_visual_segment_keeps_structural_sanity(invalid):
    visual, first, second = _visual_structure_fixture()
    row = visual["segment_views"][0]
    if invalid == "duplicate_visible":
        row["visible_entity_ids"] = ["e1", "e1"]
    elif invalid == "duplicate_observation":
        row["entity_observations"] = [first, first]
    else:
        row["visible_entity_ids"] = ["e1"]
        row["entity_observations"] = [second]
    with pytest.raises(ValidationError, match="visual observations must be unique"):
        mb.MimoVisualDraft.model_validate(visual)


def test_duplicate_visual_rows_do_not_hide_conflicting_observations():
    visual, first, _ = _visual_structure_fixture()
    row = visual["segment_views"][0]
    visual["segment_views"].append({
        **row, "entity_observations": [{**first, "orientation": "back"}],
    })
    canonical, _ = mb._canonicalize_visual_draft_payload(json.dumps(visual))
    with pytest.raises(ValidationError, match="visual observations must be unique"):
        mb.MimoVisualDraft.model_validate_json(canonical)


@pytest.mark.parametrize("invalid", ["missing", "extra"])
def test_duplicate_visual_rows_do_not_mask_invalid_fields(invalid):
    visual, _, _ = _visual_structure_fixture()
    row = dict(visual["segment_views"][0])
    if invalid == "missing":
        del row["entity_observations"]
    else:
        row["unexpected"] = True
    visual["segment_views"].append(row)
    raw = json.dumps(visual)
    assert mb._canonicalize_visual_draft_payload(raw) == (raw, {})
    with pytest.raises(ValidationError):
        mb.MimoVisualDraft.model_validate_json(raw)


@pytest.mark.parametrize("pattern", ["duplicate_0262", "partial_2ec", "picture_5cb"])
def test_visual_structure_recovery_preserves_raw_prefix_and_call_count(tmp_path, pattern):
    visual, _, second = _visual_structure_fixture()
    if pattern == "duplicate_0262":
        visual["segment_views"].append({
            "segment_id": "segment_1", "visible_entity_ids": ["e2"],
            "entity_observations": [second],
        })
    if pattern == "picture_5cb":
        visual["shot1_visual_description"] = "<Subject 1> from <Picture 1> stands by the door."
    visual_raw = json.dumps(visual, indent=2)
    _, speech_raw, finalize_raw = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(visual_raw, 0), (speech_raw, 8), (finalize_raw, 8)])
    result = backend.reconcile(
        _job_fixture(tmp_path), segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1", "e2"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
    )
    views = result.annotation.visual_observation.segment_views
    assert len(views) == 1 and views[0].visible_entity_ids == ["e1", "e2"]
    expected_ids = ["e1", "e2"] if pattern == "duplicate_0262" else ["e1"]
    assert [obs.entity_id for obs in views[0].entity_observations] == expected_ids
    assert result.annotation.visual_observation.visual_blocks[0].text == visual["shot1_visual_description"]
    assert result.visual_raw_response == visual_raw
    assert result.raw_responses == (visual_raw, speech_raw, finalize_raw)
    assert all(message["role"] != "assistant" for message in calls.requests[1]["messages"])
    supplied = calls.requests[1]["messages"][-1]["content"][-1]["text"].split("TURN 1 VISUAL DRAFT:\n")[1]
    assert json.loads(supplied)["segment_views"] == [view.model_dump() for view in views]
    assert result.model_call_count == len(calls.requests) == 3
    assert result.recheck_count == result.http_retry_count == 0
    if pattern == "duplicate_0262":
        assert result.deterministic_correction_counts["visual_segment_row_merge"] == 1


def test_visual_block_preserves_valid_subject_and_picture_labels():
    text = "<Subject 1> wears the coat in <Picture 1>."
    assert mb.MimoVisualBlock(block_id="v1", text=text).text == text


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
        "audio_observation", "av_grounding", "summary", "shot1_caption", "warnings",
    }
    finalize = mb.MimoAudioFinalizeDraft.model_json_schema()
    assert set(finalize["properties"]) == {
        "speaker_voice_profiles", "overall_soundscape", "non_diegetic_music",
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
    raws = split_annotation(expected.model_dump_json())
    job = _job_fixture(tmp_path)
    backend, calls = _backend(tmp_path, [(raws[0], 0), (raws[1], 8), (raws[2], 8)],
                              transport=transport, icl="official_ref2va_v1")
    original = calls.create
    def no_reference_usage_after_visual(**request):
        response = original(**request)
        if len(calls.requests) > 1:
            response.usage["prompt_tokens_details"]["image_tokens"] = 0
        return response
    calls.create = no_reference_usage_after_visual
    result = _run(backend, job)
    assert result.annotation == expected
    assert result.raw_responses == raws
    assert result.model_call_count == len(calls.requests) == 3
    assert result.recheck_count == result.http_retry_count == 0
    first, second, third = calls.requests
    assert first["messages"][0]["content"] == mb.VISUAL_SYSTEM_PROMPT
    assert first["messages"][1:-1] == mb._official_detailed_description_icl_messages()
    for request, prompt in ((second, mb.SYSTEM_PROMPT), (third, mb.AUDIO_FINALIZE_SYSTEM_PROMPT)):
        assert [message["role"] for message in request["messages"]] == ["system", "user"]
        assert request["messages"][0]["content"] == prompt
        media = request["messages"][1]["content"]
        assert sum(item["type"] == "video_url" for item in media) == 1
        assert not any(item["type"] == "image_url" for item in media)
        assert request["extra_body"]["use_audio_in_video"] is True
    assert first["extra_body"]["use_audio_in_video"] is False
    speech_content = second["messages"][-1]["content"]
    assert not any(item["type"] == "audio_url" for item in speech_content)
    speech_text = speech_content[-1]["text"]
    contract = json.loads(speech_text.split("AUTHORITATIVE INPUT:\n")[1].split("\nTURN 1 VISUAL DRAFT:")[0])
    assert contract["entity_subject_mapping"] == {"e1": "<Subject 1>"}
    assert contract["allowed_segment_ids"] == ["segment_1"]
    assert job.segments[0].asr_text in speech_text
    assert not {"reference_selection", "reference_image_mapping", "subject_definition_requirements"} & contract.keys()
    assert json.loads(speech_text.split("TURN 1 VISUAL DRAFT:\n")[1]) == json.loads(raws[0])
    final_media = third["messages"][-1]["content"]
    audio = [item["audio_url"]["url"] for item in final_media if item["type"] == "audio_url"]
    assert audio == [backend.config.media_resolver.resolve(Path(job.target_full_audio_path))] * 3
    assert job.segments[0].asr_text not in json.dumps(final_media)
    assert "music stem:" in json.dumps(final_media) and "sfx stem:" in json.dumps(final_media)
    assert "separator_candidate" not in json.dumps(calls.requests)
    target_text = final_media[1]["text"].split("FINALIZED SPEAKER-PROFILE TARGETS:\n", 1)[1]
    targets = json.loads(target_text.split("\nRESPONSE SCHEMA:", 1)[0])
    assert targets == [{
        "speaker_group": "g1", "speaker_id": "S1", "segments": [{
            "segment_id": "segment_1", "start_time": 0.0, "end_time": 1.0,
            "final_binding": "<Subject 1>", "speech_presentation": "onscreen_spoken",
        }],
    }]
    assert json.loads(raws[1])["audio_observation"]["speaker_voice_profiles"] == []
    assert all("image_tokens_unavailable" not in d.warnings for d in result.diagnostics[1:])
    for request, schema in zip(calls.requests, (mb.MimoVisualDraft, mb.MimoSpeechAVAssemblyDraft, mb.MimoAudioFinalizeDraft), strict=True):
        if transport == "sglang":
            assert request["response_format"] == {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True,
            }}
        else:
            assert request["response_format"] == {"type": "json_object"}


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
    assert "Observe the full target video" in mb.VISUAL_SYSTEM_PROMPT
    assert "early-to-middle-to-late" in mb.VISUAL_SYSTEM_PROMPT
    assert "Do not infer sounds, dialogue" in mb.VISUAL_SYSTEM_PROMPT
    assert "No hard word-count requirement" in mb.VISUAL_SYSTEM_PROMPT
    assert "Turn 1 visual draft owns observable visual facts" in mb.SYSTEM_PROMPT
    assert "minimum local grammatical edits" in mb.SYSTEM_PROMPT
    assert "speaker markers and exact dialogue" in mb.SYSTEM_PROMPT


def test_speech_prompt_defers_sound_and_finalizer_is_short():
    assert "Defer ALL non-dialogue audio to Turn 3" in mb.SYSTEM_PROMPT
    assert "TARGET-VIDEO SUMMARY" in mb.SYSTEM_PROMPT
    for rule in ("main visible subject/action", "not an audio report", "materializer owns that prefix"):
        assert rule in mb.SYSTEM_PROMPT
    prompt = mb.AUDIO_FINALIZE_SYSTEM_PROMPT
    for rule in ("Return ONLY speaker_voice_profiles, overall_soundscape and non_diegetic_music",
                 "three separated audio views of that SAME target", "Original AV remains primary authority",
                 "residuals are not automatically true", "quiet in the original mix",
                 "coherent piano/music compatible", "excluding dialogue and non-diegetic music"):
        assert rule in prompt
    assert len(prompt.split()) < 350
    assert "threshold" not in prompt


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
        "ROLE / OUTPUT", "SPEECH / AV ASSEMBLY", "VISUAL OWNERSHIP", "AUTHORITY",
        "AUDIO + AV GROUNDING", "VISIBLE SPEAKER BINDING", "SHOT1 CAPTION", "TARGET-VIDEO SUMMARY", "SPEAKER MARKERS",
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
    assert len(prompt) < 8500  # Prompt cleanup only, not a generated-caption length gate.


def test_assembly_schema_field_ownership():
    speech = mb.MimoSpeechAVAssemblyDraft.model_json_schema()
    final = mb.MimoAudioFinalizeDraft.model_json_schema()
    assert "summary" in speech["required"]
    assert set(final["properties"]) == set(final["required"]) == {"speaker_voice_profiles", "overall_soundscape", "non_diegetic_music"}


def test_visual_v3_factual_writing_and_attribute_subjects():
    prompt = mb.VISUAL_SYSTEM_PROMPT
    for requirement in ("generation-quality visual description", "composition/framing",
                        "environment and lighting/color", "observable actions/state changes",
                        "camera behavior", "Do not mechanically repeat attribute Subjects",
                        "when they appear", "ONE concise sentence",
                        "Picture provenance and attribute ownership", "describe visible appearance instead of guessing"):
        assert requirement in prompt
    assert "visible background Subject when it defines the current scene/background" in prompt
    assert "parenthetical metadata" in prompt
    assert "subject_definitions" in prompt
    assert "action, composition, state change or disambiguation" in prompt
    assert "Emphasize spatial relations, action, interaction, camera and temporal progression" in prompt
    assert "(e1), (e2), g1, v1 or segment_0001" in prompt
    assert "Attribute-only Subjects do not need mechanical insertion" not in prompt
    assert len(prompt.split()) < 450
    payload = json.loads(split_annotation(_annotation().model_dump_json())[0])
    payload["shot1_visual_description"] = "A still figure. " * 500
    assert mb.MimoVisualDraft.model_validate(payload).shot1_visual_description == payload["shot1_visual_description"]


@pytest.mark.parametrize("syntax", ["<Audio 1>", "<Video 1>", "(S1)", "<d>x</d>", "[Shot 1]", "[[segment:x]]", "<Subject 0>", "<Picture 0>", "<Unknown 1>"])
def test_reused_visual_caption_still_rejects_non_visual_syntax(syntax):
    with pytest.raises(ValidationError, match="non-visual or pipeline syntax"):
        mb.MimoVisualBlock(block_id="v1", text=f"<Subject 1> stands. {syntax}")


def test_visual_prompt_and_schema_remain_frozen():
    # Visual JSON shape stays frozen; only the writing prompt changes.
    assert mb.MIMO25_VISUAL_PROMPT_VERSION == "h3_mimo25_visual_only_v4"
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
    assert provenance.schema_version == "r2v.h3.mimo25_backend.55"
    assert provenance.prompt_version == "h3_mimo25_speech_assembly_v45"
    assert provenance.visual_prompt_version == "h3_mimo25_visual_only_v4"
    assert provenance.materializer_version == "h3_mimo25_materializer_v25"
    assert provenance.policy_version == "h3_mimo25_av_authority_contract_v17"
    values = provenance.model_dump(mode="json", exclude={"configuration_fingerprint"})
    assert provenance.configuration_fingerprint == mb._sha256_text(mb._compact_json(values))
    assert provenance.audio_finalize_prompt_version == "h3_mimo25_audio_finalize_v4"
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
    speech["summary"] = "A standing person speaks."
    speech["audio_observation"]["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "Turn 2 must not own this"}
    ]
    finalized["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "low register, dry texture, measured cadence"}
    ]
    finalized.update(
        overall_soundscape="A low steady hum.",
        non_diegetic_music="A soft instrumental score.",
    )
    raws = tuple(json.dumps(item) for item in (visual, speech, finalized))
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws])
    result = _run(backend, _job_fixture(tmp_path))
    annotation = result.annotation
    assert annotation.visual_observation.visual_blocks[0].text == visual["shot1_visual_description"]
    assert annotation.visual_observation.segment_views == source.visual_observation.segment_views
    assert annotation.audio_observation.segment_decisions == source.audio_observation.segment_decisions
    assert annotation.audio_observation.speaker_voice_profiles == mb.MimoAudioFinalizeDraft.model_validate(finalized).speaker_voice_profiles
    assert annotation.av_grounding == source.av_grounding
    assert annotation.warnings == source.warnings
    for key in ("subject_definitions", "visual_retention_analysis", "style_opening"):
        assert annotation.h3_semantics.model_dump(mode="json")[key] == visual[key]
    for key in ("overall_soundscape", "non_diegetic_music"):
        assert getattr(annotation.h3_semantics, key) == finalized[key]
    assert annotation.h3_semantics.shot1_caption == speech["shot1_caption"]
    assert annotation.h3_semantics.summary == speech["summary"]
    assert result.raw_responses == raws
    assert result.model_call_count == len(calls.requests) == 3
    assert not result.speaker_marker_polish.attempted


def test_unsupported_turn3_voice_profile_stays_null(tmp_path):
    raws = list(split_annotation(_annotation().model_dump_json()))
    finalized = json.loads(raws[2])
    finalized["speaker_voice_profiles"][0]["voice_characteristics"] = None
    raws[2] = json.dumps(finalized)
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws])
    result = _run(backend, _job_fixture(tmp_path))
    assert result.annotation.speaker_voice_profiles[0].voice_characteristics is None
    assert result.model_call_count == len(calls.requests) == 3


def test_profile_targets_ignore_nontranscribed_segments_and_preserve_group_order(tmp_path):
    job = _job_fixture(tmp_path)
    _, raw, _ = split_annotation(_annotation().model_dump_json())
    assembly = mb.MimoSpeechAVAssemblyDraft.model_validate_json(raw)
    segment = job.segments[0]
    extra = segment.model_copy(update={"segment_id": "empty", "asr_status": "empty", "asr_text": None})
    repeated = segment.model_copy(update={"segment_id": "repeat", "start_time": 1.0, "end_time": 2.0})
    job = job.model_copy(update={"segments": [extra, segment, repeated]})
    for inventory in (assembly.audio_observation.segment_decisions, assembly.av_grounding.segment_groundings):
        inventory.extend([inventory[0].model_copy(update={"segment_id": "empty"}),
                          inventory[0].model_copy(update={"segment_id": "repeat"})])
    targets = mb._speaker_profile_targets(assembly, job)
    assert len(targets) == 1
    assert targets[0]["speaker_group"] == "g1" and targets[0]["speaker_id"] == "S1"
    assert [s["segment_id"] for s in targets[0]["segments"]] == ["segment_1", "repeat"]
    assert job.segments[1].asr_text not in json.dumps(targets)
    assert mb._speaker_profile_targets(assembly, job.model_copy(update={"segments": []})) == []


@pytest.mark.parametrize("schema,foreign_field", [
    (mb.MimoAudioFinalizeDraft, "summary"),
    (mb.MimoAudioFinalizeDraft, "shot1_caption"),
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


def test_spatial_presentation_and_audible_music_prompt_contract():
    speech = mb.SYSTEM_PROMPT
    for rule in (
        "Keep binding_status, speech_presentation and entity_id semantically consistent",
        "If the vocal source is judged offscreen: binding_status=offscreen, speech_presentation=offscreen_spoken, entity_id=null",
        "represent the vocal source separately from visible Subjects in shot1_caption",
        "If a visible entity is judged to be the speaker: binding_status=visible_entity, speech_presentation=onscreen_spoken, entity_id=that visible entity",
        "Do not attach (Sx) to a visible Subject when the final AV judgment says the voice is offscreen",
        "Keep the visible Subject in the visual prose and introduce the offscreen vocal source separately",
        "supporting clues, NOT mandatory prerequisites",
        "bind that entity directly",
    ):
        assert rule in speech
    for forbidden in ("lips", "mouth", "articulation", "visible_lip_motion", "no_visible_lip_motion"):
        assert forbidden not in speech
    assert "Do not output N/A for clearly established music" in mb.AUDIO_FINALIZE_SYSTEM_PROMPT
    assert "only or most salient visible person" in speech
    assert "keep it offscreen even when LR-ASD has no binding" in speech
    assert "Return audio_observation.speaker_voice_profiles=[]" in speech
    for rule in (
        "speech stem supplies acoustic voice evidence", "pitch/register", "timbre/texture",
        "cadence/speaking rate", "energy/delivery", "voice_characteristics=null",
        "Never infer identity, profession, personality, nationality, role or demographics",
        "Do not re-decide speaker identity, grouping, Subject binding, on/offscreen presentation or ASR",
    ):
        assert rule in mb.AUDIO_FINALIZE_SYSTEM_PROMPT


def test_correction_rules_reach_only_their_owning_request(tmp_path):
    responses = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in responses])
    result = _run(backend, _job_fixture(tmp_path))
    first, second, third = calls.requests
    consistency = "Keep binding_status, speech_presentation and entity_id semantically consistent"
    music = "Do not output N/A for clearly established music"
    assert consistency not in json.dumps(first["messages"])
    assert consistency in second["messages"][0]["content"]
    assert music not in second["messages"][0]["content"]
    assert music in third["messages"][0]["content"]
    assert consistency not in third["messages"][0]["content"]
    assert result.model_call_count == len(calls.requests) == 3
    assert result.recheck_count == result.http_retry_count == 0
