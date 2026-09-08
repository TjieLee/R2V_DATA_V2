from __future__ import annotations

import json
from dataclasses import replace

import pytest

from r2v_data_v2.h3 import mimo25_backend as mb
from r2v_data_v2.h3 import mimo25_stem_shadow as stem
from tests.h3_mimo_two_turn_helpers import assembly_raw, split_annotation
from tests.test_h3_audio_shadow_qa import _fixture
from tests.test_h3_mimo25_av_shadow import _Completions
from tests.test_h3_mimo25_raw_stems import _backend, _raw, _records, _run, qa

SINGLE = "The woman says, <d>[Chinese] a</d>"
SINGLE_FIXED = "The woman (S1) says, <d>[Chinese] a</d>"
MULTI = "(S1)<d>[Chinese] a</d> A woman says, <d>[Chinese] b</d>"
MULTI_FIXED = "(S1)<d>[Chinese] a</d> A woman (S2) says, <d>[Chinese] b</d>"


def _case(tmp_path, monkeypatch, original, response, speakers=("S1",), transport="sglang"):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = original
    raw = json.dumps(payload)
    responses = [response]
    backend, completions, stems, jobs = _backend(
        tmp_path, shadow, [(raw, 8)], polish_responses=responses,
    )
    facts = [
        {"segment_id": f"segment_{index:04d}", "speaker_id": speaker,
         "language": "Chinese", "text": chr(96 + index)}
        for index, speaker in enumerate(speakers, 1)
    ]
    monkeypatch.setattr(mb, "direct_speech_facts", lambda annotation, segments: facts)
    if transport == "xiaomi":
        backend.config = replace(backend.config, transport="xiaomi", thinking="enabled")
        original_create = completions.create

        def create(**request):
            if request["messages"][0]["content"] == mb.SPEAKER_MARKER_POLISH_PROMPT:
                completions.requests.append(request)
                return _Completions([(responses.pop(0), 0)]).create(**request)
            if request["response_format"] == {"type": "json_object"}:
                completions.requests.append(request)
                return _Completions([(split_annotation(raw)[0] if not request["extra_body"]["use_audio_in_video"] else split_annotation(raw)[2 if mb.AUDIO_FINALIZE_SYSTEM_PROMPT == request["messages"][0]["content"] else 1], 8)]).create(**request)
            return original_create(**request)

        completions.create = create
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert row["audio_finalize_raw_response"] == assembly_raw(raw)
    if row["annotation"] is not None:
        published = json.loads(json.dumps(row["annotation"]))
        published["h3_semantics"]["shot1_caption"] = original
        assert published == payload
    assert summary.model_call_count == len(completions.requests)
    assert summary.model_call_count == 3 + summary.text_model_call_count
    assert row["visual_model_call_count"] == 1
    assert row["av_model_call_count"] == 2 and row["audio_model_call_count"] == 0
    assert row["model_call_count"] == summary.model_call_count
    return row, completions.requests, summary, responses, facts


@pytest.mark.parametrize("original,candidate,speakers,projection_issue", [
    (SINGLE, SINGLE_FIXED, ("S1",), "direct_single_speaker_marker_missing"),
    (MULTI, MULTI_FIXED, ("S1", "S2"), "direct_dialogue_speaker_marker_missing"),
    ("An offscreen male voice (S1) says, <d>[Chinese] a</d> <Subject 1> replies, <d>[Chinese] b</d>",
     "An offscreen male voice (S1) says, <d>[Chinese] a</d> <Subject 1> (S2) replies, <d>[Chinese] b</d>",
     ("S1", "S2"), "direct_dialogue_speaker_marker_missing"),
    (MULTI.replace("A woman says", "A woman (S1) says"), MULTI_FIXED, ("S1", "S2"), "direct_dialogue_speaker_marker_mismatch"),
])
def test_polish_applies_only_sx_and_rescues_marker_projection(
    tmp_path, monkeypatch, original, candidate, speakers, projection_issue,
):
    raw_polish = json.dumps({"shot1_caption": candidate, "needs_review": False})
    row, requests, summary, pending, facts = _case(
        tmp_path, monkeypatch, original, raw_polish, speakers,
    )
    assert not pending
    assert summary.ready_count == 1 and summary.text_model_call_count == 1
    assert row["failure_issues"] == []
    assert row["speaker_marker_polish_attempted"] and row["speaker_marker_polish_applied"]
    assert row["speaker_marker_polish_needs_review"] is False
    assert row["speaker_marker_polish_error"] is None
    assert row["speaker_marker_polish_raw_response"] == raw_polish
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == candidate
    assert not any(
        warning in mb._SPEAKER_MARKER_POLISH_ISSUES
        for diagnostic in row["diagnostics"] for warning in diagnostic["warnings"]
    )
    request = requests[-1]
    assert all(isinstance(message["content"], str) for message in request["messages"])
    assert len(request["messages"]) == 2
    assert not any(key in json.dumps(request) for key in ("video_url", "audio_url", "image_url", "use_audio_in_video"))
    assert request["temperature"] == 0 and request["max_completion_tokens"] == 2048
    assert request["reasoning_effort"] == "none"
    assert request["extra_body"] == {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    user = json.loads(request["messages"][-1]["content"])
    assert user["authoritative_speaker_facts"] == facts
    assert user["dialogue_marker_targets"] == [
        {"dialogue_index": index, "speaker_id": speaker}
        for index, speaker in enumerate(speakers, 1)
    ]
    assert user["existing_shot_caption"] == original
    assert user["speaker_marker_projection_issues"] == [projection_issue]
    assert user["allowed_h3_reference_labels"] == ["<Picture 1>", "<Subject 1>"]
    assert request["response_format"]["json_schema"]["name"] == "MimoSpeakerMarkerPolish"
    diagnostic = row["diagnostics"][-1]
    assert diagnostic["input_modality"] == "speaker_marker_text_only"
    assert all(diagnostic["usage"][f"{kind}_tokens"] is None for kind in ("image", "video", "audio"))
    assert row["backend_provenance"]["speaker_marker_polish_prompt_version"] == mb.MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION


def test_polish_v4_prompt_makes_aligned_speaker_projection_unambiguous():
    assert mb.MIMO25_PROMPT_VERSION == "h3_mimo25_speech_assembly_v45"
    assert mb.MIMO25_BACKEND_VERSION == "r2v.h3.mimo25_backend.55"
    assert mb.MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION == "h3_mimo25_speaker_marker_polish_v4"
    prompt = mb.SPEAKER_MARKER_POLISH_PROMPT
    assert "exactly one distinct speaker_id" in prompt
    assert "missing its speaker marker is NOT ambiguous" in prompt
    assert "smallest natural location and set needs_review=false" in prompt
    assert "Subject identity, visual binding, pronouns, or on/offscreen presentation" in prompt
    assert "needs_review=true is only for cases where the supplied authoritative speaker sequence itself" in prompt
    assert 'AUTHORITATIVE: [{"speaker_id":"S1","text":"..."}]' in prompt
    assert "EXISTING: He says, <d>[Chinese] ...</d>" in prompt
    assert "CORRECT: He (S1) says, <d>[Chinese] ...</d>" in prompt
    assert 'Keep "He says" and all other prose unchanged' in prompt
    assert "1-based chronological dialogue index" in prompt
    assert "if a different known Sx is present, replace it" in prompt
    assert "Do not question or infer the mapping" in prompt
    assert "already uniquely aligned. Set needs_review=false" in prompt
    assert '<Subject 1> (S2) replies, <d>[Chinese] b</d>' in prompt
    assert "MUST appear in the natural lead-in immediately before its corresponding <d> block" in prompt
    assert "Never place an (Sx) marker after the closing </d> tag" in prompt
    assert "visible to the reader before that dialogue begins" in prompt
    assert "existing speaker lead-in without changing any other prose" in prompt
    for lead in ("<Subject 1>", "She"):
        assert f"WRONG: {lead} responds, <d>[Chinese] b</d> (S2)" in prompt
        assert f"CORRECT: {lead} (S2) responds, <d>[Chinese] b</d>" in prompt


@pytest.mark.parametrize("trailing", [False, True])
def test_polish_marker_must_precede_its_dialogue(tmp_path, monkeypatch, trailing):
    original = "(S1)<d>[Chinese] a</d> <Subject 1> responds, <d>[Chinese] b</d>"
    candidate = (
        original + " (S2)" if trailing
        else original.replace("<Subject 1> responds", "<Subject 1> (S2) responds")
    )
    response = json.dumps({"shot1_caption": candidate, "needs_review": False})
    row, _, summary, pending, _ = _case(
        tmp_path, monkeypatch, original, response, ("S1", "S2"),
    )
    assert not pending
    assert summary.model_call_count == 4 and summary.text_model_call_count == 1
    assert row["speaker_marker_polish_applied"] is (not trailing)
    assert row["speaker_marker_polish_raw_response"] == response
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == (original if trailing else candidate)
    if trailing:
        assert row["status"] == "failed"
        assert "speaker_marker_polish_unresolved" in row["diagnostics"][-1]["warnings"]
        assert {issue["code"] for issue in row["failure_issues"]} == {"direct_dialogue_speaker_marker_missing"}
    else:
        assert row["status"] == "ready" and row["failure_issues"] == []


@pytest.mark.parametrize("speakers,caption", [
    (("S1",), "She says <d>[Chinese] a</d> Then <d>[Chinese] b</d> Finally <d>[Chinese] c</d>"),
    (("S1", "S1"), SINGLE),
])
def test_unaligned_dialogue_never_polishes_or_rewrites(tmp_path, monkeypatch, speakers, caption):
    row, _, summary, pending, _ = _case(
        tmp_path, monkeypatch, caption, "MUST NOT BE USED", speakers,
    )
    assert pending == ["MUST NOT BE USED"]
    assert summary.ready_count == 1 and summary.model_call_count == 3
    assert summary.text_model_call_count == 0
    assert not row["speaker_marker_polish_attempted"]
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == caption
    assert row["failure_issues"] == []
    assert "direct_single_speaker_marker_missing" in row["diagnostics"][-1]["warnings"]


@pytest.mark.parametrize("offscreen", [False, True])
def test_raw_visible_offscreen_downgrade_preserves_audio_and_caption(tmp_path, monkeypatch, offscreen):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = SINGLE_FIXED
    grounding = payload["av_grounding"]["segment_groundings"][0]
    grounding["speech_presentation"] = "offscreen_spoken"
    grounding["evidence_codes"] = ["offscreen_audio"] if offscreen else ["visible_lip_motion"]
    with pytest.raises(ValueError, match="visible_entity"):
        mb.MimoAVAnnotationDraft.model_validate(payload)
    raw = json.dumps(payload)
    canonical, counts = mb._canonicalize_raw_annotation_payload(raw)
    assert counts == {"raw_visible_offscreen_binding_downgrade": 1}
    parsed = mb.MimoAVAnnotationDraft.model_validate_json(canonical)
    fixed = parsed.av_grounding.segment_groundings[0]
    assert fixed.binding_status == ("offscreen" if offscreen else "no_reliable_entity")
    assert fixed.speech_presentation == ("offscreen_spoken" if offscreen else "uncertain")
    assert fixed.entity_id is None and fixed.confidence == "low"
    assert fixed.evidence_codes == (["offscreen_audio"] if offscreen else ["visible_lip_motion", "insufficient_evidence"])
    assert parsed.audio_observation.model_dump(mode="json") == payload["audio_observation"]
    assert parsed.h3_semantics.model_dump(mode="json") == payload["h3_semantics"]
    assert parsed.visual_observation.model_dump(mode="json") == payload["visual_observation"]
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(raw, 8)])
    original_segments = [segment.model_dump() for segment in jobs[0].segments]
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.ready_count == 1 and row["failure_issues"] == []
    assert summary.model_call_count == len(completions.requests) == 3
    assert row["audio_model_call_count"] == 0 and row["av_model_call_count"] == 2
    assert row["text_model_call_count"] == 0
    assert row["audio_finalize_raw_response"] == assembly_raw(raw)
    assert row["annotation"]["audio_observation"] == payload["audio_observation"]
    assert row["annotation"]["h3_semantics"] == payload["h3_semantics"]
    assert [segment.model_dump() for segment in jobs[0].segments] == original_segments
    annotation = mb.MimoAVAnnotationDraft.model_validate(row["annotation"])
    assert mb.direct_speech_facts(annotation, jobs[0].segments)[0]["speaker_id"] == "S1"
    assert "deterministic_correction:raw_visible_offscreen_binding_downgrade" in row["diagnostics"][-1]["warnings"]


@pytest.mark.parametrize("presentation", ["voice_over", "device_playback", "message_voice_over", "onscreen_spoken"])
def test_raw_downgrade_does_not_expand_to_other_presentations(presentation):
    payload = json.loads(_raw())
    payload["av_grounding"]["segment_groundings"][0]["speech_presentation"] = presentation
    raw = json.dumps(payload)
    canonical, corrections = mb._canonicalize_raw_annotation_payload(raw)
    assert canonical == raw and corrections == {}


@pytest.mark.parametrize("evidence", [
    ["insufficient_evidence"],
    ["visible_lip_motion", "speaker_visible_mouth_occluded", "av_temporal_alignment",
     "voice_continuity", "speaker_turn_change", "lr_asd_support", "lr_asd_conflict",
     "source_cluster_support"],
])
def test_raw_downgrade_preserves_evidence_uniqueness_and_capacity(evidence):
    payload = json.loads(_raw())
    grounding = payload["av_grounding"]["segment_groundings"][0]
    grounding["speech_presentation"] = "offscreen_spoken"
    grounding["evidence_codes"] = evidence
    canonical, counts = mb._canonicalize_raw_annotation_payload(json.dumps(payload))
    parsed = mb.MimoAVAnnotationDraft.model_validate_json(canonical)
    assert parsed.av_grounding.segment_groundings[0].evidence_codes == evidence
    assert counts == {"raw_visible_offscreen_binding_downgrade": 1}
    assert mb._canonicalize_raw_annotation_payload(canonical) == (canonical, {})


@pytest.mark.parametrize("original,code", [
    (SINGLE_FIXED, None),
    ("(S2)<d>[Chinese] a</d>", "direct_unknown_speaker"),
    ("<Audio 1> says <d>[Chinese] a</d>", "direct_unknown_reference"),
    ("She says <d>[Chinese] a", "direct_dialogue_format"),
    ("She says <d>a</d>", "direct_dialogue_language_missing"),
    ("She says <d>[Chinese] a</d> [[segment:x]]", "direct_internal_syntax"),
    ("[Shot 1] She says <d>[Chinese] a</d>", "schema_validation"),
])
def test_clean_and_non_marker_failures_never_polish(tmp_path, monkeypatch, original, code):
    row, _, summary, pending, _ = _case(tmp_path, monkeypatch, original, "MUST NOT BE USED")
    assert pending == ["MUST NOT BE USED"]
    assert summary.text_model_call_count == 0 and summary.model_call_count == 3
    assert not row["speaker_marker_polish_attempted"]
    assert row["speaker_marker_polish_raw_response"] is None
    assert row["status"] == ("failed" if code else "ready")
    if code:
        assert code in {issue["code"] for issue in row["failure_issues"]}


@pytest.mark.parametrize("speakers,original,candidate", [
    (("S1",), SINGLE, SINGLE_FIXED),
    (("S1", "S2"), MULTI, MULTI_FIXED),
])
@pytest.mark.parametrize("failure", ["dialogue", "language", "prose", "reference", "subject", "review", "malformed", "http", "unresolved", "unknown", "truncated"])
def test_rejected_polish_preserves_original_status_and_caption(
    tmp_path, monkeypatch, speakers, original, candidate, failure,
):
    expected_warning = "speaker_marker_polish_non_marker_edit_rejected"
    response = {"shot1_caption": candidate, "needs_review": False}
    if failure == "dialogue":
        response["shot1_caption"] = candidate.replace("[Chinese] a", "[Chinese] rewritten")
    elif failure == "language":
        response["shot1_caption"] = candidate.replace("[Chinese]", "[English]")
    elif failure == "prose":
        response["shot1_caption"] = candidate.replace("says", "softly replies")
    elif failure == "reference":
        response["shot1_caption"] = "<Audio 1> " + candidate
    elif failure == "subject":
        response["shot1_caption"] = "<Subject 1> " + candidate
    elif failure == "review":
        response["needs_review"] = True
        expected_warning = "speaker_marker_polish_needs_review"
    elif failure in {"unresolved", "unknown"}:
        response["shot1_caption"] = original if failure == "unresolved" else candidate.replace("(S1)", "(S9)")
        expected_warning = "speaker_marker_polish_unresolved"
    raw_response = json.dumps(response)
    if failure in {"malformed", "http", "truncated"}:
        raw_response = OSError("synthetic text HTTP failure") if failure == "http" else ('{"shot1_caption":' if failure == "truncated" else "not JSON")
        expected_warning = "speaker_marker_polish_failed"
    row, _, summary, pending, _ = _case(tmp_path, monkeypatch, original, raw_response, speakers)
    assert not pending
    assert summary.text_model_call_count == 1 and summary.model_call_count == 4
    assert not row["speaker_marker_polish_applied"]
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == original
    assert row["status"] == ("ready" if len(set(speakers)) == 1 else "failed")
    assert expected_warning in row["diagnostics"][-1]["warnings"]
    assert row["speaker_marker_polish_raw_response"] == (None if failure == "http" else raw_response)


def test_xiaomi_polish_disables_thinking_and_has_no_media(tmp_path, monkeypatch):
    row, requests, _, _, _ = _case(
        tmp_path, monkeypatch, SINGLE,
        json.dumps({"shot1_caption": SINGLE_FIXED}), transport="xiaomi",
    )
    assert row["speaker_marker_polish_applied"]
    assert requests[-1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert requests[-1]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in requests[-1]
    assert all(isinstance(message["content"], str) for message in requests[-1]["messages"])


@pytest.mark.parametrize("original,candidate,accepted", [
    (SINGLE, SINGLE_FIXED.replace(" says", "  says"), True),
    (SINGLE, SINGLE_FIXED.replace("[Chinese] a", "[Chinese]  a"), False),
    ("<Picture 1> " + SINGLE, SINGLE_FIXED, False),
    ("<Subject 1> " + SINGLE, "<Subject 2> " + SINGLE_FIXED, False),
    (SINGLE, SINGLE_FIXED.replace("[Chinese] a", "[Chinese] (S1) a"), False),
])
def test_marker_only_guard_preserves_dialogue_bytes_and_non_marker_tokens(original, candidate, accepted):
    assert mb._speaker_marker_only_edit(original, candidate) is accepted


def test_token_limited_polish_keeps_single_speaker_ready(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = SINGLE
    raw_polish = json.dumps({"shot1_caption": SINGLE_FIXED, "needs_review": False})
    backend, completions, stems, jobs = _backend(
        tmp_path, shadow, [(json.dumps(payload), 8)], polish_responses=[raw_polish],
    )
    original = completions.create

    def create(**request):
        completion = original(**request)
        if request["response_format"]["json_schema"]["name"] == "MimoSpeakerMarkerPolish":
            completion.choices[0].finish_reason = "length"
        return completion

    completions.create = create
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.ready_count == 1 and summary.model_call_count == 4
    assert not row["speaker_marker_polish_applied"]
    assert row["speaker_marker_polish_raw_response"] == raw_polish
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == SINGLE
    assert "speaker_marker_polish_failed" in row["diagnostics"][-1]["warnings"]
    assert row["diagnostics"][-1]["finish_reason"] == "length"


def test_polish_summary_audit_and_qa_keep_av_raw_separate(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = SINGLE
    raw = json.dumps(payload)
    polish = json.dumps({"shot1_caption": SINGLE_FIXED, "needs_review": False})
    backend, _, stems, jobs = _backend(
        tmp_path, shadow, [(raw, 8)] * 3, polish_responses=[polish] * 3,
    )
    summary = _run(shadow, backend, stems, jobs)
    assert (summary.audio_model_call_count, summary.av_model_call_count, summary.text_model_call_count, summary.model_call_count) == (0, 6, 3, 12)
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    first = data["clips"][0]
    assert first["reconcile"]["audio_finalize_raw_response"] == assembly_raw(raw)
    assert first["reconcile"]["speaker_marker_polish_raw_response"] == polish
    assert first["final_h3"]["status"] == "ready"
    assert SINGLE_FIXED in first["final_h3"]["text"]
    page = (shadow / "qa/review.html").read_text()
    assert 'details("Speaker marker polish"' in page
    assert 'details("Turn 3 audio finalizer raw response"' in page
    for field in ("attempted", "applied", "needs_review", "error", "raw_response"):
        assert f"result.speaker_marker_polish_{field}" in page
    row = _records(shadow)[0]
    row["text_model_call_count"] = 0
    with pytest.raises(ValueError, match="counts differ"):
        stem.MimoStemReconcileRecord.model_validate(row)
    values = summary.model_dump()
    values["text_model_call_count"] = 0
    with pytest.raises(ValueError, match="counts"):
        stem.MimoStemReconcileSummary.model_validate(values)
