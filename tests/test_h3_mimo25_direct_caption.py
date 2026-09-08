from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_ICL_VERSION,
    SYSTEM_PROMPT,
    VISUAL_SYSTEM_PROMPT,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    _official_detailed_description_icl_messages,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _backend,
    _job_fixture,
    _record_fixture,
    _sample,
)

LABELS = {"<Subject 1>", "<Picture 1>"}
SPEECH = [{"segment_id": "segment_1", "speaker_id": "S1", "language": "English", "text": "Exact, text!"}]


def _run(backend, job):
    return backend.reconcile(
        job, segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"}, allowed_reference_labels=LABELS,
    )


def test_official_examples_preserved_in_two_turn_prefix(tmp_path):
    job = _job_fixture(tmp_path)
    visual, speech, finalized = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(visual, 0), (speech, 8), (finalized, 8)],
                             icl="official_ref2va_v1")
    result = _run(backend, job)
    defaults = MimoBackendConfig(media_resolver=backend.config.media_resolver, api_key="test")
    assert (defaults.thinking, defaults.icl, defaults.temperature) == (
        "disabled", "official_ref2va_v1", 0.0,
    )
    assert result.model_call_count == len(calls.requests) == 3
    messages = calls.requests[0]["messages"]
    assert calls.requests[1]["messages"][:len(messages)] == messages
    assert [m["role"] for m in messages] == ["system", *(["user", "assistant"] * 4), "user"]
    assert messages[1:-1] == _official_detailed_description_icl_messages()
    guide = (Path(__file__).parents[1] / "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md").read_text()
    example = guide.split("## 7. Complete Example", 1)[1].split("```text", 1)[1].split("```", 1)[0].strip()
    detailed = example.split("detailed_description:\n", 1)[1]
    opening, body = detailed.split("\n[Shot 1] ", 1)
    body = body.split("\n[Shot 2]", 1)[0]
    assert json.loads(messages[2]["content"]) == {
        "style_opening": opening, "shot1_caption": body,
        "overall_soundscape": example.split("\noverall_soundscape:\n", 1)[1].split("\n\nnon_diegetic_music:\n", 1)[0],
        "non_diegetic_music": example.split("\nnon_diegetic_music:\n", 1)[1],
    }
    assert "official H3 writing/field-boundary demonstration" in messages[1]["content"]
    assert "not the response-schema definition" in messages[1]["content"]
    for excluded in ("subject_definitions:", "summary:", "retention_analysis:",
                     "overall_soundscape:", "non_diegetic_music:", "[Shot 1]", "[Shot 2]", "[Shot 3]"):
        assert excluded not in messages[2]["content"]
    assert not re.search(r"\d{2}:\d{2}(?:\.\d+)?", messages[2]["content"])
    base = (Path(__file__).parents[1] / "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md").read_text()
    for number, assistant in zip((2, 3, 4), messages[4:9:2], strict=True):
        section = base.split(f"### Case {number}:", 1)[1].split("### Case ", 1)[0]
        source = section.split("```text\n", 1)[1].split("\n```", 1)[0]
        lines = source.splitlines()
        expected = {
            "shot1_caption": next(line.removeprefix("integrated_multimodal_description: [Shot 1] ") for line in lines if line.startswith("integrated_multimodal_description:")),
            "overall_soundscape": next(line.removeprefix("overall_soundscape: ") for line in lines if line.startswith("overall_soundscape:")),
            "non_diegetic_music": next(line.removeprefix("non_diegetic_music: ") for line in lines if line.startswith("non_diegetic_music:")),
        }
        assert json.loads(assistant["content"]) == expected
    examples = [json.loads(message["content"]) for message in messages[2:9:2]]
    assert any(item["non_diegetic_music"] == "N/A" for item in examples)
    positive = [item for item in examples if item["non_diegetic_music"] != "N/A"]
    assert len(positive) == 2
    assert all(item["non_diegetic_music"] not in item["shot1_caption"] for item in positive)
    assert "cello" in examples[1]["non_diegetic_music"] and "cello" not in examples[1]["shot1_caption"]
    assert "electronic pulse" in examples[3]["non_diegetic_music"] and "electronic pulse" not in examples[3]["shot1_caption"]
    for message in messages[1:-1]:
        assert not re.search(r"\[Shot \d+\]|\d{2}:\d{2}|\d+\.\d+[- ]second", message["content"])
        assert "For the target video, at" not in message["content"]
        assert "How the reference pictures align" not in message["content"]
        assert "Case 1: T2VA" not in message["content"]
    assert backend.provenance.icl_version == MIMO25_ICL_VERSION == "h3_official_ref2va_detailed_shot1_v4"
    assert backend.provenance.prompt_version == "h3_mimo25_speech_assembly_v43"
    assert backend.provenance.schema_version == "r2v.h3.mimo25_backend.53"
    assert backend.provenance.annotation_schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert backend.provenance.materializer_version == "h3_mimo25_materializer_v24"
    assert "The pipeline owns [Shot 1]" in SYSTEM_PROMPT
    assert "Do not repeat the marker before every utterance" not in SYSTEM_PROMPT
    assert "style_opening is one concise global style/camera/lighting sentence" in VISUAL_SYSTEM_PROMPT


def test_direct_caption_materializes_without_reconstructing_prose(tmp_path):
    job = _job_fixture(tmp_path)
    sample = _sample(tmp_path)
    annotation = _annotation()
    record = _record_fixture(tmp_path, annotation, job=job)
    raw = annotation.h3_semantics.shot1_caption
    _, final, warnings = _materialize_sample(sample, job, record)
    direct = final.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    opening = annotation.h3_semantics.style_opening + " [Shot 1] "
    assert direct.startswith(opening)
    assert direct.count("[Shot 1]") == 1 and "[Shot 2]" not in direct
    direct = direct[len(opening):]
    assert re.sub(r"<d>.*?</d>", "<d></d>", direct) == re.sub(r"<d>.*?</d>", "<d></d>", raw)
    assert "<d>[English] Exact, text!</d>" in direct
    assert "asr_dialogue_payload_corrected" in warnings
    assert annotation.h3_semantics.shot1_caption == raw
    sections = ["subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape", "non_diegetic_music"]
    offsets = [final.index(name + ":\n") for name in sections]
    assert offsets == sorted(offsets)
    assert "[[" not in final
    assert "h3_projection" not in MimoAVAnnotationDraft.model_json_schema()["properties"]


@pytest.mark.parametrize("label", ["<Subject 9>", "<Audio 1>", "(S9)", "<Picture 0>"])
def test_unknown_labels_detected(label):
    text = f"{label} <Subject 1> (S1) asks, <d>[English] Exact, text!</d>"
    corrected, issues, _ = protect_direct_dialogue(text, SPEECH, allowed_labels=LABELS)
    assert issues and corrected == text


@pytest.mark.parametrize("text", [
    "Only a room.",
    "(S1) asks <d>[English] Exact, text!</d> then (S1) says <d>extra</d>",
])
def test_dialogue_count_cannot_be_guessed(text):
    corrected, issues, _ = protect_direct_dialogue(text, SPEECH, allowed_labels=LABELS)
    assert corrected == text
    assert "direct_dialogue_inventory_mismatch" in {issue.code for issue in issues}


def test_recognizable_reordered_dialogue_is_not_rewritten():
    speech = [*SPEECH, {**SPEECH[0], "segment_id": "segment_2", "text": "Second."}]
    text = "(S1) says <d>[English] Second.</d> then (S1) replies <d>[English] Exact, text!</d>"
    corrected, issues, _ = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert issues and corrected == text


def test_one_full_av_recheck_remains_the_limit(tmp_path):
    job = _job_fixture(tmp_path)
    valid = _annotation().model_dump_json()
    backend, calls = _backend(tmp_path, [("not JSON", 8), (valid, 8)])
    assert _run(backend, job).model_call_count == len(calls.requests) == 2
    backend, calls = _backend(tmp_path, [("not JSON", 8), ("still not JSON", 8)])
    with pytest.raises(MimoBackendFailure):
        _run(backend, job)
    assert len(calls.requests) == 2


def test_internal_absence_and_audio_wording_do_not_block_caption(tmp_path):
    job = _job_fixture(tmp_path)
    values = _annotation().model_dump(mode="json")
    values["audio_observation"]["audio_semantics"].update(
        overall_soundscape_status="absent", overall_soundscape=None, complete_silence_verified=False,
    )
    values["h3_semantics"]["overall_soundscape"] = "Dialogue and background music fill the room."
    annotation = MimoAVAnnotationDraft.model_validate(values)
    backend, calls = _backend(tmp_path, [(annotation.model_dump_json(), 8)])
    result = _run(backend, job)
    assert result.model_call_count == len(calls.requests) == 1
    assert "direct_soundscape_contains_music_or_speech" in result.diagnostics[0].warnings


def test_one_shot_schema_has_no_model_owned_timing():
    schema = MimoAVAnnotationDraft.model_json_schema()
    visual = schema["$defs"]["MimoVisualObservation"]
    assert set(visual["properties"]) == {"visual_blocks", "segment_views"}
    assert "MimoVisualShotObservation" not in schema["$defs"]
    assert "start_time" not in json.dumps(visual)
    semantics = schema["$defs"]["MimoH3Semantics"]["properties"]
    assert {"style_opening", "shot1_caption"} <= set(semantics)
    assert "detailed_description" not in semantics


@pytest.mark.parametrize("field,marker", [
    ("style_opening", "[Shot 1]"), ("shot1_caption", "[Shot 2]"),
])
def test_model_shot_markers_fail_and_use_only_existing_recheck(tmp_path, field, marker):
    job = _job_fixture(tmp_path)
    values = _annotation().model_dump(mode="json")
    values["h3_semantics"][field] += " " + marker
    with pytest.raises(ValidationError, match="pipeline-owned shot markers"):
        MimoAVAnnotationDraft.model_validate(values)
    backend, calls = _backend(tmp_path, [
        (json.dumps(values), 8), (_annotation().model_dump_json(), 8),
    ])
    assert _run(backend, job).model_call_count == len(calls.requests) == 2


@pytest.mark.parametrize("speakers,introductions,valid", [
    (["S1"] * 4, ["<Subject 1> (S1) speaks softly,", "He (S1) continues,", "He (S1) then adds,", "Finally, he (S1) concludes,"], True),
    (["S1"] * 4, ["The woman speaks,"] * 4, False),
    (["S1", "S1"], ["(S1) speaks,", "He continues,"], False),
    (["S1", "S1", "S2", "S2"], ["(S1) speaks,", "He (S1) adds,", "An off-screen voice (S2) replies,", "It (S2) continues,"], True),
    (["S1", "S2"], ["(S1) listens to (S2), then speaks,", "(S2) answers (S1) with a pause,"], True),
    (["S1", "S2"], ["(S1) speaks,", "A voice replies,"], False),
    (["S1", "S1"], ["He speaks,", "(S1) continues,"], False),
])
def test_each_dialogue_needs_its_speaker_in_event_lead_in(speakers, introductions, valid):
    speech = [
        {"segment_id": f"segment_{i}", "speaker_id": speaker,
         "language": "Chinese", "text": f"原文{i}"}
        for i, speaker in enumerate(speakers)
    ]
    text = " ".join(f"{intro} <d>[Chinese] 原文{i}</d>"
                    for i, intro in enumerate(introductions))
    corrected, issues, _ = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert corrected == text
    assert (not issues) == valid
    if not valid:
        assert {issue.code for issue in issues} == {"direct_dialogue_speaker_marker_missing"}


def test_long_natural_lead_in_preserved_with_exact_asr_payload():
    lead = ("After watching the other person for a moment, <Subject 1> (S1) "
            "turns away and answers quietly, ")
    tail = " while lowering his gaze."
    text = lead + "<d>[Chinese] wrong text</d>" + tail
    speech = [{**SPEECH[0], "language": "Chinese", "text": "EXACT ASR"}]
    corrected, issues, warnings = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert not issues
    assert corrected == lead + "<d>[Chinese] EXACT ASR</d>" + tail
    assert warnings == ["asr_dialogue_payload_corrected"]


def test_old_caption_excluded_and_single_authoritative_payload(tmp_path):
    job = _job_fixture(tmp_path)
    sentinel = "OLD_CAPTION_MUST_NOT_ANCHOR_RECAPTION"
    old_instruction = f"summary:\nOld scene\ndetailed_description:\n{sentinel}\noverall_soundscape:\nOld audio"
    job = job.model_copy(update={"r2v_instruction": old_instruction})
    backend, calls = _backend(tmp_path, [("bad JSON", 8), (_annotation().model_dump_json(), 8)],
                              transport="sglang", icl="official_ref2va_v1")
    result = _run(backend, job)
    assert result.model_call_count == len(calls.requests) == 2
    assert job.r2v_instruction == old_instruction
    for request in calls.requests:
        assert sentinel not in json.dumps(request["messages"])
        text = request["messages"][-1]["content"][-1]["text"]
        assert text.count('"subject_definition_requirements"') == 1
        assert text.count('"reference_image_mapping"') == 1
        assert text.count('"segments"') == 1
        assert "authoritative_speech_facts" not in text
        assert "MANDATORY MACHINE CONTRACT" not in text
        assert "r2v_instruction" not in text
    contract = backend.build_compact_task_contract(job)
    assert contract["segments"] == [segment.model_dump(mode="json") for segment in job.segments]
    assert calls.requests[1]["messages"][-1]["content"][-1]["text"].startswith(
        "Reinspect the same AV and fix ONLY these hard issues:"
    )
