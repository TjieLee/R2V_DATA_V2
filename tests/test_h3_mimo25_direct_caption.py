from __future__ import annotations

import re
from pathlib import Path

import pytest

from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_ICL_VERSION,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    _official_icl_messages,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
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


def test_official_example_and_single_av_message_order(tmp_path):
    job = _job_fixture(tmp_path)
    backend, calls = _backend(tmp_path, [(_annotation().model_dump_json(), 8)],
                             icl="official_ref2va_v1")
    result = _run(backend, job)
    defaults = MimoBackendConfig(media_resolver=backend.config.media_resolver, api_key="test")
    assert (defaults.thinking, defaults.icl, defaults.temperature) == (
        "disabled", "official_ref2va_v1", 0.0,
    )
    assert result.model_call_count == len(calls.requests) == 1
    messages = calls.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == _official_icl_messages()
    guide = (Path(__file__).parents[1] / "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md").read_text()
    example = guide.split("## 7. Complete Example", 1)[1].split("```text", 1)[1].split("```", 1)[0].strip()
    assert messages[2]["content"] == example
    assert "Samoyed" in example and "canned" in example.lower()
    assert "[[" not in example
    assert backend.provenance.icl_version == MIMO25_ICL_VERSION


def test_direct_caption_materializes_without_reconstructing_prose(tmp_path):
    job = _job_fixture(tmp_path)
    sample = _sample(tmp_path)
    annotation = _annotation()
    record = _record_fixture(tmp_path, annotation, job=job)
    raw = annotation.h3_semantics.detailed_description
    _, final, warnings = _materialize_sample(sample, job, record)
    direct = final.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    assert re.sub(r"<d>.*?</d>", "<d></d>", direct) == re.sub(r"<d>.*?</d>", "<d></d>", raw)
    assert "<d>[English] Exact, text!</d>" in direct
    assert "asr_dialogue_payload_corrected" in warnings
    assert annotation.h3_semantics.detailed_description == raw
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
