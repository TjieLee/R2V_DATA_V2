"""Synthetic direct-template, single-AV-call and immutable rendering regressions."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_POLICY_VERSION,
    MIMO25_SCHEMA_VERSION,
    MimoAVAnnotationDraft,
    MimoH3Shot,
    MimoVisualBlock,
    OpenAIMimo25Backend,
    _synthetic_icl_messages,
    validate_template_inventory,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3MaterializationContractError,
    _materialize_sample,
)
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _backend,
    _job_fixture,
    _playback_inputs,
    _record_fixture,
    _validate,
)


def _with_template(text):
    values = _annotation().model_dump(mode="json")
    values["h3_projection"]["shots"][0]["description_template"] = text
    return MimoAVAnnotationDraft.model_validate(values)


def test_direct_schema_no_visual_times_or_typed_timeline():
    assert MIMO25_SCHEMA_VERSION.endswith(".16")
    assert MIMO25_POLICY_VERSION.endswith("_v17")
    assert set(MimoH3Shot.model_json_schema()["properties"]) == {
        "shot_index", "start_time", "description_template",
    }
    assert set(MimoVisualBlock.model_json_schema()["properties"]) == {"block_id", "text"}
    with pytest.raises(ValidationError):
        MimoH3Shot(shot_index=1, timeline_parts=[{"type": "visual", "block_id": "v1"}])
    with pytest.raises(ValidationError):
        MimoVisualBlock(block_id="v1", text="A person sits.", start_time=0, end_time=1)
    values = _annotation().model_dump(mode="json")
    values["schema_version"] = "r2v.h3.mimo25_av_annotation.15"
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(values)


@pytest.mark.parametrize("text,code", [
    ("View. [[audio_event:ae1]]", "speech_placeholder_inventory_mismatch"),
    ("[[segment_1]] [[segment_1]] [[audio_event:ae1]]", "speech_placeholder_inventory_mismatch"),
    ("[[segment_9]] [[audio_event:ae1]]", "speech_placeholder_inventory_mismatch"),
    ("[[segment_2]] [[segment_1]] [[audio_event:ae1]]", "speech_placeholder_inventory_mismatch"),
    ("[[segment_1]]", "missing_audio_event_placeholder"),
    ("[[segment_1]] [[audio_event:ae9]]", "unknown_audio_event_placeholder"),
    ("[[segment_1]] [[audio_event:ae1]] [[audio_event:ae1]]", "duplicate_audio_event_placeholder"),
    ("[[segment_1]] [[audio_event:ae1]] [[unknown]]", "unknown_pipeline_placeholder"),
    ("[[segment_1]] [[audio_event:ae1]] <d>[Chinese] invented</d>", "draft_contains_pipeline_owned_syntax"),
    ("[[segment_1]] [[audio_event:ae1]] (S1)", "draft_contains_pipeline_owned_syntax"),
    ("[[segment_1]] [[audio_event:ae1]] S2", "draft_contains_pipeline_owned_syntax"),
    ("The person says [[segment_1]] [[audio_event:ae1]]", "draft_prefixes_complete_speech_placeholder"),
])
def test_placeholder_inventory(text, code):
    expected = ["segment_1", "segment_2"] if "segment_2" in text else ["segment_1"]
    issues = validate_template_inventory(_with_template(text), expected)
    assert code in {i.code for i in issues}


def test_event_chronology_and_non_diegetic_exclusion():
    values = _annotation().model_dump(mode="json")
    events = values["audio_observation"]["audio_semantics"]["temporal_non_speech_events"]
    events.append({**events[0], "event_id": "ae2", "approximate_start_time": 0.8,
                   "approximate_end_time": 0.9})
    values["h3_projection"]["shots"][0]["description_template"] = (
        "View. [[segment_1]] [[audio_event:ae2]] [[audio_event:ae1]]"
    )
    issues = validate_template_inventory(MimoAVAnnotationDraft.model_validate(values), ["segment_1"])
    assert "audio_event_placeholder_order_mismatch" in {i.code for i in issues}
    events[1]["category"] = "non_diegetic_music"
    values["audio_observation"]["audio_semantics"].update(
        non_diegetic_music_status="present", non_diegetic_music="A quiet musical pad plays.",
    )
    values["h3_projection"]["shots"][0]["description_template"] = "View. [[segment_1]] [[audio_event:ae1]]"
    assert not validate_template_inventory(MimoAVAnnotationDraft.model_validate(values), ["segment_1"])


@pytest.mark.parametrize("early", [False, True])
def test_natural_template_exact_speech_and_official_sections(tmp_path, early):
    sample, job, record = _playback_inputs(tmp_path)
    values = record.annotation.model_dump(mode="json")
    template = ("A woman turns toward the group. [[segment_1]] Two people look toward her. "
                "[[audio_event:ae1]] She leans back.")
    if early:
        template = "[[segment_1]] A woman turns toward the group. [[audio_event:ae1]] She leans back."
    values["h3_projection"]["shots"][0]["description_template"] = template
    annotation = MimoAVAnnotationDraft.model_validate(values)
    assert not _validate(annotation, segment_intervals={"segment_1": (1, 2)}, target_duration_seconds=5)
    record = _record_fixture(tmp_path, annotation, job=job)
    _, rendered, _ = _materialize_sample(sample, job, record, conditioning_variant="visual_only")
    assert "<d>[English] Exact, text!</d>" in rendered
    assert "[[" not in rendered
    assert rendered.index("turns toward the group") < rendered.index("She leans back")
    if not early:
        assert rendered.index("turns toward the group") < rendered.index("<d>") < rendered.index("Two people")
    for section in ("subject_definitions", "summary", "retention_analysis",
                    "detailed_description", "overall_soundscape", "non_diegetic_music"):
        assert rendered.count(section + ":") == 1


def test_wrong_shot_fails_but_no_sentence_alignment(tmp_path):
    _, _, record = _playback_inputs(tmp_path, two_shots=True)
    values = record.annotation.model_dump(mode="json")
    values["h3_projection"]["shots"][0]["description_template"] = "A seated person waits."
    values["h3_projection"]["shots"][1]["description_template"] = (
        "Another view. [[segment_1]] [[audio_event:ae1]]"
    )
    issues = _validate(MimoAVAnnotationDraft.model_validate(values),
                       segment_intervals={"segment_1": (1, 2)}, target_duration_seconds=5)
    assert "speech_placeholder_wrong_shot" in {i.code for i in issues}


def test_two_speech_clauses_interleave_without_changing_other_sections(tmp_path):
    from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
    from r2v_data_v2.h3.mimo25_av_reconcile import _job

    sample, job, record = _playback_inputs(tmp_path)
    job_values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    second = {**job_values["segments"][0], "segment_id": "segment_2",
              "start_time": 2.1, "end_time": 2.8, "source_start_sample": 67200,
              "source_end_sample": 89600, "asr_text": "A second utterance."}
    job_values["segments"].append(second)
    job = _job(job_values)
    sample_values = sample.model_dump(mode="json")
    sample_values["speech_segments"].append({
        **sample_values["speech_segments"][0], "segment_id": "segment_2",
        "start_time": 2.1, "end_time": 2.8, "source_start_sample": 67200,
        "source_end_sample": 89600, "text": "A second utterance.",
    })
    sample = FinalH3SampleV2.model_validate(sample_values)
    values = record.annotation.model_dump(mode="json")
    for section, key in (("visual_observation", "segment_views"),
                         ("audio_observation", "segment_decisions"),
                         ("av_grounding", "segment_groundings")):
        rows = values[section][key]
        rows.append({**rows[0], "segment_id": "segment_2"})
    shot = values["h3_projection"]["shots"][0]
    shot["description_template"] = (
        "A woman turns toward the group. [[segment_1]] Two people look toward her. "
        "[[segment_2]] She leans back. [[audio_event:ae1]]"
    )
    annotation = MimoAVAnnotationDraft.model_validate(values)
    rendered = _materialize_sample(sample, job, _record_fixture(tmp_path, annotation, job=job))[1]
    assert rendered.index("turns toward") < rendered.index("<d>[English] Exact, text!</d>")
    assert rendered.index("Exact, text!</d>") < rendered.index("Two people") < rendered.index("A second utterance.</d>")
    assert rendered.index("A second utterance.</d>") < rendered.index("She leans back")
    shot["description_template"] = (
        "[[segment_1]] [[segment_2]] A woman turns toward the group. "
        "Two people look toward her. She leans back. [[audio_event:ae1]]"
    )
    alternate = _materialize_sample(sample, job, _record_fixture(
        tmp_path, MimoAVAnnotationDraft.model_validate(values), job=job,
    ))[1]
    def without_detail(text):
        before, rest = text.split("detailed_description:\n", 1)
        return before, rest.split("overall_soundscape:\n", 1)[1]
    assert without_detail(rendered) == without_detail(alternate)


@pytest.mark.parametrize("bad_first", [False, True])
def test_one_av_call_or_one_full_av_recheck(tmp_path, bad_first):
    good = _annotation().model_dump_json()
    values = json.loads(good)
    values["audio_observation"]["audio_semantics"].update(
        overall_soundscape_status="absent", overall_soundscape=None, complete_silence_verified=False,
        temporal_non_speech_events=[],
    )
    values["h3_projection"]["shots"][0]["description_template"] = "A person sits. [[segment_1]]"
    bad = json.dumps(values)
    assert "soundscape_absence_requires_explicit_original_av_judgment" in {
        i.code for i in _validate(MimoAVAnnotationDraft.model_validate_json(bad))
    }
    responses = [(bad, 8, "stop")] if bad_first else []
    backend, client = _backend(tmp_path, responses + [(good, 8, "stop")], transport="sglang")
    result = backend.reconcile(
        _job_fixture(tmp_path), segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"}, allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == len(client.requests) == (2 if bad_first else 1)
    assert result.recheck_count == int(bad_first)
    assert result.annotation == _annotation()
    assert not hasattr(backend, "compose")
    for request in client.requests:
        assert request["extra_body"]["use_audio_in_video"] is True
        assert any(item["type"] == "video_url" for item in request["messages"][-1]["content"])


def test_materializer_defensively_rejects_missing_placeholder(tmp_path):
    sample, job, record = _playback_inputs(tmp_path)
    values = record.annotation.model_dump(mode="json")
    values["h3_projection"]["shots"][0]["description_template"] = "View. [[audio_event:ae1]]"
    with pytest.raises(MimoH3MaterializationContractError, match="speech_placeholder_inventory_mismatch"):
        _materialize_sample(sample, job, _record_fixture(
            tmp_path, MimoAVAnnotationDraft.model_validate(values), job=job,
        ))


def test_no_production_compositor_and_current_icl():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "r2v_data_v2/h3/mimo25_text_compositor.py").exists()
    for path in (root / "r2v_data_v2").rglob("*.py"):
        source = path.read_text()
        for name in ("split_visual_sentences", "DescriptionAtom", "CompositorInput", "TextComposition"):
            assert name not in source, (path, name)
    assert not hasattr(OpenAIMimo25Backend, "compose")
    example = MimoAVAnnotationDraft.model_validate_json(_synthetic_icl_messages()[1]["content"])
    template = example.h3_projection.shots[0].description_template
    assert template.index("[[segment_0001]]") < template.index("[[audio_event:ae1]]")
