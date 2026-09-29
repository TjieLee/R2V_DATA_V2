from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3 import mimo25_single_backend as single
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoMediaResolver,
    direct_speech_facts,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    _validated_auxiliary_stems,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import load_stem_shadow
from tests.test_h3_audio_shadow_qa import _fixture
from tests.test_h3_mimo25_av_shadow import _Completions
from tests.test_h3_mimo25_raw_stems import _single_raw


def _compact_payload() -> dict:
    return json.loads(_single_raw())


def _job(*, segments=None, subjects=None):
    segment = SimpleNamespace(
        segment_id="segment_0001", asr_status="transcribed",
        asr_language="English", asr_text="exact transcript",
        source_speaker_cluster_id="cluster_a", start_time=0.0, end_time=1.0,
    )
    return SimpleNamespace(
        segments=segments if segments is not None else [segment],
        reference_subjects=subjects if subjects is not None else [SimpleNamespace(subject_label="<Subject 1>")],
    )


def _validated(payload: dict, job=None, *, entity_ids=None):
    draft = single.MimoSingleCompactAnnotationDraft.model_validate(payload)
    issues, warnings = single._validate_single_compact_annotation(
        draft, job or _job(), allowed_entity_ids=entity_ids or {"e1"},
        allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
    )
    return draft, issues, warnings


def test_compact_schema_removes_self_reported_evidence_and_uses_new_version():
    schema = json.dumps(single.MimoSingleCompactAnnotationDraft.model_json_schema())
    for forbidden in (
        "orientation", "face_visibility", "mouth_visibility", "speech_correlated_articulation",
        "confidence", "audio_evidence_codes", "evidence_codes", "speech_presentation",
    ):
        assert forbidden not in schema
    assert single.SINGLE_COMPACT_SCHEMA_VERSION == "r2v.h3.mimo26_single_compact.1"
    assert single.MimoSingleCompactAnnotationDraft.model_validate(_compact_payload())
    assert "schema_version" in single.MimoSingleCompactAnnotationDraft.model_json_schema()["required"]


def test_compact_inventory_visibility_and_dialogue_are_hard_contracts():
    payload = _compact_payload()
    assert _validated(payload)[1] == []
    for section, field in (
        ("visual_observation", "segment_views"),
        ("audio_observation", "segment_decisions"),
        ("av_grounding", "segment_groundings"),
    ):
        broken = json.loads(json.dumps(payload))
        broken[section][field].append(broken[section][field][0])
        assert _validated(broken)[1]
    broken = json.loads(json.dumps(payload))
    broken["av_grounding"]["segment_groundings"][0]["entity_id"] = "e2"
    assert "visible_entity_absent_from_visual_segment" in {
        issue.code for issue in _validated(broken, entity_ids={"e1", "e2"})[1]
    }
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["shot1_caption"] = "<Subject 1> says nothing."
    assert "direct_transcribed_dialogue_inventory_mismatch" in {
        issue.code for issue in _validated(broken)[1]
    }
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["subject_definitions"] = []
    with pytest.raises(ValueError):
        single.MimoSingleCompactAnnotationDraft.model_validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["shot1_caption"] += " <d>[English] invented words</d>"
    assert "direct_transcribed_dialogue_inventory_mismatch" in {
        issue.code for issue in _validated(broken)[1]
    }
    broken = json.loads(json.dumps(payload))
    broken["audio_observation"]["segment_decisions"][0]["primary_speaker_group"] = "g2"
    assert "non_contiguous_speaker_groups" in {issue.code for issue in _validated(broken)[1]}
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["summary"] += " (S9)"
    assert "pipeline_owned_syntax_outside_shot_caption" in {
        issue.code for issue in _validated(broken)[1]
    }


def test_compact_group_entity_conflicts_are_warnings_without_repair():
    payload = _compact_payload()
    second = SimpleNamespace(
        segment_id="segment_0002", asr_status="empty", asr_language=None,
        asr_text=None, source_speaker_cluster_id="cluster_b", start_time=1.0, end_time=2.0,
    )
    payload["visual_observation"]["segment_views"].append({
        "segment_id": "segment_0002", "visible_entity_ids": ["e1", "e2"],
    })
    payload["audio_observation"]["segment_decisions"].append({
        "segment_id": "segment_0002", "primary_speaker_group": "g1", "delivery_style": None,
    })
    payload["av_grounding"]["segment_groundings"].append({
        "segment_id": "segment_0002", "binding_status": "visible_entity", "entity_id": "e2",
    })
    draft, issues, warnings = _validated(
        payload, _job(segments=[*_job().segments, second]), entity_ids={"e1", "e2"},
    )
    assert issues == []
    assert "compact_group_maps_multiple_entities" in warnings
    assert draft.av_grounding.segment_groundings[1].entity_id == "e2"
    payload["audio_observation"]["segment_decisions"][1]["primary_speaker_group"] = "g2"
    payload["av_grounding"]["segment_groundings"][1]["entity_id"] = "e1"
    draft, issues, warnings = _validated(
        payload, _job(segments=[*_job().segments, second]), entity_ids={"e1", "e2"},
    )
    assert issues == []
    assert "compact_entity_maps_multiple_groups" in warnings
    assert draft.av_grounding.segment_groundings[1].entity_id == "e1"


def test_compact_sx_projection_uses_transcribed_groups_and_strips_model_markers():
    payload = _compact_payload()
    text = payload["h3_semantics"]["shot1_caption"]
    block = "<d>[English] exact transcript</d>"
    assert block in text
    payload["h3_semantics"]["shot1_caption"] = text.replace(block, "(S9) " + block)
    draft, issues, _ = _validated(payload)
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, _job())
    assert isinstance(projected, MimoAVAnnotationDraft)
    assert "(S9)" not in projected.h3_semantics.shot1_caption
    assert "(S1) <d>[English] exact transcript</d>" in projected.h3_semantics.shot1_caption
    assert direct_speech_facts(projected, _job().segments)[0]["speaker_id"] == "S1"
    assert projected.visual_observation.segment_views[0].entity_observations == []
    decision = projected.audio_observation.segment_decisions[0]
    assert decision.confidence == "low"
    assert decision.audio_evidence_codes == ["insufficient_evidence"]
    assert decision.vocal_composition == "uncertain"
    assert projected.av_grounding.segment_groundings[0].evidence_codes == ["insufficient_evidence"]


def test_compact_sx_projection_repeats_same_speaker_and_uses_cluster_fallback():
    payload = _compact_payload()
    segment = _job().segments[0]
    segments = [
        segment,
        SimpleNamespace(**{**vars(segment), "segment_id": "segment_0002", "start_time": 1.0}),
        SimpleNamespace(**{**vars(segment), "segment_id": "segment_0003", "start_time": 2.0,
                           "source_speaker_cluster_id": "cluster_b"}),
    ]
    for index, group in ((2, "g1"), (3, None)):
        segment_id = f"segment_{index:04d}"
        payload["visual_observation"]["segment_views"].append({
            "segment_id": segment_id, "visible_entity_ids": ["e1"],
        })
        payload["audio_observation"]["segment_decisions"].append({
            "segment_id": segment_id, "primary_speaker_group": group, "delivery_style": "calm",
        })
        payload["av_grounding"]["segment_groundings"].append({
            "segment_id": segment_id, "binding_status": "no_reliable_entity", "entity_id": None,
        })
    block = "<d>[English] exact transcript</d>"
    payload["h3_semantics"]["shot1_caption"] += " Then another voice says " + block + " and later asks " + block
    draft, issues, _ = _validated(payload, _job(segments=segments))
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, _job(segments=segments))
    assert [fact["speaker_id"] for fact in direct_speech_facts(projected, segments)] == ["S1", "S1", "S2"]
    assert projected.h3_semantics.shot1_caption.count("(S1) <d>") == 2
    assert projected.h3_semantics.shot1_caption.count("(S2) <d>") == 1


def test_compact_zero_transcript_strips_accidental_sx_without_dialogue():
    payload = _compact_payload()
    segment = SimpleNamespace(**{
        **vars(_job().segments[0]), "asr_status": "empty", "asr_text": None,
        "asr_language": None,
    })
    payload["h3_semantics"]["shot1_caption"] = "<Subject 1> (S9) waits by the door."
    payload["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
    payload["audio_observation"]["speaker_voice_profiles"] = []
    draft, issues, _ = _validated(payload, _job(segments=[segment]))
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, _job(segments=[segment]))
    assert "(S" not in projected.h3_semantics.shot1_caption
    assert "<d>" not in projected.h3_semantics.shot1_caption
    assert direct_speech_facts(projected, [segment]) == []


def test_compact_visible_choice_is_not_remapped_or_evidence_validated():
    payload = _compact_payload()
    payload["visual_observation"]["segment_views"][0]["visible_entity_ids"] = ["e1", "e2"]
    payload["av_grounding"]["segment_groundings"][0]["entity_id"] = "e2"
    draft, issues, _ = _validated(payload, entity_ids={"e1", "e2"})
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, _job())
    assert projected.av_grounding.segment_groundings[0].entity_id == "e2"


def test_single_compact_backend_keeps_raw_response_and_legacy_annotation(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    stems = load_stem_shadow(shadow / "separation")[1]
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.build_mimo25_inventory(),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr", route="music_first",
    )[:1]
    raw = json.dumps(_compact_payload())
    completions = _Completions([(raw, 8), (raw, 8)])
    backend = single.SingleCallOpenAIMimo25Backend(
        MimoBackendConfig(
            api_key="fake", transport="sglang", model="mimo-v2.6-flash-rl", icl="none",
            media_resolver=MimoMediaResolver(
                mode="http", media_root=tmp_path, media_base_url="http://media.invalid",
            ),
        ),
        stem_records_by_clip={record.clip_uid: record for record in stems},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    result = backend.reconcile(
        jobs[0], segment_ids=[s.segment_id for s in jobs[0].segments],
        transcribed_segment_ids=[s.segment_id for s in jobs[0].segments if s.asr_status == "transcribed"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={r.picture_label for r in jobs[0].reference_images}
        | {s.subject_label for s in jobs[0].reference_subjects},
        auxiliary_audio_paths=_validated_auxiliary_stems(stems[0]),
    )
    assert len(completions.requests) == result.model_call_count == 1
    request = completions.requests[0]
    assert request["response_format"]["json_schema"]["name"] == "MimoSingleCompactAnnotationDraft"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["reasoning_effort"] == "none"
    assert request["extra_body"]["chat_template_kwargs"] == {
        "thinking": False, "enable_thinking": False,
    }
    assert result.speech_av_raw_response == raw
    assert result.raw_responses == (raw,)
    assert result.annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert "(S1) <d>" in result.annotation.h3_semantics.shot1_caption
    root = shadow / "single-v5"
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_records=stems, backend=backend, output_root=root,
        route="music_first", allow_unverified=True,
    )
    record = MimoStemReconcileRecord.model_validate_json((root / "records.jsonl").read_text().splitlines()[0])
    assert record.speech_av_raw_response == raw
    assert record.annotation is not None and record.annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"


def test_single_compact_annotation_uses_existing_h3_materializer(tmp_path, monkeypatch):
    from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample

    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    job = build_stem_reconcile_jobs(
        base_inventory=qa.build_mimo25_inventory(),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr", route="music_first",
    )[0]
    draft, issues, _ = _validated(_compact_payload(), job)
    assert issues == []
    annotation = single._project_compact_to_legacy_annotation(draft, job)
    sample_path = kwargs["audio_production_root"] / "h3/samples.jsonl"
    sample = qa.FinalH3SampleV2.model_validate_json(sample_path.read_text().splitlines()[0])
    source = job.segments[0]
    sample = qa.FinalH3SampleV2.model_validate({
        **sample.model_dump(mode="python"),
        "speech_segments": [qa.FinalQwen3SpeechSegment(
            segment_id=source.segment_id, speaker_cluster_id=source.source_speaker_cluster_id,
            entity_id=source.current_entity_id, entity_occurrence_id=source.entity_occurrence_id,
            source_start_sample=source.source_start_sample,
            source_end_sample=source.source_end_sample,
            source_sample_rate_hz=source.source_sample_rate_hz,
            start_time=source.start_time, end_time=source.end_time,
            text=source.asr_text, language=source.asr_language,
        ).model_dump(mode="python")],
    })
    corrected, rendered, _ = _materialize_sample(
        sample, job, qa._MaterializerInput(annotation, job.request_fingerprint),
    )
    assert corrected[0].text == source.asr_text
    assert "detailed_description:" in rendered
    assert "(S1) <d>[English] exact transcript</d>" in rendered


def test_single_compact_invalid_output_has_no_retry_or_polish(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    stems = load_stem_shadow(shadow / "separation")[1]
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.build_mimo25_inventory(),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr", route="music_first",
    )
    raw = json.dumps({**_compact_payload(), "visual_observation": {"visual_blocks": [], "segment_views": []}})
    completions = _Completions([(raw, 8)])
    backend = single.SingleCallOpenAIMimo25Backend(
        MimoBackendConfig(
            api_key="fake", transport="sglang", model="mimo-v2.6-flash-rl",
            media_resolver=MimoMediaResolver(mode="http", media_root=tmp_path, media_base_url="http://media.invalid"),
        ),
        stem_records_by_clip={record.clip_uid: record for record in stems},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    with pytest.raises(MimoBackendFailure) as failure:
        backend.reconcile(
            jobs[0], segment_ids=[s.segment_id for s in jobs[0].segments],
            transcribed_segment_ids=[s.segment_id for s in jobs[0].segments if s.asr_status == "transcribed"],
            allowed_entity_ids={"e1"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
            auxiliary_audio_paths=_validated_auxiliary_stems(stems[0]),
        )
    assert len(completions.requests) == failure.value.model_call_count == 1
    assert failure.value.speech_av_raw_response == raw
