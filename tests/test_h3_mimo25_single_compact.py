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


def test_v6_raw_schema_has_one_segment_inventory_and_no_entity_evidence_fields():
    schema = json.dumps(single.MimoSingleCompactAnnotationDraftV2.model_json_schema())
    assert single.SINGLE_COMPACT_SCHEMA_VERSION == "r2v.h3.mimo26_single_compact.2"
    for forbidden in (
        "segment_views", "visible_entity_ids", "entity_id", "orientation",
        "face_visibility", "mouth_visibility", "speech_correlated_articulation",
        "confidence", "audio_evidence_codes", "evidence_codes", "speech_presentation",
    ):
        assert forbidden not in schema
    payload = _compact_payload()
    assert [row["segment_id"] for row in payload["segments"]] == ["segment_0001"]
    assert "visual_observation" not in payload
    assert "audio_observation" not in payload
    assert "av_grounding" not in payload
    assert single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)
    payload["segments"][0]["entity_id"] = "e1"
    with pytest.raises(ValueError):
        single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)


def _job(*, segments=None, subjects=None):
    segment = SimpleNamespace(
        segment_id="segment_0001", asr_status="transcribed",
        asr_language="English", asr_text="exact transcript",
        source_speaker_cluster_id="cluster_a", start_time=0.0, end_time=1.0,
    )
    return SimpleNamespace(
        segments=segments if segments is not None else [segment],
        reference_subjects=subjects if subjects is not None else [
            SimpleNamespace(subject_label="<Subject 1>", kind="entity", entity_id="e1")
        ],
    )


def _validated(payload: dict, job=None, *, entity_ids=None):
    job = job or _job()
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)
    issues, warnings = single._validate_single_compact_annotation(
        draft, job, allowed_entity_ids=entity_ids or {"e1"},
        allowed_reference_labels={"<Picture 1>"}
        | {subject.subject_label for subject in job.reference_subjects},
    )
    return draft, issues, warnings


def test_compact_inventory_subjects_and_dialogue_are_hard_contracts():
    payload = _compact_payload()
    assert _validated(payload)[1] == []
    broken = json.loads(json.dumps(payload))
    broken["segments"].append(broken["segments"][0])
    assert "segment_inventory_mismatch" in {issue.code for issue in _validated(broken)[1]}
    broken = json.loads(json.dumps(payload))
    broken["segments"] = []
    assert "segment_inventory_mismatch" in {issue.code for issue in _validated(broken)[1]}
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["shot1_caption"] = "<Subject 1> says nothing."
    assert "direct_transcribed_dialogue_inventory_mismatch" in {
        issue.code for issue in _validated(broken)[1]
    }
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["subject_definitions"] = []
    with pytest.raises(ValueError):
        single.MimoSingleCompactAnnotationDraftV2.model_validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["visual_retention_analysis"][0]["subject_label"] = "<Subject 2>"
    assert "subject_inventory_mismatch" in {issue.code for issue in _validated(broken)[1]}
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["subject_definitions"][0]["description"] += " from <Picture 1>"
    with pytest.raises(ValueError, match="Picture provenance"):
        single.MimoSingleCompactAnnotationDraftV2.model_validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["shot1_caption"] += " <d>[English] invented words</d>"
    assert "direct_transcribed_dialogue_inventory_mismatch" in {
        issue.code for issue in _validated(broken)[1]
    }
    broken = json.loads(json.dumps(payload))
    broken["segments"][0]["primary_speaker_group"] = "g2"
    assert "non_contiguous_speaker_groups" in {issue.code for issue in _validated(broken)[1]}
    broken = json.loads(json.dumps(payload))
    broken["h3_semantics"]["summary"] += " (S9)"
    assert "pipeline_owned_syntax_outside_shot_caption" in {
        issue.code for issue in _validated(broken)[1]
    }


def test_compact_segment_order_must_follow_authoritative_order():
    payload = _compact_payload()
    second = SimpleNamespace(**{
        **vars(_job().segments[0]), "segment_id": "segment_0002",
        "start_time": 1.0, "asr_status": "empty", "asr_text": None,
    })
    payload["segments"].append({
        "segment_id": "segment_0002", "primary_speaker_group": "g1",
        "delivery_style": None, "binding_status": "uncertain",
        "speaker_subject_label": None,
    })
    job = _job(segments=[*_job().segments, second])
    assert _validated(payload, job)[1] == []
    payload["segments"].reverse()
    assert "segment_inventory_mismatch" in {
        issue.code for issue in _validated(payload, job)[1]
    }


def test_compact_visible_subject_is_frozen_entity_only():
    payload = _compact_payload()
    assert _validated(payload)[1] == []
    payload["segments"][0]["speaker_subject_label"] = "<Subject 99>"
    assert "unknown_speaker_subject" in {issue.code for issue in _validated(payload)[1]}
    payload["segments"][0]["speaker_subject_label"] = "<Subject 1>"
    for kind in ("attribute", "background"):
        subject = SimpleNamespace(subject_label="<Subject 1>", kind=kind, entity_id=None)
        assert "unknown_speaker_subject" in {
            issue.code for issue in _validated(payload, _job(subjects=[subject]))[1]
        }
    payload["segments"][0]["binding_status"] = "offscreen"
    with pytest.raises(ValueError, match="other bindings require null"):
        single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)


@pytest.mark.parametrize("description", [
    "a man shown in <Picture 1>",
    "a man depicted in <Picture 1>",
    "a man with its visual detail sourced from <Picture 1>",
    "a man <Picture 1>",
])
def test_compact_picture_provenance_is_removed_before_parse(description):
    payload = _compact_payload()
    payload["h3_semantics"]["subject_definitions"][0]["description"] = description
    raw = json.dumps(payload)
    assert single.parse_structured_json_issues(
        raw, single.MimoSingleCompactAnnotationDraftV2,
    )[0] is None
    canonical, corrections = single._canonicalize_single_compact_raw(raw)
    draft, issues = single.parse_structured_json_issues(
        canonical, single.MimoSingleCompactAnnotationDraftV2,
    )
    assert issues == []
    assert draft.h3_semantics.subject_definitions[0].description == "a man"
    assert corrections == {"compact_subject_picture_provenance_removed": 1}
    assert json.loads(raw) == payload


def test_compact_picture_cleanup_is_local_and_empty_definition_still_fails():
    payload = _compact_payload()
    payload["visual_blocks"][0]["text"] += " <Picture 1>"
    payload["h3_semantics"]["shot1_caption"] += " <Picture 1>"
    payload["h3_semantics"]["subject_definitions"][0]["description"] = (
        "a man shown in <Picture 1> depicted in <Picture 2>"
    )
    canonical, corrections = single._canonicalize_single_compact_raw(json.dumps(payload))
    cleaned = json.loads(canonical)
    assert cleaned["visual_blocks"] == payload["visual_blocks"]
    assert cleaned["h3_semantics"]["shot1_caption"] == payload["h3_semantics"]["shot1_caption"]
    assert cleaned["h3_semantics"]["subject_definitions"][0]["description"] == "a man"
    assert corrections == {"compact_subject_picture_provenance_removed": 1}
    payload["h3_semantics"]["subject_definitions"][0]["description"] = "shown in <Picture 1>."
    canonical, _ = single._canonicalize_single_compact_raw(json.dumps(payload))
    assert single.parse_structured_json_issues(
        canonical, single.MimoSingleCompactAnnotationDraftV2,
    )[0] is None


def test_compact_attribute_speaker_promotes_to_frozen_entity_owner():
    payload = _compact_payload()
    payload["segments"][0]["speaker_subject_label"] = "<Subject 2>"
    payload["h3_semantics"]["subject_definitions"].append({
        "subject_label": "<Subject 2>", "description": "a dark coat",
    })
    payload["h3_semantics"]["visual_retention_analysis"].append({
        "subject_label": "<Subject 2>", "marker": "fully_preserved",
        "description": "The dark coat remains visible.",
    })
    job = _job(subjects=[
        SimpleNamespace(subject_label="<Subject 1>", kind="entity", entity_id="e1"),
        SimpleNamespace(subject_label="<Subject 2>", kind="attribute", entity_id=None,
                        owner_entity_id="e1"),
    ])
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)
    normalized, corrections = single._normalize_single_compact_speaker_subjects(draft, job)
    assert normalized.segments[0].speaker_subject_label == "<Subject 1>"
    assert corrections == {"compact_speaker_attribute_promoted_to_owner": 1}
    assert draft.segments[0].speaker_subject_label == "<Subject 2>"
    assert single._validate_single_compact_annotation(
        normalized, job, allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>", "<Subject 2>"},
    )[0] == []
    projected = single._project_compact_to_legacy_annotation(normalized, job)
    assert projected.av_grounding.segment_groundings[0].entity_id == "e1"


def test_compact_background_speaker_downgrades_without_guessing_entity():
    payload = _compact_payload()
    job = _job(subjects=[SimpleNamespace(
        subject_label="<Subject 1>", kind="background", entity_id=None,
        owner_entity_id=None,
    )])
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)
    normalized, corrections = single._normalize_single_compact_speaker_subjects(draft, job)
    decision = normalized.segments[0]
    assert (decision.binding_status, decision.speaker_subject_label) == (
        "no_reliable_subject", None,
    )
    assert decision.primary_speaker_group == "g1"
    assert corrections == {"compact_invalid_speaker_subject_downgraded": 1}
    assert single._validate_single_compact_annotation(
        normalized, job, allowed_entity_ids=set(),
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )[0] == []


def test_compact_valid_entity_speaker_is_unchanged():
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(_compact_payload())
    normalized, corrections = single._normalize_single_compact_speaker_subjects(draft, _job())
    assert normalized == draft
    assert corrections == {}


def test_compact_attribute_without_unique_frozen_owner_downgrades():
    payload = _compact_payload()
    payload["segments"][0]["speaker_subject_label"] = "<Subject 3>"
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(payload)
    job = _job(subjects=[
        SimpleNamespace(subject_label="<Subject 1>", kind="entity", entity_id="e1"),
        SimpleNamespace(subject_label="<Subject 2>", kind="entity", entity_id="e1"),
        SimpleNamespace(subject_label="<Subject 3>", kind="attribute", entity_id=None,
                        owner_entity_id="e1"),
    ])
    normalized, corrections = single._normalize_single_compact_speaker_subjects(draft, job)
    assert normalized.segments[0].binding_status == "no_reliable_subject"
    assert normalized.segments[0].speaker_subject_label is None
    assert corrections == {"compact_invalid_speaker_subject_downgraded": 1}


def test_compact_attribute_without_owner_downgrades():
    draft = single.MimoSingleCompactAnnotationDraftV2.model_validate(_compact_payload())
    job = _job(subjects=[SimpleNamespace(
        subject_label="<Subject 1>", kind="attribute", entity_id=None,
    )])
    normalized, corrections = single._normalize_single_compact_speaker_subjects(draft, job)
    assert normalized.segments[0].binding_status == "no_reliable_subject"
    assert normalized.segments[0].speaker_subject_label is None
    assert corrections == {"compact_invalid_speaker_subject_downgraded": 1}


def test_compact_group_subject_conflicts_are_warnings_without_repair():
    payload = _compact_payload()
    payload["h3_semantics"]["subject_definitions"].append({
        "subject_label": "<Subject 2>", "description": "a second person in a dark coat",
    })
    payload["h3_semantics"]["visual_retention_analysis"].append({
        "subject_label": "<Subject 2>", "marker": "fully_preserved",
        "description": "The second person's coat remains visible.",
    })
    subjects = [
        SimpleNamespace(subject_label="<Subject 1>", kind="entity", entity_id="e1"),
        SimpleNamespace(subject_label="<Subject 2>", kind="entity", entity_id="e2"),
    ]
    second = SimpleNamespace(
        segment_id="segment_0002", asr_status="empty", asr_language=None,
        asr_text=None, source_speaker_cluster_id="cluster_b", start_time=1.0, end_time=2.0,
    )
    payload["segments"].append({
        "segment_id": "segment_0002", "primary_speaker_group": "g1", "delivery_style": None,
        "binding_status": "visible_subject", "speaker_subject_label": "<Subject 2>",
    })
    draft, issues, warnings = _validated(
        payload, _job(segments=[*_job().segments, second], subjects=subjects),
        entity_ids={"e1", "e2"},
    )
    assert issues == []
    assert "compact_group_maps_multiple_subjects" in warnings
    assert draft.segments[1].speaker_subject_label == "<Subject 2>"
    projected = single._project_compact_to_legacy_annotation(
        draft, _job(segments=[*_job().segments, second], subjects=subjects),
    )
    assert [item.entity_id for item in projected.av_grounding.segment_groundings] == ["e1", "e2"]
    payload["segments"][1]["primary_speaker_group"] = "g2"
    payload["segments"][1]["speaker_subject_label"] = "<Subject 1>"
    draft, issues, warnings = _validated(
        payload, _job(segments=[*_job().segments, second], subjects=subjects),
        entity_ids={"e1", "e2"},
    )
    assert issues == []
    assert "compact_subject_maps_multiple_groups" in warnings
    assert draft.segments[1].speaker_subject_label == "<Subject 1>"
    projected = single._project_compact_to_legacy_annotation(
        draft, _job(segments=[*_job().segments, second], subjects=subjects),
    )
    assert [item.entity_id for item in projected.av_grounding.segment_groundings] == ["e1", "e1"]


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
    assert projected.visual_observation.segment_views[0].visible_entity_ids == ["e1"]


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
        payload["segments"].append({
            "segment_id": segment_id, "primary_speaker_group": group, "delivery_style": "calm",
            "binding_status": "no_reliable_subject", "speaker_subject_label": None,
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
    payload["segments"][0]["delivery_style"] = None
    payload["speaker_voice_profiles"] = []
    draft, issues, _ = _validated(payload, _job(segments=[segment]))
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, _job(segments=[segment]))
    assert "(S" not in projected.h3_semantics.shot1_caption
    assert "<d>" not in projected.h3_semantics.shot1_caption
    assert direct_speech_facts(projected, [segment]) == []


def test_compact_visible_subject_maps_to_frozen_entity_without_remap():
    payload = _compact_payload()
    subject = SimpleNamespace(subject_label="<Subject 1>", kind="entity", entity_id="e2")
    job = _job(subjects=[subject])
    draft, issues, _ = _validated(payload, job, entity_ids={"e2"})
    assert issues == []
    projected = single._project_compact_to_legacy_annotation(draft, job)
    assert projected.av_grounding.segment_groundings[0].entity_id == "e2"
    assert projected.visual_observation.segment_views[0].visible_entity_ids == ["e2"]
    assert projected.visual_observation.segment_views[0].entity_observations == []
    assert "e2" not in json.dumps(payload)


def test_single_compact_backend_keeps_raw_response_and_legacy_annotation(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    stems = load_stem_shadow(shadow / "separation")[1]
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.build_mimo25_inventory(),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr", route="music_first",
    )[:1]
    payload = _compact_payload()
    payload["h3_semantics"]["subject_definitions"][0]["description"] += " shown in <Picture 1>"
    raw = json.dumps(payload)
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
    assert request["response_format"]["json_schema"]["name"] == "MimoSingleCompactAnnotationDraftV2"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["reasoning_effort"] == "none"
    assert request["extra_body"]["chat_template_kwargs"] == {
        "thinking": False, "enable_thinking": False,
    }
    assert result.speech_av_raw_response == raw
    assert result.raw_responses == (raw,)
    assert "compact_subject_picture_provenance_removed" in result.diagnostics[0].warnings
    assert result.deterministic_correction_counts == {
        "compact_subject_picture_provenance_removed": 1,
    }
    assert "<Picture 1>" not in result.annotation.h3_semantics.subject_definitions[0].description
    assert result.annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert "(S1) <d>" in result.annotation.h3_semantics.shot1_caption
    root = shadow / "single-v5"
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_records=stems, backend=backend, output_root=root,
        route="music_first", allow_unverified=True,
    )
    record = MimoStemReconcileRecord.model_validate_json((root / "records.jsonl").read_text().splitlines()[0])
    assert record.speech_av_raw_response == raw
    assert "compact_subject_picture_provenance_removed" in record.diagnostics[0].warnings
    assert record.annotation is not None and record.annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"


@pytest.mark.parametrize(("version", "prompt"), [
    (".67", "h3_mimo26_ra2va_single_v1"),
    (".68", "h3_mimo26_ra2va_single_v2"),
    (".69", "h3_mimo26_ra2va_single_v3"),
    (".70", "h3_mimo26_ra2va_single_v4"),
    (".71", "h3_mimo26_ra2va_single_v5_compact"),
    (".72", "h3_mimo26_ra2va_single_v6_compact2"),
])
def test_historical_single_record_envelopes_remain_readable(tmp_path, monkeypatch, version, prompt):
    _, shadow = _fixture(tmp_path, monkeypatch)
    stems = load_stem_shadow(shadow / "separation")[1]
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.build_mimo25_inventory(),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr", route="music_first",
    )[:1]
    backend = single.SingleCallOpenAIMimo25Backend(
        MimoBackendConfig(
            api_key="fake", transport="sglang", model="mimo-v2.6-flash-rl",
            media_resolver=MimoMediaResolver(
                mode="http", media_root=tmp_path, media_base_url="http://media.invalid",
            ),
        ),
        stem_records_by_clip={record.clip_uid: record for record in stems},
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=_Completions([(_single_raw(), 8)]),
        )),
    )
    with monkeypatch.context() as patch:
        patch.setattr(single, "SINGLE_BACKEND_VERSION", f"r2v.h3.mimo25_backend{version}")
        patch.setattr(single, "SINGLE_PROMPT_VERSION", prompt)
        root = shadow / "historical-record"
        run_mimo25_stem_reconcile_shadow(
            jobs=jobs, stem_records=stems, backend=backend, output_root=root,
            route="music_first", allow_unverified=True,
        )
    record = MimoStemReconcileRecord.model_validate_json(
        (root / "records.jsonl").read_text().splitlines()[0]
    )
    assert record.status == "ready"
    assert record.backend_provenance.prompt_version == prompt
    assert record.annotation is not None


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
    raw = json.dumps({**_compact_payload(), "visual_blocks": []})
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
