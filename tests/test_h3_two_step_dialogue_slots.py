from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from r2v_data_v2.h3.ra2va_group_export import (
    GroupAudioContract,
    _reference_contract,
    _sample,
)
from r2v_data_v2.h3.ra2va_group_pipeline import GroupFrozenJob
from r2v_data_v2.h3.ra2va_group_source import GroupTask
from r2v_data_v2.h3.speaker_ownership import two_step_identity_restricted_groups

CASES = json.loads((Path(__file__).parent / "fixtures/h3_group_pilot8_dialogue_cases.json").read_text())
CAPTION_CASES = ("cb8ebbea64d11f6f037822a1", "615ddb95dfd6375d6cc19f18")


@pytest.mark.parametrize("clip", CAPTION_CASES)
@pytest.mark.parametrize("variant", ["visual_only", "target_speech_reuse", "full_audio_reuse"])
def test_real_complete_dialogue_slots_do_not_depend_on_english_lead_in(tmp_path, clip, variant):
    case = CASES[clip]
    job = GroupFrozenJob.model_validate(case["job"])
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference fixture")
    task = replace(GroupTask(**case["task"]), pictures=[{**p, "image_path": str(image)} for p in case["task"]["pictures"]])
    job = job.model_copy(update={"reference_images": [p.model_copy(update={"image_artifact_path": str(image)})
                                                    for p in job.reference_images]})
    annotation = MimoAVAnnotationDraft.model_validate(case["record"]["annotation"])
    before = annotation.model_dump(mode="json")
    record = SimpleNamespace(annotation=annotation, source_backend_provenance=SimpleNamespace(
        **case["record"]["backend_provenance"]))
    legacy_job = SimpleNamespace(**job.__dict__, target_full_audio_sha256=None,
                                target_video_sha256=None, request_fingerprint=None)
    audios = []
    if variant != "visual_only":
        audios = [GroupAudioContract(audio_index=1, audio_label="<Audio 1>",
            kind="speaker_speech_reuse" if variant == "target_speech_reuse" else "full_audio_reuse",
            path=job.target_full_audio_path,
            retention_marker="partially_copy" if variant == "target_speech_reuse" else "fully_copy",
            **({"speaker_id": "S1", "entity_id": "e1", "subject_label": "<Subject 1>"}
               if variant == "target_speech_reuse" else {}))]
    corrected, prompt, warnings = _materialize_sample(_sample(task, job), legacy_job, record,
        conditioning_variant=variant, reuse_audio_contracts=audios, reference_contract=_reference_contract(job))
    detail = prompt.split("[Shot 1] ", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    expected = [f"<d>[{s.asr_language}] {s.asr_text}</d>" for s in job.segments if s.asr_status == "transcribed"]
    assert re.findall(r"<d>.*?</d>", detail) == expected
    original = annotation.h3_semantics.shot1_caption
    if variant == "target_speech_reuse":
        assert detail.count("copied directly from <Audio 1>") == 1
        detail = detail.replace(", with the synchronized speech signal copied directly from <Audio 1>", "")
    assert re.sub(r"\s*\(S1\)", "", detail) == re.sub(r"\s*\(S1\)", "", original)
    assert detail.split("<d>", 1)[0].count("(S1)") == 1
    assert detail.count("(S1)") == original.count("(S1)") + 1
    assert "deterministic_correction_count:two_step_speaker_markers_inserted=1" in warnings
    assert {(s.speaker_cluster_id, s.entity_id) for s in corrected} == {("g1", "e1")}
    assert annotation.model_dump(mode="json") == before


@pytest.mark.parametrize("phrase", ["raises a question", "quietly murmurs", "utters a reply", "declares", "says"])
def test_structured_multi_speaker_slots_accept_arbitrary_neutral_prose(tmp_path, phrase):
    from r2v_data_v2.h3.mimo25_h3_materializer import (
        _audio_facts,
        _prepare_materialization_context,
        _two_step_speaker_markers,
    )
    from tests.test_h3_audio_reuse_materializer import _case

    caption = (f"<Subject 1> sits. <Subject 1> {phrase}, <d>[English] Yes.</d> "
               f"A voice {phrase}, <d>[English] Yes.</d> <Subject 1> {phrase}, <d>[English] Done.</d>")
    _, sample, source = _case(tmp_path, ("g1", "g2", "g1"), caption=caption,
        speech_payloads=[("English", "Yes."), ("English", "Yes."), ("English", "Done.")])
    record = source.record.model_copy(update={"source_backend_provenance": SimpleNamespace(
        prompt_version="h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity")})
    context = _prepare_materialization_context(sample, source.job, record, reuse_audio_contracts=[])
    facts = _audio_facts(sample=context.sample, corrected=context.corrected, record=record, contract=context.contract)
    result, count = _two_step_speaker_markers(caption, source.job, record, facts.speech)
    assert count == 3
    assert re.findall(r"\((S\d+)\)", result) == ["S1", "S2", "S1"]
    assert re.sub(r"\s*\(S\d+\)", "", result) == caption


def test_uncertain_audio_with_stable_group_is_identity_restricted_not_promoted():
    case = CASES[CAPTION_CASES[1]]
    payload = copy.deepcopy(case["record"]["annotation"])
    for decision in payload["audio_observation"]["segment_decisions"]:
        decision["resolution"] = "uncertain"
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    original = annotation.model_dump(mode="json")
    provenance = SimpleNamespace(**{**case["record"]["backend_provenance"], "schema_version": "r2v.h3.mimo25_backend.86"})
    assert two_step_identity_restricted_groups(annotation, provenance) == {"g1"}
    assert annotation.model_dump(mode="json") == original
    for prompt in ("h3_mimo25_speech_assembly_v49", "h3_mimo26_ra2va_single_v10_speaker_subject_consistency"):
        assert not two_step_identity_restricted_groups(annotation, SimpleNamespace(prompt_version=prompt))


def test_marker_does_not_infer_actor_from_last_mentioned_subject(tmp_path):
    from r2v_data_v2.h3.mimo25_h3_materializer import (
        MimoH3MaterializationContractError,
        _audio_facts,
        _prepare_materialization_context,
        _two_step_speaker_markers,
    )
    from tests.test_h3_audio_reuse_materializer import _case

    caption = "<Subject 3> quietly questions <Subject 1>, <d>[English] Exact, text!</d>"
    _, sample, source = _case(tmp_path, ("g1",), caption=caption, all_visible=True)
    record = source.record.model_copy(update={"source_backend_provenance": SimpleNamespace(
        prompt_version="h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity")})
    context = _prepare_materialization_context(sample, source.job, record, reuse_audio_contracts=[])
    facts = _audio_facts(sample=context.sample, corrected=context.corrected, record=record, contract=context.contract)
    other_subject = source.job.reference_subjects[0].model_copy(update={
        "subject_index": 3, "subject_label": "<Subject 3>", "entity_id": "e2"})
    job = source.job.model_copy(update={"reference_subjects": [*source.job.reference_subjects, other_subject]})
    with pytest.raises(MimoH3MaterializationContractError, match="two_step_speaker_marker_ambiguous"):
        _two_step_speaker_markers(caption, job, record, facts.speech)


def test_real_uncertain_audio_can_retain_identity_independent_caption(tmp_path, monkeypatch):
    from tests.test_h3_mimo26_two_step import _setup

    case = CASES["0afaaa4c6a76733e1d645dd9"]
    record = case["record"]
    _, _, backend, completions, stems, _, _, _ = _setup(tmp_path, monkeypatch,
        visual=json.loads(record["visual_raw_response"]), joint=json.loads(record["speech_av_raw_response"]))
    completions.responses = [(record["visual_raw_response"], 0), (record["speech_av_raw_response"], 12)]
    backend.config.media_resolver.resolve = lambda path: "http://recorded.invalid/" + Path(path).name
    job = GroupFrozenJob.model_validate(case["job"])
    original = copy.deepcopy(record["annotation"])
    result = backend.reconcile(job, segment_ids=[s.segment_id for s in job.segments],
        transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
        allowed_entity_ids={s.entity_id for s in job.reference_subjects if s.kind == "entity"},
        allowed_reference_labels={s.subject_label for s in job.reference_subjects}
                                 | {p.picture_label for p in job.reference_images},
        auxiliary_audio_paths={s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems})
    assert len(completions.requests) == result.model_call_count == 2
    assert result.annotation.audio_observation.model_dump(mode="json") == original["audio_observation"]
    assert result.annotation.av_grounding.model_dump(mode="json") == original["av_grounding"]
    assert result.visual_raw_response == record["visual_raw_response"]
    assert result.speech_av_raw_response == record["speech_av_raw_response"]
    caption = result.annotation.h3_semantics.shot1_caption
    assert "A voice (S1) says," in caption and "<Subject 1> (S1)" not in caption
    assert two_step_identity_restricted_groups(result.annotation, backend.provenance) == {"g1"}


@pytest.mark.parametrize("conflict", ["offscreen", "absent_entity", "different_speaker", "group_mismatch"])
def test_uncertain_audio_policy_never_overrides_actual_binding_conflicts(tmp_path, monkeypatch, conflict):
    from r2v_data_v2.h3.mimo25_backend import MimoBackendFailure
    from tests.test_h3_mimo26_two_step import _setup

    case = CASES["0afaaa4c6a76733e1d645dd9"]
    visual = json.loads(case["record"]["visual_raw_response"])
    joint = json.loads(case["record"]["speech_av_raw_response"])
    decision = joint["speech_av"]["audio_observation"]["segment_decisions"][1]
    grounding = joint["speech_av"]["av_grounding"]["segment_groundings"][1]
    if conflict == "offscreen":
        grounding["evidence_codes"].append("offscreen_audio")
    elif conflict == "absent_entity":
        visual["segment_views"][1].update(visible_entity_ids=[], entity_observations=[])
    elif conflict == "different_speaker":
        decision["vocal_composition"] = "uncertain"
        decision["secondary_vocal_activity"] = {"present": True, "speaker_relation": "different_speaker", "kind": "speech"}
    else:
        grounding["primary_speaker_group"] = "g2"
    _, _, backend, completions, stems, _, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    backend.config.media_resolver.resolve = lambda path: "http://recorded.invalid/" + Path(path).name
    job = GroupFrozenJob.model_validate(case["job"])
    with pytest.raises(MimoBackendFailure):
        backend.reconcile(job, segment_ids=[s.segment_id for s in job.segments],
            transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
            allowed_entity_ids={"e2"}, allowed_reference_labels={"<Subject 1>", "<Subject 2>", "<Picture 1>", "<Picture 2>"},
            auxiliary_audio_paths={s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems})
    assert len(completions.requests) == 2


def test_export_uses_postprocessing_policy_without_relabeling_historical_inference(tmp_path):
    from r2v_data_v2.h3.ra2va_group_export import export_group_clip
    from tests.test_h3_ra2va_group_export import export_case
    from tests.test_h3_ra2va_group_pipeline import ffmpeg

    task, job, result, stems = export_case(tmp_path)
    result["backend_provenance"]["schema_version"] = "r2v.h3.mimo25_backend.85"
    result["postprocessing_backend_provenance"] = {
        **result["backend_provenance"], "schema_version": "r2v.h3.mimo25_backend.86"}
    for decision in result["annotation"]["audio_observation"]["segment_decisions"]:
        decision["resolution"] = "uncertain"
    result["annotation"]["h3_semantics"]["shot1_caption"] = (
        "<Subject 1> stands in a room. <Subject 1> (S1) says, <d>[English] Exact, text!</d>. "
        "A voice (S1) says, <d>[English] Exact, text!</d>.")
    before = copy.deepcopy(result)
    summary = export_group_clip(root=tmp_path / "bundle", task=task, job=job,
                                result=result, stems=stems, ffmpeg=ffmpeg())
    assert (summary["product_ready_count"], summary["product_failed_count"], summary["donor_count"]) == (2, 0, 0), summary
    assert sum(summary["task_counts"].values()) == 8
    assert all("<Subject 1> (S1)" not in json.loads(line)["caption"]
               for path in (tmp_path / "bundle/training").glob("*.jsonl") if path.name != "videos.jsonl"
               for line in path.read_text().splitlines())
    assert result == before
