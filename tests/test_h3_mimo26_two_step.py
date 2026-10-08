from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.mimo25_av_reconcile import _job
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendFailure,
    MimoBackendProvenance,
    _speaker_profile_targets,
    direct_speech_facts,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    MimoStemReconcileRecord,
)
from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_audio_shadow_qa import _fixture
from tests.test_h3_mimo25_av_shadow import _Completions
from tests.test_h3_mimo25_raw_stems import _backend, _raw, _records, _run
from tools import run_h3_mimo25_stem_reconcile_shadow as cli


def _drafts(payload=None):
    payload = copy.deepcopy(payload or json.loads(_raw()))
    visual, speech, profiles, finalized = map(json.loads, split_annotation(json.dumps(payload)))
    visual["shot1_visual_description"] = "<Subject 1> raises a hand, turns toward the doorway, then lowers it."
    speech["shot1_caption"] = (
        "<Subject 1> raises a hand, turns toward the doorway and (S1) says, "
        "<d>[English] exact transcript</d> then lowers it."
    )
    finalized.update(overall_soundscape="Low room ambience and a distant clink.", non_diegetic_music="Gentle piano score.")
    return visual, {"speech_av": speech, **profiles, "audio_finalize": finalized}


def _setup(tmp_path, monkeypatch, *, visual=None, joint=None):
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend

    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    multi, multi_completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    multi.config = replace(multi.config, model="mimo-v2.6-flash-rl")
    default_visual, default_joint = _drafts()
    completions = _Completions([
        (json.dumps(default_visual if visual is None else visual), 0),
        (json.dumps(default_joint if joint is None else joint), 12),
    ])
    backend = TwoStepOpenAIMimo26Backend(
        multi.config, stem_records_by_clip=multi.stem_records_by_clip,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return kwargs, shadow, backend, completions, stems, jobs, multi, multi_completions


def _reconcile(backend, job, stems):
    return backend.reconcile(
        job, segment_ids=[s.segment_id for s in job.segments],
        transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
        allowed_entity_ids={"e1"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_paths={s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems},
    )


def test_runner_two_step_is_opt_in_and_multi_default_unchanged():
    action = next(a for a in cli._parser()._actions if a.dest == "call_mode")
    assert action.default == "multi"
    assert "two_step" in action.choices
    with pytest.raises(ValueError, match="single-call only"):
        cli._parse_arguments([
            "--visual-production-root", "/v", "--visual-runs-root", "/r",
            "--audio-production-root", "/a", "--case-manifest", "/c",
            "--call-mode", "two_step", "--single-contract", "compact3",
        ])


def test_two_step_reuses_visual_request_and_common_annotation(tmp_path, monkeypatch):
    _, _, backend, completions, stems, jobs, multi, multi_completions = _setup(tmp_path, monkeypatch)
    result = _reconcile(backend, jobs[0], stems)
    _reconcile(multi, jobs[0], stems)
    assert completions.requests[0] == multi_completions.requests[0]
    assert len(completions.requests) == result.model_call_count == 2
    assert result.http_attempt_count == 2 and result.recheck_count == result.http_retry_count == 0
    assert not result.speaker_marker_polish.attempted
    assert result.speaker_profile_raw_response is result.audio_finalize_raw_response is None
    assert result.raw_responses == (result.visual_raw_response, result.speech_av_raw_response)
    annotation = result.annotation
    assert annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert annotation.visual_observation.visual_blocks[0].text == _drafts()[0]["shot1_visual_description"]
    assert "raises a hand" in annotation.h3_semantics.shot1_caption
    assert "then lowers it" in annotation.h3_semantics.shot1_caption
    assert "piano" not in annotation.h3_semantics.shot1_caption
    assert "clink" not in annotation.h3_semantics.shot1_caption
    assert annotation.speaker_voice_profiles[0].speaker_group == "g1"
    request = completions.requests[1]
    assert request["extra_body"]["use_audio_in_video"] is True
    content = request["messages"][-1]["content"]
    assert sum(item["type"] == "video_url" for item in content) == 1
    assert sum(item["type"] == "audio_url" for item in content) == 3
    assert not any(item["type"] == "image_url" for item in content)
    assert "TURN 1 VISUAL DRAFT" in content[-1]["text"]
    assert "speaker snippet:" not in str(content)


@pytest.mark.parametrize("profiles", [[], [{"speaker_group": "g2", "voice_characteristics": None}],
    [{"speaker_group": "g1", "voice_characteristics": None}] * 2])
def test_joint_profile_inventory_is_checked_before_normalization(tmp_path, monkeypatch, profiles):
    _, joint = _drafts()
    joint["speaker_voice_profiles"] = profiles
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert [i.code for i in caught.value.issues] == ["speaker_voice_profile_inventory_mismatch"]
    assert len(completions.requests) == caught.value.model_call_count == 2
    assert caught.value.speaker_profile_raw_response is caught.value.audio_finalize_raw_response is None


@pytest.mark.parametrize("binding", ["offscreen", "no_reliable_entity"])
def test_offscreen_or_unbound_profile_can_be_null(tmp_path, monkeypatch, binding):
    _, joint = _drafts()
    grounding = joint["speech_av"]["av_grounding"]["segment_groundings"][0]
    grounding.update(binding_status=binding, entity_id=None,
                     speech_presentation="offscreen_spoken" if binding == "offscreen" else "uncertain",
                     evidence_codes=["offscreen_audio" if binding == "offscreen" else "insufficient_evidence"])
    joint["speech_av"]["shot1_caption"] = (
        "<Subject 1> raises a hand. An offscreen voice (S1) says, <d>[English] exact transcript</d>."
        if binding == "offscreen" else "<Subject 1> raises a hand. A voice (S1) says, <d>[English] exact transcript</d>."
    )
    joint["speaker_voice_profiles"][0]["voice_characteristics"] = None
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert result.annotation.segment_decisions[0].binding_status == binding
    assert result.annotation.speaker_voice_profiles[0].voice_characteristics is None
    assert len(completions.requests) == 2


@pytest.mark.parametrize("nonlexical", [False, True])
def test_no_transcript_does_not_create_profiles_sx_or_dialogue(tmp_path, monkeypatch, nonlexical):
    visual, joint = _drafts()
    joint["speaker_voice_profiles"] = []
    if nonlexical:
        joint["speech_av"]["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
        joint["speech_av"]["av_grounding"]["segment_groundings"][0].update(
            binding_status="no_reliable_entity", entity_id=None, speech_presentation="uncertain",
            evidence_codes=["insufficient_evidence"],
        )
    else:
        visual["segment_views"] = []
        joint["speech_av"]["audio_observation"]["segment_decisions"] = []
        joint["speech_av"]["av_grounding"]["segment_groundings"] = []
    joint["speech_av"]["shot1_caption"] = visual["shot1_visual_description"]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    if nonlexical:
        values["segments"][0].update(asr_status="empty", asr_text=None, asr_language=None)
    else:
        values["segments"] = []
    job = _job(values)
    result = _reconcile(backend, job, stems)
    assert result.annotation.speaker_voice_profiles == []
    assert direct_speech_facts(result.annotation, list(job.segments)) == []
    assert result.annotation.h3_semantics.shot1_caption == visual["shot1_visual_description"]
    assert "<d>" not in result.annotation.h3_semantics.shot1_caption
    assert "(S1)" not in result.annotation.h3_semantics.shot1_caption
    assert result.annotation.h3_semantics.non_diegetic_music == "Gentle piano score."
    assert len(completions.requests) == 2


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_multiple_speaker_safety_is_not_relaxed(tmp_path, monkeypatch, composition):
    visual, joint = _drafts()
    audio = joint["speech_av"]["audio_observation"]["segment_decisions"][0]
    audio.update(vocal_composition=composition, resolution="needs_acoustic_refinement",
                 secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"})
    grounding = joint["speech_av"]["av_grounding"]["segment_groundings"][0]
    grounding.update(binding_status="no_reliable_entity", entity_id=None, speech_presentation="uncertain",
                     evidence_codes=["insufficient_evidence"])
    kwargs, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert "non_single_speaker" in speaker_ownership_reasons(result.annotation.audio_observation.segment_decisions[0])
    with pytest.raises(qa.MimoH3MaterializationContractError) as caught:
        _materialize(kwargs, jobs[0], result.annotation)
    assert "multi_speaker_segment_requires_turn_refinement" in {i.code for i in caught.value.issues}
    assert len(completions.requests) == 2


def test_joint_schema_and_prompt_reuse_without_stage_conflicts():
    from r2v_data_v2.h3.mimo26_two_step_backend import (
        JOINT_SYSTEM_PROMPT,
        MimoJointAVAudioDraft,
    )

    _, joint = _drafts()
    parsed = MimoJointAVAudioDraft.model_validate(joint)
    assert [t["speaker_group"] for t in _speaker_profile_targets(parsed.speech_av, SimpleNamespace(
        segments=[SimpleNamespace(segment_id="segment_0001", asr_status="transcribed", start_time=0.0,
                                  end_time=1.0, source_speaker_cluster_id="c1")], reference_subjects=[],
    ))] == ["g1"]
    schema = MimoJointAVAudioDraft.model_json_schema()
    assert schema["properties"]["speech_av"]["$ref"].endswith("/MimoSpeechAVAssemblyDraft")
    assert schema["properties"]["audio_finalize"]["$ref"].endswith("/MimoAudioFinalizeDraft")
    assert schema["properties"]["speaker_voice_profiles"]["items"]["$ref"].endswith("/MimoSpeakerVoiceProfile")
    assert "Defer ALL non-dialogue audio to Turn 4" not in JOINT_SYSTEM_PROMPT
    assert "localized speech snippets" not in JOINT_SYSTEM_PROMPT
    assert "full speech stem" in JOINT_SYSTEM_PROMPT
    assert "voice_characteristics=null" in JOINT_SYSTEM_PROMPT
    assert "Do not add non-dialogue audio" in JOINT_SYSTEM_PROMPT
    assert "Original AV remains primary authority" in JOINT_SYSTEM_PROMPT
    assert "return ONLY" not in JOINT_SYSTEM_PROMPT
    assert "Return ONLY" not in JOINT_SYSTEM_PROMPT
    assert "first actual vocal source -> S1" not in JOINT_SYSTEM_PROMPT
    assert "Non-transcribed-only groups consume no Sx" in JOINT_SYSTEM_PROMPT


def _materialize(kwargs, job, annotation):
    sample = qa.FinalH3SampleV2.model_validate_json(
        (kwargs["audio_production_root"] / "h3/samples.jsonl").read_text().splitlines()[0],
    )
    source = job.segments[0]
    sample = sample.model_copy(update={"speech_segments": [qa.FinalQwen3SpeechSegment(
        segment_id=source.segment_id, speaker_cluster_id=source.source_speaker_cluster_id,
        entity_id=source.current_entity_id, entity_occurrence_id=source.entity_occurrence_id,
        source_start_sample=source.source_start_sample, source_end_sample=source.source_end_sample,
        source_sample_rate_hz=source.source_sample_rate_hz, start_time=source.start_time, end_time=source.end_time,
        text=source.asr_text, language=source.asr_language,
    )]})
    return qa._materialize_sample(sample, job, qa._MaterializerInput(annotation, job.request_fingerprint))


def test_two_step_record_and_existing_h3_materializer(tmp_path, monkeypatch):
    kwargs, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    summary = _run(shadow, backend, stems, jobs[:1])
    assert summary.ready_count == 1 and summary.model_call_count == 2
    row = _records(shadow)[0]
    record = MimoStemReconcileRecord.model_validate(row)
    assert record.visual_model_call_count == record.av_model_call_count == 1
    assert record.audio_model_call_count == record.text_model_call_count == 0
    assert len(record.raw_responses) == len(record.diagnostics) == len(completions.requests) == 2
    assert MimoBackendProvenance.model_validate_json(backend.provenance.model_dump_json()) == backend.provenance
    corrected, rendered, _ = _materialize(kwargs, jobs[0], MimoAVAnnotationDraft.model_validate(row["annotation"]))
    assert corrected[0].entity_id == "e1"
    assert "<d>[English] exact transcript</d>" in rendered
    assert "Gentle piano score." in rendered
    assert "overall_soundscape:" in rendered
    assert (shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl").is_file()


@pytest.mark.parametrize("first_transcribed", [False, True])
def test_groups_profiles_and_sx_use_their_existing_chronological_inventories(tmp_path, monkeypatch, first_transcribed):
    visual, joint = _drafts()
    speech = joint["speech_av"]
    second_view = copy.deepcopy(visual["segment_views"][0])
    second_view["segment_id"] = "segment_0002"
    visual["segment_views"].append(second_view)
    second_audio = copy.deepcopy(speech["audio_observation"]["segment_decisions"][0])
    second_audio.update(segment_id="segment_0002", primary_speaker_group="g2")
    speech["audio_observation"]["segment_decisions"].append(second_audio)
    second_grounding = copy.deepcopy(speech["av_grounding"]["segment_groundings"][0])
    second_grounding.update(segment_id="segment_0002", primary_speaker_group="g2", binding_status="offscreen",
                            entity_id=None, speech_presentation="offscreen_spoken", evidence_codes=["offscreen_audio"])
    speech["av_grounding"]["segment_groundings"].append(second_grounding)
    joint["speaker_voice_profiles"].append({"speaker_group": "g2", "voice_characteristics": "Low register with a soft texture."})
    if first_transcribed:
        speech["shot1_caption"] += " An offscreen voice (S2) replies, <d>[Chinese] 好，我知道了。</d>."
    else:
        speech["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
        joint["speaker_voice_profiles"] = joint["speaker_voice_profiles"][1:]
        speech["shot1_caption"] = "<Subject 1> turns toward the doorway. An offscreen voice (S1) says, <d>[Chinese] 好，我知道了。</d>."
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    second = copy.deepcopy(values["segments"][0])
    start = second["end_time"] + 0.1
    second.update(segment_id="segment_0002", source_speaker_cluster_id="cluster_2", current_entity_id=None,
                  start_time=start, end_time=start + 0.5, source_start_sample=round(start * 32000),
                  source_end_sample=round((start + 0.5) * 32000), asr_text="好，我知道了。", asr_language="Chinese")
    values["segments"].append(second)
    if not first_transcribed:
        values["segments"][0].update(asr_status="empty", asr_text=None, asr_language=None)
    job = _job(values)
    result = _reconcile(backend, job, stems)
    facts = direct_speech_facts(result.annotation, list(job.segments))
    assert [f["speaker_id"] for f in facts] == (["S1", "S2"] if first_transcribed else ["S1"])
    assert [p.speaker_group for p in result.annotation.speaker_voice_profiles] == (["g1", "g2"] if first_transcribed else ["g2"])
    assert [d.primary_speaker_group for d in result.annotation.audio_observation.segment_decisions] == ["g1", "g2"]
    assert "<d>[Chinese] 好，我知道了。</d>" in result.annotation.h3_semantics.shot1_caption
    assert len(completions.requests) == 2


@pytest.mark.parametrize("stage", ["visual", "joint", "missing_marker"])
def test_failed_stages_keep_real_raws_without_retry_or_polish(tmp_path, monkeypatch, stage):
    visual, joint = _drafts()
    if stage == "visual":
        visual.pop("style_opening")
    elif stage == "joint":
        joint.pop("audio_finalize")
    else:
        joint["speech_av"]["shot1_caption"] = "Another voice (S2) says, <d>[English] exact transcript</d>."
    _, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    summary = _run(shadow, backend, stems, jobs[:1])
    row = MimoStemReconcileRecord.model_validate(_records(shadow)[0])
    expected = 1 if stage == "visual" else 2
    assert summary.failed_count == 1 and row.model_call_count == len(completions.requests) == expected
    assert len(row.raw_responses) == expected
    assert not row.speaker_marker_polish_attempted
    assert row.speaker_profile_raw_response is row.audio_finalize_raw_response is None


def test_eligible_marker_warning_does_not_trigger_third_polish_call(tmp_path, monkeypatch):
    _, joint = _drafts()
    joint["speech_av"]["shot1_caption"] = "<Subject 1> turns and says, <d>[English] exact transcript</d>."
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert len(completions.requests) == result.model_call_count == 2
    assert not result.speaker_marker_polish.attempted
    assert "direct_single_speaker_marker_missing" in result.diagnostics[-1].warnings


def test_empty_segment_input_rejects_invented_av_inventory(tmp_path, monkeypatch):
    visual, joint = _drafts()
    visual["segment_views"] = []
    joint["speaker_voice_profiles"] = []
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    job = _job({**jobs[0].model_dump(mode="json", exclude={"request_fingerprint"}), "segments": []})
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, job, stems)
    assert {i.code for i in caught.value.issues} == {"segment_inventory_mismatch"}
    assert len(completions.requests) == 2


def test_two_step_has_independent_provenance_without_changing_multi(tmp_path, monkeypatch):
    from r2v_data_v2.h3.audio_reuse_prepared import FrozenReuseBackendProvenance

    _, _, backend, _, _, _, multi, _ = _setup(tmp_path, monkeypatch)
    assert backend.provenance.schema_version == "r2v.h3.mimo25_backend.80"
    assert backend.provenance.prompt_version == "h3_mimo26_ra2va_two_step_joint_v1"
    assert multi.provenance.schema_version == "r2v.h3.mimo25_backend.66"
    assert multi.provenance.prompt_version == "h3_mimo25_speech_assembly_v49"
    FrozenReuseBackendProvenance.model_validate_json(backend.provenance.model_dump_json())
    values = backend.provenance.model_dump()
    values["prompt_version"] = multi.provenance.prompt_version
    with pytest.raises(ValueError, match="provenance differs"):
        MimoBackendProvenance.model_validate(values)


def test_two_step_flows_through_existing_audio_reuse_and_h3_cpu_path(tmp_path):
    from r2v_data_v2.h3 import audio_reuse_materializer as product
    from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
    from r2v_data_v2.h3.audio_reuse_prepared import load_reconcile_sources
    from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
    from tests.test_h3_audio_reuse_materializer import _case

    args, sample, _ = _case(tmp_path / "source", music="Quiet piano score.")
    visual, speech, profiles, finalized = map(json.loads, split_annotation(args["annotation"].model_dump_json()))
    completions = _Completions([
        (json.dumps(visual), 0), (json.dumps({"speech_av": speech, **profiles, "audio_finalize": finalized}), 12),
    ])
    backend = TwoStepOpenAIMimo26Backend(
        MimoBackendConfig(api_key="fake", transport="sglang", model="mimo-v2.6-flash-rl",
                          media_resolver=MimoMediaResolver(mode="http", media_root=tmp_path, media_base_url="http://media.invalid")),
        stem_records_by_clip={args["job"].clip_uid: args["stem_record"]},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    root = tmp_path / "joint-result"
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=[args["job"]], stem_records=[args["stem_record"]], backend=backend,
        output_root=root, route=args["stem_record"].route, allow_unverified=True,
    )
    assert summary.ready_count == 1 and summary.model_call_count == 2
    record = load_reconcile_sources(root)[0]
    values = {**args, "annotation": record.annotation, "output_root": tmp_path / "joint-reuse"}
    manifest = build_audio_reuse_assets(**values)
    assert len(manifest.speakers) == 1 and manifest.music is not None
    source = product.load_reuse_source(manifest_path=values["output_root"] / "manifest.json", job=args["job"],
                                       record=record, stem_record=args["stem_record"])
    refs, _ = product.select_reuse_audio(sample, source, {source.job.clip_uid: source})
    assert [r.contract.kind for r in refs] == ["speaker_speech_reuse", "music_reuse"]
    _, rendered, _ = qa._materialize_sample(sample, args["job"], record, conditioning_variant="target_speech_reuse",
                                           reuse_audio_contracts=[r.contract for r in refs])
    assert "<Audio 1>" in rendered and "<Audio 2>" in rendered
    assert "<d>[English] Exact, text!</d>" in rendered
