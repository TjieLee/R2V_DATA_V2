from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft, MimoBackendFailure
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from r2v_data_v2.h3.ra2va_group_export import (
    GroupAudioContract,
    _reference_contract,
    _sample,
)
from r2v_data_v2.h3.ra2va_group_pipeline import GroupFrozenJob
from r2v_data_v2.h3.ra2va_group_source import GroupTask
from r2v_data_v2.h3.speaker_ownership import (
    neutralize_two_step_caption_identity,
    two_step_identity_restricted_groups,
)

CASES = json.loads((Path(__file__).parent / "fixtures/h3_group_pilot10_failures.json").read_text())
CAPTION_CASES = ["ab27449eaf98b06401161810", "83871dd0401c1f1ce9986ec2"]
RECOVERABLE_CASES = [*CAPTION_CASES, "d897696ed13b1ea1b2104ea2"]


@pytest.mark.parametrize("clip", CAPTION_CASES)
def test_real_complex_lead_neutralization_preserves_visual_facts_and_exact_asr(clip):
    case = copy.deepcopy(CASES[clip])
    record = case["record"]
    annotation = MimoAVAnnotationDraft.model_validate(record["annotation"])
    job = GroupFrozenJob.model_validate(case["job"])
    provenance = SimpleNamespace(**record["backend_provenance"])
    before = annotation.model_dump(mode="json")
    caption, summary, counts, publishable = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert publishable
    assert two_step_identity_restricted_groups(annotation, provenance) == {"g1"}
    assert "A voice (S1) says, <d>" in caption
    assert "<Subject 1> (S1)" not in caption
    assert "he speaks (S1)" not in caption
    assert re.findall(r"<d>.*?</d>", caption) == re.findall(r"<d>.*?</d>", before["h3_semantics"]["shot1_caption"])
    assert counts["joint_unconfirmed_caption_identity_neutralized"] > 0
    assert counts["joint_unconfirmed_summary_identity_neutralized"] == 1
    original = before["h3_semantics"]["shot1_caption"]
    if clip.startswith("ab2744"):
        expected = original.replace("<Subject 1> (S1)", "<Subject 1>").replace(
            ", saying, <d>", ". A voice (S1) says, <d>")
        assert "with a voice delivering a warning" in summary
    else:
        expected = original.replace("His mouth moves as he speaks (S1), <d>", "His mouth moves. A voice (S1) says, <d>")
        assert "A hooded man stands in heavy rain while a voice delivers" in summary
    assert caption == expected
    assert annotation.model_dump(mode="json") == before
    assert case == CASES[clip]


@pytest.mark.parametrize("clip", RECOVERABLE_CASES)
@pytest.mark.parametrize("variant", ["visual_only", "full_audio_reuse"])
def test_real_restricted_captions_publish_identity_independent_h3_only(tmp_path, clip, variant):
    case = CASES[clip]
    job = GroupFrozenJob.model_validate(case["job"])
    task = GroupTask(**case["task"])
    # Local render tests need file handles, not the remote media's visual pixels.
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference fixture")
    task = replace(task, pictures=[{**p, "image_path": str(image)} for p in task.pictures])
    job = job.model_copy(update={"reference_images": [p.model_copy(update={"image_artifact_path": str(image)})
                                                     for p in job.reference_images]})
    annotation = MimoAVAnnotationDraft.model_validate(case["record"]["annotation"])
    record = SimpleNamespace(annotation=annotation, source_backend_provenance=SimpleNamespace(
        **case["record"]["backend_provenance"]))
    legacy_job = SimpleNamespace(**job.__dict__, target_full_audio_sha256=None,
                                 target_video_sha256=None, request_fingerprint=None)
    audios = [] if variant == "visual_only" else [GroupAudioContract(
        audio_index=1, audio_label="<Audio 1>", kind="full_audio_reuse",
        path=job.target_full_audio_path, retention_marker="fully_copy")]
    corrected, prompt, warnings = _materialize_sample(_sample(task, job), legacy_job, record,
        conditioning_variant=variant, reuse_audio_contracts=audios, reference_contract=_reference_contract(job))
    assert "<Subject 1> (S1)" not in prompt
    assert "A voice (S1)" in prompt
    assert "g1:identity_publication_restricted" in warnings
    assert [s.text for s in corrected] == [s.asr_text for s in job.segments if s.asr_status == "transcribed"]
    assert all(f"<d>[{s.asr_language}] {s.asr_text}</d>" in prompt for s in job.segments)
    assert ("<Audio 1>" in prompt) == (variant == "full_audio_reuse")


@pytest.mark.parametrize("mutation", ["conflicting_marker", "wrong_text", "wrong_language", "unknown_lead", "extra_identity_claim"])
def test_real_caption_ambiguous_or_modified_dialogue_remains_unpublishable(mutation):
    case = CASES[CAPTION_CASES[0]]
    payload = copy.deepcopy(case["record"]["annotation"])
    caption = payload["h3_semantics"]["shot1_caption"]
    if mutation == "conflicting_marker":
        caption = caption.replace("(S1)", "(S2)")
    elif mutation == "wrong_text":
        caption = caption.replace("大人亲手", "大人自己")
    elif mutation == "wrong_language":
        caption = caption.replace("[Chinese]", "[English]")
    elif mutation == "unknown_lead":
        caption = caption.replace(", saying, <d>", ", asking the other man to speak, <d>")
    else:
        caption = caption.replace(", saying, <d>", ", identifying himself as the speaker, saying, <d>")
    payload["h3_semantics"]["shot1_caption"] = caption
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    job = GroupFrozenJob.model_validate(case["job"])
    result = neutralize_two_step_caption_identity(annotation, job.segments, SimpleNamespace(
        **case["record"]["backend_provenance"]))
    assert result == (caption, payload["h3_semantics"]["summary"], {}, False)


def test_summary_cannot_hide_unknown_subject_by_neutralizing_unrelated_identity():
    case = CASES[CAPTION_CASES[0]]
    payload = copy.deepcopy(case["record"]["annotation"])
    payload["h3_semantics"]["summary"] = "An exchange, with <Subject 99> delivering a warning."
    job = GroupFrozenJob.model_validate(case["job"])
    _, summary, _, publishable = neutralize_two_step_caption_identity(
        MimoAVAnnotationDraft.model_validate(payload), job.segments,
        SimpleNamespace(**case["record"]["backend_provenance"]))
    assert not publishable or "<Subject 99>" in summary


def test_visual_uncertainty_prefix_cannot_authorize_affirmative_posture_clause():
    case = CASES[CAPTION_CASES[0]]
    payload = copy.deepcopy(case["record"]["annotation"])
    visual = payload["visual_observation"]["visual_blocks"][0]
    visual["text"] = visual["text"].replace("<Subject 1> faces", "It is unclear whether <Subject 1> faces")
    job = GroupFrozenJob.model_validate(case["job"])
    *_, publishable = neutralize_two_step_caption_identity(MimoAVAnnotationDraft.model_validate(payload),
        job.segments, SimpleNamespace(**case["record"]["backend_provenance"]), reference_subjects=job.reference_subjects)
    assert not publishable


@pytest.mark.parametrize("prompt_version", ["h3_mimo25_av_v1", "h3_mimo26_ra2va_single_v10_speaker_subject_consistency"])
def test_multi_and_single_caption_paths_remain_unchanged(prompt_version):
    case = CASES[CAPTION_CASES[0]]
    annotation = MimoAVAnnotationDraft.model_validate(case["record"]["annotation"])
    job = GroupFrozenJob.model_validate(case["job"])
    assert neutralize_two_step_caption_identity(annotation, job.segments, SimpleNamespace(
        prompt_version=prompt_version)) == (
        annotation.h3_semantics.shot1_caption, annotation.h3_semantics.summary, {}, True)


@pytest.mark.parametrize("clip", list(CASES))
def test_four_real_raw_replays_preserve_audio_and_keep_nonrecoverable_failures(tmp_path, monkeypatch, clip):
    from tests.test_h3_mimo26_two_step import _setup

    case = CASES[clip]
    record = case["record"]
    _, _, backend, completions, stems, _, _, _ = _setup(tmp_path, monkeypatch,
        visual=json.loads(record["visual_raw_response"]), joint=json.loads(record["speech_av_raw_response"]))
    # Only filesystem-to-URL resolution is replaced; both recorded drafts pass
    # through the real Two-step parser, normalizer, and shared annotation validator.
    backend.config.media_resolver.resolve = lambda path: "http://recorded.invalid/" + Path(path).name
    job = GroupFrozenJob.model_validate(case["job"])
    kwargs = {"segment_ids": [s.segment_id for s in job.segments],
        "transcribed_segment_ids": [s.segment_id for s in job.segments if s.asr_status == "transcribed"],
        "allowed_entity_ids": {s.entity_id for s in job.reference_subjects if s.kind == "entity"},
        "allowed_reference_labels": {s.subject_label for s in job.reference_subjects}
                                    | {p.picture_label for p in job.reference_images},
        "auxiliary_audio_paths": {s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems}}
    if clip in RECOVERABLE_CASES:
        result = backend.reconcile(job, **kwargs)
        assert result.model_call_count == 2
        assert result.annotation.audio_observation.model_dump(mode="json") == record["annotation"]["audio_observation"]
        assert result.annotation.av_grounding.model_dump(mode="json") == record["annotation"]["av_grounding"]
        assert result.annotation.visual_observation.model_dump(mode="json") == record["annotation"]["visual_observation"]
        assert result.deterministic_correction_counts["joint_unconfirmed_caption_identity_neutralized"] > 0
    else:
        with pytest.raises(MimoBackendFailure) as caught:
            backend.reconcile(job, **kwargs)
        result = caught.value
        assert result.reason == record["failure_reason"]
        assert [i.code for i in result.issues] == [i["code"] for i in record["failure_issues"]]
    assert len(completions.requests) == 2
    assert json.loads(result.visual_raw_response) == json.loads(record["visual_raw_response"])
    assert json.loads(result.speech_av_raw_response) == json.loads(record["speech_av_raw_response"])
