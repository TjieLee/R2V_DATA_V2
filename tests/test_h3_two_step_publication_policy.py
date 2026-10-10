from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from r2v_data_v2.h3.ra2va_group_pipeline import GroupFrozenJob
from r2v_data_v2.h3.speaker_ownership import (
    neutralize_two_step_caption_identity,
    two_step_identity_restricted_groups,
)
from tests.test_h3_group_pilot_caption_recovery import CASES


def _case(clip="83871dd0401c1f1ce9986ec2"):
    case = CASES[clip]
    return (
        MimoAVAnnotationDraft.model_validate(case["record"]["annotation"]),
        GroupFrozenJob.model_validate(case["job"]),
        SimpleNamespace(**case["record"]["backend_provenance"]),
    )


@pytest.mark.parametrize("lead", [
    "A voice (S1) answers, ", "A voice (S1) says: ", "A voice (S1) quietly says, ",
    "An unidentified voice (S1) responds, ",
])
def test_already_neutral_speech_does_not_fail_on_verb_adverb_or_delimiter(lead):
    annotation, job, provenance = _case()
    caption, summary, _, valid = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert valid
    caption = caption.replace("A voice (S1) says, ", lead)
    annotation.h3_semantics.shot1_caption = caption
    annotation.h3_semantics.summary = summary
    result = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert result[3]
    assert result[0] == caption


def test_real_uncertain_articulation_keeps_facts_and_neutralizes_only_unconfirmed_group():
    annotation, job, provenance = _case("d897696ed13b1ea1b2104ea2")
    original = annotation.model_dump(mode="json")
    assert two_step_identity_restricted_groups(annotation, provenance) == {"g1"}
    caption, summary, corrections, valid = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert valid
    assert caption == original["h3_semantics"]["shot1_caption"].replace(
        "<Subject 1> (S1) says,", "A voice (S1) says,")
    assert "</d> and adds, <d>" in caption
    assert "<Subject 2> (S2) reports," in caption
    assert re.findall(r"<d>.*?</d>", caption) == [
        f"<d>[{s.asr_language}] {s.asr_text}</d>" for s in job.segments
    ]
    assert "armored runner reports to" not in summary
    assert "2 acoustic speaker groups" in summary
    assert corrections["joint_unconfirmed_summary_identity_projected"] == 1
    assert annotation.model_dump(mode="json") == original


@pytest.mark.parametrize("change", ["wrong_marker", "switched_group_continuation", "wrong_text"])
def test_continuation_cannot_hide_speaker_or_asr_conflicts(change):
    annotation, job, provenance = _case("d897696ed13b1ea1b2104ea2")
    if change == "wrong_marker":
        annotation.h3_semantics.shot1_caption = annotation.h3_semantics.shot1_caption.replace(
            "and adds,", "and adds (S2),")
    elif change == "switched_group_continuation":
        annotation.audio_observation.segment_decisions[1].primary_speaker_group = "g2"
        annotation.av_grounding.segment_groundings[1].primary_speaker_group = "g2"
        annotation.av_grounding.segment_groundings[1].entity_id = "e2"
        annotation.audio_observation.segment_decisions[2].primary_speaker_group = "g1"
        annotation.av_grounding.segment_groundings[2].primary_speaker_group = "g1"
        annotation.av_grounding.segment_groundings[2].entity_id = "e1"
        annotation.av_grounding.segment_groundings[2].evidence_codes = ["av_temporal_alignment", "no_visible_lip_motion"]
    else:
        annotation.h3_semantics.shot1_caption = annotation.h3_semantics.shot1_caption.replace("懂了，大人。", "知道了，大人。")
    assert not neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)[3]


@pytest.mark.parametrize("where", ["caption", "summary"])
def test_residual_pronoun_identity_is_not_silently_published(where):
    annotation, job, provenance = _case()
    caption, summary, _, _ = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    annotation.h3_semantics.shot1_caption = caption + (" He is the speaker." if where == "caption" else "")
    annotation.h3_semantics.summary = summary + (" He is the speaker." if where == "summary" else "")
    result = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert not result[3] or "He is the speaker" not in result[0] + result[1]


def test_unknown_summary_subject_is_not_hidden_by_summary_projection():
    annotation, job, provenance = _case()
    annotation.h3_semantics.summary = "<Subject 99> is the speaker."
    result = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert not result[3] or "<Subject 99>" in result[1]


def test_identity_wording_inside_frozen_asr_is_never_rewritten():
    annotation, job, provenance = _case()
    caption, summary, _, valid = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert valid
    segment = job.segments[0]
    original_block = f"<d>[{segment.asr_language}] {segment.asr_text}</d>"
    segment.asr_language, segment.asr_text = "English", "He is the speaker."
    block = "<d>[English] He is the speaker.</d>"
    annotation.h3_semantics.shot1_caption = caption.replace(original_block, block)
    annotation.h3_semantics.summary = summary
    result = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert result[3]
    assert block in result[0]


@pytest.mark.parametrize("contradiction", ["not_observed", "offscreen_audio", "missing_alignment", "absent_entity", "competing_speaker"])
def test_uncertain_articulation_does_not_override_real_contradictions(contradiction):
    annotation, _, provenance = _case("d897696ed13b1ea1b2104ea2")
    grounding = annotation.av_grounding.segment_groundings[0]
    view = annotation.visual_observation.segment_views[0]
    if contradiction == "not_observed":
        view.entity_observations[0].speech_correlated_articulation = "not_observed"
    elif contradiction == "offscreen_audio":
        grounding.evidence_codes.append("offscreen_audio")
    elif contradiction == "missing_alignment":
        grounding.evidence_codes.remove("av_temporal_alignment")
    elif contradiction == "absent_entity":
        view.visible_entity_ids = []
    else:
        other = view.entity_observations[0].model_copy(update={
            "entity_id": "e2", "speech_correlated_articulation": "observed"})
        view.visible_entity_ids.append("e2")
        view.entity_observations.append(other)
    assert not two_step_identity_restricted_groups(annotation, provenance)


def test_confirmed_group_attribution_is_not_rejected_when_another_group_is_restricted():
    annotation, job, provenance = _case("d897696ed13b1ea1b2104ea2")
    annotation.h3_semantics.shot1_caption = annotation.h3_semantics.shot1_caption.replace(
        "<Subject 2> (S2) reports,", "<Subject 2> (S2) says,")
    result = neutralize_two_step_caption_identity(
        annotation, job.segments, provenance, reference_subjects=job.reference_subjects)
    assert result[3]
    assert "<Subject 2> (S2) says," in result[0]
