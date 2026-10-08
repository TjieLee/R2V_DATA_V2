"""Waveform ownership checks, independent of identity and voice quality."""

from __future__ import annotations

from typing import Literal, Protocol

from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoSecondaryVocalActivity,
    VocalComposition,
    _observed_articulation_entity_ids,
    _visible_binding_is_permitted,
)

SpeakerOwnershipReason = Literal[
    "unresolved", "non_single_speaker", "secondary_vocal_activity",
]


class SpeakerOwnershipEvidence(Protocol):
    @property
    def primary_speaker_group(self) -> str | None: ...

    @property
    def vocal_composition(self) -> VocalComposition: ...

    @property
    def secondary_vocal_activity(self) -> MimoSecondaryVocalActivity: ...


def speaker_ownership_reasons(
    decision: SpeakerOwnershipEvidence,
) -> list[SpeakerOwnershipReason]:
    """Return only reasons that prevent isolated primary-speaker ownership."""
    reasons: list[SpeakerOwnershipReason] = []
    if decision.primary_speaker_group is None:
        reasons.append("unresolved")
    same_speaker_nonlexical = (
        decision.vocal_composition == "same_speaker_nonlexical"
        and decision.secondary_vocal_activity.present
        and decision.secondary_vocal_activity.speaker_relation == "same_speaker"
        and decision.secondary_vocal_activity.kind != "speech"
    )
    if decision.vocal_composition != "single_speaker" and not same_speaker_nonlexical:
        reasons.append("non_single_speaker")
    if decision.secondary_vocal_activity.present and not same_speaker_nonlexical:
        reasons.append("secondary_vocal_activity")
    return reasons


def unconfirmed_visible_binding_segments(annotation: MimoAVAnnotationDraft) -> set[str]:
    """Missing positive AV cues only, never explicit contradictory evidence."""
    views = {view.segment_id: view for view in annotation.visual_observation.segment_views}
    result = set()
    for grounding in annotation.av_grounding.segment_groundings:
        view = views.get(grounding.segment_id)
        if (view is None or grounding.entity_id not in view.visible_entity_ids
                or grounding.binding_status != "visible_entity" or grounding.speech_presentation != "onscreen_spoken"
                or set(grounding.evidence_codes) & {
                    "no_visible_lip_motion", "offscreen_audio", "voice_over_context", "device_playback_context",
                    "source_cluster_conflict", "lr_asd_conflict",
                } or _visible_binding_is_permitted(grounding, view)):
            continue
        articulated = _observed_articulation_entity_ids(view, allowed_entity_ids=set(view.visible_entity_ids))
        if not articulated or grounding.entity_id in articulated:
            result.add(grounding.segment_id)
    return result


def two_step_identity_restricted_groups(annotation: MimoAVAnnotationDraft, provenance: object) -> set[str]:
    """A QA-ready Two-step binding is not an identity-specific Audio source."""
    if not getattr(provenance, "prompt_version", "").startswith("h3_mimo26_ra2va_two_step_joint_"):
        return set()
    segments = unconfirmed_visible_binding_segments(annotation)
    return {g.primary_speaker_group for g in annotation.av_grounding.segment_groundings
            if g.segment_id in segments and g.primary_speaker_group is not None}
