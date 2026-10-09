"""Waveform ownership checks, independent of identity and voice quality."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal, Protocol

from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoSecondaryVocalActivity,
    VocalComposition,
    _observed_articulation_entity_ids,
    _visible_binding_is_permitted,
    direct_speech_facts,
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
    """Two-step sparse-sampling uncertainty, never a conflicting speech source."""
    views = {view.segment_id: view for view in annotation.visual_observation.segment_views}
    result = set()
    for grounding in annotation.av_grounding.segment_groundings:
        view = views.get(grounding.segment_id)
        if (view is None or grounding.entity_id not in view.visible_entity_ids
                or grounding.binding_status != "visible_entity" or grounding.speech_presentation != "onscreen_spoken"
                or set(grounding.evidence_codes) & {
                    "offscreen_audio", "voice_over_context", "device_playback_context",
                    "source_cluster_conflict", "lr_asd_conflict",
                } or _visible_binding_is_permitted(grounding, view)):
            continue
        if "no_visible_lip_motion" in grounding.evidence_codes:
            target = next((item for item in view.entity_observations if item.entity_id == grounding.entity_id), None)
            if (target is None or target.speech_correlated_articulation != "not_assessable"
                    or "av_temporal_alignment" not in grounding.evidence_codes):
                continue
        articulated = _observed_articulation_entity_ids(view, allowed_entity_ids=set(view.visible_entity_ids))
        if not articulated or grounding.entity_id in articulated:
            result.add(grounding.segment_id)
    return result


def two_step_identity_restricted_groups(annotation: MimoAVAnnotationDraft, provenance: object) -> set[str]:
    """An unconfirmed Two-step identity cannot authorize identity-specific Audio."""
    if not getattr(provenance, "prompt_version", "").startswith("h3_mimo26_ra2va_two_step_joint_"):
        return set()
    segments = unconfirmed_visible_binding_segments(annotation)
    return {g.primary_speaker_group for g in annotation.av_grounding.segment_groundings
            if g.segment_id in segments and g.primary_speaker_group is not None}


def neutralize_two_step_caption_identity(annotation, segments, provenance) -> tuple[str, str, dict[str, int], bool]:
    """Neutralize simple speech lead-ins only; complex attribution stays unpublished."""
    caption, summary = annotation.h3_semantics.shot1_caption, annotation.h3_semantics.summary
    restricted = two_step_identity_restricted_groups(annotation, provenance)
    if not restricted:
        return caption, summary, {}, True
    speech = direct_speech_facts(annotation, list(segments))
    decisions = {d.segment_id: d for d in annotation.audio_observation.segment_decisions}
    speakers = {s["speaker_id"] for s in speech if decisions[s["segment_id"]].primary_speaker_group in restricted}
    blocks = list(re.finditer(r"<d>[\s\S]*?</d>", caption))
    if len(blocks) != len(speech):
        return caption, summary, {}, False
    pieces, counts, previous_end = [], Counter(), 0

    def strip_identity_markers(text, correction):
        for speaker in sorted(speakers):
            text, removed = re.subn(
                r"(<Subject [1-9]\d*>)\s+\(" + re.escape(speaker) + r"\)", r"\1", text,
            )
            counts[correction] += removed
        return text

    for block, fact in zip(blocks, speech, strict=True):
        lead = strip_identity_markers(caption[previous_end:block.start()], "joint_unconfirmed_caption_identity_neutralized")
        if fact["speaker_id"] in speakers:
            speaker = re.escape(fact["speaker_id"])
            # Only a standalone, explicit says/asks clause; never infer a pronoun's referent.
            clause = re.search(
                r"(?:^|(?<=[.!?])\s+)(?:<Subject [1-9]\d*>|He|She|They|A voice|The voice)"
                r"\s*(?:\(" + speaker + r"\)\s*)?(says|asks|replies|adds|continues),\s*$", lead,
            )
            if clause is None or re.search(r"\(" + speaker + r"\)", lead[:clause.start()]):
                return caption, summary, {}, False
            replacement = f"A voice ({fact['speaker_id']}) {clause.group(1)}, "
            leading_space = " " if clause.group().startswith(" ") else ""
            new_lead = lead[:clause.start()] + leading_space + replacement
            counts["joint_unconfirmed_caption_identity_neutralized"] += int(new_lead != lead)
            lead = new_lead
        pieces.extend((lead, block.group()))
        previous_end = block.end()
    tail = strip_identity_markers(caption[previous_end:], "joint_unconfirmed_caption_identity_neutralized")
    if any(f"({speaker})" in tail for speaker in speakers):
        return caption, summary, {}, False
    pieces.append(tail)
    projected_summary, removed = re.subn(
        r"the only visible entity, <Subject [1-9]\d*>, is bound as the speaker", "a voice speaks", summary,
    )
    counts["joint_unconfirmed_summary_identity_neutralized"] += removed
    projected_summary = strip_identity_markers(projected_summary, "joint_unconfirmed_summary_identity_neutralized")
    # An unhandled explicit ownership clause is not an independent visual/audio observation.
    ownership = r"\b(?:voice|speaker)\s+(?:belongs to|comes from|is bound to|is)\s+<Subject [1-9]\d*>"
    summary_binding = r"<Subject [1-9]\d*>\s+(?:speaks|says|asks|replies|is (?:the |a )?speaker|is bound as the speaker)\b"
    prose = re.sub(r"<d>[\s\S]*?</d>", "", "".join(pieces))
    if any(re.search(ownership + "|" + summary_binding, text, re.IGNORECASE) for text in (prose, projected_summary)):
        return caption, summary, {}, False
    return "".join(pieces), projected_summary, {key: value for key, value in counts.items() if value}, True
