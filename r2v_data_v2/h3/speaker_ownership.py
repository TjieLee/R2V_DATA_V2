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


def unconfirmed_visible_binding_segments(
    annotation: MimoAVAnnotationDraft, *, include_acoustic_uncertainty: bool = False,
) -> set[str]:
    """Two-step sparse-sampling uncertainty, never a conflicting speech source."""
    views = {view.segment_id: view for view in annotation.visual_observation.segment_views}
    audio = {decision.segment_id: decision for decision in annotation.audio_observation.segment_decisions}
    result = set()
    for grounding in annotation.av_grounding.segment_groundings:
        view = views.get(grounding.segment_id)
        decision = audio.get(grounding.segment_id)
        acoustic_uncertainty = (include_acoustic_uncertainty and decision is not None and decision.resolution == "uncertain"
                                and not speaker_ownership_reasons(decision))
        if (view is None or grounding.entity_id not in view.visible_entity_ids
                or grounding.binding_status != "visible_entity" or grounding.speech_presentation != "onscreen_spoken"
                or set(grounding.evidence_codes) & {
                    "offscreen_audio", "voice_over_context", "device_playback_context",
                    "source_cluster_conflict", "lr_asd_conflict",
                } or (_visible_binding_is_permitted(grounding, view) and not acoustic_uncertainty)):
            continue
        if (include_acoustic_uncertainty and decision is not None
                and decision.resolution == "uncertain" and not acoustic_uncertainty):
            continue
        if "no_visible_lip_motion" in grounding.evidence_codes:
            target = next((item for item in view.entity_observations if item.entity_id == grounding.entity_id), None)
            if (target is None or target.speech_correlated_articulation not in {"not_assessable", "uncertain"}
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
    segments = unconfirmed_visible_binding_segments(
        annotation, include_acoustic_uncertainty=getattr(provenance, "schema_version", "").endswith(".86"),
    )
    return {g.primary_speaker_group for g in annotation.av_grounding.segment_groundings
            if g.segment_id in segments and g.primary_speaker_group is not None}


def neutralize_two_step_caption_identity(
    annotation, segments, provenance, *, reference_subjects=(),
) -> tuple[str, str, dict[str, int], bool]:
    """Separate supported visual clauses from unconfirmed speech attribution."""
    caption, summary = annotation.h3_semantics.shot1_caption, annotation.h3_semantics.summary
    restricted = two_step_identity_restricted_groups(annotation, provenance)
    if not restricted:
        return caption, summary, {}, True
    speech = direct_speech_facts(annotation, list(segments))
    decisions = {d.segment_id: d for d in annotation.audio_observation.segment_decisions}
    speakers = {s["speaker_id"] for s in speech if decisions[s["segment_id"]].primary_speaker_group in restricted}
    blocks = list(re.finditer(r"<d>[\s\S]*?</d>", caption))
    if [b.group() for b in blocks] != [f"<d>[{s['language'] or 'Unknown'}] {s['text']}</d>" for s in speech]:
        return caption, summary, {}, False
    pieces, counts, previous_end = [], Counter(), 0
    verbs = r"says|asks|replies|answers|adds|continues|reports|responds|states|explains|whispers|shouts"
    neutral_actors = {"A voice", "The voice", "An unidentified voice"}

    def strip_identity_markers(text, correction):
        for speaker in sorted(speakers):
            text, removed = re.subn(
                r"(<Subject [1-9]\d*>)\s+\(" + re.escape(speaker) + r"\)", r"\1", text,
            )
            counts[correction] += removed
        return text

    for index, (block, fact) in enumerate(zip(blocks, speech, strict=True)):
        lead = strip_identity_markers(caption[previous_end:block.start()], "joint_unconfirmed_caption_identity_neutralized")
        if fact["speaker_id"] in speakers:
            speaker = re.escape(fact["speaker_id"])
            same_speaker = index > 0 and speech[index - 1]["speaker_id"] == fact["speaker_id"]
            # A continuation inherits the preceding acoustic source, never a new entity.
            if same_speaker and (not lead.strip() or re.fullmatch(r"\s*and (?:" + verbs + r")[,:]\s*", lead)):
                pieces.extend((lead, block.group()))
                previous_end = block.end()
                continue
            clause = re.search(
                r"(?:^|(?<=[.!?])\s+)(?P<actor><Subject [1-9]\d*>|He|She|They|A voice|The voice|An unidentified voice)"
                r"\s*(?:\(" + speaker + r"\)\s*)?(?P<speech>(?:[a-z]+ly\s+)*(?:" + verbs + r")[,:])\s*$", lead,
            )
            replacement = None
            if clause is not None:
                actor = clause.group("actor") if clause.group("actor") in neutral_actors else "A voice"
                replacement = f"{actor} ({fact['speaker_id']}) {clause.group('speech')} "
            else:
                # Keep the visible posture/mouth movement, but not its asserted
                # identity link to the utterance. Other embedded prose stays closed.
                clause = re.search(
                    r"(?:^|(?<=[.!?])\s+)(<Subject [1-9]\d*> faces [^.!?<>]*), saying,\s*$", lead,
                )
                if clause is not None and any(
                    re.search(r"(?:^|(?<=[.!?])\s+)" + re.escape(clause.group(1)) + r"\.", visual.text)
                    for visual in annotation.visual_observation.visual_blocks
                ):
                    replacement = f"{clause.group(1)}. A voice ({fact['speaker_id']}) says, "
                else:
                    clause = re.search(
                        r"(?:^|(?<=[.!?])\s+)(His|Her) mouth moves as (?:he|she) speaks"
                        r"\s*(?:\(" + speaker + r"\))?,\s*$", lead,
                    )
                    if clause is not None:
                        replacement = f"{clause.group(1)} mouth moves. A voice ({fact['speaker_id']}) says, "
            if (clause is None or replacement is None
                    or re.search(r"\(S[1-9]\d*\)", lead[:clause.start()])
                    or any(marker != fact["speaker_id"] for marker in re.findall(r"\((S[1-9]\d*)\)", clause.group()))):
                return caption, summary, {}, False
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
    for index in range(0, len(pieces), 2):
        pieces[index], removed = re.subn(
            r"\b(?:He|She|They) (?:is|are) (?:the |a )?speaker\b", "A voice is heard", pieces[index],
        )
        counts["joint_unconfirmed_caption_identity_neutralized"] += removed
    projected_caption = "".join(pieces)
    projected_summary, removed = re.subn(
        r"the only visible entity, <Subject [1-9]\d*>, is bound as the speaker", "a voice speaks", summary,
    )
    counts["joint_unconfirmed_summary_identity_neutralized"] += removed
    restricted_entities = {g.entity_id for g in annotation.av_grounding.segment_groundings
                           if g.primary_speaker_group in restricted and g.entity_id is not None}
    for subject in reference_subjects:
        if subject.kind != "entity" or subject.entity_id not in restricted_entities:
            continue
        projected_summary, removed = re.subn(
            r"\bwith " + re.escape(subject.subject_label) + r" delivering a warning\b",
            "with a voice delivering a warning", projected_summary,
        )
        counts["joint_unconfirmed_summary_identity_neutralized"] += removed
    projected_summary, removed = re.subn(
        r"^(A [^.!?<>]+ stands in [^.!?<>]+) and speaks (a single [^.!?<>]+)\.$",
        r"\1 while a voice delivers \2.", projected_summary,
    )
    counts["joint_unconfirmed_summary_identity_neutralized"] += removed
    projected_summary = strip_identity_markers(projected_summary, "joint_unconfirmed_summary_identity_neutralized")
    # An unhandled explicit ownership clause is not an independent visual/audio observation.
    ownership = r"\b(?:voice|speaker)\s+(?:belongs to|comes from|is bound to|is)\s+<Subject [1-9]\d*>"
    restricted_labels = [re.escape(s.subject_label) for s in reference_subjects
                         if s.kind == "entity" and s.entity_id in restricted_entities]
    subject_actor = "(?:" + "|".join(restricted_labels) + ")" if restricted_labels else r"<Subject [1-9]\d*>"
    binding = subject_actor + r"\s+(?:speaks|says|asks|replies|is (?:the |a )?speaker|is bound as the speaker)\b"
    prose = re.sub(r"<d>[\s\S]*?</d>", "", projected_caption)
    if re.search(ownership + "|" + binding, prose, re.IGNORECASE):
        return caption, summary, {}, False
    summary_actor = re.search(
        r"(?:^|(?<=[.!?,])\s+)(?P<actor><Subject [1-9]\d*>|He|She|They|(?:An?|The) [^,.!?<>]+?)\s+"
        r"(?:(?:" + verbs + r")|speaks|is (?:the |a )?speaker)\b", projected_summary, re.IGNORECASE,
    )
    if (re.search(ownership + "|" + binding, projected_summary, re.IGNORECASE)
            or (summary_actor and summary_actor.group("actor").casefold() not in {a.casefold() for a in neutral_actors})):
        # Summary wording must not discard an otherwise usable caption. Project
        # existing visual context plus acoustic inventory, not a guessed identity.
        labels = set(re.findall(r"<Subject [1-9]\d*>", projected_summary))
        if labels - {s.subject_label for s in reference_subjects}:
            return caption, summary, {}, False
        opening = re.split(r"(?<=[.!?])\s+", annotation.visual_observation.visual_blocks[0].text, maxsplit=1)[0]
        opening = re.sub(r"\s*\(S[1-9]\d*\)", "", opening)
        group_count = len({s["speaker_id"] for s in speech})
        projected_summary = f"{opening} The clip contains transcribed dialogue from {group_count} acoustic speaker groups."
        counts["joint_unconfirmed_summary_identity_projected"] += 1
    return projected_caption, projected_summary, {key: value for key, value in counts.items() if value}, True
