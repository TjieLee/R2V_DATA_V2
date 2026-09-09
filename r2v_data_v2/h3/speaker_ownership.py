"""Waveform ownership checks, independent of identity and voice quality."""

from __future__ import annotations

from typing import Literal

from r2v_data_v2.h3.mimo25_backend import MimoAudioSegmentDecision

SpeakerOwnershipReason = Literal[
    "unresolved", "non_single_speaker", "secondary_vocal_activity",
]


def speaker_ownership_reasons(
    decision: MimoAudioSegmentDecision,
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
