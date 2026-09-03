from __future__ import annotations

from typing import Literal, Protocol

SpeechPresentation = Literal[
    "onscreen_spoken",
    "offscreen_spoken",
    "voice_over",
    "message_voice_over",
    "device_playback",
    "uncertain",
]


class SpeechPresentationFact(Protocol):
    speaker_id: str
    locked_dialogue_block: str


def render_speech_presentation_clause(
    *,
    speech: SpeechPresentationFact,
    base_clause: str,
    presentation: SpeechPresentation,
) -> str:
    if presentation == "onscreen_spoken":
        return base_clause
    prefix = {
        "offscreen_spoken": f"({speech.speaker_id}), speaking offscreen",
        "voice_over": (
            f"({speech.speaker_id}), as a voice-over rather than visible speech"
        ),
        "message_voice_over": (
            f"({speech.speaker_id}), as a message voice-over rather than visible speech"
        ),
        "device_playback": (
            f"({speech.speaker_id}), heard through an in-scene device rather than "
            "visible speech"
        ),
        "uncertain": f"({speech.speaker_id}), with speech presentation uncertain",
    }[presentation]
    return f"{prefix}: {speech.locked_dialogue_block}"


__all__ = ["SpeechPresentation", "render_speech_presentation_clause"]
