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


def semantic_speech_source(
    *, speaker_id: str, subject_label: str | None, presentation: SpeechPresentation,
) -> tuple[str, str]:
    """Official Ref2VA source phrase and action; never invent speaker attributes."""
    if subject_label is not None:
        source = f"{subject_label} ({speaker_id})"
        if presentation == "offscreen_spoken":
            source += ", speaking off-screen,"
    else:
        description = {
            "onscreen_spoken": "An unidentified visible speaker",
            "offscreen_spoken": "An off-screen voice",
            "voice_over": "A voice-over",
            "message_voice_over": "A message voice-over",
            "device_playback": "A voice from an in-scene device",
            "uncertain": "An unidentified voice",
        }[presentation]
        source = f"{description} ({speaker_id})"
    action = ("says in an off-screen voiceover:"
              if presentation in {"voice_over", "message_voice_over"} else "says,")
    return source, action


def render_speech_presentation_clause(
    *,
    speech: SpeechPresentationFact,
    base_clause: str,
    presentation: SpeechPresentation,
) -> str:
    if presentation in {"onscreen_spoken", "uncertain"}:
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
    }[presentation]
    return f"{prefix}: {speech.locked_dialogue_block}"


__all__ = ["SpeechPresentation", "render_speech_presentation_clause"]
