"""One original-AV request; deliberately independent of Ref2VA's staged backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_CANONICAL_ABSENT_SOUNDSCAPE,
    MimoMediaResolver,
)
from r2v_data_v2.h3.t2va_shadow import (
    T2VABackendProvenance,
    T2VAJob,
    T2VAMimoDraft,
    T2VARawResponse,
    fingerprint,
)

# No-reference adaptation of visual_only_v5, speech_assembly_v48,
# speaker_profile_v2 and audio_finalize_v6. Existing prompt constants stay frozen.
T2VA_SYSTEM_PROMPT = (
    """Observe the ENTIRE original target video with its embedded audio and return one T2VAMimoDraft JSON object.
Original target AV is factual authority. The target video is observation input only, not conditioning media. The pipeline owns the three final section labels; do not put section headers inside the JSON strings.

INTEGRATED MULTIMODAL DESCRIPTION
Write a full generation-quality audiovisual description, not a short caption. Cover composition/framing, foreground/background, appearance when useful, environment and lighting/color, observable actions/state changes, gaze/expression, camera behavior, and meaningful early-to-middle-to-late progression.
Prefer observable description over plot interpretation. If object identity, function, material, setting, relationship, intention or psychology is uncertain, describe visible appearance instead of guessing. No hard word-count requirement. Do not pad or repeat facts merely to increase length.
Emphasize spatial relations, action, interaction, camera and temporal progression. Continue through the end of the clip rather than stopping after the opening composition and dialogue. Profile/back/occluded/silent people remain visually present; visual presence alone never assigns a speaker.
Begin integrated_multimodal_description with [Shot 1], followed by overall visual style and initial composition. The first shot has no timestamp. Use later sequential [Shot N] markers only for real cuts/transitions, beginning each with At MM:SS.mmm, with strictly increasing cut times inside the supplied target duration. Describe camera motion, amplitude and speed when observable, or the static camera. Preserve readable on-screen text verbatim in double quotes.
Use ordinary natural descriptions of people and objects. No production identities, internal cluster/segment IDs or conditioning labels belong in consumer prose. State observable events, not grounding rationale, diagnostic explanation or speculative source inference.

SPEECH AUTHORITY AND SPEAKERS
DiariZen owns exact supplied segment boundaries. Never split, filter, invent or change the speech facts. speaker_assignments must cover every supplied segment exactly once in supplied order and repeat its source_speaker_cluster unchanged.
Supplied Qwen3-ASR text and language are exact and authoritative. Never paraphrase, translate, correct, normalize, invent, drop or reorder dialogue. ONLY segments with non-null text receive a <d> block: exactly one block per such segment, in supplied chronological order. Inside each <d> put ONLY [Language] exact ASR text, with the supplied punctuation unchanged. No transcribed text means no <d> blocks; non-transcribed vocal events may be described without invented words.
Assign stable S1/S2/... only to actual vocalizing sources, numbered contiguously by first vocal appearance, not visible-person order. The same acoustic source cluster must not split across different speaker IDs. Acoustic grouping is evidence, not visual identity; clusters may share an Sx only when original AV strongly supports the same speaker. Pauses, language, sentences or segment boundaries alone never create speakers.
Pitch/register, timbre/texture, cadence/speaking rate and energy/delivery may support stable speaker assignment. Include only traits supported by audible evidence; never infer personality, nationality or profession from voice. No separate voice profile is requested.
The final target AV may identify an onscreen speaker directly. Do not require an independent visual speech cue. A visible person is not automatically the speaker merely because they are the only or most salient visible person. A visible listener must not inherit speech when the AV clearly indicates another source.
Keep each assignment's speech_presentation consistent with the final AV judgment: onscreen_spoken, offscreen_spoken, voice_over or uncertain. Offscreen is spatial, not a lack of supporting evidence; do not infer offscreen from uncertainty. When the source is offscreen, preserve visible people in visual prose and introduce the vocal source separately; do not attach its (Sx) to a visible listener.
Place the correct (Sx) in the natural speaker lead-in BEFORE every corresponding <d> block, never after </d>. Describe the actual source and delivery outside <d>. For a voiceover, naturally use says in an off-screen voiceover. Preserve playback-order dialogue placement alongside actions/reactions. Do not invent a visual action to justify attribution.

AUDIO FIELD BOUNDARIES
Original AV remains primary authority for all sounds. Only localized diegetic or shot-synchronized sound events that need a specific position in playback order belong in integrated_multimodal_description. Singing, in-scene instruments, radio/TV/phone music audible to characters belong there, not in audience-only score.
overall_soundscape describes ambience and physical/environmental, mechanical/electronic and non-verbal human sound, excluding dialogue, singing and music. Continuous room tone/hum/rumble belongs here, not as an appended sound summary in the integrated description. Do not hallucinate sounds from visible actions.
If no clearly discernible non-dialogue sound is established, use exactly: """
    + MIMO25_CANONICAL_ABSENT_SOUNDSCAPE
    + """
non_diegetic_music describes audience-only BGM, with audible instrumentation, tempo/rhythm and dynamics rather than emotional function. Preserve clearly audible music; do not output N/A for clearly established score. Use N/A only when no such music is established.
Do not summarize overall_soundscape or non_diegetic_music at the end of the integrated description. Return warnings only for genuinely uncertain observations; do not invent facts to fill a field."""
)


@dataclass(frozen=True)
class T2VAMimoConfig:
    media_resolver: MimoMediaResolver
    base_url: str
    api_key: str
    model: str = "mimo-v2.5"
    transport: Literal["sglang", "xiaomi"] = "sglang"
    max_completion_tokens: int = 32768
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if (
            not self.api_key.strip()
            or not self.base_url.strip()
            or self.model != "mimo-v2.5"
            or self.transport not in {"sglang", "xiaomi"}
            or self.max_completion_tokens <= 0
            or self.timeout_seconds <= 0
        ):
            raise ValueError("invalid one-call T2VA backend configuration")

    def provenance(self) -> T2VABackendProvenance:
        return T2VABackendProvenance(
            prompt_sha256=hashlib.sha256(T2VA_SYSTEM_PROMPT.encode()).hexdigest(),
            response_schema_sha256=fingerprint(T2VAMimoDraft.model_json_schema()),
            transport=self.transport,
            model=self.model,
            base_url=self.base_url,
            media_root=str(self.media_resolver.media_root),
            media_mode=self.media_resolver.mode,
            media_base_url=self.media_resolver.media_base_url,
            max_completion_tokens=self.max_completion_tokens,
        )


def _get(value: Any, key: str, default: Any = None) -> Any:
    return (
        value.get(key, default)
        if isinstance(value, dict)
        else getattr(value, key, default)
    )


class T2VAMimoBackend:
    def __init__(self, config: T2VAMimoConfig, *, client: Any = None) -> None:
        self.config = config
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        self.client = client

    def provenance(self) -> T2VABackendProvenance:
        return self.config.provenance()

    def build_request(self, job: T2VAJob) -> dict:
        contract = {
            "clip_uid": job.clip_uid,
            "target_duration_seconds": job.target_duration_seconds,
            "speech_facts": [s.model_dump(mode="json") for s in job.speech_facts],
        }
        request = {
            "model": self.config.model,
            "temperature": 0,
            "stream": False,
            "max_completion_tokens": self.config.max_completion_tokens,
            "messages": [
                {"role": "system", "content": T2VA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": self.config.media_resolver.resolve(
                                    Path(job.target_video_path)
                                ),
                            },
                            "fps": 4.0,
                            "media_resolution": "default",
                        },
                        {
                            "type": "text",
                            "text": json.dumps(contract, ensure_ascii=False),
                        },
                    ],
                },
            ],
        }
        if self.config.transport == "sglang":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "T2VAMimoDraft",
                    "schema": T2VAMimoDraft.model_json_schema(),
                    "strict": True,
                },
            }
            request["reasoning_effort"] = "none"
            request["extra_body"] = {
                "use_audio_in_video": True,
                "chat_template_kwargs": {
                    "thinking": False,
                    "enable_thinking": False,
                },
            }
        else:
            request["response_format"] = {"type": "json_object"}
            request["extra_body"] = {
                "use_audio_in_video": True,
                "thinking": {"type": "disabled"},
            }
        return request

    def annotate(self, job: T2VAJob, request_fingerprint: str) -> T2VARawResponse:
        raw = T2VARawResponse(
            clip_uid=job.clip_uid,
            request_fingerprint=request_fingerprint,
            model_call_count=0,
            response=None,
            finish_reason=None,
            usage={},
            warnings=[],
            error=None,
        )
        try:
            request = self.build_request(job)
            raw.model_call_count = 1
            completion = self.client.chat.completions.create(**request)
            usage = _get(completion, "usage")
            if usage is not None:
                raw.usage = (
                    usage if isinstance(usage, dict) else usage.model_dump(mode="json")
                )
            choices = _get(completion, "choices", [])
            if not choices:
                raise ValueError("T2VA response has no choices")
            choice = choices[0]
            raw.finish_reason = _get(choice, "finish_reason")
            content = _get(_get(choice, "message"), "content")
            if not isinstance(content, str):
                raise TypeError("T2VA response must be text")
            raw.response = content
            if raw.finish_reason not in {None, "stop"}:
                raise ValueError(f"T2VA incomplete response: {raw.finish_reason}")
            if raw.finish_reason is None:
                raw.warnings.append("finish_reason_unavailable")
            details = raw.usage.get("prompt_tokens_details") or {}
            for modality in ("audio", "video"):
                count = details.get(
                    f"{modality}_tokens", raw.usage.get(f"{modality}_tokens")
                )
                if count == 0:
                    raise ValueError(f"T2VA original AV {modality}_tokens is zero")
                if count is None:
                    raw.warnings.append(f"{modality}_tokens_unavailable")
        except Exception as exc:  # noqa: BLE001 - persist the sole SDK attempt, never retry.
            raw.error = f"{type(exc).__name__}: {exc}"
        return raw
