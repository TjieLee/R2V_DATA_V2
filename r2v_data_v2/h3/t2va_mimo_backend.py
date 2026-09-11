"""T2VA semantics plus the frozen R2VA audio-finalize contract, without retries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from r2v_data_v2.h3.mimo25_backend import (
    AUDIO_FINALIZE_SYSTEM_PROMPT,
    MimoAudioFinalizeDraft,
    MimoMediaResolver,
    _completion_diagnostic,
    _validate_av_observation_usage,
    _validate_finish_reason,
)
from r2v_data_v2.h3.t2va_shadow import (
    T2VABackendProvenance,
    T2VACompletion,
    T2VAJob,
    T2VAMimoDraft,
    T2VARawResponse,
    check_audio_files,
    fingerprint,
    parse_t2va_semantic,
)

# No-reference adaptation of visual_only_v5, speech_assembly_v48,
# speaker_profile_v2 and audio_finalize_v6. Existing prompt constants stay frozen.
T2VA_SYSTEM_PROMPT = """Observe the ENTIRE original target video with its embedded audio and return one T2VAMimoDraft JSON object.
Original target AV is factual authority. The target video is observation input only, not conditioning media. The pipeline owns the three final section labels; do not put section headers inside the JSON strings.

INTEGRATED MULTIMODAL DESCRIPTION
Write a full generation-quality audiovisual description, not a short caption. Cover composition/framing, foreground/background, appearance when useful, environment and lighting/color, observable actions/state changes, gaze/expression, camera behavior, and meaningful early-to-middle-to-late progression.
Prefer observable description over plot interpretation. If object identity, function, material, setting, relationship, intention or psychology is uncertain, describe visible appearance instead of guessing. No hard word-count requirement. Do not pad or repeat facts merely to increase length.
Emphasize spatial relations, action, interaction, camera and temporal progression. Continue through the end of the clip rather than stopping after the opening composition and dialogue. Profile/back/occluded/silent people remain visually present; visual presence alone never assigns a speaker.
Write integrated_sequence as an ordered sequence of typed parts. A prose part contains kind=prose and text; a speech part contains kind=speech, segment_id and lead_in ONLY. This sequence becomes the final integrated multimodal description. Begin its first prose part with [Shot 1], followed by overall visual style and initial composition. The first shot has no timestamp. Use later sequential [Shot N] markers only for real cuts/transitions, beginning each with At MM:SS.mmm, with strictly increasing cut times inside the supplied target duration. Describe camera motion, amplitude and speed when observable, or the static camera. Preserve readable on-screen text verbatim in double quotes.
Use ordinary natural descriptions of people and objects. No production identities, internal cluster/segment IDs or conditioning labels belong in consumer prose. State observable events, not grounding rationale, diagnostic explanation or speculative source inference.

SPEECH AUTHORITY AND SPEAKERS
DiariZen owns exact supplied segment boundaries. Never split, filter, invent or change the speech facts. speaker_assignments must cover every supplied segment exactly once in supplied order and repeat its source_speaker_cluster unchanged.
Qwen3-ASR text and language remain exact and authoritative internally; transcript words are deliberately NOT supplied to you. Reproducing them is NOT your task. Never paraphrase, translate, correct, normalize, invent, drop or reorder dialogue. The deterministic renderer alone copies exact language/text and creates <d> syntax and speaker markers. Do NOT return ASR words, translations, transliterations or <d> tags in any prose or lead_in.
MANDATORY INVENTORIES: speaker_assignments = every supplied segment, exactly once in supplied order. Speech slots = dialogue_segment_ids only, exactly once in that explicit chronological order. Each supplied segment exposes segment_id, source_speaker_cluster, start_time, end_time, language and has_transcript, never transcript words. The flattened speech-part segment_id sequence MUST equal dialogue_segment_ids exactly. Non-transcribed segments still receive speaker assignments, but NEVER speech slots. If dialogue_segment_ids is empty, return no speech parts. Audible non-transcribed vocal events may be described naturally in prose without invented words.
A speech part has no dialogue-text field. Its lead_in describes ONLY the vocal source, presentation and audible delivery, not what was said. Sx identities belong ONLY in structured speaker_assignments.speaker_id. Model-owned prose, lead_in and audio fields must contain NO standalone S1/S2/... tokens, bare or parenthesized; neither S1 nor (S1) belongs there. The deterministic renderer is the sole producer of final (Sx) syntax.
Assign stable S1/S2/... only to actual vocalizing sources, numbered contiguously by first vocal appearance, not visible-person order. The same acoustic source cluster must not split across different speaker IDs. Acoustic grouping is evidence, not visual identity; clusters may share an Sx only when original AV strongly supports the same speaker. Pauses, language, sentences or segment boundaries alone never create speakers.
Pitch/register, timbre/texture, cadence/speaking rate and energy/delivery may support stable speaker assignment. Include only traits supported by audible evidence; never infer personality, nationality or profession from voice. No separate voice profile is requested.
The final target AV may identify an onscreen speaker directly. Do not require an independent visual speech cue. A visible person is not automatically the speaker merely because they are the only or most salient visible person. A visible listener must not inherit speech when the AV clearly indicates another source.
Keep each assignment's speech_presentation consistent with the final AV judgment: onscreen_spoken, offscreen_spoken, voice_over or uncertain. Offscreen is spatial, not a lack of supporting evidence; do not infer offscreen from uncertainty. When the source is offscreen, preserve visible people in visual prose and introduce the vocal source separately; do not attach its (Sx) to a visible listener.
Place each speech part among the surrounding visual/action prose at its observed playback position. speaker_assignments owns its Sx and presentation; the renderer places that correct (Sx) BEFORE every corresponding <d> block. Describe the actual source and delivery in lead_in without copying speech. For a voiceover, naturally use says in an off-screen voiceover. Preserve playback-order speech placement alongside actions/reactions. Do not invent a visual action to justify attribution.

Describe each stable visual fact once. After the initial composition, describe only meaningful new actions, state, camera or shot changes. Do not repeatedly restate unchanged posture, gaze, composition, atmosphere, silence or relationship merely to extend the description. Stop once the observed clip progression has been covered.
Only localized diegetic or shot-synchronized sounds that need a specific playback position belong in prose; do not append general ambience or background score summaries. Return warnings only for genuinely uncertain observations."""


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
            "speech_facts": [
                {
                    "segment_id": s.segment_id,
                    "source_speaker_cluster": s.source_speaker_cluster,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "language": s.language,
                    "has_transcript": s.text is not None,
                }
                for s in job.speech_facts
            ],
            "dialogue_segment_ids": [
                s.segment_id for s in job.speech_facts if s.text is not None
            ],
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

    def build_audio_finalize_request(self, job: T2VAJob) -> dict:
        if job.audio_evidence is None:
            raise ValueError("T2VA requires resolved music/sfx evidence")
        request = self.build_request(job)
        target_video = request["messages"][1]["content"][0]
        instruction = "Judge the non-dialogue target audio."
        schema = MimoAudioFinalizeDraft.model_json_schema()
        if self.config.transport == "xiaomi":
            instruction += "\nRESPONSE SCHEMA:\n" + json.dumps(
                schema, ensure_ascii=False, separators=(",", ":")
            )
        content = [target_video, {"type": "text", "text": instruction}]
        for kind in ("music", "sfx"):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"{kind} stem: separated audio of the SAME target",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": self.config.media_resolver.resolve(
                                Path(getattr(job.audio_evidence, f"{kind}_path"))
                            )
                        },
                    },
                ]
            )
        request["messages"] = [
            {"role": "system", "content": AUDIO_FINALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        if self.config.transport == "sglang":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "MimoAudioFinalizeDraft",
                    "schema": schema,
                    "strict": True,
                },
            }
        return request

    def _complete(
        self, request: dict, *, audio_finalize: bool = False
    ) -> T2VACompletion:
        raw = T2VACompletion(
            model_call_count=0,
            response=None,
            finish_reason=None,
            usage={},
            warnings=[],
            error=None,
        )
        try:
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
            if audio_finalize:
                diagnostic = _completion_diagnostic(
                    completion,
                    choice,
                    modality="target_video_audio_finalize",
                    http_attempt_count=1,
                    thinking="disabled",
                )
                _validate_finish_reason(diagnostic)
                _validate_av_observation_usage(
                    diagnostic,
                    require_explicit_audio=False,
                    require_reference_images=False,
                )
                if diagnostic.usage.audio_tokens == 0:
                    diagnostic.warnings.append("embedded_audio_tokens_zero")
                raw.warnings = diagnostic.warnings
                return raw
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

    def annotate(self, job: T2VAJob, request_fingerprint: str) -> T2VARawResponse:
        from r2v_data_v2.structured_output import parse_structured_json_response

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
            if job.audio_evidence is None:
                raise ValueError("T2VA requires resolved music/sfx evidence")
            check_audio_files(job.audio_evidence)
            semantic = self._complete(self.build_request(job))
            for key, value in semantic.model_dump().items():
                setattr(raw, key, value)
            raw.semantic_model_call_count = semantic.model_call_count
            if semantic.error:
                return raw
            _, corrections = parse_t2va_semantic(job, semantic.response or "")
            raw.deterministic_corrections = corrections
            raw.audio_finalize = self._complete(
                self.build_audio_finalize_request(job), audio_finalize=True
            )
            raw.model_call_count += raw.audio_finalize.model_call_count
            if raw.audio_finalize.error:
                raw.error = raw.audio_finalize.error
                return raw
            parse_structured_json_response(
                raw.audio_finalize.response or "", MimoAudioFinalizeDraft
            )
            check_audio_files(job.audio_evidence)
        except Exception as exc:  # noqa: BLE001 - persist both stages, never retry.
            raw.error = f"{type(exc).__name__}: {exc}"
        return raw
