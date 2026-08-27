from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import threading
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from openai import OpenAI
from pydantic import Field, StrictStr, ValidationError, model_validator

from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    CanonicalAudioClipSummary,
)
from r2v_data_v2.h3.jea_diarization import JEAReadableDiarizationSegment
from r2v_data_v2.h3.jea_target_audio_caption import (
    JEA_TARGET_AUDIO_CAPTION_FALLBACK_POLICY_VERSION,
    JEATargetAudioCaptionBackendFailure,
    JEATargetAudioCaptionBackendResult,
    JEATargetAudioCaptionConfig,
    JEATargetAudioCaptionInventory,
    JEATargetAudioCaptionJob,
    JEATargetAudioCaptionPromptAdapter,
    OpenAIJEATargetAudioCaptionBackend,
    _is_all_semantic_null,
)
from r2v_data_v2.h3.jea_target_audio_caption import (
    _inventory_fingerprint as _legacy_inventory_fingerprint,
)
from r2v_data_v2.h3.jea_target_audio_caption import (
    _model_input as _local_model_input,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption_contract import (
    ModelSpeakerDelivery,
    SpeakerClusterEvidence,
    SpeakerTimeRange,
    TargetAudioCaptionResponse,
    TargetSpeakerDelivery,
    TemporalAudioEvent,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

SPECIALIZED_ROOT_NAME = "audio_semantics_specialized_v1"
CAPTIONER_RECORD_VERSION = "r2v.h3.specialized_audio_captioner.1"
GLOBAL_RECORD_VERSION = "r2v.h3.specialized_global_audio_semantics.3"
LOCAL_RECORD_VERSION = "r2v.h3.specialized_local_audio_semantics.3"
ASSEMBLED_RECORD_VERSION = "r2v.h3.specialized_audio_semantics.3"
STAGE_SUMMARY_VERSION = "r2v.h3.specialized_audio_stage_summary.1"
ASSEMBLED_SUMMARY_VERSION = "r2v.h3.specialized_audio_semantics_summary.3"
BACKEND_PROVENANCE_VERSION = "r2v.h3.specialized_audio_backend_provenance.1"

CAPTIONER_POLICY_VERSION = "qwen3_omni_native_audio_caption_v2"
GLOBAL_PROMPT_VERSION = "qwen3_vl_global_audio_extraction_v2"
GLOBAL_FALLBACK_PROMPT_VERSION = "qwen3_vl_global_audio_extraction_v2_recheck"
GLOBAL_FALLBACK_POLICY_VERSION = "global_audio_all_null_or_empty_recheck_v1"
GLOBAL_RECAPTION_RESCUE_POLICY_VERSION = (
    "global_failed_or_missing_description_recaption_once_v2"
)
MUSIC_RESCUE_PROMPT_VERSION = "h3_non_diegetic_music_direct_audio_v2"
MUSIC_RESCUE_POLICY_VERSION = "global_non_diegetic_music_direct_audio_once_v2"
GLOBAL_RUNTIME_POLICY_VERSION = (
    f"{GLOBAL_FALLBACK_POLICY_VERSION}+{GLOBAL_RECAPTION_RESCUE_POLICY_VERSION}"
    f"+{MUSIC_RESCUE_POLICY_VERSION}"
)
LOCAL_PROMPT_VERSION = "h3_specialized_local_audio_semantics_v2"
LOCAL_FALLBACK_PROMPT_VERSION = "h3_specialized_local_audio_semantics_v2_recheck"
LOCAL_FALLBACK_POLICY_VERSION = JEA_TARGET_AUDIO_CAPTION_FALLBACK_POLICY_VERSION
LOCAL_FIELD_POLICY_VERSION = "specialized_local_field_salvage_v2"
LOCAL_CAPTION_HINT_POLICY_VERSION = "specialized_local_caption_hint_audio_truth_v1"
LOCAL_EVENT_FILTER_POLICY_VERSION = "specialized_local_drop_speech_only_events_v2"
LOCAL_RUNTIME_POLICY_VERSION = (
    f"{LOCAL_FALLBACK_POLICY_VERSION}+{LOCAL_FIELD_POLICY_VERSION}"
    f"+{LOCAL_CAPTION_HINT_POLICY_VERSION}+{LOCAL_EVENT_FILTER_POLICY_VERSION}"
)

DEFAULT_CAPTIONER_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
DEFAULT_CAPTIONER_CHECKPOINT_ID = (
    "/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Captioner"
)
DEFAULT_GLOBAL_VL_MODEL = (
    "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
)
DEFAULT_GLOBAL_VL_CHECKPOINT_ID = DEFAULT_GLOBAL_VL_MODEL
DEFAULT_GLOBAL_VL_BASE_URL = "http://127.0.0.1:8000/v1"
EVENT_TIME_TOLERANCE_SECONDS = 0.10

StageRole = Literal["captioner", "global_semantics", "local_semantics"]
BackendRole = Literal[
    "captioner",
    "global_semantics",
    "local_semantics",
    "music_rescue",
]
StageStatus = Literal["ready", "failed", "blocked"]
InputModality = Literal[
    "canonical_full_audio_only",
    "captioner_text_only",
    "target_video_plus_canonical_full_audio",
]
SemanticSource = Literal[
    "primary",
    "fallback",
    "primary_all_null_confirmed",
    "fallback_all_null_confirmed",
]

SPECIALIZED_INVENTORY_VERSION = "r2v.h3.specialized_audio_semantics_inventory.1"


class SpecializedAudioSemanticsJob(SchemaModel):
    target_clip_uid: str
    clip_display_path: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    speaker_clusters: list[SpeakerClusterEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_job(self) -> SpecializedAudioSemanticsJob:
        if not self.target_clip_uid.strip() or not self.clip_display_path.strip():
            raise ValueError("specialized audio target identity must not be empty")
        if (self.target_audio_binding_path is None) != (
            self.target_audio_binding_sha256 is None
        ):
            raise ValueError("specialized audio binding path/hash must be paired")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("specialized speaker clusters must be unique")
        return self


class SpecializedAudioSemanticsInventory(SchemaModel):
    schema_version: Literal[
        "r2v.h3.specialized_audio_semantics_inventory.1"
    ] = SPECIALIZED_INVENTORY_VERSION
    source_audio_production_root: str
    source_canonical_audio_manifest_path: str
    source_canonical_audio_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_readable_segments_path: str | None = None
    source_readable_segments_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    target_clip_count: int = Field(gt=0)
    readable_segment_count: int = Field(ge=0)
    transcript_supplied_to_model: Literal[False] = False
    entity_id_supplied_to_model: Literal[False] = False
    donor_media_used: Literal[False] = False
    jobs: list[SpecializedAudioSemanticsJob]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> SpecializedAudioSemanticsInventory:
        if self.target_clip_count != len(self.jobs):
            raise ValueError("specialized target clip count is inconsistent")
        if (self.source_readable_segments_path is None) != (
            self.source_readable_segments_sha256 is None
        ):
            raise ValueError("specialized readable path/hash must be paired")
        if self.readable_segment_count and self.source_readable_segments_path is None:
            raise ValueError("readable segments require source provenance")
        clip_ids = [item.target_clip_uid for item in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("specialized target clips must be unique")
        if self.inventory_fingerprint != _specialized_inventory_fingerprint(self):
            raise ValueError("specialized inventory fingerprint is invalid")
        return self


GLOBAL_SYSTEM_PROMPT = """You extract reusable GLOBAL NON-VOCAL AUDIO facts from
one raw audio caption. EXTRACTION ONLY: every audible fact must be explicitly
supported by the supplied caption text. Never add facts through world knowledge,
dialogue meaning, identity, culture, history, genre, setting, likely scene, or
visual imagination. You may select, remove, compress, normalize, and conservatively
classify caption facts; you are not a fact-correction model.

Discard every human-language or vocal description, including transcript, quoted
or paraphrased dialogue, spoken words, language, gender, age, nationality,
identity, conversation, speech, voice, delivery, timbre, pitch, pace, prosody,
singing, lyrics, laughter, cough, sigh, crying, breathing, gasp, cheering,
shouting, and crowd chatter.

Preserve acoustic evidence while removing speculative source identity. If the
caption says "possibly a gong or war horn", use a generic description such as
"a sharp metallic or resonant sound". If it says "possibly guzheng or pipa",
use "a plucked-string instrumental melody". Never choose one uncertain source.

overall_audio_description: a concise high-recall whole-clip summary of supported
NON-VOCAL sound, including music, ambience, environmental or physical sounds,
effects, foreground/background, continuous/brief. No timestamps.

overall_soundscape: supported NON-MUSICAL room tone, ambience, environmental
noise, traffic, machinery, nature, physical sounds, or recurring/salient effects.
No music, vocals, scene semantics, or source speculation.

non_diegetic_music: extract clearly musical or music-like content supported by
the raw caption. Musical evidence includes music, musical, melody, melodic,
harmony, harmonic, instrumental passages, piano, strings, guitar, orchestral
instrumentation, synth or electronic musical layers, sustained tonal or harmonic
accompaniment, rhythmic musical backing, beats, instrumental beats, score,
soundtrack, BGM, background music, or a continuous musical bed beneath speech.
Do not require the caption to explicitly say "background music", "BGM", "score",
"soundtrack", or "non-diegetic". If clearly musical content is supported and its
source is unspecified, preserve it as non-diegetic/background music. Prefer recall
over abstention when evidence is clearly musical.

Exclude music only when the caption explicitly identifies an in-scene playback or
performance source, such as a radio, television, phone, loudspeaker, live band, a
person playing guitar or piano, or another explicitly described visible/in-scene
instrument performance. A hum, buzz, whine, ringing, drone, tone, or electronic
tone alone is not music unless accompanied by clear musical evidence such as
melodic, melody, harmonic, rhythmic, instrumental, accompaniment, musical, synth
melody, or a repeating musical phrase. Do not infer mood from plot.

Return exactly one compact JSON object matching the supplied schema. No markdown,
reasoning, explanation, or extra fields."""

GLOBAL_FALLBACK_SYSTEM_PROMPT = """Recheck the supplied raw caption for GLOBAL
NON-VOCAL AUDIO evidence only. Use only facts explicitly present in the caption.
Ignore all speech, language, identity, delivery, voice, and other human-vocal
content. Preserve generic acoustic evidence but never resolve speculative source
identity. Recheck for musical, melodic, harmonic, instrumental, rhythmic, synth
melody, or accompaniment evidence. Clearly musical content with an unspecified
source may be preserved as non_diegetic_music; exclude it only when the caption
explicitly identifies an in-scene playback or performance source such as radio,
television, phone, loudspeaker, live performance, or a person playing an
instrument. A hum, buzz, whine, drone, tone, or electronic tone alone is not
music. Return the same three-field schema as one compact JSON object."""

MUSIC_RESCUE_SYSTEM_PROMPT = """Listen specifically for clearly audible musical
accompaniment in this audio. This is a precision-oriented rescue pass. Return
non_diegetic_music only when the audio contains clear musical structure.

Strong musical evidence includes at least one of: a clearly audible melody or
repeating melodic phrase; harmony or chord progression; a rhythmic musical
pattern or beat; or clearly identifiable instrumental musical performance.

A sustained pad, ambient pad, ambient drone, electronic drone, low-frequency
drone, hum, buzz, hiss, whine, ringing, rumble, continuous tone, isolated
electronic tone, or vague tonal texture is not sufficient by itself. If it is
ambiguous whether a background layer is music or an environmental or electronic
tonal sound, return null. Do not invent a specific instrument unless its timbre
is clearly audible.

If clearly structured music is audible and there is no explicit audible evidence
that it comes from an in-scene radio, television, phone, loudspeaker, live
performance, or a person playing an instrument, describe it as
non_diegetic_music. Do not infer music from dialogue, emotion, plot, genre, or
expected scene context.

Return exactly one compact JSON object:
{"non_diegetic_music":"<concise English audible music description>"}
or {"non_diegetic_music":null}. No markdown, reasoning, explanation, or extra
fields."""

LOCAL_SYSTEM_PROMPT = """You extract two reusable LOCAL AUDIO SEMANTICS from one
target clip. The attached audio is the only source of truth. If video is also
attached, visual evidence may only disambiguate a sound already audible and can
never establish that a sound exists.

PASS 3 - TEMPORAL NON-LINGUISTIC AUDIO EVENTS

Actively scan the entire audio from beginning to end for discrete or temporally
localized NON-LINGUISTIC audible events. Include reliably audible events even
when they are brief, quiet, in the background, or partially masked by speech.
Examples include impacts, knocks, taps, metallic clinks, mechanical clicks,
object handling, cup or glass placement, pouring, splashing, footsteps, doors,
chair or fabric movement, rustling, paper movement, phone rings, electronic
chimes, bells, machinery changes, engine sounds, applause, laughter, coughs,
gasps, sighs, crying, non-lyrical singing, and diegetic music.

Never emit ordinary speech, dialogue, conversation, talking, an utterance, a
speech turn, "a person speaking", or "someone speaking" as an audio event.
Speech timing is already represented by authoritative upstream data and must not
be duplicated here. When speech overlaps a real non-linguistic event, describe
the non-linguistic event and do not treat the speech itself as the event. Return
an empty event list only when no localized non-linguistic event is reliably
audible after inspecting the whole clip. Do not manufacture events for recall.

Describe the audible sound itself. Do not copy or preserve uncertain source
guesses from the Captioner hint. Prefer "sharp metallic clink" over "sharp
metallic clink likely caused by a sword being drawn" when the source cannot be
reliably confirmed from the audio. Background score, BGM, and non-diegetic music
are not temporal_audio_events. Include music only when there is sufficient
audible evidence that it is localized diegetic or in-scene playback/performance,
such as a phone, radio, television, live performance, or another clearly
localized musical source. Do not duplicate Global non_diegetic_music here.

The supplied Captioner observation is AUXILIARY, UNTRUSTED ACOUSTIC SEARCH HINTS
ONLY. It may contain omissions, hallucinations, incorrect source or instrument
guesses, and false-negative statements. Use positive mentions only to remind
you what to actively search for in the audio. Publish an event only if listening
to the audio verifies it. Do not copy an event merely because the Captioner
mentions it, and do not suppress an audible event because the Captioner says it
is absent. The Captioner observation never supplies event timestamps.

PASS 4 - SPEAKER DELIVERY

For every supplied speaker cluster, describe concise audible speech prosody and
performance only: emotion, pace, energy, loudness, pitch tendency, rhythm,
articulation, hesitation, pauses, whispering, shouting, questioning, commanding,
or similarly useful traits. Ignore the Captioner observation completely for
speaker_delivery. Determine delivery only from the attached audio and supplied
speaker-cluster timing evidence.

Never transcribe, quote, paraphrase, correct, or summarize dialogue. Never infer
identity, gender, age, nationality, intrinsic voice identity, or timbre. Do not
emit entity_id. Return every supplied speaker_cluster_id exactly once; array
order is not semantic. If no speaker clusters are supplied, do not invent any;
return speaker_delivery as an empty array and still perform the full temporal
non-linguistic event pass. Set overall_soundscape and non_diegetic_music to null;
this specialized Local pass does not own those fields. Return exactly one compact
JSON object matching the supplied schema, with no markdown or explanation."""

LOCAL_FALLBACK_SYSTEM_PROMPT = """Reinspect the entire attached audio for the
same specialized LOCAL AUDIO SEMANTICS. Audio remains the only source of truth.
Actively check for brief, quiet, background, or speech-masked non-linguistic
events, but publish only sounds verified by listening. Never emit ordinary
speech, dialogue, conversation, talking, utterances, or speech turns as events.
Describe the audible sound itself. Do not copy or preserve uncertain source
guesses from the Captioner hint. Prefer "sharp metallic clink" over "sharp
metallic clink likely caused by a sword being drawn" when the source cannot be
reliably confirmed from the audio.

Background score, BGM, and non-diegetic music are not temporal_audio_events.
Include music as a temporal event only when there is sufficient audible evidence
that it is localized diegetic or in-scene playback/performance, such as a phone,
radio, television, live performance, or clearly localized musical source. Do not
duplicate Global non_diegetic_music in this Local pass.

The auxiliary Captioner observation is untrusted search hints only: positive
mentions require audio verification and negative claims cannot veto audible
sound. Ignore it completely for speaker_delivery, which must come only from the
audio and supplied speaker timing. If no speaker clusters are supplied, do not
invent any; return speaker_delivery as an empty array and still perform the full
temporal non-linguistic event pass. Preserve all identity, transcript, timing,
and output-schema constraints from the primary pass. Set overall_soundscape and
non_diegetic_music to null and return one compact JSON object only."""


def _local_user_prompt(
    job: JEATargetAudioCaptionJob,
    raw_audio_caption: str | None,
    *,
    fallback: bool,
) -> str:
    instruction = (
        "Recheck the attached audio under the specialized Local policy."
        if fallback
        else "Extract the specialized Local audio semantics."
    )
    caption_hint = (
        "[unavailable]" if raw_audio_caption is None else raw_audio_caption
    )
    return (
        f"{instruction}\nTask metadata / supplied speaker clusters:\n"
        f"{_compact_json(_local_model_input(job))}\n"
        "Auxiliary Captioner observation "
        "(UNTRUSTED SEARCH HINTS ONLY; AUDIO IS SOURCE OF TRUTH):\n"
        f"{caption_hint}\nJSON schema:\n"
        f"{_compact_json(TargetAudioCaptionResponse.model_json_schema())}"
    )


def _local_repair_prompt(
    *,
    job: JEATargetAudioCaptionJob,
    raw_audio_caption: str | None,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
    fallback: bool,
) -> str:
    return (
        "Repair the previous JSON format only. Preserve the same audio-only fact "
        "policy: Captioner text remains untrusted event-search hints, event facts "
        "and times require audio verification, and speaker_delivery must ignore "
        "Captioner text. Do not add facts, transcript, identity, or entity_id. "
        "Return one compact JSON object.\nOriginal request:\n"
        f"{_local_user_prompt(job, raw_audio_caption, fallback=fallback)}\n"
        "Validation issues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\n"
        f"Invalid response:\n{invalid_response}"
    )


@dataclass(frozen=True)
class _SpecializedLocalPromptAdapter(JEATargetAudioCaptionPromptAdapter):
    system_prompt: str = LOCAL_SYSTEM_PROMPT
    fallback_system_prompt: str = LOCAL_FALLBACK_SYSTEM_PROMPT

    def user_prompt(
        self,
        job: JEATargetAudioCaptionJob,
        auxiliary_observation: str | None,
    ) -> str:
        return _local_user_prompt(job, auxiliary_observation, fallback=False)

    def fallback_user_prompt(
        self,
        job: JEATargetAudioCaptionJob,
        auxiliary_observation: str | None,
    ) -> str:
        return _local_user_prompt(job, auxiliary_observation, fallback=True)

    def repair_prompt(
        self,
        *,
        job: JEATargetAudioCaptionJob,
        auxiliary_observation: str | None,
        invalid_response: str,
        issues: Sequence[ValidationIssue],
    ) -> str:
        return _local_repair_prompt(
            job=job,
            raw_audio_caption=auxiliary_observation,
            invalid_response=invalid_response,
            issues=issues,
            fallback=False,
        )

    def fallback_repair_prompt(
        self,
        *,
        job: JEATargetAudioCaptionJob,
        auxiliary_observation: str | None,
        invalid_response: str,
        issues: Sequence[ValidationIssue],
    ) -> str:
        return _local_repair_prompt(
            job=job,
            raw_audio_caption=auxiliary_observation,
            invalid_response=invalid_response,
            issues=issues,
            fallback=True,
        )

def _fingerprint_payload(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _fingerprint_payload(item)
            for key, item in value.items()
            if not (key == "runtime_dependency_fingerprints" and item is None)
        }
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    return value


def _compact_json(value: object) -> str:
    return json.dumps(
        _fingerprint_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _specialized_inventory_fingerprint(
    inventory: SpecializedAudioSemanticsInventory | dict[str, object],
) -> str:
    if isinstance(inventory, SpecializedAudioSemanticsInventory):
        values = inventory.model_dump(mode="json")
    else:
        payload = dict(inventory)
        payload.pop("inventory_fingerprint", None)
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            raise TypeError("specialized inventory jobs must be a list")
        payload["jobs"] = [
            SpecializedAudioSemanticsJob.model_validate(item) for item in jobs
        ]
        values = SpecializedAudioSemanticsInventory.model_construct(
            **payload,
            inventory_fingerprint="",
        ).model_dump(mode="json")
    values.pop("inventory_fingerprint", None)
    return _sha256_text(_compact_json(values))


def _verify_file(path_value: str, expected_hash: str, *, field_name: str) -> Path:
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise ValueError(f"{field_name} changed or is unavailable")
    return path


def _json_schema_response_format(
    model: type[SchemaModel], *, name: str
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }


class CompletionDiagnostic(SchemaModel):
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    raw_content_char_count: int = Field(ge=0)
    non_whitespace_content_char_count: int = Field(ge=0)
    whitespace_only: bool


class SpecializedStageFailure(SchemaModel):
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    attempt_count: int = Field(ge=0)
    issues: list[dict[str, str | None]] = Field(default_factory=list)


class SpecializedBackendProvenance(SchemaModel):
    schema_version: Literal[
        "r2v.h3.specialized_audio_backend_provenance.1"
    ] = BACKEND_PROVENANCE_VERSION
    backend: Literal["vllm"] = "vllm"
    role: BackendRole
    served_model_name: str
    checkpoint_id: str
    base_url: str
    input_modality: InputModality
    media_mode: Literal["none", "file", "http"]
    media_root: str | None = None
    media_base_url: str | None = None
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    prompt_version: str
    fallback_prompt_version: str | None = None
    fallback_policy_version: str
    temperature: float = Field(ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    runtime_dependency_fingerprints: dict[str, str] | None = None
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> SpecializedBackendProvenance:
        if any(
            not value.strip()
            for value in (
                self.served_model_name,
                self.checkpoint_id,
                self.base_url,
                self.prompt_version,
                self.fallback_policy_version,
            )
        ):
            raise ValueError("specialized backend provenance is incomplete")
        if self.media_mode == "none":
            if self.media_root is not None or self.media_base_url is not None:
                raise ValueError("text-only provenance cannot publish media roots")
        else:
            if self.media_root is None or not self.media_root.strip():
                raise ValueError("media backend provenance requires media root")
            if (self.media_mode == "http") != (self.media_base_url is not None):
                raise ValueError("specialized HTTP media provenance is inconsistent")
        expected_modalities: dict[BackendRole, set[InputModality]] = {
            "captioner": {"canonical_full_audio_only"},
            "global_semantics": {"captioner_text_only"},
            "local_semantics": {
                "canonical_full_audio_only",
                "target_video_plus_canonical_full_audio",
            },
            "music_rescue": {"canonical_full_audio_only"},
        }
        if self.input_modality not in expected_modalities[self.role]:
            raise ValueError("specialized backend role and modality differ")
        if self.role == "captioner":
            if self.temperature <= 0 or self.top_p is None or self.top_k is None:
                raise ValueError("captioner provenance requires sampling parameters")
        elif self.temperature != 0 or self.top_p is not None or self.top_k is not None:
            raise ValueError("structured semantics provenance must be deterministic")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.runtime_dependency_fingerprints is None:
            values.pop("runtime_dependency_fingerprints")
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("specialized backend fingerprint is invalid")
        return self


class GlobalAudioSemanticsResponse(SchemaModel):
    overall_audio_description: StrictStr | None = None
    overall_soundscape: StrictStr | None = None
    non_diegetic_music: StrictStr | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> GlobalAudioSemanticsResponse:
        for field_name in (
            "overall_audio_description",
            "overall_soundscape",
            "non_diegetic_music",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty or null")
        return self


class DirectAudioMusicResponse(SchemaModel):
    non_diegetic_music: StrictStr | None = None

    @model_validator(mode="after")
    def validate_music(self) -> DirectAudioMusicResponse:
        if (
            self.non_diegetic_music is not None
            and not self.non_diegetic_music.strip()
        ):
            raise ValueError("non_diegetic_music must be non-empty or null")
        return self


class LocalAudioSemanticsResponse(SchemaModel):
    temporal_audio_events: list[TemporalAudioEvent] = Field(default_factory=list)
    speaker_delivery: list[ModelSpeakerDelivery]

    @model_validator(mode="after")
    def validate_fields(self) -> LocalAudioSemanticsResponse:
        self.temporal_audio_events = sorted(
            self.temporal_audio_events,
            key=lambda item: (item.start_time, item.end_time, item.description),
        )
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("local speaker delivery cluster IDs must be unique")
        return self


def _is_global_null(response: GlobalAudioSemanticsResponse) -> bool:
    return all(
        getattr(response, field_name) is None
        for field_name in (
            "overall_audio_description",
            "overall_soundscape",
            "non_diegetic_music",
        )
    )


def _local_response_issues(
    response: LocalAudioSemanticsResponse,
    job: JEATargetAudioCaptionJob,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected = [item.speaker_cluster_id for item in job.speaker_clusters]
    actual = [item.speaker_cluster_id for item in response.speaker_delivery]
    if len(actual) != len(set(actual)):
        issues.append(
            ValidationIssue(
                code="duplicate_speaker_cluster",
                field="speaker_delivery",
                message="speaker delivery cluster IDs must be unique",
            )
        )
    elif set(actual) - set(expected):
        issues.append(
            ValidationIssue(
                code="unknown_speaker_cluster",
                field="speaker_delivery",
                message="speaker delivery contains an unknown cluster",
            )
        )
    elif set(expected) - set(actual):
        issues.append(
            ValidationIssue(
                code="missing_speaker_cluster",
                field="speaker_delivery",
                message="speaker delivery is missing a supplied cluster",
            )
        )
    for index, event in enumerate(response.temporal_audio_events):
        if event.end_time > job.target_duration_seconds + EVENT_TIME_TOLERANCE_SECONDS:
            issues.append(
                ValidationIssue(
                    code="audio_event_out_of_bounds",
                    field=f"temporal_audio_events.{index}.end_time",
                    message="temporal audio event exceeds target duration",
                )
            )
    return issues


_PURE_SPEECH_EVENT_DESCRIPTIONS = {
    "conversation",
    "dialogue",
    "person speaking",
    "person talking",
    "someone speaking",
    "someone talking",
    "speech",
    "speech turn",
    "talking",
    "utterance",
}

_PURE_SPEECH_TEMPLATE = re.compile(
    r"^(?:(?:male|female)\s+)?(?:person|speaker|voice)\s+"
    r"(?:(?:continues?|continued)\s+)?(?:speaks?|speaking|talks?|talking)\b"
    r"(?P<qualifiers>.*)$"
)
_SPEECH_QUALIFIER_WORDS = {
    "a",
    "accented",
    "an",
    "and",
    "anxious",
    "breathless",
    "calm",
    "cantonese",
    "chinese",
    "commanding",
    "conversational",
    "delivery",
    "desperate",
    "emotional",
    "energetic",
    "energy",
    "english",
    "even",
    "falling",
    "fast",
    "french",
    "hesitation",
    "high",
    "in",
    "japanese",
    "korean",
    "language",
    "loud",
    "low",
    "mandarin",
    "measured",
    "neutral",
    "pace",
    "pause",
    "pauses",
    "pitch",
    "prosody",
    "questioning",
    "quiet",
    "rapid",
    "rhythm",
    "rising",
    "shouting",
    "slightly",
    "slow",
    "soft",
    "spanish",
    "strained",
    "style",
    "subdued",
    "the",
    "tone",
    "urgent",
    "very",
    "whispering",
    "with",
}


def _is_pure_speech_event(description: str) -> bool:
    normalized = " ".join(description.strip().lower().split()).rstrip(" .,!?:;")
    for article in ("a ", "an ", "the "):
        if normalized.startswith(article):
            normalized = normalized[len(article) :]
            break
    if normalized in _PURE_SPEECH_EVENT_DESCRIPTIONS:
        return True
    match = _PURE_SPEECH_TEMPLATE.fullmatch(normalized)
    if match is None:
        return False
    qualifier_words = re.findall(r"[a-z]+(?:-[a-z]+)?", match.group("qualifiers"))
    return all(word in _SPEECH_QUALIFIER_WORDS for word in qualifier_words)


def _normalize_local_response(
    response: LocalAudioSemanticsResponse,
    job: JEATargetAudioCaptionJob,
) -> LocalAudioSemanticsResponse:
    delivery_by_cluster = {
        item.speaker_cluster_id: item for item in response.speaker_delivery
    }
    return response.model_copy(
        update={
            "temporal_audio_events": [
                event
                for event in response.temporal_audio_events
                if not _is_pure_speech_event(event.description)
            ],
            "speaker_delivery": [
                delivery_by_cluster[item.speaker_cluster_id]
                for item in job.speaker_clusters
            ]
        }
    )


def _json_string_end(value: str, start: int) -> int | None:
    escaped = False
    for index in range(start + 1, len(value)):
        character = value[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return index + 1
    return None


def _matching_json_array_end(value: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(value):
        character = value[index]
        if character == '"':
            string_end = _json_string_end(value, index)
            if string_end is None:
                return None
            index = string_end
            continue
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
        index += 1
    return None


def _extract_top_level_json_array(raw: str, key: str) -> list[object] | None:
    try:
        value = normalize_structured_json_envelope(raw)
    except ValueError:
        return None
    if not value.startswith("{") or not value.endswith("}"):
        return None
    object_depth = 0
    array_depth = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == '"':
            string_end = _json_string_end(value, index)
            if string_end is None:
                return None
            if object_depth == 1 and array_depth == 0:
                try:
                    candidate_key = json.loads(value[index:string_end])
                except json.JSONDecodeError:
                    return None
                cursor = string_end
                while cursor < len(value) and value[cursor].isspace():
                    cursor += 1
                if candidate_key == key and cursor < len(value) and value[cursor] == ":":
                    cursor += 1
                    while cursor < len(value) and value[cursor].isspace():
                        cursor += 1
                    if cursor >= len(value) or value[cursor] != "[":
                        return None
                    array_end = _matching_json_array_end(value, cursor)
                    if array_end is None:
                        return None
                    try:
                        payload = json.loads(value[cursor:array_end])
                    except json.JSONDecodeError:
                        return None
                    return payload if isinstance(payload, list) else None
            index = string_end
            continue
        if character == "{":
            object_depth += 1
        elif character == "}":
            object_depth -= 1
            if object_depth < 0:
                return None
        elif character == "[":
            array_depth += 1
        elif character == "]":
            array_depth -= 1
            if array_depth < 0:
                return None
        index += 1
    return None


def _salvage_speaker_delivery(
    raw_responses: Sequence[str],
    job: JEATargetAudioCaptionJob,
) -> list[ModelSpeakerDelivery]:
    expected = [item.speaker_cluster_id for item in job.speaker_clusters]
    for raw in reversed(raw_responses):
        response, _ = parse_structured_json_issues(raw, TargetAudioCaptionResponse)
        payload: Sequence[object] | None = (
            response.speaker_delivery
            if response is not None
            else _extract_top_level_json_array(raw, "speaker_delivery")
        )
        if payload is None:
            continue
        try:
            delivery = [ModelSpeakerDelivery.model_validate(item) for item in payload]
        except (TypeError, ValidationError):
            continue
        actual = [item.speaker_cluster_id for item in delivery]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            continue
        delivery_by_cluster = {item.speaker_cluster_id: item for item in delivery}
        return [delivery_by_cluster[cluster_id] for cluster_id in expected]
    return []


@dataclass(frozen=True)
class CaptionerConfig:
    base_url: str
    api_key: str
    served_model_name: str
    checkpoint_id: str
    media_resolver: MediaURLResolver
    timeout_seconds: float = 600.0
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 16384

    def __post_init__(self) -> None:
        _validate_runtime_config(self)

    def provenance(self) -> SpecializedBackendProvenance:
        return _provenance(
            role="captioner",
            served_model_name=self.served_model_name,
            checkpoint_id=self.checkpoint_id,
            base_url=self.base_url,
            input_modality="canonical_full_audio_only",
            media_resolver=self.media_resolver,
            prompt_version=CAPTIONER_POLICY_VERSION,
            fallback_prompt_version=None,
            fallback_policy_version="captioner_empty_retry_once_v1",
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class GlobalSemanticsConfig:
    base_url: str
    api_key: str
    served_model_name: str = DEFAULT_GLOBAL_VL_MODEL
    checkpoint_id: str = DEFAULT_GLOBAL_VL_CHECKPOINT_ID
    timeout_seconds: float = 600.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        _validate_runtime_config(self)

    def provenance(self) -> SpecializedBackendProvenance:
        return _provenance(
            role="global_semantics",
            served_model_name=self.served_model_name,
            checkpoint_id=self.checkpoint_id,
            base_url=self.base_url,
            input_modality="captioner_text_only",
            media_resolver=None,
            prompt_version=GLOBAL_PROMPT_VERSION,
            fallback_prompt_version=GLOBAL_FALLBACK_PROMPT_VERSION,
            fallback_policy_version=GLOBAL_RUNTIME_POLICY_VERSION,
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class LocalSemanticsConfig:
    base_url: str
    api_key: str
    served_model_name: str
    checkpoint_id: str
    media_resolver: MediaURLResolver
    include_video: bool = False
    timeout_seconds: float = 600.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        _validate_runtime_config(self)

    def provenance(self) -> SpecializedBackendProvenance:
        return _provenance(
            role="local_semantics",
            served_model_name=self.served_model_name,
            checkpoint_id=self.checkpoint_id,
            base_url=self.base_url,
            input_modality=(
                "target_video_plus_canonical_full_audio"
                if self.include_video
                else "canonical_full_audio_only"
            ),
            media_resolver=self.media_resolver,
            prompt_version=LOCAL_PROMPT_VERSION,
            fallback_prompt_version=LOCAL_FALLBACK_PROMPT_VERSION,
            fallback_policy_version=LOCAL_RUNTIME_POLICY_VERSION,
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class MusicRescueConfig:
    base_url: str
    api_key: str
    served_model_name: str
    checkpoint_id: str
    media_resolver: MediaURLResolver
    timeout_seconds: float = 600.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        _validate_runtime_config(self)

    def provenance(self) -> SpecializedBackendProvenance:
        return _provenance(
            role="music_rescue",
            served_model_name=self.served_model_name,
            checkpoint_id=self.checkpoint_id,
            base_url=self.base_url,
            input_modality="canonical_full_audio_only",
            media_resolver=self.media_resolver,
            prompt_version=MUSIC_RESCUE_PROMPT_VERSION,
            fallback_prompt_version=f"{MUSIC_RESCUE_PROMPT_VERSION}_repair",
            fallback_policy_version=MUSIC_RESCUE_POLICY_VERSION,
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=self.max_tokens,
        )


def _validate_runtime_config(
    config: (
        CaptionerConfig
        | GlobalSemanticsConfig
        | LocalSemanticsConfig
        | MusicRescueConfig
    ),
) -> None:
    values = [
        getattr(config, field_name)
        for field_name in ("base_url", "api_key", "served_model_name", "checkpoint_id")
    ]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("specialized backend endpoint and model are required")
    timeout_seconds = float(config.timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("specialized backend timeout must be positive")
    if (
        not isinstance(config.max_tokens, int)
        or isinstance(config.max_tokens, bool)
        or config.max_tokens < 1
    ):
        raise ValueError("specialized backend max tokens must be positive")
    if isinstance(config, CaptionerConfig):
        if (
            isinstance(config.temperature, bool)
            or not isinstance(config.temperature, (int, float))
            or not math.isfinite(config.temperature)
            or config.temperature <= 0
        ):
            raise ValueError("captioner temperature must be positive")
        if (
            isinstance(config.top_p, bool)
            or not isinstance(config.top_p, (int, float))
            or not math.isfinite(config.top_p)
            or not 0 < config.top_p <= 1
        ):
            raise ValueError("captioner top-p must be in (0, 1]")
        if (
            not isinstance(config.top_k, int)
            or isinstance(config.top_k, bool)
            or config.top_k < 1
        ):
            raise ValueError("captioner top-k must be positive")


def _provenance(
    *,
    role: BackendRole,
    served_model_name: str,
    checkpoint_id: str,
    base_url: str,
    input_modality: InputModality,
    media_resolver: MediaURLResolver | None,
    prompt_version: str,
    fallback_prompt_version: str | None,
    fallback_policy_version: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
) -> SpecializedBackendProvenance:
    values: dict[str, object] = {
        "schema_version": BACKEND_PROVENANCE_VERSION,
        "backend": "vllm",
        "role": role,
        "served_model_name": served_model_name,
        "checkpoint_id": checkpoint_id,
        "base_url": base_url,
        "input_modality": input_modality,
        "media_mode": "none" if media_resolver is None else media_resolver.mode,
        "media_root": None if media_resolver is None else str(media_resolver.media_root),
        "media_base_url": (
            None if media_resolver is None else media_resolver.media_base_url
        ),
        "output_modalities": ["text"],
        "prompt_version": prompt_version,
        "fallback_prompt_version": fallback_prompt_version,
        "fallback_policy_version": fallback_policy_version,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "repair_retries": 1,
    }
    return SpecializedBackendProvenance(
        **values,
        configuration_fingerprint=_sha256_text(_compact_json(values)),
    )


def _global_runtime_provenance(
    global_provenance: SpecializedBackendProvenance,
    captioner_provenance: SpecializedBackendProvenance,
    music_provenance: SpecializedBackendProvenance,
) -> SpecializedBackendProvenance:
    if global_provenance.role != "global_semantics":
        raise ValueError("global runtime requires global semantics provenance")
    if captioner_provenance.role != "captioner":
        raise ValueError("global runtime requires captioner provenance")
    if music_provenance.role != "music_rescue":
        raise ValueError("global runtime requires music rescue provenance")
    values = global_provenance.model_dump(
        mode="json",
        exclude={"configuration_fingerprint", "runtime_dependency_fingerprints"},
    )
    values["runtime_dependency_fingerprints"] = {
        "recaption_captioner": captioner_provenance.configuration_fingerprint,
        "direct_audio_music_rescue": music_provenance.configuration_fingerprint,
    }
    return SpecializedBackendProvenance(
        **values,
        configuration_fingerprint=_sha256_text(_compact_json(values)),
    )


def _optional_value(value: object, field_name: str) -> object | None:
    return value.get(field_name) if isinstance(value, dict) else getattr(
        value, field_name, None
    )


def _diagnostic(completion: object, choice: object, content: object) -> CompletionDiagnostic:
    usage = getattr(completion, "usage", None)
    text = content if isinstance(content, str) else None

    def token(name: str) -> int | None:
        value = _optional_value(usage, name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    finish_reason = _optional_value(choice, "finish_reason")
    return CompletionDiagnostic(
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        prompt_tokens=token("prompt_tokens"),
        completion_tokens=token("completion_tokens"),
        total_tokens=token("total_tokens"),
        raw_content_char_count=0 if text is None else len(text),
        non_whitespace_content_char_count=sum(
            not character.isspace() for character in (text or "")
        ),
        whitespace_only=text is not None and not text.strip(),
    )


class SpecializedBackendFailure(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: Sequence[str] = (),
        diagnostics: Sequence[CompletionDiagnostic] = (),
        issues: Sequence[ValidationIssue] = (),
        model_call_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = tuple(raw_responses)
        self.diagnostics = tuple(diagnostics)
        self.issues = tuple(issues)
        self.model_call_count = model_call_count


class _StaleSpecializedDependencyError(ValueError):
    pass


def _combined_fallback_failure(
    *,
    primary_raw_responses: Sequence[str],
    primary_diagnostics: Sequence[CompletionDiagnostic],
    primary_model_call_count: int,
    fallback: SpecializedBackendFailure,
) -> SpecializedBackendFailure:
    return SpecializedBackendFailure(
        code=fallback.code,
        reason=fallback.reason,
        raw_responses=(*primary_raw_responses, *fallback.raw_responses),
        diagnostics=(*primary_diagnostics, *fallback.diagnostics),
        issues=fallback.issues,
        model_call_count=primary_model_call_count + fallback.model_call_count,
    )


ResponseT = TypeVar("ResponseT", bound=SchemaModel)


@dataclass(frozen=True)
class SpecializedBackendResult[ResponseT]:
    response: ResponseT
    raw_responses: tuple[str, ...]
    diagnostics: tuple[CompletionDiagnostic, ...]
    model_call_count: int
    semantic_source: SemanticSource = "primary"
    fallback_attempted: bool = False


@dataclass(frozen=True)
class CaptionerBackendResult:
    raw_audio_caption: str
    raw_responses: tuple[str, ...]
    diagnostics: tuple[CompletionDiagnostic, ...]
    model_call_count: int


class CaptionerBackend(Protocol):
    @property
    def provenance(self) -> SpecializedBackendProvenance: ...

    def caption(self, job: JEATargetAudioCaptionJob) -> CaptionerBackendResult: ...


class GlobalSemanticsBackend(Protocol):
    @property
    def provenance(self) -> SpecializedBackendProvenance: ...

    def extract(
        self, job: JEATargetAudioCaptionJob, raw_audio_caption: str
    ) -> SpecializedBackendResult[GlobalAudioSemanticsResponse]: ...


class LocalSemanticsBackend(Protocol):
    @property
    def provenance(self) -> SpecializedBackendProvenance: ...

    def describe(
        self,
        job: JEATargetAudioCaptionJob,
        raw_audio_caption: str | None = None,
    ) -> SpecializedBackendResult[LocalAudioSemanticsResponse]: ...


class MusicRescueBackend(Protocol):
    @property
    def provenance(self) -> SpecializedBackendProvenance: ...

    def detect(
        self, job: JEATargetAudioCaptionJob
    ) -> SpecializedBackendResult[DirectAudioMusicResponse]: ...


class _OpenAIBackendBase:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        client: Any | None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._injected_client = client
        self._thread_local = threading.local()

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout_seconds,
            )
            self._thread_local.client = client
        return client

    def _call(
        self, request: dict[str, object], *, role: str
    ) -> tuple[str, CompletionDiagnostic]:
        try:
            completion = self._client().chat.completions.create(**request)
        except Exception as exc:
            raise SpecializedBackendFailure(
                code=f"{role}_vllm_request_failed",
                reason=f"{type(exc).__name__}: {exc}",
                model_call_count=1,
            ) from exc
        diagnostic: CompletionDiagnostic | None = None
        try:
            choices = getattr(completion, "choices", None)
            if not choices:
                raise TypeError("specialized audio response has no choices")
            choice = choices[0]
            content = getattr(choice.message, "content", None)
            diagnostic = _diagnostic(completion, choice, content)
            if not isinstance(content, str):
                raise TypeError("specialized audio response text must be a string")
            return content, diagnostic
        except Exception as exc:
            raise SpecializedBackendFailure(
                code=f"{role}_invalid_response",
                reason=f"{type(exc).__name__}: {exc}",
                diagnostics=(() if diagnostic is None else (diagnostic,)),
                model_call_count=1,
            ) from exc


class OpenAICaptionerBackend(_OpenAIBackendBase):
    def __init__(self, config: CaptionerConfig, *, client: Any | None = None) -> None:
        super().__init__(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            client=client,
        )
        self.config = config

    @property
    def provenance(self) -> SpecializedBackendProvenance:
        return self.config.provenance()

    def caption(self, job: JEATargetAudioCaptionJob) -> CaptionerBackendResult:
        try:
            audio_path = _verify_file(
                job.target_full_audio_path,
                job.target_full_audio_sha256,
                field_name="target full audio",
            )
        except (OSError, ValueError) as exc:
            raise SpecializedBackendFailure(
                code="captioner_media_unavailable",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        request: dict[str, object] = {
            "model": self.config.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": self.config.media_resolver.resolve(audio_path)
                            },
                        }
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "modalities": ["text"],
            "stream": False,
            "extra_body": {"top_k": self.config.top_k},
        }
        raw_responses: list[str] = []
        diagnostics: list[CompletionDiagnostic] = []
        invalid_failure: SpecializedBackendFailure | None = None
        for _ in range(2):
            try:
                raw, diagnostic = self._call(request, role="captioner")
            except SpecializedBackendFailure as exc:
                if exc.code != "captioner_invalid_response":
                    raise
                invalid_failure = exc
                raw_responses.append("")
                diagnostics.extend(exc.diagnostics)
                continue
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            if raw.strip():
                return CaptionerBackendResult(
                    raw_audio_caption=raw,
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    model_call_count=len(raw_responses),
                )
        raise SpecializedBackendFailure(
            code=(
                "captioner_invalid_response"
                if invalid_failure is not None
                else "captioner_empty_response"
            ),
            reason=(
                "captioner returned invalid text after one retry"
                if invalid_failure is not None
                else "captioner returned empty text after one retry"
            ),
            raw_responses=raw_responses,
            diagnostics=diagnostics,
            model_call_count=len(raw_responses),
        )


def _global_user_prompt(raw_audio_caption: str, *, fallback: bool) -> str:
    instruction = (
        "Recheck the raw caption for supported non-vocal facts."
        if fallback
        else "Extract supported global non-vocal audio facts."
    )
    return (
        f"{instruction}\nRaw Captioner observation (untrusted except as the only "
        "available evidence):\n"
        f"{raw_audio_caption}\nJSON schema:\n"
        f"{_compact_json(GlobalAudioSemanticsResponse.model_json_schema())}"
    )


def _global_repair_prompt(
    raw_audio_caption: str,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
    *,
    fallback: bool,
) -> str:
    return (
        "Repair the prior JSON only. Keep the same extraction-only policy and do "
        "not add facts. Return one compact JSON object.\nOriginal request:\n"
        f"{_global_user_prompt(raw_audio_caption, fallback=fallback)}\nIssues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\nInvalid response:\n"
        f"{invalid_response}"
    )


class OpenAIGlobalSemanticsBackend(_OpenAIBackendBase):
    def __init__(
        self, config: GlobalSemanticsConfig, *, client: Any | None = None
    ) -> None:
        super().__init__(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            client=client,
        )
        self.config = config

    @property
    def provenance(self) -> SpecializedBackendProvenance:
        return self.config.provenance()

    def _pass(
        self,
        raw_audio_caption: str,
        *,
        fallback: bool,
    ) -> SpecializedBackendResult[GlobalAudioSemanticsResponse]:
        raw_responses: list[str] = []
        diagnostics: list[CompletionDiagnostic] = []
        issues: list[ValidationIssue] = []
        system_prompt = (
            GLOBAL_FALLBACK_SYSTEM_PROMPT if fallback else GLOBAL_SYSTEM_PROMPT
        )
        for attempt in range(2):
            prompt = (
                _global_user_prompt(raw_audio_caption, fallback=fallback)
                if attempt == 0
                else _global_repair_prompt(
                    raw_audio_caption,
                    raw_responses[-1],
                    issues,
                    fallback=fallback,
                )
            )
            request: dict[str, object] = {
                "model": self.config.served_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                "response_format": _json_schema_response_format(
                    GlobalAudioSemanticsResponse,
                    name="global_audio_semantics",
                ),
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            try:
                raw, diagnostic = self._call(request, role="global_semantics")
            except SpecializedBackendFailure as exc:
                if exc.code != "global_semantics_invalid_response":
                    raise
                raw = ""
                diagnostics.extend(exc.diagnostics)
                raw_responses.append(raw)
                issues = [
                    ValidationIssue(
                        code="invalid_response_type",
                        field=None,
                        message=exc.reason,
                    )
                ]
                continue
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            if not raw.strip():
                raise SpecializedBackendFailure(
                    code="global_semantics_empty_response",
                    reason="global semantics returned empty text",
                    raw_responses=raw_responses,
                    diagnostics=diagnostics,
                    model_call_count=len(raw_responses),
                )
            response, issues = parse_structured_json_issues(
                raw,
                GlobalAudioSemanticsResponse,
            )
            if response is not None:
                return SpecializedBackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    model_call_count=len(raw_responses),
                    semantic_source="fallback" if fallback else "primary",
                    fallback_attempted=fallback,
                )
        raise SpecializedBackendFailure(
            code="global_semantics_structured_output_failed",
            reason="global semantics failed closed after one repair",
            raw_responses=raw_responses,
            diagnostics=diagnostics,
            issues=issues,
            model_call_count=len(raw_responses),
        )

    def extract(
        self, job: JEATargetAudioCaptionJob, raw_audio_caption: str
    ) -> SpecializedBackendResult[GlobalAudioSemanticsResponse]:
        del job
        try:
            primary = self._pass(raw_audio_caption, fallback=False)
        except SpecializedBackendFailure as exc:
            if exc.code != "global_semantics_empty_response":
                raise
            try:
                fallback = self._pass(raw_audio_caption, fallback=True)
            except SpecializedBackendFailure as fallback_exc:
                raise _combined_fallback_failure(
                    primary_raw_responses=exc.raw_responses,
                    primary_diagnostics=exc.diagnostics,
                    primary_model_call_count=exc.model_call_count,
                    fallback=fallback_exc,
                ) from fallback_exc
            return SpecializedBackendResult(
                response=fallback.response,
                raw_responses=(*exc.raw_responses, *fallback.raw_responses),
                diagnostics=(*exc.diagnostics, *fallback.diagnostics),
                model_call_count=exc.model_call_count + fallback.model_call_count,
                semantic_source=(
                    "fallback_all_null_confirmed"
                    if _is_global_null(fallback.response)
                    else "fallback"
                ),
                fallback_attempted=True,
            )
        if not _is_global_null(primary.response):
            return primary
        try:
            fallback = self._pass(raw_audio_caption, fallback=True)
        except SpecializedBackendFailure as fallback_exc:
            raise _combined_fallback_failure(
                primary_raw_responses=primary.raw_responses,
                primary_diagnostics=primary.diagnostics,
                primary_model_call_count=primary.model_call_count,
                fallback=fallback_exc,
            ) from fallback_exc
        return SpecializedBackendResult(
            response=fallback.response,
            raw_responses=(*primary.raw_responses, *fallback.raw_responses),
            diagnostics=(*primary.diagnostics, *fallback.diagnostics),
            model_call_count=primary.model_call_count + fallback.model_call_count,
            semantic_source=(
                "fallback_all_null_confirmed"
                if _is_global_null(fallback.response)
                else "fallback"
            ),
            fallback_attempted=True,
        )


class OpenAIMusicRescueBackend(_OpenAIBackendBase):
    def __init__(
        self, config: MusicRescueConfig, *, client: Any | None = None
    ) -> None:
        super().__init__(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            client=client,
        )
        self.config = config

    @property
    def provenance(self) -> SpecializedBackendProvenance:
        return self.config.provenance()

    def detect(
        self, job: JEATargetAudioCaptionJob
    ) -> SpecializedBackendResult[DirectAudioMusicResponse]:
        try:
            audio_path = _verify_file(
                job.target_full_audio_path,
                job.target_full_audio_sha256,
                field_name="target full audio",
            )
        except (OSError, ValueError) as exc:
            raise SpecializedBackendFailure(
                code="music_rescue_media_unavailable",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        raw_responses: list[str] = []
        diagnostics: list[CompletionDiagnostic] = []
        issues: list[ValidationIssue] = []
        audio_item = {
            "type": "audio_url",
            "audio_url": {"url": self.config.media_resolver.resolve(audio_path)},
        }
        for attempt in range(2):
            content: list[dict[str, object]] = [audio_item]
            if attempt:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Repair the prior JSON format only. Follow the same "
                            "music-listening policy, add no fields, and return one "
                            "compact JSON object.\nIssues:\n"
                            f"{_compact_json([item.to_dict() for item in issues])}"
                            "\nInvalid response:\n"
                            f"{raw_responses[-1]}"
                        ),
                    }
                )
            request: dict[str, object] = {
                "model": self.config.served_model_name,
                "messages": [
                    {"role": "system", "content": MUSIC_RESCUE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "max_tokens": self.config.max_tokens,
                "modalities": ["text"],
                "stream": False,
                "response_format": _json_schema_response_format(
                    DirectAudioMusicResponse,
                    name="direct_audio_music",
                ),
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            try:
                raw, diagnostic = self._call(request, role="music_rescue")
            except SpecializedBackendFailure as exc:
                if exc.code != "music_rescue_invalid_response":
                    raise
                raw_responses.append("")
                diagnostics.extend(exc.diagnostics)
                issues = [
                    ValidationIssue(
                        code="invalid_response_type",
                        field=None,
                        message=exc.reason,
                    )
                ]
                continue
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            if not raw.strip():
                raise SpecializedBackendFailure(
                    code="music_rescue_empty_response",
                    reason="direct-audio music rescue returned empty text",
                    raw_responses=raw_responses,
                    diagnostics=diagnostics,
                    model_call_count=len(raw_responses),
                )
            response, issues = parse_structured_json_issues(
                raw,
                DirectAudioMusicResponse,
            )
            if response is not None:
                return SpecializedBackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    model_call_count=len(raw_responses),
                )
        raise SpecializedBackendFailure(
            code="music_rescue_structured_output_failed",
            reason="direct-audio music rescue failed closed after one repair",
            raw_responses=raw_responses,
            diagnostics=diagnostics,
            issues=issues,
            model_call_count=len(raw_responses),
        )


def _specialized_diagnostics(
    diagnostics: Sequence[Any],
) -> tuple[CompletionDiagnostic, ...]:
    return tuple(
        CompletionDiagnostic.model_validate(item.to_dict()) for item in diagnostics
    )


def _specialized_local_failure(
    failure: JEATargetAudioCaptionBackendFailure,
) -> SpecializedBackendFailure:
    code = {
        "qwen3_omni_vllm_empty_response": "local_semantics_empty_response",
        "qwen3_omni_vllm_request_failed": "local_semantics_vllm_request_failed",
        "structured_output_failed": "local_semantics_structured_output_failed",
    }.get(failure.code, f"local_semantics_{failure.code}")
    return SpecializedBackendFailure(
        code=code,
        reason=failure.reason,
        raw_responses=failure.raw_responses,
        diagnostics=_specialized_diagnostics(failure.completion_diagnostics),
        issues=failure.issues,
        model_call_count=failure.model_call_count,
    )


def _project_local_result(
    result: JEATargetAudioCaptionBackendResult,
    job: JEATargetAudioCaptionJob,
    *,
    semantic_source: SemanticSource,
    fallback_attempted: bool,
    primary_raw_responses: Sequence[str] = (),
    primary_diagnostics: Sequence[CompletionDiagnostic] = (),
    primary_model_call_count: int = 0,
) -> SpecializedBackendResult[LocalAudioSemanticsResponse]:
    diagnostics = _specialized_diagnostics(result.completion_diagnostics)
    response = _normalize_local_response(
        LocalAudioSemanticsResponse(
            temporal_audio_events=result.response.temporal_audio_events,
            speaker_delivery=result.response.speaker_delivery,
        ),
        job,
    )
    return SpecializedBackendResult(
        response=response,
        raw_responses=(*primary_raw_responses, *result.raw_responses),
        diagnostics=(*primary_diagnostics, *diagnostics),
        model_call_count=primary_model_call_count + (result.model_call_count or 0),
        semantic_source=semantic_source,
        fallback_attempted=fallback_attempted,
    )


class OpenAILocalSemanticsBackend:
    def __init__(
        self, config: LocalSemanticsConfig, *, client: Any | None = None
    ) -> None:
        self.config = config
        self._delegate = OpenAIJEATargetAudioCaptionBackend(
            JEATargetAudioCaptionConfig(
                backend_family="qwen3_omni",
                base_url=config.base_url,
                api_key=config.api_key,
                served_model_name=config.served_model_name,
                checkpoint_id=config.checkpoint_id,
                media_resolver=config.media_resolver,
                include_video=config.include_video,
                timeout_seconds=config.timeout_seconds,
                max_tokens=config.max_tokens,
            ),
            client=client,
            prompt_adapter=_SpecializedLocalPromptAdapter(),
        )

    @property
    def provenance(self) -> SpecializedBackendProvenance:
        return self.config.provenance()

    def _verify_media(self, job: JEATargetAudioCaptionJob) -> None:
        try:
            _verify_file(
                job.target_full_audio_path,
                job.target_full_audio_sha256,
                field_name="target full audio",
            )
            if self.config.include_video:
                _verify_file(
                    job.target_video_path,
                    job.target_video_sha256,
                    field_name="target video",
                )
        except (OSError, ValueError) as exc:
            raise SpecializedBackendFailure(
                code="local_semantics_media_unavailable",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _fallback_after_failure(
        self,
        job: JEATargetAudioCaptionJob,
        raw_audio_caption: str | None,
        primary: SpecializedBackendFailure,
    ) -> SpecializedBackendResult[LocalAudioSemanticsResponse]:
        try:
            fallback = self._delegate.describe_semantic_fallback(
                job,
                auxiliary_observation=raw_audio_caption,
            )
        except JEATargetAudioCaptionBackendFailure as exc:
            fallback_failure = _specialized_local_failure(exc)
            raise _combined_fallback_failure(
                primary_raw_responses=primary.raw_responses,
                primary_diagnostics=primary.diagnostics,
                primary_model_call_count=primary.model_call_count,
                fallback=fallback_failure,
            ) from exc
        return _project_local_result(
            fallback,
            job,
            semantic_source=(
                "fallback_all_null_confirmed"
                if _is_all_semantic_null(fallback.response)
                else "fallback"
            ),
            fallback_attempted=True,
            primary_raw_responses=primary.raw_responses,
            primary_diagnostics=primary.diagnostics,
            primary_model_call_count=primary.model_call_count,
        )

    def describe(
        self,
        job: JEATargetAudioCaptionJob,
        raw_audio_caption: str | None = None,
    ) -> SpecializedBackendResult[LocalAudioSemanticsResponse]:
        self._verify_media(job)
        try:
            primary = self._delegate.describe(
                job,
                auxiliary_observation=raw_audio_caption,
            )
        except JEATargetAudioCaptionBackendFailure as exc:
            primary_failure = _specialized_local_failure(exc)
            if primary_failure.code != "local_semantics_empty_response":
                raise primary_failure from exc
            return self._fallback_after_failure(
                job,
                raw_audio_caption,
                primary_failure,
            )
        if not _is_all_semantic_null(primary.response) or not job.speaker_clusters:
            return _project_local_result(
                primary,
                job,
                semantic_source="primary",
                fallback_attempted=False,
            )
        primary_diagnostics = _specialized_diagnostics(
            primary.completion_diagnostics
        )
        try:
            fallback = self._delegate.describe_semantic_fallback(
                job,
                auxiliary_observation=raw_audio_caption,
            )
        except JEATargetAudioCaptionBackendFailure as exc:
            fallback_failure = _specialized_local_failure(exc)
            raise _combined_fallback_failure(
                primary_raw_responses=primary.raw_responses,
                primary_diagnostics=primary_diagnostics,
                primary_model_call_count=primary.model_call_count or 0,
                fallback=fallback_failure,
            ) from exc
        return _project_local_result(
            fallback,
            job,
            semantic_source=(
                "fallback_all_null_confirmed"
                if _is_all_semantic_null(fallback.response)
                else "fallback"
            ),
            fallback_attempted=True,
            primary_raw_responses=primary.raw_responses,
            primary_diagnostics=primary_diagnostics,
            primary_model_call_count=primary.model_call_count or 0,
        )


class CaptionerRecord(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_audio_captioner.1"] = (
        CAPTIONER_RECORD_VERSION
    )
    target_clip_uid: str
    clip_display_path: str
    status: Literal["ready", "failed"]
    raw_audio_caption: str | None = None
    backend_provenance: SpecializedBackendProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    completion_diagnostics: list[CompletionDiagnostic] = Field(default_factory=list)
    failure: SpecializedStageFailure | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> CaptionerRecord:
        if self.backend_provenance.role != "captioner":
            raise ValueError("captioner record provenance role differs")
        if self.raw_response_count != len(self.completion_diagnostics):
            raise ValueError("captioner response diagnostics differ")
        if self.status == "ready":
            if self.raw_audio_caption is None or not self.raw_audio_caption.strip():
                raise ValueError("ready captioner record requires raw caption")
            if self.failure is not None:
                raise ValueError("ready captioner record cannot contain failure")
        elif self.raw_audio_caption is not None or self.failure is None:
            raise ValueError("failed captioner record must contain only failure")
        _validate_record_fingerprint(self)
        return self


class GlobalSemanticsRecord(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_global_audio_semantics.3"] = (
        GLOBAL_RECORD_VERSION
    )
    target_clip_uid: str
    clip_display_path: str
    status: StageStatus
    overall_audio_description: str | None = None
    overall_soundscape: str | None = None
    non_diegetic_music: str | None = None
    semantic_source: SemanticSource | None = None
    fallback_attempted: bool = False
    recaption_rescue_attempted: bool = False
    recaption_rescue_used: bool = False
    music_rescue_attempted: bool = False
    music_rescue_used: bool = False
    captioner_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_audio_caption_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    backend_provenance: SpecializedBackendProvenance
    request_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    completion_diagnostics: list[CompletionDiagnostic] = Field(default_factory=list)
    failure: SpecializedStageFailure | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> GlobalSemanticsRecord:
        if self.backend_provenance.role != "global_semantics":
            raise ValueError("global record provenance role differs")
        if self.raw_response_count != len(self.completion_diagnostics):
            raise ValueError("global response diagnostics differ")
        if self.recaption_rescue_used and not self.recaption_rescue_attempted:
            raise ValueError("used recaption rescue must be attempted")
        if self.music_rescue_used and not self.music_rescue_attempted:
            raise ValueError("used music rescue must be attempted")
        values = (
            self.overall_audio_description,
            self.overall_soundscape,
            self.non_diegetic_music,
        )
        if self.status == "ready":
            if (
                self.semantic_source is None
                or self.raw_audio_caption_sha256 is None
                or self.request_fingerprint is None
                or self.failure is not None
            ):
                raise ValueError("ready global semantics record is incomplete")
        elif any(value is not None for value in values) or self.semantic_source is not None:
            raise ValueError("non-ready global record cannot publish semantics")
        if self.status == "blocked" and (
            self.request_fingerprint is not None
            or self.raw_audio_caption_sha256 is not None
            or self.model_call_count
            or self.raw_response_count
            or self.recaption_rescue_attempted
            or self.recaption_rescue_used
            or self.music_rescue_attempted
            or self.music_rescue_used
            or self.failure is None
        ):
            raise ValueError("blocked global record must not claim model work")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed global record requires failure")
        _validate_record_fingerprint(self)
        return self


class LocalSemanticsRecord(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_local_audio_semantics.3"] = (
        LOCAL_RECORD_VERSION
    )
    target_clip_uid: str
    clip_display_path: str
    status: Literal["ready", "failed"]
    temporal_audio_events: list[TemporalAudioEvent] = Field(default_factory=list)
    speaker_delivery: list[ModelSpeakerDelivery] = Field(default_factory=list)
    semantic_source: SemanticSource | None = None
    fallback_attempted: bool = False
    captioner_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_audio_caption_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    backend_provenance: SpecializedBackendProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    completion_diagnostics: list[CompletionDiagnostic] = Field(default_factory=list)
    failure: SpecializedStageFailure | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> LocalSemanticsRecord:
        if self.backend_provenance.role != "local_semantics":
            raise ValueError("local record provenance role differs")
        if self.raw_response_count != len(self.completion_diagnostics):
            raise ValueError("local response diagnostics differ")
        if self.status == "ready":
            if self.semantic_source is None or self.failure is not None:
                raise ValueError("ready local semantics record is incomplete")
        elif self.temporal_audio_events:
            raise ValueError("failed local semantics record cannot publish audio events")
        elif self.semantic_source is not None or self.failure is None:
            raise ValueError("failed local semantics record requires only failure state")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("local record speaker cluster IDs must be unique")
        _validate_record_fingerprint(self)
        return self


class SpecializedAudioSemanticsRecord(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_audio_semantics.3"] = (
        ASSEMBLED_RECORD_VERSION
    )
    target_clip_uid: str
    clip_display_path: str
    status: Literal["complete", "partial", "failed"]
    raw_audio_caption: str | None = None
    overall_audio_description: str | None = None
    overall_soundscape: str | None = None
    non_diegetic_music: str | None = None
    temporal_audio_events: list[TemporalAudioEvent] = Field(default_factory=list)
    speaker_delivery: list[TargetSpeakerDelivery] = Field(default_factory=list)
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    captioner_status: Literal["ready", "failed"]
    global_semantics_status: StageStatus
    local_semantics_status: Literal["ready", "failed"]
    captioner_provenance: SpecializedBackendProvenance
    global_semantics_provenance: SpecializedBackendProvenance
    local_semantics_provenance: SpecializedBackendProvenance
    captioner_failure: SpecializedStageFailure | None = None
    global_semantics_failure: SpecializedStageFailure | None = None
    local_semantics_failure: SpecializedStageFailure | None = None
    captioner_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_semantics_request_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    local_semantics_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    captioner_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_semantics_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_semantics_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assemble_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> SpecializedAudioSemanticsRecord:
        statuses = (
            self.captioner_status,
            self.global_semantics_status,
            self.local_semantics_status,
        )
        ready_count = sum(status == "ready" for status in statuses)
        expected = "complete" if ready_count == 3 else "partial" if ready_count else "failed"
        if self.status != expected:
            raise ValueError("assembled specialized status is inconsistent")
        if (self.target_audio_binding_path is None) != (
            self.target_audio_binding_sha256 is None
        ):
            raise ValueError("assembled audio binding path/hash must be paired")
        if self.captioner_status != ("failed" if self.raw_audio_caption is None else "ready"):
            raise ValueError("assembled raw caption status is inconsistent")
        if self.global_semantics_status != "ready" and any(
            value is not None
            for value in (
                self.overall_audio_description,
                self.overall_soundscape,
                self.non_diegetic_music,
            )
        ):
            raise ValueError("non-ready global stage cannot publish global semantics")
        if self.local_semantics_status != "ready" and self.temporal_audio_events:
            raise ValueError("failed local stage cannot publish temporal audio events")
        if (self.local_semantics_status == "ready") != (
            self.local_semantics_failure is None
        ):
            raise ValueError("assembled local failure state is inconsistent")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("assembled speaker delivery cluster IDs must be unique")
        values = self.model_dump(mode="json", exclude={"assemble_fingerprint"})
        if self.assemble_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("assembled specialized fingerprint is invalid")
        return self


class SpecializedStageSummary(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_audio_stage_summary.1"] = (
        STAGE_SUMMARY_VERSION
    )
    stage: StageRole
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: SpecializedBackendProvenance
    record_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    failure_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_summary(self) -> SpecializedStageSummary:
        if self.stage != self.backend_provenance.role:
            raise ValueError("stage summary provenance role differs")
        if self.record_count != self.ready_count + self.failed_count + self.blocked_count:
            raise ValueError("stage summary record counts differ")
        if self.stage != "global_semantics" and self.blocked_count:
            raise ValueError("only global semantics may be blocked")
        return self


class SpecializedAssembledSummary(SchemaModel):
    schema_version: Literal["r2v.h3.specialized_audio_semantics_summary.3"] = (
        ASSEMBLED_SUMMARY_VERSION
    )
    assembled_record_schema_version: Literal[
        "r2v.h3.specialized_audio_semantics.3"
    ] = ASSEMBLED_RECORD_VERSION
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=0)
    complete_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    captioner_ready_count: int = Field(ge=0)
    global_ready_count: int = Field(ge=0)
    global_blocked_count: int = Field(ge=0)
    local_ready_count: int = Field(ge=0)
    model_call_count: Literal[0] = 0

    @model_validator(mode="after")
    def validate_summary(self) -> SpecializedAssembledSummary:
        if self.record_count != self.complete_count + self.partial_count + self.failed_count:
            raise ValueError("assembled summary record counts differ")
        return self


def _record_fingerprint(model: SchemaModel) -> str:
    values = model.model_dump(mode="json", exclude={"record_fingerprint"})
    return _sha256_text(_compact_json(values))


def _validate_record_fingerprint(model: Any) -> None:
    if model.record_fingerprint != _record_fingerprint(model):
        raise ValueError("specialized stage record fingerprint is invalid")


def captioner_request_fingerprint(
    job: JEATargetAudioCaptionJob,
    provenance: SpecializedBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "target_full_audio_sha256": job.target_full_audio_sha256,
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
                "policy_version": CAPTIONER_POLICY_VERSION,
            }
        )
    )


def global_request_fingerprint(
    raw_audio_caption: str,
    provenance: SpecializedBackendProvenance,
    *,
    target_full_audio_sha256: str | None = None,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "raw_audio_caption_sha256": _sha256_text(raw_audio_caption),
                "target_full_audio_sha256": target_full_audio_sha256,
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
                "prompt_version": GLOBAL_PROMPT_VERSION,
                "fallback_prompt_version": GLOBAL_FALLBACK_PROMPT_VERSION,
                "fallback_policy_version": GLOBAL_FALLBACK_POLICY_VERSION,
                "recaption_rescue_policy_version": (
                    GLOBAL_RECAPTION_RESCUE_POLICY_VERSION
                ),
                "music_rescue_prompt_version": MUSIC_RESCUE_PROMPT_VERSION,
                "music_rescue_policy_version": MUSIC_RESCUE_POLICY_VERSION,
            }
        )
    )


def local_request_fingerprint(
    job: JEATargetAudioCaptionJob,
    provenance: SpecializedBackendProvenance,
    captioner_record: CaptionerRecord,
) -> str:
    raw_caption = captioner_record.raw_audio_caption
    return _sha256_text(
        _compact_json(
            {
                "target_full_audio_sha256": job.target_full_audio_sha256,
                "target_video_sha256": (
                    job.target_video_sha256
                    if provenance.input_modality
                    == "target_video_plus_canonical_full_audio"
                    else None
                ),
                "model_input": _local_model_input(job),
                "speaker_cluster_evidence": [
                    cluster.model_dump(mode="json")
                    for cluster in job.speaker_clusters
                ],
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
                "prompt_version": LOCAL_PROMPT_VERSION,
                "fallback_prompt_version": LOCAL_FALLBACK_PROMPT_VERSION,
                "fallback_policy_version": LOCAL_FALLBACK_POLICY_VERSION,
                "field_policy_version": LOCAL_FIELD_POLICY_VERSION,
                "caption_hint_policy_version": LOCAL_CAPTION_HINT_POLICY_VERSION,
                "event_filter_policy_version": LOCAL_EVENT_FILTER_POLICY_VERSION,
                "captioner_record_fingerprint": (
                    captioner_record.record_fingerprint
                ),
                "raw_audio_caption_sha256": (
                    None if raw_caption is None else _sha256_text(raw_caption)
                ),
            }
        )
    )


def specialized_output_root(audio_production_root: Path) -> Path:
    return audio_production_root.expanduser().resolve(strict=True) / SPECIALIZED_ROOT_NAME


def _stage_path(root: Path, role: StageRole | Literal["assembled"]) -> Path:
    return root / role


def _write_json(path: Path, value: SchemaModel | dict[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            _compact_json(record.model_dump(mode="json")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path, model: type[ResponseT]) -> list[ResponseT]:
    records: list[ResponseT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid specialized record {line_number}: {path}") from exc
    return records


def _ensure_root(
    inventory: JEATargetAudioCaptionInventory | SpecializedAudioSemanticsInventory,
    output_root: Path,
) -> Path:
    if isinstance(inventory, SpecializedAudioSemanticsInventory):
        if inventory.inventory_fingerprint != _specialized_inventory_fingerprint(
            inventory
        ):
            raise ValueError("specialized inventory fingerprint is inconsistent")
        sources = [
            (
                inventory.source_canonical_audio_manifest_path,
                inventory.source_canonical_audio_manifest_sha256,
                "source canonical audio manifest",
            )
        ]
        if inventory.source_readable_segments_path is not None:
            assert inventory.source_readable_segments_sha256 is not None
            sources.append(
                (
                    inventory.source_readable_segments_path,
                    inventory.source_readable_segments_sha256,
                    "source readable segments",
                )
            )
    else:
        if inventory.inventory_fingerprint != _legacy_inventory_fingerprint(inventory):
            raise ValueError("specialized inventory fingerprint is inconsistent")
        sources = [
            (inventory.source_pairs_path, inventory.source_pairs_sha256, "source pairs"),
            (
                inventory.source_readable_segments_path,
                inventory.source_readable_segments_sha256,
                "source readable segments",
            ),
            (
                inventory.source_qwen3_asr_segments_path,
                inventory.source_qwen3_asr_segments_sha256,
                "source Qwen3 ASR segments",
            ),
        ]
    for path_value, expected_hash, field_name in sources:
        _verify_file(path_value, expected_hash, field_name=field_name)
    production_root = Path(inventory.source_audio_production_root).resolve(strict=True)
    expected_root = production_root / SPECIALIZED_ROOT_NAME
    root = output_root.expanduser().resolve(strict=False)
    if root != expected_root:
        raise ValueError("specialized audio semantics must use the fixed output root")
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "inventory.json"
    if inventory_path.exists():
        inventory_model = (
            SpecializedAudioSemanticsInventory
            if isinstance(inventory, SpecializedAudioSemanticsInventory)
            else JEATargetAudioCaptionInventory
        )
        existing = inventory_model.model_validate_json(
            inventory_path.read_text(encoding="utf-8")
        )
        if existing != inventory:
            raise ValueError("specialized output inventory differs from current input")
    else:
        _write_json(inventory_path, inventory)
    return root


def _replace_stage_directory(temporary: Path, destination: Path) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def _assert_overwrite_ownership(
    destination: Path,
    *,
    expected_role: StageRole | Literal["assembled"],
) -> None:
    try:
        payload = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(
            f"cannot establish specialized output ownership: {destination}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"specialized output summary is not an object: {destination}")
    if expected_role == "assembled":
        owned = (
            payload.get("schema_version"),
            payload.get("assembled_record_schema_version"),
        ) in {
            (
                "r2v.h3.specialized_audio_semantics_summary.1",
                "r2v.h3.specialized_audio_semantics.1",
            ),
            (ASSEMBLED_SUMMARY_VERSION, ASSEMBLED_RECORD_VERSION),
        }
    else:
        provenance = payload.get("backend_provenance")
        owned = (
            payload.get("schema_version") == STAGE_SUMMARY_VERSION
            and payload.get("stage") == expected_role
            and isinstance(provenance, dict)
            and provenance.get("role") == expected_role
        )
    if not owned:
        raise ValueError(
            f"existing output is not owned by specialized {expected_role}: "
            f"{destination}"
        )


@dataclass(frozen=True)
class _ProcessedRecord[ResponseT]:
    record: ResponseT
    raw: dict[str, object]


def _published_failure(exc: SpecializedBackendFailure) -> SpecializedStageFailure:
    return SpecializedStageFailure(
        code=exc.code,
        reason=exc.reason,
        attempt_count=max(exc.model_call_count, len(exc.raw_responses)),
        issues=[item.to_dict() for item in exc.issues],
    )


def _raw_payload(
    *,
    status: StageStatus,
    raw_responses: Sequence[str],
    diagnostics: Sequence[CompletionDiagnostic],
    semantic_source: SemanticSource | None = None,
    failure: SpecializedBackendFailure | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "semantic_source": semantic_source,
        "raw_responses": list(raw_responses),
        "completion_diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        "failure": (
            None
            if failure is None
            else {
                "code": failure.code,
                "reason": failure.reason,
                "issues": [item.to_dict() for item in failure.issues],
            }
        ),
    }


def _captioner_record(
    job: JEATargetAudioCaptionJob,
    backend: CaptionerBackend,
) -> _ProcessedRecord[CaptionerRecord]:
    provenance = backend.provenance
    request_fingerprint = captioner_request_fingerprint(job, provenance)
    try:
        result = backend.caption(job)
        values: dict[str, object] = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "ready",
            "raw_audio_caption": result.raw_audio_caption,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": result.model_call_count,
            "raw_response_count": len(result.raw_responses),
            "completion_diagnostics": list(result.diagnostics),
            "failure": None,
        }
        record = CaptionerRecord(
            **values,
            record_fingerprint=_sha256_text(
                _compact_json(
                    CaptionerRecord.model_construct(
                        **values,
                        record_fingerprint="",
                    ).model_dump(mode="json", exclude={"record_fingerprint"})
                )
            ),
        )
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="ready",
                raw_responses=result.raw_responses,
                diagnostics=result.diagnostics,
            ),
        )
    except SpecializedBackendFailure as exc:
        values = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "failed",
            "raw_audio_caption": None,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": exc.model_call_count,
            "raw_response_count": len(exc.raw_responses),
            "completion_diagnostics": list(exc.diagnostics),
            "failure": _published_failure(exc),
        }
        record = CaptionerRecord(
            **values,
            record_fingerprint=_sha256_text(
                _compact_json(
                    CaptionerRecord.model_construct(
                        **values,
                        record_fingerprint="",
                    ).model_dump(mode="json", exclude={"record_fingerprint"})
                )
            ),
        )
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="failed",
                raw_responses=exc.raw_responses,
                diagnostics=exc.diagnostics,
                failure=exc,
            ),
        )


@dataclass(frozen=True)
class _RecaptionCaptionOutcome:
    job: JEATargetAudioCaptionJob
    backend_provenance: SpecializedBackendProvenance
    result: CaptionerBackendResult | None = None
    failure: SpecializedBackendFailure | None = None


@dataclass(frozen=True)
class _RecaptionGlobalOutcome:
    captioner: _RecaptionCaptionOutcome
    backend_provenance: SpecializedBackendProvenance
    result: SpecializedBackendResult[GlobalAudioSemanticsResponse] | None = None
    failure: SpecializedBackendFailure | None = None


@dataclass(frozen=True)
class _MusicRescueOutcome:
    job: JEATargetAudioCaptionJob
    backend_provenance: SpecializedBackendProvenance
    result: SpecializedBackendResult[DirectAudioMusicResponse] | None = None
    failure: SpecializedBackendFailure | None = None


def _recaption_caption(
    job: JEATargetAudioCaptionJob,
    backend: CaptionerBackend,
) -> _RecaptionCaptionOutcome:
    try:
        return _RecaptionCaptionOutcome(
            job=job,
            backend_provenance=backend.provenance,
            result=backend.caption(job),
        )
    except SpecializedBackendFailure as exc:
        return _RecaptionCaptionOutcome(
            job=job,
            backend_provenance=backend.provenance,
            failure=exc,
        )


def _recaption_global(
    captioner: _RecaptionCaptionOutcome,
    backend: GlobalSemanticsBackend,
) -> _RecaptionGlobalOutcome:
    if captioner.result is None:
        return _RecaptionGlobalOutcome(
            captioner=captioner,
            backend_provenance=backend.provenance,
        )
    try:
        return _RecaptionGlobalOutcome(
            captioner=captioner,
            backend_provenance=backend.provenance,
            result=backend.extract(
                captioner.job,
                captioner.result.raw_audio_caption,
            ),
        )
    except SpecializedBackendFailure as exc:
        return _RecaptionGlobalOutcome(
            captioner=captioner,
            backend_provenance=backend.provenance,
            failure=exc,
        )


def _detect_music(
    job: JEATargetAudioCaptionJob,
    backend: MusicRescueBackend,
) -> _MusicRescueOutcome:
    try:
        return _MusicRescueOutcome(
            job=job,
            backend_provenance=backend.provenance,
            result=backend.detect(job),
        )
    except SpecializedBackendFailure as exc:
        return _MusicRescueOutcome(
            job=job,
            backend_provenance=backend.provenance,
            failure=exc,
        )


def _needs_recaption_rescue(record: GlobalSemanticsRecord) -> bool:
    return record.status == "failed" or (
        record.status == "ready"
        and record.overall_audio_description is None
    )


def _outcome_raw(
    outcome: _RecaptionGlobalOutcome,
    *,
    used: bool,
) -> dict[str, object]:
    caption_result = outcome.captioner.result
    caption_failure = outcome.captioner.failure
    global_result = outcome.result
    global_failure = outcome.failure
    if caption_result is not None:
        captioner_raw = _raw_payload(
            status="ready",
            raw_responses=caption_result.raw_responses,
            diagnostics=caption_result.diagnostics,
        )
        captioner_raw["raw_audio_caption"] = caption_result.raw_audio_caption
    else:
        assert caption_failure is not None
        captioner_raw = _raw_payload(
            status="failed",
            raw_responses=caption_failure.raw_responses,
            diagnostics=caption_failure.diagnostics,
            failure=caption_failure,
        )
    if global_result is not None:
        global_raw: dict[str, object] | None = _raw_payload(
            status="ready",
            raw_responses=global_result.raw_responses,
            diagnostics=global_result.diagnostics,
            semantic_source=global_result.semantic_source,
        )
    elif global_failure is not None:
        global_raw = _raw_payload(
            status="failed",
            raw_responses=global_failure.raw_responses,
            diagnostics=global_failure.diagnostics,
            failure=global_failure,
        )
    else:
        global_raw = None
    return {
        "attempted": True,
        "used": used,
        "policy_version": GLOBAL_RECAPTION_RESCUE_POLICY_VERSION,
        "captioner": {
            **captioner_raw,
            "backend_provenance": outcome.captioner.backend_provenance.model_dump(
                mode="json"
            ),
        },
        "global": (
            None
            if global_raw is None
            else {
                **global_raw,
                "backend_provenance": outcome.backend_provenance.model_dump(
                    mode="json"
                ),
            }
        ),
    }


def _merge_recaption_outcome(
    primary: _ProcessedRecord[GlobalSemanticsRecord],
    outcome: _RecaptionGlobalOutcome,
) -> _ProcessedRecord[GlobalSemanticsRecord]:
    primary_record = primary.record
    caption_result = outcome.captioner.result
    caption_failure = outcome.captioner.failure
    rescue_result = outcome.result
    rescue_failure = outcome.failure
    primary_ready = primary_record.status == "ready"
    rescue_ready = rescue_result is not None

    used = False
    if primary_ready:
        response = rescue_result.response if rescue_result is not None else None
        description = primary_record.overall_audio_description
        soundscape = primary_record.overall_soundscape
        music = primary_record.non_diegetic_music
        if description is None and response is not None:
            description = response.overall_audio_description
            used = used or description is not None
        if soundscape is None and response is not None:
            soundscape = response.overall_soundscape
            used = used or soundscape is not None
        if music is None and response is not None:
            music = response.non_diegetic_music
            used = used or music is not None
        status: StageStatus = "ready"
        semantic_source = primary_record.semantic_source
        fallback_attempted = primary_record.fallback_attempted or (
            used and rescue_result is not None and rescue_result.fallback_attempted
        )
        failure = None
    elif rescue_ready:
        assert rescue_result is not None
        description = rescue_result.response.overall_audio_description
        soundscape = rescue_result.response.overall_soundscape
        music = rescue_result.response.non_diegetic_music
        status = "ready"
        semantic_source = rescue_result.semantic_source
        fallback_attempted = rescue_result.fallback_attempted
        failure = None
        used = True
    else:
        description = None
        soundscape = None
        music = None
        status = "failed"
        semantic_source = None
        fallback_attempted = primary_record.fallback_attempted
        failure = primary_record.failure

    caption_diagnostics = (
        caption_result.diagnostics
        if caption_result is not None
        else (() if caption_failure is None else caption_failure.diagnostics)
    )
    rescue_diagnostics = (
        rescue_result.diagnostics
        if rescue_result is not None
        else (() if rescue_failure is None else rescue_failure.diagnostics)
    )
    diagnostics = [
        *primary_record.completion_diagnostics,
        *caption_diagnostics,
        *rescue_diagnostics,
    ]
    caption_call_count = (
        caption_result.model_call_count
        if caption_result is not None
        else (0 if caption_failure is None else caption_failure.model_call_count)
    )
    rescue_call_count = (
        rescue_result.model_call_count
        if rescue_result is not None
        else (0 if rescue_failure is None else rescue_failure.model_call_count)
    )
    values = primary_record.model_dump(
        mode="python",
        exclude={"schema_version", "record_fingerprint"},
    )
    values["backend_provenance"] = primary_record.backend_provenance
    values.update(
        {
            "status": status,
            "overall_audio_description": description,
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
            "semantic_source": semantic_source,
            "fallback_attempted": fallback_attempted,
            "recaption_rescue_attempted": True,
            "recaption_rescue_used": used,
            "model_call_count": (
                primary_record.model_call_count
                + caption_call_count
                + rescue_call_count
            ),
            "raw_response_count": len(diagnostics),
            "completion_diagnostics": diagnostics,
            "failure": failure,
        }
    )
    record = _make_record(GlobalSemanticsRecord, values)

    primary_raw_responses = list(primary.raw.get("raw_responses", []))
    caption_raw_responses = (
        caption_result.raw_responses
        if caption_result is not None
        else (() if caption_failure is None else caption_failure.raw_responses)
    )
    rescue_raw_responses = (
        rescue_result.raw_responses
        if rescue_result is not None
        else (() if rescue_failure is None else rescue_failure.raw_responses)
    )
    raw = {
        "status": status,
        "semantic_source": semantic_source,
        "raw_responses": [
            *primary_raw_responses,
            *caption_raw_responses,
            *rescue_raw_responses,
        ],
        "completion_diagnostics": [
            item.model_dump(mode="json") for item in diagnostics
        ],
        "failure": None if failure is None else failure.model_dump(mode="json"),
        "primary_global": primary.raw.get("primary_global", primary.raw),
        "recaption_rescue": _outcome_raw(outcome, used=used),
    }
    return _ProcessedRecord(record=record, raw=raw)


def _music_outcome_raw(
    outcome: _MusicRescueOutcome,
    *,
    used: bool,
) -> dict[str, object]:
    result = outcome.result
    failure = outcome.failure
    if result is not None:
        payload = _raw_payload(
            status="ready",
            raw_responses=result.raw_responses,
            diagnostics=result.diagnostics,
        )
    else:
        assert failure is not None
        payload = _raw_payload(
            status="failed",
            raw_responses=failure.raw_responses,
            diagnostics=failure.diagnostics,
            failure=failure,
        )
    return {
        "attempted": True,
        "used": used,
        "prompt_version": MUSIC_RESCUE_PROMPT_VERSION,
        "policy_version": MUSIC_RESCUE_POLICY_VERSION,
        "backend_provenance": outcome.backend_provenance.model_dump(mode="json"),
        **payload,
    }


def _unattempted_rescue_raw(
    processed: _ProcessedRecord[GlobalSemanticsRecord],
) -> _ProcessedRecord[GlobalSemanticsRecord]:
    raw = dict(processed.raw)
    raw.setdefault("primary_global", processed.raw)
    raw.setdefault(
        "recaption_rescue",
        {
            "attempted": False,
            "used": False,
            "policy_version": GLOBAL_RECAPTION_RESCUE_POLICY_VERSION,
        },
    )
    raw["music_rescue"] = {
        "attempted": False,
        "used": False,
        "prompt_version": MUSIC_RESCUE_PROMPT_VERSION,
        "policy_version": MUSIC_RESCUE_POLICY_VERSION,
    }
    return _ProcessedRecord(record=processed.record, raw=raw)


def _merge_music_outcome(
    current: _ProcessedRecord[GlobalSemanticsRecord],
    outcome: _MusicRescueOutcome,
) -> _ProcessedRecord[GlobalSemanticsRecord]:
    record = current.record
    if record.status != "ready" or record.non_diegetic_music is not None:
        raise ValueError("music rescue requires ready global semantics without music")
    result = outcome.result
    failure = outcome.failure
    music = None if result is None else result.response.non_diegetic_music
    used = music is not None
    diagnostics = [
        *record.completion_diagnostics,
        *(
            result.diagnostics
            if result is not None
            else (() if failure is None else failure.diagnostics)
        ),
    ]
    model_call_count = (
        result.model_call_count
        if result is not None
        else (0 if failure is None else failure.model_call_count)
    )
    values = record.model_dump(
        mode="python",
        exclude={"schema_version", "record_fingerprint"},
    )
    values["backend_provenance"] = record.backend_provenance
    values.update(
        {
            "non_diegetic_music": music,
            "music_rescue_attempted": True,
            "music_rescue_used": used,
            "model_call_count": record.model_call_count + model_call_count,
            "raw_response_count": len(diagnostics),
            "completion_diagnostics": diagnostics,
        }
    )
    merged_record = _make_record(GlobalSemanticsRecord, values)
    result_raw_responses = (
        result.raw_responses
        if result is not None
        else (() if failure is None else failure.raw_responses)
    )
    raw = dict(current.raw)
    raw.setdefault("primary_global", current.raw)
    raw.setdefault(
        "recaption_rescue",
        {
            "attempted": False,
            "used": False,
            "policy_version": GLOBAL_RECAPTION_RESCUE_POLICY_VERSION,
        },
    )
    raw.update(
        {
            "status": "ready",
            "raw_responses": [
                *current.raw.get("raw_responses", []),
                *result_raw_responses,
            ],
            "completion_diagnostics": [
                item.model_dump(mode="json") for item in diagnostics
            ],
            "music_rescue": _music_outcome_raw(outcome, used=used),
        }
    )
    return _ProcessedRecord(record=merged_record, raw=raw)


def _global_record(
    job: JEATargetAudioCaptionJob,
    captioner: CaptionerRecord,
    backend: GlobalSemanticsBackend,
    runtime_provenance: SpecializedBackendProvenance | None = None,
) -> _ProcessedRecord[GlobalSemanticsRecord]:
    provenance = runtime_provenance or backend.provenance
    if captioner.status != "ready":
        failure = SpecializedStageFailure(
            code="captioner_not_ready",
            reason="global semantics blocked because captioner failed",
            attempt_count=0,
        )
        values: dict[str, object] = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "blocked",
            "recaption_rescue_attempted": False,
            "recaption_rescue_used": False,
            "music_rescue_attempted": False,
            "music_rescue_used": False,
            "captioner_record_fingerprint": captioner.record_fingerprint,
            "raw_audio_caption_sha256": None,
            "backend_provenance": provenance,
            "request_fingerprint": None,
            "model_call_count": 0,
            "raw_response_count": 0,
            "completion_diagnostics": [],
            "failure": failure,
        }
        record = _make_record(GlobalSemanticsRecord, values)
        return _ProcessedRecord(
            record=record,
            raw={"status": "blocked", "raw_responses": [], "failure": failure.model_dump(mode="json")},
        )
    assert captioner.raw_audio_caption is not None
    raw_caption_hash = _sha256_text(captioner.raw_audio_caption)
    request_fingerprint = global_request_fingerprint(
        captioner.raw_audio_caption,
        provenance,
        target_full_audio_sha256=job.target_full_audio_sha256,
    )
    try:
        result = backend.extract(job, captioner.raw_audio_caption)
        values = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "ready",
            "overall_audio_description": result.response.overall_audio_description,
            "overall_soundscape": result.response.overall_soundscape,
            "non_diegetic_music": result.response.non_diegetic_music,
            "semantic_source": result.semantic_source,
            "fallback_attempted": result.fallback_attempted,
            "recaption_rescue_attempted": False,
            "recaption_rescue_used": False,
            "music_rescue_attempted": False,
            "music_rescue_used": False,
            "captioner_record_fingerprint": captioner.record_fingerprint,
            "raw_audio_caption_sha256": raw_caption_hash,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": result.model_call_count,
            "raw_response_count": len(result.raw_responses),
            "completion_diagnostics": list(result.diagnostics),
            "failure": None,
        }
        record = _make_record(GlobalSemanticsRecord, values)
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="ready",
                raw_responses=result.raw_responses,
                diagnostics=result.diagnostics,
                semantic_source=result.semantic_source,
            ),
        )
    except SpecializedBackendFailure as exc:
        values = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "failed",
            "recaption_rescue_attempted": False,
            "recaption_rescue_used": False,
            "music_rescue_attempted": False,
            "music_rescue_used": False,
            "captioner_record_fingerprint": captioner.record_fingerprint,
            "raw_audio_caption_sha256": raw_caption_hash,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": exc.model_call_count,
            "raw_response_count": len(exc.raw_responses),
            "completion_diagnostics": list(exc.diagnostics),
            "failure": _published_failure(exc),
        }
        record = _make_record(GlobalSemanticsRecord, values)
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="failed",
                raw_responses=exc.raw_responses,
                diagnostics=exc.diagnostics,
                failure=exc,
            ),
        )


def _local_record(
    job: JEATargetAudioCaptionJob,
    captioner: CaptionerRecord,
    backend: LocalSemanticsBackend,
) -> _ProcessedRecord[LocalSemanticsRecord]:
    provenance = backend.provenance
    raw_audio_caption = captioner.raw_audio_caption
    raw_caption_hash = (
        None if raw_audio_caption is None else _sha256_text(raw_audio_caption)
    )
    request_fingerprint = local_request_fingerprint(
        job,
        provenance,
        captioner,
    )
    try:
        result = backend.describe(job, raw_audio_caption)
        issues = _local_response_issues(result.response, job)
        if issues:
            raise RuntimeError("local semantics backend returned unvalidated output")
        response = _normalize_local_response(result.response, job)
        values: dict[str, object] = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "ready",
            "temporal_audio_events": response.temporal_audio_events,
            "speaker_delivery": response.speaker_delivery,
            "semantic_source": result.semantic_source,
            "fallback_attempted": result.fallback_attempted,
            "captioner_record_fingerprint": captioner.record_fingerprint,
            "raw_audio_caption_sha256": raw_caption_hash,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": result.model_call_count,
            "raw_response_count": len(result.raw_responses),
            "completion_diagnostics": list(result.diagnostics),
            "failure": None,
        }
        record = _make_record(LocalSemanticsRecord, values)
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="ready",
                raw_responses=result.raw_responses,
                diagnostics=result.diagnostics,
                semantic_source=result.semantic_source,
            ),
        )
    except SpecializedBackendFailure as exc:
        speaker_delivery = _salvage_speaker_delivery(exc.raw_responses, job)
        values = {
            "target_clip_uid": job.target_clip_uid,
            "clip_display_path": job.clip_display_path,
            "status": "failed",
            "temporal_audio_events": [],
            "speaker_delivery": speaker_delivery,
            "captioner_record_fingerprint": captioner.record_fingerprint,
            "raw_audio_caption_sha256": raw_caption_hash,
            "backend_provenance": provenance,
            "request_fingerprint": request_fingerprint,
            "model_call_count": exc.model_call_count,
            "raw_response_count": len(exc.raw_responses),
            "completion_diagnostics": list(exc.diagnostics),
            "failure": _published_failure(exc),
        }
        record = _make_record(LocalSemanticsRecord, values)
        return _ProcessedRecord(
            record=record,
            raw=_raw_payload(
                status="failed",
                raw_responses=exc.raw_responses,
                diagnostics=exc.diagnostics,
                failure=exc,
            ),
        )


RecordT = TypeVar("RecordT", bound=SchemaModel)


def _make_record(model: type[RecordT], values: dict[str, object]) -> RecordT:
    provisional = model.model_construct(**values, record_fingerprint="")
    fingerprint = _sha256_text(
        _compact_json(provisional.model_dump(mode="json", exclude={"record_fingerprint"}))
    )
    return model(**values, record_fingerprint=fingerprint)


def _summary(
    *,
    role: StageRole,
    inventory: JEATargetAudioCaptionInventory,
    provenance: SpecializedBackendProvenance,
    records: Sequence[CaptionerRecord | GlobalSemanticsRecord | LocalSemanticsRecord],
) -> SpecializedStageSummary:
    failure_counts = Counter(
        record.failure.code for record in records if record.failure is not None
    )
    return SpecializedStageSummary(
        stage=role,
        inventory_fingerprint=inventory.inventory_fingerprint,
        backend_provenance=provenance,
        record_count=len(records),
        ready_count=sum(record.status == "ready" for record in records),
        failed_count=sum(record.status == "failed" for record in records),
        blocked_count=sum(record.status == "blocked" for record in records),
        model_call_count=sum(record.model_call_count for record in records),
        raw_response_count=sum(record.raw_response_count for record in records),
        failure_counts=dict(sorted(failure_counts.items())),
    )


def _publish_stage(
    *,
    destination: Path,
    records: Sequence[SchemaModel],
    summary: SpecializedStageSummary,
    raw_by_clip: dict[str, dict[str, object]],
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir(parents=True)
        (temporary / "raw").mkdir()
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        for clip_uid, payload in sorted(raw_by_clip.items()):
            _write_json(temporary / "raw" / f"{clip_uid}.json", payload)
        _replace_stage_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_stage(
    *,
    destination: Path,
    role: StageRole,
    inventory: JEATargetAudioCaptionInventory,
    provenance: SpecializedBackendProvenance,
    model: type[RecordT],
) -> tuple[list[RecordT], SpecializedStageSummary]:
    try:
        summary = SpecializedStageSummary.model_validate_json(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        records = _read_jsonl(destination / "records.jsonl", model)
    except Exception as exc:
        raise ValueError(f"invalid existing specialized {role} output") from exc
    if (
        summary.stage != role
        or summary.inventory_fingerprint != inventory.inventory_fingerprint
        or summary.backend_provenance != provenance
        or summary.record_count != len(records)
        or [record.target_clip_uid for record in records]
        != [job.target_clip_uid for job in inventory.jobs]
    ):
        raise ValueError(f"existing specialized {role} output is incompatible")
    return records, summary


def _validate_global_dependencies(
    global_records: Sequence[GlobalSemanticsRecord],
    captioner_records: Sequence[CaptionerRecord],
) -> None:
    if len(global_records) != len(captioner_records):
        raise _StaleSpecializedDependencyError(
            "existing global output depends on stale captioner records; use --overwrite"
        )
    for global_record, captioner_record in zip(
        global_records,
        captioner_records,
        strict=True,
    ):
        caption = captioner_record.raw_audio_caption
        if (
            global_record.target_clip_uid != captioner_record.target_clip_uid
            or global_record.captioner_record_fingerprint
            != captioner_record.record_fingerprint
            or (
                global_record.status == "ready"
                and (
                    caption is None
                    or global_record.raw_audio_caption_sha256
                    != _sha256_text(caption)
                )
            )
        ):
            raise _StaleSpecializedDependencyError(
                "existing global output depends on stale captioner records; "
                "use --overwrite"
            )


def _validate_local_dependencies(
    local_records: Sequence[LocalSemanticsRecord],
    captioner_records: Sequence[CaptionerRecord],
) -> None:
    if len(local_records) != len(captioner_records):
        raise _StaleSpecializedDependencyError(
            "existing local output depends on stale captioner records; use --overwrite"
        )
    for local_record, captioner_record in zip(
        local_records,
        captioner_records,
        strict=True,
    ):
        caption = captioner_record.raw_audio_caption
        expected_caption_hash = (
            None if caption is None else _sha256_text(caption)
        )
        if (
            local_record.target_clip_uid != captioner_record.target_clip_uid
            or local_record.captioner_record_fingerprint
            != captioner_record.record_fingerprint
            or local_record.raw_audio_caption_sha256 != expected_caption_hash
        ):
            raise _StaleSpecializedDependencyError(
                "existing local output depends on stale captioner records; "
                "use --overwrite"
            )


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


def _bounded_map(
    items: Sequence[ItemT],
    process: Callable[[ItemT], ResultT],
    *,
    max_inflight: int,
) -> list[ResultT]:
    if max_inflight < 1:
        raise ValueError("specialized max inflight must be positive")
    if max_inflight == 1:
        return [process(item) for item in items]
    results: dict[int, ResultT] = {}
    pending: dict[Future[ResultT], int] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=max_inflight) as executor:
        while next_index < len(items) or pending:
            while next_index < len(items) and len(pending) < max_inflight:
                pending[executor.submit(process, items[next_index])] = next_index
                next_index += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
    return [results[index] for index in range(len(items))]


def _bounded_process(
    jobs: Sequence[JEATargetAudioCaptionJob],
    process: Callable[[JEATargetAudioCaptionJob], _ProcessedRecord[RecordT]],
    *,
    max_inflight: int,
) -> list[_ProcessedRecord[RecordT]]:
    return _bounded_map(jobs, process, max_inflight=max_inflight)


def _run_recaption_rescues(
    *,
    jobs: Sequence[JEATargetAudioCaptionJob],
    processed: Sequence[_ProcessedRecord[GlobalSemanticsRecord]],
    captioner_backend: CaptionerBackend | None,
    global_backend: GlobalSemanticsBackend,
    captioner_max_inflight: int,
    global_max_inflight: int,
) -> list[_ProcessedRecord[GlobalSemanticsRecord]]:
    jobs_by_clip = {job.target_clip_uid: job for job in jobs}
    candidates = [item for item in processed if _needs_recaption_rescue(item.record)]
    if not candidates:
        return list(processed)
    if captioner_backend is None:
        raise ValueError("recaption rescue requires a captioner backend")
    recaptions = _bounded_map(
        [jobs_by_clip[item.record.target_clip_uid] for item in candidates],
        lambda job: _recaption_caption(job, captioner_backend),
        max_inflight=captioner_max_inflight,
    )
    rescues = _bounded_map(
        recaptions,
        lambda captioner: _recaption_global(captioner, global_backend),
        max_inflight=global_max_inflight,
    )
    primary_by_clip = {item.record.target_clip_uid: item for item in candidates}
    merged_by_clip = {
        outcome.captioner.job.target_clip_uid: _merge_recaption_outcome(
            primary_by_clip[outcome.captioner.job.target_clip_uid],
            outcome,
        )
        for outcome in rescues
    }
    return [
        merged_by_clip.get(item.record.target_clip_uid, item) for item in processed
    ]


def _run_music_rescues(
    *,
    jobs: Sequence[JEATargetAudioCaptionJob],
    processed: Sequence[_ProcessedRecord[GlobalSemanticsRecord]],
    music_backend: MusicRescueBackend,
    max_inflight: int,
) -> list[_ProcessedRecord[GlobalSemanticsRecord]]:
    jobs_by_clip = {job.target_clip_uid: job for job in jobs}
    candidates = [
        item
        for item in processed
        if item.record.status == "ready"
        and item.record.non_diegetic_music is None
    ]
    outcomes = _bounded_map(
        [jobs_by_clip[item.record.target_clip_uid] for item in candidates],
        lambda job: _detect_music(job, music_backend),
        max_inflight=max_inflight,
    )
    current_by_clip = {item.record.target_clip_uid: item for item in candidates}
    merged_by_clip = {
        outcome.job.target_clip_uid: _merge_music_outcome(
            current_by_clip[outcome.job.target_clip_uid],
            outcome,
        )
        for outcome in outcomes
    }
    return [
        merged_by_clip[item.record.target_clip_uid]
        if item.record.target_clip_uid in merged_by_clip
        else _unattempted_rescue_raw(item)
        for item in processed
    ]


def run_captioner_phase(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    backend: CaptionerBackend,
    overwrite: bool = False,
    max_inflight: int = 1,
) -> tuple[list[CaptionerRecord], SpecializedStageSummary]:
    root = _ensure_root(inventory, output_root)
    destination = _stage_path(root, "captioner")
    provenance = backend.provenance
    if destination.exists() and not overwrite:
        return _load_stage(
            destination=destination,
            role="captioner",
            inventory=inventory,
            provenance=provenance,
            model=CaptionerRecord,
        )
    if destination.exists():
        _assert_overwrite_ownership(destination, expected_role="captioner")
    processed = _bounded_process(
        inventory.jobs,
        lambda job: _captioner_record(job, backend),
        max_inflight=max_inflight,
    )
    records = [item.record for item in processed]
    summary = _summary(
        role="captioner",
        inventory=inventory,
        provenance=provenance,
        records=records,
    )
    _publish_stage(
        destination=destination,
        records=records,
        summary=summary,
        raw_by_clip={
            record.target_clip_uid: item.raw
            for record, item in zip(records, processed, strict=True)
        },
    )
    return records, summary


def run_global_semantics_phase(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    backend: GlobalSemanticsBackend,
    captioner_backend: CaptionerBackend,
    music_backend: MusicRescueBackend,
    overwrite: bool = False,
    max_inflight: int = 4,
    captioner_max_inflight: int = 1,
    music_max_inflight: int = 1,
) -> tuple[list[GlobalSemanticsRecord], SpecializedStageSummary]:
    root = _ensure_root(inventory, output_root)
    destination = _stage_path(root, "global_semantics")
    provenance = _global_runtime_provenance(
        backend.provenance,
        captioner_backend.provenance,
        music_backend.provenance,
    )
    captioner_records, captioner_summary = _load_stage(
        destination=_stage_path(root, "captioner"),
        role="captioner",
        inventory=inventory,
        provenance=_existing_stage_provenance(_stage_path(root, "captioner")),
        model=CaptionerRecord,
    )
    if captioner_backend.provenance != captioner_summary.backend_provenance:
        raise ValueError("recaption backend differs from primary captioner stage")
    if destination.exists() and not overwrite:
        records, summary = _load_stage(
            destination=destination,
            role="global_semantics",
            inventory=inventory,
            provenance=provenance,
            model=GlobalSemanticsRecord,
        )
        _validate_global_dependencies(records, captioner_records)
        return records, summary
    if destination.exists():
        _assert_overwrite_ownership(destination, expected_role="global_semantics")
    by_clip = {record.target_clip_uid: record for record in captioner_records}
    processed = _bounded_process(
        inventory.jobs,
        lambda job: _global_record(
            job,
            by_clip[job.target_clip_uid],
            backend,
            provenance,
        ),
        max_inflight=max_inflight,
    )
    processed = _run_recaption_rescues(
        jobs=inventory.jobs,
        processed=processed,
        captioner_backend=captioner_backend,
        global_backend=backend,
        captioner_max_inflight=captioner_max_inflight,
        global_max_inflight=max_inflight,
    )
    processed = _run_music_rescues(
        jobs=inventory.jobs,
        processed=processed,
        music_backend=music_backend,
        max_inflight=music_max_inflight,
    )
    records = [item.record for item in processed]
    summary = _summary(
        role="global_semantics",
        inventory=inventory,
        provenance=provenance,
        records=records,
    )
    _publish_stage(
        destination=destination,
        records=records,
        summary=summary,
        raw_by_clip={
            record.target_clip_uid: item.raw
            for record, item in zip(records, processed, strict=True)
        },
    )
    return records, summary


def run_local_semantics_phase(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    backend: LocalSemanticsBackend,
    overwrite: bool = False,
    max_inflight: int = 1,
) -> tuple[list[LocalSemanticsRecord], SpecializedStageSummary]:
    root = _ensure_root(inventory, output_root)
    destination = _stage_path(root, "local_semantics")
    provenance = backend.provenance
    captioner_destination = _stage_path(root, "captioner")
    captioner_records, _ = _load_stage(
        destination=captioner_destination,
        role="captioner",
        inventory=inventory,
        provenance=_existing_stage_provenance(captioner_destination),
        model=CaptionerRecord,
    )
    if destination.exists() and not overwrite:
        records, summary = _load_stage(
            destination=destination,
            role="local_semantics",
            inventory=inventory,
            provenance=provenance,
            model=LocalSemanticsRecord,
        )
        _validate_local_dependencies(records, captioner_records)
        return records, summary
    if destination.exists():
        _assert_overwrite_ownership(destination, expected_role="local_semantics")
    processed = _bounded_map(
        list(zip(inventory.jobs, captioner_records, strict=True)),
        lambda item: _local_record(item[0], item[1], backend),
        max_inflight=max_inflight,
    )
    records = [item.record for item in processed]
    summary = _summary(
        role="local_semantics",
        inventory=inventory,
        provenance=provenance,
        records=records,
    )
    _publish_stage(
        destination=destination,
        records=records,
        summary=summary,
        raw_by_clip={
            record.target_clip_uid: item.raw
            for record, item in zip(records, processed, strict=True)
        },
    )
    return records, summary


def _existing_stage_provenance(destination: Path) -> SpecializedBackendProvenance:
    try:
        summary = SpecializedStageSummary.model_validate_json(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"cannot establish specialized stage ownership: {destination}") from exc
    return summary.backend_provenance


def _assembled_record(
    job: JEATargetAudioCaptionJob,
    captioner: CaptionerRecord,
    global_record: GlobalSemanticsRecord,
    local: LocalSemanticsRecord,
) -> SpecializedAudioSemanticsRecord:
    if global_record.captioner_record_fingerprint != captioner.record_fingerprint:
        raise ValueError("global semantics does not match captioner record")
    if (
        global_record.status == "ready"
        and captioner.raw_audio_caption is not None
        and global_record.raw_audio_caption_sha256
        != _sha256_text(captioner.raw_audio_caption)
    ):
        raise ValueError("global semantics caption hash differs")
    if local.status == "ready" or local.speaker_delivery:
        issues = _local_response_issues(
            LocalAudioSemanticsResponse(
                temporal_audio_events=local.temporal_audio_events,
                speaker_delivery=local.speaker_delivery,
            ),
            job,
        )
        if issues:
            raise ValueError("local semantics record no longer matches inventory")
    local_response = LocalAudioSemanticsResponse(
        temporal_audio_events=local.temporal_audio_events,
        speaker_delivery=local.speaker_delivery,
    )
    normalized_local = (
        _normalize_local_response(local_response, job)
        if local.status == "ready" or local.speaker_delivery
        else local_response
    )
    entity_by_cluster = {
        cluster.speaker_cluster_id: cluster.entity_id for cluster in job.speaker_clusters
    }
    target_delivery = [
        TargetSpeakerDelivery(
            speaker_cluster_id=item.speaker_cluster_id,
            delivery_style=item.delivery_style,
            entity_id=entity_by_cluster[item.speaker_cluster_id],
        )
        for item in normalized_local.speaker_delivery
    ]
    statuses = (captioner.status, global_record.status, local.status)
    ready_count = sum(status == "ready" for status in statuses)
    status = "complete" if ready_count == 3 else "partial" if ready_count else "failed"
    values: dict[str, object] = {
        "target_clip_uid": job.target_clip_uid,
        "clip_display_path": job.clip_display_path,
        "status": status,
        "raw_audio_caption": captioner.raw_audio_caption,
        "overall_audio_description": global_record.overall_audio_description,
        "overall_soundscape": global_record.overall_soundscape,
        "non_diegetic_music": global_record.non_diegetic_music,
        "temporal_audio_events": normalized_local.temporal_audio_events,
        "speaker_delivery": target_delivery,
        "target_video_path": job.target_video_path,
        "target_video_sha256": job.target_video_sha256,
        "target_full_audio_path": job.target_full_audio_path,
        "target_full_audio_sha256": job.target_full_audio_sha256,
        "target_duration_seconds": job.target_duration_seconds,
        "target_audio_binding_path": job.target_audio_binding_path,
        "target_audio_binding_sha256": job.target_audio_binding_sha256,
        "captioner_status": captioner.status,
        "global_semantics_status": global_record.status,
        "local_semantics_status": local.status,
        "captioner_provenance": captioner.backend_provenance,
        "global_semantics_provenance": global_record.backend_provenance,
        "local_semantics_provenance": local.backend_provenance,
        "captioner_failure": captioner.failure,
        "global_semantics_failure": global_record.failure,
        "local_semantics_failure": local.failure,
        "captioner_request_fingerprint": captioner.request_fingerprint,
        "global_semantics_request_fingerprint": global_record.request_fingerprint,
        "local_semantics_request_fingerprint": local.request_fingerprint,
        "captioner_record_fingerprint": captioner.record_fingerprint,
        "global_semantics_record_fingerprint": global_record.record_fingerprint,
        "local_semantics_record_fingerprint": local.record_fingerprint,
    }
    provisional = SpecializedAudioSemanticsRecord.model_construct(
        **values,
        assemble_fingerprint="",
    )
    fingerprint = _sha256_text(
        _compact_json(
            provisional.model_dump(mode="json", exclude={"assemble_fingerprint"})
        )
    )
    return SpecializedAudioSemanticsRecord(
        **values,
        assemble_fingerprint=fingerprint,
    )


QA_FLAGS = (
    "hallucinated_global_audio",
    "missed_global_audio",
    "vocal_leakage_global",
    "unsupported_captioner_inference",
    "wrong_music_classification",
    "hallucinated_audio_event",
    "missed_audio_event",
    "wrong_audio_event_timing",
    "wrong_speaker_style",
    "dialogue_leakage",
    "captioner_audio_hallucination",
    "captioner_missed_background_audio",
    "other",
)


def review_configuration_fingerprint(
    captioner: SpecializedBackendProvenance,
    global_semantics: SpecializedBackendProvenance,
    local_semantics: SpecializedBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "captioner": captioner.configuration_fingerprint,
                "global": global_semantics.configuration_fingerprint,
                "local": local_semantics.configuration_fingerprint,
            }
        )
    )


def _review_media_name(job: JEATargetAudioCaptionJob) -> str:
    suffix = Path(job.target_full_audio_path).suffix.lower() or ".audio"
    return _sha256_text(job.target_clip_uid) + suffix


def _review_html(
    *,
    inventory: JEATargetAudioCaptionInventory,
    records: Sequence[SpecializedAudioSemanticsRecord],
    media_names: dict[str, str],
    review_config_fingerprint: str,
) -> str:
    cards: list[str] = []
    for job, record in zip(inventory.jobs, records, strict=True):
        events = "".join(
            "<li>"
            + f"{event.start_time:.3f}-{event.end_time:.3f}s | "
            + html.escape(event.description)
            + "</li>"
            for event in record.temporal_audio_events
        )
        deliveries = "".join(
            "<li><code>"
            + html.escape(item.speaker_cluster_id)
            + "</code> entity="
            + html.escape(item.entity_id or "UNBOUND")
            + " | "
            + html.escape(item.delivery_style or "[null]")
            + "</li>"
            for item in record.speaker_delivery
        )
        events_html = (
            "<ul>" + events + "</ul>"
            if events
            else (
                "<p>[none]</p>"
                if record.local_semantics_status == "ready"
                else "<p>[unavailable]</p>"
            )
        )
        provenance = "".join(
            "<li>"
            + label
            + ": "
            + html.escape(stage.role)
            + " | "
            + html.escape(stage.served_model_name)
            + " | "
            + html.escape(stage.configuration_fingerprint)
            + "</li>"
            for label, stage in (
                ("Captioner", record.captioner_provenance),
                ("Global", record.global_semantics_provenance),
                ("Local", record.local_semantics_provenance),
            )
        )
        flags = " ".join(
            f"<label><input type='checkbox' data-flag='{flag}' onchange='saveQA()'>{flag}</label>"
            for flag in QA_FLAGS
        )
        cards.append(
            f"<article class='case' data-clip='{html.escape(job.target_clip_uid)}'>"
            f"<h2>{html.escape(job.clip_display_path)}</h2>"
            f"<audio controls preload='metadata' src='media/{html.escape(media_names[job.target_clip_uid])}'></audio>"
            f"<p><b>Overall:</b> {html.escape(record.status)}</p>"
            f"<p><b>Stages:</b> captioner={record.captioner_status}, "
            f"global={record.global_semantics_status}, local={record.local_semantics_status}</p>"
            "<details><summary>Captioner raw audio caption</summary><pre>"
            + html.escape(record.raw_audio_caption or "[unavailable]")
            + "</pre></details><h3>Overall audio description</h3><p>"
            + html.escape(record.overall_audio_description or "[null/unavailable]")
            + "</p><h3>Overall soundscape</h3><p>"
            + html.escape(record.overall_soundscape or "[null/unavailable]")
            + "</p><h3>Non-diegetic music</h3><p>"
            + html.escape(record.non_diegetic_music or "[null/unavailable]")
            + "</p><h3>Temporal audio events</h3>"
            + events_html
            + "<h3>Speaker delivery</h3>"
            + (
                "<ul>" + deliveries + "</ul>"
                if deliveries
                else (
                    "<p>[none]</p>"
                    if record.local_semantics_status == "ready"
                    else "<p>[unavailable]</p>"
                )
            )
            + "<details><summary>Stage provenance</summary><ul>"
            + provenance
            + "</ul></details><div class='qa'><b>QA:</b> "
            + flags
            + "<br><label>Note <input class='note' maxlength='500' oninput='saveQA()'></label></div></article>"
        )
    clip_order = [job.target_clip_uid for job in inventory.jobs]
    namespace = (
        f"h3-specialized-audio-qa:{ASSEMBLED_RECORD_VERSION}:"
        f"{inventory.inventory_fingerprint}:{review_config_fingerprint}:"
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>H3 Specialized Audio Semantics Review</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f6;color:#171717}}
header,.case{{max-width:1100px;margin:0 auto 22px;background:white;border:1px solid #bbb;padding:18px}}
audio{{width:100%}}pre{{white-space:pre-wrap}}.qa{{margin-top:14px;padding:12px;background:#eef2f5}}
.qa label{{margin-right:12px}}.note{{min-width:320px}}
</style></head><body><header><h1>H3 Specialized Audio Semantics V1</h1>
<p id='progress'>Reviewed 0 / {len(records)}</p></header>{''.join(cards)}<script>
const clipOrder={json.dumps(clip_order)};const flags={json.dumps(list(QA_FLAGS))};
const keyPrefix={json.dumps(namespace)};
function saveQA(){{document.querySelectorAll('.case').forEach(card=>{{const selected=Array.from(card.querySelectorAll('[data-flag]')).filter(x=>x.checked).map(x=>x.dataset.flag);const note=card.querySelector('.note').value;localStorage.setItem(keyPrefix+card.dataset.clip,JSON.stringify({{flags:selected,note}}));}});update();}}
function restore(){{document.querySelectorAll('.case').forEach(card=>{{let value=null;try{{value=JSON.parse(localStorage.getItem(keyPrefix+card.dataset.clip)||'null')}}catch(error){{value=null}}if(!value)return;card.querySelectorAll('[data-flag]').forEach(x=>x.checked=(value.flags||[]).includes(x.dataset.flag));card.querySelector('.note').value=value.note||'';}});update();}}
function update(){{let count=0;clipOrder.forEach(clip=>{{if(localStorage.getItem(keyPrefix+clip))count++;}});document.getElementById('progress').textContent=`Reviewed ${{count}} / ${{clipOrder.length}}`;}}
restore();</script></body></html>"""


def _stage_review_media(
    temporary: Path,
    inventory: JEATargetAudioCaptionInventory,
) -> dict[str, str]:
    media_directory = temporary / "media"
    media_directory.mkdir()
    media_names: dict[str, str] = {}
    for job in inventory.jobs:
        source = _verify_file(
            job.target_full_audio_path,
            job.target_full_audio_sha256,
            field_name="target full audio",
        )
        media_name = _review_media_name(job)
        destination = media_directory / media_name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if _sha256_file(destination) != job.target_full_audio_sha256:
            raise ValueError("assembled review media hash differs from canonical audio")
        media_names[job.target_clip_uid] = media_name
    return media_names


def _validate_review_media(
    destination: Path,
    inventory: JEATargetAudioCaptionInventory,
) -> None:
    try:
        for job in inventory.jobs:
            _verify_file(
                str(destination / "media" / _review_media_name(job)),
                job.target_full_audio_sha256,
                field_name="assembled review audio",
            )
    except (OSError, ValueError) as exc:
        raise _StaleSpecializedDependencyError(
            "existing assembled review media is stale; use --overwrite"
        ) from exc


def _publish_assembled(
    *,
    destination: Path,
    inventory: JEATargetAudioCaptionInventory,
    records: Sequence[SpecializedAudioSemanticsRecord],
    summary: SpecializedAssembledSummary,
    review_config_fingerprint: str,
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir(parents=True)
        media_names = _stage_review_media(temporary, inventory)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        (temporary / "review.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                media_names=media_names,
                review_config_fingerprint=review_config_fingerprint,
            ),
            encoding="utf-8",
        )
        _replace_stage_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_assembled(
    destination: Path,
    inventory: JEATargetAudioCaptionInventory,
    captioner_records: Sequence[CaptionerRecord],
    global_records: Sequence[GlobalSemanticsRecord],
    local_records: Sequence[LocalSemanticsRecord],
) -> tuple[list[SpecializedAudioSemanticsRecord], SpecializedAssembledSummary]:
    try:
        records = _read_jsonl(
            destination / "records.jsonl",
            SpecializedAudioSemanticsRecord,
        )
        summary = SpecializedAssembledSummary.model_validate_json(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError("invalid existing specialized assembled output") from exc
    if (
        summary.inventory_fingerprint != inventory.inventory_fingerprint
        or summary.record_count != len(records)
        or [record.target_clip_uid for record in records]
        != [job.target_clip_uid for job in inventory.jobs]
    ):
        raise ValueError("existing specialized assembled output is incompatible")
    for assembled, captioner, global_record, local in zip(
        records,
        captioner_records,
        global_records,
        local_records,
        strict=True,
    ):
        if (
            assembled.captioner_record_fingerprint != captioner.record_fingerprint
            or assembled.global_semantics_record_fingerprint
            != global_record.record_fingerprint
            or assembled.local_semantics_record_fingerprint != local.record_fingerprint
        ):
            raise _StaleSpecializedDependencyError(
                "existing assembled output depends on stale stage records; "
                "use --overwrite"
            )
    _validate_review_media(destination, inventory)
    return records, summary


def run_assemble_phase(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    overwrite: bool = False,
    regenerate_stale: bool = False,
) -> tuple[list[SpecializedAudioSemanticsRecord], SpecializedAssembledSummary]:
    root = _ensure_root(inventory, output_root)
    destination = _stage_path(root, "assembled")
    captioner_records, captioner_summary = _load_stage(
        destination=_stage_path(root, "captioner"),
        role="captioner",
        inventory=inventory,
        provenance=_existing_stage_provenance(_stage_path(root, "captioner")),
        model=CaptionerRecord,
    )
    global_records, global_summary = _load_stage(
        destination=_stage_path(root, "global_semantics"),
        role="global_semantics",
        inventory=inventory,
        provenance=_existing_stage_provenance(_stage_path(root, "global_semantics")),
        model=GlobalSemanticsRecord,
    )
    local_records, local_summary = _load_stage(
        destination=_stage_path(root, "local_semantics"),
        role="local_semantics",
        inventory=inventory,
        provenance=_existing_stage_provenance(_stage_path(root, "local_semantics")),
        model=LocalSemanticsRecord,
    )
    _validate_global_dependencies(global_records, captioner_records)
    _validate_local_dependencies(local_records, captioner_records)
    if destination.exists() and not overwrite:
        try:
            return _load_assembled(
                destination,
                inventory,
                captioner_records,
                global_records,
                local_records,
            )
        except _StaleSpecializedDependencyError:
            if not regenerate_stale:
                raise
    if destination.exists():
        _assert_overwrite_ownership(destination, expected_role="assembled")
    records = [
        _assembled_record(job, captioner, global_record, local)
        for job, captioner, global_record, local in zip(
            inventory.jobs,
            captioner_records,
            global_records,
            local_records,
            strict=True,
        )
    ]
    summary = SpecializedAssembledSummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        record_count=len(records),
        complete_count=sum(record.status == "complete" for record in records),
        partial_count=sum(record.status == "partial" for record in records),
        failed_count=sum(record.status == "failed" for record in records),
        captioner_ready_count=sum(
            record.captioner_status == "ready" for record in records
        ),
        global_ready_count=sum(
            record.global_semantics_status == "ready" for record in records
        ),
        global_blocked_count=sum(
            record.global_semantics_status == "blocked" for record in records
        ),
        local_ready_count=sum(
            record.local_semantics_status == "ready" for record in records
        ),
    )
    _publish_assembled(
        destination=destination,
        inventory=inventory,
        records=records,
        summary=summary,
        review_config_fingerprint=review_configuration_fingerprint(
            captioner_summary.backend_provenance,
            global_summary.backend_provenance,
            local_summary.backend_provenance,
        ),
    )
    return records, summary


class SpecializedPipelineResult(SchemaModel):
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_clip_count: int = Field(gt=0)
    captioner_reused: bool
    global_semantics_reused: bool
    local_semantics_reused: bool
    captioner_max_inflight: int = Field(gt=0)
    global_vl_max_inflight: int = Field(gt=0)
    local_instruct_max_inflight: int = Field(gt=0)
    peak_captioner_inflight: int = Field(ge=0)
    peak_global_vl_inflight: int = Field(ge=0)
    peak_local_instruct_inflight: int = Field(ge=0)
    peak_global_backlog: int = Field(ge=0)
    captioner_summary: SpecializedStageSummary
    global_semantics_summary: SpecializedStageSummary
    local_semantics_summary: SpecializedStageSummary
    assembled_summary: SpecializedAssembledSummary


def _existing_configuration_fingerprint(
    destination: Path,
    *,
    role: StageRole,
) -> str:
    _assert_overwrite_ownership(destination, expected_role=role)
    try:
        payload = json.loads(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        provenance = payload["backend_provenance"]
        fingerprint = provenance["configuration_fingerprint"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"cannot establish specialized stage configuration: {destination}"
        ) from exc
    if not isinstance(fingerprint, str):
        raise TypeError("specialized stage configuration fingerprint is not text")
    return fingerprint


def _pipeline_existing_stage(
    *,
    root: Path,
    role: StageRole,
    inventory: JEATargetAudioCaptionInventory,
    provenance: SpecializedBackendProvenance,
    model: type[RecordT],
    overwrite: bool,
) -> tuple[list[RecordT] | None, SpecializedStageSummary | None]:
    destination = _stage_path(root, role)
    if overwrite:
        if destination.exists():
            _assert_overwrite_ownership(destination, expected_role=role)
        return None, None
    if not destination.exists():
        return None, None
    if _existing_configuration_fingerprint(destination, role=role) != (
        provenance.configuration_fingerprint
    ):
        return None, None
    return _load_stage(
        destination=destination,
        role=role,
        inventory=inventory,
        provenance=provenance,
        model=model,
    )


def run_specialized_pipeline(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    captioner_backend: CaptionerBackend,
    global_backend: GlobalSemanticsBackend,
    local_backend: LocalSemanticsBackend,
    music_backend: MusicRescueBackend,
    overwrite: bool = False,
    captioner_max_inflight: int = 1,
    global_vl_max_inflight: int = 4,
    local_instruct_max_inflight: int = 1,
) -> SpecializedPipelineResult:
    limits = (
        captioner_max_inflight,
        global_vl_max_inflight,
        local_instruct_max_inflight,
    )
    if any(value < 1 for value in limits):
        raise ValueError("specialized pipeline inflight limits must be positive")
    root = _ensure_root(inventory, output_root)
    global_runtime_provenance = _global_runtime_provenance(
        global_backend.provenance,
        captioner_backend.provenance,
        music_backend.provenance,
    )
    assembled_destination = _stage_path(root, "assembled")
    if overwrite and assembled_destination.exists():
        _assert_overwrite_ownership(
            assembled_destination,
            expected_role="assembled",
        )
    caption_existing, caption_summary = _pipeline_existing_stage(
        root=root,
        role="captioner",
        inventory=inventory,
        provenance=captioner_backend.provenance,
        model=CaptionerRecord,
        overwrite=overwrite,
    )
    global_existing, global_summary = _pipeline_existing_stage(
        root=root,
        role="global_semantics",
        inventory=inventory,
        provenance=global_runtime_provenance,
        model=GlobalSemanticsRecord,
        overwrite=overwrite,
    )
    if global_existing is not None:
        if caption_existing is None:
            global_existing = None
            global_summary = None
        else:
            try:
                _validate_global_dependencies(global_existing, caption_existing)
            except _StaleSpecializedDependencyError:
                global_existing = None
                global_summary = None
    local_existing, local_summary = _pipeline_existing_stage(
        root=root,
        role="local_semantics",
        inventory=inventory,
        provenance=local_backend.provenance,
        model=LocalSemanticsRecord,
        overwrite=overwrite,
    )
    if local_existing is not None:
        if caption_existing is None:
            local_existing = None
            local_summary = None
        else:
            try:
                _validate_local_dependencies(local_existing, caption_existing)
            except _StaleSpecializedDependencyError:
                local_existing = None
                local_summary = None
    caption_reused = caption_existing is not None
    global_reused = global_existing is not None
    local_reused = local_existing is not None
    caption_by_clip = {
        record.target_clip_uid: _ProcessedRecord(record=record, raw={})
        for record in (caption_existing or [])
    }
    global_by_clip = {
        record.target_clip_uid: _ProcessedRecord(record=record, raw={})
        for record in (global_existing or [])
    }
    local_by_clip = {
        record.target_clip_uid: _ProcessedRecord(record=record, raw={})
        for record in (local_existing or [])
    }
    jobs_by_clip = {job.target_clip_uid: job for job in inventory.jobs}
    caption_pending = deque(
        job for job in inventory.jobs if job.target_clip_uid not in caption_by_clip
    )
    local_pending = deque(
        jobs_by_clip[clip_uid]
        for clip_uid in caption_by_clip
        if clip_uid not in local_by_clip
    )
    global_pending = deque(
        jobs_by_clip[clip_uid]
        for clip_uid in caption_by_clip
        if clip_uid not in global_by_clip
    )
    global_capacity = max(1, global_vl_max_inflight * 2)
    caption_futures: dict[Future[_ProcessedRecord[CaptionerRecord]], str] = {}
    global_futures: dict[Future[_ProcessedRecord[GlobalSemanticsRecord]], str] = {}
    local_futures: dict[Future[_ProcessedRecord[LocalSemanticsRecord]], str] = {}
    peaks = {"captioner": 0, "global": 0, "local": 0, "backlog": len(global_pending)}

    with (
        ThreadPoolExecutor(
            max_workers=captioner_max_inflight,
            thread_name_prefix="h3-captioner",
        ) as caption_executor,
        ThreadPoolExecutor(
            max_workers=global_vl_max_inflight,
            thread_name_prefix="h3-global-vl",
        ) as global_executor,
        ThreadPoolExecutor(
            max_workers=local_instruct_max_inflight,
            thread_name_prefix="h3-local-instruct",
        ) as local_executor,
    ):
        while (
            caption_pending
            or local_pending
            or global_pending
            or caption_futures
            or global_futures
            or local_futures
        ):
            while local_pending and len(local_futures) < local_instruct_max_inflight:
                job = local_pending.popleft()
                captioner = caption_by_clip[job.target_clip_uid].record
                local_futures[
                    local_executor.submit(
                        _local_record,
                        job,
                        captioner,
                        local_backend,
                    )
                ] = job.target_clip_uid
            while global_pending and len(global_futures) < global_vl_max_inflight:
                job = global_pending.popleft()
                caption = caption_by_clip[job.target_clip_uid].record
                global_futures[
                    global_executor.submit(
                        _global_record,
                        job,
                        caption,
                        global_backend,
                        global_runtime_provenance,
                    )
                ] = job.target_clip_uid
            while (
                caption_pending
                and len(caption_futures) < captioner_max_inflight
                and len(global_pending) + len(global_futures) < global_capacity
            ):
                job = caption_pending.popleft()
                caption_futures[
                    caption_executor.submit(_captioner_record, job, captioner_backend)
                ] = job.target_clip_uid
            peaks["captioner"] = max(peaks["captioner"], len(caption_futures))
            peaks["global"] = max(peaks["global"], len(global_futures))
            peaks["local"] = max(peaks["local"], len(local_futures))
            peaks["backlog"] = max(
                peaks["backlog"],
                len(global_pending) + len(global_futures),
            )
            all_futures = [*caption_futures, *global_futures, *local_futures]
            if not all_futures:
                raise RuntimeError("specialized pipeline scheduler deadlocked")
            done, _ = wait(all_futures, return_when=FIRST_COMPLETED)
            for future in done:
                if future in caption_futures:
                    clip_uid = caption_futures.pop(future)
                    caption_by_clip[clip_uid] = future.result()
                    if clip_uid not in global_by_clip:
                        global_pending.append(jobs_by_clip[clip_uid])
                    if clip_uid not in local_by_clip:
                        local_pending.append(jobs_by_clip[clip_uid])
                elif future in global_futures:
                    clip_uid = global_futures.pop(future)
                    global_by_clip[clip_uid] = future.result()
                else:
                    clip_uid = local_futures.pop(future)  # type: ignore[arg-type]
                    local_by_clip[clip_uid] = future.result()  # type: ignore[assignment]

    caption_processed = [caption_by_clip[job.target_clip_uid] for job in inventory.jobs]
    global_processed = [global_by_clip[job.target_clip_uid] for job in inventory.jobs]
    local_processed = [local_by_clip[job.target_clip_uid] for job in inventory.jobs]
    if not global_reused:
        global_processed = _run_recaption_rescues(
            jobs=inventory.jobs,
            processed=global_processed,
            captioner_backend=captioner_backend,
            global_backend=global_backend,
            captioner_max_inflight=captioner_max_inflight,
            global_max_inflight=global_vl_max_inflight,
        )
        global_processed = _run_music_rescues(
            jobs=inventory.jobs,
            processed=global_processed,
            music_backend=music_backend,
            max_inflight=local_instruct_max_inflight,
        )
    caption_records = [item.record for item in caption_processed]
    global_records = [item.record for item in global_processed]
    local_records = [item.record for item in local_processed]
    if not caption_reused:
        caption_summary = _summary(
            role="captioner",
            inventory=inventory,
            provenance=captioner_backend.provenance,
            records=caption_records,
        )
        _publish_stage(
            destination=_stage_path(root, "captioner"),
            records=caption_records,
            summary=caption_summary,
            raw_by_clip={
                item.record.target_clip_uid: item.raw for item in caption_processed
            },
        )
    if not global_reused:
        global_summary = _summary(
            role="global_semantics",
            inventory=inventory,
            provenance=global_runtime_provenance,
            records=global_records,
        )
        _publish_stage(
            destination=_stage_path(root, "global_semantics"),
            records=global_records,
            summary=global_summary,
            raw_by_clip={
                item.record.target_clip_uid: item.raw for item in global_processed
            },
        )
    if not local_reused:
        local_summary = _summary(
            role="local_semantics",
            inventory=inventory,
            provenance=local_backend.provenance,
            records=local_records,
        )
        _publish_stage(
            destination=_stage_path(root, "local_semantics"),
            records=local_records,
            summary=local_summary,
            raw_by_clip={
                item.record.target_clip_uid: item.raw for item in local_processed
            },
        )
    assert caption_summary is not None
    assert global_summary is not None
    assert local_summary is not None
    _, assembled_summary = run_assemble_phase(
        inventory=inventory,
        output_root=root,
        overwrite=overwrite,
        regenerate_stale=True,
    )
    return SpecializedPipelineResult(
        inventory_fingerprint=inventory.inventory_fingerprint,
        target_clip_count=len(inventory.jobs),
        captioner_reused=caption_reused,
        global_semantics_reused=global_reused,
        local_semantics_reused=local_reused,
        captioner_max_inflight=captioner_max_inflight,
        global_vl_max_inflight=global_vl_max_inflight,
        local_instruct_max_inflight=local_instruct_max_inflight,
        peak_captioner_inflight=peaks["captioner"],
        peak_global_vl_inflight=peaks["global"],
        peak_local_instruct_inflight=peaks["local"],
        peak_global_backlog=peaks["backlog"],
        captioner_summary=caption_summary,
        global_semantics_summary=global_summary,
        local_semantics_summary=local_summary,
        assembled_summary=assembled_summary,
    )


def build_specialized_inventory(
    *,
    audio_production_root: Path,
) -> SpecializedAudioSemanticsInventory:
    root = audio_production_root.expanduser().resolve(strict=True)
    canonical_path = (root / "audio/canonical_clips.jsonl").resolve(strict=True)
    canonical_summary_path = (
        root / "audio/canonical_clips_summary.json"
    ).resolve(strict=True)
    canonical = [
        CanonicalAudioClip.model_validate_json(line)
        for line in canonical_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = CanonicalAudioClipSummary.model_validate_json(
        canonical_summary_path.read_text(encoding="utf-8")
    )
    if not canonical or summary.canonical_audio_clip_count != len(canonical):
        raise ValueError("canonical audio manifest count is inconsistent")
    canonical_ids = [item.clip_uid for item in canonical]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("canonical audio manifest contains duplicate clips")

    readable_path = root / "diarization/readable_segments.jsonl"
    readable: list[JEAReadableDiarizationSegment] = []
    if readable_path.is_file():
        readable = [
            JEAReadableDiarizationSegment.model_validate_json(line)
            for line in readable_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    canonical_by_clip = {item.clip_uid: item for item in canonical}
    readable_by_clip: dict[str, list[JEAReadableDiarizationSegment]] = defaultdict(list)
    for row in readable:
        if row.clip_uid not in canonical_by_clip:
            raise ValueError("DiariZen clip is outside canonical Audio universe")
        readable_by_clip[row.clip_uid].append(row)

    jobs: list[SpecializedAudioSemanticsJob] = []
    for item in canonical:
        _verify_file(
            item.target_video_path,
            item.target_video_sha256,
            field_name="canonical target video",
        )
        _verify_file(
            item.target_full_audio_path,
            item.target_full_audio_sha256,
            field_name="canonical target full audio",
        )
        if item.target_audio_binding_path is not None:
            assert item.target_audio_binding_sha256 is not None
            _verify_file(
                item.target_audio_binding_path,
                item.target_audio_binding_sha256,
                field_name="canonical target audio binding",
            )
        clip_rows = sorted(
            readable_by_clip.get(item.clip_uid, []),
            key=lambda row: (row.source_start_sample, row.segment_id),
        )
        if any(row.clip_display_path != item.clip_display_path for row in clip_rows):
            raise ValueError("canonical Audio and DiariZen clip paths differ")
        if any(
            row.end_time > item.target_duration_seconds + EVENT_TIME_TOLERANCE_SECONDS
            for row in clip_rows
        ):
            raise ValueError("DiariZen segment exceeds canonical Audio timeline")
        rows_by_cluster: dict[str, list[JEAReadableDiarizationSegment]] = defaultdict(
            list
        )
        cluster_order: list[str] = []
        for row in clip_rows:
            if row.speaker_cluster_id not in rows_by_cluster:
                cluster_order.append(row.speaker_cluster_id)
            rows_by_cluster[row.speaker_cluster_id].append(row)
        clusters: list[SpeakerClusterEvidence] = []
        for cluster_id in cluster_order:
            cluster_rows = rows_by_cluster[cluster_id]
            bound_ids = {
                row.entity_id for row in cluster_rows if row.entity_id is not None
            }
            if len(bound_ids) > 1:
                raise ValueError("speaker cluster resolves to conflicting entity IDs")
            clusters.append(
                SpeakerClusterEvidence(
                    speaker_cluster_id=cluster_id,
                    entity_id=next(iter(bound_ids), None),
                    active_time_ranges=[
                        SpeakerTimeRange(
                            start_time=row.start_time,
                            end_time=row.end_time,
                        )
                        for row in cluster_rows
                    ],
                )
            )
        jobs.append(
            SpecializedAudioSemanticsJob(
                target_clip_uid=item.clip_uid,
                clip_display_path=item.clip_display_path,
                target_video_path=item.target_video_path,
                target_video_sha256=item.target_video_sha256,
                target_full_audio_path=item.target_full_audio_path,
                target_full_audio_sha256=item.target_full_audio_sha256,
                target_duration_seconds=item.target_duration_seconds,
                target_audio_binding_path=item.target_audio_binding_path,
                target_audio_binding_sha256=item.target_audio_binding_sha256,
                speaker_clusters=clusters,
            )
        )
    values: dict[str, object] = {
        "schema_version": SPECIALIZED_INVENTORY_VERSION,
        "source_audio_production_root": str(root),
        "source_canonical_audio_manifest_path": str(canonical_path),
        "source_canonical_audio_manifest_sha256": _sha256_file(canonical_path),
        "source_readable_segments_path": (
            str(readable_path.resolve(strict=True)) if readable_path.is_file() else None
        ),
        "source_readable_segments_sha256": (
            _sha256_file(readable_path) if readable_path.is_file() else None
        ),
        "target_clip_count": len(jobs),
        "readable_segment_count": len(readable),
        "transcript_supplied_to_model": False,
        "entity_id_supplied_to_model": False,
        "donor_media_used": False,
        "jobs": [job.model_dump(mode="json") for job in jobs],
    }
    return SpecializedAudioSemanticsInventory(
        **values,
        inventory_fingerprint=_specialized_inventory_fingerprint(values),
    )


__all__ = [
    "ASSEMBLED_RECORD_VERSION",
    "ASSEMBLED_SUMMARY_VERSION",
    "CAPTIONER_POLICY_VERSION",
    "CAPTIONER_RECORD_VERSION",
    "DEFAULT_CAPTIONER_CHECKPOINT_ID",
    "DEFAULT_CAPTIONER_MODEL",
    "DEFAULT_GLOBAL_VL_BASE_URL",
    "DEFAULT_GLOBAL_VL_CHECKPOINT_ID",
    "DEFAULT_GLOBAL_VL_MODEL",
    "GLOBAL_FALLBACK_PROMPT_VERSION",
    "GLOBAL_PROMPT_VERSION",
    "GLOBAL_RECAPTION_RESCUE_POLICY_VERSION",
    "GLOBAL_RECORD_VERSION",
    "GLOBAL_SYSTEM_PROMPT",
    "LOCAL_CAPTION_HINT_POLICY_VERSION",
    "LOCAL_EVENT_FILTER_POLICY_VERSION",
    "LOCAL_FALLBACK_PROMPT_VERSION",
    "LOCAL_FALLBACK_SYSTEM_PROMPT",
    "LOCAL_FIELD_POLICY_VERSION",
    "LOCAL_PROMPT_VERSION",
    "LOCAL_RECORD_VERSION",
    "LOCAL_RUNTIME_POLICY_VERSION",
    "LOCAL_SYSTEM_PROMPT",
    "MUSIC_RESCUE_POLICY_VERSION",
    "MUSIC_RESCUE_PROMPT_VERSION",
    "MUSIC_RESCUE_SYSTEM_PROMPT",
    "CaptionerBackend",
    "CaptionerBackendResult",
    "CaptionerConfig",
    "CaptionerRecord",
    "DirectAudioMusicResponse",
    "GlobalAudioSemanticsResponse",
    "GlobalSemanticsBackend",
    "GlobalSemanticsConfig",
    "GlobalSemanticsRecord",
    "LocalAudioSemanticsResponse",
    "LocalSemanticsBackend",
    "LocalSemanticsConfig",
    "LocalSemanticsRecord",
    "MusicRescueBackend",
    "MusicRescueConfig",
    "OpenAICaptionerBackend",
    "OpenAIGlobalSemanticsBackend",
    "OpenAILocalSemanticsBackend",
    "OpenAIMusicRescueBackend",
    "SpecializedAssembledSummary",
    "SpecializedAudioSemanticsInventory",
    "SpecializedAudioSemanticsJob",
    "SpecializedAudioSemanticsRecord",
    "SpecializedBackendFailure",
    "SpecializedBackendProvenance",
    "SpecializedBackendResult",
    "SpecializedPipelineResult",
    "SpecializedStageFailure",
    "SpecializedStageSummary",
    "build_specialized_inventory",
    "captioner_request_fingerprint",
    "global_request_fingerprint",
    "local_request_fingerprint",
    "run_assemble_phase",
    "run_captioner_phase",
    "run_global_semantics_phase",
    "run_local_semantics_phase",
    "run_specialized_pipeline",
    "specialized_output_root",
]
