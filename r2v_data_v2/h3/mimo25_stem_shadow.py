from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.audio_backends import AudioMediaBackend
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoInventory,
    MimoSegmentEvidence,
    _inventory,
)
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendProvenance,
    MimoBackendResult,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    MimoSpeakerMarkerPolishAudit,
    OpenAIMimo25Backend,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemInventory,
    SAMAudioStemRecord,
    SAMRoute,
    StemDiarizationClipFailure,
    StemDiarizationShadowProvenance,
    StemShadowClipSkip,
    StemType,
    load_stem_shadow,
    require_shadow_output_path,
    selected_stem_records,
    separation_skips,
    sha256_file,
    validate_stem_asr_lineage,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import ValidationIssue

MIMO25_STEM_FACTS_VERSION = "r2v.h3.mimo25_stem_facts.3"
MIMO25_STEM_FACTS_SUMMARY_VERSION = "r2v.h3.mimo25_stem_facts_summary.4"
MIMO25_STEM_FACT_RAW_VERSION = "r2v.h3.mimo25_stem_fact_raw.1"
MIMO25_STEM_FACT_PROMPT_VERSION = "h3_mimo25_stem_fact_prompt_v1"
MIMO25_STEM_RECONCILE_VERSION = "r2v.h3.mimo25_stem_reconcile.13"
MIMO25_STEM_RECONCILE_SUMMARY_VERSION = "r2v.h3.mimo25_stem_reconcile_summary.15"
MIMO25_STEM_RECONCILE_POLICY_VERSION = "h3_mimo25_stem_reconcile_v6"
MIMO25_STEM_RECONCILE_STAGE = "mimo_reconcile_stemtext_final_av_markerpolish_v1"
STEM_VIEW_VERSION = "r2v.h3.sam_audio_stem_view.1"
STEM_RECONCILE_UPSTREAM_FAILURE_VERSION = (
    "r2v.h3.mimo25_stem_reconcile_upstream_failure.1"
)

Confidence = Literal["high", "medium", "low", "unknown"]
MusicStatus = Literal["present", "absent", "unknown"]
SFXCategory = Literal[
    "physical",
    "environmental",
    "mechanical",
    "electronic",
    "human_non_speech",
    "other",
]


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: SchemaModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[SchemaModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_compact_json(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved = True
        temporary.replace(destination)
        if moved:
            shutil.rmtree(backup)
    except Exception:
        if moved and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise


class StemView(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_view.1"] = STEM_VIEW_VERSION
    clip_uid: str
    route: SAMRoute
    stem_type: StemType
    original_video_path: str
    original_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_path: str
    source_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    view_path: str
    view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_codec_policy: Literal["stream_copy"] = "stream_copy"
    audio_codec_policy: Literal["aac_320k_annotation_proxy"] = (
        "aac_320k_annotation_proxy"
    )
    audio_authority_remains_source_stem: Literal[True] = True


class StemViewBackend(Protocol):
    def create(
        self,
        *,
        clip_uid: str,
        route: SAMRoute,
        stem_type: StemType,
        original_video_path: Path,
        source_stem_path: Path,
        destination: Path,
    ) -> StemView: ...


class FFmpegStemViewBackend:
    def __init__(self, *, ffmpeg: str = "ffmpeg") -> None:
        self.ffmpeg = ffmpeg

    def create(
        self,
        *,
        clip_uid: str,
        route: SAMRoute,
        stem_type: StemType,
        original_video_path: Path,
        source_stem_path: Path,
        destination: Path,
    ) -> StemView:
        video = original_video_path.expanduser().resolve(strict=True)
        stem = source_stem_path.expanduser().resolve(strict=True)
        output = destination.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp-{uuid.uuid4().hex}.mp4")
        try:
            subprocess.run(
                [
                    self.ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-i",
                    str(stem),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k",
                    "-shortest",
                    "-y",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return StemView(
            clip_uid=clip_uid,
            route=route,
            stem_type=stem_type,
            original_video_path=str(video),
            original_video_sha256=sha256_file(video),
            source_stem_path=str(stem),
            source_stem_sha256=sha256_file(stem),
            view_path=str(output.resolve(strict=True)),
            view_sha256=sha256_file(output),
        )


class SpeechStemSegmentFact(SchemaModel):
    segment_id: str
    vocal_composition: str | None = None
    delivery_style: str | None = None
    nonlexical_vocalizations: list[str] = Field(default_factory=list)
    speaker_behavior: str | None = None
    confidence: Confidence


class SpeechStemFacts(SchemaModel):
    clip_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    segment_facts: list[SpeechStemSegmentFact]
    acoustic_notes: str | None = None


class MusicInterval(SchemaModel):
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    confidence: Confidence
    onset: Literal["hard", "fade_in", "uncertain"] | None = None
    offset: Literal["hard", "fade_out", "uncertain"] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> MusicInterval:
        if self.end_time <= self.start_time or not self.description.strip():
            raise ValueError("music interval must be positive and described")
        return self


class MusicStemFacts(SchemaModel):
    clip_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    music_status: MusicStatus
    intervals: list[MusicInterval]

    @model_validator(mode="after")
    def validate_music(self) -> MusicStemFacts:
        if self.music_status == "present" and not self.intervals:
            raise ValueError("present music requires at least one interval")
        if self.music_status != "present" and self.intervals:
            raise ValueError("absent or unknown music cannot fabricate intervals")
        order = [
            (item.start_time, item.end_time, item.description) for item in self.intervals
        ]
        if order != sorted(order):
            raise ValueError("music intervals must be chronological")
        previous_end = 0.0
        for item in self.intervals:
            if item.end_time > self.clip_duration_seconds or item.start_time < previous_end:
                raise ValueError("music intervals must be bounded and non-overlapping")
            previous_end = item.end_time
        return self


class SFXContinuousLayer(SchemaModel):
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_layer(self) -> SFXContinuousLayer:
        if self.end_time <= self.start_time or not self.description.strip():
            raise ValueError("SFX layer must be positive and described")
        return self


class SFXEvent(SFXContinuousLayer):
    category: SFXCategory


class SFXStemFacts(SchemaModel):
    clip_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    continuous_layers: list[SFXContinuousLayer]
    events: list[SFXEvent]
    residual_artifact_notes: str | None = None

    @model_validator(mode="after")
    def validate_timeline(self) -> SFXStemFacts:
        for label, values in (
            ("continuous layers", self.continuous_layers),
            ("events", self.events),
        ):
            order = [(item.start_time, item.end_time, item.description) for item in values]
            if order != sorted(order):
                raise ValueError(f"SFX {label} must be chronological")
            if any(item.end_time > self.clip_duration_seconds for item in values):
                raise ValueError(f"SFX {label} must be within clip duration")
        return self


class StemFactSegment(SchemaModel):
    segment_id: str
    speaker_cluster_id: str
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    asr_status: Literal["transcribed", "empty", "failed"]
    asr_text: str | None = None
    asr_language: str | None = None

    @model_validator(mode="after")
    def validate_asr_fact(self) -> StemFactSegment:
        if self.end_time <= self.start_time:
            raise ValueError("stem fact segment interval must be positive")
        if self.asr_status == "transcribed":
            if self.asr_text is None or not self.asr_text.strip():
                raise ValueError("transcribed stem fact requires authoritative ASR text")
        elif self.asr_text is not None or self.asr_language is not None:
            raise ValueError("non-transcribed stem fact cannot publish ASR text")
        return self


class StemFactJob(SchemaModel):
    clip_uid: str
    route: SAMRoute
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    speech_stem_path: str
    speech_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_stem_path: str
    music_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sfx_stem_path: str
    sfx_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speech_stem_available_duration_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    music_stem_available_duration_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    sfx_stem_available_duration_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    segments: list[StemFactSegment]
    job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_job(self) -> StemFactJob:
        available = (
            self.speech_stem_available_duration_seconds,
            self.music_stem_available_duration_seconds,
            self.sfx_stem_available_duration_seconds,
        )
        if any(item > self.clip_duration_seconds for item in available) or any(
            item.end_time > self.speech_stem_available_duration_seconds
            for item in self.segments
        ):
            raise ValueError("stem fact timeline exceeds available source media")
        values = self.model_dump(mode="json", exclude={"job_fingerprint"})
        if self.job_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("stem fact job fingerprint is invalid")
        return self


class StemFactsBackendProvenance(SchemaModel):
    backend: str
    model: str
    base_url: str
    media_mode: Literal["base64", "http"]
    prompt_version: Literal["h3_mimo25_stem_fact_prompt_v1"] = (
        MIMO25_STEM_FACT_PROMPT_VERSION
    )
    response_format: Literal["json_schema"] = "json_schema"
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_completion_tokens: int = Field(gt=0)
    original_av_is_final_authority: Literal[True] = True
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class StemFactsBackend(Protocol):
    provenance: StemFactsBackendProvenance

    def extract(
        self,
        *,
        job: StemFactJob,
        stem_type: StemType,
        view: StemView,
    ) -> StemFactsBackendResult: ...


@dataclass(frozen=True)
class StemFactsBackendResult:
    facts: SpeechStemFacts | MusicStemFacts | SFXStemFacts
    raw_response: str
    diagnostics: dict[str, object]


class StemFactsBackendFailure(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_response: str | None,
        diagnostics: dict[str, object],
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_response = raw_response
        self.diagnostics = diagnostics


class StemFactCallAudit(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_fact_raw.1"] = (
        MIMO25_STEM_FACT_RAW_VERSION
    )
    clip_uid: str
    route: SAMRoute
    stem_type: StemType
    job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_index: int = Field(ge=1, le=3)
    status: Literal["ready", "failed"]
    raw_response: str | None = None
    request_diagnostics: dict[str, object]
    failure_code: str | None = None
    failure_type: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_audit(self) -> StemFactCallAudit:
        if self.status == "ready":
            if (
                self.raw_response is None
                or self.failure_code is not None
                or self.failure_type is not None
                or self.failure_reason is not None
            ):
                raise ValueError("ready stem fact call requires raw response only")
        elif not self.failure_code or not self.failure_type or not self.failure_reason:
            raise ValueError("failed stem fact call requires failure provenance")
        return self


class OpenAIStemFactsBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        media_resolver: MimoMediaResolver,
        temperature: float = 0.2,
        max_completion_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        if not model.strip() or not base_url.strip() or not api_key.strip():
            raise ValueError("stem facts backend configuration is incomplete")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.base_url = base_url
        self.media_resolver = media_resolver
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        values = {
            "backend": "sglang_openai_compatible",
            "model": model,
            "base_url": base_url,
            "media_mode": media_resolver.mode,
            "prompt_version": MIMO25_STEM_FACT_PROMPT_VERSION,
            "response_format": "json_schema",
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "original_av_is_final_authority": True,
        }
        self.provenance = StemFactsBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )

    @staticmethod
    def _schema(stem_type: StemType) -> type[SchemaModel]:
        return {
            "speech": SpeechStemFacts,
            "music": MusicStemFacts,
            "sfx": SFXStemFacts,
        }[stem_type]

    @staticmethod
    def _prompt(job: StemFactJob, stem_type: StemType) -> str:
        common = (
            "Observe the complete clip-local timeline through this separated-stem AV "
            "view. The separated stem is auxiliary evidence, not ground truth. Never "
            "invent a sound, rewrite ASR, bind a speaker to a person/entity, or change "
            "supplied segment boundaries. Keep all times within clip_duration_seconds. "
        )
        specific = {
            "speech": (
                "Describe only vocal composition, delivery, nonlexical vocalizations, "
                "sequential/overlap behavior, and acoustic confidence for each exact "
                "segment_id. Supplied asr_text and asr_language are authoritative facts; "
                "do not rewrite them. The response schema intentionally contains no "
                "transcript field."
            ),
            "music": (
                "Report a chronological non-overlapping timeline of genuinely audible "
                "music. Do not decide diegetic versus non-diegetic from the stem alone. "
                "Present requires intervals; absent/unknown require none."
            ),
            "sfx": (
                "Separate sustained layers from discrete events. Residual separator "
                "artifacts may be noted and must not be promoted to factual sound."
            ),
        }[stem_type]
        return common + specific + "\nAUTHORITATIVE JOB:\n" + _compact_json(
            job.model_dump(mode="json")
        )

    def extract(
        self,
        *,
        job: StemFactJob,
        stem_type: StemType,
        view: StemView,
    ) -> StemFactsBackendResult:
        schema = self._schema(stem_type)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return exactly one compact JSON object with no prose.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": self.media_resolver.resolve(Path(view.view_path))
                            },
                            "fps": 4.0,
                            "media_resolution": "default",
                        },
                        {"type": "text", "text": self._prompt(job, stem_type)},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stream=False,
            reasoning_effort="none",
            extra_body={
                "use_audio_in_video": True,
                "chat_template_kwargs": {
                    "thinking": False,
                    "enable_thinking": False,
                },
            },
        )
        raw = completion.choices[0].message.content
        usage = getattr(completion, "usage", None)
        diagnostics = {
            "finish_reason": getattr(completion.choices[0], "finish_reason", None),
            "usage": (
                usage.model_dump(mode="json")
                if hasattr(usage, "model_dump")
                else None
            ),
            "view_sha256": view.view_sha256,
            "input_modality": "derived_video_with_embedded_stem_audio",
        }
        if not isinstance(raw, str):
            raise StemFactsBackendFailure(
                code="stem_fact_response_not_text",
                reason="stem fact response content must be text JSON",
                raw_response=None,
                diagnostics=diagnostics,
            )
        try:
            facts = schema.model_validate_json(raw)
        except Exception as exc:
            raise StemFactsBackendFailure(
                code="stem_fact_structured_output_failed",
                reason=str(exc),
                raw_response=raw,
                diagnostics=diagnostics,
            ) from exc
        return StemFactsBackendResult(
            facts=facts,
            raw_response=raw,
            diagnostics=diagnostics,
        )


class MimoStemFactsRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_facts.3"] = (
        MIMO25_STEM_FACTS_VERSION
    )
    clip_uid: str
    route: SAMRoute
    status: Literal["ready", "failed"]
    job_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_stem_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_verification_state: Literal["success", "uncertain", "unverified"]
    unverified_consumption_authorized: bool
    backend_provenance: StemFactsBackendProvenance
    stem_views: list[StemView]
    speech: SpeechStemFacts | None = None
    music: MusicStemFacts | None = None
    sfx: SFXStemFacts | None = None
    failing_stem_type: StemType | None = None
    failure_code: str | None = None
    failure_type: str | None = None
    failure_reason: str | None = None
    model_call_count: int = Field(ge=0, le=3)
    raw_responses: list[StemFactCallAudit]
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoStemFactsRecord:
        view_types = [item.stem_type for item in self.stem_views]
        if view_types != ["speech", "music", "sfx"][: len(view_types)]:
            raise ValueError("stem fact record views must preserve stem order")
        if (
            self.source_verification_state == "unverified"
            and not self.unverified_consumption_authorized
        ):
            raise ValueError("unverified stem facts lack explicit authorization")
        if self.model_call_count != len(self.raw_responses):
            raise ValueError("stem fact model-call count differs from raw audit")
        if [item.call_index for item in self.raw_responses] != list(
            range(1, self.model_call_count + 1)
        ):
            raise ValueError("stem fact raw audit call order is invalid")
        if any(
            item.clip_uid != self.clip_uid
            or item.route != self.route
            or item.job_fingerprint != self.job_fingerprint
            or item.backend_configuration_fingerprint
            != self.backend_provenance.configuration_fingerprint
            for item in self.raw_responses
        ):
            raise ValueError("stem fact raw audit provenance differs")
        if self.status == "ready":
            if (
                self.job_fingerprint is None
                or view_types != ["speech", "music", "sfx"]
                or self.speech is None
                or self.music is None
                or self.sfx is None
                or self.failing_stem_type is not None
                or self.failure_code is not None
                or self.failure_type is not None
                or self.failure_reason is not None
                or self.model_call_count != 3
                or any(item.status != "ready" for item in self.raw_responses)
            ):
                raise ValueError("ready stem facts require three successful stem calls")
        elif (
            self.speech is not None
            or self.music is not None
            or self.sfx is not None
            or not self.failure_code
            or not self.failure_type
            or not self.failure_reason
        ):
            raise ValueError("failed stem facts require only failure provenance")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("stem fact record fingerprint is invalid")
        return self


class MimoStemFactsSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_facts_summary.4"] = (
        MIMO25_STEM_FACTS_SUMMARY_VERSION
    )
    record_count: int = Field(ge=0)
    source_stem_root: str
    source_stem_diarization_root: str
    source_stem_asr_root: str
    source_stem_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_diarization_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_asr_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route: SAMRoute
    source_clip_uids: list[str] = Field(min_length=1)
    clip_uids: list[str]
    ready_clip_uids: list[str]
    failed_clip_uids: list[str]
    diarization_ready_clip_uids: list[str]
    diarization_empty_clip_uids: list[str]
    skipped_clips: list[StemShadowClipSkip]
    diarization_failed_clips: list[StemDiarizationClipFailure]
    unverified_clip_uids: list[str]
    unverified_consumption_authorized: bool
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoStemFactsSummary:
        if any(
            not item.strip()
            for item in (
                self.source_stem_root,
                self.source_stem_diarization_root,
                self.source_stem_asr_root,
            )
        ):
            raise ValueError("stem facts source stage root is empty")
        excluded = {item.clip_uid for item in self.skipped_clips}.union(
            item.clip_uid for item in self.diarization_failed_clips
        )
        if (
            self.record_count != len(self.clip_uids)
            or self.record_count != self.ready_count + self.failed_count
            or self.ready_count != len(self.ready_clip_uids)
            or self.failed_count != len(self.failed_clip_uids)
            or [
                item for item in self.clip_uids if item in set(self.ready_clip_uids)
            ]
            != self.ready_clip_uids
            or [
                item for item in self.clip_uids if item in set(self.failed_clip_uids)
            ]
            != self.failed_clip_uids
            or set(self.ready_clip_uids).intersection(self.failed_clip_uids)
            or set(self.ready_clip_uids).union(self.failed_clip_uids)
            != set(self.clip_uids)
            or len(self.source_clip_uids) != len(set(self.source_clip_uids))
            or [
                item
                for item in self.source_clip_uids
                if item in set(self.clip_uids)
            ]
            != self.clip_uids
            or excluded != set(self.source_clip_uids) - set(self.clip_uids)
            or self.clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item
                in set(self.diarization_ready_clip_uids).union(
                    self.diarization_empty_clip_uids
                )
            ]
            or self.diarization_ready_clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item in set(self.diarization_ready_clip_uids)
            ]
            or self.diarization_empty_clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item in set(self.diarization_empty_clip_uids)
            ]
            or set(self.diarization_ready_clip_uids).intersection(
                self.diarization_empty_clip_uids
            )
            or [item.clip_uid for item in self.diarization_failed_clips]
            != [
                item
                for item in self.source_clip_uids
                if item
                in {failure.clip_uid for failure in self.diarization_failed_clips}
            ]
            or any(item.route != self.route for item in self.diarization_failed_clips)
            or set(self.unverified_clip_uids) - set(self.clip_uids)
            or (
                self.unverified_clip_uids
                and not self.unverified_consumption_authorized
            )
        ):
            raise ValueError("stem facts ordered clip provenance is inconsistent")
        return self


def _stem_fact_job(
    *,
    stem_inventory: SAMAudioStemInventory,
    record: SAMAudioStemRecord,
    segments: Sequence[RawDiarizationSegment],
    asr: Sequence[Qwen3ASRSegment],
) -> StemFactJob:
    source = next(item for item in stem_inventory.jobs if item.clip_uid == record.clip_uid)
    target_video = Path(source.target_video_path).resolve(strict=True)
    if sha256_file(target_video) != source.target_video_sha256:
        raise ValueError("stem fact target video hash changed")
    stems = {stem_type: record.stem(stem_type) for stem_type in ("speech", "music", "sfx")}
    for stem in stems.values():
        stem_path = Path(stem.canonical_stem_path).resolve(strict=True)
        if sha256_file(stem_path) != stem.canonical_stem_sha256:
            raise ValueError("stem fact canonical stem hash changed")
    asr_by_id = {item.segment_id: item for item in asr if item.clip_uid == record.clip_uid}
    clip_segments = sorted(
        (item for item in segments if item.target_clip_uid == record.clip_uid),
        key=lambda item: (item.start_time, item.end_time, item.segment_id),
    )
    if set(asr_by_id) != {item.segment_id for item in clip_segments}:
        raise ValueError("stem facts DiariZen and ASR inventories differ")
    segment_payload = []
    for item in clip_segments:
        transcript = asr_by_id[item.segment_id]
        if (
            item.source_audio_path != stems["speech"].canonical_stem_path
            or item.source_audio_sha256 != stems["speech"].canonical_stem_sha256
            or transcript.source_audio_path != stems["speech"].canonical_stem_path
        ):
            raise ValueError("stem ASR did not consume the selected speech stem")
        segment_payload.append(
            {
                "segment_id": item.segment_id,
                "speaker_cluster_id": item.speaker_cluster_id,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "asr_status": transcript.status,
                "asr_text": transcript.text,
                "asr_language": transcript.language,
            }
        )
    values = {
        "clip_uid": record.clip_uid,
        "route": record.route,
        "target_video_path": str(target_video),
        "target_video_sha256": source.target_video_sha256,
        "clip_duration_seconds": source.source_duration_seconds,
        "speech_stem_path": stems["speech"].canonical_stem_path,
        "speech_stem_sha256": stems["speech"].canonical_stem_sha256,
        "music_stem_path": stems["music"].canonical_stem_path,
        "music_stem_sha256": stems["music"].canonical_stem_sha256,
        "sfx_stem_path": stems["sfx"].canonical_stem_path,
        "sfx_stem_sha256": stems["sfx"].canonical_stem_sha256,
        "speech_stem_available_duration_seconds": min(
            source.source_duration_seconds,
            stems["speech"].canonical_duration_seconds,
        ),
        "music_stem_available_duration_seconds": min(
            source.source_duration_seconds,
            stems["music"].canonical_duration_seconds,
        ),
        "sfx_stem_available_duration_seconds": min(
            source.source_duration_seconds,
            stems["sfx"].canonical_duration_seconds,
        ),
        "segments": segment_payload,
    }
    return StemFactJob(
        **values,
        job_fingerprint=_sha256_text(_compact_json(values)),
    )


def _validate_speech_fact_inventory(
    facts: SpeechStemFacts,
    job: StemFactJob,
) -> None:
    expected = [item.segment_id for item in job.segments]
    actual = [item.segment_id for item in facts.segment_facts]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("speech stem facts must preserve exact segment inventory")
    if facts.clip_duration_seconds != job.clip_duration_seconds:
        raise ValueError("speech stem facts clip duration differs")


def _validate_music_fact_extent(
    music: MusicStemFacts,
    job: StemFactJob,
) -> None:
    if any(
        item.end_time > job.music_stem_available_duration_seconds
        for item in music.intervals
    ):
        raise ValueError("music stem fact exceeds available source media")


def _validate_sfx_fact_extent(
    sfx: SFXStemFacts,
    job: StemFactJob,
) -> None:
    if any(
        item.end_time > job.sfx_stem_available_duration_seconds
        for item in (*sfx.continuous_layers, *sfx.events)
    ):
        raise ValueError("SFX stem fact exceeds available source media")


def _published_stem_views(
    views: Sequence[StemView],
    *,
    temporary_root: Path,
    destination_root: Path,
) -> list[StemView]:
    old = str(temporary_root)
    new = str(destination_root)
    output: list[StemView] = []
    for view in views:
        values = view.model_dump(mode="json")
        path = values["view_path"]
        if not isinstance(path, str) or not (path == old or path.startswith(old + "/")):
            raise ValueError("stem view is outside the owned facts stage")
        values["view_path"] = new + path[len(old) :]
        output.append(StemView.model_validate(values))
    return output


def run_mimo25_stem_facts_shadow(
    *,
    stem_root: Path,
    route: SAMRoute,
    backend: StemFactsBackend,
    view_backend: StemViewBackend,
    output_root: Path,
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> MimoStemFactsSummary:
    separation_root = stem_root.expanduser().resolve(strict=True)
    if separation_root.name != "separation":
        raise ValueError("stem facts require the owned SAM Audio separation root")
    shadow_root = separation_root.parent
    stem_inventory, stem_records, _ = load_stem_shadow(separation_root)
    separation_selected = selected_stem_records(
        stem_records,
        route=route,
        allow_unverified=allow_unverified,
    )
    selected_by_clip = {item.clip_uid: item for item in separation_selected}
    separation_selected = [
        selected_by_clip[clip_uid]
        for clip_uid in stem_inventory.clip_uids
        if clip_uid in selected_by_clip
    ]
    skips = separation_skips(
        inventory=stem_inventory,
        records=stem_records,
        route=route,
    )
    diarization_root = shadow_root / "diarization"
    asr_root = shadow_root / "asr"
    diarization_provenance, _, _ = validate_stem_diarization_lineage(
        diarization_root, expected_shadow_root=shadow_root,
    )
    asr_provenance, asr_source_provenance = validate_stem_asr_lineage(
        asr_root, expected_shadow_root=shadow_root,
    )
    separation_selected_ids = [item.clip_uid for item in separation_selected]
    selected_ids = diarization_provenance.usable_clip_uids
    selected = [selected_by_clip[item] for item in selected_ids]
    if (
        Path(diarization_provenance.source_stem_root).resolve(strict=True)
        != separation_root
        or asr_source_provenance != diarization_provenance
        or diarization_provenance.route != route
        or asr_provenance.route != route
        or diarization_provenance.target_count != len(separation_selected_ids)
        or [
            item
            for item in stem_inventory.clip_uids
            if item in set(separation_selected_ids)
        ]
        != separation_selected_ids
        or asr_provenance.clip_uids != selected_ids
    ):
        raise ValueError("stem facts upstream route or ordered inventory differs")
    raw = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization_root / "raw_segments.jsonl")
    ]
    asr = [
        Qwen3ASRSegment.model_validate(item)
        for item in _read_jsonl(asr_root / "segments.jsonl")
    ]
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    records: list[MimoStemFactsRecord] = []
    try:
        temporary.mkdir(parents=True)
        views_root = temporary / "stem_views"
        for record in selected:
            job: StemFactJob | None = None
            views: list[StemView] = []
            audits: list[StemFactCallAudit] = []
            failing_stem_type: StemType | None = None
            failure_code: str | None = None
            failure_type: str | None = None
            failure_reason: str | None = None
            facts: dict[StemType, SpeechStemFacts | MusicStemFacts | SFXStemFacts] = {}
            try:
                job = _stem_fact_job(
                    stem_inventory=stem_inventory,
                    record=record,
                    segments=raw,
                    asr=asr,
                )
                source = next(
                    item
                    for item in stem_inventory.jobs
                    if item.clip_uid == record.clip_uid
                )
                for stem_type in ("speech", "music", "sfx"):
                    failing_stem_type = stem_type
                    view = view_backend.create(
                        clip_uid=record.clip_uid,
                        route=route,
                        stem_type=stem_type,
                        original_video_path=Path(source.target_video_path),
                        source_stem_path=Path(
                            record.stem(stem_type).canonical_stem_path
                        ),
                        destination=(
                            views_root / record.clip_uid / route / f"{stem_type}.mp4"
                        ),
                    )
                    stem = record.stem(view.stem_type)
                    if (
                        view.clip_uid != record.clip_uid
                        or view.route != route
                        or view.original_video_path != job.target_video_path
                        or view.original_video_sha256 != job.target_video_sha256
                        or view.source_stem_path != stem.canonical_stem_path
                        or view.source_stem_sha256 != stem.canonical_stem_sha256
                    ):
                        raise ValueError("stem annotation view provenance differs")
                    views.append(view)
                for view in views:
                    failing_stem_type = view.stem_type
                    call_index = len(audits) + 1
                    try:
                        result = backend.extract(
                            job=job,
                            stem_type=view.stem_type,
                            view=view,
                        )
                    except StemFactsBackendFailure as exc:
                        audits.append(
                            StemFactCallAudit(
                                clip_uid=record.clip_uid,
                                route=route,
                                stem_type=view.stem_type,
                                job_fingerprint=job.job_fingerprint,
                                backend_configuration_fingerprint=(
                                    backend.provenance.configuration_fingerprint
                                ),
                                call_index=call_index,
                                status="failed",
                                raw_response=exc.raw_response,
                                request_diagnostics=exc.diagnostics,
                                failure_code=exc.code,
                                failure_type=type(exc).__name__,
                                failure_reason=exc.reason,
                            )
                        )
                        failure_code = exc.code
                        failure_type = type(exc).__name__
                        failure_reason = exc.reason
                        raise
                    except Exception as exc:
                        audits.append(
                            StemFactCallAudit(
                                clip_uid=record.clip_uid,
                                route=route,
                                stem_type=view.stem_type,
                                job_fingerprint=job.job_fingerprint,
                                backend_configuration_fingerprint=(
                                    backend.provenance.configuration_fingerprint
                                ),
                                call_index=call_index,
                                status="failed",
                                raw_response=None,
                                request_diagnostics={
                                    "view_sha256": view.view_sha256,
                                    "input_modality": (
                                        "derived_video_with_embedded_stem_audio"
                                    ),
                                },
                                failure_code="stem_fact_backend_failed",
                                failure_type=type(exc).__name__,
                                failure_reason=str(exc),
                            )
                        )
                        failure_code = "stem_fact_backend_failed"
                        failure_type = type(exc).__name__
                        failure_reason = str(exc)
                        raise
                    audits.append(
                        StemFactCallAudit(
                            clip_uid=record.clip_uid,
                            route=route,
                            stem_type=view.stem_type,
                            job_fingerprint=job.job_fingerprint,
                            backend_configuration_fingerprint=(
                                backend.provenance.configuration_fingerprint
                            ),
                            call_index=call_index,
                            status="ready",
                            raw_response=result.raw_response,
                            request_diagnostics=result.diagnostics,
                        )
                    )
                    facts[view.stem_type] = result.facts
                speech = facts["speech"]
                music = facts["music"]
                sfx = facts["sfx"]
                if not isinstance(speech, SpeechStemFacts):
                    raise TypeError("speech stem backend returned the wrong schema")
                if not isinstance(music, MusicStemFacts):
                    raise TypeError("music stem backend returned the wrong schema")
                if not isinstance(sfx, SFXStemFacts):
                    raise TypeError("SFX stem backend returned the wrong schema")
                failing_stem_type = "speech"
                _validate_speech_fact_inventory(speech, job)
                failing_stem_type = "music"
                if music.clip_duration_seconds != job.clip_duration_seconds:
                    raise ValueError("stem fact duration differs from source clip")
                _validate_music_fact_extent(music, job)
                failing_stem_type = "sfx"
                if sfx.clip_duration_seconds != job.clip_duration_seconds:
                    raise ValueError("stem fact duration differs from source clip")
                _validate_sfx_fact_extent(sfx, job)
                failing_stem_type = None
                values = {
                    "schema_version": MIMO25_STEM_FACTS_VERSION,
                    "clip_uid": record.clip_uid,
                    "route": route,
                    "status": "ready",
                    "job_fingerprint": job.job_fingerprint,
                    "source_stem_record_fingerprint": record.record_fingerprint,
                    "source_verification_state": record.separation_state,
                    "unverified_consumption_authorized": (
                        record.separation_state != "unverified" or allow_unverified
                    ),
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "stem_views": [
                        item.model_dump(mode="json")
                        for item in _published_stem_views(
                            views,
                            temporary_root=temporary,
                            destination_root=destination,
                        )
                    ],
                    "speech": speech.model_dump(mode="json"),
                    "music": music.model_dump(mode="json"),
                    "sfx": sfx.model_dump(mode="json"),
                    "failing_stem_type": None,
                    "failure_code": None,
                    "failure_type": None,
                    "failure_reason": None,
                    "model_call_count": len(audits),
                    "raw_responses": [item.model_dump(mode="json") for item in audits],
                }
            except Exception as exc:  # noqa: BLE001 - isolate one clip
                failure_code = failure_code or "stem_fact_clip_failed"
                failure_type = failure_type or type(exc).__name__
                failure_reason = failure_reason or str(exc)
                previous_views = destination / "stem_views" / record.clip_uid / route
                failed_views = views_root / record.clip_uid / route
                if previous_views.is_dir():
                    if failed_views.exists():
                        shutil.rmtree(failed_views)
                    failed_views.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(previous_views, failed_views)
                elif failed_views.exists():
                    shutil.rmtree(failed_views)
                values = {
                    "schema_version": MIMO25_STEM_FACTS_VERSION,
                    "clip_uid": record.clip_uid,
                    "route": route,
                    "status": "failed",
                    "job_fingerprint": job.job_fingerprint if job is not None else None,
                    "source_stem_record_fingerprint": record.record_fingerprint,
                    "source_verification_state": record.separation_state,
                    "unverified_consumption_authorized": (
                        record.separation_state != "unverified" or allow_unverified
                    ),
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "stem_views": [],
                    "speech": None,
                    "music": None,
                    "sfx": None,
                    "failing_stem_type": failing_stem_type,
                    "failure_code": failure_code,
                    "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "model_call_count": len(audits),
                    "raw_responses": [item.model_dump(mode="json") for item in audits],
                }
            records.append(
                MimoStemFactsRecord(
                    **values,
                    record_fingerprint=_sha256_text(_compact_json(values)),
                )
            )
            _write_json(
                temporary / "raw_responses" / f"{record.clip_uid}.json",
                {
                    "clip_uid": record.clip_uid,
                    "route": route,
                    "record_status": values["status"],
                    "calls": [item.model_dump(mode="json") for item in audits],
                },
            )
        counts = Counter(item.status for item in records)
        records_path = temporary / "records.jsonl"
        _write_jsonl(records_path, records)
        summary = MimoStemFactsSummary(
            record_count=len(records),
            source_stem_root=str(separation_root),
            source_stem_diarization_root=str(diarization_root),
            source_stem_asr_root=str(asr_root),
            source_stem_inventory_fingerprint=stem_inventory.inventory_fingerprint,
            source_stem_records_sha256=sha256_file(separation_root / "records.jsonl"),
            source_stem_diarization_provenance_sha256=sha256_file(
                diarization_root / "stem_provenance.json"
            ),
            source_stem_asr_provenance_sha256=sha256_file(
                asr_root / "stem_provenance.json"
            ),
            records_sha256=sha256_file(records_path),
            route=route,
            source_clip_uids=stem_inventory.clip_uids,
            clip_uids=[item.clip_uid for item in records],
            ready_clip_uids=[item.clip_uid for item in records if item.status == "ready"],
            failed_clip_uids=[
                item.clip_uid for item in records if item.status == "failed"
            ],
            diarization_ready_clip_uids=diarization_provenance.ready_clip_uids,
            diarization_empty_clip_uids=diarization_provenance.empty_clip_uids,
            skipped_clips=skips,
            diarization_failed_clips=(
                diarization_provenance.diarization_failed_clips
            ),
            unverified_clip_uids=[
                item.clip_uid
                for item in selected
                if item.separation_state == "unverified"
            ],
            unverified_consumption_authorized=allow_unverified,
            ready_count=counts["ready"],
            failed_count=counts["failed"],
            model_call_count=sum(item.model_call_count for item in records),
        )
        _write_jsonl(temporary / "skipped_clips.jsonl", skips)
        _write_jsonl(
            temporary / "diarization_failed_clips.jsonl",
            diarization_provenance.diarization_failed_clips,
        )
        _write_json(temporary / "summary.json", summary)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_stem_fact_records(path: Path) -> list[MimoStemFactsRecord]:
    return [
        MimoStemFactsRecord.model_validate(item)
        for item in _read_jsonl(path.expanduser().resolve(strict=True) / "records.jsonl")
    ]


def validate_stem_facts_lineage(
    *,
    facts_root: Path,
    stem_diarization_root: Path,
    stem_asr_root: Path,
    route: SAMRoute,
) -> tuple[MimoStemFactsSummary, list[MimoStemFactsRecord]]:
    facts_path = facts_root.expanduser().resolve(strict=True)
    diarization = stem_diarization_root.expanduser().resolve(strict=True)
    asr_root = stem_asr_root.expanduser().resolve(strict=True)
    require_shadow_output_path(
        shadow_root=diarization.parent, output_path=facts_path,
    )
    diarization_provenance, stem_inventory, stem_records = (
        validate_stem_diarization_lineage(
            diarization, expected_shadow_root=diarization.parent,
        )
    )
    asr_provenance, asr_source = validate_stem_asr_lineage(
        asr_root, expected_shadow_root=diarization.parent,
    )
    if asr_source != diarization_provenance:
        raise ValueError("stem facts ASR does not consume current DiariZen provenance")
    summary = MimoStemFactsSummary.model_validate_json(
        (facts_path / "summary.json").read_text(encoding="utf-8")
    )
    records = load_stem_fact_records(facts_path)
    if (
        summary.route != route
        or diarization_provenance.route != route
        or asr_provenance.route != route
    ):
        raise ValueError("stem facts route differs across shadow stages")
    if (
        Path(summary.source_stem_root).expanduser().resolve(strict=True)
        != Path(diarization_provenance.source_stem_root).resolve(strict=True)
        or Path(summary.source_stem_diarization_root).expanduser().resolve(strict=True)
        != diarization
        or Path(summary.source_stem_asr_root).expanduser().resolve(strict=True)
        != asr_root
        or summary.source_stem_inventory_fingerprint
        != stem_inventory.inventory_fingerprint
        or summary.source_stem_records_sha256
        != sha256_file(Path(diarization_provenance.source_stem_root) / "records.jsonl")
        or summary.source_stem_diarization_provenance_sha256
        != sha256_file(diarization / "stem_provenance.json")
        or summary.source_stem_asr_provenance_sha256
        != sha256_file(asr_root / "stem_provenance.json")
        or summary.records_sha256 != sha256_file(facts_path / "records.jsonl")
    ):
        raise ValueError("stem facts source lineage changed")
    if summary.source_clip_uids != stem_inventory.clip_uids:
        raise ValueError("stem facts source clip order changed")
    selected = selected_stem_records(stem_records, route=route, allow_unverified=True)
    selected_by_clip = {item.clip_uid: item for item in selected}
    expected_ids = diarization_provenance.usable_clip_uids
    if (
        expected_ids != asr_provenance.clip_uids
        or expected_ids != summary.clip_uids
        or [item.clip_uid for item in records] != expected_ids
        or summary.ready_clip_uids
        != [item.clip_uid for item in records if item.status == "ready"]
        or summary.failed_clip_uids
        != [item.clip_uid for item in records if item.status == "failed"]
        or summary.diarization_ready_clip_uids
        != diarization_provenance.ready_clip_uids
        or summary.diarization_empty_clip_uids
        != diarization_provenance.empty_clip_uids
        or summary.diarization_failed_clips
        != diarization_provenance.diarization_failed_clips
        or summary.model_call_count
        != sum(item.model_call_count for item in records)
        or summary.skipped_clips
        != separation_skips(
            inventory=stem_inventory,
            records=stem_records,
            route=route,
        )
    ):
        raise ValueError("stem facts ordered inventory differs from upstream")
    raw_segments = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization / "raw_segments.jsonl")
    ]
    asr_segments = [
        Qwen3ASRSegment.model_validate(item)
        for item in _read_jsonl(asr_root / "segments.jsonl")
    ]
    for record in records:
        source_record = selected_by_clip[record.clip_uid]
        if (
            record.route != route
            or record.source_stem_record_fingerprint
            != source_record.record_fingerprint
        ):
            raise ValueError("stem fact record differs from current separation")
        expected_job = _stem_fact_job(
            stem_inventory=stem_inventory,
            record=source_record,
            segments=raw_segments,
            asr=asr_segments,
        )
        if record.job_fingerprint is not None and (
            record.job_fingerprint != expected_job.job_fingerprint
        ):
            raise ValueError("stem fact job fingerprint changed")
        raw_payload = json.loads(
            (facts_path / "raw_responses" / f"{record.clip_uid}.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(raw_payload, dict) or raw_payload != {
            "clip_uid": record.clip_uid,
            "route": record.route,
            "record_status": record.status,
            "calls": [item.model_dump(mode="json") for item in record.raw_responses],
        }:
            raise ValueError("stem fact raw response audit differs from record")
    return summary, records


def stem_reconcile_auxiliary_contract(facts: MimoStemFactsRecord) -> dict[str, object]:
    if facts.status != "ready":
        raise ValueError("stem reconcile requires ready stem facts")
    published_facts = facts.model_dump(
        mode="json",
        exclude={
            "status",
            "failing_stem_type",
            "failure_code",
            "failure_type",
            "failure_reason",
            "model_call_count",
            "raw_responses",
        },
    )
    return {
        "policy_version": MIMO25_STEM_RECONCILE_POLICY_VERSION,
        "authority_order": [
            "original_target_full_av",
            "direct_audiovisual_evidence",
            "stem_derived_facts",
            "separator_labels",
        ],
        "rules": [
            "Original target AV is final factual authority.",
            "Stem facts may increase recall but cannot override contradictory original AV.",
            "Negative stem evidence is non-confirmatory; positive evidence may increase recall.",
            "Music stem absent does NOT establish that original target has no music.",
            "Empty SFX items do NOT establish absent original ambience or physical sounds.",
            "Separator labels are never semantic truth.",
            "Independently decide soundscape/music from original target AV; uncertainty is unknown, not absent.",
            "Music-stem labels do not decide diegetic versus non-diegetic music.",
            "Preserve verified music onset and offset naturally in the existing H3 section.",
            "Do not duplicate music across detailed description, soundscape, and music.",
        ],
        "stem_facts": published_facts,
    }


class StemAwareOpenAIMimo25Backend(OpenAIMimo25Backend):
    def __init__(
        self,
        config: MimoBackendConfig,
        *,
        stem_records_by_clip: dict[str, SAMAudioStemRecord],
        client: Any | None = None,
        audio_media_backend: AudioMediaBackend | None = None,
    ) -> None:
        super().__init__(config, client=client, audio_media_backend=audio_media_backend)
        self.stem_records_by_clip = stem_records_by_clip


def _mimo_job(values: dict[str, object]) -> MimoClipJob:
    return MimoClipJob(
        **values,
        request_fingerprint=_sha256_text(_compact_json(values)),
    )


def usable_stem_reconcile_inventory(
    inventory: MimoInventory, provenance: StemDiarizationShadowProvenance,
) -> MimoInventory:
    """Separate known upstream failures before strict reconcile job selection."""
    requested = [job.clip_uid for job in inventory.jobs]
    selected = set(requested)
    if [uid for uid in provenance.clip_uids if uid in selected] != requested:
        raise ValueError("stem reconcile inventory is not an ordered upstream subset")
    usable = set(provenance.usable_clip_uids)
    values = inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
    if values["source_diarization_inventory_sha256"] is None:
        values.pop("source_diarization_inventory_sha256")
    values["jobs"] = [job.model_dump(mode="json") for job in inventory.jobs if job.clip_uid in usable]
    values["clip_count"] = len(values["jobs"])
    return _inventory(values)


def build_stem_reconcile_jobs(
    *,
    base_inventory: MimoInventory,
    stem_diarization_root: Path,
    stem_asr_root: Path,
    route: SAMRoute | None = None,
) -> list[MimoClipJob]:
    diarization = stem_diarization_root.expanduser().resolve(strict=True)
    asr_root = stem_asr_root.expanduser().resolve(strict=True)
    diarization_provenance, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=diarization.parent,
    )
    asr_provenance, asr_source = validate_stem_asr_lineage(
        asr_root, expected_shadow_root=diarization.parent,
    )
    if (
        asr_source != diarization_provenance
        or asr_provenance.source_clip_uids != diarization_provenance.clip_uids
        or asr_provenance.clip_uids != diarization_provenance.usable_clip_uids
        or asr_provenance.source_stem_inventory_fingerprint
        != diarization_provenance.source_stem_inventory_fingerprint
        or (route is not None and diarization_provenance.route != route)
    ):
        raise ValueError("stem reconcile stage provenance differs")
    raw = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization / "raw_segments.jsonl")
    ]
    bound = [
        BoundDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization / "bound_segments.jsonl")
    ]
    clusters = [
        DiarizationClusterBinding.model_validate(item)
        for item in _read_jsonl(diarization / "cluster_bindings.jsonl")
    ]
    asr = [
        Qwen3ASRSegment.model_validate(item)
        for item in _read_jsonl(asr_root / "segments.jsonl")
    ]
    raw_by_key = {(item.target_clip_uid, item.segment_id): item for item in raw}
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    asr_by_key = {(item.clip_uid, item.segment_id): item for item in asr}
    support_by_cluster = {
        (item.target_clip_uid, item.speaker_cluster_id): {
            support.entity_id: support.direct_support_seconds
            for support in item.entity_supports
        }
        for item in clusters
    }
    if not (set(raw_by_key) == set(bound_by_key) == set(asr_by_key)):
        raise ValueError("stem reconcile DiariZen/ASR inventories differ")
    available_ids = set(diarization_provenance.usable_clip_uids)
    if {key[0] for key in raw_by_key} - available_ids:
        raise ValueError("stem reconcile segments contain a skipped clip")
    base_by_clip = {item.clip_uid: item for item in base_inventory.jobs}
    requested_ids = [item.clip_uid for item in base_inventory.jobs]
    if len(base_by_clip) != len(requested_ids) or not set(requested_ids) <= available_ids:
        raise ValueError("stem reconcile requested clips must be unique and usable")
    projected = [
        uid for uid in diarization_provenance.usable_clip_uids if uid in base_by_clip
    ]
    if projected != requested_ids:
        raise ValueError("stem reconcile requested clips are not an ordered usable subset")
    jobs: list[MimoClipJob] = []
    for clip_uid in requested_ids:
        base = base_by_clip[clip_uid]
        keys = sorted(
            (key for key in raw_by_key if key[0] == base.clip_uid),
            key=lambda key: (
                raw_by_key[key].start_time,
                raw_by_key[key].end_time,
                key[1],
            ),
        )
        segments: list[MimoSegmentEvidence] = []
        for key in keys:
            raw_row = raw_by_key[key]
            bound_row = bound_by_key[key]
            asr_row = asr_by_key[key]
            support = support_by_cluster.get(
                (base.clip_uid, raw_row.speaker_cluster_id),
                {},
            )
            segments.append(
                MimoSegmentEvidence(
                    segment_id=raw_row.segment_id,
                    start_time=raw_row.start_time,
                    end_time=raw_row.end_time,
                    source_start_sample=raw_row.source_start_sample,
                    source_end_sample=raw_row.source_end_sample,
                    source_sample_rate_hz=raw_row.source_sample_rate_hz,
                    source_speaker_cluster_id=raw_row.speaker_cluster_id,
                    current_entity_id=bound_row.entity_id,
                    entity_occurrence_id=bound_row.entity_occurrence_id,
                    identity_scope=bound_row.identity_scope,
                    direct_anchor_seconds=bound_row.direct_anchor_seconds,
                    cluster_binding_status=bound_row.cluster_binding_status,
                    overlapping_visible_entities=sorted(support),
                    direct_support_seconds_by_entity=dict(sorted(support.items())),
                    competing_visible_speaker_evidence=[],
                    asr_status=asr_row.status,
                    asr_text=asr_row.text,
                    asr_language=asr_row.language,
                )
            )
        values = base.model_dump(mode="json", exclude={"request_fingerprint"})
        values["segments"] = [item.model_dump(mode="json") for item in segments]
        jobs.append(_mimo_job(values))
    return jobs


class MimoStemReconcileRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_reconcile.13"] = (
        MIMO25_STEM_RECONCILE_VERSION
    )
    clip_uid: str
    source_job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["h3_mimo25_stem_reconcile_v6"] = (
        MIMO25_STEM_RECONCILE_POLICY_VERSION
    )
    backend_provenance: MimoBackendProvenance
    status: Literal["ready", "failed"]
    annotation: MimoAVAnnotationDraft | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    failure_issues: list[ValidationIssue]
    raw_responses: list[StrictStr]
    visual_raw_response: StrictStr | None
    speech_av_raw_response: StrictStr | None
    audio_finalize_raw_response: StrictStr | None
    diagnostics: list[MimoCompletionDiagnostic]
    music_stem_description: StrictStr | None
    music_stem_raw_response: StrictStr | None
    music_stem_error: str | None
    sfx_stem_description: StrictStr | None
    sfx_stem_raw_response: StrictStr | None
    sfx_stem_error: str | None
    speaker_marker_polish_attempted: bool
    speaker_marker_polish_applied: bool
    speaker_marker_polish_needs_review: bool | None
    speaker_marker_polish_raw_response: str | None
    speaker_marker_polish_error: str | None
    av_model_call_count: int = Field(ge=0, le=2)
    visual_model_call_count: int = Field(ge=0, le=1)
    audio_model_call_count: int = Field(ge=0, le=2)
    text_model_call_count: int = Field(ge=0, le=1)
    model_call_count: int = Field(ge=0, le=6)
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoStemReconcileRecord:
        if self.status == "ready":
            if (
                self.annotation is None
                or self.failure_code is not None
                or self.failure_reason is not None
                or self.failure_issues
            ):
                raise ValueError("ready stem reconcile requires only annotation")
        elif not self.failure_code or not self.failure_reason:
            raise ValueError("failed stem reconcile requires failure provenance")
        if self.model_call_count != (
            self.visual_model_call_count + self.av_model_call_count
            + self.audio_model_call_count + self.text_model_call_count
        ):
            raise ValueError("stem reconcile model call counts differ")
        turn_modalities = (
            "target_video_visual_only", "target_video_speech_assembly",
            "target_video_audio_finalize",
        )
        turns = [d.input_modality for d in self.diagnostics if d.input_modality in turn_modalities]
        turn_raws = (self.visual_raw_response, self.speech_av_raw_response, self.audio_finalize_raw_response)
        if (
            turns != list(turn_modalities[:len(turns)])
            or self.visual_model_call_count != int(bool(turns))
            or self.av_model_call_count != max(0, len(turns) - 1)
            or (self.status == "ready" and (len(turns) != 3 or any(raw is None for raw in turn_raws)))
            or any(raw is not None and index >= len(turns) for index, raw in enumerate(turn_raws))
            or any(turn_raws[index] is None for index in range(max(0, len(turns) - 1)))
            or self.raw_responses != [raw for raw in turn_raws if raw is not None]
        ):
            raise ValueError("stem reconcile three-turn raw/diagnostic counts differ")
        if (
            self.text_model_call_count != int(self.speaker_marker_polish_attempted)
            or self.text_model_call_count != sum(
                diagnostic.input_modality == "speaker_marker_text_only"
                for diagnostic in self.diagnostics
            )
            or (self.speaker_marker_polish_attempted and self.av_model_call_count != 2)
            or (not self.speaker_marker_polish_attempted and (
                self.speaker_marker_polish_applied
                or self.speaker_marker_polish_needs_review is not None
                or self.speaker_marker_polish_raw_response is not None
                or self.speaker_marker_polish_error is not None
            ))
            or (self.speaker_marker_polish_applied and (
                self.speaker_marker_polish_needs_review is not False
                or self.speaker_marker_polish_raw_response is None
                or self.speaker_marker_polish_error is not None
            ))
        ):
            raise ValueError("stem reconcile speaker marker polish audit differs")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("stem reconcile record fingerprint is invalid")
        return self


class StemReconcileUpstreamFailure(SchemaModel):
    schema_version: Literal[
        "r2v.h3.mimo25_stem_reconcile_upstream_failure.1"
    ] = STEM_RECONCILE_UPSTREAM_FAILURE_VERSION
    clip_uid: str
    route: SAMRoute
    source_stage: Literal["mimo_stem_facts"] = "mimo_stem_facts"
    source_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_code: str
    reason_type: str
    reason: str


class MimoStemReconcileSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_reconcile_summary.15"] = (
        MIMO25_STEM_RECONCILE_SUMMARY_VERSION
    )
    route: SAMRoute
    clip_count: int = Field(ge=1)
    processed_clip_count: int = Field(ge=0)
    skipped_clip_count: int = Field(ge=0)
    clip_uids: list[str] = Field(min_length=1)
    processed_clip_uids: list[str]
    skipped_clips: list[StemShadowClipSkip]
    diarization_failed_clips: list[StemDiarizationClipFailure]
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    av_model_call_count: int = Field(ge=0)
    visual_model_call_count: int = Field(ge=0)
    audio_model_call_count: int = Field(ge=0)
    text_model_call_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    original_target_av_is_highest_authority: Literal[True] = True
    current_mimo_versions_modified: Literal[True] = True
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoStemReconcileSummary:
        if (
            self.model_call_count != (
                self.visual_model_call_count + self.av_model_call_count
                + self.audio_model_call_count + self.text_model_call_count
            )
            or self.av_model_call_count > 2 * self.visual_model_call_count
            or self.visual_model_call_count > self.processed_clip_count
            or self.text_model_call_count > self.av_model_call_count // 2
            or self.processed_clip_count != self.ready_count + self.failed_count
            or self.clip_count != self.processed_clip_count + self.skipped_clip_count
            or self.clip_count != len(self.clip_uids)
            or self.processed_clip_count != len(self.processed_clip_uids)
            or self.skipped_clip_count
            != len(self.skipped_clips)
            + len(self.diarization_failed_clips)
            or [item for item in self.clip_uids if item in set(self.processed_clip_uids)]
            != self.processed_clip_uids
            or [
                item
                for item in self.clip_uids
                if item
                in {failure.clip_uid for failure in self.diarization_failed_clips}
            ]
            != [item.clip_uid for item in self.diarization_failed_clips]
            or set(self.processed_clip_uids)
            .union(item.clip_uid for item in self.skipped_clips)
            .union(item.clip_uid for item in self.diarization_failed_clips)
            != set(self.clip_uids)
            or any(item.route != self.route for item in self.skipped_clips)
            or any(
                item.route != self.route for item in self.diarization_failed_clips
            )
        ):
            raise ValueError("stem reconcile summary counts do not reconcile")
        return self


class StemReconcileBackend(Protocol):
    @property
    def provenance(self) -> MimoBackendProvenance: ...

    def reconcile(
        self,
        job: MimoClipJob,
        *,
        segment_ids: list[str],
        transcribed_segment_ids: list[str],
        allowed_entity_ids: set[str],
        allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> MimoBackendResult: ...


def _validated_auxiliary_stems(record: SAMAudioStemRecord) -> dict[str, Path]:
    paths = {}
    try:
        for kind in ("speech", "music", "sfx"):
            stem = record.stem(kind)
            path = Path(stem.canonical_stem_path)
            if sha256_file(path) != stem.canonical_stem_sha256:
                raise ValueError(f"{kind} auxiliary stem changed")
            paths[kind] = path
    except (OSError, ValueError) as exc:
        raise MimoBackendFailure(
            code="mimo_stem_media_invalid", reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    return paths


def run_mimo25_stem_reconcile_shadow(
    *,
    jobs: Sequence[MimoClipJob],
    stem_records: Sequence[SAMAudioStemRecord],
    backend: StemReconcileBackend,
    output_root: Path,
    source_clip_uids: Sequence[str] | None = None,
    skipped_clips: Sequence[StemShadowClipSkip] = (),
    diarization_failed_clips: Sequence[StemDiarizationClipFailure] = (),
    route: SAMRoute,
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> MimoStemReconcileSummary:
    job_ids = [job.clip_uid for job in jobs]
    selected = selected_stem_records(
        [record for record in stem_records if record.clip_uid in set(job_ids)],
        route=route,
        allow_unverified=allow_unverified,
    )
    stems_by_clip = {record.clip_uid: record for record in selected}
    if set(stems_by_clip) != set(job_ids) or len(set(job_ids)) != len(job_ids):
        raise ValueError("stem reconcile jobs and usable separation records differ")
    ordered_source = list(source_clip_uids or job_ids)
    if (
        len(ordered_source) != len(set(ordered_source))
        or [clip for clip in ordered_source if clip in set(job_ids)] != job_ids
    ):
        raise ValueError("stem reconcile jobs differ from source clip order")
    skipped_ids = [
        item.clip_uid for item in [*skipped_clips, *diarization_failed_clips]
    ]
    if (
        len(skipped_ids) != len(set(skipped_ids))
        or set(skipped_ids) != set(ordered_source) - set(job_ids)
        or any(
            item.route != route for item in [*skipped_clips, *diarization_failed_clips]
        )
    ):
        raise ValueError("stem reconcile skipped clips differ from source inventory")
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    records: list[MimoStemReconcileRecord] = []
    try:
        temporary.mkdir(parents=True)
        for job in jobs:
            audio_calls = 0
            annotation = None
            failure_code = failure_reason = None
            issues: list[ValidationIssue] = []
            try:
                result = backend.reconcile(
                    job,
                    auxiliary_audio_paths=_validated_auxiliary_stems(stems_by_clip[job.clip_uid]),
                    segment_ids=[s.segment_id for s in job.segments],
                    transcribed_segment_ids=[
                        s.segment_id
                        for s in job.segments
                        if s.asr_status == "transcribed"
                    ],
                    allowed_entity_ids={
                        r.entity_id
                        for r in job.reference_images
                        if r.kind == "subject" and r.entity_id is not None
                    },
                    allowed_reference_labels={
                        r.picture_label for r in job.reference_images
                    }
                    | {s.subject_label for s in job.reference_subjects},
                )
                annotation = result.annotation
                raw = list(result.raw_responses)
                diagnostics = list(result.diagnostics)
                text_calls = getattr(result, "text_model_call_count", 0)
                visual_calls = result.visual_model_call_count
                visual_raw = result.visual_raw_response
                speech_av_raw = result.speech_av_raw_response
                audio_finalize_raw = result.audio_finalize_raw_response
                polish = getattr(result, "speaker_marker_polish", MimoSpeakerMarkerPolishAudit())
                av_calls = result.model_call_count - text_calls - visual_calls
            except MimoBackendFailure as exc:
                annotation = exc.annotation
                failure_code, failure_reason = exc.code, exc.reason
                issues = list(exc.issues)
                raw, diagnostics, av_calls = (
                    list(exc.raw_responses),
                    list(exc.diagnostics),
                    exc.model_call_count - exc.text_model_call_count - exc.visual_model_call_count,
                )
                text_calls = exc.text_model_call_count
                visual_calls = exc.visual_model_call_count
                visual_raw = exc.visual_raw_response
                speech_av_raw = exc.speech_av_raw_response
                audio_finalize_raw = exc.audio_finalize_raw_response
                polish = exc.speaker_marker_polish
            values = {
                "schema_version": MIMO25_STEM_RECONCILE_VERSION,
                "clip_uid": job.clip_uid,
                "source_job_fingerprint": job.request_fingerprint,
                "source_stem_record_fingerprint": stems_by_clip[
                    job.clip_uid
                ].record_fingerprint,
                "policy_version": MIMO25_STEM_RECONCILE_POLICY_VERSION,
                "backend_provenance": backend.provenance.model_dump(mode="json"),
                "status": "ready"
                if failure_code is None and annotation is not None
                else "failed",
                "annotation": annotation.model_dump(mode="json")
                if annotation
                else None,
                "failure_code": failure_code,
                "failure_reason": failure_reason,
                "failure_issues": [issue.to_dict() for issue in issues],
                "raw_responses": raw,
                "visual_raw_response": visual_raw,
                "speech_av_raw_response": speech_av_raw,
                "audio_finalize_raw_response": audio_finalize_raw,
                "diagnostics": [d.model_dump(mode="json") for d in diagnostics],
                "music_stem_description": None,
                "music_stem_raw_response": None,
                "music_stem_error": None,
                "sfx_stem_description": None,
                "sfx_stem_raw_response": None,
                "sfx_stem_error": None,
                **{f"speaker_marker_polish_{key}": value for key, value in polish.model_dump(mode="json").items()},
                "av_model_call_count": av_calls,
                "visual_model_call_count": visual_calls,
                "audio_model_call_count": audio_calls,
                "text_model_call_count": text_calls,
                "model_call_count": visual_calls + av_calls + audio_calls + text_calls,
            }
            records.append(
                MimoStemReconcileRecord(
                    **values,
                    record_fingerprint=_sha256_text(_compact_json(values)),
                )
            )
        counts = Counter(record.status for record in records)
        summary = MimoStemReconcileSummary(
            route=route,
            clip_count=len(ordered_source),
            processed_clip_count=len(records),
            skipped_clip_count=len(skipped_ids),
            clip_uids=ordered_source,
            processed_clip_uids=job_ids,
            skipped_clips=list(skipped_clips),
            diarization_failed_clips=list(diarization_failed_clips),
            ready_count=counts["ready"],
            failed_count=counts["failed"],
            av_model_call_count=sum(r.av_model_call_count for r in records),
            visual_model_call_count=sum(r.visual_model_call_count for r in records),
            audio_model_call_count=sum(r.audio_model_call_count for r in records),
            text_model_call_count=sum(r.text_model_call_count for r in records),
            model_call_count=sum(r.model_call_count for r in records),
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "skipped_clips.jsonl", list(skipped_clips))
        _write_jsonl(
            temporary / "diarization_failed_clips.jsonl", list(diarization_failed_clips)
        )
        _write_json(temporary / "summary.json", summary)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "MIMO25_STEM_FACT_PROMPT_VERSION",
    "MIMO25_STEM_RECONCILE_POLICY_VERSION",
    "MIMO25_STEM_RECONCILE_STAGE",
    "FFmpegStemViewBackend",
    "MimoStemFactsRecord",
    "MimoStemFactsSummary",
    "MimoStemReconcileRecord",
    "MimoStemReconcileSummary",
    "MusicInterval",
    "MusicStemFacts",
    "OpenAIStemFactsBackend",
    "SFXContinuousLayer",
    "SFXEvent",
    "SFXStemFacts",
    "SpeechStemFacts",
    "SpeechStemSegmentFact",
    "StemAwareOpenAIMimo25Backend",
    "StemFactCallAudit",
    "StemFactJob",
    "StemFactSegment",
    "StemFactsBackend",
    "StemFactsBackendFailure",
    "StemFactsBackendProvenance",
    "StemFactsBackendResult",
    "StemReconcileUpstreamFailure",
    "StemView",
    "StemViewBackend",
    "build_stem_reconcile_jobs",
    "load_stem_fact_records",
    "run_mimo25_stem_facts_shadow",
    "run_mimo25_stem_reconcile_shadow",
    "stem_reconcile_auxiliary_contract",
    "validate_stem_facts_lineage",
]
