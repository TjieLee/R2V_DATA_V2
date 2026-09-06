from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoInventory,
    MimoSegmentEvidence,
)
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendProvenance,
    MimoBackendResult,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    OpenAIMimo25Backend,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemInventory,
    SAMAudioStemRecord,
    SAMRoute,
    StemASRShadowProvenance,
    StemDiarizationShadowProvenance,
    StemShadowClipSkip,
    StemType,
    load_stem_shadow,
    selected_stem_records,
    separation_skips,
    sha256_file,
)
from r2v_data_v2.h3.schemas import SchemaModel

MIMO25_STEM_FACTS_VERSION = "r2v.h3.mimo25_stem_facts.2"
MIMO25_STEM_FACTS_SUMMARY_VERSION = "r2v.h3.mimo25_stem_facts_summary.2"
MIMO25_STEM_FACT_PROMPT_VERSION = "h3_mimo25_stem_fact_prompt_v1"
MIMO25_STEM_RECONCILE_VERSION = "r2v.h3.mimo25_stem_reconcile.1"
MIMO25_STEM_RECONCILE_SUMMARY_VERSION = "r2v.h3.mimo25_stem_reconcile_summary.2"
MIMO25_STEM_RECONCILE_POLICY_VERSION = "h3_mimo25_stem_reconcile_v1"
STEM_VIEW_VERSION = "r2v.h3.sam_audio_stem_view.1"

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


def _write_json(path: Path, value: SchemaModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


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
    segments: list[StemFactSegment]
    job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_job(self) -> StemFactJob:
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
    ) -> SpeechStemFacts | MusicStemFacts | SFXStemFacts: ...


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
    ) -> SpeechStemFacts | MusicStemFacts | SFXStemFacts:
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
        if not isinstance(raw, str):
            raise TypeError("stem fact response content must be text JSON")
        return schema.model_validate_json(raw)


class MimoStemFactsRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_facts.2"] = (
        MIMO25_STEM_FACTS_VERSION
    )
    clip_uid: str
    route: SAMRoute
    job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_verification_state: Literal["success", "uncertain", "unverified"]
    unverified_consumption_authorized: bool
    backend_provenance: StemFactsBackendProvenance
    stem_views: list[StemView]
    speech: SpeechStemFacts
    music: MusicStemFacts
    sfx: SFXStemFacts
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoStemFactsRecord:
        if [item.stem_type for item in self.stem_views] != ["speech", "music", "sfx"]:
            raise ValueError("stem fact record requires three ordered AV views")
        if (
            self.source_verification_state == "unverified"
            and not self.unverified_consumption_authorized
        ):
            raise ValueError("unverified stem facts lack explicit authorization")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("stem fact record fingerprint is invalid")
        return self


class MimoStemFactsSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_facts_summary.2"] = (
        MIMO25_STEM_FACTS_SUMMARY_VERSION
    )
    record_count: int = Field(ge=1)
    source_stem_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    route: SAMRoute
    source_clip_uids: list[str] = Field(min_length=1)
    clip_uids: list[str] = Field(min_length=1)
    skipped_clips: list[StemShadowClipSkip]
    unverified_clip_uids: list[str]
    unverified_consumption_authorized: bool
    model_call_count: int = Field(ge=0)
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoStemFactsSummary:
        if (
            self.record_count != len(self.clip_uids)
            or len(self.source_clip_uids) != len(set(self.source_clip_uids))
            or [
                item
                for item in self.source_clip_uids
                if item in set(self.clip_uids)
            ]
            != self.clip_uids
            or {item.clip_uid for item in self.skipped_clips}
            != set(self.source_clip_uids) - set(self.clip_uids)
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
    selected = selected_stem_records(
        stem_records,
        route=route,
        allow_unverified=allow_unverified,
    )
    selected_by_clip = {item.clip_uid: item for item in selected}
    selected = [
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
            job = _stem_fact_job(
                stem_inventory=stem_inventory,
                record=record,
                segments=raw,
                asr=asr,
            )
            source = next(
                item for item in stem_inventory.jobs if item.clip_uid == record.clip_uid
            )
            views = [
                view_backend.create(
                    clip_uid=record.clip_uid,
                    route=route,
                    stem_type=stem_type,
                    original_video_path=Path(source.target_video_path),
                    source_stem_path=Path(record.stem(stem_type).canonical_stem_path),
                    destination=(
                        views_root / record.clip_uid / route / f"{stem_type}.mp4"
                    ),
                )
                for stem_type in ("speech", "music", "sfx")
            ]
            for view in views:
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
            facts = {
                view.stem_type: backend.extract(
                    job=job,
                    stem_type=view.stem_type,
                    view=view,
                )
                for view in views
            }
            speech = facts["speech"]
            music = facts["music"]
            sfx = facts["sfx"]
            if not isinstance(speech, SpeechStemFacts):
                raise TypeError("speech stem backend returned the wrong schema")
            if not isinstance(music, MusicStemFacts):
                raise TypeError("music stem backend returned the wrong schema")
            if not isinstance(sfx, SFXStemFacts):
                raise TypeError("SFX stem backend returned the wrong schema")
            _validate_speech_fact_inventory(speech, job)
            if (
                music.clip_duration_seconds != job.clip_duration_seconds
                or sfx.clip_duration_seconds != job.clip_duration_seconds
            ):
                raise ValueError("stem fact duration differs from source clip")
            values = {
                "schema_version": MIMO25_STEM_FACTS_VERSION,
                "clip_uid": record.clip_uid,
                "route": route,
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
            }
            records.append(
                MimoStemFactsRecord(
                    **values,
                    record_fingerprint=_sha256_text(_compact_json(values)),
                )
            )
        summary = MimoStemFactsSummary(
            record_count=len(records),
            source_stem_inventory_fingerprint=stem_inventory.inventory_fingerprint,
            route=route,
            source_clip_uids=stem_inventory.clip_uids,
            clip_uids=[item.clip_uid for item in records],
            skipped_clips=skips,
            unverified_clip_uids=[
                item.clip_uid
                for item in selected
                if item.separation_state == "unverified"
            ],
            unverified_consumption_authorized=allow_unverified,
            model_call_count=len(records) * 3,
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "skipped_clips.jsonl", skips)
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


def stem_reconcile_auxiliary_contract(facts: MimoStemFactsRecord) -> dict[str, object]:
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
            "Music-stem labels do not decide diegetic versus non-diegetic music.",
            "Preserve verified music onset and offset naturally in the existing H3 section.",
            "Do not duplicate music across detailed description, soundscape, and music.",
        ],
        "stem_facts": facts.model_dump(mode="json"),
    }


class StemAwareOpenAIMimo25Backend(OpenAIMimo25Backend):
    def __init__(
        self,
        config: MimoBackendConfig,
        *,
        stem_facts_by_clip: dict[str, MimoStemFactsRecord],
        client: Any | None = None,
    ) -> None:
        super().__init__(config, client=client)
        self.stem_facts_by_clip = stem_facts_by_clip

    def _auxiliary_text(self, job: MimoClipJob) -> str:
        facts = self.stem_facts_by_clip.get(job.clip_uid)
        if facts is None:
            raise ValueError("stem-aware MiMo job lacks stem facts")
        return "\nSTEM AUXILIARY EVIDENCE:\n" + _compact_json(
            stem_reconcile_auxiliary_contract(facts)
        )

    def _prompt(self, job: MimoClipJob) -> str:
        return super()._prompt(job) + self._auxiliary_text(job)

    def _full_av_recheck_prompt(
        self,
        job: MimoClipJob,
        *,
        invalid_response: str,
        issues: list[Any],
    ) -> str:
        return super()._full_av_recheck_prompt(
            job,
            invalid_response=invalid_response,
            issues=issues,
        ) + self._auxiliary_text(job)


def _mimo_job(values: dict[str, object]) -> MimoClipJob:
    return MimoClipJob(
        **values,
        request_fingerprint=_sha256_text(_compact_json(values)),
    )


def build_stem_reconcile_jobs(
    *,
    base_inventory: MimoInventory,
    stem_diarization_root: Path,
    stem_asr_root: Path,
) -> list[MimoClipJob]:
    diarization = stem_diarization_root.expanduser().resolve(strict=True)
    asr_root = stem_asr_root.expanduser().resolve(strict=True)
    diarization_provenance = StemDiarizationShadowProvenance.model_validate_json(
        (diarization / "stem_provenance.json").read_text(encoding="utf-8")
    )
    asr_provenance = StemASRShadowProvenance.model_validate_json(
        (asr_root / "stem_provenance.json").read_text(encoding="utf-8")
    )
    if (
        asr_provenance.source_clip_uids != diarization_provenance.clip_uids
        or asr_provenance.clip_uids != diarization_provenance.usable_clip_uids
        or asr_provenance.source_stem_inventory_fingerprint
        != diarization_provenance.source_stem_inventory_fingerprint
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
    if len(base_by_clip) != len(base_inventory.jobs) or not available_ids <= set(
        base_by_clip
    ):
        raise ValueError("stem reconcile base inventory lacks usable clips")
    jobs: list[MimoClipJob] = []
    for clip_uid in diarization_provenance.usable_clip_uids:
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
    schema_version: Literal["r2v.h3.mimo25_stem_reconcile.1"] = (
        MIMO25_STEM_RECONCILE_VERSION
    )
    clip_uid: str
    source_job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_facts_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["h3_mimo25_stem_reconcile_v1"] = (
        MIMO25_STEM_RECONCILE_POLICY_VERSION
    )
    backend_provenance: MimoBackendProvenance
    status: Literal["ready", "failed"]
    annotation: MimoAVAnnotationDraft | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    diagnostics: list[MimoCompletionDiagnostic]
    model_call_count: int = Field(ge=0)
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoStemReconcileRecord:
        if self.status == "ready":
            if self.annotation is None or self.failure_code is not None or self.failure_reason is not None:
                raise ValueError("ready stem reconcile requires only annotation")
        elif self.annotation is not None or not self.failure_code or not self.failure_reason:
            raise ValueError("failed stem reconcile requires failure provenance")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("stem reconcile record fingerprint is invalid")
        return self


class MimoStemReconcileSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_stem_reconcile_summary.2"] = (
        MIMO25_STEM_RECONCILE_SUMMARY_VERSION
    )
    clip_count: int = Field(ge=1)
    processed_clip_count: int = Field(ge=0)
    skipped_clip_count: int = Field(ge=0)
    clip_uids: list[str] = Field(min_length=1)
    processed_clip_uids: list[str]
    skipped_clips: list[StemShadowClipSkip]
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    original_target_av_is_highest_authority: Literal[True] = True
    current_mimo_versions_modified: Literal[False] = False
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoStemReconcileSummary:
        if (
            self.processed_clip_count != self.ready_count + self.failed_count
            or self.clip_count != self.processed_clip_count + self.skipped_clip_count
            or self.clip_count != len(self.clip_uids)
            or self.processed_clip_count != len(self.processed_clip_uids)
            or self.skipped_clip_count != len(self.skipped_clips)
            or [item for item in self.clip_uids if item in set(self.processed_clip_uids)]
            != self.processed_clip_uids
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
    ) -> MimoBackendResult: ...


def run_mimo25_stem_reconcile_shadow(
    *,
    jobs: Sequence[MimoClipJob],
    stem_facts: Sequence[MimoStemFactsRecord],
    backend: StemReconcileBackend,
    output_root: Path,
    source_clip_uids: Sequence[str] | None = None,
    skipped_clips: Sequence[StemShadowClipSkip] = (),
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> MimoStemReconcileSummary:
    facts_by_clip = {item.clip_uid: item for item in stem_facts}
    if len(facts_by_clip) != len(stem_facts) or set(facts_by_clip) != {
        item.clip_uid for item in jobs
    }:
        raise ValueError("stem reconcile jobs and facts differ")
    if any(
        item.source_verification_state == "unverified" for item in stem_facts
    ) and not allow_unverified:
        raise ValueError(
            "unverified SAM Audio stems require explicit allow_unverified opt-in"
        )
    ordered_source = list(source_clip_uids or [item.clip_uid for item in jobs])
    if len(ordered_source) != len(set(ordered_source)):
        raise ValueError("stem reconcile source clip order contains duplicates")
    processed_ids = [item.clip_uid for item in jobs]
    if [item for item in ordered_source if item in set(processed_ids)] != processed_ids:
        raise ValueError("stem reconcile jobs differ from source clip order")
    if {item.clip_uid for item in skipped_clips} != set(ordered_source) - set(
        processed_ids
    ):
        raise ValueError("stem reconcile skipped clips differ from source inventory")
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    records: list[MimoStemReconcileRecord] = []
    try:
        temporary.mkdir(parents=True)
        for job in jobs:
            facts = facts_by_clip[job.clip_uid]
            segments = [item.segment_id for item in job.segments]
            transcribed = [
                item.segment_id for item in job.segments if item.asr_status == "transcribed"
            ]
            entities = {
                item.entity_id
                for item in job.reference_images
                if item.kind == "subject" and item.entity_id is not None
            }
            labels = {item.picture_label for item in job.reference_images} | {
                item.subject_label for item in job.reference_subjects
            }
            try:
                result = backend.reconcile(
                    job,
                    segment_ids=segments,
                    transcribed_segment_ids=transcribed,
                    allowed_entity_ids=entities,
                    allowed_reference_labels=labels,
                )
                values = {
                    "schema_version": MIMO25_STEM_RECONCILE_VERSION,
                    "clip_uid": job.clip_uid,
                    "source_job_fingerprint": job.request_fingerprint,
                    "source_stem_facts_fingerprint": facts.record_fingerprint,
                    "policy_version": MIMO25_STEM_RECONCILE_POLICY_VERSION,
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "status": "ready",
                    "annotation": result.annotation.model_dump(mode="json"),
                    "failure_code": None,
                    "failure_reason": None,
                    "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics],
                    "model_call_count": result.model_call_count,
                }
            except MimoBackendFailure as exc:
                values = {
                    "schema_version": MIMO25_STEM_RECONCILE_VERSION,
                    "clip_uid": job.clip_uid,
                    "source_job_fingerprint": job.request_fingerprint,
                    "source_stem_facts_fingerprint": facts.record_fingerprint,
                    "policy_version": MIMO25_STEM_RECONCILE_POLICY_VERSION,
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "status": "failed",
                    "annotation": None,
                    "failure_code": exc.code,
                    "failure_reason": exc.reason,
                    "diagnostics": [item.model_dump(mode="json") for item in exc.diagnostics],
                    "model_call_count": exc.model_call_count,
                }
            records.append(
                MimoStemReconcileRecord(
                    **values,
                    record_fingerprint=_sha256_text(_compact_json(values)),
                )
            )
        counts = Counter(item.status for item in records)
        summary = MimoStemReconcileSummary(
            clip_count=len(ordered_source),
            processed_clip_count=len(records),
            skipped_clip_count=len(skipped_clips),
            clip_uids=ordered_source,
            processed_clip_uids=processed_ids,
            skipped_clips=list(skipped_clips),
            ready_count=counts["ready"],
            failed_count=counts["failed"],
            model_call_count=sum(item.model_call_count for item in records),
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "skipped_clips.jsonl", list(skipped_clips))
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
    "StemFactJob",
    "StemFactSegment",
    "StemFactsBackend",
    "StemFactsBackendProvenance",
    "StemView",
    "StemViewBackend",
    "build_stem_reconcile_jobs",
    "load_stem_fact_records",
    "run_mimo25_stem_facts_shadow",
    "run_mimo25_stem_reconcile_shadow",
    "stem_reconcile_auxiliary_contract",
]
