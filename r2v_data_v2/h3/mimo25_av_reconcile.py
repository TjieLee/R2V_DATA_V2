from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from r2v_data_v2.h3.binding_audit import SpeakerBindingSegmentAudit
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationInventory,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    jea_production_paths,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MODEL,
    MimoAVAnnotationDraft,
    MimoBackendFailure,
    MimoBackendProvenance,
    MimoBackendResult,
    sha256_file,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionSubjectContract,
    build_reference_contract,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory

MIMO25_INVENTORY_VERSION = "r2v.h3.mimo25_inventory.3"
MIMO25_RECORD_VERSION = "r2v.h3.mimo25_record.5"
MIMO25_SUMMARY_VERSION = "r2v.h3.mimo25_summary.5"
MIMO25_FAILURE_VERSION = "r2v.h3.mimo25_failure.5"
MIMO25_RAW_VERSION = "r2v.h3.mimo25_raw_response.5"
MIMO25_CASE_MANIFEST_VERSION = "r2v.h3.mimo25_case_manifest.1"
MIMO25_INVENTORY_SCOPE = "canonical_visual_target_inventory"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} must be an object: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(value.model_dump(mode="json")) + "\n" for value in values),
        encoding="utf-8",
    )


class MimoReferenceImage(SchemaModel):
    image_index: int = Field(gt=0)
    picture_label: str = Field(pattern=r"^<Picture [1-9]\d*>$")
    kind: Literal["subject", "object", "group", "background", "attribute"]
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None
    image_artifact_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_image(self) -> MimoReferenceImage:
        if self.picture_label != f"<Picture {self.image_index}>":
            raise ValueError("MiMo Picture label must match image index")
        if not Path(self.image_artifact_path).is_absolute():
            raise ValueError("MiMo reference artifact path must be absolute")
        if self.kind == "attribute":
            if any(
                value is None
                for value in (self.attribute_id, self.owner_entity_id, self.attribute_type)
            ) or self.entity_id is not None:
                raise ValueError("MiMo attribute reference requires owner provenance")
        elif self.kind == "background":
            if any(
                value is not None
                for value in (
                    self.entity_id,
                    self.attribute_id,
                    self.owner_entity_id,
                    self.attribute_type,
                )
            ):
                raise ValueError("MiMo background reference cannot claim ownership")
        elif self.entity_id is None or any(
            value is not None
            for value in (self.attribute_id, self.owner_entity_id, self.attribute_type)
        ):
            raise ValueError("MiMo entity reference requires only entity_id")
        return self


class MimoSegmentEvidence(SchemaModel):
    segment_id: str
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    source_speaker_cluster_id: str
    current_entity_id: str | None = None
    entity_occurrence_id: str | None = None
    identity_scope: Literal[
        "direct_anchor_present", "cluster_propagated_only", "unresolved"
    ]
    direct_anchor_seconds: float = Field(ge=0, allow_inf_nan=False)
    cluster_binding_status: Literal[
        "candidate_mapped", "unbound", "ambiguous", "conflict"
    ]
    overlapping_visible_entities: list[str]
    direct_support_seconds_by_entity: dict[str, float]
    competing_visible_speaker_evidence: list[str]
    asr_status: Literal["transcribed", "empty", "failed"]
    asr_text: str | None = None
    asr_language: str | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> MimoSegmentEvidence:
        if self.end_time <= self.start_time or self.source_end_sample <= self.source_start_sample:
            raise ValueError("MiMo segment evidence interval must be positive")
        if self.asr_status == "transcribed":
            if self.asr_text is None or not self.asr_text.strip():
                raise ValueError("transcribed MiMo evidence requires authoritative text")
        elif self.asr_text is not None or self.asr_language is not None:
            raise ValueError("non-transcribed MiMo evidence cannot publish ASR text")
        return self


class MimoClipJob(SchemaModel):
    clip_uid: str
    r2v_instruction: str = Field(min_length=1)
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    reference_images: list[MimoReferenceImage] = Field(min_length=1)
    reference_subjects: list[RecaptionSubjectContract] = Field(min_length=1)
    segments: list[MimoSegmentEvidence]
    source_h3_sample_ids: list[str] = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_job(self) -> MimoClipJob:
        if not self.r2v_instruction.strip():
            raise ValueError("MiMo R2V instruction must not be blank")
        indexes = [item.image_index for item in self.reference_images]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("MiMo reference images must preserve canonical order")
        segment_ids = [item.segment_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("MiMo job segments must be unique")
        order = [(item.start_time, item.end_time, item.segment_id) for item in self.segments]
        if order != sorted(order):
            raise ValueError("MiMo job segments must be chronological")
        values = self.model_dump(mode="json", exclude={"request_fingerprint"})
        if self.request_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo job fingerprint is invalid")
        return self


class MimoCaseManifest(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_case_manifest.1"] = (
        MIMO25_CASE_MANIFEST_VERSION
    )
    clip_uids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> MimoCaseManifest:
        if len(self.clip_uids) != len(set(self.clip_uids)):
            raise ValueError("MiMo case manifest clip IDs must be unique")
        return self


class MimoInventory(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_inventory.3"] = MIMO25_INVENTORY_VERSION
    inventory_scope: Literal[
        "current_diarization_asr_target_inventory",
        "canonical_visual_target_inventory",
        "explicit_case_subset",
    ] = MIMO25_INVENTORY_SCOPE
    canonical_wide_coverage: bool = True
    source_visual_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_canonical_audio_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_diarization_raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_bound_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_qwen3_asr_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_audit_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_count: int = Field(ge=0)
    jobs: list[MimoClipJob]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> MimoInventory:
        if self.clip_count != len(self.jobs):
            raise ValueError("MiMo inventory clip count is inconsistent")
        clip_ids = [job.clip_uid for job in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("MiMo inventory clip IDs must be unique")
        values = self.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if values["source_diarization_inventory_sha256"] is None:
            values.pop("source_diarization_inventory_sha256")
        if self.inventory_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo inventory fingerprint is invalid")
        return self


class MimoFailure(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_failure.5"] = MIMO25_FAILURE_VERSION
    clip_uid: str
    code: str
    reason: str
    attempt_count: int = Field(ge=0)
    http_attempt_count: int = Field(ge=0)
    http_retry_count: int = Field(ge=0)
    recheck_count: int = Field(ge=0, le=1)
    issues: list[dict[str, str | None]]


class MimoRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_record.5"] = MIMO25_RECORD_VERSION
    clip_uid: str
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["ready", "failed", "unsupported"]
    backend_provenance: MimoBackendProvenance
    annotation: MimoAVAnnotationDraft | None = None
    failure: MimoFailure | None = None
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
    ] | None = None
    model_call_count: int = Field(ge=0)
    http_attempt_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    http_retry_count: int = Field(ge=0)
    recheck_count: int = Field(ge=0, le=1)
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoRecord:
        if self.status == "ready":
            if self.annotation is None or self.failure is not None or self.input_modality is None:
                raise ValueError("ready MiMo record requires only annotation")
        elif self.annotation is not None or self.failure is None:
            raise ValueError("non-ready MiMo record requires only failure")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo record fingerprint is invalid")
        return self


class MimoRawResponse(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_raw_response.5"] = MIMO25_RAW_VERSION
    clip_uid: str
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_responses: list[str]
    diagnostics: list[dict[str, object]]


class MimoSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_summary.5"] = MIMO25_SUMMARY_VERSION
    inventory_scope: Literal[
        "current_diarization_asr_target_inventory",
        "canonical_visual_target_inventory",
        "explicit_case_subset",
    ] = MIMO25_INVENTORY_SCOPE
    canonical_wide_coverage: bool = True
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_hashes: dict[str, str]
    model: Literal["mimo-v2.5"] = MIMO25_MODEL
    fps: Literal[4.0] = 4.0
    media_resolution: Literal["default"] = "default"
    clip_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    http_attempt_count: int = Field(ge=0)
    http_retry_count: int = Field(ge=0)
    recheck_count: int = Field(ge=0)
    input_modality_counts: dict[str, int]
    usage_totals: dict[str, int]
    diagnostic_warning_counts: dict[str, int]
    responses_with_nonzero_reasoning_tokens: int = Field(ge=0)
    correction_counts: dict[str, int]
    audio_event_count: int = Field(ge=0)
    music_status_counts: dict[str, int]
    production_binding_modified: Literal[False] = False
    production_diarization_modified: Literal[False] = False
    production_asr_modified: Literal[False] = False
    production_h3_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoSummary:
        if self.clip_count != self.ready_count + self.failed_count + self.unsupported_count:
            raise ValueError("MiMo summary counts must reconcile")
        return self


class MimoBackend(Protocol):
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


def _job(values: dict[str, object]) -> MimoClipJob:
    return MimoClipJob(
        **values,
        request_fingerprint=_sha256_text(_compact_json(values)),
    )


def _inventory(values: dict[str, object]) -> MimoInventory:
    return MimoInventory(
        **values,
        inventory_fingerprint=_sha256_text(_compact_json(values)),
    )


def _record(values: dict[str, object]) -> MimoRecord:
    return MimoRecord(
        **values,
        record_fingerprint=_sha256_text(_compact_json(values)),
    )


def _index(rows: Sequence[SchemaModel], *fields: str) -> dict[tuple[object, ...], SchemaModel]:
    result: dict[tuple[object, ...], SchemaModel] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in fields)
        if key in result:
            raise ValueError(f"duplicate source row: {key}")
        result[key] = row
    return result


def _validate_h3_variant_observations(
    clip_uid: str,
    samples: Sequence[FinalH3SampleV2],
) -> FinalH3SampleV2:
    if not samples:
        raise ValueError(f"MiMo H3 sample inventory is empty: {clip_uid}")
    canonical = [item for item in samples if item.pair_type == "canonical"]
    if len(canonical) != 1:
        raise ValueError(f"MiMo requires one canonical H3 base sample: {clip_uid}")
    representative = canonical[0]
    if any(
        item.target_video != representative.target_video
        or item.target_full_audio_path != representative.target_full_audio_path
        or item.visual_references != representative.visual_references
        or item.r2v_instruction != representative.r2v_instruction
        or item.full_clip_audio_semantics != representative.full_clip_audio_semantics
        for item in samples
        if item is not representative
    ):
        raise ValueError(f"MiMo H3 variants disagree on target observations: {clip_uid}")
    return representative


def _validate_clip_segment_inventory(
    clip_uid: str,
    *,
    raw: Sequence[RawDiarizationSegment],
    bound: Sequence[BoundDiarizationSegment],
    asr: Sequence[Qwen3ASRSegment],
    audits: Sequence[SpeakerBindingSegmentAudit],
) -> None:
    inventories = {
        "raw": {item.segment_id for item in raw if item.target_clip_uid == clip_uid},
        "bound": {
            item.segment_id for item in bound if item.target_clip_uid == clip_uid
        },
        "audit": {item.segment_id for item in audits if item.clip_uid == clip_uid},
        "asr": {item.segment_id for item in asr if item.clip_uid == clip_uid},
    }
    expected = inventories["raw"]
    if any(value != expected for value in inventories.values()):
        raise ValueError(
            f"MiMo segment inventories differ for {clip_uid}: "
            + _compact_json({key: sorted(value) for key, value in inventories.items()})
        )


def build_mimo25_inventory(
    *,
    visual_production_root: Path,
    visual_runs_root: Path,
    audio_production_root: Path,
    case_manifest_path: Path | None = None,
    case_manifest: MimoCaseManifest | None = None,
) -> MimoInventory:
    if case_manifest_path is not None and case_manifest is not None:
        raise ValueError("provide only one MiMo case manifest source")
    paths = jea_production_paths(audio_production_root)
    visual = load_visual_production_inventory(
        visual_production_root=visual_production_root,
        visual_runs_root=visual_runs_root,
    )
    visual_by_clip = {item.identity.clip_uid: item for item in visual.canonical_clips}
    source_paths = {
        "visual_inventory": visual_production_root.resolve(strict=True) / "samples.jsonl",
        "canonical_audio_manifest": paths.audio / "canonical_clips.jsonl",
        "diarization_inventory": paths.diarization / "inventory.json",
        "diarization_raw_segments": paths.diarization / "raw_segments.jsonl",
        "diarization_bound_segments": paths.diarization / "bound_segments.jsonl",
        "qwen3_asr_segments": paths.asr / "segments.jsonl",
        "binding_audit_segments": paths.root / "binding_audit_v1" / "segments.jsonl",
        "h3_samples": paths.h3 / "samples.jsonl",
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"MiMo source {name} is missing: {path}")
    canonical = [
        CanonicalAudioClip.model_validate(row)
        for row in _read_jsonl(source_paths["canonical_audio_manifest"])
    ]
    diarization_inventory = DiarizationInventory.model_validate_json(
        source_paths["diarization_inventory"].read_text(encoding="utf-8")
    )
    if (
        diarization_inventory.source_inventory_kind != "canonical_audio_manifest"
        or diarization_inventory.selection_mode
        != "canonical_visual_target_inventory_v1"
        or diarization_inventory.source_canonical_audio_manifest_path is None
        or Path(
            diarization_inventory.source_canonical_audio_manifest_path
        ).resolve(strict=True)
        != source_paths["canonical_audio_manifest"].resolve(strict=True)
        or diarization_inventory.source_canonical_audio_manifest_sha256
        != sha256_file(source_paths["canonical_audio_manifest"])
    ):
        raise ValueError("MiMo DiariZen inventory is not canonical-rooted")
    raw = [
        RawDiarizationSegment.model_validate(row)
        for row in _read_jsonl(source_paths["diarization_raw_segments"])
    ]
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read_jsonl(source_paths["diarization_bound_segments"])
    ]
    asr = [
        Qwen3ASRSegment.model_validate(row)
        for row in _read_jsonl(source_paths["qwen3_asr_segments"])
    ]
    audits = [
        SpeakerBindingSegmentAudit.model_validate(row)
        for row in _read_jsonl(source_paths["binding_audit_segments"])
    ]
    samples = [
        FinalH3SampleV2.model_validate(row)
        for row in _read_jsonl(source_paths["h3_samples"])
    ]
    canonical_by_clip = {item.clip_uid: item for item in canonical}
    visual_clip_ids = [item.identity.clip_uid for item in visual.canonical_clips]
    diarization_clip_ids = [
        item.target_clip_uid for item in diarization_inventory.targets
    ]
    if (
        len(canonical_by_clip) != len(canonical)
        or len(visual_clip_ids) != len(set(visual_clip_ids))
        or len(diarization_clip_ids) != len(set(diarization_clip_ids))
        or set(canonical_by_clip) != set(visual_clip_ids)
        or set(canonical_by_clip) != set(diarization_clip_ids)
    ):
        raise ValueError("canonical Audio manifest contains duplicate clips")
    raw_by_key = _index(raw, "target_clip_uid", "segment_id")
    bound_by_key = _index(bound, "target_clip_uid", "segment_id")
    _index(asr, "clip_uid", "segment_id")
    audit_by_key = _index(audits, "clip_uid", "segment_id")
    samples_by_clip: dict[str, list[FinalH3SampleV2]] = defaultdict(list)
    for sample in samples:
        samples_by_clip[sample.clip_uid].append(sample)
    canonical_h3_ids = [
        sample.clip_uid for sample in samples if sample.pair_type == "canonical"
    ]
    if (
        len(canonical_h3_ids) != len(set(canonical_h3_ids))
        or set(canonical_h3_ids) != set(visual_clip_ids)
        or set(samples_by_clip) != set(visual_clip_ids)
    ):
        raise ValueError("MiMo H3 inventory is not canonical-wide")
    available_clip_ids = visual_clip_ids
    if case_manifest_path is not None or case_manifest is not None:
        if case_manifest is not None:
            manifest = case_manifest
        else:
            assert case_manifest_path is not None
            manifest = MimoCaseManifest.model_validate_json(
                case_manifest_path.read_text(encoding="utf-8")
            )
        unknown = set(manifest.clip_uids) - set(available_clip_ids)
        if unknown:
            raise ValueError(f"MiMo manifest contains unknown clips: {sorted(unknown)}")
        selected_clip_ids = manifest.clip_uids
    else:
        selected_clip_ids = available_clip_ids
    jobs: list[MimoClipJob] = []
    for clip_uid in selected_clip_ids:
        canonical_clip = canonical_by_clip.get(clip_uid)
        visual_clip = visual_by_clip.get(clip_uid)
        clip_samples = sorted(samples_by_clip.get(clip_uid, []), key=lambda item: item.sample_id)
        if canonical_clip is None or visual_clip is None or not clip_samples:
            raise ValueError(f"MiMo clip lacks canonical Visual/Audio/H3 input: {clip_uid}")
        representative = _validate_h3_variant_observations(clip_uid, clip_samples)
        if (
            Path(representative.target_video).resolve(strict=True)
            != Path(canonical_clip.target_video_path).resolve(strict=True)
            or Path(representative.target_full_audio_path).resolve(strict=True)
            != Path(canonical_clip.target_full_audio_path).resolve(strict=True)
            or Path(visual_clip.sample.target_video).resolve(strict=True)
            != Path(representative.target_video).resolve(strict=True)
        ):
            raise ValueError(f"MiMo target media provenance differs: {clip_uid}")
        target_video = Path(representative.target_video).resolve(strict=True)
        target_audio = Path(representative.target_full_audio_path).resolve(strict=True)
        if sha256_file(target_video) != canonical_clip.target_video_sha256:
            raise ValueError(f"MiMo target video hash differs: {clip_uid}")
        if sha256_file(target_audio) != canonical_clip.target_full_audio_sha256:
            raise ValueError(f"MiMo target audio hash differs: {clip_uid}")
        reference_images = []
        for reference in representative.visual_references:
            artifact = Path(reference.image_artifact_path).resolve(strict=True)
            reference_images.append(
                MimoReferenceImage(
                    image_index=reference.image_index,
                    picture_label=f"<Picture {reference.image_index}>",
                    kind=reference.kind,
                    entity_id=reference.entity_id,
                    attribute_id=reference.attribute_id,
                    owner_entity_id=reference.owner_entity_id,
                    attribute_type=reference.attribute_type,
                    image_artifact_path=str(artifact),
                    image_sha256=sha256_file(artifact),
                )
            )
        variant = "visual_only"
        reference_subjects = build_reference_contract(
            representative, variant
        ).subjects
        _validate_clip_segment_inventory(
            clip_uid,
            raw=raw,
            bound=bound,
            asr=asr,
            audits=audits,
        )
        clip_asr = sorted(
            (item for item in asr if item.clip_uid == clip_uid),
            key=lambda item: (item.start_time, item.end_time, item.segment_id),
        )
        segment_evidence = []
        for asr_row in clip_asr:
            key = (clip_uid, asr_row.segment_id)
            raw_row = raw_by_key.get(key)
            bound_row = bound_by_key.get(key)
            audit_row = audit_by_key.get(key)
            if not isinstance(raw_row, RawDiarizationSegment) or not isinstance(
                bound_row, BoundDiarizationSegment
            ) or not isinstance(audit_row, SpeakerBindingSegmentAudit):
                raise TypeError(f"MiMo segment source coverage differs: {key}")
            exact = (
                raw_row.speaker_cluster_id
                == bound_row.speaker_cluster_id
                == asr_row.speaker_cluster_id
                == audit_row.speaker_cluster_id,
                raw_row.start_time == bound_row.start_time == asr_row.start_time == audit_row.start_time,
                raw_row.end_time == bound_row.end_time == asr_row.end_time == audit_row.end_time,
                raw_row.source_start_sample == bound_row.source_start_sample == asr_row.source_start_sample,
                raw_row.source_end_sample == bound_row.source_end_sample == asr_row.source_end_sample,
                raw_row.source_sample_rate_hz == asr_row.source_sample_rate_hz,
                bound_row.entity_id == asr_row.entity_id == audit_row.current_entity_id,
                bound_row.cluster_binding_status == audit_row.current_mapping_status,
            )
            if not all(exact):
                raise ValueError(f"MiMo DiariZen/ASR/audit reconciliation differs: {key}")
            visible = sorted(
                set(audit_row.direct_support_seconds_by_entity)
                | set(audit_row.multi_active_mapped_entity_ids)
                | {item.entity_id for item in audit_row.exclusive_active_entities}
            )
            competing = sorted(
                {
                    item.face_track_id
                    for item in audit_row.exclusive_active_faces
                    if item.entity_id != bound_row.entity_id
                }
                | set(audit_row.multi_active_unmatched_face_track_ids)
            )
            segment_evidence.append(
                MimoSegmentEvidence(
                    segment_id=asr_row.segment_id,
                    start_time=asr_row.start_time,
                    end_time=asr_row.end_time,
                    source_start_sample=asr_row.source_start_sample,
                    source_end_sample=asr_row.source_end_sample,
                    source_sample_rate_hz=asr_row.source_sample_rate_hz,
                    source_speaker_cluster_id=asr_row.speaker_cluster_id,
                    current_entity_id=asr_row.entity_id,
                    entity_occurrence_id=asr_row.entity_occurrence_id,
                    identity_scope=bound_row.identity_scope,
                    direct_anchor_seconds=bound_row.direct_anchor_seconds,
                    cluster_binding_status=bound_row.cluster_binding_status,
                    overlapping_visible_entities=visible,
                    direct_support_seconds_by_entity=audit_row.direct_support_seconds_by_entity,
                    competing_visible_speaker_evidence=competing,
                    asr_status=asr_row.status,
                    asr_text=asr_row.text,
                    asr_language=asr_row.language,
                )
            )
        values = {
            "clip_uid": clip_uid,
            "r2v_instruction": representative.r2v_instruction,
            "target_video_path": str(target_video),
            "target_video_sha256": canonical_clip.target_video_sha256,
            "target_full_audio_path": str(target_audio),
            "target_full_audio_sha256": canonical_clip.target_full_audio_sha256,
            "target_duration_seconds": canonical_clip.target_duration_seconds,
            "reference_images": [item.model_dump(mode="json") for item in reference_images],
            "reference_subjects": [
                item.model_dump(mode="json") for item in reference_subjects
            ],
            "segments": [item.model_dump(mode="json") for item in segment_evidence],
            "source_h3_sample_ids": [item.sample_id for item in clip_samples],
        }
        jobs.append(_job(values))
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    values = {
        "schema_version": MIMO25_INVENTORY_VERSION,
        "inventory_scope": (
            "explicit_case_subset"
            if case_manifest_path is not None or case_manifest is not None
            else MIMO25_INVENTORY_SCOPE
        ),
        "canonical_wide_coverage": (
            case_manifest_path is None and case_manifest is None
        ),
        "source_visual_inventory_sha256": source_hashes["visual_inventory"],
        "source_canonical_audio_manifest_sha256": source_hashes["canonical_audio_manifest"],
        "source_diarization_inventory_sha256": source_hashes[
            "diarization_inventory"
        ],
        "source_diarization_raw_segments_sha256": source_hashes["diarization_raw_segments"],
        "source_diarization_bound_segments_sha256": source_hashes["diarization_bound_segments"],
        "source_qwen3_asr_segments_sha256": source_hashes["qwen3_asr_segments"],
        "source_binding_audit_segments_sha256": source_hashes["binding_audit_segments"],
        "source_h3_samples_sha256": source_hashes["h3_samples"],
        "clip_count": len(jobs),
        "jobs": [job.model_dump(mode="json") for job in jobs],
    }
    return _inventory(values)


def _correction_counts(job: MimoClipJob, annotation: MimoAVAnnotationDraft) -> Counter[str]:
    counts: Counter[str] = Counter()
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    sources_by_group: dict[str, set[str]] = defaultdict(set)
    for source, decision in zip(job.segments, annotation.segment_decisions, strict=True):
        if decision.primary_speaker_group is not None:
            groups_by_source[source.source_speaker_cluster_id].add(decision.primary_speaker_group)
            sources_by_group[decision.primary_speaker_group].add(source.source_speaker_cluster_id)
        old, new = source.current_entity_id, decision.entity_id
        key = (
            "entity_binding_preserved"
            if old == new
            else "entity_binding_added"
            if old is None and new is not None
            else "entity_binding_removed"
            if old is not None and new is None
            else "entity_binding_changed"
        )
        counts[key] += 1
        counts[f"{decision.binding_status}_segments"] += 1
        counts[f"{decision.resolution}_segments"] += 1
    counts["source_cluster_split"] = sum(len(groups) > 1 for groups in groups_by_source.values())
    counts["source_clusters_merged"] = sum(len(sources) > 1 for sources in sources_by_group.values())
    counts["source_cluster_preserved"] = sum(len(groups) == 1 for groups in groups_by_source.values())
    return counts


def run_mimo25_av_reconcile(
    *,
    inventory: MimoInventory,
    backend: MimoBackend,
    output_root: Path,
    overwrite: bool = False,
) -> MimoSummary:
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    raw_root = temporary / "raw_responses"
    raw_root.mkdir()
    records: list[MimoRecord] = []
    failures: list[MimoFailure] = []
    modality_counts: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    diagnostic_warning_counts: Counter[str] = Counter()
    corrections: Counter[str] = Counter()
    music_counts: Counter[str] = Counter()
    audio_event_count = 0
    nonzero_reasoning_count = 0
    try:
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        for job in inventory.jobs:
            segment_ids = [item.segment_id for item in job.segments]
            transcribed = [item.segment_id for item in job.segments if item.asr_status == "transcribed"]
            entities = {
                item.entity_id
                for item in job.reference_images
                if item.kind == "subject" and item.entity_id is not None
            }
            labels = {item.picture_label for item in job.reference_images}
            labels |= {item.subject_label for item in job.reference_subjects}
            try:
                result = backend.reconcile(
                    job,
                    segment_ids=segment_ids,
                    transcribed_segment_ids=transcribed,
                    allowed_entity_ids=entities,
                    allowed_reference_labels=labels,
                )
                values = {
                    "schema_version": MIMO25_RECORD_VERSION,
                    "clip_uid": job.clip_uid,
                    "request_fingerprint": job.request_fingerprint,
                    "inventory_fingerprint": inventory.inventory_fingerprint,
                    "status": "ready",
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "annotation": result.annotation.model_dump(mode="json"),
                    "failure": None,
                    "input_modality": result.input_modality,
                    "model_call_count": result.model_call_count,
                    "http_attempt_count": result.http_attempt_count,
                    "raw_response_count": len(result.raw_responses),
                    "http_retry_count": result.http_retry_count,
                    "recheck_count": result.recheck_count,
                }
                records.append(_record(values))
                modality_counts[result.input_modality] += 1
                corrections.update(_correction_counts(job, result.annotation))
                audio_event_count += len(result.annotation.audio_semantics.temporal_non_speech_events)
                music_counts[result.annotation.audio_semantics.non_diegetic_music_status] += 1
                case_diagnostics = result.diagnostics
                raw = MimoRawResponse(
                    clip_uid=job.clip_uid,
                    request_fingerprint=job.request_fingerprint,
                    raw_responses=list(result.raw_responses),
                    diagnostics=[item.model_dump(mode="json") for item in result.diagnostics],
                )
            except MimoBackendFailure as exc:
                failure = MimoFailure(
                    clip_uid=job.clip_uid,
                    code=exc.code,
                    reason=exc.reason,
                    attempt_count=exc.model_call_count,
                    http_attempt_count=exc.http_attempt_count,
                    http_retry_count=exc.http_retry_count,
                    recheck_count=exc.recheck_count,
                    issues=[item.to_dict() for item in exc.issues],
                )
                failures.append(failure)
                values = {
                    "schema_version": MIMO25_RECORD_VERSION,
                    "clip_uid": job.clip_uid,
                    "request_fingerprint": job.request_fingerprint,
                    "inventory_fingerprint": inventory.inventory_fingerprint,
                    "status": "failed",
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "annotation": None,
                    "failure": failure.model_dump(mode="json"),
                    "input_modality": None,
                    "model_call_count": exc.model_call_count,
                    "http_attempt_count": exc.http_attempt_count,
                    "raw_response_count": len(exc.raw_responses),
                    "http_retry_count": exc.http_retry_count,
                    "recheck_count": exc.recheck_count,
                }
                records.append(_record(values))
                case_diagnostics = exc.diagnostics
                raw = MimoRawResponse(
                    clip_uid=job.clip_uid,
                    request_fingerprint=job.request_fingerprint,
                    raw_responses=list(exc.raw_responses),
                    diagnostics=[item.model_dump(mode="json") for item in exc.diagnostics],
                )
            for diagnostic in case_diagnostics:
                for field, value in diagnostic.usage.model_dump(mode="python").items():
                    if value is not None:
                        usage_totals[field] += value
                diagnostic_warning_counts.update(diagnostic.warnings)
                if (diagnostic.usage.reasoning_tokens or 0) > 0:
                    nonzero_reasoning_count += 1
            _write_json(raw_root / f"{job.clip_uid}.json", raw.model_dump(mode="json"))
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "failures.jsonl", failures)
        summary = MimoSummary(
            inventory_scope=inventory.inventory_scope,
            canonical_wide_coverage=inventory.canonical_wide_coverage,
            inventory_fingerprint=inventory.inventory_fingerprint,
            source_artifact_hashes={
                "visual_inventory": inventory.source_visual_inventory_sha256,
                "canonical_audio_manifest": inventory.source_canonical_audio_manifest_sha256,
                "diarization_raw_segments": inventory.source_diarization_raw_segments_sha256,
                "diarization_bound_segments": inventory.source_diarization_bound_segments_sha256,
                "qwen3_asr_segments": inventory.source_qwen3_asr_segments_sha256,
                "binding_audit_segments": inventory.source_binding_audit_segments_sha256,
                "h3_samples": inventory.source_h3_samples_sha256,
                **(
                    {}
                    if inventory.source_diarization_inventory_sha256 is None
                    else {
                        "diarization_inventory": (
                            inventory.source_diarization_inventory_sha256
                        )
                    }
                ),
            },
            clip_count=len(records),
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            unsupported_count=sum(item.status == "unsupported" for item in records),
            model_call_count=sum(item.model_call_count for item in records),
            http_attempt_count=sum(item.http_attempt_count for item in records),
            http_retry_count=sum(item.http_retry_count for item in records),
            recheck_count=sum(item.recheck_count for item in records),
            input_modality_counts=dict(sorted(modality_counts.items())),
            usage_totals=dict(sorted(usage_totals.items())),
            diagnostic_warning_counts=dict(
                sorted(diagnostic_warning_counts.items())
            ),
            responses_with_nonzero_reasoning_tokens=nonzero_reasoning_count,
            correction_counts=dict(sorted(corrections.items())),
            audio_event_count=audio_event_count,
            music_status_counts=dict(sorted(music_counts.items())),
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.backup"
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except Exception:
            if backup.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def known_case_manifest() -> MimoCaseManifest:
    return MimoCaseManifest(
        clip_uids=[
            "938050f7193d7c065dc8e249",
            "711b1dbc74932c5d1d720495",
            "a073596149def028cff1305e",
        ]
    )


__all__ = [
    "MIMO25_CASE_MANIFEST_VERSION",
    "MIMO25_INVENTORY_SCOPE",
    "MimoBackend",
    "MimoCaseManifest",
    "MimoClipJob",
    "MimoFailure",
    "MimoInventory",
    "MimoRecord",
    "MimoReferenceImage",
    "MimoSegmentEvidence",
    "MimoSummary",
    "build_mimo25_inventory",
    "known_case_manifest",
    "run_mimo25_av_reconcile",
]
