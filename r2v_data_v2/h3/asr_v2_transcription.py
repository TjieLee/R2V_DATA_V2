from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import statistics
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.asr_transcription import (
    ASR_HUMAN_QA_LABELS,
    ASR_INPUT_SAMPLE_RATE_HZ,
    ASRBackend,
    ASRBackendProvenance,
    ASRBackendResult,
    ASRDecoderDiagnostics,
    ASRFailure,
    ASRHumanQACounts,
    ASRInventory,
    ASRSummary,
    ASRTranscriptionFailure,
    ASRTurnJob,
    ASRTurnRecord,
    PreparedExactASRWaveform,
    prepare_exact_asr_waveform,
)
from r2v_data_v2.h3.asr_transcription import (
    _inventory_fingerprint as _asr_v1_inventory_fingerprint,
)
from r2v_data_v2.h3.asr_transcription import (
    _write_review_wav as _write_asr_review_wav,
)
from r2v_data_v2.h3.diarization_binding import (
    DIARIZATION_MAPPING_POLICY_VERSION,
    DIARIZATION_SEGMENT_VERSION,
    BoundDiarizationSegment,
    DiarizationBoundaryReconciliation,
    DiarizationClusterBinding,
    DiarizationInventory,
    DiarizationSummary,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.schemas import SchemaModel

ASR_V2_SEGMENT_SCHEMA_VERSION = "r2v.h3.asr_v2_segment.1"
ASR_V2_INVENTORY_SCHEMA_VERSION = "r2v.h3.asr_v2_inventory.1"
ASR_V2_SUMMARY_SCHEMA_VERSION = "r2v.h3.asr_v2_summary.1"
ASR_V2_PRODUCTION_INVENTORY_SCHEMA_VERSION = "r2v.h3.asr_v2_inventory.2"
ASR_V2_PRODUCTION_SUMMARY_SCHEMA_VERSION = "r2v.h3.asr_v2_summary.2"
ASR_V2_HUMAN_QA_SCHEMA_VERSION = "r2v.h3.asr_v2_human_qa.1"
ASR_V2_REQUEST_CONTRACT_VERSION = "h3_whisper_diarizen_segment_asr_v2"
ASR_V2_PREPROCESSING_VERSION = "pcm16_exact_diarizen_segment_crop_v1"
ASR_V2_BOUNDARY_SOURCE = "production_diarizen_segment_v1"
ASR_V2_ENTITY_BINDING_SOURCE = DIARIZATION_MAPPING_POLICY_VERSION
ASR_V2_SOURCE_DIARIZATION_INVENTORY_FINGERPRINT = (
    "bc750320ab0488c122c18c8a70afb6c44c70dd5b9b5c001d791cfaf9ff4fb1d9"
)
ASR_V2_SOURCE_ASR_V1_INVENTORY_FINGERPRINT = (
    "ead8ce8aad5dc587517c4d38e74962152fbae96721fe1f92797b832d648c6a75"
)
ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT = (
    "e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0"
)
ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT = (
    "10ea6fb8ae7cdd1fa26495deeb1f32e79c1fc882c19f80ae711bf5f0dd671db3"
)
ASR_V2_CALIBRATION_HUMAN_QA_TOTAL = 50
ASR_V2_CALIBRATION_HUMAN_QA_CORRECT = 41
ASR_V2_CALIBRATION_HUMAN_QA_WRONG = 3
ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN = 6
ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED = 0
EXPECTED_PRODUCTION_TARGET_COUNT = 75


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"ASR V2 JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(item.model_dump(mode="json")) + "\n" for item in values),
        encoding="utf-8",
    )


class ASRV2TargetClip(SchemaModel):
    target_clip_uid: str
    target_video_path: str
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    source_frame_count: int = Field(gt=0)
    segment_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_target(self) -> ASRV2TargetClip:
        if not self.target_clip_uid.strip():
            raise ValueError("ASR V2 target clip identity must not be empty")
        for value in (self.target_video_path, self.source_audio_path):
            if not value.strip():
                raise ValueError("ASR V2 target media path must not be empty")
        return self


class ASRV2SegmentJob(SchemaModel):
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    cluster_binding_status: Literal[
        "candidate_mapped", "ambiguous", "unbound", "conflict"
    ]
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    identity_scope: Literal[
        "direct_anchor_present", "cluster_propagated_only", "unresolved"
    ]
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_segment_schema_version: Literal[
        "r2v.h3.diarization_segment.2"
    ] = DIARIZATION_SEGMENT_VERSION
    source_diarization_mapping_policy_version: Literal[
        "h3_diarizen_sparse_anchor_policy_v1"
    ] = DIARIZATION_MAPPING_POLICY_VERSION
    source_segment_id: str
    boundary_source: Literal["production_diarizen_segment_v1"] = ASR_V2_BOUNDARY_SOURCE
    entity_binding_source: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        ASR_V2_ENTITY_BINDING_SOURCE
    )
    boundary_reconciliation: DiarizationBoundaryReconciliation

    @model_validator(mode="after")
    def validate_job(self) -> ASRV2SegmentJob:
        if any(
            not value.strip()
            for value in (
                self.target_clip_uid,
                self.segment_id,
                self.speaker_cluster_id,
                self.source_segment_id,
                self.source_audio_path,
            )
        ):
            raise ValueError("ASR V2 segment identity and provenance must be complete")
        if self.source_segment_id != self.segment_id:
            raise ValueError("ASR V2 source segment identity is inconsistent")
        mapped = self.cluster_binding_status == "candidate_mapped"
        if mapped != (self.entity_id is not None):
            raise ValueError("only candidate_mapped ASR V2 segments may bind an entity")
        expected_occurrence = (
            f"{self.target_clip_uid}/{self.entity_id}"
            if self.entity_id is not None
            else None
        )
        if self.entity_occurrence_id != expected_occurrence:
            raise ValueError("ASR V2 entity occurrence identity is inconsistent")
        if not mapped and self.identity_scope != "unresolved":
            raise ValueError("unresolved ASR V2 segment must preserve unresolved scope")
        if mapped and self.identity_scope == "unresolved":
            raise ValueError("mapped ASR V2 segment cannot use unresolved scope")
        if self.end_time <= self.start_time:
            raise ValueError("ASR V2 segment interval must be positive")
        if self.source_start_sample != round(
            self.start_time * self.source_sample_rate_hz
        ) or self.source_end_sample != round(
            self.end_time * self.source_sample_rate_hz
        ):
            raise ValueError("ASR V2 samples must match effective DiariZen times")
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("ASR V2 sample interval must be positive")
        return self


class ASRV2Inventory(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_inventory.1"] = (
        ASR_V2_INVENTORY_SCHEMA_VERSION
    )
    mode: Literal["pilot20", "production"]
    source_diarization_root: str
    source_diarization_inventory_path: str
    source_diarization_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_raw_segments_path: str
    source_diarization_raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_bound_segments_path: str
    source_diarization_bound_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_cluster_bindings_path: str
    source_diarization_cluster_bindings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_summary_path: str
    source_diarization_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_mapping_policy_version: Literal[
        "h3_diarizen_sparse_anchor_policy_v1"
    ] = DIARIZATION_MAPPING_POLICY_VERSION
    source_asr_v1_inventory_path: str
    source_asr_v1_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_turns_path: str
    source_asr_v1_turns_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_summary_path: str
    source_asr_v1_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_asr_v1_backend_provenance: ASRBackendProvenance
    baseline_asr_v1_turn_count: int = Field(ge=0)
    baseline_asr_v1_turn_median_duration: float | None = Field(default=None, gt=0)
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_count: int = Field(ge=0)
    selected_target_count: int = Field(ge=0)
    selected_segment_count: int = Field(ge=0)
    selection_mode: Literal[
        "same_ordered_asr_v1_pilot20_targets_v1",
        "complete_production_diarization_targets_v1",
    ]
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    cross_pair_jobs_created: Literal[0] = 0
    production_inference_blocked: bool
    targets: list[ASRV2TargetClip]
    jobs: list[ASRV2SegmentJob]

    @model_validator(mode="after")
    def validate_inventory(self) -> ASRV2Inventory:
        target_ids = [item.target_clip_uid for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("ASR V2 targets must be unique")
        if self.selected_target_count != len(self.targets):
            raise ValueError("ASR V2 selected target count is inconsistent")
        if self.selected_segment_count != len(self.jobs):
            raise ValueError("ASR V2 selected segment count is inconsistent")
        job_ids = [(item.target_clip_uid, item.segment_id) for item in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("ASR V2 segment jobs must be unique")
        target_set = set(target_ids)
        if any(item.target_clip_uid not in target_set for item in self.jobs):
            raise ValueError("ASR V2 segment references an unknown target")
        counts = Counter(item.target_clip_uid for item in self.jobs)
        if any(
            item.segment_count != counts[item.target_clip_uid] for item in self.targets
        ):
            raise ValueError("ASR V2 per-target segment count is inconsistent")
        if self.mode == "pilot20":
            if (
                self.selection_mode != "same_ordered_asr_v1_pilot20_targets_v1"
                or self.selected_target_count != 20
                or not self.bounded_selection_applied
                or self.production_inference_blocked
            ):
                raise ValueError("ASR V2 pilot must reuse the frozen ASR V1 target set")
        elif (
            self.selection_mode != "complete_production_diarization_targets_v1"
            or self.selected_target_count != self.source_target_count
            or self.bounded_selection_applied
            or not self.production_inference_blocked
        ):
            raise ValueError("production ASR V2 inventory must be complete and blocked")
        return self


class ASRV2ProductionInventory(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_inventory.2"] = (
        ASR_V2_PRODUCTION_INVENTORY_SCHEMA_VERSION
    )
    mode: Literal["production"] = "production"
    source_diarization_root: str
    source_diarization_inventory_path: str
    source_diarization_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_raw_segments_path: str
    source_diarization_raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_bound_segments_path: str
    source_diarization_bound_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_cluster_bindings_path: str
    source_diarization_cluster_bindings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_summary_path: str
    source_diarization_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_mapping_policy_version: Literal[
        "h3_diarizen_sparse_anchor_policy_v1"
    ] = DIARIZATION_MAPPING_POLICY_VERSION
    source_asr_v1_inventory_path: str
    source_asr_v1_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_turns_path: str
    source_asr_v1_turns_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v1_summary_path: str
    source_asr_v1_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_asr_v1_backend_provenance: ASRBackendProvenance
    baseline_asr_v1_turn_count: int = Field(ge=0)
    baseline_asr_v1_turn_median_duration: float | None = Field(default=None, gt=0)
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_count: int = Field(ge=0)
    selected_target_count: int = Field(ge=0)
    selected_segment_count: int = Field(ge=0)
    selection_mode: Literal["complete_production_diarization_targets_v1"] = (
        "complete_production_diarization_targets_v1"
    )
    bounded_selection_applied: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    cross_pair_jobs_created: Literal[0] = 0
    production_inference_enabled: Literal[True] = True
    asr_v2_policy_validated: Literal[True] = True
    calibration_inventory_fingerprint: Literal[
        "e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0"
    ] = ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT
    calibration_checkpoint_fingerprint: Literal[
        "10ea6fb8ae7cdd1fa26495deeb1f32e79c1fc882c19f80ae711bf5f0dd671db3"
    ] = ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    calibration_human_qa_total: Literal[50] = ASR_V2_CALIBRATION_HUMAN_QA_TOTAL
    calibration_human_qa_correct: Literal[41] = ASR_V2_CALIBRATION_HUMAN_QA_CORRECT
    calibration_human_qa_wrong: Literal[3] = ASR_V2_CALIBRATION_HUMAN_QA_WRONG
    calibration_human_qa_uncertain: Literal[6] = ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN
    calibration_human_qa_unlabeled: Literal[0] = ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED
    text_usability_gate_applied: Literal[False] = False
    transcript_confidence_threshold_used: Literal[False] = False
    targets: list[ASRV2TargetClip]
    jobs: list[ASRV2SegmentJob]

    @model_validator(mode="after")
    def validate_inventory(self) -> ASRV2ProductionInventory:
        target_ids = [item.target_clip_uid for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("ASR V2 production targets must be unique")
        if (
            self.selected_target_count != len(self.targets)
            or self.selected_target_count != self.source_target_count
            or self.selected_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
        ):
            raise ValueError("ASR V2 production target inventory must be complete")
        if self.selected_segment_count != len(self.jobs):
            raise ValueError("ASR V2 production segment count is inconsistent")
        job_ids = [(item.target_clip_uid, item.segment_id) for item in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("ASR V2 production segment jobs must be unique")
        target_set = set(target_ids)
        if any(item.target_clip_uid not in target_set for item in self.jobs):
            raise ValueError("ASR V2 production segment references an unknown target")
        counts = Counter(item.target_clip_uid for item in self.jobs)
        if any(
            item.segment_count != counts[item.target_clip_uid] for item in self.targets
        ):
            raise ValueError("ASR V2 production per-target count is inconsistent")
        if self.calibration_human_qa_total != (
            self.calibration_human_qa_correct
            + self.calibration_human_qa_wrong
            + self.calibration_human_qa_uncertain
            + self.calibration_human_qa_unlabeled
        ):
            raise ValueError("ASR V2 calibration human-QA counts are inconsistent")
        return self


ASRV2InventoryRecord = ASRV2Inventory | ASRV2ProductionInventory


class ASRV2PreprocessingProvenance(SchemaModel):
    version: Literal["pcm16_exact_diarizen_segment_crop_v1"] = (
        ASR_V2_PREPROCESSING_VERSION
    )
    source_encoding: Literal["pcm_s16le_wave"] = "pcm_s16le_wave"
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    asr_input_sample_rate_hz: Literal[16000] = ASR_INPUT_SAMPLE_RATE_HZ
    asr_input_channels: Literal[1] = 1
    resampled: bool
    downmixed: bool
    padding_seconds: Literal[0.0] = 0.0
    look_behind_seconds: Literal[0.0] = 0.0
    look_ahead_seconds: Literal[0.0] = 0.0
    denoising_applied: Literal[False] = False
    enhancement_applied: Literal[False] = False
    vad_resegmentation_applied: Literal[False] = False
    diarization_resegmentation_applied: Literal[False] = False


class ASRV2BackendProvenance(SchemaModel):
    backend: Literal["faster_whisper"] = "faster_whisper"
    model_identifier: str
    checkpoint_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_comparison: Literal["matched", "unavailable_in_asr_v1"]
    device: str
    compute_type: str
    task: Literal["transcribe"] = "transcribe"
    condition_on_previous_text: Literal[False] = False
    vad_filter: Literal[False] = False
    word_timestamps: Literal[False] = False
    local_files_only: Literal[True] = True
    request_contract_version: Literal["h3_whisper_diarizen_segment_asr_v2"] = (
        ASR_V2_REQUEST_CONTRACT_VERSION
    )
    preprocessing_version: Literal["pcm16_exact_diarizen_segment_crop_v1"] = (
        ASR_V2_PREPROCESSING_VERSION
    )
    baseline_asr_v1_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_device_may_differ_from_asr_v1: Literal[True] = True
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> ASRV2BackendProvenance:
        if any(
            not value.strip()
            for value in (self.model_identifier, self.device, self.compute_type)
        ):
            raise ValueError("ASR V2 backend identity must not be empty")
        if (self.checkpoint_comparison == "matched") != (
            self.checkpoint_fingerprint is not None
        ):
            raise ValueError("ASR V2 checkpoint comparison provenance is inconsistent")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("ASR V2 backend configuration fingerprint is invalid")
        return self


def build_asr_v2_backend_provenance(
    *,
    runtime: ASRBackendProvenance,
    baseline: ASRBackendProvenance,
) -> ASRV2BackendProvenance:
    comparable_fields = (
        "backend",
        "model_identifier",
        "compute_type",
        "task",
        "condition_on_previous_text",
        "vad_filter",
        "word_timestamps",
        "local_files_only",
    )
    if any(
        getattr(runtime, field) != getattr(baseline, field)
        for field in comparable_fields
    ):
        raise ValueError("ASR V2 runtime does not match the frozen ASR V1 decoder")
    if baseline.checkpoint_fingerprint is not None:
        if runtime.checkpoint_fingerprint != baseline.checkpoint_fingerprint:
            raise ValueError("ASR V2 checkpoint fingerprint differs from ASR V1")
        checkpoint_fingerprint = baseline.checkpoint_fingerprint
        checkpoint_comparison = "matched"
    else:
        checkpoint_fingerprint = None
        checkpoint_comparison = "unavailable_in_asr_v1"
    values = {
        "backend": runtime.backend,
        "model_identifier": runtime.model_identifier,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_comparison": checkpoint_comparison,
        "device": runtime.device,
        "compute_type": runtime.compute_type,
        "task": runtime.task,
        "condition_on_previous_text": runtime.condition_on_previous_text,
        "vad_filter": runtime.vad_filter,
        "word_timestamps": runtime.word_timestamps,
        "local_files_only": runtime.local_files_only,
        "request_contract_version": ASR_V2_REQUEST_CONTRACT_VERSION,
        "preprocessing_version": ASR_V2_PREPROCESSING_VERSION,
        "baseline_asr_v1_configuration_fingerprint": baseline.configuration_fingerprint,
        "physical_device_may_differ_from_asr_v1": True,
    }
    return ASRV2BackendProvenance(
        **values,
        configuration_fingerprint=_sha256_text(_compact_json(values)),
    )


class ASRV2SegmentRecord(ASRV2SegmentJob):
    schema_version: Literal["r2v.h3.asr_v2_segment.1"] = ASR_V2_SEGMENT_SCHEMA_VERSION
    backend: Literal["faster_whisper"] = "faster_whisper"
    model_identifier: str
    task: Literal["transcribe"] = "transcribe"
    backend_provenance: ASRV2BackendProvenance
    preprocessing: ASRV2PreprocessingProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["transcribed", "uncertain", "failed"]
    text: str | None = None
    language: str | None = None
    diagnostics: ASRDecoderDiagnostics | None = None
    warnings: list[str] = Field(default_factory=list)
    failure: ASRFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> ASRV2SegmentRecord:
        if self.model_identifier != self.backend_provenance.model_identifier:
            raise ValueError("ASR V2 model provenance is inconsistent")
        if any(not item.strip() for item in self.warnings):
            raise ValueError("ASR V2 warnings must not be empty")
        if self.status == "transcribed":
            if self.text is None or not self.text.strip() or self.failure is not None:
                raise ValueError("transcribed ASR V2 segment requires text")
        elif self.status == "uncertain":
            if self.text is not None or self.failure is not None:
                raise ValueError("uncertain ASR V2 segment requires null text")
        elif self.text is not None or self.failure is None:
            raise ValueError("failed ASR V2 segment requires failure details")
        return self


class ASRV2Summary(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_summary.1"] = ASR_V2_SUMMARY_SCHEMA_VERSION
    mode: Literal["pilot20", "production"]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_mapping_policy_version: Literal[
        "h3_diarizen_sparse_anchor_policy_v1"
    ] = DIARIZATION_MAPPING_POLICY_VERSION
    source_asr_v1_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: ASRV2BackendProvenance
    source_target_count: int = Field(ge=0)
    target_clip_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    cross_pair_jobs_created: Literal[0] = 0
    candidate_mapped_segment_count: int = Field(ge=0)
    ambiguous_segment_count: int = Field(ge=0)
    unbound_segment_count: int = Field(ge=0)
    conflict_segment_count: int = Field(ge=0)
    direct_anchor_present_segment_count: int = Field(ge=0)
    cluster_propagated_only_segment_count: int = Field(ge=0)
    unresolved_segment_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    backend_call_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    total_segment_seconds: float = Field(ge=0)
    segment_median_duration: float | None = Field(default=None, gt=0)
    baseline_asr_v1_turn_count: int = Field(ge=0)
    baseline_asr_v1_turn_median_duration: float | None = Field(default=None, gt=0)
    transcript_confidence_threshold_used: Literal[False] = False
    pair_assets_modified: Literal[False] = False
    primary_voice_assets_modified: Literal[False] = False
    diarization_assets_modified: Literal[False] = False
    production_inference_blocked_pending_human_calibration: Literal[True] = True

    @model_validator(mode="after")
    def validate_summary(self) -> ASRV2Summary:
        if self.segment_count != (
            self.candidate_mapped_segment_count
            + self.ambiguous_segment_count
            + self.unbound_segment_count
            + self.conflict_segment_count
        ):
            raise ValueError("ASR V2 identity status counts must reconcile")
        if self.segment_count != (
            self.direct_anchor_present_segment_count
            + self.cluster_propagated_only_segment_count
            + self.unresolved_segment_count
        ):
            raise ValueError("ASR V2 identity scope counts must reconcile")
        if self.segment_count != (
            self.transcribed_count + self.uncertain_count + self.failed_count
        ):
            raise ValueError("ASR V2 transcript status counts must reconcile")
        if self.backend_call_count > self.segment_count:
            raise ValueError("ASR V2 backend calls cannot exceed segment count")
        return self


class ASRV2ProductionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_summary.2"] = (
        ASR_V2_PRODUCTION_SUMMARY_SCHEMA_VERSION
    )
    mode: Literal["production"] = "production"
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_mapping_policy_version: Literal[
        "h3_diarizen_sparse_anchor_policy_v1"
    ] = DIARIZATION_MAPPING_POLICY_VERSION
    source_asr_v1_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_inventory_fingerprint: Literal[
        "e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0"
    ] = ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT
    calibration_checkpoint_fingerprint: Literal[
        "10ea6fb8ae7cdd1fa26495deeb1f32e79c1fc882c19f80ae711bf5f0dd671db3"
    ] = ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    backend_provenance: ASRV2BackendProvenance
    source_target_count: int = Field(ge=0)
    target_clip_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    bounded_selection_applied: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    cross_pair_jobs_created: Literal[0] = 0
    candidate_mapped_segment_count: int = Field(ge=0)
    ambiguous_segment_count: int = Field(ge=0)
    unbound_segment_count: int = Field(ge=0)
    conflict_segment_count: int = Field(ge=0)
    direct_anchor_present_segment_count: int = Field(ge=0)
    cluster_propagated_only_segment_count: int = Field(ge=0)
    unresolved_segment_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    backend_call_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    total_segment_seconds: float = Field(ge=0)
    segment_median_duration: float | None = Field(default=None, gt=0)
    baseline_asr_v1_turn_count: int = Field(ge=0)
    baseline_asr_v1_turn_median_duration: float | None = Field(default=None, gt=0)
    asr_v2_policy_validated: Literal[True] = True
    calibration_human_qa_total: Literal[50] = ASR_V2_CALIBRATION_HUMAN_QA_TOTAL
    calibration_human_qa_correct: Literal[41] = ASR_V2_CALIBRATION_HUMAN_QA_CORRECT
    calibration_human_qa_wrong: Literal[3] = ASR_V2_CALIBRATION_HUMAN_QA_WRONG
    calibration_human_qa_uncertain: Literal[6] = ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN
    calibration_human_qa_unlabeled: Literal[0] = ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED
    text_usability_gate_applied: Literal[False] = False
    transcript_confidence_threshold_used: Literal[False] = False
    pair_assets_modified: Literal[False] = False
    primary_voice_assets_modified: Literal[False] = False
    embedding_assets_modified: Literal[False] = False
    diarization_assets_modified: Literal[False] = False
    asr_v1_assets_modified: Literal[False] = False
    calibration_pilot_assets_modified: Literal[False] = False
    production_inference_enabled: Literal[True] = True

    @model_validator(mode="after")
    def validate_summary(self) -> ASRV2ProductionSummary:
        if self.target_clip_count != self.source_target_count:
            raise ValueError("ASR V2 production target count must be complete")
        if self.segment_count != (
            self.candidate_mapped_segment_count
            + self.ambiguous_segment_count
            + self.unbound_segment_count
            + self.conflict_segment_count
        ):
            raise ValueError("ASR V2 production identity counts must reconcile")
        if self.segment_count != (
            self.direct_anchor_present_segment_count
            + self.cluster_propagated_only_segment_count
            + self.unresolved_segment_count
        ):
            raise ValueError("ASR V2 production identity scopes must reconcile")
        if self.segment_count != (
            self.transcribed_count + self.uncertain_count + self.failed_count
        ):
            raise ValueError("ASR V2 production ASR status counts must reconcile")
        if self.backend_call_count > self.segment_count:
            raise ValueError("ASR V2 production backend calls cannot exceed segments")
        if self.calibration_human_qa_total != (
            self.calibration_human_qa_correct
            + self.calibration_human_qa_wrong
            + self.calibration_human_qa_uncertain
            + self.calibration_human_qa_unlabeled
        ):
            raise ValueError("ASR V2 calibration human-QA counts are inconsistent")
        return self


ASRV2SummaryRecord = ASRV2Summary | ASRV2ProductionSummary


class ASRV2HumanQALabel(SchemaModel):
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    label: Literal["CORRECT", "WRONG", "UNCERTAIN"]

    @model_validator(mode="after")
    def validate_label(self) -> ASRV2HumanQALabel:
        if any(
            not value.strip()
            for value in (
                self.target_clip_uid,
                self.segment_id,
                self.speaker_cluster_id,
            )
        ):
            raise ValueError("ASR V2 human-QA identity must not be empty")
        return self


class ASRV2HumanQAExport(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_human_qa.1"] = ASR_V2_HUMAN_QA_SCHEMA_VERSION
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20", "production"]
    label_count: int = Field(ge=0)
    total_segment_count: int = Field(ge=0)
    counts: ASRHumanQACounts
    labels: list[ASRV2HumanQALabel]

    @model_validator(mode="after")
    def validate_export(self) -> ASRV2HumanQAExport:
        keys = [
            (item.target_clip_uid, item.segment_id, item.speaker_cluster_id)
            for item in self.labels
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("ASR V2 human-QA labels must be unique")
        actual = Counter(item.label for item in self.labels)
        if any(
            getattr(self.counts, label) != actual[label]
            for label in ASR_HUMAN_QA_LABELS
        ):
            raise ValueError("ASR V2 human-QA counts are inconsistent")
        labeled = self.counts.CORRECT + self.counts.WRONG + self.counts.UNCERTAIN
        if self.label_count != len(self.labels) or self.label_count != labeled:
            raise ValueError("ASR V2 human-QA label count is inconsistent")
        if self.total_segment_count != self.label_count + self.counts.UNLABELED:
            raise ValueError("ASR V2 human-QA total count is inconsistent")
        return self


def _inventory_fingerprint(
    inventory: ASRV2InventoryRecord | dict[str, object],
) -> str:
    values = (
        inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if isinstance(inventory, (ASRV2Inventory, ASRV2ProductionInventory))
        else {
            key: value
            for key, value in inventory.items()
            if key != "inventory_fingerprint"
        }
    )
    return _sha256_text(_compact_json(values))


def _validate_diarization_inventory(inventory: DiarizationInventory) -> None:
    expected = _diarization_inventory_fingerprint(
        source_pairs_sha256=inventory.source_pairs_sha256,
        source_asr_inventory_fingerprint=inventory.source_asr_inventory_fingerprint,
        mode=inventory.mode,
        targets=inventory.targets,
    )
    if inventory.inventory_fingerprint != expected:
        raise ValueError("source DiariZen inventory fingerprint is inconsistent")
    if (
        inventory.mode != "production"
        or inventory.selected_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
        or inventory.selected_target_count != inventory.source_target_count
        or inventory.bounded_selection_applied
        or inventory.parent_quota_applied
        or inventory.cross_pair_jobs_created != 0
    ):
        raise ValueError("ASR V2 requires the complete formal DiariZen production")


def _validate_asr_v1_inventory(inventory: ASRInventory) -> None:
    expected = _asr_v1_inventory_fingerprint(
        source_pairs_sha256=inventory.source_pairs_sha256,
        mode=inventory.mode,
        targets=inventory.targets,
        jobs=inventory.jobs,
    )
    if inventory.inventory_fingerprint != expected:
        raise ValueError("source ASR V1 inventory fingerprint is inconsistent")
    if inventory.mode != "pilot20" or inventory.selected_target_count != 20:
        raise ValueError("ASR V2 pilot requires the frozen ASR V1 pilot20")


def _median_duration(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def build_asr_v2_inventory(
    *,
    audio_run_root: Path,
    mode: Literal["pilot20", "production"],
    expected_diarization_inventory_fingerprint: str | None = (
        ASR_V2_SOURCE_DIARIZATION_INVENTORY_FINGERPRINT
    ),
    expected_asr_v1_inventory_fingerprint: str | None = (
        ASR_V2_SOURCE_ASR_V1_INVENTORY_FINGERPRINT
    ),
    expected_calibration_checkpoint_fingerprint: str | None = (
        ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    ),
) -> ASRV2InventoryRecord:
    root = audio_run_root.expanduser().resolve(strict=True)
    diarization_root = (root / "production" / "diarization").resolve(strict=True)
    source_paths = {
        "inventory": (diarization_root / "inventory.json").resolve(strict=True),
        "raw": (diarization_root / "raw_segments.jsonl").resolve(strict=True),
        "bound": (diarization_root / "bound_segments.jsonl").resolve(strict=True),
        "clusters": (diarization_root / "cluster_bindings.jsonl").resolve(strict=True),
        "summary": (diarization_root / "summary.json").resolve(strict=True),
    }
    if any(path.parent != diarization_root for path in source_paths.values()):
        raise ValueError("ASR V2 DiariZen artifact path escaped its source root")
    diarization_inventory = DiarizationInventory.model_validate_json(
        source_paths["inventory"].read_text(encoding="utf-8")
    )
    _validate_diarization_inventory(diarization_inventory)
    if (
        expected_diarization_inventory_fingerprint is not None
        and diarization_inventory.inventory_fingerprint
        != expected_diarization_inventory_fingerprint
    ):
        raise ValueError("formal DiariZen production fingerprint is not accepted")
    diarization_summary = DiarizationSummary.model_validate_json(
        source_paths["summary"].read_text(encoding="utf-8")
    )
    if (
        diarization_summary.mode != "production"
        or diarization_summary.inventory_fingerprint
        != diarization_inventory.inventory_fingerprint
        or diarization_summary.mapping_policy_version
        != DIARIZATION_MAPPING_POLICY_VERSION
        or not diarization_summary.mapping_policy_validated
        or diarization_summary.numeric_mapping_thresholds_used
        or diarization_summary.failed_clip_count != 0
        or diarization_summary.empty_clip_count != 0
        or diarization_summary.ready_clip_count
        != diarization_inventory.selected_target_count
    ):
        raise ValueError("formal DiariZen production summary is not accepted")

    raw_segments = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(source_paths["raw"])
    ]
    bound_segments = [
        BoundDiarizationSegment.model_validate(item)
        for item in _read_jsonl(source_paths["bound"])
    ]
    cluster_bindings = [
        DiarizationClusterBinding.model_validate(item)
        for item in _read_jsonl(source_paths["clusters"])
    ]
    if diarization_summary.raw_segment_count != len(raw_segments):
        raise ValueError("DiariZen raw segment count does not match its summary")
    if diarization_summary.speaker_cluster_count != len(cluster_bindings):
        raise ValueError("DiariZen cluster count does not match its summary")
    raw_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in raw_segments
    }
    bound_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in bound_segments
    }
    cluster_by_key = {
        (item.target_clip_uid, item.speaker_cluster_id): item
        for item in cluster_bindings
    }
    if (
        len(raw_by_key) != len(raw_segments)
        or len(bound_by_key) != len(bound_segments)
        or len(cluster_by_key) != len(cluster_bindings)
        or set(raw_by_key) != set(bound_by_key)
    ):
        raise ValueError("DiariZen segment or cluster identities are inconsistent")

    asr_v1_root = (root / "asr_pilot20").resolve(strict=True)
    asr_v1_paths = {
        "inventory": (asr_v1_root / "inventory.json").resolve(strict=True),
        "turns": (asr_v1_root / "turns.jsonl").resolve(strict=True),
        "summary": (asr_v1_root / "summary.json").resolve(strict=True),
    }
    if any(path.parent != asr_v1_root for path in asr_v1_paths.values()):
        raise ValueError("ASR V1 artifact path escaped its source root")
    asr_v1_inventory = ASRInventory.model_validate_json(
        asr_v1_paths["inventory"].read_text(encoding="utf-8")
    )
    _validate_asr_v1_inventory(asr_v1_inventory)
    if (
        expected_asr_v1_inventory_fingerprint is not None
        and asr_v1_inventory.inventory_fingerprint
        != expected_asr_v1_inventory_fingerprint
    ):
        raise ValueError("frozen ASR V1 pilot fingerprint is not accepted")
    asr_v1_summary = ASRSummary.model_validate_json(
        asr_v1_paths["summary"].read_text(encoding="utf-8")
    )
    asr_v1_records = [
        ASRTurnRecord.model_validate(item)
        for item in _read_jsonl(asr_v1_paths["turns"])
    ]
    if (
        asr_v1_summary.inventory_fingerprint != asr_v1_inventory.inventory_fingerprint
        or asr_v1_summary.turn_count != len(asr_v1_records)
        or len(asr_v1_records) != len(asr_v1_inventory.jobs)
    ):
        raise ValueError("frozen ASR V1 pilot artifacts are inconsistent")
    if (
        mode == "production"
        and expected_calibration_checkpoint_fingerprint is not None
        and asr_v1_summary.backend_provenance.checkpoint_fingerprint
        != expected_calibration_checkpoint_fingerprint
    ):
        raise ValueError("ASR V2 production checkpoint is not the calibrated baseline")
    v1_job_fields = tuple(ASRTurnJob.model_fields)
    for job, record in zip(asr_v1_inventory.jobs, asr_v1_records, strict=True):
        record_job = ASRTurnJob.model_validate(
            {field: getattr(record, field) for field in v1_job_fields}
        )
        if (
            record_job != job
            or record.backend_provenance != asr_v1_summary.backend_provenance
        ):
            raise ValueError("frozen ASR V1 record provenance is inconsistent")

    diarization_targets = {
        item.target_clip_uid: item for item in diarization_inventory.targets
    }
    if len(diarization_targets) != len(diarization_inventory.targets):
        raise ValueError("formal DiariZen target identities are duplicated")
    if mode == "pilot20":
        selected_ids = [item.target_clip_uid for item in asr_v1_inventory.targets]
        selection_mode = "same_ordered_asr_v1_pilot20_targets_v1"
        bounded = True
    else:
        selected_ids = [item.target_clip_uid for item in diarization_inventory.targets]
        selection_mode = "complete_production_diarization_targets_v1"
        bounded = False
    if any(clip_uid not in diarization_targets for clip_uid in selected_ids):
        raise ValueError("ASR V1 pilot target is missing from DiariZen production")

    raw_by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    for segment in raw_segments:
        raw_by_clip[segment.target_clip_uid].append(segment)
    targets: list[ASRV2TargetClip] = []
    jobs: list[ASRV2SegmentJob] = []
    for clip_uid in selected_ids:
        source_target = diarization_targets[clip_uid]
        video_path = (
            Path(source_target.target_video_path).expanduser().resolve(strict=True)
        )
        audio_path = (
            Path(source_target.source_audio_path).expanduser().resolve(strict=True)
        )
        if not video_path.is_file() or not audio_path.is_file():
            raise ValueError("ASR V2 target media is unavailable")
        if _sha256_file(audio_path) != source_target.source_audio_sha256:
            raise ValueError("ASR V2 canonical source audio hash changed")
        selected_raw = raw_by_clip[clip_uid]
        targets.append(
            ASRV2TargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video_path),
                source_audio_path=str(audio_path),
                source_audio_sha256=source_target.source_audio_sha256,
                source_sample_rate_hz=source_target.source_sample_rate_hz,
                source_channels=source_target.source_channels,
                source_frame_count=source_target.source_frame_count,
                segment_count=len(selected_raw),
            )
        )
        for raw in selected_raw:
            bound = bound_by_key[(clip_uid, raw.segment_id)]
            cluster = cluster_by_key[(clip_uid, raw.speaker_cluster_id)]
            if (
                bound.speaker_cluster_id != raw.speaker_cluster_id
                or bound.start_time != raw.start_time
                or bound.end_time != raw.end_time
                or bound.source_start_sample != raw.source_start_sample
                or bound.source_end_sample != raw.source_end_sample
                or bound.cluster_binding_status != cluster.status
                or bound.entity_id != cluster.entity_id
                or raw.source_audio_path != str(audio_path)
                or raw.source_audio_sha256 != source_target.source_audio_sha256
                or raw.source_sample_rate_hz != source_target.source_sample_rate_hz
                or raw.source_end_sample > source_target.source_frame_count
            ):
                raise ValueError("ASR V2 DiariZen segment join is inconsistent")
            jobs.append(
                ASRV2SegmentJob(
                    target_clip_uid=clip_uid,
                    segment_id=raw.segment_id,
                    speaker_cluster_id=raw.speaker_cluster_id,
                    cluster_binding_status=bound.cluster_binding_status,
                    entity_id=bound.entity_id,
                    entity_occurrence_id=bound.entity_occurrence_id,
                    identity_scope=bound.identity_scope,
                    start_time=raw.start_time,
                    end_time=raw.end_time,
                    source_start_sample=raw.source_start_sample,
                    source_end_sample=raw.source_end_sample,
                    source_audio_path=str(audio_path),
                    source_audio_sha256=source_target.source_audio_sha256,
                    source_sample_rate_hz=source_target.source_sample_rate_hz,
                    source_channels=source_target.source_channels,
                    source_diarization_inventory_fingerprint=(
                        diarization_inventory.inventory_fingerprint
                    ),
                    source_segment_id=raw.segment_id,
                    boundary_reconciliation=raw.boundary_reconciliation,
                )
            )

    values: dict[str, object] = {
        "schema_version": (
            ASR_V2_INVENTORY_SCHEMA_VERSION
            if mode == "pilot20"
            else ASR_V2_PRODUCTION_INVENTORY_SCHEMA_VERSION
        ),
        "mode": mode,
        "source_diarization_root": str(diarization_root),
        "source_diarization_inventory_path": str(source_paths["inventory"]),
        "source_diarization_inventory_sha256": _sha256_file(source_paths["inventory"]),
        "source_diarization_inventory_fingerprint": (
            diarization_inventory.inventory_fingerprint
        ),
        "source_diarization_raw_segments_path": str(source_paths["raw"]),
        "source_diarization_raw_segments_sha256": _sha256_file(source_paths["raw"]),
        "source_diarization_bound_segments_path": str(source_paths["bound"]),
        "source_diarization_bound_segments_sha256": _sha256_file(source_paths["bound"]),
        "source_diarization_cluster_bindings_path": str(source_paths["clusters"]),
        "source_diarization_cluster_bindings_sha256": _sha256_file(
            source_paths["clusters"]
        ),
        "source_diarization_summary_path": str(source_paths["summary"]),
        "source_diarization_summary_sha256": _sha256_file(source_paths["summary"]),
        "source_diarization_mapping_policy_version": (
            diarization_inventory.mapping_policy_version
        ),
        "source_asr_v1_inventory_path": str(asr_v1_paths["inventory"]),
        "source_asr_v1_inventory_sha256": _sha256_file(asr_v1_paths["inventory"]),
        "source_asr_v1_inventory_fingerprint": asr_v1_inventory.inventory_fingerprint,
        "source_asr_v1_turns_path": str(asr_v1_paths["turns"]),
        "source_asr_v1_turns_sha256": _sha256_file(asr_v1_paths["turns"]),
        "source_asr_v1_summary_path": str(asr_v1_paths["summary"]),
        "source_asr_v1_summary_sha256": _sha256_file(asr_v1_paths["summary"]),
        "baseline_asr_v1_backend_provenance": (
            asr_v1_summary.backend_provenance.model_dump(mode="json")
        ),
        "baseline_asr_v1_turn_count": len(asr_v1_records),
        "baseline_asr_v1_turn_median_duration": _median_duration(
            [item.end_time - item.start_time for item in asr_v1_records]
        ),
        "source_target_count": len(diarization_inventory.targets),
        "selected_target_count": len(targets),
        "selected_segment_count": len(jobs),
        "selection_mode": selection_mode,
        "bounded_selection_applied": bounded,
        "parent_quota_applied": False,
        "donor_media_used": False,
        "cross_pair_jobs_created": 0,
        "targets": [item.model_dump(mode="json") for item in targets],
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    if mode == "production":
        values.update(
            {
                "production_inference_enabled": True,
                "asr_v2_policy_validated": True,
                "calibration_inventory_fingerprint": (
                    ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT
                ),
                "calibration_checkpoint_fingerprint": (
                    ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
                ),
                "calibration_human_qa_total": ASR_V2_CALIBRATION_HUMAN_QA_TOTAL,
                "calibration_human_qa_correct": (ASR_V2_CALIBRATION_HUMAN_QA_CORRECT),
                "calibration_human_qa_wrong": ASR_V2_CALIBRATION_HUMAN_QA_WRONG,
                "calibration_human_qa_uncertain": (
                    ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN
                ),
                "calibration_human_qa_unlabeled": (
                    ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED
                ),
                "text_usability_gate_applied": False,
                "transcript_confidence_threshold_used": False,
            }
        )
        inventory_type = ASRV2ProductionInventory
    else:
        values["production_inference_blocked"] = False
        inventory_type = ASRV2Inventory
    return inventory_type(
        **values,
        inventory_fingerprint=_sha256_text(_compact_json(values)),
    )


def _verify_inventory_sources(inventory: ASRV2InventoryRecord) -> None:
    expected = {
        inventory.source_diarization_inventory_path: inventory.source_diarization_inventory_sha256,
        inventory.source_diarization_raw_segments_path: inventory.source_diarization_raw_segments_sha256,
        inventory.source_diarization_bound_segments_path: inventory.source_diarization_bound_segments_sha256,
        inventory.source_diarization_cluster_bindings_path: inventory.source_diarization_cluster_bindings_sha256,
        inventory.source_diarization_summary_path: inventory.source_diarization_summary_sha256,
        inventory.source_asr_v1_inventory_path: inventory.source_asr_v1_inventory_sha256,
        inventory.source_asr_v1_turns_path: inventory.source_asr_v1_turns_sha256,
        inventory.source_asr_v1_summary_path: inventory.source_asr_v1_summary_sha256,
    }
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"ASR V2 source artifact changed: {path.name}")


def asr_v2_request_fingerprint(
    job: ASRV2SegmentJob,
    *,
    backend_provenance: ASRV2BackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "request_contract_version": ASR_V2_REQUEST_CONTRACT_VERSION,
                "backend_configuration_fingerprint": (
                    backend_provenance.configuration_fingerprint
                ),
                "job": job.model_dump(mode="json"),
            }
        )
    )


def _preprocessing(
    job: ASRV2SegmentJob,
) -> tuple[PreparedExactASRWaveform, ASRV2PreprocessingProvenance]:
    prepared = prepare_exact_asr_waveform(job, unit_label="segment")
    provenance = ASRV2PreprocessingProvenance(
        source_sample_rate_hz=prepared.source_sample_rate_hz,
        source_channels=prepared.source_channels,
        resampled=prepared.resampled,
        downmixed=prepared.downmixed,
    )
    return prepared, provenance


def _record(
    *,
    job: ASRV2SegmentJob,
    backend_provenance: ASRV2BackendProvenance,
    preprocessing: ASRV2PreprocessingProvenance,
    result: ASRBackendResult,
) -> ASRV2SegmentRecord:
    text = None if result.text is None else result.text.strip() or None
    status: Literal["transcribed", "uncertain"] = (
        "transcribed" if text is not None else "uncertain"
    )
    warnings = list(result.warnings)
    if status == "uncertain":
        warnings.append("empty_transcript")
    return ASRV2SegmentRecord(
        **job.model_dump(mode="python"),
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        preprocessing=preprocessing,
        request_fingerprint=asr_v2_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status=status,
        text=text,
        language=result.language,
        diagnostics=result.diagnostics,
        warnings=warnings,
    )


def _failed_record(
    *,
    job: ASRV2SegmentJob,
    backend_provenance: ASRV2BackendProvenance,
    failure: ASRTranscriptionFailure,
) -> ASRV2SegmentRecord:
    return ASRV2SegmentRecord(
        **job.model_dump(mode="python"),
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        preprocessing=ASRV2PreprocessingProvenance(
            source_sample_rate_hz=job.source_sample_rate_hz,
            source_channels=job.source_channels,
            resampled=job.source_sample_rate_hz != ASR_INPUT_SAMPLE_RATE_HZ,
            downmixed=job.source_channels != 1,
        ),
        request_fingerprint=asr_v2_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status="failed",
        text=None,
        language=None,
        diagnostics=None,
        warnings=[failure.code],
        failure=ASRFailure(code=failure.code, reason=failure.reason),
    )


def _summary(
    *,
    inventory: ASRV2InventoryRecord,
    records: Sequence[ASRV2SegmentRecord],
    backend_provenance: ASRV2BackendProvenance,
    backend_call_count: int,
    failure_counts: Counter[str],
) -> ASRV2SummaryRecord:
    status_counts = Counter(item.status for item in records)
    binding_counts = Counter(item.cluster_binding_status for item in records)
    scope_counts = Counter(item.identity_scope for item in records)
    durations = [item.end_time - item.start_time for item in records]
    values: dict[str, object] = {
        "mode": inventory.mode,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "source_diarization_inventory_fingerprint": (
            inventory.source_diarization_inventory_fingerprint
        ),
        "source_asr_v1_inventory_fingerprint": (
            inventory.source_asr_v1_inventory_fingerprint
        ),
        "backend_provenance": backend_provenance,
        "source_target_count": inventory.source_target_count,
        "target_clip_count": inventory.selected_target_count,
        "segment_count": len(records),
        "bounded_selection_applied": inventory.bounded_selection_applied,
        "candidate_mapped_segment_count": binding_counts["candidate_mapped"],
        "ambiguous_segment_count": binding_counts["ambiguous"],
        "unbound_segment_count": binding_counts["unbound"],
        "conflict_segment_count": binding_counts["conflict"],
        "direct_anchor_present_segment_count": scope_counts["direct_anchor_present"],
        "cluster_propagated_only_segment_count": scope_counts[
            "cluster_propagated_only"
        ],
        "unresolved_segment_count": scope_counts["unresolved"],
        "transcribed_count": status_counts["transcribed"],
        "uncertain_count": status_counts["uncertain"],
        "failed_count": status_counts["failed"],
        "backend_call_count": backend_call_count,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "total_segment_seconds": float(math.fsum(durations)),
        "segment_median_duration": _median_duration(durations),
        "baseline_asr_v1_turn_count": inventory.baseline_asr_v1_turn_count,
        "baseline_asr_v1_turn_median_duration": (
            inventory.baseline_asr_v1_turn_median_duration
        ),
    }
    if isinstance(inventory, ASRV2ProductionInventory):
        values.update(
            {
                "calibration_inventory_fingerprint": (
                    inventory.calibration_inventory_fingerprint
                ),
                "calibration_checkpoint_fingerprint": (
                    inventory.calibration_checkpoint_fingerprint
                ),
                "asr_v2_policy_validated": True,
                "calibration_human_qa_total": inventory.calibration_human_qa_total,
                "calibration_human_qa_correct": (
                    inventory.calibration_human_qa_correct
                ),
                "calibration_human_qa_wrong": inventory.calibration_human_qa_wrong,
                "calibration_human_qa_uncertain": (
                    inventory.calibration_human_qa_uncertain
                ),
                "calibration_human_qa_unlabeled": (
                    inventory.calibration_human_qa_unlabeled
                ),
                "text_usability_gate_applied": False,
                "transcript_confidence_threshold_used": False,
                "production_inference_enabled": True,
            }
        )
        return ASRV2ProductionSummary(**values)
    return ASRV2Summary(**values)


def _review_segment_name(job: ASRV2SegmentJob) -> str:
    safe = all(
        value and "/" not in value and "\\" not in value and value not in {".", ".."}
        for value in (job.target_clip_uid, job.segment_id)
    )
    if not safe:
        raise ValueError("ASR V2 review segment identity is not path-safe")
    return f"{job.target_clip_uid}/{job.segment_id}.wav"


def _review_video_name(target: ASRV2TargetClip) -> str:
    suffix = Path(target.target_video_path).suffix.lower() or ".media"
    return f"{target.target_clip_uid}.video{suffix}"


def _load_v1_records(inventory: ASRV2InventoryRecord) -> list[ASRTurnRecord]:
    return [
        ASRTurnRecord.model_validate(item)
        for item in _read_jsonl(Path(inventory.source_asr_v1_turns_path))
    ]


def _qa_name(record: ASRV2SegmentRecord) -> str:
    return (
        f"qa-{record.target_clip_uid}-{record.segment_id}-{record.speaker_cluster_id}"
    )


def _qa_key(inventory: ASRV2InventoryRecord, record: ASRV2SegmentRecord) -> str:
    return f"h3-asr-v2-{inventory.inventory_fingerprint}-{_qa_name(record)}"


def _diagnostic_text(diagnostics: ASRDecoderDiagnostics | None) -> str:
    if diagnostics is None:
        return "[unavailable]"
    return (
        f"lang_p={diagnostics.language_probability!r}; "
        f"avg_logprob={diagnostics.avg_log_probability!r}; "
        f"no_speech={diagnostics.no_speech_probability!r}; "
        f"compression={diagnostics.compression_ratio!r}; "
        f"segments={diagnostics.decoder_segment_count}"
    )


def _review_html(
    *,
    inventory: ASRV2InventoryRecord,
    records: Sequence[ASRV2SegmentRecord],
    video_names: dict[str, str],
    segment_names: dict[tuple[str, str], str | None],
    v1_records: Sequence[ASRTurnRecord],
) -> str:
    qa_metadata = {
        "schema_version": ASR_V2_HUMAN_QA_SCHEMA_VERSION,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "mode": inventory.mode,
        "total_segment_count": len(records),
        "allowed_labels": list(ASR_HUMAN_QA_LABELS),
        "export_filename": (
            "asr_v2_pilot20_human_qa.json"
            if inventory.mode == "pilot20"
            else "asr_v2_production_human_qa.json"
        ),
        "segments": [
            {
                "target_clip_uid": item.target_clip_uid,
                "segment_id": item.segment_id,
                "speaker_cluster_id": item.speaker_cluster_id,
                "input_name": _qa_name(item),
                "storage_key": _qa_key(inventory, item),
            }
            for item in records
        ],
    }
    qa_json = _compact_json(qa_metadata).replace("<", "\\u003c")
    v1_by_clip: dict[str, list[ASRTurnRecord]] = defaultdict(list)
    for record in v1_records:
        v1_by_clip[record.target_clip_uid].append(record)
    records_by_clip: dict[str, list[ASRV2SegmentRecord]] = defaultdict(list)
    for record in records:
        records_by_clip[record.target_clip_uid].append(record)
    cards: list[str] = []
    for target in inventory.targets:
        rows: list[str] = []
        for record in records_by_clip[target.target_clip_uid]:
            media_name = segment_names[(record.target_clip_uid, record.segment_id)]
            player = (
                "[unavailable]"
                if media_name is None
                else (
                    "<audio controls preload='metadata' "
                    f"src='review_media/{html.escape(media_name)}'></audio>"
                )
            )
            overlapping = [
                item
                for item in v1_by_clip.get(record.target_clip_uid, [])
                if item.start_time < record.end_time
                and item.end_time > record.start_time
            ]
            v1_reference = (
                "[none]"
                if not overlapping
                else "<br>".join(
                    f"{html.escape(item.turn_id)} {item.start_time:.3f}-{item.end_time:.3f}: "
                    f"{html.escape(item.text or '[null]')} ({html.escape(_diagnostic_text(item.diagnostics))})"
                    for item in overlapping
                )
            )
            qa_name = _qa_name(record)
            qa_key = _qa_key(inventory, record)
            rows.append(
                "<tr>"
                f"<td>{html.escape(record.segment_id)}<br>{html.escape(record.speaker_cluster_id)}</td>"
                f"<td>{html.escape(record.cluster_binding_status)}<br>"
                f"{html.escape(record.entity_id or 'UNRESOLVED')}<br>"
                f"{html.escape(record.identity_scope)}</td>"
                f"<td>{record.start_time:.3f}-{record.end_time:.3f}<br>"
                f"{record.end_time - record.start_time:.3f}s</td>"
                f"<td>{player}</td>"
                f"<td>{html.escape(record.status)}<br>"
                f"{html.escape(record.text or '[null]')}<br>"
                f"lang={html.escape(record.language or '[null]')}<br>"
                f"{html.escape(_diagnostic_text(record.diagnostics))}</td>"
                f"<td><strong>ASR V1 REFERENCE - NOT GROUND TRUTH</strong><br>{v1_reference}</td>"
                "<td>"
                + " ".join(
                    f"<label><input type='radio' name='{html.escape(qa_name)}' "
                    f"data-qa-storage-key='{html.escape(qa_key)}' value='{label}' "
                    f"onchange='saveLabel(this)'>{label}</label>"
                    for label in ASR_HUMAN_QA_LABELS
                )
                + "</td></tr>"
            )
        cards.append(
            f"<article class='case'><h2>{html.escape(target.target_clip_uid)}</h2>"
            f"<video controls preload='metadata' src='review_media/{html.escape(video_names[target.target_clip_uid])}'></video>"
            "<table><thead><tr><th>segment / speaker</th><th>identity</th><th>time</th>"
            "<th>exact crop</th><th>ASR V2</th><th>V1 A/B reference</th><th>human QA</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></article>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>H3 ASR V2 review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f7;color:#171717}}
header,.case{{max-width:1500px;margin:0 auto 24px;background:#fff;border:1px solid #ccc;padding:18px}}
h1,h2{{letter-spacing:0}}video{{width:100%;max-height:520px;background:#111}}audio{{width:100%;min-width:180px}}
table{{width:100%;border-collapse:collapse;margin-top:14px}}th,td{{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}}
label{{display:block;white-space:nowrap}}.qa-controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:14px;padding:12px;border:1px solid #bbb}}
.qa-counts,.qa-actions{{display:flex;flex-wrap:wrap;gap:10px}}.qa-actions{{margin-left:auto}}button{{padding:7px 11px}}
</style></head><body>
<header><h1>H3 Whisper-large-v3 ASR V2 DiariZen-segment review</h1>
<p>V1 baseline: 82 turns; 59 CORRECT, 15 WRONG, 8 UNCERTAIN. V1 text below is A/B reference only, never ground truth.</p>
<section class="qa-controls"><strong id="qa-progress">labeled 0 / 0</strong>
<div class="qa-counts"><span>CORRECT: <strong id="qa-correct-count">0</strong></span>
<span>WRONG: <strong id="qa-wrong-count">0</strong></span>
<span>UNCERTAIN: <strong id="qa-uncertain-count">0</strong></span>
<span>UNLABELED: <strong id="qa-unlabeled-count">0</strong></span></div>
<div class="qa-actions"><button onclick="exportQAJSON()">Export QA JSON</button>
<button onclick="clearQALabels()">Clear QA labels</button></div></section></header>
{"".join(cards)}
<script>
const qaMetadata = {qa_json};
const qaAllowedLabels = new Set(qaMetadata.allowed_labels);
function collectQACounts() {{
  const counts = {{CORRECT:0, WRONG:0, UNCERTAIN:0, UNLABELED:0}};
  qaMetadata.segments.forEach(function(segment) {{
    const label = localStorage.getItem(segment.storage_key);
    if (qaAllowedLabels.has(label)) counts[label] += 1; else counts.UNLABELED += 1;
  }});
  return counts;
}}
function updateQACounts() {{
  const counts = collectQACounts();
  document.getElementById('qa-progress').textContent = 'labeled ' +
    (qaMetadata.total_segment_count - counts.UNLABELED) + ' / ' + qaMetadata.total_segment_count;
  document.getElementById('qa-correct-count').textContent = counts.CORRECT;
  document.getElementById('qa-wrong-count').textContent = counts.WRONG;
  document.getElementById('qa-uncertain-count').textContent = counts.UNCERTAIN;
  document.getElementById('qa-unlabeled-count').textContent = counts.UNLABELED;
}}
function saveLabel(input) {{
  if (!qaAllowedLabels.has(input.value)) return;
  localStorage.setItem(input.dataset.qaStorageKey, input.value); updateQACounts();
}}
function restoreQALabels() {{
  document.querySelectorAll('input[data-qa-storage-key]').forEach(function(input) {{
    const saved = localStorage.getItem(input.dataset.qaStorageKey);
    input.checked = qaAllowedLabels.has(saved) && saved === input.value;
  }}); updateQACounts();
}}
function exportQAJSON() {{
  const labels = [];
  qaMetadata.segments.forEach(function(segment) {{
    const label = localStorage.getItem(segment.storage_key);
    if (qaAllowedLabels.has(label)) labels.push({{
      target_clip_uid: segment.target_clip_uid,
      segment_id: segment.segment_id,
      speaker_cluster_id: segment.speaker_cluster_id,
      label: label
    }});
  }});
  const payload = {{schema_version:qaMetadata.schema_version,
    inventory_fingerprint:qaMetadata.inventory_fingerprint, mode:qaMetadata.mode,
    label_count:labels.length, total_segment_count:qaMetadata.total_segment_count,
    counts:collectQACounts(), labels:labels}};
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\\n'], {{type:'application/json'}});
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = qaMetadata.export_filename; link.click();
  URL.revokeObjectURL(url);
}}
function clearQALabels() {{
  if (!window.confirm('Clear QA labels for this ASR V2 review?')) return;
  qaMetadata.segments.forEach(function(segment) {{localStorage.removeItem(segment.storage_key);}});
  document.querySelectorAll('input[data-qa-storage-key]').forEach(function(input) {{input.checked=false;}});
  updateQACounts();
}}
restoreQALabels();
</script></body></html>"""


def _ensure_symlink(*, destination: Path, source: Path) -> None:
    resolved = source.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"ASR V2 review source is not a file: {resolved}")
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True) == resolved:
                return
        except FileNotFoundError:
            pass
    elif destination.exists():
        raise ValueError(f"ASR V2 review media is not a symlink: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.symlink_to(resolved)
    temporary.replace(destination)


def _materialize_review_video_links(
    *,
    inventory: ASRV2InventoryRecord,
    review_root: Path,
) -> dict[str, str]:
    video_names: dict[str, str] = {}
    for target in inventory.targets:
        name = _review_video_name(target)
        _ensure_symlink(
            destination=review_root / name, source=Path(target.target_video_path)
        )
        video_names[target.target_clip_uid] = name
    return video_names


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"ASR V2 output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def load_asr_v2_inventory(path: Path) -> ASRV2InventoryRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version == ASR_V2_INVENTORY_SCHEMA_VERSION:
        return ASRV2Inventory.model_validate(payload)
    if schema_version == ASR_V2_PRODUCTION_INVENTORY_SCHEMA_VERSION:
        return ASRV2ProductionInventory.model_validate(payload)
    raise ValueError(f"unsupported ASR V2 inventory schema: {schema_version!r}")


def load_asr_v2_summary(path: Path) -> ASRV2SummaryRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version == ASR_V2_SUMMARY_SCHEMA_VERSION:
        return ASRV2Summary.model_validate(payload)
    if schema_version == ASR_V2_PRODUCTION_SUMMARY_SCHEMA_VERSION:
        return ASRV2ProductionSummary.model_validate(payload)
    raise ValueError(f"unsupported ASR V2 summary schema: {schema_version!r}")


def run_asr_v2_transcription(
    *,
    inventory: ASRV2InventoryRecord,
    output_root: Path,
    backend: ASRBackend,
    overwrite: bool = False,
) -> ASRV2SummaryRecord:
    if inventory.inventory_fingerprint != _inventory_fingerprint(inventory):
        raise ValueError("ASR V2 inventory fingerprint is inconsistent")
    _verify_inventory_sources(inventory)
    backend_provenance = build_asr_v2_backend_provenance(
        runtime=backend.provenance,
        baseline=inventory.baseline_asr_v1_backend_provenance,
    )
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[ASRV2SegmentRecord] = []
    failure_counts: Counter[str] = Counter()
    backend_call_count = 0
    try:
        temporary.mkdir()
        review_root = temporary / "review_media"
        review_root.mkdir()
        video_names = _materialize_review_video_links(
            inventory=inventory,
            review_root=review_root,
        )
        segment_names: dict[tuple[str, str], str | None] = {}
        for job in inventory.jobs:
            try:
                prepared, preprocessing = _preprocessing(job)
                segment_name = _review_segment_name(job)
                _write_asr_review_wav(
                    review_root / segment_name,
                    prepared.waveform,
                )
                segment_names[(job.target_clip_uid, job.segment_id)] = segment_name
                backend_call_count += 1
                result = backend.transcribe(
                    audio=prepared.waveform,
                    sample_rate_hz=ASR_INPUT_SAMPLE_RATE_HZ,
                )
                record = _record(
                    job=job,
                    backend_provenance=backend_provenance,
                    preprocessing=preprocessing,
                    result=result,
                )
            except ASRTranscriptionFailure as exc:
                segment_names[(job.target_clip_uid, job.segment_id)] = None
                failure_counts.update([exc.code])
                record = _failed_record(
                    job=job,
                    backend_provenance=backend_provenance,
                    failure=exc,
                )
            records.append(record)
        summary = _summary(
            inventory=inventory,
            records=records,
            backend_provenance=backend_provenance,
            backend_call_count=backend_call_count,
            failure_counts=failure_counts,
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "segments.jsonl", records)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "review.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                video_names=video_names,
                segment_names=segment_names,
                v1_records=_load_v1_records(inventory),
            ),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_review_source(
    *,
    inventory: ASRV2InventoryRecord,
    records: Sequence[ASRV2SegmentRecord],
    summary: ASRV2SummaryRecord,
    expected_mode: Literal["pilot20", "production"],
) -> None:
    if inventory.inventory_fingerprint != _inventory_fingerprint(inventory):
        raise ValueError("stored ASR V2 inventory fingerprint is inconsistent")
    if inventory.mode != expected_mode or summary.mode != expected_mode:
        raise ValueError("stored ASR V2 review mode does not match request")
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("stored ASR V2 summary fingerprint is inconsistent")
    if len(records) != len(inventory.jobs) or summary.segment_count != len(records):
        raise ValueError("stored ASR V2 segment count is inconsistent")
    job_fields = tuple(ASRV2SegmentJob.model_fields)
    for job, record in zip(inventory.jobs, records, strict=True):
        record_job = ASRV2SegmentJob.model_validate(
            {field: getattr(record, field) for field in job_fields}
        )
        if record_job != job:
            raise ValueError("stored ASR V2 record does not match inventory order")


def regenerate_asr_v2_review(
    *,
    output_root: Path,
    expected_mode: Literal["pilot20", "production"],
) -> dict[str, object]:
    root = output_root.expanduser().resolve(strict=True)
    inventory = load_asr_v2_inventory(root / "inventory.json")
    records = [
        ASRV2SegmentRecord.model_validate(item)
        for item in _read_jsonl(root / "segments.jsonl")
    ]
    summary = load_asr_v2_summary(root / "summary.json")
    _validate_review_source(
        inventory=inventory,
        records=records,
        summary=summary,
        expected_mode=expected_mode,
    )
    _verify_inventory_sources(inventory)
    review_root = root / "review_media"
    review_root.mkdir(exist_ok=True)
    review_root = review_root.resolve(strict=True)
    review_root.relative_to(root)
    video_names: dict[str, str] = {}
    for target in inventory.targets:
        name = _review_video_name(target)
        _ensure_symlink(
            destination=review_root / name, source=Path(target.target_video_path)
        )
        video_names[target.target_clip_uid] = name
    regenerated = 0
    segment_names: dict[tuple[str, str], str | None] = {}
    for job in inventory.jobs:
        name = _review_segment_name(job)
        path = review_root / name
        if not path.is_file():
            if path.exists() or path.is_symlink():
                raise ValueError(f"ASR V2 segment review media is invalid: {path}")
            prepared, _ = _preprocessing(job)
            temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
            _write_asr_review_wav(temporary, prepared.waveform)
            temporary.replace(path)
            regenerated += 1
        segment_names[(job.target_clip_uid, job.segment_id)] = name
    review_path = root / "review.html"
    temporary_review = review_path.with_name(
        f".{review_path.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        temporary_review.write_text(
            _review_html(
                inventory=inventory,
                records=records,
                video_names=video_names,
                segment_names=segment_names,
                v1_records=_load_v1_records(inventory),
            ),
            encoding="utf-8",
        )
        temporary_review.replace(review_path)
    finally:
        if temporary_review.exists():
            temporary_review.unlink()
    return {
        "mode": inventory.mode,
        "review_regenerated": True,
        "model_loaded": False,
        "backend_calls": 0,
        "segment_count": len(records),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "regenerated_segment_media_count": regenerated,
    }


def asr_v2_output_root(
    audio_run_root: Path,
    *,
    mode: Literal["pilot20", "production"],
) -> Path:
    root = audio_run_root.expanduser().resolve(strict=False)
    if mode == "pilot20":
        return root / "asr_v2_pilot20"
    return root / "production" / "asr_v2"


def asr_v2_stage_is_complete(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file()
        for name in ("inventory.json", "segments.jsonl", "summary.json", "review.html")
    )
