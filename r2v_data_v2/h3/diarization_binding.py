from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import select
import shutil
import subprocess
import uuid
import wave
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import IO, Literal, Protocol, Self

from pydantic import Field, StrictFloat, model_validator

from r2v_data_v2.h3.asr_transcription import (
    PILOT_TARGET_COUNT,
    ASRInventory,
)
from r2v_data_v2.h3.asr_transcription import (
    _inventory_fingerprint as _asr_inventory_fingerprint,
)
from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    coalesce_audio_bindings,
)
from r2v_data_v2.h3.audio_production import H3ProductionInPair
from r2v_data_v2.h3.schemas import (
    AudioBindingSidecar,
    AudioEntityBinding,
    SchemaModel,
)

DIARIZATION_INVENTORY_VERSION = "r2v.h3.diarization_inventory.4"
DIARIZATION_SEGMENT_VERSION = "r2v.h3.diarization_segment.3"
DIARIZATION_CLUSTER_BINDING_VERSION = "r2v.h3.diarization_cluster_binding.3"
DIARIZATION_BOUND_SEGMENT_VERSION = "r2v.h3.diarization_bound_segment.2"
DIARIZATION_CLIP_RESULT_VERSION = "r2v.h3.diarization_clip_result.2"
DIARIZATION_SUMMARY_VERSION = "r2v.h3.diarization_summary.4"
DIARIZATION_HUMAN_QA_VERSION = "r2v.h3.diarization_human_qa.1"
DIARIZATION_MAPPING_POLICY_VERSION = "h3_diarizen_sparse_anchor_policy_v1"
DIARIZATION_REQUEST_VERSION = "h3_diarizen_clip_diarization_v2"
DIARIZATION_CANONICAL_PREPROCESSING_VERSION = (
    "h3_diarizen_torchaudio_kaiser_32k_stereo_to_16k_mono_v1"
)
DIARIZATION_LEGACY_PREPROCESSING_VERSION = (
    "h3_diarizen_native_16k_mono_passthrough_v1"
)
DIARIZATION_PREPROCESSING_VERSION = DIARIZATION_CANONICAL_PREPROCESSING_VERSION
DIARIZATION_BOUNDARY_POLICY_VERSION = "canonical_source_intersection_v1"
DIARIZATION_CALIBRATION_INVENTORY_FINGERPRINT = (
    "776761abc1ffa1822766eb29c1ecf61f9e32beda35f2246cb3ef6dc3f096e7b7"
)
DIARIZATION_CALIBRATION_SOURCE_ASR_FINGERPRINT = (
    "ead8ce8aad5dc587517c4d38e74962152fbae96721fe1f92797b832d648c6a75"
)
DEFAULT_DIARIZEN_MODEL_IDENTIFIER = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DEFAULT_DIARIZEN_DEVICE = "cuda:0"
DEFAULT_DIARIZEN_TIMEOUT_SECONDS = 900.0
EXPECTED_PRODUCTION_TARGET_COUNT = 75
DIARIZATION_HUMAN_QA_LABELS = ("CORRECT", "WRONG", "UNCERTAIN")

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

DiarizationInputProfile = Literal["canonical_32k_stereo", "legacy_16k_mono"]


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            raise TypeError(f"diarization JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


class DiarizationVisualReference(SchemaModel):
    entity_id: str
    image_path: str

    @model_validator(mode="after")
    def validate_reference(self) -> DiarizationVisualReference:
        if not self.entity_id.strip() or not self.image_path.strip():
            raise ValueError("diarization visual reference must be complete")
        return self


class DiarizationTargetClip(SchemaModel):
    target_clip_uid: str
    target_video_path: str
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    source_frame_count: int = Field(gt=0)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    visual_references: list[DiarizationVisualReference]

    @model_validator(mode="after")
    def validate_target(self) -> DiarizationTargetClip:
        if _SAFE_ID.fullmatch(self.target_clip_uid) is None:
            raise ValueError("diarization target clip UID is not path-safe")
        for value in (self.target_video_path, self.source_audio_path):
            if not value.strip():
                raise ValueError("diarization target provenance path is empty")
        if (
            self.target_audio_binding_path is not None
            and not self.target_audio_binding_path.strip()
        ):
            raise ValueError("diarization Audio binding path must be non-empty or null")
        if self.target_audio_binding_path is None and (
            self.target_audio_binding_sha256 is not None
        ):
            raise ValueError("diarization Audio binding hash requires a path")
        entity_ids = [item.entity_id for item in self.visual_references]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("diarization visual reference entities must be unique")
        return self


class DiarizationInventory(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_inventory.4"] = (
        DIARIZATION_INVENTORY_VERSION
    )
    mode: Literal["pilot20", "production"]
    source_inventory_kind: Literal[
        "pair_inventory", "canonical_audio_manifest"
    ] = "pair_inventory"
    source_pairs_path: str | None = None
    source_pairs_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_visual_production_root: str | None = None
    source_visual_inventory_path: str | None = None
    source_visual_inventory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_canonical_audio_manifest_path: str | None = None
    source_canonical_audio_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_asr_inventory_path: str | None = None
    source_asr_inventory_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    mapping_policy_version: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        DIARIZATION_MAPPING_POLICY_VERSION
    )
    mapping_policy_validated: Literal[True] = True
    numeric_mapping_thresholds_used: Literal[False] = False
    calibration_inventory_fingerprint: Literal[
        "776761abc1ffa1822766eb29c1ecf61f9e32beda35f2246cb3ef6dc3f096e7b7"
    ] = DIARIZATION_CALIBRATION_INVENTORY_FINGERPRINT
    calibration_source_asr_inventory_fingerprint: Literal[
        "ead8ce8aad5dc587517c4d38e74962152fbae96721fe1f92797b832d648c6a75"
    ] = DIARIZATION_CALIBRATION_SOURCE_ASR_FINGERPRINT
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_count: int = Field(ge=0)
    selected_target_count: int = Field(ge=0)
    selection_mode: Literal[
        "same_ordered_asr_pilot20_targets_v1",
        "complete_in_pair_target_inventory_v1",
        "canonical_visual_target_inventory_v1",
    ]
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    cross_pair_jobs_created: Literal[0] = 0
    production_blocked: Literal[False] = False
    targets: list[DiarizationTargetClip]

    @model_validator(mode="after")
    def validate_inventory(self) -> DiarizationInventory:
        target_ids = [item.target_clip_uid for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("diarization inventory targets must be unique")
        if self.selected_target_count != len(self.targets):
            raise ValueError("diarization selected target count is inconsistent")
        if self.source_target_count < self.selected_target_count:
            raise ValueError("diarization selection cannot exceed source targets")
        canonical_fields = (
            self.source_visual_production_root,
            self.source_visual_inventory_path,
            self.source_visual_inventory_sha256,
            self.source_canonical_audio_manifest_path,
            self.source_canonical_audio_manifest_sha256,
        )
        if self.source_inventory_kind == "pair_inventory":
            if self.source_pairs_path is None or self.source_pairs_sha256 is None:
                raise ValueError("pair-rooted diarization inventory requires pair provenance")
            if any(value is not None for value in canonical_fields):
                raise ValueError("pair-rooted diarization cannot claim canonical provenance")
        elif (
            self.source_pairs_path is not None
            or self.source_pairs_sha256 is not None
            or any(value is None for value in canonical_fields)
        ):
            raise ValueError(
                "canonical diarization inventory requires only Visual/Audio provenance"
            )
        if self.mode == "pilot20":
            if (
                self.source_inventory_kind != "pair_inventory"
                or self.selection_mode != "same_ordered_asr_pilot20_targets_v1"
                or self.selected_target_count != PILOT_TARGET_COUNT
                or self.source_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
                or not self.bounded_selection_applied
                or self.source_asr_inventory_path is None
                or self.source_asr_inventory_fingerprint is None
            ):
                raise ValueError("diarization pilot must reuse the frozen ASR pilot20")
        elif (
            self.selection_mode
            not in {
                "complete_in_pair_target_inventory_v1",
                "canonical_visual_target_inventory_v1",
            }
            or self.selected_target_count != self.source_target_count
            or self.bounded_selection_applied
            or self.source_asr_inventory_path is not None
            or self.source_asr_inventory_fingerprint is not None
        ):
            raise ValueError(
                "production diarization inventory must independently be complete"
            )
        elif (
            self.source_inventory_kind == "canonical_audio_manifest"
        ) != (self.selection_mode == "canonical_visual_target_inventory_v1"):
            raise ValueError("production diarization source kind and selection differ")
        if self.source_inventory_kind == "canonical_audio_manifest" and any(
            (target.target_audio_binding_path is None)
            != (target.target_audio_binding_sha256 is None)
            for target in self.targets
        ):
            raise ValueError("canonical diarization binding path/hash must be paired")
        return self


class DiarizationBackendProvenance(SchemaModel):
    backend: str
    model_identifier: str
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_contract_version: Literal["h3_diarizen_clip_diarization_v2"] = (
        DIARIZATION_REQUEST_VERSION
    )
    input_profile: DiarizationInputProfile
    input_preprocessing: Literal[
        "h3_diarizen_torchaudio_kaiser_32k_stereo_to_16k_mono_v1",
        "h3_diarizen_native_16k_mono_passthrough_v1",
    ]
    source_sample_rate_hz: Literal[16000, 32000]
    source_channels: Literal[1, 2]
    model_input_sample_rate_hz: Literal[16000] = 16000
    model_input_channels: Literal[1] = 1

    @model_validator(mode="after")
    def validate_provenance(self) -> DiarizationBackendProvenance:
        if not self.backend.strip() or not self.model_identifier.strip():
            raise ValueError("diarization backend provenance must not be empty")
        expected = {
            "canonical_32k_stereo": (
                32000,
                2,
                DIARIZATION_CANONICAL_PREPROCESSING_VERSION,
            ),
            "legacy_16k_mono": (
                16000,
                1,
                DIARIZATION_LEGACY_PREPROCESSING_VERSION,
            ),
        }[self.input_profile]
        if (
            self.source_sample_rate_hz,
            self.source_channels,
            self.input_preprocessing,
        ) != expected:
            raise ValueError("diarization input profile provenance is inconsistent")
        return self


class DiarizationBackendSegment(SchemaModel):
    start_time: StrictFloat = Field(ge=0)
    end_time: StrictFloat = Field(gt=0)
    speaker_label: str

    @model_validator(mode="after")
    def validate_segment(self) -> DiarizationBackendSegment:
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("diarization backend segment times must be finite")
        if self.end_time <= self.start_time or not self.speaker_label.strip():
            raise ValueError("diarization backend segment is invalid")
        return self


class DiarizationBoundaryReconciliation(SchemaModel):
    policy_version: Literal["canonical_source_intersection_v1"] = (
        DIARIZATION_BOUNDARY_POLICY_VERSION
    )
    adjusted: bool
    end_clamped: bool
    end_overrun_samples: int = Field(ge=0)
    end_overrun_seconds: float = Field(ge=0)
    reason: Literal["end_clamped_to_canonical_source"] | None = None

    @model_validator(mode="after")
    def validate_reconciliation(self) -> DiarizationBoundaryReconciliation:
        if self.adjusted:
            if (
                not self.end_clamped
                or self.end_overrun_samples <= 0
                or self.end_overrun_seconds <= 0
                or self.reason != "end_clamped_to_canonical_source"
            ):
                raise ValueError("adjusted diarization boundary evidence is incomplete")
        elif (
            self.end_clamped
            or self.end_overrun_samples != 0
            or self.end_overrun_seconds != 0
            or self.reason is not None
        ):
            raise ValueError(
                "unchanged diarization boundary cannot claim reconciliation"
            )
        return self


class RawDiarizationSegment(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_segment.3"] = (
        DIARIZATION_SEGMENT_VERSION
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    backend_speaker_label: str
    backend_reported_start_time: float = Field(ge=0)
    backend_reported_end_time: float = Field(gt=0)
    backend_reported_start_sample: int = Field(ge=0)
    backend_reported_end_sample: int = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    backend: str
    model_identifier: str
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_preprocessing: Literal[
        "h3_diarizen_torchaudio_kaiser_32k_stereo_to_16k_mono_v1",
        "h3_diarizen_native_16k_mono_passthrough_v1",
    ]
    boundary_reconciliation: DiarizationBoundaryReconciliation

    @model_validator(mode="after")
    def validate_times(self) -> RawDiarizationSegment:
        if (
            not math.isfinite(self.backend_reported_start_time)
            or not math.isfinite(self.backend_reported_end_time)
            or self.backend_reported_end_time <= self.backend_reported_start_time
        ):
            raise ValueError("backend-reported diarization interval is invalid")
        if self.end_time <= self.start_time:
            raise ValueError("raw diarization segment must have positive duration")
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("raw diarization sample range must be positive")
        if self.source_start_sample != round(
            self.start_time * self.source_sample_rate_hz
        ) or self.source_end_sample != round(
            self.end_time * self.source_sample_rate_hz
        ):
            raise ValueError("raw diarization samples must match source times")
        if self.backend_reported_start_sample != round(
            self.backend_reported_start_time * self.source_sample_rate_hz
        ) or self.backend_reported_end_sample != round(
            self.backend_reported_end_time * self.source_sample_rate_hz
        ):
            raise ValueError("backend-reported samples must match backend times")
        if self.source_start_sample != self.backend_reported_start_sample:
            raise ValueError("boundary reconciliation must not shift segment start")
        reconciliation = self.boundary_reconciliation
        if reconciliation.adjusted:
            if (
                self.backend_reported_end_sample - self.source_end_sample
                != reconciliation.end_overrun_samples
                or not math.isclose(
                    reconciliation.end_overrun_seconds,
                    reconciliation.end_overrun_samples / self.source_sample_rate_hz,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("diarization end-clamp provenance is inconsistent")
        elif (
            self.backend_reported_start_sample != self.source_start_sample
            or self.backend_reported_end_sample != self.source_end_sample
        ):
            raise ValueError("unchanged diarization segment must preserve samples")
        return self


class DiarizationEntitySupport(SchemaModel):
    entity_id: str
    direct_support_samples: int = Field(gt=0)
    direct_support_seconds: float = Field(gt=0)
    weighted_support: float = Field(gt=0)
    contributing_binding_count: int = Field(gt=0)


class DiarizationClusterBinding(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_cluster_binding.3"] = (
        DIARIZATION_CLUSTER_BINDING_VERSION
    )
    target_clip_uid: str
    speaker_cluster_id: str
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    status: Literal["candidate_mapped", "unbound", "ambiguous", "conflict"]
    entity_id: str | None = None
    cluster_segment_count: int = Field(gt=0)
    cluster_speaker_seconds: float = Field(gt=0)
    usable_anchor_sample_count: int = Field(ge=0)
    usable_anchor_duration: float = Field(ge=0)
    contested_anchor_sample_count: int = Field(ge=0)
    contested_anchor_duration: float = Field(ge=0)
    unmatched_anchor_sample_count: int = Field(ge=0)
    unmatched_anchor_duration: float = Field(ge=0)
    entity_supports: list[DiarizationEntitySupport]
    top1_entity_id: str | None = None
    top1_support: float = Field(ge=0)
    top2_support: float = Field(ge=0)
    top1_share: float | None = Field(default=None, ge=0, le=1)
    top1_top2_margin: float | None = Field(default=None, ge=0)
    visual_anchor_coverage_ratio: float = Field(ge=0, le=1)
    visual_anchor_coverage_is_diagnostic_only: Literal[True] = True
    mapping_policy_version: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        DIARIZATION_MAPPING_POLICY_VERSION
    )
    warnings: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_binding(self) -> DiarizationClusterBinding:
        if (self.status == "candidate_mapped") != (self.entity_id is not None):
            raise ValueError("only candidate_mapped clusters may publish an entity")
        if self.entity_supports != sorted(
            self.entity_supports,
            key=lambda item: (
                -item.direct_support_samples,
                -item.weighted_support,
                item.entity_id,
            ),
        ):
            raise ValueError("cluster entity supports must use deterministic order")
        if self.entity_supports:
            if self.top1_entity_id != self.entity_supports[0].entity_id:
                raise ValueError("cluster top1 entity must match ordered supports")
        elif self.top1_entity_id is not None or self.top1_share is not None:
            raise ValueError("unsupported cluster cannot claim top entity statistics")
        return self


class BoundDiarizationSegment(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_bound_segment.2"] = (
        DIARIZATION_BOUND_SEGMENT_VERSION
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    cluster_binding_status: Literal[
        "candidate_mapped", "unbound", "ambiguous", "conflict"
    ]
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    direct_anchor_samples: int = Field(ge=0)
    direct_anchor_seconds: float = Field(ge=0)
    identity_scope: Literal[
        "direct_anchor_present", "cluster_propagated_only", "unresolved"
    ]

    @model_validator(mode="after")
    def validate_binding(self) -> BoundDiarizationSegment:
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("bound diarization sample range must be positive")
        if self.source_start_sample != round(
            self.start_time * self.source_sample_rate_hz
        ) or self.source_end_sample != round(
            self.end_time * self.source_sample_rate_hz
        ):
            raise ValueError("bound diarization samples must match source time domain")
        mapped = self.cluster_binding_status == "candidate_mapped"
        if mapped != (self.entity_id is not None):
            raise ValueError("bound diarization entity must match cluster status")
        expected_occurrence = (
            f"{self.target_clip_uid}/{self.entity_id}"
            if self.entity_id is not None
            else None
        )
        if self.entity_occurrence_id != expected_occurrence:
            raise ValueError("bound diarization occurrence identity is inconsistent")
        expected_scope = (
            "unresolved"
            if not mapped
            else "direct_anchor_present"
            if self.direct_anchor_samples > 0
            else "cluster_propagated_only"
        )
        if self.identity_scope != expected_scope:
            raise ValueError("bound diarization identity scope is inconsistent")
        return self


class DiarizationClipResult(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_clip_result.2"] = (
        DIARIZATION_CLIP_RESULT_VERSION
    )
    target_clip_uid: str
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    status: Literal["ready", "empty", "failed"]
    backend_called: bool
    raw_segment_count: int = Field(ge=0)
    speaker_cluster_count: int = Field(ge=0)
    legacy_bound_interval_count: int = Field(ge=0)
    legacy_coalesced_bound_turn_count: int = Field(ge=0)
    legacy_lr_asd_bound_samples: int = Field(ge=0)
    usable_direct_anchor_samples: int = Field(ge=0)
    contested_anchor_samples: int = Field(ge=0)
    unmatched_anchor_samples: int = Field(ge=0)
    legacy_bound_turn_durations: list[float] = Field(default_factory=list)
    diarization_segment_durations: list[float] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> DiarizationClipResult:
        if self.status == "failed":
            if self.reason is None or not self.reason.strip():
                raise ValueError("failed diarization clip requires a reason")
        elif self.reason is not None:
            raise ValueError("non-failed diarization clip cannot have a reason")
        if self.status == "empty" and self.raw_segment_count != 0:
            raise ValueError("empty diarization clip cannot contain segments")
        return self


class DiarizationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_summary.4"] = (
        DIARIZATION_SUMMARY_VERSION
    )
    mode: Literal["pilot20", "production"]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_inventory_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    mapping_policy_version: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        DIARIZATION_MAPPING_POLICY_VERSION
    )
    mapping_policy_validated: Literal[True] = True
    numeric_mapping_thresholds_used: Literal[False] = False
    calibration_inventory_fingerprint: Literal[
        "776761abc1ffa1822766eb29c1ecf61f9e32beda35f2246cb3ef6dc3f096e7b7"
    ] = DIARIZATION_CALIBRATION_INVENTORY_FINGERPRINT
    calibration_source_asr_inventory_fingerprint: Literal[
        "ead8ce8aad5dc587517c4d38e74962152fbae96721fe1f92797b832d648c6a75"
    ] = DIARIZATION_CALIBRATION_SOURCE_ASR_FINGERPRINT
    backend_provenance: DiarizationBackendProvenance
    target_clip_count: int = Field(ge=0)
    ready_clip_count: int = Field(ge=0)
    empty_clip_count: int = Field(ge=0)
    failed_clip_count: int = Field(ge=0)
    backend_call_count: int = Field(ge=0)
    raw_segment_count: int = Field(ge=0)
    speaker_cluster_count: int = Field(ge=0)
    candidate_mapped_cluster_count: int = Field(ge=0)
    unbound_cluster_count: int = Field(ge=0)
    ambiguous_cluster_count: int = Field(ge=0)
    conflict_cluster_count: int = Field(ge=0)
    diarized_speaker_seconds: float = Field(ge=0)
    diarized_wallclock_speech_seconds: float = Field(ge=0)
    legacy_lr_asd_bound_seconds: float = Field(ge=0)
    usable_direct_anchor_seconds: float = Field(ge=0)
    contested_anchor_seconds: float = Field(ge=0)
    mapped_speaker_seconds: float = Field(ge=0)
    mapped_direct_anchor_speaker_seconds: float = Field(ge=0)
    identity_propagated_speaker_seconds: float = Field(ge=0)
    fully_propagated_segment_speaker_seconds: float = Field(ge=0)
    unbound_speaker_seconds: float = Field(ge=0)
    ambiguous_speaker_seconds: float = Field(ge=0)
    conflict_speaker_seconds: float = Field(ge=0)
    legacy_coalesced_bound_turn_count: int = Field(ge=0)
    legacy_bound_turn_median_duration: float | None = Field(default=None, ge=0)
    diarization_segment_median_duration: float | None = Field(default=None, ge=0)
    boundary_adjusted_segment_count: int = Field(ge=0)
    boundary_adjusted_clip_count: int = Field(ge=0)
    end_clamped_segment_count: int = Field(ge=0)
    total_end_overrun_seconds: float = Field(ge=0)
    max_end_overrun_seconds: float = Field(ge=0)
    max_end_overrun_samples: int = Field(ge=0)
    median_positive_end_overrun_seconds: float | None = Field(default=None, gt=0)
    failure_reason_counts: dict[str, int]
    thresholds_calibrated: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    transitive_clustering_performed: Literal[False] = False
    visual_anchor_coverage_used_as_gate: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> DiarizationSummary:
        if (self.mode == "pilot20") != (
            self.source_asr_inventory_fingerprint is not None
        ):
            raise ValueError("summary ASR pilot provenance must match its mode")
        if self.target_clip_count != (
            self.ready_clip_count + self.empty_clip_count + self.failed_clip_count
        ):
            raise ValueError("diarization clip counts must reconcile")
        if self.speaker_cluster_count != (
            self.candidate_mapped_cluster_count
            + self.unbound_cluster_count
            + self.ambiguous_cluster_count
            + self.conflict_cluster_count
        ):
            raise ValueError("diarization cluster counts must reconcile")
        if not math.isclose(
            self.mapped_speaker_seconds,
            self.mapped_direct_anchor_speaker_seconds
            + self.identity_propagated_speaker_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("mapped speaker identity seconds must reconcile")
        if (
            self.fully_propagated_segment_speaker_seconds
            > self.identity_propagated_speaker_seconds + 1e-9
        ):
            raise ValueError("fully propagated segments exceed propagated identity")
        if self.boundary_adjusted_segment_count != self.end_clamped_segment_count:
            raise ValueError("boundary-adjusted and end-clamped counts must reconcile")
        if self.boundary_adjusted_clip_count > self.boundary_adjusted_segment_count:
            raise ValueError("adjusted clip count cannot exceed adjusted segments")
        if self.boundary_adjusted_segment_count == 0:
            if (
                self.total_end_overrun_seconds != 0
                or self.max_end_overrun_seconds != 0
                or self.max_end_overrun_samples != 0
                or self.median_positive_end_overrun_seconds is not None
            ):
                raise ValueError("empty boundary diagnostics must be zero")
        elif (
            self.total_end_overrun_seconds <= 0
            or self.max_end_overrun_seconds <= 0
            or self.max_end_overrun_samples <= 0
            or self.median_positive_end_overrun_seconds is None
        ):
            raise ValueError("adjusted boundary diagnostics must be positive")
        return self


class DiarizationHumanQALabel(SchemaModel):
    target_clip_uid: str
    speaker_cluster_id: str
    predicted_binding_status: Literal[
        "candidate_mapped", "unbound", "ambiguous", "conflict"
    ]
    predicted_entity_id: str | None = None
    label: Literal["CORRECT", "WRONG", "UNCERTAIN"]


class DiarizationHumanQAExport(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_human_qa.1"] = (
        DIARIZATION_HUMAN_QA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20"] = "pilot20"
    label_count: int = Field(ge=0)
    total_cluster_count: int = Field(ge=0)
    counts: dict[str, int]
    labels: list[DiarizationHumanQALabel]

    @model_validator(mode="after")
    def validate_export(self) -> DiarizationHumanQAExport:
        allowed_count_keys = {*DIARIZATION_HUMAN_QA_LABELS, "UNLABELED"}
        if set(self.counts) != allowed_count_keys:
            raise ValueError("diarization QA counts use unknown labels")
        if self.label_count != len(self.labels) or self.label_count != sum(
            self.counts[label] for label in DIARIZATION_HUMAN_QA_LABELS
        ):
            raise ValueError("diarization QA label counts are inconsistent")
        if self.total_cluster_count != self.label_count + self.counts["UNLABELED"]:
            raise ValueError("diarization QA total count is inconsistent")
        order = [
            (item.target_clip_uid, item.speaker_cluster_id) for item in self.labels
        ]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("diarization QA labels must be unique and ordered")
        return self


class DiarizationBackendFailure(RuntimeError):
    pass


class DiarizationBackend(Protocol):
    provenance: DiarizationBackendProvenance

    def diarize(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
    ) -> list[DiarizationBackendSegment]: ...


class PersistentDiariZenBackend:
    """Long-lived JSONL bridge that loads DiariZen once per stage run."""

    def __init__(
        self,
        *,
        executable: list[str],
        provenance: DiarizationBackendProvenance,
        timeout_seconds: float,
        diagnostics_root: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not executable or timeout_seconds <= 0:
            raise ValueError("persistent DiariZen backend needs command and timeout")
        executable_path = Path(executable[0]).expanduser()
        if executable_path.parent != Path(".") and not executable_path.exists():
            raise FileNotFoundError(f"DiariZen Python is missing: {executable[0]}")
        self.executable = list(executable)
        self.provenance = provenance
        self.timeout_seconds = timeout_seconds
        self.diagnostics_root = diagnostics_root
        self.environment = environment or {}
        self._process: subprocess.Popen[str] | None = None
        self._stderr_stream: IO[str] | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()

    def _readline(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("DiariZen worker is not running")
        ready, _, _ = select.select(
            [self._process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            raise TimeoutError("DiariZen worker response timed out")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(
                "DiariZen worker exited without response "
                f"(returncode={self._process.poll()})"
            )
        return line

    def start(self) -> None:
        if self._process is not None:
            return
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        stderr_stream = (self.diagnostics_root / "worker.stderr.log").open(
            "w", encoding="utf-8"
        )
        try:
            process = subprocess.Popen(
                self.executable,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_stream,
                text=True,
                bufsize=1,
                env={**os.environ, **self.environment},
            )
        except Exception:
            stderr_stream.close()
            raise
        if process.stdin is None or process.stdout is None:
            process.terminate()
            stderr_stream.close()
            raise RuntimeError("DiariZen worker pipes are unavailable")
        self._process = process
        self._stderr_stream = stderr_stream
        try:
            ready = json.loads(self._readline())
        except Exception:
            self.close()
            raise
        if (
            not isinstance(ready, dict)
            or ready.get("request_id") != "startup"
            or ready.get("status") != "ready"
            or ready.get("model_identifier") != self.provenance.model_identifier
            or ready.get("model_fingerprint") != self.provenance.model_fingerprint
        ):
            self.close()
            raise RuntimeError("DiariZen worker failed startup validation")

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        request_id = str(payload["request_id"])
        try:
            self._process.stdin.write(_compact_json(payload) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DiarizationBackendFailure(
                "persistent DiariZen worker request failed"
            ) from exc
        try:
            response = json.loads(self._readline())
        except json.JSONDecodeError as exc:
            raise DiarizationBackendFailure(
                "persistent DiariZen worker returned invalid JSON"
            ) from exc
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise DiarizationBackendFailure(
                "persistent DiariZen worker request_id mismatch"
            )
        return response

    def diarize(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
    ) -> list[DiarizationBackendSegment]:
        request_id = _sha256_text(f"diarize\0{clip_uid}")[:20]
        response = self._request(
            {
                "request_id": request_id,
                "operation": "diarize",
                "clip_uid": clip_uid,
                "audio_path": str(audio_path),
                "model_identifier": self.provenance.model_identifier,
                "input_profile": self.provenance.input_profile,
            }
        )
        if response.get("status") != "ready":
            raise DiarizationBackendFailure(
                str(response.get("reason") or "diarization_runtime_failed")
            )
        if response.get("model_identifier") != self.provenance.model_identifier:
            raise DiarizationBackendFailure("DiariZen model identifier mismatch")
        metadata = response.get("backend_metadata")
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != expected
            for key, expected in (
                ("input_profile", self.provenance.input_profile),
                ("input_preprocessing", self.provenance.input_preprocessing),
                ("source_sample_rate_hz", self.provenance.source_sample_rate_hz),
                ("source_channels", self.provenance.source_channels),
                (
                    "model_input_sample_rate_hz",
                    self.provenance.model_input_sample_rate_hz,
                ),
                ("model_input_channels", self.provenance.model_input_channels),
            )
        ):
            raise DiarizationBackendFailure(
                "DiariZen runtime preprocessing provenance mismatch"
            )
        segments = response.get("segments")
        if not isinstance(segments, list):
            raise DiarizationBackendFailure("DiariZen worker segments are invalid")
        return [DiarizationBackendSegment.model_validate(item) for item in segments]

    def copy_diagnostics(self, destination: Path) -> None:
        if self._stderr_stream is not None:
            self._stderr_stream.flush()
        source = self.diagnostics_root / "worker.stderr.log"
        if source.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination / source.name)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    response = self._request(
                        {"request_id": "shutdown", "operation": "shutdown"}
                    )
                    if response.get("status") != "shutdown":
                        raise RuntimeError("DiariZen worker refused shutdown")
                except (
                    BrokenPipeError,
                    OSError,
                    RuntimeError,
                    TimeoutError,
                    ValueError,
                ):
                    process.terminate()
                try:
                    process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=self.timeout_seconds)
        finally:
            if process.stdin is not None:
                with suppress(BrokenPipeError, OSError):
                    process.stdin.close()
            if process.stdout is not None:
                with suppress(BrokenPipeError, OSError):
                    process.stdout.close()
            if self._stderr_stream is not None:
                with suppress(BrokenPipeError, OSError):
                    self._stderr_stream.close()
            self._process = None
            self._stderr_stream = None


def diarization_output_root(
    audio_run_root: Path,
    *,
    mode: Literal["pilot20", "production"],
) -> Path:
    root = audio_run_root.expanduser().resolve(strict=False)
    if mode == "pilot20":
        return root / "diarization_pilot20"
    return root / "production" / "diarization"


def _source_frame_count(path: Path, *, sample_rate: int, channels: int) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
                or source.getframerate() != sample_rate
                or source.getnchannels() != channels
            ):
                raise ValueError(
                    "canonical diarization audio must be matching PCM16 WAV"
                )
            return source.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("canonical diarization audio is unreadable") from exc


def _inventory_fingerprint(
    *,
    source_pairs_sha256: str | None,
    source_asr_inventory_fingerprint: str | None,
    mode: str,
    targets: Sequence[DiarizationTargetClip],
    source_inventory_kind: str = "pair_inventory",
    source_visual_inventory_sha256: str | None = None,
    source_canonical_audio_manifest_sha256: str | None = None,
) -> str:
    source: dict[str, str | None] = {}
    if source_inventory_kind == "canonical_audio_manifest":
        source.update(
            {
                "source_inventory_kind": source_inventory_kind,
                "source_visual_inventory_sha256": source_visual_inventory_sha256,
                "source_canonical_audio_manifest_sha256": (
                    source_canonical_audio_manifest_sha256
                ),
            }
        )
    else:
        source["source_pairs_sha256"] = source_pairs_sha256
    return _sha256_text(
        _compact_json(
            {
                **source,
                "source_asr_inventory_fingerprint": (source_asr_inventory_fingerprint),
                "mapping_policy_version": DIARIZATION_MAPPING_POLICY_VERSION,
                "mapping_policy_validated": True,
                "numeric_mapping_thresholds_used": False,
                "calibration_inventory_fingerprint": (
                    DIARIZATION_CALIBRATION_INVENTORY_FINGERPRINT
                ),
                "calibration_source_asr_inventory_fingerprint": (
                    DIARIZATION_CALIBRATION_SOURCE_ASR_FINGERPRINT
                ),
                "mode": mode,
                "targets": [item.model_dump(mode="json") for item in targets],
            }
        )
    )


def build_complete_diarization_inventory(
    *,
    source_pairs_path: Path,
    targets: Sequence[DiarizationTargetClip],
) -> DiarizationInventory:
    """Build the frozen production inventory from an explicit complete pair set."""
    pairs_path = source_pairs_path.expanduser().resolve(strict=True)
    ordered = list(targets)
    target_ids = [item.target_clip_uid for item in ordered]
    if target_ids != sorted(target_ids) or len(target_ids) != len(set(target_ids)):
        raise ValueError("complete diarization targets must be unique and ordered")
    pairs_sha256 = _sha256_file(pairs_path)
    fingerprint = _inventory_fingerprint(
        source_pairs_sha256=pairs_sha256,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=ordered,
    )
    return DiarizationInventory(
        schema_version=DIARIZATION_INVENTORY_VERSION,
        mode="production",
        source_pairs_path=str(pairs_path),
        source_pairs_sha256=pairs_sha256,
        inventory_fingerprint=fingerprint,
        source_target_count=len(ordered),
        selected_target_count=len(ordered),
        selection_mode="complete_in_pair_target_inventory_v1",
        bounded_selection_applied=False,
        targets=ordered,
    )


def build_diarization_inventory(
    *,
    audio_run_root: Path,
    mode: Literal["pilot20", "production"],
) -> DiarizationInventory:
    root = audio_run_root.expanduser().resolve(strict=True)
    pairs_path = (root / "production" / "pairs" / "in_pairs.jsonl").resolve(strict=True)
    pairs_sha256 = _sha256_file(pairs_path)
    pairs = [
        H3ProductionInPair.model_validate(item) for item in _read_jsonl(pairs_path)
    ]
    pair_by_clip = {item.target_clip_uid: item for item in pairs}
    if len(pair_by_clip) != len(pairs):
        raise ValueError("production in-pairs contain duplicate target clips")
    if len(pairs) != EXPECTED_PRODUCTION_TARGET_COUNT:
        raise ValueError("production in-pair target count is not the frozen 75")
    asr_path: Path | None = None
    source_asr_inventory_fingerprint: str | None = None
    if mode == "pilot20":
        asr_path = (root / "asr_pilot20" / "inventory.json").resolve(strict=True)
        asr_inventory = ASRInventory.model_validate_json(
            asr_path.read_text(encoding="utf-8")
        )
        expected_asr_fingerprint = _asr_inventory_fingerprint(
            source_pairs_sha256=asr_inventory.source_pairs_sha256,
            mode=asr_inventory.mode,
            targets=asr_inventory.targets,
            jobs=asr_inventory.jobs,
        )
        if asr_inventory.inventory_fingerprint != expected_asr_fingerprint:
            raise ValueError("source ASR pilot inventory fingerprint is inconsistent")
        if (
            asr_inventory.mode != "pilot20"
            or asr_inventory.selected_target_count != PILOT_TARGET_COUNT
            or asr_inventory.source_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
        ):
            raise ValueError("source ASR inventory is not the frozen 20-of-75 pilot")
        if (
            str(pairs_path)
            != str(Path(asr_inventory.source_pairs_path).resolve(strict=True))
            or pairs_sha256 != asr_inventory.source_pairs_sha256
        ):
            raise ValueError("production in-pairs changed after ASR pilot selection")
        selected_ids = [item.target_clip_uid for item in asr_inventory.targets]
        source_asr_inventory_fingerprint = asr_inventory.inventory_fingerprint
        if any(clip_uid not in pair_by_clip for clip_uid in selected_ids):
            raise ValueError("ASR pilot target is missing from production in-pairs")
    else:
        selected_ids = sorted(pair_by_clip)

    targets: list[DiarizationTargetClip] = []
    for clip_uid in selected_ids:
        pair = pair_by_clip[clip_uid]
        video_path = Path(pair.target_video_path).expanduser().resolve(strict=True)
        audio_path = Path(pair.target_full_audio_path).expanduser().resolve(strict=True)
        sidecar_path = (
            Path(pair.target_audio_binding_path).expanduser().resolve(strict=True)
        )
        if (
            not video_path.is_file()
            or not audio_path.is_file()
            or not sidecar_path.is_file()
        ):
            raise ValueError("diarization target media is unavailable")
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if (
            sidecar.clip_uid != clip_uid
            or sidecar.status != "ready"
            or sidecar.evidence is None
        ):
            raise ValueError("diarization requires a matching ready Audio sidecar")
        audio = sidecar.evidence.audio
        if (
            audio.full_audio_path is None
            or Path(audio.full_audio_path).expanduser().resolve(strict=True)
            != audio_path
            or audio.sample_rate_hz is None
            or audio.channels is None
        ):
            raise ValueError("diarization Audio sidecar provenance is incomplete")
        frame_count = _source_frame_count(
            audio_path,
            sample_rate=audio.sample_rate_hz,
            channels=audio.channels,
        )
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video_path),
                source_audio_path=str(audio_path),
                source_audio_sha256=_sha256_file(audio_path),
                source_sample_rate_hz=audio.sample_rate_hz,
                source_channels=audio.channels,
                source_frame_count=frame_count,
                target_audio_binding_path=str(sidecar_path),
                target_audio_binding_sha256=_sha256_file(sidecar_path),
                visual_references=[
                    DiarizationVisualReference(
                        entity_id=subject.target_entity_id,
                        image_path=subject.target_visual_reference_path,
                    )
                    for subject in pair.subjects
                ],
            )
        )
    fingerprint = _inventory_fingerprint(
        source_pairs_sha256=pairs_sha256,
        source_asr_inventory_fingerprint=source_asr_inventory_fingerprint,
        mode=mode,
        targets=targets,
    )
    return DiarizationInventory(
        schema_version=DIARIZATION_INVENTORY_VERSION,
        mode=mode,
        source_pairs_path=str(pairs_path),
        source_pairs_sha256=pairs_sha256,
        source_asr_inventory_path=str(asr_path) if asr_path is not None else None,
        source_asr_inventory_fingerprint=source_asr_inventory_fingerprint,
        inventory_fingerprint=fingerprint,
        source_target_count=len(pairs),
        selected_target_count=len(targets),
        selection_mode=(
            "same_ordered_asr_pilot20_targets_v1"
            if mode == "pilot20"
            else "complete_in_pair_target_inventory_v1"
        ),
        bounded_selection_applied=mode == "pilot20",
        targets=targets,
    )


@dataclass(frozen=True)
class _SampleSpan:
    start: int
    end: int

    @property
    def samples(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _Anchor:
    index: int
    start: int
    end: int
    entity_id: str
    confidence: float


@dataclass
class _SupportAccumulator:
    samples: int = 0
    weighted: float = 0.0
    binding_indices: set[int] | None = None

    def __post_init__(self) -> None:
        if self.binding_indices is None:
            self.binding_indices = set()


def _union_sample_count(spans: Sequence[_SampleSpan]) -> int:
    if not spans:
        return 0
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    total = 0
    start, end = ordered[0].start, ordered[0].end
    for span in ordered[1:]:
        if span.start <= end:
            end = max(end, span.end)
        else:
            total += end - start
            start, end = span.start, span.end
    return total + end - start


def _spans_overlap(left: Sequence[_SampleSpan], right: Sequence[_SampleSpan]) -> bool:
    for first in left:
        for second in right:
            if max(first.start, second.start) < min(first.end, second.end):
                return True
    return False


def _normalize_segments(
    *,
    target: DiarizationTargetClip,
    segments: Sequence[DiarizationBackendSegment],
    provenance: DiarizationBackendProvenance,
) -> list[RawDiarizationSegment]:
    ordered = sorted(
        segments,
        key=lambda item: (item.start_time, item.end_time, item.speaker_label),
    )
    cluster_by_label: dict[str, str] = {}
    output: list[RawDiarizationSegment] = []
    for index, item in enumerate(ordered, start=1):
        reported_start_sample = round(item.start_time * target.source_sample_rate_hz)
        reported_end_sample = round(item.end_time * target.source_sample_rate_hz)
        if reported_start_sample >= target.source_frame_count:
            raise DiarizationBackendFailure(
                "diarization segment starts at or after canonical source EOF"
            )
        end_sample = min(reported_end_sample, target.source_frame_count)
        if reported_start_sample < 0 or end_sample <= reported_start_sample:
            raise DiarizationBackendFailure(
                "diarization segment has no positive canonical source intersection"
            )
        end_overrun_samples = reported_end_sample - end_sample
        end_clamped = end_overrun_samples > 0
        if item.speaker_label not in cluster_by_label:
            cluster_by_label[item.speaker_label] = f"speaker_{len(cluster_by_label)}"
        output.append(
            RawDiarizationSegment(
                target_clip_uid=target.target_clip_uid,
                segment_id=f"segment_{index:04d}",
                speaker_cluster_id=cluster_by_label[item.speaker_label],
                backend_speaker_label=item.speaker_label,
                backend_reported_start_time=item.start_time,
                backend_reported_end_time=item.end_time,
                backend_reported_start_sample=reported_start_sample,
                backend_reported_end_sample=reported_end_sample,
                start_time=reported_start_sample / target.source_sample_rate_hz,
                end_time=end_sample / target.source_sample_rate_hz,
                source_start_sample=reported_start_sample,
                source_end_sample=end_sample,
                source_audio_path=target.source_audio_path,
                source_audio_sha256=target.source_audio_sha256,
                source_sample_rate_hz=target.source_sample_rate_hz,
                source_channels=target.source_channels,
                backend=provenance.backend,
                model_identifier=provenance.model_identifier,
                model_fingerprint=provenance.model_fingerprint,
                backend_configuration_fingerprint=(
                    provenance.configuration_fingerprint
                ),
                input_preprocessing=provenance.input_preprocessing,
                boundary_reconciliation=DiarizationBoundaryReconciliation(
                    adjusted=end_clamped,
                    end_clamped=end_clamped,
                    end_overrun_samples=end_overrun_samples,
                    end_overrun_seconds=(
                        end_overrun_samples / target.source_sample_rate_hz
                    ),
                    reason=("end_clamped_to_canonical_source" if end_clamped else None),
                ),
            )
        )
    return output


def _usable_anchors(
    bindings: Sequence[AudioEntityBinding],
    *,
    target: DiarizationTargetClip,
) -> list[_Anchor]:
    minimum = AudioBindingProductionConfig().minimum_binding_confidence
    output: list[_Anchor] = []
    for index, binding in enumerate(bindings):
        if (
            binding.status != "bound"
            or binding.entity_id is None
            or binding.face_track_id is None
            or binding.confidence < minimum
            or not binding.evidence.synchronization_plausible
        ):
            continue
        start = round(binding.start_time * target.source_sample_rate_hz)
        end = round(binding.end_time * target.source_sample_rate_hz)
        if start < 0 or end <= start or end > target.source_frame_count:
            raise ValueError("frozen Audio binding exceeds canonical source audio")
        output.append(
            _Anchor(
                index=index,
                start=start,
                end=end,
                entity_id=binding.entity_id,
                confidence=binding.confidence,
            )
        )
    return output


@dataclass
class _MappingResult:
    bindings: list[DiarizationClusterBinding]
    bound_segments: list[BoundDiarizationSegment]
    usable_spans_by_cluster: dict[str, list[_SampleSpan]]
    usable_samples: int
    contested_samples: int
    unmatched_samples: int


def bind_diarization_segments(
    *,
    target: DiarizationTargetClip,
    raw_segments: Sequence[RawDiarizationSegment],
    frozen_bindings: Sequence[AudioEntityBinding],
) -> _MappingResult:
    clusters: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    for segment in raw_segments:
        if segment.target_clip_uid != target.target_clip_uid:
            raise ValueError("raw diarization segment belongs to another clip")
        clusters[segment.speaker_cluster_id].append(segment)
    anchors = _usable_anchors(frozen_bindings, target=target)
    support: dict[tuple[str, str], _SupportAccumulator] = {}
    usable_spans: dict[str, list[_SampleSpan]] = defaultdict(list)
    contested_spans: dict[str, list[_SampleSpan]] = defaultdict(list)
    all_usable: list[_SampleSpan] = []
    all_contested: list[_SampleSpan] = []
    all_unmatched: list[_SampleSpan] = []

    for anchor in anchors:
        boundaries = {anchor.start, anchor.end}
        for segment in raw_segments:
            start = max(anchor.start, segment.source_start_sample)
            end = min(anchor.end, segment.source_end_sample)
            if start < end:
                boundaries.update((start, end))
        points = sorted(boundaries)
        for start, end in pairwise(points):
            if start >= end:
                continue
            active = {
                segment.speaker_cluster_id
                for segment in raw_segments
                if segment.source_start_sample < end
                and segment.source_end_sample > start
            }
            span = _SampleSpan(start, end)
            if len(active) == 1:
                cluster_id = next(iter(active))
                key = (cluster_id, anchor.entity_id)
                item = support.setdefault(key, _SupportAccumulator())
                item.samples += span.samples
                item.weighted += span.samples * anchor.confidence
                assert item.binding_indices is not None
                item.binding_indices.add(anchor.index)
                usable_spans[cluster_id].append(span)
                all_usable.append(span)
            elif len(active) > 1:
                for cluster_id in active:
                    contested_spans[cluster_id].append(span)
                all_contested.append(span)
            else:
                all_unmatched.append(span)

    cluster_bindings: list[DiarizationClusterBinding] = []
    cluster_spans: dict[str, list[_SampleSpan]] = {}
    sample_rate = target.source_sample_rate_hz
    for cluster_id, segments in sorted(clusters.items()):
        spans = [
            _SampleSpan(item.source_start_sample, item.source_end_sample)
            for item in segments
        ]
        cluster_spans[cluster_id] = spans
        cluster_samples = _union_sample_count(spans)
        supports = [
            DiarizationEntitySupport(
                entity_id=entity_id,
                direct_support_samples=value.samples,
                direct_support_seconds=value.samples / sample_rate,
                weighted_support=value.weighted,
                contributing_binding_count=len(value.binding_indices or ()),
            )
            for (support_cluster, entity_id), value in support.items()
            if support_cluster == cluster_id
        ]
        supports.sort(
            key=lambda item: (
                -item.direct_support_samples,
                -item.weighted_support,
                item.entity_id,
            )
        )
        top1 = supports[0] if supports else None
        top2 = supports[1] if len(supports) > 1 else None
        total_weight = sum(item.weighted_support for item in supports)
        if not supports:
            status: Literal["candidate_mapped", "unbound", "ambiguous", "conflict"] = (
                "unbound"
            )
            entity_id = None
            reasons = ["no_usable_direct_entity_support"]
        elif len(supports) == 1:
            status = "candidate_mapped"
            entity_id = supports[0].entity_id
            reasons = []
        else:
            status = "ambiguous"
            entity_id = None
            reasons = ["multiple_entities_have_direct_support"]
        usable_count = _union_sample_count(usable_spans[cluster_id])
        contested_count = _union_sample_count(contested_spans[cluster_id])
        cluster_bindings.append(
            DiarizationClusterBinding(
                target_clip_uid=target.target_clip_uid,
                speaker_cluster_id=cluster_id,
                source_sample_rate_hz=target.source_sample_rate_hz,
                source_channels=target.source_channels,
                status=status,
                entity_id=entity_id,
                cluster_segment_count=len(segments),
                cluster_speaker_seconds=cluster_samples / sample_rate,
                usable_anchor_sample_count=usable_count,
                usable_anchor_duration=usable_count / sample_rate,
                contested_anchor_sample_count=contested_count,
                contested_anchor_duration=contested_count / sample_rate,
                unmatched_anchor_sample_count=0,
                unmatched_anchor_duration=0.0,
                entity_supports=supports,
                top1_entity_id=top1.entity_id if top1 is not None else None,
                top1_support=top1.weighted_support if top1 is not None else 0.0,
                top2_support=top2.weighted_support if top2 is not None else 0.0,
                top1_share=(
                    top1.weighted_support / total_weight
                    if top1 is not None and total_weight > 0
                    else None
                ),
                top1_top2_margin=(
                    top1.weighted_support
                    - (top2.weighted_support if top2 is not None else 0.0)
                    if top1 is not None
                    else None
                ),
                visual_anchor_coverage_ratio=(
                    usable_count / cluster_samples if cluster_samples else 0.0
                ),
                warnings=(
                    ["contested_overlapping_speaker_anchor_excluded"]
                    if contested_count
                    else []
                ),
                reason_codes=reasons,
            )
        )

    mapped_by_entity: dict[str, list[int]] = defaultdict(list)
    for index, binding in enumerate(cluster_bindings):
        if binding.status == "candidate_mapped" and binding.entity_id is not None:
            mapped_by_entity[binding.entity_id].append(index)
    conflict_indices: set[int] = set()
    for indices in mapped_by_entity.values():
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                if _spans_overlap(
                    cluster_spans[cluster_bindings[left_index].speaker_cluster_id],
                    cluster_spans[cluster_bindings[right_index].speaker_cluster_id],
                ):
                    conflict_indices.update((left_index, right_index))
    for index in sorted(conflict_indices):
        current = cluster_bindings[index]
        cluster_bindings[index] = DiarizationClusterBinding.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": "conflict",
                "entity_id": None,
                "warnings": [
                    *current.warnings,
                    "same_entity_mapped_to_overlapping_clusters",
                ],
                "reason_codes": ["same_entity_temporal_identity_conflict"],
            }
        )

    binding_by_cluster = {item.speaker_cluster_id: item for item in cluster_bindings}
    bound_segments: list[BoundDiarizationSegment] = []
    for segment in raw_segments:
        binding = binding_by_cluster[segment.speaker_cluster_id]
        direct_spans = [
            _SampleSpan(
                max(span.start, segment.source_start_sample),
                min(span.end, segment.source_end_sample),
            )
            for span in usable_spans[segment.speaker_cluster_id]
            if max(span.start, segment.source_start_sample)
            < min(span.end, segment.source_end_sample)
        ]
        direct_samples = _union_sample_count(direct_spans)
        entity_id = binding.entity_id
        bound_segments.append(
            BoundDiarizationSegment(
                target_clip_uid=segment.target_clip_uid,
                segment_id=segment.segment_id,
                speaker_cluster_id=segment.speaker_cluster_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                source_start_sample=segment.source_start_sample,
                source_end_sample=segment.source_end_sample,
                source_sample_rate_hz=segment.source_sample_rate_hz,
                source_channels=segment.source_channels,
                cluster_binding_status=binding.status,
                entity_id=entity_id,
                entity_occurrence_id=(
                    f"{segment.target_clip_uid}/{entity_id}"
                    if entity_id is not None
                    else None
                ),
                direct_anchor_samples=direct_samples,
                direct_anchor_seconds=direct_samples / sample_rate,
                identity_scope=(
                    "unresolved"
                    if entity_id is None
                    else "direct_anchor_present"
                    if direct_samples
                    else "cluster_propagated_only"
                ),
            )
        )
    return _MappingResult(
        bindings=cluster_bindings,
        bound_segments=bound_segments,
        usable_spans_by_cluster=dict(usable_spans),
        usable_samples=_union_sample_count(all_usable),
        contested_samples=_union_sample_count(all_contested),
        unmatched_samples=_union_sample_count(all_unmatched),
    )


def _load_optional_sidecar(
    target: DiarizationTargetClip,
) -> AudioBindingSidecar | None:
    if target.target_audio_binding_path is None:
        return None
    path = Path(target.target_audio_binding_path).expanduser().resolve(strict=True)
    if (
        target.target_audio_binding_sha256 is not None
        and _sha256_file(path) != target.target_audio_binding_sha256
    ):
        raise ValueError("diarization source Audio sidecar hash changed")
    sidecar = AudioBindingSidecar.model_validate_json(path.read_text(encoding="utf-8"))
    if sidecar.clip_uid != target.target_clip_uid:
        raise ValueError("diarization source Audio sidecar is inconsistent")
    if sidecar.status != "ready" or sidecar.evidence is None:
        return None
    return sidecar


def _legacy_metrics(
    target: DiarizationTargetClip,
    sidecar: AudioBindingSidecar | None,
) -> tuple[list[_Anchor], list[float]]:
    if sidecar is None:
        return [], []
    anchors = _usable_anchors(sidecar.bindings, target=target)
    turns = coalesce_audio_bindings(
        sidecar.bindings,
        clip_uid=target.target_clip_uid,
        sample_rate_hz=target.source_sample_rate_hz,
        maximum_gap_seconds=AudioBindingProductionConfig().speech_merge_gap_seconds,
        minimum_voice_reference_duration_seconds=(
            AudioBindingProductionConfig().minimum_voice_reference_duration_seconds
        ),
    )
    durations = [
        item.end_time - item.start_time
        for item in turns
        if item.status == "bound" and item.entity_id is not None
    ]
    return anchors, durations


def _clip_result(
    *,
    target: DiarizationTargetClip,
    raw_segments: Sequence[RawDiarizationSegment],
    sidecar: AudioBindingSidecar | None,
    mapping: _MappingResult,
    status: Literal["ready", "empty"],
) -> DiarizationClipResult:
    anchors, legacy_turn_durations = _legacy_metrics(target, sidecar)
    return DiarizationClipResult(
        target_clip_uid=target.target_clip_uid,
        source_sample_rate_hz=target.source_sample_rate_hz,
        source_channels=target.source_channels,
        status=status,
        backend_called=True,
        raw_segment_count=len(raw_segments),
        speaker_cluster_count=len({item.speaker_cluster_id for item in raw_segments}),
        legacy_bound_interval_count=len(anchors),
        legacy_coalesced_bound_turn_count=len(legacy_turn_durations),
        legacy_lr_asd_bound_samples=_union_sample_count(
            [_SampleSpan(item.start, item.end) for item in anchors]
        ),
        usable_direct_anchor_samples=mapping.usable_samples,
        contested_anchor_samples=mapping.contested_samples,
        unmatched_anchor_samples=mapping.unmatched_samples,
        legacy_bound_turn_durations=legacy_turn_durations,
        diarization_segment_durations=[
            item.end_time - item.start_time for item in raw_segments
        ],
    )


def _failed_clip_result(
    *,
    target: DiarizationTargetClip,
    backend_called: bool,
    reason: str,
) -> DiarizationClipResult:
    return DiarizationClipResult(
        target_clip_uid=target.target_clip_uid,
        source_sample_rate_hz=target.source_sample_rate_hz,
        source_channels=target.source_channels,
        status="failed",
        backend_called=backend_called,
        raw_segment_count=0,
        speaker_cluster_count=0,
        legacy_bound_interval_count=0,
        legacy_coalesced_bound_turn_count=0,
        legacy_lr_asd_bound_samples=0,
        usable_direct_anchor_samples=0,
        contested_anchor_samples=0,
        unmatched_anchor_samples=0,
        reason=reason,
    )


def _timeline_union_samples(segments: Sequence[RawDiarizationSegment]) -> int:
    return _union_sample_count(
        [
            _SampleSpan(item.source_start_sample, item.source_end_sample)
            for item in segments
        ]
    )


def _mapped_identity_duration_metrics(
    *,
    raw_segments: Sequence[RawDiarizationSegment],
    cluster_bindings: Sequence[DiarizationClusterBinding],
    bound_segments: Sequence[BoundDiarizationSegment],
) -> tuple[float, float, float]:
    bound_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in bound_segments
    }
    fully_propagated_by_cluster: dict[tuple[str, str], list[_SampleSpan]] = defaultdict(
        list
    )
    sample_rate_by_cluster: dict[tuple[str, str], int] = {}
    for raw in raw_segments:
        if (
            bound_by_key[(raw.target_clip_uid, raw.segment_id)].identity_scope
            != "cluster_propagated_only"
        ):
            continue
        key = (raw.target_clip_uid, raw.speaker_cluster_id)
        fully_propagated_by_cluster[key].append(
            _SampleSpan(raw.source_start_sample, raw.source_end_sample)
        )
        sample_rate_by_cluster[key] = raw.source_sample_rate_hz
    fully_propagated_segments = sum(
        _union_sample_count(spans) / sample_rate_by_cluster[key]
        for key, spans in fully_propagated_by_cluster.items()
    )
    mapped_clusters = [
        item for item in cluster_bindings if item.status == "candidate_mapped"
    ]
    mapped_direct_anchor_seconds = sum(
        item.usable_anchor_duration for item in mapped_clusters
    )
    identity_propagated_seconds = sum(
        item.cluster_speaker_seconds - item.usable_anchor_duration
        for item in mapped_clusters
    )
    return (
        mapped_direct_anchor_seconds,
        identity_propagated_seconds,
        fully_propagated_segments,
    )


def _summary(
    *,
    inventory: DiarizationInventory,
    provenance: DiarizationBackendProvenance,
    raw_segments: Sequence[RawDiarizationSegment],
    cluster_bindings: Sequence[DiarizationClusterBinding],
    bound_segments: Sequence[BoundDiarizationSegment],
    clip_results: Sequence[DiarizationClipResult],
) -> DiarizationSummary:
    status_counts = Counter(item.status for item in clip_results)
    cluster_counts = Counter(item.status for item in cluster_bindings)
    by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    for item in raw_segments:
        by_clip[item.target_clip_uid].append(item)
    wallclock_seconds = 0.0
    for target in inventory.targets:
        wallclock_seconds += (
            _timeline_union_samples(by_clip[target.target_clip_uid])
            / target.source_sample_rate_hz
        )
    speaker_seconds_by_status: Counter[str] = Counter()
    for cluster in cluster_bindings:
        speaker_seconds_by_status.update(
            {cluster.status: cluster.cluster_speaker_seconds}
        )
    (
        mapped_direct_anchor_seconds,
        identity_propagated_seconds,
        fully_propagated_segments,
    ) = _mapped_identity_duration_metrics(
        raw_segments=raw_segments,
        cluster_bindings=cluster_bindings,
        bound_segments=bound_segments,
    )
    adjusted_segments = [
        item for item in raw_segments if item.boundary_reconciliation.adjusted
    ]
    positive_overruns = [
        item.boundary_reconciliation.end_overrun_seconds for item in adjusted_segments
    ]
    all_legacy_durations = [
        value for item in clip_results for value in item.legacy_bound_turn_durations
    ]
    all_diarization_durations = [
        value for item in clip_results for value in item.diarization_segment_durations
    ]
    failure_counts = Counter(
        item.reason for item in clip_results if item.reason is not None
    )
    source_rate_by_clip = {
        item.target_clip_uid: item.source_sample_rate_hz for item in inventory.targets
    }
    return DiarizationSummary(
        mode=inventory.mode,
        inventory_fingerprint=inventory.inventory_fingerprint,
        source_asr_inventory_fingerprint=(inventory.source_asr_inventory_fingerprint),
        backend_provenance=provenance,
        target_clip_count=len(clip_results),
        ready_clip_count=status_counts["ready"],
        empty_clip_count=status_counts["empty"],
        failed_clip_count=status_counts["failed"],
        backend_call_count=sum(item.backend_called for item in clip_results),
        raw_segment_count=len(raw_segments),
        speaker_cluster_count=len(cluster_bindings),
        candidate_mapped_cluster_count=cluster_counts["candidate_mapped"],
        unbound_cluster_count=cluster_counts["unbound"],
        ambiguous_cluster_count=cluster_counts["ambiguous"],
        conflict_cluster_count=cluster_counts["conflict"],
        diarized_speaker_seconds=sum(
            item.cluster_speaker_seconds for item in cluster_bindings
        ),
        diarized_wallclock_speech_seconds=wallclock_seconds,
        legacy_lr_asd_bound_seconds=sum(
            item.legacy_lr_asd_bound_samples / source_rate_by_clip[item.target_clip_uid]
            for item in clip_results
        ),
        usable_direct_anchor_seconds=sum(
            item.usable_direct_anchor_samples
            / source_rate_by_clip[item.target_clip_uid]
            for item in clip_results
        ),
        contested_anchor_seconds=sum(
            item.contested_anchor_samples / source_rate_by_clip[item.target_clip_uid]
            for item in clip_results
        ),
        mapped_speaker_seconds=speaker_seconds_by_status["candidate_mapped"],
        mapped_direct_anchor_speaker_seconds=mapped_direct_anchor_seconds,
        identity_propagated_speaker_seconds=identity_propagated_seconds,
        fully_propagated_segment_speaker_seconds=fully_propagated_segments,
        unbound_speaker_seconds=speaker_seconds_by_status["unbound"],
        ambiguous_speaker_seconds=speaker_seconds_by_status["ambiguous"],
        conflict_speaker_seconds=speaker_seconds_by_status["conflict"],
        legacy_coalesced_bound_turn_count=sum(
            item.legacy_coalesced_bound_turn_count for item in clip_results
        ),
        legacy_bound_turn_median_duration=(
            median(all_legacy_durations) if all_legacy_durations else None
        ),
        diarization_segment_median_duration=(
            median(all_diarization_durations) if all_diarization_durations else None
        ),
        boundary_adjusted_segment_count=len(adjusted_segments),
        boundary_adjusted_clip_count=len(
            {item.target_clip_uid for item in adjusted_segments}
        ),
        end_clamped_segment_count=sum(
            item.boundary_reconciliation.end_clamped for item in adjusted_segments
        ),
        total_end_overrun_seconds=sum(positive_overruns),
        max_end_overrun_seconds=max(positive_overruns, default=0.0),
        max_end_overrun_samples=max(
            (
                item.boundary_reconciliation.end_overrun_samples
                for item in adjusted_segments
            ),
            default=0,
        ),
        median_positive_end_overrun_seconds=(
            median(positive_overruns) if positive_overruns else None
        ),
        failure_reason_counts=dict(sorted(failure_counts.items())),
    )


def _write_segment_wav(
    *,
    source_path: Path,
    destination: Path,
    start_sample: int,
    end_sample: int,
    expected_sample_rate: int,
    expected_channels: int,
) -> None:
    if source_path.suffix.lower() == ".flac":
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is required to render canonical FLAC review audio"
            ) from exc
        info = sf.info(str(source_path))
        if (
            info.samplerate != expected_sample_rate
            or info.channels != expected_channels
            or end_sample > info.frames
        ):
            raise ValueError("canonical review audio metadata changed")
        frames, sample_rate = sf.read(
            str(source_path),
            start=start_sample,
            stop=end_sample,
            dtype="int16",
            always_2d=True,
        )
        if sample_rate != expected_sample_rate or frames.shape != (
            end_sample - start_sample,
            expected_channels,
        ):
            raise ValueError("canonical review audio ended unexpectedly")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), frames, expected_sample_rate, subtype="PCM_16")
        return
    with wave.open(str(source_path), "rb") as source:
        if (
            source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
            or source.getframerate() != expected_sample_rate
            or source.getnchannels() != expected_channels
            or end_sample > source.getnframes()
        ):
            raise ValueError("canonical review audio metadata changed")
        source.setpos(start_sample)
        frames = source.readframes(end_sample - start_sample)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(expected_channels)
        output.setsampwidth(2)
        output.setframerate(expected_sample_rate)
        output.writeframes(frames)


def _review_html(
    *,
    inventory: DiarizationInventory,
    raw_segments: Sequence[RawDiarizationSegment],
    cluster_bindings: Sequence[DiarizationClusterBinding],
    bound_segments: Sequence[BoundDiarizationSegment],
    clip_results: Sequence[DiarizationClipResult],
    anchors_by_clip: dict[str, list[AudioEntityBinding]],
    media: dict[str, dict[str, object]],
) -> str:
    raw_by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    clusters_by_clip: dict[str, list[DiarizationClusterBinding]] = defaultdict(list)
    bound_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in bound_segments
    }
    result_by_clip = {item.target_clip_uid: item for item in clip_results}
    for item in raw_segments:
        raw_by_clip[item.target_clip_uid].append(item)
    for item in cluster_bindings:
        clusters_by_clip[item.target_clip_uid].append(item)
    qa_rows = [
        {
            "target_clip_uid": item.target_clip_uid,
            "speaker_cluster_id": item.speaker_cluster_id,
            "predicted_binding_status": item.status,
            "predicted_entity_id": item.entity_id,
        }
        for item in sorted(
            cluster_bindings,
            key=lambda value: (value.target_clip_uid, value.speaker_cluster_id),
        )
    ]
    sections: list[str] = []
    for target in inventory.targets:
        clip_id = target.target_clip_uid
        clip_segments = raw_by_clip[clip_id]
        duration = target.source_frame_count / target.source_sample_rate_hz
        lanes: list[str] = []
        anchor_blocks = []
        for anchor in anchors_by_clip.get(clip_id, []):
            left = 100 * anchor.start_time / duration
            width = 100 * (anchor.end_time - anchor.start_time) / duration
            anchor_blocks.append(
                f'<span class="block anchor" style="left:{left:.5f}%;width:{width:.5f}%" '
                f'title="{html.escape(str(anchor.entity_id))} {anchor.start_time:.2f}-{anchor.end_time:.2f}"></span>'
            )
        lanes.append(
            '<div class="lane"><b>LR-ASD anchors</b><div class="track">'
            + "".join(anchor_blocks)
            + "</div></div>"
        )
        for cluster in clusters_by_clip.get(clip_id, []):
            blocks = []
            for segment in clip_segments:
                if segment.speaker_cluster_id != cluster.speaker_cluster_id:
                    continue
                bound = bound_by_key[(clip_id, segment.segment_id)]
                left = 100 * segment.start_time / duration
                width = 100 * (segment.end_time - segment.start_time) / duration
                blocks.append(
                    f'<span class="block speaker {html.escape(bound.identity_scope)}" '
                    f'style="left:{left:.5f}%;width:{width:.5f}%" '
                    f'title="{segment.start_time:.2f}-{segment.end_time:.2f} '
                    f'{html.escape(str(bound.entity_id))}"></span>'
                )
            lanes.append(
                f'<div class="lane"><b>{html.escape(cluster.speaker_cluster_id)}</b>'
                '<div class="track">' + "".join(blocks) + "</div></div>"
            )
        visual_html = "".join(
            f'<figure><img src="{html.escape(str(item["path"]))}" alt="{html.escape(str(item["entity_id"]))}"><figcaption>{html.escape(str(item["entity_id"]))}</figcaption></figure>'
            for item in media[clip_id]["visuals"]  # type: ignore[index]
        )
        cluster_html = []
        for cluster in clusters_by_clip.get(clip_id, []):
            supports = (
                ", ".join(
                    f"{item.entity_id}: {item.direct_support_seconds:.3f}s / {item.weighted_support:.1f}"
                    for item in cluster.entity_supports
                )
                or "none"
            )
            segment_rows = []
            for segment in clip_segments:
                if segment.speaker_cluster_id != cluster.speaker_cluster_id:
                    continue
                bound = bound_by_key[(clip_id, segment.segment_id)]
                segment_path = media[clip_id]["segments"][segment.segment_id]  # type: ignore[index]
                reconciliation = segment.boundary_reconciliation
                boundary_html = ""
                if reconciliation.adjusted:
                    boundary_html = (
                        '<span class="warning-badge">EOF CLAMP</span> '
                        f"REPORTED: {segment.backend_reported_start_time:.3f}-"
                        f"{segment.backend_reported_end_time:.3f}s; "
                        f"CANONICAL: {segment.start_time:.3f}-{segment.end_time:.3f}s; "
                        f"EOF CLAMP: +{reconciliation.end_overrun_seconds:.3f}s "
                    )
                segment_rows.append(
                    "<li>"
                    f"{boundary_html}"
                    f"{segment.segment_id} {segment.start_time:.3f}-{segment.end_time:.3f}s "
                    f"{html.escape(bound.identity_scope)} entity={html.escape(str(bound.entity_id))} "
                    f'<audio controls preload="none" src="{html.escape(str(segment_path))}"></audio>'
                    "</li>"
                )
            qa_name = f"qa-{clip_id}-{cluster.speaker_cluster_id}"
            radios = "".join(
                f'<label><input type="radio" name="{qa_name}" value="{label}" onchange="saveLabel(this)">{label}</label>'
                for label in DIARIZATION_HUMAN_QA_LABELS
            )
            cluster_html.append(
                '<article class="cluster" '
                f'data-clip="{html.escape(clip_id)}" data-cluster="{html.escape(cluster.speaker_cluster_id)}">'
                f"<h3>{html.escape(cluster.speaker_cluster_id)}: {cluster.status} -> {html.escape(str(cluster.entity_id))}</h3>"
                f"<p>segments={cluster.cluster_segment_count}; speaker={cluster.cluster_speaker_seconds:.3f}s; "
                f"usable={cluster.usable_anchor_duration:.3f}s; contested={cluster.contested_anchor_duration:.3f}s; "
                f"coverage(diagnostic only)={cluster.visual_anchor_coverage_ratio:.4f}; "
                f"top1_share={cluster.top1_share}; margin={cluster.top1_top2_margin}</p>"
                f"<p>supports: {html.escape(supports)}</p><ul>{''.join(segment_rows)}</ul>"
                f'<div class="qa">{radios}</div></article>'
            )
        legacy_rows = "".join(
            f"<li>{item.start_time:.3f}-{item.end_time:.3f}s entity={html.escape(str(item.entity_id))} face={html.escape(str(item.face_track_id))} confidence={item.confidence:.3f}</li>"
            for item in anchors_by_clip.get(clip_id, [])
        )
        result = result_by_clip[clip_id]
        sections.append(
            f"<section><h2>{html.escape(clip_id)} ({result.status})</h2>"
            f'<video controls preload="metadata" src="{html.escape(str(media[clip_id]["video"]))}"></video>'
            f'<audio controls preload="none" src="{html.escape(str(media[clip_id]["audio"]))}"></audio>'
            f'<div class="visuals">{visual_html}</div>'
            f"<p>raw segments={result.raw_segment_count}; clusters={result.speaker_cluster_count}; "
            f"usable direct anchor={result.usable_direct_anchor_samples / target.source_sample_rate_hz:.3f}s; "
            f"contested={result.contested_anchor_samples / target.source_sample_rate_hz:.3f}s</p>"
            + "".join(lanes)
            + f"<details><summary>Frozen raw identity anchors</summary><ul>{legacy_rows}</ul></details>"
            + "".join(cluster_html)
            + "</section>"
        )
    qa_metadata = _compact_json(
        {
            "schema_version": DIARIZATION_HUMAN_QA_VERSION,
            "inventory_fingerprint": inventory.inventory_fingerprint,
            "source_asr_inventory_fingerprint": (
                inventory.source_asr_inventory_fingerprint
            ),
            "mode": "pilot20",
            "rows": qa_rows,
        }
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>H3 DiariZen Binding Review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f6f7;color:#17191c}}section{{background:white;border:1px solid #ccd0d5;margin:18px 0;padding:16px;border-radius:6px}}video{{max-width:640px;max-height:360px;display:block}}audio{{vertical-align:middle}}.visuals{{display:flex;gap:10px;flex-wrap:wrap}}figure{{margin:8px}}figure img{{max-width:180px;max-height:180px}}.lane{{display:grid;grid-template-columns:150px 1fr;gap:8px;margin:5px 0}}.track{{height:24px;background:#e7e9ec;position:relative}}.block{{position:absolute;top:2px;height:20px;min-width:2px}}.anchor{{background:#d1495b}}.speaker{{background:#247ba0}}.cluster_propagated_only{{background:#2a9d8f}}.unresolved{{background:#7b7f86}}.warning-badge{{display:inline-block;background:#b42318;color:white;font-weight:700;padding:2px 5px;margin-right:4px;border-radius:3px}}.cluster{{border-top:1px solid #ddd;padding-top:8px;margin-top:12px}}.qa label{{margin-right:12px}}#qa-controls{{position:sticky;top:0;background:#fff;border:1px solid #aaa;padding:12px;z-index:3}}button{{margin-right:8px}}
</style></head><body>
<h1>DiariZen-assisted speaker binding pilot20</h1>
<div id="qa-controls"><b id="qa-progress">labeled 0 / {len(qa_rows)}</b> <span id="qa-counts"></span><br><button onclick="exportQA()">Export QA JSON</button><button onclick="clearQA()">Clear QA labels</button></div>
{"".join(sections)}
<script>
const qaMetadata={qa_metadata};
const qaPrefix=`h3-diarization-${{qaMetadata.inventory_fingerprint}}-`;
function keyFor(row){{return qaPrefix+row.target_clip_uid+'-'+row.speaker_cluster_id;}}
function counts(){{const c={{CORRECT:0,WRONG:0,UNCERTAIN:0,UNLABELED:0}};for(const row of qaMetadata.rows){{const value=localStorage.getItem(keyFor(row));if(value&&Object.hasOwn(c,value))c[value]++;else c.UNLABELED++;}}return c;}}
function updateCounts(){{const c=counts();const labeled=c.CORRECT+c.WRONG+c.UNCERTAIN;document.getElementById('qa-progress').textContent=`labeled ${{labeled}} / ${{qaMetadata.rows.length}}`;document.getElementById('qa-counts').textContent=`CORRECT ${{c.CORRECT}} | WRONG ${{c.WRONG}} | UNCERTAIN ${{c.UNCERTAIN}} | UNLABELED ${{c.UNLABELED}}`;}}
function saveLabel(input){{const parts=input.name.slice(3).split('-');const article=input.closest('.cluster');const row={{target_clip_uid:article.dataset.clip,speaker_cluster_id:article.dataset.cluster}};localStorage.setItem(keyFor(row),input.value);updateCounts();}}
function restoreLabels(){{for(const article of document.querySelectorAll('.cluster')){{const row={{target_clip_uid:article.dataset.clip,speaker_cluster_id:article.dataset.cluster}};const value=localStorage.getItem(keyFor(row));if(value){{const input=article.querySelector(`input[value="${{value}}"]`);if(input)input.checked=true;}}}}updateCounts();}}
function clearQA(){{if(!confirm('Clear only labels for this diarization review?'))return;for(const row of qaMetadata.rows)localStorage.removeItem(keyFor(row));restoreLabels();}}
function exportQA(){{const labels=[];for(const row of qaMetadata.rows){{const label=localStorage.getItem(keyFor(row));if(label&&['CORRECT','WRONG','UNCERTAIN'].includes(label))labels.push({{...row,label}});}}const payload={{schema_version:qaMetadata.schema_version,inventory_fingerprint:qaMetadata.inventory_fingerprint,source_asr_inventory_fingerprint:qaMetadata.source_asr_inventory_fingerprint,mode:qaMetadata.mode,label_count:labels.length,total_cluster_count:qaMetadata.rows.length,counts:counts(),labels}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='diarization_pilot20_human_qa.json';link.click();URL.revokeObjectURL(link.href);}}
restoreLabels();
</script></body></html>"""


def _materialize_review_media(
    *,
    inventory: DiarizationInventory,
    raw_segments: Sequence[RawDiarizationSegment],
    review_root: Path,
) -> dict[str, dict[str, object]]:
    raw_by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    for item in raw_segments:
        raw_by_clip[item.target_clip_uid].append(item)
    media: dict[str, dict[str, object]] = {}
    for target in inventory.targets:
        clip_uid = target.target_clip_uid
        video_name = (
            f"{clip_uid}.video{Path(target.target_video_path).suffix or '.media'}"
        )
        audio_name = (
            f"{clip_uid}.audio{Path(target.source_audio_path).suffix or '.media'}"
        )
        (review_root / video_name).symlink_to(Path(target.target_video_path))
        (review_root / audio_name).symlink_to(Path(target.source_audio_path))
        visuals: list[dict[str, str]] = []
        for reference in target.visual_references:
            source = Path(reference.image_path).expanduser()
            if not source.is_file():
                continue
            suffix = source.suffix.lower() or ".image"
            relative = Path("visual") / clip_uid / f"{reference.entity_id}{suffix}"
            destination_image = review_root / relative
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination_image)
            visuals.append(
                {
                    "entity_id": reference.entity_id,
                    "path": f"review_media/{relative.as_posix()}",
                }
            )
        segment_paths: dict[str, str] = {}
        for segment in raw_by_clip[clip_uid]:
            relative = Path("segments") / clip_uid / f"{segment.segment_id}.wav"
            _write_segment_wav(
                source_path=Path(target.source_audio_path),
                destination=review_root / relative,
                start_sample=segment.source_start_sample,
                end_sample=segment.source_end_sample,
                expected_sample_rate=target.source_sample_rate_hz,
                expected_channels=target.source_channels,
            )
            segment_paths[segment.segment_id] = f"review_media/{relative.as_posix()}"
        media[clip_uid] = {
            "video": f"review_media/{video_name}",
            "audio": f"review_media/{audio_name}",
            "visuals": visuals,
            "segments": segment_paths,
        }
    return media


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"diarization output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_diarization_binding_pilot(
    *,
    inventory: DiarizationInventory,
    output_root: Path,
    backend: DiarizationBackend,
    overwrite: bool = False,
) -> DiarizationSummary:
    expected_source = (
        backend.provenance.source_sample_rate_hz,
        backend.provenance.source_channels,
    )
    if any(
        (target.source_sample_rate_hz, target.source_channels) != expected_source
        for target in inventory.targets
    ):
        raise ValueError("DiariZen backend input profile differs from inventory")
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    raw_segments: list[RawDiarizationSegment] = []
    cluster_bindings: list[DiarizationClusterBinding] = []
    bound_segments: list[BoundDiarizationSegment] = []
    clip_results: list[DiarizationClipResult] = []
    anchors_by_clip: dict[str, list[AudioEntityBinding]] = {}
    try:
        temporary.mkdir()
        (temporary / "diagnostics").mkdir()
        for target in inventory.targets:
            audio_path = Path(target.source_audio_path)
            backend_called = False
            try:
                if (
                    not audio_path.is_file()
                    or _sha256_file(audio_path) != target.source_audio_sha256
                ):
                    raise ValueError("source_audio_hash_mismatch")
                sidecar = _load_optional_sidecar(target)
                frozen_bindings = [] if sidecar is None else list(sidecar.bindings)
                anchors_by_clip[target.target_clip_uid] = frozen_bindings
                backend_called = True
                backend_segments = backend.diarize(
                    clip_uid=target.target_clip_uid,
                    audio_path=audio_path,
                )
                clip_raw = _normalize_segments(
                    target=target,
                    segments=backend_segments,
                    provenance=backend.provenance,
                )
                mapping = bind_diarization_segments(
                    target=target,
                    raw_segments=clip_raw,
                    frozen_bindings=frozen_bindings,
                )
                raw_segments.extend(clip_raw)
                cluster_bindings.extend(mapping.bindings)
                bound_segments.extend(mapping.bound_segments)
                clip_results.append(
                    _clip_result(
                        target=target,
                        raw_segments=clip_raw,
                        sidecar=sidecar,
                        mapping=mapping,
                        status="ready" if clip_raw else "empty",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate one diarization clip
                reason = f"{type(exc).__name__}:{exc}"
                clip_results.append(
                    _failed_clip_result(
                        target=target,
                        backend_called=backend_called,
                        reason=reason,
                    )
                )
                (
                    temporary / "diagnostics" / f"{target.target_clip_uid}.json"
                ).write_text(
                    _compact_json(
                        {"target_clip_uid": target.target_clip_uid, "reason": reason}
                    )
                    + "\n",
                    encoding="utf-8",
                )

        raw_segments.sort(
            key=lambda item: (
                item.target_clip_uid,
                item.source_start_sample,
                item.source_end_sample,
                item.speaker_cluster_id,
                item.segment_id,
            )
        )
        cluster_bindings.sort(
            key=lambda item: (item.target_clip_uid, item.speaker_cluster_id)
        )
        bound_segments.sort(
            key=lambda item: (
                item.target_clip_uid,
                item.source_start_sample,
                item.source_end_sample,
                item.speaker_cluster_id,
                item.segment_id,
            )
        )
        clip_results.sort(key=lambda item: item.target_clip_uid)

        summary = _summary(
            inventory=inventory,
            provenance=backend.provenance,
            raw_segments=raw_segments,
            cluster_bindings=cluster_bindings,
            bound_segments=bound_segments,
            clip_results=clip_results,
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "raw_segments.jsonl", raw_segments)
        _write_jsonl(temporary / "cluster_bindings.jsonl", cluster_bindings)
        _write_jsonl(temporary / "bound_segments.jsonl", bound_segments)
        _write_jsonl(temporary / "clip_results.jsonl", clip_results)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        if inventory.mode == "pilot20":
            review_root = temporary / "review_media"
            review_root.mkdir()
            media = _materialize_review_media(
                inventory=inventory,
                raw_segments=raw_segments,
                review_root=review_root,
            )
            (temporary / "review.html").write_text(
                _review_html(
                    inventory=inventory,
                    raw_segments=raw_segments,
                    cluster_bindings=cluster_bindings,
                    bound_segments=bound_segments,
                    clip_results=clip_results,
                    anchors_by_clip=anchors_by_clip,
                    media=media,
                ),
                encoding="utf-8",
            )
        copy_diagnostics = getattr(backend, "copy_diagnostics", None)
        if callable(copy_diagnostics):
            copy_diagnostics(temporary / "diagnostics")
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
