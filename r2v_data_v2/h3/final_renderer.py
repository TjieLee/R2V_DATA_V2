from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.asr_v2_transcription import (
    ASR_V2_SEGMENT_SCHEMA_VERSION,
    ASRV2InventoryRecord,
    ASRV2SegmentJob,
    ASRV2SegmentRecord,
    load_asr_v2_inventory,
    load_asr_v2_summary,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    _inventory_fingerprint as _asr_v2_inventory_fingerprint,
)
from r2v_data_v2.h3.audio_production import (
    H3ProductionCrossPair,
    H3ProductionInPair,
    H3ProductionSummary,
)
from r2v_data_v2.h3.diarization_binding import DiarizationInventory
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.text_usability import (
    TEXT_USABILITY_POLICY_VERSION,
    TEXT_USABILITY_SEGMENT_SCHEMA_VERSION,
    TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT,
    TextUsabilityInventory,
    TextUsabilityPolicy,
    TextUsabilitySegment,
    TextUsabilitySummary,
)
from r2v_data_v2.h3.text_usability import (
    _inventory_fingerprint as _text_usability_inventory_fingerprint,
)
from r2v_data_v2.v3.schemas import AnnotationEntity, ClipRecord, EntityReferenceState

FINAL_RENDERER_VERSION = "h3_deterministic_final_renderer_v1"
FINAL_SAMPLE_SCHEMA_VERSION = "r2v.h3.final_sample.1"
FINAL_SPEECH_SEGMENT_SCHEMA_VERSION = "r2v.h3.final_speech_segment.1"
FINAL_INVENTORY_SCHEMA_VERSION = "r2v.h3.final_inventory.1"
FINAL_SUMMARY_SCHEMA_VERSION = "r2v.h3.final_summary.1"
FINAL_FAILURE_SCHEMA_VERSION = "r2v.h3.final_failure.1"
FINAL_OUTPUT_DIRECTORY = "production/h3"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUBJECT_ID = re.compile(r"subject_([1-9]\d*)")
_PICTURE_ID = re.compile(r"picture_([1-9]\d*)")
_AUDIO_ID = re.compile(r"audio_([1-9]\d*)")

FinalPair = H3ProductionInPair | H3ProductionCrossPair
FinalPairKind = Literal["in_pair", "cross_pair"]
FinalTextStatus = Literal["trusted", "hidden"]
FinalIdentityStatus = Literal[
    "subject_bound",
    "background_identified",
    "unresolved",
]
FinalDialogueStatus = Literal[
    "rendered_dialogue",
    "hidden_text",
    "trusted_text_unrendered_no_subject",
]


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            _compact_json(value.model_dump(mode="json")) + "\n" for value in values
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} in {path} must be an object")
        rows.append(value)
    return rows


class FinalMediaAsset(SchemaModel):
    logical_id: str
    role: Literal[
        "target_video",
        "target_full_audio",
        "subject_picture",
        "voice_reference",
    ]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    media_type: str
    source_path: str
    source_occurrence_id: str | None = None

    @model_validator(mode="after")
    def validate_asset(self) -> FinalMediaAsset:
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts or not self.path.strip():
            raise ValueError("final H3 asset path must be safe and relative")
        if not Path(self.source_path).is_absolute():
            raise ValueError("final H3 source asset path must be absolute")
        if not self.media_type.strip():
            raise ValueError("final H3 asset media type must not be empty")
        if self.role == "target_video":
            if self.logical_id != "video_1" or self.source_occurrence_id is not None:
                raise ValueError("target video must use video_1 without an occurrence")
        elif self.role == "target_full_audio":
            if self.logical_id != "audio_1" or self.source_occurrence_id is not None:
                raise ValueError("target full audio must use audio_1")
        elif self.role == "subject_picture":
            if (
                _PICTURE_ID.fullmatch(self.logical_id) is None
                or self.source_occurrence_id is None
            ):
                raise ValueError("subject picture requires picture_N and an occurrence")
        elif (
            _AUDIO_ID.fullmatch(self.logical_id) is None
            or self.logical_id == "audio_1"
            or self.source_occurrence_id is None
        ):
            raise ValueError("voice reference requires audio_N after audio_1")
        return self


class FinalSubject(SchemaModel):
    subject_index: int = Field(gt=0)
    subject_id: str
    target_entity_id: str
    target_occurrence_id: str
    phrase: str
    picture: FinalMediaAsset
    voice_reference: FinalMediaAsset
    voice_occurrence_id: str
    donor_occurrence_id: str | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> FinalSubject:
        if self.subject_id != f"subject_{self.subject_index}":
            raise ValueError("final H3 subject ID must follow subject order")
        if self.picture.logical_id != f"picture_{self.subject_index}":
            raise ValueError("final H3 subject picture ID is inconsistent")
        if self.voice_reference.logical_id != f"audio_{self.subject_index + 1}":
            raise ValueError("final H3 subject voice ID is inconsistent")
        if self.picture.source_occurrence_id != self.target_occurrence_id:
            raise ValueError("final H3 picture must come from the target occurrence")
        if self.voice_reference.source_occurrence_id != self.voice_occurrence_id:
            raise ValueError("final H3 voice occurrence provenance is inconsistent")
        if not self.phrase.strip() or not self.target_entity_id.strip():
            raise ValueError("final H3 subject semantics must not be empty")
        return self


class FinalSpeechSegment(SchemaModel):
    schema_version: Literal["r2v.h3.final_speech_segment.1"] = (
        FINAL_SPEECH_SEGMENT_SCHEMA_VERSION
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    subject_id: str | None = None
    identity_status: FinalIdentityStatus
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    text_status: FinalTextStatus
    trusted_text: str | None = None
    dialogue_status: FinalDialogueStatus
    rendered_dialogue: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    detected_language: str | None = None
    identity_scope: Literal[
        "direct_anchor_present", "cluster_propagated_only", "unresolved"
    ]
    cluster_binding_status: Literal[
        "candidate_mapped", "ambiguous", "unbound", "conflict"
    ]
    source_diarization_segment_schema_version: Literal[
        "r2v.h3.diarization_segment.2"
    ] = "r2v.h3.diarization_segment.2"
    source_asr_v2_segment_schema_version: Literal["r2v.h3.asr_v2_segment.1"] = (
        ASR_V2_SEGMENT_SCHEMA_VERSION
    )
    source_asr_v2_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_usability_segment_schema_version: Literal[
        "r2v.h3.text_usability_segment.1"
    ] = TEXT_USABILITY_SEGMENT_SCHEMA_VERSION
    source_text_usability_inventory_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    text_usability_policy_version: Literal[
        "h3_asr_v2_text_usability_policy_v1"
    ] = TEXT_USABILITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_segment(self) -> FinalSpeechSegment:
        if self.end_time <= self.start_time:
            raise ValueError("final speech segment interval must be positive")
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("final speech sample interval must be positive")
        expected_occurrence = (
            f"{self.target_clip_uid}/{self.entity_id}"
            if self.entity_id is not None
            else None
        )
        if self.entity_occurrence_id != expected_occurrence:
            raise ValueError("final speech entity occurrence is inconsistent")
        if self.identity_status == "subject_bound":
            if self.subject_id is None or self.entity_id is None:
                raise ValueError("subject-bound speech requires subject and entity")
        elif self.subject_id is not None:
            raise ValueError("non-subject speech cannot claim a subject ID")
        if self.identity_status == "background_identified" and (
            self.cluster_binding_status != "candidate_mapped"
            or self.entity_id is None
        ):
            raise ValueError("background speaker must retain mapped entity evidence")
        if self.identity_status == "unresolved" and self.entity_id is not None:
            raise ValueError("unresolved speech cannot claim an entity")
        if self.text_status == "trusted":
            if self.trusted_text is None or not self.trusted_text.strip():
                raise ValueError("trusted speech text must not be empty")
            expected_dialogue_status = (
                "rendered_dialogue"
                if self.subject_id is not None
                else "trusted_text_unrendered_no_subject"
            )
        else:
            if self.trusted_text is not None:
                raise ValueError("hidden text must remain null in final output")
            expected_dialogue_status = "hidden_text"
        if self.dialogue_status != expected_dialogue_status:
            raise ValueError("final dialogue status is inconsistent")
        if self.dialogue_status == "rendered_dialogue":
            assert self.subject_id is not None and self.trusted_text is not None
            index = _SUBJECT_ID.fullmatch(self.subject_id)
            expected = f"<Subject {index.group(1)}> says <d>{self.trusted_text}</d>"
            if self.rendered_dialogue != expected:
                raise ValueError("rendered dialogue must preserve trusted text exactly")
        elif self.rendered_dialogue is not None:
            raise ValueError("unrendered speech cannot publish dialogue text")
        if self.detected_language is not None and not self.detected_language.strip():
            raise ValueError("detected language must be null or non-empty")
        return self


class FinalSourceFingerprints(SchemaModel):
    pair_production: str = Field(pattern=r"^[0-9a-f]{64}$")
    diarization_inventory: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_v2_inventory: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_usability_inventory: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_usability_policy: str = Field(pattern=r"^[0-9a-f]{64}$")


class FinalH3Sample(SchemaModel):
    schema_version: Literal["r2v.h3.final_sample.1"] = FINAL_SAMPLE_SCHEMA_VERSION
    sample_id: str
    pair_id: str
    pair_kind: FinalPairKind
    target_clip_uid: str
    target_video: FinalMediaAsset
    target_full_audio: FinalMediaAsset
    subjects: list[FinalSubject]
    speech_segments: list[FinalSpeechSegment]
    detected_languages: list[str]
    language_metadata_source: Literal["whisper_large_v3_asr_v2"] = (
        "whisper_large_v3_asr_v2"
    )
    language_conditioning_applied: Literal[False] = False
    visual_r2v_instruction: str
    visual_r2v_instruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_annotation: str
    source_visual_clip_path: str
    source_visual_clip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprints: FinalSourceFingerprints
    renderer_version: Literal["h3_deterministic_final_renderer_v1"] = (
        FINAL_RENDERER_VERSION
    )
    parent_quota_applied: Literal[False] = False
    model_calls: Literal[0] = 0
    mllm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_sample(self) -> FinalH3Sample:
        if self.sample_id != self.pair_id or not self.subjects:
            raise ValueError("final H3 sample identity/subjects are incomplete")
        if self.target_video.logical_id != "video_1":
            raise ValueError("final H3 target video must use video_1")
        if self.target_full_audio.logical_id != "audio_1":
            raise ValueError("final H3 target full audio must use audio_1")
        indices = [item.subject_index for item in self.subjects]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("final H3 subject indices must be contiguous")
        if [item.subject_id for item in self.subjects] != [
            f"subject_{index}" for index in indices
        ]:
            raise ValueError("final H3 subject IDs must follow pair order")
        target_occurrences = [item.target_occurrence_id for item in self.subjects]
        if len(target_occurrences) != len(set(target_occurrences)) or any(
            not value.startswith(f"{self.target_clip_uid}/")
            for value in target_occurrences
        ):
            raise ValueError("final H3 target occurrences are invalid")
        if self.pair_kind == "in_pair":
            if any(
                item.voice_occurrence_id != item.target_occurrence_id
                or item.donor_occurrence_id is not None
                for item in self.subjects
            ):
                raise ValueError("in-pair voice must come from the target occurrence")
        elif any(
            item.voice_occurrence_id == item.target_occurrence_id
            or item.donor_occurrence_id != item.voice_occurrence_id
            for item in self.subjects
        ):
            raise ValueError("cross-pair may change only voice occurrence provenance")
        ordered = sorted(
            self.speech_segments,
            key=lambda item: (
                item.start_time,
                item.end_time,
                item.segment_id,
                item.speaker_cluster_id,
            ),
        )
        if self.speech_segments != ordered or any(
            item.target_clip_uid != self.target_clip_uid
            for item in self.speech_segments
        ):
            raise ValueError("final H3 speech must preserve ordered target timeline")
        expected_languages = sorted(
            {
                item.detected_language
                for item in self.speech_segments
                if item.detected_language is not None
            }
        )
        if self.detected_languages != expected_languages:
            raise ValueError("final H3 detected language metadata is inconsistent")
        if not self.visual_r2v_instruction.strip():
            raise ValueError("final H3 sample requires frozen visual instruction")
        if self.visual_r2v_instruction_sha256 != _sha256_bytes(
            self.visual_r2v_instruction.encode("utf-8")
        ):
            raise ValueError("final H3 visual instruction hash is inconsistent")
        expected_annotation = render_final_annotation(
            visual_instruction=self.visual_r2v_instruction,
            subjects=self.subjects,
            speech_segments=self.speech_segments,
        )
        if self.rendered_annotation != expected_annotation:
            raise ValueError("final H3 annotation is not deterministic")
        return self


class FinalSourceFile(SchemaModel):
    role: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> FinalSourceFile:
        if not self.role.strip() or not Path(self.path).is_absolute():
            raise ValueError("final H3 source file provenance is incomplete")
        return self


class FinalH3Inventory(SchemaModel):
    schema_version: Literal["r2v.h3.final_inventory.1"] = (
        FINAL_INVENTORY_SCHEMA_VERSION
    )
    output_root: str
    source_files: list[FinalSourceFile]
    source_fingerprints: FinalSourceFingerprints
    renderer_version: Literal["h3_deterministic_final_renderer_v1"] = (
        FINAL_RENDERER_VERSION
    )
    pair_row_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    failed_sample_count: int = Field(ge=0)
    text_usability_policy_version: Literal[
        "h3_asr_v2_text_usability_policy_v1"
    ] = TEXT_USABILITY_POLICY_VERSION
    language_conditioning_applied: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    model_calls: Literal[0] = 0
    mllm_calls: Literal[0] = 0
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> FinalH3Inventory:
        if self.pair_row_count != self.sample_count + self.failed_sample_count:
            raise ValueError("final H3 pair/sample counts must reconcile")
        roles = [item.role for item in self.source_files]
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise ValueError("final H3 source files must be unique and ordered")
        if self.inventory_fingerprint != _final_inventory_fingerprint(self):
            raise ValueError("final H3 inventory fingerprint is inconsistent")
        return self


class FinalH3Summary(SchemaModel):
    schema_version: Literal["r2v.h3.final_summary.1"] = FINAL_SUMMARY_SCHEMA_VERSION
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    renderer_version: Literal["h3_deterministic_final_renderer_v1"] = (
        FINAL_RENDERER_VERSION
    )
    in_pair_sample_count: int = Field(ge=0)
    cross_pair_sample_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    speech_segment_count: int = Field(ge=0)
    subject_bound_segment_count: int = Field(ge=0)
    background_identified_segment_count: int = Field(ge=0)
    unresolved_segment_count: int = Field(ge=0)
    trusted_text_segment_count: int = Field(ge=0)
    hidden_text_segment_count: int = Field(ge=0)
    rendered_dialogue_segment_count: int = Field(ge=0)
    trusted_text_unrendered_no_subject_count: int = Field(ge=0)
    detected_language_counts: dict[str, int]
    multi_language_sample_count: int = Field(ge=0)
    target_video_asset_count: int = Field(ge=0)
    full_audio_asset_count: int = Field(ge=0)
    picture_asset_count: int = Field(ge=0)
    voice_asset_count: int = Field(ge=0)
    unique_asset_count: int = Field(ge=0)
    failed_sample_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    parent_quota_applied: Literal[False] = False
    model_calls: Literal[0] = 0
    mllm_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_counts(self) -> FinalH3Summary:
        if self.total_sample_count != (
            self.in_pair_sample_count + self.cross_pair_sample_count
        ):
            raise ValueError("final H3 pair-kind counts must reconcile")
        if self.speech_segment_count != (
            self.subject_bound_segment_count
            + self.background_identified_segment_count
            + self.unresolved_segment_count
        ):
            raise ValueError("final H3 identity counts must reconcile")
        if self.speech_segment_count != (
            self.trusted_text_segment_count + self.hidden_text_segment_count
        ):
            raise ValueError("final H3 text counts must reconcile")
        if self.rendered_dialogue_segment_count > self.trusted_text_segment_count:
            raise ValueError("rendered dialogue cannot exceed trusted text")
        if self.trusted_text_segment_count != (
            self.rendered_dialogue_segment_count
            + self.trusted_text_unrendered_no_subject_count
        ):
            raise ValueError("trusted final H3 text disposition must reconcile")
        if self.failed_sample_count != sum(self.failure_reason_counts.values()):
            raise ValueError("final H3 failure counts must reconcile")
        asset_sum = (
            self.target_video_asset_count
            + self.full_audio_asset_count
            + self.picture_asset_count
            + self.voice_asset_count
        )
        if self.unique_asset_count != asset_sum:
            raise ValueError("final H3 asset counts must reconcile")
        return self


class FinalH3Failure(SchemaModel):
    schema_version: Literal["r2v.h3.final_failure.1"] = FINAL_FAILURE_SCHEMA_VERSION
    pair_id: str
    pair_kind: FinalPairKind
    target_clip_uid: str
    reason_code: str
    message: str

    @model_validator(mode="after")
    def validate_failure(self) -> FinalH3Failure:
        if any(
            not value.strip()
            for value in (self.pair_id, self.target_clip_uid, self.reason_code, self.message)
        ):
            raise ValueError("final H3 failure evidence must not be empty")
        return self


def _final_inventory_fingerprint(
    inventory: FinalH3Inventory | Mapping[str, object],
) -> str:
    values = (
        inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if isinstance(inventory, FinalH3Inventory)
        else {
            key: value
            for key, value in inventory.items()
            if key != "inventory_fingerprint"
        }
    )
    return _sha256_bytes(_compact_json(values).encode("utf-8"))


def render_final_annotation(
    *,
    visual_instruction: str,
    subjects: Sequence[FinalSubject],
    speech_segments: Sequence[FinalSpeechSegment],
) -> str:
    if not visual_instruction.strip():
        raise ValueError("final H3 annotation requires frozen visual instruction")
    mapping_lines = ["<Audio 1> is the target full audio."]
    for subject in subjects:
        mapping_lines.extend(
            [
                (
                    f"<Subject {subject.subject_index}> uses "
                    f"<Picture {subject.subject_index}>."
                ),
                (
                    f"<Audio {subject.subject_index + 1}> is the voice reference "
                    f"for <Subject {subject.subject_index}>."
                ),
            ]
        )
    dialogue_lines = [
        segment.rendered_dialogue
        for segment in speech_segments
        if segment.rendered_dialogue is not None
    ]
    sections = [visual_instruction, "\n".join(mapping_lines)]
    if dialogue_lines:
        sections.append("\n".join(dialogue_lines))
    return "\n\n".join(sections)


@dataclass(frozen=True)
class _SourceBundle:
    audio_run_root: Path
    production_root: Path
    pairs_root: Path
    diarization_root: Path
    asr_root: Path
    text_root: Path
    in_pairs: tuple[H3ProductionInPair, ...]
    cross_pairs: tuple[H3ProductionCrossPair, ...]
    asr_inventory: ASRV2InventoryRecord
    asr_records_by_clip: dict[str, tuple[ASRV2SegmentRecord, ...]]
    text_records_by_key: dict[tuple[str, str, str], TextUsabilitySegment]
    text_inventory: TextUsabilityInventory
    fingerprints: FinalSourceFingerprints
    source_hashes: dict[Path, str]


@dataclass(frozen=True)
class _VisualTarget:
    clip: ClipRecord
    clip_path: Path
    clip_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    source_run_root: Path
    entities_by_id: dict[str, AnnotationEntity]
    references_by_id: dict[str, EntityReferenceState]


class _SampleFailure(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _absolute_file(value: str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise _SampleFailure("source_asset_path_invalid", f"{label} must be absolute")
    try:
        path = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _SampleFailure("source_asset_missing", f"{label} is missing: {raw}") from exc
    if not path.is_file():
        raise _SampleFailure("source_asset_missing", f"{label} is not a file: {path}")
    return path


def _path_inside(root: Path, value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {root}") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a file")
    return path


def _source_file(role: str, path: Path, hashes: Mapping[Path, str]) -> FinalSourceFile:
    return FinalSourceFile(role=role, path=str(path), sha256=hashes[path])


def _pair_production_fingerprint(
    *, in_pairs_sha256: str, cross_pairs_sha256: str, summary_sha256: str
) -> str:
    return _sha256_bytes(
        _compact_json(
            {
                "contract": "h3_production_pair_bundle_v1",
                "in_pairs_sha256": in_pairs_sha256,
                "cross_pairs_sha256": cross_pairs_sha256,
                "summary_sha256": summary_sha256,
            }
        ).encode("utf-8")
    )


def _load_sources(
    *, audio_run_root: Path, expected_asr_inventory_fingerprint: str
) -> _SourceBundle:
    root = audio_run_root.expanduser().resolve(strict=True)
    production = (root / "production").resolve(strict=True)
    pairs_root = (production / "pairs").resolve(strict=True)
    diarization_root = (production / "diarization").resolve(strict=True)
    asr_root = (production / "asr_v2").resolve(strict=True)
    text_root = (production / "text_usability").resolve(strict=True)

    pairs_paths = {
        "pair_in_pairs": pairs_root / "in_pairs.jsonl",
        "pair_cross_pairs": pairs_root / "cross_pairs.jsonl",
        "pair_summary": pairs_root / "summary.json",
    }
    for path in pairs_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen pair artifact: {path}")
    in_pairs = tuple(
        H3ProductionInPair.model_validate(row)
        for row in _read_jsonl(pairs_paths["pair_in_pairs"])
    )
    cross_pairs = tuple(
        H3ProductionCrossPair.model_validate(row)
        for row in _read_jsonl(pairs_paths["pair_cross_pairs"])
    )
    pair_ids = [item.pair_id for item in (*in_pairs, *cross_pairs)]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("frozen pair rows contain duplicate pair IDs")
    pair_summary = H3ProductionSummary.model_validate_json(
        pairs_paths["pair_summary"].read_text(encoding="utf-8")
    )
    if (
        pair_summary.in_pair_clip_sample_count != len(in_pairs)
        or pair_summary.cross_pair_clip_sample_count != len(cross_pairs)
    ):
        raise ValueError("frozen pair rows differ from pair summary")

    asr_paths = {
        "asr_v2_inventory": asr_root / "inventory.json",
        "asr_v2_segments": asr_root / "segments.jsonl",
        "asr_v2_summary": asr_root / "summary.json",
    }
    for path in asr_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen ASR V2 artifact: {path}")
    asr_inventory = load_asr_v2_inventory(asr_paths["asr_v2_inventory"])
    if asr_inventory.mode != "production":
        raise ValueError("final H3 renderer requires production ASR V2")
    if asr_inventory.inventory_fingerprint != _asr_v2_inventory_fingerprint(
        asr_inventory
    ):
        raise ValueError("production ASR V2 inventory fingerprint is inconsistent")
    if asr_inventory.inventory_fingerprint != expected_asr_inventory_fingerprint:
        raise ValueError("production ASR V2 inventory is not the frozen source")
    asr_summary = load_asr_v2_summary(asr_paths["asr_v2_summary"])
    if (
        asr_summary.mode != "production"
        or asr_summary.inventory_fingerprint != asr_inventory.inventory_fingerprint
    ):
        raise ValueError("production ASR V2 summary provenance is inconsistent")
    asr_records = tuple(
        ASRV2SegmentRecord.model_validate(row)
        for row in _read_jsonl(asr_paths["asr_v2_segments"])
    )
    if len(asr_records) != len(asr_inventory.jobs):
        raise ValueError("production ASR V2 records do not cover inventory jobs")
    job_fields = tuple(ASRV2SegmentJob.model_fields)
    for job, record in zip(asr_inventory.jobs, asr_records, strict=True):
        if any(getattr(record, field) != getattr(job, field) for field in job_fields):
            raise ValueError("production ASR V2 records differ from inventory order")
    status_counts = Counter(item.status for item in asr_records)
    if (
        status_counts["transcribed"] != asr_summary.transcribed_count
        or status_counts["uncertain"] != asr_summary.uncertain_count
        or status_counts["failed"] != asr_summary.failed_count
    ):
        raise ValueError("production ASR V2 records differ from summary counts")

    text_paths = {
        "text_usability_inventory": text_root / "inventory.json",
        "text_usability_segments": text_root / "segments.jsonl",
        "text_usability_summary": text_root / "summary.json",
    }
    for path in text_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen text-usability artifact: {path}")
    text_inventory = TextUsabilityInventory.model_validate_json(
        text_paths["text_usability_inventory"].read_text(encoding="utf-8")
    )
    text_summary = TextUsabilitySummary.model_validate_json(
        text_paths["text_usability_summary"].read_text(encoding="utf-8")
    )
    if text_inventory.inventory_fingerprint != _text_usability_inventory_fingerprint(
        text_inventory
    ):
        raise ValueError("text-usability inventory fingerprint is inconsistent")
    policy = TextUsabilityPolicy()
    if (
        text_inventory.source_asr_v2_inventory_fingerprint
        != asr_inventory.inventory_fingerprint
        or text_inventory.policy_version != policy.version
        or text_inventory.policy_fingerprint != policy.fingerprint()
        or text_summary.inventory_fingerprint != text_inventory.inventory_fingerprint
        or text_summary.policy_fingerprint != policy.fingerprint()
    ):
        raise ValueError("text-usability source or policy provenance is inconsistent")
    actual_asr_hashes = {path: _sha256_file(path) for path in asr_paths.values()}
    expected_text_sources = (
        (
            Path(text_inventory.source_asr_v2_inventory_path),
            asr_paths["asr_v2_inventory"],
            text_inventory.source_asr_v2_inventory_sha256,
        ),
        (
            Path(text_inventory.source_asr_v2_segments_path),
            asr_paths["asr_v2_segments"],
            text_inventory.source_asr_v2_segments_sha256,
        ),
        (
            Path(text_inventory.source_asr_v2_summary_path),
            asr_paths["asr_v2_summary"],
            text_inventory.source_asr_v2_summary_sha256,
        ),
    )
    for declared, actual, expected_hash in expected_text_sources:
        if (
            declared.expanduser().resolve(strict=True) != actual
            or actual_asr_hashes[actual] != expected_hash
        ):
            raise ValueError("text-usability source ASR V2 evidence changed")
    text_records = tuple(
        TextUsabilitySegment.model_validate(row)
        for row in _read_jsonl(text_paths["text_usability_segments"])
    )
    if (
        len(text_records) != len(asr_records)
        or text_summary.segment_count != len(text_records)
    ):
        raise ValueError("text-usability records do not cover ASR V2 records")
    text_by_key: dict[tuple[str, str, str], TextUsabilitySegment] = {}
    for asr, text in zip(asr_records, text_records, strict=True):
        key = (asr.target_clip_uid, asr.segment_id, asr.speaker_cluster_id)
        if key in text_by_key:
            raise ValueError("text-usability segment identity is duplicated")
        if (
            key
            != (text.target_clip_uid, text.segment_id, text.speaker_cluster_id)
            or text.entity_id != asr.entity_id
            or text.entity_occurrence_id != asr.entity_occurrence_id
            or text.cluster_binding_status != asr.cluster_binding_status
            or text.identity_scope != asr.identity_scope
            or text.source_asr_v2_segment_status != asr.status
            or text.source_asr_v2_request_fingerprint != asr.request_fingerprint
            or text.source_asr_v2_inventory_fingerprint
            != asr_inventory.inventory_fingerprint
        ):
            raise ValueError("text-usability record differs from ASR V2 source")
        text_by_key[key] = text

    diarization_inventory_path = _path_inside(
        diarization_root,
        asr_inventory.source_diarization_inventory_path,
        label="diarization inventory",
    )
    if diarization_inventory_path != diarization_root / "inventory.json":
        raise ValueError("ASR V2 does not reference fixed production diarization")
    diarization_inventory = DiarizationInventory.model_validate_json(
        diarization_inventory_path.read_text(encoding="utf-8")
    )
    if (
        diarization_inventory.mode != "production"
        or diarization_inventory.inventory_fingerprint
        != _diarization_inventory_fingerprint(
            source_pairs_sha256=diarization_inventory.source_pairs_sha256,
            source_asr_inventory_fingerprint=(
                diarization_inventory.source_asr_inventory_fingerprint
            ),
            mode=diarization_inventory.mode,
            targets=diarization_inventory.targets,
        )
        or diarization_inventory.inventory_fingerprint
        != asr_inventory.source_diarization_inventory_fingerprint
    ):
        raise ValueError("production diarization inventory provenance is inconsistent")
    if any(
        job.source_diarization_inventory_fingerprint
        != diarization_inventory.inventory_fingerprint
        for job in asr_inventory.jobs
    ):
        raise ValueError("ASR V2 segment provenance differs from diarization source")
    if (
        Path(diarization_inventory.source_pairs_path)
        .expanduser()
        .resolve(strict=True)
        != pairs_paths["pair_in_pairs"]
        or diarization_inventory.source_pairs_sha256
        != _sha256_file(pairs_paths["pair_in_pairs"])
    ):
        raise ValueError("production diarization pair source changed")
    diarization_sources = {
        "diarization_inventory": (
            diarization_inventory_path,
            asr_inventory.source_diarization_inventory_sha256,
        ),
        "diarization_raw_segments": (
            _path_inside(
                diarization_root,
                asr_inventory.source_diarization_raw_segments_path,
                label="diarization raw segments",
            ),
            asr_inventory.source_diarization_raw_segments_sha256,
        ),
        "diarization_bound_segments": (
            _path_inside(
                diarization_root,
                asr_inventory.source_diarization_bound_segments_path,
                label="diarization bound segments",
            ),
            asr_inventory.source_diarization_bound_segments_sha256,
        ),
        "diarization_cluster_bindings": (
            _path_inside(
                diarization_root,
                asr_inventory.source_diarization_cluster_bindings_path,
                label="diarization cluster bindings",
            ),
            asr_inventory.source_diarization_cluster_bindings_sha256,
        ),
        "diarization_summary": (
            _path_inside(
                diarization_root,
                asr_inventory.source_diarization_summary_path,
                label="diarization summary",
            ),
            asr_inventory.source_diarization_summary_sha256,
        ),
    }
    for path, expected_hash in diarization_sources.values():
        if _sha256_file(path) != expected_hash:
            raise ValueError("production diarization source hash changed")

    source_paths = {
        **pairs_paths,
        **asr_paths,
        **text_paths,
        **{role: value[0] for role, value in diarization_sources.items()},
    }
    source_hashes = {path: _sha256_file(path) for path in source_paths.values()}
    pair_fingerprint = _pair_production_fingerprint(
        in_pairs_sha256=source_hashes[pairs_paths["pair_in_pairs"]],
        cross_pairs_sha256=source_hashes[pairs_paths["pair_cross_pairs"]],
        summary_sha256=source_hashes[pairs_paths["pair_summary"]],
    )
    records_by_clip: dict[str, list[ASRV2SegmentRecord]] = defaultdict(list)
    for record in asr_records:
        records_by_clip[record.target_clip_uid].append(record)
    target_ids = {item.target_clip_uid for item in asr_inventory.targets}
    pair_target_ids = {
        item.target_clip_uid for item in (*in_pairs, *cross_pairs)
    }
    if pair_target_ids - target_ids:
        raise ValueError("frozen pair target is absent from ASR V2 inventory")
    return _SourceBundle(
        audio_run_root=root,
        production_root=production,
        pairs_root=pairs_root,
        diarization_root=diarization_root,
        asr_root=asr_root,
        text_root=text_root,
        in_pairs=in_pairs,
        cross_pairs=cross_pairs,
        asr_inventory=asr_inventory,
        asr_records_by_clip={
            key: tuple(value) for key, value in records_by_clip.items()
        },
        text_records_by_key=text_by_key,
        text_inventory=text_inventory,
        fingerprints=FinalSourceFingerprints(
            pair_production=pair_fingerprint,
            diarization_inventory=diarization_inventory.inventory_fingerprint,
            asr_v2_inventory=asr_inventory.inventory_fingerprint,
            text_usability_inventory=text_inventory.inventory_fingerprint,
            text_usability_policy=policy.fingerprint(),
        ),
        source_hashes=source_hashes,
    )


_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class _AssetStore:
    def __init__(self, temporary_root: Path | None) -> None:
        self.temporary_root = temporary_root
        self.source_hashes: dict[Path, str] = {}

    def asset(
        self,
        *,
        source_value: str,
        category: Literal["videos", "full_audio", "pictures", "voices"],
        logical_id: str,
        role: Literal[
            "target_video",
            "target_full_audio",
            "subject_picture",
            "voice_reference",
        ],
        source_occurrence_id: str | None = None,
    ) -> FinalMediaAsset:
        source = _absolute_file(source_value, label=role)
        digest = _sha256_file(source)
        self.source_hashes[source] = digest
        suffix = source.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ".bin"
        relative = Path("assets") / category / f"{digest}{suffix}"
        if self.temporary_root is not None:
            destination = self.temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _sha256_file(destination) != digest:
                    raise ValueError("content-addressed H3 asset hash collision")
            else:
                shutil.copyfile(source, destination)
                if _sha256_file(destination) != digest:
                    raise ValueError("copied H3 asset differs from source")
        return FinalMediaAsset(
            logical_id=logical_id,
            role=role,
            path=relative.as_posix(),
            sha256=digest,
            byte_size=source.stat().st_size,
            media_type=_MEDIA_TYPES.get(suffix, "application/octet-stream"),
            source_path=str(source),
            source_occurrence_id=source_occurrence_id,
        )


def _resolve_visual_reference(
    *, source_run: Path, clip_uid: str, value: str
) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _SampleFailure(
            "visual_reference_provenance_invalid",
            "published Visual reference path must be safe and relative",
        )
    base = source_run if relative.parts and relative.parts[0] == "clips" else (
        source_run / "clips" / clip_uid
    )
    try:
        path = (base / relative).resolve(strict=True)
        path.relative_to(source_run)
    except (FileNotFoundError, ValueError) as exc:
        raise _SampleFailure(
            "visual_reference_provenance_invalid",
            "published Visual reference cannot be resolved under source run",
        ) from exc
    if not path.is_file():
        raise _SampleFailure(
            "visual_reference_provenance_invalid",
            "published Visual reference is not a file",
        )
    return path


def _load_visual_target(
    *, pair: FinalPair, cache: dict[str, _VisualTarget]
) -> _VisualTarget:
    cached = cache.get(pair.target_clip_uid)
    if cached is not None:
        sidecar = _absolute_file(
            pair.target_audio_binding_path, label="target audio binding sidecar"
        )
        pair_video = _absolute_file(pair.target_video_path, label="target video")
        if sidecar != cached.sidecar_path or pair_video != Path(
            cached.clip.source.video_path
        ).expanduser().resolve(strict=True):
            raise _SampleFailure(
                "visual_provenance_invalid",
                "repeated pair target uses inconsistent Visual provenance",
            )
        return cached
    sidecar = _absolute_file(
        pair.target_audio_binding_path, label="target audio binding sidecar"
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise _SampleFailure(
            "visual_provenance_invalid", "target audio binding must be an object"
        )
    source_run_value = payload.get("source_run_root")
    sidecar_video_value = payload.get("source_video_path")
    if (
        payload.get("clip_uid") != pair.target_clip_uid
        or payload.get("status") != "ready"
        or not isinstance(source_run_value, str)
        or not source_run_value.strip()
        or not isinstance(sidecar_video_value, str)
        or not sidecar_video_value.strip()
    ):
        raise _SampleFailure(
            "visual_provenance_invalid",
            "target audio binding lacks ready Visual source provenance",
        )
    source_run = Path(source_run_value).expanduser().resolve(strict=True)
    clip_path = (source_run / "clips" / pair.target_clip_uid / "clip.json").resolve(
        strict=True
    )
    try:
        clip_path.relative_to(source_run)
    except ValueError as exc:
        raise _SampleFailure(
            "visual_provenance_invalid", "Visual clip path escapes source run"
        ) from exc
    clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
    if (
        clip.clip_uid != pair.target_clip_uid
        or clip.instruction is None
        or clip.instruction.status != "ready"
        or not clip.instruction.r2v_instruction.strip()
        or not clip.export.accepted
    ):
        raise _SampleFailure(
            "visual_instruction_unavailable",
            "frozen Visual clip lacks an accepted canonical R2V instruction",
        )
    source_video = Path(clip.source.video_path).expanduser().resolve(strict=True)
    pair_video = _absolute_file(pair.target_video_path, label="target video")
    sidecar_video = _absolute_file(sidecar_video_value, label="sidecar source video")
    if source_video != pair_video or sidecar_video != pair_video:
        raise _SampleFailure(
            "visual_provenance_invalid",
            "pair target video differs from frozen Visual clip",
        )
    if clip.annotation is None or clip.annotation.status != "ready":
        raise _SampleFailure(
            "visual_provenance_invalid", "frozen Visual annotation is not ready"
        )
    result = _VisualTarget(
        clip=clip,
        clip_path=clip_path,
        clip_sha256=_sha256_file(clip_path),
        sidecar_path=sidecar,
        sidecar_sha256=_sha256_file(sidecar),
        source_run_root=source_run,
        entities_by_id={item.entity_id: item for item in clip.annotation.entities},
        references_by_id={item.entity_id: item for item in clip.references.entities},
    )
    cache[pair.target_clip_uid] = result
    return result


def _subject_values(pair: FinalPair) -> list[dict[str, object]]:
    if isinstance(pair, H3ProductionInPair):
        return [
            {
                "subject_index": item.subject_index,
                "target_occurrence_id": item.target_occurrence_id,
                "target_entity_id": item.target_entity_id,
                "target_visual_reference_path": item.target_visual_reference_path,
                "voice_occurrence_id": item.target_occurrence_id,
                "voice_reference_path": item.target_primary_voice_reference_path,
                "donor_occurrence_id": None,
            }
            for item in pair.subjects
        ]
    return [
        {
            "subject_index": item.subject_index,
            "target_occurrence_id": item.target_occurrence_id,
            "target_entity_id": item.target_occurrence_id.rsplit("/", 1)[-1],
            "target_visual_reference_path": item.target_visual_reference_path,
            "voice_occurrence_id": item.donor_occurrence_id,
            "voice_reference_path": item.donor_primary_voice_reference_path,
            "donor_occurrence_id": item.donor_occurrence_id,
        }
        for item in pair.mappings
    ]


def _final_speech_segments(
    *,
    source: _SourceBundle,
    target_clip_uid: str,
    subject_by_entity: Mapping[str, str],
) -> list[FinalSpeechSegment]:
    output: list[FinalSpeechSegment] = []
    for asr in source.asr_records_by_clip.get(target_clip_uid, ()):
        key = (asr.target_clip_uid, asr.segment_id, asr.speaker_cluster_id)
        text = source.text_records_by_key[key]
        subject_id = (
            subject_by_entity.get(asr.entity_id)
            if asr.entity_id is not None
            else None
        )
        if subject_id is not None:
            identity_status: FinalIdentityStatus = "subject_bound"
        elif asr.entity_id is not None:
            identity_status = "background_identified"
        else:
            identity_status = "unresolved"
        if text.text_status == "hidden":
            dialogue_status: FinalDialogueStatus = "hidden_text"
            rendered = None
        elif subject_id is None:
            dialogue_status = "trusted_text_unrendered_no_subject"
            rendered = None
        else:
            dialogue_status = "rendered_dialogue"
            subject_match = _SUBJECT_ID.fullmatch(subject_id)
            assert subject_match is not None and text.trusted_text is not None
            rendered = (
                f"<Subject {subject_match.group(1)}> says "
                f"<d>{text.trusted_text}</d>"
            )
        output.append(
            FinalSpeechSegment(
                target_clip_uid=asr.target_clip_uid,
                segment_id=asr.segment_id,
                speaker_cluster_id=asr.speaker_cluster_id,
                entity_id=asr.entity_id,
                entity_occurrence_id=asr.entity_occurrence_id,
                subject_id=subject_id,
                identity_status=identity_status,
                start_time=asr.start_time,
                end_time=asr.end_time,
                source_start_sample=asr.source_start_sample,
                source_end_sample=asr.source_end_sample,
                source_sample_rate_hz=asr.source_sample_rate_hz,
                text_status=text.text_status,
                trusted_text=text.trusted_text,
                dialogue_status=dialogue_status,
                rendered_dialogue=rendered,
                language_probability=text.language_probability,
                detected_language=asr.language,
                identity_scope=asr.identity_scope,
                cluster_binding_status=asr.cluster_binding_status,
                source_asr_v2_request_fingerprint=asr.request_fingerprint,
                source_text_usability_inventory_fingerprint=(
                    source.text_inventory.inventory_fingerprint
                ),
            )
        )
    output.sort(
        key=lambda item: (
            item.start_time,
            item.end_time,
            item.segment_id,
            item.speaker_cluster_id,
        )
    )
    return output


def _build_sample(
    *,
    pair: FinalPair,
    pair_kind: FinalPairKind,
    source: _SourceBundle,
    assets: _AssetStore,
    visual_cache: dict[str, _VisualTarget],
) -> FinalH3Sample:
    visual = _load_visual_target(pair=pair, cache=visual_cache)
    target_audio = _absolute_file(pair.target_full_audio_path, label="target full audio")
    asr_target = next(
        (
            item
            for item in source.asr_inventory.targets
            if item.target_clip_uid == pair.target_clip_uid
        ),
        None,
    )
    if asr_target is None:
        raise _SampleFailure(
            "asr_target_missing", "pair target is absent from ASR V2 inventory"
        )
    if (
        Path(asr_target.source_audio_path).expanduser().resolve(strict=True)
        != target_audio
        or _sha256_file(target_audio) != asr_target.source_audio_sha256
    ):
        raise _SampleFailure(
            "target_audio_provenance_invalid",
            "pair target full audio differs from ASR V2 source",
        )
    final_subjects: list[FinalSubject] = []
    for values in _subject_values(pair):
        subject_index = int(values["subject_index"])
        target_occurrence_id = str(values["target_occurrence_id"])
        target_entity_id = str(values["target_entity_id"])
        if target_occurrence_id != f"{pair.target_clip_uid}/{target_entity_id}":
            raise _SampleFailure(
                "pair_subject_identity_invalid",
                "pair target occurrence and entity identity differ",
            )
        entity = visual.entities_by_id.get(target_entity_id)
        reference = visual.references_by_id.get(target_entity_id)
        if entity is None or entity.reference_type != "subject":
            raise _SampleFailure(
                "pair_subject_identity_invalid",
                "pair subject is absent from the frozen Visual annotation",
            )
        if reference is None or reference.status != "ready":
            raise _SampleFailure(
                "visual_reference_unavailable",
                "pair subject lacks a ready frozen Visual reference",
            )
        published_visual = _resolve_visual_reference(
            source_run=visual.source_run_root,
            clip_uid=pair.target_clip_uid,
            value=str(reference.image_path),
        )
        pair_visual = _absolute_file(
            str(values["target_visual_reference_path"]),
            label="target visual reference",
        )
        if _sha256_file(published_visual) != _sha256_file(pair_visual):
            raise _SampleFailure(
                "visual_reference_provenance_invalid",
                "pair picture differs from frozen Visual reference",
            )
        voice_occurrence_id = str(values["voice_occurrence_id"])
        final_subjects.append(
            FinalSubject(
                subject_index=subject_index,
                subject_id=f"subject_{subject_index}",
                target_entity_id=target_entity_id,
                target_occurrence_id=target_occurrence_id,
                phrase=entity.phrase,
                picture=assets.asset(
                    source_value=str(pair_visual),
                    category="pictures",
                    logical_id=f"picture_{subject_index}",
                    role="subject_picture",
                    source_occurrence_id=target_occurrence_id,
                ),
                voice_reference=assets.asset(
                    source_value=str(values["voice_reference_path"]),
                    category="voices",
                    logical_id=f"audio_{subject_index + 1}",
                    role="voice_reference",
                    source_occurrence_id=voice_occurrence_id,
                ),
                voice_occurrence_id=voice_occurrence_id,
                donor_occurrence_id=(
                    str(values["donor_occurrence_id"])
                    if values["donor_occurrence_id"] is not None
                    else None
                ),
            )
        )
    subject_by_entity = {
        item.target_entity_id: item.subject_id for item in final_subjects
    }
    speech = _final_speech_segments(
        source=source,
        target_clip_uid=pair.target_clip_uid,
        subject_by_entity=subject_by_entity,
    )
    visual_instruction = visual.clip.instruction.r2v_instruction
    return FinalH3Sample(
        sample_id=pair.pair_id,
        pair_id=pair.pair_id,
        pair_kind=pair_kind,
        target_clip_uid=pair.target_clip_uid,
        target_video=assets.asset(
            source_value=pair.target_video_path,
            category="videos",
            logical_id="video_1",
            role="target_video",
        ),
        target_full_audio=assets.asset(
            source_value=str(target_audio),
            category="full_audio",
            logical_id="audio_1",
            role="target_full_audio",
        ),
        subjects=final_subjects,
        speech_segments=speech,
        detected_languages=sorted(
            {
                item.detected_language
                for item in speech
                if item.detected_language is not None
            }
        ),
        visual_r2v_instruction=visual_instruction,
        visual_r2v_instruction_sha256=_sha256_bytes(
            visual_instruction.encode("utf-8")
        ),
        rendered_annotation=render_final_annotation(
            visual_instruction=visual_instruction,
            subjects=final_subjects,
            speech_segments=speech,
        ),
        source_visual_clip_path=str(visual.clip_path),
        source_visual_clip_sha256=visual.clip_sha256,
        source_fingerprints=source.fingerprints,
    )


@dataclass(frozen=True)
class _BuildResult:
    source: _SourceBundle
    samples: tuple[FinalH3Sample, ...]
    failures: tuple[FinalH3Failure, ...]
    asset_source_hashes: dict[Path, str]
    visual_source_hashes: dict[Path, str]


def _build(
    *,
    audio_run_root: Path,
    temporary_root: Path | None,
    expected_asr_inventory_fingerprint: str,
) -> _BuildResult:
    source = _load_sources(
        audio_run_root=audio_run_root,
        expected_asr_inventory_fingerprint=expected_asr_inventory_fingerprint,
    )
    assets = _AssetStore(temporary_root)
    visual_cache: dict[str, _VisualTarget] = {}
    samples: list[FinalH3Sample] = []
    failures: list[FinalH3Failure] = []
    pairs: list[tuple[FinalPair, FinalPairKind]] = [
        *((item, "in_pair") for item in source.in_pairs),
        *((item, "cross_pair") for item in source.cross_pairs),
    ]
    for pair, pair_kind in pairs:
        try:
            samples.append(
                _build_sample(
                    pair=pair,
                    pair_kind=pair_kind,
                    source=source,
                    assets=assets,
                    visual_cache=visual_cache,
                )
            )
        except _SampleFailure as exc:
            failures.append(
                FinalH3Failure(
                    pair_id=pair.pair_id,
                    pair_kind=pair_kind,
                    target_clip_uid=pair.target_clip_uid,
                    reason_code=exc.reason_code,
                    message=str(exc),
                )
            )
    visual_hashes: dict[Path, str] = {}
    for item in visual_cache.values():
        visual_hashes[item.clip_path] = item.clip_sha256
        visual_hashes[item.sidecar_path] = item.sidecar_sha256
    return _BuildResult(
        source=source,
        samples=tuple(samples),
        failures=tuple(failures),
        asset_source_hashes=assets.source_hashes,
        visual_source_hashes=visual_hashes,
    )


def _asset_paths(samples: Sequence[FinalH3Sample]) -> dict[str, set[str]]:
    output = {
        "target_video": set(),
        "target_full_audio": set(),
        "subject_picture": set(),
        "voice_reference": set(),
    }
    for sample in samples:
        output["target_video"].add(sample.target_video.path)
        output["target_full_audio"].add(sample.target_full_audio.path)
        output["subject_picture"].update(
            item.picture.path for item in sample.subjects
        )
        output["voice_reference"].update(
            item.voice_reference.path for item in sample.subjects
        )
    return output


def _build_summary(
    *, inventory_fingerprint: str, samples: Sequence[FinalH3Sample], failures: Sequence[FinalH3Failure]
) -> FinalH3Summary:
    speech = [segment for sample in samples for segment in sample.speech_segments]
    identity = Counter(item.identity_status for item in speech)
    text = Counter(item.text_status for item in speech)
    dialogue = Counter(item.dialogue_status for item in speech)
    languages = Counter(
        item.detected_language
        for item in speech
        if item.detected_language is not None
    )
    failure_reasons = Counter(item.reason_code for item in failures)
    asset_paths = _asset_paths(samples)
    return FinalH3Summary(
        inventory_fingerprint=inventory_fingerprint,
        in_pair_sample_count=sum(item.pair_kind == "in_pair" for item in samples),
        cross_pair_sample_count=sum(
            item.pair_kind == "cross_pair" for item in samples
        ),
        total_sample_count=len(samples),
        speech_segment_count=len(speech),
        subject_bound_segment_count=identity["subject_bound"],
        background_identified_segment_count=identity["background_identified"],
        unresolved_segment_count=identity["unresolved"],
        trusted_text_segment_count=text["trusted"],
        hidden_text_segment_count=text["hidden"],
        rendered_dialogue_segment_count=dialogue["rendered_dialogue"],
        trusted_text_unrendered_no_subject_count=dialogue[
            "trusted_text_unrendered_no_subject"
        ],
        detected_language_counts=dict(sorted(languages.items())),
        multi_language_sample_count=sum(
            len(item.detected_languages) > 1 for item in samples
        ),
        target_video_asset_count=len(asset_paths["target_video"]),
        full_audio_asset_count=len(asset_paths["target_full_audio"]),
        picture_asset_count=len(asset_paths["subject_picture"]),
        voice_asset_count=len(asset_paths["voice_reference"]),
        unique_asset_count=sum(len(paths) for paths in asset_paths.values()),
        failed_sample_count=len(failures),
        failure_reason_counts=dict(sorted(failure_reasons.items())),
    )


def _source_file_records(source: _SourceBundle) -> list[FinalSourceFile]:
    role_by_path: dict[Path, str] = {}
    for path in source.source_hashes:
        relative = path.relative_to(source.production_root).as_posix()
        role_by_path[path] = relative.replace("/", "_").replace(".", "_")
    return [
        _source_file(role_by_path[path], path, source.source_hashes)
        for path in sorted(role_by_path, key=lambda item: role_by_path[item])
    ]


def _inventory(
    *, output_root: Path, build: _BuildResult
) -> FinalH3Inventory:
    values: dict[str, object] = {
        "schema_version": FINAL_INVENTORY_SCHEMA_VERSION,
        "output_root": str(output_root),
        "source_files": [
            item.model_dump(mode="json")
            for item in _source_file_records(build.source)
        ],
        "source_fingerprints": build.source.fingerprints.model_dump(mode="json"),
        "renderer_version": FINAL_RENDERER_VERSION,
        "pair_row_count": len(build.source.in_pairs) + len(build.source.cross_pairs),
        "sample_count": len(build.samples),
        "failed_sample_count": len(build.failures),
        "text_usability_policy_version": TEXT_USABILITY_POLICY_VERSION,
        "language_conditioning_applied": False,
        "parent_quota_applied": False,
        "model_calls": 0,
        "mllm_calls": 0,
    }
    return FinalH3Inventory(
        **values,
        inventory_fingerprint=_final_inventory_fingerprint(values),
    )


def _verify_sources_unchanged(build: _BuildResult) -> None:
    expected = {
        **build.source.source_hashes,
        **build.asset_source_hashes,
        **build.visual_source_hashes,
    }
    current = {path: _sha256_file(path) for path in expected}
    if current != expected:
        raise ValueError("frozen H3 source changed during final rendering")


def _remove_unused_assets(temporary: Path, samples: Sequence[FinalH3Sample]) -> None:
    used = {path for paths in _asset_paths(samples).values() for path in paths}
    assets_root = temporary / "assets"
    if not assets_root.exists():
        return
    for path in sorted(assets_root.rglob("*"), reverse=True):
        if path.is_file() and path.relative_to(temporary).as_posix() not in used:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"final H3 output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def final_h3_output_root(audio_run_root: Path) -> Path:
    return audio_run_root / "production" / "h3"


def plan_final_h3_renderer(
    *,
    audio_run_root: Path,
    expected_asr_inventory_fingerprint: str = (
        TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT
    ),
) -> dict[str, object]:
    root = audio_run_root.expanduser().resolve(strict=True)
    build = _build(
        audio_run_root=root,
        temporary_root=None,
        expected_asr_inventory_fingerprint=expected_asr_inventory_fingerprint,
    )
    return {
        "output_root": str(final_h3_output_root(root)),
        "pair_row_count": len(build.source.in_pairs) + len(build.source.cross_pairs),
        "in_pair_sample_count": sum(
            item.pair_kind == "in_pair" for item in build.samples
        ),
        "cross_pair_sample_count": sum(
            item.pair_kind == "cross_pair" for item in build.samples
        ),
        "renderable_sample_count": len(build.samples),
        "failed_sample_count": len(build.failures),
        "source_fingerprints": build.source.fingerprints.model_dump(mode="json"),
        "renderer_version": FINAL_RENDERER_VERSION,
        "dry_run": True,
        "model_calls": 0,
        "mllm_calls": 0,
    }


def publish_final_h3_renderer(
    *,
    audio_run_root: Path,
    overwrite: bool = False,
    expected_asr_inventory_fingerprint: str = (
        TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT
    ),
    before_publish: Callable[[], None] | None = None,
) -> FinalH3Summary:
    root = audio_run_root.expanduser().resolve(strict=True)
    production = (root / "production").resolve(strict=True)
    destination = final_h3_output_root(root).resolve(strict=False)
    if destination.parent != production or destination.name != "h3":
        raise ValueError("final H3 output must use fixed production/h3 root")
    if destination.is_symlink():
        raise ValueError("final H3 output cannot be a symlink")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir()
        build = _build(
            audio_run_root=root,
            temporary_root=temporary,
            expected_asr_inventory_fingerprint=expected_asr_inventory_fingerprint,
        )
        _remove_unused_assets(temporary, build.samples)
        inventory = _inventory(output_root=destination, build=build)
        summary = _build_summary(
            inventory_fingerprint=inventory.inventory_fingerprint,
            samples=build.samples,
            failures=build.failures,
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "samples.jsonl", build.samples)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        _write_jsonl(temporary / "failures.jsonl", build.failures)
        if before_publish is not None:
            before_publish()
        _verify_sources_unchanged(build)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
