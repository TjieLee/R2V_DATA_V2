from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
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
from r2v_data_v2.h3.schemas import SchemaModel

TEXT_USABILITY_POLICY_VERSION = "h3_asr_v2_text_usability_policy_v1"
TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD = 0.65
TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT = (
    "53550fd6206f90368023caaf5629eab3b9cc6e7a5ca9f9b8081d6e5d49c173de"
)
TEXT_USABILITY_OUTPUT_DIRECTORY = "production/text_usability"
TEXT_USABILITY_SOURCE_DIRECTORY = "production/asr_v2"
TEXT_USABILITY_SEGMENT_SCHEMA_VERSION = "r2v.h3.text_usability_segment.1"
TEXT_USABILITY_INVENTORY_SCHEMA_VERSION = "r2v.h3.text_usability_inventory.1"
TEXT_USABILITY_SUMMARY_SCHEMA_VERSION = "r2v.h3.text_usability_summary.1"

TextUsabilityReasonCode = Literal[
    "raw_text_unavailable",
    "language_probability_unavailable",
    "language_probability_below_threshold",
]
TextUsabilityStatus = Literal["trusted", "hidden"]

_REASON_CODE_ORDER: tuple[TextUsabilityReasonCode, ...] = (
    "raw_text_unavailable",
    "language_probability_unavailable",
    "language_probability_below_threshold",
)
_BINDING_STATUS_ORDER = ("candidate_mapped", "ambiguous", "unbound", "conflict")
_IDENTITY_SCOPE_ORDER = (
    "direct_anchor_present",
    "cluster_propagated_only",
    "unresolved",
)


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


@dataclass(frozen=True)
class TextUsabilityPolicy:
    version: str = TEXT_USABILITY_POLICY_VERSION
    language_probability_threshold: float = (
        TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
    )
    policy_validated: bool = True

    def __post_init__(self) -> None:
        if self.version != TEXT_USABILITY_POLICY_VERSION:
            raise ValueError("text-usability policy version is frozen")
        if (
            not math.isfinite(self.language_probability_threshold)
            or self.language_probability_threshold
            != TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
        ):
            raise ValueError("text-usability language threshold is frozen at 0.65")
        if not self.policy_validated:
            raise ValueError("formal text-usability policy must be validated")

    def fingerprint(self) -> str:
        return _sha256_bytes(_compact_json(asdict(self)).encode("utf-8"))


class TextUsabilitySegment(SchemaModel):
    schema_version: Literal["r2v.h3.text_usability_segment.1"] = (
        TEXT_USABILITY_SEGMENT_SCHEMA_VERSION
    )
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
    source_asr_v2_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_segment_schema_version: Literal["r2v.h3.asr_v2_segment.1"] = (
        ASR_V2_SEGMENT_SCHEMA_VERSION
    )
    source_asr_v2_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_segment_status: Literal["transcribed", "uncertain", "failed"]
    source_raw_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_text_present: bool
    language_probability: float | None = Field(default=None, ge=0, le=1)
    policy_version: Literal["h3_asr_v2_text_usability_policy_v1"] = (
        TEXT_USABILITY_POLICY_VERSION
    )
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_probability_threshold: float = (
        TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
    )
    text_status: TextUsabilityStatus
    trusted_text: str | None = None
    reason_codes: list[TextUsabilityReasonCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record(self) -> TextUsabilitySegment:
        if any(
            not value.strip()
            for value in (
                self.target_clip_uid,
                self.segment_id,
                self.speaker_cluster_id,
            )
        ):
            raise ValueError("text-usability segment identity must not be empty")
        if (
            self.language_probability_threshold
            != TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
        ):
            raise ValueError("text-usability threshold must be exactly 0.65")
        if self.policy_fingerprint != TextUsabilityPolicy().fingerprint():
            raise ValueError("text-usability policy fingerprint is inconsistent")
        expected_reasons = _expected_reason_codes(
            status=self.source_asr_v2_segment_status,
            raw_text_present=self.raw_text_present,
            language_probability=self.language_probability,
            threshold=self.language_probability_threshold,
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("text-usability reason codes are inconsistent")
        expected_trusted = not expected_reasons
        if (self.text_status == "trusted") != expected_trusted:
            raise ValueError("text-usability status is inconsistent")
        if expected_trusted:
            if self.trusted_text is None or not self.trusted_text.strip():
                raise ValueError("trusted text must preserve the source transcript")
            if self.source_raw_text_sha256 != _sha256_bytes(
                self.trusted_text.encode("utf-8")
            ):
                raise ValueError("trusted text differs from raw ASR evidence")
        elif self.trusted_text is not None:
            raise ValueError("hidden text must not publish transcript content")
        if self.raw_text_present != (self.source_raw_text_sha256 is not None):
            raise ValueError("raw text presence and fingerprint are inconsistent")
        return self


class TextUsabilityInventory(SchemaModel):
    schema_version: Literal["r2v.h3.text_usability_inventory.1"] = (
        TEXT_USABILITY_INVENTORY_SCHEMA_VERSION
    )
    output_root: str
    source_asr_v2_root: str
    source_asr_v2_inventory_path: str
    source_asr_v2_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_segments_path: str
    source_asr_v2_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_summary_path: str
    source_asr_v2_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_segment_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    policy_version: Literal["h3_asr_v2_text_usability_policy_v1"] = (
        TEXT_USABILITY_POLICY_VERSION
    )
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_probability_threshold: float = (
        TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
    )
    policy_validated: Literal[True] = True
    model_calls: Literal[0] = 0
    whisper_calls: Literal[0] = 0
    diarizen_calls: Literal[0] = 0
    gpu_calls: Literal[0] = 0
    source_asr_v2_modified: Literal[False] = False
    voice_pair_embedding_modified: Literal[False] = False
    final_renderer_applied: Literal[False] = False
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> TextUsabilityInventory:
        if self.segment_count != self.source_segment_count:
            raise ValueError("text-usability inventory must cover every source segment")
        if (
            self.language_probability_threshold
            != TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
        ):
            raise ValueError("text-usability inventory threshold must be exactly 0.65")
        if self.policy_fingerprint != TextUsabilityPolicy().fingerprint():
            raise ValueError("text-usability inventory policy is inconsistent")
        expected = _inventory_fingerprint(self)
        if self.inventory_fingerprint != expected:
            raise ValueError("text-usability inventory fingerprint is inconsistent")
        return self


class TextUsabilitySummary(SchemaModel):
    schema_version: Literal["r2v.h3.text_usability_summary.1"] = (
        TEXT_USABILITY_SUMMARY_SCHEMA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["h3_asr_v2_text_usability_policy_v1"] = (
        TEXT_USABILITY_POLICY_VERSION
    )
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_probability_threshold: float = (
        TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
    )
    policy_validated: Literal[True] = True
    segment_count: int = Field(ge=0)
    raw_text_available_count: int = Field(ge=0)
    trusted_text_count: int = Field(ge=0)
    hidden_text_count: int = Field(ge=0)
    hidden_reason_counts: dict[str, int]
    trusted_by_cluster_binding_status: dict[str, int]
    hidden_by_cluster_binding_status: dict[str, int]
    trusted_by_identity_scope: dict[str, int]
    hidden_by_identity_scope: dict[str, int]
    identity_used_as_text_gate: Literal[False] = False
    raw_asr_preserved: Literal[True] = True
    voice_reference_quality_independent: Literal[True] = True
    speaker_entity_identity_independent: Literal[True] = True
    final_renderer_applied: Literal[False] = False
    model_calls: Literal[0] = 0
    gpu_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_summary(self) -> TextUsabilitySummary:
        if self.segment_count != self.trusted_text_count + self.hidden_text_count:
            raise ValueError("trusted and hidden text counts must reconcile")
        if self.raw_text_available_count > self.segment_count:
            raise ValueError("raw text availability cannot exceed segment count")
        if sum(self.hidden_reason_counts.values()) != self.hidden_text_count:
            raise ValueError("hidden text reason counts must reconcile")
        expected_reasons = set(_REASON_CODE_ORDER)
        if set(self.hidden_reason_counts) != expected_reasons:
            raise ValueError("hidden text reason histogram is incomplete")
        for trusted, hidden, expected_keys in (
            (
                self.trusted_by_cluster_binding_status,
                self.hidden_by_cluster_binding_status,
                set(_BINDING_STATUS_ORDER),
            ),
            (
                self.trusted_by_identity_scope,
                self.hidden_by_identity_scope,
                set(_IDENTITY_SCOPE_ORDER),
            ),
        ):
            if set(trusted) != expected_keys or set(hidden) != expected_keys:
                raise ValueError("text-usability diagnostic breakdown is incomplete")
            if sum(trusted.values()) != self.trusted_text_count:
                raise ValueError("trusted diagnostic breakdown does not reconcile")
            if sum(hidden.values()) != self.hidden_text_count:
                raise ValueError("hidden diagnostic breakdown does not reconcile")
        if (
            self.language_probability_threshold
            != TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
        ):
            raise ValueError("text-usability summary threshold must be exactly 0.65")
        if self.policy_fingerprint != TextUsabilityPolicy().fingerprint():
            raise ValueError("text-usability summary policy is inconsistent")
        return self


@dataclass(frozen=True)
class _SourceASRV2:
    root: Path
    inventory_path: Path
    segments_path: Path
    summary_path: Path
    inventory: ASRV2InventoryRecord
    records: tuple[ASRV2SegmentRecord, ...]
    file_hashes: dict[Path, str]


def _inventory_fingerprint(
    inventory: TextUsabilityInventory | dict[str, object],
) -> str:
    values = (
        inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if isinstance(inventory, TextUsabilityInventory)
        else {
            key: value
            for key, value in inventory.items()
            if key != "inventory_fingerprint"
        }
    )
    return _sha256_bytes(_compact_json(values).encode("utf-8"))


def _expected_reason_codes(
    *,
    status: str,
    raw_text_present: bool,
    language_probability: float | None,
    threshold: float,
) -> list[TextUsabilityReasonCode]:
    if status != "transcribed" or not raw_text_present:
        return ["raw_text_unavailable"]
    if language_probability is None:
        return ["language_probability_unavailable"]
    if language_probability < threshold:
        return ["language_probability_below_threshold"]
    return []


def text_is_trusted(
    record: ASRV2SegmentRecord,
    *,
    policy: TextUsabilityPolicy | None = None,
) -> bool:
    active = policy or TextUsabilityPolicy()
    language_probability = (
        record.diagnostics.language_probability
        if record.diagnostics is not None
        else None
    )
    return (
        record.status == "transcribed"
        and record.text is not None
        and bool(record.text.strip())
        and language_probability is not None
        and language_probability >= active.language_probability_threshold
    )


def assess_text_usability(
    record: ASRV2SegmentRecord,
    *,
    source_inventory_fingerprint: str,
    policy: TextUsabilityPolicy | None = None,
) -> TextUsabilitySegment:
    active = policy or TextUsabilityPolicy()
    raw_text_present = record.text is not None and bool(record.text.strip())
    language_probability = (
        record.diagnostics.language_probability
        if record.diagnostics is not None
        else None
    )
    trusted = text_is_trusted(record, policy=active)
    reasons = _expected_reason_codes(
        status=record.status,
        raw_text_present=raw_text_present,
        language_probability=language_probability,
        threshold=active.language_probability_threshold,
    )
    if trusted != (not reasons):
        raise ValueError("text-usability predicate and reasons disagree")
    return TextUsabilitySegment(
        target_clip_uid=record.target_clip_uid,
        segment_id=record.segment_id,
        speaker_cluster_id=record.speaker_cluster_id,
        cluster_binding_status=record.cluster_binding_status,
        entity_id=record.entity_id,
        entity_occurrence_id=record.entity_occurrence_id,
        identity_scope=record.identity_scope,
        source_asr_v2_inventory_fingerprint=source_inventory_fingerprint,
        source_asr_v2_request_fingerprint=record.request_fingerprint,
        source_asr_v2_segment_status=record.status,
        source_raw_text_sha256=(
            _sha256_bytes(record.text.encode("utf-8"))
            if raw_text_present and record.text is not None
            else None
        ),
        raw_text_present=raw_text_present,
        language_probability=language_probability,
        policy_fingerprint=active.fingerprint(),
        text_status="trusted" if trusted else "hidden",
        trusted_text=record.text if trusted else None,
        reason_codes=reasons,
    )


def text_usability_output_root(audio_run_root: Path) -> Path:
    return audio_run_root / "production" / "text_usability"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"ASR V2 source JSONL line {line_number} must be an object")
        rows.append(payload)
    return rows


def _load_source_asr_v2(
    *,
    audio_run_root: Path,
    expected_inventory_fingerprint: str,
) -> _SourceASRV2:
    root = audio_run_root.expanduser().resolve(strict=True)
    source_root = (root / TEXT_USABILITY_SOURCE_DIRECTORY).resolve(strict=True)
    if root not in source_root.parents:
        raise ValueError("production ASR V2 source must remain under audio run root")
    inventory_path = source_root / "inventory.json"
    segments_path = source_root / "segments.jsonl"
    summary_path = source_root / "summary.json"
    for path in (inventory_path, segments_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing production ASR V2 source artifact: {path}"
            )
    inventory = load_asr_v2_inventory(inventory_path)
    if inventory.mode != "production":
        raise ValueError("text-usability policy requires production ASR V2 input")
    if inventory.inventory_fingerprint != _asr_v2_inventory_fingerprint(inventory):
        raise ValueError("production ASR V2 inventory fingerprint is inconsistent")
    if inventory.inventory_fingerprint != expected_inventory_fingerprint:
        raise ValueError("production ASR V2 inventory is not the frozen source")
    summary = load_asr_v2_summary(summary_path)
    if summary.mode != "production":
        raise ValueError("production ASR V2 summary mode is inconsistent")
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("production ASR V2 summary fingerprint is inconsistent")
    records = tuple(
        ASRV2SegmentRecord.model_validate(row) for row in _read_jsonl(segments_path)
    )
    if len(records) != len(inventory.jobs) or len(records) != summary.segment_count:
        raise ValueError("production ASR V2 segment count is inconsistent")
    status_counts = Counter(record.status for record in records)
    if (
        status_counts["transcribed"] != summary.transcribed_count
        or status_counts["uncertain"] != summary.uncertain_count
        or status_counts["failed"] != summary.failed_count
    ):
        raise ValueError("production ASR V2 status counts differ from source summary")
    job_fields = tuple(ASRV2SegmentJob.model_fields)
    for job, record in zip(inventory.jobs, records, strict=True):
        record_job = ASRV2SegmentJob.model_validate(
            {field: getattr(record, field) for field in job_fields}
        )
        if record_job != job:
            raise ValueError("production ASR V2 record order differs from inventory")
    identities = [
        (record.target_clip_uid, record.segment_id, record.speaker_cluster_id)
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("production ASR V2 segment identities must be unique")
    paths = (inventory_path, segments_path, summary_path)
    return _SourceASRV2(
        root=source_root,
        inventory_path=inventory_path,
        segments_path=segments_path,
        summary_path=summary_path,
        inventory=inventory,
        records=records,
        file_hashes={path: _sha256_file(path) for path in paths},
    )


def _verify_source_unchanged(source: _SourceASRV2) -> None:
    current = {path: _sha256_file(path) for path in source.file_hashes}
    if current != source.file_hashes:
        raise ValueError("production ASR V2 source changed during text-usability build")


def _counts(
    records: Sequence[TextUsabilitySegment],
    *,
    field: str,
    values: Sequence[str],
    status: TextUsabilityStatus,
) -> dict[str, int]:
    counter = Counter(
        str(getattr(record, field))
        for record in records
        if record.text_status == status
    )
    return {value: counter[value] for value in values}


def _build_summary(
    *,
    records: Sequence[TextUsabilitySegment],
    inventory: TextUsabilityInventory,
) -> TextUsabilitySummary:
    hidden_reasons = Counter(
        reason
        for record in records
        if record.text_status == "hidden"
        for reason in record.reason_codes
    )
    return TextUsabilitySummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        source_asr_v2_inventory_fingerprint=(
            inventory.source_asr_v2_inventory_fingerprint
        ),
        policy_fingerprint=inventory.policy_fingerprint,
        segment_count=len(records),
        raw_text_available_count=sum(record.raw_text_present for record in records),
        trusted_text_count=sum(record.text_status == "trusted" for record in records),
        hidden_text_count=sum(record.text_status == "hidden" for record in records),
        hidden_reason_counts={
            reason: hidden_reasons[reason] for reason in _REASON_CODE_ORDER
        },
        trusted_by_cluster_binding_status=_counts(
            records,
            field="cluster_binding_status",
            values=_BINDING_STATUS_ORDER,
            status="trusted",
        ),
        hidden_by_cluster_binding_status=_counts(
            records,
            field="cluster_binding_status",
            values=_BINDING_STATUS_ORDER,
            status="hidden",
        ),
        trusted_by_identity_scope=_counts(
            records,
            field="identity_scope",
            values=_IDENTITY_SCOPE_ORDER,
            status="trusted",
        ),
        hidden_by_identity_scope=_counts(
            records,
            field="identity_scope",
            values=_IDENTITY_SCOPE_ORDER,
            status="hidden",
        ),
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"text-usability output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def plan_text_usability(
    *,
    audio_run_root: Path,
    expected_source_inventory_fingerprint: str = (
        TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT
    ),
) -> dict[str, object]:
    root = audio_run_root.expanduser().resolve(strict=True)
    source = _load_source_asr_v2(
        audio_run_root=root,
        expected_inventory_fingerprint=expected_source_inventory_fingerprint,
    )
    return {
        "source_asr_v2_inventory_fingerprint": source.inventory.inventory_fingerprint,
        "source_segment_count": len(source.records),
        "policy_version": TEXT_USABILITY_POLICY_VERSION,
        "language_probability_threshold": (
            TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD
        ),
        "output_root": str(text_usability_output_root(root)),
        "dry_run": True,
        "model_calls": 0,
        "gpu_calls": 0,
    }


def publish_text_usability(
    *,
    audio_run_root: Path,
    overwrite: bool = False,
    expected_source_inventory_fingerprint: str = (
        TEXT_USABILITY_SOURCE_INVENTORY_FINGERPRINT
    ),
    before_publish: Callable[[], None] | None = None,
) -> TextUsabilitySummary:
    root = audio_run_root.expanduser().resolve(strict=True)
    source = _load_source_asr_v2(
        audio_run_root=root,
        expected_inventory_fingerprint=expected_source_inventory_fingerprint,
    )
    output_path = text_usability_output_root(root)
    if output_path.is_symlink():
        raise ValueError("text-usability output cannot be a symlink")
    production_root = (root / "production").resolve(strict=True)
    destination = output_path.resolve(strict=False)
    if destination.parent != production_root:
        raise ValueError("text-usability output must remain under production root")
    policy = TextUsabilityPolicy()
    records = [
        assess_text_usability(
            record,
            source_inventory_fingerprint=source.inventory.inventory_fingerprint,
            policy=policy,
        )
        for record in source.records
    ]
    inventory_values: dict[str, object] = {
        "schema_version": TEXT_USABILITY_INVENTORY_SCHEMA_VERSION,
        "output_root": str(destination),
        "source_asr_v2_root": str(source.root),
        "source_asr_v2_inventory_path": str(source.inventory_path),
        "source_asr_v2_inventory_sha256": source.file_hashes[source.inventory_path],
        "source_asr_v2_inventory_fingerprint": source.inventory.inventory_fingerprint,
        "source_asr_v2_segments_path": str(source.segments_path),
        "source_asr_v2_segments_sha256": source.file_hashes[source.segments_path],
        "source_asr_v2_summary_path": str(source.summary_path),
        "source_asr_v2_summary_sha256": source.file_hashes[source.summary_path],
        "source_segment_count": len(source.records),
        "segment_count": len(records),
        "policy_version": policy.version,
        "policy_fingerprint": policy.fingerprint(),
        "language_probability_threshold": policy.language_probability_threshold,
        "policy_validated": policy.policy_validated,
        "model_calls": 0,
        "whisper_calls": 0,
        "diarizen_calls": 0,
        "gpu_calls": 0,
        "source_asr_v2_modified": False,
        "voice_pair_embedding_modified": False,
        "final_renderer_applied": False,
    }
    inventory = TextUsabilityInventory(
        **inventory_values,
        inventory_fingerprint=_inventory_fingerprint(inventory_values),
    )
    summary = _build_summary(records=records, inventory=inventory)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "segments.jsonl", records)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        if before_publish is not None:
            before_publish()
        _verify_source_unchanged(source)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
