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

from r2v_data_v2.h3.audio_backends import AudioMediaBackend
from r2v_data_v2.h3.audio_schemas import FileAsset, VoiceReferenceArtifact
from r2v_data_v2.h3.pilot_schemas import (
    VoiceReferenceClipDiagnostics,
    VoiceReferenceTurnDiagnostics,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel

VOICE_REFERENCE_QUALITY_POLICY_VERSION = "voice_reference_quality_v1"

VoiceQualityReasonCode = Literal[
    "voice_duration_too_short",
    "voice_association_confidence_low",
    "voice_lr_asd_mean_low",
    "voice_lr_asd_p10_low",
    "voice_rms_too_low",
    "voice_clipping_excessive",
    "voice_noise_context_unavailable",
    "voice_snr_too_low",
]

_REASON_CODE_ORDER: tuple[VoiceQualityReasonCode, ...] = (
    "voice_duration_too_short",
    "voice_association_confidence_low",
    "voice_lr_asd_mean_low",
    "voice_lr_asd_p10_low",
    "voice_rms_too_low",
    "voice_clipping_excessive",
    "voice_noise_context_unavailable",
    "voice_snr_too_low",
)


@dataclass(frozen=True)
class VoiceReferenceQualityPolicy:
    version: str = VOICE_REFERENCE_QUALITY_POLICY_VERSION
    minimum_duration_seconds: float = 1.0
    minimum_association_confidence: float = 0.85
    minimum_lr_asd_mean: float = 0.50
    minimum_lr_asd_p10: float = 0.20
    minimum_rms_dbfs: float = -40.0
    maximum_clipping_ratio: float = 0.0001
    require_local_noise_context: bool = True
    minimum_estimated_snr_db: float = 10.0
    thresholds_calibrated: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.minimum_duration_seconds,
            self.minimum_association_confidence,
            self.minimum_lr_asd_mean,
            self.minimum_lr_asd_p10,
            self.minimum_rms_dbfs,
            self.maximum_clipping_ratio,
            self.minimum_estimated_snr_db,
        )
        if not self.version.strip() or not all(math.isfinite(value) for value in numeric):
            raise ValueError("voice-reference quality policy must be finite and versioned")
        if self.minimum_duration_seconds <= 0:
            raise ValueError("voice-reference minimum duration must be positive")
        if not 0 <= self.minimum_association_confidence <= 1:
            raise ValueError("voice-reference association threshold must be in [0, 1]")
        if not 0 <= self.maximum_clipping_ratio <= 1:
            raise ValueError("voice-reference clipping threshold must be in [0, 1]")
        if not self.thresholds_calibrated:
            raise ValueError("formal V1 voice-reference thresholds must be calibrated")

    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class VoiceReferenceQualityAssessment(SchemaModel):
    schema_version: Literal["r2v.h3.voice_reference_assessment.1"] = (
        "r2v.h3.voice_reference_assessment.1"
    )
    clip_uid: str
    entity_occurrence_id: str
    turn_id: str
    status: Literal["accepted", "rejected"]
    reason_codes: list[VoiceQualityReasonCode] = Field(default_factory=list)
    metrics: VoiceReferenceTurnDiagnostics
    policy_version: str
    policy_fingerprint: str
    thresholds_calibrated: Literal[True] = True

    @model_validator(mode="after")
    def validate_assessment(self) -> VoiceReferenceQualityAssessment:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.metrics.entity_id}":
            raise ValueError("voice assessment occurrence identity is inconsistent")
        if self.turn_id != self.metrics.turn_id or self.clip_uid != self.metrics.clip_uid:
            raise ValueError("voice assessment turn identity is inconsistent")
        if not self.policy_version.strip() or len(self.policy_fingerprint) != 64:
            raise ValueError("voice assessment policy provenance is invalid")
        expected = [code for code in _REASON_CODE_ORDER if code in self.reason_codes]
        if self.reason_codes != expected or len(expected) != len(set(expected)):
            raise ValueError("voice assessment reason codes must be unique and ordered")
        if (self.status == "accepted") != (not self.reason_codes):
            raise ValueError("voice assessment status must match its hard-gate reasons")
        return self


class PrimaryVoiceReferenceSelection(SchemaModel):
    schema_version: Literal["r2v.h3.primary_voice_reference_selection.1"] = (
        "r2v.h3.primary_voice_reference_selection.1"
    )
    clip_uid: str
    entity_id: str
    entity_occurrence_id: str
    primary_voice_reference: VoiceReferenceArtifact | None = None
    reason_codes: list[str] = Field(default_factory=list)
    candidate_turn_ids: list[str] = Field(default_factory=list)
    accepted_turn_ids: list[str] = Field(default_factory=list)
    policy_version: str
    policy_fingerprint: str

    @model_validator(mode="after")
    def validate_selection(self) -> PrimaryVoiceReferenceSelection:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("primary voice occurrence identity is inconsistent")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("primary voice selection reason codes must be unique")
        if self.primary_voice_reference is None:
            if not self.reason_codes:
                raise ValueError("missing primary voice reference requires a reason")
        elif (
            self.reason_codes
            or self.primary_voice_reference.entity_occurrence_id
            != self.entity_occurrence_id
            or self.primary_voice_reference.source_turn_id not in self.accepted_turn_ids
        ):
            raise ValueError("published primary voice reference is inconsistent")
        return self


class PrimaryVoiceReferenceExportSummary(SchemaModel):
    schema_version: Literal["r2v.h3.primary_voice_reference_export.1"] = (
        "r2v.h3.primary_voice_reference_export.1"
    )
    policy_version: str
    policy_fingerprint: str
    thresholds_calibrated: Literal[True] = True
    candidate_turn_count: int = Field(ge=0)
    accepted_turn_count: int = Field(ge=0)
    rejected_turn_count: int = Field(ge=0)
    entity_occurrence_count: int = Field(ge=0)
    occurrences_with_primary_voice_reference: int = Field(ge=0)
    occurrences_without_primary_voice_reference: int = Field(ge=0)
    rejection_reason_counts: dict[str, int]
    selected_reference_rows: list[dict[str, object]]


def assess_voice_reference_turn(
    turn: VoiceReferenceTurnDiagnostics,
    *,
    policy: VoiceReferenceQualityPolicy | None = None,
) -> VoiceReferenceQualityAssessment:
    active = policy or VoiceReferenceQualityPolicy()
    failed: set[VoiceQualityReasonCode] = set()
    if turn.duration_seconds < active.minimum_duration_seconds:
        failed.add("voice_duration_too_short")
    if turn.association_confidence.min < active.minimum_association_confidence:
        failed.add("voice_association_confidence_low")
    if turn.lr_asd_raw_native_score.mean < active.minimum_lr_asd_mean:
        failed.add("voice_lr_asd_mean_low")
    if turn.lr_asd_raw_native_score.p10 < active.minimum_lr_asd_p10:
        failed.add("voice_lr_asd_p10_low")
    if turn.rms_dbfs is None or turn.rms_dbfs < active.minimum_rms_dbfs:
        failed.add("voice_rms_too_low")
    if turn.clipping_ratio > active.maximum_clipping_ratio:
        failed.add("voice_clipping_excessive")
    if active.require_local_noise_context and (
        not turn.local_noise_context_available or turn.estimated_snr_db is None
    ):
        failed.add("voice_noise_context_unavailable")
    elif (
        turn.estimated_snr_db is not None
        and turn.estimated_snr_db < active.minimum_estimated_snr_db
    ):
        failed.add("voice_snr_too_low")
    reasons = [code for code in _REASON_CODE_ORDER if code in failed]
    return VoiceReferenceQualityAssessment(
        clip_uid=turn.clip_uid,
        entity_occurrence_id=f"{turn.clip_uid}/{turn.entity_id}",
        turn_id=turn.turn_id,
        status="accepted" if not reasons else "rejected",
        reason_codes=reasons,
        metrics=turn,
        policy_version=active.version,
        policy_fingerprint=active.fingerprint(),
    )


def assess_voice_reference_clip(
    report: VoiceReferenceClipDiagnostics,
    *,
    policy: VoiceReferenceQualityPolicy | None = None,
) -> list[VoiceReferenceQualityAssessment]:
    if report.status != "ready":
        raise ValueError("formal voice-reference gate requires ready diagnostics")
    return [
        assess_voice_reference_turn(turn, policy=policy)
        for turn in report.candidate_turns
    ]


def select_primary_voice_assessments(
    assessments: Sequence[VoiceReferenceQualityAssessment],
    *,
    entity_order: Sequence[str],
) -> dict[str, VoiceReferenceQualityAssessment]:
    selected: dict[str, VoiceReferenceQualityAssessment] = {}
    for entity_id in entity_order:
        accepted = [
            item
            for item in assessments
            if item.metrics.entity_id == entity_id and item.status == "accepted"
        ]
        if accepted:
            selected[entity_id] = min(
                accepted,
                key=lambda item: (
                    -float(item.metrics.estimated_snr_db),  # accepted requires SNR
                    -item.metrics.lr_asd_raw_native_score.p10,
                    -item.metrics.lr_asd_raw_native_score.mean,
                    -item.metrics.duration_seconds,
                    -item.metrics.association_confidence.min,
                    item.metrics.start_time,
                    item.turn_id,
                ),
            )
    return selected


def voice_turn_sample_range(
    turn: VoiceReferenceTurnDiagnostics,
    *,
    sample_rate_hz: int = 16000,
) -> tuple[int, int]:
    start = round(turn.start_time * sample_rate_hz)
    end = start + turn.sample_count
    if end != round(turn.end_time * sample_rate_hz):
        raise ValueError("voice diagnostics sample count does not match its time extent")
    return start, end


def load_voice_reference_quality_diagnostics(
    *,
    sidecar_root: Path,
    clip_uid: str,
) -> VoiceReferenceClipDiagnostics:
    path = sidecar_root / "clips" / clip_uid / "voice_reference_quality.json"
    report = VoiceReferenceClipDiagnostics.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if report.clip_uid != clip_uid:
        raise ValueError("voice-quality diagnostics clip identity does not match")
    return report


def resolve_voice_quality_audio_path(
    *,
    pilot_root: Path,
    source_audio_path: str,
) -> Path:
    root = pilot_root.resolve(strict=True)
    source = Path(source_audio_path).expanduser()
    candidate = source if source.is_absolute() else root / source
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError("voice-quality source audio is not a file")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_asset(path: Path, root: Path) -> FileAsset:
    return FileAsset(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        media_type="audio/flac",
    )


def build_voice_reference_artifact(
    *,
    assessment: VoiceReferenceQualityAssessment,
    asset_path: Path,
    output_root: Path,
    source_sample_rate_hz: int = 16000,
    source_channels: int = 1,
    sample_mapping_policy: str = "native_16k_turn_sample_extent_v1",
    source_start_sample: int | None = None,
    source_end_sample: int | None = None,
) -> VoiceReferenceArtifact:
    turn = assessment.metrics
    if source_start_sample is None and source_end_sample is None:
        start_sample, end_sample = voice_turn_sample_range(
            turn,
            sample_rate_hz=source_sample_rate_hz,
        )
    elif source_start_sample is not None and source_end_sample is not None:
        start_sample, end_sample = source_start_sample, source_end_sample
    else:
        raise ValueError("voice reference source sample extent must be complete")
    return VoiceReferenceArtifact(
        voice_reference_id="voice_ref_1",
        entity_occurrence_id=assessment.entity_occurrence_id,
        source_turn_id=turn.turn_id,
        source_start=turn.start_time,
        source_end=turn.end_time,
        source_start_sample=start_sample,
        source_end_sample=end_sample,
        asset=_file_asset(asset_path, output_root),
        quality_score=1.0,
        quality_metadata={
            "assessment": assessment.model_dump(mode="json"),
            "policy_version": assessment.policy_version,
            "policy_fingerprint": assessment.policy_fingerprint,
            "thresholds_calibrated": True,
            "quality_score_definition": "binary_all_hard_gates_pass_v1",
            "selection_policy": (
                "snr_lr_asd_p10_lr_asd_mean_duration_association_time_v1"
            ),
            "source_sample_rate_hz": source_sample_rate_hz,
            "source_channels": source_channels,
            "sample_mapping_policy": sample_mapping_policy,
        },
    )


def _jsonl(values: Sequence[SchemaModel]) -> str:
    return "".join(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for value in values
    )


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"primary voice export already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    published = False
    try:
        temporary.replace(destination)
        published = True
    finally:
        if published:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)


def export_primary_voice_references(
    *,
    pilot_root: Path,
    output_root: Path,
    audio_backend: AudioMediaBackend,
    policy: VoiceReferenceQualityPolicy | None = None,
    overwrite: bool = False,
    output_path_for_entity: Callable[[str, str], Path] | None = None,
    source_audio_for_clip: Callable[[str], Path] | None = None,
    output_sample_rate_hz: int = 16000,
    output_channels: int = 1,
    sample_mapping_policy: str = "native_16k_turn_sample_extent_v1",
) -> PrimaryVoiceReferenceExportSummary:
    source = pilot_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("primary voice export must be outside the source pilot root")
    active = policy or VoiceReferenceQualityPolicy()
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    assessments: list[VoiceReferenceQualityAssessment] = []
    selections: list[PrimaryVoiceReferenceSelection] = []
    selected_rows: list[dict[str, object]] = []
    try:
        temporary.mkdir()
        sidecar_paths = sorted((source / "clips").glob("**/audio_binding.json"))
        if not sidecar_paths:
            raise ValueError("primary voice export found no audio binding artifacts")
        for sidecar_path in sidecar_paths:
            sidecar = AudioBindingSidecar.model_validate_json(
                sidecar_path.read_text(encoding="utf-8")
            )
            clip_uid = sidecar.clip_uid
            if sidecar.status != "ready":
                raise ValueError("primary voice export requires identity-matched ready clips")
            if sidecar.h3_ir is None:
                raise ValueError("primary voice export requires H3 subject metadata")
            report = VoiceReferenceClipDiagnostics.model_validate_json(
                sidecar_path.with_name("voice_reference_quality.json").read_text(
                    encoding="utf-8"
                )
            )
            if report.clip_uid != clip_uid:
                raise ValueError("primary voice diagnostics differ from Audio sidecar")
            clip_assessments = assess_voice_reference_clip(report, policy=active)
            assessments.extend(clip_assessments)
            subject_ids = [
                item.entity_id
                for item in sidecar.h3_ir.subjects
                if item.reference_type == "subject"
            ]
            selected = select_primary_voice_assessments(
                clip_assessments,
                entity_order=subject_ids,
            )
            source_audio = (
                source_audio_for_clip(clip_uid).expanduser().resolve(strict=True)
                if source_audio_for_clip is not None
                else resolve_voice_quality_audio_path(
                    pilot_root=source,
                    source_audio_path=report.source_audio_path,
                )
            )
            for entity_id in subject_ids:
                occurrence_id = f"{clip_uid}/{entity_id}"
                entity_assessments = [
                    item
                    for item in clip_assessments
                    if item.metrics.entity_id == entity_id
                ]
                accepted = [
                    item for item in entity_assessments if item.status == "accepted"
                ]
                chosen = selected.get(entity_id)
                artifact = None
                reasons: list[str] = []
                if chosen is None:
                    reasons.append(
                        "no_voice_reference_candidate_turn"
                        if not entity_assessments
                        else "no_voice_reference_passed_quality_gate"
                    )
                    reasons.extend(
                        code
                        for code in _REASON_CODE_ORDER
                        if any(code in item.reason_codes for item in entity_assessments)
                    )
                else:
                    if source_audio_for_clip is None:
                        start_sample, end_sample = voice_turn_sample_range(
                            chosen.metrics,
                            sample_rate_hz=output_sample_rate_hz,
                        )
                    else:
                        start_sample = round(
                            chosen.metrics.start_time * output_sample_rate_hz
                        )
                        end_sample = round(
                            chosen.metrics.end_time * output_sample_rate_hz
                        )
                    relative_voice_path = (
                        output_path_for_entity(clip_uid, entity_id)
                        if output_path_for_entity is not None
                        else Path("voice_refs")
                        / clip_uid
                        / entity_id
                        / "voice_ref_1.flac"
                    )
                    if (
                        relative_voice_path.is_absolute()
                        or ".." in relative_voice_path.parts
                    ):
                        raise ValueError("primary voice output path must be safe and relative")
                    voice_path = temporary / relative_voice_path
                    exact_source = (
                        {
                            "source_audio_path": source_audio,
                            "source_start_sample": start_sample,
                            "source_end_sample": end_sample,
                        }
                        if source_audio_for_clip is None
                        else {}
                    )
                    audio_backend.extract_voice_reference(
                        clip_uid=clip_uid,
                        entity_id=entity_id,
                        full_audio_path=source_audio,
                        start_time=chosen.metrics.start_time,
                        end_time=chosen.metrics.end_time,
                        destination=voice_path,
                        sample_rate_hz=output_sample_rate_hz,
                        channels=output_channels,
                        output_format="flac",
                        **exact_source,
                    )
                    artifact = build_voice_reference_artifact(
                        assessment=chosen,
                        asset_path=voice_path,
                        output_root=temporary,
                        source_sample_rate_hz=output_sample_rate_hz,
                        source_channels=output_channels,
                        sample_mapping_policy=sample_mapping_policy,
                        source_start_sample=start_sample,
                        source_end_sample=end_sample,
                    )
                    selected_rows.append(
                        {
                            "clip_uid": clip_uid,
                            "entity_id": entity_id,
                            "entity_occurrence_id": occurrence_id,
                            "source_turn_id": chosen.turn_id,
                            "source_start": chosen.metrics.start_time,
                            "source_end": chosen.metrics.end_time,
                            "source_start_sample": start_sample,
                            "source_end_sample": end_sample,
                            "estimated_snr_db": chosen.metrics.estimated_snr_db,
                            "lr_asd_p10": chosen.metrics.lr_asd_raw_native_score.p10,
                            "lr_asd_mean": chosen.metrics.lr_asd_raw_native_score.mean,
                            "association_confidence_min": (
                                chosen.metrics.association_confidence.min
                            ),
                            "asset_path": artifact.asset.path,
                        }
                    )
                selections.append(
                    PrimaryVoiceReferenceSelection(
                        clip_uid=clip_uid,
                        entity_id=entity_id,
                        entity_occurrence_id=occurrence_id,
                        primary_voice_reference=artifact,
                        reason_codes=reasons,
                        candidate_turn_ids=[item.turn_id for item in entity_assessments],
                        accepted_turn_ids=[item.turn_id for item in accepted],
                        policy_version=active.version,
                        policy_fingerprint=active.fingerprint(),
                    )
                )

        assessments.sort(key=lambda item: (item.clip_uid, item.metrics.start_time, item.turn_id))
        selections.sort(key=lambda item: (item.clip_uid, item.entity_id))
        selected_rows.sort(key=lambda item: (str(item["clip_uid"]), str(item["entity_id"])))
        reason_counts = Counter(
            code
            for item in assessments
            if item.status == "rejected"
            for code in item.reason_codes
        )
        summary = PrimaryVoiceReferenceExportSummary(
            policy_version=active.version,
            policy_fingerprint=active.fingerprint(),
            candidate_turn_count=len(assessments),
            accepted_turn_count=sum(item.status == "accepted" for item in assessments),
            rejected_turn_count=sum(item.status == "rejected" for item in assessments),
            entity_occurrence_count=len(selections),
            occurrences_with_primary_voice_reference=sum(
                item.primary_voice_reference is not None for item in selections
            ),
            occurrences_without_primary_voice_reference=sum(
                item.primary_voice_reference is None for item in selections
            ),
            rejection_reason_counts=dict(sorted(reason_counts.items())),
            selected_reference_rows=selected_rows,
        )
        (temporary / "voice_quality_assessments.jsonl").write_text(
            _jsonl(assessments),
            encoding="utf-8",
        )
        (temporary / "primary_voice_references.jsonl").write_text(
            _jsonl(selections),
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
