from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import uuid
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import SpeakerEmbeddingBackend
from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    DurationBucket,
    IdentityScope,
    VoiceConsistencyAuditRecord,
    cosine_similarity,
)
from r2v_data_v2.h3.diarization_voice_consistency_review import (
    VoiceConsistencyReviewAnnotation,
    load_current_review_annotations,
    load_review_context,
)
from r2v_data_v2.h3.schemas import SchemaModel

ERES2NETV2_SHADOW_RECORD_VERSION = (
    "r2v.h3.eres2netv2_voice_consistency_shadow.1"
)
ERES2NETV2_SHADOW_ERROR_VERSION = (
    "r2v.h3.eres2netv2_voice_consistency_shadow_error.1"
)
ERES2NETV2_SHADOW_SUMMARY_VERSION = (
    "r2v.h3.eres2netv2_voice_consistency_shadow_summary.1"
)
DEFAULT_OUTPUT_DIRECTORY = "eres2netv2_shadow_v1"

BinaryDecision = Literal["same", "different"]
ModelName = Literal["ecapa", "eres2netv2"]
ErrorStage = Literal["primary_voice_embedding", "segment_embedding"]

_KNOWN_CASES = (
    ("938050f7193d7c065dc8e249", "segment_0002"),
    ("938050f7193d7c065dc8e249", "segment_0003"),
    ("a073596149def028cff1305e", "segment_0001"),
    ("a073596149def028cff1305e", "segment_0003"),
    ("d5382a9bcae9ba3fd6f8879d", "segment_0001"),
    ("711b1dbc74932c5d1d720495", "segment_0001"),
)


class ScoreDistribution(SchemaModel):
    count: int = Field(ge=0)
    min: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    median: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    mean: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    max: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_distribution(self) -> ScoreDistribution:
        values = (self.min, self.median, self.mean, self.max)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty score distribution cannot publish statistics")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty score distribution is incomplete")
        return self


class ThresholdMetrics(SchemaModel):
    threshold: float = Field(allow_inf_nan=False)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    same_precision: float | None = Field(default=None, ge=0, le=1)
    same_recall: float | None = Field(default=None, ge=0, le=1)
    different_precision: float | None = Field(default=None, ge=0, le=1)
    different_recall: float | None = Field(default=None, ge=0, le=1)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)


class ModelEvaluation(SchemaModel):
    same_scores: ScoreDistribution
    different_scores: ScoreDistribution
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    diagnostic_threshold_count: int = Field(ge=0)
    best_balanced_accuracy_threshold: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    best_balanced_accuracy_metrics: ThresholdMetrics | None = None
    production_threshold_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_best_threshold(self) -> ModelEvaluation:
        if (self.best_balanced_accuracy_threshold is None) != (
            self.best_balanced_accuracy_metrics is None
        ):
            raise ValueError("diagnostic best threshold fields are inconsistent")
        if self.best_balanced_accuracy_metrics is not None and (
            self.best_balanced_accuracy_metrics.threshold
            != self.best_balanced_accuracy_threshold
        ):
            raise ValueError("diagnostic best threshold metric is inconsistent")
        return self


class EvaluationSlice(SchemaModel):
    labeled_same_count: int = Field(ge=0)
    labeled_different_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    evaluation_error_count: int = Field(ge=0)
    evaluated_binary_count: int = Field(ge=0)
    models: dict[ModelName, ModelEvaluation]

    @model_validator(mode="after")
    def validate_counts(self) -> EvaluationSlice:
        if self.evaluated_binary_count + self.evaluation_error_count != (
            self.labeled_same_count + self.labeled_different_count
        ):
            raise ValueError("shadow evaluation slice counts do not reconcile")
        return self


class ERes2NetV2ShadowRecord(SchemaModel):
    schema_version: Literal[
        "r2v.h3.eres2netv2_voice_consistency_shadow.1"
    ] = ERES2NETV2_SHADOW_RECORD_VERSION
    clip_uid: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    identity_scope: IdentityScope
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    duration_bucket: DurationBucket
    human_decision: BinaryDecision
    ecapa_cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    eres2netv2_cosine_similarity: float = Field(
        ge=-1,
        le=1,
        allow_inf_nan=False,
    )
    segment_audio_path: str
    segment_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_path: str
    primary_voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    ecapa_model_identifier: str
    eres2netv2_model_identifier: str
    eres2netv2_model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ERes2NetV2ShadowError(SchemaModel):
    schema_version: Literal[
        "r2v.h3.eres2netv2_voice_consistency_shadow_error.1"
    ] = ERES2NETV2_SHADOW_ERROR_VERSION
    clip_uid: str
    segment_id: str
    entity_id: str
    identity_scope: IdentityScope
    duration_bucket: DurationBucket
    human_decision: BinaryDecision
    stage: ErrorStage
    reason: str
    source_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_audio_path: str
    segment_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_path: str
    primary_voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnownCaseComparison(SchemaModel):
    clip_uid: str
    segment_id: str
    human_decision: BinaryDecision
    identity_scope: IdentityScope
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    ecapa_cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    eres2netv2_cosine_similarity: float = Field(
        ge=-1,
        le=1,
        allow_inf_nan=False,
    )


class ERes2NetV2ShadowSummary(SchemaModel):
    schema_version: Literal[
        "r2v.h3.eres2netv2_voice_consistency_shadow_summary.1"
    ] = ERES2NETV2_SHADOW_SUMMARY_VERSION
    source_audit_root: str
    source_audit_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audit_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_human_annotations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_media_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_annotation_count: int = Field(ge=0)
    stale_annotation_count: int = Field(ge=0)
    labeled_same_count: int = Field(ge=0)
    labeled_different_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    evaluated_record_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    primary_voice_embedding_call_count: int = Field(ge=0)
    segment_embedding_call_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    ecapa_model_identifier: str
    eres2netv2_model_identifier: str
    eres2netv2_model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    speakerlab_source_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    overall: EvaluationSlice
    by_identity_scope: dict[IdentityScope, EvaluationSlice]
    by_duration_bucket: dict[DurationBucket, EvaluationSlice]
    known_cases: list[KnownCaseComparison]
    production_threshold_applied: Literal[False] = False
    binding_modified: Literal[False] = False
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> ERes2NetV2ShadowSummary:
        if self.current_annotation_count != (
            self.labeled_same_count
            + self.labeled_different_count
            + self.uncertain_count
        ):
            raise ValueError("shadow current annotation counts do not reconcile")
        if self.evaluated_record_count + self.error_count != (
            self.labeled_same_count + self.labeled_different_count
        ):
            raise ValueError("shadow evaluated/error counts do not reconcile")
        if self.model_call_count != (
            self.primary_voice_embedding_call_count
            + self.segment_embedding_call_count
        ):
            raise ValueError("shadow model-call counts do not reconcile")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _aggregate_file_fingerprint(values: dict[str, str]) -> str:
    return hashlib.sha256(
        _compact_json(sorted(values.items())).encode("utf-8")
    ).hexdigest()


def _score_distribution(values: Sequence[float]) -> ScoreDistribution:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ScoreDistribution(count=0)
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in ordered):
        raise ValueError("shadow score distribution contains invalid values")
    return ScoreDistribution(
        count=len(ordered),
        min=ordered[0],
        median=float(median(ordered)),
        mean=sum(ordered) / len(ordered),
        max=ordered[-1],
    )


def binary_roc_auc(scores: Sequence[float], decisions: Sequence[BinaryDecision]) -> float | None:
    if len(scores) != len(decisions):
        raise ValueError("ROC-AUC scores and decisions differ in length")
    positives = [score for score, label in zip(scores, decisions, strict=True) if label == "same"]
    negatives = [score for score, label in zip(scores, decisions, strict=True) if label == "different"]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def threshold_metrics(
    scores: Sequence[float],
    decisions: Sequence[BinaryDecision],
    threshold: float,
) -> ThresholdMetrics:
    if len(scores) != len(decisions) or not scores or not math.isfinite(threshold):
        raise ValueError("threshold evaluation input is invalid")
    true_positive = false_positive = true_negative = false_negative = 0
    for score, label in zip(scores, decisions, strict=True):
        predicted_same = score >= threshold
        if label == "same" and predicted_same:
            true_positive += 1
        elif label == "same":
            false_negative += 1
        elif predicted_same:
            false_positive += 1
        else:
            true_negative += 1

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    same_recall = ratio(true_positive, true_positive + false_negative)
    different_recall = ratio(true_negative, true_negative + false_positive)
    balanced = (
        None
        if same_recall is None or different_recall is None
        else (same_recall + different_recall) / 2
    )
    return ThresholdMetrics(
        threshold=threshold,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        same_precision=ratio(true_positive, true_positive + false_positive),
        same_recall=same_recall,
        different_precision=ratio(true_negative, true_negative + false_negative),
        different_recall=different_recall,
        balanced_accuracy=balanced,
        accuracy=(true_positive + true_negative) / len(scores),
    )


def diagnostic_threshold_sweep(
    scores: Sequence[float],
    decisions: Sequence[BinaryDecision],
) -> list[ThresholdMetrics]:
    if len(scores) != len(decisions):
        raise ValueError("threshold sweep scores and decisions differ in length")
    if not scores:
        return []
    unique = sorted({float(score) for score in scores})
    if not all(math.isfinite(score) and -1 <= score <= 1 for score in unique):
        raise ValueError("threshold sweep contains an invalid score")
    thresholds = [math.nextafter(unique[0], -math.inf)]
    thresholds.extend((left + right) / 2 for left, right in pairwise(unique))
    thresholds.append(math.nextafter(unique[-1], math.inf))
    return [threshold_metrics(scores, decisions, value) for value in thresholds]


def _model_evaluation(
    records: Sequence[ERes2NetV2ShadowRecord],
    *,
    score: Callable[[ERes2NetV2ShadowRecord], float],
) -> ModelEvaluation:
    scores = [score(item) for item in records]
    decisions = [item.human_decision for item in records]
    sweep = diagnostic_threshold_sweep(scores, decisions)
    eligible = [item for item in sweep if item.balanced_accuracy is not None]
    best = (
        max(
            eligible,
            key=lambda item: (
                float(item.balanced_accuracy),
                item.accuracy,
                item.threshold,
            ),
        )
        if eligible
        else None
    )
    return ModelEvaluation(
        same_scores=_score_distribution(
            [score(item) for item in records if item.human_decision == "same"]
        ),
        different_scores=_score_distribution(
            [score(item) for item in records if item.human_decision == "different"]
        ),
        roc_auc=binary_roc_auc(scores, decisions),
        diagnostic_threshold_count=len(sweep),
        best_balanced_accuracy_threshold=(None if best is None else best.threshold),
        best_balanced_accuracy_metrics=best,
    )


def _evaluation_slice(
    annotations: Sequence[VoiceConsistencyReviewAnnotation],
    records: Sequence[ERes2NetV2ShadowRecord],
    errors: Sequence[ERes2NetV2ShadowError],
) -> EvaluationSlice:
    return EvaluationSlice(
        labeled_same_count=sum(item.decision == "same" for item in annotations),
        labeled_different_count=sum(
            item.decision == "different" for item in annotations
        ),
        uncertain_count=sum(item.decision == "uncertain" for item in annotations),
        evaluation_error_count=len(errors),
        evaluated_binary_count=len(records),
        models={
            "ecapa": _model_evaluation(
                records,
                score=lambda item: item.ecapa_cosine_similarity,
            ),
            "eres2netv2": _model_evaluation(
                records,
                score=lambda item: item.eres2netv2_cosine_similarity,
            ),
        },
    )


def _safe_segment_path(audit_root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("shadow segment audio path is unsafe")
    path = audit_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        path.relative_to(audit_root)
    except ValueError as exc:
        raise ValueError("shadow segment audio escapes audit root") from exc
    if not path.is_file():
        raise ValueError("shadow segment audio is not a file")
    return path


def _validate_media(path: Path, expected_sha256: str, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or _sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{label} differs from audit provenance")
    return resolved


def _backend_identity(backend: SpeakerEmbeddingBackend) -> tuple[str, str]:
    identifier = str(getattr(backend, "model_identifier", type(backend).__name__))
    fingerprint = str(getattr(backend, "checkpoint_sha256", ""))
    if not identifier or len(fingerprint) != 64:
        raise ValueError("ERes2NetV2 backend identity is incomplete")
    return identifier, fingerprint


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            _compact_json(item.model_dump(mode="json")) + "\n" for item in values
        ),
        encoding="utf-8",
    )


def _comparison_csv(records: Sequence[ERes2NetV2ShadowRecord]) -> str:
    output = io.StringIO(newline="")
    fields = tuple(ERes2NetV2ShadowRecord.model_fields)
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(fields)
    for record in records:
        value = record.model_dump(mode="json")
        writer.writerow(value[field] for field in fields)
    return output.getvalue()


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"ERes2NetV2 shadow output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_eres2netv2_voice_consistency_shadow(
    *,
    audit_root: Path,
    speaker_backend: SpeakerEmbeddingBackend,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> ERes2NetV2ShadowSummary:
    context = load_review_context(audit_root)
    root = context.audit_root
    annotation_path = root / "human_review/annotations.jsonl"
    if not annotation_path.is_file():
        raise ValueError("voice-consistency human annotations are unavailable")
    annotations, stale_count = load_current_review_annotations(context)
    records_by_key = {
        (item.clip_uid, item.segment_id): item for item in context.records
    }
    audit_summary = context.audit_summary
    ecapa_identifier = audit_summary.speaker_model_identifier
    if any(
        item.speaker_model_identifier != ecapa_identifier
        for item in context.records
    ):
        raise ValueError("ECAPA record model identity differs from audit summary")
    binary_annotations = [
        item for item in annotations if item.decision in {"same", "different"}
    ]
    destination = (
        output_root.expanduser().resolve(strict=False)
        if output_root is not None
        else root / DEFAULT_OUTPUT_DIRECTORY
    )
    if destination in {root, root / "human_review"}:
        raise ValueError("shadow output cannot replace source audit artifacts")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"ERes2NetV2 shadow output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "summary": root / "summary.json",
        "records": root / "records.jsonl",
        "annotations": annotation_path,
        "h3_samples": Path(
            context.audit_summary.source_audio_production_root
        ).expanduser().resolve(strict=True)
        / "h3/samples.jsonl",
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    model_identifier, model_fingerprint = _backend_identity(speaker_backend)
    primary_cache: dict[tuple[str, str], np.ndarray | str] = {}
    segment_cache: dict[tuple[str, str], np.ndarray | str] = {}
    primary_calls = 0
    segment_calls = 0
    media_hashes: dict[str, str] = {}
    speakerlab_fingerprints: set[str] = set()
    comparison_records: list[ERes2NetV2ShadowRecord] = []
    errors: list[ERes2NetV2ShadowError] = []

    def embed(
        *,
        cache: dict[tuple[str, str], np.ndarray | str],
        path: Path,
        sha256: str,
        request_identity: str,
    ) -> tuple[np.ndarray | str, bool]:
        key = (str(path), sha256)
        if key in cache:
            return cache[key], False
        try:
            result = speaker_backend.embed_speaker(
                entity_occurrence_id=request_identity,
                audio_path=path,
            )
            if (
                result.model_identifier != model_identifier
                or result.checkpoint_sha256 != model_fingerprint
            ):
                raise ValueError("ERes2NetV2 backend result identity differs")
            metadata = result.backend_metadata or {}
            source_fingerprint = metadata.get("speakerlab_source_fingerprint")
            if source_fingerprint is not None:
                if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
                    raise ValueError("SpeakerLab source fingerprint is invalid")
                speakerlab_fingerprints.add(source_fingerprint)
            value: np.ndarray | str = np.asarray(result.vector, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - fail soft per labeled record
            value = f"{type(exc).__name__}: {exc}"
        cache[key] = value
        return value, True

    for annotation in binary_annotations:
        key = (annotation.clip_uid, annotation.segment_id)
        record = records_by_key[key]
        segment_path = _safe_segment_path(root, record.segment_audio_path)
        segment_path = _validate_media(
            segment_path,
            record.segment_audio_sha256,
            label="segment audio",
        )
        primary_path = _validate_media(
            Path(record.primary_voice_reference_path),
            record.primary_voice_reference_sha256,
            label="primary voice",
        )
        media_hashes[str(segment_path)] = record.segment_audio_sha256
        media_hashes[str(primary_path)] = record.primary_voice_reference_sha256
        primary_value, called = embed(
            cache=primary_cache,
            path=primary_path,
            sha256=record.primary_voice_reference_sha256,
            request_identity=f"primary/{record.entity_occurrence_id}",
        )
        primary_calls += int(called)
        if isinstance(primary_value, str):
            errors.append(
                ERes2NetV2ShadowError(
                    **_error_fields(record, annotation),
                    stage="primary_voice_embedding",
                    reason=primary_value,
                )
            )
            continue
        segment_value, called = embed(
            cache=segment_cache,
            path=segment_path,
            sha256=record.segment_audio_sha256,
            request_identity=f"segment/{record.clip_uid}/{record.segment_id}",
        )
        segment_calls += int(called)
        if isinstance(segment_value, str):
            errors.append(
                ERes2NetV2ShadowError(
                    **_error_fields(record, annotation),
                    stage="segment_embedding",
                    reason=segment_value,
                )
            )
            continue
        comparison_records.append(
            ERes2NetV2ShadowRecord(
                clip_uid=record.clip_uid,
                segment_id=record.segment_id,
                entity_id=record.entity_id,
                identity_scope=record.identity_scope,
                duration_seconds=record.duration_seconds,
                duration_bucket=record.duration_bucket,
                human_decision=annotation.decision,
                ecapa_cosine_similarity=record.cosine_similarity,
                eres2netv2_cosine_similarity=cosine_similarity(
                    primary_value,
                    segment_value,
                ),
                segment_audio_path=record.segment_audio_path,
                segment_audio_sha256=record.segment_audio_sha256,
                primary_voice_reference_path=record.primary_voice_reference_path,
                primary_voice_reference_sha256=(
                    record.primary_voice_reference_sha256
                ),
                source_record_fingerprint=annotation.record_fingerprint,
                ecapa_model_identifier=record.speaker_model_identifier,
                eres2netv2_model_identifier=model_identifier,
                eres2netv2_model_fingerprint=model_fingerprint,
            )
        )

    comparison_records.sort(key=lambda item: (item.clip_uid, item.segment_id))
    errors.sort(key=lambda item: (item.clip_uid, item.segment_id, item.stage))
    if len(speakerlab_fingerprints) > 1:
        raise ValueError("SpeakerLab source changed during shadow evaluation")
    record_by_key = {
        (item.clip_uid, item.segment_id): item for item in comparison_records
    }
    error_keys = {(item.clip_uid, item.segment_id) for item in errors}

    def slice_for(
        predicate: Callable[[VoiceConsistencyAuditRecord], bool],
    ) -> EvaluationSlice:
        selected_annotations = [
            item for item in annotations if predicate(records_by_key[(item.clip_uid, item.segment_id)])
        ]
        selected_records = [
            item
            for item in comparison_records
            if predicate(records_by_key[(item.clip_uid, item.segment_id)])
        ]
        selected_errors = [
            item
            for item in errors
            if predicate(records_by_key[(item.clip_uid, item.segment_id)])
        ]
        return _evaluation_slice(selected_annotations, selected_records, selected_errors)

    summary = ERes2NetV2ShadowSummary(
        source_audit_root=str(root),
        source_audit_summary_sha256=source_hashes["summary"],
        source_audit_records_sha256=source_hashes["records"],
        source_human_annotations_sha256=source_hashes["annotations"],
        source_h3_samples_sha256=source_hashes["h3_samples"],
        source_media_set_fingerprint=_aggregate_file_fingerprint(media_hashes),
        current_annotation_count=len(annotations),
        stale_annotation_count=stale_count,
        labeled_same_count=sum(item.decision == "same" for item in annotations),
        labeled_different_count=sum(
            item.decision == "different" for item in annotations
        ),
        uncertain_count=sum(item.decision == "uncertain" for item in annotations),
        evaluated_record_count=len(comparison_records),
        error_count=len(errors),
        primary_voice_embedding_call_count=primary_calls,
        segment_embedding_call_count=segment_calls,
        model_call_count=primary_calls + segment_calls,
        ecapa_model_identifier=ecapa_identifier,
        eres2netv2_model_identifier=model_identifier,
        eres2netv2_model_fingerprint=model_fingerprint,
        speakerlab_source_fingerprint=(
            next(iter(speakerlab_fingerprints)) if speakerlab_fingerprints else None
        ),
        overall=_evaluation_slice(annotations, comparison_records, errors),
        by_identity_scope={
            scope: slice_for(lambda item, scope=scope: item.identity_scope == scope)
            for scope in ("direct_anchor_present", "cluster_propagated_only")
        },
        by_duration_bucket={
            bucket: slice_for(lambda item, bucket=bucket: item.duration_bucket == bucket)
            for bucket in ("<0.75s", "0.75-1.0s", "1.0-2.0s", ">=2.0s")
        },
        known_cases=[
            KnownCaseComparison(
                clip_uid=item.clip_uid,
                segment_id=item.segment_id,
                human_decision=item.human_decision,
                identity_scope=item.identity_scope,
                duration_seconds=item.duration_seconds,
                ecapa_cosine_similarity=item.ecapa_cosine_similarity,
                eres2netv2_cosine_similarity=item.eres2netv2_cosine_similarity,
            )
            for key in _KNOWN_CASES
            if (item := record_by_key.get(key)) is not None
        ],
    )
    if error_keys != {
        (item.clip_uid, item.segment_id)
        for item in binary_annotations
        if (item.clip_uid, item.segment_id) not in record_by_key
    }:
        raise ValueError("shadow error inventory does not reconcile")

    current_source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    current_media_hashes = {path: _sha256_file(Path(path)) for path in media_hashes}
    if current_source_hashes != source_hashes or current_media_hashes != media_hashes:
        raise ValueError("voice-consistency inputs changed during shadow evaluation")

    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir()
        _write_jsonl(temporary / "records.jsonl", comparison_records)
        _write_jsonl(temporary / "errors.jsonl", errors)
        (temporary / "comparison.csv").write_text(
            _comparison_csv(comparison_records),
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary


def _error_fields(
    record: VoiceConsistencyAuditRecord,
    annotation: VoiceConsistencyReviewAnnotation,
) -> dict[str, object]:
    return {
        "clip_uid": record.clip_uid,
        "segment_id": record.segment_id,
        "entity_id": record.entity_id,
        "identity_scope": record.identity_scope,
        "duration_bucket": record.duration_bucket,
        "human_decision": annotation.decision,
        "source_record_fingerprint": annotation.record_fingerprint,
        "segment_audio_path": record.segment_audio_path,
        "segment_audio_sha256": record.segment_audio_sha256,
        "primary_voice_reference_path": record.primary_voice_reference_path,
        "primary_voice_reference_sha256": record.primary_voice_reference_sha256,
    }


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "ERes2NetV2ShadowError",
    "ERes2NetV2ShadowRecord",
    "ERes2NetV2ShadowSummary",
    "EvaluationSlice",
    "ModelEvaluation",
    "ScoreDistribution",
    "ThresholdMetrics",
    "binary_roc_auc",
    "diagnostic_threshold_sweep",
    "run_eres2netv2_voice_consistency_shadow",
    "threshold_metrics",
]
