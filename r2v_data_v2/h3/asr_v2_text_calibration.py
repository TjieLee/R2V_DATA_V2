from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import statistics
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.asr_transcription import ASRHumanQACounts
from r2v_data_v2.h3.asr_v2_transcription import (
    ASR_V2_CALIBRATION_HUMAN_QA_CORRECT,
    ASR_V2_CALIBRATION_HUMAN_QA_TOTAL,
    ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN,
    ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED,
    ASR_V2_CALIBRATION_HUMAN_QA_WRONG,
    ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT,
    ASRV2HumanQAExport,
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

CALIBRATION_INVENTORY_SCHEMA_VERSION = "r2v.h3.asr_v2_text_calibration_inventory.1"
CALIBRATION_SEGMENT_SCHEMA_VERSION = "r2v.h3.asr_v2_text_calibration_segment.1"
CALIBRATION_SUMMARY_SCHEMA_VERSION = "r2v.h3.asr_v2_text_calibration_summary.1"
CALIBRATION_SWEEP_SCHEMA_VERSION = "r2v.h3.asr_v2_text_calibration_sweep.1"
CALIBRATION_OUTPUT_DIRECTORY = "asr_v2_text_calibration"
CALIBRATION_PILOT_DIRECTORY = "asr_v2_pilot20"
CALIBRATION_PRODUCTION_DIRECTORY = "production/asr_v2"
CALIBRATION_CANDIDATE_LABEL = "CALIBRATION CANDIDATE ONLY"
MAX_PAIRWISE_THRESHOLDS_PER_DIAGNOSTIC = 9
SHORTLIST_LIMIT = 20

QALabel = Literal["CORRECT", "WRONG", "UNCERTAIN"]
DiagnosticName = Literal[
    "avg_log_probability",
    "no_speech_probability",
    "language_probability",
    "compression_ratio",
    "duration_seconds",
]
ConditionOperator = Literal["ge", "le"]

DIAGNOSTIC_ORDER: tuple[DiagnosticName, ...] = (
    "avg_log_probability",
    "no_speech_probability",
    "language_probability",
    "compression_ratio",
    "duration_seconds",
)
DIAGNOSTIC_OPERATORS: dict[DiagnosticName, ConditionOperator] = {
    "avg_log_probability": "ge",
    "no_speech_probability": "le",
    "language_probability": "ge",
    "compression_ratio": "le",
    "duration_seconds": "ge",
}


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"calibration JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


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


class TextCalibrationSegment(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_text_calibration_segment.1"] = (
        CALIBRATION_SEGMENT_SCHEMA_VERSION
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    qa_label: QALabel
    status: Literal["transcribed", "uncertain", "failed"]
    text: str | None = None
    text_present: bool
    raw_text_available: bool
    duration_seconds: float = Field(gt=0)
    language_probability: float | None = Field(default=None, ge=0, le=1)
    avg_log_probability: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)
    compression_ratio: float | None = Field(default=None, ge=0)
    decoder_segment_count: int | None = Field(default=None, ge=0)
    cluster_binding_status: Literal[
        "candidate_mapped", "ambiguous", "unbound", "conflict"
    ]
    identity_scope: Literal[
        "direct_anchor_present", "cluster_propagated_only", "unresolved"
    ]

    @model_validator(mode="after")
    def validate_segment(self) -> TextCalibrationSegment:
        expected_present = self.text is not None and bool(self.text.strip())
        if self.text_present != expected_present:
            raise ValueError("calibration text presence is inconsistent")
        if self.raw_text_available != (
            self.status == "transcribed" and self.text_present
        ):
            raise ValueError(
                "raw text availability must require transcribed non-empty text"
            )
        values = (
            self.duration_seconds,
            self.language_probability,
            self.avg_log_probability,
            self.no_speech_probability,
            self.compression_ratio,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("calibration diagnostics must be finite")
        return self


class DiagnosticDistribution(SchemaModel):
    count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    minimum: float | None = None
    p10: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p90: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> DiagnosticDistribution:
        values = (
            self.minimum,
            self.p10,
            self.p25,
            self.median,
            self.p75,
            self.p90,
            self.maximum,
        )
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty diagnostic distribution cannot publish values")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty diagnostic distribution requires all values")
        return self


class CalibrationCondition(SchemaModel):
    diagnostic: DiagnosticName
    operator: ConditionOperator
    threshold: float
    missing_value_behavior: Literal["not_retained"] = "not_retained"

    @model_validator(mode="after")
    def validate_condition(self) -> CalibrationCondition:
        if self.operator != DIAGNOSTIC_OPERATORS[self.diagnostic]:
            raise ValueError("calibration diagnostic uses an unsupported direction")
        if not math.isfinite(self.threshold):
            raise ValueError("calibration threshold must be finite")
        return self


class CalibrationRuleResult(SchemaModel):
    rule_id: str
    rule: str
    label: Literal["CALIBRATION CANDIDATE ONLY"] = CALIBRATION_CANDIDATE_LABEL
    conditions: list[CalibrationCondition]
    rule_complexity: int = Field(ge=0, le=2)
    missing_value_count: int = Field(ge=0)
    retained_segment_count: int = Field(ge=0)
    correct_retained: int = Field(ge=0)
    wrong_retained: int = Field(ge=0)
    uncertain_retained: int = Field(ge=0)
    correct_rejected: int = Field(ge=0)
    wrong_rejected: int = Field(ge=0)
    uncertain_rejected: int = Field(ge=0)
    correct_retention: float | None = Field(default=None, ge=0, le=1)
    wrong_leakage: float | None = Field(default=None, ge=0, le=1)
    uncertain_retention: float | None = Field(default=None, ge=0, le=1)
    explicit_precision: float | None = Field(default=None, ge=0, le=1)
    conservative_precision: float | None = Field(default=None, ge=0, le=1)
    retained_clip_count: int = Field(ge=0)
    clips_with_retained_wrong: list[str]
    empirical_wrong_margin: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_rule(self) -> CalibrationRuleResult:
        if self.rule_complexity != len(self.conditions):
            raise ValueError("calibration rule complexity is inconsistent")
        if self.retained_segment_count != (
            self.correct_retained + self.wrong_retained + self.uncertain_retained
        ):
            raise ValueError("calibration retained labels do not reconcile")
        return self


class ProductionShadowResult(SchemaModel):
    rule_id: str
    rule: str
    production_segment_count: int = Field(ge=0)
    production_raw_text_available_count: int = Field(ge=0)
    production_retained_count: int = Field(ge=0)
    production_hidden_count: int = Field(ge=0)
    retained_by_cluster_binding_status: dict[str, int]
    hidden_by_cluster_binding_status: dict[str, int]
    retained_by_identity_scope: dict[str, int]
    hidden_by_identity_scope: dict[str, int]
    production_precision_claimed: Literal[False] = False

    @model_validator(mode="after")
    def validate_shadow(self) -> ProductionShadowResult:
        if self.production_raw_text_available_count != (
            self.production_retained_count + self.production_hidden_count
        ):
            raise ValueError(
                "production shadow retained/hidden counts do not reconcile"
            )
        if sum(self.retained_by_cluster_binding_status.values()) != (
            self.production_retained_count
        ) or sum(self.hidden_by_cluster_binding_status.values()) != (
            self.production_hidden_count
        ):
            raise ValueError("production shadow binding-status counts do not reconcile")
        if sum(self.retained_by_identity_scope.values()) != (
            self.production_retained_count
        ) or sum(self.hidden_by_identity_scope.values()) != (
            self.production_hidden_count
        ):
            raise ValueError("production shadow identity-scope counts do not reconcile")
        return self


class TextCalibrationInventory(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_text_calibration_inventory.1"] = (
        CALIBRATION_INVENTORY_SCHEMA_VERSION
    )
    output_root: str
    pilot_root: str
    pilot_inventory_path: str
    pilot_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_segments_path: str
    pilot_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_root: str
    production_inventory_path: str
    production_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_segments_path: str
    production_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_qa_source_path: str
    human_qa_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_segment_count: int = Field(ge=0)
    production_segment_count: int = Field(ge=0)
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_only: Literal[True] = True
    model_calls: Literal[0] = 0
    gpu_calls: Literal[0] = 0
    text_usability_policy_validated: Literal[False] = False
    text_usability_gate_applied: Literal[False] = False
    transcript_confidence_threshold_used: Literal[False] = False
    production_asr_modified: Literal[False] = False
    diarization_modified: Literal[False] = False
    voice_pair_embedding_modified: Literal[False] = False
    language_probability_is_transcript_correctness_probability: Literal[False] = False

    @model_validator(mode="after")
    def validate_inventory(self) -> TextCalibrationInventory:
        values = self.model_dump(mode="json", exclude={"inventory_fingerprint"})
        expected = _sha256_bytes(_compact_json(values).encode())
        if self.inventory_fingerprint != expected:
            raise ValueError("text calibration inventory fingerprint is inconsistent")
        return self


class TextCalibrationSweep(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_text_calibration_sweep.1"] = (
        CALIBRATION_SWEEP_SCHEMA_VERSION
    )
    baseline: CalibrationRuleResult
    one_dimensional_rules: list[CalibrationRuleResult]
    two_condition_rules: list[CalibrationRuleResult]
    zero_wrong_shortlist: list[CalibrationRuleResult]
    at_most_one_wrong_shortlist: list[CalibrationRuleResult]
    pareto_frontier: list[CalibrationRuleResult]
    production_shadow: list[ProductionShadowResult]
    learned_classifier_used: Literal[False] = False
    maximum_rule_condition_count: Literal[2] = 2
    fixed_half_threshold_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_sweep(self) -> TextCalibrationSweep:
        if self.baseline.conditions or self.baseline.rule_complexity != 0:
            raise ValueError("text calibration baseline must not contain a threshold")
        if any(item.rule_complexity != 1 for item in self.one_dimensional_rules):
            raise ValueError("one-dimensional sweep contains an invalid rule")
        if any(item.rule_complexity != 2 for item in self.two_condition_rules):
            raise ValueError("two-condition sweep contains an invalid rule")
        if any(item.wrong_retained != 0 for item in self.zero_wrong_shortlist):
            raise ValueError("zero-WRONG shortlist contains a leaking rule")
        if any(item.wrong_retained > 1 for item in self.at_most_one_wrong_shortlist):
            raise ValueError("secondary shortlist contains more than one WRONG")
        return self


class TextCalibrationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.asr_v2_text_calibration_summary.1"] = (
        CALIBRATION_SUMMARY_SCHEMA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_segment_count: int = Field(ge=0)
    production_segment_count: int = Field(ge=0)
    qa_counts: ASRHumanQACounts
    raw_text_available_count: int = Field(ge=0)
    diagnostic_distributions: dict[str, dict[str, DiagnosticDistribution]]
    baseline: CalibrationRuleResult
    one_dimensional_rule_count: int = Field(ge=0)
    two_condition_rule_count: int = Field(ge=0)
    zero_wrong_shortlist_count: int = Field(ge=0)
    at_most_one_wrong_shortlist_count: int = Field(ge=0)
    pareto_frontier_count: int = Field(ge=0)
    hard_case_count: int = Field(ge=0)
    production_shadow_rule_count: int = Field(ge=0)
    text_usability_policy_validated: Literal[False] = False
    text_usability_gate_applied: Literal[False] = False
    transcript_confidence_threshold_used: Literal[False] = False
    raw_asr_preserved: Literal[True] = True
    speaker_identity_independent: Literal[True] = True
    voice_reference_quality_independent: Literal[True] = True
    production_precision_claimed: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> TextCalibrationSummary:
        reviewed = (
            self.qa_counts.CORRECT
            + self.qa_counts.WRONG
            + self.qa_counts.UNCERTAIN
            + self.qa_counts.UNLABELED
        )
        if reviewed != self.pilot_segment_count:
            raise ValueError("text calibration QA counts do not reconcile")
        if self.hard_case_count != self.qa_counts.WRONG + self.qa_counts.UNCERTAIN:
            raise ValueError("text calibration hard-case count is inconsistent")
        return self


def _validate_inventory(
    path: Path, *, expected_mode: Literal["pilot20", "production"]
) -> ASRV2InventoryRecord:
    inventory = load_asr_v2_inventory(path)
    if inventory.mode != expected_mode:
        raise ValueError(f"ASR V2 {expected_mode} inventory mode is inconsistent")
    if inventory.inventory_fingerprint != _asr_v2_inventory_fingerprint(inventory):
        raise ValueError(
            f"ASR V2 {expected_mode} inventory fingerprint is inconsistent"
        )
    return inventory


def _load_records(
    *, root: Path, inventory: ASRV2InventoryRecord
) -> list[ASRV2SegmentRecord]:
    records = [
        ASRV2SegmentRecord.model_validate(row)
        for row in _read_jsonl(root / "segments.jsonl")
    ]
    summary = load_asr_v2_summary(root / "summary.json")
    if summary.mode != inventory.mode:
        raise ValueError("ASR V2 source summary mode is inconsistent")
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("ASR V2 source summary fingerprint is inconsistent")
    if len(records) != len(inventory.jobs) or summary.segment_count != len(records):
        raise ValueError("ASR V2 source segment count is inconsistent")
    job_fields = tuple(ASRV2SegmentJob.model_fields)
    for job, record in zip(inventory.jobs, records, strict=True):
        record_job = ASRV2SegmentJob.model_validate(
            {field: getattr(record, field) for field in job_fields}
        )
        if record_job != job:
            raise ValueError("ASR V2 source record does not match inventory order")
    identities = [
        (row.target_clip_uid, row.segment_id, row.speaker_cluster_id) for row in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("ASR V2 source segment identities must be unique")
    return records


def _validate_qa(
    *,
    qa_path: Path,
    expected_inventory_fingerprint: str,
    expected_counts: ASRHumanQACounts,
) -> tuple[ASRV2HumanQAExport, bytes]:
    raw = qa_path.read_bytes()
    qa = ASRV2HumanQAExport.model_validate_json(raw)
    if qa.mode != "pilot20":
        raise ValueError("ASR V2 text calibration requires pilot20 human QA")
    if qa.inventory_fingerprint != expected_inventory_fingerprint:
        raise ValueError("ASR V2 human-QA inventory fingerprint is inconsistent")
    if qa.total_segment_count != ASR_V2_CALIBRATION_HUMAN_QA_TOTAL:
        raise ValueError("ASR V2 human-QA total segment count is inconsistent")
    if qa.label_count != ASR_V2_CALIBRATION_HUMAN_QA_TOTAL:
        raise ValueError("ASR V2 human-QA must cover every pilot segment")
    if qa.counts != expected_counts:
        raise ValueError("ASR V2 human-QA frozen counts are inconsistent")
    return qa, raw


def _join_segments(
    *, records: Sequence[ASRV2SegmentRecord], qa: ASRV2HumanQAExport
) -> list[TextCalibrationSegment]:
    labels = {
        (row.target_clip_uid, row.segment_id, row.speaker_cluster_id): row.label
        for row in qa.labels
    }
    record_keys = [
        (row.target_clip_uid, row.segment_id, row.speaker_cluster_id) for row in records
    ]
    if set(labels) != set(record_keys) or len(labels) != len(record_keys):
        missing = sorted(set(record_keys) - set(labels))
        extra = sorted(set(labels) - set(record_keys))
        raise ValueError(
            f"ASR V2 human-QA coverage is not one-to-one; missing={missing!r}, extra={extra!r}"
        )
    joined: list[TextCalibrationSegment] = []
    for record, key in zip(records, record_keys, strict=True):
        diagnostics = record.diagnostics
        text_present = record.text is not None and bool(record.text.strip())
        joined.append(
            TextCalibrationSegment(
                target_clip_uid=record.target_clip_uid,
                segment_id=record.segment_id,
                speaker_cluster_id=record.speaker_cluster_id,
                qa_label=labels[key],
                status=record.status,
                text=record.text,
                text_present=text_present,
                raw_text_available=record.status == "transcribed" and text_present,
                duration_seconds=record.end_time - record.start_time,
                language_probability=(
                    diagnostics.language_probability
                    if diagnostics is not None
                    else None
                ),
                avg_log_probability=(
                    diagnostics.avg_log_probability if diagnostics is not None else None
                ),
                no_speech_probability=(
                    diagnostics.no_speech_probability
                    if diagnostics is not None
                    else None
                ),
                compression_ratio=(
                    diagnostics.compression_ratio if diagnostics is not None else None
                ),
                decoder_segment_count=(
                    diagnostics.decoder_segment_count
                    if diagnostics is not None
                    else None
                ),
                cluster_binding_status=record.cluster_binding_status,
                identity_scope=record.identity_scope,
            )
        )
    return joined


def raw_text_available(record: ASRV2SegmentRecord | TextCalibrationSegment) -> bool:
    return (
        record.status == "transcribed"
        and record.text is not None
        and bool(record.text.strip())
    )


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(
    rows: Sequence[TextCalibrationSegment], diagnostic: DiagnosticName
) -> DiagnosticDistribution:
    values = [
        float(value) for row in rows if (value := getattr(row, diagnostic)) is not None
    ]
    if not values:
        return DiagnosticDistribution(count=0, missing_count=len(rows))
    return DiagnosticDistribution(
        count=len(values),
        missing_count=len(rows) - len(values),
        minimum=min(values),
        p10=_quantile(values, 0.10),
        p25=_quantile(values, 0.25),
        median=statistics.median(values),
        p75=_quantile(values, 0.75),
        p90=_quantile(values, 0.90),
        maximum=max(values),
    )


def _diagnostic_distributions(
    rows: Sequence[TextCalibrationSegment],
) -> dict[str, dict[str, DiagnosticDistribution]]:
    grouped = {
        label: [row for row in rows if row.qa_label == label]
        for label in ("CORRECT", "WRONG", "UNCERTAIN")
    }
    return {
        diagnostic: {
            label: _distribution(label_rows, diagnostic)
            for label, label_rows in grouped.items()
        }
        for diagnostic in DIAGNOSTIC_ORDER
    }


def _threshold_candidates(
    values: Sequence[float], *, operator: ConditionOperator
) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return []
    if len(unique) == 1:
        return unique
    midpoints = [(left + right) / 2.0 for left, right in pairwise(unique)]
    return [unique[0], *midpoints] if operator == "ge" else [*midpoints, unique[-1]]


def _format_threshold(value: float) -> str:
    return format(value, ".12g")


def _condition_text(condition: CalibrationCondition) -> str:
    symbol = ">=" if condition.operator == "ge" else "<="
    return f"{condition.diagnostic} {symbol} {_format_threshold(condition.threshold)}"


def _rule_identity(conditions: Sequence[CalibrationCondition]) -> tuple[str, str]:
    if not conditions:
        return "raw_text_available_only", "raw_text_available_only"
    representation = " AND ".join(_condition_text(item) for item in conditions)
    digest = hashlib.sha256(representation.encode("utf-8")).hexdigest()[:16]
    return f"diagnostic_gate_{digest}", representation


def _condition_passes(row: object, condition: CalibrationCondition) -> bool:
    value = getattr(row, condition.diagnostic)
    if value is None:
        return False
    if condition.operator == "ge":
        return value >= condition.threshold
    return value <= condition.threshold


def _rule_passes(
    row: ASRV2SegmentRecord | TextCalibrationSegment,
    conditions: Sequence[CalibrationCondition],
) -> bool:
    if not raw_text_available(row):
        return False
    return all(_condition_passes(row, condition) for condition in conditions)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _empirical_wrong_margin(
    *,
    rows: Sequence[TextCalibrationSegment],
    conditions: Sequence[CalibrationCondition],
    ranges: dict[DiagnosticName, float],
) -> float | None:
    if not conditions:
        return None
    margins: list[float] = []
    for row in rows:
        if row.qa_label != "WRONG" or _rule_passes(row, conditions):
            continue
        deficits: list[float] = []
        missing = False
        for condition in conditions:
            value = getattr(row, condition.diagnostic)
            if value is None:
                missing = True
                break
            raw_deficit = (
                condition.threshold - value
                if condition.operator == "ge"
                else value - condition.threshold
            )
            scale = ranges[condition.diagnostic]
            deficits.append(max(0.0, raw_deficit) / scale if scale > 0 else 0.0)
        margins.append(0.0 if missing else max(deficits, default=0.0))
    return min(margins) if margins else None


def _evaluate_rule(
    *,
    rows: Sequence[TextCalibrationSegment],
    conditions: Sequence[CalibrationCondition],
    ranges: dict[DiagnosticName, float],
) -> CalibrationRuleResult:
    rule_id, representation = _rule_identity(conditions)
    retained = [row for row in rows if _rule_passes(row, conditions)]
    retained_counts = Counter(row.qa_label for row in retained)
    total_counts = Counter(row.qa_label for row in rows)
    rejected_counts = total_counts - retained_counts
    missing_count = sum(
        row.raw_text_available
        and any(getattr(row, condition.diagnostic) is None for condition in conditions)
        for row in rows
    )
    retained_clips = sorted({row.target_clip_uid for row in retained})
    wrong_clips = sorted(
        {row.target_clip_uid for row in retained if row.qa_label == "WRONG"}
    )
    correct = retained_counts["CORRECT"]
    wrong = retained_counts["WRONG"]
    uncertain = retained_counts["UNCERTAIN"]
    return CalibrationRuleResult(
        rule_id=rule_id,
        rule=representation,
        conditions=list(conditions),
        rule_complexity=len(conditions),
        missing_value_count=missing_count,
        retained_segment_count=len(retained),
        correct_retained=correct,
        wrong_retained=wrong,
        uncertain_retained=uncertain,
        correct_rejected=rejected_counts["CORRECT"],
        wrong_rejected=rejected_counts["WRONG"],
        uncertain_rejected=rejected_counts["UNCERTAIN"],
        correct_retention=_ratio(correct, total_counts["CORRECT"]),
        wrong_leakage=_ratio(wrong, total_counts["WRONG"]),
        uncertain_retention=_ratio(uncertain, total_counts["UNCERTAIN"]),
        explicit_precision=_ratio(correct, correct + wrong),
        conservative_precision=_ratio(correct, correct + wrong + uncertain),
        retained_clip_count=len(retained_clips),
        clips_with_retained_wrong=wrong_clips,
        empirical_wrong_margin=_empirical_wrong_margin(
            rows=rows, conditions=conditions, ranges=ranges
        ),
    )


def _conditions_for_diagnostic(
    rows: Sequence[TextCalibrationSegment], diagnostic: DiagnosticName
) -> list[CalibrationCondition]:
    values = [
        float(value)
        for row in rows
        if row.raw_text_available and (value := getattr(row, diagnostic)) is not None
    ]
    operator = DIAGNOSTIC_OPERATORS[diagnostic]
    return [
        CalibrationCondition(
            diagnostic=diagnostic,
            operator=operator,
            threshold=threshold,
        )
        for threshold in _threshold_candidates(values, operator=operator)
    ]


def _bounded_conditions(
    conditions: Sequence[CalibrationCondition],
) -> list[CalibrationCondition]:
    if len(conditions) <= MAX_PAIRWISE_THRESHOLDS_PER_DIAGNOSTIC:
        return list(conditions)
    count = MAX_PAIRWISE_THRESHOLDS_PER_DIAGNOSTIC
    indices = [
        round(index * (len(conditions) - 1) / (count - 1)) for index in range(count)
    ]
    return [conditions[index] for index in dict.fromkeys(indices)]


def _shortlist_key(rule: CalibrationRuleResult) -> tuple[object, ...]:
    margin = rule.empirical_wrong_margin
    return (
        -rule.correct_retained,
        rule.uncertain_retained,
        rule.rule_complexity,
        -(margin if margin is not None else -1.0),
        rule.rule,
    )


def _shortlist(
    rules: Sequence[CalibrationRuleResult], *, maximum_wrong: int
) -> list[CalibrationRuleResult]:
    eligible = [rule for rule in rules if rule.wrong_retained <= maximum_wrong]
    eligible.sort(key=_shortlist_key)
    return eligible[:SHORTLIST_LIMIT]


def _dominates(left: CalibrationRuleResult, right: CalibrationRuleResult) -> bool:
    no_worse = (
        left.wrong_retained <= right.wrong_retained
        and left.uncertain_retained <= right.uncertain_retained
        and left.rule_complexity <= right.rule_complexity
        and left.correct_retained >= right.correct_retained
    )
    strictly_better = (
        left.wrong_retained < right.wrong_retained
        or left.uncertain_retained < right.uncertain_retained
        or left.rule_complexity < right.rule_complexity
        or left.correct_retained > right.correct_retained
    )
    return no_worse and strictly_better


def _pareto_frontier(
    rules: Sequence[CalibrationRuleResult],
) -> list[CalibrationRuleResult]:
    representatives: dict[tuple[int, int, int, int], CalibrationRuleResult] = {}
    for rule in sorted(rules, key=lambda item: item.rule):
        objective = (
            rule.wrong_retained,
            rule.uncertain_retained,
            rule.rule_complexity,
            rule.correct_retained,
        )
        representatives.setdefault(objective, rule)
    candidates = list(representatives.values())
    frontier = [
        rule
        for rule in candidates
        if not any(_dominates(other, rule) for other in candidates if other != rule)
    ]
    return sorted(
        frontier,
        key=lambda item: (
            item.wrong_retained,
            item.uncertain_retained,
            -item.correct_retained,
            item.rule_complexity,
            item.rule,
        ),
    )


def _record_diagnostic(
    record: ASRV2SegmentRecord, diagnostic: DiagnosticName
) -> float | None:
    if diagnostic == "duration_seconds":
        return record.end_time - record.start_time
    diagnostics = record.diagnostics
    return getattr(diagnostics, diagnostic) if diagnostics is not None else None


def _production_rule_passes(
    record: ASRV2SegmentRecord, conditions: Sequence[CalibrationCondition]
) -> bool:
    if not raw_text_available(record):
        return False
    for condition in conditions:
        value = _record_diagnostic(record, condition.diagnostic)
        if value is None:
            return False
        if condition.operator == "ge" and value < condition.threshold:
            return False
        if condition.operator == "le" and value > condition.threshold:
            return False
    return True


def _production_shadow(
    *, records: Sequence[ASRV2SegmentRecord], rule: CalibrationRuleResult
) -> ProductionShadowResult:
    raw = [record for record in records if raw_text_available(record)]
    retained = [
        record for record in raw if _production_rule_passes(record, rule.conditions)
    ]
    retained_keys = {(record.target_clip_uid, record.segment_id) for record in retained}
    hidden = [
        record
        for record in raw
        if (record.target_clip_uid, record.segment_id) not in retained_keys
    ]

    def counts(
        values: Sequence[ASRV2SegmentRecord], field: str, allowed: Sequence[str]
    ) -> dict[str, int]:
        counter = Counter(getattr(value, field) for value in values)
        return {key: counter[key] for key in allowed}

    statuses = ("candidate_mapped", "ambiguous", "unbound", "conflict")
    scopes = ("direct_anchor_present", "cluster_propagated_only", "unresolved")
    return ProductionShadowResult(
        rule_id=rule.rule_id,
        rule=rule.rule,
        production_segment_count=len(records),
        production_raw_text_available_count=len(raw),
        production_retained_count=len(retained),
        production_hidden_count=len(hidden),
        retained_by_cluster_binding_status=counts(
            retained, "cluster_binding_status", statuses
        ),
        hidden_by_cluster_binding_status=counts(
            hidden, "cluster_binding_status", statuses
        ),
        retained_by_identity_scope=counts(retained, "identity_scope", scopes),
        hidden_by_identity_scope=counts(hidden, "identity_scope", scopes),
    )


def build_calibration_sweep(
    *,
    joined: Sequence[TextCalibrationSegment],
    production_records: Sequence[ASRV2SegmentRecord],
) -> tuple[TextCalibrationSweep, dict[str, dict[str, DiagnosticDistribution]]]:
    finite_ranges: dict[DiagnosticName, float] = {}
    grouped_conditions: dict[DiagnosticName, list[CalibrationCondition]] = {}
    for diagnostic in DIAGNOSTIC_ORDER:
        values = [
            float(value)
            for row in joined
            if row.raw_text_available
            and (value := getattr(row, diagnostic)) is not None
        ]
        finite_ranges[diagnostic] = max(values) - min(values) if values else 0.0
        grouped_conditions[diagnostic] = _conditions_for_diagnostic(joined, diagnostic)

    baseline = _evaluate_rule(rows=joined, conditions=[], ranges=finite_ranges)
    one_dimensional = [
        _evaluate_rule(rows=joined, conditions=[condition], ranges=finite_ranges)
        for diagnostic in DIAGNOSTIC_ORDER
        for condition in grouped_conditions[diagnostic]
    ]
    two_condition: list[CalibrationRuleResult] = []
    for left_index, left_name in enumerate(DIAGNOSTIC_ORDER):
        for right_name in DIAGNOSTIC_ORDER[left_index + 1 :]:
            for left in _bounded_conditions(grouped_conditions[left_name]):
                for right in _bounded_conditions(grouped_conditions[right_name]):
                    two_condition.append(
                        _evaluate_rule(
                            rows=joined,
                            conditions=[left, right],
                            ranges=finite_ranges,
                        )
                    )
    all_rules = [baseline, *one_dimensional, *two_condition]
    zero_wrong = _shortlist(all_rules, maximum_wrong=0)
    one_wrong = _shortlist(all_rules, maximum_wrong=1)
    pareto = _pareto_frontier(all_rules)
    shadow_rules = {rule.rule_id: rule for rule in [*zero_wrong, *one_wrong]}
    shadows = [
        _production_shadow(records=production_records, rule=shadow_rules[rule_id])
        for rule_id in sorted(shadow_rules)
    ]
    return (
        TextCalibrationSweep(
            baseline=baseline,
            one_dimensional_rules=one_dimensional,
            two_condition_rules=two_condition,
            zero_wrong_shortlist=zero_wrong,
            at_most_one_wrong_shortlist=one_wrong,
            pareto_frontier=pareto,
            production_shadow=shadows,
        ),
        _diagnostic_distributions(joined),
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _rule_rows(rules: Sequence[CalibrationRuleResult]) -> list[list[object]]:
    return [
        [
            item.rule,
            item.correct_retained,
            item.wrong_retained,
            item.uncertain_retained,
            item.correct_retention,
            item.explicit_precision,
            item.conservative_precision,
            item.empirical_wrong_margin,
        ]
        for item in rules
    ]


def _report_html(
    *,
    summary: TextCalibrationSummary,
    joined: Sequence[TextCalibrationSegment],
    sweep: TextCalibrationSweep,
) -> str:
    distribution_rows: list[list[object]] = []
    for diagnostic, labels in summary.diagnostic_distributions.items():
        for label, item in labels.items():
            distribution_rows.append(
                [
                    diagnostic,
                    label,
                    item.count,
                    item.missing_count,
                    item.minimum,
                    item.p10,
                    item.p25,
                    item.median,
                    item.p75,
                    item.p90,
                    item.maximum,
                ]
            )
    shortlist_rules = {
        rule.rule_id: rule
        for rule in [
            *sweep.zero_wrong_shortlist,
            *sweep.at_most_one_wrong_shortlist,
        ]
    }
    hard_cases = [row for row in joined if row.qa_label != "CORRECT"]
    hard_rows = []
    for row in hard_cases:
        decisions = "; ".join(
            f"{rule.rule_id}={'KEEP' if _rule_passes(row, rule.conditions) else 'HIDE'}"
            for rule in shortlist_rules.values()
        )
        hard_rows.append(
            [
                row.target_clip_uid,
                row.segment_id,
                row.speaker_cluster_id,
                row.duration_seconds,
                row.text,
                row.qa_label,
                row.language_probability,
                row.avg_log_probability,
                row.no_speech_probability,
                row.compression_ratio,
                row.identity_scope,
                decisions,
            ]
        )
    shadow_rows = [
        [
            row.rule,
            row.production_raw_text_available_count,
            row.production_retained_count,
            row.production_hidden_count,
            row.retained_by_cluster_binding_status,
            row.hidden_by_cluster_binding_status,
            row.retained_by_identity_scope,
            row.hidden_by_identity_scope,
        ]
        for row in sweep.production_shadow
    ]
    rule_headers = (
        "rule",
        "correct keep",
        "wrong keep",
        "uncertain keep",
        "correct retention",
        "explicit precision",
        "conservative precision",
        "wrong margin",
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 ASR V2 Text Usability Calibration</title><style>
body{{margin:0;background:#f6f7f9;color:#202124;font:14px system-ui,sans-serif}}header{{background:#fff;border-bottom:1px solid #d9dce1;padding:20px 28px}}main{{max-width:1440px;margin:auto;padding:22px}}section{{margin:0 0 26px}}h1{{font-size:24px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 10px}}.notice{{border-left:4px solid #b42318;background:#fff;padding:12px}}.stats{{display:flex;gap:10px;flex-wrap:wrap}}.stat{{background:#fff;border:1px solid #d9dce1;padding:12px;min-width:130px}}.scroll{{overflow:auto;background:#fff;border:1px solid #d9dce1}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #e8eaed;text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#f1f3f4;position:sticky;top:0}}td:last-child{{white-space:normal;min-width:240px}}code{{background:#eceff3;padding:2px 4px}}
</style></head><body><header><h1>H3 ASR V2 Text Usability Calibration</h1><div>Model-free diagnostic analysis. Every rule is <strong>{CALIBRATION_CANDIDATE_LABEL}</strong>.</div></header><main>
<section class=notice><strong>No policy is frozen.</strong> Raw ASR remains preserved. Transcript usability is independent from speaker identity and voice-reference quality. <code>language_probability</code> is not transcript correctness probability.</section>
<section><h2>1. Calibration snapshot</h2><div class=stats><div class=stat>Reviewed<br><strong>{summary.pilot_segment_count}</strong></div><div class=stat>CORRECT<br><strong>{summary.qa_counts.CORRECT}</strong></div><div class=stat>WRONG<br><strong>{summary.qa_counts.WRONG}</strong></div><div class=stat>UNCERTAIN<br><strong>{summary.qa_counts.UNCERTAIN}</strong></div></div></section>
<section><h2>2. Raw-text baseline</h2>{_table(rule_headers, _rule_rows([sweep.baseline]))}</section>
<section><h2>3. Diagnostic distributions by human QA</h2>{_table(("diagnostic", "label", "count", "missing", "min", "p10", "p25", "median", "p75", "p90", "max"), distribution_rows)}</section>
<section><h2>4. One-dimensional sweep results</h2>{_table(rule_headers, _rule_rows(sweep.one_dimensional_rules))}</section>
<section><h2>5. Two-condition candidate results</h2>{_table(rule_headers, _rule_rows(sweep.two_condition_rules))}</section>
<section><h2>6. Zero-WRONG shortlist</h2>{_table(rule_headers, _rule_rows(sweep.zero_wrong_shortlist))}</section>
<section><h2>7. At-most-one-WRONG shortlist</h2>{_table(rule_headers, _rule_rows(sweep.at_most_one_wrong_shortlist))}</section>
<section><h2>8. Pareto frontier</h2>{_table(rule_headers, _rule_rows(sweep.pareto_frontier))}</section>
<section><h2>9. All WRONG / UNCERTAIN hard cases</h2>{_table(("clip", "segment", "speaker", "duration", "transcript", "QA", "language p", "avg log p", "no speech p", "compression", "identity scope", "shortlist decisions"), hard_rows)}</section>
<section><h2>10. Production shadow coverage</h2><p>Coverage only; production has no human correctness labels.</p>{_table(("rule", "raw available", "retained", "hidden", "retained binding", "hidden binding", "retained scope", "hidden scope"), shadow_rows)}</section>
</main></body></html>"""


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"ASR V2 text calibration output exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def analyze_asr_v2_text_usability(
    *,
    audio_run_root: Path,
    qa_json: Path,
    overwrite: bool = False,
    expected_pilot_inventory_fingerprint: str = ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT,
    before_publish: Callable[[], None] | None = None,
) -> TextCalibrationSummary:
    root = audio_run_root.expanduser().resolve(strict=True)
    pilot_root = (root / CALIBRATION_PILOT_DIRECTORY).resolve(strict=True)
    production_root = (root / CALIBRATION_PRODUCTION_DIRECTORY).resolve(strict=True)
    output_path = root / CALIBRATION_OUTPUT_DIRECTORY
    if output_path.is_symlink():
        raise ValueError("ASR V2 text calibration output cannot be a symlink")
    destination = output_path.resolve(strict=False)
    if destination.parent != root:
        raise ValueError("ASR V2 text calibration output must remain under run root")
    qa_path = qa_json.expanduser().resolve(strict=True)

    pilot_inventory_path = pilot_root / "inventory.json"
    pilot_segments_path = pilot_root / "segments.jsonl"
    production_inventory_path = production_root / "inventory.json"
    production_segments_path = production_root / "segments.jsonl"
    pilot_inventory = _validate_inventory(pilot_inventory_path, expected_mode="pilot20")
    if pilot_inventory.inventory_fingerprint != expected_pilot_inventory_fingerprint:
        raise ValueError("ASR V2 pilot inventory is not the frozen calibration input")
    pilot_records = _load_records(root=pilot_root, inventory=pilot_inventory)
    production_inventory = _validate_inventory(
        production_inventory_path, expected_mode="production"
    )
    production_records = _load_records(
        root=production_root, inventory=production_inventory
    )
    expected_counts = ASRHumanQACounts(
        CORRECT=ASR_V2_CALIBRATION_HUMAN_QA_CORRECT,
        WRONG=ASR_V2_CALIBRATION_HUMAN_QA_WRONG,
        UNCERTAIN=ASR_V2_CALIBRATION_HUMAN_QA_UNCERTAIN,
        UNLABELED=ASR_V2_CALIBRATION_HUMAN_QA_UNLABELED,
    )
    qa, qa_bytes = _validate_qa(
        qa_path=qa_path,
        expected_inventory_fingerprint=expected_pilot_inventory_fingerprint,
        expected_counts=expected_counts,
    )
    joined = _join_segments(records=pilot_records, qa=qa)
    if len(joined) != ASR_V2_CALIBRATION_HUMAN_QA_TOTAL:
        raise ValueError("ASR V2 calibration must join exactly 50 pilot segments")
    sweep, distributions = build_calibration_sweep(
        joined=joined, production_records=production_records
    )

    inventory_values = {
        "schema_version": CALIBRATION_INVENTORY_SCHEMA_VERSION,
        "output_root": str(destination),
        "pilot_root": str(pilot_root),
        "pilot_inventory_path": str(pilot_inventory_path),
        "pilot_inventory_sha256": _sha256_file(pilot_inventory_path),
        "pilot_inventory_fingerprint": pilot_inventory.inventory_fingerprint,
        "pilot_segments_path": str(pilot_segments_path),
        "pilot_segments_sha256": _sha256_file(pilot_segments_path),
        "production_root": str(production_root),
        "production_inventory_path": str(production_inventory_path),
        "production_inventory_sha256": _sha256_file(production_inventory_path),
        "production_inventory_fingerprint": production_inventory.inventory_fingerprint,
        "production_segments_path": str(production_segments_path),
        "production_segments_sha256": _sha256_file(production_segments_path),
        "human_qa_source_path": str(qa_path),
        "human_qa_sha256": _sha256_bytes(qa_bytes),
        "pilot_segment_count": len(joined),
        "production_segment_count": len(production_records),
        "analysis_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "text_usability_policy_validated": False,
        "text_usability_gate_applied": False,
        "transcript_confidence_threshold_used": False,
        "production_asr_modified": False,
        "diarization_modified": False,
        "voice_pair_embedding_modified": False,
        "language_probability_is_transcript_correctness_probability": False,
    }
    inventory = TextCalibrationInventory(
        **inventory_values,
        inventory_fingerprint=_sha256_bytes(_compact_json(inventory_values).encode()),
    )
    hard_case_count = sum(row.qa_label != "CORRECT" for row in joined)
    summary = TextCalibrationSummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        pilot_segment_count=len(joined),
        production_segment_count=len(production_records),
        qa_counts=qa.counts,
        raw_text_available_count=sum(row.raw_text_available for row in joined),
        diagnostic_distributions=distributions,
        baseline=sweep.baseline,
        one_dimensional_rule_count=len(sweep.one_dimensional_rules),
        two_condition_rule_count=len(sweep.two_condition_rules),
        zero_wrong_shortlist_count=len(sweep.zero_wrong_shortlist),
        at_most_one_wrong_shortlist_count=len(sweep.at_most_one_wrong_shortlist),
        pareto_frontier_count=len(sweep.pareto_frontier),
        hard_case_count=hard_case_count,
        production_shadow_rule_count=len(sweep.production_shadow),
    )

    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        (temporary / "human_qa.json").write_bytes(qa_bytes)
        _write_jsonl(temporary / "joined_segments.jsonl", joined)
        _write_json(temporary / "sweep.json", sweep.model_dump(mode="json"))
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "report.html").write_text(
            _report_html(summary=summary, joined=joined, sweep=sweep),
            encoding="utf-8",
        )
        if before_publish is not None:
            before_publish()
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
