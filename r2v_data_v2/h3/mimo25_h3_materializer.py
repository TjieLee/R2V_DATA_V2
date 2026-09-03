from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalQwen3SpeechSegment,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoInventory,
    MimoRecord,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MATERIALIZER_VERSION,
    MimoSegmentDecision,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    UNGROUNDED_NON_DIEGETIC_MUSIC,
    UNGROUNDED_OVERALL_SOUNDSCAPE,
    AudioFactAuditItem,
    ConditioningVariant,
    Qwen38DraftShot,
    Qwen38H3DraftResponse,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRequest,
    RecaptionAudioFacts,
    RecaptionNonSpeechFact,
    RecaptionSpeechFact,
    build_reference_contract,
    materialize_h3_draft,
    render_h3_prompt,
    validate_h3_response,
)
from r2v_data_v2.h3.schemas import SchemaModel

MIMO25_SHADOW_RECORD_VERSION = "r2v.h3.mimo25_h3_shadow.4"
MIMO25_SHADOW_SUMMARY_VERSION = "r2v.h3.mimo25_h3_shadow_summary.4"
_MIMO_SEGMENT_PLACEHOLDER = re.compile(r"\[\[segment:([^\[\]\r\n]+)\]\]")
_MIMO_AUDIO_EVENT_PLACEHOLDER = re.compile(
    r"\[\[audio_event:([^\[\]\r\n]+)\]\]"
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(item.model_dump(mode="json")) + "\n" for item in values),
        encoding="utf-8",
    )


class MimoH3ShadowRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_h3_shadow.4"] = (
        MIMO25_SHADOW_RECORD_VERSION
    )
    sample_id: str
    clip_uid: str
    pair_type: Literal["canonical", "in_pair", "cross_pair"]
    status: Literal["ready", "failed"]
    source_mimo_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materializer_version: Literal["h3_mimo25_materializer_v4"] = (
        MIMO25_MATERIALIZER_VERSION
    )
    corrected_speech_segments: list[FinalQwen3SpeechSegment]
    rendered_h3_prompt: str | None = None
    warnings: list[str]
    failure_reason: str | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoH3ShadowRecord:
        if self.status == "ready":
            if self.rendered_h3_prompt is None or self.failure_reason is not None:
                raise ValueError("ready MiMo H3 shadow record is incomplete")
        elif self.rendered_h3_prompt is not None or not self.failure_reason:
            raise ValueError("failed MiMo H3 shadow record requires only failure")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo H3 shadow record fingerprint is invalid")
        return self


class MimoH3ShadowSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_h3_shadow_summary.4"] = (
        MIMO25_SHADOW_SUMMARY_VERSION
    )
    source_mimo_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_mimo_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    warning_counts: dict[str, int]
    target_clip_annotation_reuse_count: int = Field(ge=0)
    asr_text_modified: Literal[False] = False
    asr_language_modified: Literal[False] = False
    source_segments_deleted: Literal[False] = False
    production_h3_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoH3ShadowSummary:
        if self.sample_count != self.ready_count + self.failed_count:
            raise ValueError("MiMo H3 shadow summary counts must reconcile")
        return self


def _variant(sample: FinalH3SampleV2) -> ConditioningVariant:
    return {
        "canonical": "visual_only",
        "in_pair": "target_voice_reference",
        "cross_pair": "cross_voice_reference",
    }[sample.pair_type]


def _corrected_segments(
    sample: FinalH3SampleV2,
    job: MimoClipJob,
    record: MimoRecord,
) -> tuple[list[FinalQwen3SpeechSegment], list[str]]:
    assert record.annotation is not None
    decisions = {item.segment_id: item for item in record.annotation.segment_decisions}
    source_by_id = {item.segment_id: item for item in job.segments}
    sample_by_id = {item.segment_id: item for item in sample.speech_segments}
    transcribed_ids = [item.segment_id for item in job.segments if item.asr_status == "transcribed"]
    if set(sample_by_id) != set(transcribed_ids):
        raise ValueError("source H3 speech inventory differs from authoritative ASR")
    corrected: list[FinalQwen3SpeechSegment] = []
    warnings: list[str] = []
    for segment_id in transcribed_ids:
        source = source_by_id[segment_id]
        original = sample_by_id[segment_id]
        decision = decisions[segment_id]
        if (
            original.text != source.asr_text
            or original.language != source.asr_language
            or original.start_time != source.start_time
            or original.end_time != source.end_time
            or original.source_start_sample != source.source_start_sample
            or original.source_end_sample != source.source_end_sample
            or original.source_sample_rate_hz != source.source_sample_rate_hz
            or original.speaker_cluster_id != source.source_speaker_cluster_id
        ):
            raise ValueError("source H3 speech differs from authoritative Qwen3-ASR")
        resolved = decision.resolution == "resolved" and decision.primary_speaker_group is not None
        group = (
            decision.primary_speaker_group
            if resolved
            else f"fallback__{source.source_speaker_cluster_id}"
        )
        entity_id = (
            decision.entity_id
            if resolved
            and decision.binding_status == "visible_entity"
            and decision.speech_presentation == "onscreen_spoken"
            else None
        )
        if not resolved:
            warnings.append(f"{segment_id}:acoustic_refinement_unresolved")
        corrected.append(
            FinalQwen3SpeechSegment(
                segment_id=segment_id,
                speaker_cluster_id=group,
                entity_id=entity_id,
                entity_occurrence_id=(
                    None if entity_id is None else f"{sample.clip_uid}/{entity_id}"
                ),
                source_start_sample=source.source_start_sample,
                source_end_sample=source.source_end_sample,
                source_sample_rate_hz=source.source_sample_rate_hz,
                start_time=source.start_time,
                end_time=source.end_time,
                text=source.asr_text or "",
                language=source.asr_language,
            )
        )
    return corrected, warnings


def _speaker_ids(segments: Sequence[FinalQwen3SpeechSegment]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in segments:
        if segment.speaker_cluster_id not in result:
            result[segment.speaker_cluster_id] = f"S{len(result) + 1}"
    return result


def _audio_facts(
    *,
    sample: FinalH3SampleV2,
    corrected: list[FinalQwen3SpeechSegment],
    record: MimoRecord,
    contract: object,
) -> RecaptionAudioFacts:
    from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionReferenceContract

    assert record.annotation is not None
    typed_contract = RecaptionReferenceContract.model_validate(contract)
    entity_subject = {
        item.entity_id: item.subject_label
        for item in typed_contract.subjects
        if item.kind == "entity" and item.entity_id is not None
    }
    speaker_ids = _speaker_ids(corrected)
    delivery_by_group = {
        item.speaker_group: item.delivery_style
        for item in record.annotation.audio_semantics.speaker_delivery
    }
    speech = []
    for segment in corrected:
        language = segment.language or "Unknown"
        speech.append(
            RecaptionSpeechFact(
                fact_id=segment.segment_id,
                segment_id=segment.segment_id,
                speaker_cluster_id=segment.speaker_cluster_id,
                entity_id=segment.entity_id,
                entity_subject_label=(
                    None if segment.entity_id is None else entity_subject.get(segment.entity_id)
                ),
                speaker_id=speaker_ids[segment.speaker_cluster_id],
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                language=language,
                delivery=delivery_by_group.get(segment.speaker_cluster_id),
                locked_dialogue_block=f"<d>[{language}] {segment.text}</d>",
            )
        )
    events = [
        RecaptionNonSpeechFact(
            fact_id=item.event_id,
            start_time=item.approximate_start_time,
            end_time=item.approximate_end_time,
            category=item.category,
            description=item.description,
            source_attribution=item.source_grounding,
            provenance="mimo25.audio_semantics.temporal_non_speech_events",
        )
        for item in record.annotation.audio_semantics.temporal_non_speech_events
    ]
    semantics = record.annotation.audio_semantics
    return RecaptionAudioFacts(
        speech=speech,
        non_speech_events=events,
        overall_soundscape_hint=semantics.overall_soundscape,
        non_diegetic_music_hint=semantics.non_diegetic_music,
        audio_grounding_complete=semantics.non_diegetic_music_status != "unknown",
        provenance={
            "speech": "qwen3_asr_segments_exact",
            "speaker_structure": "mimo25_shadow_annotation",
            "audio_semantics": "mimo25_shadow_annotation",
            "source_sample_id": sample.sample_id,
        },
    )


def _render_mimo_speech_clause(
    speech: RecaptionSpeechFact,
    base_clause: str,
    decisions: dict[str, MimoSegmentDecision],
) -> str:
    presentation = decisions[speech.segment_id].speech_presentation
    if presentation == "onscreen_spoken":
        return base_clause
    prefix = {
        "offscreen_spoken": f"({speech.speaker_id}), speaking offscreen",
        "voice_over": (
            f"({speech.speaker_id}), as a voice-over rather than visible speech"
        ),
        "message_voice_over": (
            f"({speech.speaker_id}), as a message voice-over rather than visible speech"
        ),
        "device_playback": (
            f"({speech.speaker_id}), heard through an in-scene device rather than "
            "visible speech"
        ),
        "uncertain": f"({speech.speaker_id}), with speech presentation uncertain",
    }[presentation]
    return f"{prefix}: {speech.locked_dialogue_block}"


def _materialize_sample(
    sample: FinalH3SampleV2,
    job: MimoClipJob,
    record: MimoRecord,
) -> tuple[list[FinalQwen3SpeechSegment], str, list[str]]:
    assert record.annotation is not None
    corrected, warnings = _corrected_segments(sample, job, record)
    corrected_payload = sample.model_dump(mode="python")
    corrected_payload["speech_segments"] = [item.model_dump(mode="python") for item in corrected]
    corrected_sample = FinalH3SampleV2.model_validate(corrected_payload)
    variant = _variant(corrected_sample)
    contract = build_reference_contract(corrected_sample, variant)
    facts = _audio_facts(
        sample=corrected_sample,
        corrected=corrected,
        record=record,
        contract=contract,
    )
    case = Qwen38RecaptionManifestCase(
        sample_id=sample.sample_id,
        conditioning_variant=variant,
        note="MiMo-V2.5 target-level AV shadow annotation",
    )
    request = Qwen38RecaptionRequest(
        sample=corrected_sample,
        case=case,
        reference_contract=contract,
        audio_facts=facts,
        request_fingerprint=record.request_fingerprint,
    )
    semantics = record.annotation.audio_semantics
    audio_event_by_id = {
        item.event_id: item.description
        for item in semantics.temporal_non_speech_events
    }
    prefix = (
        "[reference generation]"
        if variant == "visual_only"
        else "[reference generation + audio reference]"
    )
    music = (
        semantics.non_diegetic_music
        if semantics.non_diegetic_music_status == "present"
        else "N/A"
        if semantics.non_diegetic_music_status == "absent"
        else UNGROUNDED_NON_DIEGETIC_MUSIC
    )
    draft = Qwen38H3DraftResponse(
        subject_definitions=record.annotation.h3_draft.subject_definitions,
        summary=f"{prefix} {record.annotation.h3_draft.summary}",
        retention_analysis=record.annotation.h3_draft.visual_retention_analysis,
        shots=[
            Qwen38DraftShot(
                shot_index=item.shot_index,
                start_time=item.start_time,
                description_template=_MIMO_SEGMENT_PLACEHOLDER.sub(
                    lambda match: f"[[{match.group(1)}]]",
                    _MIMO_AUDIO_EVENT_PLACEHOLDER.sub(
                        lambda match: audio_event_by_id[match.group(1)],
                        item.description_template,
                    ),
                ),
            )
            for item in record.annotation.h3_draft.shots
        ],
        overall_soundscape=(
            semantics.overall_soundscape
            or UNGROUNDED_OVERALL_SOUNDSCAPE
        ),
        non_diegetic_music=music,
        audio_fact_audit=[
            AudioFactAuditItem(fact_id=item.fact_id, action="preserved")
            for item in facts.non_speech_events
        ],
    )
    decisions = {
        item.segment_id: item for item in record.annotation.segment_decisions
    }
    structured = materialize_h3_draft(
        draft,
        request,
        speech_clause_transform=lambda speech, clause: _render_mimo_speech_clause(
            speech, clause, decisions
        ),
    )
    if "[[audio_event:" in structured.detailed_description:
        raise ValueError("MiMo Audio-event placeholder survived materialization")
    issues, validation_warnings = validate_h3_response(structured, request)
    if issues:
        raise ValueError(
            "MiMo H3 materialization violates reference contract: "
            + _compact_json([item.to_dict() for item in issues])
        )
    warnings.extend(validation_warnings)
    if record.annotation.warnings:
        warnings.extend(
            f"{item.segment_id}:{item.code}"
            for item in record.annotation.warnings
        )
    return corrected, render_h3_prompt(structured), sorted(set(warnings))


def _record(values: dict[str, object]) -> MimoH3ShadowRecord:
    return MimoH3ShadowRecord(
        **values,
        record_fingerprint=_sha256_text(_compact_json(values)),
    )


def _validate_job_media_integrity(job: MimoClipJob) -> None:
    media = (
        (
            Path(job.target_video_path),
            job.target_video_sha256,
            "MiMo target video changed after AV annotation",
        ),
        (
            Path(job.target_full_audio_path),
            job.target_full_audio_sha256,
            "MiMo target full audio changed after AV annotation",
        ),
        *(
            (
                Path(reference.image_artifact_path),
                reference.image_sha256,
                "MiMo frozen reference image changed after AV annotation",
            )
            for reference in job.reference_images
        ),
    )
    for path, expected_sha256, message in media:
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(message)


def _validate_materializer_provenance(
    *,
    inventory: MimoInventory,
    records: Sequence[MimoRecord],
    source_samples: Sequence[FinalH3SampleV2],
) -> None:
    job_by_clip = {item.clip_uid: item for item in inventory.jobs}
    sample_ids_by_clip: dict[str, list[str]] = {
        clip_uid: [] for clip_uid in job_by_clip
    }
    for sample in source_samples:
        if sample.clip_uid in sample_ids_by_clip:
            sample_ids_by_clip[sample.clip_uid].append(sample.sample_id)
    for record in records:
        job = job_by_clip.get(record.clip_uid)
        if record.inventory_fingerprint != inventory.inventory_fingerprint:
            raise ValueError("MiMo record inventory fingerprint mismatch")
        if job is None or record.request_fingerprint != job.request_fingerprint:
            raise ValueError("MiMo record request fingerprint mismatch")
    for job in inventory.jobs:
        if sorted(sample_ids_by_clip[job.clip_uid]) != sorted(
            job.source_h3_sample_ids
        ):
            raise ValueError("MiMo source H3 sample IDs changed after AV annotation")
        _validate_job_media_integrity(job)


def materialize_mimo25_h3_shadow(
    *,
    mimo_root: Path,
    source_h3_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> MimoH3ShadowSummary:
    source_mimo = mimo_root.expanduser().resolve(strict=True)
    source_h3 = source_h3_root.expanduser().resolve(strict=True)
    source_h3_samples_path = source_h3 / "samples.jsonl"
    inventory = MimoInventory.model_validate_json(
        (source_mimo / "inventory.json").read_text(encoding="utf-8")
    )
    if _sha256_file(source_h3_samples_path) != inventory.source_h3_samples_sha256:
        raise ValueError("MiMo source H3 inventory changed after AV annotation")
    mimo_records = [
        MimoRecord.model_validate(row)
        for row in _read_jsonl(source_mimo / "records.jsonl")
    ]
    all_source_samples = [
        FinalH3SampleV2.model_validate(row)
        for row in _read_jsonl(source_h3_samples_path)
    ]
    job_by_clip = {item.clip_uid: item for item in inventory.jobs}
    source_samples = [
        item for item in all_source_samples if item.clip_uid in job_by_clip
    ]
    record_by_clip = {item.clip_uid: item for item in mimo_records}
    if len(record_by_clip) != len(mimo_records):
        raise ValueError("MiMo records contain duplicate target clips")
    if set(record_by_clip) != set(job_by_clip):
        raise ValueError("MiMo records do not exactly cover the selected inventory")
    _validate_materializer_provenance(
        inventory=inventory,
        records=mimo_records,
        source_samples=all_source_samples,
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    records: list[MimoH3ShadowRecord] = []
    warning_counts: Counter[str] = Counter()
    try:
        for sample in source_samples:
            source_hash = _sha256_text(_compact_json(sample.model_dump(mode="json")))
            job = job_by_clip.get(sample.clip_uid)
            mimo_record = record_by_clip.get(sample.clip_uid)
            base = {
                "schema_version": MIMO25_SHADOW_RECORD_VERSION,
                "sample_id": sample.sample_id,
                "clip_uid": sample.clip_uid,
                "pair_type": sample.pair_type,
                "source_mimo_record_fingerprint": (
                    "0" * 64 if mimo_record is None else mimo_record.record_fingerprint
                ),
                "source_h3_sample_sha256": source_hash,
                "materializer_version": MIMO25_MATERIALIZER_VERSION,
            }
            if job is None or mimo_record is None or mimo_record.status != "ready":
                values = {
                    **base,
                    "status": "failed",
                    "corrected_speech_segments": [],
                    "rendered_h3_prompt": None,
                    "warnings": [],
                    "failure_reason": "ready MiMo target annotation is unavailable",
                }
            else:
                try:
                    corrected, rendered, warnings = _materialize_sample(
                        sample, job, mimo_record
                    )
                    warning_counts.update(warnings)
                    values = {
                        **base,
                        "status": "ready",
                        "corrected_speech_segments": [
                            item.model_dump(mode="json") for item in corrected
                        ],
                        "rendered_h3_prompt": rendered,
                        "warnings": warnings,
                        "failure_reason": None,
                    }
                except (OSError, ValueError) as exc:
                    values = {
                        **base,
                        "status": "failed",
                        "corrected_speech_segments": [],
                        "rendered_h3_prompt": None,
                        "warnings": [],
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
            records.append(_record(values))
        _write_jsonl(temporary / "records.jsonl", records)
        summary = MimoH3ShadowSummary(
            source_mimo_inventory_fingerprint=inventory.inventory_fingerprint,
            source_mimo_records_sha256=_sha256_file(source_mimo / "records.jsonl"),
            source_h3_samples_sha256=_sha256_file(source_h3_samples_path),
            sample_count=len(records),
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            warning_counts=dict(sorted(warning_counts.items())),
            target_clip_annotation_reuse_count=sum(
                max(0, count - 1) for count in Counter(item.clip_uid for item in source_samples).values()
            ),
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


__all__ = [
    "MimoH3ShadowRecord",
    "MimoH3ShadowSummary",
    "materialize_mimo25_h3_shadow",
]
