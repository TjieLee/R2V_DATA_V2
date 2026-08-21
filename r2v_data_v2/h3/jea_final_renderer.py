from __future__ import annotations

import html
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_binding import BoundDiarizationSegment
from r2v_data_v2.h3.jea_audio_production import JEACrossPair, JEAInPair
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.visual_production_source import VisualProductionInventory
from r2v_data_v2.v3.production_export import ProductionReference

FINAL_SAMPLE_VERSION = "r2v.h3.final_sample.2"


class FinalVisualReference(SchemaModel):
    image_id: str
    image_index: int = Field(gt=0)
    kind: Literal["subject", "object", "group", "background", "attribute"]
    image_path: str
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None
    source_frame_index: int = Field(ge=0)
    scope: str | None = None
    visible_region: dict[str, object] | None = None
    synthetic: bool

    @classmethod
    def from_production(cls, value: ProductionReference) -> FinalVisualReference:
        return cls.model_validate(value.model_dump(mode="json"))


class FinalSubjectVoice(SchemaModel):
    subject_index: int = Field(gt=0)
    entity_id: str
    target_occurrence_id: str
    voice_reference_path: str
    voice_source: Literal["target", "cross_donor"]
    donor_occurrence_id: str | None = None
    donor_clip_uid: str | None = None
    donor_clip_display_path: str | None = None

    @model_validator(mode="after")
    def validate_voice(self) -> FinalSubjectVoice:
        donor_values = (
            self.donor_occurrence_id,
            self.donor_clip_uid,
            self.donor_clip_display_path,
        )
        if self.voice_source == "target" and any(
            value is not None for value in donor_values
        ):
            raise ValueError("in-pair voice cannot publish donor provenance")
        if self.voice_source == "cross_donor" and any(
            value is None for value in donor_values
        ):
            raise ValueError("cross voice requires complete donor provenance")
        return self


class FinalQwen3SpeechSegment(SchemaModel):
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str
    language: str | None = None
    asr_model: Literal["Qwen/Qwen3-ASR-1.7B"] = "Qwen/Qwen3-ASR-1.7B"


class FinalH3SampleV2(SchemaModel):
    schema_version: Literal["r2v.h3.final_sample.2"] = FINAL_SAMPLE_VERSION
    sample_id: str
    pair_id: str
    pair_type: Literal["in_pair", "cross_pair"]
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video: str
    target_full_audio_path: str
    r2v_instruction: str
    visual_references: list[FinalVisualReference]
    subject_voices: list[FinalSubjectVoice]
    speech_segments: list[FinalQwen3SpeechSegment]

    @model_validator(mode="after")
    def validate_sample(self) -> FinalH3SampleV2:
        indexes = [item.image_index for item in self.visual_references]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("final Visual references must preserve canonical order")
        subject_ids = {
            item.entity_id for item in self.visual_references if item.kind == "subject"
        }
        if {item.entity_id for item in self.subject_voices} != subject_ids:
            raise ValueError(
                "only canonical subject references may receive voice binding"
            )
        if any(not item.text.strip() for item in self.speech_segments):
            raise ValueError("final Qwen3 speech segments must be non-empty")
        return self


class FinalH3SummaryV2(SchemaModel):
    schema_version: Literal["r2v.h3.final_summary.2"] = "r2v.h3.final_summary.2"
    canonical_clip_count: int = Field(ge=0)
    in_pair_sample_count: int = Field(ge=0)
    cross_pair_sample_count: int = Field(ge=0)
    final_sample_count: int = Field(ge=0)
    speech_segment_count: int = Field(ge=0)
    visual_reference_kind_counts: dict[str, int]
    asr_model: Literal["Qwen/Qwen3-ASR-1.7B"] = "Qwen/Qwen3-ASR-1.7B"
    whisper_rows_consumed: Literal[0] = 0
    language_probability_gate_applied: Literal[False] = False
    dots3_used: Literal[False] = False


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target_voices(pair: JEAInPair) -> list[FinalSubjectVoice]:
    return [
        FinalSubjectVoice(
            subject_index=item.subject_index,
            entity_id=item.target_entity_id,
            target_occurrence_id=item.target_occurrence_id,
            voice_reference_path=item.target_primary_voice_reference_path,
            voice_source="target",
        )
        for item in pair.subjects
    ]


def _cross_voices(pair: JEAInPair, cross: JEACrossPair) -> list[FinalSubjectVoice]:
    entity_by_occurrence = {
        item.target_occurrence_id: item.target_entity_id for item in pair.subjects
    }
    return [
        FinalSubjectVoice(
            subject_index=item.subject_index,
            entity_id=entity_by_occurrence[item.target_occurrence_id],
            target_occurrence_id=item.target_occurrence_id,
            voice_reference_path=item.donor_primary_voice_reference_path,
            voice_source="cross_donor",
            donor_occurrence_id=item.donor_occurrence_id,
            donor_clip_uid=item.donor_clip_uid,
            donor_clip_display_path=item.donor_clip_display_path,
        )
        for item in cross.mappings
    ]


def _speech(value: Qwen3ASRSegment) -> FinalQwen3SpeechSegment:
    assert value.text is not None
    return FinalQwen3SpeechSegment(
        segment_id=value.segment_id,
        speaker_cluster_id=value.speaker_cluster_id,
        entity_id=value.entity_id,
        entity_occurrence_id=value.entity_occurrence_id,
        source_start_sample=value.source_start_sample,
        source_end_sample=value.source_end_sample,
        source_sample_rate_hz=value.source_sample_rate_hz,
        start_time=value.start_time,
        end_time=value.end_time,
        text=value.text,
        language=value.language,
    )


def render_jea_final_samples(
    *,
    visual_inventory: VisualProductionInventory,
    pairs_root: Path,
    diarization_root: Path,
    qwen3_asr_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> FinalH3SummaryV2:
    pairs = pairs_root.expanduser().resolve(strict=True)
    in_pairs = [
        JEAInPair.model_validate(row) for row in _read_rows(pairs / "in_pairs.jsonl")
    ]
    cross_pairs = [
        JEACrossPair.model_validate(row)
        for row in _read_rows(pairs / "cross_pairs.jsonl")
    ]
    cross_by_clip = {item.target_clip_uid: item for item in cross_pairs}
    if len(cross_by_clip) != len(cross_pairs):
        raise ValueError("more than one cross pair was published for a target clip")

    diarization = diarization_root.expanduser().resolve(strict=True)
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read_rows(diarization / "bound_segments.jsonl")
    ]
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    qwen_rows = [
        Qwen3ASRSegment.model_validate(row)
        for row in _read_rows(
            qwen3_asr_root.expanduser().resolve(strict=True) / "segments.jsonl"
        )
    ]
    for row in qwen_rows:
        source = bound_by_key.get((row.clip_uid, row.segment_id))
        if source is None or (
            source.source_start_sample,
            source.source_end_sample,
            source.speaker_cluster_id,
            source.entity_id,
        ) != (
            row.source_start_sample,
            row.source_end_sample,
            row.speaker_cluster_id,
            row.entity_id,
        ):
            raise ValueError("Qwen3 row differs from its exact DiariZen segment")
    speech_by_clip: dict[str, list[FinalQwen3SpeechSegment]] = {}
    for row in qwen_rows:
        if row.status != "transcribed":
            continue
        speech_by_clip.setdefault(row.clip_uid, []).append(_speech(row))
    for rows in speech_by_clip.values():
        rows.sort(key=lambda item: (item.source_start_sample, item.segment_id))

    visual_by_clip = {item.identity.clip_uid: item for item in visual_inventory.clips}
    samples: list[FinalH3SampleV2] = []
    for pair in sorted(in_pairs, key=lambda item: item.target_clip_display_path):
        visual = visual_by_clip[pair.target_clip_uid]
        references = [
            FinalVisualReference.from_production(item)
            for item in visual.sample.references
        ]
        common = {
            **visual.identity.model_dump(mode="python", exclude={"clip_uid"}),
            "clip_uid": pair.target_clip_uid,
            "target_video": visual.sample.target_video,
            "target_full_audio_path": pair.target_full_audio_path,
            "r2v_instruction": visual.sample.r2v_instruction,
            "visual_references": references,
            "speech_segments": speech_by_clip.get(pair.target_clip_uid, []),
        }
        samples.append(
            FinalH3SampleV2(
                sample_id=f"{visual.sample.sample_id}/in_pair",
                pair_id=pair.pair_id,
                pair_type="in_pair",
                subject_voices=_target_voices(pair),
                **common,
            )
        )
        cross = cross_by_clip.get(pair.target_clip_uid)
        if cross is not None:
            samples.append(
                FinalH3SampleV2(
                    sample_id=f"{visual.sample.sample_id}/cross_pair/1",
                    pair_id=cross.pair_id,
                    pair_type="cross_pair",
                    subject_voices=_cross_voices(pair, cross),
                    **common,
                )
            )

    kind_counts = Counter(
        reference.kind for sample in samples for reference in sample.visual_references
    )
    summary = FinalH3SummaryV2(
        canonical_clip_count=visual_inventory.canonical_sample_count,
        in_pair_sample_count=sum(item.pair_type == "in_pair" for item in samples),
        cross_pair_sample_count=sum(item.pair_type == "cross_pair" for item in samples),
        final_sample_count=len(samples),
        speech_segment_count=sum(len(item.speech_segments) for item in samples),
        visual_reference_kind_counts=dict(sorted(kind_counts.items())),
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"final H3 output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "samples.jsonl").write_text(
            "".join(
                json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for item in samples
            ),
            encoding="utf-8",
        )
        for sample in samples:
            relative = PurePosixPath(sample.clip_display_path)
            name = (
                "in_pair.json" if sample.pair_type == "in_pair" else "cross_pair_1.json"
            )
            path = temporary / "samples" / Path(*relative.parts) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sample.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        rows = "".join(
            f"<tr><td>{html.escape(item.clip_display_path)}</td><td>{item.pair_type}</td><td>{len(item.visual_references)}</td><td>{len(item.speech_segments)}</td></tr>"
            for item in samples
        )
        (temporary / "review.html").write_text(
            "<!doctype html><meta charset='utf-8'><table><tr><th>clip</th><th>pair</th><th>visual refs</th><th>speech</th></tr>"
            + rows
            + "</table>",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
