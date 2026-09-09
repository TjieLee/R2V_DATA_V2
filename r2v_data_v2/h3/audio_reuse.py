"""Additive PCM reuse assets; no H3 contract or voice-quality-policy integration."""

from __future__ import annotations

import hashlib
import json
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft, direct_speech_facts
from r2v_data_v2.h3.mimo25_recovered_voice import _semantic_reasons
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    STEM_ALIGNMENT_TOLERANCE_SECONDS,
    SAMAudioStemRecord,
    StemArtifact,
    VerificationState,
    sha256_file,
)
from r2v_data_v2.h3.schemas import SchemaModel

SAMPLE_RATE = 32000
SAMPLE_MAPPING_POLICY = "source_sample_rational_to_32k_round_half_even_v1"


class AudioReuseIntegrityError(ValueError):
    """No assets are published when source or output integrity cannot be proved."""


class ReuseSampleRange(SchemaModel):
    segment_id: str = Field(min_length=1)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    target_start_sample: int = Field(ge=0)
    target_end_sample: int = Field(gt=0)
    sample_mapping_policy: Literal["source_sample_rational_to_32k_round_half_even_v1"] = SAMPLE_MAPPING_POLICY

    @model_validator(mode="after")
    def validate_range(self) -> ReuseSampleRange:
        if self.source_end_sample <= self.source_start_sample or self.end_time <= self.start_time:
            raise ValueError("reuse source range must be positive")
        for source, target in ((self.source_start_sample, self.target_start_sample),
                               (self.source_end_sample, self.target_end_sample)):
            if target != round(Fraction(source * SAMPLE_RATE, self.source_sample_rate_hz)):
                raise ValueError("reuse sample mapping differs from exact rational policy")
        if self.target_end_sample <= self.target_start_sample:
            raise ValueError("reuse target range must be positive")
        return self


class _ReuseAsset(SchemaModel):
    clip_uid: str = Field(min_length=1)
    source_stem_path: str
    source_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_verification_state: VerificationState
    source_stem_frame_count: int = Field(gt=0)
    target_audio_path: str
    target_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_frame_count: int = Field(gt=0)
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    sample_rate_hz: Literal[32000] = SAMPLE_RATE
    channels: Literal[2] = 2
    sample_format: Literal["PCM_16"] = "PCM_16"
    timeline_preserved: Literal[True] = True
    alignment_offset_seconds: Literal[0] = 0
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_extent(self) -> _ReuseAsset:
        if self.target_duration_seconds != self.target_frame_count / SAMPLE_RATE:
            raise ValueError("reuse duration differs from target frame count")
        if any(not Path(path).is_absolute() for path in (
            self.source_stem_path, self.target_audio_path, self.output_path,
        )):
            raise ValueError("reuse media paths must be absolute")
        return self


class SpeakerSpeechReuseAsset(_ReuseAsset):
    schema_version: Literal["r2v.h3.speaker_speech_reuse_asset.1"] = "r2v.h3.speaker_speech_reuse_asset.1"
    source_stem_type: Literal["speech"] = "speech"
    speaker_group: str = Field(pattern=r"^g[1-9]\d*$")
    speaker_id: str = Field(pattern=r"^S[1-9]\d*$")
    entity_id: str | None = None
    subject_index: int | None = Field(default=None, gt=0)
    source_segment_ids: list[str] = Field(min_length=1)
    source_sample_ranges: list[ReuseSampleRange] = Field(min_length=1)
    source_job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_annotation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    silence_fill: Literal[True] = True

    @model_validator(mode="after")
    def validate_speaker(self) -> SpeakerSpeechReuseAsset:
        if (self.entity_id is None) != (self.subject_index is None):
            raise ValueError("reuse entity/Subject provenance must be paired")
        ids = [item.segment_id for item in self.source_sample_ranges]
        if ids != self.source_segment_ids or len(ids) != len(set(ids)):
            raise ValueError("reuse source segments differ or repeat")
        if any(item.target_end_sample > self.target_frame_count
               or item.target_end_sample > self.source_stem_frame_count
               for item in self.source_sample_ranges):
            raise ValueError("reuse speech range exceeds available media")
        return self


class MusicReuseAsset(_ReuseAsset):
    schema_version: Literal["r2v.h3.music_reuse_asset.1"] = "r2v.h3.music_reuse_asset.1"
    source_stem_type: Literal["music"] = "music"
    copied_frame_count: int = Field(gt=0)
    silence_tail_frame_count: int = Field(ge=0)
    truncated_tail_frame_count: int = Field(ge=0)
    semantic_presence_evaluated: Literal[False] = False

    @model_validator(mode="after")
    def validate_music(self) -> MusicReuseAsset:
        copied = min(self.target_frame_count, self.source_stem_frame_count)
        if (self.copied_frame_count, self.silence_tail_frame_count, self.truncated_tail_frame_count) != (
            copied, self.target_frame_count - copied, self.source_stem_frame_count - copied,
        ):
            raise ValueError("music reuse extent diagnostics differ")
        return self


class ReuseExclusion(SchemaModel):
    segment_id: str
    speaker_group: str | None
    reason: Literal[
        "unresolved", "non_single_speaker", "secondary_vocal_activity",
        "cross_speaker_overlap", "speech_range_outside_stem",
        "speech_range_outside_target", "speaker_id_group_inconsistent",
        "conflicting_entity_provenance", "nonpositive_mapped_range",
    ]


class AudioReuseManifest(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_manifest.1"] = "r2v.h3.audio_reuse_manifest.1"
    clip_uid: str
    source_job_fingerprint: str
    source_annotation_sha256: str
    source_stem_record_fingerprint: str
    speakers: list[SpeakerSpeechReuseAsset]
    music: MusicReuseAsset
    exclusions: list[ReuseExclusion]
    production_artifacts_modified: Literal[False] = False
    model_call_count: Literal[0] = 0


def _fingerprint(value: SchemaModel) -> str:
    raw = json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_pcm(path: Path, expected_hash: str) -> np.ndarray:
    import soundfile as sf

    if not path.is_file() or sha256_file(path) != expected_hash:
        raise AudioReuseIntegrityError(f"audio_reuse_source_hash_mismatch: {path}")
    info = sf.info(str(path))
    if (info.samplerate, info.channels, info.subtype) != (SAMPLE_RATE, 2, "PCM_16"):
        raise AudioReuseIntegrityError(f"audio_reuse_requires_32k_stereo_pcm16: {path}")
    pcm, _ = sf.read(str(path), dtype="int16", always_2d=True)
    if not len(pcm) or len(pcm) != info.frames:
        raise AudioReuseIntegrityError(f"audio_reuse_invalid_frame_count: {path}")
    return pcm


def _write_verified(path: Path, pcm: np.ndarray) -> str:
    import soundfile as sf

    sf.write(str(path), pcm, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    digest = sha256_file(path)
    if sf.info(str(path)).format != "FLAC" or not np.array_equal(_read_pcm(path, digest), pcm):
        raise AudioReuseIntegrityError("audio_reuse_decoded_samples_differ")
    return digest


def _check_stem(stem: StemArtifact, target: Path, target_hash: str, frame_count: int) -> np.ndarray:
    if (Path(stem.source_audio_path).resolve() != target
            or stem.source_audio_sha256 != target_hash or stem.source_end_sample != frame_count):
        raise AudioReuseIntegrityError("audio_reuse_stem_target_lineage_mismatch")
    pcm = _read_pcm(Path(stem.canonical_stem_path), stem.canonical_stem_sha256)
    if len(pcm) != stem.canonical_frame_count:
        raise AudioReuseIntegrityError("audio_reuse_stem_frame_count_mismatch")
    if abs(len(pcm) - frame_count) / SAMPLE_RATE > STEM_ALIGNMENT_TOLERANCE_SECONDS:
        raise AudioReuseIntegrityError("audio_reuse_stem_timeline_mismatch")
    return pcm


def build_audio_reuse_assets(
    *, job: MimoClipJob, annotation: MimoAVAnnotationDraft,
    stem_record: SAMAudioStemRecord, audio_production_root: Path,
    output_root: Path, allow_unverified: bool = False,
) -> AudioReuseManifest:
    """Build one clip into a NEW external shadow directory, never overwrite.

    Call with the finalized annotation and its authoritative stem-reconcile job.
    PCM support requires soundfile/libsndfile. No model/runtime construction occurs.
    """
    job = MimoClipJob.model_validate(job.model_dump())
    annotation = MimoAVAnnotationDraft.model_validate(annotation.model_dump())
    stem_record = SAMAudioStemRecord.model_validate(stem_record.model_dump())
    output = output_root.expanduser().resolve()
    production = audio_production_root.expanduser().resolve(strict=True)
    if output.is_relative_to(production) or production.is_relative_to(output):
        raise ValueError("audio reuse output must be separate from the production root")
    if output.exists():
        raise FileExistsError(output)
    if stem_record.clip_uid != job.clip_uid or stem_record.separation_state == "failure":
        raise AudioReuseIntegrityError("audio_reuse_stem_record_unusable")
    if stem_record.separation_state != "success" and not allow_unverified:
        raise ValueError("unverified SAM stems require explicit allow_unverified")
    target = Path(job.target_full_audio_path).resolve(strict=True)
    speech = stem_record.stem("speech")
    music = stem_record.stem("music")
    sources = {target: job.target_full_audio_sha256}
    for stem in (speech, music):
        path = Path(stem.canonical_stem_path).resolve(strict=True)
        if path.is_relative_to(output):
            raise ValueError("audio reuse output cannot contain source media")
        sources[path] = stem.canonical_stem_sha256
    target_pcm = _read_pcm(target, job.target_full_audio_sha256)
    frames = len(target_pcm)
    if abs(frames / SAMPLE_RATE - job.target_duration_seconds) > STEM_ALIGNMENT_TOLERANCE_SECONDS:
        raise AudioReuseIntegrityError("audio_reuse_job_target_timeline_mismatch")
    speech_pcm = _check_stem(speech, target, job.target_full_audio_sha256, frames)
    music_pcm = _check_stem(music, target, job.target_full_audio_sha256, frames)
    ids = [s.segment_id for s in job.segments]
    decisions = annotation.audio_observation.segment_decisions
    groundings = annotation.av_grounding.segment_groundings
    if [d.segment_id for d in decisions] != ids or [g.segment_id for g in groundings] != ids:
        raise AudioReuseIntegrityError("audio_reuse_segment_inventory_mismatch")
    if any(d.primary_speaker_group != g.primary_speaker_group
           for d, g in zip(decisions, groundings, strict=True)):
        raise AudioReuseIntegrityError("audio_reuse_speaker_group_grounding_mismatch")
    facts = {f["segment_id"]: f["speaker_id"] for f in direct_speech_facts(annotation, list(job.segments))}
    annotation_hash = _fingerprint(annotation)
    exclusions: list[ReuseExclusion] = []
    ranges: dict[str, ReuseSampleRange] = {}
    grouped: dict[str, list[str]] = {}
    group_entities: dict[str, set[str]] = {}
    unsafe: set[str] = set()
    blocked_groups: set[str] = set()
    groups = {d.segment_id: d.primary_speaker_group for d in decisions}

    def exclude(segment_id: str, reason: str, *, block_group: bool = False) -> None:
        group = groups[segment_id]
        exclusions.append(ReuseExclusion(segment_id=segment_id, speaker_group=group, reason=reason))
        unsafe.add(segment_id)
        if block_group and group is not None:
            blocked_groups.add(group)

    for segment, decision, grounding in zip(job.segments, decisions, groundings, strict=True):
        group = decision.primary_speaker_group
        # Reuse semantic exclusions, NOT legacy visibility or quality gates.
        ownership_reasons = [r for r in _semantic_reasons(decision, grounding)
                             if r in {"unresolved", "non_single_speaker", "secondary_vocal_activity"}]
        if not ownership_reasons and grounding.binding_status == "visible_entity" and grounding.entity_id is not None:
            group_entities.setdefault(group, set()).add(grounding.entity_id)
        start = round(Fraction(segment.source_start_sample * SAMPLE_RATE, segment.source_sample_rate_hz))
        end = round(Fraction(segment.source_end_sample * SAMPLE_RATE, segment.source_sample_rate_hz))
        if end <= start:
            exclude(segment.segment_id, "nonpositive_mapped_range", block_group=True)
            continue
        ranges[segment.segment_id] = ReuseSampleRange(
            **segment.model_dump(include={"segment_id", "source_start_sample", "source_end_sample", "source_sample_rate_hz", "start_time", "end_time"}),
            target_start_sample=start, target_end_sample=end,
        )
        if segment.asr_status != "transcribed":
            if decision.vocal_composition != "single_speaker" or decision.secondary_vocal_activity.present:
                unsafe.add(segment.segment_id)
            continue
        for reason in ownership_reasons:
            exclude(segment.segment_id, reason)
        if end > frames:
            exclude(segment.segment_id, "speech_range_outside_target", block_group=True)
        if end > len(speech_pcm):
            exclude(segment.segment_id, "speech_range_outside_stem", block_group=True)
        if group is not None:
            grouped.setdefault(group, []).append(segment.segment_id)

    # A shared stem cannot disentangle overlapping speakers (or mixed intervals).
    items = list(ranges.items())
    for index, (left, a) in enumerate(items):
        for right, b in items[index + 1:]:
            if (max(a.target_start_sample, b.target_start_sample) < min(a.target_end_sample, b.target_end_sample)
                    and (groups[left] != groups[right] or left in unsafe or right in unsafe)):
                exclude(left, "cross_speaker_overlap", block_group=True)
                exclude(right, "cross_speaker_overlap", block_group=True)

    speaker_groups_by_id: dict[str, set[str]] = {}
    for group, segment_ids in grouped.items():
        for sid in segment_ids:
            speaker_groups_by_id.setdefault(facts[sid], set()).add(group)
    for group, segment_ids in grouped.items():
        if any(len(speaker_groups_by_id[facts[sid]]) != 1 for sid in segment_ids):
            for sid in segment_ids:
                exclude(sid, "speaker_id_group_inconsistent", block_group=True)

    speakers: list[SpeakerSpeechReuseAsset] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audio-reuse-", dir=output.parent) as temporary:
        stage = Path(temporary) / "assets"
        stage.mkdir()

        def common(stem: StemArtifact) -> dict:
            return {
                "clip_uid": job.clip_uid, "source_stem_path": str(Path(stem.canonical_stem_path).resolve()),
                "source_stem_sha256": stem.canonical_stem_sha256,
                "source_stem_record_fingerprint": stem_record.record_fingerprint,
                "source_stem_verification_state": stem_record.separation_state,
                "source_stem_frame_count": stem.canonical_frame_count,
                "target_audio_path": str(target), "target_audio_sha256": job.target_full_audio_sha256,
                "target_frame_count": frames, "target_duration_seconds": frames / SAMPLE_RATE,
            }

        for group, segment_ids in grouped.items():
            included = [sid for sid in segment_ids if sid not in unsafe]
            if group in blocked_groups or not included:
                continue
            speaker_ids = {facts[sid] for sid in segment_ids}
            entities = group_entities.get(group, set())
            if len(speaker_ids) != 1 or len(entities) > 1:
                reason = "speaker_id_group_inconsistent" if len(speaker_ids) != 1 else "conflicting_entity_provenance"
                for sid in segment_ids:
                    exclude(sid, reason)
                continue
            entity = next(iter(entities), None)
            subjects = [s for s in job.reference_subjects if s.kind == "entity" and s.entity_id == entity] if entity else []
            if entity is not None and len(subjects) != 1:
                raise AudioReuseIntegrityError("audio_reuse_entity_subject_provenance_mismatch")
            speaker_id = next(iter(speaker_ids))
            name = f"{speaker_id}_{group}.flac"
            pcm = np.zeros((frames, 2), dtype=np.int16)
            for sid in included:
                r = ranges[sid]
                pcm[r.target_start_sample:r.target_end_sample] = speech_pcm[r.target_start_sample:r.target_end_sample]
            speakers.append(SpeakerSpeechReuseAsset(
                **common(speech), speaker_group=group, speaker_id=speaker_id,
                entity_id=entity, subject_index=subjects[0].subject_index if subjects else None,
                source_segment_ids=included, source_sample_ranges=[ranges[sid] for sid in included],
                source_job_fingerprint=job.request_fingerprint, source_annotation_sha256=annotation_hash,
                output_path=str(output / name), output_sha256=_write_verified(stage / name, pcm),
            ))
        copied = min(frames, len(music_pcm))
        pcm = np.zeros((frames, 2), dtype=np.int16)
        pcm[:copied] = music_pcm[:copied]
        music_asset = MusicReuseAsset(
            **common(music), copied_frame_count=copied, silence_tail_frame_count=frames-copied,
            truncated_tail_frame_count=len(music_pcm)-copied,
            output_path=str(output / "music.flac"), output_sha256=_write_verified(stage / "music.flac", pcm),
        )
        # Recheck inputs before atomic publication; never overwrite a source/stage.
        for path, digest in sources.items():
            if sha256_file(path) != digest:
                raise AudioReuseIntegrityError("audio_reuse_source_changed_during_build")
        exclusions = list({(e.segment_id, e.reason): e for e in exclusions}.values())
        manifest = AudioReuseManifest(
            clip_uid=job.clip_uid, source_job_fingerprint=job.request_fingerprint,
            source_annotation_sha256=annotation_hash, source_stem_record_fingerprint=stem_record.record_fingerprint,
            speakers=speakers, music=music_asset, exclusions=exclusions,
        )
        (stage / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return manifest
