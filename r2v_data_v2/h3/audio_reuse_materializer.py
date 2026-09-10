"""Model-free H3 products from frozen MiMo and validated Phase-1 PCM assets."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from r2v_data_v2.h3.audio_reuse import (
    AudioReuseManifest,
    MusicReuseAsset,
    SpeakerSpeechReuseAsset,
    probe_canonical_target_frames,
    read_canonical_stem_pcm16,
    read_reuse_asset_pcm16,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    validate_prepared_inputs,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2, FinalQwen3SpeechSegment
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob, MimoInventory
from r2v_data_v2.h3.mimo25_backend import direct_speech_facts
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3MaterializationContractError,
    _materialize_sample,
    _validate_job_media_integrity,
    validate_product_dialogue_sections,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    ConditioningVariant,
    RecaptionAudioContract,
    audio_task_prefix,
)
from r2v_data_v2.h3.resolved_audio_stems import StemRecord, validate_stem_record
from r2v_data_v2.h3.sam_audio_stem_shadow import SAMAudioStemRecord, sha256_file
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons

AUDIO_REUSE_MATERIALIZER_VERSION = "h3_mimo25_audio_reuse_materializer_v6"


def _hash(value: SchemaModel | dict) -> str:
    values = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(path: Path, model: type[SchemaModel]) -> list:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class ValidatedReuseSource:
    job: MimoClipJob
    record: AudioReusePreparedSource
    manifest: AudioReuseManifest
    manifest_path: Path
    manifest_sha256: str


def load_reuse_source(
    *, manifest_path: Path, job: MimoClipJob, record: AudioReusePreparedSource, stem_record: StemRecord,
) -> ValidatedReuseSource:
    """Validate published assets against independent frozen job/annotation/stem inputs."""
    job = MimoClipJob.model_validate(job.model_dump())
    record = AudioReusePreparedSource.model_validate(record.model_dump())
    stem_record = validate_stem_record(stem_record)
    path = manifest_path.resolve(strict=True)
    digest = sha256_file(path)
    manifest = AudioReuseManifest.model_validate_json(path.read_text())
    annotation = record.annotation
    if record.status != "ready" or annotation is None:
        raise ValueError("reuse needs a ready frozen MiMo annotation")
    if (
        record.clip_uid != job.clip_uid or record.source_job_fingerprint != job.request_fingerprint
        or record.source_stem_record_fingerprint != stem_record.record_fingerprint
        or stem_record.clip_uid != job.clip_uid or stem_record.separation_state == "failure"
        or manifest.clip_uid != job.clip_uid
        or manifest.source_job_fingerprint != job.request_fingerprint
        or manifest.source_annotation_sha256 != _hash(annotation)
        or manifest.source_stem_record_fingerprint != stem_record.record_fingerprint
    ):
        raise ValueError("reuse manifest lineage mismatch")
    frames = probe_canonical_target_frames(Path(job.target_full_audio_path), job.target_full_audio_sha256)
    if sha256_file(Path(job.target_video_path)) != job.target_video_sha256:
        raise ValueError("reuse target video changed")
    segments = {s.segment_id: s for s in job.segments}
    decisions = annotation.audio_observation.segment_decisions
    groundings = annotation.av_grounding.segment_groundings
    if [d.segment_id for d in decisions] != list(segments) or [g.segment_id for g in groundings] != list(segments):
        raise ValueError("reuse annotation segment inventory mismatch")
    decisions_by_id = {d.segment_id: d for d in decisions}
    facts = {s["segment_id"]: s["speaker_id"] for s in direct_speech_facts(annotation, list(job.segments))}
    for asset in [*manifest.speakers, manifest.music]:
        stem = stem_record.stem(asset.source_stem_type)
        if (
            asset.clip_uid != job.clip_uid or asset.target_audio_path != str(Path(job.target_full_audio_path).resolve())
            or asset.target_audio_sha256 != job.target_full_audio_sha256
            or asset.source_stem_record_fingerprint != stem_record.record_fingerprint
            or asset.source_stem_verification_state != stem_record.separation_state
            or asset.source_stem_path != str(Path(stem.canonical_stem_path).resolve())
            or asset.source_stem_sha256 != stem.canonical_stem_sha256
            or asset.source_stem_frame_count != stem.canonical_frame_count
            or Path(stem.source_audio_path).resolve() != Path(job.target_full_audio_path).resolve()
            or stem.source_audio_sha256 != job.target_full_audio_sha256
            or asset.target_frame_count != frames
            or Path(asset.output_path).resolve().parent != path.parent
        ):
            raise ValueError("reuse asset source provenance mismatch")
        if len(read_canonical_stem_pcm16(Path(asset.source_stem_path), asset.source_stem_sha256)) != asset.source_stem_frame_count:
            raise ValueError("reuse source stem frame count mismatch")
        if len(read_reuse_asset_pcm16(Path(asset.output_path), asset.output_sha256)) != frames:
            raise ValueError("reuse asset frame count mismatch")
        if isinstance(asset, SpeakerSpeechReuseAsset):
            if asset.source_job_fingerprint != job.request_fingerprint or asset.source_annotation_sha256 != _hash(annotation):
                raise ValueError("reuse speaker source mismatch")
            if asset.source_segment_ids != [s for s in segments if s in set(asset.source_segment_ids)]:
                raise ValueError("reuse speaker segment order mismatch")
            for interval in asset.source_sample_ranges:
                source = segments[interval.segment_id]
                decision = decisions_by_id[source.segment_id]
                if (
                    source.asr_status != "transcribed" or speaker_ownership_reasons(decision)
                    or decision.primary_speaker_group != asset.speaker_group
                    or facts.get(source.segment_id) != asset.speaker_id
                    or any(getattr(interval, name) != getattr(source, name) for name in (
                        "source_start_sample", "source_end_sample", "source_sample_rate_hz", "start_time", "end_time",
                    ))
                ):
                    raise ValueError("reuse speaker interval differs from frozen ownership/timing")
            entities = {g.entity_id for g in groundings
                        if g.primary_speaker_group == asset.speaker_group and g.binding_status == "visible_entity"
                        and g.entity_id is not None and not speaker_ownership_reasons(decisions_by_id[g.segment_id])}
            if entities != ({asset.entity_id} if asset.entity_id else set()):
                raise ValueError("reuse entity ownership differs from frozen annotation")
            if asset.entity_id is not None and not any(
                s.kind == "entity" and s.entity_id == asset.entity_id and s.subject_index == asset.subject_index
                for s in job.reference_subjects
            ):
                raise ValueError("reuse Subject ownership mismatch")
    if len({s.speaker_group for s in manifest.speakers}) != len(manifest.speakers) or len({s.speaker_id for s in manifest.speakers}) != len(manifest.speakers):
        raise ValueError("reuse manifest repeats speaker ownership")
    return ValidatedReuseSource(job, record, manifest, path, digest)


class ReuseAudioReference(SchemaModel):
    """General Audio provenance; multi-segment sources never become one interval."""
    contract: RecaptionAudioContract
    role: Literal["speaker_speech_reuse", "voice_reference", "music_reuse", "full_audio_reuse", "music_reference"]
    source_type: Literal["target_speaker_reuse_asset", "donor_speaker_reuse_asset", "target_music_reuse_asset", "canonical_full_audio", "legacy_music_reference"]
    source_clip_uid: str
    reuse_manifest_path: str | None = None
    reuse_manifest_sha256: str | None = None
    reuse_manifest_fingerprint: str | None = None
    reuse_asset: SpeakerSpeechReuseAsset | MusicReuseAsset | None = None
    source_mimo_record_fingerprint: str
    donor_occurrence_id: str | None = None
    donor_entity_id: str | None = None
    donor_clip_display_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> ReuseAudioReference:
        expected = {"speaker_speech_reuse": ("speaker_speech_reuse", "target_speaker_reuse_asset"),
                    "cross_voice": ("voice_reference", "donor_speaker_reuse_asset"),
                    "music_reuse": ("music_reuse", "target_music_reuse_asset"),
                    "full_audio_reuse": ("full_audio_reuse", "canonical_full_audio"),
                    "music_reference": ("music_reference", "legacy_music_reference")}
        if expected.get(self.contract.kind) != (self.role, self.source_type):
            raise ValueError("reuse Audio role differs from contract")
        if self.reuse_asset is not None:
            if not all((self.reuse_manifest_path, self.reuse_manifest_sha256, self.reuse_manifest_fingerprint)):
                raise ValueError("reuse Audio lacks manifest provenance")
            if (self.contract.path, self.contract.sha256, self.source_clip_uid) != (
                self.reuse_asset.output_path, self.reuse_asset.output_sha256, self.reuse_asset.clip_uid,
            ):
                raise ValueError("reuse Audio differs from asset")
        elif self.contract.kind not in {"full_audio_reuse", "music_reference"}:
            raise ValueError("reuse Audio lacks asset provenance")
        if self.contract.kind == "cross_voice":
            if not all((self.contract.speaker_id, self.donor_occurrence_id, self.donor_entity_id, self.donor_clip_display_path)):
                raise ValueError("cross Audio lacks speaker/donor provenance")
            if not isinstance(self.reuse_asset, SpeakerSpeechReuseAsset) or self.reuse_asset.entity_id != self.donor_entity_id:
                raise ValueError("cross Audio donor entity differs from asset")
        return self


def _reference(source: ValidatedReuseSource, asset: SpeakerSpeechReuseAsset | MusicReuseAsset, **values) -> ReuseAudioReference:
    return ReuseAudioReference(
        source_clip_uid=source.job.clip_uid, reuse_manifest_path=str(source.manifest_path),
        reuse_manifest_sha256=source.manifest_sha256, reuse_manifest_fingerprint=_hash(source.manifest),
        reuse_asset=asset, source_mimo_record_fingerprint=source.record.source_reconcile_record_fingerprint, **values,
    )


def select_reuse_audio(
    sample: FinalH3SampleV2, source: ValidatedReuseSource, sources: dict[str, ValidatedReuseSource],
) -> tuple[list[ReuseAudioReference], list[str]]:
    """Use frozen donor mappings; speaker capacity is chronological, never quality-ranked."""
    refs: list[ReuseAudioReference] = []
    warnings = [f"{e.segment_id}:reuse_asset_excluded:{e.reason}" for e in source.manifest.exclusions]
    targets = [(s.speaker_id, s.entity_id, s.subject_index, s) for s in source.manifest.speakers]
    if sample.pair_type == "cross_pair":
        # Donor conditioning does not depend on the target having a usable reuse track.
        annotation = source.record.annotation
        facts = {f["segment_id"]: f["speaker_id"] for f in direct_speech_facts(annotation, list(source.job.segments))}
        decisions = {d.segment_id: d for d in annotation.audio_observation.segment_decisions}
        groups: dict[str, set[str]] = {}
        for g in annotation.segment_decisions:
            if (g.segment_id in facts and g.binding_status == "visible_entity"
                    and g.speech_presentation == "onscreen_spoken" and g.entity_id
                    and not speaker_ownership_reasons(decisions[g.segment_id])):
                groups.setdefault(facts[g.segment_id], set()).add(g.entity_id)
        subjects = {s.entity_id: s.subject_index for s in source.job.reference_subjects if s.kind == "entity"}
        targets = []
        for speaker, entities in groups.items():
            if len(entities) != 1 or next(iter(entities)) not in subjects:
                warnings.append(f"{speaker}:cross_target_ownership_not_unique")
                continue
            entity = next(iter(entities))
            targets.append((speaker, entity, subjects[entity], None))
    for speaker_id, entity_id, subject_index, target in sorted(targets, key=lambda t: int(t[0][1:])):
        donor_voice = None
        owner, asset = source, target
        kind = "speaker_speech_reuse"
        profile = None
        if sample.pair_type == "cross_pair":
            matches = [v for v in sample.subject_voices if v.voice_source == "cross_donor" and v.entity_id == entity_id]
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError("duplicate frozen target donor mapping")
            donor_voice = matches[0]
            owner = sources.get(donor_voice.donor_clip_uid)
            prefix = f"{donor_voice.donor_clip_uid}/"
            occurrence = donor_voice.donor_occurrence_id or ""
            if owner is None or not occurrence.startswith(prefix):
                warnings.append(f"{speaker_id}:donor_reuse_unavailable")
                continue
            donor_entity = occurrence[len(prefix):]
            candidates = [s for s in owner.manifest.speakers if s.entity_id == donor_entity]
            if len(candidates) != 1:
                warnings.append(f"{speaker_id}:donor_speaker_reuse_not_unique")
                continue
            asset = candidates[0]
            kind = "cross_voice"
            profiles = [p.voice_characteristics for p in owner.record.annotation.speaker_voice_profiles
                        if p.speaker_group == asset.speaker_group]
            profile = profiles[0] if len(profiles) == 1 else None
        if len(refs) == 3:
            warnings.append(f"{speaker_id}:audio_capacity_omitted_speaker")
            continue
        index = len(refs) + 1
        contract = RecaptionAudioContract(
            audio_index=index, audio_label=f"<Audio {index}>", kind=kind,
            path=asset.output_path, sha256=asset.output_sha256,
            subject_label=f"<Subject {subject_index}>" if subject_index else None,
            entity_id=entity_id, speaker_id=speaker_id,
            voice_characteristics=profile, retention_marker="reference" if donor_voice else "partially_copy",
        )
        refs.append(_reference(
            owner, asset, contract=contract,
            role="voice_reference" if donor_voice else "speaker_speech_reuse",
            source_type="donor_speaker_reuse_asset" if donor_voice else "target_speaker_reuse_asset",
            donor_occurrence_id=donor_voice.donor_occurrence_id if donor_voice else None,
            donor_entity_id=asset.entity_id if donor_voice else None,
            donor_clip_display_path=donor_voice.donor_clip_display_path if donor_voice else None,
        ))
    # Annotation .20 uses the final section's N/A sentinel, not a separate status enum.
    music = source.record.annotation.h3_semantics.non_diegetic_music.strip()
    if music and music.casefold() not in {"n/a", "unknown"}:
        if len(refs) == 3:
            warnings.append("audio_capacity_omitted_music")
        else:
            index = len(refs) + 1
            asset = source.manifest.music
            refs.append(_reference(source, asset, role="music_reuse", source_type="target_music_reuse_asset",
                                   contract=RecaptionAudioContract(
                                       audio_index=index, audio_label=f"<Audio {index}>", kind="music_reuse",
                                       path=asset.output_path, sha256=asset.output_sha256, retention_marker="partially_copy",
                                   )))
    return refs, warnings


class AudioReuseProduct(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_product.1"] = "r2v.h3.audio_reuse_product.1"
    materializer_version: Literal["h3_mimo25_audio_reuse_materializer_v3", "h3_mimo25_audio_reuse_materializer_v4",
                                "h3_mimo25_audio_reuse_materializer_v5", "h3_mimo25_audio_reuse_materializer_v6"] = AUDIO_REUSE_MATERIALIZER_VERSION
    sample_id: str
    source_h3_sample_id: str
    source_h3_sample_sha256: str
    clip_uid: str
    pair_type: Literal["canonical", "in_pair", "cross_pair"]
    conditioning_variant: ConditioningVariant
    source_mimo_record_fingerprint: str
    status: Literal["ready", "failed"]
    audio_references: list[ReuseAudioReference]
    corrected_speech_segments: list[FinalQwen3SpeechSegment]
    rendered_h3_prompt: str | None
    task_prefix: str
    warnings: list[str]
    failure_reason: str | None = None
    record_fingerprint: str

    @model_validator(mode="after")
    def validate_record(self) -> AudioReuseProduct:
        if self.task_prefix != audio_task_prefix([r.contract for r in self.audio_references]):
            raise ValueError("product task prefix differs from actual Audio")
        if [r.contract.audio_index for r in self.audio_references] != list(range(1, len(self.audio_references) + 1)) or len(self.audio_references) > 3:
            raise ValueError("product Audio capacity/order mismatch")
        if self.status == "ready":
            kinds = [r.contract.kind for r in self.audio_references]
            if (self.conditioning_variant == "target_speech_reuse" and "speaker_speech_reuse" not in kinds
                    or self.conditioning_variant == "cross_voice_reference" and "cross_voice" not in kinds
                    or self.conditioning_variant == "visual_only" and kinds
                    or self.conditioning_variant == "full_audio_reuse" and (
                        kinds != ["full_audio_reuse"] or self.audio_references[0].contract.retention_marker != "fully_copy"
                        or self.audio_references[0].source_type != "canonical_full_audio")):
                raise ValueError("product conditioning variant differs from actual Audio")
            if not self.rendered_h3_prompt or self.failure_reason:
                raise ValueError("ready reuse product is incomplete")
            validate_product_dialogue_sections(
                self.rendered_h3_prompt,
                [f"<d>[{s.language or 'Unknown'}] {s.text}</d>" for s in self.corrected_speech_segments],
            )
        elif self.rendered_h3_prompt is not None or not self.failure_reason or self.audio_references:
            raise ValueError("failed reuse product cannot publish Audio")
        if self.record_fingerprint != _hash(self.model_dump(mode="json", exclude={"record_fingerprint"})):
            raise ValueError("reuse product fingerprint mismatch")
        return self


class AudioReuseProductSummary(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_product_summary.1"] = "r2v.h3.audio_reuse_product_summary.1"
    materializer_version: Literal["h3_mimo25_audio_reuse_materializer_v1", "h3_mimo25_audio_reuse_materializer_v3",
                                "h3_mimo25_audio_reuse_materializer_v4", "h3_mimo25_audio_reuse_materializer_v5",
                                "h3_mimo25_audio_reuse_materializer_v6"] = AUDIO_REUSE_MATERIALIZER_VERSION
    source_hashes: dict[str, str]
    clip_uids: list[str]
    sample_count: int
    ready_count: int
    failed_count: int
    audio_kind_counts: dict[str, int]
    task_prefix_counts: dict[str, int]
    warning_counts: dict[str, int]
    model_call_count: Literal[0] = 0
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> AudioReuseProductSummary:
        if self.sample_count != self.ready_count + self.failed_count or sum(self.task_prefix_counts.values()) != self.sample_count:
            raise ValueError("reuse product summary counts mismatch")
        return self


def materialize_audio_reuse_products(
    *, mimo_root: Path, source_h3_root: Path, separation_root: Path, reuse_root: Path,
    audio_production_root: Path, output_root: Path, enable_full_audio_reuse: bool = False,
) -> AudioReuseProductSummary:
    """Read prepared frozen roots; publish a new shadow directory without model calls."""
    mimo_root, source_h3_root, separation_root, reuse_root = (
        path.resolve(strict=True) for path in (mimo_root, source_h3_root, separation_root, reuse_root)
    )
    output = output_root.resolve()
    paths = jea_production_paths(audio_production_root)
    protected = [getattr(paths, f.name).resolve() for f in fields(paths) if f.name != "root"]
    protected += [mimo_root, source_h3_root, separation_root, reuse_root]
    if paths.root.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("reuse product output overlaps protected inputs/stages")
    if output.exists():
        raise FileExistsError(output)
    input_files = [mimo_root / "inventory.json", mimo_root / "records.jsonl",
                   source_h3_root / "samples.jsonl", separation_root / "records.jsonl"]
    hashes = {str(p): sha256_file(p) for p in input_files}
    inventory = MimoInventory.model_validate_json(input_files[0].read_text())
    records = _rows(input_files[1], AudioReusePreparedSource)
    samples = _rows(input_files[2], FinalH3SampleV2)
    from r2v_data_v2.h3.resolved_audio_stems import RESOLVED_STAGE, load_stem_source

    stems = (
        load_stem_source(separation_root)[1] if separation_root.name == RESOLVED_STAGE
        else _rows(input_files[3], SAMAudioStemRecord)
    )
    if hashes[str(input_files[2])] != inventory.source_h3_samples_sha256:
        raise ValueError("frozen H3 samples changed")
    by_clip = {r.clip_uid: r for r in records}
    stem_by_clip = {r.clip_uid: r for r in stems}
    if len(by_clip) != len(records) or set(by_clip) != {j.clip_uid for j in inventory.jobs} or len(stem_by_clip) != len(stems):
        raise ValueError("duplicate or missing frozen clip inventory")
    validate_prepared_inputs(inventory, records, samples, stems)
    for job in inventory.jobs:
        _validate_job_media_integrity(job)
    sources = {}
    for job in inventory.jobs:
        record = by_clip[job.clip_uid]
        if record.status != "ready":
            continue
        manifest = reuse_root / job.clip_uid / "manifest.json"
        if not manifest.resolve().is_relative_to(reuse_root):
            raise ValueError("unsafe reuse clip path")
        if job.clip_uid not in stem_by_clip:
            raise ValueError("missing frozen SAM record")
        source = load_reuse_source(manifest_path=manifest, job=job, record=record, stem_record=stem_by_clip[job.clip_uid])
        sources[job.clip_uid] = source
        hashes[str(source.manifest_path)] = source.manifest_sha256
        hashes[job.target_video_path] = job.target_video_sha256
        hashes[job.target_full_audio_path] = job.target_full_audio_sha256
        for asset in [*source.manifest.speakers, source.manifest.music]:
            hashes[asset.output_path] = asset.output_sha256
            hashes[asset.source_stem_path] = asset.source_stem_sha256
    if any(Path(path).resolve().is_relative_to(output) for path in hashes):
        raise ValueError("reuse product output contains source media")
    products: list[AudioReuseProduct] = []
    for job in inventory.jobs:
        clip_samples = [s for s in samples if s.clip_uid == job.clip_uid]
        source = sources.get(job.clip_uid)
        has_in_pair = any(s.pair_type == "in_pair" for s in clip_samples)
        plans = [(s, "visual_only" if s.pair_type == "canonical" else
                  "cross_voice_reference" if s.pair_type == "cross_pair" else "target_speech_reuse", s.sample_id)
                 for s in clip_samples]
        canonical = next((s for s in clip_samples if s.pair_type == "canonical"), None)
        if canonical is None:
            raise ValueError("missing canonical H3 sample")
        if not has_in_pair:
            plans.append((canonical, "target_speech_reuse", f"{job.clip_uid}/target_speech_reuse"))
        if enable_full_audio_reuse:
            plans.append((canonical, "full_audio_reuse", f"{job.clip_uid}/audio_reuse"))
        for sample, variant, sample_id in plans:
            refs, warnings, corrected = [], [], []
            rendered, failure = None, None
            try:
                if source is None:
                    raise ValueError("ready frozen annotation unavailable")
                if variant == "full_audio_reuse":
                    refs = [ReuseAudioReference(
                        contract=RecaptionAudioContract(audio_index=1, audio_label="<Audio 1>", kind="full_audio_reuse",
                                                       path=job.target_full_audio_path, sha256=job.target_full_audio_sha256,
                                                       retention_marker="fully_copy"),
                        role="full_audio_reuse", source_type="canonical_full_audio", source_clip_uid=job.clip_uid,
                        source_mimo_record_fingerprint=source.record.source_reconcile_record_fingerprint,
                    )]
                elif variant != "visual_only":
                    refs, warnings = select_reuse_audio(sample, source, sources)
                    required_kind = "cross_voice" if variant == "cross_voice_reference" else "speaker_speech_reuse"
                    if not any(r.contract.kind == required_kind for r in refs):
                        continue  # Unavailable variant, not a failed materialization.
                corrected, rendered, render_warnings = _materialize_sample(
                    sample, job, source.record, conditioning_variant=variant,
                    reuse_audio_contracts=[r.contract for r in refs],
                )
                warnings.extend(render_warnings)
            except (ValueError, MimoH3MaterializationContractError) as exc:
                refs, corrected = [], []
                failure = str(exc)
            values = {
                "schema_version": "r2v.h3.audio_reuse_product.1", "materializer_version": AUDIO_REUSE_MATERIALIZER_VERSION,
                "sample_id": sample_id, "source_h3_sample_id": sample.sample_id, "source_h3_sample_sha256": _hash(sample),
                "clip_uid": job.clip_uid, "pair_type": sample.pair_type, "conditioning_variant": variant,
                "source_mimo_record_fingerprint": by_clip[job.clip_uid].source_reconcile_record_fingerprint,
                "status": "failed" if failure else "ready", "audio_references": [r.model_dump(mode="json") for r in refs],
                "corrected_speech_segments": [s.model_dump(mode="json") for s in corrected], "rendered_h3_prompt": rendered,
                "task_prefix": audio_task_prefix([r.contract for r in refs]), "warnings": sorted(set(warnings)), "failure_reason": failure,
            }
            products.append(AudioReuseProduct(**values, record_fingerprint=_hash(values)))
    summary = AudioReuseProductSummary(
        source_hashes=dict(sorted(hashes.items())), clip_uids=[j.clip_uid for j in inventory.jobs],
        sample_count=len(products), ready_count=sum(p.status == "ready" for p in products),
        failed_count=sum(p.status == "failed" for p in products),
        audio_kind_counts=dict(sorted(Counter(r.contract.kind for p in products for r in p.audio_references).items())),
        task_prefix_counts=dict(sorted(Counter(p.task_prefix for p in products).items())),
        warning_counts=dict(sorted(Counter(w for p in products for w in p.warnings).items())),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reuse-products-", dir=output.parent) as temporary:
        stage = Path(temporary) / "products"
        stage.mkdir()
        (stage / "records.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in products))
        (stage / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
        for path, digest in hashes.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("frozen reuse input changed during materialization")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return summary
