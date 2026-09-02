from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.binding_audit import (
    SpeakerBindingAuditSummary,
    SpeakerBindingSegmentAudit,
)
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClipResult,
    DiarizationInventory,
    DiarizationTargetClip,
)
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    JEAOccurrenceEmbedding,
    build_jea_pairs,
    full_audio_path,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalSubjectVoice,
    FinalVisualReference,
    render_jea_final_samples,
)
from r2v_data_v2.h3.qwen3_asr import (
    Qwen3ASRConfiguration,
    Qwen3ASRSegment,
)
from r2v_data_v2.h3.specialized_audio_semantics import (
    SpecializedAudioSemanticsRecord,
    SpecializedBackendProvenance,
)
from r2v_data_v2.h3.specialized_audio_semantics import (
    _compact_json as _specialized_compact_json,
)
from r2v_data_v2.h3.target_audio_caption_contract import (
    TargetSpeakerDelivery,
    TemporalAudioEvent,
)
from r2v_data_v2.h3.visual_production_source import (
    NormalizedVisualReference,
    NormalizedVisualSample,
    ReadableClipIdentity,
    VisualProductionClip,
    VisualProductionInventory,
)
from r2v_data_v2.v3.production_export import ProductionSample
from r2v_data_v2.v3.schemas import ClipRecord


def _identity(clip_uid: str, clip_name: str) -> ReadableClipIdentity:
    return ReadableClipIdentity(
        clip_uid=clip_uid,
        clip_display_path=f"节目/集合/{clip_name}",
        media_collection_relpath="节目/集合",
        media_collection_name="集合",
        episode_name=clip_name.split("_")[0],
        clip_name=clip_name,
        shard_id=f"shard-{clip_uid}",
    )


def _visual_clip(
    tmp_path: Path,
    clip_uid: str,
    clip_name: str,
    *,
    subject_scope: str = "full",
    subject_visible_region: str | None = None,
    subject_entity_ids: tuple[str, ...] = ("entity_1",),
) -> VisualProductionClip:
    identity = _identity(clip_uid, clip_name)
    target = tmp_path / f"{clip_uid}.mp4"
    target.write_bytes(b"processed")
    subjects = [
        tmp_path / f"{clip_uid}-subject-{index}.png"
        for index in range(1, len(subject_entity_ids) + 1)
    ]
    attribute = tmp_path / f"{clip_uid}-hair.png"
    for index, subject in enumerate(subjects, start=1):
        subject.write_bytes(f"subject-{index}".encode())
    attribute.write_bytes(b"hair")
    reference_rows = [
        {
            "image_id": f"image_{index}",
            "image_index": index,
            "kind": "subject",
            "entity_id": entity_id,
            "image_path": subject.name,
            "source_frame_index": 0,
            "scope": subject_scope,
            "visible_region": subject_visible_region,
            "synthetic": False,
        }
        for index, (entity_id, subject) in enumerate(
            zip(subject_entity_ids, subjects, strict=True), start=1
        )
    ]
    attribute_index = len(reference_rows) + 1
    reference_rows.append(
        {
            "image_id": f"image_{attribute_index}",
            "image_index": attribute_index,
            "kind": "attribute",
            "attribute_id": "attribute_hair",
            "owner_entity_id": "entity_1",
            "attribute_type": "hair",
            "image_path": attribute.name,
            "source_frame_index": 0,
            "synthetic": False,
        }
    )
    sample = ProductionSample.model_validate(
        {
            "sample_id": clip_uid,
            "clip_uid": clip_uid,
            "target_video": str(target),
            "t2v_caption": "canonical caption",
            "r2v_instruction": f"canonical {clip_uid}: "
            + " and ".join(
                f"Image {index}" for index in range(1, attribute_index + 1)
            ),
            "references": reference_rows,
            "source": {
                "parent_video_id": "legacy-parent",
                "clip_suffix": clip_uid,
                "shard_id": identity.shard_id,
            },
        }
    )
    clip = ClipRecord.model_validate(
        {
            "clip_uid": clip_uid,
            "source": {
                "video_path": str(target),
                "parent_video_id": "legacy-parent",
                "clip_suffix": clip_uid,
                "source_index": 0,
                "caption_raw": "",
                "metadata": {
                    "source_relative_video_path": f"节目/集合/{clip_name}.mp4",
                    "source_relative_source_video_path": f"节目/集合/{clip_name}.mkv",
                },
            },
        }
    )
    references = [
        NormalizedVisualReference(
            **reference.model_dump(mode="python"),
            artifact_path=str(tmp_path / reference.image_path),
        )
        for reference in sample.references
    ]
    normalized = NormalizedVisualSample(
        sample_id=sample.sample_id,
        clip_uid=sample.clip_uid,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=sample.r2v_instruction,
        references=references,
    )
    return VisualProductionClip(
        identity=identity,
        sample=normalized,
        clip=clip,
        clip_record_path=str(tmp_path / identity.shard_id / clip_uid / "clip.json"),
        subject_references=[
            reference for reference in references if reference.kind == "subject"
        ],
    )


def _visual_inventory(tmp_path: Path) -> VisualProductionInventory:
    clips = [
        _visual_clip(
            tmp_path,
            "clip-a",
            "episode_a_0001",
            subject_scope="local",
            subject_visible_region="upper_body",
        ),
        _visual_clip(
            tmp_path,
            "clip-b",
            "episode_b_0002",
            subject_visible_region="whole",
        ),
    ]
    return VisualProductionInventory(
        visual_production_root=str(tmp_path),
        visual_runs_root=str(tmp_path / "runs"),
        visual_input_schema="r2v.v3.production_sample.1",
        visual_input_mode="compacted_production",
        canonical_sample_count=2,
        eligible_clip_count=2,
        eligible_subject_occurrence_count=2,
        media_collection_count=1,
        media_collection_clip_counts={"节目/集合": 2},
        shard_count=2,
        canonical_clips=clips,
        clips=clips,
        skip_reason_counts={},
    )


def _canonical_wide_visual_inventory(tmp_path: Path) -> VisualProductionInventory:
    base = _visual_inventory(tmp_path)
    clip_c = _visual_clip(tmp_path, "clip-c", "episode_c_0003")
    clips = [*base.canonical_clips, clip_c]
    return base.model_copy(
        update={
            "canonical_sample_count": len(clips),
            "eligible_clip_count": len(clips),
            "eligible_subject_occurrence_count": len(clips),
            "media_collection_clip_counts": {"节目/集合": len(clips)},
            "shard_count": len(clips),
            "canonical_clips": clips,
            "clips": clips,
        }
    )


def _occurrences(inventory: VisualProductionInventory) -> list[JEAOccurrenceEmbedding]:
    return [
        JEAOccurrenceEmbedding(
            occurrence_id=f"{clip.identity.clip_uid}/entity_1",
            clip_uid=clip.identity.clip_uid,
            entity_id="entity_1",
            subject_index=1,
            identity=clip.identity,
            visual_reference_path=str(
                Path(inventory.visual_production_root)
                / clip.subject_references[0].image_path
            ),
            primary_voice_reference_path=str(
                Path(inventory.visual_production_root)
                / "primary_voice"
                / clip.identity.clip_display_path
                / "e1.flac"
            ),
            face_embedding=[1.0, 0.0],
            voice_embedding=[1.0, 0.0],
        )
        for clip in inventory.clips
    ]


def _jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                value if isinstance(value, dict) else value.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_binding_audit(
    tmp_path: Path,
    bound: list[BoundDiarizationSegment],
    *,
    clip_count: int | None = None,
) -> Path:
    root = tmp_path / "binding_audit_v1"
    root.mkdir()
    segments = []
    for item in bound:
        duration = item.end_time - item.start_time
        direct_support = (
            {item.entity_id: item.direct_anchor_seconds}
            if item.entity_id is not None and item.direct_anchor_seconds > 0
            else {}
        )
        segments.append(
            SpeakerBindingSegmentAudit(
                clip_uid=item.target_clip_uid,
                segment_id=item.segment_id,
                speaker_cluster_id=item.speaker_cluster_id,
                current_mapping_status=item.cluster_binding_status,
                current_entity_id=item.entity_id,
                start_time=item.start_time,
                end_time=item.end_time,
                speaker_duration_seconds=duration,
                direct_anchor_seconds=item.direct_anchor_seconds,
                direct_support_seconds_by_entity=direct_support,
                directly_supported_entity_count=len(direct_support),
                contested_anchor_seconds=0,
                identity_propagated_seconds=(
                    duration - item.direct_anchor_seconds
                    if item.cluster_binding_status == "candidate_mapped"
                    else 0
                ),
                exclusive_active_faces=[],
                exclusive_active_entities=[],
                unmatched_exclusive_active_face_track_ids=[],
                multi_active_lr_asd_seconds=0,
                multi_active_frame_count=0,
                max_simultaneous_active_face_count=0,
                multi_active_face_track_ids=[],
                multi_active_mapped_entity_ids=[],
                multi_active_unmatched_face_track_ids=[],
                flags={
                    "conflict": item.cluster_binding_status == "conflict",
                    "ambiguous": item.cluster_binding_status == "ambiguous",
                    "unbound": item.cluster_binding_status == "unbound",
                    "fully_propagated_segment": (
                        item.identity_scope == "cluster_propagated_only"
                    ),
                    "multiple_direct_entity_support": False,
                    "contested_anchor": False,
                    "multiple_exclusive_active_face_tracks": False,
                    "exclusive_active_entity_contradiction": False,
                    "mapped_entity_vs_unmatched_face_contradiction": False,
                    "has_multi_active_lr_asd_frames": False,
                    "multi_active_contains_current_entity_and_other_face": False,
                },
            )
        )
    _jsonl(root / "segments.jsonl", segments)
    cluster_statuses = {
        (item.target_clip_uid, item.speaker_cluster_id): item.cluster_binding_status
        for item in bound
    }
    status_counts = {
        status: sum(value == status for value in cluster_statuses.values())
        for status in ("candidate_mapped", "conflict", "ambiguous", "unbound")
    }
    summary = SpeakerBindingAuditSummary(
        source_audio_production_root=str(tmp_path),
        source_diarization_root=str(tmp_path / "diarization"),
        source_artifact_sha256={
            "bound_segments": _sha256(tmp_path / "diarization/bound_segments.jsonl"),
            "inventory": _sha256(tmp_path / "diarization/inventory.json"),
        },
        audio_binding_input_set_sha256="a" * 64,
        lr_asd_native_input_set_sha256="b" * 64,
        clip_count=(
            len({item.target_clip_uid for item in bound})
            if clip_count is None
            else clip_count
        ),
        cluster_count=len(cluster_statuses),
        segment_count=len(segments),
        candidate_mapped_count=status_counts["candidate_mapped"],
        conflict_count=status_counts["conflict"],
        ambiguous_count=status_counts["ambiguous"],
        unbound_count=status_counts["unbound"],
        clusters_with_multiple_exclusive_active_face_tracks=0,
        clusters_with_exclusive_active_entity_contradiction=0,
        clusters_with_mapped_entity_vs_unmatched_face=0,
        clusters_with_multiple_direct_entity_support=0,
        clusters_with_fully_propagated_segments=sum(
            item.identity_scope == "cluster_propagated_only" for item in bound
        ),
        clusters_with_multi_active_lr_asd_frames=0,
        clusters_with_current_entity_plus_other_active_face=0,
        multiple_exclusive_active_face_track_speaker_seconds=0,
        exclusive_active_entity_contradiction_speaker_seconds=0,
        mapped_entity_vs_unmatched_face_speaker_seconds=0,
        fully_propagated_segment_speaker_seconds=sum(
            item.end_time - item.start_time
            for item in bound
            if item.identity_scope == "cluster_propagated_only"
        ),
        identity_propagated_speaker_seconds=sum(
            item.end_time - item.start_time - item.direct_anchor_seconds
            for item in bound
            if item.cluster_binding_status == "candidate_mapped"
        ),
        multi_active_lr_asd_speaker_seconds=0,
        review_priority_counts={
            "candidate_mapped_lowest_direct_support_ratio": len(cluster_statuses)
        },
    )
    (root / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _write_canonical_sources(
    tmp_path: Path,
    inventory: VisualProductionInventory,
    *,
    clips_with_segments: set[str],
) -> None:
    canonical = []
    targets = []
    results = []
    for clip in inventory.canonical_clips:
        clip_uid = clip.identity.clip_uid
        audio = full_audio_path(tmp_path / "audio", clip.identity)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"canonical-audio:{clip_uid}".encode())
        video = Path(clip.sample.target_video).resolve(strict=True)
        canonical.append(
            CanonicalAudioClip(
                **clip.identity.model_dump(mode="python"),
                target_video_path=str(video),
                target_video_sha256=_sha256(video),
                target_full_audio_path=str(audio),
                target_full_audio_sha256=_sha256(audio),
                target_duration_seconds=0.1,
                subject_reference_count=len(clip.subject_references),
            )
        )
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video),
                source_audio_path=str(audio),
                source_audio_sha256=_sha256(audio),
                source_sample_rate_hz=32000,
                source_channels=2,
                source_frame_count=3200,
                visual_references=[],
            )
        )
        results.append(
            DiarizationClipResult(
                target_clip_uid=clip_uid,
                status="ready" if clip_uid in clips_with_segments else "empty",
                backend_called=True,
                raw_segment_count=1 if clip_uid in clips_with_segments else 0,
                speaker_cluster_count=1 if clip_uid in clips_with_segments else 0,
                legacy_bound_interval_count=0,
                legacy_coalesced_bound_turn_count=0,
                legacy_lr_asd_bound_samples=0,
                usable_direct_anchor_samples=0,
                contested_anchor_samples=0,
                unmatched_anchor_samples=0,
                diarization_segment_durations=([0.1] if clip_uid in clips_with_segments else []),
                legacy_bound_turn_durations=[],
            )
        )
    _jsonl(tmp_path / "audio/canonical_clips.jsonl", canonical)
    primary_voices = []
    voice_sources: dict[str, str] = {}
    pairs_path = tmp_path / "pairs/in_pairs.jsonl"
    if pairs_path.is_file():
        for pair in (json.loads(line) for line in pairs_path.read_text().splitlines()):
            for subject in pair["subjects"]:
                voice_sources[subject["target_occurrence_id"]] = subject[
                    "target_primary_voice_reference_path"
                ]
    cross_path = tmp_path / "pairs/cross_pairs.jsonl"
    if cross_path.is_file():
        for pair in (json.loads(line) for line in cross_path.read_text().splitlines()):
            for mapping in pair["mappings"]:
                voice_sources[mapping["donor_occurrence_id"]] = mapping[
                    "donor_primary_voice_reference_path"
                ]
    for occurrence_id, voice_path_value in sorted(voice_sources.items()):
        clip_uid, entity_id = occurrence_id.split("/", maxsplit=1)
        voice_path = Path(voice_path_value)
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        voice_path.write_bytes(f"canonical-voice:{occurrence_id}".encode())
        relative = voice_path.relative_to(tmp_path / "primary_voice").as_posix()
        primary_voices.append(
            {
                "schema_version": "r2v.h3.primary_voice_reference_selection.1",
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "entity_occurrence_id": occurrence_id,
                "primary_voice_reference": {
                    "voice_reference_id": "voice_ref_1",
                    "entity_occurrence_id": occurrence_id,
                    "source_turn_id": "turn_1",
                    "source_start": 0.0,
                    "source_end": 0.1,
                    "source_start_sample": 0,
                    "source_end_sample": 3200,
                    "asset": {
                        "path": relative,
                        "sha256": _sha256(voice_path),
                        "byte_size": voice_path.stat().st_size,
                        "media_type": "audio/flac",
                    },
                    "quality_score": 1.0,
                    "quality_metadata": {
                        "source_sample_rate_hz": 32000,
                        "source_channels": 2,
                        "sample_mapping_policy": "round_time_seconds_times_32000_v1",
                    },
                },
                "reason_codes": [],
                "candidate_turn_ids": ["turn_1"],
                "accepted_turn_ids": ["turn_1"],
                "policy_version": "fixture",
                "policy_fingerprint": "f" * 64,
            }
        )
    _jsonl(tmp_path / "primary_voice/primary_voice_references.jsonl", primary_voices)
    diarization = DiarizationInventory(
        mode="production",
        source_inventory_kind="canonical_audio_manifest",
        source_visual_production_root=inventory.visual_production_root,
        source_visual_inventory_path=str(tmp_path / "samples.jsonl"),
        source_visual_inventory_sha256="a" * 64,
        source_canonical_audio_manifest_path=str(
            tmp_path / "audio/canonical_clips.jsonl"
        ),
        source_canonical_audio_manifest_sha256=_sha256(
            tmp_path / "audio/canonical_clips.jsonl"
        ),
        inventory_fingerprint="b" * 64,
        source_target_count=len(targets),
        selected_target_count=len(targets),
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )
    (tmp_path / "diarization").mkdir(parents=True, exist_ok=True)
    (tmp_path / "diarization/inventory.json").write_text(
        diarization.model_dump_json(indent=2), encoding="utf-8"
    )
    _jsonl(tmp_path / "diarization/clip_results.jsonl", results)


def _specialized_provenance(
    *,
    role: str,
    media_root: Path,
) -> SpecializedBackendProvenance:
    input_modality = {
        "captioner": "canonical_full_audio_only",
        "global_semantics": "captioner_text_only",
        "local_semantics": "canonical_full_audio_only",
    }[role]
    media_mode = "none" if role == "global_semantics" else "file"
    values = {
        "backend": "vllm",
        "role": role,
        "served_model_name": f"fixture/{role}",
        "checkpoint_id": f"fixture/{role}",
        "base_url": "http://127.0.0.1:8000/v1",
        "input_modality": input_modality,
        "media_mode": media_mode,
        "media_root": None if media_mode == "none" else str(media_root),
        "media_base_url": None,
        "output_modalities": ["text"],
        "prompt_version": f"fixture_{role}_v1",
        "fallback_prompt_version": None,
        "fallback_policy_version": "fixture_fallback_v1",
        "temperature": 0.6 if role == "captioner" else 0.0,
        "top_p": 0.95 if role == "captioner" else None,
        "top_k": 20 if role == "captioner" else None,
        "max_tokens": 256,
        "repair_retries": 1,
        "runtime_dependency_fingerprints": None,
    }
    provisional = SpecializedBackendProvenance.model_construct(
        **values,
        configuration_fingerprint="",
    )
    fingerprint_values = provisional.model_dump(
        mode="json",
        exclude={"configuration_fingerprint"},
    )
    fingerprint_values.pop("runtime_dependency_fingerprints")
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return SpecializedBackendProvenance(
        **values,
        configuration_fingerprint=fingerprint,
    )


def _specialized_semantics_record(
    canonical: CanonicalAudioClip,
    *,
    media_root: Path,
) -> SpecializedAudioSemanticsRecord:
    captioner = _specialized_provenance(role="captioner", media_root=media_root)
    global_semantics = _specialized_provenance(
        role="global_semantics",
        media_root=media_root,
    )
    local_semantics = _specialized_provenance(
        role="local_semantics",
        media_root=media_root,
    )
    values = {
        "target_clip_uid": canonical.clip_uid,
        "clip_display_path": canonical.clip_display_path,
        "status": "complete",
        "raw_audio_caption": "raw caption must not enter Final H3",
        "overall_audio_description": f"Audio for {canonical.clip_uid}.",
        "overall_soundscape": "quiet interior ambience",
        "non_diegetic_music": None,
        "temporal_audio_events": [
            TemporalAudioEvent(
                start_time=0.02,
                end_time=0.08,
                description="a brief object sound",
            )
        ],
        "speaker_delivery": [
            TargetSpeakerDelivery(
                speaker_cluster_id="speaker_1",
                delivery_style="calm",
                entity_id="entity_1",
            )
        ],
        "target_video_path": canonical.target_video_path,
        "target_video_sha256": canonical.target_video_sha256,
        "target_full_audio_path": canonical.target_full_audio_path,
        "target_full_audio_sha256": canonical.target_full_audio_sha256,
        "target_duration_seconds": canonical.target_duration_seconds,
        "target_audio_binding_path": canonical.target_audio_binding_path,
        "target_audio_binding_sha256": canonical.target_audio_binding_sha256,
        "captioner_status": "ready",
        "global_semantics_status": "ready",
        "local_semantics_status": "ready",
        "captioner_provenance": captioner,
        "global_semantics_provenance": global_semantics,
        "local_semantics_provenance": local_semantics,
        "captioner_failure": None,
        "global_semantics_failure": None,
        "local_semantics_failure": None,
        "captioner_request_fingerprint": "1" * 64,
        "global_semantics_request_fingerprint": "2" * 64,
        "local_semantics_request_fingerprint": "3" * 64,
        "captioner_record_fingerprint": "4" * 64,
        "global_semantics_record_fingerprint": "5" * 64,
        "local_semantics_record_fingerprint": "6" * 64,
    }
    provisional = SpecializedAudioSemanticsRecord.model_construct(
        **values,
        assemble_fingerprint="",
    )
    fingerprint = hashlib.sha256(
        _specialized_compact_json(
            provisional.model_dump(
                mode="json",
                exclude={"assemble_fingerprint"},
            )
        ).encode()
    ).hexdigest()
    return SpecializedAudioSemanticsRecord(
        **values,
        assemble_fingerprint=fingerprint,
    )


def _final_sample(
    tmp_path: Path,
    voices: list[FinalSubjectVoice],
    *,
    clip_uid: str = "clip-a",
    references: list[FinalVisualReference] | None = None,
) -> FinalH3SampleV2:
    if references is None:
        references = []
        for index in (1, 2):
            artifact = tmp_path / f"entity_{index}.png"
            artifact.write_bytes(f"entity-{index}".encode())
            references.append(
                FinalVisualReference(
                    image_id=f"image_{index}",
                    image_index=index,
                    kind="subject",
                    image_path=f"references/entity_{index}.png",
                    image_artifact_path=str(artifact),
                    entity_id=f"entity_{index}",
                    source_frame_index=0,
                    scope="full",
                    visible_region="whole",
                    synthetic=False,
                )
            )
    return FinalH3SampleV2(
        sample_id=f"{clip_uid}/in_pair",
        pair_id=f"in_pair/{clip_uid}",
        pair_type="in_pair",
        clip_uid=clip_uid,
        clip_display_path=f"节目/集合/{clip_uid}",
        media_collection_relpath="节目/集合",
        media_collection_name="集合",
        episode_name="episode",
        clip_name=clip_uid,
        shard_id="shard-a",
        target_video=f"{clip_uid}.mp4",
        target_full_audio_path=f"{clip_uid}.flac",
        target_full_audio_sha256="a" * 64,
        r2v_instruction="Image 1 and Image 2",
        visual_references=references,
        subject_voices=voices,
        speech_segments=[],
    )


def _voice(
    entity_index: int,
    *,
    subject_index: int | None = None,
    clip_uid: str = "clip-a",
    target_occurrence_id: str | None = None,
) -> FinalSubjectVoice:
    entity_id = f"entity_{entity_index}"
    return FinalSubjectVoice(
        subject_index=entity_index if subject_index is None else subject_index,
        entity_id=entity_id,
        target_occurrence_id=target_occurrence_id
        or f"{clip_uid}/{entity_id}",
        voice_reference_path=f"voices/{entity_id}.flac",
        voice_reference_sha256="c" * 64,
        source_start=0.0,
        source_end=1.0,
        source_start_sample=0,
        source_end_sample=32000,
        sample_mapping_policy="round_time_seconds_times_32000_v1",
        voice_source="target",
    )


def _render(tmp_path: Path) -> list[dict[str, object]]:
    inventory = _visual_inventory(tmp_path)
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    bound = []
    qwen = []
    for clip in inventory.clips:
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid=clip.identity.clip_uid,
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                start_time=0.0,
                end_time=0.1,
                source_start_sample=0,
                source_end_sample=3200,
                cluster_binding_status="candidate_mapped",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip.identity.clip_uid}/entity_1",
                direct_anchor_samples=100,
                direct_anchor_seconds=0.00625,
                identity_scope="direct_anchor_present",
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **clip.identity.model_dump(mode="python"),
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip.identity.clip_uid}/entity_1",
                source_audio_path=str(tmp_path / f"{clip.identity.clip_uid}.flac"),
                source_start_sample=0,
                source_end_sample=3200,
                source_sample_rate_hz=32000,
                start_time=0.0,
                end_time=0.1,
                status="transcribed",
                text=f"raw {clip.identity.clip_uid}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)
    _write_canonical_sources(
        tmp_path,
        inventory,
        clips_with_segments={item.target_clip_uid for item in bound},
    )
    audit_root = _write_binding_audit(
        tmp_path,
        bound,
        clip_count=len(inventory.canonical_clips),
    )
    render_jea_final_samples(
        visual_inventory=inventory,
        audio_root=tmp_path / "audio",
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        binding_audit_root=audit_root,
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )
    return [
        json.loads(line)
        for line in (tmp_path / "h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_final_visual_reference_preserves_canonical_visible_region_values(
    tmp_path: Path,
) -> None:
    (tmp_path / "local.png").write_bytes(b"local")
    (tmp_path / "full.png").write_bytes(b"full")
    local = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/local.png",
            entity_id="entity_1",
            source_frame_index=1,
            scope="local",
            visible_region="upper_body",
            synthetic=False,
            artifact_path=str(tmp_path / "local.png"),
        )
    )
    full = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_2",
            image_index=2,
            kind="object",
            image_path="references/full.png",
            entity_id="entity_2",
            source_frame_index=2,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(tmp_path / "full.png"),
        )
    )

    assert local.visible_region == "upper_body"
    assert isinstance(local.visible_region, str)
    assert full.visible_region == "whole"


def test_final_visual_reference_keeps_nullable_background_and_attribute_region(
    tmp_path: Path,
) -> None:
    (tmp_path / "background.png").write_bytes(b"background")
    (tmp_path / "hair.png").write_bytes(b"hair")
    background = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="background",
            image_path="references/background.png",
            source_frame_index=1,
            scope="scene",
            visible_region=None,
            synthetic=False,
            artifact_path=str(tmp_path / "background.png"),
        )
    )
    attribute = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_2",
            image_index=2,
            kind="attribute",
            image_path="references/hair.png",
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=2,
            scope=None,
            visible_region=None,
            synthetic=False,
            artifact_path=str(tmp_path / "hair.png"),
        )
    )

    assert background.visible_region is None
    assert attribute.visible_region is None


def test_final_visual_reference_publishes_exact_normalized_artifact_paths(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "visual-export/references/ordinary.png"
    enriched = tmp_path / "visual-runs/clips/clip-a/selected/local.png"
    background_path = tmp_path / "visual-runs/clips/clip-a/selected/background.png"
    attribute_path = (
        tmp_path / "visual-runs/subject_attributes/clip-a/attribute_hair.png"
    )
    compacted = tmp_path / "compacted-production/references/subject.png"
    for path in (ordinary, enriched, background_path, attribute_path, compacted):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    values = [
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/ordinary.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(ordinary),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="clips/clip-a/selected/local.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="local",
            visible_region="upper_body",
            synthetic=True,
            artifact_path=str(enriched),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="background",
            image_path="clips/clip-a/selected/background.png",
            source_frame_index=0,
            scope="scene",
            synthetic=False,
            artifact_path=str(background_path),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="attribute",
            image_path="clip-a/attribute_hair.png",
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=0,
            synthetic=False,
            artifact_path=str(attribute_path),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/subject.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(compacted),
        ),
    ]

    published = [FinalVisualReference.from_visual(value) for value in values]

    assert [item.image_path for item in published] == [
        value.image_path for value in values
    ]
    assert [item.image_artifact_path for item in published] == [
        str(ordinary),
        str(enriched),
        str(background_path),
        str(attribute_path),
        str(compacted),
    ]


def test_final_sample_supports_visual_assets_from_different_owner_roots(
    tmp_path: Path,
) -> None:
    subject_path = tmp_path / "visual-export/references/subject.png"
    attribute_path = (
        tmp_path / "visual-runs/subject_attributes/clip-a/attribute_hair.png"
    )
    for path in (subject_path, attribute_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    references = [
        FinalVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/subject.png",
            image_artifact_path=str(subject_path),
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
        ),
        FinalVisualReference(
            image_id="image_2",
            image_index=2,
            kind="attribute",
            image_path="clip-a/attribute_hair.png",
            image_artifact_path=str(attribute_path),
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=0,
            synthetic=False,
        ),
    ]

    sample = _final_sample(tmp_path, [], references=references)

    assert [Path(item.image_artifact_path).read_bytes() for item in sample.visual_references] == [
        b"subject.png",
        b"attribute_hair.png",
    ]


def test_final_visual_reference_rejects_relative_or_missing_artifact_path(
    tmp_path: Path,
) -> None:
    values = {
        "image_id": "image_1",
        "image_index": 1,
        "kind": "subject",
        "image_path": "references/subject.png",
        "entity_id": "entity_1",
        "source_frame_index": 0,
        "scope": "full",
        "visible_region": "whole",
        "synthetic": False,
    }
    with pytest.raises(ValueError, match="must be absolute"):
        FinalVisualReference(**values, image_artifact_path="subject.png")
    with pytest.raises(ValueError, match="must be an existing file"):
        FinalVisualReference(
            **values,
            image_artifact_path=str(tmp_path / "missing.png"),
        )


def test_resolved_path_contract_requires_new_final_sample_schema(
    tmp_path: Path,
) -> None:
    sample = _final_sample(tmp_path, [_voice(1)])
    payload = sample.model_dump(mode="json")

    assert sample.schema_version == "r2v.h3.final_sample.5"
    without_artifact = json.loads(json.dumps(payload))
    without_artifact["visual_references"][0].pop("image_artifact_path")
    with pytest.raises(ValueError, match="image_artifact_path"):
        FinalH3SampleV2.model_validate(without_artifact)

    old_version = json.loads(json.dumps(payload))
    old_version["schema_version"] = "r2v.h3.final_sample.2"
    with pytest.raises(ValueError, match="r2v.h3.final_sample.5"):
        FinalH3SampleV2.model_validate(old_version)


def test_final_sample_allows_partial_subject_voice_coverage(tmp_path: Path) -> None:
    sample = _final_sample(tmp_path, [_voice(1)])

    assert sample.schema_version == "r2v.h3.final_sample.5"
    assert [item.entity_id for item in sample.visual_references] == [
        "entity_1",
        "entity_2",
    ]
    assert [item.entity_id for item in sample.subject_voices] == ["entity_1"]


def test_final_sample_still_allows_all_subjects_to_have_voice(tmp_path: Path) -> None:
    sample = _final_sample(tmp_path, [_voice(1), _voice(2)])

    assert [item.entity_id for item in sample.subject_voices] == [
        "entity_1",
        "entity_2",
    ]


def test_final_sample_rejects_noncanonical_or_duplicate_voice_bindings(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only canonical subject references"):
        _final_sample(tmp_path, [_voice(3)])
    with pytest.raises(ValueError, match="unique entity IDs"):
        _final_sample(tmp_path, [_voice(1), _voice(1)])


def test_final_sample_validates_voice_occurrence_and_subject_index(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="target occurrence is inconsistent"):
        _final_sample(
            tmp_path,
            [_voice(1, target_occurrence_id="other/entity_1")],
        )
    with pytest.raises(ValueError, match="canonical subject order"):
        _final_sample(tmp_path, [_voice(1, subject_index=2)])


def test_final_renderer_uses_exact_canonical_instruction_and_ordered_references(
    tmp_path: Path,
) -> None:
    rows = _render(tmp_path)
    assert {row["schema_version"] for row in rows} == {"r2v.h3.final_sample.5"}
    summary = json.loads((tmp_path / "h3/summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "r2v.h3.final_summary.6"
    assert summary["final_sample_schema_version"] == "r2v.h3.final_sample.5"
    first = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    assert first["r2v_instruction"] == "canonical clip-a: Image 1 and Image 2"
    assert [item["kind"] for item in first["visual_references"]] == [
        "subject",
        "attribute",
    ]
    assert first["visual_references"][1]["attribute_id"] == "attribute_hair"
    assert first["visual_references"][1]["owner_entity_id"] == "entity_1"
    assert first["visual_references"][0]["scope"] == "local"
    assert first["visual_references"][0]["visible_region"] == "upper_body"
    assert not isinstance(first["visual_references"][0]["visible_region"], dict)
    assert first["visual_references"][0]["image_path"].endswith("subject-1.png")
    assert Path(first["visual_references"][0]["image_artifact_path"]).is_file()
    assert "full_audio" in Path(first["target_full_audio_path"]).parts
    assert first["target_audio_sample_rate_hz"] == 32000
    assert first["target_audio_channels"] == 2
    target_voice = first["subject_voices"][0]
    assert "primary_voice" in Path(target_voice["voice_reference_path"]).parts
    assert target_voice["voice_sample_rate_hz"] == 32000
    assert target_voice["voice_channels"] == 2
    assert target_voice["source_start_sample"] == round(
        target_voice["source_start"] * 32000
    )
    assert target_voice["source_end_sample"] == round(
        target_voice["source_end"] * 32000
    )
    second = next(row for row in rows if row["sample_id"] == "clip-b/in_pair")
    assert second["visual_references"][0]["scope"] == "full"
    assert second["visual_references"][0]["visible_region"] == "whole"
    assert len(first["subject_voices"]) == 1
    cross = next(row for row in rows if row["sample_id"] == "clip-a/cross_pair/1")
    donor_voice = cross["subject_voices"][0]
    assert donor_voice["voice_source"] == "cross_donor"
    assert "primary_voice" in Path(donor_voice["voice_reference_path"]).parts
    assert donor_voice["voice_sample_rate_hz"] == 32000
    assert donor_voice["voice_channels"] == 2


def test_final_renderer_projects_canonical_audio_semantics_for_every_variant(
    tmp_path: Path,
) -> None:
    _render(tmp_path)
    inventory = _visual_inventory(tmp_path)
    canonical = [
        CanonicalAudioClip.model_validate_json(line)
        for line in (tmp_path / "audio/canonical_clips.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    semantics_root = tmp_path / "audio_semantics_specialized_v1"
    _jsonl(
        semantics_root / "assembled/records.jsonl",
        [
            _specialized_semantics_record(item, media_root=tmp_path)
            for item in canonical
        ],
    )

    summary = render_jea_final_samples(
        visual_inventory=inventory,
        audio_root=tmp_path / "audio",
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        binding_audit_root=tmp_path / "binding_audit_v1",
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3-with-semantics",
        audio_semantics_root=semantics_root,
    )

    samples = [
        json.loads(line)
        for line in (tmp_path / "h3-with-semantics/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for clip_uid in {item["clip_uid"] for item in samples}:
        clip_semantics = {
            json.dumps(item["full_clip_audio_semantics"], sort_keys=True)
            for item in samples
            if item["clip_uid"] == clip_uid
        }
        assert len(clip_semantics) == 1
    assert all(item["full_clip_audio_semantics"] is not None for item in samples)
    assert all(
        "raw_audio_caption" not in item["full_clip_audio_semantics"]
        for item in samples
    )
    assert summary.audio_semantics_available_clip_count == len(canonical)
    assert summary.audio_semantics_complete_clip_count == len(canonical)
    assert summary.audio_semantics_missing_clip_count == 0
    assert summary.source_audio_semantics_records_sha256 == _sha256(
        semantics_root / "assembled/records.jsonl"
    )


def test_final_renderer_publishes_all_canonical_clips_with_optional_voice_variants(
    tmp_path: Path,
) -> None:
    inventory = _canonical_wide_visual_inventory(tmp_path)
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory)[:2],
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    for name in ("in_pairs.jsonl", "cross_pairs.jsonl"):
        path = tmp_path / "pairs" / name
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = [row for row in rows if row["target_clip_uid"] == "clip-a"]
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in selected
            ),
            encoding="utf-8",
        )
    bound = []
    qwen = []
    for clip in inventory.canonical_clips[:2]:
        clip_uid = clip.identity.clip_uid
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid=clip_uid,
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                start_time=0.0,
                end_time=0.1,
                source_start_sample=0,
                source_end_sample=3200,
                cluster_binding_status="candidate_mapped",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip_uid}/entity_1",
                direct_anchor_samples=100,
                direct_anchor_seconds=0.00625,
                identity_scope="direct_anchor_present",
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **clip.identity.model_dump(mode="python"),
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip_uid}/entity_1",
                source_audio_path=str(tmp_path / f"{clip_uid}.flac"),
                source_start_sample=0,
                source_end_sample=3200,
                source_sample_rate_hz=32000,
                start_time=0.0,
                end_time=0.1,
                status="transcribed",
                text=f"speech from {clip_uid}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)
    _write_canonical_sources(
        tmp_path,
        inventory,
        clips_with_segments={"clip-a", "clip-b"},
    )
    audit_root = _write_binding_audit(
        tmp_path,
        bound,
        clip_count=len(inventory.canonical_clips),
    )

    summary = render_jea_final_samples(
        visual_inventory=inventory,
        audio_root=tmp_path / "audio",
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        binding_audit_root=audit_root,
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )

    samples = [
        FinalH3SampleV2.model_validate_json(line)
        for line in (tmp_path / "h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    canonical = [item for item in samples if item.pair_type == "canonical"]
    assert [item.clip_uid for item in canonical] == ["clip-a", "clip-b", "clip-c"]
    assert all(not item.subject_voices for item in canonical)
    assert [item.pair_type for item in samples if item.clip_uid == "clip-a"] == [
        "canonical",
        "in_pair",
        "cross_pair",
    ]
    assert [item.pair_type for item in samples if item.clip_uid == "clip-b"] == [
        "canonical"
    ]
    clip_c = next(item for item in canonical if item.clip_uid == "clip-c")
    assert clip_c.speech_segments == []
    assert summary.canonical_clip_count == len(inventory.canonical_clips)
    assert summary.canonical_base_sample_count == len(inventory.canonical_clips)
    assert summary.in_pair_sample_count == 1
    assert summary.cross_pair_sample_count == 1
    assert summary.final_sample_count == len(inventory.canonical_clips) + 2
    assert summary.canonical_clips_without_target_voice_variant_count == 2
    assert summary.canonical_clips_with_empty_speech_count == 1
    assert summary.audio_semantics_available_clip_count == 0
    assert summary.audio_semantics_missing_clip_count == len(
        inventory.canonical_clips
    )
    assert (tmp_path / "h3/samples/节目/集合/episode_c_0003/canonical.json").is_file()


def test_final_renderer_preserves_unvoiced_subject_and_its_speech(tmp_path: Path) -> None:
    visual = _visual_clip(
        tmp_path,
        "clip-a",
        "episode_a_0001",
        subject_entity_ids=("entity_1", "entity_2"),
    )
    inventory = VisualProductionInventory(
        visual_production_root=str(tmp_path),
        visual_runs_root=str(tmp_path / "runs"),
        visual_input_schema="r2v.v3.production_sample.1",
        visual_input_mode="compacted_production",
        canonical_sample_count=1,
        eligible_clip_count=1,
        eligible_subject_occurrence_count=2,
        media_collection_count=1,
        media_collection_clip_counts={"节目/集合": 1},
        shard_count=1,
        canonical_clips=[visual],
        clips=[visual],
        skip_reason_counts={},
    )
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    bound = []
    qwen = []
    for index in (1, 2):
        entity_id = f"entity_{index}"
        segment_id = f"segment_{index:04d}"
        start_sample = (index - 1) * 3200
        end_sample = index * 3200
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid="clip-a",
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{index}",
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                cluster_binding_status="candidate_mapped",
                entity_id=entity_id,
                entity_occurrence_id=f"clip-a/{entity_id}",
                direct_anchor_samples=100,
                direct_anchor_seconds=0.00625,
                identity_scope="direct_anchor_present",
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **visual.identity.model_dump(mode="python"),
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{index}",
                entity_id=entity_id,
                entity_occurrence_id=f"clip-a/{entity_id}",
                source_audio_path=str(tmp_path / "clip-a.flac"),
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                source_sample_rate_hz=32000,
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                status="transcribed",
                text=f"speech from {entity_id}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)
    _write_canonical_sources(
        tmp_path,
        inventory,
        clips_with_segments={item.target_clip_uid for item in bound},
    )
    audit_root = _write_binding_audit(
        tmp_path,
        bound,
        clip_count=inventory.canonical_sample_count,
    )

    render_jea_final_samples(
        visual_inventory=inventory,
        audio_root=tmp_path / "audio",
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        binding_audit_root=audit_root,
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    row = next(item for item in rows if item["pair_type"] == "in_pair")
    assert [item["entity_id"] for item in row["visual_references"][:2]] == [
        "entity_1",
        "entity_2",
    ]
    assert [item["entity_id"] for item in row["subject_voices"]] == ["entity_1"]
    assert [item["entity_id"] for item in row["speech_segments"]] == [
        "entity_1",
        "entity_2",
    ]


def test_final_renderer_only_publishes_directly_anchored_segment_identity(
    tmp_path: Path,
) -> None:
    inventory = _visual_inventory(tmp_path)
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    identity = inventory.clips[0].identity
    statuses = (
        ("direct", "candidate_mapped", "entity_1", 100, "direct_anchor_present"),
        (
            "propagated",
            "candidate_mapped",
            "entity_1",
            0,
            "cluster_propagated_only",
        ),
        ("conflict", "conflict", None, 0, "unresolved"),
        ("ambiguous", "ambiguous", None, 0, "unresolved"),
        ("unbound", "unbound", None, 0, "unresolved"),
    )
    bound = []
    qwen = []
    for index, (name, status, entity_id, direct_samples, scope) in enumerate(
        statuses, start=1
    ):
        segment_id = f"segment_{index:04d}"
        start_sample = (index - 1) * 3200
        end_sample = index * 3200
        occurrence_id = (
            f"{identity.clip_uid}/{entity_id}" if entity_id is not None else None
        )
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid=identity.clip_uid,
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{name}",
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                cluster_binding_status=status,
                entity_id=entity_id,
                entity_occurrence_id=occurrence_id,
                direct_anchor_samples=direct_samples,
                direct_anchor_seconds=direct_samples / 32000,
                identity_scope=scope,
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **identity.model_dump(mode="python"),
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{name}",
                entity_id=entity_id,
                entity_occurrence_id=occurrence_id,
                source_audio_path=str(tmp_path / f"{identity.clip_uid}.flac"),
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                source_sample_rate_hz=32000,
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                status="transcribed",
                text=f"speech {name}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)
    _write_canonical_sources(
        tmp_path,
        inventory,
        clips_with_segments={item.target_clip_uid for item in bound},
    )
    audit_root = _write_binding_audit(
        tmp_path,
        bound,
        clip_count=inventory.canonical_sample_count,
    )

    render_jea_final_samples(
        visual_inventory=inventory,
        audio_root=tmp_path / "audio",
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        binding_audit_root=audit_root,
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sample = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    assert [item["entity_id"] for item in sample["speech_segments"]] == [
        "entity_1",
        None,
        None,
        None,
        None,
    ]
    assert [item["entity_occurrence_id"] for item in sample["speech_segments"]] == [
        "clip-a/entity_1",
        None,
        None,
        None,
        None,
    ]
    propagated = sample["speech_segments"][1]
    assert propagated["speaker_cluster_id"] == "speaker_propagated"
    assert propagated["start_time"] == 0.1
    assert propagated["end_time"] == 0.2
    assert propagated["text"] == "speech propagated"
    assert propagated["language"] == "en"
    assert "subject_id" not in propagated
    assert "rendered_dialogue" not in propagated
    summary = json.loads((tmp_path / "h3/summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "r2v.h3.final_summary.6"
    assert summary["entity_bindings_removed_by_direct_anchor_gate"] == 1
    assert summary["speaker_binding_audit_policy_version"] == (
        "h3_speaker_binding_structural_audit_v1"
    )


def test_final_cross_pair_swaps_only_donor_voice(tmp_path: Path) -> None:
    rows = _render(tmp_path)
    in_pair = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    cross = next(row for row in rows if row["sample_id"] == "clip-a/cross_pair/1")
    for key in (
        "target_video",
        "target_full_audio_path",
        "r2v_instruction",
        "visual_references",
        "speech_segments",
    ):
        assert cross[key] == in_pair[key]
    assert (
        cross["subject_voices"][0]["voice_reference_path"]
        != in_pair["subject_voices"][0]["voice_reference_path"]
    )
    assert cross["subject_voices"][0]["voice_source"] == "cross_donor"


def test_final_renderer_publishes_qwen_text_without_confidence(tmp_path: Path) -> None:
    rows = _render(tmp_path)
    segment = rows[0]["speech_segments"][0]
    assert segment["text"].startswith("raw clip-")
    assert segment["language"] == "en"
    assert "confidence" not in segment
    assert "language_probability" not in segment


def test_multi_subject_cross_mapping_is_one_to_one_and_capped_at_one(
    tmp_path: Path,
) -> None:
    inventory = _visual_inventory(tmp_path)
    occurrences = []
    for clip in inventory.clips:
        for index in (1, 2):
            occurrences.append(
                JEAOccurrenceEmbedding(
                    occurrence_id=f"{clip.identity.clip_uid}/entity_{index}",
                    clip_uid=clip.identity.clip_uid,
                    entity_id=f"entity_{index}",
                    subject_index=index,
                    identity=clip.identity,
                    visual_reference_path=f"visual/{clip.identity.clip_uid}/e{index}.png",
                    primary_voice_reference_path=str(
                        tmp_path
                        / "primary_voice"
                        / clip.identity.clip_uid
                        / f"e{index}.flac"
                    ),
                    face_embedding=[1.0, 0.0],
                    voice_embedding=[1.0, 0.0],
                )
            )
    summary = build_jea_pairs(
        visual_inventory=inventory,
        occurrences=occurrences,
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    assert summary.cross_pair_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "pairs/cross_pairs.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["pair_id"].endswith("/1")
        donors = [item["donor_occurrence_id"] for item in row["mappings"]]
        assert len(donors) == len(set(donors)) == 2
