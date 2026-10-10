"""Group-native publication using existing Audio/H3/JSONL business consumers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from r2v_data_v2.h3.audio_reuse import (
    AudioReuseManifest,
    MusicReuseAsset,
    SpeakerSpeechReuseAsset,
    _build_audio_reuse_assets,
)
from r2v_data_v2.h3.frame_conditioning_projection import (
    FrameMetadata,
    extract_frames,
    project_prompt,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionAudioContract,
    RecaptionPictureContract,
    RecaptionReferenceContract,
)
from r2v_data_v2.h3.speaker_ownership import two_step_identity_restricted_groups
from r2v_data_v2.h3.t2va_production import atomic_json
from r2v_data_v2.h3.training_manifest_export import (
    RA2VA_TASK_ORDER,
    build_ra2va_training_task_rows,
    write_training_manifests,
)
from r2v_data_v2.h3.voice_donor_reserve import _donor


# These native contracts explicitly carry unavailable legacy provenance as null,
# never a fabricated digest. All inherited ownership/timeline validators remain.
class GroupSpeakerAsset(SpeakerSpeechReuseAsset):
    schema_version: Literal["r2v.h3.group_speaker_speech_reuse_asset.1"] = "r2v.h3.group_speaker_speech_reuse_asset.1"
    source_stem_sha256: None = None
    source_stem_record_fingerprint: None = None
    target_audio_sha256: None = None
    source_job_fingerprint: None = None


class GroupMusicAsset(MusicReuseAsset):
    schema_version: Literal["r2v.h3.group_music_reuse_asset.1"] = "r2v.h3.group_music_reuse_asset.1"
    source_stem_sha256: None = None
    source_stem_record_fingerprint: None = None
    target_audio_sha256: None = None


class GroupReuseManifest(AudioReuseManifest):
    schema_version: Literal["r2v.h3.group_audio_reuse_manifest.1"] = "r2v.h3.group_audio_reuse_manifest.1"
    source_job_fingerprint: None = None
    source_stem_record_fingerprint: None = None
    speakers: list[GroupSpeakerAsset]
    music: GroupMusicAsset


class GroupH3Sample(FinalH3SampleV2):
    target_full_audio_sha256: None = None


class GroupPictureContract(RecaptionPictureContract):
    image_sha256: None = None


class GroupAudioContract(RecaptionAudioContract):
    sha256: None = None


class GroupReferenceContract(RecaptionReferenceContract):
    pictures: list[GroupPictureContract]
    audios: list[RecaptionAudioContract | GroupAudioContract]


class GroupFrameMetadata(FrameMetadata):
    source_video_sha256: None = None


def _sample(task, job):
    return GroupH3Sample(
        sample_id=f"{task.sample_id}/canonical", pair_id=task.sample_id, pair_type="canonical",
        clip_uid=task.clip_uid, clip_display_path=task.clip_uid,
        media_collection_relpath=task.sample["source"]["parent_video_id"],
        media_collection_name=task.sample["source"]["parent_video_id"],
        episode_name=Path(job.target_video_path).parent.name, clip_name=Path(job.target_video_path).stem,
        shard_id=task.sample["source"]["shard_id"], target_video=job.target_video_path,
        target_full_audio_path=job.target_full_audio_path, r2v_instruction=job.r2v_instruction,
        visual_references=[{**r, "image_artifact_path": p["image_path"]}
                           for r, p in zip(task.sample["references"], task.pictures, strict=True)],
        subject_voices=[], speech_segments=[{
            "segment_id": s.segment_id, "speaker_cluster_id": s.source_speaker_cluster_id,
            "entity_id": None, "entity_occurrence_id": None,
            **s.model_dump(include={"source_start_sample", "source_end_sample", "source_sample_rate_hz", "start_time", "end_time"}),
            "text": s.asr_text, "language": s.asr_language,
        } for s in job.segments if s.asr_status == "transcribed"],
    )


def _reference_contract(job):
    return GroupReferenceContract(subjects=job.reference_subjects, audios=[], pictures=[
        GroupPictureContract(**p.model_dump(include={"image_index", "picture_label", "kind", "entity_id",
            "attribute_id", "owner_entity_id", "attribute_type"}), image_id=p.source_image_id,
            image_path=p.image_artifact_path) for p in job.reference_images
    ])


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def export_group_clip(*, root, task, job, result, stems, ffmpeg="ffmpeg") -> dict:
    if result["status"] != "ready" or result.get("annotation") is None:
        raise ValueError("Group export requires a ready annotation")
    root = Path(root)
    summary_path = root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    annotation = MimoAVAnnotationDraft.model_validate(result["annotation"])
    provenance = SimpleNamespace(**result.get("postprocessing_backend_provenance", result["backend_provenance"]))
    record = SimpleNamespace(annotation=annotation, source_backend_provenance=provenance)
    legacy_job = SimpleNamespace(**job.__dict__, target_full_audio_sha256=None,
                                 target_video_sha256=None, request_fingerprint=None)
    stem_items = [SimpleNamespace(stem_type=kind, canonical_stem_path=stems[kind]["path"],
        canonical_frame_count=stems[kind]["frame_count"], canonical_stem_sha256=None,
        source_audio_path=job.target_full_audio_path, source_audio_sha256=None,
        source_end_sample=round(job.target_duration_seconds * 32000)) for kind in ("speech", "music", "sfx")]
    stem_record = SimpleNamespace(clip_uid=job.clip_uid, stems=stem_items, record_fingerprint=None,
        separation_state=stems["separation_state"], stem=lambda kind: next(s for s in stem_items if s.stem_type == kind))
    assets_root = root / "audio_reuse_assets_v1" / job.clip_uid
    manifest_path = assets_root / "manifest.json"
    if manifest_path.exists():
        manifest = GroupReuseManifest.model_validate_json(manifest_path.read_text())
    else:
        manifest = _build_audio_reuse_assets(job=legacy_job, annotation=annotation, stem_record=stem_record,
            audio_production_root=root, output_root=assets_root, backend_provenance=provenance,
            frozen_bundle=True, allow_unverified=True,
            asset_models=(GroupSpeakerAsset, GroupMusicAsset, GroupReuseManifest))
    sample, contract = _sample(task, job), _reference_contract(job)
    products = []
    restricted = two_step_identity_restricted_groups(annotation, provenance)
    speech_refs = []
    for asset in sorted(manifest.speakers, key=lambda a: int(a.speaker_id[1:]))[:3]:
        if asset.speaker_group in restricted:
            continue
        index = len(speech_refs) + 1
        speech_refs.append(RecaptionAudioContract(audio_index=index, audio_label=f"<Audio {index}>",
            kind="speaker_speech_reuse", path=asset.output_path, sha256=asset.output_sha256,
            speaker_id=asset.speaker_id, entity_id=asset.entity_id,
            subject_label=f"<Subject {asset.subject_index}>" if asset.entity_id else None,
            voice_characteristics=None, retention_marker="partially_copy"))
    music = annotation.h3_semantics.non_diegetic_music.strip()
    if music and music.casefold() not in {"n/a", "unknown"} and len(speech_refs) < 3:
        index = len(speech_refs) + 1
        speech_refs.append(RecaptionAudioContract(audio_index=index, audio_label=f"<Audio {index}>",
            kind="music_reuse", path=manifest.music.output_path, sha256=manifest.music.output_sha256,
            retention_marker="partially_copy"))
    for variant, audios in (("visual_only", []), ("target_speech_reuse", speech_refs),
        ("full_audio_reuse", [GroupAudioContract(audio_index=1, audio_label="<Audio 1>",
            kind="full_audio_reuse", path=job.target_full_audio_path, retention_marker="fully_copy")])):
        if variant == "target_speech_reuse" and not any(a.kind == "speaker_speech_reuse" for a in audios):
            continue
        failure, rendered, corrected = None, None, []
        try:
            corrected, rendered, warnings = _materialize_sample(sample, legacy_job, record,
                conditioning_variant=variant, reuse_audio_contracts=audios, reference_contract=contract)
        except ValueError as exc:
            failure, warnings = str(exc), []
        products.append({"sample_id": f"{task.sample_id}/{variant}", "source_h3_sample_id": sample.sample_id,
            "clip_uid": task.clip_uid, "pair_type": "canonical", "conditioning_variant": variant,
            "status": "failed" if failure else "ready", "failure_reason": failure,
            "audio_references": [{"contract": a.model_dump(mode="json")} for a in audios],
            "corrected_speech_segments": [s.model_dump(mode="json") for s in corrected],
            "rendered_h3_prompt": rendered, "warnings": warnings})
    frames_root = root / "frames"
    if not frames_root.exists():
        with tempfile.TemporaryDirectory(prefix=".frames-", dir=root) as temporary:
            stage = Path(temporary) / "frames"
            extract_frames(legacy_job, stage, frames_root, ffmpeg=ffmpeg, metadata_type=GroupFrameMetadata)
            stage.rename(frames_root)
    frames = json.loads((frames_root / "metadata.json").read_text())
    projected = []
    for product in products:
        if product["status"] != "ready":
            continue
        for mode in ("first_frame", "last_frame", "first_last_frame"):
            roles = ["first", "last"] if mode == "first_last_frame" else [mode.split("_")[0]]
            projected.append({**product, "visual_reference_mode": mode, "target_video_path": job.target_video_path,
                "frame_references": [{"picture_label": f"<Picture {i}>", "image_path": frames[f"{role}_frame_path"]}
                                     for i, role in enumerate(roles, 1)],
                "rendered_h3_prompt": project_prompt(product["rendered_h3_prompt"], mode, job.target_duration_seconds)})
    atomic_json(root / "audio_reuse_prepared_v1/inventory.json", {"jobs": [job.model_dump(mode="json")]})
    _write_rows(root / "h3_audio_reuse_products_v1/records.jsonl", products)
    _write_rows(root / "h3_frame_conditioned_products_v1/records.jsonl", projected)
    rows = build_ra2va_training_task_rows(root)
    training = {key: [r.row for r in value] for key, value in rows.items()}
    training_root = root / "training"
    if not training_root.exists():
        with tempfile.TemporaryDirectory(prefix=".training-", dir=root) as temporary:
            stage = Path(temporary) / "training"
            write_training_manifests(output_root=stage, rows=training, task_order=RA2VA_TASK_ORDER)
            stage.rename(training_root)
    donors = [_donor(root, job, record, asset, manifest_path) for asset in manifest.speakers
              if asset.entity_id is not None and asset.speaker_group not in restricted]
    reserve = {"schema_version": "r2v.h3.voice_donor_reserve.1", "donors": donors}
    reserve_path = root / "voice_donor_reserve.json"
    if not reserve_path.exists():
        atomic_json(reserve_path, reserve)
    summary = {"clip_uid": job.clip_uid, "product_ready_count": sum(p["status"] == "ready" for p in products),
        "product_failed_count": sum(p["status"] == "failed" for p in products),
        "failure_reasons": [p["failure_reason"] for p in products if p["status"] == "failed"],
        "task_counts": {key: len(value) for key, value in rows.items()}, "donor_count": len(donors),
        "donor_segment_count": sum(d["segment_count"] for d in donors),
        "identity_exclusions": [e.model_dump(mode="json") for e in manifest.exclusions],
        "bundle_root": str(root), "model_call_count": 0}
    atomic_json(summary_path, summary)
    return summary
