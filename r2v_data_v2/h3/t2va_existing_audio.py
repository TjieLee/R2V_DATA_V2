"""Select fixed T2VA cases from already-published Audio shadow records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from r2v_data_v2.h3.diarization_binding import RawDiarizationSegment
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.resolved_audio_stems import (
    downstream_stem_root,
    load_resolved_stems,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import stem_shadow_root
from r2v_data_v2.h3.t2va_shadow import (
    T2VABackendProvenance,
    T2VACaseManifest,
    T2VAInventory,
    T2VAJob,
    T2VASpeechFact,
    audio_evidence,
    fingerprint,
    read_rows,
    t2va_root,
)
from r2v_data_v2.h3.t2va_source import T2VAShot, T2VAShotSelection
from r2v_data_v2.naming import parse_clip_identity


def _selected_media_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"selected T2VA media is missing: {path}")


def build_t2va_inventory_from_existing_audio(
    *,
    audio_production_root: Path,
    audio_shadow_run_id: str,
    case_manifest: Path,
    t2va_run_id: str,
    backend: T2VABackendProvenance,
) -> T2VAInventory:
    production = audio_production_root.expanduser().resolve(strict=True)
    t2va_root(production, t2va_run_id)
    case_path = case_manifest.expanduser().resolve(strict=True)
    case_bytes = case_path.read_bytes()
    selected = T2VACaseManifest.model_validate_json(case_bytes).clip_uids
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate fixed-case clip UID")

    canonical_path = production / "audio/canonical_clips.jsonl"
    canonical = read_rows(canonical_path, CanonicalAudioClip)
    canonical_by_uid = {clip.clip_uid: (index, clip) for index, clip in enumerate(canonical)}
    if len(canonical_by_uid) != len(canonical):
        raise ValueError("duplicate published canonical Audio clip UID")
    missing = [uid for uid in selected if uid not in canonical_by_uid]
    if missing:
        raise ValueError(f"fixed-case clip UID missing from published Audio: {missing}")

    resolved_root = downstream_stem_root(production, audio_shadow_run_id)
    resolved, stem_records, _ = load_resolved_stems(resolved_root)
    stems_by_uid = {record.clip_uid: record for record in stem_records}
    missing = [uid for uid in selected if uid not in stems_by_uid]
    if missing:
        raise ValueError(f"fixed-case clip UID missing from resolved stems: {missing}")

    shadow = stem_shadow_root(production, audio_shadow_run_id)
    raw_path = shadow / "diarization/raw_segments.jsonl"
    asr_path = shadow / "asr/segments.jsonl"
    raw = read_rows(raw_path, RawDiarizationSegment)
    asr = read_rows(asr_path, Qwen3ASRSegment)
    diarization_meta = json.loads((shadow / "diarization/stem_provenance.json").read_text())
    asr_meta = json.loads((shadow / "asr/stem_provenance.json").read_text())
    raw_by_uid: dict[str, list[RawDiarizationSegment]] = {}
    for segment in raw:
        raw_by_uid.setdefault(segment.target_clip_uid, []).append(segment)
    for segments in raw_by_uid.values():
        segments.sort(key=lambda item: (item.start_time, item.end_time, item.segment_id))
    asr_by_key = {(segment.clip_uid, segment.segment_id): segment for segment in asr}
    asr_ready = set(asr_meta["clip_uids"])

    shots, jobs = [], []
    for uid in selected:
        source_index, clip = canonical_by_uid[uid]
        identity = parse_clip_identity(clip.target_video_path)
        if identity.clip_uid != uid:
            raise ValueError("fixed-case published clip identity differs")
        _selected_media_exists(clip.target_video_path)
        _selected_media_exists(clip.target_full_audio_path)
        shots.append(
            T2VAShot(
                clip_uid=uid,
                source_index=source_index,
                source_row_sha256=fingerprint(clip.model_dump(mode="json")),
                source_video_id=identity.parent_video_id,
                shot_index=identity.clip_order[-1],
                clip_display_path=clip.clip_display_path,
                video_path=clip.target_video_path,
                video_sha256=clip.target_video_sha256,
                duration_seconds=clip.target_duration_seconds,
            )
        )
        reason = None if uid in asr_ready else "resolved_speech_unavailable"
        facts = []
        for segment in raw_by_uid.get(uid, ()):
            result = asr_by_key.get((uid, segment.segment_id))
            if result is None:
                raise ValueError(f"published ASR segment missing: {uid}/{segment.segment_id}")
            if result.status == "failed" or (
                result.status == "transcribed" and not result.language
            ):
                reason, facts = "asr_failed", []
                break
            facts.append(
                T2VASpeechFact(
                    segment_id=segment.segment_id,
                    source_speaker_cluster=segment.speaker_cluster_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    language=result.language,
                    text=result.text,
                )
            )
        record = stems_by_uid[uid]
        evidence = None
        if record.status == "ready":
            evidence = audio_evidence(clip, record, resolved_root, verify_files=False)
            for kind in ("speech", "music", "sfx"):
                _selected_media_exists(getattr(evidence, f"{kind}_path"))
        else:
            reason = reason or "resolved_audio_unavailable"
        jobs.append(
            T2VAJob(
                clip_uid=uid,
                clip_display_path=clip.clip_display_path,
                target_video_path=clip.target_video_path,
                target_video_sha256=clip.target_video_sha256,
                target_duration_seconds=clip.target_duration_seconds,
                speech_facts=facts,
                audio_evidence=evidence,
                upstream_failure=reason,
            )
        )

    # The legacy selection container identifies the published canonical rows here.
    # Its required roots are metadata, not inputs to this fixed-case selection.
    selection = T2VAShotSelection(
        shot_manifest_path=str(canonical_path),
        shot_manifest_sha256=resolved.source_canonical_audio_manifest_sha256,
        clips_root=str(production),
        source_videos_root=str(production),
        selection_mode="case_manifest",
        sample_seed=None,
        case_manifest_path=str(case_path),
        case_manifest_sha256=hashlib.sha256(case_bytes).hexdigest(),
        valid_row_count=len(canonical),
        excluded_rows=[],
        shots=shots,
    )
    values = {
        "shot_selection": selection.model_dump(mode="json"),
        "audio_production_root": str(production),
        "audio_shadow_run_id": audio_shadow_run_id,
        "t2va_run_id": t2va_run_id,
        "source_hashes": {
            str(canonical_path): resolved.source_canonical_audio_manifest_sha256,
            str(raw_path): diarization_meta["raw_segments_sha256"],
            str(asr_path): asr_meta["segments_sha256"],
            str(case_path): selection.case_manifest_sha256,
        },
        "selection_mode": "case_manifest",
        "sample_seed": None,
        "clip_uids": selected,
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "backend": backend.model_dump(mode="json"),
    }
    return T2VAInventory(**values, inventory_fingerprint=fingerprint({
        "schema_version": "r2v.h3.t2va_inventory.3", **values,
    }))
