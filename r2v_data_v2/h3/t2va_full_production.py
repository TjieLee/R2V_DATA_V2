"""Node-owned full Audio production; semantic producers stay frozen."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from r2v_data_v2.h3 import t2va_production as production
from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    DiarizationTargetClip,
    _inventory_fingerprint,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file
from r2v_data_v2.h3.t2va_source import (
    T2VAShotSelection,
    prepare_t2va_audio,
    select_t2va_shots,
    validate_cached_target,
)

RUN_ID = "production"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w") as handle:
            for row in rows:
                if hasattr(row, "model_dump"):
                    row = row.model_dump(mode="json")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def shard_selection(root, index, shard_id, clips_root, source_videos_root):
    manifest = production.materialize_shard(index, shard_id, root)
    selection = select_t2va_shots(
        manifest, clips_root=clips_root, source_videos_root=source_videos_root
    )
    start = shard_id * production.SHARD_SIZE
    selection = selection.model_copy(
        update={
            "shots": [
                s.model_copy(update={"source_index": s.source_index + start})
                for s in selection.shots
            ],
            "excluded_rows": [
                {**r, "source_index": r["source_index"] + start}
                for r in selection.excluded_rows
            ],
        }
    )
    source = root / "shards" / production.shard_name(shard_id) / "source"
    production.atomic_json(source / "selection.json", selection.model_dump(mode="json"))
    production.atomic_json(
        source / "case_manifest.json",
        {"clip_uids": [s.clip_uid for s in selection.shots]},
    )
    return selection


def bootstrap_audio(shard: Path, selection: T2VAShotSelection, backend):
    """Reuse the frozen bootstrap per clip, then publish an ordered shard inventory."""
    audio = shard / "audio_production"
    items = audio / "canonical_items"
    clips, states = [], []
    for shot in selection.shots:
        destination = items / shot.clip_uid
        try:
            if not destination.exists():
                single = selection.model_copy(update={"shots": [shot]})
                prepare_t2va_audio(
                    single, output_root=destination, audio_backend=backend
                )
            rows = list(
                production.complete_rows(destination / "audio/canonical_clips.jsonl")
            )
            if len(rows) != 1:
                raise ValueError("canonical clip receipt differs")
            clip = CanonicalAudioClip.model_validate(rows[0])
            validate_cached_target(shot, clip)
            if (
                sha256_file(Path(clip.target_full_audio_path))
                != clip.target_full_audio_sha256
            ):
                raise ValueError("canonical audio changed")
            clips.append(clip)
            states.append({"clip_uid": shot.clip_uid, "status": "ready"})
        except (ValueError, OSError, RuntimeError) as exc:
            states.append(
                {"clip_uid": shot.clip_uid, "status": "failed", "reason": str(exc)}
            )
        production.atomic_json(shard / "stage_state/canonical_audio.json", states)
    canonical = audio / "audio/canonical_clips.jsonl"
    write_rows(canonical, clips)
    targets = [
        DiarizationTargetClip(
            target_clip_uid=c.clip_uid,
            target_video_path=c.target_video_path,
            source_audio_path=c.target_full_audio_path,
            source_audio_sha256=c.target_full_audio_sha256,
            source_sample_rate_hz=32000,
            source_channels=2,
            source_frame_count=c.frame_count,
            visual_references=[],
        )
        for c in clips
    ]
    values = dict(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
        source_inventory_kind="jea_shot_manifest",
        source_shot_manifest_sha256=selection.shot_manifest_sha256,
        source_canonical_audio_manifest_sha256=sha256_file(canonical),
    )
    inventory = DiarizationInventory(
        schema_version="r2v.h3.diarization_inventory.5",
        mode="production",
        source_inventory_kind="jea_shot_manifest",
        source_shot_manifest_path=selection.shot_manifest_path,
        source_shot_manifest_sha256=selection.shot_manifest_sha256,
        source_canonical_audio_manifest_path=str(canonical),
        source_canonical_audio_manifest_sha256=sha256_file(canonical),
        inventory_fingerprint=_inventory_fingerprint(**values),
        source_target_count=len(targets),
        selected_target_count=len(targets),
        selection_mode="selected_jea_shot_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )
    production.atomic_json(
        audio / "diarization/inventory.json", inventory.model_dump(mode="json")
    )
    production.atomic_json(
        audio / "case_manifest.json", {"clip_uids": [c.clip_uid for c in clips]}
    )
    return audio
