"""JEA shot population and no-reference Audio bootstrap for T2VA."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal

import soundfile as sf
from pydantic import Field

from r2v_data_v2.h3.audio_backends import AudioMediaBackend
from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    DiarizationTargetClip,
    _inventory_fingerprint,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.jea_target_audio_caption import (
    AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import _publish_directory, sha256_file
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter


class T2VAShot(SchemaModel):
    clip_uid: str
    source_index: int = Field(ge=0)
    source_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_id: str
    shot_index: int = Field(ge=0)
    clip_display_path: str
    video_path: str
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)


class T2VAShotSelection(SchemaModel):
    schema_version: Literal["r2v.h3.t2va_shot_selection.1"] = (
        "r2v.h3.t2va_shot_selection.1"
    )
    shot_manifest_path: str
    shot_manifest_sha256: str
    clips_root: str
    source_videos_root: str
    selection_mode: Literal["all", "case_manifest", "random"]
    sample_seed: int | None
    case_manifest_path: str | None
    case_manifest_sha256: str | None
    valid_row_count: int
    excluded_rows: list[dict]
    shots: list[T2VAShot]


def select_t2va_shots(
    shot_manifest: Path,
    *,
    clips_root: Path | None = None,
    source_videos_root: Path | None = None,
    case_manifest: Path | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
) -> T2VAShotSelection:
    from r2v_data_v2.h3.t2va_shadow import fingerprint, select_clip_uids

    path = shot_manifest.expanduser().resolve(strict=True)
    source_hash = sha256_file(path)
    clips = (clips_root or path.parent).expanduser().resolve(strict=True)
    sources = (source_videos_root or path.parent).expanduser().resolve(strict=True)
    # Reuse the Video ingestion adapter and its path containment/identity checks.
    # Explicit roots also support local synthetic datasets without a public mount.
    adapter = JeaVideoMotionAdapter(clips_root=clips, source_videos_root=sources)
    rows, excluded = {}, []
    for index, raw in enumerate(iter_source_records(path)):
        try:
            item = adapter.parse(raw, source_index=index)
            duration = float(raw["duration"])
            if not 0 < duration < float("inf"):
                raise ValueError("shot duration must be finite and positive")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            excluded.append({"source_index": index, "reason": str(exc)})
            continue
        uid = item["clip_uid"]
        if uid in rows:
            raise ValueError("duplicate JEA shot clip identity")
        rows[uid] = {
            "clip_uid": uid,
            "source_index": index,
            "source_row_sha256": fingerprint(raw),
            "source_video_id": item["parent_video_id"],
            "shot_index": raw["shot_index"],
            "clip_display_path": str(
                Path(item["metadata"]["source_relative_video_path"]).with_suffix("")
            ),
            "video_path": item["video_path"],
            "duration_seconds": duration,
        }
    selected = select_clip_uids(
        list(rows),
        case_manifest=case_manifest,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    shots = [
        T2VAShot(**rows[uid], video_sha256=sha256_file(Path(rows[uid]["video_path"])))
        for uid in selected
    ]
    if sha256_file(path) != source_hash:
        raise ValueError("JEA shot manifest changed during selection")
    return T2VAShotSelection(
        shot_manifest_path=str(path),
        shot_manifest_sha256=source_hash,
        clips_root=str(clips),
        source_videos_root=str(sources),
        selection_mode="case_manifest"
        if case_manifest
        else "random"
        if sample_size is not None
        else "all",
        sample_seed=sample_seed,
        case_manifest_path=str(case_manifest.resolve()) if case_manifest else None,
        case_manifest_sha256=sha256_file(case_manifest) if case_manifest else None,
        valid_row_count=len(rows),
        excluded_rows=excluded,
        shots=shots,
    )


def validate_cached_target(shot: T2VAShot, clip: CanonicalAudioClip) -> None:
    if (
        shot.clip_uid != clip.clip_uid
        or shot.video_path != clip.target_video_path
        or shot.video_sha256 != clip.target_video_sha256
        or abs(shot.duration_seconds - clip.target_duration_seconds)
        > AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "T2VA selected JEA shot differs from cached Audio identity/timeline"
        )


def prepare_t2va_audio(
    selection: T2VAShotSelection,
    *,
    output_root: Path,
    audio_backend: AudioMediaBackend,
) -> Path:
    """Create a NEW caller-owned Audio workspace; never modify an existing population."""
    from r2v_data_v2.h3.t2va_shadow import write_json

    destination = output_root.expanduser().resolve()
    if destination.exists() or output_root.is_symlink():
        raise FileExistsError(destination)
    inputs = [
        Path(selection.shot_manifest_path),
        *(Path(s.video_path) for s in selection.shots),
    ]
    if any(destination == p or destination in p.parents for p in inputs):
        raise ValueError("T2VA Audio workspace overlaps source media")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir(parents=True)
        clips = []
        for shot in selection.shots:
            if sha256_file(Path(shot.video_path)) != shot.video_sha256:
                raise ValueError("selected JEA target video changed")
            relative = Path("audio/full_audio") / f"{shot.clip_uid}.flac"
            actual = temporary / relative
            audio_backend.materialize_full_audio(
                clip_uid=shot.clip_uid,
                source_video_path=Path(shot.video_path),
                destination=actual,
                sample_rate_hz=32000,
                channels=2,
                output_format="flac",
            )
            info = sf.info(actual)
            if (info.format, info.samplerate, info.channels) != (
                "FLAC",
                32000,
                2,
            ) or info.frames <= 0:
                raise ValueError("T2VA canonical Audio media contract differs")
            clip = CanonicalAudioClip(
                clip_uid=shot.clip_uid,
                clip_display_path=shot.clip_display_path,
                media_collection_relpath="jea",
                media_collection_name="jea",
                episode_name=shot.source_video_id,
                clip_name=Path(shot.video_path).stem,
                shard_id="t2va",
                target_video_path=shot.video_path,
                target_video_sha256=shot.video_sha256,
                target_full_audio_path=str(destination / relative),
                target_full_audio_sha256=sha256_file(actual),
                frame_count=info.frames,
                target_duration_seconds=info.frames / 32000,
                subject_reference_count=0,
            )
            validate_cached_target(shot, clip)
            clips.append(clip)
        canonical = temporary / "audio/canonical_clips.jsonl"
        canonical.write_text("".join(c.model_dump_json() + "\n" for c in clips))
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
        values = {
            "source_pairs_sha256": None,
            "source_asr_inventory_fingerprint": None,
            "mode": "production",
            "targets": targets,
            "source_inventory_kind": "jea_shot_manifest",
            "source_shot_manifest_sha256": selection.shot_manifest_sha256,
            "source_canonical_audio_manifest_sha256": sha256_file(canonical),
        }
        inventory = DiarizationInventory(
            schema_version="r2v.h3.diarization_inventory.5",
            mode="production",
            source_inventory_kind="jea_shot_manifest",
            source_shot_manifest_path=selection.shot_manifest_path,
            source_shot_manifest_sha256=selection.shot_manifest_sha256,
            source_canonical_audio_manifest_path=str(
                destination / "audio/canonical_clips.jsonl"
            ),
            source_canonical_audio_manifest_sha256=sha256_file(canonical),
            inventory_fingerprint=_inventory_fingerprint(**values),
            source_target_count=len(targets),
            selected_target_count=len(targets),
            selection_mode="selected_jea_shot_inventory_v1",
            bounded_selection_applied=False,
            targets=targets,
        )
        write_json(temporary / "diarization/inventory.json", inventory)
        write_json(temporary / "selection.json", selection)
        write_json(
            temporary / "case_manifest.json",
            {
                "schema_version": "r2v.h3.mimo25_case_manifest.1",
                "clip_uids": [s.clip_uid for s in selection.shots],
            },
        )
        if (
            sha256_file(Path(selection.shot_manifest_path))
            != selection.shot_manifest_sha256
        ):
            raise ValueError("JEA shot manifest changed during Audio preparation")
        for shot in selection.shots:
            if sha256_file(Path(shot.video_path)) != shot.video_sha256:
                raise ValueError("selected JEA target video changed")
        _publish_directory(temporary, destination, overwrite=False)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
