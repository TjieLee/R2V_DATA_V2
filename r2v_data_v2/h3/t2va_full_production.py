"""Node-owned full Audio production; semantic producers stay frozen."""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
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
    T2VAShot,
    T2VAShotSelection,
    prepare_t2va_audio,
    validate_cached_target,
)

RUN_ID = "production"
STAGES = ("canonical", "sam", "auk", "resolve", "diarizen", "asr", "mimo")


@dataclass(frozen=True)
class PreparedShard:
    audio_root: Path
    summary: dict


def resume_first_assigned_shards(root, shard_ids):
    """Prioritize existing incomplete shard directories without content reads.

    Cross-node ownership is fixed before this call. Scheduling performs one
    directory enumeration of shards/, then inspects only existing assigned shard
    directories for a top-level COMPLETE marker. It never opens stage JSON,
    receipts, media, or hashes.
    """
    root = Path(root)
    assigned = list(shard_ids)
    assigned_by_name = {production.shard_name(shard_id): shard_id for shard_id in assigned}
    existing = {}
    shards_root = root / "shards"
    if shards_root.is_dir():
        with os.scandir(shards_root) as entries:
            for entry in entries:
                shard_id = assigned_by_name.get(entry.name)
                if shard_id is not None and entry.is_dir(follow_symlinks=False):
                    existing[shard_id] = Path(entry.path)

    complete = set()
    for shard_id, shard in existing.items():
        with os.scandir(shard) as entries:
            if any(
                entry.name == "COMPLETE" and entry.is_file(follow_symlinks=False)
                for entry in entries
            ):
                complete.add(shard_id)

    unfinished = [
        shard_id for shard_id in assigned if shard_id in existing and shard_id not in complete
    ]
    fresh = [shard_id for shard_id in assigned if shard_id not in existing]
    return unfinished + fresh, {
        "unfinished": len(unfinished),
        "fresh": len(fresh),
        "complete_skipped": len(complete),
    }


def run_assigned_shards(root, shard_ids, pipeline):
    """One node owns a shard through all barriers, including downstream writes."""
    results = {}
    for position, shard_id in enumerate(shard_ids):
        shard = root / "shards" / production.shard_name(shard_id)
        try:
            if hasattr(pipeline, "prepare_ahead"):
                pipeline.prepare_ahead(
                    shard_id,
                    shard_ids[position + 1]
                    if position + 1 < len(shard_ids)
                    else None,
                )
            with production.file_lock(shard / "invocation.lock"):
                results[shard_id] = {}
                for name in STAGES:
                    log_name = {
                        "canonical": "canonical_audio",
                        "mimo": "mimo-downstream",
                    }.get(name, name)
                    log_path = shard / "logs" / f"{log_name}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    try:
                        result = pipeline.stage(name, shard_id)
                    except BaseException:
                        with log_path.open("a") as handle:
                            traceback.print_exc(file=handle)
                        raise
                    if hasattr(result, "model_dump"):
                        result = result.model_dump(mode="json")
                    production.atomic_json(
                        shard / "stage_state" / f"{name}.json", result
                    )
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(result, sort_keys=True) + "\n")
                    counters = {k: v for k, v in result.items() if k != "clip_uids"}
                    results[shard_id][name] = counters
                    print(
                        f"shard={shard_id} stage={name} elapsed_seconds={time.monotonic() - started:.3f} "
                        f"{json.dumps(counters, sort_keys=True)}",
                        flush=True,
                    )
        except production.ShardLockedError:
            print(f"shard={shard_id} locked; skipped", flush=True)
    return results


class FullPipeline:
    def __init__(
        self,
        *,
        root,
        index,
        clips_root,
        source_videos_root,
        gpu_ids,
        sam_configuration,
        auk_configuration,
        backend,
        profiles,
        allow_unverified=False,
        request_workers=1,
        canonical_workers=16,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        mimo_lifecycle=None,
    ):
        self.root, self.index = root, index
        self.clips_root, self.source_videos_root = clips_root, source_videos_root
        self.gpu_ids = gpu_ids
        self.sam_configuration, self.auk_configuration = (
            sam_configuration,
            auk_configuration,
        )
        self.backend, self.profiles = backend, profiles
        self.allow_unverified, self.request_workers = allow_unverified, request_workers
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe
        self.mimo_lifecycle = mimo_lifecycle
        if canonical_workers < 1:
            raise ValueError("canonical workers must be positive")
        self.canonical_workers = canonical_workers
        self.prepared = {}
        self.prefetch = None
        self.pools = None

    @contextmanager
    def node(self, shard_ids):
        from r2v_data_v2.h3.t2va_full_prefetch import CanonicalPrefetch

        with CanonicalPrefetch(
            root=self.root,
            index=self.index,
            clips_root=self.clips_root,
            source_videos_root=self.source_videos_root,
            canonical_workers=self.canonical_workers,
            ffmpeg=self.ffmpeg,
            ffprobe=self.ffprobe,
        ) as prefetch:
            self.prefetch = prefetch
            try:
                if shard_ids:
                    prefetch.start(shard_ids[0])
                with self:
                    yield self
            finally:
                self.prefetch = None

    def prepare_ahead(self, shard_id, next_shard_id):
        if self.prefetch is None:
            return
        try:
            result = self.prefetch.wait(shard_id)
        except production.ShardLockedError:
            if next_shard_id is not None:
                self.prefetch.start(next_shard_id)
            raise
        self.prepared[shard_id] = PreparedShard(
            Path(result["audio_root"]), result["summary"]
        )
        if next_shard_id is not None:
            self.prefetch.start(next_shard_id)

    def __enter__(self):
        # MiMo is owned by the outer launcher and remains resident. Upstream
        # SAM/AuK/DiariZen/ASR workers are intentionally stage-local: stage()
        # leaves execution unset so the existing ephemeral executor loads one
        # worker per GPU for the active stage and releases it at the barrier.
        self.pools = None
        return self

    def __exit__(self, *_args):
        self.pools = None
        return False

    def stage(self, name, shard_id):
        from r2v_data_v2.h3 import auk_speech_shadow as auk
        from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
        from r2v_data_v2.h3 import t2va_full_stems as stems
        from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
        from r2v_data_v2.h3.resolved_audio_stems import resolve_audio_stems

        shard = self.root / "shards" / production.shard_name(shard_id)
        state = shard / "stage_state"
        execution = {}
        if name == "canonical":
            if self.prefetch is not None and shard_id in self.prepared:
                # Re-read after taking invocation ownership: another node may
                # have recovered canonical failures since our prefetch finished.
                audio_root = self.prepared[shard_id].audio_root
                selection = T2VAShotSelection.model_validate_json(
                    (shard / "source/selection.json").read_text()
                )
                count = sum(
                    1
                    for _ in production.complete_rows(
                        audio_root / "audio/canonical_clips.jsonl"
                    )
                )
                summary = {
                    "ready": count,
                    "failed": len(selection.shots) - count,
                    "skipped": len(selection.excluded_rows),
                }
                self.prepared[shard_id] = PreparedShard(audio_root, summary)
                return summary
            with production.file_lock(shard / "canonical.lock", blocking=True):
                selection = shard_selection(
                    self.root,
                    self.index,
                    shard_id,
                    self.clips_root,
                    self.source_videos_root,
                )
                audio_root = bootstrap_audio(
                    shard,
                    selection,
                    FFmpegAudioMediaBackend(ffmpeg=self.ffmpeg, ffprobe=self.ffprobe),
                    canonical_workers=self.canonical_workers,
                )
                count = sum(
                    1
                    for _ in production.complete_rows(
                        audio_root / "audio/canonical_clips.jsonl"
                    )
                )
                summary = {
                    "ready": count,
                    "failed": len(selection.shots) - count,
                    "skipped": len(selection.excluded_rows),
                }
                self.prepared[shard_id] = PreparedShard(audio_root, summary)
                return summary
        prepared = self.prepared[shard_id]
        audio_root = prepared.audio_root
        if not prepared.summary["ready"] and name != "mimo":
            return {"ready": 0, "skipped": "no canonical audio"}
        shadow = sam.stem_shadow_root(audio_root, RUN_ID)
        if name == "sam":
            inventory = sam.build_sam_audio_stem_inventory(
                canonical_audio_manifest_path=audio_root
                / "audio/canonical_clips.jsonl",
                model_configuration=self.sam_configuration,
                route="music_first",
            )
            return stems.run_sam(
                inventory,
                shadow / "separation",
                state / "sam",
                self.gpu_ids,
                ffmpeg=self.ffmpeg,
                ffprobe=self.ffprobe,
                **execution,
            )
        if name == "auk":
            _, records, _ = sam.load_stem_shadow(shadow / "separation")
            eligible = {r.clip_uid for r in records if r.separation_state != "failure"}
            inventory = auk.build_auk_inventory(
                audio_production_root=audio_root,
                shadow_run_id=RUN_ID,
                case_manifest_path=audio_root / "case_manifest.json",
                configuration=self.auk_configuration,
            )
            return stems.run_auk(
                inventory,
                state / "auk",
                self.gpu_ids,
                eligible=eligible,
                ffmpeg=self.ffmpeg,
                **execution,
            )
        if name == "resolve":
            return resolve_audio_stems(
                audio_production_root=audio_root,
                shadow_run_id=RUN_ID,
                overwrite=(shadow / "resolved_stems_v1").exists(),
            )
        if name in {"diarizen", "asr"}:
            from r2v_data_v2.h3.t2va_full_speech import run_asr, run_diarizen

            result = (run_diarizen if name == "diarizen" else run_asr)(
                audio_root,
                RUN_ID,
                state / name,
                self.gpu_ids,
                self.allow_unverified,
                ffmpeg=self.ffmpeg,
                **execution,
            )
            provenance = result["provenance"]
            return {
                "job_count": result["job_count"],
                "ready": len(
                    provenance.get("ready_clip_uids", provenance.get("clip_uids", []))
                ),
                "failed": len(provenance.get("failed_clip_uids", [])),
            }
        if name == "mimo":
            from contextlib import nullcontext

            from r2v_data_v2.h3.t2va_full_downstream import run_downstream

            lifecycle = (
                self.mimo_lifecycle.stage(shard_id)
                if self.mimo_lifecycle is not None
                else nullcontext()
            )
            with lifecycle:
                states = run_downstream(
                    self.root,
                    self.index,
                    shard_id,
                    audio_root,
                    self.clips_root,
                    self.source_videos_root,
                    self.backend,
                    self.profiles,
                    allow_unverified=self.allow_unverified,
                    request_workers=self.request_workers,
                    run_id=RUN_ID,
                )
            return {
                f"{stage}_{status}": sum(
                    row[f"{stage}_status"] == status for row in states.values()
                )
                for stage in ("t2va", "ta2va")
                for status in ("ready", "failed", "skipped", "pending")
            }
        raise ValueError(f"unknown stage {name}")


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
    from r2v_data_v2.h3.t2va_shadow import fingerprint
    from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter

    manifest = production.materialize_shard(index, shard_id, root)
    source = root / "shards" / production.shard_name(shard_id) / "source"
    cached = source / "selection.json"
    if cached.is_file():
        selection = T2VAShotSelection.model_validate_json(
            cached.read_text(encoding="utf-8")
        )
        if (
            Path(selection.shot_manifest_path).resolve() != manifest.resolve()
            or Path(selection.clips_root).resolve() != Path(clips_root).resolve()
            or (
                selection.source_videos_root is not None
                and source_videos_root is not None
                and Path(selection.source_videos_root).resolve()
                != Path(source_videos_root).resolve()
            )
        ):
            raise ValueError("cached shard selection identity differs")
        return selection
    start = shard_id * production.SHARD_SIZE
    adapter = JeaVideoMotionAdapter(
        clips_root=clips_root, source_videos_root=source_videos_root
    )
    shots, excluded, seen = [], [], set()
    with manifest.open("rb") as handle:
        for offset, line in enumerate(handle):
            source_index = start + offset
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("shot row must be an object")
                item = adapter.parse(raw, source_index=source_index)
                if item["clip_uid"] in seen:
                    raise ValueError("duplicate clip identity")
                shot = T2VAShot(
                    clip_uid=item["clip_uid"],
                    source_index=source_index,
                    source_row_sha256=fingerprint(raw),
                    source_video_id=item["parent_video_id"],
                    shot_index=raw["shot_index"],
                    clip_display_path=str(
                        Path(
                            item["metadata"]["source_relative_video_path"]
                        ).with_suffix("")
                    ),
                    video_path=item["video_path"],
                    video_sha256=sha256_file(Path(item["video_path"])),
                    duration_seconds=float(raw["duration"]),
                )
                shots.append(shot)
                seen.add(shot.clip_uid)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                excluded.append({"source_index": source_index, "reason": str(exc)})
    selection = T2VAShotSelection(
        shot_manifest_path=str(manifest),
        shot_manifest_sha256=sha256_file(manifest),
        clips_root=str(clips_root),
        source_videos_root=str(source_videos_root),
        selection_mode="all",
        sample_seed=None,
        case_manifest_path=None,
        case_manifest_sha256=None,
        valid_row_count=len(shots),
        excluded_rows=excluded,
        shots=shots,
    )
    production.atomic_json(source / "selection.json", selection.model_dump(mode="json"))
    production.atomic_json(
        source / "case_manifest.json",
        {"clip_uids": [s.clip_uid for s in selection.shots]},
    )
    return selection


def bootstrap_audio(
    shard: Path, selection: T2VAShotSelection, backend, *, canonical_workers=16
):
    """Reuse the frozen bootstrap per clip, then publish an ordered shard inventory."""
    if canonical_workers < 1:
        raise ValueError("canonical workers must be positive")
    audio = shard / "audio_production"
    items = audio / "canonical_items"

    def prepare(shot):
        destination = items / shot.clip_uid
        try:
            if not destination.exists():
                single = selection.model_copy(update={"shots": [shot]})
                prepare_t2va_audio(
                    single,
                    output_root=destination,
                    audio_backend=backend,
                    verify_source_videos=False,
                )
            rows = list(
                production.complete_rows(destination / "audio/canonical_clips.jsonl")
            )
            if len(rows) != 1:
                raise ValueError("canonical clip receipt differs")
            clip = CanonicalAudioClip.model_validate(rows[0])
            validate_cached_target(shot, clip)
            return clip, {"clip_uid": shot.clip_uid, "status": "ready"}
        except (ValueError, OSError, RuntimeError) as exc:
            return None, {
                "clip_uid": shot.clip_uid,
                "status": "failed",
                "reason": str(exc),
            }

    # map consumes results in source order, independently of completion order.
    with ThreadPoolExecutor(max_workers=canonical_workers) as executor:
        prepared = list(executor.map(prepare, selection.shots))
    clips = [clip for clip, _ in prepared if clip is not None]
    states = [state for _, state in prepared]
    # Per-clip canonical directories are already atomically durable. The summary
    # is derived once per barrier, rather than rewriting growing snapshots.
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
