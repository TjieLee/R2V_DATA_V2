"""Standalone, model-free reserve of all eligible frozen speaker segments."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from r2v_data_v2.h3.audio_reuse import (
    SAMPLE_RATE,
    AudioReuseManifest,
    SpeakerSpeechReuseAsset,
    _fingerprint,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    load_prepared_inventory,
)
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.speaker_ownership import two_step_identity_restricted_groups

RESERVE_FILENAME = "voice_donor_reserve.json"
SEGMENTS_STAGE = "voice_donor_segments_v1"


def _read_pcm16(path: Path) -> np.ndarray:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        if (info.format, info.subtype, info.samplerate, info.channels) != ("FLAC", "PCM_16", SAMPLE_RATE, 2):
            raise ValueError(f"reserve requires PCM16 32 kHz stereo FLAC: {path}")
        pcm, rate = sf.read(str(path), dtype="int16", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError(f"unreadable reserve PCM16 FLAC: {path}") from exc
    if rate != SAMPLE_RATE or not len(pcm) or len(pcm) != info.frames:
        raise ValueError(f"invalid reserve PCM16 frame count: {path}")
    return pcm


@contextmanager
def _temporary_path(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _publish(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    # Linking a complete sibling file is atomic and cannot replace an old name.
    os.link(temporary, destination)


def _check_existing_crop(path: Path, pcm: np.ndarray) -> None:
    if not np.array_equal(_read_pcm16(path), pcm):
        raise FileExistsError(f"existing reserve crop differs from timeline: {path}")


def _export_crop(path: Path, pcm: np.ndarray) -> None:
    import soundfile as sf

    if os.path.lexists(path):
        _check_existing_crop(path, pcm)
        return
    with _temporary_path(path) as temporary:
        sf.write(str(temporary), pcm, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        try:
            _publish(temporary, path)
        except FileExistsError:
            _check_existing_crop(path, pcm)


def _donor(root: Path, job: MimoClipJob, record: AudioReusePreparedSource,
           asset: SpeakerSpeechReuseAsset, manifest_path: Path) -> dict:
    if asset.clip_uid != job.clip_uid or Path(asset.output_path).resolve().parent != manifest_path.parent:
        raise ValueError("reserve timeline asset differs from its bundle clip")
    subjects = [s for s in job.reference_subjects if s.kind == "entity" and s.entity_id == asset.entity_id]
    if len(subjects) != 1 or subjects[0].subject_index != asset.subject_index:
        raise ValueError("reserve entity/Subject differs from prepared job")
    known = {s.segment_id: s for s in job.segments}
    ranges = sorted(asset.source_sample_ranges, key=lambda r: (r.start_time, r.end_time, r.segment_id))
    for interval in ranges:
        if interval.segment_id not in known:
            raise ValueError(f"unknown reserve source segment: {interval.segment_id}")
        source = known[interval.segment_id]
        if any(getattr(source, name) != getattr(interval, name) for name in (
            "source_start_sample", "source_end_sample", "source_sample_rate_hz", "start_time", "end_time",
        )):
            raise ValueError(f"reserve boundary differs from prepared job: {interval.segment_id}")
    pcm = _read_pcm16(Path(asset.output_path))
    if len(pcm) != asset.target_frame_count:
        raise ValueError("reserve timeline frame count differs from manifest")
    segments = []
    for interval in ranges:
        path = root / SEGMENTS_STAGE / job.clip_uid / asset.speaker_group / f"{interval.segment_id}.flac"
        crop = pcm[interval.target_start_sample:interval.target_end_sample]
        if len(crop) != interval.target_end_sample - interval.target_start_sample:
            raise ValueError(f"reserve range exceeds timeline: {interval.segment_id}")
        _export_crop(path, crop)
        segments.append({
            "segment_id": interval.segment_id, "audio_path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "start_time": interval.start_time, "end_time": interval.end_time,
            "source_start_sample": interval.source_start_sample, "source_end_sample": interval.source_end_sample,
            "source_sample_rate_hz": interval.source_sample_rate_hz,
        })
    profiles = [p.voice_characteristics for p in record.annotation.speaker_voice_profiles
                if p.speaker_group == asset.speaker_group]
    return {
        "clip_uid": job.clip_uid, "entity_id": asset.entity_id,
        "donor_occurrence_id": f"{job.clip_uid}/{asset.entity_id}",
        "subject_label": f"<Subject {asset.subject_index}>", "speaker_group": asset.speaker_group,
        "speaker_id": asset.speaker_id, "voice_characteristics": profiles[0] if len(profiles) == 1 else None,
        "timeline_speech_asset_path": asset.output_path, "segment_count": len(segments),
        "total_speech_duration_seconds": sum(r.target_end_sample - r.target_start_sample for r in ranges) / SAMPLE_RATE,
        "segments": segments,
    }


def export_voice_donor_reserve(*, bundle_root: Path) -> dict:
    """Add immutable reserve sidecars to a prepared bundle, without upstream reads.

    A completed JSON sidecar is never replaced. Interrupted exports can reuse
    existing sample-identical crops; conflicting crops fail without modification.
    """
    root = bundle_root.expanduser().resolve(strict=True)
    destination = root / RESERVE_FILENAME
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    prepared = root / "audio_reuse_prepared_v1"
    inventory = load_prepared_inventory(prepared / "inventory.json")
    records = [AudioReusePreparedSource.model_validate_json(line)
               for line in (prepared / "records.jsonl").read_text().splitlines() if line.strip()]
    by_clip = {r.clip_uid: r for r in records}
    if len(by_clip) != len(records) or set(by_clip) != {j.clip_uid for j in inventory.jobs}:
        raise ValueError("duplicate or missing reserve prepared clip")
    donors = []
    for job in inventory.jobs:
        record = by_clip[job.clip_uid]
        if record.status != "ready":
            continue
        manifest_path = root / "audio_reuse_assets_v1" / job.clip_uid / "manifest.json"
        manifest = AudioReuseManifest.model_validate_json(manifest_path.read_text())
        if (manifest.clip_uid != job.clip_uid or manifest.source_job_fingerprint != job.request_fingerprint
                or record.source_job_fingerprint != job.request_fingerprint
                or manifest.source_annotation_sha256 != _fingerprint(record.annotation)
                or manifest.source_stem_record_fingerprint != record.source_stem_record_fingerprint):
            raise ValueError("reserve manifest differs from prepared clip")
        restricted = two_step_identity_restricted_groups(record.annotation, record.source_backend_provenance)
        groups, speakers = set(), set()
        for asset in manifest.speakers:
            if (asset.source_job_fingerprint != manifest.source_job_fingerprint
                    or asset.source_annotation_sha256 != manifest.source_annotation_sha256
                    or asset.source_stem_record_fingerprint != manifest.source_stem_record_fingerprint):
                raise ValueError("reserve speaker ownership differs from manifest")
            if asset.speaker_group in groups or asset.speaker_id in speakers:
                raise ValueError("reserve manifest repeats speaker ownership")
            groups.add(asset.speaker_group)
            speakers.add(asset.speaker_id)
            if asset.entity_id is None or asset.speaker_group in restricted:
                continue
            donors.append(_donor(root, job, record, asset, manifest_path))
    result = {"schema_version": "r2v.h3.voice_donor_reserve.1", "donors": donors}
    with _temporary_path(destination) as temporary:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _publish(temporary, destination)
    return result
