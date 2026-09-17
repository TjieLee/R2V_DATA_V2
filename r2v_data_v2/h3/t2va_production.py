"""Target-side production orchestration; frozen semantic modules remain unchanged."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

SHARD_SIZE = 10_000
DEFAULT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA"
)


def shard_bounds(shard_id: int) -> tuple[int, int]:
    if shard_id < 0:
        raise ValueError("negative shard")
    start = shard_id * SHARD_SIZE
    return start, start + SHARD_SIZE - 1


def shard_name(shard_id: int) -> str:
    start, end = shard_bounds(shard_id)
    return f"shard-{start:09d}-{end:09d}"


def shard_order(*, explicit=None, start=0, end=0, seed=20260917, rank=0, world_size=1):
    if explicit is not None:
        result = list(explicit)
        if not result or len(set(result)) != len(result) or min(result) < 0:
            raise ValueError("shards must be unique nonnegative IDs")
        return result
    if start < 0 or end < start or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid shard schedule")
    result = list(range(start, end + 1))
    random.Random(seed).shuffle(result)
    return result[rank::world_size]


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class ShardLockedError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ShardLockedError(f"locked: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _source_identity(path):
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def build_source_index(source: Path, root: Path) -> dict:
    source = source.resolve(strict=True)
    root = root.resolve()
    if source.is_relative_to(root):
        raise ValueError("source must not be inside production output")
    path = root / "manifests/source_index.json"
    with file_lock(root / "manifests/source_index.lock"):
        identity = _source_identity(source)
        if path.exists():
            index = json.loads(path.read_text())
            if (
                index["source_identity"] != identity
                or index["shard_size"] != SHARD_SIZE
            ):
                raise ValueError("source changed; use a fresh production root")
            return index
        digest = hashlib.sha256()
        shards, count = [], 0
        with source.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if count % SHARD_SIZE == 0:
                    shards.append(
                        {
                            "shard_id": count // SHARD_SIZE,
                            "start_index": count,
                            "end_index": count,
                            "byte_offset": offset,
                        }
                    )
                shards[-1]["end_index"] = count
                count += 1
                digest.update(line)
        if _source_identity(source) != identity:
            raise ValueError("source changed during indexing")
        index = {
            "shard_size": SHARD_SIZE,
            "source_record_count": count,
            "source_identity": identity,
            "source_sha256": digest.hexdigest(),
            "shards": shards,
        }
        atomic_json(path, index)
        return index


def materialize_shard(index: dict, shard_id: int, root: Path) -> Path:
    source = Path(index["source_identity"]["path"])
    if _source_identity(source) != index["source_identity"]:
        raise ValueError("source changed after indexing")
    if not 0 <= shard_id < len(index["shards"]):
        raise ValueError("shard outside source population")
    shard = index["shards"][shard_id]
    path = root / "manifests" / f"{shard_name(shard_id)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            return path
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with source.open("rb") as handle, temporary.open("wb") as output:
                handle.seek(shard["byte_offset"])
                for _ in range(shard["end_index"] - shard["start_index"] + 1):
                    output.write(handle.readline())
                output.flush()
                os.fsync(output.fileno())
            if _source_identity(source) != index["source_identity"]:
                raise ValueError("source changed during shard extraction")
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return path


class EndpointUnavailable(RuntimeError):
    """Infrastructure outage: stop scheduling, do not label remaining samples."""


def complete_rows(path: Path):
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def _repair_tail(path: Path):
    if not path.exists():
        return
    with path.open("r+b") as handle:
        end = 0
        for line in handle:
            if not line.endswith(b"\n"):
                handle.truncate(end)
                handle.flush()
                os.fsync(handle.fileno())
                return
            end = handle.tell()


def append_row(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_states(shard: Path) -> dict:
    return {r["clip_uid"]: r for r in complete_rows(shard / "state.jsonl.partial")}


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _recover_stage(destination: Path, identity: str):
    if not destination.exists():
        return None
    receipt = json.loads((destination / "stage.json").read_text())
    if receipt["identity"] != identity:
        raise ValueError("production stage identity changed; use a fresh root")
    for relative, digest in receipt["files"].items():
        if _sha(destination / relative) != digest:
            raise ValueError("published stage artifact changed")
    return receipt["result"]


def _publish_stage(temporary: Path, destination: Path, identity: str, result):
    files = {}
    for path in temporary.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(temporary))] = _sha(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    atomic_json(
        temporary / "stage.json",
        {
            "identity": identity,
            "files": files,
            "result": result,
        },
    )
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    _sync_directory(destination.parent)


def process_shard(
    root: Path,
    shard_id: int,
    rows: list[dict],
    processor,
    *,
    request_workers=8,
    progress_every=100,
):
    """Processor owns semantics; this function owns durable scheduling only."""
    if request_workers < 1:
        raise ValueError("request_workers must be positive")
    shard = root / "shards" / shard_name(shard_id)
    with file_lock(shard / "shard.lock"):
        journal = shard / "state.jsonl.partial"
        _repair_tail(journal)
        states = load_states(shard)
        writer = threading.Lock()
        stopped = threading.Event()
        processed = 0

        def save(state):
            with writer:
                append_row(journal, state)
                states[state["clip_uid"]] = dict(state)

        def process(row):
            nonlocal processed
            uid = row["clip_uid"]
            if Path(uid).name != uid or uid in {".", ".."}:
                raise ValueError("unsafe clip identity")
            identity = processor.identity(row)
            state = dict(
                states.get(uid)
                or {
                    "source_index": row["source_index"],
                    "clip_uid": uid,
                    "t2va_status": "pending",
                    "ta2va_status": "pending",
                    "failure_stage": None,
                    "failure_reason": None,
                    "identity": identity,
                }
            )
            if state["identity"] != identity:
                raise ValueError("production source/config identity changed")
            if state["t2va_status"] == "skipped":
                return
            if row.get("upstream_failure"):
                state.update(
                    t2va_status="skipped",
                    ta2va_status="skipped",
                    failure_reason=row["upstream_failure"],
                    failure_stage="upstream",
                )
                save(state)
                return
            for stage in ("t2va", "ta2va"):
                if stopped.is_set():
                    return
                destination = shard / "artifacts" / uid / stage
                result = _recover_stage(destination, identity)
                if result is None:
                    temporary = destination.with_name(
                        f".{stage}.tmp-{uuid.uuid4().hex}"
                    )
                    temporary.mkdir(parents=True)
                    try:
                        try:
                            result = processor.process(
                                stage, row, temporary, destination
                            )
                        except EndpointUnavailable:
                            stopped.set()
                            raise
                        except Exception as exc:  # noqa: BLE001 - sample isolation boundary.
                            state.update(
                                {
                                    f"{stage}_status": "failed",
                                    "failure_stage": stage,
                                    "failure_reason": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            if stage == "t2va":
                                state["ta2va_status"] = "skipped"
                            save(state)
                            return
                        _publish_stage(temporary, destination, identity, result)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                state.update(
                    {
                        f"{stage}_status": "ready",
                        "failure_stage": None,
                        "failure_reason": None,
                    }
                )
                save(state)
            with writer:
                processed += 1
                if progress_every and processed % progress_every == 0:
                    counts = {
                        f"{s}_{v}": sum(x[f"{s}_status"] == v for x in states.values())
                        for s in ("t2va", "ta2va")
                        for v in ("ready", "failed")
                    }
                    print(
                        f"shard={shard_id} processed={processed}/{len(rows)} {counts}",
                        flush=True,
                    )

        pool = ThreadPoolExecutor(max_workers=request_workers)
        pending = set()
        iterator = iter(rows)
        try:
            while not stopped.is_set():
                while len(pending) < request_workers:
                    row = next(iterator, None)
                    if row is None:
                        break
                    pending.add(pool.submit(process, row))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException:
            stopped.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if stopped.is_set():
            raise EndpointUnavailable("endpoint unavailable; resume after recovery")
        return states


def _endpoint_error(error):
    if error and any(
        token in str(error).lower()
        for token in (
            "apiconnectionerror",
            "connection refused",
            "connecterror",
            "failed to establish a new connection",
        )
    ):
        raise EndpointUnavailable(str(error))


def training_row(video, caption, audios=()):
    return {"video": video, "images": [], "audios": list(audios), "caption": caption}


def t2va_stage(job, backend, temporary: Path):
    """Mirror shadow orchestration using the frozen parser/validator/renderer."""
    from r2v_data_v2.h3 import t2va_shadow as frozen
    from r2v_data_v2.structured_output import parse_structured_json_response

    identity = frozen.request_fingerprint(job, backend.provenance())
    if _sha(Path(job.target_video_path)) != job.target_video_sha256:
        raise ValueError("original target video hash differs")
    try:
        raw = backend.annotate(job, identity)
    except Exception as exc:
        _endpoint_error(exc)
        raise
    frozen.write_json(temporary / "raw.json", raw)
    if raw.clip_uid != job.clip_uid or raw.request_fingerprint != identity:
        raise ValueError("T2VA raw response ownership differs")
    if raw.error:
        _endpoint_error(raw.error)
        raise ValueError(raw.error)
    draft, corrections = frozen.parse_t2va_semantic(job, raw.response or "")
    if (
        corrections != raw.deterministic_corrections
        or raw.audio_finalize is None
        or raw.audio_finalize.error
    ):
        raise ValueError("T2VA audio-finalize/correction provenance differs")
    audio = parse_structured_json_response(
        raw.audio_finalize.response or "", frozen.MimoAudioFinalizeDraft
    )
    core = frozen.validate_t2va_draft(job, draft, audio)
    if _sha(Path(job.target_video_path)) != job.target_video_sha256:
        raise ValueError("original target video changed during annotation")
    frozen.write_json(temporary / "job.json", job)
    frozen.write_json(temporary / "core.json", core)
    prompt = frozen.render_t2va_prompt(core)
    (temporary / "prompt.txt").write_text(prompt, encoding="utf-8")
    return {
        "model_call_count": raw.model_call_count,
        "exports": {"t2va": training_row(job.target_video_path, prompt)},
        "warnings": [*raw.warnings, *raw.audio_finalize.warnings, *core.warnings],
    }


def ta2va_stage(
    job,
    core,
    segments,
    stem,
    source_fingerprint,
    core_hash,
    backend,
    temporary: Path,
    destination: Path,
    *,
    allow_unverified=False,
):
    """Production writer around frozen TA2VA waveform and rendering operations."""
    import numpy as np

    from r2v_data_v2.h3 import ta2va_shadow as ta

    evidence = job.audio_evidence
    frames = ta.probe_canonical_target_frames(
        Path(evidence.full_audio_path), evidence.full_audio_sha256
    )
    products, exclusions = [], []

    def publish(variant, refs, assets, profiles, warnings, calls):
        fields = {
            "schema_version": ta.VERSION,
            "clip_uid": job.clip_uid,
            "source_inventory_fingerprint": source_fingerprint,
            "source_core_sha256": core_hash,
            "source_resolved_record_fingerprint": stem.record_fingerprint,
            "variant": variant,
            "status": "ready",
            "failure_reason": None,
            "audio_references": [r.model_dump(mode="json") for r in refs],
            "assets": [a.model_dump(mode="json") for a in assets],
            "speaker_profiles": [p.model_dump(mode="json") for p in profiles],
            "prompt": ta.render_product(core, variant, refs),
            "warnings": warnings,
            "model_call_count": calls,
        }
        product = ta.TA2VAProduct(**fields, record_fingerprint=ta.fingerprint(fields))
        products.append(product)
        (temporary / f"{variant}.txt").write_text(product.prompt, encoding="utf-8")

    full = ta.RecaptionAudioContract(
        audio_index=1,
        audio_label="<Audio 1>",
        kind="full_audio_reuse",
        path=evidence.full_audio_path,
        sha256=evidence.full_audio_sha256,
        retention_marker="fully_copy",
    )
    publish("full_audio_reuse", [full], [], [], [], 0)
    if any(f.text is not None for f in core.speech_facts):
        if stem.separation_state != "success" and not allow_unverified:
            raise ValueError(
                "TA2VA unverified stems require explicit --allow-unverified"
            )
        speech_pcm = ta._check_stem(
            stem.stem("speech"),
            Path(evidence.full_audio_path),
            evidence.full_audio_sha256,
            frames,
        )
        groups, exclusions = ta.sample_ranges(
            job, core, segments, frames, len(speech_pcm)
        )
        selected, include_music, warnings = ta.select_speakers(
            groups, ta._music_present(core.non_diegetic_music)
        )
        if selected:
            directory = temporary / "assets"
            directory.mkdir()
            assets, snippets, targets = [], [], []

            def write_asset(
                name, pcm, kind, sx=None, ranges=None, source_kind="speech"
            ):
                digest = ta._write_verified(directory / name, pcm)
                source_stem = stem.stem(source_kind)
                copied = min(frames, source_stem.canonical_frame_count)
                tail = frames - copied
                if ranges:
                    copied, previous_end = 0, 0
                    for r in ranges:
                        copied += max(
                            0,
                            r.target_end_sample
                            - max(previous_end, r.target_start_sample),
                        )
                        previous_end = max(previous_end, r.target_end_sample)
                    tail = frames - previous_end
                return ta.TA2VAAsset(
                    kind=kind,
                    speaker_id=sx,
                    ranges=ranges or [],
                    source_path=source_stem.canonical_stem_path,
                    source_sha256=source_stem.canonical_stem_sha256,
                    source_frame_count=source_stem.canonical_frame_count,
                    output_path=str(destination / "assets" / name),
                    output_sha256=digest,
                    frame_count=frames,
                    copied_frame_count=copied,
                    silence_tail_frame_count=tail,
                )

            for sx in selected:
                for r in groups[sx]:
                    name = f"{sx}.{r.segment_id}.snippet.flac"
                    ta._write_verified(
                        directory / name,
                        speech_pcm[r.target_start_sample : r.target_end_sample],
                    )
                    snippets.append(
                        (
                            {
                                "speaker_group": sx,
                                "speaker_id": sx,
                                **r.model_dump(mode="json"),
                            },
                            directory / name,
                        )
                    )
                assets.append(
                    write_asset(
                        f"{sx}.flac",
                        ta.speech_track(speech_pcm, frames, groups[sx]),
                        "speaker_speech_reuse",
                        sx,
                        groups[sx],
                    )
                )
                targets.append(
                    {
                        "speaker_group": sx,
                        "speaker_id": sx,
                        "segments": [r.model_dump(mode="json") for r in groups[sx]],
                    }
                )
            try:
                profiles, raw = backend.profile(targets, snippets)
            except Exception as exc:
                _endpoint_error(exc)
                raise
            raw["snippets"] = [
                {
                    "label": label,
                    "path": str(destination / "assets" / p.name),
                    "sha256": _sha(p),
                }
                for label, p in snippets
            ]
            atomic_json(temporary / "raw_profile.json", raw)
            if raw["error"]:
                _endpoint_error(raw["error"])
                raise ValueError(raw["error"])
            if include_music:
                music = ta._check_stem(
                    stem.stem("music"),
                    Path(evidence.full_audio_path),
                    evidence.full_audio_sha256,
                    frames,
                )
                pcm = np.zeros((frames, 2), dtype=np.int16)
                copied = min(frames, len(music))
                pcm[:copied] = music[:copied]
                assets.append(
                    write_asset("music.flac", pcm, "music_reuse", source_kind="music")
                )
            profile_by_sx = {p.speaker_group: p.voice_characteristics for p in profiles}
            refs = [
                ta.RecaptionAudioContract(
                    audio_index=i,
                    audio_label=f"<Audio {i}>",
                    kind=a.kind,
                    path=a.output_path,
                    sha256=a.output_sha256,
                    speaker_id=a.speaker_id,
                    voice_characteristics=profile_by_sx.get(a.speaker_id),
                    retention_marker="partially_copy",
                )
                for i, a in enumerate(assets, 1)
            ]
            publish(
                "target_speech_reuse",
                refs,
                assets,
                profiles,
                [*exclusions, *warnings],
                raw["model_call_count"],
            )
    (temporary / "records.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in products), encoding="utf-8"
    )
    return {
        "model_call_count": sum(p.model_call_count for p in products),
        "reuse_exclusions": exclusions,
        "exports": {
            "ta2va_full_audio"
            if p.variant == "full_audio_reuse"
            else "ta2va_speech_bgm": training_row(
                job.target_video_path, p.prompt, [a.path for a in p.audio_references]
            )
            for p in products
        },
    }
