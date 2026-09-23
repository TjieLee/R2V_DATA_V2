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

SHARD_SIZE = 2_000
MAX_CLIP_DURATION_SECONDS = 200.0
CLIP_DURATION_EXCLUSION_REASON = "clip_duration_over_200s"
DEFAULT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA"
)
TASKS = ("t2va", "ta2va_full_audio", "ta2va_speech_bgm")
TERMINAL_STATUSES = frozenset({"ready", "failed", "skipped"})


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
def file_lock(path: Path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
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


def _same_source_content_metadata(left, right):
    """Cross-mount-stable fast identity; full SHA is the fallback on change."""
    return all(left.get(key) == right.get(key) for key in ("path", "size", "mtime_ns"))


def build_source_index(source: Path, root: Path) -> dict:
    source = source.resolve(strict=True)
    root = root.resolve()
    if source.is_relative_to(root):
        raise ValueError("source must not be inside production output")
    path = root / "manifests/source_index.json"
    with file_lock(root / "manifests/source_index.lock", blocking=True):
        identity = _source_identity(source)
        if path.exists():
            index = json.loads(path.read_text())
            if index["shard_size"] != SHARD_SIZE:
                raise ValueError("source index shard size mismatch; use a fresh production root")
            if (
                not _same_source_content_metadata(index["source_identity"], identity)
                and _sha(source) != index["source_sha256"]
            ):
                raise ValueError("source changed; use a fresh production root")
            # device/inode/ctime can differ across shared mounts. Stable path,
            # size and mtime avoid a full-file SHA on every cluster node; SHA
            # remains the fallback whenever those content-facing fields change.
            index["source_identity"] = identity
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
    current_identity = _source_identity(source)
    if (
        not _same_source_content_metadata(current_identity, index["source_identity"])
        and _sha(source) != index["source_sha256"]
    ):
        raise ValueError("source changed after indexing")
    identity = _source_identity(source)
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
            if _source_identity(source) != identity:
                raise ValueError("source changed during shard extraction")
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return path


class EndpointUnavailable(RuntimeError):
    """Infrastructure outage: stop scheduling, do not label remaining samples."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class PartialStageFailure(ValueError):
    """A failed variant must not discard independent ready variants."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


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


def seal_export(shard: Path, task: str) -> None:
    """Seal complete JSONL rows; recover an interrupted final/partial overlap."""
    if task not in TASKS:
        raise ValueError("unknown training task")
    partial = shard / "exports" / f"{task}.jsonl.partial"
    final = partial.with_suffix("")
    if not partial.exists():
        return
    if not final.exists():
        _repair_tail(partial)
        os.replace(partial, final)
        _sync_directory(final.parent)
        return
    rows = {}
    for path in (final, partial):
        for row in complete_rows(path):
            if set(row) != {"video", "images", "audios", "caption"}:
                raise ValueError("invalid published training row")
            video = row["video"]
            if video in rows and rows[video] != row:
                raise ValueError("conflicting published training rows")
            rows[video] = row
    temporary = final.with_name(f".{final.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows.values():
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        _sync_directory(final.parent)
        partial.unlink()
        _sync_directory(final.parent)
    finally:
        temporary.unlink(missing_ok=True)


def seal_terminal_shard(shard: Path) -> dict | None:
    """Use only durable source/state JSON to finish attempted shard stages."""
    if (shard / "COMPLETE").exists():
        return load_states(shard)
    source_path = shard / "sources.json"
    if not source_path.is_file():
        return None
    sources = json.loads(source_path.read_text())
    source_ids = [row["clip_uid"] for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate shard source clip identity")
    _repair_tail(shard / "state.jsonl.partial")
    states = load_states(shard)
    if set(states) - set(source_ids):
        raise ValueError("state contains a clip outside the shard source population")
    if len(states) != len(sources):
        return None
    t2va_done = all(
        states[uid]["t2va_status"] in TERMINAL_STATUSES for uid in source_ids
    )
    ta2va_done = all(
        states[uid]["ta2va_status"] in TERMINAL_STATUSES for uid in source_ids
    )
    if t2va_done:
        seal_export(shard, "t2va")
    if not (t2va_done and ta2va_done):
        return None
    for task in TASKS[1:]:
        seal_export(shard, task)
    counters = {
        f"{stage}_{status}": sum(
            states[uid][f"{stage}_status"] == status for uid in source_ids
        )
        for stage in ("t2va", "ta2va")
        for status in ("ready", "failed", "skipped")
    }
    atomic_json(shard / "COMPLETE", {"source_count": len(sources), **counters})
    return states


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _recover_stage(
    destination: Path,
    identity: str,
    *,
    allow_identity_mismatch: bool = False,
):
    if not destination.exists():
        return None
    receipt = json.loads((destination / "stage.json").read_text())
    if receipt["identity"] != identity and not allow_identity_mismatch:
        raise ValueError("production stage identity changed; use a fresh root")
    for relative, digest in receipt["files"].items():
        artifact = destination / relative
        if not artifact.is_file():
            raise ValueError("published stage artifact is missing")
        if not allow_identity_mismatch and _sha(artifact) != digest:
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
    allow_existing_identity_mismatch=False,
):
    """Processor owns semantics; this function owns durable scheduling only."""
    if request_workers < 1:
        raise ValueError("request_workers must be positive")
    shard = root / "shards" / shard_name(shard_id)
    with file_lock(shard / "shard.lock"):
        sources = [
            {k: r[k] for k in ("source_index", "clip_uid", "video")} for r in rows
        ]
        if len({r["clip_uid"] for r in sources}) != len(sources):
            raise ValueError("duplicate source clip identity within shard")
        source_path = shard / "sources.json"
        if source_path.exists() and json.loads(source_path.read_text()) != sources:
            raise ValueError("shard source population changed")
        if not source_path.exists():
            atomic_json(source_path, sources)
        journal = shard / "state.jsonl.partial"
        completed = seal_terminal_shard(shard)
        if completed is not None:
            return completed
        _repair_tail(journal)
        states = load_states(shard)
        writer = threading.Lock()
        stopped = threading.Event()
        processed = 0
        exported = {}
        for task in TASKS:
            for path in (
                shard / "exports" / f"{task}.jsonl",
                shard / "exports" / f"{task}.jsonl.partial",
            ):
                _repair_tail(path)
                for item in complete_rows(path):
                    key = (task, item["video"])
                    if key in exported and exported[key] != item:
                        raise ValueError("conflicting published training rows")
                    exported[key] = item

        def publish_exports(result):
            with writer:
                for task, item in result.get("exports", {}).items():
                    if task not in TASKS or set(item) != {
                        "video",
                        "images",
                        "audios",
                        "caption",
                    }:
                        raise ValueError("invalid training row")
                    key = (task, item["video"])
                    if key in exported:
                        if exported[key] != item:
                            raise ValueError("conflicting published training rows")
                        continue
                    append_row(shard / "exports" / f"{task}.jsonl.partial", item)
                    exported[key] = item

        def save(state):
            with writer:
                append_row(journal, state)
                states[state["clip_uid"]] = dict(state)

        def process(row):
            uid = row["clip_uid"]
            if Path(uid).name != uid or uid in {".", ".."}:
                raise ValueError("unsafe clip identity")
            previous = states.get(uid)
            if previous and (
                previous["t2va_status"] in {"failed", "skipped"}
                or (
                    previous["t2va_status"] in TERMINAL_STATUSES
                    and previous["ta2va_status"] in TERMINAL_STATUSES
                )
            ):
                return
            identity = processor.identity(row)
            state = dict(
                previous
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
                if allow_existing_identity_mismatch or (
                    state.get("preparation_failed")
                    and not (shard / "artifacts" / uid / "t2va").exists()
                ):
                    state["identity"] = identity
                else:
                    raise ValueError("production source/config identity changed")
            state["preparation_failed"] = bool(row.get("preparation_error"))
            if row.get("upstream_failure") and state["t2va_status"] == "pending":
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
                if state[f"{stage}_status"] in TERMINAL_STATUSES:
                    continue
                destination = shard / "artifacts" / uid / stage
                failed_destination = destination.with_name(f"{stage}_failed")
                previous_partial = _recover_stage(
                    failed_destination,
                    identity,
                    allow_identity_mismatch=allow_existing_identity_mismatch,
                )
                if previous_partial is not None:
                    publish_exports(previous_partial)
                result = _recover_stage(
                    destination,
                    identity,
                    allow_identity_mismatch=allow_existing_identity_mismatch,
                )
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
                        except EndpointUnavailable as exc:
                            stopped.set()
                            if exc.result is not None:
                                if failed_destination.exists():
                                    shutil.rmtree(failed_destination)
                                _publish_stage(
                                    temporary, failed_destination, identity, exc.result
                                )
                                publish_exports(exc.result)
                                state.update(
                                    {
                                        f"{stage}_status": "failed",
                                        "failure_stage": stage,
                                        "failure_reason": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                                save(state)
                            raise
                        except Exception as exc:  # noqa: BLE001 - sample isolation boundary.
                            if isinstance(exc, PartialStageFailure):
                                if failed_destination.exists():
                                    shutil.rmtree(failed_destination)
                                _publish_stage(
                                    temporary, failed_destination, identity, exc.result
                                )
                                publish_exports(exc.result)
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
                publish_exports(result)
                if state[f"{stage}_status"] == "ready":
                    continue
                state.update(
                    {
                        f"{stage}_status": "ready",
                        "failure_stage": None,
                        "failure_reason": None,
                    }
                )
                save(state)

        def process_and_report(row):
            nonlocal processed
            try:
                process(row)
            finally:
                with writer:
                    processed += 1
                    if progress_every and processed % progress_every == 0:
                        counts = {
                            f"{s}_{v}": sum(
                                x[f"{s}_status"] == v for x in states.values()
                            )
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
                    pending.add(pool.submit(process_and_report, row))
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
        seal_terminal_shard(shard)
        return states


def _endpoint_error(error, result=None):
    if error and any(
        token in str(error).lower()
        for token in (
            "apiconnectionerror",
            "connection refused",
            "connecterror",
            "failed to establish a new connection",
        )
    ):
        raise EndpointUnavailable(str(error), result)


def training_row(video, caption, audios=()):
    return {"video": video, "images": [], "audios": list(audios), "caption": caption}


def t2va_stage(job, backend, temporary: Path, *, verify_video: bool = True):
    """Mirror shadow orchestration using the frozen parser/validator/renderer."""
    from r2v_data_v2.h3 import t2va_shadow as frozen

    identity = frozen.request_fingerprint(job, backend.provenance())
    if verify_video and _sha(Path(job.target_video_path)) != job.target_video_sha256:
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
    draft, audio, corrections = frozen.parse_t2va_completion(job, raw)
    if corrections != raw.deterministic_corrections:
        raise ValueError("T2VA correction provenance differs")
    core = frozen.validate_t2va_draft(job, draft, audio)
    if verify_video and _sha(Path(job.target_video_path)) != job.target_video_sha256:
        raise ValueError("original target video changed during annotation")
    frozen.write_json(temporary / "job.json", job)
    frozen.write_json(temporary / "core.json", core)
    prompt = frozen.render_t2va_prompt(core)
    (temporary / "prompt.txt").write_text(prompt, encoding="utf-8")
    return {
        "model_call_count": raw.model_call_count,
        "exports": {"t2va": training_row(job.target_video_path, prompt)},
        "warnings": [
            *raw.warnings,
            *(raw.audio_finalize.warnings if raw.audio_finalize else []),
            *core.warnings,
        ],
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
    source_hashes=None,
    verify_media=True,
):
    """Production writer around frozen TA2VA waveform and rendering operations."""
    import numpy as np

    from r2v_data_v2.h3 import ta2va_shadow as ta

    evidence = job.audio_evidence
    hashes = {
        **(source_hashes or {}),
        job.target_video_path: job.target_video_sha256,
        **{
            getattr(evidence, f"{kind}_path"): getattr(evidence, f"{kind}_sha256")
            for kind in ("full_audio", "speech", "music", "sfx")
        },
    }
    if verify_media:
        ta._verify(hashes)
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

    def finish(failed_calls=0):
        if verify_media:
            ta._verify(hashes)
        (temporary / "records.jsonl").write_text(
            "".join(p.model_dump_json() + "\n" for p in products), encoding="utf-8"
        )
        return {
            "model_call_count": failed_calls
            + sum(p.model_call_count for p in products),
            "reuse_exclusions": exclusions,
            "exports": {
                "ta2va_full_audio"
                if p.variant == "full_audio_reuse"
                else "ta2va_speech_bgm": training_row(
                    job.target_video_path,
                    p.prompt,
                    [a.path for a in p.audio_references],
                )
                for p in products
            },
        }

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
                atomic_json(
                    temporary / "raw_profile.json",
                    {
                        "model_call_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                partial = finish()
                _endpoint_error(exc, partial)
                raise PartialStageFailure(str(exc), partial) from exc
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
                partial = finish(raw["model_call_count"])
                _endpoint_error(raw["error"], partial)
                raise PartialStageFailure(raw["error"], partial)
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
    return finish()


def build_snapshot(root: Path, snapshot_id: str) -> Path:
    if (
        not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", "..", "LATEST"}
    ):
        raise ValueError("invalid snapshot ID")
    root = root.resolve()
    snapshots = root / "snapshots"
    destination = snapshots / snapshot_id
    with file_lock(snapshots / "snapshot.lock"):
        if destination.exists():
            raise FileExistsError(destination)
        temporary = snapshots / f".{snapshot_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        handles = {}
        try:
            handles = {
                task: (temporary / f"{task}.jsonl").open("w", encoding="utf-8")
                for task in TASKS
            }
            seen, videos = {}, {}
            counts = dict.fromkeys(TASKS, 0)
            for shard in sorted((root / "shards").glob("shard-*")):
                source_path = shard / "sources.json"
                if not source_path.exists():
                    continue
                sources = sorted(
                    json.loads(source_path.read_text()), key=lambda r: r["source_index"]
                )
                by_video = {r["video"]: r for r in sources if r["video"]}
                if len(by_video) != sum(bool(r["video"]) for r in sources):
                    raise ValueError("conflicting video identity in shard")
                # Bound caption memory to one 10k shard. Global dedup stores hashes only.
                rows_by_task = {}
                for task in TASKS:
                    rows = {}
                    paths = [
                        shard / "exports" / f"{task}.jsonl",
                        shard / "exports" / f"{task}.jsonl.partial",
                    ]
                    # Final may appear after the first check but before partial
                    # is opened. Re-read final to close that rename window.
                    paths.append(paths[0])
                    for path in paths:
                        try:
                            items = list(complete_rows(path))
                        except FileNotFoundError:
                            # A writer may have just renamed partial to final.
                            items = list(complete_rows(paths[0]))
                        for item in items:
                            if set(item) != {"video", "images", "audios", "caption"}:
                                raise ValueError("invalid loader-facing row")
                            source = by_video.get(item["video"])
                            if source is None:
                                raise ValueError("export has no source clip identity")
                            uid = source["clip_uid"]
                            if uid in rows and rows[uid] != item:
                                raise ValueError(
                                    f"conflicting snapshot key: {uid}/{task}"
                                )
                            rows[uid] = item
                    rows_by_task[task] = rows
                for source in sources:
                    uid = source["clip_uid"]
                    for task in TASKS:
                        item = rows_by_task[task].get(uid)
                        if item is None:
                            continue
                        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
                        digest = hashlib.sha256(encoded.encode()).hexdigest()
                        key = (uid, task)
                        if key in seen:
                            if seen[key] != digest:
                                raise ValueError(
                                    f"conflicting snapshot key: {uid}/{task}"
                                )
                            continue
                        seen[key] = digest
                        handles[task].write(encoded + "\n")
                        counts[task] += 1
                        video = videos.setdefault(
                            uid, {"video": item["video"], "tasks": []}
                        )
                        if video["video"] != item["video"]:
                            raise ValueError(f"conflicting clip video: {uid}")
                        video["tasks"].append(task)
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            with (temporary / "videos.jsonl").open("w", encoding="utf-8") as handle:
                for video in videos.values():
                    video["tasks"].sort(key=TASKS.index)
                    handle.write(json.dumps(video, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_json(
                temporary / "summary.json",
                {
                    "snapshot_id": snapshot_id,
                    "counts": counts,
                    "video_count": len(videos),
                    "production_root": str(root),
                },
            )
            temporary.rename(destination)
            _sync_directory(snapshots)
            latest = snapshots / f".LATEST.tmp-{uuid.uuid4().hex}"
            with latest.open("w") as handle:
                handle.write(snapshot_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(latest, snapshots / "LATEST")
            _sync_directory(snapshots)
        finally:
            for handle in handles.values():
                handle.close()
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination


def prepare_shard(
    root,
    index,
    shard_id,
    *,
    audio_production_root,
    audio_shadow_run_id,
    clips_root,
    source_videos_root,
    backend,
):
    """Bound selection to one fixed shard and reuse the frozen inventory builder."""
    from r2v_data_v2.h3.t2va_shadow import build_t2va_inventory, write_json
    from r2v_data_v2.naming import parse_clip_identity
    from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter, _path_below_root

    manifest = materialize_shard(index, shard_id, root)
    prepared = root / "shards" / shard_name(shard_id) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    preselected = None
    selection_path = root / "shards" / shard_name(shard_id) / "source/selection.json"
    if selection_path.is_file():
        from r2v_data_v2.h3.t2va_source import T2VAShotSelection

        preselected = T2VAShotSelection.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )

    print(f"shard={shard_id} downstream_source_projection_start", flush=True)
    rows, contexts = [], {}
    if preselected is not None:
        shots_by_index = {shot.source_index: shot for shot in preselected.shots}
        if len(shots_by_index) != len(preselected.shots):
            raise ValueError("cached T2VA selection has duplicate source indexes")
        excluded_by_index = {
            row["source_index"]: row["reason"] for row in preselected.excluded_rows
        }
        if len(excluded_by_index) != len(preselected.excluded_rows):
            raise ValueError("cached T2VA selection has duplicate excluded indexes")
        with manifest.open("rb") as handle:
            for offset, line in enumerate(handle):
                source_index = shard_id * SHARD_SIZE + offset
                row = {
                    "source_index": source_index,
                    "clip_uid": f"invalid-source-{source_index}",
                    "video": "",
                    "source_row_sha256": hashlib.sha256(line).hexdigest(),
                }
                shot = shots_by_index.get(source_index)
                if shot is not None:
                    row.update(clip_uid=shot.clip_uid, video=shot.video_path)
                else:
                    reason = excluded_by_index.get(source_index)
                    if reason is None:
                        raise ValueError(
                            "cached T2VA selection does not cover source row "
                            f"{source_index}"
                        )
                    if reason == CLIP_DURATION_EXCLUSION_REASON:
                        row["upstream_failure"] = reason
                    else:
                        row["preparation_error"] = (
                            "ValueError: canonical selection excluded source row: "
                            + reason
                        )
                rows.append(row)
        expected = len(preselected.shots) + len(preselected.excluded_rows)
        if len(rows) != expected:
            raise ValueError("cached T2VA selection row count differs")
    else:
        adapter = JeaVideoMotionAdapter(
            clips_root=clips_root, source_videos_root=source_videos_root
        )
        seen = set()
        with manifest.open("rb") as handle:
            for offset, line in enumerate(handle):
                source_index = shard_id * SHARD_SIZE + offset
                row = {
                    "source_index": source_index,
                    "clip_uid": f"invalid-source-{source_index}",
                    "video": "",
                    "source_row_sha256": hashlib.sha256(line).hexdigest(),
                }
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("shot manifest row must be a JSON object")
                    # Preserve ingestion identity across transient missing-media
                    # failures when no canonical selection exists.
                    candidate, _ = _path_below_root(
                        raw.get("video_path"),
                        root=clips_root,
                        field_name="video_path",
                        require_file=False,
                    )
                    uid = parse_clip_identity(candidate).clip_uid
                    if uid in seen:
                        raise ValueError("duplicate JEA clip identity")
                    seen.add(uid)
                    row.update(clip_uid=uid, video=str(candidate))
                    item = adapter.parse(raw, source_index=source_index)
                    row.update(clip_uid=item["clip_uid"], video=item["video_path"])
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    row["preparation_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    print(
        f"shard={shard_id} downstream_source_projection_ready rows={len(rows)}",
        flush=True,
    )

    def build(path, *, preselected=None, verify_audio_files=True):
        return build_t2va_inventory(
            shot_manifest=path,
            clips_root=clips_root,
            source_videos_root=source_videos_root,
            audio_production_root=audio_production_root,
            audio_shadow_run_id=audio_shadow_run_id,
            t2va_run_id="production",
            backend=backend,
            preselected=preselected,
            verify_audio_files=verify_audio_files,
        )

    print(f"shard={shard_id} downstream_inventory_build_start", flush=True)
    try:
        inventory = build(
            manifest,
            preselected=preselected,
            verify_audio_files=preselected is None,
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(
            f"shard={shard_id} downstream_inventory_build_failed "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    print(
        f"shard={shard_id} downstream_inventory_build_ready "
        f"jobs={len(inventory.jobs)}",
        flush=True,
    )
    write_json(prepared / "inventory.json", inventory)
    contexts = {job.clip_uid: (job, inventory) for job in inventory.jobs}
    for row in rows:
        uid = row["clip_uid"]
        if row.get("preparation_error") or row.get("upstream_failure"):
            continue
        if uid not in contexts:
            row["preparation_error"] = (
                "ValueError: valid source row is absent from bulk T2VA inventory"
            )
            continue
        job, _ = contexts[uid]
        if job.upstream_failure:
            row["upstream_failure"] = job.upstream_failure
    return rows, contexts


class FrozenProductionProcessor:
    def __init__(
        self,
        contexts,
        backend,
        profile_backend,
        *,
        allow_unverified=False,
        verify_sources_per_sample=True,
    ):
        self.contexts = contexts
        self.backend = backend
        self.profile_backend = profile_backend
        self.allow_unverified = allow_unverified
        self.verify_sources_per_sample = verify_sources_per_sample
        self._sources = {}
        # Load immutable source records once, never once per clip.
        from r2v_data_v2.h3 import t2va_shadow as t

        for _, inventory in contexts.values():
            root = t.stem_shadow_root(
                Path(inventory.audio_production_root), inventory.audio_shadow_run_id
            )
            if root in self._sources or not (root / "asr").exists():
                continue
            _, stems, _ = t.load_resolved_stems(root / "resolved_stems_v1")
            raw = t.read_rows(
                root / "diarization/raw_segments.jsonl", t.RawDiarizationSegment
            )
            asr = t.read_rows(root / "asr/segments.jsonl", t.Qwen3ASRSegment)
            self._sources[root] = (
                {s.clip_uid: s for s in stems},
                {(s.target_clip_uid, s.segment_id): s for s in raw},
                {(s.clip_uid, s.segment_id): s for s in asr},
            )

    def identity(self, row):
        from r2v_data_v2.h3.t2va_shadow import fingerprint

        context = self.contexts.get(row["clip_uid"])
        hashes = {}
        if context:
            inventory = context[1]
            local_selection = {
                inventory.shot_selection.shot_manifest_path,
                inventory.shot_selection.case_manifest_path,
            }
            hashes = {
                p: h
                for p, h in inventory.source_hashes.items()
                if p not in local_selection
            }
        return fingerprint(
            {
                "source": {
                    k: v
                    for k, v in row.items()
                    if k not in {"preparation_error", "upstream_failure"}
                },
                "job": context[0].model_dump(mode="json") if context else None,
                "source_hashes": hashes,
                "backend": self.backend.provenance().model_dump(mode="json"),
                "profile_backend": self.profile_backend.provenance(),
                "allow_unverified": self.allow_unverified,
            }
        )

    def process(self, stage, row, temporary, destination):
        from r2v_data_v2.h3 import t2va_shadow as t
        from r2v_data_v2.h3 import ta2va_shadow as ta

        if row.get("preparation_error"):
            raise ValueError(row["preparation_error"])
        job, inventory = self.contexts[row["clip_uid"]]
        if stage == "t2va":
            if self.verify_sources_per_sample:
                ta._verify(inventory.source_hashes)
                result = t2va_stage(job, self.backend, temporary)
                ta._verify(inventory.source_hashes)
                return result
            if job.audio_evidence is None or any(
                not Path(getattr(job.audio_evidence, f"{kind}_path")).is_file()
                for kind in ("full_audio", "speech", "music", "sfx")
            ):
                raise ValueError("T2VA audio evidence is missing")
            return t2va_stage(
                job,
                self.backend,
                temporary,
                verify_video=False,
            )
        core_path = destination.parent / "t2va/core.json"
        core = t.H3NoReferenceAVCore.model_validate_json(core_path.read_text())
        t.validate_t2va_draft(job, core)
        if self.verify_sources_per_sample:
            t.check_audio_files(job.audio_evidence)
        root = t.stem_shadow_root(
            Path(inventory.audio_production_root), inventory.audio_shadow_run_id
        )
        stems, raw, asr = self._sources[root]
        segments = []
        for fact in job.speech_facts:
            key = (job.clip_uid, fact.segment_id)
            r, a = raw[key], asr[key]
            if (r.speaker_cluster_id, r.start_time, r.end_time) != (
                fact.source_speaker_cluster,
                fact.start_time,
                fact.end_time,
            ) or (a.text, a.language) != (fact.text, fact.language):
                raise ValueError("TA2VA frozen speech identity differs")
            for name in (
                "speaker_cluster_id",
                "source_audio_path",
                "source_start_sample",
                "source_end_sample",
                "source_sample_rate_hz",
                "source_channels",
                "start_time",
                "end_time",
            ):
                if getattr(r, name) != getattr(a, name):
                    raise ValueError("TA2VA raw/ASR sample identity differs")
            segments.append(r)
        return ta2va_stage(
            job,
            core,
            segments,
            stems[job.clip_uid],
            inventory.inventory_fingerprint,
            _sha(core_path),
            ta.profile_backend_for_t2va_raw(
                destination.parent / "t2va/raw.json", self.profile_backend
            ),
            temporary,
            destination,
            allow_unverified=self.allow_unverified,
            source_hashes=(
                inventory.source_hashes if self.verify_sources_per_sample else None
            ),
            verify_media=self.verify_sources_per_sample,
        )
