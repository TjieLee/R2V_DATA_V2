"""One owned CPU-only canonical preparation process, with one-shard lookahead."""

from __future__ import annotations

import importlib
import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.h3.t2va_full_workers import (
    _atomic_json,
    _interrupted,
    _read_json,
    _stop_workers,
)
from r2v_data_v2.h3.t2va_production import file_lock, shard_name

_SIGNALS = {signal.SIGINT, signal.SIGTERM}
_DEFAULT_WORKER = "r2v_data_v2.h3.t2va_full_prefetch:prepare_canonical"


def prepare_canonical(request: dict) -> dict:
    """Child-only import boundary; the enclosing worker owns canonical.lock."""
    full = importlib.import_module("r2v_data_v2.h3.t2va_full_production")
    media = importlib.import_module("r2v_data_v2.h3.audio_backends")
    selection = full.shard_selection(
        Path(request["root"]),
        request["index"],
        request["shard_id"],
        Path(request["clips_root"]),
        Path(request["source_videos_root"]) if request["source_videos_root"] else None,
    )
    audio_root = full.bootstrap_audio(
        Path(request["shard_root"]),
        selection,
        media.FFmpegAudioMediaBackend(
            ffmpeg=request["ffmpeg"], ffprobe=request["ffprobe"]
        ),
        canonical_workers=request["canonical_workers"],
    )
    ready = sum(
        1
        for _ in full.production.complete_rows(
            audio_root / "audio/canonical_clips.jsonl"
        )
    )
    return {
        "audio_root": str(audio_root),
        "summary": {
            "ready": ready,
            "failed": len(selection.shots) - ready,
            "skipped": len(selection.excluded_rows),
        },
    }


def _validate_result(result):
    if not isinstance(result, dict) or set(result) != {"audio_root", "summary"}:
        raise ValueError("invalid canonical prefetch result")
    if (
        not isinstance(result["audio_root"], str)
        or not Path(result["audio_root"]).is_absolute()
    ):
        raise ValueError("invalid canonical prefetch audio_root")
    summary = result["summary"]
    if not isinstance(summary, dict) or set(summary) != {"ready", "failed", "skipped"}:
        raise ValueError("invalid canonical prefetch summary")
    if any(type(value) is not int or value < 0 for value in summary.values()):
        raise ValueError("invalid canonical prefetch counts")
    return result


class CanonicalPrefetch:
    """Main-thread context owning at most one shard until wait() consumes it.

    worker_factory is an importable ``module:callable`` (or dotted callable)
    accepting a JSON request and returning {audio_root: str, summary: dict}.
    It runs inside the same blocking canonical.lock as production preparation.
    Use the default worker in production; injection is for model-free CPU tests.
    """

    def __init__(
        self,
        root,
        index,
        clips_root,
        source_videos_root,
        canonical_workers,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        *,
        worker_factory=_DEFAULT_WORKER,
    ):
        if type(canonical_workers) is not int or canonical_workers < 1:
            raise ValueError("canonical_workers must be a positive integer")
        if not isinstance(worker_factory, str) or not worker_factory.strip():
            raise ValueError("worker_factory must be an importable callable")
        self.root = Path(root).expanduser().resolve()
        self._configuration = {
            "root": str(self.root),
            "index": json.loads(json.dumps(index, allow_nan=False)),
            "clips_root": str(Path(clips_root).expanduser().resolve()),
            "source_videos_root": str(Path(source_videos_root).expanduser().resolve())
            if source_videos_root is not None
            else None,
            "canonical_workers": canonical_workers,
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
            "worker_factory": worker_factory,
        }
        self._pending = None
        self._previous_handlers = None
        self._closed = False
        self._failed = False
        self._handler = self._handle_signal

    def __enter__(self):
        if self._closed:
            raise RuntimeError("canonical prefetch is closed")
        if self._previous_handlers is not None:
            raise RuntimeError("canonical prefetch context is already entered")
        previous = {sig: signal.getsignal(sig) for sig in _SIGNALS}
        try:
            for sig in _SIGNALS:
                signal.signal(sig, self._handler)
        except BaseException:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            raise
        self._previous_handlers = previous
        return self

    def __exit__(self, *_):
        self.close()

    def _handle_signal(self, signum, frame):
        self.close()
        _interrupted(signum, frame)

    def start(self, shard_id: int) -> None:
        if self._closed:
            raise RuntimeError("canonical prefetch is closed")
        if self._previous_handlers is None:
            raise RuntimeError("enter the canonical prefetch context before start")
        if self._failed:
            raise RuntimeError("canonical prefetch failed; close this invocation")
        if self._pending is not None:
            raise RuntimeError("canonical prefetch already has an outstanding shard")
        if type(shard_id) is not int or shard_id < 0:
            raise ValueError("shard_id must be a nonnegative integer")
        shard = self.root / "shards" / shard_name(shard_id)
        attempt = shard / "stage_state/canonical_prefetch" / uuid.uuid4().hex
        request_path = attempt / "request.json"
        _atomic_json(
            request_path,
            {**self._configuration, "shard_id": shard_id, "shard_root": str(shard)},
        )
        log_path = shard / "logs/canonical-prefetch.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            # A signal after Popen but before registration must not orphan FFmpeg.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        str(request_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._pending = (shard_id, process, attempt, log_path)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        print(f"shard={shard_id} canonical_prefetch_started", flush=True)

    def _release(self):
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        try:
            if self._pending is not None:
                _stop_workers([self._pending[1]])
                self._pending = None
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    def wait(self, shard_id: int) -> dict:
        if self._pending is None:
            raise RuntimeError("canonical prefetch has no outstanding shard")
        expected, process, attempt, log_path = self._pending
        if shard_id != expected:
            raise ValueError(
                f"waiting for shard {shard_id}, but outstanding shard is {expected}"
            )
        try:
            while True:
                payload = _read_json(attempt / "result.json")
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(
                        f"canonical prefetch shard={shard_id} exited {code}; log={log_path}"
                    )
                if payload is not None:
                    break
                if code is not None:
                    raise RuntimeError(
                        f"canonical prefetch shard={shard_id} exited without result; log={log_path}"
                    )
                time.sleep(0.02)
            if not isinstance(payload, dict) or payload.get("shard_id") != shard_id:
                raise ValueError("invalid canonical prefetch response identity")
            if payload.get("error"):
                raise RuntimeError(
                    f"canonical prefetch shard={shard_id}: {payload['error']}; log={log_path}"
                )
            result = _validate_result(payload.get("result"))
            elapsed = payload["elapsed_seconds"]
            print(
                f"shard={shard_id} canonical_prefetch_completed elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
            return result
        except Exception as exc:
            self._failed = True
            raise RuntimeError(f"canonical prefetch failed: {exc}") from exc
        finally:
            self._release()

    def close(self) -> None:
        if self._closed:
            return
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        try:
            self._release()
        finally:
            self._closed = True
            if self._previous_handlers is not None:
                for sig, handler in self._previous_handlers.items():
                    if signal.getsignal(sig) == self._handler:
                        signal.signal(sig, handler)
                self._previous_handlers = None
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _worker(request_path: Path) -> None:
    for sig in _SIGNALS:
        signal.signal(sig, _interrupted)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
    started = time.monotonic()
    request = json.loads(request_path.read_text())
    shard_id = request["shard_id"]
    print(f"shard={shard_id} canonical_prefetch_started", flush=True)
    try:
        name = request["worker_factory"]
        module, attribute = name.split(":") if ":" in name else name.rsplit(".", 1)
        prepare = getattr(importlib.import_module(module), attribute)
        # A duplicate node must not republish canonical manifests while the
        # current owner is consuming them in GPU stages. Keep lock order aligned
        # with the synchronous path (invocation, then canonical).
        with (
            file_lock(Path(request["shard_root"]) / "invocation.lock", blocking=True),
            file_lock(Path(request["shard_root"]) / "canonical.lock", blocking=True),
        ):
            result = _validate_result(prepare(request))
        payload = {"shard_id": shard_id, "result": result}
    except Exception as exc:  # noqa: BLE001 - return child failure through the control file
        payload = {"shard_id": shard_id, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.monotonic() - started
    payload["elapsed_seconds"] = elapsed
    _atomic_json(request_path.with_name("result.json"), payload)
    event = "failed" if "error" in payload else "completed"
    print(
        f"shard={shard_id} canonical_prefetch_{event} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    # Remain owned until the parent consumes the result. This lets cleanup see
    # any nested process tree even after a preparation exception.
    while True:
        signal.pause()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
