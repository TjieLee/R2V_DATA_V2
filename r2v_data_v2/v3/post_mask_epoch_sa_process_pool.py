"""Execution-only SA SAM processes; no durable state or writer lives here.

Each spawned child has one serial model and one synchronous Pipe RPC at a time.
The parent SA admission budget also covers returned masks awaiting persistence;
there is no feeder thread, hidden result queue, or unbounded affinity table.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .post_mask_epoch_resources import EpochResourceError, WorkerPoolConfig


def _build_backend(config: Any) -> Any:
    # Import only after the spawned entry point has isolated the environment.
    from .pre_qwen_production import enable_sam3_session_reuse
    from .sam3_backend import Sam3SegmentationBackend
    from .subject_attributes import Sam3AttributeFrameSegmenter

    backend = Sam3SegmentationBackend(config)
    try:
        enable_sam3_session_reuse(backend, mode="clip_reset_v1")
        backend._load_predictor()
    except BaseException:
        backend.close()
        raise
    return Sam3AttributeFrameSegmenter(config, backend=backend)


def _error_response(exc: BaseException) -> tuple:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return ("fatal", type(exc).__name__, exc.args)
    # Once inside model code, even TypeError/ValueError may be CUDA/Triton
    # failures. Only the explicit input checks in _request are clip-local.
    return ("infrastructure", type(exc).__name__, str(exc))


def _worker_main(connection: Any, config: Any, gpu_id: int, factory: Any) -> None:
    # This module and its spawn-time imports must remain Torch-free.
    for key in tuple(os.environ):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                   "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK",
                   "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
            os.environ.pop(key, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    backend = None
    try:
        try:
            backend = (factory or _build_backend)(config)
        except BaseException as exc:  # noqa: BLE001 - transport child startup failures
            connection.send(_error_response(exc))
            return
        connection.send(("ready",))
        while True:
            request = connection.recv()
            if request is None:
                return
            operation, arguments = request
            try:
                returned = getattr(backend, operation)(**arguments)
            except BaseException as exc:  # noqa: BLE001 - transport fatal control flow too
                connection.send(_error_response(exc))
                return
            try:
                import numpy as np

                masks = []
                for mask in returned:
                    if hasattr(mask, "detach"):
                        mask = mask.detach().cpu().numpy()
                    masks.append(np.array(mask, copy=True))
                connection.send(("ok", tuple(masks)))
            except BaseException as exc:  # noqa: BLE001 - snapshot/IPC errors are infrastructure
                connection.send(_error_response(exc))
                return
    except (EOFError, OSError):
        return
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            connection.close()


class _SAMHandle:
    def __init__(self, owner: SASamProcessPool, slot: int) -> None:
        self._owner, self._slot = owner, slot

    def segment_frame(self, *, frame_path: Path, frame_slot: int,
                      grounding_prompt: str) -> tuple:
        return self._owner._request(self._slot, "segment_frame", {
            "frame_path": frame_path, "frame_slot": frame_slot,
            "grounding_prompt": grounding_prompt,
        })

    def segment_generated_frame(self, *, frame_path: Path,
                                grounding_prompt: str) -> tuple:
        return self._owner._request(self._slot, "segment_generated_frame", {
            "frame_path": frame_path, "grounding_prompt": grounding_prompt,
        })


class SASamProcessPool:
    """Stage-owned spawn pool, with at most one request/result per child.

    Slot handles are lightweight parent proxies, not shared CUDA predictors.
    Dispatch prefers an idle worker that last saw the same frames directory,
    then the caller's idle slot, then any idle worker. Only one affinity key per
    child is retained. Busy affinity never blocks a different idle worker.
    """

    def __init__(self, config: Any, *, pool: WorkerPoolConfig,
                 workers_per_gpu: int, backend_factory: Any = None) -> None:
        if isinstance(workers_per_gpu, bool) or workers_per_gpu not in (1, 2, 4):
            raise ValueError("workers_per_gpu must be one of 1, 2, 4")
        if not pool.gpu_ids or min(pool.timeout_seconds, pool.shutdown_grace_seconds) <= 0:
            raise ValueError("SA pool needs GPUs and positive timeout/grace")
        self.config, self.pool = config, pool
        self.workers_per_gpu = workers_per_gpu
        self.backend_factory = backend_factory
        self.slot_count = len(pool.gpu_ids) * workers_per_gpu
        self._children: list[Any] = []
        self._connections: list[Any] = []
        self._available: set[int] = set()
        self._last_keys: dict[int, str] = {}
        self._condition = threading.Condition()
        self._stop_lock = threading.Lock()
        self._started = False
        self._closing = False
        self._failure: str | None = None
        self._active = 0
        self._active_peak = 0
        self._calls = 0

    def start(self) -> None:
        if self._started or self._children or self._active:
            raise EpochResourceError("SA SAM pool is already started")
        self._closing = False
        self._failure = None
        context = multiprocessing.get_context("spawn")
        # Keep worker creation injectable for CPU-only validation.
        if hasattr(self.config, "__dataclass_fields__"):
            config = replace(self.config, device="cuda:0")
        else:
            import copy

            config = copy.copy(self.config)
            config.device = "cuda:0"
        try:
            for gpu_id in self.pool.gpu_ids:
                for _ in range(self.workers_per_gpu):
                    parent, child = context.Pipe(duplex=True)
                    process = context.Process(target=_worker_main,
                        args=(child, config, gpu_id, self.backend_factory),
                        name=f"sa-sam-gpu-{gpu_id}-worker-{len(self._children)}")
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    self._children.append(process)
                    self._connections.append(parent)
            deadline = time.monotonic() + self.pool.timeout_seconds
            for index in range(self.slot_count):
                response = self._receive(index, deadline)
                if response != ("ready",):
                    self._raise_response(response)
                    raise EpochResourceError("invalid SA worker ready response")
            self._available = set(range(self.slot_count))
            self._started = True
        except BaseException as exc:
            self.stop()
            if isinstance(exc, Exception) and not isinstance(exc, EpochResourceError):
                raise EpochResourceError(f"SA SAM startup failed: {exc}") from exc
            raise

    def _receive(self, index: int, deadline: float) -> tuple:
        connection = self._connections[index]
        process = self._children[index]
        while True:
            if connection.poll(0.05):
                return connection.recv()
            if not process.is_alive():
                raise EpochResourceError(
                    f"SA SAM child {index} exited: {process.exitcode}")
            if time.monotonic() >= deadline:
                raise EpochResourceError(f"SA SAM child {index} timed out")

    @staticmethod
    def _raise_response(response: tuple) -> None:
        if not isinstance(response, tuple) or len(response) != 3:
            raise EpochResourceError("invalid SA SAM worker reply")
        kind, name, detail = response
        if kind == "fatal":
            fatal = KeyboardInterrupt if name == "KeyboardInterrupt" else SystemExit
            raise fatal(*detail)
        raise EpochResourceError(f"SA SAM {name}: {detail}")

    def handle_for_slot(self, index: int) -> _SAMHandle:
        if not 0 <= index < self.slot_count:
            raise EpochResourceError(f"invalid SA SAM slot: {index}")
        return _SAMHandle(self, index)

    def _request(self, preferred: int, operation: str, arguments: dict) -> tuple:
        # A narrow pre-model input boundary, not exception-type inference.
        # Keep missing frames/empty prompts local; model and IPC failures below
        # must reach the scheduler uncommitted as EpochResourceError.
        if not arguments["grounding_prompt"].strip():
            raise ValueError("attribute grounding prompt must not be empty")
        try:
            frame = Path(arguments["frame_path"]).expanduser().resolve()
            exists = frame.is_file()
        except OSError as exc:
            raise EpochResourceError(f"SA SAM input access failed: {exc}") from exc
        if not exists:
            raise FileNotFoundError(f"attribute candidate frame is missing: {frame}")
        arguments = {**arguments, "frame_path": frame}
        key = str(frame.parent)
        with self._condition:
            while not self._available:
                if not self._started or self._closing or self._failure:
                    raise EpochResourceError(self._failure or "SA SAM pool is closed")
                self._condition.wait(0.05)
            if not self._started or self._closing or self._failure:
                raise EpochResourceError(self._failure or "SA SAM pool is closed")
            matching = [index for index in sorted(self._available)
                        if self._last_keys.get(index) == key]
            index = matching[0] if matching else (
                preferred if preferred in self._available else min(self._available))
            self._available.remove(index)
            self._active += 1
            self._active_peak = max(self._active_peak, self._active)
        try:
            try:
                self._connections[index].send((operation, arguments))
                response = self._receive(index, time.monotonic() + self.pool.timeout_seconds)
            except (EOFError, OSError, ValueError) as exc:
                raise EpochResourceError(f"SA SAM IPC failed: {exc}") from exc
            if not isinstance(response, tuple) or not response:
                raise EpochResourceError("invalid SA SAM worker reply")
            if response[0] != "ok":
                self._raise_response(response)
            if len(response) != 2 or not isinstance(response[1], tuple):
                raise EpochResourceError("invalid SA SAM mask reply")
            return response[1]
        except EpochResourceError as exc:
            with self._condition:
                self._failure = str(exc)
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._calls += 1
                if not self._closing:
                    self._last_keys[index] = key
                if self._started and index < len(self._children):
                    self._available.add(index)
                self._condition.notify_all()

    def counters(self) -> dict[str, int]:
        with self._condition:
            return {"sa_sam_processes": len(self._children),
                    "sa_sam_active_peak": self._active_peak,
                    "sa_sam_requests": self._calls}

    def stop(self) -> None:
        with self._stop_lock:
            deadline = time.monotonic() + self.pool.shutdown_grace_seconds
            with self._condition:
                self._closing = True
                self._condition.notify_all()
                while self._active and time.monotonic() < deadline:
                    self._condition.wait(min(0.05, max(0, deadline - time.monotonic())))
                idle = set(self._available)
                # Startup failures have no admitted requests.
                if not self._started:
                    idle = set(range(len(self._children)))
            for index in idle:
                try:
                    if self._children[index].is_alive():
                        self._connections[index].send(None)
                except (EOFError, OSError, ValueError):
                    pass
            for child in self._children:
                child.join(max(0, deadline - time.monotonic()))
            forced = set()
            for child in self._children:
                if child.is_alive():
                    forced.add(child.pid)
                    child.terminate()
            terminate_deadline = time.monotonic() + 0.2
            for child in self._children:
                child.join(max(0, terminate_deadline - time.monotonic()))
            for child in self._children:
                if child.is_alive():
                    child.kill()
            kill_deadline = time.monotonic() + 1
            for child in self._children:
                child.join(max(0, kill_deadline - time.monotonic()))
            failures = [child.exitcode for child in self._children
                        if child.pid not in forced and child.exitcode != 0]
            for connection in self._connections:
                connection.close()
            with self._condition:
                self._started = False
                self._available.clear()
                self._condition.notify_all()
            unreaped = [child.pid for child in self._children if child.is_alive()]
            if unreaped:
                # Keep owned Process references (and their closed Pipe handles)
                # until a later stop can reap them; never forget a live child.
                self._failure = f"SA SAM shutdown left unreaped owned children: {unreaped}"
                raise EpochResourceError(self._failure)
            self._children = []
            self._connections = []
            self._last_keys.clear()
            if failures and not self._failure:
                raise EpochResourceError(f"SA SAM shutdown failed: child exit codes {failures}")
