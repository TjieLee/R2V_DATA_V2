from __future__ import annotations

import contextvars
import json
import os
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.profiling import V3Profiler
from r2v_data_v2.v3.schemas import ClipRecord
from r2v_data_v2.v3.storage import RunStorage

StageHandler = Callable[[str], Mapping[str, object]]


def streaming_model_stage_enabled(config: V3Config, stage: str) -> bool:
    if stage == "remove":
        return config.remove.enabled
    if stage == "reference_edit":
        return config.reference_edit.enabled
    return stage == "segment"


class ClipScopedStorage:
    """Expose one clip to an existing stage while serializing shared writes."""

    def __init__(
        self,
        storage: RunStorage,
        clip_uid: str,
        *,
        shared_write_lock: threading.Lock | None = None,
    ) -> None:
        self._storage = storage
        self._clip_uid = clip_uid
        self._shared_write_lock = shared_write_lock or threading.Lock()
        self.stage_counts: dict[str, dict[str, object]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage, name)

    def iter_clips(self) -> Iterator[ClipRecord]:
        yield self._storage.read_clip(self._clip_uid)

    def update_stage_counts(
        self,
        stage: str,
        counts: Mapping[str, object],
    ) -> None:
        self.stage_counts[stage] = dict(counts)

    def append_failure(self, **kwargs: object) -> None:
        with self._shared_write_lock:
            self._storage.append_failure(**kwargs)


@dataclass(frozen=True)
class StreamingStage:
    name: str
    workers: int
    handler: StageHandler
    resource: str = "cpu"


@dataclass(frozen=True)
class StreamingRuntimeResult:
    stage_counts: dict[str, dict[str, object]]
    failed_tasks: list[dict[str, str]]
    completion_order: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_counts": self.stage_counts,
            "failed_tasks": self.failed_tasks,
            "completion_order": self.completion_order,
        }


@dataclass
class _Task:
    clip_uid: str
    stage_index: int
    enqueued_at: float


@dataclass(frozen=True)
class _TaskOutcome:
    counts: dict[str, object]
    queue_wait_seconds: float
    service_seconds: float
    inflight: int
    error: Exception | None = None


def _merge_counts(
    destination: dict[str, object],
    values: Mapping[str, object],
) -> None:
    for name, value in values.items():
        if name in {"sam3_compile_requested", "sam3_compile_effective"}:
            destination[name] = bool(destination.get(name, False)) or bool(value)
        elif name == "sam3_compile_failure_reason":
            if value is not None and destination.get(name) is None:
                destination[name] = value
            elif name not in destination:
                destination[name] = None
        elif isinstance(value, int) and not isinstance(value, bool):
            destination[name] = int(destination.get(name, 0)) + value
        elif isinstance(value, float):
            destination[name] = float(destination.get(name, 0.0)) + value
        elif name not in destination:
            destination[name] = value


class StreamingDAGScheduler:
    """Run a linear per-clip DAG with bounded per-stage concurrency."""

    def __init__(
        self,
        stages: list[StreamingStage],
        *,
        cpu_workers: int,
        profiler: V3Profiler | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if not stages:
            raise ValueError("streaming scheduler requires at least one stage")
        if not isinstance(cpu_workers, int) or cpu_workers < 1:
            raise ValueError("cpu_workers must be a positive integer")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("streaming stage names must be unique")
        if any(stage.workers < 1 for stage in stages):
            raise ValueError("streaming stage workers must be positive")
        self.stages = stages
        self.cpu_workers = cpu_workers
        self.profiler = profiler
        self.clock = clock
        self._clip_locks: dict[str, threading.Lock] = {}
        self._cpu_budget = threading.BoundedSemaphore(cpu_workers)
        self._state_lock = threading.Lock()
        self._inflight = {stage.name: 0 for stage in stages}

    def _run_task(
        self,
        task: _Task,
    ) -> _TaskOutcome:
        stage = self.stages[task.stage_index]
        cpu_context: AbstractContextManager[object]
        if stage.resource == "cpu":
            cpu_context = _SemaphoreContext(self._cpu_budget)
        else:
            cpu_context = _NullContext()
        with cpu_context, self._clip_locks[task.clip_uid]:
            started = float(self.clock())
            with self._state_lock:
                self._inflight[stage.name] += 1
                inflight = self._inflight[stage.name]
            try:
                try:
                    result = dict(stage.handler(task.clip_uid))
                    error = None
                except Exception as exc:  # noqa: BLE001 - isolate clip tasks
                    result = {"runtime_failed": 1}
                    error = exc
            finally:
                with self._state_lock:
                    self._inflight[stage.name] -= 1
        finished = float(self.clock())
        return _TaskOutcome(
            counts=result,
            queue_wait_seconds=started - task.enqueued_at,
            service_seconds=finished - started,
            inflight=inflight,
            error=error,
        )

    def run(self, clip_uids: list[str]) -> StreamingRuntimeResult:
        ordered_uids = sorted(dict.fromkeys(clip_uids))
        self._clip_locks = {clip_uid: threading.Lock() for clip_uid in ordered_uids}
        started_by_clip = {clip_uid: float(self.clock()) for clip_uid in ordered_uids}
        executors = {
            stage.name: ThreadPoolExecutor(
                max_workers=stage.workers,
                thread_name_prefix=f"v3-{stage.name}",
            )
            for stage in self.stages
        }
        futures: dict[Future[_TaskOutcome], _Task] = {}
        stage_counts = {stage.name: {} for stage in self.stages}
        failures: list[dict[str, str]] = []
        completion_order: list[str] = []

        def submit(clip_uid: str, stage_index: int) -> None:
            stage = self.stages[stage_index]
            task = _Task(
                clip_uid=clip_uid,
                stage_index=stage_index,
                enqueued_at=float(self.clock()),
            )
            context = contextvars.copy_context()
            future = executors[stage.name].submit(context.run, self._run_task, task)
            futures[future] = task

        try:
            for clip_uid in ordered_uids:
                submit(clip_uid, 0)
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(
                    completed,
                    key=lambda item: (
                        futures[item].stage_index,
                        futures[item].clip_uid,
                    ),
                ):
                    task = futures.pop(future)
                    stage = self.stages[task.stage_index]
                    outcome = future.result()
                    if outcome.error is not None:
                        failures.append(
                            {
                                "clip_uid": task.clip_uid,
                                "stage": stage.name,
                                "error_type": type(outcome.error).__name__,
                                "reason": str(outcome.error),
                            }
                        )
                    _merge_counts(stage_counts[stage.name], outcome.counts)
                    if self.profiler is not None:
                        self.profiler.record_runtime_stage(
                            stage=stage.name,
                            clip_uid=task.clip_uid,
                            queue_wait_seconds=outcome.queue_wait_seconds,
                            service_seconds=outcome.service_seconds,
                            success=outcome.error is None,
                            inflight=outcome.inflight,
                            resource=stage.resource,
                            error_type=(
                                type(outcome.error).__name__
                                if outcome.error is not None
                                else None
                            ),
                        )
                    next_index = task.stage_index + 1
                    if outcome.error is not None:
                        completion_order.append(task.clip_uid)
                    elif next_index < len(self.stages):
                        submit(task.clip_uid, next_index)
                    else:
                        completion_order.append(task.clip_uid)
                    if (
                        outcome.error is not None
                        or next_index == len(self.stages)
                    ) and self.profiler is not None:
                        self.profiler.record_clip_runtime(
                            clip_uid=task.clip_uid,
                            end_to_end_clip_seconds=(
                                float(self.clock()) - started_by_clip[task.clip_uid]
                            ),
                        )
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
        segment_counts = stage_counts.get("segment")
        if segment_counts is not None:
            steady_seconds = segment_counts.get(
                "sam3_steady_state_segment_seconds", 0.0
            )
            steady_clips = segment_counts.get("sam3_steady_state_segment_clips", 0)
            if (
                isinstance(steady_seconds, (int, float))
                and not isinstance(steady_seconds, bool)
                and isinstance(steady_clips, int)
                and not isinstance(steady_clips, bool)
                and steady_clips > 0
            ):
                segment_counts["steady_state_segment_mean_seconds"] = (
                    float(steady_seconds) / steady_clips
                )
            else:
                segment_counts["steady_state_segment_mean_seconds"] = 0.0
        return StreamingRuntimeResult(
            stage_counts=stage_counts,
            failed_tasks=sorted(
                failures,
                key=lambda item: (item["clip_uid"], item["stage"]),
            ),
            completion_order=sorted(completion_order),
        )


class _SemaphoreContext:
    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self.semaphore = semaphore

    def __enter__(self) -> None:
        self.semaphore.acquire()

    def __exit__(self, *_: object) -> None:
        self.semaphore.release()


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


@dataclass(frozen=True)
class StageWorkerConfig:
    stage: str
    config_path: Path
    cuda_visible_devices: str
    overwrite: bool
    timeout_seconds: int
    profile: bool
    stderr_log_path: Path
    attribute_probe_only: bool = False
    worker_script: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "run_v3_streaming_stage_worker.py"
        )
    )


class PersistentStageProcess:
    """Own one model-bearing stage process and exchange deterministic JSONL."""

    def __init__(self, config: StageWorkerConfig) -> None:
        if config.stage not in {"segment", "remove", "reference_edit"}:
            raise ValueError("persistent worker stage is not model-bearing")
        if not config.cuda_visible_devices.isdigit():
            raise ValueError("CUDA_VISIBLE_DEVICES must be one physical CUDA index")
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr: TextIO | None = None
        self._request_lock = threading.Lock()
        self.starts = 0

    def _read(self) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("streaming stage worker is not running")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(self.config.timeout_seconds)
        finally:
            selector.close()
        if not events:
            raise TimeoutError(
                f"{self.config.stage} worker response timed out after "
                f"{self.config.timeout_seconds}s"
            )
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(
                f"{self.config.stage} worker exited without a response: {code}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.config.stage} worker returned invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise TypeError(f"{self.config.stage} worker response must be an object")
        return value

    def _write(self, payload: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("streaming stage worker is not running")
        process.stdin.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        process.stdin.flush()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("streaming stage worker is already started")
        config = self.config
        config.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = config.stderr_log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        command = [
            sys.executable,
            str(config.worker_script),
            "--stage",
            config.stage,
            "--config",
            str(config.config_path),
        ]
        if config.overwrite:
            command.append("--overwrite")
        if config.profile:
            command.append("--profile")
        if config.attribute_probe_only:
            command.append("--attribute-probe-only")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
            )
            ready = self._read()
            if ready != {"schema_version": 1, "status": "ok", "type": "ready"}:
                raise RuntimeError(
                    f"invalid {config.stage} worker startup response: {ready}"
                )
            self.starts += 1
        except Exception:
            self.terminate()
            raise

    def _exchange(self, payload: Mapping[str, object]) -> dict[str, object]:
        request_id = uuid.uuid4().hex
        self._write(
            {
                "schema_version": 1,
                "request_id": request_id,
                **dict(payload),
            }
        )
        response = self._read()
        if response.get("request_id") != request_id:
            self.terminate()
            raise RuntimeError("streaming stage worker response ID mismatch")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("reason") or "worker failed"))
        return response

    def request(self, clip_uid: str) -> dict[str, object]:
        with self._request_lock:
            response = self._exchange(
                {
                    "type": "run_clip",
                    "clip_uid": clip_uid,
                }
            )
            counts = response.get("counts")
            if not isinstance(counts, dict):
                raise TypeError("streaming stage worker omitted stage counts")
            return counts

    def attribute_probe(
        self,
        *,
        clip_uid: str,
        frame_slot: int,
        source_frame_index: int,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]:
        if self.config.stage != "segment":
            raise RuntimeError("attribute probes require the persistent segment worker")
        with self._request_lock:
            response = self._exchange(
                {
                    "type": "attribute_probe",
                    "clip_uid": clip_uid,
                    "frame_slot": frame_slot,
                    "source_frame_index": source_frame_index,
                    "grounding_prompt": grounding_prompt,
                }
            )
        masks = response.get("masks")
        if not isinstance(masks, list):
            raise TypeError("segment worker omitted attribute probe masks")
        return tuple(decode_binary_mask(mask) for mask in masks)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            request_id = uuid.uuid4().hex
            self._write(
                {
                    "schema_version": 1,
                    "type": "shutdown",
                    "request_id": request_id,
                }
            )
            response = self._read()
            if response.get("request_id") != request_id or response.get("status") != "ok":
                raise RuntimeError("streaming stage worker shutdown failed")
            process.wait(timeout=10)
        finally:
            self.terminate()

    def terminate(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


def runtime_worker_config(
    config: V3Config,
    *,
    config_path: str | Path,
    stage: str,
    gpu_assignment: str | None = None,
    overwrite: bool,
    profile: bool,
) -> StageWorkerConfig:
    assignment = gpu_assignment or stage
    visible = getattr(config.runtime.gpu_workers, assignment)
    if visible is None:
        raise ValueError(
            f"runtime.gpu_workers.{assignment} is required for streaming model "
            "isolation"
        )
    return StageWorkerConfig(
        stage=stage,
        config_path=Path(config_path).expanduser().resolve(strict=True),
        cuda_visible_devices=visible,
        overwrite=overwrite,
        timeout_seconds=config.runtime.worker_timeout_seconds,
        profile=profile,
        attribute_probe_only=assignment == "subject_attributes_segment",
        stderr_log_path=(
            config.resolved_run_root
            / "logs"
            / f"streaming_{assignment}_worker.stderr.log"
        ),
    )
