from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from r2v_data_v2.reconciliation import write_json_atomic

_ACTIVE_PROFILER: ContextVar[V3Profiler | None] = ContextVar(
    "v3_active_profiler",
    default=None,
)
_MODEL_PROFILE_CONTEXT: ContextVar[ModelProfileContext | None] = ContextVar(
    "v3_model_profile_context",
    default=None,
)
_QWEN_CONCURRENCY_GATE: ContextVar[Any | None] = ContextVar(
    "v3_qwen_concurrency_gate",
    default=None,
)
_FORBIDDEN_EVENT_KEYS = frozenset(
    {"messages", "prompt", "response", "image_bytes", "video_bytes"}
)


def _usage_value(usage: object, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _response_content(response: object) -> str | None:
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return str(content) if content is not None else None


def summarize_openai_messages(
    messages: list[dict[str, object]],
) -> tuple[int, int, dict[str, object]]:
    text_chars = 0
    image_count = 0
    video_input = False
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                text_chars += len(item["text"])
            elif item_type == "image_url":
                image_count += 1
            elif item_type == "video_url":
                video_input = True
    metadata: dict[str, object] = {}
    if video_input:
        metadata["video_input"] = True
    return text_chars, image_count, metadata


@dataclass
class ModelCallObservation:
    output_text_chars: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def observe_response(self, response: object) -> None:
        content = _response_content(response)
        self.output_text_chars = len(content) if content is not None else None
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, Mapping):
            usage = response.get("usage")
        if usage is None:
            return
        self.prompt_tokens = _usage_value(usage, "prompt_tokens")
        self.completion_tokens = _usage_value(usage, "completion_tokens")
        self.total_tokens = _usage_value(usage, "total_tokens")


@dataclass
class StageObservation:
    counters: dict[str, object] = field(default_factory=dict)

    def set_counters(self, counters: Mapping[str, object]) -> None:
        self.counters = dict(counters)


@dataclass(frozen=True)
class ModelProfileContext:
    retry_index: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QwenGateObservation:
    queue_wait_seconds: float
    inflight: int
    qwen_slot: int


class QwenConcurrencyGate:
    """Bound Qwen calls across threads and isolated local worker processes."""

    def __init__(
        self,
        maximum: int,
        *,
        lock_directory: str | Path,
        clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("Qwen concurrency maximum must be a positive integer")
        self.maximum = maximum
        self.lock_directory = Path(lock_directory).expanduser().resolve(strict=False)
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._local_budget = threading.BoundedSemaphore(maximum)
        self._slot_locks = [threading.Lock() for _ in range(maximum)]
        self._state_lock = threading.Lock()
        self._ticket = 0
        self._inflight = 0

    @contextmanager
    def acquire(self) -> Iterator[QwenGateObservation]:
        started = float(self._clock())
        self._local_budget.acquire()
        with self._state_lock:
            start_slot = (os.getpid() + self._ticket) % self.maximum
            self._ticket += 1
        slot_index: int | None = None
        slot_lock: threading.Lock | None = None
        handle: Any | None = None
        entered = False
        locked = False
        try:
            while slot_index is None:
                for offset in range(self.maximum):
                    candidate_index = (start_slot + offset) % self.maximum
                    candidate_lock = self._slot_locks[candidate_index]
                    if not candidate_lock.acquire(blocking=False):
                        continue
                    candidate_handle = None
                    try:
                        candidate_handle = (
                            self.lock_directory / f"slot-{candidate_index}.lock"
                        ).open("a+")
                        try:
                            fcntl.flock(
                                candidate_handle.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                        except BlockingIOError:
                            continue
                        slot_index = candidate_index
                        slot_lock = candidate_lock
                        handle = candidate_handle
                        locked = True
                        break
                    finally:
                        if slot_index != candidate_index:
                            if candidate_handle is not None:
                                candidate_handle.close()
                            candidate_lock.release()
                if slot_index is None:
                    time.sleep(0.005)
                    start_slot = (start_slot + 1) % self.maximum
            with self._state_lock:
                self._inflight += 1
                inflight = self._inflight
                entered = True
            yield QwenGateObservation(
                queue_wait_seconds=float(self._clock()) - started,
                inflight=inflight,
                qwen_slot=slot_index,
            )
        finally:
            if entered:
                with self._state_lock:
                    self._inflight -= 1
            if locked:
                assert handle is not None
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if handle is not None:
                handle.close()
            if slot_lock is not None:
                slot_lock.release()
            self._local_budget.release()


class V3Profiler:
    def __init__(
        self,
        run_root: str | Path,
        *,
        git_commit: str,
        clock: Any = time.monotonic,
    ) -> None:
        self.run_root = Path(run_root).expanduser().resolve(strict=False)
        self.directory = self.run_root / "profiling"
        self.events_path = self.directory / "events.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.git_commit = git_commit
        self._clock = clock
        self._started = float(clock())
        self._thread_lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._events_lock_path = self.directory / "events.lock"
        self._events = self._load_existing_events()
        self._previous_profiled_seconds = self._load_previous_profiled_seconds()
        self.events_path.touch(exist_ok=True)

    def _load_existing_events(self) -> list[dict[str, object]]:
        if not self.events_path.is_file():
            return []
        events: list[dict[str, object]] = []
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"profiling event line {line_number} must contain an object"
                )
            events.append(value)
        return events

    def _load_previous_profiled_seconds(self) -> float:
        if not self.summary_path.is_file():
            return 0.0
        value = json.loads(self.summary_path.read_text(encoding="utf-8"))
        duration = value.get("total_profiled_seconds") if isinstance(value, dict) else 0
        return float(duration) if isinstance(duration, (int, float)) else 0.0

    def _validate_event(self, event: dict[str, object]) -> str:
        forbidden = _FORBIDDEN_EVENT_KEYS.intersection(event)
        if forbidden:
            raise ValueError(f"profiling event contains forbidden keys: {sorted(forbidden)}")
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
        lowered = encoded.lower()
        if "data:image" in lowered or "base64" in lowered:
            raise ValueError("profiling event contains encoded media")
        return encoded

    def record_call(self, event: dict[str, object]) -> None:
        encoded = self._validate_event(event)
        with self._thread_lock:
            with self._events_lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    with self.events_path.open("a", encoding="utf-8") as handle:
                        handle.write(encoded)
                        handle.write("\n")
                        handle.flush()
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            self._events.append(event)

    def record_runtime_stage(
        self,
        *,
        stage: str,
        clip_uid: str,
        queue_wait_seconds: float,
        service_seconds: float,
        success: bool,
        inflight: int,
        resource: str,
        error_type: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "kind": "runtime_stage",
            "stage": stage,
            "clip_uid": clip_uid,
            "queue_wait_seconds": queue_wait_seconds,
            "service_seconds": service_seconds,
            "success": success,
            "inflight": inflight,
            "resource": resource,
        }
        if error_type is not None:
            event["error_type"] = error_type
        self.record_call(event)

    def record_clip_runtime(
        self,
        *,
        clip_uid: str,
        end_to_end_clip_seconds: float,
    ) -> None:
        self.record_call(
            {
                "kind": "runtime_clip",
                "clip_uid": clip_uid,
                "end_to_end_clip_seconds": end_to_end_clip_seconds,
            }
        )

    def _stage_summary(self) -> dict[str, dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for event in self._events:
            if event.get("kind") != "stage":
                continue
            stage = event.get("stage")
            if not isinstance(stage, str):
                continue
            item = summary.setdefault(
                stage,
                {"calls": 0, "total_seconds": 0.0, "successes": 0},
            )
            item["calls"] = int(item["calls"]) + 1
            item["total_seconds"] = float(item["total_seconds"]) + float(
                event.get("duration_seconds", 0.0)
            )
            item["successes"] = int(item["successes"]) + int(
                event.get("success") is True
            )
        return summary

    def _component_summary(self) -> dict[str, dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for event in self._events:
            component = event.get("component")
            if event.get("kind") == "model_call" and isinstance(component, str):
                grouped.setdefault(component, []).append(event)
        summary: dict[str, dict[str, object]] = {}
        for component, events in grouped.items():
            durations = [float(event.get("duration_seconds", 0.0)) for event in events]
            prompt_values = [
                event["prompt_tokens"]
                for event in events
                if isinstance(event.get("prompt_tokens"), int)
            ]
            completion_values = [
                event["completion_tokens"]
                for event in events
                if isinstance(event.get("completion_tokens"), int)
            ]
            summary[component] = {
                "calls": len(events),
                "successful_calls": sum(event.get("success") is True for event in events),
                "failed_calls": sum(event.get("success") is False for event in events),
                "total_seconds": sum(durations),
                "mean_seconds": sum(durations) / len(durations),
                "max_seconds": max(durations),
                "initial_calls": sum(event.get("retry_index") == 0 for event in events),
                "repair_calls": sum(
                    isinstance(event.get("retry_index"), int)
                    and int(event["retry_index"]) > 0
                    for event in events
                ),
                "input_images_total": sum(
                    int(event.get("input_image_count", 0)) for event in events
                ),
                "input_text_chars_total": sum(
                    int(event.get("input_text_chars", 0)) for event in events
                ),
                "prompt_tokens_total": sum(prompt_values) if prompt_values else None,
                "completion_tokens_total": (
                    sum(completion_values) if completion_values else None
                ),
            }
        return summary

    def write_summary(self) -> dict[str, object]:
        elapsed = float(self._clock()) - self._started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("profiling clock produced an invalid elapsed duration")
        with self._thread_lock:
            self._events = self._load_existing_events()
        runtime_stages: dict[str, dict[str, object]] = {}
        for event in self._events:
            if event.get("kind") != "runtime_stage":
                continue
            stage = event.get("stage")
            if not isinstance(stage, str):
                continue
            item = runtime_stages.setdefault(
                stage,
                {
                    "calls": 0,
                    "successes": 0,
                    "queue_wait_seconds": 0.0,
                    "service_seconds": 0.0,
                    "maximum_inflight": 0,
                },
            )
            item["calls"] = int(item["calls"]) + 1
            item["successes"] = int(item["successes"]) + int(
                event.get("success") is True
            )
            item["queue_wait_seconds"] = float(item["queue_wait_seconds"]) + float(
                event.get("queue_wait_seconds", 0.0)
            )
            item["service_seconds"] = float(item["service_seconds"]) + float(
                event.get("service_seconds", 0.0)
            )
            item["maximum_inflight"] = max(
                int(item["maximum_inflight"]), int(event.get("inflight", 0))
            )
        for item in runtime_stages.values():
            service = float(item["service_seconds"])
            item["throughput_clips_per_hour"] = (
                3600.0 * int(item["successes"]) / service if service > 0 else None
            )
        bottleneck = None
        throughputs = {
            stage: float(item["throughput_clips_per_hour"])
            for stage, item in runtime_stages.items()
            if isinstance(item.get("throughput_clips_per_hour"), (int, float))
        }
        if throughputs:
            bottleneck = min(throughputs, key=throughputs.get)
        clip_durations = [
            float(event["end_to_end_clip_seconds"])
            for event in self._events
            if event.get("kind") == "runtime_clip"
            and isinstance(event.get("end_to_end_clip_seconds"), (int, float))
        ]
        gpu_busy = {
            stage: float(item["service_seconds"])
            for stage, item in runtime_stages.items()
            if stage in {"segment", "remove", "reference_edit"}
        }
        summary: dict[str, object] = {
            "schema_version": 1,
            "git_commit": self.git_commit,
            "total_profiled_seconds": self._previous_profiled_seconds + elapsed,
            "stages": self._stage_summary(),
            "components": self._component_summary(),
            "runtime": {
                "stages": runtime_stages,
                "critical_bottleneck": bottleneck,
                "mean_end_to_end_clip_seconds": (
                    sum(clip_durations) / len(clip_durations)
                    if clip_durations
                    else None
                ),
                "gpu_worker_busy_seconds": gpu_busy,
                "gpu_worker_idle_seconds": {
                    stage: max(0.0, elapsed - busy)
                    for stage, busy in gpu_busy.items()
                },
            },
        }
        write_json_atomic(self.summary_path, summary)
        return summary


def get_active_profiler() -> V3Profiler | None:
    return _ACTIVE_PROFILER.get()


def get_model_profile_context() -> ModelProfileContext:
    return _MODEL_PROFILE_CONTEXT.get() or ModelProfileContext()


@contextmanager
def model_profile_context(
    *,
    retry_index: int,
    metadata: Mapping[str, object] | None = None,
) -> Iterator[None]:
    if get_active_profiler() is None:
        yield
        return
    token = _MODEL_PROFILE_CONTEXT.set(
        ModelProfileContext(
            retry_index=retry_index,
            metadata=dict(metadata or {}),
        )
    )
    try:
        yield
    finally:
        _MODEL_PROFILE_CONTEXT.reset(token)


@contextmanager
def active_profiler(profiler: V3Profiler) -> Iterator[V3Profiler]:
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def qwen_concurrency_gate(gate: QwenConcurrencyGate) -> Iterator[None]:
    token = _QWEN_CONCURRENCY_GATE.set(gate)
    try:
        yield
    finally:
        _QWEN_CONCURRENCY_GATE.reset(token)


@contextmanager
def profile_stage(stage: str) -> Iterator[StageObservation]:
    observation = StageObservation()
    profiler = get_active_profiler()
    if profiler is None:
        yield observation
        return
    started = profiler._clock()
    try:
        yield observation
    except BaseException as exc:
        profiler.record_call(
            {
                "kind": "stage",
                "stage": stage,
                "duration_seconds": float(profiler._clock() - started),
                "success": False,
                "counters": observation.counters,
                "error_type": type(exc).__name__,
            }
        )
        raise
    else:
        profiler.record_call(
            {
                "kind": "stage",
                "stage": stage,
                "duration_seconds": float(profiler._clock() - started),
                "success": True,
                "counters": observation.counters,
            }
        )


@contextmanager
def profile_model_call(
    *,
    component: str,
    operation: str,
    retry_index: int,
    model: str | None = None,
    input_text_chars: int = 0,
    input_image_count: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> Iterator[ModelCallObservation]:
    observation = ModelCallObservation()
    profiler = get_active_profiler()
    if profiler is None:
        yield observation
        return
    started = profiler._clock()
    base_event: dict[str, object] = {
        "kind": "model_call",
        "component": component,
        "operation": operation,
        "retry_index": retry_index,
        "model": model,
        "input_text_chars": input_text_chars,
        "input_image_count": input_image_count,
        "metadata": dict(metadata or {}),
    }
    try:
        yield observation
    except BaseException as exc:
        profiler.record_call(
            {
                **base_event,
                "duration_seconds": float(profiler._clock() - started),
                "success": False,
                "output_text_chars": observation.output_text_chars,
                "prompt_tokens": observation.prompt_tokens,
                "completion_tokens": observation.completion_tokens,
                "total_tokens": observation.total_tokens,
                "error_type": type(exc).__name__,
            }
        )
        raise
    else:
        profiler.record_call(
            {
                **base_event,
                "duration_seconds": float(profiler._clock() - started),
                "success": True,
                "output_text_chars": observation.output_text_chars,
                "prompt_tokens": observation.prompt_tokens,
                "completion_tokens": observation.completion_tokens,
                "total_tokens": observation.total_tokens,
            }
        )


def profiled_openai_call(
    request: Callable[[], object],
    *,
    component: str,
    operation: str,
    retry_index: int,
    model: str | None,
    messages: list[dict[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> object:
    text_chars, image_count, message_metadata = summarize_openai_messages(messages)
    combined_metadata = {**message_metadata, **dict(metadata or {})}
    gate = _QWEN_CONCURRENCY_GATE.get()
    gate_context = gate.acquire() if gate is not None else nullcontext(None)
    with gate_context as gate_observation:
        if gate_observation is not None:
            combined_metadata.update(
                {
                    "queue_wait_seconds": gate_observation.queue_wait_seconds,
                    "qwen_inflight": gate_observation.inflight,
                    "qwen_slot": gate_observation.qwen_slot,
                }
            )
        with profile_model_call(
            component=component,
            operation=operation,
            retry_index=retry_index,
            model=model,
            input_text_chars=text_chars,
            input_image_count=image_count,
            metadata=combined_metadata,
        ) as observation:
            response = request()
            observation.observe_response(response)
            return response


def _print_report(summary: Mapping[str, object]) -> None:
    print("===== STAGES =====")
    stages = summary.get("stages")
    if isinstance(stages, Mapping):
        for stage, value in stages.items():
            if isinstance(value, Mapping):
                print(f"{stage:<18} {float(value.get('total_seconds', 0.0)):8.1f}s")
    print("\n===== MODEL CALLS =====")
    print(f"{'component':<36} {'calls':>5} {'repairs':>7} {'total_s':>9} {'mean_s':>8} {'max_s':>8}")
    components = summary.get("components")
    if isinstance(components, Mapping):
        for component, value in components.items():
            if not isinstance(value, Mapping):
                continue
            print(
                f"{component:<36} {int(value.get('calls', 0)):>5} "
                f"{int(value.get('repair_calls', 0)):>7} "
                f"{float(value.get('total_seconds', 0.0)):>9.1f} "
                f"{float(value.get('mean_seconds', 0.0)):>8.1f} "
                f"{float(value.get('max_seconds', 0.0)):>8.1f}"
            )
    print("\n===== QWEN PAYLOAD =====")
    candidate = components.get("qwen_candidate_judge") if isinstance(components, Mapping) else None
    if isinstance(candidate, Mapping) and int(candidate.get("calls", 0)):
        calls = int(candidate["calls"])
        initial = int(candidate.get("initial_calls", 0))
        repairs = int(candidate.get("repair_calls", 0))
        print(
            "candidate judge avg images/request: "
            f"{int(candidate.get('input_images_total', 0)) / calls:.2f}"
        )
        print(
            "candidate judge avg text chars/request: "
            f"{int(candidate.get('input_text_chars_total', 0)) / calls:.2f}"
        )
        repair_rate = repairs / initial if initial else 0.0
        print(f"candidate judge repair rate: {repair_rate:.2%}")
    else:
        print("candidate judge: no calls")
    runtime = summary.get("runtime")
    if isinstance(runtime, Mapping):
        print("\n===== STREAMING RUNTIME =====")
        runtime_stages = runtime.get("stages")
        if isinstance(runtime_stages, Mapping):
            for stage, value in runtime_stages.items():
                if not isinstance(value, Mapping):
                    continue
                throughput = value.get("throughput_clips_per_hour")
                rendered = (
                    f"{float(throughput):.1f} clips/hour"
                    if isinstance(throughput, (int, float))
                    else "n/a"
                )
                print(f"{stage:<24} {rendered}")
        print(f"critical bottleneck: {runtime.get('critical_bottleneck') or 'n/a'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a V3 profiling summary")
    parser.add_argument("run_root")
    args = parser.parse_args()
    summary_path = Path(args.run_root).expanduser().resolve(strict=False) / "profiling" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError("profiling summary must contain a JSON object")
    _print_report(summary)


if __name__ == "__main__":
    main()
