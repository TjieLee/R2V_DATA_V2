"""Isolated per-GPU SAM3 subprocesses for Subject Attributes only.

Each child owns one predictor and one CUDA context. The parent retains semantic
planning, CPU/IO persistence, and the existing durable receipt boundary. Large
mask arrays are copied back through a local pipe before freeing the infer slot;
no child ever writes a production artifact or touches GroupLedger.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


class SAProcessTransportFatal(BaseException):
    """Infrastructure failure: never turn a dead worker into 'sam_failed'."""


class SAProcessInferenceError(RuntimeError):
    """Ordinary model failure: handled exactly like an existing SAM exception."""


def _worker_main(connection: Any, config: Any, gpu_id: int) -> None:
    """Spawn target. Do not import torch or CUDA backend in the parent module."""
    backend = None
    try:
        from r2v_data_v2.v3.pre_qwen_production import enable_sam3_session_reuse
        from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend
        from r2v_data_v2.v3.subject_attributes import Sam3AttributeFrameSegmenter

        # Respect the parent's CUDA-visible device numbering; physical ids
        # here are the existing WorkerPoolConfig gpu_ids, not slot numbers.
        sam_config = replace(config, device=f"cuda:{gpu_id}")
        backend = Sam3SegmentationBackend(sam_config)
        enable_sam3_session_reuse(backend, mode="clip_reset_v1")
        adapter = Sam3AttributeFrameSegmenter(sam_config, backend=backend)
        connection.send(("ready", None))
        while True:
            try:
                operation, arguments = connection.recv()
            except EOFError:
                break
            if operation == "close":
                break
            try:
                if operation == "frame":
                    masks = adapter.segment_frame(
                        frame_path=Path(arguments["frame_path"]),
                        frame_slot=int(arguments["frame_slot"]),
                        grounding_prompt=str(arguments["grounding_prompt"]),
                    )
                elif operation == "generated":
                    masks = adapter.segment_generated_frame(
                        frame_path=Path(arguments["frame_path"]),
                        grounding_prompt=str(arguments["grounding_prompt"]),
                    )
                else:
                    raise ValueError(f"unknown SAM operation: {operation}")
                connection.send(("ok", masks))
            except Exception as exc:  # noqa: BLE001 - preserves legacy per-call failure
                connection.send(
                    ("error", (type(exc).__name__, str(exc)))
                )
    except BaseException as exc:  # noqa: BLE001 - tell parent to abort epoch
        try:
            connection.send(("fatal", (type(exc).__name__, str(exc))))
        except (OSError, EOFError):
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass
        connection.close()


class SAProcessHandle:
    """One pipe is owned exclusively by one SAM infer thread."""

    def __init__(self, connection: Any, process: Any, *, timeout_seconds: float):
        self.connection = connection
        self.process = process
        self.timeout_seconds = timeout_seconds

    def _call(self, name: str, arguments: dict[str, Any]) -> tuple:
        try:
            self.connection.send((name, arguments))
            if not self.connection.poll(self.timeout_seconds):
                raise SAProcessTransportFatal(
                    f"SAM worker timed out after {self.timeout_seconds}s "
                    f"(pid={self.process.pid}, alive={self.process.is_alive()})"
                )
            kind, payload = self.connection.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            raise SAProcessTransportFatal(
                f"SAM subprocess exited or IPC failed (pid={self.process.pid})"
            ) from exc
        if kind == "ok":
            return tuple(payload)
        if kind == "error":
            original_type, message = payload
            raise SAProcessInferenceError(f"{original_type}: {message}")
        raise SAProcessTransportFatal(
            f"SAM worker fatal reply (pid={self.process.pid}): {payload}"
        )

    def segment_frame(
        self, *, frame_path: Path, frame_slot: int, grounding_prompt: str
    ) -> tuple:
        return self._call(
            "frame",
            {
                "frame_path": str(frame_path),
                "frame_slot": frame_slot,
                "grounding_prompt": grounding_prompt,
            },
        )

    def segment_generated_frame(
        self, *, frame_path: Path, grounding_prompt: str
    ) -> tuple:
        return self._call(
            "generated",
            {"frame_path": str(frame_path), "grounding_prompt": grounding_prompt},
        )


class SAProcessPool:
    """Four independent SAM model processes per GPU, stage-scoped and bounded."""

    affinity_by_clip = True

    def __init__(
        self,
        config: Any,
        gpu_ids: tuple[int, ...],
        *,
        workers_per_gpu: int = 4,
        timeout_seconds: float = 900.0,
        shutdown_seconds: float = 30.0,
    ) -> None:
        if not gpu_ids or workers_per_gpu not in (2, 4):
            raise ValueError("SA multi-process pool requires 2 or 4 workers per GPU")
        self.config = config
        self.gpu_ids = tuple(gpu_ids)
        self.workers_per_gpu = workers_per_gpu
        self.timeout_seconds = timeout_seconds
        self.shutdown_seconds = shutdown_seconds
        self.slot_count = len(gpu_ids) * workers_per_gpu
        self._context = mp.get_context("spawn")
        self._processes: list[Any] = []
        self._connections: list[Any] = []
        self._handles: list[SAProcessHandle] = []

    def start(self) -> None:
        if self._processes:
            raise RuntimeError("SAM SA multi-process pool already started")
        try:
            # Start all processes before waiting, rather than serializing 32
            # Python interpreters / torch imports. Model weights remain lazy.
            for gpu_id in self.gpu_ids:
                for _ in range(self.workers_per_gpu):
                    parent, child = self._context.Pipe(duplex=True)
                    process = self._context.Process(
                        target=_worker_main,
                        args=(child, self.config, gpu_id),
                        name=f"sa-sam-gpu{gpu_id}-worker",
                    )
                    process.start()
                    child.close()
                    self._processes.append(process)
                    self._connections.append(parent)
            for process, connection in zip(self._processes, self._connections):
                if not connection.poll(self.timeout_seconds):
                    raise SAProcessTransportFatal(
                        f"SAM worker startup timed out (pid={process.pid})"
                    )
                try:
                    kind, details = connection.recv()
                except (EOFError, OSError) as exc:
                    raise SAProcessTransportFatal(
                        f"SAM worker failed to start (pid={process.pid})"
                    ) from exc
                if kind != "ready":
                    raise SAProcessTransportFatal(
                        f"SAM worker startup failed (pid={process.pid}): {details}"
                    )
            self._handles = [
                SAProcessHandle(c, p, timeout_seconds=self.timeout_seconds)
                for c, p in zip(self._connections, self._processes)
            ]
        except BaseException:
            self.close()
            raise

    def handle_for_slot(self, slot: int) -> SAProcessHandle:
        if not 0 <= slot < len(self._handles):
            raise SAProcessTransportFatal(f"invalid SAM subprocess slot: {slot}")
        return self._handles[slot]

    def close(self) -> None:
        handles, self._handles = self._handles, []
        del handles
        for connection in self._connections:
            try:
                connection.send(("close", {}))
            except (OSError, EOFError, BrokenPipeError):
                pass
        for process in self._processes:
            process.join(timeout=self.shutdown_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()
