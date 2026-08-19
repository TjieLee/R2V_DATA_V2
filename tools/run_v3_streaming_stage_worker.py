"""Persistent isolated model worker for V3 streaming stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.config import V3Config, load_config
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.profiling import (
    QwenConcurrencyGate,
    V3Profiler,
    active_profiler,
    qwen_concurrency_gate,
)
from r2v_data_v2.v3.runtime import ClipScopedStorage
from r2v_data_v2.v3.storage import RunStorage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("segment", "remove", "reference_edit"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    sys.stdout.flush()


class _StageRuntime:
    def __init__(
        self,
        config: V3Config,
        storage: RunStorage,
        *,
        stage: str,
        overwrite: bool,
    ) -> None:
        self.config = config
        self.storage = storage
        self.stage = stage
        self.overwrite = overwrite
        self._shared_lock = threading.Lock()
        self._closers: list[Any] = []
        self._segment_backend: Any | None = None
        self._attribute_segmenter: Any | None = None
        self._first_reference_edit_request = True
        self._handler = self._initialize()

    def _initialize(self) -> Any:
        if self.stage == "segment":
            from r2v_data_v2.v3.segment import (
                build_sam3_segment_backend,
                segment_clips,
            )
            from r2v_data_v2.v3.subject_attributes import (
                Sam3AttributeFrameSegmenter,
            )

            backend = build_sam3_segment_backend(self.config)
            self._segment_backend = backend
            self._attribute_segmenter = Sam3AttributeFrameSegmenter(
                self.config.sam3,
                backend=backend,
            )
            self._closers.append(backend)
            return lambda scoped: segment_clips(
                self.config,
                scoped,
                overwrite=self.overwrite,
                backend=backend,
            )
        if self.stage == "remove":
            from r2v_data_v2.v3.qwen_image_edit_backend import (
                QwenImageEditRemovalBackend,
            )
            from r2v_data_v2.v3.removal_judge import QwenBackgroundRemovalJudge
            from r2v_data_v2.v3.remove import remove_backgrounds

            service = self.config.qwen.background_remove_judge
            if service is None:
                raise ValueError("qwen.background_remove_judge is required")
            backend = QwenImageEditRemovalBackend(self.config.remove)
            judge = QwenBackgroundRemovalJudge(service)
            self._closers.extend((judge, backend))
            return lambda scoped: remove_backgrounds(
                self.config,
                scoped,
                overwrite=self.overwrite,
                backend=backend,
                judge=judge,
            )

        from r2v_data_v2.v3.reference_edit import reference_edit_clips
        from r2v_data_v2.v3.reference_edit_boogu import (
            BooguSubprocessBackend,
            BooguWorkerConfig,
            QwenBooguReferenceEditJudge,
            Sam3BooguReferenceReviewer,
        )
        from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend
        from r2v_data_v2.v3.scale_collapse_fallback_guard import (
            QwenScaleCollapseFallbackJudge,
        )

        service = self.config.qwen.reference_edit_judge
        if service is None:
            raise ValueError("qwen.reference_edit_judge is required")
        physical_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if physical_device is None or not physical_device.isdigit():
            raise ValueError("reference_edit worker requires one visible physical GPU")
        backend = BooguSubprocessBackend(
            BooguWorkerConfig(
                python_executable=self.config.reference_edit.python_executable,
                code_root=self.config.reference_edit.code_root,
                model_path=self.config.reference_edit.model_path,
                model_revision=self.config.reference_edit.model_revision,
                device="cuda:0",
                timeout_seconds=self.config.reference_edit.timeout_seconds,
                cuda_visible_devices=physical_device,
                temporary_root=self.storage.reference_edit_temporary_dir(),
            )
        )
        backend.start(stderr_log_path=self.storage.reference_edit_worker_log_path())
        judge = QwenBooguReferenceEditJudge(service)
        segmenter = Sam3SegmentationBackend(self.config.sam3)
        sam_reviewer = Sam3BooguReferenceReviewer(
            segmenter,
            temporary_root=self.storage.reference_edit_temporary_dir(),
            max_area_growth_ratio=self.config.reference_edit.sam_max_area_growth_ratio,
            max_significant_components=(
                self.config.reference_edit.sam_max_significant_components
            ),
            min_candidate_scale_ratio=(
                self.config.reference_edit.min_candidate_scale_ratio
            ),
            max_candidate_center_shift=(
                self.config.reference_edit.max_candidate_center_shift
            ),
        )
        scale_judge = (
            QwenScaleCollapseFallbackJudge(service)
            if self.config.reference_edit.scale_collapse_fallback_guard_mode
            == "qwen_v1"
            else None
        )
        self._closers.extend((scale_judge, segmenter, judge, backend))
        return lambda scoped: reference_edit_clips(
            self.config,
            scoped,
            overwrite=self.overwrite,
            backend=backend,
            judge=judge,
            sam_reviewer=sam_reviewer,
            scale_collapse_judge=scale_judge,
            manage_backend_lifecycle=False,
        )

    def run_clip(self, clip_uid: str) -> dict[str, object]:
        scoped = ClipScopedStorage(
            self.storage,
            clip_uid,
            shared_write_lock=self._shared_lock,
        )
        before_anchor = (
            self._segment_backend.anchor_search_counters()
            if self._segment_backend is not None
            else None
        )
        before_recall = (
            self._segment_backend.recall_rescue_counters()
            if self._segment_backend is not None
            else None
        )
        counts = self._handler(scoped).to_dict()
        if before_anchor is not None:
            after_anchor = self._segment_backend.anchor_search_counters()
            for name, value in after_anchor.items():
                counts[name] = value - before_anchor.get(name, 0)
        if before_recall is not None:
            after_recall = self._segment_backend.recall_rescue_counters()
            for name, value in after_recall.items():
                counts[name] = int(counts.get(name, value)) - before_recall.get(
                    name,
                    0,
                )
        if self.stage == "reference_edit" and self._first_reference_edit_request:
            counts["worker_starts"] = 1
            self._first_reference_edit_request = False
        return counts

    def attribute_probe(
        self,
        *,
        clip_uid: str,
        frame_slot: int,
        source_frame_index: int,
        grounding_prompt: str,
    ) -> list[dict[str, object]]:
        if self.stage != "segment" or self._attribute_segmenter is None:
            raise RuntimeError("attribute probes require the segment worker")
        frames = self.storage.read_frames(clip_uid)
        frame = next((item for item in frames.frames if item.slot == frame_slot), None)
        if frame is None or frame.source_frame_index != source_frame_index:
            raise ValueError("attribute probe frame provenance does not match")
        frame_path = self.storage.clip_dir(clip_uid) / frame.image_path
        masks = self._attribute_segmenter.segment_frame(
            frame_path=frame_path,
            frame_slot=frame_slot,
            grounding_prompt=grounding_prompt,
        )
        return [encode_binary_mask(mask).model_dump(mode="json") for mask in masks]

    def close(self) -> None:
        first_error: BaseException | None = None
        for resource in self._closers:
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - finish remaining resources
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _local_device_config(config: V3Config, stage: str) -> V3Config:
    if stage == "segment":
        return replace(config, sam3=replace(config.sam3, device="cuda"))
    if stage == "remove":
        return replace(config, remove=replace(config.remove, device="cuda"))
    return replace(config, sam3=replace(config.sam3, device="cuda"))


def serve(args: argparse.Namespace) -> int:
    config = _local_device_config(load_config(args.config), args.stage)
    storage = RunStorage(config)
    run = storage.read_run()
    profiler = V3Profiler(storage.root, git_commit=run.git_commit) if args.profile else None
    gate = QwenConcurrencyGate(
        config.runtime.qwen_max_inflight,
        lock_directory=storage.root / "profiling" / "qwen_slots",
    )
    with ExitStack() as stack:
        if profiler is not None:
            stack.enter_context(active_profiler(profiler))
        stack.enter_context(qwen_concurrency_gate(gate))
        runtime = _StageRuntime(
            config,
            storage,
            stage=args.stage,
            overwrite=args.overwrite,
        )
        _write({"schema_version": 1, "status": "ok", "type": "ready"})
        try:
            for line in sys.stdin:
                request_id: str | None = None
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("worker request must be an object")
                    raw_request_id = payload.get("request_id")
                    if isinstance(raw_request_id, str):
                        request_id = raw_request_id
                    if payload.get("schema_version") != 1:
                        raise ValueError("unsupported worker schema version")
                    if payload.get("type") == "shutdown":
                        _write(
                            {
                                "schema_version": 1,
                                "type": "shutdown",
                                "request_id": request_id,
                                "status": "ok",
                            }
                        )
                        return 0
                    request_type = payload.get("type")
                    if request_type not in {"run_clip", "attribute_probe"}:
                        raise ValueError("unsupported worker request type")
                    clip_uid = payload.get("clip_uid")
                    if not isinstance(clip_uid, str) or not clip_uid:
                        raise ValueError("worker clip_uid must be non-empty")
                    if request_type == "run_clip":
                        counts = runtime.run_clip(clip_uid)
                        response = {
                            "schema_version": 1,
                            "type": "response",
                            "request_id": request_id,
                            "status": "ok",
                            "counts": counts,
                        }
                    else:
                        frame_slot = payload.get("frame_slot")
                        source_frame_index = payload.get("source_frame_index")
                        grounding_prompt = payload.get("grounding_prompt")
                        if (
                            isinstance(frame_slot, bool)
                            or not isinstance(frame_slot, int)
                            or frame_slot < 0
                        ):
                            raise ValueError(
                                "attribute probe frame_slot must be non-negative"
                            )
                        if (
                            isinstance(source_frame_index, bool)
                            or not isinstance(source_frame_index, int)
                            or source_frame_index < 0
                        ):
                            raise ValueError(
                                "attribute probe source_frame_index must be non-negative"
                            )
                        if (
                            not isinstance(grounding_prompt, str)
                            or not grounding_prompt.strip()
                        ):
                            raise ValueError(
                                "attribute probe grounding_prompt must be non-empty"
                            )
                        masks = runtime.attribute_probe(
                            clip_uid=clip_uid,
                            frame_slot=frame_slot,
                            source_frame_index=source_frame_index,
                            grounding_prompt=grounding_prompt,
                        )
                        response = {
                            "schema_version": 1,
                            "type": "response",
                            "request_id": request_id,
                            "status": "ok",
                            "masks": masks,
                        }
                except Exception as exc:  # noqa: BLE001 - process boundary response
                    response = {
                        "schema_version": 1,
                        "type": "response",
                        "request_id": request_id,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                _write(response)
        finally:
            runtime.close()
    return 0


def main() -> int:
    return serve(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
