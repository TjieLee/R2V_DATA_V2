#!/usr/bin/env python3
"""Run one isolated, contiguous JEA V3 functional canary."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter
from tools.compact_v3_production_exports import compact_production_exports
from tools.prepare_v3_production_shards import (
    _atomic_write,
    _base_config_identity,
    _selection_bytes,
    _source_descriptor,
    _write_immutable,
)

DEFAULT_SOURCE_JSONL = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/shots_f03_motion.jsonl"
)
DEFAULT_CLIPS_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/clips_clean_cropped"
)
DEFAULT_SOURCE_VIDEOS_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808"
)
DEFAULT_BASE_CONFIG = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_configs/"
    "stream-attr10-dp4-sam7-c1c056-20260819-223752.yaml"
)
DEFAULT_COUNT = 20
CANARY_STAGES = (
    "manifest",
    "annotate",
    "frames",
    "segment",
    "rank",
    "background",
    "remove",
    "pair",
    "reference_edit",
    "reference_integrity",
    "instruct",
    "subject_attributes",
    "export",
)
_SAFE_TAG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class CanarySelection:
    source_video: Path
    records: tuple[dict[str, object], ...]

    @property
    def start_index(self) -> int:
        return int(self.records[0]["source_index"])

    @property
    def end_index(self) -> int:
        return int(self.records[-1]["source_index"])

    @property
    def shard_id(self) -> str:
        return f"shard-{self.start_index:09d}-{self.end_index:09d}"


@dataclass(frozen=True)
class CanaryPaths:
    tag: str
    config_root: Path
    canary_runs_root: Path
    run_root: Path
    export_root: Path
    shard_export_root: Path
    source_yaml: Path
    selection_manifest: Path
    shard_config: Path
    log_path: Path


@dataclass(frozen=True)
class PipelineExecution:
    returncode: int
    output: str
    result: dict[str, object] | None


class CanaryPipelineError(RuntimeError):
    def __init__(self, returncode: int, log_path: Path) -> None:
        super().__init__(f"canary pipeline exited with status {returncode}: {log_path}")
        self.returncode = returncode
        self.log_path = log_path


PipelineRunner = Callable[..., PipelineExecution]
Compactor = Callable[..., dict[str, object]]


def _resolve_source_jsonl(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if (
        not resolved.is_file()
        or resolved.suffix.lower() != ".jsonl"
        or dataset_root not in resolved.parents
    ):
        raise ValueError("source_jsonl must be a JSONL below public dataset")
    return resolved


def _resolve_requested_source_video(
    adapter: JeaVideoMotionAdapter,
    source_video: str | Path | None,
) -> Path | None:
    if source_video is None:
        return None
    resolved, _ = adapter.resolve_source_video_path(
        {"source_video_path": str(source_video)},
        require_file=True,
    )
    return resolved


def select_source_records(
    *,
    source_jsonl: str | Path,
    clips_root: str | Path,
    source_videos_root: str | Path,
    count: int = DEFAULT_COUNT,
    exclude_source_names: Sequence[str] = (),
    source_video: str | Path | None = None,
) -> CanarySelection:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if source_video is not None and exclude_source_names:
        raise ValueError("source_video and exclude_source_names are mutually exclusive")
    if any(not isinstance(value, str) or not value for value in exclude_source_names):
        raise ValueError("exclude_source_names must contain non-empty text")

    source_path = _resolve_source_jsonl(source_jsonl)
    adapter = JeaVideoMotionAdapter.create(
        clips_root=clips_root,
        source_videos_root=source_videos_root,
    )
    requested_source = _resolve_requested_source_video(adapter, source_video)
    active_source: Path | None = None
    active_records: list[dict[str, object]] = []
    source_index = 0

    def reset() -> None:
        nonlocal active_source, active_records
        active_source = None
        active_records = []

    with source_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            if not line.strip():
                continue
            current_index = source_index
            source_index += 1
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("source record must be a JSON object")
                selected = adapter.parse(raw, source_index=current_index)
                clip_path, _ = adapter.resolve_clip_path(raw)
                if clip_path.suffix.lower() != ".mp4":
                    raise ValueError("record.video_path must identify an MP4")
                resolved_source, _ = adapter.resolve_source_video_path(
                    raw,
                    require_file=True,
                )
            except Exception:  # noqa: BLE001 - invalid records break a contiguous run
                reset()
                continue

            if requested_source is not None and resolved_source != requested_source:
                reset()
                continue
            if any(value in str(resolved_source) for value in exclude_source_names):
                reset()
                continue
            is_continuation = (
                active_source == resolved_source
                and bool(active_records)
                and int(active_records[-1]["source_index"]) + 1 == current_index
            )
            if not is_continuation:
                active_source = resolved_source
                active_records = []
            active_records.append(selected)
            if len(active_records) == count:
                return CanarySelection(
                    source_video=resolved_source,
                    records=tuple(active_records),
                )

    qualifier = (
        f" for source video {requested_source}"
        if requested_source is not None
        else " after applying source-name exclusions"
    )
    raise ValueError(
        f"no source video has {count} consecutive valid records{qualifier}"
    )


def _canary_tag(
    selection: CanarySelection,
    *,
    count: int,
    now: datetime,
    sam3_compile: bool = False,
) -> str:
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = (
        f"canary-e2e{count}-jea-{timestamp}-"
        f"s{selection.start_index:09d}-{selection.end_index:09d}"
    )
    return f"{tag}-sam3compile" if sam3_compile else tag


def build_canary_paths(
    selection: CanarySelection,
    *,
    count: int,
    now: datetime,
    sam3_compile: bool = False,
) -> CanaryPaths:
    tag = _canary_tag(
        selection,
        count=count,
        now=now,
        sam3_compile=sam3_compile,
    )
    if _SAFE_TAG.fullmatch(tag) is None:
        raise ValueError("generated canary tag is not ASCII-safe")
    writable = config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    config_root = (
        writable
        / "r2v_v3_configs"
        / "production"
        / "jea_motion_v1"
        / tag
    )
    canary_runs_root = (
        writable / "r2v_v3_runs" / "production" / "jea_motion_v1" / tag
    )
    run_root = canary_runs_root / selection.shard_id
    export_root = (
        writable / "r2v_v3_exports" / "production" / "jea_motion_v1" / tag
    )
    return CanaryPaths(
        tag=tag,
        config_root=config_root,
        canary_runs_root=canary_runs_root,
        run_root=run_root,
        export_root=export_root,
        shard_export_root=export_root / "shards" / selection.shard_id,
        source_yaml=config_root / "source.yaml",
        selection_manifest=config_root / "selection.jsonl",
        shard_config=config_root / f"{selection.shard_id}.yaml",
        log_path=run_root / "canary.log",
    )


def _canary_config_bytes(
    base: dict[str, object],
    *,
    source_jsonl: Path,
    paths: CanaryPaths,
    sam3_compile: bool = False,
    dual_main_sam3: bool = False,
    defer_subject_attributes: bool = False,
    attribute_completion: bool = False,
    qwen_max_inflight: int | None = None,
    qwen_stage_workers: int | None = None,
) -> bytes:
    value = dict(base)
    source = dict(value.get("source") or {})
    source.update(
        {
            "selection_mode": "fixed_selection_v1",
            "selection_manifest": str(paths.selection_manifest),
            "start_index": 0,
            "limit": None,
            "allow_full_run": False,
        }
    )
    runtime = dict(value.get("runtime") or {})
    runtime["sam3_compile_enabled"] = sam3_compile
    runtime["subject_attributes_deferred"] = defer_subject_attributes
    stage_workers = dict(runtime.get("stage_workers") or {})
    gpu_workers = dict(runtime.get("gpu_workers") or {})
    if dual_main_sam3:
        gpu_workers["segment_pool"] = ["5", "7"]
        stage_workers["segment"] = 2
        runtime["sam3_compile_enabled"] = False
    if qwen_max_inflight is not None:
        runtime["qwen_max_inflight"] = qwen_max_inflight
    if qwen_stage_workers is not None:
        qwen_stages = (
            "annotate",
            "pair",
            "reference_integrity",
            "instruct",
        )
        for stage in qwen_stages:
            stage_workers[stage] = qwen_stage_workers
        if not defer_subject_attributes:
            stage_workers["subject_attributes"] = qwen_stage_workers
    runtime["stage_workers"] = stage_workers
    runtime["gpu_workers"] = gpu_workers
    if "subject_attribute_gme" in value:
        gme = dict(value.get("subject_attribute_gme") or {})
        gme["enabled"] = False
        value["subject_attribute_gme"] = gme
    if attribute_completion:
        subject_attributes = dict(value.get("subject_attributes") or {})
        completion = dict(subject_attributes.get("completion") or {})
        completion["enabled"] = True
        subject_attributes["completion"] = completion
        value["subject_attributes"] = subject_attributes
        gpu_workers.setdefault("subject_attributes_completion", "6")
    value.update(
        {
            "dataset_json": str(source_jsonl),
            "run_root": str(paths.run_root),
            "export_root": str(paths.shard_export_root),
            "source": source,
            "runtime": runtime,
        }
    )
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def prepare_canary_artifacts(
    *,
    selection: CanarySelection,
    paths: CanaryPaths,
    source_jsonl: str | Path,
    clips_root: str | Path,
    source_videos_root: str | Path,
    base_config: str | Path,
    sam3_compile: bool = False,
    dual_main_sam3: bool = False,
    defer_subject_attributes: bool = False,
    attribute_completion: bool = False,
    qwen_max_inflight: int | None = None,
    qwen_stage_workers: int | None = None,
) -> None:
    source_path = _resolve_source_jsonl(source_jsonl)
    base_path = Path(base_config).expanduser().resolve(strict=True)
    base_bytes, base_sha256, base_fingerprint = _base_config_identity(base_path)
    base = yaml.safe_load(base_bytes.decode("utf-8"))
    if not isinstance(base, dict):
        raise TypeError("base V3 config must be a YAML mapping")
    adapter = JeaVideoMotionAdapter.create(
        clips_root=clips_root,
        source_videos_root=source_videos_root,
    )

    paths.config_root.mkdir(parents=True, exist_ok=False)
    _write_immutable(
        paths.source_yaml,
        _source_descriptor(
            source_jsonl=source_path,
            base_config=base_path,
            adapter=adapter,
            shard_size=len(selection.records),
            path_probe_records=len(selection.records),
            base_config_sha256=base_sha256,
            base_config_fingerprint=base_fingerprint,
        ),
    )
    _write_immutable(
        paths.selection_manifest,
        _selection_bytes(list(selection.records)),
    )
    _write_immutable(
        paths.shard_config,
        _canary_config_bytes(
            base,
            source_jsonl=source_path,
            paths=paths,
            sam3_compile=sam3_compile,
            dual_main_sam3=dual_main_sam3,
            defer_subject_attributes=defer_subject_attributes,
            attribute_completion=attribute_completion,
            qwen_max_inflight=qwen_max_inflight,
            qwen_stage_workers=qwen_stage_workers,
        ),
    )
    load_config(paths.shard_config)


def _parse_pipeline_result(output: str) -> dict[str, object] | None:
    starts = [match.start() for match in re.finditer(r"(?m)^\{", output)]
    for start in reversed(starts):
        try:
            value = json.loads(output[start:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_pipeline_process(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    environment: dict[str, str],
) -> PipelineExecution:
    output_parts: list[str] = []
    pending_log_parts: list[str] = []
    log_handle: TextIO | None = None
    process = subprocess.Popen(  # noqa: S603 - fixed local executable and arguments
        list(command),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_parts.append(line)
            if log_handle is None and log_path.parent.is_dir():
                log_handle = log_path.open("x", encoding="utf-8")
                log_handle.writelines(pending_log_parts)
                pending_log_parts = []
            if log_handle is None:
                pending_log_parts.append(line)
            else:
                log_handle.write(line)
                log_handle.flush()
        returncode = process.wait()
        if log_handle is None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("x", encoding="utf-8")
            log_handle.writelines(pending_log_parts)
        log_handle.flush()
        os.fsync(log_handle.fileno())
    finally:
        if log_handle is not None:
            log_handle.close()
    output = "".join(output_parts)
    return PipelineExecution(
        returncode=returncode,
        output=output,
        result=_parse_pipeline_result(output),
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed local git command
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _elapsed_hms(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _failed_tasks(result: dict[str, object] | None) -> list[object]:
    if not isinstance(result, dict):
        return []
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        return []
    failures = runtime.get("failed_tasks")
    return list(failures) if isinstance(failures, list) else []


def _print_selection(
    *,
    selection: CanarySelection,
    paths: CanaryPaths,
    git_commit: str,
) -> None:
    first = Path(str(selection.records[0]["video_path"]))
    last = Path(str(selection.records[-1]["video_path"]))
    print("Selected source video:")
    print(f"  {selection.source_video}")
    print("\nSelected processed shots:")
    print(f"  source_index: {selection.start_index}..{selection.end_index}")
    print(f"  count: {len(selection.records)}")
    print(f"  first: {first}")
    print(f"  last:  {last}")
    print(f"\ngit_commit: {git_commit}")
    print(f"config: {paths.shard_config}")
    print(f"run_root: {paths.run_root}")
    print(f"export_root: {paths.export_root}")
    print(f"log: {paths.log_path}")
    sys.stdout.flush()


def _print_summary(summary: dict[str, object]) -> None:
    print("\nCanary summary:")
    for field_name in (
        "status",
        "selected_source_video",
        "selected_source_indices",
        "input_clips",
        "subject_attributes_deferred",
        "elapsed_seconds",
        "elapsed_hms",
        "sam3_compile_requested",
        "sam3_compile_effective",
        "sam3_compile_fallbacks",
        "sam3_compile_failure_reason",
        "sam3_predictor_startup_seconds",
        "sam3_compile_warmup_seconds",
        "sam3_segment_model_call_time_seconds",
        "sam3_segment_clips",
        "sam3_segment_entities",
        "first_segment_clip_seconds",
        "steady_state_segment_mean_seconds",
        "segment_worker_pool_size",
        "segment_worker_requests_by_gpu",
        "segment_worker_service_seconds_by_gpu",
        "sam_pool_main_requests_by_gpu",
        "sam_pool_attribute_probe_requests_by_gpu",
        "sam_pool_main_service_seconds_by_gpu",
        "sam_pool_attribute_service_seconds_by_gpu",
        "sam_pool_main_wait_seconds_total",
        "sam_pool_attribute_wait_seconds_total",
        "sam_pool_max_concurrent_requests",
        "qwen_calls",
        "qwen_gate_wait_seconds_total",
        "qwen_gate_wait_seconds_mean",
        "qwen_gate_wait_seconds_max",
        "qwen_max_local_inflight_observed",
        "qwen_slot_usage",
        "visual_samples",
        "visual_references",
        "visual_background_references",
        "canonical_samples",
        "canonical_visual_references",
        "canonical_background_references",
        "canonical_attribute_references",
        "samples_with_background",
        "enriched_samples",
        "remove_candidates_generated",
        "remove_candidates_accepted",
        "remove_candidates_rejected",
        "ready_removed",
        "gme_calls",
        "gme_candidates_screened",
        "gme_candidates_passed",
        "gme_candidates_rejected",
        "gme_retried_next_frame",
        "gme_failures",
        "gme_model_call_time_seconds",
        "completion_attempts",
        "completion_accepted",
        "completion_rejected",
        "completion_failures",
        "completion_qwen_review_rejects",
        "raw_attribute_review_accepted",
        "raw_attribute_review_repair_recommended",
        "raw_attribute_review_hard_rejected",
        "completion_raw_usable_attempts",
        "completion_raw_unusable_attempts",
        "completion_selected_completed",
        "completion_fallback_to_raw",
        "completion_backend_failures",
        "completion_postcheck_rejects",
        "completion_sam_zero_mask_rejects",
        "completion_sam_single_mask",
        "completion_sam_multi_mask",
        "completion_sam_masks_returned_total",
        "completion_identity_review_rejects",
        "completion_final_review_rejects",
        "repaired_attribute_final_review_accepted",
        "repaired_attribute_final_review_rejected",
        "completion_attempts_by_type",
        "completion_accepted_by_type",
        "completion_model_call_time_seconds",
        "failed_tasks",
        "config",
        "run_root",
        "export_root",
        "samples_jsonl",
        "references_root",
        "log",
    ):
        value = summary[field_name]
        rendered = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
        )
        print(f"{field_name}: {rendered}")


def _background_counts(
    samples_path: Path,
    *,
    kind_field: str,
) -> tuple[int, int]:
    if not samples_path.is_file():
        return 0, 0
    reference_count = 0
    sample_count = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise TypeError("samples.jsonl line must contain an object")
            references = sample.get("references", [])
            if not isinstance(references, list):
                raise TypeError("samples.jsonl references must contain an array")
            backgrounds = sum(
                isinstance(reference, dict)
                and reference.get(kind_field) == "background"
                for reference in references
            )
            reference_count += backgrounds
            sample_count += backgrounds > 0
    return reference_count, sample_count


def _stage_count(
    result: dict[str, object] | None,
    *,
    stage: str,
    field: str,
) -> int:
    if not isinstance(result, dict):
        return 0
    stage_result = result.get(stage)
    if not isinstance(stage_result, dict):
        return 0
    value = stage_result.get(field, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return 0
    return value


def _stage_metric(
    result: dict[str, object] | None,
    *,
    stage: str,
    field: str,
    default: object,
) -> object:
    if not isinstance(result, dict):
        return default
    stage_result = result.get(stage)
    if not isinstance(stage_result, dict):
        return default
    return stage_result.get(field, default)


def _subject_attribute_metric(
    result: dict[str, object] | None,
    field: str,
) -> int | float:
    if not isinstance(result, dict):
        return 0
    summary = result.get("subject_attributes_summary")
    if not isinstance(summary, dict):
        return 0
    value = summary.get(field, 0)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        return 0
    return value


def _subject_attribute_metric_object(
    result: dict[str, object] | None,
    field: str,
) -> dict[str, int | float]:
    if not isinstance(result, dict):
        return {}
    summary = result.get("subject_attributes_summary")
    if not isinstance(summary, dict):
        return {}
    value = summary.get(field, {})
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, int | float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            or item < 0
        ):
            continue
        cleaned[key] = item
    return cleaned


def _profiling_summary(run_root: Path) -> dict[str, object]:
    path = run_root / "profiling" / "summary.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def run_canary(
    *,
    source_jsonl: str | Path = DEFAULT_SOURCE_JSONL,
    clips_root: str | Path = DEFAULT_CLIPS_ROOT,
    source_videos_root: str | Path = DEFAULT_SOURCE_VIDEOS_ROOT,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    count: int = DEFAULT_COUNT,
    exclude_source_names: Sequence[str] = (),
    source_video: str | Path | None = None,
    now: datetime | None = None,
    pipeline_runner: PipelineRunner = run_pipeline_process,
    compactor: Compactor = compact_production_exports,
    clock: Callable[[], float] = time.monotonic,
    sam3_compile: bool = False,
    dual_main_sam3: bool = False,
    defer_subject_attributes: bool = False,
    attribute_completion: bool = False,
    qwen_max_inflight: int | None = None,
    qwen_stage_workers: int | None = None,
) -> dict[str, object]:
    for name, value in (
        ("qwen_max_inflight", qwen_max_inflight),
        ("qwen_stage_workers", qwen_stage_workers),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if dual_main_sam3 and sam3_compile:
        raise ValueError("--dual-main-sam3 cannot be combined with --sam3-compile")
    selection = select_source_records(
        source_jsonl=source_jsonl,
        clips_root=clips_root,
        source_videos_root=source_videos_root,
        count=count,
        exclude_source_names=exclude_source_names,
        source_video=source_video,
    )
    paths = build_canary_paths(
        selection,
        count=count,
        now=now or datetime.now(timezone.utc),
        sam3_compile=sam3_compile,
    )
    prepare_canary_artifacts(
        selection=selection,
        paths=paths,
        source_jsonl=source_jsonl,
        clips_root=clips_root,
        source_videos_root=source_videos_root,
        base_config=base_config,
        sam3_compile=sam3_compile,
        dual_main_sam3=dual_main_sam3,
        defer_subject_attributes=defer_subject_attributes,
        attribute_completion=attribute_completion,
        qwen_max_inflight=qwen_max_inflight,
        qwen_stage_workers=qwen_stage_workers,
    )
    git_commit = _git_commit()
    _print_selection(selection=selection, paths=paths, git_commit=git_commit)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "run_pipeline_v3.py"),
        "--config",
        str(paths.shard_config),
        "--stages",
        ",".join(CANARY_STAGES),
        "--profile",
    ]
    started = clock()
    execution = pipeline_runner(
        command,
        log_path=paths.log_path,
        cwd=REPOSITORY_ROOT,
        environment=environment,
    )
    if execution.returncode != 0:
        print(f"\nstatus: FAIL\nlog: {paths.log_path}", file=sys.stderr)
        raise CanaryPipelineError(execution.returncode, paths.log_path)

    shard_dataset_path = paths.shard_export_root / "dataset.json"
    shard_dataset = json.loads(shard_dataset_path.read_text(encoding="utf-8"))
    if not isinstance(shard_dataset, dict):
        raise TypeError("shard dataset.json must contain an object")
    catalog = compactor(
        shards_root=paths.export_root / "shards",
        output_root=paths.export_root,
        source_jsonl=source_jsonl,
        source_yaml=paths.source_yaml,
        runs_root=paths.canary_runs_root,
    )
    canonical_attribute_references = int(catalog["total_attribute_references"])
    if defer_subject_attributes and canonical_attribute_references != 0:
        raise RuntimeError(
            "deferred subject-attribute canary unexpectedly produced attributes"
        )
    profiling = _profiling_summary(paths.run_root)
    visual_background_references, _ = _background_counts(
        paths.shard_export_root / "samples.jsonl",
        kind_field="type",
    )
    canonical_background_references, samples_with_background = (
        _background_counts(
            paths.export_root / "samples.jsonl",
            kind_field="kind",
        )
    )
    elapsed_seconds = clock() - started
    summary: dict[str, object] = {
        "status": "PASS",
        "tag": paths.tag,
        "git_commit": git_commit,
        "selected_source_video": str(selection.source_video),
        "selected_source_indices": f"{selection.start_index}-{selection.end_index}",
        "input_clips": len(selection.records),
        "subject_attributes_deferred": defer_subject_attributes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_hms": _elapsed_hms(elapsed_seconds),
        "sam3_compile_requested": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_compile_requested",
            default=sam3_compile,
        ),
        "sam3_compile_effective": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_compile_effective",
            default=False,
        ),
        "sam3_compile_fallbacks": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_compile_fallbacks",
            default=0,
        ),
        "sam3_compile_failure_reason": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_compile_failure_reason",
            default=None,
        ),
        "sam3_predictor_startup_seconds": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_predictor_startup_seconds",
            default=0.0,
        ),
        "sam3_compile_warmup_seconds": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_compile_warmup_seconds",
            default=0.0,
        ),
        "sam3_segment_model_call_time_seconds": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_segment_model_call_time_seconds",
            default=0.0,
        ),
        "sam3_segment_clips": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_segment_clips",
            default=0,
        ),
        "sam3_segment_entities": _stage_metric(
            execution.result,
            stage="segment",
            field="sam3_segment_entities",
            default=0,
        ),
        "first_segment_clip_seconds": _stage_metric(
            execution.result,
            stage="segment",
            field="first_segment_clip_seconds",
            default=0.0,
        ),
        "steady_state_segment_mean_seconds": _stage_metric(
            execution.result,
            stage="segment",
            field="steady_state_segment_mean_seconds",
            default=0.0,
        ),
        "segment_worker_pool_size": _stage_metric(
            execution.result,
            stage="segment",
            field="segment_worker_pool_size",
            default=1,
        ),
        "segment_worker_requests_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="segment_worker_requests_by_gpu",
            default={},
        ),
        "segment_worker_service_seconds_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="segment_worker_service_seconds_by_gpu",
            default={},
        ),
        "sam_pool_main_requests_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_main_requests_by_gpu",
            default={},
        ),
        "sam_pool_attribute_probe_requests_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_attribute_probe_requests_by_gpu",
            default={},
        ),
        "sam_pool_main_service_seconds_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_main_service_seconds_by_gpu",
            default={},
        ),
        "sam_pool_attribute_service_seconds_by_gpu": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_attribute_service_seconds_by_gpu",
            default={},
        ),
        "sam_pool_main_wait_seconds_total": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_main_wait_seconds_total",
            default=0.0,
        ),
        "sam_pool_attribute_wait_seconds_total": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_attribute_wait_seconds_total",
            default=0.0,
        ),
        "sam_pool_max_concurrent_requests": _stage_metric(
            execution.result,
            stage="segment",
            field="sam_pool_max_concurrent_requests",
            default=0,
        ),
        "qwen_calls": profiling.get("qwen_calls", 0),
        "qwen_gate_wait_seconds_total": profiling.get(
            "qwen_gate_wait_seconds_total", 0.0
        ),
        "qwen_gate_wait_seconds_mean": profiling.get(
            "qwen_gate_wait_seconds_mean", 0.0
        ),
        "qwen_gate_wait_seconds_max": profiling.get(
            "qwen_gate_wait_seconds_max", 0.0
        ),
        "qwen_max_local_inflight_observed": profiling.get(
            "qwen_max_local_inflight_observed", 0
        ),
        "qwen_slot_usage": profiling.get("qwen_slot_usage", {}),
        "visual_samples": int(shard_dataset["sample_count"]),
        "visual_references": int(shard_dataset["reference_count"]),
        "visual_background_references": visual_background_references,
        "canonical_samples": int(catalog["total_samples"]),
        "canonical_visual_references": int(catalog["total_visual_references"]),
        "canonical_background_references": canonical_background_references,
        "canonical_attribute_references": canonical_attribute_references,
        "samples_with_background": samples_with_background,
        "enriched_samples": int(catalog["total_enriched_samples"]),
        "remove_candidates_generated": _stage_count(
            execution.result,
            stage="remove",
            field="candidates_generated",
        ),
        "remove_candidates_accepted": _stage_count(
            execution.result,
            stage="remove",
            field="ready_removed",
        ),
        "remove_candidates_rejected": _stage_count(
            execution.result,
            stage="remove",
            field="candidates_rejected",
        ),
        "ready_removed": _stage_count(
            execution.result,
            stage="remove",
            field="ready_removed",
        ),
        "gme_calls": _subject_attribute_metric(execution.result, "gme_calls"),
        "gme_candidates_screened": _subject_attribute_metric(
            execution.result,
            "gme_candidates_screened",
        ),
        "gme_candidates_passed": _subject_attribute_metric(
            execution.result,
            "gme_candidates_passed",
        ),
        "gme_candidates_rejected": _subject_attribute_metric(
            execution.result,
            "gme_candidates_rejected",
        ),
        "gme_retried_next_frame": _subject_attribute_metric(
            execution.result,
            "gme_retried_next_frame",
        ),
        "gme_failures": _subject_attribute_metric(
            execution.result,
            "gme_failures",
        ),
        "gme_model_call_time_seconds": _subject_attribute_metric(
            execution.result,
            "gme_model_call_time_seconds",
        ),
        "completion_attempts": _subject_attribute_metric(
            execution.result,
            "completion_attempts",
        ),
        "completion_accepted": _subject_attribute_metric(
            execution.result,
            "completion_accepted",
        ),
        "completion_rejected": _subject_attribute_metric(
            execution.result,
            "completion_rejected",
        ),
        "completion_failures": _subject_attribute_metric(
            execution.result,
            "completion_failures",
        ),
        "completion_qwen_review_rejects": _subject_attribute_metric(
            execution.result,
            "completion_qwen_review_rejects",
        ),
        "raw_attribute_review_accepted": _subject_attribute_metric(
            execution.result,
            "raw_attribute_review_accepted",
        ),
        "raw_attribute_review_repair_recommended": _subject_attribute_metric(
            execution.result,
            "raw_attribute_review_repair_recommended",
        ),
        "raw_attribute_review_hard_rejected": _subject_attribute_metric(
            execution.result,
            "raw_attribute_review_hard_rejected",
        ),
        "completion_raw_usable_attempts": _subject_attribute_metric(
            execution.result,
            "completion_raw_usable_attempts",
        ),
        "completion_raw_unusable_attempts": _subject_attribute_metric(
            execution.result,
            "completion_raw_unusable_attempts",
        ),
        "completion_selected_completed": _subject_attribute_metric(
            execution.result,
            "completion_selected_completed",
        ),
        "completion_fallback_to_raw": _subject_attribute_metric(
            execution.result,
            "completion_fallback_to_raw",
        ),
        "completion_backend_failures": _subject_attribute_metric(
            execution.result,
            "completion_backend_failures",
        ),
        "completion_postcheck_rejects": _subject_attribute_metric(
            execution.result,
            "completion_postcheck_rejects",
        ),
        "completion_sam_zero_mask_rejects": _subject_attribute_metric(
            execution.result,
            "completion_sam_zero_mask_rejects",
        ),
        "completion_sam_single_mask": _subject_attribute_metric(
            execution.result,
            "completion_sam_single_mask",
        ),
        "completion_sam_multi_mask": _subject_attribute_metric(
            execution.result,
            "completion_sam_multi_mask",
        ),
        "completion_sam_masks_returned_total": _subject_attribute_metric(
            execution.result,
            "completion_sam_masks_returned_total",
        ),
        "completion_identity_review_rejects": _subject_attribute_metric(
            execution.result,
            "completion_identity_review_rejects",
        ),
        "completion_final_review_rejects": _subject_attribute_metric(
            execution.result,
            "completion_final_review_rejects",
        ),
        "repaired_attribute_final_review_accepted": _subject_attribute_metric(
            execution.result,
            "repaired_attribute_final_review_accepted",
        ),
        "repaired_attribute_final_review_rejected": _subject_attribute_metric(
            execution.result,
            "repaired_attribute_final_review_rejected",
        ),
        "completion_attempts_by_type": _subject_attribute_metric_object(
            execution.result,
            "completion_attempts_by_type",
        ),
        "completion_accepted_by_type": _subject_attribute_metric_object(
            execution.result,
            "completion_accepted_by_type",
        ),
        "completion_model_call_time_seconds": _subject_attribute_metric(
            execution.result,
            "completion_model_call_time_seconds",
        ),
        "failed_tasks": _failed_tasks(execution.result),
        "config": str(paths.shard_config),
        "run_root": str(paths.run_root),
        "export_root": str(paths.export_root),
        "samples_jsonl": str(paths.export_root / "samples.jsonl"),
        "references_root": str(paths.export_root / "references"),
        "log": str(paths.log_path),
    }
    _atomic_write(
        paths.export_root / "canary_summary.json",
        (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    _print_summary(summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--exclude-source-name", action="append", default=[])
    selection.add_argument("--source-video", type=Path)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT)
    parser.add_argument(
        "--source-videos-root",
        type=Path,
        default=DEFAULT_SOURCE_VIDEOS_ROOT,
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--sam3-compile", action="store_true")
    parser.add_argument("--dual-main-sam3", action="store_true")
    parser.add_argument("--defer-subject-attributes", action="store_true")
    parser.add_argument("--attribute-completion", action="store_true")
    parser.add_argument("--qwen-max-inflight", type=int)
    parser.add_argument("--qwen-stage-workers", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        run_canary(
            source_jsonl=args.source_jsonl,
            clips_root=args.clips_root,
            source_videos_root=args.source_videos_root,
            base_config=args.base_config,
            count=args.count,
            exclude_source_names=args.exclude_source_name,
            source_video=args.source_video,
            sam3_compile=args.sam3_compile,
            dual_main_sam3=args.dual_main_sam3,
            defer_subject_attributes=args.defer_subject_attributes,
            attribute_completion=args.attribute_completion,
            qwen_max_inflight=args.qwen_max_inflight,
            qwen_stage_workers=args.qwen_stage_workers,
        )
    except CanaryPipelineError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 1) from exc


if __name__ == "__main__":
    main()
