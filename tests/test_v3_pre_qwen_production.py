from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import yaml
from PIL import Image

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.pre_qwen_production as production
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.frames import DecodedFrameInfo, DecodedVideoFrame
from r2v_data_v2.v3.pre_qwen_production import (
    DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS,
    AnnotationShard,
    ConfigIdentity,
    ExecutionChunk,
    ShardBusyError,
    build_execution_chunks,
    canary_summary,
    check_stage2_candidate_judge_health,
    compact_execution_chunks,
    ensure_execution_identity,
    execution_chunk_path,
    inspect_stage2_shard,
    inventory,
    load_annotation_shard,
    prepare_canary,
    process_execution_chunk,
    process_ready_clip,
    process_shard,
    run_canary_worker,
    select_canary_rows,
    shard_lock,
    validate_stage2_preflight,
)
from r2v_data_v2.v3.sam3_anchor_selector import QwenSam3AnchorSelector
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
    Sam3SegmentationBackend,
)
from tools import run_v3_entity_mask_auto as entity_mask_tool
from tools import run_v3_pre_qwen_auto as auto_tool
from tools import run_v3_pre_qwen_batch as batch_tool
from tools import run_v3_pre_qwen_canary as canary_tool


@dataclass
class FakeDecoder:
    calls: list[str] = field(default_factory=list)

    def inspect(self, video_path: Path) -> list[DecodedFrameInfo]:
        self.calls.append(f"inspect:{video_path.name}")
        return [
            DecodedFrameInfo(
                source_frame_index=index,
                timestamp_seconds=index * 0.1,
                width=16,
                height=12,
            )
            for index in range(20)
        ]

    def decode_indices(
        self,
        video_path: Path,
        source_frame_indices: list[int],
    ) -> list[DecodedVideoFrame]:
        self.calls.append(f"decode:{video_path.name}")
        return [
            DecodedVideoFrame(
                source_frame_index=index,
                image=Image.new("RGB", (16, 12), (index, 20, 30)),
            )
            for index in source_frame_indices
        ]


@dataclass
class FakeBackend:
    visible_slots: int = 10
    failure: Exception | None = None
    fail_on_call: int | None = None
    calls: list[str] = field(default_factory=list)
    close_calls: int = 0

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
        **kwargs: object,
    ) -> EntityTrackResult:
        del reference_type, grounding_prompt, kwargs
        self.calls.append(entity_id)
        if self.failure is not None and (
            self.fail_on_call is None or len(self.calls) == self.fail_on_call
        ):
            raise self.failure
        observations = []
        for slot in range(self.visible_slots):
            mask = np.zeros((12, 16), dtype=bool)
            mask[4:8, 6:10] = True
            observations.append(
                BackendMaskObservation(
                    slot=slot,
                    mask=mask,
                    confidence=0.9,
                    object_id=f"track-{entity_id}",
                )
            )
        assert len(frame_paths) == 10
        return EntityTrackResult(status="ready", observations=tuple(observations))

    def close(self) -> None:
        self.close_calls += 1


class LifecycleProcess:
    def __init__(
        self,
        returncode: int | None,
        *,
        signal_on_first_wait: int | None = None,
        interrupt_on_first_wait: BaseException | None = None,
    ) -> None:
        self.returncode = returncode
        self.signal_on_first_wait = signal_on_first_wait
        self.interrupt_on_first_wait = interrupt_on_first_wait
        self.wait_calls: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if len(self.wait_calls) == 1 and timeout is None:
            if self.signal_on_first_wait is not None:
                handler = signal.getsignal(self.signal_on_first_wait)
                assert callable(handler)
                handler(self.signal_on_first_wait, None)
            if self.interrupt_on_first_wait is not None:
                raise self.interrupt_on_first_wait
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


@dataclass
class CountingBackend(FakeBackend):
    def anchor_search_counters(self) -> dict[str, int]:
        return {
            "anchor_probe_calls": len(self.calls) * 2,
            "anchor_fast_path_hits": len(self.calls),
            "anchor_fallback_attempted": 0,
            "anchor_fallback_hits": 0,
            "anchor_all_frames_not_found": 0,
        }

    def recall_rescue_counters(self) -> dict[str, int]:
        return {
            "multi_instance_rescue_attempted": len(self.calls),
            "multi_instance_rescue_selected": len(self.calls),
            "multi_instance_rescue_rejected": 0,
        }


@dataclass
class FakeIdleClock:
    current: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds


@dataclass(frozen=True)
class Fixture:
    writable: Path
    dataset: Path
    input_root: Path
    output_root: Path
    base_path: Path
    config: V3Config
    identity: ConfigIdentity


def _paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    models = (writable / "models").resolve()
    for path in (writable, dataset, pretrained, models):
        path.mkdir(parents=True, exist_ok=True)
    for module in (config_module, production):
        monkeypatch.setattr(module, "ALLOWED_WRITABLE_ROOT", writable)
        monkeypatch.setattr(module, "ALLOWED_DATASET_ROOT", dataset)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", models)
    dataset_json = dataset / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    qwen_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "base-run",
        export_root=writable / "base-export",
        source=SourceConfig(limit=100),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(qwen_model)),
            instruction_writer=QwenServiceConfig(model=str(qwen_model)),
            candidate_judge=QwenServiceConfig(model=str(qwen_model)),
            background_remove_judge=QwenServiceConfig(model=str(qwen_model)),
        ),
        sam3=Sam3Config(model_path=models / "sam3.pt"),
        remove=RemoveConfig(
            enabled=False,
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=models / "object-remover",
            generation_mask_dilation_pixels=0,
        ),
    )
    config.validate()
    base_path = writable / "base.yaml"
    base_path.write_text("base: fixture\n", encoding="utf-8")
    identity = ConfigIdentity(
        path=base_path,
        sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        fingerprint=config.fingerprint(),
        config=config,
    )
    input_root = dataset / "entity_annotations"
    (input_root / "parts").mkdir(parents=True)
    return Fixture(
        writable=writable,
        dataset=dataset,
        input_root=input_root,
        output_root=writable / "pre-qwen-v1",
        base_path=base_path,
        config=config,
        identity=identity,
    )


def _write_loadable_config(fixture: Fixture, *, qwen_rescue: bool = False) -> Path:
    path = fixture.writable / ("qwen.yaml" if qwen_rescue else "off.yaml")
    payload = {
        "dataset_json": str(fixture.config.dataset_json),
        "run_root": str(fixture.writable / "yaml-run"),
        "export_root": str(fixture.writable / "yaml-export"),
        "source": {"limit": 100},
        "qwen": {
            "candidate_judge": {
                "model": str(
                    fixture.dataset.parent
                    / "pretrained"
                    / "Qwen"
                    / "Qwen3-VL-32B-Instruct"
                )
            },
            "background_remove_judge": {
                "model": str(
                    fixture.dataset.parent
                    / "pretrained"
                    / "Qwen"
                    / "Qwen3-VL-32B-Instruct"
                )
            },
        },
        "sam3": {
            "model_path": str(fixture.writable / "models" / "sam3.pt"),
            "multi_instance_rescue_mode": (
                "qwen_anchor_select_v1" if qwen_rescue else "off"
            ),
        },
        "remove": {
            "enabled": False,
            "base_model_path": str(
                fixture.dataset.parent / "pretrained" / "Qwen" / "Qwen-Image-Edit-2511"
            ),
            "adapter_path": str(fixture.writable / "models" / "object-remover"),
            "generation_mask_dilation_pixels": 0,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _entity(index: int = 1) -> dict[str, object]:
    return {
        "entity_id": f"e{index}",
        "reference_type": "subject" if index == 1 else "object",
        "phrase": f"entity {index}",
        "grounding_prompt": f"distinct entity {index}",
    }


def _row(
    fixture: Fixture,
    source_index: int,
    *,
    status: str = "ready",
    entities: int = 1,
    background: bool = False,
) -> dict[str, object]:
    video = fixture.dataset / "clips" / f"clip-{source_index}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-mp4")
    if status == "failed":
        return {
            "source_index": source_index,
            "clip_uid": f"clip-{source_index}",
            "video_path": str(video),
            "source_video_id": "movie",
            "source_video_path": "movie.mkv",
            "shot_index": source_index,
            "status": "failed",
            "entities": [],
            "background": None,
            "instruction_template": "",
            "reason": "annotation failed",
            "repair_attempts": 0,
            "warnings": [],
        }
    return {
        "source_index": source_index,
        "clip_uid": f"clip-{source_index}",
        "video_path": str(video),
        "source_video_id": "movie",
        "source_video_path": "movie.mkv",
        "shot_index": source_index,
        "status": "ready",
        "entities": [_entity(index + 1) for index in range(entities)],
        "background": (
            {"phrase": "a room", "grounding_prompt": "empty room"}
            if background
            else None
        ),
        "instruction_template": "An entity moves in the scene.",
        "reason": None,
        "repair_attempts": 0,
        "warnings": [],
    }


def _write_shard(
    fixture: Fixture,
    rows: list[dict[str, object]],
    *,
    nominal_end: int | None = None,
) -> Path:
    start = int(rows[0]["source_index"])
    end = nominal_end if nominal_end is not None else int(rows[-1]["source_index"])
    path = fixture.input_root / "parts" / f"shard-{start:09d}-{end:09d}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_canary_shards(
    fixture: Fixture,
    *,
    shard_count: int = 16,
    eligible_per_shard: int = 5,
) -> list[Path]:
    paths = []
    for shard_index in range(shard_count):
        start = shard_index * 10_000
        rows = [
            _row(fixture, start + offset) for offset in range(eligible_per_shard)
        ]
        paths.append(_write_shard(fixture, rows, nominal_end=start + 9_999))
    return paths


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_qwen_anchor_mode_without_candidate_judge_fails_preflight() -> None:
    config = V3Config(
        dataset_json=Path("/mnt/workspace/public/dataset/source.jsonl"),
        run_root=Path("/mnt/workspace/litengjie/data/run"),
        export_root=Path("/mnt/workspace/litengjie/data/export"),
        source=SourceConfig(limit=1),
        sam3=Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
    )

    with pytest.raises(ValueError, match="qwen.candidate_judge is required"):
        validate_stage2_preflight(config)


def test_qwen_anchor_mode_passes_preflight_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    config = replace(
        fixture.config,
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )

    policy = validate_stage2_preflight(config)

    assert config.sam3.multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert policy["sam3_multi_instance_rescue_mode"] == "qwen_anchor_select_v1"
    assert policy["sam3_anchor_qwen_enabled"] is True
    assert policy["candidate_judge_base_url"] == "http://127.0.0.1:8000/v1"
    assert policy["candidate_judge_model"] == config.qwen.candidate_judge.model


def test_candidate_judge_health_uses_models_endpoint_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    service = replace(
        fixture.config.qwen.candidate_judge,
        base_url="http://6.167.57.88:8000/v1",
    )
    config = replace(
        fixture.config,
        qwen=replace(fixture.config.qwen, candidate_judge=service),
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )
    requests: list[tuple[str, float]] = []

    class Response:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def opener(request: object, *, timeout: float) -> Response:
        requests.append((request.full_url, timeout))
        return Response()

    health = check_stage2_candidate_judge_health(config, opener=opener)

    assert requests == [("http://6.167.57.88:8000/v1/models", 5.0)]
    assert health["candidate_judge_health"] == "available"


def test_candidate_judge_health_failure_prevents_sam3_backend_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    config = replace(
        fixture.config,
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )
    identity = replace(fixture.identity, config=config)
    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: identity)
    monkeypatch.setattr(batch_tool, "enumerate_annotation_shards", lambda root: [shard])
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda current: (_ for _ in ()).throw(RuntimeError("gateway unavailable")),
    )
    monkeypatch.setattr(
        batch_tool,
        "default_backend_factory",
        lambda current: pytest.fail("SAM3 must not load before Qwen health passes"),
    )

    with pytest.raises(RuntimeError, match="gateway unavailable"):
        batch_tool.run_claim_loop(
            input_root=fixture.input_root,
            base_config=fixture.base_path,
            output_root=fixture.output_root,
            scan_offset=0,
        )


def test_full_stage2_shard_is_one_to_one_and_skips_ineligible_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(
        fixture,
        [
            _row(fixture, 0),
            _row(fixture, 1, status="failed"),
            _row(fixture, 2, entities=0),
        ],
        nominal_end=9999,
    )
    backend = FakeBackend()
    decoder = FakeDecoder()

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=decoder,
    )

    rows = _read_jsonl(Path(result["path"]))
    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert [row["status"] for row in rows] == [
        "ready_no_background",
        "skipped_annotation_failed",
        "skipped_no_entities",
    ]
    assert backend.calls == ["e1"]
    assert decoder.calls == ["inspect:clip-0.mp4", "decode:clip-0.mp4"]
    assert Path(result["path"]).name == shard.name


def test_annotation_shard_rejects_duplicate_ready_clip_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    rows = [_row(fixture, 0), _row(fixture, 1)]
    rows[1]["clip_uid"] = rows[0]["clip_uid"]
    shard = _write_shard(fixture, rows)

    with pytest.raises(ValueError, match="duplicate ready clip_uid"):
        load_annotation_shard(shard)


def test_completed_shard_validates_and_skips_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    backend = FakeBackend(failure=RuntimeError("must not run"))

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=FakeDecoder(),
    )

    assert result["skipped"] is True
    assert backend.calls == []


def test_artifacts_published_before_row_are_reused_without_sam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard_path.stem / "clip-0"
    first = FakeBackend()
    process_ready_clip(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend(failure=RuntimeError("must not run"))

    process_shard(
        shard_path,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=second,
        decoder=FakeDecoder(),
    )

    assert first.calls == ["e1"]
    assert second.calls == []


def test_retryable_sam_failure_does_not_commit_terminal_row_and_resumes_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    first_decoder = FakeDecoder()
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("CUDA OOM")),
        decoder=first_decoder,
    )

    assert result["retryable"] is True
    assert not (fixture.output_root / "parts" / shard.name).exists()
    assert not (fixture.output_root / "parts" / f"{shard.name}.partial").exists()
    second_decoder = FakeDecoder()
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=second_decoder,
    )
    assert first_decoder.calls
    assert second_decoder.calls == []


def test_coverage_rejection_is_business_terminal_and_skips_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(visible_slots=6),
        decoder=FakeDecoder(),
    )

    rows = _read_jsonl(Path(result["path"]))
    assert rows[0]["status"] == "coverage_rejected"
    assert rows[0]["background_status"] is None


def test_background_pending_remove_is_published_after_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    row = _read_jsonl(Path(result["path"]))[0]
    assert row["status"] == "ready_background_pending_remove"
    mask_paths = list((fixture.output_root / row["artifact_root"]).rglob("source_mask_*.png"))
    assert len(mask_paths) == 1


def test_torn_partial_tail_is_truncated_and_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    completed = Path(result["path"])
    first = completed.read_bytes().splitlines(keepends=True)[0]
    partial = completed.with_name(f"{completed.name}.partial")
    completed.unlink()
    partial.write_bytes(first + b'{"source_index":1')

    resumed = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("artifacts must be reused")),
        decoder=FakeDecoder(),
    )

    assert len(_read_jsonl(Path(resumed["path"]))) == 2


def test_changed_stage1_shard_sha_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    changed = _row(fixture, 0)
    changed["warnings"] = ["changed"]
    shard.write_text(json.dumps(changed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        process_shard(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=FakeBackend(),
            decoder=FakeDecoder(),
        )


@pytest.mark.parametrize("artifact", ["frames", "masks", "background"])
def test_completed_shard_fails_closed_on_corrupt_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    row = _read_jsonl(Path(result["path"]))[0]
    workspace = fixture.output_root / row["artifact_root"] / "run" / "clips" / "clip-0"
    if artifact == "frames":
        (workspace / "frames" / "00.jpg").write_bytes(b"corrupt")
    elif artifact == "masks":
        (workspace / "masks.rle.json").write_text("{}", encoding="utf-8")
    else:
        mask = next((workspace / "background").glob("source_mask_*.png"))
        mask.write_bytes(b"corrupt")

    with pytest.raises((ValueError, OSError)):
        process_shard(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=FakeBackend(failure=RuntimeError("must not rerun")),
            decoder=FakeDecoder(),
        )


def test_invalid_source_index_and_clip_uid_prefix_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    bad = _row(fixture, 1)
    shard = fixture.input_root / "parts" / "shard-000000000-000000009.jsonl"
    shard.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not contiguous"):
        load_annotation_shard(shard)


def test_flock_claim_is_nonblocking_and_released(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.lock"
    with shard_lock(path), pytest.raises(ShardBusyError), shard_lock(path):
        pass
    with shard_lock(path):
        pass


def test_inspect_and_inventory_report_stage_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    inspected = inspect_stage2_shard(result["path"])
    report = inventory(fixture.input_root, fixture.output_root)
    assert inspected["rows"] == 1
    assert inspected["sam3_entity_ready"] == 1
    assert report["stage2_completed_shards"] == 1
    assert report["outstanding_shards"] == 0


def test_canary_selection_is_deterministic_and_counts_only_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    first_rows = [
        _row(fixture, 0, status="failed"),
        _row(fixture, 1, entities=0),
        *[_row(fixture, index) for index in range(2, 7)],
    ]
    _write_shard(fixture, first_rows, nominal_end=9_999)
    for shard_index in range(1, 17):
        start = shard_index * 10_000
        _write_shard(
            fixture,
            [_row(fixture, start + offset) for offset in range(5)],
            nominal_end=start + 9_999,
        )

    first = select_canary_rows(
        fixture.input_root,
        gpu_count=8,
        canary_shards=16,
        samples_per_shard=5,
    )
    second = select_canary_rows(
        fixture.input_root,
        gpu_count=8,
        canary_shards=16,
        samples_per_shard=5,
    )

    assert first == second
    assert len(first) == 80
    assert [sum(item.gpu_slot == slot for item in first) for slot in range(8)] == [10] * 8
    assert [item.source_index for item in first[:5]] == [2, 3, 4, 5, 6]
    assert {Path(item.input_annotation_shard).name for item in first}.isdisjoint(
        {"shard-000160000-000169999.jsonl"}
    )


def test_canary_selection_rejects_short_shard_without_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture, shard_count=2, eligible_per_shard=4)
    start = 20_000
    _write_shard(
        fixture,
        [_row(fixture, start + offset) for offset in range(5)],
        nominal_end=start + 9_999,
    )

    with pytest.raises(ValueError, match=r"shard-000000000-000009999.*4 eligible.*5 required"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=1,
            canary_shards=2,
            samples_per_shard=5,
        )


def test_canary_selection_rejects_cross_shard_duplicate_clip_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture, shard_count=2, eligible_per_shard=5)
    second = fixture.input_root / "parts" / "shard-000010000-000019999.jsonl"
    rows = _read_jsonl(second)
    rows[0]["clip_uid"] = "clip-0"
    second.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate clip_uid across Stage1 shards"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=1,
            canary_shards=2,
            samples_per_shard=5,
        )


def test_canary_selection_requires_divisible_gpu_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must be divisible"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=8,
            canary_shards=3,
            samples_per_shard=5,
        )


def test_canary_output_cannot_enter_production_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    production_root = fixture.writable / "formal" / "pre-qwen-v1"
    monkeypatch.setattr(production, "DEFAULT_PRODUCTION_OUTPUT_ROOT", production_root)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)

    for output_root in (
        production_root,
        production_root / "fake-canary",
        production_root.parent,
    ):
        with pytest.raises(ValueError, match="fully disjoint"):
            prepare_canary(
                input_root=fixture.input_root,
                base_config=base,
                output_root=output_root,
                gpus=["0"],
                canary_shards=1,
                samples_per_shard=1,
            )


def test_exact_entity_mask_public_output_has_stage2_only_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )

    assert production._safe_output_root(entity_mask_root) == entity_mask_root
    assert production._safe_output_root(fixture.output_root) == fixture.output_root
    for rejected in (
        fixture.dataset,
        entity_mask_root.parent,
        entity_mask_root.parent / "foo",
    ):
        with pytest.raises(ValueError, match="writable root|public dataset"):
            production._safe_output_root(rejected)
    for rejected_canary in (
        entity_mask_root,
        entity_mask_root / "fake-canary",
        entity_mask_root.parent,
    ):
        with pytest.raises(ValueError, match="fully disjoint"):
            production._safe_output_root(rejected_canary, canary=True)


def test_entity_mask_public_workspace_does_not_loosen_normal_v3_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    workspace = entity_mask_root / "artifacts" / "shard-0" / "clip-0"
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )

    public_config = replace(
        fixture.config,
        run_root=workspace / "run",
        export_root=workspace / "unused-export",
    )
    with pytest.raises(ValueError, match="run_root must be inside"):
        public_config.validate()
    with pytest.raises(ValueError, match="run_root must be inside"):
        production.RunStorage(public_config)

    runtime_config, storage = production._stage2_workspace_storage(
        fixture.config,
        workspace,
        output_root=entity_mask_root,
    )
    assert runtime_config.resolved_run_root == (workspace / "run").resolve()
    assert storage.root == (workspace / "run").resolve()
    storage.initialize(git_commit=production.VISUAL_ALGORITHM_FREEZE)
    assert storage.run_path.is_file()


def test_stage2_processes_into_exact_entity_mask_public_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )
    shard_path = _write_shard(fixture, [_row(fixture, 0)])

    result = process_shard(
        shard_path,
        output_root=entity_mask_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    assert result["rows"] == 1
    assert (entity_mask_root / "parts" / shard_path.name).is_file()
    assert (
        entity_mask_root
        / "artifacts"
        / shard_path.stem
        / "clip-0"
        / "run"
        / "run.json"
    ).is_file()


def test_prepare_canary_persists_and_reuses_exact_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture)
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "run-1"

    manifest, first = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )
    monkeypatch.setattr(
        production,
        "select_canary_rows",
        lambda *args, **kwargs: pytest.fail("persisted selection must be reused"),
    )
    resumed_manifest, second = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )

    assert manifest == resumed_manifest
    assert first == second
    with pytest.raises(ValueError, match="identity"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=base,
            output_root=output,
            gpus=["0", "1"],
            canary_shards=16,
            samples_per_shard=5,
        )
    assert manifest.selection_policy == "first_n_shards_first_k_eligible_v1"
    assert manifest.selected_shard_names == [
        f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl"
        for index in range(16)
    ]
    assert manifest.duplicate_clip_uid_skipped == 0


def test_canary_resume_rejects_changed_input_shard_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "sha"
    prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    changed = _row(fixture, 0)
    changed["warnings"] = ["mutated"]
    shard.write_text(json.dumps(changed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA changed"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=base,
            output_root=output,
            gpus=["0"],
            canary_shards=1,
            samples_per_shard=1,
        )


def test_qwen_anchor_canary_uses_original_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture, qwen_rescue=True)
    output = fixture.writable / "canary" / "qwen-anchor"

    manifest, _selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    assert manifest.sam3_multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert manifest.sam3_anchor_qwen_enabled is True
    assert manifest.candidate_judge_base_url == "http://127.0.0.1:8000/v1"
    assert "qwen_required" not in manifest.model_dump()
    assert "qwen_calls" not in manifest.model_dump()


def test_off_mode_canary_cannot_resume_with_qwen_anchor_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    output = fixture.writable / "canary" / "off-mode-history"
    prepare_canary(
        input_root=fixture.input_root,
        base_config=_write_loadable_config(fixture),
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    with pytest.raises(ValueError, match="identity"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=_write_loadable_config(fixture, qwen_rescue=True),
            output_root=output,
            gpus=["0"],
            canary_shards=1,
            samples_per_shard=1,
        )


def test_canary_worker_resume_reuses_completed_masks_and_summary_counts_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "run"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    first = FakeBackend()
    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend(failure=RuntimeError("must not run"))
    resumed = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=second,
        decoder=FakeDecoder(),
    )
    summary = canary_summary(output, manifest)

    assert result["completed"] == 1
    assert resumed["skipped"] is True
    assert second.calls == []
    assert summary["status"] == "complete"
    assert summary["selection_complete"] is True
    assert summary["functional_status"] == "pass"
    assert summary["failed_input"] == 0
    assert summary["failed_frames"] == 0
    assert summary["terminal_failures"] == 0
    assert summary["duplicate_clip_uid_skipped"] == 0
    assert canary_tool._canary_exit_code([], summary) == 0
    assert summary["frames_bytes"] > 0
    assert summary["masks_bytes"] > 0
    assert summary["annotation_qwen_calls"] == 0
    assert summary["background_remove_qwen_calls"] == 0
    assert "qwen_calls" not in summary


def test_canary_sam3_counters_use_backend_deltas_not_per_clip_cumulative_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "counter-deltas"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=2,
    )
    backend = CountingBackend()

    run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=backend,
        decoder=FakeDecoder(),
    )
    summary = canary_summary(output, manifest)

    assert summary["anchor_probe_calls"] == 4
    assert summary["anchor_fast_path_hits"] == 2
    assert summary["multi_instance_rescue_attempted"] == 2
    assert summary["multi_instance_rescue_selected"] == 2


def test_canary_retryable_sam_failure_keeps_worker_partial_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "retry"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=FakeBackend(failure=RuntimeError("CUDA context error")),
        decoder=FakeDecoder(),
    )

    assert result["retryable"] is True
    assert not (output / "workers" / "gpu-0.jsonl").exists()
    summary = canary_summary(output, manifest)
    assert summary["status"] == "partial"
    assert summary["selection_complete"] is False
    assert summary["functional_status"] == "fail"
    assert summary["outstanding_retryable_work"] == 1
    assert canary_tool._canary_exit_code([], summary) == 2


@pytest.mark.parametrize("terminal_status", ["failed_input", "failed_frames"])
def test_canary_terminal_row_completes_selection_but_fails_functionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "terminal"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    item = selection[0]
    terminal = production.Stage2Row(
        source_index=item.source_index,
        clip_uid=item.clip_uid,
        input_annotation_shard=item.input_annotation_shard,
        input_annotation_shard_sha256=item.input_annotation_shard_sha256,
        input_row_sha256=item.input_row_sha256,
        status=terminal_status,
        reason="invalid input or frame stream",
    )
    worker_path = output / "workers" / "gpu-0.jsonl"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(
        json.dumps(terminal.model_dump(mode="json")) + "\n", encoding="utf-8"
    )

    summary = canary_summary(output, manifest)

    assert summary["status"] == "complete"
    assert summary["selection_complete"] is True
    assert summary["functional_status"] == "fail"
    assert summary[terminal_status] == 1
    assert summary["terminal_failures"] == 1
    assert canary_tool._canary_exit_code([], summary) == 2


def test_retryable_canary_worker_cli_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "worker-exit"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    monkeypatch.setattr(
        canary_tool, "_load_identity_files", lambda root: (manifest, selection)
    )
    monkeypatch.setattr(canary_tool, "default_backend_factory", lambda config: object())
    monkeypatch.setattr(
        canary_tool,
        "run_canary_worker",
        lambda **kwargs: {
            "gpu_slot": 0,
            "selected": 1,
            "completed": 0,
            "retryable": True,
        },
    )

    with pytest.raises(SystemExit) as raised:
        canary_tool.main(["--output-root", str(output), "--worker-slot", "0"])

    assert raised.value.code == 2


def test_canary_partial_worker_resumes_from_exact_next_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "partial"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=2,
    )
    first = FakeBackend(
        failure=RuntimeError("CUDA OOM"),
        fail_on_call=2,
    )
    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend()
    second_decoder = FakeDecoder()

    resumed = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=second,
        decoder=second_decoder,
    )

    assert result["completed"] == 1
    assert resumed["completed"] == 2
    assert second.calls == ["e1"]
    assert second_decoder.calls == []


def test_stage2_backend_factory_uses_builder_default_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    candidate_judge = replace(
        fixture.config.qwen.candidate_judge,
        base_url="http://6.167.57.88:8000/v1",
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    physical_index_config = replace(
        fixture.config,
        qwen=replace(fixture.config.qwen, candidate_judge=candidate_judge),
        sam3=replace(
            fixture.config.sam3,
            device="cuda:7",
            object_rescue_mode="phrase_retry_v1",
            not_found_rescue_mode="entity_phrase_retry_v1",
            multi_instance_rescue_mode="qwen_anchor_select_v1",
            anchor_search_mode="progressive_v1",
        ),
    )

    backend = production.default_backend_factory(physical_index_config)

    assert isinstance(backend, Sam3SegmentationBackend)
    assert backend.config.device == "cuda"
    assert backend.config.object_rescue_mode == "phrase_retry_v1"
    assert backend.config.not_found_rescue_mode == "entity_phrase_retry_v1"
    assert backend.config.multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert backend.config.anchor_search_mode == "progressive_v1"
    assert isinstance(backend._anchor_selector, QwenSam3AnchorSelector)
    assert backend._anchor_selector.config is candidate_judge
    assert backend._anchor_selector.config.base_url == "http://6.167.57.88:8000/v1"
    assert physical_index_config.sam3.device == "cuda:7"
    assert production._workspace_config(
        physical_index_config, fixture.writable / "workspace"
    ).sam3.device == "cuda:0"


def test_stage2_backend_lazy_init_supports_builder_without_device_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    assert fixture.config.sam3.model_path is not None
    fixture.config.sam3.model_path.write_bytes(b"sam3-checkpoint")
    backend = production.default_backend_factory(
        replace(
            fixture.config,
            sam3=replace(fixture.config.sam3, device="cuda:5"),
        )
    )
    assert isinstance(backend, Sam3SegmentationBackend)
    calls: list[str] = []
    predictor = object()

    def installed_builder_without_device(checkpoint_path: str) -> object:
        calls.append(checkpoint_path)
        return predictor

    backend._builder = installed_builder_without_device

    assert backend._load_predictor() is predictor
    assert calls == [str(fixture.config.sam3.model_path.resolve())]


def test_frames_ready_checkpoint_resumes_with_fixed_runtime_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    first_decoder = FakeDecoder()
    first = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("SAM3 startup failed")),
        decoder=first_decoder,
    )
    workspace = fixture.output_root / "artifacts" / shard.stem / "clip-0"
    checkpoint = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
    fixed_backend = FakeBackend()
    effective_devices: list[str] = []

    def build_backend(config: V3Config) -> FakeBackend:
        effective_devices.append(config.sam3.device)
        return fixed_backend

    monkeypatch.setattr(production, "build_sam3_segment_backend", build_backend)
    backend = production.default_backend_factory(fixture.config)
    resumed_decoder = FakeDecoder()
    resumed = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=resumed_decoder,
    )

    assert first["retryable"] is True
    assert checkpoint["stage"] == "frames_ready"
    assert first_decoder.calls
    assert resumed["rows"] == 1
    assert effective_devices == ["cuda"]
    assert fixed_backend.calls == ["e1"]
    assert resumed_decoder.calls == []


def test_canary_launcher_isolates_physical_gpus_with_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture, qwen_rescue=True)
    payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    payload["qwen"]["candidate_judge"]["base_url"] = (
        "http://6.167.57.88:8000/v1"
    )
    base.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = fixture.writable / "canary" / "gpu-isolation"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["5", "7"],
        canary_shards=1,
        samples_per_shard=2,
    )
    launched: list[dict[str, object]] = []

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command: list[str], **kwargs: object) -> Process:
        launched.append({"command": command, **kwargs})
        return Process()

    monkeypatch.setattr(
        canary_tool,
        "prepare_canary",
        lambda **kwargs: (manifest, selection),
    )
    monkeypatch.setattr(
        canary_tool,
        "canary_summary",
        lambda output_root, current_manifest: {
            "status": "complete",
            "selection_complete": True,
            "functional_status": "pass",
        },
    )
    monkeypatch.setattr(
        canary_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "available"},
    )
    monkeypatch.setattr(canary_tool.subprocess, "Popen", popen)

    canary_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(base),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "5,7",
            "--canary-shards",
            "1",
            "--samples-per-shard",
            "2",
        ]
    )

    assert [item["env"]["CUDA_VISIBLE_DEVICES"] for item in launched] == ["5", "7"]
    assert manifest.candidate_judge_base_url == "http://6.167.57.88:8000/v1"
    assert all("8001" not in " ".join(item["command"]) for item in launched)
    assert all("8008" not in " ".join(item["command"]) for item in launched)


def test_new_stage1_shard_is_discovered_without_topology_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    first = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        first,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    _write_shard(fixture, [_row(fixture, 1)])

    report = inventory(fixture.input_root, fixture.output_root)

    assert report["stage1_completed_shards"] == 2
    assert report["stage2_completed_shards"] == 1
    assert report["outstanding_shards"] == 1


def test_claim_worker_builds_one_persistent_backend_for_multiple_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(2)],
    )
    backend = FakeBackend()
    factory_calls: list[V3Config] = []
    process_backends: list[object] = []
    monkeypatch.setattr(
        batch_tool,
        "load_config_identity",
        lambda path: fixture.identity,
    )
    monkeypatch.setattr(
        batch_tool,
        "enumerate_annotation_shards",
        lambda root: [shard_path],
    )

    def factory(config: V3Config) -> FakeBackend:
        factory_calls.append(config)
        return backend

    def process(
        shard: AnnotationShard,
        chunk: ExecutionChunk,
        **kwargs: object,
    ) -> dict[str, object]:
        process_backends.append(kwargs["backend"])
        destination = execution_chunk_path(fixture.output_root, shard, chunk)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("done\n", encoding="utf-8")
        return {
            "path": str(destination),
            "rows": 1,
            "skipped": False,
            "chunk": chunk.stem,
        }

    def compact(
        shard: AnnotationShard, **kwargs: object
    ) -> dict[str, object] | None:
        del kwargs
        chunks = build_execution_chunks(shard, chunk_rows=1)
        if any(
            not execution_chunk_path(fixture.output_root, shard, chunk).is_file()
            for chunk in chunks
        ):
            return None
        destination = fixture.output_root / "parts" / shard.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("done\n", encoding="utf-8")
        return {"path": str(destination), "rows": 1, "skipped": False}

    monkeypatch.setattr(batch_tool, "default_backend_factory", factory)
    monkeypatch.setattr(batch_tool, "process_execution_chunk", process)
    monkeypatch.setattr(batch_tool, "compact_execution_chunks", compact)

    batch_tool.run_claim_loop(
        input_root=fixture.input_root,
        base_config=fixture.base_path,
        output_root=fixture.output_root,
        scan_offset=1,
        chunk_rows=1,
    )

    assert len(factory_calls) == 1
    assert process_backends == [backend, backend]
    assert backend.close_calls == 1


def test_claim_worker_timing_keeps_one_lazy_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    factory_calls = 0

    class TimingBackend(FakeBackend):
        _predictor = None

        def __init__(self) -> None:
            super().__init__()
            self.predictor_builds = 0

        def _build_predictor(self, *, compile_enabled: bool) -> object:
            del compile_enabled
            self.predictor_builds += 1
            return _TransparentPredictor()

        def performance_counters(self) -> dict[str, object]:
            return {"sam3_segment_model_call_time_seconds": 0.0}

    backend = TimingBackend()

    def factory(config: V3Config) -> TimingBackend:
        nonlocal factory_calls
        del config
        factory_calls += 1
        return backend

    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        batch_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(batch_tool, "default_backend_factory", factory)
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )

    result = batch_tool.run_claim_loop(
        input_root=fixture.input_root,
        base_config=fixture.base_path,
        output_root=fixture.output_root,
        scan_offset=0,
        chunk_rows=1,
        sam3_request_timing=True,
        sam3_session_reuse_mode="clip_reset_v1",
    )

    assert factory_calls == 1
    assert backend.predictor_builds == 0
    assert backend.close_calls == 1
    assert result["sam3_request_timing"] == {}
    assert result["sam3_backend_performance_counters"] == {
        "sam3_segment_model_call_time_seconds": 0.0
    }
    assert result["sam3_session_reuse_mode"] == "clip_reset_v1"
    assert result["sam3_physical_start_session_calls"] == 0
    assert (
        fixture.output_root / "_internal" / "sam3-session-reuse.json"
    ).is_file()


def test_final_partial_sized_stage1_shard_keeps_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(
        fixture,
        [_row(fixture, 10000), _row(fixture, 10001)],
        nominal_end=19999,
    )

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    assert Path(result["path"]).name == "shard-000010000-000019999.jsonl"


def test_machine_topology_is_absent_from_shard_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    meta = json.loads(
        (fixture.output_root / "parts" / f"{shard.stem}.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert not {"rank", "world_size", "gpu", "hostname"}.intersection(meta)


def test_no_remove_or_qwen_modules_are_constructed_by_stage2_source() -> None:
    source = Path(production.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "QwenAnnotationClient",
        "QwenBackgroundRemovalJudge",
        "QwenBackgroundFinalGuard",
        "QwenCrossPairJudge",
        "QwenReferenceEditJudge",
        "QwenReferenceIntegrityJudge",
        "QwenInstructionWriter",
        "QwenSubjectAttributeClient",
        "Boogu",
    ):
        assert forbidden not in source
    assert "QwenSam3AnchorSelector" not in source
    assert "build_sam3_segment_backend" in source
    assert "build_background_candidates" in source


def test_execution_chunk_identity_is_fixed_contiguous_and_topology_independent(
    tmp_path: Path,
) -> None:
    rows = tuple({"source_index": index} for index in range(10_000))
    shard = AnnotationShard(
        path=tmp_path / "shard-000000000-000009999.jsonl",
        sha256="0" * 64,
        nominal_start=0,
        nominal_end=9_999,
        rows=rows,
    )

    chunks = build_execution_chunks(shard)

    assert DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS == 100
    assert len(chunks) == 100
    assert chunks[0].stem == "chunk-000000000-000000099"
    assert chunks[-1].stem == "chunk-000009900-000009999"
    assert all(chunk.row_count == 100 for chunk in chunks)
    assert all(
        left.source_index_end + 1 == right.source_index_start
        and left.row_offset_end == right.row_offset_start
        for left, right in pairwise(chunks)
    )
    assert chunks == build_execution_chunks(shard)


def test_execution_chunk_partial_final_shard_has_short_last_chunk(
    tmp_path: Path,
) -> None:
    rows = tuple({"source_index": 10_000 + index} for index in range(205))
    shard = AnnotationShard(
        path=tmp_path / "shard-000010000-000019999.jsonl",
        sha256="0" * 64,
        nominal_start=10_000,
        nominal_end=19_999,
        rows=rows,
    )

    chunks = build_execution_chunks(shard)

    assert [(chunk.source_index_start, chunk.source_index_end) for chunk in chunks] == [
        (10_000, 10_099),
        (10_100, 10_199),
        (10_200, 10_204),
    ]
    assert [chunk.row_count for chunk in chunks] == [100, 100, 5]


def test_chunk_claim_is_nonblocking_released_after_crash_and_reclaimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, 0, status="failed"), _row(fixture, 1, status="failed")],
    )
    shard = load_annotation_shard(shard_path)
    first, second = build_execution_chunks(shard, chunk_rows=1)
    lock_path = production._chunk_lock_path(fixture.output_root, shard, first)

    with shard_lock(lock_path), pytest.raises(ShardBusyError):
        process_execution_chunk(
            shard,
            first,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=None,
            chunk_rows=1,
        )
    with pytest.raises(RuntimeError, match="worker crash"), shard_lock(lock_path):
        raise RuntimeError("worker crash")

    first_result = process_execution_chunk(
        shard,
        first,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )
    second_result = process_execution_chunk(
        shard,
        second,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )

    assert first_result["chunk"] == first.stem
    assert second_result["chunk"] == second.stem
    assert first.stem != second.stem


def test_claim_loop_skips_busy_chunk_and_work_steals_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, 0, status="failed"), _row(fixture, 1, status="failed")],
    )
    shard = load_annotation_shard(shard_path)
    first, second = build_execution_chunks(shard, chunk_rows=1)
    backend = FakeBackend()
    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        batch_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(batch_tool, "default_backend_factory", lambda config: backend)
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    first_lock = production._chunk_lock_path(fixture.output_root, shard, first)

    with shard_lock(first_lock):
        initial = batch_tool.run_claim_loop(
            input_root=fixture.input_root,
            base_config=fixture.base_path,
            output_root=fixture.output_root,
            scan_offset=0,
            chunk_rows=1,
            idle_exit_seconds=0,
        )

    assert initial["chunks_completed"] == 1
    assert not execution_chunk_path(fixture.output_root, shard, first).exists()
    assert execution_chunk_path(fixture.output_root, shard, second).is_file()
    resumed = batch_tool.run_claim_loop(
        input_root=fixture.input_root,
        base_config=fixture.base_path,
        output_root=fixture.output_root,
        scan_offset=999,
        chunk_rows=1,
    )
    assert resumed["chunks_completed"] == 1
    assert (fixture.output_root / "parts" / shard_path.name).is_file()
    assert backend.close_calls == 2


def test_claim_loop_with_all_work_complete_exits_without_idle_or_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    process_shard(
        shard_path,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )
    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        batch_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(
        batch_tool,
        "default_backend_factory",
        lambda config: pytest.fail("completed work must not create SAM3 backend"),
    )
    clock = FakeIdleClock()

    result = batch_tool.run_claim_loop(
        input_root=fixture.input_root,
        base_config=fixture.base_path,
        output_root=fixture.output_root,
        scan_offset=0,
        chunk_rows=1,
        idle_exit_seconds=60,
        sleep=lambda seconds: pytest.fail("completed work must not sleep"),
        monotonic=clock.monotonic,
    )

    assert result["outstanding_or_busy"] is False
    assert result["idle_rescans"] == 0
    assert result["worker_idle_scan_seconds"] == 0.0


@pytest.mark.parametrize("release_mode", ["unlock", "owner_crash_close"])
def test_busy_chunk_released_during_idle_grace_is_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_mode: str,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    shard = load_annotation_shard(shard_path)
    chunk = build_execution_chunks(shard, chunk_rows=1)[0]
    lock_path = production._chunk_lock_path(fixture.output_root, shard, chunk)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_handle = lock_path.open("a+b")
    fcntl.flock(owner_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    backend = FakeBackend()
    factory_calls = 0
    clock = FakeIdleClock()

    def factory(config: V3Config) -> FakeBackend:
        nonlocal factory_calls
        del config
        factory_calls += 1
        return backend

    def release_during_backoff(seconds: float) -> None:
        clock.sleep(seconds)
        if release_mode == "unlock":
            fcntl.flock(owner_handle.fileno(), fcntl.LOCK_UN)
        else:
            owner_handle.close()

    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        batch_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(batch_tool, "default_backend_factory", factory)
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    try:
        result = batch_tool.run_claim_loop(
            input_root=fixture.input_root,
            base_config=fixture.base_path,
            output_root=fixture.output_root,
            scan_offset=0,
            chunk_rows=1,
            idle_exit_seconds=5,
            sleep=release_during_backoff,
            monotonic=clock.monotonic,
        )
    finally:
        if not owner_handle.closed:
            owner_handle.close()

    assert result["chunks_completed"] == 1
    assert result["outstanding_or_busy"] is False
    assert result["idle_rescans"] == 1
    assert clock.sleeps == [1.0]
    assert factory_calls == 1
    assert backend.close_calls == 1
    assert (fixture.output_root / "parts" / shard_path.name).is_file()


def test_continuously_busy_chunk_uses_backoff_then_exits_after_idle_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    shard = load_annotation_shard(shard_path)
    chunk = build_execution_chunks(shard, chunk_rows=1)[0]
    lock_path = production._chunk_lock_path(fixture.output_root, shard, chunk)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_handle = lock_path.open("a+b")
    fcntl.flock(owner_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    backend = FakeBackend()
    factory_calls = 0
    clock = FakeIdleClock()

    def factory(config: V3Config) -> FakeBackend:
        nonlocal factory_calls
        del config
        factory_calls += 1
        return backend

    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        batch_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(batch_tool, "default_backend_factory", factory)
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    try:
        result = batch_tool.run_claim_loop(
            input_root=fixture.input_root,
            base_config=fixture.base_path,
            output_root=fixture.output_root,
            scan_offset=0,
            chunk_rows=1,
            idle_exit_seconds=2,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    finally:
        owner_handle.close()

    assert result["outstanding_or_busy"] is True
    assert result["chunks_claimed"] == 0
    assert result["idle_rescans"] == 2
    assert result["idle_exit_seconds"] == 2
    assert clock.sleeps == [1.0, 1.0]
    assert result["worker_idle_scan_seconds"] >= 2.0
    assert factory_calls == 1
    assert backend.close_calls == 1


def test_chunk_partial_resumes_and_truncates_incomplete_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(3)],
    )
    shard = load_annotation_shard(shard_path)
    chunk = build_execution_chunks(shard, chunk_rows=3)[0]
    first = process_execution_chunk(
        shard,
        chunk,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=3,
    )
    final = Path(first["path"])
    first_line = final.read_bytes().splitlines(keepends=True)[0]
    partial = final.with_name(f"{final.name}.partial")
    final.unlink()
    partial.write_bytes(first_line + b'{"source_index":1')

    resumed = process_execution_chunk(
        shard,
        chunk,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=3,
    )

    assert resumed["resumed"] is True
    assert [row["source_index"] for row in _read_jsonl(Path(resumed["path"]))] == [
        0,
        1,
        2,
    ]
    assert not partial.exists()


def test_chunk_partial_row_hash_and_config_mismatch_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    shard = load_annotation_shard(shard_path)
    chunk = build_execution_chunks(shard, chunk_rows=1)[0]
    result = process_execution_chunk(
        shard,
        chunk,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )
    final = Path(result["path"])
    partial = final.with_name(f"{final.name}.partial")
    payload = _read_jsonl(final)[0]
    payload["input_row_sha256"] = "f" * 64
    final.unlink()
    partial.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact annotation prefix"):
        process_execution_chunk(
            shard,
            chunk,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=None,
            chunk_rows=1,
        )

    partial.unlink()
    other_path = fixture.writable / "other-base.yaml"
    other_path.write_text("other: config\n", encoding="utf-8")
    other_identity = replace(
        fixture.identity,
        path=other_path,
        sha256=hashlib.sha256(other_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        process_execution_chunk(
            shard,
            chunk,
            output_root=fixture.output_root,
            config_identity=other_identity,
            backend=None,
            chunk_rows=1,
        )


def test_execution_chunk_size_identity_and_legacy_partial_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    ensure_execution_identity(fixture.output_root, chunk_rows=100)
    with pytest.raises(ValueError, match="execution_chunk_rows"):
        ensure_execution_identity(fixture.output_root, chunk_rows=50)

    legacy_root = fixture.writable / "legacy-stage2"
    partial = legacy_root / "parts" / "shard-000000000-000009999.jsonl.partial"
    partial.parent.mkdir(parents=True)
    partial.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy unfinished whole-shard"):
        ensure_execution_identity(legacy_root, chunk_rows=100)


def test_static_process_shard_uses_prevalidated_execution_identity_without_ensure_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(3)],
    )
    ensure_execution_identity(fixture.output_root, chunk_rows=1)
    monkeypatch.setattr(
        production,
        "_ensure_execution_identity",
        lambda *args, **kwargs: pytest.fail(
            "static shard processing must not acquire execution identity lock"
        ),
    )
    monkeypatch.setattr(
        production,
        "shard_lock",
        lambda *args, **kwargs: pytest.fail(
            "static shard processing must not acquire internal locks"
        ),
    )

    result = process_shard(
        shard_path,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        acquire_lock=False,
        execution_identity_prevalidated=True,
        static_owner=True,
        chunk_rows=1,
    )

    assert result["rows"] == 3
    assert (fixture.output_root / "parts" / shard_path.name).is_file()


def test_static_ensure_meta_skips_lock_and_validates_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = load_annotation_shard(_write_shard(fixture, [_row(fixture, 0)]))
    monkeypatch.setattr(
        production,
        "shard_lock",
        lambda *args, **kwargs: pytest.fail("static metadata must not lock"),
    )

    path = production._ensure_meta(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        static_owner=True,
    )
    repeated = production._ensure_meta(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        static_owner=True,
    )

    assert repeated == path
    assert json.loads(path.read_text(encoding="utf-8"))["input_row_count"] == 1


def test_dynamic_ensure_meta_keeps_blocking_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = load_annotation_shard(_write_shard(fixture, [_row(fixture, 0)]))
    calls: list[tuple[Path, bool]] = []

    @contextmanager
    def recording_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
        calls.append((path, blocking))
        yield

    monkeypatch.setattr(production, "shard_lock", recording_lock)

    production._ensure_meta(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
    )

    assert calls == [
        (
            fixture.output_root / "locks" / shard.path.stem / "metadata.lock",
            True,
        )
    ]


def test_chunk_compaction_is_canonical_and_incomplete_chunks_do_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(3)],
        nominal_end=9_999,
    )
    shard = load_annotation_shard(shard_path)
    chunks = build_execution_chunks(shard, chunk_rows=1)
    for chunk in chunks[:2]:
        process_execution_chunk(
            shard,
            chunk,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=None,
            chunk_rows=1,
        )
    assert compact_execution_chunks(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        chunk_rows=1,
    ) is None
    assert not (fixture.output_root / "parts" / shard.path.name).exists()

    process_execution_chunk(
        shard,
        chunks[2],
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )
    compacted = compact_execution_chunks(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        chunk_rows=1,
    )
    assert compacted is not None
    canonical = Path(compacted["path"])
    rows = _read_jsonl(canonical)
    assert canonical.name == shard.path.name
    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert {row["schema_version"] for row in rows} == {
        production.STAGE2_SCHEMA_VERSION
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_index", 999),
        ("input_row_sha256", "f" * 64),
    ],
)
def test_chunk_compaction_rejects_wrong_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(2)],
    )
    shard = load_annotation_shard(shard_path)
    chunks = build_execution_chunks(shard, chunk_rows=1)
    for chunk in chunks:
        process_execution_chunk(
            shard,
            chunk,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=None,
            chunk_rows=1,
        )
    corrupt = execution_chunk_path(fixture.output_root, shard, chunks[1])
    payload = _read_jsonl(corrupt)[0]
    payload[field] = value
    corrupt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact annotation prefix"):
        compact_execution_chunks(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            chunk_rows=1,
        )


def test_chunk_compaction_rejects_duplicate_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [_row(fixture, index, status="failed") for index in range(2)],
    )
    shard = load_annotation_shard(shard_path)
    chunks = build_execution_chunks(shard, chunk_rows=1)
    for chunk in chunks:
        process_execution_chunk(
            shard,
            chunk,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=None,
            chunk_rows=1,
        )
    first_payload = execution_chunk_path(
        fixture.output_root, shard, chunks[0]
    ).read_bytes()
    execution_chunk_path(fixture.output_root, shard, chunks[1]).write_bytes(
        first_payload
    )

    with pytest.raises(ValueError, match="exact annotation prefix"):
        compact_execution_chunks(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            chunk_rows=1,
        )


def test_compaction_lock_is_nonblocking_and_single_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0, status="failed")])
    shard = load_annotation_shard(shard_path)
    chunk = build_execution_chunks(shard, chunk_rows=1)[0]
    process_execution_chunk(
        shard,
        chunk,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=None,
        chunk_rows=1,
    )
    lock_path = production._compaction_lock_path(fixture.output_root, shard)
    with shard_lock(lock_path), pytest.raises(ShardBusyError):
        compact_execution_chunks(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            chunk_rows=1,
        )
    published = compact_execution_chunks(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        chunk_rows=1,
    )
    skipped = compact_execution_chunks(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        chunk_rows=1,
    )
    assert published is not None and published["skipped"] is False
    assert skipped is not None and skipped["skipped"] is True


def test_chunk_strategy_preserves_artifact_and_canonical_interfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
        chunk_rows=1,
    )
    row = _read_jsonl(Path(result["path"]))[0]

    assert Path(result["path"]) == fixture.output_root / "parts" / shard.name
    assert row["artifact_root"] == f"artifacts/{shard.stem}/clip-0"
    assert "chunk" not in row["artifact_root"]
    assert row["schema_version"] == production.STAGE2_SCHEMA_VERSION


def test_prepare_canary_resume_loads_each_unique_shard_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture)
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "cached-parent"
    prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )
    original = production.load_annotation_shard
    calls: list[Path] = []

    def counting_load(path: str | Path) -> AnnotationShard:
        calls.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(production, "load_annotation_shard", counting_load)
    prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )

    assert len(calls) == 16
    assert len(set(calls)) == 16


def test_canary_worker_loads_each_unique_shard_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    for shard_index in range(2):
        start = shard_index * 10_000
        _write_shard(
            fixture,
            [_row(fixture, start + offset) for offset in range(5)],
            nominal_end=start + 9_999,
        )
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "cached-worker"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=2,
        samples_per_shard=5,
    )
    original = production.load_annotation_shard
    calls: list[Path] = []

    def counting_load(path: str | Path) -> AnnotationShard:
        calls.append(Path(path).resolve())
        return original(path)

    def terminal(**kwargs: object) -> production.Stage2Row:
        row = kwargs["row"]
        shard = kwargs["shard"]
        assert isinstance(row, dict)
        assert isinstance(shard, AnnotationShard)
        return production.Stage2Row(
            source_index=int(row["source_index"]),
            clip_uid=str(row["clip_uid"]),
            input_annotation_shard=str(shard.path),
            input_annotation_shard_sha256=shard.sha256,
            input_row_sha256=production._row_sha256(row),
            status="coverage_rejected",
            annotation_entity_count=1,
            coverage_passed=False,
        )

    monkeypatch.setattr(production, "load_annotation_shard", counting_load)
    monkeypatch.setattr(production, "process_ready_clip", terminal)
    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=FakeBackend(),
    )

    assert result["completed"] == 10
    assert len(calls) == 2
    assert len(set(calls)) == 2


def test_auto_normal_start_skips_inventory_but_dry_run_keeps_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    inventory_calls = 0
    commands: list[list[str]] = []

    class Process:
        def wait(self) -> int:
            return 0

    def fake_inventory(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal inventory_calls
        del args, kwargs
        inventory_calls += 1
        return {"stage1_total_rows": 123}

    def popen(command: list[str], **kwargs: object) -> Process:
        del kwargs
        commands.append(command)
        return Process()

    monkeypatch.setattr(auto_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(auto_tool, "enumerate_annotation_shards", lambda root: [])
    monkeypatch.setattr(auto_tool, "inventory", fake_inventory)
    monkeypatch.setattr(
        auto_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    monkeypatch.setattr(auto_tool.subprocess, "Popen", popen)

    worker_log_root = fixture.writable / "entity-mask-worker-logs"
    normal = auto_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(fixture.base_path),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "0",
        ],
        worker_log_root=worker_log_root,
    )
    assert inventory_calls == 0
    assert normal["worker_exit_codes"] == [0]
    assert normal["frame_prefetch_workers"] == 0
    assert normal["frame_prefetch_submitted"] == 0
    assert "sam3_request_timing" not in normal
    assert "sam3_timing_total_track_seconds" not in normal
    assert "sam3_session_reuse_mode" not in normal
    idle_flag = commands[0].index("--idle-exit-seconds")
    assert commands[0][idle_flag + 1] == "60.0"
    assert (worker_log_root / "rank-0-gpu-0.log").is_file()

    dry = auto_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(fixture.base_path),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "0",
            "--dry-run",
        ]
    )
    assert inventory_calls == 1
    assert dry["stage1_total_rows"] == 123


def test_zero_argument_entity_mask_wrapper_uses_fixed_production_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    private_config = fixture.writable / "entity_mask_configs" / "production.yaml"
    private_config.parent.mkdir(parents=True)
    private_config.write_text("validated: true\n", encoding="utf-8")
    private_logs = fixture.writable / "entity_mask_logs"
    service = replace(
        fixture.config.qwen.candidate_judge,
        base_url=entity_mask_tool.PRODUCTION_CANDIDATE_JUDGE_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
        timeout_seconds=3600,
    )
    config = replace(
        fixture.config,
        qwen=replace(fixture.config.qwen, candidate_judge=service),
        sam3=replace(
            fixture.config.sam3,
            save_debug_overlays=False,
            object_rescue_mode="phrase_retry_v1",
            not_found_rescue_mode="entity_phrase_retry_v1",
            multi_instance_rescue_mode="qwen_anchor_select_v1",
            anchor_search_mode="progressive_v1",
        ),
        debug=replace(fixture.config.debug, save_diagnostics=False),
    )
    identity = replace(fixture.identity, path=private_config, config=config)
    shard_paths = [
        fixture.input_root
        / "parts"
        / f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl"
        for index in range(16)
    ]
    launched: list[dict[str, object]] = []
    startup: list[dict[str, object]] = []
    output_before_spawn: list[str] = []

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command: list[str], **kwargs: object) -> Process:
        if not launched:
            output_before_spawn.append(capsys.readouterr().out)
        launched.append({"command": command, **kwargs})
        return Process()

    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_INPUT_ROOT", fixture.input_root)
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_OUTPUT_ROOT", entity_mask_root)
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_BASE_CONFIG", private_config)
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_LOG_ROOT", private_logs)
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_CANDIDATE_JUDGE_MODEL",
        service.model,
    )
    monkeypatch.setattr(entity_mask_tool, "load_config_identity", lambda path: identity)
    monkeypatch.setattr(entity_mask_tool, "parse_gpus", lambda value: ["5", "7"])
    monkeypatch.setattr(
        entity_mask_tool,
        "enumerate_annotation_shards",
        lambda root: shard_paths,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "check_stage2_candidate_judge_health",
        lambda current: {"candidate_judge_health": "available"},
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "coordinate_startup_identity",
        lambda **kwargs: startup.append(kwargs),
    )
    monkeypatch.setattr(entity_mask_tool.subprocess, "Popen", popen)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    result = entity_mask_tool.main([])

    assert len(launched) == 2
    assert len(startup) == 1
    startup_plan = json.loads(output_before_spawn[0])
    assert startup_plan["event"] == "startup_plan"
    assert startup_plan["rank"] == 3
    assert startup_plan["world_size"] == 4
    assert startup_plan["stage1_completed_shards"] == 16
    assert len(startup_plan["workers"]) == 2
    commands = [item["command"] for item in launched]
    assert all("run_v3_entity_mask_worker.py" in command[1] for command in commands)
    assert all("--claim-loop" not in command for command in commands)
    assert all("--scan-offset" not in command for command in commands)
    assert [command[command.index("--global-worker-id") + 1] for command in commands] == [
        "6",
        "7",
    ]
    assert [
        (
            command[command.index("--shard-start-position") + 1],
            command[command.index("--shard-end-position") + 1],
        )
        for command in commands
    ] == [("12", "14"), ("14", "16")]
    assert [item["env"]["CUDA_VISIBLE_DEVICES"] for item in launched] == ["5", "7"]
    assert result["rank"] == 3
    assert result["world_size"] == 4
    assert result["global_worker_count"] == 8
    assert result["frame_prefetch_workers"] == 0
    assert result["sam3_request_timing"] is False
    assert result["worker_exit_codes"] == [0, 0]


def _configure_entity_mask_lifecycle_test(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gpu_count: int,
) -> None:
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    shard_paths = [
        fixture.input_root
        / "parts"
        / f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl"
        for index in range(gpu_count)
    ]
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_INPUT_ROOT", fixture.input_root)
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_OUTPUT_ROOT", entity_mask_root)
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_BASE_CONFIG",
        fixture.base_path,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_LOG_ROOT",
        fixture.writable / "entity-mask-logs",
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "load_config_identity",
        lambda path: fixture.identity,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "validate_entity_mask_production_config",
        lambda config: None,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "parse_gpus",
        lambda value: [str(index) for index in range(gpu_count)],
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "enumerate_annotation_shards",
        lambda root: shard_paths,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "available"},
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "coordinate_startup_identity",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def test_entity_mask_worker_signal_exit_is_nonzero_parent_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=3)
    processes = [
        LifecycleProcess(0),
        LifecycleProcess(-signal.SIGKILL),
        LifecycleProcess(0),
    ]
    spawned = iter(processes)
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: next(spawned),
    )

    with pytest.raises(SystemExit) as exc_info:
        entity_mask_tool.main([])

    assert exc_info.value.code == 137
    assert [process.wait_calls for process in processes] == [[None], [None], [None]]
    assert all(process.terminate_calls == 0 for process in processes)
    completed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert completed["event"] == "completed"
    assert completed["worker_exit_codes"] == [0, -9, 0]
    assert completed["success"] is False
    assert completed["parent_exit_code"] == 137


def test_entity_mask_worker_failure_waits_for_other_fixed_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=2)
    processes = [LifecycleProcess(2), LifecycleProcess(0)]
    spawned = iter(processes)
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: next(spawned),
    )

    with pytest.raises(SystemExit) as exc_info:
        entity_mask_tool.main([])

    assert exc_info.value.code == 2
    assert [process.wait_calls for process in processes] == [[None], [None]]
    assert all(process.terminate_calls == 0 for process in processes)


@pytest.mark.parametrize(
    ("termination_signal", "expected_exit"),
    [(signal.SIGTERM, 143), (signal.SIGINT, 130)],
)
def test_entity_mask_launcher_signal_cleans_workers_and_restores_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_signal: int,
    expected_exit: int,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=2)
    processes = [
        LifecycleProcess(None, signal_on_first_wait=termination_signal),
        LifecycleProcess(None),
    ]
    spawned = iter(processes)
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: next(spawned),
    )
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }

    with pytest.raises(SystemExit) as exc_info:
        entity_mask_tool.main([])

    assert exc_info.value.code == expected_exit
    assert [process.terminate_calls for process in processes] == [1, 1]
    assert all(process.kill_calls == 0 for process in processes)
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    } == previous


def test_entity_mask_launcher_keyboard_interrupt_cleans_all_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=2)
    processes = [
        LifecycleProcess(None, interrupt_on_first_wait=KeyboardInterrupt()),
        LifecycleProcess(None),
    ]
    spawned = iter(processes)
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: next(spawned),
    )

    with pytest.raises(SystemExit) as exc_info:
        entity_mask_tool.main([])

    assert exc_info.value.code == 130
    assert [process.terminate_calls for process in processes] == [1, 1]
    assert all(process.kill_calls == 0 for process in processes)


def test_entity_mask_partial_spawn_failure_cleans_started_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=3)
    processes = [LifecycleProcess(None), LifecycleProcess(None)]
    popen_calls = 0

    def popen(*args: object, **kwargs: object) -> LifecycleProcess:
        nonlocal popen_calls
        del args, kwargs
        popen_calls += 1
        if popen_calls > len(processes):
            raise RuntimeError("spawn exploded")
        return processes[popen_calls - 1]

    monkeypatch.setattr(entity_mask_tool.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="spawn exploded"):
        entity_mask_tool.main([])

    assert [process.terminate_calls for process in processes] == [1, 1]
    assert all(process.kill_calls == 0 for process in processes)


def test_entity_mask_success_does_not_terminate_workers_and_restores_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _configure_entity_mask_lifecycle_test(fixture, monkeypatch, gpu_count=2)
    processes = [LifecycleProcess(0), LifecycleProcess(0)]
    spawned = iter(processes)
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: next(spawned),
    )
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }

    result = entity_mask_tool.main([])

    assert result["success"] is True
    assert result["parent_exit_code"] == 0
    assert all(process.terminate_calls == 0 for process in processes)
    assert all(process.kill_calls == 0 for process in processes)
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    } == previous


def test_entity_mask_wrapper_missing_private_config_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "entity_mask_configs" / "production.yaml"
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_BASE_CONFIG", missing)

    with pytest.raises(FileNotFoundError, match=str(missing)):
        entity_mask_tool.main([])


def test_entity_mask_dry_run_only_builds_lightweight_static_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    shards = [
        Path(f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl")
        for index in range(384)
    ]
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_OUTPUT_ROOT", entity_mask_root)
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_BASE_CONFIG",
        fixture.base_path,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "load_config_identity",
        lambda path: fixture.identity,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "validate_entity_mask_production_config",
        lambda config: None,
    )
    monkeypatch.setattr(entity_mask_tool, "parse_gpus", lambda value: [str(i) for i in range(8)])
    monkeypatch.setattr(
        entity_mask_tool,
        "enumerate_annotation_shards",
        lambda root: shards,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "check_stage2_candidate_judge_health",
        lambda config: pytest.fail("dry-run must not check Qwen health"),
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "coordinate_startup_identity",
        lambda **kwargs: pytest.fail("dry-run must not write startup identity"),
    )
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry-run must not launch workers"),
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    result = entity_mask_tool.main(["--dry-run"])

    assert result["stage1_completed_shards"] == 384
    assert result["global_worker_count"] == 8
    assert [worker["shard_count"] for worker in result["workers"]] == [48] * 8
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 1
    assert json.loads(output_lines[0])["event"] == "startup_plan"


def test_entity_mask_wrapper_exits_cleanly_when_no_shards_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    entity_mask_root = fixture.dataset / "jea" / "processed" / "entity_mask"
    monkeypatch.setattr(
        production,
        "ENTITY_MASK_PRODUCTION_OUTPUT_ROOT",
        entity_mask_root,
    )
    monkeypatch.setattr(entity_mask_tool, "PRODUCTION_OUTPUT_ROOT", entity_mask_root)
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_BASE_CONFIG",
        fixture.base_path,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "load_config_identity",
        lambda path: fixture.identity,
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "validate_entity_mask_production_config",
        lambda config: None,
    )
    monkeypatch.setattr(entity_mask_tool, "parse_gpus", lambda value: ["0", "1"])
    monkeypatch.setattr(
        entity_mask_tool,
        "enumerate_annotation_shards",
        lambda root: [],
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "check_stage2_candidate_judge_health",
        lambda config: pytest.fail("zero-shard run must not check Qwen health"),
    )
    monkeypatch.setattr(
        entity_mask_tool,
        "coordinate_startup_identity",
        lambda **kwargs: pytest.fail("zero-shard run must not write identity"),
    )
    monkeypatch.setattr(
        entity_mask_tool.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("zero-shard run must not launch workers"),
    )

    result = entity_mask_tool.main([])

    assert result["stage1_completed_shards"] == 0
    assert result["nothing_to_do"] is True
    assert result["worker_exit_codes"] == []


def test_entity_mask_wrapper_rejects_non_production_visual_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        entity_mask_tool,
        "PRODUCTION_CANDIDATE_JUDGE_MODEL",
        fixture.config.qwen.candidate_judge.model,
    )
    invalid = replace(
        fixture.config,
        sam3=replace(fixture.config.sam3, save_debug_overlays=True),
    )

    with pytest.raises(ValueError, match="sam3.save_debug_overlays"):
        entity_mask_tool.validate_entity_mask_production_config(invalid)


def test_frame_prefetch_uses_frozen_frame_builder_and_exact_frame_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    decoder = FakeDecoder()
    original = production._build_clip_frames
    calls: list[dict[str, object]] = []

    def recording_builder(
        config: V3Config,
        storage: object,
        **kwargs: object,
    ) -> object:
        calls.append(
            {
                "frames": config.frames,
                "clip_uid": kwargs["clip_uid"],
                "video_path": kwargs["video_path"],
                "decoder": kwargs["decoder"],
            }
        )
        return original(config, storage, **kwargs)

    monkeypatch.setattr(production, "_build_clip_frames", recording_builder)
    prepared = production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=decoder,
    )
    storage = production.RunStorage(
        production._workspace_config(fixture.config, workspace)
    )
    frames = production.validate_sampled_frames(storage, "clip-0")

    assert prepared.built is True
    assert prepared.checkpoint.stage == "frames_ready"
    assert calls == [
        {
            "frames": fixture.config.frames,
            "clip_uid": "clip-0",
            "video_path": fixture.dataset / "clips" / "clip-0.mp4",
            "decoder": decoder,
        }
    ]
    assert len(frames.frames) == 10


def test_static_frame_preparation_skips_frame_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard = load_annotation_shard(_write_shard(fixture, [row]))
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    monkeypatch.setattr(
        production,
        "shard_lock",
        lambda *args, **kwargs: pytest.fail("static frame preparation must not lock"),
    )

    prepared = production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=FakeDecoder(),
        static_owner=True,
    )

    assert prepared.built is True
    assert prepared.checkpoint.stage == "frames_ready"


def test_dynamic_frame_preparation_keeps_blocking_frame_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard = load_annotation_shard(_write_shard(fixture, [row]))
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    calls: list[tuple[Path, bool]] = []

    @contextmanager
    def recording_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
        calls.append((path, blocking))
        yield

    monkeypatch.setattr(production, "shard_lock", recording_lock)

    production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=FakeDecoder(),
    )

    assert calls == [
        (production.frame_lock_path(fixture.output_root, shard, "clip-0"), True)
    ]


def test_prefetched_frames_ready_is_reused_by_sam_without_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    prefetch_decoder = FakeDecoder()
    production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=prefetch_decoder,
    )
    sam_decoder = FakeDecoder()
    backend = FakeBackend()

    result = process_ready_clip(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        backend=backend,
        decoder=sam_decoder,
    )

    assert prefetch_decoder.calls
    assert sam_decoder.calls == []
    assert backend.calls == ["e1"]
    assert result.status == "ready_no_background"


def test_prefetch_and_sam_frame_race_builds_frames_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    original = production._build_clip_frames
    build_calls = 0
    counter_lock = threading.Lock()

    def slow_builder(
        config: V3Config,
        storage: object,
        **kwargs: object,
    ) -> object:
        nonlocal build_calls
        with counter_lock:
            build_calls += 1
        time.sleep(0.05)
        return original(config, storage, **kwargs)

    monkeypatch.setattr(production, "_build_clip_frames", slow_builder)

    def prepare() -> production.FramePreparationResult:
        return production.prepare_clip_frames(
            row=row,
            shard=shard,
            output_root=fixture.output_root,
            workspace=workspace,
            config_identity=fixture.identity,
            decoder=FakeDecoder(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare(), range(2)))

    assert build_calls == 1
    assert sorted(result.built for result in results) == [False, True]
    assert all(result.checkpoint.stage == "frames_ready" for result in results)


def test_existing_frames_ready_prefetch_skips_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard.path.stem / "clip-0"
    first = production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=FakeDecoder(),
    )
    second_decoder = FakeDecoder()
    second = production.prepare_clip_frames(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        decoder=second_decoder,
    )

    assert first.built is True
    assert second.built is False
    assert second_decoder.calls == []


def test_prefetch_failure_is_diagnostic_only_and_sam_sync_fallback_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    task = next(production.iter_frame_prefetch_tasks([shard_path]))
    monkeypatch.setattr(
        production,
        "_FRAME_PREFETCH_CONFIG_IDENTITY",
        fixture.identity,
    )
    monkeypatch.setattr(
        production,
        "_FRAME_PREFETCH_OUTPUT_ROOT",
        fixture.output_root,
    )
    original = production.prepare_clip_frames
    monkeypatch.setattr(
        production,
        "prepare_clip_frames",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("decode unavailable")),
    )

    diagnostic = production.run_frame_prefetch_task(task)

    assert diagnostic["status"] == "failed"
    assert "decode unavailable" in str(diagnostic["reason"])
    assert not list((fixture.output_root / "parts").glob("*.jsonl"))

    monkeypatch.setattr(production, "prepare_clip_frames", original)
    result = process_shard(
        shard_path,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
        chunk_rows=1,
    )
    assert Path(result["path"]).name == shard_path.name


def test_prefetch_task_identity_is_clip_level_and_independent_of_chunk_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(
        fixture,
        [
            _row(fixture, 0),
            _row(fixture, 1, status="failed"),
            _row(fixture, 2, entities=0),
            _row(fixture, 3),
        ],
    )
    shard = load_annotation_shard(shard_path)
    tasks = list(production.iter_frame_prefetch_tasks([shard_path]))

    assert [task.row["source_index"] for task in tasks] == [0, 3]
    assert [task.row["clip_uid"] for task in tasks] == ["clip-0", "clip-3"]
    assert all(task.input_annotation_shard_sha256 == shard.sha256 for task in tasks)
    assert [task.row["source_index"] for task in tasks] == [
        task.row["source_index"]
        for task in production.iter_frame_prefetch_tasks([shard_path])
    ]


def test_auto_frame_prefetch_pool_is_once_per_node_not_per_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard_path = _write_shard(fixture, [_row(fixture, 0)])
    controllers: list[object] = []
    commands: list[list[str]] = []

    class Process:
        def wait(self) -> int:
            return 0

    class Controller:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            controllers.append(self)

        def start(self) -> None:
            self.started = True

        def stop_and_join(self) -> dict[str, object]:
            self.stopped = True
            return {
                "frame_prefetch_workers": 32,
                "frame_prefetch_submitted": 1,
                "frame_prefetch_completed": 1,
                "frame_prefetch_skipped_existing": 0,
                "frame_prefetch_failed": 0,
                "frame_prefetch_wall_seconds": 0.5,
            }

    def popen(command: list[str], **kwargs: object) -> Process:
        del kwargs
        commands.append(command)
        return Process()

    monkeypatch.setattr(auto_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        auto_tool, "enumerate_annotation_shards", lambda root: [shard_path]
    )
    monkeypatch.setattr(
        auto_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    monkeypatch.setattr(auto_tool, "_FramePrefetchController", Controller)
    monkeypatch.setattr(auto_tool.subprocess, "Popen", popen)

    result = auto_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(fixture.base_path),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "0,1,2,3,4,5,6,7",
            "--frame-prefetch-workers",
            "32",
            "--chunk-rows",
            "1",
        ]
    )

    assert len(controllers) == 1
    controller = controllers[0]
    assert controller.kwargs["workers"] == 32
    assert controller.kwargs["shard_paths"] == [shard_path]
    assert controller.started is True
    assert controller.stopped is True
    assert len(commands) == 8
    assert all("--frame-prefetch-workers" not in command for command in commands)
    assert all("--sam3-session-reuse-mode" not in command for command in commands)
    assert result["frame_prefetch_completed"] == 1
    source = Path(auto_tool.__file__).read_text(encoding="utf-8")
    assert "ProcessPoolExecutor" in source
    assert "ThreadPoolExecutor" not in source


@dataclass
class _ManualClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


class _TransparentPredictor:
    def __init__(self) -> None:
        self.requests: list[tuple[str, object]] = []
        self.return_value = object()
        self.stream_values = (object(), object())

    def handle_request(self, request: object) -> object:
        self.requests.append(("request", request))
        return self.return_value

    def handle_stream_request(self, request: object):
        self.requests.append(("stream", request))
        yield from self.stream_values


class _StatefulSessionPredictor:
    def __init__(
        self,
        *,
        fail_add_prompt: BaseException | None = None,
        fail_reset: BaseException | None = None,
    ) -> None:
        self.fail_add_prompt = fail_add_prompt
        self.fail_reset = fail_reset
        self.next_session = 0
        self.sessions: dict[str, dict[str, int]] = {}
        self.requests: list[tuple[str, object]] = []
        self.add_prompt_outputs: list[np.ndarray] = []
        self.propagation_outputs: list[tuple[str, tuple[np.ndarray, ...]]] = []
        self.shutdown_calls = 0

    @staticmethod
    def _output(mask: np.ndarray) -> dict[str, object]:
        return {
            "out_binary_masks": mask[None, None].astype(np.uint8),
            "out_probs": np.asarray([0.9], dtype=np.float32),
            "out_obj_ids": np.asarray(["object-0"]),
        }

    @staticmethod
    def _mask(version: int) -> np.ndarray:
        mask = np.zeros((12, 16), dtype=bool)
        start = 5 + min(version, 3)
        mask[4:8, start : start + 4] = True
        return mask

    def handle_request(self, request: object) -> dict[str, object]:
        assert isinstance(request, dict)
        request_type = str(request["type"])
        self.requests.append((request_type, request))
        if request_type == "start_session":
            self.next_session += 1
            session_id = f"session-{self.next_session}"
            self.sessions[session_id] = {"prompts": 0, "propagations": 0}
            return {"session_id": session_id}
        session_id = str(request["session_id"])
        if request_type == "close_session":
            self.sessions.pop(session_id)
            return {}
        state = self.sessions[session_id]
        if request_type == "reset_session":
            if self.fail_reset is not None:
                raise self.fail_reset
            state.update(prompts=0, propagations=0)
            return {"reset": True}
        if request_type == "add_prompt":
            if self.fail_add_prompt is not None:
                raise self.fail_add_prompt
            mask = self._mask(state["prompts"])
            state["prompts"] += 1
            self.add_prompt_outputs.append(mask.copy())
            return {
                "frame_index": int(request["frame_index"]),
                "outputs": self._output(mask),
            }
        raise AssertionError(f"unexpected request type: {request_type}")

    def handle_stream_request(self, request: object):
        assert isinstance(request, dict)
        self.requests.append(("propagate_in_video", request))
        session_id = str(request["session_id"])
        state = self.sessions[session_id]
        if state["prompts"] != 1:
            raise RuntimeError("propagation requires one fresh prompt")
        mask = self._mask(state["prompts"] - 1)
        state["propagations"] += 1
        direction = str(request["propagation_direction"])
        responses = tuple(mask.copy() for _ in range(10))
        self.propagation_outputs.append((direction, responses))
        for slot, current in enumerate(responses):
            yield {"frame_index": slot, "outputs": self._output(current)}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _logical_sam3_round(
    predictor: object,
    *,
    resource_path: str,
    frame_index: int = 5,
    direction: str = "forward",
) -> tuple[dict[str, object], list[object]]:
    start = predictor.handle_request(  # type: ignore[attr-defined]
        {"type": "start_session", "resource_path": resource_path}
    )
    session_id = str(start["session_id"])
    prompted = predictor.handle_request(  # type: ignore[attr-defined]
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": "unchanged prompt",
        }
    )
    propagated = list(
        predictor.handle_stream_request(  # type: ignore[attr-defined]
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
            }
        )
    )
    predictor.handle_request(  # type: ignore[attr-defined]
        {"type": "close_session", "session_id": session_id}
    )
    return prompted, propagated


def test_timed_sam3_predictor_handle_request_is_transparent() -> None:
    clock = _ManualClock(10.0)
    collector = production.Sam3RequestTimingCollector()

    class Predictor(_TransparentPredictor):
        def handle_request(self, request: object) -> object:
            result = super().handle_request(request)
            clock.value = 15.0
            return result

    predictor = Predictor()
    proxy = production.TimedSam3PredictorProxy(
        predictor,
        collector,
        clock=clock,
    )
    request = {"type": "start_session", "resource_path": "/unchanged"}

    clock.value = 12.0
    result = proxy.handle_request(request)
    assert result is predictor.return_value
    assert predictor.requests == [("request", request)]
    assert predictor.requests[0][1] is request
    assert request == {"type": "start_session", "resource_path": "/unchanged"}
    timing = collector.summary()["handle_request.start_session"]
    assert timing["calls"] == 1
    assert timing["total_seconds"] == 3.0


def test_timed_sam3_stream_covers_full_iteration_and_preserves_responses() -> None:
    clock = _ManualClock(1.0)
    collector = production.Sam3RequestTimingCollector()
    predictor = _TransparentPredictor()
    proxy = production.TimedSam3PredictorProxy(
        predictor,
        collector,
        clock=clock,
    )
    request = {"type": "propagate_in_video", "session_id": "unchanged"}

    stream = proxy.handle_stream_request(request)
    assert predictor.requests == []
    assert collector.summary() == {}
    clock.value = 2.0
    first = next(stream)
    assert collector.summary() == {}
    clock.value = 4.0
    second = next(stream)
    assert collector.summary() == {}
    clock.value = 9.0
    with pytest.raises(StopIteration):
        next(stream)

    assert first is predictor.stream_values[0]
    assert second is predictor.stream_values[1]
    assert predictor.requests == [("stream", request)]
    assert predictor.requests[0][1] is request
    timing = collector.summary()["handle_stream_request.propagate_in_video"]
    assert timing == {
        "calls": 1,
        "total_seconds": 8.0,
        "mean_seconds": 8.0,
        "max_seconds": 8.0,
    }


def test_timed_sam3_predictor_preserves_request_and_stream_exceptions() -> None:
    class RequestError(RuntimeError):
        pass

    class StreamError(ValueError):
        pass

    request_error = RequestError("request failed")
    stream_error = StreamError("stream failed")

    class Predictor:
        def handle_request(self, request: object) -> object:
            del request
            raise request_error

        def handle_stream_request(self, request: object):
            del request
            yield "before failure"
            raise stream_error

    clock = _ManualClock(3.0)
    collector = production.Sam3RequestTimingCollector()
    proxy = production.TimedSam3PredictorProxy(
        Predictor(),
        collector,
        clock=clock,
    )

    with pytest.raises(RequestError) as request_info:
        proxy.handle_request({"type": "add_prompt"})
    assert request_info.value is request_error
    clock.value = 5.0
    stream = proxy.handle_stream_request({"type": "propagate_in_video"})
    assert next(stream) == "before failure"
    clock.value = 8.0
    with pytest.raises(StreamError) as stream_info:
        next(stream)
    assert stream_info.value is stream_error
    assert collector.summary()["handle_request.add_prompt"]["calls"] == 1
    assert collector.summary()["handle_stream_request.propagate_in_video"] == {
        "calls": 1,
        "total_seconds": 3.0,
        "mean_seconds": 3.0,
        "max_seconds": 3.0,
    }


def test_sam3_timing_on_off_sequence_is_identical_and_lazy() -> None:
    predictor = _TransparentPredictor()

    class Backend:
        _predictor = None

        def __init__(self) -> None:
            self.build_calls: list[bool] = []

        def _build_predictor(self, *, compile_enabled: bool) -> object:
            self.build_calls.append(compile_enabled)
            return predictor

    backend = Backend()
    collector = production.enable_sam3_request_timing(backend)

    assert backend.build_calls == []
    assert predictor.requests == []
    timed_predictor = backend._build_predictor(compile_enabled=True)
    assert backend.build_calls == [True]
    assert isinstance(timed_predictor, production.TimedSam3PredictorProxy)

    requests = [
        {"type": "start_session"},
        {"type": "add_prompt"},
        {"type": "propagate_in_video"},
        {"type": "close_session"},
    ]
    raw_predictor = _TransparentPredictor()
    raw_predictor.handle_request(requests[0])
    raw_predictor.handle_request(requests[1])
    list(raw_predictor.handle_stream_request(requests[2]))
    raw_predictor.handle_request(requests[3])
    timed_predictor.handle_request(requests[0])
    timed_predictor.handle_request(requests[1])
    list(timed_predictor.handle_stream_request(requests[2]))
    timed_predictor.handle_request(requests[3])

    assert predictor.requests == raw_predictor.requests
    assert sum(value["calls"] for value in collector.summary().values()) == 4
    assert production.enable_sam3_request_timing(backend) is collector
    assert backend.build_calls == [True]


def test_clip_session_reuse_matches_fresh_logical_outputs() -> None:
    baseline = _StatefulSessionPredictor()
    reused_raw = _StatefulSessionPredictor()
    diagnostics = production.Sam3SessionReuseDiagnostics()
    reused = production.Sam3ClipSessionReuseProxy(reused_raw, diagnostics)

    baseline_results = [
        _logical_sam3_round(baseline, resource_path="/frames/clip-a")
        for _ in range(3)
    ]
    reused_results = [
        _logical_sam3_round(reused, resource_path="/frames/clip-a")
        for _ in range(3)
    ]

    for (baseline_prompt, baseline_stream), (reuse_prompt, reuse_stream) in zip(
        baseline_results,
        reused_results,
    ):
        assert baseline_prompt["frame_index"] == reuse_prompt["frame_index"]
        assert np.array_equal(
            baseline_prompt["outputs"]["out_binary_masks"],  # type: ignore[index]
            reuse_prompt["outputs"]["out_binary_masks"],  # type: ignore[index]
        )
        assert len(baseline_stream) == len(reuse_stream)
        for baseline_response, reuse_response in zip(
            baseline_stream,
            reuse_stream,
        ):
            assert baseline_response["frame_index"] == reuse_response["frame_index"]
            assert np.array_equal(
                baseline_response["outputs"]["out_binary_masks"],
                reuse_response["outputs"]["out_binary_masks"],
            )

    assert [name for name, _ in reused_raw.requests].count("start_session") == 1
    assert [name for name, _ in reused_raw.requests].count("reset_session") == 3
    assert [name for name, _ in reused_raw.requests].count("close_session") == 0
    assert diagnostics.summary() == {
        "sam3_session_reuse_mode": "clip_reset_v1",
        "sam3_logical_start_session_calls": 3,
        "sam3_logical_close_session_calls": 3,
        "sam3_physical_start_session_calls": 1,
        "sam3_physical_reset_session_calls": 3,
        "sam3_physical_close_session_calls": 0,
        "sam3_reused_start_session_calls": 2,
        "sam3_session_resource_switches": 0,
    }


def test_clip_session_reuse_preserves_frozen_track_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    frames_dir = tmp_path / "track-frames"
    frames_dir.mkdir()
    frame_paths = []
    for slot in range(10):
        path = frames_dir / f"{slot:02d}.jpg"
        path.write_bytes(b"frame")
        frame_paths.append(path)
    baseline_predictor = _StatefulSessionPredictor()
    reused_predictor = _StatefulSessionPredictor()
    baseline_backend = Sam3SegmentationBackend(
        fixture.config.sam3,
        predictor=baseline_predictor,
    )
    reused_backend = Sam3SegmentationBackend(
        fixture.config.sam3,
        predictor=reused_predictor,
    )
    diagnostics = production.enable_sam3_session_reuse(
        reused_backend,
        mode="clip_reset_v1",
    )

    baseline_result = baseline_backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="person",
        entity_phrase="person",
    )
    reused_result = reused_backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="person",
        entity_phrase="person",
    )

    assert baseline_result.status == reused_result.status == "ready"
    assert len(baseline_result.observations) == len(reused_result.observations)
    for baseline_observation, reused_observation in zip(
        baseline_result.observations,
        reused_result.observations,
    ):
        assert baseline_observation.slot == reused_observation.slot
        assert baseline_observation.object_id == reused_observation.object_id
        assert baseline_observation.confidence == reused_observation.confidence
        assert np.array_equal(baseline_observation.mask, reused_observation.mask)
    assert diagnostics is not None
    assert diagnostics.logical_start_session_calls == 3
    assert diagnostics.physical_start_session_calls == 1
    assert diagnostics.physical_reset_session_calls == 3
    assert diagnostics.reused_start_session_calls == 2
    assert [name for name, _ in reused_predictor.requests].count("start_session") == 1

    reused_backend.close()
    assert [name for name, _ in reused_predictor.requests].count("close_session") == 1
    assert reused_predictor.shutdown_calls == 1


def test_clip_session_reuse_switches_resources_and_shutdown_closes_once() -> None:
    raw = _StatefulSessionPredictor()
    diagnostics = production.Sam3SessionReuseDiagnostics()
    proxy = production.Sam3ClipSessionReuseProxy(raw, diagnostics)

    _logical_sam3_round(proxy, resource_path="/frames/clip-a")
    _logical_sam3_round(proxy, resource_path="/frames/clip-b")
    proxy.shutdown()

    request_types = [name for name, _ in raw.requests]
    assert request_types.count("start_session") == 2
    assert request_types.count("reset_session") == 2
    assert request_types.count("close_session") == 2
    assert diagnostics.physical_start_session_calls == 2
    assert diagnostics.physical_close_session_calls == 2
    assert diagnostics.session_resource_switches == 1
    assert raw.shutdown_calls == 1
    assert raw.sessions == {}


def test_clip_session_reuse_fails_closed_on_invalid_sequence() -> None:
    raw = _StatefulSessionPredictor()
    proxy = production.Sam3ClipSessionReuseProxy(
        raw,
        production.Sam3SessionReuseDiagnostics(),
    )
    start = proxy.handle_request(
        {"type": "start_session", "resource_path": "/frames/clip-a"}
    )
    session_id = str(start["session_id"])

    with pytest.raises(RuntimeError, match="already active"):
        proxy.handle_request(
            {"type": "start_session", "resource_path": "/frames/clip-a"}
        )
    with pytest.raises(RuntimeError, match="inactive session"):
        proxy.handle_request(
            {"type": "add_prompt", "session_id": "wrong", "frame_index": 5}
        )
    with pytest.raises(RuntimeError, match="inactive session"):
        proxy.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "wrong",
                "propagation_direction": "forward",
            }
        )
    with pytest.raises(RuntimeError, match="inactive session"):
        proxy.handle_request({"type": "close_session", "session_id": "wrong"})

    proxy.handle_request({"type": "close_session", "session_id": session_id})
    proxy.shutdown()


def test_clip_session_reuse_cleanup_does_not_override_original_exception() -> None:
    original = LookupError("original add_prompt failure")
    reset_failure = RuntimeError("reset unsupported")
    raw = _StatefulSessionPredictor(
        fail_add_prompt=original,
        fail_reset=reset_failure,
    )
    proxy = production.Sam3ClipSessionReuseProxy(
        raw,
        production.Sam3SessionReuseDiagnostics(),
    )
    start = proxy.handle_request(
        {"type": "start_session", "resource_path": "/frames/clip-a"}
    )
    session_id = str(start["session_id"])

    with pytest.raises(LookupError) as exc_info:
        try:
            proxy.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 5,
                    "text": "prompt",
                }
            )
        finally:
            proxy.handle_request(
                {"type": "close_session", "session_id": session_id}
            )

    assert exc_info.value is original
    assert [name for name, _ in raw.requests].count("reset_session") == 1
    assert [name for name, _ in raw.requests].count("close_session") == 1
    assert raw.sessions == {}


def test_clip_session_reuse_fails_if_reset_session_is_unsupported() -> None:
    reset_failure = RuntimeError("reset_session is unsupported")
    raw = _StatefulSessionPredictor(fail_reset=reset_failure)
    proxy = production.Sam3ClipSessionReuseProxy(
        raw,
        production.Sam3SessionReuseDiagnostics(),
    )
    start = proxy.handle_request(
        {"type": "start_session", "resource_path": "/frames/clip-a"}
    )

    with pytest.raises(RuntimeError) as exc_info:
        proxy.handle_request(
            {"type": "close_session", "session_id": str(start["session_id"])}
        )

    assert exc_info.value is reset_failure
    assert [name for name, _ in raw.requests] == [
        "start_session",
        "reset_session",
        "close_session",
    ]
    assert raw.sessions == {}


def test_timing_and_reuse_count_only_physical_requests() -> None:
    raw = _StatefulSessionPredictor()
    collector = production.Sam3RequestTimingCollector()
    timed = production.TimedSam3PredictorProxy(raw, collector)
    diagnostics = production.Sam3SessionReuseDiagnostics()
    reused = production.Sam3ClipSessionReuseProxy(timed, diagnostics)

    add_requests: list[dict[str, object]] = []
    stream_requests: list[dict[str, object]] = []
    for _ in range(2):
        start = reused.handle_request(
            {"type": "start_session", "resource_path": "/frames/clip-a"}
        )
        session_id = str(start["session_id"])
        add_request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 5,
            "text": "unchanged prompt",
        }
        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
        }
        add_requests.append(add_request)
        stream_requests.append(stream_request)
        reused.handle_request(add_request)
        list(reused.handle_stream_request(stream_request))
        reused.handle_request(
            {"type": "close_session", "session_id": session_id}
        )
    reused.shutdown()

    timing = collector.summary()
    assert timing["handle_request.start_session"]["calls"] == 1
    assert timing["handle_request.reset_session"]["calls"] == 2
    assert timing["handle_request.close_session"]["calls"] == 1
    assert timing["handle_request.add_prompt"]["calls"] == 2
    assert timing["handle_stream_request.propagate_in_video"]["calls"] == 2
    assert diagnostics.logical_start_session_calls == 2
    assert diagnostics.reused_start_session_calls == 1
    raw_add_requests = [request for name, request in raw.requests if name == "add_prompt"]
    raw_stream_requests = [
        request for name, request in raw.requests if name == "propagate_in_video"
    ]
    assert raw_add_requests == add_requests
    assert raw_stream_requests == stream_requests
    assert all(
        raw_request is supplied
        for raw_request, supplied in zip(raw_add_requests, add_requests)
    )
    assert all(
        raw_request is supplied
        for raw_request, supplied in zip(raw_stream_requests, stream_requests)
    )


def test_sam3_session_reuse_marker_prevents_runtime_mode_mixing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    reuse_root = fixture.writable / "reuse-root"

    marker = production.ensure_sam3_session_reuse_identity(
        reuse_root,
        mode="clip_reset_v1",
    )
    assert marker is not None
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "mode": "clip_reset_v1",
    }
    execution = production.ensure_execution_identity(reuse_root, chunk_rows=1)
    assert json.loads(execution.read_text(encoding="utf-8"))[
        "execution_strategy"
    ] == production.STAGE2_EXECUTION_STRATEGY
    assert "sam3_session_reuse_mode" not in production.Stage2Row.model_fields
    assert (
        production.ensure_sam3_session_reuse_identity(
            reuse_root,
            mode="clip_reset_v1",
        )
        == marker
    )
    with pytest.raises(ValueError, match="reuse mode mismatch"):
        production.ensure_sam3_session_reuse_identity(reuse_root, mode="off")

    legacy_root = fixture.writable / "legacy-off-root"
    production.ensure_execution_identity(legacy_root, chunk_rows=1)
    assert (
        production.ensure_sam3_session_reuse_identity(legacy_root, mode="off")
        is None
    )
    assert not (legacy_root / "_internal" / "sam3-session-reuse.json").exists()


def test_sam3_session_reuse_rejects_existing_unmarked_stage2_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    root = fixture.writable / "unmarked-stage2"
    artifact = root / "artifacts" / "shard-000000000-000009999" / "clip-0"
    artifact.mkdir(parents=True)
    (artifact / "state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no SAM3 session reuse marker"):
        production.ensure_sam3_session_reuse_identity(
            root,
            mode="clip_reset_v1",
        )


def test_sam3_worker_timing_includes_frozen_backend_counters() -> None:
    class Backend:
        def performance_counters(self) -> dict[str, object]:
            return {
                "sam3_predictor_startup_seconds": 4.0,
                "sam3_segment_model_call_time_seconds": 9.0,
                "sam3_segment_clips": 2,
                "sam3_segment_entities": 3,
                "first_segment_clip_seconds": 6.0,
                "steady_state_segment_mean_seconds": 3.0,
            }

        def anchor_search_counters(self) -> dict[str, int]:
            return {"anchor_probe_calls": 7, "anchor_fast_path_hits": 2}

        def recall_rescue_counters(self) -> dict[str, int]:
            return {"multi_instance_rescue_attempted": 1}

    collector = production.Sam3RequestTimingCollector()
    collector.record("handle_request.start_session", 2.5)
    diagnostics = production.sam3_worker_timing_diagnostics(Backend(), collector)

    assert diagnostics["sam3_request_timing"] == {
        "handle_request.start_session": {
            "calls": 1,
            "total_seconds": 2.5,
            "mean_seconds": 2.5,
            "max_seconds": 2.5,
        }
    }
    assert diagnostics["sam3_backend_performance_counters"] == {
        "sam3_predictor_startup_seconds": 4.0,
        "sam3_segment_model_call_time_seconds": 9.0,
        "sam3_segment_clips": 2,
        "sam3_segment_entities": 3,
        "first_segment_clip_seconds": 6.0,
        "steady_state_segment_mean_seconds": 3.0,
    }
    assert diagnostics["sam3_anchor_search_counters"] == {
        "anchor_probe_calls": 7,
        "anchor_fast_path_hits": 2,
    }
    assert diagnostics["sam3_recall_rescue_counters"] == {
        "multi_instance_rescue_attempted": 1
    }


def test_auto_sam3_timing_aggregates_current_worker_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command: list[str], **kwargs: object) -> Process:
        commands.append(command)
        gpu = command[command.index("--gpu") + 1]
        multiplier = int(gpu) + 1
        output = kwargs["stdout"]
        payload = {
            "sam3_request_timing": {
                "handle_request.start_session": {
                    "calls": multiplier,
                    "total_seconds": 10.0 * multiplier,
                    "mean_seconds": 10.0,
                    "max_seconds": 10.0,
                },
                "handle_request.add_prompt": {
                    "calls": multiplier,
                    "total_seconds": 2.0 * multiplier,
                    "mean_seconds": 2.0,
                    "max_seconds": 2.0,
                },
                "handle_request.reset_session": {
                    "calls": multiplier,
                    "total_seconds": 4.0 * multiplier,
                    "mean_seconds": 4.0,
                    "max_seconds": 4.0,
                },
                "handle_stream_request.propagate_in_video": {
                    "calls": multiplier,
                    "total_seconds": 3.0 * multiplier,
                    "mean_seconds": 3.0,
                    "max_seconds": 3.0,
                },
                "handle_request.close_session": {
                    "calls": multiplier,
                    "total_seconds": 1.0 * multiplier,
                    "mean_seconds": 1.0,
                    "max_seconds": 1.0,
                },
            },
            "sam3_backend_performance_counters": {
                "sam3_predictor_startup_seconds": 5.0 * multiplier,
                "sam3_segment_model_call_time_seconds": 20.0 * multiplier,
                "sam3_segment_clips": multiplier,
                "sam3_segment_entities": multiplier,
                "first_segment_clip_seconds": 20.0,
                "steady_state_segment_mean_seconds": 0.0,
            },
            "sam3_anchor_search_counters": {
                "anchor_probe_calls": multiplier,
            },
            "sam3_recall_rescue_counters": {
                "multi_instance_rescue_attempted": multiplier,
            },
            "sam3_session_reuse_mode": "clip_reset_v1",
            "sam3_logical_start_session_calls": 4 * multiplier,
            "sam3_logical_close_session_calls": 4 * multiplier,
            "sam3_physical_start_session_calls": multiplier,
            "sam3_physical_reset_session_calls": 4 * multiplier,
            "sam3_physical_close_session_calls": multiplier,
            "sam3_reused_start_session_calls": 3 * multiplier,
            "sam3_session_resource_switches": multiplier,
        }
        output.write((json.dumps(payload) + "\n").encode())
        output.flush()
        return Process()

    monkeypatch.setattr(auto_tool, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(auto_tool, "enumerate_annotation_shards", lambda root: [])
    monkeypatch.setattr(
        auto_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "not_required"},
    )
    monkeypatch.setattr(auto_tool.subprocess, "Popen", popen)

    result = auto_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(fixture.base_path),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "0,1",
            "--sam3-request-timing",
            "--sam3-session-reuse-mode",
            "clip_reset_v1",
        ]
    )

    assert all("--sam3-request-timing" in command for command in commands)
    assert all("--sam3-session-reuse-mode" in command for command in commands)
    assert result["sam3_timing_total_start_session_calls"] == 3
    assert result["sam3_timing_total_start_session_seconds"] == 30.0
    assert result["sam3_timing_total_add_prompt_seconds"] == 6.0
    assert result["sam3_timing_total_reset_session_calls"] == 3
    assert result["sam3_timing_total_reset_session_seconds"] == 12.0
    assert result["sam3_timing_total_propagate_seconds"] == 9.0
    assert result["sam3_timing_total_close_session_seconds"] == 3.0
    assert result["sam3_timing_total_request_seconds"] == 60.0
    assert result["sam3_timing_total_track_seconds"] == 60.0
    assert result["sam3_timing_request_seconds_fraction_of_track"] == 1.0
    assert result["sam3_timing_unattributed_seconds"] == 0.0
    assert set(result["sam3_timing_per_gpu"]) == {"rank0_gpu0", "rank0_gpu1"}
    assert result["sam3_anchor_search_totals"] == {"anchor_probe_calls": 3}
    assert result["sam3_recall_rescue_totals"] == {
        "multi_instance_rescue_attempted": 3
    }
    assert result["sam3_session_reuse_mode"] == "clip_reset_v1"
    assert result["sam3_physical_start_session_calls"] == 3
    assert result["sam3_physical_reset_session_calls"] == 12
    assert result["sam3_physical_close_session_calls"] == 3
    assert result["sam3_reused_start_session_calls"] == 9


def test_stage2_orchestration_keeps_frozen_visual_file_hashes() -> None:
    repository = Path(production.__file__).resolve().parents[2]
    expected = {
        "frames.py": "3f527c834205cd53de4587e1390afed932944291836ee81dc9c5af426c9efefc",
        "segment.py": "65c3b3ec99e710cafb7c97bbc9f813ef915deb40b617bc6c0f9309b38fe31f92",
        "sam3_backend.py": "67ea1794983a4a04b604a1a5162b0e98f17688f9403192d43ec55840b4d453a3",
        "sam3_anchor_selector.py": "4011e01580bcf55b20bc131683d8f3a7cf25bdfc07ef4817894d97a7fdb18552",
        "rank.py": "e93129bdbac660f45e2b970733c5e7bbf83637db01a127a789554ae9a9c6cf86",
        "background.py": "2231a4e40b6172e053932c84b4041e3d866d2db90f07e18be1d7db4765aabb68",
    }

    for name, digest in expected.items():
        path = repository / "r2v_data_v2" / "v3" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
